import pytest
import torch
import torch.nn.functional as F

from ndm.triton.e88_triton_backward import e88_triton
from ndm.triton.e88_triton_forward import e88_torch_reference


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests"),
]


def _inputs(seed=0, requires_grad=False):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    T, B, H, N, V = 16, 1, 64, 4, 4

    S0 = 0.05 * torch.randn(B, H, N, V, device=device)
    k = 0.20 * torch.randn(T, B, H, N, device=device)
    v = 0.20 * torch.randn(T, B, H, V, device=device)
    q = 0.20 * torch.randn(T, B, H, N, device=device)
    decay = 0.35 + 0.55 * torch.rand(T, B, H, device=device)

    tensors = (S0, k, v, q, decay)
    if requires_grad:
        tensors = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    return tensors


def _l2(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def test_e88_triton_forward_matches_reference():
    S0, k, v, q, decay = _inputs(seed=1)

    out_tri, S_tri = e88_triton(S0, k, v, q, decay)
    out_ref, S_ref, _ = e88_torch_reference(S0, k, v, q, decay)

    torch.testing.assert_close(out_tri, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(S_tri, S_ref, rtol=2e-4, atol=2e-4)


def test_e88_triton_unaligned_inference_returns_prepadding_state():
    """Regression: alignment padding must not become the recurrent cache."""
    S0, k, v, q, decay = _inputs(seed=8)
    valid_length = 5

    torch.manual_seed(88)
    erase = torch.sigmoid(0.20 * torch.randn_like(k))
    write = torch.sigmoid(0.20 * torch.randn_like(v))

    out_ref, state_ref, _ = e88_torch_reference(
        S0,
        k[:valid_length],
        v[:valid_length],
        q[:valid_length],
        decay[:valid_length],
        erase_gate=erase[:valid_length],
        value_write_gate=write[:valid_length],
    )

    # Mirror the optimized inference wrapper: real projections followed by
    # zero padding to the 16-step sparse-checkpoint interval.
    padded = [tensor.clone() for tensor in (k, v, q, decay, erase, write)]
    for tensor in padded:
        tensor[valid_length:] = 0
    k_pad, v_pad, q_pad, decay_pad, erase_pad, write_pad = padded
    out_padded, state_valid = e88_triton(
        S0, k_pad, v_pad, q_pad, decay_pad,
        erase_gate=erase_pad,
        value_write_gate=write_pad,
        valid_length=valid_length,
    )

    torch.testing.assert_close(
        out_padded[:valid_length], out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_valid, state_ref, rtol=2e-4, atol=2e-4)
    assert state_valid.abs().max().item() > 0.0


def test_e88_triton_fp32_cache_is_chunk_boundary_invariant():
    """Token-at-a-time inference must retain the fused fp32 state."""
    S0, k, v, q, decay = _inputs(seed=9)
    S0 = S0.float()
    k, v, q, decay = (tensor.bfloat16() for tensor in (k, v, q, decay))

    torch.manual_seed(99)
    erase = torch.sigmoid(0.20 * torch.randn_like(k))
    write = torch.sigmoid(0.20 * torch.randn_like(v))

    out_full, state_full = e88_triton(
        S0, k, v, q, decay,
        erase_gate=erase,
        value_write_gate=write,
    )

    state_cached = S0
    outputs = []
    for index in range(k.shape[0]):
        def padded_step(tensor):
            result = torch.zeros_like(tensor)
            result[0] = tensor[index]
            return result

        out_step, state_cached = e88_triton(
            state_cached,
            padded_step(k),
            padded_step(v),
            padded_step(q),
            padded_step(decay),
            erase_gate=padded_step(erase),
            value_write_gate=padded_step(write),
            valid_length=1,
        )
        assert state_cached.dtype == torch.float32
        outputs.append(out_step[:1])

    out_cached = torch.cat(outputs, dim=0)
    torch.testing.assert_close(out_cached, out_full, rtol=0, atol=0)
    torch.testing.assert_close(state_cached, state_full, rtol=0, atol=0)


def test_e88_triton_fused_norm_and_gate_match_reference():
    S0, k, v, q, decay = _inputs(seed=2)
    gate = 0.20 * torch.randn_like(v)

    out_tri, S_tri = e88_triton(S0, k, v, q, decay, g=gate, normalize_kq=True)
    out_ref, S_ref, _ = e88_torch_reference(S0, _l2(k), v, _l2(q), decay)
    out_ref = F.silu(gate) * out_ref

    torch.testing.assert_close(out_tri, out_ref, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(S_tri, S_ref, rtol=3e-4, atol=3e-4)


def test_e88_triton_backward_matches_reference():
    tri_inputs = _inputs(seed=3, requires_grad=True)
    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in tri_inputs)

    out_tri, S_tri = e88_triton(*tri_inputs)
    out_ref, S_ref, _ = e88_torch_reference(*ref_inputs)

    loss_tri = out_tri.float().square().mean() + 0.3 * S_tri.float().square().mean()
    loss_ref = out_ref.float().square().mean() + 0.3 * S_ref.float().square().mean()

    grads_tri = torch.autograd.grad(loss_tri, tri_inputs)
    grads_ref = torch.autograd.grad(loss_ref, ref_inputs)

    for grad_tri, grad_ref in zip(grads_tri, grads_ref):
        torch.testing.assert_close(grad_tri, grad_ref, rtol=5e-3, atol=2e-3)


def test_e88_triton_raw_write_matches_reference():
    tri_inputs = _inputs(seed=4, requires_grad=True)
    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in tri_inputs)

    out_tri, S_tri = e88_triton(*tri_inputs, raw_write=True)
    out_ref, S_ref, _ = e88_torch_reference(*ref_inputs, raw_write=True)

    torch.testing.assert_close(out_tri, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(S_tri, S_ref, rtol=2e-4, atol=2e-4)

    loss_tri = out_tri.float().square().mean() + 0.3 * S_tri.float().square().mean()
    loss_ref = out_ref.float().square().mean() + 0.3 * S_ref.float().square().mean()

    grads_tri = torch.autograd.grad(loss_tri, tri_inputs)
    grads_ref = torch.autograd.grad(loss_ref, ref_inputs)

    for grad_tri, grad_ref in zip(grads_tri, grads_ref):
        torch.testing.assert_close(grad_tri, grad_ref, rtol=5e-3, atol=2e-3)


def test_e88_triton_linear_state_matches_reference():
    tri_inputs = _inputs(seed=6, requires_grad=True)
    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in tri_inputs)

    out_tri, S_tri = e88_triton(*tri_inputs, linear_state=True)
    out_ref, S_ref, _ = e88_torch_reference(*ref_inputs, linear_state=True)

    torch.testing.assert_close(out_tri, out_ref, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(S_tri, S_ref, rtol=3e-4, atol=3e-4)

    loss_tri = out_tri.float().square().mean() + 0.3 * S_tri.float().square().mean()
    loss_ref = out_ref.float().square().mean() + 0.3 * S_ref.float().square().mean()

    grads_tri = torch.autograd.grad(loss_tri, tri_inputs)
    grads_ref = torch.autograd.grad(loss_ref, ref_inputs)

    for grad_tri, grad_ref in zip(grads_tri, grads_ref):
        torch.testing.assert_close(grad_tri, grad_ref, rtol=5e-3, atol=2e-3)


def test_e97_split_edit_triton_matches_reference():
    tri_base = _inputs(seed=5, requires_grad=True)
    ref_base = tuple(t.detach().clone().requires_grad_(True) for t in tri_base)
    S0, k, v, q, decay = tri_base
    S0_ref, k_ref, v_ref, q_ref, decay_ref = ref_base

    torch.manual_seed(55)
    gate = (0.20 * torch.randn_like(v)).requires_grad_(True)
    erase_gate = torch.sigmoid(0.20 * torch.randn_like(k)).detach().requires_grad_(True)
    value_write_gate = torch.sigmoid(0.20 * torch.randn_like(v)).detach().requires_grad_(True)

    gate_ref = gate.detach().clone().requires_grad_(True)
    erase_ref = erase_gate.detach().clone().requires_grad_(True)
    value_write_ref = value_write_gate.detach().clone().requires_grad_(True)

    out_tri, S_tri = e88_triton(
        S0, k, v, q, decay,
        g=gate,
        normalize_kq=True,
        apply_silu_qkv=True,
        erase_gate=erase_gate,
        value_write_gate=value_write_gate,
    )

    k_ref_base = F.silu(k_ref)
    q_ref_base = F.silu(q_ref)
    v_ref_base = F.silu(v_ref)
    out_ref, S_ref, _ = e88_torch_reference(
        S0_ref,
        _l2(k_ref_base),
        v_ref_base,
        _l2(q_ref_base),
        decay_ref,
        erase_gate=erase_ref,
        value_write_gate=value_write_ref,
    )
    out_ref = F.silu(gate_ref) * out_ref

    torch.testing.assert_close(out_tri, out_ref, rtol=4e-4, atol=4e-4)
    torch.testing.assert_close(S_tri, S_ref, rtol=4e-4, atol=4e-4)

    loss_tri = out_tri.float().square().mean() + 0.3 * S_tri.float().square().mean()
    loss_ref = out_ref.float().square().mean() + 0.3 * S_ref.float().square().mean()

    grads_tri = torch.autograd.grad(
        loss_tri,
        tri_base + (gate, erase_gate, value_write_gate),
    )
    grads_ref = torch.autograd.grad(
        loss_ref,
        ref_base + (gate_ref, erase_ref, value_write_ref),
    )

    for grad_tri, grad_ref in zip(grads_tri, grads_ref):
        torch.testing.assert_close(grad_tri, grad_ref, rtol=6e-3, atol=3e-3)


def test_e97_split_edit_linear_triton_matches_reference():
    tri_base = _inputs(seed=7, requires_grad=True)
    ref_base = tuple(t.detach().clone().requires_grad_(True) for t in tri_base)
    S0, k, v, q, decay = tri_base
    S0_ref, k_ref, v_ref, q_ref, decay_ref = ref_base

    torch.manual_seed(77)
    gate = (0.20 * torch.randn_like(v)).requires_grad_(True)
    erase_gate = torch.sigmoid(0.20 * torch.randn_like(k)).detach().requires_grad_(True)
    value_write_gate = torch.sigmoid(0.20 * torch.randn_like(v)).detach().requires_grad_(True)

    gate_ref = gate.detach().clone().requires_grad_(True)
    erase_ref = erase_gate.detach().clone().requires_grad_(True)
    value_write_ref = value_write_gate.detach().clone().requires_grad_(True)

    out_tri, S_tri = e88_triton(
        S0, k, v, q, decay,
        g=gate,
        normalize_kq=True,
        apply_silu_qkv=True,
        erase_gate=erase_gate,
        value_write_gate=value_write_gate,
        linear_state=True,
    )

    k_ref_base = F.silu(k_ref)
    q_ref_base = F.silu(q_ref)
    v_ref_base = F.silu(v_ref)
    out_ref, S_ref, _ = e88_torch_reference(
        S0_ref,
        _l2(k_ref_base),
        v_ref_base,
        _l2(q_ref_base),
        decay_ref,
        linear_state=True,
        erase_gate=erase_ref,
        value_write_gate=value_write_ref,
    )
    out_ref = F.silu(gate_ref) * out_ref

    torch.testing.assert_close(out_tri, out_ref, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(S_tri, S_ref, rtol=5e-4, atol=5e-4)

    loss_tri = out_tri.float().square().mean() + 0.3 * S_tri.float().square().mean()
    loss_ref = out_ref.float().square().mean() + 0.3 * S_ref.float().square().mean()

    grads_tri = torch.autograd.grad(
        loss_tri,
        tri_base + (gate, erase_gate, value_write_gate),
    )
    grads_ref = torch.autograd.grad(
        loss_ref,
        ref_base + (gate_ref, erase_ref, value_write_ref),
    )

    for grad_tri, grad_ref in zip(grads_tri, grads_ref):
        torch.testing.assert_close(grad_tri, grad_ref, rtol=7e-3, atol=4e-3)
