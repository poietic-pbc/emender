"""Executable gates for the fused E97 MoE Triton path.

These tests never use the Python MoE module as the system under test.  The
small tensor expressions are assertion oracles only; every produced MoE value
comes from the real @triton.jit routing and expert kernels.
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ndm.models.triton_ops import mamba2_decay as triton_mamba2_decay
from ndm.triton.e97_moe_fused import (
    FUSED_E97_MOE_ABI,
    fused_routed_swiglu_forward,
    fused_schedulefree_adamw_update_,
    fused_shared_routed_swiglu_autograd_parity,
    fused_shared_routed_swiglu_forward,
    fused_shared_routed_swiglu_forward_backward_parity,
)


HIP = bool(torch.cuda.is_available() and torch.version.hip)
pytestmark = pytest.mark.skipif(not HIP, reason="requires ROCm/HIP Triton")


def _fixed_tensors():
    device = torch.device("cuda")
    tokens, dim, hidden, experts = 8, 64, 128, 4
    x = (torch.arange(tokens * dim, device=device, dtype=torch.float32)
         .reshape(tokens, dim).remainder(37).sub(18).div(19).to(torch.bfloat16))
    router = (torch.arange(experts * dim, device=device, dtype=torch.float32)
              .reshape(experts, dim).remainder(29).sub(14).mul_(1.0e-4))
    gate = (torch.arange(experts * hidden * dim, device=device, dtype=torch.float32)
            .reshape(experts, hidden, dim).remainder(31).sub(15).mul_(0.002)
            .to(torch.bfloat16).contiguous())
    up = (torch.arange(experts * hidden * dim, device=device, dtype=torch.float32)
          .reshape(experts, hidden, dim).remainder(23).sub(11).mul_(0.002)
          .to(torch.bfloat16).contiguous())
    down = (torch.arange(experts * dim * hidden, device=device, dtype=torch.float32)
            .reshape(experts, dim, hidden).remainder(19).sub(9).mul_(0.002)
            .to(torch.bfloat16).contiguous())
    return x, router, gate, up, down


def test_fused_source_has_no_eager_hot_path_and_all_required_forward_kernels():
    source = (Path(__file__).parents[1] / "ndm/triton/e97_moe_fused.py").read_text()
    assert source.count("@triton.jit") >= 8
    for kernel in (
        "_router_fp32_kernel", "_top3_softmax_metrics_kernel",
        "_padded_prefix_kernel", "_assign_packed_rows_kernel",
        "_pack_tokens_kernel", "_grouped_gate_up_silu_kernel",
        "_dense_gate_up_silu_kernel", "_grouped_down_kernel",
        "_dense_down_kernel", "_combine_top3_kernel",
        "_combine_shared_top3_kernel", "_combine_backward_kernel",
        "_grouped_down_input_backward_kernel",
        "_grouped_down_weight_backward_kernel", "_silu_backward_kernel",
        "_grouped_gate_up_input_backward_kernel",
        "_grouped_gate_up_weight_backward_kernel",
        "_top3_router_backward_kernel", "_schedulefree_adamw_update_kernel",
    ):
        assert f"def {kernel}(" in source
    assert "EMENDER_E97_MOE_ALLOW_EAGER" in source


def test_e97_decay_backward_preserves_dt_bias_gradient():
    alpha = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    a_log = torch.randn(16, device="cuda", dtype=torch.float32).mul(0.1).requires_grad_(True)
    dt_bias = torch.randn(16, device="cuda", dtype=torch.float32).mul(0.1).requires_grad_(True)
    grad = torch.randn_like(alpha)
    decay = triton_mamba2_decay(alpha, a_log, dt_bias)
    actual = torch.autograd.grad(decay, (alpha, a_log, dt_bias), grad)

    oa = alpha.float().detach().requires_grad_(True)
    olog = a_log.detach().requires_grad_(True)
    odt = dt_bias.detach().requires_grad_(True)
    expected_decay = torch.exp(-torch.exp(olog) * F.softplus(oa + odt))
    expected = torch.autograd.grad(expected_decay, (oa, olog, odt), grad.float())
    assert actual[2] is not None
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=2e-2, atol=3e-3)


def test_real_fused_triton_forward_matches_assertion_oracle():
    x, router, gate, up, down = _fixed_tensors()
    result = fused_routed_swiglu_forward(x, router, gate, up, down)

    logits = x.float() @ router.T
    indices = logits.topk(3, dim=-1).indices
    weights = logits.gather(1, indices).softmax(dim=-1)
    expected = torch.zeros_like(x, dtype=torch.float32)
    # Oracle only. This loop is not reachable from the fused implementation.
    for token in range(x.shape[0]):
        for route in range(3):
            expert = indices[token, route]
            activation = F.silu(x[token].float() @ gate[expert].float().T)
            activation *= x[token].float() @ up[expert].float().T
            expected[token] += weights[token, route] * (activation @ down[expert].float().T)

    assert result.kernel_abi == FUSED_E97_MOE_ABI
    assert torch.equal(result.top_indices.long(), indices)
    torch.testing.assert_close(result.top_weights, weights, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(result.output.float(), expected, rtol=8e-3, atol=5e-5)
    torch.testing.assert_close(
        result.top_weights.sum(dim=-1), torch.ones(x.shape[0], device=x.device),
        rtol=0, atol=1e-6)
    assert int(result.expert_counts.sum()) == x.shape[0] * 3
    assert torch.isfinite(result.router_z_per_token).all()
    assert torch.isfinite(result.router_entropy_per_token).all()
    assert torch.isfinite(result.router_max_probability_per_token).all()
    counts = torch.bincount(indices.reshape(-1), minlength=router.shape[0]).float()
    probabilities = logits.softmax(dim=-1)
    expected_balance = router.shape[0] * torch.sum(
        counts / (x.shape[0] * 3) * probabilities.mean(dim=0))
    expected_z = torch.logsumexp(logits, dim=-1).square().mean()
    torch.testing.assert_close(result.load_balance_loss, expected_balance, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(result.z_loss, expected_z, rtol=2e-6, atol=2e-7)


def test_real_fused_shared_plus_routed_forward_matches_assertion_oracle():
    x, router, gate, up, down = _fixed_tensors()
    shared_gate = gate[0].contiguous()
    shared_up = up[1].contiguous()
    shared_down = down[2].contiguous()
    result = fused_routed_swiglu_forward(
        x, router, gate, up, down,
        shared_gate_weight=shared_gate,
        shared_up_weight=shared_up,
        shared_down_weight=shared_down)

    logits = x.float() @ router.T
    indices = logits.topk(3, dim=-1).indices
    weights = logits.gather(1, indices).softmax(dim=-1)
    expected = F.silu(x.float() @ shared_gate.float().T)
    expected *= x.float() @ shared_up.float().T
    expected = expected @ shared_down.float().T
    for token in range(x.shape[0]):
        for route in range(3):
            expert = indices[token, route]
            activation = F.silu(x[token].float() @ gate[expert].float().T)
            activation *= x[token].float() @ up[expert].float().T
            expected[token] += weights[token, route] * (activation @ down[expert].float().T)

    torch.testing.assert_close(result.output.float(), expected, rtol=8e-3, atol=7e-5)
    assert result.kernel_abi == FUSED_E97_MOE_ABI


def test_real_fused_shared_routed_output_backward_matches_autograd_oracle():
    x, router, gate, up, down = _fixed_tensors()
    shared_gate = gate[0].contiguous()
    shared_up = up[1].contiguous()
    shared_down = down[2].contiguous()
    grad_output = (torch.arange(x.numel(), device=x.device, dtype=torch.float32)
                   .reshape_as(x).remainder(17).sub(8).div(11).to(torch.bfloat16))
    grad_aux = torch.tensor([0.07, 0.03], device=x.device, dtype=torch.float32)
    _, actual = fused_shared_routed_swiglu_forward_backward_parity(
        x, router, gate, up, down,
        shared_gate, shared_up, shared_down, grad_output, grad_aux)

    oracle_tensors = [tensor.float().detach().requires_grad_(True) for tensor in (
        x, router, gate, up, down, shared_gate, shared_up, shared_down)]
    ox, orouter, ogate, oup, odown, osg, osu, osd = oracle_tensors
    logits = ox @ orouter.T
    indices = logits.topk(3, dim=-1).indices
    weights = logits.gather(1, indices).softmax(dim=-1)
    shared_activation = F.silu(ox @ osg.T) * (ox @ osu.T)
    expected_rows = []
    for token in range(ox.shape[0]):
        routed = torch.zeros_like(ox[token])
        for route in range(3):
            expert = indices[token, route]
            activation = F.silu(ox[token] @ ogate[expert].T)
            activation = activation * (ox[token] @ oup[expert].T)
            routed = routed + weights[token, route] * (activation @ odown[expert].T)
        expected_rows.append(shared_activation[token] @ osd.T + routed)
    expected = torch.stack(expected_rows)
    counts = torch.bincount(indices.reshape(-1), minlength=orouter.shape[0]).float()
    probabilities = logits.softmax(dim=-1)
    balance = orouter.shape[0] * torch.sum(
        counts / (ox.shape[0] * 3) * probabilities.mean(dim=0))
    z_loss = torch.logsumexp(logits, dim=-1).square().mean()
    objective = ((expected * grad_output.float()).sum()
                 + grad_aux[0] * balance + grad_aux[1] * z_loss)
    oracle = torch.autograd.grad(objective, oracle_tensors)

    for fused_grad, expected_grad in zip(actual, oracle):
        torch.testing.assert_close(
            fused_grad.float(), expected_grad.float(), rtol=3e-2, atol=2e-4)


def test_custom_autograd_executes_fused_output_and_auxiliary_backward():
    x, router, gate, up, down = _fixed_tensors()
    tensors = [tensor.detach().clone().requires_grad_(True) for tensor in (
        x, router, gate, up, down,
        gate[0].contiguous(), up[1].contiguous(), down[2].contiguous())]
    output, balance, z_loss = fused_shared_routed_swiglu_autograd_parity(*tensors)
    objective = output.float().square().mean() + 0.07 * balance + 0.03 * z_loss
    objective.backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_fused_schedulefree_adamw_update_matches_assertion_oracle_without_master_copy():
    parameter = (torch.arange(513, device="cuda", dtype=torch.float32)
                 .remainder(23).sub(11).div(13).to(torch.bfloat16))
    grad = (torch.arange(513, device="cuda", dtype=torch.float32)
            .remainder(19).sub(9).div(17).to(torch.bfloat16))
    z = (parameter.float() * 0.9 + 0.03).to(torch.bfloat16)
    variance = (grad.float().square() * 0.2 + 0.01).to(torch.bfloat16)
    y0, z0, v0 = parameter.float(), z.float(), variance.float()
    lr, beta1, beta2 = 2.5e-3, 0.9, 0.999
    correction, eps, decay, ckp1 = 1.0 - beta2**7, 1.0e-8, 0.1, 0.17
    expected_v = beta2 * v0 + (1.0 - beta2) * grad.float().square()
    normalized = grad.float() / (expected_v / correction).sqrt().add(eps)
    normalized = normalized + decay * y0
    expected_y = y0 + ckp1 * (z0 - y0)
    expected_y = expected_y + lr * (beta1 * (1.0 - ckp1) - 1.0) * normalized
    expected_z = z0 - lr * normalized

    fused_schedulefree_adamw_update_(
        parameter, grad, z, variance, lr=lr, beta1=beta1, beta2=beta2,
        bias_correction2=correction, eps=eps, weight_decay=decay, ckp1=ckp1)
    assert parameter.dtype == z.dtype == variance.dtype == torch.bfloat16
    torch.testing.assert_close(parameter.float(), expected_y.to(torch.bfloat16).float(), rtol=0, atol=0)
    torch.testing.assert_close(z.float(), expected_z.to(torch.bfloat16).float(), rtol=0, atol=0)
    torch.testing.assert_close(variance.float(), expected_v.to(torch.bfloat16).float(), rtol=0, atol=0)


def test_training_and_non_gpu_paths_fail_closed():
    x, router, gate, up, down = _fixed_tensors()
    with pytest.raises(ValueError, match="exactly 64 routed experts"):
        fused_shared_routed_swiglu_forward(
            x, router, gate, up, down,
            gate[0].contiguous(), up[0].contiguous(), down[0].contiguous())
    with pytest.raises(RuntimeError, match="backward/optimizer gate is not complete"):
        fused_routed_swiglu_forward(
            x, router, gate, up, down,
            shared_gate_weight=gate[0].contiguous(),
            shared_up_weight=up[0].contiguous(),
            shared_down_weight=down[0].contiguous(),
            require_training=True)
    with pytest.raises(RuntimeError, match="requires a GPU tensor"):
        fused_routed_swiglu_forward(
            x.cpu(), router.cpu(), gate.cpu(), up.cpu(), down.cpu())
