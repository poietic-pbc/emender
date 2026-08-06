"""Executable gates for the fused E97 MoE Triton path.

These tests never use the Python MoE module as the system under test.  The
small tensor expressions are assertion oracles only; every produced MoE value
comes from the real @triton.jit routing and expert kernels.
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ndm.triton.e97_moe_fused import (
    FUSED_E97_MOE_ABI,
    fused_routed_swiglu_forward,
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
        "_pack_tokens_kernel", "_grouped_gate_up_kernel",
        "_silu_mul_kernel", "_grouped_down_kernel", "_combine_top3_kernel",
    ):
        assert f"def {kernel}(" in source
    assert "EMENDER_E97_MOE_ALLOW_EAGER" in source


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


def test_training_and_non_gpu_paths_fail_closed():
    x, router, gate, up, down = _fixed_tensors()
    with pytest.raises(RuntimeError, match="backward/optimizer gate is not complete"):
        fused_routed_swiglu_forward(
            x, router, gate, up, down, require_training=True)
    with pytest.raises(RuntimeError, match="requires a GPU tensor"):
        fused_routed_swiglu_forward(
            x.cpu(), router.cpu(), gate.cpu(), up.cpu(), down.cpu())
