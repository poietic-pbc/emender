import pytest
import torch

from ndm.e97_moe_optimizer import FusedScheduleFreeAdamW


HIP = bool(torch.cuda.is_available() and torch.version.hip)
pytestmark = pytest.mark.skipif(not HIP, reason="requires ROCm/HIP Triton")


def test_fused_schedulefree_trajectory_modes_and_no_master_weights():
    bf16 = torch.nn.Parameter(
        torch.linspace(-1, 1, 513, device="cuda", dtype=torch.bfloat16))
    fp32 = torch.nn.Parameter(
        torch.linspace(-0.5, 0.5, 257, device="cuda", dtype=torch.float32))
    optimizer = FusedScheduleFreeAdamW(
        [bf16, fp32], lr=2.5e-3, betas=(0.9, 0.999),
        weight_decay=0.1, warmup_steps=2)
    with pytest.raises(RuntimeError, match="call fused ScheduleFree train"):
        optimizer.step()
    optimizer.train()
    for step in range(3):
        bf16.grad = torch.full_like(bf16, 0.02 * (step + 1))
        fp32.grad = torch.full_like(fp32, -0.03 * (step + 1))
        optimizer.step()
    optimizer.assert_no_master_weights()
    optimizer.offload_z_()
    for parameter in (bf16, fp32):
        assert optimizer.state[parameter]["z"].device.type == "cpu"
        assert optimizer.state[parameter]["z"].is_pinned()
    bf16.grad = torch.full_like(bf16, 0.01)
    fp32.grad = torch.full_like(fp32, -0.01)
    optimizer.step()
    for parameter in (bf16, fp32):
        assert optimizer.state[parameter]["z"].device.type == "cpu"
    optimizer.offload_state_()
    for parameter in (bf16, fp32):
        assert optimizer.state[parameter]["exp_avg_sq"].device.type == "cpu"
    bf16.grad = torch.full_like(bf16, 0.005)
    fp32.grad = torch.full_like(fp32, -0.005)
    optimizer.step()
    for parameter in (bf16, fp32):
        assert optimizer.state[parameter]["z"].device.type == "cpu"
        assert optimizer.state[parameter]["exp_avg_sq"].device.type == "cpu"
    optimizer.assert_no_master_weights()
    for parameter in (bf16, fp32):
        assert optimizer.state[parameter]["z"].dtype == parameter.dtype
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == parameter.dtype
        assert torch.isfinite(parameter).all()
    train_bf16 = bf16.detach().clone()
    train_fp32 = fp32.detach().clone()
    optimizer.eval()
    assert not torch.equal(bf16, train_bf16)
    assert not torch.equal(fp32, train_fp32)
    optimizer.train()
    torch.testing.assert_close(bf16, train_bf16, rtol=0, atol=2 ** -7)
    torch.testing.assert_close(fp32, train_fp32, rtol=2e-6, atol=2e-7)
