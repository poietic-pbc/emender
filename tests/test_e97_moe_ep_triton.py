"""ROCm execution gates for node-local E97 expert-parallel packing."""
import pytest
import torch
import torch.nn.functional as F

from ndm.triton.e97_moe_fused import (
    checkpointed_packed_local_experts_rocblas,
    fused_packed_local_experts_autograd,
    fused_shared_expert_autograd,
    fused_top3_router_autograd,
)
from ndm.triton.e97_moe_ep import (
    EP_SIZE,
    EXPERTS_PER_RANK,
    build_ep_send_plan,
    build_local_expert_plan,
    unpack_local_expert_rows,
)


HIP = bool(torch.cuda.is_available() and torch.version.hip)
pytestmark = pytest.mark.skipif(not HIP, reason="requires ROCm/HIP Triton")


def test_bounded_ep_send_pack_is_complete_and_destination_contiguous():
    tokens, dim = 19, 64
    x = (torch.arange(tokens * dim, device="cuda", dtype=torch.float32)
         .reshape(tokens, dim).remainder(31).to(torch.bfloat16).contiguous())
    top = torch.empty((tokens, 3), device="cuda", dtype=torch.int32)
    expected = []
    for token in range(tokens):
        values = [(token * 11 + slot * 19) % 64 for slot in range(3)]
        top[token] = torch.tensor(values, device="cuda", dtype=torch.int32)
        expected.extend(values)

    plan = build_ep_send_plan(x, top)
    counts = plan.send_counts.cpu().tolist()
    offsets = plan.send_offsets.cpu().tolist()
    assert sum(counts) == tokens * 3
    assert offsets[0] == 0 and offsets[-1] == tokens * 3
    assert offsets[1:] == [sum(counts[:i + 1]) for i in range(EP_SIZE)]

    inverse = plan.assignment_to_send_row.cpu().tolist()
    send_local = plan.send_local_expert.cpu().tolist()
    for assignment, expert in enumerate(expected):
        row = inverse[assignment]
        owner = expert // EXPERTS_PER_RANK
        assert offsets[owner] <= row < offsets[owner + 1]
        assert send_local[row] == expert % EXPERTS_PER_RANK
        torch.testing.assert_close(plan.send_x[row], x[assignment // 3], rtol=0, atol=0)


def test_received_rows_repack_by_local_expert_and_unpack_exactly():
    rows, dim = 37, 64
    received = (torch.arange(rows * dim, device="cuda", dtype=torch.float32)
                .reshape(rows, dim).remainder(43).to(torch.bfloat16).contiguous())
    local = (torch.arange(rows, device="cuda", dtype=torch.int32)
             .mul(5).remainder(EXPERTS_PER_RANK).contiguous())
    plan = build_local_expert_plan(received, local)
    offsets = plan.expert_offsets.cpu().tolist()
    inverse = plan.received_to_packed_row.cpu().tolist()
    local_cpu = local.cpu().tolist()
    for row, expert in enumerate(local_cpu):
        assert offsets[expert] <= inverse[row] < offsets[expert + 1]
        torch.testing.assert_close(plan.packed_x[inverse[row]], received[row], rtol=0, atol=0)

    transformed = plan.packed_x + torch.tensor(2, device="cuda", dtype=torch.bfloat16)
    unpacked = unpack_local_expert_rows(transformed, plan)
    torch.testing.assert_close(unpacked, received + 2, rtol=0, atol=0)


def test_send_and_local_permutations_preserve_autograd_exactly():
    tokens, dim = 11, 64
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    top = torch.tensor(
        [[(token * 7 + slot * 17) % 64 for slot in range(3)]
         for token in range(tokens)], device="cuda", dtype=torch.int32)
    send = build_ep_send_plan(x, top)
    send.send_x.float().sum().backward()
    torch.testing.assert_close(x.grad.float(), torch.full_like(x.float(), 3), rtol=0, atol=0)

    received = torch.randn(29, dim, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    local = torch.arange(29, device="cuda", dtype=torch.int32).remainder(8)
    plan = build_local_expert_plan(received, local)
    unpacked = unpack_local_expert_rows(plan.packed_x, plan)
    grad = torch.randn_like(unpacked)
    unpacked.backward(grad)
    torch.testing.assert_close(unpacked, received, rtol=0, atol=0)
    torch.testing.assert_close(received.grad, grad, rtol=0, atol=0)


def test_production_router_and_shared_expert_custom_autograd_match_oracle():
    tokens, dim, hidden = 9, 64, 128
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    router = torch.randn(64, dim, device="cuda", dtype=torch.float32).mul(1e-3).requires_grad_(True)
    gate = torch.randn(hidden, dim, device="cuda", dtype=torch.bfloat16).mul(0.01).requires_grad_(True)
    up = torch.randn(hidden, dim, device="cuda", dtype=torch.bfloat16).mul(0.01).requires_grad_(True)
    down = torch.randn(dim, hidden, device="cuda", dtype=torch.bfloat16).mul(0.01).requires_grad_(True)
    indices, weights, counts, balance, z_loss = fused_top3_router_autograd(x, router)
    shared = fused_shared_expert_autograd(x, gate, up, down)
    grad_output = torch.randn_like(shared)
    grad_weights = torch.randn_like(weights)
    objective = ((shared * grad_output).float().sum() + (weights * grad_weights).sum()
                 + 0.07 * balance + 0.03 * z_loss)
    actual = torch.autograd.grad(objective, (x, router, gate, up, down))

    ox, orouter, ogate, oup, odown = [
        tensor.float().detach().requires_grad_(True)
        for tensor in (x, router, gate, up, down)]
    logits = ox @ orouter.T
    expected_indices = logits.topk(3, dim=-1).indices
    expected_weights = logits.gather(1, expected_indices).softmax(dim=-1)
    expected_counts = torch.bincount(expected_indices.reshape(-1), minlength=64).float()
    probabilities = logits.softmax(dim=-1)
    expected_balance = 64 * torch.sum(expected_counts / (tokens * 3) * probabilities.mean(0))
    expected_z = torch.logsumexp(logits, -1).square().mean()
    activation = F.silu(ox @ ogate.T) * (ox @ oup.T)
    expected_shared = activation @ odown.T
    expected_objective = ((expected_shared * grad_output.float()).sum()
                          + (expected_weights * grad_weights).sum()
                          + 0.07 * expected_balance + 0.03 * expected_z)
    expected = torch.autograd.grad(expected_objective, (ox, orouter, ogate, oup, odown))
    assert torch.equal(indices.long(), expected_indices)
    assert torch.equal(counts.long(), expected_counts.long())
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-7)
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad.float(), expected_grad, rtol=3e-2, atol=3e-4)


def test_fused_eight_local_experts_forward_and_backward_match_oracle():
    rows, dim, hidden = 23, 64, 128
    received = (torch.arange(rows * dim, device="cuda", dtype=torch.float32)
                .reshape(rows, dim).remainder(29).sub(14).div(17)
                .to(torch.bfloat16).contiguous())
    local = (torch.arange(rows, device="cuda", dtype=torch.int32)
             .mul(3).remainder(8).contiguous())
    plan = build_local_expert_plan(received, local)
    packed = plan.packed_x.detach().requires_grad_(True)
    gate = (torch.arange(8 * hidden * dim, device="cuda", dtype=torch.float32)
            .reshape(8, hidden, dim).remainder(31).sub(15).mul(0.002)
            .to(torch.bfloat16).contiguous().requires_grad_(True))
    up = (torch.arange(8 * hidden * dim, device="cuda", dtype=torch.float32)
          .reshape(8, hidden, dim).remainder(23).sub(11).mul(0.002)
          .to(torch.bfloat16).contiguous().requires_grad_(True))
    down = (torch.arange(8 * dim * hidden, device="cuda", dtype=torch.float32)
            .reshape(8, dim, hidden).remainder(19).sub(9).mul(0.002)
            .to(torch.bfloat16).contiguous().requires_grad_(True))
    packed_output = fused_packed_local_experts_autograd(
        packed, plan.expert_offsets, gate, up, down)
    actual = unpack_local_expert_rows(packed_output, plan)
    rocblas_output = checkpointed_packed_local_experts_rocblas(
        packed, plan.expert_offsets, gate, up, down)
    real_packed_rows = plan.received_to_packed_row.long()
    torch.testing.assert_close(
        rocblas_output[real_packed_rows].float(), packed_output[real_packed_rows].float(),
        rtol=8e-3, atol=7e-5)

    oreceived = received.float().detach().requires_grad_(True)
    ogate = gate.float().detach().requires_grad_(True)
    oup = up.float().detach().requires_grad_(True)
    odown = down.float().detach().requires_grad_(True)
    expected_rows = []
    for row, expert in enumerate(local.cpu().tolist()):
        activation = F.silu(oreceived[row] @ ogate[expert].T)
        activation = activation * (oreceived[row] @ oup[expert].T)
        expected_rows.append(activation @ odown[expert].T)
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=7e-5)

    grad_rows = (torch.arange(rows * dim, device="cuda", dtype=torch.float32)
                 .reshape(rows, dim).remainder(17).sub(8).div(13))
    grad_packed = torch.zeros_like(packed_output)
    inverse = plan.received_to_packed_row.long()
    grad_packed[inverse] = grad_rows.to(torch.bfloat16)
    packed_grads = torch.autograd.grad(
        packed_output, (packed, gate, up, down), grad_outputs=grad_packed)
    rocblas_grads = torch.autograd.grad(
        rocblas_output, (packed, gate, up, down), grad_outputs=grad_packed)
    for triton_grad, rocblas_grad in zip(packed_grads, rocblas_grads):
        torch.testing.assert_close(
            triton_grad.float(), rocblas_grad.float(), rtol=3e-2, atol=3e-4)
    oracle_grads = torch.autograd.grad(
        (expected * grad_rows).sum(), (oreceived, ogate, oup, odown))
    torch.testing.assert_close(
        packed_grads[0][inverse].float(), oracle_grads[0], rtol=3e-2, atol=3e-4)
    for actual_grad, expected_grad in zip(packed_grads[1:], oracle_grads[1:]):
        torch.testing.assert_close(
            actual_grad.float(), expected_grad, rtol=3e-2, atol=3e-4)
