"""ROCm execution gates for node-local E97 expert-parallel packing."""
import pytest
import torch

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
