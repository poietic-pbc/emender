"""Node-local expert-parallel routing primitives for the E97 MoE.

Triton owns assignment, packing, local-expert repacking, and unpacking. RCCL is
the only admissible transport and is orchestrated in :mod:`ndm.e97_moe_ep`.
These primitives allocate exactly one row per routed assignment plus bounded
local-expert GEMM padding; they never allocate a per-destination capacity.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .e97_moe_fused import _pack_tokens_kernel, _padded_prefix_kernel, _require_hip_triton


EP_SIZE = 8
ROUTED_EXPERTS = 64
EXPERTS_PER_RANK = ROUTED_EXPERTS // EP_SIZE
TOP_K = 3


@triton.jit
def _ep_rank_counts_kernel(TOP_INDEX, RANK_COUNTS, ASSIGNMENTS: tl.constexpr,
                           EXPERTS_PER_RANK: tl.constexpr):
    assignment = tl.program_id(0)
    if assignment < ASSIGNMENTS:
        expert = tl.load(TOP_INDEX + assignment)
        tl.atomic_add(RANK_COUNTS + expert // EXPERTS_PER_RANK, 1)


@triton.jit
def _ep_prefix8_kernel(COUNTS, OFFSETS, CURSOR, EP: tl.constexpr):
    rank = tl.arange(0, EP)
    counts = tl.load(COUNTS + rank)
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(OFFSETS + rank, inclusive - counts)
    tl.store(OFFSETS + EP, tl.sum(counts, axis=0))
    tl.store(CURSOR + rank, 0)


@triton.jit
def _ep_assign_send_rows_kernel(
    TOP_INDEX, OFFSETS, CURSOR, INVERSE, SEND_LOCAL_EXPERT,
    ASSIGNMENTS: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
):
    assignment = tl.program_id(0)
    if assignment < ASSIGNMENTS:
        expert = tl.load(TOP_INDEX + assignment)
        owner = expert // EXPERTS_PER_RANK
        slot = tl.atomic_add(CURSOR + owner, 1)
        row = tl.load(OFFSETS + owner) + slot
        tl.store(INVERSE + assignment, row)
        tl.store(SEND_LOCAL_EXPERT + row, expert % EXPERTS_PER_RANK)


@triton.jit
def _local_expert_counts_kernel(LOCAL_EXPERT, COUNTS, ROWS: tl.constexpr):
    row = tl.program_id(0)
    if row < ROWS:
        expert = tl.load(LOCAL_EXPERT + row)
        tl.atomic_add(COUNTS + expert, 1)


@triton.jit
def _assign_local_packed_rows_kernel(
    LOCAL_EXPERT, OFFSETS, CURSOR, INVERSE, ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    if row < ROWS:
        expert = tl.load(LOCAL_EXPERT + row)
        slot = tl.atomic_add(CURSOR + expert, 1)
        packed_row = tl.load(OFFSETS + expert) + slot
        tl.store(INVERSE + row, packed_row)


@triton.jit
def _repack_rows_kernel(X, INVERSE, PACKED, ROWS: tl.constexpr,
                        D: tl.constexpr, BLOCK_D: tl.constexpr):
    row, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    packed_row = tl.load(INVERSE + row)
    value = tl.load(X + row * D + d, mask=d < D, other=0.0)
    tl.store(PACKED + packed_row * D + d, value, mask=d < D)


@triton.jit
def _unpack_rows_kernel(PACKED, INVERSE, OUT, ROWS: tl.constexpr,
                        D: tl.constexpr, BLOCK_D: tl.constexpr):
    row, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    packed_row = tl.load(INVERSE + row)
    value = tl.load(PACKED + packed_row * D + d, mask=d < D, other=0.0)
    tl.store(OUT + row * D + d, value, mask=d < D)


@dataclass(frozen=True)
class EPSendPlan:
    send_x: torch.Tensor
    send_local_expert: torch.Tensor
    send_counts: torch.Tensor
    send_offsets: torch.Tensor
    assignment_to_send_row: torch.Tensor


@dataclass(frozen=True)
class EPLocalPlan:
    packed_x: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    received_to_packed_row: torch.Tensor


def build_ep_send_plan(x: torch.Tensor, top_indices: torch.Tensor) -> EPSendPlan:
    """Pack top-3 assignments into eight contiguous destination-rank spans."""
    _require_hip_triton(x)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("x must be contiguous BF16 [tokens, dim]")
    if (top_indices.shape != (x.shape[0], TOP_K) or
            top_indices.dtype != torch.int32 or not top_indices.is_contiguous()):
        raise ValueError("top_indices must be contiguous int32 [tokens, 3]")
    assignments = x.shape[0] * TOP_K
    counts = torch.zeros(EP_SIZE, device=x.device, dtype=torch.int32)
    _ep_rank_counts_kernel[(assignments,)](
        top_indices, counts, assignments, EXPERTS_PER_RANK, num_warps=1)
    offsets = torch.empty(EP_SIZE + 1, device=x.device, dtype=torch.int32)
    cursor = torch.empty(EP_SIZE, device=x.device, dtype=torch.int32)
    _ep_prefix8_kernel[(1,)](counts, offsets, cursor, EP_SIZE, num_warps=1)
    inverse = torch.empty(assignments, device=x.device, dtype=torch.int32)
    local_expert = torch.empty(assignments, device=x.device, dtype=torch.int32)
    _ep_assign_send_rows_kernel[(assignments,)](
        top_indices, offsets, cursor, inverse, local_expert,
        assignments, EXPERTS_PER_RANK, num_warps=1)
    send_x = torch.empty((assignments, x.shape[1]), device=x.device, dtype=x.dtype)
    _pack_tokens_kernel[(assignments, triton.cdiv(x.shape[1], 256))](
        x, inverse, send_x, x.shape[0], x.shape[1], BLOCK_D=256, num_warps=4)
    return EPSendPlan(send_x, local_expert, counts, offsets, inverse)


def build_local_expert_plan(received_x: torch.Tensor,
                            received_local_expert: torch.Tensor) -> EPLocalPlan:
    """Repack received rows into eight padded local-expert spans."""
    _require_hip_triton(received_x)
    if (received_x.ndim != 2 or received_x.dtype != torch.bfloat16 or
            not received_x.is_contiguous()):
        raise ValueError("received_x must be contiguous BF16 [rows, dim]")
    rows, dim = received_x.shape
    if (received_local_expert.shape != (rows,) or
            received_local_expert.dtype != torch.int32 or
            not received_local_expert.is_contiguous()):
        raise ValueError("received_local_expert must be contiguous int32 [rows]")
    counts = torch.zeros(EXPERTS_PER_RANK, device=received_x.device, dtype=torch.int32)
    if rows:
        _local_expert_counts_kernel[(rows,)](
            received_local_expert, counts, rows, num_warps=1)
    offsets = torch.empty(EXPERTS_PER_RANK + 1, device=received_x.device, dtype=torch.int32)
    cursor = torch.empty(EXPERTS_PER_RANK, device=received_x.device, dtype=torch.int32)
    total = torch.empty((), device=received_x.device, dtype=torch.int32)
    _padded_prefix_kernel[(1,)](
        counts, offsets, cursor, total, EXPERTS_PER_RANK, BLOCK_E=8, BLOCK_M=16,
        num_warps=1)
    padded_max = rows + EXPERTS_PER_RANK * 15
    inverse = torch.empty(rows, device=received_x.device, dtype=torch.int32)
    if rows:
        _assign_local_packed_rows_kernel[(rows,)](
            received_local_expert, offsets, cursor, inverse, rows, num_warps=1)
    packed = torch.zeros((padded_max, dim), device=received_x.device, dtype=received_x.dtype)
    if rows:
        _repack_rows_kernel[(rows, triton.cdiv(dim, 256))](
            received_x, inverse, packed, rows, dim, BLOCK_D=256, num_warps=4)
    return EPLocalPlan(packed, counts, offsets, inverse)


def unpack_local_expert_rows(packed_output: torch.Tensor,
                             plan: EPLocalPlan) -> torch.Tensor:
    rows = plan.received_to_packed_row.numel()
    output = torch.empty((rows, packed_output.shape[1]), device=packed_output.device,
                         dtype=packed_output.dtype)
    if rows:
        _unpack_rows_kernel[(rows, triton.cdiv(packed_output.shape[1], 256))](
            packed_output, plan.received_to_packed_row, output,
            rows, packed_output.shape[1], BLOCK_D=256, num_warps=4)
    return output
