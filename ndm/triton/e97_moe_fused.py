"""Fail-closed fused Triton kernels for the E97 MoE data path.

This file is the only admissible compute path for E97 MoE experiments.  Python
allocates bounded buffers and launches kernels; routing, packing, expert
SwiGLU, and deterministic top-k combination execute in ``@triton.jit``
kernels.  There is deliberately no eager fallback.

The current public entry point is forward-only while the fused backward and
optimizer kernels are completed.  ``require_training=True`` therefore fails
closed: it is impossible to mistake this implementation for a training-ready
system or submit a parity/training job prematurely.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import triton
import triton.language as tl


FUSED_E97_MOE_ABI = "emender-e97-moe-triton-v1"


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _require_hip_triton(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("E97 MoE is Triton-only and requires a GPU tensor")
    if not getattr(torch.version, "hip", None):
        raise RuntimeError("E97 MoE production path requires the Frontier ROCm/HIP stack")
    if os.environ.get("EMENDER_E97_MOE_ALLOW_EAGER", "0") != "0":
        raise RuntimeError("eager MoE fallback is forbidden")


@triton.jit
def _router_fp32_kernel(
    X, W, LOGITS,
    M: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
    stride_xm: tl.constexpr, stride_xd: tl.constexpr,
    stride_we: tl.constexpr, stride_wd: tl.constexpr,
    stride_lm: tl.constexpr, stride_le: tl.constexpr,
    BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
):
    pm, pe = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    re = pe * BE + tl.arange(0, BE)
    acc = tl.zeros((BM, BE), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(X + rm[:, None] * stride_xm + rk[None, :] * stride_xd,
                    mask=(rm[:, None] < M) & (rk[None, :] < D), other=0.0).to(tl.float32)
        b = tl.load(W + re[None, :] * stride_we + rk[:, None] * stride_wd,
                    mask=(re[None, :] < E) & (rk[:, None] < D), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(LOGITS + rm[:, None] * stride_lm + re[None, :] * stride_le,
             acc, mask=(rm[:, None] < M) & (re[None, :] < E))


@triton.jit
def _top3_softmax_metrics_kernel(
    LOGITS, TOP_INDEX, TOP_WEIGHT, COUNTS, PROB_SUM, Z, ENTROPY, MAX_PROB,
    M: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
):
    m = tl.program_id(0)
    e = tl.arange(0, BLOCK_E)
    mask = e < E
    values = tl.load(LOGITS + m * E + e, mask=mask, other=-float("inf")).to(tl.float32)
    vmax = tl.max(values, axis=0)
    expv = tl.exp(values - vmax)
    denom = tl.sum(tl.where(mask, expv, 0.0), axis=0)
    probs = expv / denom

    i0 = tl.argmax(values, axis=0)
    v1s = tl.where(e == i0, -float("inf"), values)
    i1 = tl.argmax(v1s, axis=0)
    v2s = tl.where((e == i0) | (e == i1), -float("inf"), values)
    i2 = tl.argmax(v2s, axis=0)
    a0 = tl.max(values, axis=0)
    a1 = tl.max(v1s, axis=0)
    a2 = tl.max(v2s, axis=0)
    am = tl.maximum(a0, tl.maximum(a1, a2))
    q0, q1, q2 = tl.exp(a0-am), tl.exp(a1-am), tl.exp(a2-am)
    qsum = q0 + q1 + q2
    tl.store(TOP_INDEX + m * 3 + 0, i0)
    tl.store(TOP_INDEX + m * 3 + 1, i1)
    tl.store(TOP_INDEX + m * 3 + 2, i2)
    tl.store(TOP_WEIGHT + m * 3 + 0, q0 / qsum)
    tl.store(TOP_WEIGHT + m * 3 + 1, q1 / qsum)
    tl.store(TOP_WEIGHT + m * 3 + 2, q2 / qsum)
    tl.atomic_add(COUNTS + i0, 1)
    tl.atomic_add(COUNTS + i1, 1)
    tl.atomic_add(COUNTS + i2, 1)
    tl.atomic_add(PROB_SUM + e, tl.where(mask, probs, 0.0), mask=mask)
    lse = tl.log(denom) + vmax
    tl.store(Z + m, lse * lse)
    entropy = -tl.sum(tl.where(mask, probs * tl.log(tl.maximum(probs, 1.0e-30)), 0.0), axis=0)
    tl.store(ENTROPY + m, entropy)
    tl.store(MAX_PROB + m, tl.max(tl.where(mask, probs, 0.0), axis=0))


@triton.jit
def _padded_prefix_kernel(COUNTS, OFFSETS, CURSOR, TOTAL,
                          E: tl.constexpr, BLOCK_E: tl.constexpr,
                          BLOCK_M: tl.constexpr):
    e = tl.arange(0, BLOCK_E)
    mask = e < E
    counts = tl.load(COUNTS + e, mask=mask, other=0)
    padded = ((counts + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    inclusive = tl.cumsum(padded, axis=0)
    start = inclusive - padded
    tl.store(OFFSETS + e, start, mask=mask)
    tl.store(OFFSETS + E, tl.sum(tl.where(mask, padded, 0), axis=0))
    tl.store(CURSOR + e, 0, mask=mask)
    tl.store(TOTAL, tl.sum(tl.where(mask, padded, 0), axis=0))


@triton.jit
def _assign_packed_rows_kernel(TOP_INDEX, OFFSETS, CURSOR, INVERSE,
                               ASSIGNMENTS: tl.constexpr):
    a = tl.program_id(0)
    if a < ASSIGNMENTS:
        expert = tl.load(TOP_INDEX + a)
        slot = tl.atomic_add(CURSOR + expert, 1)
        row = tl.load(OFFSETS + expert) + slot
        tl.store(INVERSE + a, row)


@triton.jit
def _pack_tokens_kernel(X, INVERSE, PACKED,
                        M: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    a, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    row = tl.load(INVERSE + a)
    token = a // 3
    value = tl.load(X + token * D + d, mask=d < D, other=0.0)
    tl.store(PACKED + row * D + d, value, mask=d < D)


@triton.jit
def _grouped_gate_up_kernel(
    X, W_GATE, W_UP, OFFSETS, GATE, UP,
    PADDED_M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BLOCK_E: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    row0 = pm * BM
    ev = tl.arange(0, BLOCK_E)
    ends = tl.load(OFFSETS + ev + 1, mask=ev < E, other=PADDED_M + 1)
    expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    rows = row0 + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = (rows >= start) & (rows < end) & (rows < PADDED_M)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * D + k[None, :],
                    mask=row_mask[:, None] & (k[None, :] < D), other=0.0)
        base = expert * HIDDEN * D
        wg = tl.load(W_GATE + base + cols[None, :] * D + k[:, None],
                     mask=(cols[None, :] < HIDDEN) & (k[:, None] < D), other=0.0)
        wu = tl.load(W_UP + base + cols[None, :] * D + k[:, None],
                     mask=(cols[None, :] < HIDDEN) & (k[:, None] < D), other=0.0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    out_mask = row_mask[:, None] & (cols[None, :] < HIDDEN)
    tl.store(GATE + rows[:, None] * HIDDEN + cols[None, :], gate, mask=out_mask)
    tl.store(UP + rows[:, None] * HIDDEN + cols[None, :], up, mask=out_mask)


@triton.jit
def _silu_mul_kernel(GATE, UP, ACT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GATE + offsets, mask=offsets < N, other=0.0).to(tl.float32)
    u = tl.load(UP + offsets, mask=offsets < N, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g)) * u
    tl.store(ACT + offsets, y, mask=offsets < N)


@triton.jit
def _grouped_down_kernel(
    ACT, W_DOWN, OFFSETS, OUT,
    PADDED_M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BLOCK_E: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    row0 = pm * BM
    ev = tl.arange(0, BLOCK_E)
    ends = tl.load(OFFSETS + ev + 1, mask=ev < E, other=PADDED_M + 1)
    expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    rows = row0 + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = (rows >= start) & (rows < end) & (rows < PADDED_M)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, HIDDEN, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(ACT + rows[:, None] * HIDDEN + k[None, :],
                    mask=row_mask[:, None] & (k[None, :] < HIDDEN), other=0.0)
        base = expert * D * HIDDEN
        w = tl.load(W_DOWN + base + cols[None, :] * HIDDEN + k[:, None],
                    mask=(cols[None, :] < D) & (k[:, None] < HIDDEN), other=0.0)
        acc += tl.dot(a, w)
    mask = row_mask[:, None] & (cols[None, :] < D)
    tl.store(OUT + rows[:, None] * D + cols[None, :], acc, mask=mask)


@triton.jit
def _combine_top3_kernel(PACKED_OUT, INVERSE, TOP_WEIGHT, OUT,
                         M: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    m, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    p0 = tl.load(INVERSE + m * 3 + 0)
    p1 = tl.load(INVERSE + m * 3 + 1)
    p2 = tl.load(INVERSE + m * 3 + 2)
    w0 = tl.load(TOP_WEIGHT + m * 3 + 0).to(tl.float32)
    w1 = tl.load(TOP_WEIGHT + m * 3 + 1).to(tl.float32)
    w2 = tl.load(TOP_WEIGHT + m * 3 + 2).to(tl.float32)
    y0 = tl.load(PACKED_OUT + p0 * D + d, mask=d < D, other=0.0).to(tl.float32)
    y1 = tl.load(PACKED_OUT + p1 * D + d, mask=d < D, other=0.0).to(tl.float32)
    y2 = tl.load(PACKED_OUT + p2 * D + d, mask=d < D, other=0.0).to(tl.float32)
    tl.store(OUT + m * D + d, w0*y0 + w1*y1 + w2*y2, mask=d < D)


@dataclass
class FusedMoEForwardResult:
    output: torch.Tensor
    top_indices: torch.Tensor
    top_weights: torch.Tensor
    expert_counts: torch.Tensor
    probability_sums: torch.Tensor
    router_z_per_token: torch.Tensor
    router_entropy_per_token: torch.Tensor
    router_max_probability_per_token: torch.Tensor
    kernel_abi: str = FUSED_E97_MOE_ABI


def fused_routed_swiglu_forward(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    require_training: bool = False,
) -> FusedMoEForwardResult:
    """Execute the fused, dropless top-3 routed expert forward.

    Shapes are ``x[M,D]``, router ``[E,D]``, gate/up ``[E,H,D]`` and down
    ``[E,D,H]``.  All expert tensors must be BF16 and contiguous.  The router
    weights/logits are FP32.  No CPU/eager fallback exists.
    """
    _require_hip_triton(x)
    if require_training:
        raise RuntimeError(
            "fused E97 MoE backward/optimizer gate is not complete; training is forbidden")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be contiguous [tokens, dim]")
    if x.dtype != torch.bfloat16:
        raise ValueError("fused E97 MoE expert input must be BF16")
    if router_weight.dtype != torch.float32:
        raise ValueError("router weights must be FP32")
    if any(w.dtype != torch.bfloat16 or not w.is_contiguous()
           for w in (gate_weight, up_weight, down_weight)):
        raise ValueError("expert weights must be contiguous BF16 tensors")
    M, D = x.shape
    E, router_d = router_weight.shape
    if router_d != D or E > 64 or E < 3:
        raise ValueError("router shape must be [3..64, dim]")
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate/up weights must match [experts, hidden, dim]")
    if gate_weight.shape[0] != E or gate_weight.shape[2] != D:
        raise ValueError("expert gate/up shape mismatch")
    HIDDEN = gate_weight.shape[1]
    if down_weight.shape != (E, D, HIDDEN):
        raise ValueError("down weights must be [experts, dim, hidden]")

    logits = torch.empty((M, E), device=x.device, dtype=torch.float32)
    _router_fp32_kernel[(triton.cdiv(M, 16), triton.cdiv(E, 16))](
        x, router_weight, logits, M, D, E,
        x.stride(0), x.stride(1), router_weight.stride(0), router_weight.stride(1),
        logits.stride(0), logits.stride(1), BM=16, BE=16, BK=32,
        num_warps=4, num_stages=1)

    top_indices = torch.empty((M, 3), device=x.device, dtype=torch.int32)
    top_weights = torch.empty((M, 3), device=x.device, dtype=torch.float32)
    counts = torch.zeros(E, device=x.device, dtype=torch.int32)
    probability_sums = torch.zeros(E, device=x.device, dtype=torch.float32)
    z = torch.empty(M, device=x.device, dtype=torch.float32)
    entropy = torch.empty_like(z)
    max_probability = torch.empty_like(z)
    _top3_softmax_metrics_kernel[(M,)](
        logits, top_indices, top_weights, counts, probability_sums,
        z, entropy, max_probability, M, E, BLOCK_E=_next_power_of_two(E),
        num_warps=1)

    BM = 16
    offsets = torch.empty(E + 1, device=x.device, dtype=torch.int32)
    cursor = torch.empty(E, device=x.device, dtype=torch.int32)
    total = torch.empty((), device=x.device, dtype=torch.int32)
    _padded_prefix_kernel[(1,)](
        counts, offsets, cursor, total, E, BLOCK_E=_next_power_of_two(E), BLOCK_M=BM,
        num_warps=1)
    # A fixed upper bound avoids a device-to-host synchronization for total.
    padded_max = M * 3 + E * (BM - 1)
    assignments = M * 3
    inverse = torch.empty(assignments, device=x.device, dtype=torch.int32)
    _assign_packed_rows_kernel[(assignments,)](
        top_indices, offsets, cursor, inverse, assignments, num_warps=1)
    packed = torch.zeros((padded_max, D), device=x.device, dtype=x.dtype)
    _pack_tokens_kernel[(assignments, triton.cdiv(D, 256))](
        x, inverse, packed, M, D, BLOCK_D=256, num_warps=4)

    gate = torch.empty((padded_max, HIDDEN), device=x.device, dtype=x.dtype)
    up = torch.empty_like(gate)
    grid = (triton.cdiv(padded_max, BM), triton.cdiv(HIDDEN, 64))
    _grouped_gate_up_kernel[grid](
        packed, gate_weight, up_weight, offsets, gate, up,
        padded_max, D, HIDDEN, E, BLOCK_E=_next_power_of_two(E),
        BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
    act = torch.empty_like(gate)
    n_act = padded_max * HIDDEN
    _silu_mul_kernel[(triton.cdiv(n_act, 256),)](
        gate, up, act, n_act, BLOCK=256, num_warps=4)
    packed_out = torch.empty((padded_max, D), device=x.device, dtype=x.dtype)
    _grouped_down_kernel[(triton.cdiv(padded_max, BM), triton.cdiv(D, 64))](
        act, down_weight, offsets, packed_out,
        padded_max, D, HIDDEN, E, BLOCK_E=_next_power_of_two(E),
        BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
    output = torch.empty_like(x)
    _combine_top3_kernel[(M, triton.cdiv(D, 256))](
        packed_out, inverse, top_weights, output, M, D, BLOCK_D=256,
        num_warps=4)
    return FusedMoEForwardResult(
        output=output, top_indices=top_indices, top_weights=top_weights,
        expert_counts=counts, probability_sums=probability_sums,
        router_z_per_token=z, router_entropy_per_token=entropy,
        router_max_probability_per_token=max_probability)
