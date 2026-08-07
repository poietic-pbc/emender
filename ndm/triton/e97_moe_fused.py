"""Fail-closed fused Triton kernels for the E97 MoE data path.

This file is the only admissible compute path for E97 MoE experiments.  Python
allocates bounded buffers and launches kernels; routing, packing, expert
SwiGLU, and deterministic top-k combination execute in ``@triton.jit``
kernels.  There is deliberately no eager fallback.

The production forward includes both the always-active shared expert and the
routed experts.  It remains forward-only while fused backward and optimizer
kernels are completed.  ``require_training=True`` therefore fails closed: it
is impossible to mistake this implementation for a training-ready system or
submit a parity/training job prematurely.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import triton
import triton.language as tl


FUSED_E97_MOE_ABI = "emender-e97-moe-triton-v2"


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
def _router_aux_metrics_kernel(
    COUNTS, PROB_SUM, Z_PER_TOKEN, LOAD_BALANCE, Z_MEAN,
    M: tl.constexpr, E: tl.constexpr,
):
    index = tl.program_id(0)
    if index < E:
        count = tl.load(COUNTS + index).to(tl.float32)
        probability_sum = tl.load(PROB_SUM + index).to(tl.float32)
        tl.atomic_add(LOAD_BALANCE, E * count * probability_sum / (3.0 * M * M))
    if index < M:
        tl.atomic_add(Z_MEAN, tl.load(Z_PER_TOKEN + index).to(tl.float32) / M)


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
    raw_expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    valid_block = raw_expert < E
    expert = tl.minimum(raw_expert, E - 1)
    rows = row0 + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = valid_block & (rows >= start) & (rows < end) & (rows < PADDED_M)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * D + k[None, :],
                    mask=row_mask[:, None] & (k[None, :] < D), other=0.0)
        base = expert * HIDDEN * D
        wg = tl.load(W_GATE + base + cols[None, :] * D + k[:, None],
                     mask=valid_block & (cols[None, :] < HIDDEN) &
                     (k[:, None] < D), other=0.0)
        wu = tl.load(W_UP + base + cols[None, :] * D + k[:, None],
                     mask=valid_block & (cols[None, :] < HIDDEN) &
                     (k[:, None] < D), other=0.0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    out_mask = row_mask[:, None] & (cols[None, :] < HIDDEN)
    tl.store(GATE + rows[:, None] * HIDDEN + cols[None, :], gate, mask=out_mask)
    tl.store(UP + rows[:, None] * HIDDEN + cols[None, :], up, mask=out_mask)


@triton.jit
def _grouped_gate_up_silu_kernel(
    X, W_GATE, W_UP, OFFSETS, ACT,
    PADDED_M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BLOCK_E: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Grouped gate/up GEMMs with SiLU multiplication fused into the store."""
    pm, pn = tl.program_id(0), tl.program_id(1)
    row0 = pm * BM
    ev = tl.arange(0, BLOCK_E)
    ends = tl.load(OFFSETS + ev + 1, mask=ev < E, other=PADDED_M + 1)
    raw_expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    valid_block = raw_expert < E
    expert = tl.minimum(raw_expert, E - 1)
    rows = row0 + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = valid_block & (rows >= start) & (rows < end) & (rows < PADDED_M)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * D + k[None, :],
                    mask=row_mask[:, None] & (k[None, :] < D), other=0.0)
        base = expert * HIDDEN * D
        wg = tl.load(W_GATE + base + cols[None, :] * D + k[:, None],
                     mask=valid_block & (cols[None, :] < HIDDEN) &
                     (k[:, None] < D), other=0.0)
        wu = tl.load(W_UP + base + cols[None, :] * D + k[:, None],
                     mask=valid_block & (cols[None, :] < HIDDEN) &
                     (k[:, None] < D), other=0.0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    value = (gate * tl.sigmoid(gate)) * up
    out_mask = row_mask[:, None] & (cols[None, :] < HIDDEN)
    tl.store(ACT + rows[:, None] * HIDDEN + cols[None, :], value, mask=out_mask)


@triton.jit
def _dense_gate_up_silu_kernel(
    X, W_GATE, W_UP, ACT,
    M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * D + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < D), other=0.0)
        wg = tl.load(W_GATE + rn[None, :] * D + rk[:, None],
                     mask=(rn[None, :] < HIDDEN) & (rk[:, None] < D), other=0.0)
        wu = tl.load(W_UP + rn[None, :] * D + rk[:, None],
                     mask=(rn[None, :] < HIDDEN) & (rk[:, None] < D), other=0.0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    value = (gate * tl.sigmoid(gate)) * up
    tl.store(ACT + rm[:, None] * HIDDEN + rn[None, :], value,
             mask=(rm[:, None] < M) & (rn[None, :] < HIDDEN))


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
    raw_expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    valid_block = raw_expert < E
    expert = tl.minimum(raw_expert, E - 1)
    rows = row0 + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = valid_block & (rows >= start) & (rows < end) & (rows < PADDED_M)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, HIDDEN, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(ACT + rows[:, None] * HIDDEN + k[None, :],
                    mask=row_mask[:, None] & (k[None, :] < HIDDEN), other=0.0)
        base = expert * D * HIDDEN
        w = tl.load(W_DOWN + base + cols[None, :] * HIDDEN + k[:, None],
                    mask=valid_block & (cols[None, :] < D) &
                    (k[:, None] < HIDDEN), other=0.0)
        acc += tl.dot(a, w)
    mask = row_mask[:, None] & (cols[None, :] < D)
    tl.store(OUT + rows[:, None] * D + cols[None, :], acc, mask=mask)


@triton.jit
def _dense_down_kernel(
    ACT, W_DOWN, OUT,
    M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, HIDDEN, BK):
        rk = k0 + tl.arange(0, BK)
        act = tl.load(ACT + rm[:, None] * HIDDEN + rk[None, :],
                      mask=(rm[:, None] < M) & (rk[None, :] < HIDDEN), other=0.0)
        weight = tl.load(W_DOWN + rn[None, :] * HIDDEN + rk[:, None],
                         mask=(rn[None, :] < D) & (rk[:, None] < HIDDEN), other=0.0)
        acc += tl.dot(act, weight)
    tl.store(OUT + rm[:, None] * D + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < D))


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


@triton.jit
def _combine_shared_top3_kernel(
    SHARED_OUT, PACKED_OUT, INVERSE, TOP_WEIGHT, OUT,
    M: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    m, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D
    p0 = tl.load(INVERSE + m * 3 + 0)
    p1 = tl.load(INVERSE + m * 3 + 1)
    p2 = tl.load(INVERSE + m * 3 + 2)
    w0 = tl.load(TOP_WEIGHT + m * 3 + 0).to(tl.float32)
    w1 = tl.load(TOP_WEIGHT + m * 3 + 1).to(tl.float32)
    w2 = tl.load(TOP_WEIGHT + m * 3 + 2).to(tl.float32)
    shared = tl.load(SHARED_OUT + m * D + d, mask=mask, other=0.0).to(tl.float32)
    y0 = tl.load(PACKED_OUT + p0 * D + d, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(PACKED_OUT + p1 * D + d, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(PACKED_OUT + p2 * D + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + m * D + d, shared + w0*y0 + w1*y1 + w2*y2, mask=mask)


@triton.jit
def _combine_backward_kernel(
    GRAD_OUT, PACKED_OUT, INVERSE, TOP_WEIGHT, GRAD_PACKED_OUT, GRAD_TOP_WEIGHT,
    ASSIGNMENTS: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    assignment = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    token = assignment // 3
    row = tl.load(INVERSE + assignment)
    weight = tl.load(TOP_WEIGHT + assignment).to(tl.float32)
    grad = tl.load(GRAD_OUT + token * D + d, mask=mask, other=0.0).to(tl.float32)
    expert_out = tl.load(PACKED_OUT + row * D + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(GRAD_PACKED_OUT + row * D + d, weight * grad, mask=mask)
    tl.store(GRAD_TOP_WEIGHT + assignment, tl.sum(grad * expert_out, axis=0))


@triton.jit
def _grouped_down_input_backward_kernel(
    GRAD_OUT, W_DOWN, OFFSETS, GRAD_ACT,
    PADDED_M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BLOCK_E: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    row0 = pm * BM
    ev = tl.arange(0, BLOCK_E)
    ends = tl.load(OFFSETS + ev + 1, mask=ev < E, other=PADDED_M + 1)
    raw_expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    valid_block = raw_expert < E
    expert = tl.minimum(raw_expert, E - 1)
    rows = row0 + tl.arange(0, BM)
    hidden = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = valid_block & (rows >= start) & (rows < end) & (rows < PADDED_M)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        grad = tl.load(GRAD_OUT + rows[:, None] * D + k[None, :],
                       mask=row_mask[:, None] & (k[None, :] < D), other=0.0)
        base = expert * D * HIDDEN
        weight = tl.load(W_DOWN + base + k[:, None] * HIDDEN + hidden[None, :],
                         mask=valid_block & (k[:, None] < D) &
                         (hidden[None, :] < HIDDEN), other=0.0)
        acc += tl.dot(grad, weight)
    tl.store(GRAD_ACT + rows[:, None] * HIDDEN + hidden[None, :], acc,
             mask=row_mask[:, None] & (hidden[None, :] < HIDDEN))


@triton.jit
def _grouped_down_weight_backward_kernel(
    GRAD_OUT, ACT, OFFSETS, GRAD_W,
    PADDED_M, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    expert, pn, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    k = pk * BK + tl.arange(0, BK)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    acc = tl.zeros((BN, BK), tl.float32)
    # Expert spans are padded to BM alignment.  Walk only this expert's span;
    # the former range(0, PADDED_M) made every expert rescan all eight spans.
    m0 = start
    while m0 < end:
        m = m0 + tl.arange(0, BM)
        row_mask = m < end
        grad = tl.load(GRAD_OUT + m[:, None] * D + n[None, :],
                       mask=row_mask[:, None] & (n[None, :] < D), other=0.0)
        act = tl.load(ACT + m[:, None] * HIDDEN + k[None, :],
                      mask=row_mask[:, None] & (k[None, :] < HIDDEN), other=0.0)
        acc += tl.dot(tl.trans(grad), act)
        m0 += BM
    base = expert * D * HIDDEN
    tl.store(GRAD_W + base + n[:, None] * HIDDEN + k[None, :], acc,
             mask=(n[:, None] < D) & (k[None, :] < HIDDEN))


@triton.jit
def _silu_backward_kernel(
    GATE, UP, GRAD_ACT, GRAD_GATE, GRAD_UP,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(GRAD_ACT + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    tl.store(GRAD_GATE + offsets, grad * up * sigmoid * (1.0 + gate * (1.0 - sigmoid)), mask=mask)
    tl.store(GRAD_UP + offsets, grad * silu, mask=mask)


@triton.jit
def _grouped_gate_up_input_backward_kernel(
    GRAD_GATE, GRAD_UP, W_GATE, W_UP, OFFSETS, GRAD_X,
    PADDED_M: tl.constexpr, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BLOCK_E: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    row0 = pm * BM
    ev = tl.arange(0, BLOCK_E)
    ends = tl.load(OFFSETS + ev + 1, mask=ev < E, other=PADDED_M + 1)
    raw_expert = tl.sum(tl.where((ev < E) & (row0 >= ends), 1, 0), axis=0)
    valid_block = raw_expert < E
    expert = tl.minimum(raw_expert, E - 1)
    rows = row0 + tl.arange(0, BM)
    dim = pn * BN + tl.arange(0, BN)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    row_mask = valid_block & (rows >= start) & (rows < end) & (rows < PADDED_M)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, HIDDEN, BK):
        hidden = k0 + tl.arange(0, BK)
        dg = tl.load(GRAD_GATE + rows[:, None] * HIDDEN + hidden[None, :],
                     mask=row_mask[:, None] & (hidden[None, :] < HIDDEN), other=0.0)
        du = tl.load(GRAD_UP + rows[:, None] * HIDDEN + hidden[None, :],
                     mask=row_mask[:, None] & (hidden[None, :] < HIDDEN), other=0.0)
        base = expert * HIDDEN * D
        wg = tl.load(W_GATE + base + hidden[:, None] * D + dim[None, :],
                     mask=valid_block & (hidden[:, None] < HIDDEN) &
                     (dim[None, :] < D), other=0.0)
        wu = tl.load(W_UP + base + hidden[:, None] * D + dim[None, :],
                     mask=valid_block & (hidden[:, None] < HIDDEN) &
                     (dim[None, :] < D), other=0.0)
        acc += tl.dot(dg, wg) + tl.dot(du, wu)
    tl.store(GRAD_X + rows[:, None] * D + dim[None, :], acc,
             mask=row_mask[:, None] & (dim[None, :] < D))


@triton.jit
def _grouped_gate_up_weight_backward_kernel(
    GRAD_GATE, GRAD_UP, X, OFFSETS, GRAD_W_GATE, GRAD_W_UP,
    PADDED_M, D: tl.constexpr, HIDDEN: tl.constexpr,
    E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    expert, pn, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    k = pk * BK + tl.arange(0, BK)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    acc_gate = tl.zeros((BN, BK), tl.float32)
    acc_up = tl.zeros((BN, BK), tl.float32)
    # Expert-local dynamic loop: do not perform eight masked passes over the
    # complete rank-level packed batch for each expert weight tile.
    m0 = start
    while m0 < end:
        m = m0 + tl.arange(0, BM)
        row_mask = m < end
        dg = tl.load(GRAD_GATE + m[:, None] * HIDDEN + n[None, :],
                     mask=row_mask[:, None] & (n[None, :] < HIDDEN), other=0.0)
        du = tl.load(GRAD_UP + m[:, None] * HIDDEN + n[None, :],
                     mask=row_mask[:, None] & (n[None, :] < HIDDEN), other=0.0)
        x = tl.load(X + m[:, None] * D + k[None, :],
                    mask=row_mask[:, None] & (k[None, :] < D), other=0.0)
        acc_gate += tl.dot(tl.trans(dg), x)
        acc_up += tl.dot(tl.trans(du), x)
        m0 += BM
    base = expert * HIDDEN * D
    mask = (n[:, None] < HIDDEN) & (k[None, :] < D)
    tl.store(GRAD_W_GATE + base + n[:, None] * D + k[None, :], acc_gate, mask=mask)
    tl.store(GRAD_W_UP + base + n[:, None] * D + k[None, :], acc_up, mask=mask)


@triton.jit
def _top3_router_backward_kernel(
    LOGITS, COUNTS, TOP_INDEX, TOP_WEIGHT, GRAD_TOP_WEIGHT, GRAD_AUX, GRAD_LOGITS,
    M: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    e = tl.arange(0, BLOCK_E)
    mask = e < E
    logits = tl.load(LOGITS + token * E + e, mask=mask, other=-float("inf")).to(tl.float32)
    maximum = tl.max(logits, axis=0)
    exponent = tl.exp(logits - maximum)
    denominator = tl.sum(tl.where(mask, exponent, 0.0), axis=0)
    probability = exponent / denominator
    i0 = tl.load(TOP_INDEX + token * 3 + 0)
    i1 = tl.load(TOP_INDEX + token * 3 + 1)
    i2 = tl.load(TOP_INDEX + token * 3 + 2)
    w0 = tl.load(TOP_WEIGHT + token * 3 + 0).to(tl.float32)
    w1 = tl.load(TOP_WEIGHT + token * 3 + 1).to(tl.float32)
    w2 = tl.load(TOP_WEIGHT + token * 3 + 2).to(tl.float32)
    g0 = tl.load(GRAD_TOP_WEIGHT + token * 3 + 0).to(tl.float32)
    g1 = tl.load(GRAD_TOP_WEIGHT + token * 3 + 1).to(tl.float32)
    g2 = tl.load(GRAD_TOP_WEIGHT + token * 3 + 2).to(tl.float32)
    mean = w0 * g0 + w1 * g1 + w2 * g2
    value = tl.where(e == i0, w0 * (g0 - mean), 0.0)
    value += tl.where(e == i1, w1 * (g1 - mean), 0.0)
    value += tl.where(e == i2, w2 * (g2 - mean), 0.0)
    grad_load_balance = tl.load(GRAD_AUX + 0).to(tl.float32)
    grad_z = tl.load(GRAD_AUX + 1).to(tl.float32)
    counts = tl.load(COUNTS + e, mask=mask, other=0).to(tl.float32)
    expected_count = tl.sum(tl.where(mask, probability * counts, 0.0), axis=0)
    value += (grad_load_balance * E / (3.0 * M * M)
              * probability * (counts - expected_count))
    logsumexp = tl.log(denominator) + maximum
    value += grad_z * (2.0 / M) * logsumexp * probability
    tl.store(GRAD_LOGITS + token * E + e, value, mask=mask)


@triton.jit
def _dense_weight_backward_kernel(
    GRAD_OUT, X, GRAD_W,
    M, OUT: tl.constexpr, IN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pn, pk = tl.program_id(0), tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    k = pk * BK + tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for m0 in range(0, M, BM):
        m = m0 + tl.arange(0, BM)
        grad = tl.load(GRAD_OUT + m[:, None] * OUT + n[None, :],
                       mask=(m[:, None] < M) & (n[None, :] < OUT), other=0.0).to(tl.float32)
        x = tl.load(X + m[:, None] * IN + k[None, :],
                    mask=(m[:, None] < M) & (k[None, :] < IN), other=0.0).to(tl.float32)
        acc += tl.dot(tl.trans(grad), x, input_precision="ieee")
    tl.store(GRAD_W + n[:, None] * IN + k[None, :], acc,
             mask=(n[:, None] < OUT) & (k[None, :] < IN))


@triton.jit
def _router_input_backward_kernel(
    GRAD_LOGITS, ROUTER_W, GRAD_X,
    M: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
):
    token, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    e = tl.arange(0, BLOCK_E)
    mask_d = d < D
    grad = tl.load(GRAD_LOGITS + token * E + e, mask=e < E, other=0.0)
    weight = tl.load(ROUTER_W + e[:, None] * D + d[None, :],
                     mask=(e[:, None] < E) & mask_d[None, :], other=0.0)
    tl.store(GRAD_X + token * D + d, tl.sum(grad[:, None] * weight, axis=0),
             mask=mask_d)


@triton.jit
def _router_input_and_unpack_kernel(
    GRAD_LOGITS, ROUTER_W, GRAD_PACKED_X, ROUTED_INVERSE, GRAD_SHARED_X, GRAD_X,
    M: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
):
    token, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    e = tl.arange(0, BLOCK_E)
    mask_d = d < D
    logits_grad = tl.load(GRAD_LOGITS + token * E + e, mask=e < E, other=0.0)
    router_w = tl.load(ROUTER_W + e[:, None] * D + d[None, :],
                       mask=(e[:, None] < E) & mask_d[None, :], other=0.0)
    router_dx = tl.sum(logits_grad[:, None] * router_w, axis=0)
    p0 = tl.load(ROUTED_INVERSE + token * 3 + 0)
    p1 = tl.load(ROUTED_INVERSE + token * 3 + 1)
    p2 = tl.load(ROUTED_INVERSE + token * 3 + 2)
    dx0 = tl.load(GRAD_PACKED_X + p0 * D + d, mask=mask_d, other=0.0).to(tl.float32)
    dx1 = tl.load(GRAD_PACKED_X + p1 * D + d, mask=mask_d, other=0.0).to(tl.float32)
    dx2 = tl.load(GRAD_PACKED_X + p2 * D + d, mask=mask_d, other=0.0).to(tl.float32)
    shared_dx = tl.load(GRAD_SHARED_X + token * D + d, mask=mask_d, other=0.0).to(tl.float32)
    tl.store(GRAD_X + token * D + d, router_dx + dx0 + dx1 + dx2 + shared_dx,
             mask=mask_d)


@triton.jit
def _pack_aux_grad_kernel(GRAD_LOAD_BALANCE, GRAD_Z, GRAD_AUX):
    tl.store(GRAD_AUX + 0, tl.load(GRAD_LOAD_BALANCE).to(tl.float32))
    tl.store(GRAD_AUX + 1, tl.load(GRAD_Z).to(tl.float32))


@triton.jit
def _schedulefree_lerp_kernel(PARAM, Z, N: tl.constexpr,
                              WEIGHT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    parameter = tl.load(PARAM + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(PARAM + offsets, parameter + WEIGHT * (z - parameter), mask=mask)


@triton.jit
def _schedulefree_adamw_update_kernel(
    PARAM, GRAD, Z, EXP_AVG_SQ,
    N: tl.constexpr,
    LR: tl.constexpr, BETA1: tl.constexpr, BETA2: tl.constexpr,
    BIAS_CORRECTION2: tl.constexpr, EPS: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr, CKP1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(PARAM + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(GRAD + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.load(EXP_AVG_SQ + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = BETA2 * variance + (1.0 - BETA2) * grad * grad
    normalized = grad / (tl.sqrt(variance / BIAS_CORRECTION2) + EPS)
    normalized += WEIGHT_DECAY * y
    y += CKP1 * (z - y)
    y += LR * (BETA1 * (1.0 - CKP1) - 1.0) * normalized
    z -= LR * normalized
    tl.store(PARAM + offsets, y, mask=mask)
    tl.store(Z + offsets, z, mask=mask)
    tl.store(EXP_AVG_SQ + offsets, variance, mask=mask)


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
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor
    kernel_abi: str = FUSED_E97_MOE_ABI
    # Private parity/backward payload. Production callers must not consume it.
    _intermediates: tuple[torch.Tensor, ...] | None = None


def fused_routed_swiglu_forward(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    shared_gate_weight: torch.Tensor | None = None,
    shared_up_weight: torch.Tensor | None = None,
    shared_down_weight: torch.Tensor | None = None,
    require_training: bool = False,
    _capture_intermediates: bool = False,
) -> FusedMoEForwardResult:
    """Execute the fused, dropless top-3 routed expert forward.

    Shapes are ``x[M,D]``, router ``[E,D]``, routed gate/up ``[E,H,D]`` and
    routed down ``[E,D,H]``.  When supplied, shared gate/up are ``[H,D]`` and
    shared down is ``[D,H]``.  All expert tensors must be BF16 and contiguous.
    The router weights/logits are FP32.  No CPU/eager fallback exists.

    This routed-only entry point remains available as a kernel oracle.  The
    admissible E97 architecture calls :func:`fused_shared_routed_swiglu_forward`,
    which makes the shared expert mandatory.
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
    shared_weights = (shared_gate_weight, shared_up_weight, shared_down_weight)
    has_shared = all(weight is not None for weight in shared_weights)
    if any(weight is not None for weight in shared_weights) and not has_shared:
        raise ValueError("shared gate, up, and down weights must be supplied together")
    if has_shared:
        assert shared_gate_weight is not None
        assert shared_up_weight is not None
        assert shared_down_weight is not None
        if any(weight.dtype != torch.bfloat16 or not weight.is_contiguous()
               for weight in shared_weights):
            raise ValueError("shared expert weights must be contiguous BF16 tensors")
        if shared_gate_weight.shape != (HIDDEN, D) or shared_up_weight.shape != (HIDDEN, D):
            raise ValueError("shared gate/up weights must be [hidden, dim]")
        if shared_down_weight.shape != (D, HIDDEN):
            raise ValueError("shared down weight must be [dim, hidden]")

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
    load_balance_loss = torch.zeros((), device=x.device, dtype=torch.float32)
    z_loss = torch.zeros((), device=x.device, dtype=torch.float32)
    _router_aux_metrics_kernel[(max(M, E),)](
        counts, probability_sums, z, load_balance_loss, z_loss, M, E,
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

    act = torch.empty((padded_max, HIDDEN), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(padded_max, BM), triton.cdiv(HIDDEN, 64))
    _grouped_gate_up_silu_kernel[grid](
        packed, gate_weight, up_weight, offsets, act,
        padded_max, D, HIDDEN, E, BLOCK_E=_next_power_of_two(E),
        BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
    packed_out = torch.empty((padded_max, D), device=x.device, dtype=x.dtype)
    _grouped_down_kernel[(triton.cdiv(padded_max, BM), triton.cdiv(D, 64))](
        act, down_weight, offsets, packed_out,
        padded_max, D, HIDDEN, E, BLOCK_E=_next_power_of_two(E),
        BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
    output = torch.empty_like(x)
    shared_act: torch.Tensor | None = None
    if has_shared:
        assert shared_gate_weight is not None
        assert shared_up_weight is not None
        assert shared_down_weight is not None
        shared_act = torch.empty((M, HIDDEN), device=x.device, dtype=x.dtype)
        dense_grid = (triton.cdiv(M, BM), triton.cdiv(HIDDEN, 64))
        _dense_gate_up_silu_kernel[dense_grid](
            x, shared_gate_weight, shared_up_weight, shared_act,
            M, D, HIDDEN, BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
        shared_out = torch.empty_like(x)
        _dense_down_kernel[(triton.cdiv(M, BM), triton.cdiv(D, 64))](
            shared_act, shared_down_weight, shared_out,
            M, D, HIDDEN, BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
        _combine_shared_top3_kernel[(M, triton.cdiv(D, 256))](
            shared_out, packed_out, inverse, top_weights, output,
            M, D, BLOCK_D=256, num_warps=4)
    else:
        _combine_top3_kernel[(M, triton.cdiv(D, 256))](
            packed_out, inverse, top_weights, output, M, D, BLOCK_D=256,
            num_warps=4)
    return FusedMoEForwardResult(
        output=output, top_indices=top_indices, top_weights=top_weights,
        expert_counts=counts, probability_sums=probability_sums,
        router_z_per_token=z, router_entropy_per_token=entropy,
        router_max_probability_per_token=max_probability,
        load_balance_loss=load_balance_loss, z_loss=z_loss,
        _intermediates=(logits, packed, offsets, inverse, act, packed_out, shared_act)
        if _capture_intermediates and shared_act is not None else None)


def fused_shared_routed_swiglu_backward(
    grad_output: torch.Tensor,
    forward: FusedMoEForwardResult,
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_gate_weight: torch.Tensor,
    routed_up_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    grad_aux: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Parity API for the fused output-gradient backward.

    This computes gradients from the MoE output through the shared expert,
    routed experts, normalized top-3 weights, and router. Auxiliary-router
    gradients and optimizer updates are separate gates, so this function does
    not make the module training-ready.
    """
    _require_hip_triton(grad_output)
    if forward._intermediates is None:
        raise RuntimeError("forward intermediates were not captured for fused backward")
    if grad_output.dtype != torch.bfloat16 or not grad_output.is_contiguous():
        raise ValueError("grad_output must be contiguous BF16")
    logits, packed, offsets, inverse, routed_act, packed_out, shared_act = forward._intermediates
    M, D = x.shape
    E = router_weight.shape[0]
    HIDDEN = routed_gate_weight.shape[1]
    padded_max = packed.shape[0]
    assignments = M * 3
    BM, BN, BK = 16, 32, 32
    if grad_aux is None:
        grad_aux = torch.zeros(2, device=x.device, dtype=torch.float32)
    if (grad_aux.shape != (2,) or grad_aux.dtype != torch.float32 or
            grad_aux.device != x.device or not grad_aux.is_contiguous()):
        raise ValueError("grad_aux must be contiguous FP32 [load_balance, z_loss]")

    grad_packed_out = torch.zeros_like(packed_out)
    grad_top_weight = torch.empty_like(forward.top_weights)
    _combine_backward_kernel[(assignments,)](
        grad_output, packed_out, inverse, forward.top_weights,
        grad_packed_out, grad_top_weight,
        assignments, D, BLOCK_D=_next_power_of_two(D), num_warps=4)

    grad_routed_act = torch.empty_like(routed_act)
    _grouped_down_input_backward_kernel[
        (triton.cdiv(padded_max, BM), triton.cdiv(HIDDEN, BN))
    ](
        grad_packed_out, routed_down_weight, offsets, grad_routed_act,
        padded_max, D, HIDDEN, E, BLOCK_E=_next_power_of_two(E),
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
    grad_routed_down = torch.empty_like(routed_down_weight)
    _grouped_down_weight_backward_kernel[
        (E, triton.cdiv(D, BN), triton.cdiv(HIDDEN, BK))
    ](
        grad_packed_out, routed_act, offsets, grad_routed_down,
        padded_max, D, HIDDEN, E, BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=1)

    routed_gate = torch.empty_like(routed_act)
    routed_up = torch.empty_like(routed_act)
    _grouped_gate_up_kernel[
        (triton.cdiv(padded_max, BM), triton.cdiv(HIDDEN, 64))
    ](
        packed, routed_gate_weight, routed_up_weight, offsets,
        routed_gate, routed_up, padded_max, D, HIDDEN, E,
        BLOCK_E=_next_power_of_two(E), BM=BM, BN=64, BK=32,
        num_warps=4, num_stages=1)
    grad_routed_gate = torch.empty_like(routed_gate)
    grad_routed_up = torch.empty_like(routed_up)
    _silu_backward_kernel[(triton.cdiv(padded_max * HIDDEN, 256),)](
        routed_gate, routed_up, grad_routed_act, grad_routed_gate, grad_routed_up,
        padded_max * HIDDEN, BLOCK=256, num_warps=4)
    grad_packed_x = torch.empty_like(packed)
    _grouped_gate_up_input_backward_kernel[
        (triton.cdiv(padded_max, BM), triton.cdiv(D, BN))
    ](
        grad_routed_gate, grad_routed_up, routed_gate_weight, routed_up_weight,
        offsets, grad_packed_x, padded_max, D, HIDDEN, E,
        BLOCK_E=_next_power_of_two(E), BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=1)
    grad_routed_gate_weight = torch.empty_like(routed_gate_weight)
    grad_routed_up_weight = torch.empty_like(routed_up_weight)
    _grouped_gate_up_weight_backward_kernel[
        (E, triton.cdiv(HIDDEN, BN), triton.cdiv(D, BK))
    ](
        grad_routed_gate, grad_routed_up, packed, offsets,
        grad_routed_gate_weight, grad_routed_up_weight,
        padded_max, D, HIDDEN, E, BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=1)

    # The shared expert is the E=1 grouped case over unpadded token rows.
    shared_offsets = torch.tensor([0, M], device=x.device, dtype=torch.int32)
    grad_shared_act = torch.empty_like(shared_act)
    _grouped_down_input_backward_kernel[(triton.cdiv(M, BM), triton.cdiv(HIDDEN, BN))](
        grad_output, shared_down_weight, shared_offsets, grad_shared_act,
        M, D, HIDDEN, 1, BLOCK_E=1, BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=1)
    grad_shared_down = torch.empty_like(shared_down_weight)
    _grouped_down_weight_backward_kernel[
        (1, triton.cdiv(D, BN), triton.cdiv(HIDDEN, BK))
    ](
        grad_output, shared_act, shared_offsets, grad_shared_down,
        M, D, HIDDEN, 1, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
    shared_gate = torch.empty_like(shared_act)
    shared_up = torch.empty_like(shared_act)
    _grouped_gate_up_kernel[(triton.cdiv(M, BM), triton.cdiv(HIDDEN, 64))](
        x, shared_gate_weight, shared_up_weight, shared_offsets,
        shared_gate, shared_up, M, D, HIDDEN, 1, BLOCK_E=1,
        BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
    grad_shared_gate = torch.empty_like(shared_gate)
    grad_shared_up = torch.empty_like(shared_up)
    _silu_backward_kernel[(triton.cdiv(M * HIDDEN, 256),)](
        shared_gate, shared_up, grad_shared_act, grad_shared_gate, grad_shared_up,
        M * HIDDEN, BLOCK=256, num_warps=4)
    grad_shared_x = torch.empty_like(x)
    _grouped_gate_up_input_backward_kernel[(triton.cdiv(M, BM), triton.cdiv(D, BN))](
        grad_shared_gate, grad_shared_up, shared_gate_weight, shared_up_weight,
        shared_offsets, grad_shared_x, M, D, HIDDEN, 1, BLOCK_E=1,
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
    grad_shared_gate_weight = torch.empty_like(shared_gate_weight)
    grad_shared_up_weight = torch.empty_like(shared_up_weight)
    _grouped_gate_up_weight_backward_kernel[
        (1, triton.cdiv(HIDDEN, BN), triton.cdiv(D, BK))
    ](
        grad_shared_gate, grad_shared_up, x, shared_offsets,
        grad_shared_gate_weight, grad_shared_up_weight,
        M, D, HIDDEN, 1, BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=1)

    grad_logits = torch.empty((M, E), device=x.device, dtype=torch.float32)
    _top3_router_backward_kernel[(M,)](
        logits, forward.expert_counts, forward.top_indices, forward.top_weights,
        grad_top_weight, grad_aux, grad_logits,
        M, E, BLOCK_E=_next_power_of_two(E), num_warps=1)
    grad_router_weight = torch.empty_like(router_weight)
    _dense_weight_backward_kernel[(triton.cdiv(E, BN), triton.cdiv(D, BK))](
        grad_logits, x, grad_router_weight, M, E, D,
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
    grad_x = torch.empty_like(x)
    _router_input_and_unpack_kernel[(M, triton.cdiv(D, 128))](
        grad_logits, router_weight, grad_packed_x, inverse, grad_shared_x, grad_x,
        M, D, E, BLOCK_D=128, BLOCK_E=_next_power_of_two(E), num_warps=4)
    return (
        grad_x, grad_router_weight,
        grad_routed_gate_weight, grad_routed_up_weight, grad_routed_down,
        grad_shared_gate_weight, grad_shared_up_weight, grad_shared_down,
    )


def fused_shared_routed_swiglu_forward_backward_parity(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_gate_weight: torch.Tensor,
    routed_up_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    grad_output: torch.Tensor,
    grad_aux: torch.Tensor | None = None,
) -> tuple[FusedMoEForwardResult, tuple[torch.Tensor, ...]]:
    """Run fused forward and output-gradient backward for kernel parity tests."""
    forward = fused_routed_swiglu_forward(
        x, router_weight, routed_gate_weight, routed_up_weight, routed_down_weight,
        shared_gate_weight=shared_gate_weight,
        shared_up_weight=shared_up_weight,
        shared_down_weight=shared_down_weight,
        _capture_intermediates=True,
    )
    gradients = fused_shared_routed_swiglu_backward(
        grad_output, forward, x, router_weight,
        routed_gate_weight, routed_up_weight, routed_down_weight,
        shared_gate_weight, shared_up_weight, shared_down_weight,
        grad_aux,
    )
    return forward, gradients



class _FusedTop3RouterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_weight):
        _require_hip_triton(x)
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("router input must be contiguous BF16 [tokens, dim]")
        if (router_weight.shape != (64, x.shape[1]) or
                router_weight.dtype != torch.float32 or not router_weight.is_contiguous()):
            raise ValueError("production router weight must be contiguous FP32 [64, dim]")
        M, D = x.shape
        E = 64
        logits = torch.empty((M, E), device=x.device, dtype=torch.float32)
        _router_fp32_kernel[(triton.cdiv(M, 16), 4)](
            x, router_weight, logits, M, D, E,
            x.stride(0), x.stride(1), router_weight.stride(0), router_weight.stride(1),
            logits.stride(0), logits.stride(1), BM=16, BE=16, BK=32,
            num_warps=4, num_stages=1)
        indices = torch.empty((M, 3), device=x.device, dtype=torch.int32)
        weights = torch.empty((M, 3), device=x.device, dtype=torch.float32)
        counts = torch.zeros(E, device=x.device, dtype=torch.int32)
        probability_sums = torch.zeros(E, device=x.device, dtype=torch.float32)
        z_per_token = torch.empty(M, device=x.device, dtype=torch.float32)
        entropy = torch.empty_like(z_per_token)
        max_probability = torch.empty_like(z_per_token)
        _top3_softmax_metrics_kernel[(M,)](
            logits, indices, weights, counts, probability_sums,
            z_per_token, entropy, max_probability, M, E, BLOCK_E=64, num_warps=1)
        load_balance = torch.zeros((), device=x.device, dtype=torch.float32)
        z_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        _router_aux_metrics_kernel[(max(M, E),)](
            counts, probability_sums, z_per_token, load_balance, z_loss, M, E,
            num_warps=1)
        ctx.mark_non_differentiable(indices, counts)
        ctx.save_for_backward(x, router_weight, logits, counts, indices, weights)
        return indices, weights, counts, load_balance, z_loss

    @staticmethod
    def backward(ctx, _grad_indices, grad_weights, _grad_counts,
                 grad_load_balance, grad_z):
        x, router_weight, logits, counts, indices, weights = ctx.saved_tensors
        M, D = x.shape
        if grad_weights is None:
            grad_weights = torch.zeros_like(weights)
        if grad_load_balance is None:
            grad_load_balance = torch.zeros((), device=x.device, dtype=torch.float32)
        if grad_z is None:
            grad_z = torch.zeros((), device=x.device, dtype=torch.float32)
        grad_aux = torch.empty(2, device=x.device, dtype=torch.float32)
        _pack_aux_grad_kernel[(1,)](grad_load_balance, grad_z, grad_aux, num_warps=1)
        grad_logits = torch.empty_like(logits)
        _top3_router_backward_kernel[(M,)](
            logits, counts, indices, weights, grad_weights.contiguous(), grad_aux,
            grad_logits, M, 64, BLOCK_E=64, num_warps=1)
        grad_router = torch.empty_like(router_weight)
        _dense_weight_backward_kernel[(2, triton.cdiv(D, 32))](
            grad_logits, x, grad_router, M, 64, D,
            BM=16, BN=32, BK=32, num_warps=4, num_stages=1)
        grad_x = torch.empty_like(x)
        _router_input_backward_kernel[(M, triton.cdiv(D, 128))](
            grad_logits, router_weight, grad_x, M, D, 64,
            BLOCK_D=128, BLOCK_E=64, num_warps=4)
        return grad_x, grad_router


def fused_top3_router_autograd(
    x: torch.Tensor, router_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused production 64-expert top-3 router with auxiliary gradients."""
    return _FusedTop3RouterFunction.apply(x, router_weight)


class _FusedSharedExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        _require_hip_triton(x)
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("shared expert input must be contiguous BF16 [tokens, dim]")
        M, D = x.shape
        if gate_weight.ndim != 2:
            raise ValueError("shared gate weight must be [hidden, dim]")
        H = gate_weight.shape[0]
        if (gate_weight.shape[1] != D or up_weight.shape != gate_weight.shape or
                down_weight.shape != (D, H)):
            raise ValueError("shared expert weight shape mismatch")
        if any(weight.dtype != torch.bfloat16 or not weight.is_contiguous()
               for weight in (gate_weight, up_weight, down_weight)):
            raise ValueError("shared expert weights must be contiguous BF16")
        act = torch.empty((M, H), device=x.device, dtype=x.dtype)
        _dense_gate_up_silu_kernel[(triton.cdiv(M, 16), triton.cdiv(H, 64))](
            x, gate_weight, up_weight, act, M, D, H,
            BM=16, BN=64, BK=32, num_warps=4, num_stages=1)
        output = torch.empty_like(x)
        _dense_down_kernel[(triton.cdiv(M, 16), triton.cdiv(D, 64))](
            act, down_weight, output, M, D, H,
            BM=16, BN=64, BK=32, num_warps=4, num_stages=1)
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight, act)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate_weight, up_weight, down_weight, act = ctx.saved_tensors
        M, D = x.shape
        H = gate_weight.shape[0]
        offsets = torch.tensor([0, M], device=x.device, dtype=torch.int32)
        BM, BN, BK = 16, 32, 32
        grad_act = torch.empty_like(act)
        _grouped_down_input_backward_kernel[(triton.cdiv(M, BM), triton.cdiv(H, BN))](
            grad_output.contiguous(), down_weight, offsets, grad_act,
            M, D, H, 1, BLOCK_E=1, BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=1)
        grad_down = torch.empty_like(down_weight)
        _grouped_down_weight_backward_kernel[(1, triton.cdiv(D, BN), triton.cdiv(H, BK))](
            grad_output.contiguous(), act, offsets, grad_down,
            M, D, H, 1, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
        gate = torch.empty_like(act)
        up = torch.empty_like(act)
        _grouped_gate_up_kernel[(triton.cdiv(M, BM), triton.cdiv(H, 64))](
            x, gate_weight, up_weight, offsets, gate, up,
            M, D, H, 1, BLOCK_E=1, BM=BM, BN=64, BK=32,
            num_warps=4, num_stages=1)
        grad_gate_values = torch.empty_like(gate)
        grad_up_values = torch.empty_like(up)
        _silu_backward_kernel[(triton.cdiv(M * H, 256),)](
            gate, up, grad_act, grad_gate_values, grad_up_values,
            M * H, BLOCK=256, num_warps=4)
        grad_x = torch.empty_like(x)
        _grouped_gate_up_input_backward_kernel[(triton.cdiv(M, BM), triton.cdiv(D, BN))](
            grad_gate_values, grad_up_values, gate_weight, up_weight, offsets, grad_x,
            M, D, H, 1, BLOCK_E=1, BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=1)
        grad_gate = torch.empty_like(gate_weight)
        grad_up = torch.empty_like(up_weight)
        _grouped_gate_up_weight_backward_kernel[(1, triton.cdiv(H, BN), triton.cdiv(D, BK))](
            grad_gate_values, grad_up_values, x, offsets, grad_gate, grad_up,
            M, D, H, 1, BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=1)
        return grad_x, grad_gate, grad_up, grad_down


def fused_shared_expert_autograd(x, gate_weight, up_weight, down_weight):
    """Fused always-active shared SwiGLU expert."""
    return _FusedSharedExpertFunction.apply(x, gate_weight, up_weight, down_weight)


class _FusedMoECombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, returned_expert_output, assignment_to_send_row,
                top_weights, shared_output):
        M = top_weights.shape[0]
        D = shared_output.shape[1]
        output = torch.empty_like(shared_output)
        _combine_shared_top3_kernel[(M, triton.cdiv(D, 256))](
            shared_output, returned_expert_output, assignment_to_send_row,
            top_weights, output, M, D, BLOCK_D=256, num_warps=4)
        ctx.save_for_backward(
            returned_expert_output, assignment_to_send_row, top_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        returned, inverse, weights = ctx.saved_tensors
        M, D = grad_output.shape
        grad_returned = torch.zeros_like(returned)
        grad_weights = torch.empty_like(weights)
        _combine_backward_kernel[(M * 3,)](
            grad_output.contiguous(), returned, inverse, weights,
            grad_returned, grad_weights, M * 3, D,
            BLOCK_D=_next_power_of_two(D), num_warps=4)
        return grad_returned, None, grad_weights, grad_output


def fused_shared_top3_combine_autograd(
    returned_expert_output, assignment_to_send_row, top_weights, shared_output,
):
    """Combine returned expert rows and the shared expert with fused backward."""
    return _FusedMoECombineFunction.apply(
        returned_expert_output, assignment_to_send_row, top_weights, shared_output)


class _FusedPackedLocalExpertsFunction(torch.autograd.Function):
    """Autograd for one rank's eight already-packed routed experts."""

    @staticmethod
    def forward(ctx, packed_x, expert_offsets, gate_weight, up_weight, down_weight):
        _require_hip_triton(packed_x)
        if packed_x.ndim != 2 or packed_x.dtype != torch.bfloat16 or not packed_x.is_contiguous():
            raise ValueError("packed local expert input must be contiguous BF16 [rows, dim]")
        if (expert_offsets.shape != (9,) or expert_offsets.dtype != torch.int32 or
                not expert_offsets.is_contiguous()):
            raise ValueError("local expert offsets must be contiguous int32 [9]")
        rows, dim = packed_x.shape
        if gate_weight.ndim != 3 or gate_weight.shape[0] != 8:
            raise ValueError("each GCD must own exactly eight routed experts")
        hidden = gate_weight.shape[1]
        if (gate_weight.shape[2] != dim or up_weight.shape != gate_weight.shape or
                down_weight.shape != (8, dim, hidden)):
            raise ValueError("local expert weight shape mismatch")
        if any(weight.dtype != torch.bfloat16 or not weight.is_contiguous()
               for weight in (gate_weight, up_weight, down_weight)):
            raise ValueError("local expert weights must be contiguous BF16")
        act = torch.empty((rows, hidden), device=packed_x.device, dtype=packed_x.dtype)
        _grouped_gate_up_silu_kernel[(triton.cdiv(rows, 16), triton.cdiv(hidden, 64))](
            packed_x, gate_weight, up_weight, expert_offsets, act,
            rows, dim, hidden, 8, BLOCK_E=8,
            BM=16, BN=64, BK=32, num_warps=4, num_stages=1)
        output = torch.empty_like(packed_x)
        _grouped_down_kernel[(triton.cdiv(rows, 16), triton.cdiv(dim, 64))](
            act, down_weight, expert_offsets, output,
            rows, dim, hidden, 8, BLOCK_E=8,
            BM=16, BN=64, BK=32, num_warps=4, num_stages=1)
        # Checkpoint the compact packed input and route spans, not the enormous
        # [rows, 8832] activation.  Backward already needs gate/up recomputation
        # for the SiLU derivative, so retaining ACT consumed HBM without avoiding
        # its dominant recompute.
        ctx.save_for_backward(
            packed_x, expert_offsets, gate_weight, up_weight, down_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        packed_x, offsets, gate_weight, up_weight, down_weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        rows, dim = packed_x.shape
        hidden = gate_weight.shape[1]
        BM, BN, BK = 16, 32, 32
        # One checkpoint recomputation supplies both the down-weight input and
        # the gate/up values required by the SiLU VJP.
        gate = torch.empty((rows, hidden), device=packed_x.device, dtype=packed_x.dtype)
        up = torch.empty_like(gate)
        _grouped_gate_up_kernel[(triton.cdiv(rows, BM), triton.cdiv(hidden, 64))](
            packed_x, gate_weight, up_weight, offsets, gate, up,
            rows, dim, hidden, 8, BLOCK_E=8,
            BM=BM, BN=64, BK=32, num_warps=4, num_stages=1)
        act = torch.empty_like(gate)
        _silu_mul_kernel[(triton.cdiv(rows * hidden, 256),)](
            gate, up, act, rows * hidden, BLOCK=256, num_warps=4)
        grad_act = torch.empty_like(act)
        _grouped_down_input_backward_kernel[(triton.cdiv(rows, BM), triton.cdiv(hidden, BN))](
            grad_output, down_weight, offsets, grad_act,
            rows, dim, hidden, 8, BLOCK_E=8,
            BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
        grad_down = torch.empty_like(down_weight)
        _grouped_down_weight_backward_kernel[(8, triton.cdiv(dim, BN), triton.cdiv(hidden, BK))](
            grad_output, act, offsets, grad_down,
            rows, dim, hidden, 8, BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=1)
        grad_gate_values = torch.empty_like(gate)
        grad_up_values = torch.empty_like(up)
        _silu_backward_kernel[(triton.cdiv(rows * hidden, 256),)](
            gate, up, grad_act, grad_gate_values, grad_up_values,
            rows * hidden, BLOCK=256, num_warps=4)
        grad_x = torch.empty_like(packed_x)
        _grouped_gate_up_input_backward_kernel[(triton.cdiv(rows, BM), triton.cdiv(dim, BN))](
            grad_gate_values, grad_up_values, gate_weight, up_weight,
            offsets, grad_x, rows, dim, hidden, 8, BLOCK_E=8,
            BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
        grad_gate = torch.empty_like(gate_weight)
        grad_up = torch.empty_like(up_weight)
        _grouped_gate_up_weight_backward_kernel[(8, triton.cdiv(hidden, BN), triton.cdiv(dim, BK))](
            grad_gate_values, grad_up_values, packed_x, offsets,
            grad_gate, grad_up, rows, dim, hidden, 8,
            BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=1)
        return grad_x, None, grad_gate, grad_up, grad_down


def fused_packed_local_experts_autograd(
    packed_x: torch.Tensor,
    expert_offsets: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Run fused forward/backward for the eight experts owned by one GCD."""
    return _FusedPackedLocalExpertsFunction.apply(
        packed_x, expert_offsets, gate_weight, up_weight, down_weight)


class _FusedSharedRoutedSwiGLUParityFunction(torch.autograd.Function):
    """Custom autograd bridge kept private until all runtime gates are complete."""

    @staticmethod
    def forward(ctx, x, router_weight, routed_gate_weight, routed_up_weight,
                routed_down_weight, shared_gate_weight, shared_up_weight,
                shared_down_weight):
        ctx.set_materialize_grads(False)
        result = fused_routed_swiglu_forward(
            x, router_weight, routed_gate_weight, routed_up_weight, routed_down_weight,
            shared_gate_weight=shared_gate_weight,
            shared_up_weight=shared_up_weight,
            shared_down_weight=shared_down_weight,
            _capture_intermediates=True,
        )
        ctx.result = result
        ctx.save_for_backward(
            x, router_weight, routed_gate_weight, routed_up_weight,
            routed_down_weight, shared_gate_weight, shared_up_weight,
            shared_down_weight,
        )
        return result.output, result.load_balance_loss, result.z_loss

    @staticmethod
    def backward(ctx, grad_output, grad_load_balance, grad_z):
        tensors = ctx.saved_tensors
        x = tensors[0]
        if grad_output is None:
            grad_output = torch.zeros_like(x)
        if grad_load_balance is None:
            grad_load_balance = torch.zeros((), device=x.device, dtype=torch.float32)
        if grad_z is None:
            grad_z = torch.zeros((), device=x.device, dtype=torch.float32)
        grad_aux = torch.empty(2, device=x.device, dtype=torch.float32)
        _pack_aux_grad_kernel[(1,)](grad_load_balance, grad_z, grad_aux, num_warps=1)
        return fused_shared_routed_swiglu_backward(
            grad_output.contiguous(), ctx.result, *tensors, grad_aux=grad_aux)


def fused_shared_routed_swiglu_autograd_parity(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_gate_weight: torch.Tensor,
    routed_up_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exercise the real custom-autograd path without opening the training gate."""
    return _FusedSharedRoutedSwiGLUParityFunction.apply(
        x, router_weight, routed_gate_weight, routed_up_weight,
        routed_down_weight, shared_gate_weight, shared_up_weight,
        shared_down_weight,
    )


def fused_schedulefree_lerp_(parameter: torch.Tensor, z: torch.Tensor, *,
                             weight: float) -> None:
    """Fused ScheduleFree train/eval basis interpolation."""
    _require_hip_triton(parameter)
    if (z.shape != parameter.shape or z.dtype != parameter.dtype or
            not z.is_contiguous() or not parameter.is_contiguous()):
        raise ValueError("parameter and z must be matching contiguous tensors")
    if not -1.0 <= weight <= 1.0:
        raise ValueError("ScheduleFree interpolation weight must be in [-1, 1]")
    count = parameter.numel()
    _schedulefree_lerp_kernel[(triton.cdiv(count, 256),)](
        parameter, z, count, WEIGHT=float(weight), BLOCK=256, num_warps=4)


def fused_schedulefree_adamw_update_(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    z: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    bias_correction2: float,
    eps: float,
    weight_decay: float,
    ckp1: float,
) -> None:
    """Fused in-place ScheduleFree AdamW tensor update without master weights."""
    _require_hip_triton(parameter)
    if parameter.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("fused ScheduleFree state supports BF16 or FP32 tensors")
    if any(tensor.dtype != parameter.dtype or tensor.shape != parameter.shape or
           not tensor.is_contiguous() for tensor in (grad, z, exp_avg_sq)):
        raise ValueError("parameter, grad, z, and exp_avg_sq must be matching contiguous tensors")
    if not parameter.is_contiguous():
        raise ValueError("parameter must be contiguous")
    if bias_correction2 <= 0 or lr < 0 or eps < 0 or not 0 <= ckp1 <= 1:
        raise ValueError("invalid ScheduleFree scalar state")
    count = parameter.numel()
    _schedulefree_adamw_update_kernel[(triton.cdiv(count, 256),)](
        parameter, grad, z, exp_avg_sq, count,
        LR=float(lr), BETA1=float(beta1), BETA2=float(beta2),
        BIAS_CORRECTION2=float(bias_correction2), EPS=float(eps),
        WEIGHT_DECAY=float(weight_decay), CKP1=float(ckp1),
        BLOCK=256, num_warps=4)


def fused_shared_routed_swiglu_forward(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_gate_weight: torch.Tensor,
    routed_up_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    *,
    require_training: bool = False,
) -> FusedMoEForwardResult:
    """Execute the required one-shared plus dropless top-3 E97 MoE forward."""
    if router_weight.ndim != 2 or router_weight.shape[0] != 64:
        raise ValueError("production E97 MoE requires exactly 64 routed experts")
    if routed_gate_weight.ndim != 3 or routed_gate_weight.shape[0] != 64:
        raise ValueError("production E97 MoE requires 64 packed routed expert weights")
    return fused_routed_swiglu_forward(
        x, router_weight, routed_gate_weight, routed_up_weight, routed_down_weight,
        shared_gate_weight=shared_gate_weight,
        shared_up_weight=shared_up_weight,
        shared_down_weight=shared_down_weight,
        require_training=require_training,
    )
