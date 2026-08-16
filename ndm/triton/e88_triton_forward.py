"""Triton forward kernel for the E88/NDM matrix-state recurrence.

This is the portable Triton implementation of the production E88/NDM recurrence
used by ``ndm.models.e88_fla_hybrid``. The paired backward kernel lives in
``ndm.triton.e88_triton_backward``. The PyTorch reference below defines the
same recurrence for correctness tests.

Per timestep, for each (batch, head):
    r_t = S_{t-1}.T @ k_t                # retrieve  [V]
    delta_t = v_t - r_t                  # [V]
    S_t = tanh(decay_t * S_{t-1} + outer(delta_t, k_t))   # [N, V]
    out_t = S_t.T @ q_t                  # [V]

Shapes (matching the CUDA kernel layout — caller is responsible for the
[B, T, ...] -> [T, B, ...] transpose):
    k, q:      [T, B, H, N]    bf16 or fp32
    v:         [T, B, H, V]
    decay:     [T, B, H]
    S0:        [B, H, N, V]
    output:    [T, B, H, V]
    S_final:   [B, H, N, V]
    S_checkpoints: [num_ckpts, B, H, N, V]   (sparse: every CKPT_INTERVAL steps)

Sparse forward checkpointing
----------------------------
To shrink the S_checkpoints buffer from O(T) to O(T/K) tiles, we save S
only every CKPT_INTERVAL steps (default K=16, mirroring CUDA reg-own).

Layout:
    S_ckpt[0]                    = initial S0
    S_ckpt[k] for k >= 1         = S after step (k * CKPT_INTERVAL - 1)

So segment ``seg`` (covering t in [seg*K, seg*K + K)) starts from
S_ckpt[seg], i.e. S_ckpt[seg] is the state right BEFORE step seg*K.

We currently REQUIRE T % CKPT_INTERVAL == 0 — the unaligned case can be
added later (just needs a shorter final segment in the backward replay).

Number of slots: ``num_ckpts = T // CKPT_INTERVAL + 1``  (e.g. T=512 K=16
gives 33 slots vs 513 in the dense layout — ~16x memory shrink).

The kernel currently supports N <= 64 and V <= 64 (sufficient for the
E88 configurations actually used in benchmarks: n_state in {16, 32}).

Multi-program tiling
--------------------
At high head counts (H >= ~256) we run out of useful (B, H) parallelism
(too many tiny programs, all competing for SMs and bandwidth). To fix
this, each program handles ``BLOCK_H`` consecutive heads at once:

  state tile:   [BLOCK_H, BLOCK_N, BLOCK_V]
  per-step:     loaded k/v/q/decay vectorize across BLOCK_H

Grid: (B, ceil(H / BLOCK_H)).  BLOCK_H is autotuned over {1, 2, 4, 8}.
"""
from __future__ import absolute_import

from typing import Tuple

import torch
import triton
import triton.language as tl


# Default checkpoint stride. Matches the CUDA register-owned kernel
# (``E88_REG_CHECKPOINT_INTERVAL = 16``) so that the Python-allocated
# S_checkpoints tensor has the same number of slots as the CUDA path.
DEFAULT_CKPT_INTERVAL = 16


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _e88_forward_kernel(
    # Inputs
    K_ptr,            # [T, B, H, N]
    V_ptr,            # [T, B, H, V]
    Q_ptr,            # [T, B, H, N]
    D_ptr,            # [T, B, H]
    S0_ptr,           # [B, H, N, V]
    G_ptr,            # [T, B, H, V] gate (used iff APPLY_GATE)
    E_ptr,            # [T, B, H, N] erase/read gate (used iff SPLIT_EDIT)
    W_ptr,            # [T, B, H, V] value write gate (used iff SPLIT_EDIT)
    # Outputs
    Out_ptr,          # [T, B, H, V]
    Sfinal_ptr,       # [B, H, N, V]
    Sckpt_ptr,        # [num_ckpts, B, H, N, V]
    # Strides (in elements). All tensors are made contiguous before launch,
    # so we only need one stride argument set per tensor's outer dim, but we
    # pass the full set explicitly to keep the kernel self-documenting and
    # not require contiguity assumptions in the future.
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sd_t, sd_b, sd_h,
    s0_b, s0_h, s0_n, s0_v,
    sg_t, sg_b, sg_h, sg_v,  # gate strides (zero/dummy when APPLY_GATE=False)
    se_t, se_b, se_h, se_n,  # erase/read gate strides (zero/dummy when SPLIT_EDIT=False)
    sw_t, sw_b, sw_h, sw_v,  # value write gate strides (zero/dummy when SPLIT_EDIT=False)
    so_t, so_b, so_h, so_v,
    sf_b, sf_h, sf_n, sf_v,
    sc_t, sc_b, sc_h, sc_n, sc_v,
    # Sizes
    VALID_T,  # real-token boundary; may be shorter than padded constexpr T
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CKPT_INTERVAL: tl.constexpr,
    APPLY_GATE: tl.constexpr,  # if True, output = silu(g) * S^T@q
    NORMALIZE_KQ: tl.constexpr,  # if True, L2-normalize k and q on load (per head, last dim)
    APPLY_SILU_QKV: tl.constexpr,  # if True, apply silu to raw q/k/v projection loads
    RAW_WRITE: tl.constexpr,  # if True, ablate delta correction: delta = v
    LINEAR_STATE: tl.constexpr,  # if True, ablate tanh: S = pre
    SPLIT_EDIT: tl.constexpr,  # if True, use E97 b/w edit gates
):
    """One program per (batch, head_block). Sequential time loop."""
    # 2D launch grid: (B, ceil(H / BLOCK_H))
    b = tl.program_id(0).to(tl.int64)
    hg = tl.program_id(1).to(tl.int64)

    h_start = hg * BLOCK_H
    h_idx = h_start + tl.arange(0, BLOCK_H)            # [BLOCK_H]
    h_mask = h_idx < H                                  # [BLOCK_H]

    # We assume N <= BLOCK_N and V <= BLOCK_V. Mask off padded N/V lanes.
    n_idx = tl.arange(0, BLOCK_N)
    v_idx = tl.arange(0, BLOCK_V)
    n_mask = n_idx < N
    v_mask = v_idx < V

    # Composite masks for state and intermediates
    mask_hnv = (h_mask[:, None, None] & n_mask[None, :, None] & v_mask[None, None, :])
    mask_hn = h_mask[:, None] & n_mask[None, :]
    mask_hv = h_mask[:, None] & v_mask[None, :]

    # Load initial state S as [BLOCK_H, BLOCK_N, BLOCK_V] tile.
    s0_off = (
        b * s0_b
        + h_idx[:, None, None] * s0_h
        + n_idx[None, :, None] * s0_n
        + v_idx[None, None, :] * s0_v
    )
    S = tl.load(S0_ptr + s0_off, mask=mask_hnv, other=0.0).to(tl.float32)

    # Save S0 into checkpoint slot 0.
    sc0_off = (
        0 * sc_t + b * sc_b
        + h_idx[:, None, None] * sc_h
        + n_idx[None, :, None] * sc_n
        + v_idx[None, None, :] * sc_v
    )
    tl.store(Sckpt_ptr + sc0_off, S.to(Sckpt_ptr.dtype.element_ty), mask=mask_hnv)

    # Sequential time loop.
    # NOTE: we cast (t+1) to int64 below for offset arithmetic on
    # potentially very large tensors (e.g. S_ckpt at T=4K B=2 H=386
    # N=V=32 — even the SPARSE layout is large enough to need int64 once
    # B*H*N*V exceeds ~500M elements).
    for t in range(T):
        # Load k_t, q_t, v_t, decay_t for all BLOCK_H heads in this block.
        # Use int64 for the t-stride product so we don't overflow at large
        # T*B*H*N (input tensors can be ~6 GB at production scale).
        t_i64 = tl.full([1], t, dtype=tl.int64)
        # k, q: [BLOCK_H, BLOCK_N]
        k_off = (
            t_i64 * sk_t + b * sk_b
            + h_idx[:, None] * sk_h
            + n_idx[None, :] * sk_n
        )
        q_off = (
            t_i64 * sq_t + b * sq_b
            + h_idx[:, None] * sq_h
            + n_idx[None, :] * sq_n
        )
        # v: [BLOCK_H, BLOCK_V]
        v_off = (
            t_i64 * sv_t + b * sv_b
            + h_idx[:, None] * sv_h
            + v_idx[None, :] * sv_v
        )
        # decay: [BLOCK_H]
        d_off = t_i64 * sd_t + b * sd_b + h_idx * sd_h

        k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH, BN]
        q_vec = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)   # [BH, BN]
        v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)   # [BH, BV]
        d_val = tl.load(D_ptr + d_off, mask=h_mask, other=0.0).to(tl.float32)    # [BH]

        if APPLY_SILU_QKV:
            k_vec = k_vec / (1.0 + tl.exp(-k_vec))
            q_vec = q_vec / (1.0 + tl.exp(-q_vec))
            v_vec = v_vec / (1.0 + tl.exp(-v_vec))

        # Optional fused L2 normalization on k and q (per head, last-dim).
        # Saves two PyTorch ops per (k, q) per layer call (norm, divide):
        # at depth=14 with grad_ckpt this is ~50-60 ms/step at 1.27B.
        # Matches CUDA reg-own's `normalize_kq=True` path.
        if NORMALIZE_KQ:
            k_norm_sq = tl.sum(k_vec * k_vec, axis=1)            # [BH]
            q_norm_sq = tl.sum(q_vec * q_vec, axis=1)            # [BH]
            inv_k_norm = 1.0 / (tl.sqrt(k_norm_sq) + 1e-6)        # [BH]
            inv_q_norm = 1.0 / (tl.sqrt(q_norm_sq) + 1e-6)        # [BH]
            k_vec = k_vec * inv_k_norm[:, None]
            q_vec = q_vec * inv_q_norm[:, None]

        if SPLIT_EDIT:
            e_off = (
                t_i64 * se_t + b * se_b
                + h_idx[:, None] * se_h
                + n_idx[None, :] * se_n
            )
            w_off = (
                t_i64 * sw_t + b * sw_b
                + h_idx[:, None] * sw_h
                + v_idx[None, :] * sw_v
            )
            e_vec = tl.load(E_ptr + e_off, mask=mask_hn, other=0.0).to(tl.float32)
            w_vec = tl.load(W_ptr + w_off, mask=mask_hv, other=0.0).to(tl.float32)
            read_key = k_vec * e_vec
            write_value = v_vec * w_vec
        else:
            read_key = k_vec
            write_value = v_vec

        # Retrieve: r[h, v] = sum_n S[h, n, v] * read_key[h, n]
        if RAW_WRITE:
            delta = write_value
        else:
            retrieved = tl.sum(S * read_key[:, :, None], axis=1)   # [BH, BV]
            delta = write_value - retrieved                        # [BH, BV]

        # Update: standard E88 uses tanh; linear-state ablation keeps pre.
        outer = k_vec[:, :, None] * delta[:, None, :]       # [BH, BN, BV]
        pre = d_val[:, None, None] * S + outer
        if LINEAR_STATE:
            S = pre
        else:
            # Stable tanh. The raw exp formula overflows for pre > ~44 in fp32,
            # yielding inf/inf = NaN. sigmoid saturates without forming inf/inf.
            S = 2.0 * tl.sigmoid(2.0 * pre) - 1.0

        # Output: out[h, v] = sum_n S[h, n, v] * q[h, n]   (reduce over N axis=1)
        out_vec = tl.sum(S * q_vec[:, :, None], axis=1)     # [BH, BV]

        # Optional fused output gate: output = silu(g) * out_vec.
        # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
        # We fuse here so the gate doesn't become two extra kernel
        # launches (silu, multiply) per layer in Python.
        if APPLY_GATE:
            g_off = (
                t_i64 * sg_t + b * sg_b
                + h_idx[:, None] * sg_h
                + v_idx[None, :] * sg_v
            )
            g_val = tl.load(G_ptr + g_off, mask=mask_hv, other=0.0).to(tl.float32)
            silu_g = g_val / (1.0 + tl.exp(-g_val))
            out_vec = silu_g * out_vec

        out_off = (
            t_i64 * so_t + b * so_b
            + h_idx[:, None] * so_h
            + v_idx[None, :] * so_v
        )
        tl.store(Out_ptr + out_off, out_vec.to(Out_ptr.dtype.element_ty), mask=mask_hv)

        # Stateful inference may right-pad an unaligned sequence to the sparse
        # checkpoint interval. Capture the state after the last real token,
        # before nonlinear zero-padding steps mutate it.
        sf_off = (
            b * sf_b
            + h_idx[:, None, None] * sf_h
            + n_idx[None, :, None] * sf_n
            + v_idx[None, None, :] * sf_v
        )
        tl.store(
            Sfinal_ptr + sf_off, S.to(Sfinal_ptr.dtype.element_ty),
            mask=mask_hnv & ((t + 1) == VALID_T))

        # Sparse checkpoint write: only when (t+1) is a multiple of
        # CKPT_INTERVAL. The wrapper enforces T % CKPT_INTERVAL == 0, so
        # the last step (t = T-1) is always a checkpoint.
        # Slot index: (t+1) // CKPT_INTERVAL — that is, S_ckpt[k] holds
        # S after step (k*CKPT_INTERVAL - 1) for k>=1.
        is_ckpt_step = ((t + 1) % CKPT_INTERVAL) == 0
        if is_ckpt_step:
            slot_i64 = tl.full([1], (t + 1) // CKPT_INTERVAL, dtype=tl.int64)
            sc_off = (
                slot_i64 * sc_t + b * sc_b
                + h_idx[:, None, None] * sc_h
                + n_idx[None, :, None] * sc_n
                + v_idx[None, None, :] * sc_v
            )
            tl.store(Sckpt_ptr + sc_off, S.to(Sckpt_ptr.dtype.element_ty), mask=mask_hnv)



# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(p, 16)


# Lightweight in-process autotune cache: maps (B, T, H, N, V, dtype) -> best
# (BLOCK_H, num_warps).
_AUTOTUNE_CACHE = {}

# BLOCK_H candidates. Capped so the [BLOCK_H, BLOCK_N, BLOCK_V] state tile
# stays in registers/SRAM. For BLOCK_N=BLOCK_V=32 and fp32, tile=4 KB per
# BLOCK_H slot. BLOCK_H=8 is 32 KB; BLOCK_H=16 with BLOCK_N=BLOCK_V=64 is
# 256 KB which spills, so we cap depending on N*V.
_BLOCK_H_CANDIDATES = (1, 2, 4, 8, 16)


def _select_block_h_candidates(N: int, V: int):
    """Return BLOCK_H values that won't blow SRAM for this (N, V)."""
    elem = _next_pow2(N) * _next_pow2(V)
    if elem >= 64 * 64:        # 4096 elements
        return (1, 2, 4)
    if elem >= 32 * 64:        # 2048 elements
        return (1, 2, 4, 8)
    return _BLOCK_H_CANDIDATES  # up to 16 for small (N, V)


def _autotune_kernel(
    launch_args,
    B,
    T,
    H,
    N,
    Vsz,
    dtype,
    ckpt_interval,
    normalize_kq,
    apply_silu_qkv,
    raw_write,
    linear_state,
    split_edit,
    valid_length,
):
    """Tiny in-process autotune. Tries (BLOCK_H, num_warps) and caches winner.

    Empirically, the "right" num_warps depends on BLOCK_H AND on H itself:
    at high H (many programs in flight) fewer warps per program reduces
    occupancy contention; at low H (few programs) more warps fills the SM.

    For H < 16 we just default to BLOCK_H=1 (no head-grouping helps when
    there are too few heads to begin with).
    """
    cache_key = (
        B, T, H, N, Vsz, str(dtype), ckpt_interval, bool(normalize_kq),
        bool(apply_silu_qkv), bool(raw_write), bool(linear_state),
        bool(split_edit),
    )
    if cache_key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[cache_key]

    if H < 16:
        cfg = (1, 4)
        _AUTOTUNE_CACHE[cache_key] = cfg
        return cfg

    bh_candidates = _select_block_h_candidates(N, Vsz)
    # Restrict to BHs that don't grossly oversubscribe a SM. BLOCK_H=1
    # never needs more than 4 warps; BLOCK_H=8 with 1 warp tends to
    # generate massive spills (observed 13 ms outliers in sweep), so we
    # exclude 1-warp configs for BLOCK_H >= 8.
    nw_candidates_for = {
        1: (2, 4),
        2: (2, 4),
        4: (2, 4),
        8: (2, 4),
        16: (4, 8),
    }

    import time as _time
    best = None
    best_t = float("inf")

    (
        k_c, v_c, q_c, d_c, s0_c, g_c, e_c, w_c, apply_gate, out,
        S_final, S_ckpt, strides,
    ) = launch_args

    for bh in bh_candidates:
        if bh > H:
            continue
        for nw in nw_candidates_for[bh]:
            try:
                grid = (B, (H + bh - 1) // bh)
                # Warmup
                for _ in range(3):
                    _e88_forward_kernel[grid](
                        k_c, v_c, q_c, d_c, s0_c, g_c, e_c, w_c,
                        out, S_final, S_ckpt,
                        *strides,
                        T=T, VALID_T=valid_length, B=B, H=H, N=N, V=Vsz,
                        BLOCK_N=_next_pow2(N), BLOCK_V=_next_pow2(Vsz),
                        BLOCK_H=bh,
                        CKPT_INTERVAL=ckpt_interval,
                        APPLY_GATE=apply_gate,
                        NORMALIZE_KQ=bool(normalize_kq),
                        APPLY_SILU_QKV=bool(apply_silu_qkv),
                        RAW_WRITE=bool(raw_write),
                        LINEAR_STATE=bool(linear_state),
                        SPLIT_EDIT=bool(split_edit),
                        num_warps=nw,
                    )
                torch.cuda.synchronize()
                t0 = _time.perf_counter()
                iters = 5
                for _ in range(iters):
                    _e88_forward_kernel[grid](
                        k_c, v_c, q_c, d_c, s0_c, g_c, e_c, w_c,
                        out, S_final, S_ckpt,
                        *strides,
                        T=T, VALID_T=valid_length, B=B, H=H, N=N, V=Vsz,
                        BLOCK_N=_next_pow2(N), BLOCK_V=_next_pow2(Vsz),
                        BLOCK_H=bh,
                        CKPT_INTERVAL=ckpt_interval,
                        APPLY_GATE=apply_gate,
                        NORMALIZE_KQ=bool(normalize_kq),
                        APPLY_SILU_QKV=bool(apply_silu_qkv),
                        RAW_WRITE=bool(raw_write),
                        LINEAR_STATE=bool(linear_state),
                        SPLIT_EDIT=bool(split_edit),
                        num_warps=nw,
                    )
                torch.cuda.synchronize()
                t = (_time.perf_counter() - t0) / iters
                if t < best_t:
                    best_t = t
                    best = (bh, nw)
            except Exception:
                # OOM / SRAM exceeded for this config — skip.
                continue

    if best is None:
        best = (1, 4)
    _AUTOTUNE_CACHE[cache_key] = best
    return best


def e88_triton_forward(
    S0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    head_v_dim: int = None,
    block_h: int = None,
    num_warps: int = None,
    ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
    g: torch.Tensor = None,  # [T, B, H, V] gate; if None, no fused gate
    normalize_kq: bool = False,  # if True, kernel L2-normalizes k and q on load
    apply_silu_qkv: bool = False,  # if True, kernel applies silu to raw q/k/v loads
    raw_write: bool = False,  # if True, use raw v write instead of delta correction
    linear_state: bool = False,  # if True, drop tanh and use the raw preactivation state
    erase_gate: torch.Tensor = None,  # [T, B, H, N] E97 read/erase gate
    value_write_gate: torch.Tensor = None,  # [T, B,H,V] E97 value write gate
    valid_length: int = None,  # state boundary before inference-only padding
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the E88 forward recurrence in Triton.

    Args:
        S0:    [B, H, N, V] initial state (matches CUDA kernel).
        k, q:  [T, B, H, N] keys / queries.
        v:     [T, B, H, V] values.
        decay: [T, B, H] per-step decay scalars.
        head_v_dim: ignored, present for API compatibility with CUDA wrapper.
        block_h: optional override for BLOCK_H (heads per program).
                 If None, autotune picks the best for this shape.
        num_warps: optional override; default chosen with block_h.
        ckpt_interval: store S to S_checkpoints every K steps. Default 16
                 (matches CUDA register-owned). Currently REQUIRES
                 T % ckpt_interval == 0.

    Returns:
        out:           [T, B, H, V]
        S_final:       [B, H, N, V]
        S_checkpoints: [num_ckpts, B, H, N, V] — sparse checkpoint
                       layout. ``num_ckpts = T // ckpt_interval + 1``.
                       S_ckpt[0] = S0,
                       S_ckpt[k] = S after step (k*ckpt_interval - 1) for k>=1.

    The output dtype matches the input k/v/q dtype (bf16 -> bf16, fp32 ->
    fp32). All accumulation inside the kernel is fp32.
    """
    assert k.is_cuda, "Inputs must be on CUDA"
    assert k.dim() == 4 and q.dim() == 4 and v.dim() == 4, \
        "k/v/q must be [T, B, H, *]"
    assert decay.dim() == 3, "decay must be [T, B, H]"
    assert S0.dim() == 4, "S0 must be [B, H, N, V]"

    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    valid_length = T if valid_length is None else int(valid_length)
    if not 0 < valid_length <= T:
        raise ValueError(
            f"valid_length must be in [1, {T}], got {valid_length}")
    assert q.shape == (T, B, H, N)
    assert v.shape == (T, B, H, Vsz)
    assert decay.shape == (T, B, H)
    assert S0.shape == (B, H, N, Vsz)

    BLOCK_N = _next_pow2(N)
    BLOCK_V = _next_pow2(Vsz)
    if BLOCK_N > 64 or BLOCK_V > 64:
        raise NotImplementedError(
            f"e88_triton_forward currently supports N, V <= 64 "
            f"(got N={N}, V={Vsz}). Larger sizes need a tiled implementation."
        )

    if T % ckpt_interval != 0:
        raise NotImplementedError(
            f"Sparse-checkpoint forward currently requires T % ckpt_interval == 0 "
            f"(got T={T}, ckpt_interval={ckpt_interval}). The unaligned case can "
            f"be added by handling a shorter final segment in the backward replay."
        )

    # The kernel reads via explicit strides, so non-contiguous inputs
    # (e.g. transposes from [B, T, H, *] -> [T, B, H, *]) work fine as
    # long as the last dim is contiguous (stride[-1] == 1). We only
    # call .contiguous() on the *last-dim* axis when needed; otherwise
    # we keep the view to avoid expensive copies. At production scale
    # (B=8, T=512, H=386, N=V=32) each .contiguous() is ~100 MB —
    # 14 layers * 3 invocations (fwd+fwd_replay+bwd) * 4 tensors saved
    # is tens of GB of bandwidth per training step.
    def _strided_ok(x):
        return x.stride(-1) == 1
    k_c = k if _strided_ok(k) else k.contiguous()
    v_c = v if _strided_ok(v) else v.contiguous()
    q_c = q if _strided_ok(q) else q.contiguous()
    d_c = decay if _strided_ok(decay) else decay.contiguous()
    s0_c = S0 if _strided_ok(S0) else S0.contiguous()

    apply_gate = g is not None
    if apply_gate:
        g_c = g if _strided_ok(g) else g.contiguous()
        assert g_c.shape == (T, B, H, Vsz), \
            f"gate shape must be [T, B, H, V] = {(T, B, H, Vsz)}, got {tuple(g_c.shape)}"
        g_strides = (g_c.stride(0), g_c.stride(1), g_c.stride(2), g_c.stride(3))
    else:
        # Pass a dummy pointer + zero strides; kernel guards via APPLY_GATE constexpr.
        g_c = k_c  # any valid CUDA tensor
        g_strides = (0, 0, 0, 0)

    split_edit = erase_gate is not None or value_write_gate is not None
    if split_edit:
        assert erase_gate is not None and value_write_gate is not None, \
            "erase_gate and value_write_gate must be provided together"
        e_c = erase_gate if _strided_ok(erase_gate) else erase_gate.contiguous()
        w_c = value_write_gate if _strided_ok(value_write_gate) else value_write_gate.contiguous()
        assert e_c.shape == (T, B, H, N), \
            f"erase_gate shape must be [T, B, H, N] = {(T, B, H, N)}, got {tuple(e_c.shape)}"
        assert w_c.shape == (T, B, H, Vsz), \
            f"value_write_gate shape must be [T, B, H, V] = {(T, B, H, Vsz)}, got {tuple(w_c.shape)}"
        e_strides = (e_c.stride(0), e_c.stride(1), e_c.stride(2), e_c.stride(3))
        w_strides = (w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3))
    else:
        e_c = k_c
        w_c = v_c
        e_strides = (0, 0, 0, 0)
        w_strides = (0, 0, 0, 0)

    out_dtype = k_c.dtype
    out = torch.empty((T, B, H, Vsz), dtype=out_dtype, device=k.device)
    S_final = torch.empty_like(s0_c)

    num_ckpts = T // ckpt_interval + 1
    S_ckpt = torch.empty((num_ckpts, B, H, N, Vsz), dtype=out_dtype, device=k.device)

    strides = (
        # k strides
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        # v strides
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        # q strides
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        # decay strides
        d_c.stride(0), d_c.stride(1), d_c.stride(2),
        # S0 strides
        s0_c.stride(0), s0_c.stride(1), s0_c.stride(2), s0_c.stride(3),
        # gate strides
        *g_strides,
        # split edit gate strides
        *e_strides,
        *w_strides,
        # out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # S_final strides
        S_final.stride(0), S_final.stride(1), S_final.stride(2), S_final.stride(3),
        # S_ckpt strides
        S_ckpt.stride(0), S_ckpt.stride(1), S_ckpt.stride(2),
        S_ckpt.stride(3), S_ckpt.stride(4),
    )

    # Pick BLOCK_H + num_warps.
    if block_h is None:
        # BLOCK_H=1 is always best at H >= 64 (BLOCK_H>1 spills the
        # [BH, N, V] register state). After fusing gate + L2-norm into
        # the kernel, the per-step register pressure goes UP, and
        # nw=1 (CUDA-reg-own's design philosophy) wins or ties at every
        # shape:
        #   B=1 T=512   (B*H=386):   nw=1: 0.66 ms,  nw=2: 0.66 ms (tie)
        #   B=2 T=4K    (B*H=772):   nw=1: 7.85 ms,  nw=2: 8.18 ms
        #   B=4 T=2K    (B*H=1544):  nw=1: 4.22 ms,  nw=2: 4.76 ms
        #   B=8 T=512   (B*H=3088):  nw=1: 1.88 ms,  nw=2: 2.10 ms
        #   B=1 T=16K   (B*H=386):   nw=1: 24.3 ms,  nw=2: 24.5 ms
        # so we always use nw=1 at H>=64.
        if H >= 64:
            block_h_chosen = 1
            nw = 1
        else:
            launch_args = (
                k_c, v_c, q_c, d_c, s0_c, g_c, e_c, w_c, apply_gate,
                out, S_final, S_ckpt, strides,
            )
            block_h_chosen, nw = _autotune_kernel(
                launch_args, B, T, H, N, Vsz, out_dtype, ckpt_interval,
                normalize_kq, apply_silu_qkv, raw_write, linear_state,
                split_edit, valid_length,
            )
    else:
        block_h_chosen = int(block_h)
        nw = int(num_warps) if num_warps is not None else (8 if block_h_chosen >= 8 else 4)

    grid = (B, (H + block_h_chosen - 1) // block_h_chosen)

    _e88_forward_kernel[grid](
        k_c, v_c, q_c, d_c, s0_c, g_c, e_c, w_c,
        out, S_final, S_ckpt,
        *strides,
        T=T, VALID_T=valid_length, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V,
        BLOCK_H=block_h_chosen,
        CKPT_INTERVAL=ckpt_interval,
        APPLY_GATE=apply_gate,
        NORMALIZE_KQ=bool(normalize_kq),
        APPLY_SILU_QKV=bool(apply_silu_qkv),
        RAW_WRITE=bool(raw_write),
        LINEAR_STATE=bool(linear_state),
        SPLIT_EDIT=bool(split_edit),
        num_warps=nw,
    )
    return out, S_final, S_ckpt


# ---------------------------------------------------------------------------
# PyTorch reference (mirrors the slow fallback in e88_fla_hybrid.py)
# ---------------------------------------------------------------------------

def e88_torch_reference(
    S0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    linear_state: bool = False,
    raw_write: bool = False,
    erase_gate: torch.Tensor = None,
    value_write_gate: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for parity testing.

    Returns the DENSE checkpoint history (one slot per timestep, plus
    initial). Tests that compare against this reference need to subsample
    when comparing to the sparse kernel checkpoint.

    Inputs use the SAME [T, B, H, ...] convention as the Triton wrapper
    (i.e., the CUDA kernel convention), to keep the test simple. The
    PyTorch fallback inside e88_fla_hybrid.py uses [B, T, H, ...] but the
    math is identical; we just iterate the time axis differently.

    Args:
        S0:    [B, H, N, V]
        k, q:  [T, B, H, N]
        v:     [T, B, H, V]
        decay: [T, B, H]
        linear_state: if True, drop the tanh (matches `linear_state=True`
                      in e88_fla_hybrid.py).

    Returns:
        out:           [T, B, H, V]
        S_final:       [B, H, N, V]
        ckpt:          [T+1, B, H, N, V] — DENSE per-step history.
                       ckpt[0]=S0, ckpt[t+1]=S after step t.
    """
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    out_dtype = k.dtype

    S = S0.clone().to(torch.float32)
    out = torch.empty((T, B, H, Vsz), dtype=out_dtype, device=k.device)
    ckpt = torch.empty((T + 1, B, H, N, Vsz), dtype=out_dtype, device=k.device)
    ckpt[0] = S.to(out_dtype)

    for t in range(T):
        k_t = k[t].to(torch.float32)        # [B, H, N]
        q_t = q[t].to(torch.float32)        # [B, H, N]
        v_t = v[t].to(torch.float32)        # [B, H, V]
        d_t = decay[t].to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        if erase_gate is not None or value_write_gate is not None:
            if erase_gate is None or value_write_gate is None:
                raise ValueError("erase_gate and value_write_gate must be provided together")
            read_key = k_t * erase_gate[t].to(torch.float32)
            write_value = v_t * value_write_gate[t].to(torch.float32)
        else:
            read_key = k_t
            write_value = v_t

        # retrieve: S [B,H,N,V], read_key [B,H,N] -> [B, H, V]
        if raw_write:
            delta = write_value
        else:
            retrieved = torch.einsum('bhnv,bhn->bhv', S, read_key)
            delta = write_value - retrieved

        outer = torch.einsum('bhn,bhv->bhnv', k_t, delta)

        pre = d_t * S + outer
        if linear_state:
            S = pre
        else:
            S = torch.tanh(pre)

        Sq = torch.einsum('bhnv,bhn->bhv', S, q_t)
        out[t] = Sq.to(out_dtype)
        ckpt[t + 1] = S.to(out_dtype)

    S_final = S.to(out_dtype)
    return out, S_final, ckpt
