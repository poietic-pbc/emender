"""Triton backward kernel for E88 FLA-Hybrid recurrence.

Sparse-checkpoint design (matches CUDA register-owned):
  - The forward stores S only every CKPT_INTERVAL=K steps, into
    ``S_ckpt`` of shape [num_ckpts, B, H, N, V] with num_ckpts = T/K + 1.
    S_ckpt[seg] holds the state right BEFORE step seg*K (so S_ckpt[0]=S0
    and S_ckpt[num_ckpts-1] = S after step T-1).
  - The backward processes one segment at a time, in reverse order.
    Per segment:
      1. Load S_seg_start = S_ckpt[seg].
      2. Forward-replay through the K steps of the segment, caching the
         pre-update state at each step into a per-program scratch buffer.
      3. Walk the K steps in reverse, using the scratched S_{t-1} and
         re-derived S_t to apply the same chain rule as the dense
         backward kernel. Update dS_carry across steps.
  - After the outermost (seg=0) segment finishes, dS_carry is the
    gradient w.r.t. S0.

Forward recurrence (per (b, h)):
    r_t       = S_{t-1}.T @ k_t                     # [V]
    delta_t   = v_t - r_t
    pre_t     = decay_t * S_{t-1} + outer(k_t, delta_t)
    S_t       = tanh(pre_t)                          # [N, V]
    out_t     = S_t.T @ q_t                          # [V]

Backward (let upstream gradient be d_out, d_S_final):
    dS_t          = (carry from t+1) + outer(q_t, d_out_t)
    d_q_t         = S_t @ d_out_t
    d_pre_t       = dS_t * (1 - S_t**2)
    d_decay_t     = sum_{n,v} d_pre_t * S_{t-1}
    d_outer_t     = d_pre_t                          # [N, V]
    d_k_t (from outer)     = sum_v delta_t * d_outer_t
    d_delta_t              = sum_n k_t  * d_outer_t  # [V]
    d_v_t                  = d_delta_t
    d_retrieve_t           = -d_delta_t
    d_k_t (from retrieve)  = sum_v S_{t-1} * d_retrieve_t
    dS_{t-1}_from_decay    = decay_t * d_pre_t
    dS_{t-1}_from_retrieve = outer(k_t, d_retrieve_t)
    carry        = dS_{t-1}_from_decay + dS_{t-1}_from_retrieve

Shapes (matching the forward kernel layout):
    k, q:         [T, B, H, N]  bf16 or fp32
    v:            [T, B, H, V]
    decay:        [T, B, H]
    S_ckpt:       [num_ckpts, B, H, N, V] sparse forward checkpoints.
    d_out:        [T, B, H, V]
    d_S_final:    [B, H, N, V]
    seg_scratch:  [num_programs * (K+1) * BLOCK_H * N * V] flat fp32
                  (per-program staging for replayed S history).
Outputs:
    d_k:          [T, B, H, N]
    d_v:          [T, B, H, V]
    d_q:          [T, B, H, N]
    d_decay:      [T, B, H]
    d_S0:         [B, H, N, V]
"""
from __future__ import absolute_import

from typing import Tuple

import torch
import triton
import triton.language as tl

from ndm.triton.e88_triton_forward import DEFAULT_CKPT_INTERVAL


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _e88_backward_kernel(
    # Forward-side inputs (need re-reads).
    K_ptr,              # [T, B, H, N]
    V_ptr,              # [T, B, H, V]
    Q_ptr,              # [T, B, H, N]
    D_ptr,              # [T, B, H]
    Sckpt_ptr,          # [num_ckpts, B, H, N, V]  SPARSE
    G_ptr,              # [T, B, H, V] gate (read iff APPLY_GATE)
    E_ptr,              # [T, B, H, N] erase/read gate (read iff SPLIT_EDIT)
    W_ptr,              # [T, B, H, V] value write gate (read iff SPLIT_EDIT)
    # Scratch staging buffer (per program × (K+1) × BLOCK_H × N × V, fp32).
    Scratch_ptr,
    # Upstream grads.
    DOut_ptr,           # [T, B, H, V]
    DSfinal_ptr,        # [B, H, N, V]
    # Output grads.
    DK_ptr,             # [T, B, H, N]
    DV_ptr,             # [T, B, H, V]
    DQ_ptr,             # [T, B, H, N]
    DD_ptr,             # [T, B, H]
    DG_ptr,             # [T, B, H, V] (written iff APPLY_GATE)
    DE_ptr,             # [T, B, H, N] (written iff SPLIT_EDIT)
    DW_ptr,             # [T, B, H, V] (written iff SPLIT_EDIT)
    DS0_ptr,            # [B, H, N, V]
    # Strides for every tensor (in elements).
    sk_t, sk_b, sk_h, sk_n,
    sv_t, sv_b, sv_h, sv_v,
    sq_t, sq_b, sq_h, sq_n,
    sd_t, sd_b, sd_h,
    sc_t, sc_b, sc_h, sc_n, sc_v,
    sg_t, sg_b, sg_h, sg_v,
    se_t, se_b, se_h, se_n,
    sw_t, sw_b, sw_h, sw_v,
    sdo_t, sdo_b, sdo_h, sdo_v,
    sdsf_b, sdsf_h, sdsf_n, sdsf_v,
    sdk_t, sdk_b, sdk_h, sdk_n,
    sdv_t, sdv_b, sdv_h, sdv_v,
    sdq_t, sdq_b, sdq_h, sdq_n,
    sdd_t, sdd_b, sdd_h,
    sdg_t, sdg_b, sdg_h, sdg_v,
    sde_t, sde_b, sde_h, sde_n,
    sdw_t, sdw_b, sdw_h, sdw_v,
    sds0_b, sds0_h, sds0_n, sds0_v,
    # Sizes.
    T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    N: tl.constexpr, V: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CKPT_INTERVAL: tl.constexpr,
    NUM_PROGS_H: tl.constexpr,
    APPLY_GATE: tl.constexpr,
    NORMALIZE_KQ: tl.constexpr,
    APPLY_SILU_QKV: tl.constexpr,
    RAW_WRITE: tl.constexpr,
    LINEAR_STATE: tl.constexpr,
    SPLIT_EDIT: tl.constexpr,
):
    """One program per (batch, head_block). Reverse-segment loop."""
    b = tl.program_id(0).to(tl.int64)
    hg = tl.program_id(1).to(tl.int64)

    # Linear program index for indexing into the per-program scratch buffer.
    prog_id = b * NUM_PROGS_H + hg

    h_start = hg * BLOCK_H
    h_idx = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_idx < H

    n_idx = tl.arange(0, BLOCK_N)
    v_idx = tl.arange(0, BLOCK_V)
    n_mask = n_idx < N
    v_mask = v_idx < V

    mask_hnv = (h_mask[:, None, None] & n_mask[None, :, None] & v_mask[None, None, :])
    mask_hn = h_mask[:, None] & n_mask[None, :]
    mask_hv = h_mask[:, None] & v_mask[None, :]

    # Per-program scratch base offset, in elements.
    # Scratch layout: [num_programs, (K+1), BLOCK_H, BLOCK_N, BLOCK_V] fp32.
    # We use BLOCK_N/BLOCK_V (rounded-up power-of-2) for stride, with
    # masking on N/V loads/stores.
    tile_size = BLOCK_H * BLOCK_N * BLOCK_V  # elements per S-slot
    prog_scratch_size = (CKPT_INTERVAL + 1) * tile_size
    prog_scratch_base = prog_id.to(tl.int64) * prog_scratch_size

    # Pre-compute scratch index offsets for a single tile [BLOCK_H, BLOCK_N, BLOCK_V].
    scratch_inner = (
        tl.arange(0, BLOCK_H)[:, None, None] * (BLOCK_N * BLOCK_V)
        + tl.arange(0, BLOCK_N)[None, :, None] * BLOCK_V
        + tl.arange(0, BLOCK_V)[None, None, :]
    )

    # Initialize dS_carry from upstream d_S_final.
    dsf_off = (
        b * sdsf_b
        + h_idx[:, None, None] * sdsf_h
        + n_idx[None, :, None] * sdsf_n
        + v_idx[None, None, :] * sdsf_v
    )
    dS_carry = tl.load(DSfinal_ptr + dsf_off, mask=mask_hnv, other=0.0).to(tl.float32)

    num_segments = T // CKPT_INTERVAL  # T % CKPT_INTERVAL == 0 enforced by wrapper

    # Reverse-segment loop: seg = num_segments - 1 .. 0.
    for seg_rev in range(num_segments):
        seg = num_segments - 1 - seg_rev
        seg_i64 = tl.full([1], seg, dtype=tl.int64)

        # ---- Phase 1: load S_seg_start = S_ckpt[seg] (state BEFORE step seg*K). ----
        sc_off = (
            seg_i64 * sc_t + b * sc_b
            + h_idx[:, None, None] * sc_h
            + n_idx[None, :, None] * sc_n
            + v_idx[None, None, :] * sc_v
        )
        S = tl.load(Sckpt_ptr + sc_off, mask=mask_hnv, other=0.0).to(tl.float32)

        # Store S into scratch slot 0 (the "S_{t-1}" for the first step of
        # this segment). Scratch is bf16 — store auto-casts fp32 -> bf16.
        slot0_off = prog_scratch_base + 0 * tile_size + scratch_inner
        tl.store(Scratch_ptr + slot0_off, S.to(Scratch_ptr.dtype.element_ty), mask=mask_hnv)

        # ---- Phase 2: forward-replay K steps, caching post-step S. ----
        for j in range(CKPT_INTERVAL):
            t = seg * CKPT_INTERVAL + j
            t_i64 = tl.full([1], t, dtype=tl.int64)

            k_off = (
                t_i64 * sk_t + b * sk_b
                + h_idx[:, None] * sk_h
                + n_idx[None, :] * sk_n
            )
            v_off = (
                t_i64 * sv_t + b * sv_b
                + h_idx[:, None] * sv_h
                + v_idx[None, :] * sv_v
            )
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
            d_off = t_i64 * sd_t + b * sd_b + h_idx * sd_h

            k_vec = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)
            v_vec = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)
            decay_val = tl.load(D_ptr + d_off, mask=h_mask, other=0.0).to(tl.float32)

            if APPLY_SILU_QKV:
                k_vec = k_vec / (1.0 + tl.exp(-k_vec))
                v_vec = v_vec / (1.0 + tl.exp(-v_vec))

            # Forward replay — match the forward kernel's L2-norm if enabled.
            if NORMALIZE_KQ:
                k_norm_sq = tl.sum(k_vec * k_vec, axis=1)
                inv_k_norm = 1.0 / (tl.sqrt(k_norm_sq) + 1e-6)
                k_vec = k_vec * inv_k_norm[:, None]

            if SPLIT_EDIT:
                e_vec = tl.load(E_ptr + e_off, mask=mask_hn, other=0.0).to(tl.float32)
                w_vec = tl.load(W_ptr + w_off, mask=mask_hv, other=0.0).to(tl.float32)
                read_key = k_vec * e_vec
                write_value = v_vec * w_vec
            else:
                read_key = k_vec
                write_value = v_vec

            # retrieve = S^T @ read_key:  [BH, BV]
            if RAW_WRITE:
                delta = write_value
            else:
                retrieved = tl.sum(S * read_key[:, :, None], axis=1)
                delta = write_value - retrieved
            outer = k_vec[:, :, None] * delta[:, None, :]
            pre = decay_val[:, None, None] * S + outer
            if LINEAR_STATE:
                S = pre
            else:
                # Match forward's stable tanh path. The raw exp formula can
                # overflow and turn saturation into inf/inf = NaN.
                S = 2.0 * tl.sigmoid(2.0 * pre) - 1.0

            # Save S after step t into scratch slot j+1 (bf16-cast).
            slot_off = prog_scratch_base + (j + 1) * tile_size + scratch_inner
            tl.store(Scratch_ptr + slot_off, S.to(Scratch_ptr.dtype.element_ty), mask=mask_hnv)

        # ---- Phase 3: backward through K steps in reverse. ----
        for j_rev in range(CKPT_INTERVAL):
            j = CKPT_INTERVAL - 1 - j_rev
            t = seg * CKPT_INTERVAL + j
            t_i64 = tl.full([1], t, dtype=tl.int64)

            # Load S_t (slot j+1) and S_{t-1} (slot j) from scratch.
            slot_t_off = prog_scratch_base + (j + 1) * tile_size + scratch_inner
            slot_tm1_off = prog_scratch_base + j * tile_size + scratch_inner
            # Loads from scratch (allocated bf16 in wrapper) -> cast to fp32 for compute.
            S_t = tl.load(Scratch_ptr + slot_t_off, mask=mask_hnv, other=0.0).to(tl.float32)
            S_tm1 = tl.load(Scratch_ptr + slot_tm1_off, mask=mask_hnv, other=0.0).to(tl.float32)

            # Reload forward inputs.
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
            v_off = (
                t_i64 * sv_t + b * sv_b
                + h_idx[:, None] * sv_h
                + v_idx[None, :] * sv_v
            )
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
            d_off = t_i64 * sd_t + b * sd_b + h_idx * sd_h

            k_raw = tl.load(K_ptr + k_off, mask=mask_hn, other=0.0).to(tl.float32)
            q_raw = tl.load(Q_ptr + q_off, mask=mask_hn, other=0.0).to(tl.float32)
            v_raw = tl.load(V_ptr + v_off, mask=mask_hv, other=0.0).to(tl.float32)
            decay_val = tl.load(D_ptr + d_off, mask=h_mask, other=0.0).to(tl.float32)

            if APPLY_SILU_QKV:
                sigmoid_k = 1.0 / (1.0 + tl.exp(-k_raw))
                sigmoid_q = 1.0 / (1.0 + tl.exp(-q_raw))
                sigmoid_v = 1.0 / (1.0 + tl.exp(-v_raw))
                k_base = k_raw * sigmoid_k
                q_base = q_raw * sigmoid_q
                v_vec = v_raw * sigmoid_v
            else:
                k_base = k_raw
                q_base = q_raw
                v_vec = v_raw

            # If kernel-fused L2 norm: compute k_norm, q_norm here and use
            # them as `k_vec`, `q_vec` for the recurrence backward. Save
            # 1/||k||, 1/||q|| for the post-hoc d_k_raw / d_q_raw conversion.
            if NORMALIZE_KQ:
                k_norm_sq = tl.sum(k_base * k_base, axis=1)
                q_norm_sq = tl.sum(q_base * q_base, axis=1)
                inv_k_norm = 1.0 / (tl.sqrt(k_norm_sq) + 1e-6)
                inv_q_norm = 1.0 / (tl.sqrt(q_norm_sq) + 1e-6)
                k_vec = k_base * inv_k_norm[:, None]
                q_vec = q_base * inv_q_norm[:, None]
            else:
                k_vec = k_base
                q_vec = q_base

            do_off = (
                t_i64 * sdo_t + b * sdo_b
                + h_idx[:, None] * sdo_h
                + v_idx[None, :] * sdo_v
            )
            d_out = tl.load(DOut_ptr + do_off, mask=mask_hv, other=0.0).to(tl.float32)

            # Optional fused gate: forward applied output_layer = silu(g) * out_kernel.
            # In backward, that means:
            #   d_out_kernel = d_out_layer * silu(g)
            #   d_g           = d_out_layer * out_kernel * silu_prime(g)
            # silu(g) = g * sigmoid(g);  silu_prime(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g))).
            # Compute out_kernel inline as one extra reduction; cheap relative
            # to the full step.
            if APPLY_GATE:
                g_off = (
                    t_i64 * sg_t + b * sg_b
                    + h_idx[:, None] * sg_h
                    + v_idx[None, :] * sg_v
                )
                g_val = tl.load(G_ptr + g_off, mask=mask_hv, other=0.0).to(tl.float32)
                sigmoid_g = 1.0 / (1.0 + tl.exp(-g_val))
                silu_g = g_val * sigmoid_g
                silu_prime_g = sigmoid_g * (1.0 + g_val * (1.0 - sigmoid_g))
                # out_kernel before gating: same expression as forward output.
                out_kernel = tl.sum(S_t * q_vec[:, :, None], axis=1)  # [BH, BV]
                d_g = d_out * out_kernel * silu_prime_g
                # Replace d_out with the upstream gradient w.r.t. the un-gated kernel output.
                d_out = d_out * silu_g
                # Store d_g.
                dg_off = (
                    t_i64 * sdg_t + b * sdg_b
                    + h_idx[:, None] * sdg_h
                    + v_idx[None, :] * sdg_v
                )
                tl.store(DG_ptr + dg_off, d_g.to(DG_ptr.dtype.element_ty), mask=mask_hv)

            # dS_t = carry + outer(q_t, d_out_t)
            dS_t = dS_carry + q_vec[:, :, None] * d_out[:, None, :]

            # d_q_t = sum_v S_t * d_out
            d_q = tl.sum(S_t * d_out[:, None, :], axis=2)

            # d_pre = dS_t for linear-state ablation, otherwise tanh chain rule.
            if LINEAR_STATE:
                d_pre = dS_t
            else:
                d_pre = dS_t * (1.0 - S_t * S_t)

            # d_decay_t = sum_{n,v} d_pre * S_{t-1}
            d_decay = tl.sum(tl.sum(d_pre * S_tm1, axis=2), axis=1)

            if SPLIT_EDIT:
                e_vec = tl.load(E_ptr + e_off, mask=mask_hn, other=0.0).to(tl.float32)
                w_vec = tl.load(W_ptr + w_off, mask=mask_hv, other=0.0).to(tl.float32)
                read_key = k_vec * e_vec
                write_value = v_vec * w_vec
            else:
                read_key = k_vec
                write_value = v_vec

            # Recompute delta_t from S_{t-1} and read_key unless this is the
            # raw-write ablation, where the write value is used directly.
            if RAW_WRITE:
                delta = write_value
            else:
                retrieved = tl.sum(S_tm1 * read_key[:, :, None], axis=1)
                delta = write_value - retrieved

            # outer[n,v] = k[n] * delta[v]
            d_k_outer = tl.sum(d_pre * delta[:, None, :], axis=2)
            d_delta = tl.sum(d_pre * k_vec[:, :, None], axis=1)

            if SPLIT_EDIT:
                d_v = d_delta * w_vec
                d_w = d_delta * v_vec
            else:
                d_v = d_delta
                d_w = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)

            if RAW_WRITE:
                d_k = d_k_outer
                d_e = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
            else:
                d_read_key = -tl.sum(S_tm1 * d_delta[:, None, :], axis=2)
                if SPLIT_EDIT:
                    d_e = d_read_key * k_vec
                    d_k_retrieve = d_read_key * e_vec
                else:
                    d_e = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
                    d_k_retrieve = d_read_key
                d_k = d_k_outer + d_k_retrieve

            if RAW_WRITE:
                dS_carry = decay_val[:, None, None] * d_pre
            else:
                dS_carry = (
                    decay_val[:, None, None] * d_pre
                    - read_key[:, :, None] * d_delta[:, None, :]
                )

            # If kernel-fused L2 norm: convert d_k_norm -> d_k_raw and
            # d_q_norm -> d_q_raw via the standard L2-norm chain rule:
            #   d_x_raw = (1/||x||) * (d_x_norm - x_norm * (d_x_norm . x_norm))
            # at this point d_k and d_q are gradients w.r.t. k_norm/q_norm.
            if NORMALIZE_KQ:
                # Project out the radial component, then scale by 1/||x||.
                d_k_dot_kn = tl.sum(d_k * k_vec, axis=1)             # [BH]
                d_q_dot_qn = tl.sum(d_q * q_vec, axis=1)             # [BH]
                d_k = (d_k - k_vec * d_k_dot_kn[:, None]) * inv_k_norm[:, None]
                d_q = (d_q - q_vec * d_q_dot_qn[:, None]) * inv_q_norm[:, None]

            if APPLY_SILU_QKV:
                silu_prime_k = sigmoid_k * (1.0 + k_raw * (1.0 - sigmoid_k))
                silu_prime_q = sigmoid_q * (1.0 + q_raw * (1.0 - sigmoid_q))
                silu_prime_v = sigmoid_v * (1.0 + v_raw * (1.0 - sigmoid_v))
                d_k = d_k * silu_prime_k
                d_q = d_q * silu_prime_q
                d_v = d_v * silu_prime_v

            dk_off = (
                t_i64 * sdk_t + b * sdk_b
                + h_idx[:, None] * sdk_h
                + n_idx[None, :] * sdk_n
            )
            dv_off = (
                t_i64 * sdv_t + b * sdv_b
                + h_idx[:, None] * sdv_h
                + v_idx[None, :] * sdv_v
            )
            dq_off = (
                t_i64 * sdq_t + b * sdq_b
                + h_idx[:, None] * sdq_h
                + n_idx[None, :] * sdq_n
            )
            dd_off = t_i64 * sdd_t + b * sdd_b + h_idx * sdd_h

            tl.store(DK_ptr + dk_off, d_k.to(DK_ptr.dtype.element_ty), mask=mask_hn)
            tl.store(DV_ptr + dv_off, d_v.to(DV_ptr.dtype.element_ty), mask=mask_hv)
            tl.store(DQ_ptr + dq_off, d_q.to(DQ_ptr.dtype.element_ty), mask=mask_hn)
            tl.store(DD_ptr + dd_off, d_decay.to(DD_ptr.dtype.element_ty), mask=h_mask)
            if SPLIT_EDIT:
                de_off = (
                    t_i64 * sde_t + b * sde_b
                    + h_idx[:, None] * sde_h
                    + n_idx[None, :] * sde_n
                )
                dw_off = (
                    t_i64 * sdw_t + b * sdw_b
                    + h_idx[:, None] * sdw_h
                    + v_idx[None, :] * sdw_v
                )
                tl.store(DE_ptr + de_off, d_e.to(DE_ptr.dtype.element_ty), mask=mask_hn)
                tl.store(DW_ptr + dw_off, d_w.to(DW_ptr.dtype.element_ty), mask=mask_hv)

    # Write d_S0 = remaining carry.
    ds0_off = (
        b * sds0_b
        + h_idx[:, None, None] * sds0_h
        + n_idx[None, :, None] * sds0_n
        + v_idx[None, None, :] * sds0_v
    )
    tl.store(DS0_ptr + ds0_off, dS_carry.to(DS0_ptr.dtype.element_ty), mask=mask_hnv)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return max(p, 16)


def e88_triton_backward(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    S_ckpt: torch.Tensor,
    d_out: torch.Tensor,
    d_S_final: torch.Tensor = None,
    block_h: int = None,
    num_warps: int = None,
    ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
    g: torch.Tensor = None,  # [T, B, H, V] gate; if None, no fused-gate handling
    normalize_kq: bool = False,
    apply_silu_qkv: bool = False,
    raw_write: bool = False,
    linear_state: bool = False,
    erase_gate: torch.Tensor = None,
    value_write_gate: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the E88 backward recurrence in Triton.

    Args:
        k, q:      [T, B, H, N]  -- forward inputs.
        v:         [T, B, H, V]
        decay:     [T, B, H]
        S_ckpt:    [num_ckpts, B, H, N, V]  SPARSE checkpoints from forward
                                            (S_ckpt[0]=S0). num_ckpts must
                                            equal T // ckpt_interval + 1.
        d_out:     [T, B, H, V]        upstream gradient w.r.t. output.
        d_S_final: [B, H, N, V] or None  upstream gradient w.r.t. S_final.
                                          Defaults to zero.
        block_h, num_warps: optional kernel tuning overrides.
        ckpt_interval: must match the value used in the forward pass.

    Returns:
        d_k:    [T, B, H, N]
        d_v:    [T, B, H, V]
        d_q:    [T, B, H, N]
        d_decay:[T, B, H]
        d_S0:   [B, H, N, V]
    """
    assert k.is_cuda
    T, B, H, N = k.shape
    Vsz = v.shape[-1]
    assert q.shape == (T, B, H, N)
    assert v.shape == (T, B, H, Vsz)
    assert decay.shape == (T, B, H)
    assert d_out.shape == (T, B, H, Vsz)

    if T % ckpt_interval != 0:
        raise NotImplementedError(
            f"Sparse-checkpoint backward currently requires T % ckpt_interval == 0 "
            f"(got T={T}, ckpt_interval={ckpt_interval})."
        )
    num_ckpts = T // ckpt_interval + 1
    assert S_ckpt.shape == (num_ckpts, B, H, N, Vsz), (
        f"S_ckpt shape {tuple(S_ckpt.shape)} != expected {(num_ckpts, B, H, N, Vsz)} "
        f"(T={T}, ckpt_interval={ckpt_interval})"
    )

    BLOCK_N = _next_pow2(N)
    BLOCK_V = _next_pow2(Vsz)
    if BLOCK_N > 64 or BLOCK_V > 64:
        raise NotImplementedError(
            f"e88_triton_backward currently supports N, V <= 64 (got {N},{Vsz})"
        )

    # Avoid expensive .contiguous() copies for transposed views — kernel
    # uses strides directly. Only force contiguous if last dim isn't
    # unit-stride. At production scale each copy is ~100 MB; saving 4 of
    # them per backward call across 14 layers x 3 grad_ckpt invocations
    # is several GB of bandwidth per training step.
    def _strided_ok(x):
        return x.stride(-1) == 1
    k_c = k if _strided_ok(k) else k.contiguous()
    v_c = v if _strided_ok(v) else v.contiguous()
    q_c = q if _strided_ok(q) else q.contiguous()
    d_c = decay if _strided_ok(decay) else decay.contiguous()
    sc_c = S_ckpt if _strided_ok(S_ckpt) else S_ckpt.contiguous()
    do_c = d_out if _strided_ok(d_out) else d_out.contiguous()

    if d_S_final is None:
        dsf_c = torch.zeros((B, H, N, Vsz), dtype=k_c.dtype, device=k.device)
    else:
        dsf_c = d_S_final if _strided_ok(d_S_final) else d_S_final.contiguous()
    assert dsf_c.shape == (B, H, N, Vsz)

    apply_gate = g is not None
    if apply_gate:
        g_c = g if _strided_ok(g) else g.contiguous()
        assert g_c.shape == (T, B, H, Vsz), \
            f"gate shape must be [T, B, H, V] = {(T, B, H, Vsz)}, got {tuple(g_c.shape)}"
        g_strides = (g_c.stride(0), g_c.stride(1), g_c.stride(2), g_c.stride(3))
        d_g = torch.empty_like(g_c)
        dg_strides = (d_g.stride(0), d_g.stride(1), d_g.stride(2), d_g.stride(3))
    else:
        # Pass dummy pointers + zero strides; kernel guards via APPLY_GATE constexpr.
        g_c = k_c
        g_strides = (0, 0, 0, 0)
        d_g = k_c  # same dummy
        dg_strides = (0, 0, 0, 0)

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
        d_erase = torch.empty_like(e_c)
        d_value_write = torch.empty_like(w_c)
        de_strides = (d_erase.stride(0), d_erase.stride(1), d_erase.stride(2), d_erase.stride(3))
        dw_strides = (
            d_value_write.stride(0), d_value_write.stride(1),
            d_value_write.stride(2), d_value_write.stride(3),
        )
    else:
        e_c = k_c
        w_c = v_c
        e_strides = (0, 0, 0, 0)
        w_strides = (0, 0, 0, 0)
        d_erase = k_c
        d_value_write = v_c
        de_strides = (0, 0, 0, 0)
        dw_strides = (0, 0, 0, 0)

    out_dtype = k_c.dtype
    d_k = torch.empty_like(k_c)
    d_v = torch.empty_like(v_c)
    d_q = torch.empty_like(q_c)
    d_decay = torch.empty_like(d_c)
    d_S0 = torch.empty((B, H, N, Vsz), dtype=out_dtype, device=k.device)

    # Default heads-per-program. Empirically tuned at H=386 N=V=32:
    #   - BLOCK_H=1 is best (BLOCK_H>1 spills the [BH, N, V] state tile).
    #   - num_warps depends on B: nw=1 wins when B*H is large enough to
    #     saturate the SMs already (e.g. B=8, H=386: 3088 programs, no
    #     need for extra warps per program — register pressure hurts).
    #     nw=2 wins at small B (e.g. B=1) where we need extra warps for
    #     latency hiding.
    # Empirical (B=8, T=512, H=386, N=V=32, sparse-ckpt + bf16 scratch):
    #   nw=1: 4.49 ms; nw=2: 6.58 ms; nw=4: 9.43 ms
    # At B=1: nw=2 is best (1.08 ms vs nw=1: 1.25 ms).
    # See tests/sweep_triton_block_h_at_386.py for the original sweep.
    if block_h is None:
        block_h = 1
        if num_warps is None:
            # After fusion (gate + L2-norm) lands, nw=1 wins at every
            # shape we care about at H>=64. See sweep in
            # tests/sweep_num_stages_at_386.py and inline data in the
            # forward kernel docstring.
            if H >= 64:
                num_warps = 1
            else:
                num_warps = 4
    if num_warps is None:
        num_warps = 2 if block_h == 1 else (4 if block_h <= 4 else 8)

    num_progs_h = (H + block_h - 1) // block_h
    grid = (B, num_progs_h)

    # Allocate the per-program scratch buffer. Layout:
    #   [B * num_progs_h, K+1, BLOCK_H, BLOCK_N, BLOCK_V] fp32.
    # We use fp32 for accuracy — bf16 scratch loses ~3 bits per re-load
    # of S during the backward walk and is the main source of drift.
    scratch_numel = (
        B * num_progs_h
        * (ckpt_interval + 1)
        * block_h * BLOCK_N * BLOCK_V
    )
    # Scratch dtype matches input dtype — bf16 for production (halves
    # bandwidth, matches CUDA reg-own's segment_cache); fp32 for fp32
    # inputs (preserves test parity). The kernel auto-casts on load/store.
    seg_scratch = torch.empty(scratch_numel, dtype=k.dtype, device=k.device)

    _e88_backward_kernel[grid](
        k_c, v_c, q_c, d_c, sc_c, g_c, e_c, w_c,
        seg_scratch,
        do_c, dsf_c,
        d_k, d_v, d_q, d_decay, d_g, d_erase, d_value_write, d_S0,
        # strides
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        d_c.stride(0), d_c.stride(1), d_c.stride(2),
        sc_c.stride(0), sc_c.stride(1), sc_c.stride(2),
        sc_c.stride(3), sc_c.stride(4),
        *g_strides,
        *e_strides,
        *w_strides,
        do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
        dsf_c.stride(0), dsf_c.stride(1), dsf_c.stride(2), dsf_c.stride(3),
        d_k.stride(0), d_k.stride(1), d_k.stride(2), d_k.stride(3),
        d_v.stride(0), d_v.stride(1), d_v.stride(2), d_v.stride(3),
        d_q.stride(0), d_q.stride(1), d_q.stride(2), d_q.stride(3),
        d_decay.stride(0), d_decay.stride(1), d_decay.stride(2),
        *dg_strides,
        *de_strides,
        *dw_strides,
        d_S0.stride(0), d_S0.stride(1), d_S0.stride(2), d_S0.stride(3),
        T=T, B=B, H=H, N=N, V=Vsz,
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V,
        BLOCK_H=block_h,
        CKPT_INTERVAL=ckpt_interval,
        NUM_PROGS_H=num_progs_h,
        APPLY_GATE=apply_gate,
        NORMALIZE_KQ=bool(normalize_kq),
        APPLY_SILU_QKV=bool(apply_silu_qkv),
        RAW_WRITE=bool(raw_write),
        LINEAR_STATE=bool(linear_state),
        SPLIT_EDIT=bool(split_edit),
        num_warps=num_warps,
    )

    if apply_gate and split_edit:
        return d_k, d_v, d_q, d_decay, d_g, d_erase, d_value_write, d_S0
    elif apply_gate:
        return d_k, d_v, d_q, d_decay, d_g, d_S0
    elif split_edit:
        return d_k, d_v, d_q, d_decay, d_erase, d_value_write, d_S0
    else:
        return d_k, d_v, d_q, d_decay, d_S0


# ---------------------------------------------------------------------------
# autograd.Function wrapper combining forward + backward.
# ---------------------------------------------------------------------------

class E88TritonFunction(torch.autograd.Function):
    """torch.autograd.Function gluing forward + backward Triton kernels.

    Optionally supports a fused output gate (output = silu(g) * S^T@q)
    when ``g`` is provided. The fused gate avoids two extra kernel
    launches per layer call (silu, multiply) and matches CUDA's
    register-owned forward.

    Supports the ``linear_state=True`` ablation by dropping tanh in both the
    forward replay and backward chain rule.
    """

    @staticmethod
    def forward(
        ctx,
        S0,
        k,
        v,
        q,
        decay,
        g=None,
        normalize_kq=False,
        apply_silu_qkv=False,
        raw_write=False,
        erase_gate=None,
        value_write_gate=None,
        linear_state=False,
        valid_length=None,
    ):
        from ndm.triton.e88_triton_forward import e88_triton_forward
        out, S_final, S_ckpt = e88_triton_forward(
            S0, k, v, q, decay, g=g, normalize_kq=normalize_kq,
            apply_silu_qkv=apply_silu_qkv, raw_write=raw_write,
            linear_state=linear_state,
            erase_gate=erase_gate, value_write_gate=value_write_gate,
            valid_length=valid_length,
        )
        ctx.normalize_kq = bool(normalize_kq)
        ctx.apply_silu_qkv = bool(apply_silu_qkv)
        ctx.raw_write = bool(raw_write)
        ctx.linear_state = bool(linear_state)
        ctx.has_split_edit = erase_gate is not None or value_write_gate is not None
        ctx.valid_length = k.shape[0] if valid_length is None else int(valid_length)
        # Save for backward. Note: S0 isn't strictly required (it equals
        # S_ckpt[0]), but saving it is cheap and explicit. We must save
        # g if present because backward needs it for d_g and to scale d_out.
        if g is not None and ctx.has_split_edit:
            ctx.save_for_backward(k, v, q, decay, S_ckpt, g, erase_gate, value_write_gate)
            ctx.has_gate = True
        elif g is not None:
            ctx.save_for_backward(k, v, q, decay, S_ckpt, g)
            ctx.has_gate = True
        elif ctx.has_split_edit:
            ctx.save_for_backward(k, v, q, decay, S_ckpt, erase_gate, value_write_gate)
            ctx.has_gate = False
        else:
            ctx.save_for_backward(k, v, q, decay, S_ckpt)
            ctx.has_gate = False
        return out, S_final

    @staticmethod
    def backward(ctx, d_out, d_S_final):
        if ctx.valid_length != d_out.shape[0]:
            raise RuntimeError(
                "valid_length shorter than the padded recurrence is "
                "forward-only; unaligned training must not use inference padding")
        nkq = ctx.normalize_kq
        silu_qkv = ctx.apply_silu_qkv
        raw_write = ctx.raw_write
        linear_state = ctx.linear_state
        if ctx.has_gate and ctx.has_split_edit:
            k, v, q, decay, S_ckpt, g, erase_gate, value_write_gate = ctx.saved_tensors
            d_k, d_v, d_q, d_decay, d_g, d_erase, d_value_write, d_S0 = e88_triton_backward(
                k, v, q, decay, S_ckpt,
                d_out=d_out.contiguous(),
                d_S_final=d_S_final.contiguous() if d_S_final is not None else None,
                g=g,
                normalize_kq=nkq,
                apply_silu_qkv=silu_qkv,
                raw_write=raw_write,
                linear_state=linear_state,
                erase_gate=erase_gate,
                value_write_gate=value_write_gate,
            )
            return (
                d_S0, d_k, d_v, d_q, d_decay, d_g,
                None, None, None, d_erase, d_value_write, None, None,
            )
        elif ctx.has_gate:
            k, v, q, decay, S_ckpt, g = ctx.saved_tensors
            d_k, d_v, d_q, d_decay, d_g, d_S0 = e88_triton_backward(
                k, v, q, decay, S_ckpt,
                d_out=d_out.contiguous(),
                d_S_final=d_S_final.contiguous() if d_S_final is not None else None,
                g=g,
                normalize_kq=nkq,
                apply_silu_qkv=silu_qkv,
                raw_write=raw_write,
                linear_state=linear_state,
            )
            # Match forward signature order (S0, k, v, q, decay, g, normalize_kq).
            return d_S0, d_k, d_v, d_q, d_decay, d_g, None, None, None, None, None, None, None
        elif ctx.has_split_edit:
            k, v, q, decay, S_ckpt, erase_gate, value_write_gate = ctx.saved_tensors
            d_k, d_v, d_q, d_decay, d_erase, d_value_write, d_S0 = e88_triton_backward(
                k, v, q, decay, S_ckpt,
                d_out=d_out.contiguous(),
                d_S_final=d_S_final.contiguous() if d_S_final is not None else None,
                normalize_kq=nkq,
                apply_silu_qkv=silu_qkv,
                raw_write=raw_write,
                linear_state=linear_state,
                erase_gate=erase_gate,
                value_write_gate=value_write_gate,
            )
            return (
                d_S0, d_k, d_v, d_q, d_decay, None,
                None, None, None, d_erase, d_value_write, None, None,
            )
        else:
            k, v, q, decay, S_ckpt = ctx.saved_tensors
            d_k, d_v, d_q, d_decay, d_S0 = e88_triton_backward(
                k, v, q, decay, S_ckpt,
                d_out=d_out.contiguous(),
                d_S_final=d_S_final.contiguous() if d_S_final is not None else None,
                normalize_kq=nkq,
                apply_silu_qkv=silu_qkv,
                raw_write=raw_write,
                linear_state=linear_state,
            )
            # Match forward signature order (S0, k, v, q, decay, g, normalize_kq).
            return d_S0, d_k, d_v, d_q, d_decay, None, None, None, None, None, None, None, None


def e88_triton(
    S0,
    k,
    v,
    q,
    decay,
    g=None,
    normalize_kq=False,
    apply_silu_qkv=False,
    raw_write=False,
    erase_gate=None,
    value_write_gate=None,
    linear_state=False,
    valid_length=None,
):
    """Differentiable Triton E88 — returns (out, S_final).

    If ``g`` is provided, applies the fused output gate
    ``output = silu(g) * S^T@q`` inside the kernel (saves two kernel
    launches per call). ``g`` must have shape [T, B, H, V].

    If ``normalize_kq`` is True, the kernel L2-normalizes k and q
    on-the-fly per head (saves the Python `linalg_vector_norm` and
    `aten::div` per layer call). The backward applies the standard
    norm chain rule to recover gradients w.r.t. the raw k, q.
    """
    return E88TritonFunction.apply(
        S0, k, v, q, decay, g, normalize_kq, apply_silu_qkv, raw_write,
        erase_gate, value_write_gate, linear_state, valid_length,
    )
