"""
Fused Triton kernels for E88 performance optimization.

These kernels eliminate intermediate memory traffic and kernel launch overhead
for common element-wise operations during training.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Fused SiLU + L2 Normalization
# ============================================================================

@triton.jit
def _silu_l2_norm_kernel(
    x_ptr, out_ptr,
    N,  # number of elements per row
    stride_x, stride_out,
    BLOCK: tl.constexpr,
):
    """Fused SiLU + L2 norm: out = silu(x) / ||silu(x)||"""
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * stride_x + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)

    # L2 norm
    x_sq = silu_x * silu_x
    norm_sq = tl.sum(x_sq, axis=0)
    norm = tl.sqrt(norm_sq + 1e-6)

    # Normalize
    x_normed = silu_x / norm

    # Store
    out_ptrs = out_ptr + row * stride_out + tl.arange(0, BLOCK)
    tl.store(out_ptrs, x_normed.to(tl.bfloat16), mask=mask)


@triton.jit
def _silu_l2_norm_bwd_kernel(
    grad_out_ptr, x_ptr, silu_x_ptr, norm_ptr,
    grad_x_ptr,
    N,
    stride_grad_out, stride_x, stride_silu, stride_norm, stride_grad_x,
    BLOCK: tl.constexpr,
):
    """Backward for fused SiLU + L2 norm."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < N

    # Load inputs
    grad_out = tl.load(grad_out_ptr + row * stride_grad_out + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
    silu_x = tl.load(silu_x_ptr + row * stride_silu + col_offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + row * stride_norm, mask=row >= 0, other=1.0).to(tl.float32)

    # d(silu(x)/norm) / d(silu(x)) = 1/norm - silu(x) * (silu(x) dot grad_out) / norm^3
    silu_normalized = silu_x / norm
    dot_product = tl.sum(grad_out * silu_normalized, axis=0)
    d_silu = (grad_out - silu_normalized * dot_product) / norm

    # d(silu(x)) / dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sig_x = tl.sigmoid(x)
    d_x = d_silu * sig_x * (1.0 + x * (1.0 - sig_x))

    # Store
    tl.store(grad_x_ptr + row * stride_grad_x + col_offsets, d_x.to(tl.bfloat16), mask=mask)


class SiLUL2Norm(torch.autograd.Function):
    """Fused SiLU + L2 normalization with autograd support."""

    @staticmethod
    def forward(ctx, x):
        """x: [*, N] -> out: [*, N] normalized along last dim."""
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        B, N = x_flat.shape

        out = torch.empty_like(x_flat)
        BLOCK = triton.next_power_of_2(N)
        if BLOCK > 1024:
            BLOCK = 1024

        _silu_l2_norm_kernel[(B,)](
            x_flat, out,
            N,
            x_flat.stride(0), out.stride(0),
            BLOCK=BLOCK,
        )

        # Save for backward
        if x.requires_grad:
            # Compute and save silu_x and norm for backward
            silu_x = torch.nn.functional.silu(x_flat.float()).to(x.dtype)
            norm = silu_x.float().norm(dim=-1, keepdim=True).squeeze(-1).to(x.dtype)
            ctx.save_for_backward(x_flat, silu_x, norm)
            ctx.N = N
            ctx.BLOCK = BLOCK

        return out.view(original_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_flat, silu_x, norm = ctx.saved_tensors
        N = ctx.N
        BLOCK = ctx.BLOCK
        B = x_flat.shape[0]

        # Use reshape instead of view to handle non-contiguous tensors
        grad_out_flat = grad_out.reshape(-1, N).contiguous()
        grad_x = torch.empty_like(x_flat)

        _silu_l2_norm_bwd_kernel[(B,)](
            grad_out_flat, x_flat, silu_x, norm,
            grad_x,
            N,
            grad_out_flat.stride(0), x_flat.stride(0), silu_x.stride(0), 1, grad_x.stride(0),
            BLOCK=BLOCK,
        )

        return grad_x.reshape(grad_out.shape)


def silu_l2_norm(x):
    """Fused SiLU + L2 normalize along last dimension."""
    return SiLUL2Norm.apply(x)


# ============================================================================
# Fused Mamba2-style Decay Computation
# ============================================================================

@triton.jit
def _mamba2_decay_kernel(
    alpha_ptr,      # [B, H] - input from a_proj(x)
    A_log_ptr,      # [H] - learned A_log
    dt_bias_ptr,    # [H] - learned dt_bias
    out_ptr,        # [B, H] - output decay
    N,              # H (number of heads)
    stride_alpha,
    stride_out,
    BLOCK: tl.constexpr,
):
    """Compute Mamba2-style decay: exp(-exp(A_log) * softplus(alpha + dt_bias))"""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < N

    # Load inputs
    alpha = tl.load(alpha_ptr + row * stride_alpha + col_offsets, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute decay
    x = alpha + dt_bias
    # Numerically stable softplus
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

    # g = -exp(A_log) * softplus(alpha + dt_bias)
    g = -tl.exp(A_log) * softplus_x

    # decay = exp(g)
    decay = tl.exp(g)

    # Store
    tl.store(out_ptr + row * stride_out + col_offsets, decay.to(tl.bfloat16), mask=mask)


@triton.jit
def _mamba2_decay_bwd_kernel(
    grad_decay_ptr, alpha_ptr, A_log_ptr, dt_bias_ptr, decay_ptr,
    grad_alpha_ptr, grad_A_log_ptr, grad_dt_bias_ptr,
    N,
    stride_grad_decay, stride_alpha, stride_decay, stride_grad_alpha,
    BLOCK: tl.constexpr,
):
    """Backward for Mamba2-style decay."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < N

    # Load inputs
    grad_decay = tl.load(grad_decay_ptr + row * stride_grad_decay + col_offsets, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + row * stride_alpha + col_offsets, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    decay = tl.load(decay_ptr + row * stride_decay + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Recompute intermediate values
    x = alpha + dt_bias
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    exp_A_log = tl.exp(A_log)

    # d(decay)/d(alpha) = decay * (-exp(A_log)) * d(softplus)/dx
    # d(softplus(x))/dx = sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    grad_alpha = grad_decay * decay * (-exp_A_log) * sigmoid_x

    # Store grad_alpha
    tl.store(grad_alpha_ptr + row * stride_grad_alpha + col_offsets, grad_alpha.to(tl.bfloat16), mask=mask)

    # grad_A_log needs atomic add across all rows
    # For simplicity, we accumulate in a separate pass
    # d(decay)/d(A_log) = decay * (-softplus_x) * exp(A_log)
    grad_A_log_local = grad_decay * decay * (-softplus_x) * exp_A_log
    tl.atomic_add(grad_A_log_ptr + col_offsets, grad_A_log_local, mask=mask)
    # dt_bias enters through the same x = alpha + dt_bias term, so its
    # per-row derivative equals grad_alpha and must reduce over all rows.
    tl.atomic_add(grad_dt_bias_ptr + col_offsets, grad_alpha, mask=mask)


class Mamba2Decay(torch.autograd.Function):
    """Fused Mamba2-style decay computation with autograd support."""

    @staticmethod
    def forward(ctx, alpha, A_log, dt_bias):
        """
        alpha: [B, T, H] - output from a_proj(x)
        A_log: [H] - learned parameter
        dt_bias: [H] - learned parameter
        Returns: decay [B, T, H]
        """
        original_shape = alpha.shape
        alpha_flat = alpha.view(-1, alpha.shape[-1])
        B, H = alpha_flat.shape

        out = torch.empty_like(alpha_flat)
        BLOCK = triton.next_power_of_2(H)
        if BLOCK > 1024:
            BLOCK = 1024

        _mamba2_decay_kernel[(B,)](
            alpha_flat, A_log, dt_bias, out,
            H,
            alpha_flat.stride(0), out.stride(0),
            BLOCK=BLOCK,
        )

        # Save for backward
        if alpha.requires_grad or A_log.requires_grad or dt_bias.requires_grad:
            ctx.save_for_backward(alpha_flat, A_log, dt_bias, out)
            ctx.H = H
            ctx.BLOCK = BLOCK
            ctx.original_shape = original_shape

        return out.view(original_shape)

    @staticmethod
    def backward(ctx, grad_decay):
        alpha_flat, A_log, dt_bias, decay = ctx.saved_tensors
        H = ctx.H
        BLOCK = ctx.BLOCK
        B = alpha_flat.shape[0]

        # Use reshape instead of view to handle non-contiguous tensors
        grad_decay_flat = grad_decay.reshape(-1, H).contiguous()

        grad_alpha = torch.empty_like(alpha_flat)
        grad_A_log = torch.zeros_like(A_log)
        grad_dt_bias = torch.zeros_like(dt_bias)

        _mamba2_decay_bwd_kernel[(B,)](
            grad_decay_flat, alpha_flat, A_log, dt_bias, decay,
            grad_alpha, grad_A_log, grad_dt_bias,
            H,
            grad_decay_flat.stride(0), alpha_flat.stride(0), decay.stride(0), grad_alpha.stride(0),
            BLOCK=BLOCK,
        )

        return grad_alpha.reshape(ctx.original_shape), grad_A_log, grad_dt_bias


def mamba2_decay(alpha, A_log, dt_bias):
    """Fused Mamba2-style decay: exp(-exp(A_log) * softplus(alpha + dt_bias))"""
    return Mamba2Decay.apply(alpha, A_log, dt_bias)


# ============================================================================
# Simple L2 Normalization (without SiLU)
# ============================================================================

@triton.jit
def _l2_norm_kernel(
    x_ptr, out_ptr,
    N,
    stride_x, stride_out,
    BLOCK: tl.constexpr,
):
    """L2 normalize: out = x / ||x||"""
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * stride_x + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # L2 norm
    x_sq = x * x
    norm_sq = tl.sum(x_sq, axis=0)
    norm = tl.sqrt(norm_sq + 1e-6)

    # Normalize
    x_normed = x / norm

    # Store
    out_ptrs = out_ptr + row * stride_out + tl.arange(0, BLOCK)
    tl.store(out_ptrs, x_normed.to(tl.bfloat16), mask=mask)


@triton.jit
def _l2_norm_bwd_kernel(
    grad_out_ptr, x_ptr, norm_ptr,
    grad_x_ptr,
    N,
    stride_grad_out, stride_x, stride_grad_x,
    BLOCK: tl.constexpr,
):
    """Backward for L2 norm."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < N

    grad_out = tl.load(grad_out_ptr + row * stride_grad_out + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + row, mask=row >= 0, other=1.0).to(tl.float32)

    # d(x/norm) / dx = 1/norm - x * (x dot grad_out) / norm^3
    x_normalized = x / norm
    dot_product = tl.sum(grad_out * x_normalized, axis=0)
    grad_x = (grad_out - x_normalized * dot_product) / norm

    tl.store(grad_x_ptr + row * stride_grad_x + col_offsets, grad_x.to(tl.bfloat16), mask=mask)


class L2Norm(torch.autograd.Function):
    """L2 normalization with autograd support."""

    @staticmethod
    def forward(ctx, x):
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        B, N = x_flat.shape

        out = torch.empty_like(x_flat)
        BLOCK = triton.next_power_of_2(N)
        if BLOCK > 1024:
            BLOCK = 1024

        _l2_norm_kernel[(B,)](
            x_flat, out,
            N,
            x_flat.stride(0), out.stride(0),
            BLOCK=BLOCK,
        )

        if x.requires_grad:
            norm = x_flat.float().norm(dim=-1).to(x.dtype)
            ctx.save_for_backward(x_flat, norm)
            ctx.N = N
            ctx.BLOCK = BLOCK

        return out.view(original_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_flat, norm = ctx.saved_tensors
        N = ctx.N
        BLOCK = ctx.BLOCK
        B = x_flat.shape[0]

        # Use reshape instead of view to handle non-contiguous tensors
        grad_out_flat = grad_out.reshape(-1, N).contiguous()
        grad_x = torch.empty_like(x_flat)

        _l2_norm_bwd_kernel[(B,)](
            grad_out_flat, x_flat, norm,
            grad_x,
            N,
            grad_out_flat.stride(0), x_flat.stride(0), grad_x.stride(0),
            BLOCK=BLOCK,
        )

        return grad_x.reshape(grad_out.shape)


def l2_norm(x):
    """L2 normalize along last dimension."""
    return L2Norm.apply(x)
