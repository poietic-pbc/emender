"""E97-named facade for the fused sequential split-edit Triton recurrence.

The proven implementation is shared with the older E88 recurrence engine.
Keeping delegation here avoids duplicating a large forward/backward kernel while
giving E97 a truthful public API and a stable place for future specialization.
"""

from __future__ import annotations

from typing import Tuple

import torch


def e97_split_edit_triton_apply(
    training: bool,
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    decay: torch.Tensor,
    g: torch.Tensor | None = None,
    S0: torch.Tensor | None = None,
    n_heads: int | None = None,
    apply_gate: bool = True,
    normalize_kq: bool = False,
    checkpoint_interval: int = 16,
    apply_silu_qkv: bool = False,
    raw_write: bool = False,
    linear_state: bool = False,
    *,
    erase_gate: torch.Tensor,
    value_write_gate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the fused sequential E97 split-edit recurrence.

    ``erase_gate`` and ``value_write_gate`` are required by the E97 contract.
    The shared engine receives them as the signal that selects its compile-time
    ``SPLIT_EDIT`` branch.
    """

    if erase_gate is None or value_write_gate is None:
        raise ValueError(
            "E97 requires both erase_gate and value_write_gate; "
            "use the E88 API for a recurrence without split edit"
        )

    # Resolve on every call so existing instrumentation that patches the shared
    # engine continues to observe E97 execution through this facade.
    from .e88_triton_optimized import e88_triton_optimized_apply

    return e88_triton_optimized_apply(
        training,
        k,
        v,
        q,
        decay,
        g,
        S0,
        n_heads,
        apply_gate,
        normalize_kq,
        checkpoint_interval,
        apply_silu_qkv=apply_silu_qkv,
        raw_write=raw_write,
        linear_state=linear_state,
        erase_gate=erase_gate,
        value_write_gate=value_write_gate,
    )


__all__ = ["e97_split_edit_triton_apply"]
