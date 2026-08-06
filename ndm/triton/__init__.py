"""Triton kernel implementations for Elman models.

This package mirrors selected CUDA kernels in Triton, primarily as a
research vehicle for studying alternative implementations alongside the
hand-written CUDA kernels in ``elman.cuda``.
"""

# IMPORTANT: avoid shadowing the global ``triton`` package. We use absolute
# imports inside this subpackage's modules (``import triton`` and
# ``import triton.language as tl``); Python's import machinery resolves
# those against site-packages, not against this package, because there is
# no relative-import ambiguity in those statements.

from __future__ import absolute_import

from .e88_triton_forward import (
    e88_triton_forward,
    e88_torch_reference,
)
from .e88_triton_backward import (
    e88_triton_backward,
    e88_triton,
    E88TritonFunction,
)
from .e88_triton_optimized import (
    e88_triton_optimized_apply,
)
from .e97_sequential import e97_split_edit_triton_apply
from .unified_cell_forward import (
    unified_cell_forward,
    unified_cell_torch_reference,
    PHI_NAME_TO_CODE,
    PHI_IDENTITY, PHI_TANH, PHI_GAMMA_MIX, PHI_RELU, PHI_SOFTPLUS,
)
from .unified_cell_backward import (
    unified_cell_backward,
    unified_cell,
    UnifiedCellFunction,
)
from .e97_chunked_autograd import e97_delta_chunked_triton
from .e97_multiquery_autograd import (
    e97_multiquery_chunked_triton,
    E97MultiQueryChunkedFn,
)
from .e97_moe_ep import (
    EP_SIZE,
    EXPERTS_PER_RANK,
    EPSendPlan,
    EPLocalPlan,
    build_ep_send_plan,
    build_local_expert_plan,
    unpack_local_expert_rows,
)
from .e97_moe_fused import (
    FUSED_E97_MOE_ABI,
    FusedMoEForwardResult,
    fused_packed_local_experts_autograd,
    fused_routed_swiglu_forward,
    fused_shared_expert_autograd,
    fused_shared_top3_combine_autograd,
    fused_top3_router_autograd,
    fused_schedulefree_adamw_update_,
    fused_schedulefree_lerp_,
    fused_shared_routed_swiglu_autograd_parity,
    fused_shared_routed_swiglu_backward,
    fused_shared_routed_swiglu_forward,
    fused_shared_routed_swiglu_forward_backward_parity,
)

__all__ = [
    "e88_triton_forward",
    "e88_torch_reference",
    "e88_triton_backward",
    "e88_triton",
    "E88TritonFunction",
    "e88_triton_optimized_apply",
    "e97_split_edit_triton_apply",
    "unified_cell_forward",
    "unified_cell_torch_reference",
    "unified_cell_backward",
    "unified_cell",
    "UnifiedCellFunction",
    "e97_delta_chunked_triton",
    "e97_multiquery_chunked_triton",
    "E97MultiQueryChunkedFn",
    "EP_SIZE",
    "EXPERTS_PER_RANK",
    "EPSendPlan",
    "EPLocalPlan",
    "build_ep_send_plan",
    "build_local_expert_plan",
    "unpack_local_expert_rows",
    "FUSED_E97_MOE_ABI",
    "FusedMoEForwardResult",
    "fused_packed_local_experts_autograd",
    "fused_routed_swiglu_forward",
    "fused_shared_expert_autograd",
    "fused_shared_top3_combine_autograd",
    "fused_top3_router_autograd",
    "fused_schedulefree_adamw_update_",
    "fused_schedulefree_lerp_",
    "fused_shared_routed_swiglu_autograd_parity",
    "fused_shared_routed_swiglu_backward",
    "fused_shared_routed_swiglu_forward",
    "fused_shared_routed_swiglu_forward_backward_parity",
    "PHI_NAME_TO_CODE",
    "PHI_IDENTITY", "PHI_TANH", "PHI_GAMMA_MIX", "PHI_RELU", "PHI_SOFTPLUS",
]

# Pin Triton autotune config selection (kill the init-wedge / autotune storm).
# Default ON; env-gated. Patches triton's Autotuner.run at the class level so it
# also covers FLA's @triton.autotune kernels (layernorm fwd/bwd, gated-delta
# chunk, conv) used by the gdn2-mlp arm and the E97 norms/gates. Disable with
# NDM_PIN_TRITON_AUTOTUNE=0 to restore the original sweep. See pin_autotune.py.
from . import pin_autotune as pin_autotune  # noqa: E402
pin_autotune.maybe_install_from_env()
__all__.append("pin_autotune")
