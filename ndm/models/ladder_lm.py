"""
Language Model wrapper for E-Series Elman models.

E-Series:
    e0: Stock Elman - tanh + h*silu(W_gate@x) gating
    e1: Mamba-Gated Elman - Mamba2-style split projection gating
    e2: Slot Elman - Multi-slot memory (64x like Mamba2)

Architecture matches Mamba exactly:
    - Fused add+norm using mamba_ssm.ops.triton.layer_norm
    - Block pattern: residual = x + residual; x = norm(residual); x = mixer(x)
    - RMSNorm (not LayerNorm) for efficiency
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# Import fused ops from mamba_ssm for exact architecture match
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn
    FUSED_NORM_AVAILABLE = True
except ImportError:
    FUSED_NORM_AVAILABLE = False
    RMSNorm = nn.RMSNorm  # Fallback to PyTorch

from .stock_elman import StockElman
from .counter_baseline import ReLURNNLayer, LSTMLayer
from .mamba_gated_elman import MambaGatedElman
from .softsign_elman import SoftsignElman
from .diagonal_state_elman import DiagonalStateElman
from .slot_elman import SlotElman
from .lowrank_slot_elman import LowRankSlotElman
from .lowrank_elman import LowRankElman
from .pure_lowrank_elman import PureLowRankElman
from .diagonal_elman import DiagonalElman
from .scaled_lowrank_elman import ScaledLowRankElman
from .hybrid_elman import HybridElman
from .multiscale_elman import MultiScaleElman
from .selective_elman import SelectiveElman
from .selective_gated_elman import SelectiveGatedElman
from .matrix_state_elman import MatrixStateElman
from .selective_wh_elman import SelectiveWhElman
from .haware_gate_elman import HAwareGateElman
from .simplified_gate_elman import SimplifiedGateElman
from .mamba2_informed_elman import Mamba2InformedElman
from .structured_elman import StructuredElman
from .structured_elman_attention import StructuredElmanAttention
from .dual_memory_elman import DualMemoryElman
from .e24_single_gemm import E24Layer
from .e25_entmax import E25DualMemoryElman
from .e26_parallel import E26DualMemoryElman
from .e28_conv_elman import E28ConvElman
from .e30_diagonal_gated import E30DiagonalGated
from .e31_sparse_gated import E31SparseGated
from .e32_no_presilu import E32NoPresilu
from .e33_self_gate import E33SelfGate
from .e34_diagonal_wh import E34DiagonalWh
from .e35_cubic_gate import E35CubicGate
from .e36_linear_recurrence import E36LinearRecurrence
from .e37_tied_weights import E37TiedWeights
from .e37_tied_weights_v2 import E37TiedWeightsV2
from .e38_no_wx import E38NoWx
from .e39_no_bias import E39NoBias
from .e40_no_presilu import E40NoPresilu
from .e41_diagonal_wx import E41DiagonalWx
from .e42_linear_tied import E42LinearTied
from .e43_scalar_decay import E43ScalarDecay
from .e44_diagonal_w import E44DiagonalW
from .e45_pure_accumulation import E45PureAccumulation, E45bWithDecay
from .e46_no_in_proj import E46NoInProj
from .e48_no_projections import E48NoProjections
from .e51_no_self_gate import E51NoSelfGate
from .e52_quadratic_gate import E52QuadraticGate, E52bSignedQuadratic
from .e53_sigmoid_gate import E53SigmoidGate
from .e54_diagonal_no_proj import E54DiagonalNoProj
from .e55_scalar_no_proj import E55ScalarNoProj
from .e56_concat_elman import E56ConcatElman
from .e57_learned_radius import E57LearnedRadius
from .e58_learned_radii import E58LearnedRadii
from .e59_highway import E59Highway, E59bGatedHighway, E59cMixedHighway
from .e60_residual_nonlinear import E60ResidualNonlinear, E60bGatedResidual, E60cForgetGate
from .e61_decay_gated import E61DecayGated, E61bAdditiveDecay, E61cTiedDecay
from .e62_selective_write import E62SelectiveWrite, E62bDecaySelective, E62cTiedSelective
from .e63_nonlinear_delta import E63NonlinearDelta, E63aComplementary, E63bIndependent, E63cHDependent, E63dResidual
from .e63m_matrix_nonlinear import E63mMatrixNonlinear, E63mFull, E63mLite, E63mRNN
from .e64_additive_h import E64AdditiveH
from .e65_diagonal_h import E65DiagonalH
from .e66_lowrank_h import E66LowRankH
from .e67_h_gated import E67HGated, E67HGatedDiagonal, E67HGatedLowRank
from .e68_self_gating import E68SelfGating, E68SelfGatingStandard, E68SelfGatingInverse
from .gated_delta_net import GatedDeltaNet, GatedDeltaNetVector
from .fla_gated_delta import FLAGatedDeltaNetLayer
from .external_gdn2 import GDN2ExternalLayer, GDN2ExternalMLPLayer, OFFICIAL_GDN2_MLP_RATIO
from .e91_matmat import E91MatMat
from .e92_matmat import E92MatMat
from .e93_minimal import E93Minimal
from .llama_baseline import LlamaLayer
from .e70_matrix_linear import E70MatrixLinear
from .e71_matrix_gated import E71MatrixGated
from .e72_matrix_selfgate import E72MatrixSelfGate, E72MatrixSelfGateStandard, E72MatrixSelfGateInverse
from .e73_matrix_nonlinear import E73MatrixNonlinear, E73MatrixColumn, E73MatrixRow, E73MatrixFull
from .e74_v2 import E74v2
from .e75_gated_delta import E75GatedDelta
from .e75_multihead import E75MultiHead
from .e88_fla_hybrid import E88FLAHybrid
from .e97 import E97SplitEditLayer
from .unified_cell import UnifiedCellLayer
from .typed_head_mixture import TypedHeadMixtureLayer
from .phi_shell import PhiShellLayer
from .complex_eig_head import ComplexEigHeadLayer
from .mlp_mem_head import MlpMemHeadLayer
from .complex_eig_lm import RealEigGDNShellLayer
from .e89_residual_state import E89ResidualStateCell
from .e76_logspace_delta import E76LogSpaceDelta
from .e77_linear_matrix import E77LinearMatrix
from .e78_projected_matrix import E78ProjectedMatrix
from .e79_coupled_matrix import E79CoupledMatrix
from .e83_circular_tower import E83CircularTower
from .e85_input_as_matrix import E85InputAsMatrixLayer
from .e86_input_matrix_delta import E86InputMatrixDeltaLayer
from .e87_sparse_block import E87SparseBlockLayer
from .mom_e88 import MoME88
from .e90_dual_rate import E90DualRate
from .e1_multihead import E1MultiHead


class _LadderProtocolAdapter(nn.Module):
    """Adapt a stateless full-sequence mixer to the production LadderLM layer
    protocol.

    The expressivity layers ``TypedHeadMixtureLayer`` and ``UnifiedCellLayer``
    were written for the ``HybridLadderLM`` path: their ``forward(x)`` takes a
    single ``[B,T,dim]`` tensor and returns a bare ``[B,T,dim]`` tensor (they
    process the whole sequence chunkwise and carry no hidden state across
    calls).  The production ``LadderLM.forward`` instead calls
    ``layer(x, prev_hidden)`` and unpacks ``(out, h_final)``.

    This thin wrapper bridges the two: it accepts (and ignores) the
    ``prev_hidden`` positional arg and returns ``(out, None)`` — the same
    "no recurrent carry" contract ``FLAGatedDeltaNetLayer`` already honours when
    ``use_cache=False``.  The wrapped module's parameters live under
    ``self.inner.*`` so a fresh model's checkpoint round-trips against itself.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def set_layer_idx(self, idx):
        if hasattr(self.inner, 'set_layer_idx'):
            self.inner.set_layer_idx(idx)

    def forward(self, x, prev_hidden=None, **kwargs):  # noqa: ARG002 (protocol)
        out = self.inner(x)
        if isinstance(out, tuple):
            out = out[0]
        return out, None


def get_ladder_level(level):
    """Get the module class for a specific ladder level.

    Args:
        level: Integer level (0-6, 8-10) or 'mamba2'

    Returns:
        Layer class
    """
    levels = {
        0: StockElman,
        1: MambaGatedElman,
        2: SlotElman,
        3: LowRankSlotElman,
        4: LowRankElman,
        5: PureLowRankElman,
        6: DiagonalElman,
        8: ScaledLowRankElman,
        9: HybridElman,
        10: MultiScaleElman,
        11: SelectiveElman,
        12: SelectiveGatedElman,  # E12: Hidden-state-dependent gating
        14: MatrixStateElman,  # E14: Matrix state with outer product update
        15: SoftsignElman,  # E15: E1 with softsign instead of tanh
        16: DiagonalStateElman,  # E16: Mamba2 efficiency + E1 nonlinearity
        17: SelectiveWhElman,  # E17: Input-dependent gating on W_h @ h
        '18a': lambda **kw: HAwareGateElman(gate_mode=0, **kw),  # E18-A: gate = z + h
        '18b': lambda **kw: HAwareGateElman(gate_mode=1, **kw),  # E18-B: gate = z + Rh
        '18e': lambda **kw: HAwareGateElman(gate_mode=2, **kw),  # E18-E: no gate
        '19a': lambda **kw: SimplifiedGateElman(gate_mode=0, **kw),  # E19-A: gate = Wx + h
        '19b': lambda **kw: SimplifiedGateElman(gate_mode=1, **kw),  # E19-B: gate = h-only
        '19d': lambda **kw: SimplifiedGateElman(gate_mode=2, **kw),  # E19-D: residual + z
        '19e': lambda **kw: SimplifiedGateElman(gate_mode=3, **kw),  # E19-E: residual + Wx + h
        20: Mamba2InformedElman,  # E20: Mamba2-style matrix state
        21: StructuredElman,  # E21: MIMO with nonlinear state
        30: E30DiagonalGated,  # E30: E1 + diagonal gating (SSM-style selectivity)
        31: E31SparseGated,  # E31: E1 + sparse gating via softplus (default α=1.5)
        '31a': lambda **kw: E31SparseGated(alpha=2.0, **kw),  # E31a: relu gating (strictly sparse)
        '31b': lambda **kw: E31SparseGated(alpha=1.5, **kw),  # E31b: softplus gating (smooth sparse)
        32: E32NoPresilu,  # E32: E1 without pre-silu activation (simplification test)
        33: E33SelfGate,  # E33: E1 with self-gating: output = h * silu(h) instead of h * silu(z)
        34: E34DiagonalWh,  # E34: E33 with diagonal W_h (d vector instead of matrix)
        35: E35CubicGate,  # E35: E1 with cubic gating: output = h^3 instead of h * silu(z)
        36: E36LinearRecurrence,  # E36: Linear recurrence (no tanh!) + self-gate
        37: E37TiedWeights,  # E37: Tied weights: W_x = W_h = W, single GEMM per step
        '37v2': E37TiedWeightsV2,  # E37v2: Tied weights with batched GEMM optimization
        38: E38NoWx,  # E38: No W_x: h = tanh(x + W_h @ h_prev + b), removes input transform
        39: E39NoBias,  # E39: No bias: h = tanh(x + W_h @ h_prev), simplest recurrence
        40: E40NoPresilu,  # E40: No pre-silu: x_proj = in_proj(x), NOT silu(in_proj(x))
        41: E41DiagonalWx,  # E41: Diagonal W_x (d_x vector instead of matrix)
        42: E42LinearTied,  # E42: Linear recurrence + tied weights (E36 + E37)
        43: E43ScalarDecay,  # E43: Scalar decay (λ replaces d×d matrix)
        44: E44DiagonalW,  # E44: Diagonal W (per-dimension decay, Mamba2-style)
        45: E45PureAccumulation,  # E45: Pure accumulation (W=I, no params in recurrence)
        '45b': E45bWithDecay,  # E45b: Pure accumulation + learned scalar decay
        46: E46NoInProj,  # E46: No in_proj (recurrence on raw embeddings)
        48: E48NoProjections,  # E48: No projections at all (minimal recurrent layer)
        51: E51NoSelfGate,  # E51: No self-gate (linear output)
        52: E52QuadraticGate,  # E52: Pure quadratic gate (h²)
        '52b': E52bSignedQuadratic,  # E52b: Signed quadratic (h * |h|)
        53: E53SigmoidGate,  # E53: Sigmoid gate only (silu, not h * silu)
        54: E54DiagonalNoProj,  # E54: Diagonal W + no projections (Mamba2-style minimal)
        55: E55ScalarNoProj,  # E55: Scalar + no projections (ultimate minimal)
        56: E56ConcatElman,  # E56: Concat Elman - W @ [x;h] instead of W_x @ x + W_h @ h
        57: E57LearnedRadius,  # E57: E1 with learned spectral radius (scalar)
        58: E58LearnedRadii,  # E58: E1 with per-dimension learned radii
        59: E59Highway,  # E59: Highway Elman (residual recurrence, gradient=I)
        '59b': E59bGatedHighway,  # E59b: Gated residual highway
        '59c': E59cMixedHighway,  # E59c: Mixed residual + small recurrent
        60: E60ResidualNonlinear,  # E60: Residual nonlinear (h + tanh(Wh + Ux))
        '60b': E60bGatedResidual,  # E60b: Gated residual nonlinear
        '60c': E60cForgetGate,  # E60c: Forget-gate style (GRU-like)
        61: E61DecayGated,  # E61: Decay-gated (α·h + (1-α)·v, Mamba2-style)
        '61b': E61bAdditiveDecay,  # E61b: Additive decay (α·h + v)
        '61c': E61cTiedDecay,  # E61c: Tied decay (single-gate GRU)
        62: E62SelectiveWrite,  # E62: Selective write ((1-k)·h + k·v, DeltaNet-style)
        '62b': E62bDecaySelective,  # E62b: Decay + selective (α·(1-k)·h + k·v)
        '62c': E62cTiedSelective,  # E62c: Tied selective (GRU-style)
        63: E63NonlinearDelta,  # E63: Nonlinear delta (UTM-class! v=tanh(Wh+Ux))
        '63a': E63aComplementary,  # E63a: Complementary gates (GRU-style)
        '63b': E63bIndependent,  # E63b: Independent gates (LSTM-style)
        '63c': E63cHDependent,  # E63c: H-dependent gates (maximum expressivity)
        '63d': E63dResidual,  # E63d: Residual nonlinear (h + α*tanh(Wh+Ux))
        '63m': E63mMatrixNonlinear,  # E63m: Matrix state + nonlinear retrieval (O(d²) state)
        '63m-full': E63mFull,  # E63m-full: Full d×d matrix state
        '63m-lite': E63mLite,  # E63m-lite: Reduced-rank N×d matrix
        '63m-rnn': E63mRNN,  # E63m-rnn: + output recurrence (Delta RNN)
        64: E64AdditiveH,  # E64: Additive h-dependence v=tanh(h+Wx) - O(d) UTM
        65: E65DiagonalH,  # E65: Diagonal h-dependence v=tanh(d*h+Wx) - O(d) UTM
        66: E66LowRankH,  # E66: Low-rank h-dependence v=tanh(UVh+Wx) - O(d*r) UTM
        '66r16': lambda **kw: E66LowRankH(rank=16, **kw),  # E66 rank=16
        '66r64': lambda **kw: E66LowRankH(rank=64, **kw),  # E66 rank=64
        '66r128': lambda **kw: E66LowRankH(rank=128, **kw),  # E66 rank=128
        67: E67HGated,  # E67: H-dependent gate α=σ(Wx+d*h) - O(d) UTM
        '67d': E67HGatedDiagonal,  # E67d: Diagonal h in gate
        '67lr': E67HGatedLowRank,  # E67lr: Low-rank h in gate
        68: E68SelfGating,  # E68: Self-gating v=tanh(Wx)*σ(h) - O(d) UTM
        '68s': E68SelfGatingStandard,  # E68s: Standard self-gating
        '68i': E68SelfGatingInverse,  # E68i: Inverse (resist overwrite)
        'gdn': GatedDeltaNet,  # GatedDeltaNet: ICLR 2025 baseline (matrix state)
        'gdn-vec': GatedDeltaNetVector,  # GatedDeltaNet Vector: Simplified (vector state)
        'fla-gdn': FLAGatedDeltaNetLayer,  # FLA GatedDeltaNet: Optimized Triton kernels (ICLR 2025)
        'gdn2': GDN2ExternalLayer,  # External NVIDIA GatedDeltaNet-2 checkout
        'gdn2-mlp': GDN2ExternalMLPLayer,  # External GDN-2 mixer plus official-style SwiGLU MLP
        'llama': LlamaLayer,  # Llama Transformer: attention baseline
        70: E70MatrixLinear,  # E70: Matrix Linear (E42-style) - linear accum + self-gate
        '70n32': lambda **kw: E70MatrixLinear(n_state=32, **kw),
        '70n128': lambda **kw: E70MatrixLinear(n_state=128, **kw),
        71: E71MatrixGated,  # E71: Matrix Gated (E67-style) - S affects gate
        '71n32': lambda **kw: E71MatrixGated(n_state=32, **kw),
        '71n128': lambda **kw: E71MatrixGated(n_state=128, **kw),
        72: E72MatrixSelfGate,  # E72: Matrix Self-Gate (E68-style) - S gates value
        '72s': E72MatrixSelfGateStandard,  # Standard: content enables writing
        '72i': E72MatrixSelfGateInverse,  # Inverse: content resists writing
        73: E73MatrixNonlinear,  # E73: Matrix Nonlinear (E1-style) - S inside tanh
        '73c': E73MatrixColumn,  # Column modulation
        '73r': E73MatrixRow,  # Row modulation
        '73f': E73MatrixFull,  # Full element-wise modulation
        75: E75GatedDelta,  # E75: Gated Delta (E74 delta rule + E61 forget gate)
        '75n32': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 32}),
        '75n48': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 48}),
        '75n64': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 64}),
        '75n96': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 96}),
        # E75 Multi-Head variants (H independent matrix states)
        'E75h2': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 2}),
        'E75h4': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4}),
        'E75h8': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8}),
        'E75h2n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 2, 'n_state': 48}),
        'E75h4n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 24}),
        'E75h8n16': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 16}),
        'E75h4n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 32}),
        'E75h8n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 24}),
        # E75 Multi-Head parameter scan variants (n_state must be multiple of 8)
        'E75h3n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 3, 'n_state': 24}),
        'E75h3n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 3, 'n_state': 32}),
        'E75h3n40': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 3, 'n_state': 40}),
        'E75h4n16': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 16}),
        'E75h4n40': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 40}),
        'E75h4n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 48}),
        'E75h5n16': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 5, 'n_state': 16}),
        'E75h5n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 5, 'n_state': 24}),
        'E75h5n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 5, 'n_state': 32}),
        'E75h6n16': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 6, 'n_state': 16}),
        'E75h6n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 6, 'n_state': 24}),
        'E75h6n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 6, 'n_state': 32}),
        'E75h8n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 32}),
        'E75h8n64': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 64}),
        'E75h8n96': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 96}),
        'E75h8n128': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 128}),
        'E75h8n160': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 160}),
        'E75h12n128': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 12, 'n_state': 128}),
        'E75h16n112': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 16, 'n_state': 112}),
        # High-head configs to match mamba2 state while staying in shared memory
        'E75h16n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 16, 'n_state': 32}),
        'E75h16n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 16, 'n_state': 48}),
        'E75h16n64': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 16, 'n_state': 64}),
        'E75h32n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 32, 'n_state': 32}),
        'E75h32n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 32, 'n_state': 48}),
        'E75h32n64': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 32, 'n_state': 64}),
        'E75h48n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 48, 'n_state': 32}),
        'E75h48n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 48, 'n_state': 48}),
        'E75h64n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 64, 'n_state': 32}),
        'E75h64n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 64, 'n_state': 48}),
        'E75h128n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 128, 'n_state': 32}),
        'E75h128n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 128, 'n_state': 48}),
        # E75 Post-Conv variants (FLA-GDN style: separate convs on k,v,q AFTER projections)
        # This provides per-role local context before the associative memory update
        'E75pc': lambda **kw: E75MultiHead(**{**kw, 'use_conv': True, 'conv_mode': 'post'}),
        'E75pch4': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'use_conv': True, 'conv_mode': 'post'}),
        'E75pch4n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 24, 'use_conv': True, 'conv_mode': 'post'}),
        'E75pch4n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 32, 'use_conv': True, 'conv_mode': 'post'}),
        'E75pch4n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 48, 'use_conv': True, 'conv_mode': 'post'}),
        'E75pch8n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 24, 'use_conv': True, 'conv_mode': 'post'}),
        'E75pch8n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 32, 'use_conv': True, 'conv_mode': 'post'}),
        # E76: Log-Space Gated Delta (E75 + Mamba2/FLA-GDN stability techniques)
        # Default: tanh + log_gate (nonlinear recurrence with stable params)
        76: E76LogSpaceDelta,
        '76n32': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 32}),
        '76n48': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 48}),
        '76n64': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 64}),
        '76n96': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 96}),
        # E76 configuration variants:
        # -t = tanh (nonlinear), -l = linear, -log = log_gate, -sig = sigmoid_gate
        '76-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': True, 'log_space_gate': True}),
        '76-t-sig': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': True, 'log_space_gate': False}),
        '76-l-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': False, 'log_space_gate': True}),
        '76-l-sig': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': False, 'log_space_gate': False}),
        # With n_state variants
        '76n32-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 32, 'use_tanh': True, 'log_space_gate': True}),
        '76n48-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 48, 'use_tanh': True, 'log_space_gate': True}),
        '76n64-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 64, 'use_tanh': True, 'log_space_gate': True}),
        # E74v2: Extended delta rule variants (CUDA kernel support)
        '74v2': E74v2,  # E74v2 base: delta update, output gate
        '74v2-delta': lambda **kw: E74v2(update_type='delta', **kw),
        '74v2-residual': lambda **kw: E74v2(update_type='residual', **kw),
        '74v2-ntm': lambda **kw: E74v2(update_type='ntm', **kw),
        '74v2-retrieved_gate': lambda **kw: E74v2(update_type='retrieved_gate', **kw),
        '74v2-ema': lambda **kw: E74v2(update_type='ema', **kw),
        '74v2-delta-input': lambda **kw: E74v2(update_type='delta', gate_type='input', **kw),
        '74v2-ema-input': lambda **kw: E74v2(update_type='ema', gate_type='input', **kw),
        # E77: Linear Matrix State (E42's linear recurrence + matrix state + fused projections)
        77: E77LinearMatrix,
        '77n32': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 32}),
        '77n48': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 48}),
        '77n64': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 64}),
        '77n96': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 96}),
        # E78: Projected Matrix State (E77 + random projection for sparse efficient state)
        78: E78ProjectedMatrix,
        '78n64s32': lambda **kw: E78ProjectedMatrix(**{**kw, 'n_effective': 64, 'n_small': 32}),
        '78n128s32': lambda **kw: E78ProjectedMatrix(**{**kw, 'n_effective': 128, 'n_small': 32}),
        '78n256s64': lambda **kw: E78ProjectedMatrix(**{**kw, 'n_effective': 256, 'n_small': 64}),
        # E79: Coupled Memory-Modulation Matrix System
        # Two coupled matrices (S content + M modulation) with mutual gating control
        79: E79CoupledMatrix,
        '79n32': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 32}),
        '79n48': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 48}),
        '79n64': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 64}),
        '79n96': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 96}),
        # E79 bias ablations
        '79nb': lambda **kw: E79CoupledMatrix(**{**kw, 'use_bias': False}),  # No bias
        '79ib': lambda **kw: E79CoupledMatrix(**{**kw, 'input_bias': True}),  # Input-dependent bias
        '79n32nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 32, 'use_bias': False}),
        '79n32ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 32, 'input_bias': True}),
        '79n48nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 48, 'use_bias': False}),
        '79n48ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 48, 'input_bias': True}),
        '79n64nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 64, 'use_bias': False}),
        '79n64ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 64, 'input_bias': True}),
        '79n96nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 96, 'use_bias': False}),
        '79n96ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 96, 'input_bias': True}),

        # E83: Circular K-Tower (K matrices in mutual gating circle)
        # Default: K=3, n_state=32, fixed bias
        83: E83CircularTower,
        # K=2 (like E79 but circular)
        '83k2': lambda **kw: E83CircularTower(**{**kw, 'K': 2, 'n_state': 32}),
        '83k2nb': lambda **kw: E83CircularTower(**{**kw, 'K': 2, 'n_state': 32, 'use_bias': False}),
        '83k2ib': lambda **kw: E83CircularTower(**{**kw, 'K': 2, 'n_state': 32, 'input_bias': True}),
        # K=4, n_state=32 (4 heads, 4K state)
        '83k4n32': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 32}),
        '83k4n32nb': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 32, 'use_bias': False}),
        '83k4n32ib': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 32, 'input_bias': True}),
        # K=4, n_state=24 (4 heads, 2.3K state)
        '83k4n24': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 24}),
        '83k4n24nb': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 24, 'use_bias': False}),
        '83k4n24ib': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 24, 'input_bias': True}),
        # K=8, n_state=24 (8 heads, 4.6K state)
        '83k8n24': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 24}),
        '83k8n24nb': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 24, 'use_bias': False}),
        '83k8n24ib': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 24, 'input_bias': True}),
        # K=8, n_state=16 (8 heads, 2K state)
        '83k8n16': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 16}),
        '83k8n16nb': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 16, 'use_bias': False}),
        '83k8n16ib': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 16, 'input_bias': True}),

        # E85: Input-As-Matrix (dim = n_state^2, input IS the transformation matrix)
        # Default: n_state=32 -> dim=1024
        85: E85InputAsMatrixLayer,
        # n_state variants (dim = n_state^2)
        '85n16': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 16}),  # dim=256
        '85n24': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 24}),  # dim=576
        '85n32': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 32}),  # dim=1024
        '85n48': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 48}),  # dim=2304

        # E86: Input-as-Matrix Delta Rule (E85's input-as-matrix + E75's delta rule)
        # Default: n_state=32, n_heads=1 -> cell_dim=1024, output=32
        86: E86InputMatrixDeltaLayer,
        # n_state variants (single head)
        '86n16': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 16}),  # cell_dim=256, out=16
        '86n24': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 24}),  # cell_dim=576, out=24
        '86n32': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 32}),  # cell_dim=1024, out=32
        '86n48': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 48}),  # cell_dim=2304, out=48
        # Multi-head variants for capacity scaling
        '86h2': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_heads': 2}),  # 2 heads, cell_dim=2048, out=64
        '86h4': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_heads': 4}),  # 4 heads, cell_dim=4096, out=128
        '86h2n24': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 24, 'n_heads': 2}),  # cell_dim=1152, out=48
        '86h4n24': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 24, 'n_heads': 4}),  # cell_dim=2304, out=96
        '86h4n16': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 16, 'n_heads': 4}),  # cell_dim=1024, out=64
        '86h8n16': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 16, 'n_heads': 8}),  # cell_dim=2048, out=128

        # E87: Content-Gated Sparse Block Memory
        # n_blocks blocks of n_state×n_state, top_k updated per step
        '87': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 4, 'top_k': 2}),
        '87b4k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 4, 'top_k': 2}),
        '87b4k1': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 4, 'top_k': 1}),
        '87b8k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 24, 'n_blocks': 8, 'top_k': 2}),
        '87b8k4': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 24, 'n_blocks': 8, 'top_k': 4}),
        '87b4k2n48': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 48, 'n_blocks': 4, 'top_k': 2}),
        '87b6k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 6, 'top_k': 2}),
        '87b6k3': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 6, 'top_k': 3}),
        '87b8k3': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 24, 'n_blocks': 8, 'top_k': 3}),
        # 16 blocks variants
        '87b16k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 16, 'top_k': 2}),
        '87b16k4': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 16, 'top_k': 4}),
        '87b16k6': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 16, 'top_k': 6}),
        '87b16k8': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 16, 'top_k': 8}),
        # 32 blocks variants (n_state=16 for CUDA compatibility)
        '87b32k4': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 32, 'top_k': 4}),
        '87b32k8': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 32, 'top_k': 8}),
        '87b32k12': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 32, 'top_k': 12}),
        '87b32k16': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 16, 'n_blocks': 32, 'top_k': 16}),
        # E88: FLA-GDN Hybrid with Nonlinear Matrix State
        # Combines FLA-GDN's proven design (Mamba2 decay, output gating, short conv)
        # with E75's nonlinear matrix state: S = tanh(decay * S + outer(delta, k_norm))
        88: E88FLAHybrid,
        'E88': E88FLAHybrid,
        # === UNIFIED parameterized matrix-recurrence cell (unified-cell-triton) ===
        # ONE Triton kernel; knobs (lambda gain, beta correction, gamma phi) select capability.
        # Pinned-preset corner arms:
        'unified-track':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'track'}),
        'unified-count':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'count'}),
        'unified-latch':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'latch'}),
        'unified-nonlin':  lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'nonlin'}),
        'unified-e88base': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'e88base'}),
        # LEARNED-knob arms: free gain (incl >=1) vs CLAMPED to (0,1) -- the un-cribbing demo.
        'unified-learned':       lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5}),
        'unified-learned-free':  lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5}),
        'unified-learned-clamp': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.0}),
        # LEARNABILITY: spread per-head knobs ACROSS the corners at init (free gain),
        # so descent REFINES specialization. Pair with a knob-specific higher LR
        # (train_hybrid --knob_lr_mult) so lambda/beta/gamma actually move.
        'unified-learned-spread': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5, 'spread_init': True}),
        # SPECIALIZATION STUDY (horizontal head-type hybridization). The regularizer
        # arms reuse the GENERIC-init learned-free cell below and apply the
        # specialization-pressure penalty at train time (train_hybrid --spec_reg).
        # TYPE-DICTIONARY: K shared learnable prototype knobs + per-head soft weights.
        'unified-dict4':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'dictionary', 'phi': 'gamma_mix', 'lam_max': 1.5, 'n_proto': 4}),
        'unified-dict8':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'dictionary', 'phi': 'gamma_mix', 'lam_max': 1.5, 'n_proto': 8}),
        # FIXED-TYPE POPULATION (floor): heads hard-assigned to corners, projections only.
        'unified-fixedpop': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'fixed_pop', 'phi': 'gamma_mix', 'lam_max': 1.5}),
        # === E98 = E97 split-gate (decoupled erase b*k / value-write w*v) ON TOP of
        # the unified capability-span + horizontal specialization. The split gate
        # makes the correction term E97-rich: pre = lam*S - beta*k((b*k)^T S) +
        # i*k(w*v)^T. b=w=1 recovers the unified cell so all four corners persist.
        # Pinned-corner arms (split-gate on):
        'e98-track':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'track',  'split_gate': True}),
        'e98-count':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'count',  'split_gate': True}),
        'e98-latch':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'latch',  'split_gate': True}),
        'e98-nonlin':  lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'nonlin', 'split_gate': True}),
        # E98 FIVE-CORNER 5th pinned preset: leaky-linear associative-memory
        # workhorse (the GDN/Mamba recall regime). Should WIN the MQAR recall probe
        # where the four exotic corners fail.
        'e98-leaky':   lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'leaky-linear', 'split_gate': True}),
        # E98 SIXTH pinned preset: gated-delta backbone (beta=1 clean overwrite,
        # identity phi, INPUT-DEPENDENT gated decay lambda_t via decay_gate). The
        # GDN-in-E98 operating point -- the key test of whether the GDN recall+track
        # regime is reachable inside the unified cell. neg-eig (reflection for S5)
        # arises naturally when the decay gate drives lambda_t<1 (eig=lambda_t-1<0).
        'e98-gated-delta': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'pinned', 'preset': 'gated-delta', 'split_gate': True}),
        # Generic-init learned (free gain) + split gate.
        'e98-learned-free': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5, 'split_gate': True}),
        # WINNING specialization form (SPECIALIZATION_STUDY): spread-init + knob-LR,
        # now with the E97 split gate. This is the E98 learnability arm.
        # spread-4 = the current 4 exotic corners; spread-5 adds the leaky-linear
        # workhorse so ~1/5 of the heads place on associative recall (E98 5-corner).
        'e98-learned-spread': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5, 'spread_init': True, 'n_spread_corners': 4, 'split_gate': True}),
        'e98-learned-spread5': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5, 'spread_init': True, 'n_spread_corners': 5, 'split_gate': True}),
        # spread-6 adds the gated-delta backbone -> ~1/6 of heads place on
        # clean-overwrite delta memory (the recall+track workhorse). The placed
        # corner uses the fixed-lambda + split-gate machinery (input-dependent erase
        # b_t supplies the overwrite); the fully input-dependent decay is showcased
        # in the e98-gated-delta PRESET arm.
        'e98-learned-spread6': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5, 'spread_init': True, 'n_spread_corners': 6, 'split_gate': True}),
        # Fixed-type population floor + split gate (for completeness).
        'e98-fixedpop': lambda **kw: UnifiedCellLayer(**{**kw, 'knob_mode': 'fixed_pop', 'phi': 'gamma_mix', 'lam_max': 1.5, 'split_gate': True}),
        # CMA-tunable E98 learnability arm (cma-capability): same winning FORM as
        # e98-learned-spread (spread-init + split-gate + gamma_mix), but with **kw
        # LAST so the meta-search can override lam_max / beta_max / igain_max /
        # corner_mixture per candidate. knob_lr_mult is a train-time arg.
        'e98-cma': lambda **kw: UnifiedCellLayer(**{
            'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5,
            'spread_init': True, 'split_gate': True, **kw}),
        # typed-gdn-2-head: a horizontal population of NATIVE recurrent head types
        # (native GDN-2 delta-memory recall heads + E98 corner specialists) in one
        # layer, allocated deterministically from per-type logits.
        'typed-gdn2': lambda **kw: TypedHeadMixtureLayer(**kw),
        # phi-shell (task phi-explore): FLA GDN-2 gated-delta plumbing with a
        # PER-STEP state nonlinearity phi as a swept free axis. phi='identity'
        # is the linear gdn-neg baseline realized in the same code path; every
        # other phi is one elementwise function away. The vehicle for the
        # capability-vs-phi sweep on the depth-growing modular_quadratic cliff.
        'phi-shell': lambda **kw: PhiShellLayer(**kw),
        # complex-eig (task complex-eig-capability): FLA GDN-2 gated-delta plumbing
        # with a PER-CHANNEL COMPLEX eigenvalue lambda = r*e^{i*theta} transition
        # (rotation-scaling). cplx_real_only=True snaps theta to {0,pi} and freezes
        # it -> the MATCHED real-eigenvalue (positive+negative) control with
        # identical params/kernel/compute (only rotation removed). The vehicle for
        # the periodic/mod-k/positional capability battery: does the complex axis
        # unlock a capability the real-eigenvalue cell cannot reach?
        'complex-eig': lambda **kw: ComplexEigHeadLayer(**kw),
        # complex-eigenvalue head (task complex-eig-lm): the SAME FLA GatedDeltaNet
        # shell as 'real-eig-gdn' below, with the per-head real scalar decay
        # replaced by a per-key-channel COMPLEX eigenvalue lambda=r*e^{i theta}
        # (complex-everywhere) + an optional per-step bounded-phi subset of heads.
        # 'real-eig-gdn' is the matched real-eigenvalue control (native gated-delta).
        # Both wrapped so LadderLM's (out, h) protocol works. None-valued generic
        # kwargs (e.g. n_heads when unset) are stripped so int() coercion is safe.
        'complex-eig-lm': lambda **kw: _LadderProtocolAdapter(
            ComplexEigHeadLayer(**{k: v for k, v in kw.items() if v is not None})),
        'real-eig-gdn': lambda **kw: _LadderProtocolAdapter(
            RealEigGDNShellLayer(**{k: v for k, v in kw.items() if v is not None})),
        # Production-LM-protocol variants of the two expressivity mixers: wrapped
        # so LadderLM's `out, h = layer(x, prev_hidden)` calling convention works.
        # These are the E99 typed-Emender and E98-CMA candidates wired into the
        # real train.py / LadderLM / FLA-GDN path (task: wire-e99-e98).
        'typed-gdn2-lm': lambda **kw: _LadderProtocolAdapter(TypedHeadMixtureLayer(**kw)),
        # mlp-mem (task nlmem-triton, spec NONLIN_MEMORY_SPEC.md): the nonlinear
        # MLP-memory cell — the recurrent state is the params of a 1-hidden-layer
        # MLP written by one gated inner gradient step per token (REAL fused
        # sequential Triton fwd+bwd kernel; non-associative => no chunked scan).
        # Wrapped so LadderLM's (out, h) protocol works. None-valued generic kwargs
        # are stripped so int() coercion is safe (mirrors complex-eig-lm).
        'mlp-mem-lm': lambda **kw: _LadderProtocolAdapter(
            MlpMemHeadLayer(**{k: v for k, v in kw.items() if v is not None})),
        # Exactly-param-matched GDN-2 LM baseline (nlmem-capability convergent-loss-null):
        # same MlpMemHeadLayer shell, spec §2.3 linear gated-delta cell, LadderLM-wrapped.
        'gdn-matched-lm': lambda **kw: _LadderProtocolAdapter(
            MlpMemHeadLayer(cell='gdn', **{k: v for k, v in kw.items() if v is not None})),
        # Raw single-arg variant for the HybridLadderLM expressivity harness
        # (task nlmem-capability). HybridLadderLM calls `layer(x)` and unwraps
        # tuples, so the bare MlpMemHeadLayer (forward(x) -> tensor) is used
        # directly without the LadderLM (out, h) protocol adapter — mirrors how
        # the 'complex-eig' raw level is used by run_complex_eig_battery.py.
        'mlp-mem': lambda **kw: MlpMemHeadLayer(
            **{k: v for k, v in kw.items() if v is not None}),
        # Exactly-param-matched GDN-2 baseline for the capability A/B (nlmem-capability):
        # the SAME MlpMemHeadLayer shell with the spec §2.3 degenerate LINEAR corner —
        # FLA chunked gated-delta (linear matrix memory) replaces the nonlinear MLP
        # memory. Identical projections/conv/gate/o_proj; only the recurrent cell differs.
        'gdn-matched': lambda **kw: MlpMemHeadLayer(
            cell='gdn', **{k: v for k, v in kw.items() if v is not None}),
        'e98-cma-lm': lambda **kw: _LadderProtocolAdapter(UnifiedCellLayer(**{
            'knob_mode': 'learned', 'phi': 'gamma_mix', 'lam_max': 1.5,
            'spread_init': True, 'split_gate': True, **kw})),
        # E97: E88/NDM with GDN-2-inspired split edit gates.
        # Use --use_triton 1 for the split-edit Triton recurrence.
        'E97': E97SplitEditLayer,
        97: E97SplitEditLayer,
        # E97-M2: M2 multi-query readout (paper/review/STATE_AWARE_MLP_DESIGN.md §3).
        # The state UPDATE is the unchanged E97 split-edit delta; only the READ is
        # rank-R (multiquery_r queries/head). Built on the fused chunked split-edit
        # path (linear state, GDN-2-class throughput). Pass the rank via
        # layer_kwargs, e.g. --layer_kwargs '{"multiquery_r": 4}' (default R=2 here).
        'E97-M2': lambda **kw: E97SplitEditLayer(**{
            'multiquery_r': 2, **kw, 'use_triton': True,
            'use_chunked_e97': True, 'linear_state': True}),
        'E91': E91MatMat,
        91: E91MatMat,
        'E91r1': lambda **kw: E91MatMat(**{**kw, 'rank': 1}),
        'E91r4': lambda **kw: E91MatMat(**{**kw, 'rank': 4}),
        'E91r8': lambda **kw: E91MatMat(**{**kw, 'rank': 8}),
        'E91r16': lambda **kw: E91MatMat(**{**kw, 'rank': 16}),
        'E92': E92MatMat,
        92: E92MatMat,
        'E93': E93Minimal,
        93: E93Minimal,
        # E93 ablation cells (Python loop fallback for non-vanilla)
        'E93a_vanilla': E93Minimal,
        'E93a_no_wh': lambda **kw: E93Minimal(**{**kw, 'use_w_h': False}),
        'E93a_no_delta': lambda **kw: E93Minimal(**{**kw, 'use_delta': False}),
        'E93a_no_decay': lambda **kw: E93Minimal(**{**kw, 'use_decay': False}),
        'E93a_linear': lambda **kw: E93Minimal(**{**kw, 'nonlinearity': 'linear'}),
        'E93a_min_tanh': lambda **kw: E93Minimal(**{**kw, 'use_w_h': False, 'use_delta': False, 'use_decay': False, 'nonlinearity': 'tanh'}),
        'E93a_min_lin': lambda **kw: E93Minimal(**{**kw, 'use_w_h': False, 'use_delta': False, 'use_decay': False, 'nonlinearity': 'linear'}),
        # Minimal-2: drop W_h + decay, keep tanh + delta (the proposed "shipping minimum")
        'E93a_no_wh_no_decay': lambda **kw: E93Minimal(**{**kw, 'use_w_h': False, 'use_decay': False}),
        # Minimal-2-linear: drop W_h + decay AND tanh, keep delta only
        'E93a_no_wh_no_decay_lin': lambda **kw: E93Minimal(**{**kw, 'use_w_h': False, 'use_decay': False, 'nonlinearity': 'linear'}),
        'E88h4': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 4}),
        'E88h8': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8}),
        'E88h16': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16}),
        'E88h4n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 4, 'n_state': 32}),
        'E88h8n24': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 24}),
        'E88h8n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32}),
        'E88h8n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 48}),
        'E88h8n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 64}),
        'E88h8n96': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 96}),
        'E88h16n24': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 24}),
        'E88h16n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32}),
        'E88h16n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 64}),
        # E88 with expansion=1.0 (square state, faster/less memory)
        'E88s': lambda **kw: E88FLAHybrid(**{**{k: v for k, v in kw.items() if k != 'expansion'}, 'expansion': 1.0}),
        'E88sh8': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'expansion': 1.0}),
        'E88sh8n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 64, 'expansion': 1.0}),
        'E88sh8n96': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 96, 'expansion': 1.0}),
        'E88sh16n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0}),
        'E88sh16n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 64, 'expansion': 1.0}),
        # E88 with tie_kv=True (skip v projection, v=k, only works with expansion=1.0)
        'E88t': lambda **kw: E88FLAHybrid(**{**{k: v for k, v in kw.items() if k != 'expansion'}, 'expansion': 1.0, 'tie_kv': True}),
        'E88th8': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'expansion': 1.0, 'tie_kv': True}),
        'E88th8n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 64, 'expansion': 1.0, 'tie_kv': True}),
        'E88th8n96': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 96, 'expansion': 1.0, 'tie_kv': True}),
        'E88th16n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'tie_kv': True}),
        'E88th16n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 64, 'expansion': 1.0, 'tie_kv': True}),

        # E88 ablation variants (based on best E88sh16n32)
        # No convolution ablation
        'E88a_noconv': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False}),
        # Linear state ablation (no tanh)
        'E88a_linear': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'linear_state': True}),
        # No output gating ablation
        'E88a_nogate': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_gate': False}),
        # Simple decay ablation (sigmoid instead of Mamba2-style)
        'E88a_simpledecay': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'simple_decay': True}),
        # Combined ablations
        'E88a_noconv_linear': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'linear_state': True}),
        'E88a_noconv_nogate': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False}),
        'E88a_minimal': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'simple_decay': True}),

        # E88 round 2 ablations (based on E88a_noconv as new baseline)
        # No SiLU on projections
        'E88b_nosilu': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_silu': False}),
        # No L2 normalization on k/q
        'E88b_nol2': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_l2_norm': False}),
        # No output RMSNorm
        'E88b_nonorm': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_output_norm': False}),
        # No SiLU + no L2 norm
        'E88b_nosilu_nol2': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_silu': False, 'use_l2_norm': False}),
        # No gate + no norm (test if gating was compensating for norm)
        'E88b_nogate_nonorm': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # E88 round 3 ablations (based on E88b_nonorm - no conv, no output norm)
        # More heads, smaller state (32 heads x 16x16)
        'E88c_h32n16': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 16, 'expansion': 1.0, 'use_conv': False, 'use_output_norm': False}),
        # Fewer heads, larger state (8 heads x 64x64)
        'E88c_h8n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_output_norm': False}),
        # No gate variant of E88b_nonorm
        'E88c_nogate': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # 24 heads x 24 state (different balance)
        'E88c_h24n24': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 24, 'n_state': 24, 'expansion': 1.0, 'use_conv': False, 'use_output_norm': False}),
        # Test if simple_decay works better without norm
        'E88c_simpledecay': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_output_norm': False, 'simple_decay': True}),

        # E88 round 4 ablations (based on E88c_nogate - no conv, no gate, no norm)
        # tie_kv: v=k (reduce parameters)
        'E88d_tiekv': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'tie_kv': True}),
        # Linear state (re-test without other components)
        'E88d_linear': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'linear_state': True}),
        # 12 heads (fewer than 16)
        'E88d_h12': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 12, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # 20 heads (more than 16)
        'E88d_h20': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 20, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Simple decay with winning config
        'E88d_simpledecay': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'simple_decay': True}),
        # tie_kv + linear (most minimal with params savings)
        'E88d_tiekv_linear': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'tie_kv': True, 'linear_state': True}),

        # E88 round 5: Equal-param comparisons (~75M target)
        # h12 scaled up to ~75M (dim 1920)
        'E88e_h12_75m': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 12, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # h16 scaled down to ~48M (dim 1408)
        'E88e_h16_48m': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # h8 at ~75M (dim ~2176)
        'E88e_h8_75m': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # h10 at ~75M
        'E88e_h10_75m': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 10, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Linear state version of h12 (to confirm tanh doesn't matter)
        'E88e_h12_linear': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 12, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'linear_state': True}),

        # E88 round 6: Small states (now fixed) + fewer heads exploration
        # Small states with winning config (no conv/gate/norm)
        'E88f_h32n16': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 16, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88f_h24n24': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 24, 'n_state': 24, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Even fewer heads (continuing the trend)
        'E88f_h6': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 6, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88f_h4': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 4, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Larger state with fewer heads
        'E88f_h4n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 4, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # E88 round 7: Many heads with tiny matrices (n_state=8)
        # Testing if extreme head count helps - each head has 8x8 matrix
        'E88g_h64n8': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 8, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88g_h128n8': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 128, 'n_state': 8, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88g_h256n8': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 256, 'n_state': 8, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Medium: 16 heads with 16x16 matrices (4096 total state = same as h64n8)
        'E88g_h16n16': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 16, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Compare: fewer heads with larger matrices (same total state size as h256n8: 256*64 = h4n128: 4*4096)
        'E88g_h4n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 4, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # E88 500M scaling variants
        # Best config (h8n32) needs very wide dims for 500M, so also test h8n64
        'E88_h8n32_500m': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h8n64_500m': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Properly scaled E88 (state/dim ratio ≈ 3.76 like 75M config)
        'E88_h8n48_scaled': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h8n56_scaled': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 56, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # E88 round 8: Head mixing ablation (based on best h8n32 config)
        # Testing different ways to combine head outputs
        'E88h_concat': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'head_mix': 'concat'}),
        'E88h_wsum': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'head_mix': 'weighted_sum'}),
        'E88h_perhead': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'head_mix': 'per_head'}),
        'E88h_inputw': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'head_mix': 'input_weighted'}),
        'E88h_sum': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 8, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False, 'head_mix': 'sum'}),
        # Also test h32n32 (32 heads × 32 state)
        'E88h_h32n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # State-matched configs for fair comparison with Mamba2/FLA-GDN
        # Using 32×32 matrices (well-optimized) with many heads

        # Mamba2 state: ~163,840/layer - use h160n32 = 160 × 32² = 163,840
        'E88_h160n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 160, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Alternative: h128n32 = 128 × 32² = 131,072 (slightly less state)
        'E88_h128n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 128, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # FLA-GDN state: ~524,288/layer - use h512n32 = 512 × 32² = 524,288
        'E88_h512n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 512, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Alternative: h256n32 = 256 × 32² = 262,144 (half state)
        'E88_h256n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 256, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # Keep old configs for reference
        'E88_h32n72': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 72, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h16n96': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 16, 'n_state': 96, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h24n80': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 24, 'n_state': 80, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h32n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h64n96': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 96, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # Many-head configs with smaller n_state (faster)
        # Mamba2 state: ~163,840/layer - h40n64 = 40 × 64² = 163,840 (exact match)
        'E88_h40n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 40, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h64n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h72n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 72, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_h96n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 96, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        # Smaller state configs (1/2 and 1/4 of Mamba2's 163,840)
        'E88_h80n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 80, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 81,920 state (1/2 Mamba2)
        'E88_h40n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 40, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 40,960 state (1/4 Mamba2)
        'E88_h20n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 20, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 20,480 state (1/8 Mamba2)
        # Additional configs for head/state scaling study
        'E88_h128n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 128, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 131,072 state
        'E88_h64n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 147,456 state
        'E88_h32n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 131,072 state
        'E88_h24n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 24, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 98,304 state
        # Balanced configs for ~500M params (dim ≈ d_inner)
        'E88_h48n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 48, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 49,152 state
        'E88_h64n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 65,536 state
        'E88_h68n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 68, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 69,632 state
        'E88_h72n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 72, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 73,728 state
        'E88_h76n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 76, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 77,824 state
        'E88_h48n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 48, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 110,592 state
        'E88_h48n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 48, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 196,608 state

        # E88 with write gate (FLA-GDN beta style) - gates delta before writing to memory
        'E88-wgate': E88FLAHybrid,  # Full config via kwargs: --use_gate 1 --use_write_gate 1

        # State-matched 500M configs (depth=32 like mamba2)
        # FLA-GDN at 500M has ~1.33M state/layer, Mamba2 has ~410K state/layer
        'E88_h5n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 5, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 81,920 state (1/16x FLA)
        'E88_h10n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 10, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 163,840 state (1/8x FLA)
        'E88_h20n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 20, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 327,680 state (1/4x FLA)
        'E88_h40n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 40, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 655,360 state (1/2x FLA)
        'E88_h81n128': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 81, 'n_state': 128, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 1,327,104 state (1x FLA)
        'E88_h64n80': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 80, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 409,600 state (~Mamba2)
        'E88_h81n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 81, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 82,944 state (depth=32 baseline)
        'E88_h162n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 162, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 165,888 state
        'E88_h100n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 100, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 409,600 state (~Mamba2)

        # State-matched with n_state=48 (more numerically stable than n_state=64+)
        'E88_h36n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 36, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 82,944 state (1/16x FLA)
        'E88_h72n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 72, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 165,888 state (1/8x FLA)
        'E88_h144n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 144, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 331,776 state (1/4x FLA)
        'E88_h178n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 178, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 410,112 state (~Mamba2)

        # Balanced configs: n_heads × n_state ≈ dim (ratio ~1.0 for efficient projection)
        # These avoid the projection bottleneck that causes slowdowns with high head counts
        'E88_b56n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 57K state, ratio=0.82
        'E88_b60n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 60, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 61K state, ratio=0.94
        'E88_b64n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 65K state, ratio=1.07
        'E88_b40n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 40, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 92K state, ratio=0.94
        'E88_b44n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 44, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 101K state, ratio=1.18
        'E88_b28n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 28, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 115K state, ratio=0.82
        'E88_b32n64': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 64, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # 131K state, ratio=1.07

        # Scaling study configs (all n_state=32, ratio=1.0)
        # Dimension sweep at ~500M params
        'E88_dim1536': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 48, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # narrow+deep
        'E88_dim1792': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_dim2048': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # medium
        'E88_dim2304': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 72, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_dim2560': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 80, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # wide
        'E88_dim2816': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 88, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_dim3072': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 96, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # wider

        # n_state sweep at 500M (dim=2048, depth=32)
        'E88_n16': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 128, 'n_state': 16, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # smallest state
        'E88_n24': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 85, 'n_state': 24, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_n32': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),  # current best
        'E88_n40': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 51, 'n_state': 40, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_n48': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 42, 'n_state': 48, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # Multi-scale configs (balanced, n_state=32)
        'E88_100M': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 32, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_200M': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 40, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_300M': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 48, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_500M': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 64, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_750M': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 72, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),
        'E88_1B': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 80, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # Gating ablation: compare no gating vs sigmoid vs SiLU
        # Using best config: dim=1792, depth=38, n_heads=56, n_state=32 (1.44 loss ungated)
        'E88_gated_sigmoid': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': True, 'gate_activation': 'sigmoid', 'use_output_norm': False}),
        'E88_gated_silu': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': True, 'gate_activation': 'silu', 'use_output_norm': False}),
        # With convolutions (FLA-GDN style)
        'E88_conv_silu': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': True, 'use_gate': True, 'gate_activation': 'silu', 'use_output_norm': True}),
        'E88_full_fla': lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': True, 'use_gate': True, 'gate_activation': 'silu', 'use_output_norm': True, 'use_silu': True, 'use_l2_norm': True}),

        # E89: Residual State - tanh only on outer product (better gradient flow)
        # S = decay * S + tanh(outer(delta, k)) instead of tanh(decay * S + outer)
        89: E89ResidualStateCell,
        'E89': E89ResidualStateCell,
        # E89 with optimal E88 config (h56_n32, expansion=1.0)
        'E89_opt': lambda **kw: E89ResidualStateCell(**{**kw, 'n_heads': 56, 'n_state': 32, 'expansion': 1.0, 'use_conv': False, 'use_gate': False, 'use_output_norm': False}),

        # MoM E88: Mixture of Memory - sparse routing to memory heads
        # Routes each token to top-K heads instead of all heads
        # Allows 2-3x more heads with same compute budget
        'MoME88': MoME88,
        'MoM': MoME88,  # Short alias
        # Default: 196 heads (2x E88 optimal), top_k=32
        'MoM_h196k32': lambda **kw: MoME88(**{**kw, 'n_heads': 196, 'top_k': 32}),
        # More heads, same compute
        'MoM_h256k32': lambda **kw: MoME88(**{**kw, 'n_heads': 256, 'top_k': 32}),
        'MoM_h312k32': lambda **kw: MoME88(**{**kw, 'n_heads': 312, 'top_k': 32}),  # 3x E88
        # Sparser routing
        'MoM_h256k16': lambda **kw: MoME88(**{**kw, 'n_heads': 256, 'top_k': 16}),
        'MoM_h312k16': lambda **kw: MoME88(**{**kw, 'n_heads': 312, 'top_k': 16}),
        # Denser routing (more heads active per token)
        'MoM_h196k64': lambda **kw: MoME88(**{**kw, 'n_heads': 196, 'top_k': 64}),
        'MoM_h256k64': lambda **kw: MoME88(**{**kw, 'n_heads': 256, 'top_k': 64}),
        # Match E88 optimal n_state=32
        'MoM_h196k32n32': lambda **kw: MoME88(**{**kw, 'n_heads': 196, 'top_k': 32, 'n_state': 32}),
        'MoM_h256k32n32': lambda **kw: MoME88(**{**kw, 'n_heads': 256, 'top_k': 32, 'n_state': 32}),
        # Smaller n_state for speed
        'MoM_h256k32n16': lambda **kw: MoME88(**{**kw, 'n_heads': 256, 'top_k': 32, 'n_state': 16}),
        'MoM_h312k32n16': lambda **kw: MoME88(**{**kw, 'n_heads': 312, 'top_k': 32, 'n_state': 16}),

        # E90: Dual-Rate Factorized State (fast + slow memory)
        # Fast state: small k_fast×k_fast, updated every step
        # Slow state: larger k_slow×k_slow, updated via learned soft gate
        'E90': E90DualRate,
        'E90_f16s32': lambda **kw: E90DualRate(**{**kw, 'k_fast': 16, 'k_slow': 32}),
        'E90_f16s48': lambda **kw: E90DualRate(**{**kw, 'k_fast': 16, 'k_slow': 48}),
        'E90_f16s64': lambda **kw: E90DualRate(**{**kw, 'k_fast': 16, 'k_slow': 64}),
        'E90_f32s64': lambda **kw: E90DualRate(**{**kw, 'k_fast': 32, 'k_slow': 64}),
        'E90_f32s96': lambda **kw: E90DualRate(**{**kw, 'k_fast': 32, 'k_slow': 96}),
        # High state capacity variants
        'E90_f16s96': lambda **kw: E90DualRate(**{**kw, 'k_fast': 16, 'k_slow': 96}),
        'E90_f16s128': lambda **kw: E90DualRate(**{**kw, 'k_fast': 16, 'k_slow': 128}),
        # More heads variants
        'E90_h32': lambda **kw: E90DualRate(**{**kw, 'n_heads': 32}),
        'E90_h64': lambda **kw: E90DualRate(**{**kw, 'n_heads': 64}),
        'E90_h96': lambda **kw: E90DualRate(**{**kw, 'n_heads': 96}),

        '21s': lambda **kw: StructuredElman(mimo_rank=4, **kw),  # E21-S: smaller rank
        '21t': lambda **kw: StructuredElman(nonlinearity='tanh', **kw),  # E21-T: tanh
        '21l': lambda **kw: StructuredElman(nonlinearity='linear', **kw),  # E21-L: linear (ablation)
        22: StructuredElmanAttention,  # E22: E21 + state attention (UTM class)
        '22n': lambda **kw: StructuredElmanAttention(attn_type='over_N', **kw),  # E22-N: attention over N
        '22h': lambda **kw: StructuredElmanAttention(attn_type='over_heads', **kw),  # E22-H: attention over heads
        '22k4': lambda **kw: StructuredElmanAttention(attn_period=4, **kw),  # E22-K4: attend every 4 steps
        '22k16': lambda **kw: StructuredElmanAttention(attn_period=16, **kw),  # E22-K16: attend every 16 steps
        23: DualMemoryElman,  # E23: Dual-memory (tape + working memory)
        '23n32': lambda **kw: DualMemoryElman(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '23n128': lambda **kw: DualMemoryElman(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        24: E24Layer,  # E24: True single-GEMM dual memory
        '24n32': lambda **kw: E24Layer(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '24n128': lambda **kw: E24Layer(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        25: E25DualMemoryElman,  # E25: Dual memory with 1.5-entmax attention
        '25n32': lambda **kw: E25DualMemoryElman(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '25n128': lambda **kw: E25DualMemoryElman(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        26: E26DualMemoryElman,  # E26: Parallel dual memory
        '26n32': lambda **kw: E26DualMemoryElman(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '26n128': lambda **kw: E26DualMemoryElman(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        28: E28ConvElman,  # E28: E1 + Mamba2 causal conv
        # E1H: Multi-Head E1 (independent Elman heads, vector state, no matrix/decay/L2)
        'E1H': E1MultiHead,

        'mamba2': 'mamba2',  # Special case - handled separately
        # PROBE 1 additive / non-saturating counter baselines (WGY 2018 positive
        # control: CAN realize an unbounded counter; tanh/linear-state cannot).
        'relu_rnn': ReLURNNLayer,  # additive ReLU-Elman RNN
        'lstm': LSTMLayer,         # standard LSTM (gated additive cell)
    }
    if level in levels:
        return levels[level]

    # Dynamic parsing for E75h*n* patterns (E75 Multi-Head variants)
    # Format: E75h{n_heads}n{n_state} or E75h{n_heads}
    # Examples: E75h7n24, E75h11n48, E75h4 (uses default n_state=32)
    if isinstance(level, str) and level.startswith('E75h'):
        import re
        # Match E75h{heads}n{state} or E75h{heads}
        match = re.match(r'E75h(\d+)(?:n(\d+))?', level)
        if match:
            n_heads = int(match.group(1))
            n_state = int(match.group(2)) if match.group(2) else 32  # Default to 32

            # Validate n_state is supported (must be in CUDA kernel instantiations)
            SUPPORTED_N_STATE = {8, 16, 24, 32, 40, 48, 56, 64}
            if n_state not in SUPPORTED_N_STATE:
                raise ValueError(
                    f"E75 n_state={n_state} not supported. "
                    f"Supported values: {sorted(SUPPORTED_N_STATE)}. "
                    f"For larger state sizes, use E88 instead."
                )

            return lambda **kw: E75MultiHead(**{**kw, 'n_heads': n_heads, 'n_state': n_state})

    # Dynamic parsing for E88h*n* patterns (E88 FLA Hybrid variants)
    # Format: E88h{n_heads}n{n_state} or E88h{n_heads}
    if isinstance(level, str) and level.startswith('E88h'):
        import re
        match = re.match(r'E88h(\d+)(?:n(\d+))?', level)
        if match:
            n_heads = int(match.group(1))
            n_state = int(match.group(2)) if match.group(2) else 32
            return lambda **kw: E88FLAHybrid(**{**kw, 'n_heads': n_heads, 'n_state': n_state})

    raise ValueError(f"Invalid level {level}. Available: 0-6, 8-17, 18a/b/e, 19a/b/d/e, 20-26, 28, 30-68, gdn, gdn-vec, fla-gdn, gdn2, gdn2-mlp, llama, mamba2, E75h*n*, E88h*n*")


# === SwiGLU MLP block (task e97-raw-plus) ===========================================
# Bias-free LLaMA-style SwiGLU FFN identical to the one used by the official GDN-2
# block (see ndm/models/llama_baseline.py:LlamaFFN and the gdn2-mlp reference in the
# emender research repo). Adding this post-mixer MLP turned mixer-only GDN-2 into the
# rank-2 gdn2-mlp on the 1.3B CMA leaderboard; this lets us test the same upgrade on
# the rank-1 mixer-only e97-raw cell.
OFFICIAL_GDN2_MLP_RATIO = 6208 / 2304  # ~2.694: official GDN-2 SwiGLU hidden ratio


def round_mlp_hidden(dim, mlp_ratio, multiple=64):
    """SwiGLU hidden width = dim * mlp_ratio, rounded to a multiple (default 64)."""
    return max(multiple, int(round(dim * mlp_ratio / multiple) * multiple))


class SwiGLUMLP(nn.Module):
    """Bias-free SwiGLU MLP: w3(silu(w1(x)) * w2(x)).

    extra_in (M1 state-aware MLP, STATE_AWARE_MLP_DESIGN.md §5): when >0, the gate
    and up projections (w1, w2) widen their INPUT from `dim` to `dim + extra_in`.
    The caller is responsible for passing an input already concatenated to width
    `dim + extra_in` (residual-stream norm output concatenated with the state-aware
    readout summary). w3 (down) is unchanged so the residual add stays at `dim`.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0, extra_in=0):
        super().__init__()
        self.dim = dim
        self.extra_in = int(extra_in)
        self.w1 = nn.Linear(dim + self.extra_in, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(dim + self.extra_in, hidden_dim, bias=False)  # up
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MixerMLPWrapper(nn.Module):
    """Wrap any LadderLM mixer layer with a post-mixer RMSNorm + SwiGLU MLP.

    LadderLM owns the outer Mamba-style residual stream and supplies the pre-mixer
    RMSNorm; this wrapper adds the second (post-mixer) RMSNorm and the SwiGLU MLP so
    each block becomes mixer + MLP, exactly mirroring the gdn2-mlp reference layer
    (emender ndm/models/external_gdn2.py:GDN2ExternalMLPLayer). The single value
    returned is added to LadderLM's residual stream, so:

        out = mixer(x) + mlp(norm2(x + mixer(x)))

    The inner mixer's hidden-state list is threaded through unchanged so TBPTT and
    FLA caches keep working.
    """

    def __init__(self, mixer, dim, mlp_ratio, mlp_multiple=64, dropout=0.0,
                 state_summary_dim=0, mlp_hidden_dim=None):
        super().__init__()
        self.mixer = mixer
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # mlp_hidden_dim override lets the iso-param solver pin an exact hidden
        # (state-aware-MLP A/B); otherwise derive it from mlp_ratio as before.
        self.mlp_hidden_dim = (int(mlp_hidden_dim) if mlp_hidden_dim
                               else round_mlp_hidden(dim, mlp_ratio, mlp_multiple))
        # M1 state-aware MLP (STATE_AWARE_MLP_DESIGN.md §5): when >0, the mixer
        # returns a 3rd value (the readout summary) that we concat to the MLP input.
        self.state_summary_dim = int(state_summary_dim)
        self.norm_2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, self.mlp_hidden_dim, dropout=dropout,
                             extra_in=self.state_summary_dim)

    def set_layer_idx(self, idx):
        if hasattr(self.mixer, 'set_layer_idx'):
            self.mixer.set_layer_idx(idx)

    def forward(self, x, hidden=None, **kwargs):
        out = self.mixer(x, hidden, **kwargs)
        if self.state_summary_dim > 0:
            mix_out, h_final, summary = out
            normed = self.norm_2(x + mix_out)
            mlp_in = torch.cat([normed, summary.to(normed.dtype)], dim=-1)
        else:
            mix_out, h_final = out
            mlp_in = self.norm_2(x + mix_out)
        mlp_out = self.mlp(mlp_in)
        return mix_out + mlp_out, h_final

    def extra_repr(self):
        return (f"dim={self.dim}, mlp_ratio={self.mlp_ratio:.4f}, "
                f"mlp_hidden_dim={self.mlp_hidden_dim}, "
                f"state_summary_dim={self.state_summary_dim}")


class LadderLM(nn.Module):
    """
    Language Model using Elman Ablation Ladder levels.

    Uses Mamba-style architecture with pre-norm + residual connections.

    Args:
        vocab_size: Size of vocabulary (256 for byte-level)
        dim: Model dimension
        depth: Number of layers
        level: Ablation ladder level (0-3)
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax (levels 2+)
        delta_init: Initial delta gate bias
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        level=0,
        expansion=1.0,
        n_groups=32,
        n_slots=8,
        n_banks=4,  # For E10 multi-scale: number of EMA memory banks
        n_state=64,  # For E70-E73: matrix state size (S is n_state x n_state)
        n_heads=None,  # For E88 FLA Hybrid: number of heads
        use_gate=True,  # For E88 FLA Hybrid: output gating (False for "best" config)
        gate_activation='sigmoid',  # For E88 FLA Hybrid: gate activation ('sigmoid' or 'silu')
        linear_state=False,  # For E88 FLA Hybrid: linear state update (no tanh)
        use_write_gate=False,  # For E88 FLA Hybrid: write gate (beta) for delta
        e88_decay_mode='mamba',  # For E88 FLA Hybrid: mamba, simple, none, or constant
        e88_value_residual=False,  # For E88 FLA Hybrid: add D*v residual before output gate
        e88_raw_write=False,  # For E88 FLA Hybrid: ablate delta correction
        rank=None,
        delta_init=-2.0,
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=0.1,
        core_ratio=0.125,  # For E9 hybrid: fraction of d_inner for dense core
        mamba2_init=False,  # Use Mamba2-style initialization
        state_expansion=2,  # For E16: d_state = d_inner * state_expansion
        use_conv=False,  # Conv1d hurts E-series (nonlinear RNN doesn't need it)
        d_conv=4,  # Conv kernel size (if enabled)
        gdn2_mlp_ratio=OFFICIAL_GDN2_MLP_RATIO,  # Official GDN-2 6208/2304 SwiGLU ratio
        top_k=None,  # For MoM E88: number of active memory slots (top-K routing)
        k_fast=None,  # For E90 Dual-Rate: fast state dimension
        k_slow=None,  # For E90 Dual-Rate: slow state dimension
        checkpoint_interval=16,  # For E88: steps between state checkpoints (larger = less memory)
        gradient_checkpointing=False,  # Recompute layer forward during backward (saves ~16GB at 25 layers)
        projection_chunk_size=0,  # For E88: chunk size for projection recomputation (0=disabled, saves ~5GB/layer at T=32K)
        loss_chunk_size=0,  # Chunk T dimension when computing lm_head + cross_entropy (0=disabled, saves T*V*2 bytes at long T)
        use_triton=False,  # For E88: use Triton fwd+bwd kernels instead of CUDA register-owned (portable across NVIDIA/AMD ROCm)
        use_chunked_e97=False,  # For E97 linear-state Frontier smoke: route through chunked split-edit kernel
        e97_chunk_size=32,  # Chunk size for --use_chunked_e97
        layer_kwargs=None,  # Extra per-layer kwargs merged into LayerClass(...) — e.g. head_type_logits / lam_max / beta_max / corner_mixture for typed-gdn2-lm / e98-cma-lm. None == no change to existing levels.
        mlp_ratio=0.0,  # >0 wraps every mixer layer with a post-mixer RMSNorm + SwiGLU MLP (task e97-raw-plus). 0 = mixer-only (unchanged behavior).
        mlp_multiple=64,  # Round SwiGLU hidden width to this multiple.
        state_summary_dim=0,  # M1 state-aware MLP (STATE_AWARE_MLP_DESIGN.md §5): >0 makes each E97 mixer down-project its
                              # pre-o_proj readout to this dim + RMSNorm and the MixerMLPWrapper concat it to the SwiGLU input.
        mlp_hidden=None,      # Optional exact SwiGLU hidden override (bypasses mlp_ratio rounding) for exact iso-param A/B arms.
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.level = level
        self.n_slots = n_slots
        self.n_banks = n_banks
        self.n_state = n_state
        self.n_heads = n_heads
        self.use_gate = use_gate
        self.gate_activation = gate_activation
        self.linear_state = linear_state
        self.use_write_gate = use_write_gate
        self.e88_decay_mode = e88_decay_mode
        self.e88_value_residual = e88_value_residual
        self.e88_raw_write = e88_raw_write
        self.rank = rank
        self.r_h_mode = r_h_mode
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.gradient_checkpointing = gradient_checkpointing
        self.loss_chunk_size = loss_chunk_size
        self.use_chunked_e97 = use_chunked_e97
        self.e97_chunk_size = e97_chunk_size
        self.mlp_ratio = mlp_ratio
        self.mlp_multiple = mlp_multiple

        # Get the layer class for this level
        LayerClass = get_ladder_level(level)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Use fused RMSNorm from mamba_ssm (matches Mamba architecture exactly)
        self.fused_add_norm = FUSED_NORM_AVAILABLE
        self.residual_in_fp32 = True  # Keep residual in fp32 for stability

        # Pre-normalization layers (one per recurrent layer) - RMSNorm like Mamba
        self.layer_norms = nn.ModuleList([
            RMSNorm(dim) for _ in range(depth)
        ])

        # Extra per-layer kwargs (typed-gdn2-lm / e98-cma-lm candidate knobs).
        # Merged LAST so they override the generic defaults below.
        extra_layer_kwargs = dict(layer_kwargs) if layer_kwargs else {}
        self.layer_kwargs = extra_layer_kwargs

        # Stack of recurrent layers
        self.layers = nn.ModuleList([
            LayerClass(**{**dict(
                dim=dim,
                expansion=expansion,
                n_groups=n_groups,
                n_slots=n_slots,
                n_banks=n_banks,  # For E10 multi-scale
                n_state=n_state,  # For E70-E73 matrix state
                n_heads=n_heads,  # For E88 FLA Hybrid
                use_gate=use_gate,  # For E88 FLA Hybrid: output gating
                gate_activation=gate_activation,  # For E88 FLA Hybrid: gate activation
                linear_state=linear_state,  # For E88 FLA Hybrid: linear state
                use_write_gate=use_write_gate,  # For E88 FLA Hybrid: write gate
                decay_mode=e88_decay_mode,  # For E88 FLA Hybrid: decay mode
                use_value_residual=e88_value_residual,  # For E88 FLA Hybrid: D*v residual
                raw_write=e88_raw_write,  # For E88 FLA Hybrid: raw-write ablation
                rank=rank,
                delta_init=delta_init,
                dropout=dropout,
                r_h_mode=r_h_mode,
                r_h_init_gain=r_h_init_gain,
                core_ratio=core_ratio,  # For E9 hybrid
                mamba2_init=mamba2_init,  # Mamba2-style initialization
                state_expansion=state_expansion,  # For E16
                use_conv=use_conv,  # Conv1d before recurrence (like Mamba2)
                d_conv=d_conv,  # Conv kernel size
                gdn2_mlp_ratio=gdn2_mlp_ratio,  # For gdn2-mlp
                top_k=top_k,  # For MoM E88: number of active memory slots
                k_fast=k_fast,  # For E90 Dual-Rate: fast state dimension
                k_slow=k_slow,  # For E90 Dual-Rate: slow state dimension
                checkpoint_interval=checkpoint_interval,  # For E88: state checkpoint interval
                projection_chunk_size=projection_chunk_size,  # For E88: projection recomputation chunks
                use_triton=use_triton,  # For E88: route fwd+bwd through Triton kernels
                use_chunked_e97=use_chunked_e97,  # For E97 linear-state chunked split-edit kernel
                e97_chunk_size=e97_chunk_size,  # For E97 linear-state chunked split-edit kernel
                readout_summary_dim=state_summary_dim,  # M1 state-aware MLP: E88FLAHybrid emits a 3rd readout-summary value when >0
            ), **extra_layer_kwargs})
            for _ in range(depth)
        ])

        # Optional post-mixer SwiGLU MLP block per layer (task e97-raw-plus).
        # Wrap AFTER construction so the wrapper is mixer-agnostic and applies
        # identically to E97/E88/gdn2/etc. mlp_ratio=0 leaves layers untouched.
        if mlp_ratio and mlp_ratio > 0:
            self.layers = nn.ModuleList([
                MixerMLPWrapper(layer, dim, mlp_ratio, mlp_multiple, dropout=dropout,
                                state_summary_dim=state_summary_dim, mlp_hidden_dim=mlp_hidden)
                for layer in self.layers
            ])

        # Give each layer a unique index if it supports set_layer_idx (needed
        # for FLA's Cache to populate/retrieve per-layer state correctly).
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_layer_idx'):
                layer.set_layer_idx(i)

        # Final layer norm before output - RMSNorm like Mamba
        self.norm = RMSNorm(dim)

        # Output projection to vocabulary (tied with embedding)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        x,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None,
        prev_conv_buffers=None,
        actual_length=None,
        doc_boundaries=None,
    ):
        """
        Forward pass compatible with train.py interface.

        Args:
            x: [B, T] input token indices
            return_loss: If True, compute loss (x is [B, T+1] with targets)
            return_prev_hiddens: If True, return hidden states for TBPTT
            prev_hiddens: List of [B, d_inner] per layer, or None
            prev_conv_buffers: Unused, for API compatibility
            actual_length: For masking padded chunks
            doc_boundaries: [B, T] boolean tensor for hidden state reset

        Returns:
            If return_loss: (loss, new_hiddens) or loss
            Else: (logits, new_hiddens) or logits
        """
        if return_loss:
            # x is [B, T+1], split into input and target
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        B, T = inp.shape

        # Embed tokens
        x = self.embedding(inp)  # [B, T, dim]

        # Initialize hidden states if not provided
        if prev_hiddens is None:
            prev_hiddens = [None] * self.depth

        new_hidden_states = []
        diagnostic_rank = os.environ.get("EMENDER_DIAG_FINITE_RANK")
        trace_finiteness = (
            diagnostic_rank is not None
            and int(diagnostic_rank) == int(os.environ.get("RANK", "0"))
        )

        # Run through layers with fused add+norm (exactly like Mamba's Block)
        # Pattern: residual = x + residual; x = norm(residual); x = mixer(x)
        residual = None
        for i, (ln, layer) in enumerate(zip(self.layer_norms, self.layers)):
            if self.fused_add_norm:
                # Fused add + RMSNorm (like Mamba)
                x, residual = rms_norm_fn(
                    x,
                    ln.weight,
                    None,  # bias
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=ln.eps,
                )
            else:
                # Non-fused fallback
                residual = (x + residual) if residual is not None else x
                x = ln(residual.to(dtype=ln.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

            # Elman layer forward
            mixer_input = x
            if self.gradient_checkpointing and self.training:
                x, h_final = torch_checkpoint(layer, x, prev_hiddens[i], use_reentrant=False)
            else:
                x, h_final = layer(x, prev_hiddens[i])

            if trace_finiteness and not bool(torch.isfinite(x).all().item()):
                def summary(name, tensor):
                    if tensor is None or not torch.is_tensor(tensor):
                        return f"{name}=none"
                    finite = torch.isfinite(tensor)
                    count = int(finite.sum().item())
                    total = tensor.numel()
                    max_abs = (
                        float(tensor[finite].float().abs().max().item())
                        if count else float("nan")
                    )
                    return f"{name}_finite={count}/{total} {name}_max_abs={max_abs:.6g}"

                hidden_tensors = (
                    [h_final] if torch.is_tensor(h_final)
                    else list(h_final or []) if isinstance(h_final, (list, tuple))
                    else []
                )
                hidden_summary = " ".join(
                    summary(f"hidden_{j}", value)
                    for j, value in enumerate(hidden_tensors)
                )
                print(
                    f"[finite-trace] rank={diagnostic_rank} first_nonfinite_layer={i} "
                    f"layer_type={type(layer).__name__} {summary('input', mixer_input)} "
                    f"{summary('output', x)} {hidden_summary}",
                    flush=True,
                )
                raise RuntimeError(
                    f"finite-trace: first non-finite mixer output at layer {i}"
                )

            new_hidden_states.append(h_final)

        # Final fused add + norm
        if self.fused_add_norm:
            # prenorm=False returns just the normalized output (not a tuple)
            x = rms_norm_fn(
                x,
                self.norm.weight,
                None,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        else:
            x = self.norm((x + residual).to(dtype=self.norm.weight.dtype))
        if return_loss:
            # Mask out padded positions if actual_length is provided
            if actual_length is not None:
                device = x.device
                positions = torch.arange(target.size(1), device=device).unsqueeze(0)
                valid_mask = positions < (actual_length.unsqueeze(1) - 1)
                target = target.clone()
                target[~valid_mask] = -100

            # Chunked CE: when T is large, materializing logits=(B,T,V) costs T*V*2 bytes.
            # At T=128K, V=50281, bf16 → 13GB just for logits. Stream through time in chunks.
            T_total = x.size(1)
            loss_chunk = getattr(self, 'loss_chunk_size', 0)
            if loss_chunk > 0 and T_total > loss_chunk:
                total_sum = x.new_zeros(())
                total_count = 0
                for t0 in range(0, T_total, loss_chunk):
                    t1 = min(t0 + loss_chunk, T_total)
                    logits_c = self.lm_head(x[:, t0:t1])
                    target_c = target[:, t0:t1]
                    chunk_loss_sum = F.cross_entropy(
                        logits_c.reshape(-1, self.vocab_size),
                        target_c.reshape(-1),
                        ignore_index=-100,
                        reduction='sum',
                    )
                    total_sum = total_sum + chunk_loss_sum
                    total_count = total_count + (target_c != -100).sum()
                loss = total_sum / total_count.clamp(min=1)
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    target.reshape(-1),
                    ignore_index=-100,
                )
            if return_prev_hiddens:
                return loss, (new_hidden_states, None)
            return loss

        logits = self.lm_head(x)
        if return_prev_hiddens:
            return logits, (new_hidden_states, None)
        return logits

    def get_num_params(self):
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        level_names = {
            0: "Stock Elman (e0)",
            1: "Mamba-Gated Elman (e1)",
            2: "Slot Elman (e2)",
            3: "Low-Rank Slot Elman (e3)",
        }
        return f'level={self.level} ({level_names.get(self.level, "Unknown")}), dim={self.dim}, depth={self.depth}'


def create_ladder_model(
    target_params: str = "100m",
    level: int = 0,
    vocab_size: int = 256,
    expansion: float = 1.0,
    n_groups: int = 32,
    n_slots: int = 8,
    r_h_mode: str = 'spectral_norm',
    r_h_init_gain: float = 0.1,
    state_expansion: int = 2,
    mamba2_init: bool = False,
    use_conv: bool = False,  # Conv1d hurts E-series (nonlinear RNN doesn't need it)
    d_conv: int = 4,  # Conv kernel size (if enabled)
):
    """
    Create a LadderLM with approximately target_params parameters.

    Uses dynamic parameter counting: creates 1-layer and 2-layer models to
    compute exact params_per_layer, then determines depth to reach target.

    Args:
        target_params: Target parameter count (e.g., "100m", "500m", "1b")
        level: Ablation ladder level (0-3) or 'mamba2'
        vocab_size: Vocabulary size
        expansion: Hidden state expansion
        n_groups: Number of groups for compete softmax
        n_slots: Number of slots for E2/E3 (default: 8)
        r_h_mode: Constraint mode for R_h matrix (for log-space levels)
        r_h_init_gain: Initial gain for R_h orthogonal initialization
        state_expansion: For E16: d_state = d_inner * state_expansion

    Returns:
        LadderLM or Mamba2LM model
    """
    # Handle mamba2 specially
    if level == 'mamba2':
        from .mamba2_baseline import create_mamba2_model
        return create_mamba2_model(target_params=target_params, vocab_size=vocab_size)

    # Parse target
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Dimension configs based on target size (expansion can vary per level)
    # Format: target_params -> (dim, default_expansion)
    dim_configs = {
        50_000_000: (512, 1.5),
        100_000_000: (768, 1.5),
        200_000_000: (1024, 1.5),
        350_000_000: (1024, 2.0),
        500_000_000: (1024, 2.5),
        700_000_000: (1280, 2.0),
        1_000_000_000: (1536, 2.0),
        1_300_000_000: (1792, 2.0),
    }

    # Find closest dim config
    closest = min(dim_configs.keys(), key=lambda x: abs(x - target_count))
    dim, default_expansion = dim_configs[closest]

    # Use provided expansion or default from config
    if expansion == 1.0:
        expansion = default_expansion

    # Create a 1-layer model to count base params (embeddings, output, etc)
    model_1layer = LadderLM(
        vocab_size=vocab_size, dim=dim, depth=1, level=level,
        expansion=expansion, n_groups=n_groups, n_slots=n_slots,
        r_h_mode=r_h_mode, r_h_init_gain=r_h_init_gain,
        state_expansion=state_expansion, mamba2_init=mamba2_init,
        use_conv=use_conv, d_conv=d_conv,
    )
    params_1layer = model_1layer.get_num_params()

    # Create a 2-layer model to compute params per layer
    model_2layer = LadderLM(
        vocab_size=vocab_size, dim=dim, depth=2, level=level,
        expansion=expansion, n_groups=n_groups, n_slots=n_slots,
        r_h_mode=r_h_mode, r_h_init_gain=r_h_init_gain,
        state_expansion=state_expansion, mamba2_init=mamba2_init,
        use_conv=use_conv, d_conv=d_conv,
    )
    params_2layer = model_2layer.get_num_params()

    # Compute params per layer
    params_per_layer = params_2layer - params_1layer
    base_params = params_1layer - params_per_layer  # embedding + output

    # Clean up probe models
    del model_1layer, model_2layer

    # Calculate depth needed to reach target
    if params_per_layer > 0:
        depth = max(1, round((target_count - base_params) / params_per_layer))
    else:
        depth = 12  # fallback

    # Ensure reasonable depth bounds
    depth = max(4, min(depth, 48))

    # Create the actual model
    model = LadderLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        level=level,
        expansion=expansion,
        n_groups=n_groups,
        n_slots=n_slots,
        r_h_mode=r_h_mode,
        r_h_init_gain=r_h_init_gain,
        state_expansion=state_expansion,
        mamba2_init=mamba2_init,
        use_conv=use_conv,
        d_conv=d_conv,
    )

    actual_params = model.get_num_params()
    r_h_info = f", r_h_mode={r_h_mode}" if str(level).startswith('log') else ""
    print(f"Created Level {level} model: dim={dim}, depth={depth}, params={actual_params:,}{r_h_info}")

    return model


if __name__ == "__main__":
    print("Testing LadderLM...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for level in range(4):
        print(f"\nLevel {level}:")
        model = LadderLM(
            vocab_size=256,
            dim=256,
            depth=4,
            level=level,
            expansion=1.0,
        ).to(device).bfloat16()

        x = torch.randint(0, 256, (2, 32), device=device)
        logits, hidden = model(x, return_prev_hiddens=True)
        loss = F.cross_entropy(logits.view(-1, 256), x.view(-1))
        loss.backward()

        print(f"  Params: {model.get_num_params():,}")
        print(f"  Logits: {logits.shape}, Loss: {loss.item():.4f}")

    print("\nLadderLM tests passed!")
