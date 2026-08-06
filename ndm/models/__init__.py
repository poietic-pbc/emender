"""
Elman Models - E-Series for Mamba2 comparison

E-Series:
    e0: Stock Elman - tanh recurrence + h*silu(W_gate@x) gating
    e1: Mamba-Gated Elman - Mamba2-style split projection gating
    e2: Slot Elman - Multi-slot memory (64x more capacity like Mamba2)

All support:
    - Optional conv1d for local context
    - Mamba2-style silu gating

Archived experimental levels are in elman/models/archive/.

Usage:
    from ndm.models import StockElman, MambaGatedElman, SlotElman
    from ndm.models import LadderLM, create_ladder_model

    # Create an e0 model
    model = create_ladder_model("500m", level=0)

    # Create an e1 model
    model = create_ladder_model("500m", level=1)

    # Create an e2 model (64 memory slots)
    model = create_ladder_model("500m", level=2)
"""

# E0: Stock Elman (base of e-series)
from .stock_elman import StockElman, StockElmanCell, LEVEL_0_AVAILABLE

# E1: Mamba-Gated Elman (Mamba2-style split projection)
from .mamba_gated_elman import MambaGatedElman, MambaGatedElmanCell
LEVEL_1_AVAILABLE = True

# E15: Softsign Elman (E1 with softsign instead of tanh)
from .softsign_elman import SoftsignElman, SoftsignElmanCell
LEVEL_15_AVAILABLE = True

# E16: Diagonal State-Expanded Elman (Mamba2 efficiency + E1 expressivity)
from .diagonal_state_elman import DiagonalStateElman, DiagonalStateElmanCell
LEVEL_16_AVAILABLE = True

# E2: Slot Elman (Multi-slot memory like Mamba2)
from .slot_elman import SlotElman, SlotElmanCell
LEVEL_2_AVAILABLE = True

# E3: Low-Rank Slot Elman (independent low-rank W_h per slot)
from .lowrank_slot_elman import LowRankSlotElman, LowRankSlotElmanCell
LEVEL_3_AVAILABLE = True

# E4: Low-Rank Elman (SVD-style for fat hidden state)
from .lowrank_elman import LowRankElman, LowRankElmanCell
LEVEL_4_AVAILABLE = True

# E5: Pure Low-Rank Elman (no projections, all low-rank on full dim)
from .pure_lowrank_elman import PureLowRankElman, PureLowRankElmanCell
LEVEL_5_AVAILABLE = True

# E6: Diagonal Elman (per-channel scalar recurrence + low-rank mix)
from .diagonal_elman import DiagonalElman, DiagonalElmanCell
LEVEL_6_AVAILABLE = True

# E9: Hybrid Elman (dense core + diagonal memory)
from .hybrid_elman import HybridElman, HybridElmanCell
LEVEL_9_AVAILABLE = True

# E10: Multi-Scale EMA Elman (E1 core + multiple EMA memory banks)
from .multiscale_elman import MultiScaleElman, MultiScaleElmanCell
LEVEL_10_AVAILABLE = True

# E11: Selective Memory Elman (Mamba-inspired input-dependent memory)
from .selective_elman import SelectiveElman, SelectiveElmanCell
LEVEL_11_AVAILABLE = True

# E14: Matrix State Elman (d*k matrix state with outer product update)
from .matrix_state_elman import MatrixStateElman, MatrixStateElmanCell, MATRIX_STATE_CUDA_AVAILABLE
LEVEL_14_AVAILABLE = True

# E85: Input-As-Matrix (dim = n_state^2, input IS the transformation matrix)
from .e85_input_as_matrix import E85InputAsMatrixLayer, E85InputAsMatrixCell, E85InputAsMatrix, E85_CUDA_AVAILABLE
LEVEL_85_AVAILABLE = True

# Language model wrapper
from .ladder_lm import LadderLM, create_ladder_model
from .e97_moe import (
    E97MoEConfig,
    E97MoEParameterRecipe,
    SharedRoutedMoE,
    calculate_e97_moe_recipe,
    convert_e97_ffns_to_moe,
    e97_moe_auxiliary_loss,
    expert_owner,
    experts_for_rank,
)

# Mamba2 baseline for comparison
try:
    from .mamba2_baseline import Mamba2LM, create_mamba2_model, MAMBA2_AVAILABLE
except ImportError:
    Mamba2LM = None
    create_mamba2_model = None
    MAMBA2_AVAILABLE = False

# minGRU and minLSTM baselines (from "Were RNNs All We Needed?")
from .min_rnn_baseline import (
    minGRU, minLSTM,
    MinGRULM, MinLSTMLM,
    create_mingru_model, create_minlstm_model,
)

# FLA GatedDeltaNet baseline (ICLR 2025 "Gated Delta Networks")
try:
    from .fla_gated_delta import FLAGatedDeltaNetLayer, FLA_GDN_AVAILABLE
except ImportError:
    FLAGatedDeltaNetLayer = None
    FLA_GDN_AVAILABLE = False


def get_available_levels():
    """Return dict of available ladder levels."""
    levels = {
        0: ("Stock Elman (e0)", LEVEL_0_AVAILABLE, StockElman),
        1: ("Mamba-Gated Elman (e1)", LEVEL_1_AVAILABLE, MambaGatedElman),
        2: ("Slot Elman (e2)", LEVEL_2_AVAILABLE, SlotElman),
        3: ("Low-Rank Slot Elman (e3)", LEVEL_3_AVAILABLE, LowRankSlotElman),
        4: ("Low-Rank Elman (e4)", LEVEL_4_AVAILABLE, LowRankElman),
        5: ("Pure Low-Rank Elman (e5)", LEVEL_5_AVAILABLE, PureLowRankElman),
        6: ("Diagonal Elman (e6)", LEVEL_6_AVAILABLE, DiagonalElman),
        9: ("Hybrid Elman (e9)", LEVEL_9_AVAILABLE, HybridElman),
        10: ("Multi-Scale EMA Elman (e10)", LEVEL_10_AVAILABLE, MultiScaleElman),
        11: ("Selective Memory Elman (e11)", LEVEL_11_AVAILABLE, SelectiveElman),
        14: ("Matrix State Elman (e14)", LEVEL_14_AVAILABLE, MatrixStateElman),
    }
    return levels


def get_ladder_level(level):
    """Get the module class for a specific ladder level."""
    levels = get_available_levels()
    if level not in levels:
        raise ValueError(f"Invalid level {level}. Available: {list(levels.keys())}")
    name, available, cls = levels[level]
    if not available:
        raise ImportError(f"Level {level} ({name}) is not available.")
    return cls


__all__ = [
    # E0: Stock Elman
    'StockElman', 'StockElmanCell', 'LEVEL_0_AVAILABLE',
    # E1: Mamba-Gated Elman
    'MambaGatedElman', 'MambaGatedElmanCell', 'LEVEL_1_AVAILABLE',
    # E2: Slot Elman
    'SlotElman', 'SlotElmanCell', 'LEVEL_2_AVAILABLE',
    # E3: Low-Rank Slot Elman
    'LowRankSlotElman', 'LowRankSlotElmanCell', 'LEVEL_3_AVAILABLE',
    # E4: Low-Rank Elman
    'LowRankElman', 'LowRankElmanCell', 'LEVEL_4_AVAILABLE',
    # E5: Pure Low-Rank Elman
    'PureLowRankElman', 'PureLowRankElmanCell', 'LEVEL_5_AVAILABLE',
    # E6: Diagonal Elman
    'DiagonalElman', 'DiagonalElmanCell', 'LEVEL_6_AVAILABLE',
    # E9: Hybrid Elman
    'HybridElman', 'HybridElmanCell', 'LEVEL_9_AVAILABLE',
    # E10: Multi-Scale EMA Elman
    'MultiScaleElman', 'MultiScaleElmanCell', 'LEVEL_10_AVAILABLE',
    # E11: Selective Memory Elman
    'SelectiveElman', 'SelectiveElmanCell', 'LEVEL_11_AVAILABLE',
    # E14: Matrix State Elman
    'MatrixStateElman', 'MatrixStateElmanCell', 'LEVEL_14_AVAILABLE', 'MATRIX_STATE_CUDA_AVAILABLE',
    # E85: Input-As-Matrix
    'E85InputAsMatrixLayer', 'E85InputAsMatrixCell', 'E85InputAsMatrix', 'LEVEL_85_AVAILABLE', 'E85_CUDA_AVAILABLE',
    # Language model wrapper
    'LadderLM', 'create_ladder_model',
    'E97MoEConfig', 'E97MoEParameterRecipe', 'SharedRoutedMoE',
    'calculate_e97_moe_recipe', 'convert_e97_ffns_to_moe',
    'e97_moe_auxiliary_loss', 'expert_owner', 'experts_for_rank',
    # Mamba2 baseline
    'Mamba2LM', 'create_mamba2_model', 'MAMBA2_AVAILABLE',
    # FLA GatedDeltaNet baseline
    'FLAGatedDeltaNetLayer', 'FLA_GDN_AVAILABLE',
    # Helpers
    'get_available_levels', 'get_ladder_level',
]
