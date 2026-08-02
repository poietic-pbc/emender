"""Emender nonlinear delta-memory models.

Historical E88 internals remain for compatibility. E97 has a public model type,
checkpoint loader, and generation API even while its sequential fused kernel
delegates to the shared E88-derived implementation core.
"""

from .models import (
    StockElman, StockElmanCell,
    E97SplitEditLayer, LadderLM, create_ladder_model,
    get_available_levels, get_ladder_level,
)
from .e97 import LoadedE97Checkpoint, generate_e97, load_e97_checkpoint

__all__ = [
    "E97SplitEditLayer",
    "LadderLM",
    "LoadedE97Checkpoint",
    "StockElman",
    "StockElmanCell",
    "create_ladder_model",
    "generate_e97",
    "get_available_levels",
    "get_ladder_level",
    "load_e97_checkpoint",
]

__version__ = "0.2.0"
