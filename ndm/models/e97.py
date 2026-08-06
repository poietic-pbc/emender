"""Public E97/Emender split-edit model identity.

E97 was originally implemented as a configuration of :class:`E88FLAHybrid`.
This wrapper gives model factories, checkpoint loaders, stack traces, and user
code an honest E97 type while preserving the exact module tree and state-dict
keys used by existing checkpoints.
"""

from __future__ import annotations

from .e88_fla_hybrid import E88FLAHybrid


class E97SplitEditLayer(E88FLAHybrid):
    """E97 nonlinear split-edit delta-memory layer.

    The implementation core remains shared with E88.  E97 is selected by the
    mandatory ``use_split_edit=True`` specialization, which adds independent
    key-axis erase/read and value-axis write gates.
    """

    architecture_name = "emender/nonlin"
    historical_level = "E97"
    implementation_core = "e88-shared"

    def __init__(self, *args, **kwargs):
        if kwargs.get("use_split_edit") is False:
            raise ValueError("E97SplitEditLayer requires use_split_edit=True")
        kwargs["use_split_edit"] = True
        super().__init__(*args, **kwargs)


__all__ = ["E97SplitEditLayer"]
