"""
Optional external GDN-2 baseline.

This module loads NVIDIA's GatedDeltaNet-2 layer from a local checkout instead
of vendoring the non-commercially licensed source into this repository. Set
``GDN2_PATH`` to override the default checkout location.
"""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GDN2_PATH = "/home/erikg/GatedDeltaNet-2"
BOUND_SOURCE_RECEIPT = ".emender-gdn2-source.json"
OFFICIAL_GDN2_MLP_RATIO = 6208 / 2304
_GDN2_CLASS = None
_GDN2_OPS_MODULE = None


def _external_root() -> Path:
    return Path(os.environ.get("GDN2_PATH", DEFAULT_GDN2_PATH)).expanduser().resolve()


def gdn2_source_tree_sha256(root: Path) -> str:
    """Hash staged source paths, types, link targets, and file bytes."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        if relative == BOUND_SOURCE_RECEIPT or ".git" in relative_path.parts:
            continue
        if path.is_symlink():
            kind, payload = b"L", os.readlink(path).encode()
        elif path.is_file():
            kind, payload = b"F", path.read_bytes()
        elif path.is_dir():
            kind, payload = b"D", b""
        else:
            raise RuntimeError(f"unsupported staged GDN2 source entry: {path}")
        encoded_relative = relative.encode()
        digest.update(kind)
        digest.update(len(encoded_relative).to_bytes(8, "little"))
        digest.update(encoded_relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def verify_bound_gdn2_source(
    root: Path, expected_commit: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless an extracted source tree matches its immutable receipt."""
    root = Path(root).resolve()
    receipt_path = root / BOUND_SOURCE_RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError(f"bound GDN2 source receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "emender-gdn2-source-v1":
        raise RuntimeError("unsupported GDN2 source receipt schema")
    if expected_commit is not None and receipt.get("commit") != expected_commit:
        raise RuntimeError(
            f"staged GDN2 commit {receipt.get('commit')} != required {expected_commit}")
    observed = gdn2_source_tree_sha256(root)
    if observed != receipt.get("source_tree_sha256"):
        raise RuntimeError(
            f"staged GDN2 source tree digest mismatch: {observed} != "
            f"{receipt.get('source_tree_sha256')}")
    return receipt


def _ensure_fla_cache_helpers() -> None:
    """Install cache helper aliases expected by the external GDN-2 checkout."""
    try:
        layer_utils = importlib.import_module("fla.layers.utils")
    except ImportError:
        return

    if not hasattr(layer_utils, "get_layer_cache"):

        def get_layer_cache(layer, past_key_values):
            layer_idx = getattr(layer, "layer_idx", None)
            if (
                past_key_values is None
                or layer_idx is None
                or len(past_key_values) <= layer_idx
            ):
                return None
            return past_key_values[layer_idx]

        layer_utils.get_layer_cache = get_layer_cache

    if not hasattr(layer_utils, "update_layer_cache"):

        def update_layer_cache(layer, past_key_values, **cache_kwargs):
            layer_idx = getattr(layer, "layer_idx", None)
            if past_key_values is None or layer_idx is None:
                return None
            return past_key_values.update(layer_idx=layer_idx, **cache_kwargs)

        layer_utils.update_layer_cache = update_layer_cache


def _ensure_fla_kernel_helpers() -> None:
    """Install import shims for external GDN-2 kernel helpers missing in newer FLA."""
    try:
        ops_module = importlib.import_module("fla.ops")
    except ImportError:
        return

    if "fla.ops.backends" not in sys.modules:
        backends_module = types.ModuleType("fla.ops.backends")

        def dispatch(_backend):
            def decorator(func):
                return func

            return decorator

        backends_module.dispatch = dispatch
        sys.modules["fla.ops.backends"] = backends_module
        setattr(ops_module, "backends", backends_module)

    if "fla.ops.cp" not in sys.modules:
        cp_module = types.ModuleType("fla.ops.cp")
        cp_module.__path__ = []

        class FLACPContext:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "This installed FLA package does not provide context-parallel helpers."
                )

        cp_module.FLACPContext = FLACPContext
        sys.modules["fla.ops.cp"] = cp_module
        setattr(ops_module, "cp", cp_module)

    if "fla.ops.cp.comm" not in sys.modules:
        comm_module = types.ModuleType("fla.ops.cp.comm")

        def all_gather_into_tensor(*args, **kwargs):
            raise ImportError(
                "This installed FLA package does not provide context-parallel helpers."
            )

        comm_module.all_gather_into_tensor = all_gather_into_tensor
        sys.modules["fla.ops.cp.comm"] = comm_module
        setattr(sys.modules["fla.ops.cp"], "comm", comm_module)


def _ensure_fla_compat() -> None:
    _ensure_fla_cache_helpers()
    _ensure_fla_kernel_helpers()


def _load_gdn2_class():
    global _GDN2_CLASS, _GDN2_OPS_MODULE
    if _GDN2_CLASS is not None:
        return _GDN2_CLASS

    root = _external_root()
    if os.environ.get("EMENDER_GDN2_REQUIRE_BOUND_SOURCE") == "1":
        verify_bound_gdn2_source(
            root, os.environ.get("EMENDER_GDN2_EXPECTED_COMMIT"))
    gdn2_file = root / "lit_gpt" / "gdn2.py"
    if not gdn2_file.exists():
        raise ImportError(
            f"GDN-2 checkout not found at {root}. "
            "Clone https://github.com/NVlabs/GatedDeltaNet-2 or set GDN2_PATH."
        )

    package_name = "_external_gdn2_lit_gpt"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(root / "lit_gpt")]
        sys.modules[package_name] = pkg

    module_name = f"{package_name}.gdn2"
    _ensure_fla_compat()
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, gdn2_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load GDN-2 module from {gdn2_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    ops_module = importlib.import_module(f"{package_name}.gdn2_ops.chunk_gdn2")
    original = ops_module.chunk_gla_fwd_o_gk
    if not getattr(original, "_emender_gdn2_compat", False):
        signature = inspect.signature(original)

        def compat_chunk_gla_fwd_o_gk(*args, use_exp2=None, transpose_state_layout=False, **kwargs):
            if "use_exp2" in signature.parameters:
                kwargs["use_exp2"] = use_exp2
            if "transpose_state_layout" in signature.parameters:
                kwargs["transpose_state_layout"] = transpose_state_layout
            elif "state_v_first" in signature.parameters:
                kwargs["state_v_first"] = transpose_state_layout
            return original(*args, **kwargs)

        compat_chunk_gla_fwd_o_gk._emender_gdn2_compat = True
        ops_module.chunk_gla_fwd_o_gk = compat_chunk_gla_fwd_o_gk

    _GDN2_CLASS = module.GatedDeltaNet2
    _GDN2_OPS_MODULE = ops_module
    return _GDN2_CLASS


def probe_gdn2_external_dependencies() -> dict[str, Any]:
    """Import the external GDN-2 stack and return an actionable preflight report.

    This intentionally uses the same loader and FLA compatibility shims as the
    production wrapper so a PASS means model construction will see the same
    external module and chunk-kernel namespace. It does not launch a Triton
    kernel; use ``scripts/frontier/gdn2_rocm_preflight.py --run-fwdbwd`` for the
    compile/runtime smoke inside a GPU allocation.
    """
    global _GDN2_OPS_MODULE

    root = _external_root()
    gdn2_file = root / "lit_gpt" / "gdn2.py"
    report: dict[str, Any] = {
        "gdn2_path": str(root),
        "gdn2_file": str(gdn2_file),
        "gdn2_file_exists": gdn2_file.exists(),
        "bound_source_required": (
            os.environ.get("EMENDER_GDN2_REQUIRE_BOUND_SOURCE") == "1"),
    }
    receipt_path = root / BOUND_SOURCE_RECEIPT
    if receipt_path.is_file():
        try:
            report["bound_source"] = verify_bound_gdn2_source(
                root, os.environ.get("EMENDER_GDN2_EXPECTED_COMMIT"))
        except Exception as exc:
            report["ok"] = False
            report["failure"] = f"invalid bound GDN2 source: {exc!r}"
            return report
    elif report["bound_source_required"]:
        report["ok"] = False
        report["failure"] = f"bound GDN2 source receipt missing: {receipt_path}"
        return report

    try:
        import torch

        report.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "torch_version_hip": getattr(torch.version, "hip", None),
                "torch_cuda_is_available": torch.cuda.is_available(),
                "torch_cuda_device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "torch_backend": "hip"
                if getattr(torch.version, "hip", None)
                else "cuda"
                if getattr(torch.version, "cuda", None)
                else "cpu",
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["torch_import_error"] = repr(exc)

    try:
        fla_module = importlib.import_module("fla")
        report["fla_module"] = getattr(fla_module, "__file__", None)
        report["fla_version"] = getattr(fla_module, "__version__", None)
    except Exception as exc:
        report["ok"] = False
        report["failure"] = f"could not import fla: {exc!r}"
        return report

    try:
        cls = _load_gdn2_class()
        ops_module = _GDN2_OPS_MODULE or importlib.import_module(
            "_external_gdn2_lit_gpt.gdn2_ops.chunk_gdn2"
        )
        chunk_symbols = sorted(
            name
            for name in dir(ops_module)
            if name.startswith("chunk") or name.endswith("chunk_gdn2")
        )
        required = ["chunk_gla_fwd_o_gk"]
        missing = [name for name in required if not hasattr(ops_module, name)]
        report.update(
            {
                "external_module": cls.__module__,
                "external_class": cls.__name__,
                "chunk_ops_module": getattr(ops_module, "__file__", None),
                "chunk_symbols": chunk_symbols,
                "required_symbols": required,
                "missing_required_symbols": missing,
                "ok": not missing,
            }
        )
        if missing:
            report["failure"] = (
                "external gdn2_ops.chunk_gdn2 is missing required symbols: "
                + ", ".join(missing)
            )
    except Exception as exc:
        report["ok"] = False
        report["failure"] = f"could not load external lit_gpt.gdn2 chunk path: {exc!r}"

    return report


class GDN2ExternalLayer(nn.Module):
    """LadderLM-compatible wrapper around an external GatedDeltaNet-2 layer."""

    def __init__(
        self,
        dim,
        expansion=2.0,
        dropout=0.0,
        head_dim=128,
        num_heads=None,
        use_conv=None,
        d_conv=4,
        allow_neg_eigval=False,
        **kwargs,
    ):
        super().__init__()
        GatedDeltaNet2 = _load_gdn2_class()

        if num_heads is None:
            num_heads = kwargs.get("n_heads")
        if use_conv is None or use_conv is False:
            use_conv = True

        if head_dim > 256:
            raise ValueError("GDN-2 kernels require head_dim <= 256")
        if num_heads is None:
            num_heads = max(1, dim // head_dim)

        self.dim = dim
        self.expansion = expansion
        self.num_heads = num_heads
        self.actual_head_dim = head_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gdn2 = GatedDeltaNet2(
            hidden_size=dim,
            expand_v=expansion,
            head_dim=self.actual_head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=use_conv,
            conv_size=d_conv,
            allow_neg_eigval=allow_neg_eigval,
            layer_idx=0,
        )

    def set_layer_idx(self, idx):
        self.gdn2.layer_idx = idx

    def forward(self, x, h0=None, **kwargs):
        if self.training:
            output, _, _ = self.gdn2(x, use_cache=False)
            return self.dropout(output), None

        output, _, cache = self.gdn2(x, past_key_values=h0, use_cache=True)
        return self.dropout(output), cache

    def extra_repr(self):
        return (
            f"dim={self.dim}, expansion={self.expansion}, "
            f"num_heads={self.num_heads}, head_dim={self.actual_head_dim}, "
            "external=GDN2"
        )


class _SwiGLUMLP(nn.Module):
    """Bias-free LLaMA-style SwiGLU MLP used by the official GDN-2 block."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _round_mlp_hidden(dim, mlp_ratio, multiple=64):
    return max(multiple, int(round(dim * mlp_ratio / multiple) * multiple))


class GDN2ExternalMLPLayer(nn.Module):
    """GDN-2 mixer plus the official-style post-mixer SwiGLU MLP.

    LadderLM supplies the first RMSNorm externally. This layer adds the second
    RMSNorm and MLP so its per-layer parameter count matches the official
    serial GDN-2 block. The residual placement is necessarily approximate
    because LadderLM owns the outer Mamba-style residual stream.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        head_dim=128,
        num_heads=None,
        use_conv=None,
        d_conv=4,
        gdn2_mlp_ratio=OFFICIAL_GDN2_MLP_RATIO,
        gdn2_mlp_multiple=64,
        **kwargs,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = kwargs.get("n_heads")

        self.dim = dim
        self.expansion = expansion
        self.num_heads = num_heads if num_heads is not None else max(1, dim // head_dim)
        self.actual_head_dim = head_dim
        self.mlp_ratio = gdn2_mlp_ratio
        self.mlp_hidden_dim = _round_mlp_hidden(dim, gdn2_mlp_ratio, gdn2_mlp_multiple)
        self.gdn2 = GDN2ExternalLayer(
            dim=dim,
            expansion=expansion,
            dropout=0.0,
            head_dim=head_dim,
            num_heads=self.num_heads,
            use_conv=use_conv,
            d_conv=d_conv,
            **kwargs,
        )
        self.norm_2 = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = _SwiGLUMLP(dim, self.mlp_hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_layer_idx(self, idx):
        self.gdn2.set_layer_idx(idx)

    def forward(self, x, h0=None, **kwargs):
        gdn_out, cache = self.gdn2(x, h0, **kwargs)
        mlp_out = self.mlp(self.norm_2(x + gdn_out))
        return self.dropout(gdn_out + mlp_out), cache

    def extra_repr(self):
        return (
            f"dim={self.dim}, expansion={self.expansion}, "
            f"num_heads={self.num_heads}, head_dim={self.actual_head_dim}, "
            f"mlp_hidden_dim={self.mlp_hidden_dim}, external=GDN2+MLP"
        )


def _gdn2_layer_params(dim, expansion=2.0, num_heads=None, head_dim=128, d_conv=4, use_conv=True):
    """Exact parameter count for the external GatedDeltaNet2 mixer module."""
    if num_heads is None:
        num_heads = max(1, dim // head_dim)
    value_head_dim = int(head_dim * expansion)
    key_dim = num_heads * head_dim
    value_dim = num_heads * value_head_dim
    conv_params = (key_dim * 2 + value_dim) * d_conv if use_conv else 0

    return (
        dim * key_dim * 2          # q_proj, k_proj
        + dim * value_dim          # v_proj
        + conv_params              # optional q/k/v depthwise short convs
        + dim * value_head_dim     # f_proj.0
        + value_head_dim * key_dim # f_proj.1
        + dim * key_dim            # b_proj
        + dim * value_dim          # w_proj
        + num_heads                # A_log
        + key_dim                  # dt_bias
        + dim * value_head_dim     # g_proj.0
        + value_head_dim * value_dim
        + value_dim                # g_proj.1 bias
        + value_head_dim           # o_norm weight
        + value_dim * dim          # o_proj
    )


def count_gdn2_external_params(dim, depth, vocab_size=256, expansion=2.0, num_heads=None, head_dim=128):
    """Exact LadderLM parameter count for the external GDN-2 mixer layer."""
    per_layer = _gdn2_layer_params(
        dim, expansion=expansion, num_heads=num_heads, head_dim=head_dim
    ) + dim  # LadderLM's pre-mixer RMSNorm
    return vocab_size * dim + depth * per_layer + dim


def count_gdn2_mlp_external_params(
    dim,
    depth,
    vocab_size=256,
    expansion=1.0,
    num_heads=None,
    head_dim=128,
    mlp_ratio=OFFICIAL_GDN2_MLP_RATIO,
    mlp_multiple=64,
):
    """Exact LadderLM parameter count for GDN-2 mixer + SwiGLU MLP."""
    mlp_hidden = _round_mlp_hidden(dim, mlp_ratio, mlp_multiple)
    per_layer = (
        _gdn2_layer_params(dim, expansion=expansion, num_heads=num_heads, head_dim=head_dim)
        + dim                # LadderLM pre-mixer RMSNorm
        + dim                # post-mixer RMSNorm inside GDN2ExternalMLPLayer
        + 3 * dim * mlp_hidden
    )
    return vocab_size * dim + depth * per_layer + dim
