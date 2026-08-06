"""Public checkpoint loading and generation helpers for E97/Emender.

The historical training checkpoints store a generic ``LadderLM`` state dict and
an adjacent ``args.json``.  This module is the supported E97-facing bridge from
those artifacts to the E97 model identity used by current code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from .models import E97SplitEditLayer, LadderLM


_CHECKPOINT_STEP_RE = re.compile(r"checkpoint_step_(\d+)")
_E97_LEVELS = {"97", "E97", "E97-M2"}


def is_e97_level(level: Any) -> bool:
    """Return whether a saved level selects an E97-family model."""

    return str(level) in _E97_LEVELS


def resolve_e97_checkpoint(path: str | Path) -> Path:
    """Resolve a checkpoint file, ``latest.pt`` symlink, or run directory."""

    candidate = Path(path).expanduser()
    if candidate.is_dir():
        latest = candidate / "latest.pt"
        if latest.exists() or latest.is_symlink():
            candidate = latest
        else:
            checkpoints = list(candidate.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"no E97 checkpoints found in {candidate}")

            def checkpoint_key(item: Path) -> tuple[int, str]:
                match = _CHECKPOINT_STEP_RE.search(item.name)
                return (int(match.group(1)) if match else -1, item.name)

            candidate = max(checkpoints, key=checkpoint_key)
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"E97 checkpoint not found: {candidate}") from exc


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def e97_checkpoint_config(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    args_json: str | Path | None = None,
) -> dict[str, Any]:
    """Find and validate the training configuration for an E97 checkpoint."""

    if args_json is not None:
        config = _json_object(Path(args_json).expanduser().resolve(strict=True))
    else:
        config = None
        for key in ("args", "config", "cfg"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                config = dict(value)
                break
        if config is None:
            sibling = checkpoint_path.parent / "args.json"
            if not sibling.exists():
                raise FileNotFoundError(
                    "E97 checkpoints require their training arguments. Place args.json "
                    f"beside {checkpoint_path.name} or pass args_json=..."
                )
            config = _json_object(sibling)

    if not is_e97_level(config.get("level")):
        raise ValueError(
            f"checkpoint config level={config.get('level')!r} is not E97; "
            f"expected one of {sorted(_E97_LEVELS)}"
        )
    return config


def _layer_kwargs(config: Mapping[str, Any]) -> dict[str, Any] | None:
    value = config.get("layer_kwargs")
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("layer_kwargs must be a JSON object or mapping")
    return dict(value)


def e97_model_kwargs_from_config(
    config: Mapping[str, Any],
    *,
    vocab_size: int,
    use_triton: bool | None = None,
) -> dict[str, Any]:
    """Translate a frozen E97 ``args.json`` into ``LadderLM`` arguments."""

    if not is_e97_level(config.get("level")):
        raise ValueError(f"not an E97 config: level={config.get('level')!r}")
    if config.get("dim") is None or config.get("depth") is None:
        raise ValueError("E97 checkpoint config must record explicit dim and depth")

    level = str(config["level"])
    if level == "97":
        level = "E97"
    resolved_triton = bool(config.get("use_triton", config.get("bf16", False)))
    if use_triton is not None:
        resolved_triton = bool(use_triton)
    r_h_mode = config.get("r_h_mode", "none")
    if r_h_mode == "auto":
        r_h_mode = "none"

    return {
        "vocab_size": int(vocab_size),
        "dim": int(config["dim"]),
        "depth": int(config["depth"]),
        "level": level,
        "layer_kwargs": _layer_kwargs(config),
        "expansion": float(config.get("expansion", 1.0)),
        "n_groups": int(config.get("n_groups", 32)),
        "n_state": int(config.get("n_state", 64)),
        "n_slots": int(config.get("n_slots", 64)),
        "n_heads": config.get("n_heads"),
        "top_k": config.get("top_k"),
        "k_fast": config.get("k_fast"),
        "k_slow": config.get("k_slow"),
        "use_gate": bool(config.get("use_gate", 1)),
        "gate_activation": config.get("gate_activation", "sigmoid"),
        "linear_state": bool(config.get("linear_state", 0)),
        "use_write_gate": bool(config.get("use_write_gate", 0)),
        "e88_decay_mode": config.get("e88_decay_mode", "mamba"),
        "e88_value_residual": bool(config.get("e88_value_residual", 0)),
        "e88_raw_write": bool(config.get("e88_raw_write", 0)),
        "use_chunked_e97": bool(config.get("use_chunked_e97", 0)),
        "e97_chunk_size": int(config.get("e97_chunk_size", 32)),
        "state_expansion": int(config.get("state_expansion", 2)),
        "r_h_mode": r_h_mode,
        "use_conv": bool(config.get("use_conv", 0)),
        "d_conv": int(config.get("d_conv", 4)),
        "gdn2_mlp_ratio": float(config.get("gdn2_mlp_ratio", 0.0)),
        "dropout": float(config.get("dropout", 0.0)),
        "checkpoint_interval": int(config.get("checkpoint_interval", 16)),
        "gradient_checkpointing": False,
        "projection_chunk_size": int(config.get("projection_chunk_size", 0)),
        "loss_chunk_size": int(config.get("loss_chunk_size", 0)),
        "use_triton": resolved_triton,
        "mlp_ratio": float(config.get("mlp_ratio", 0.0)),
        "mlp_multiple": int(config.get("mlp_multiple", 64)),
        "state_summary_dim": int(config.get("state_summary_dim") or 0),
        "mlp_hidden": config.get("mlp_hidden"),
    }


def _vocab_size(
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> int:
    explicit = config.get("vocab_size")
    if explicit:
        return int(explicit)
    if state_dict is not None and "embedding.weight" in state_dict:
        return int(state_dict["embedding.weight"].shape[0])
    tokenizer_name = config.get("tokenizer")
    if tokenizer_name:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "install the eval extra (`pip install 'ndm[eval]'`) to derive the "
                f"vocabulary for tokenizer {tokenizer_name!r}"
            ) from exc
        return int(tiktoken.get_encoding(tokenizer_name).n_vocab)
    return 256


def build_e97_model(
    config: Mapping[str, Any],
    *,
    vocab_size: int | None = None,
    use_triton: bool | None = None,
) -> LadderLM:
    """Construct an E97 model from frozen training arguments."""

    kwargs = e97_model_kwargs_from_config(
        config,
        vocab_size=_vocab_size(config) if vocab_size is None else vocab_size,
        use_triton=use_triton,
    )
    model = LadderLM(**kwargs)
    if not any(isinstance(module, E97SplitEditLayer) for module in model.modules()):
        raise RuntimeError("E97 model construction did not produce an E97SplitEditLayer")
    return model


def _resolve_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    names = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return names[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}; choose {sorted(names)}") from exc


def _apply_schedulefree_train_weights(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    if config.get("optimizer", "adamw") != "schedulefree":
        return False
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        raise ValueError(
            "this E97 checkpoint stores schedule-free averaged weights but has no "
            "optimizer_state_dict from which to recover generation/train weights; "
            "load with weight_mode='saved' only if averaged x-mode is intentional"
        )

    import schedulefree

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
        warmup_steps=int(config.get("warmup_steps", 0)),
    )
    optimizer.load_state_dict(optimizer_state)
    optimizer.train()
    del optimizer
    return True


def _replace_cpu_rmsnorms(module: torch.nn.Module) -> None:
    """Replace mamba-ssm's Triton-only RMSNorm modules for CPU inference."""

    for name, child in list(module.named_children()):
        if child.__class__.__module__.startswith("mamba_ssm.") and hasattr(child, "weight"):
            weight = child.weight
            replacement = torch.nn.RMSNorm(
                int(weight.numel()),
                eps=float(getattr(child, "eps", 1e-6)),
            ).to(device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(weight)
            module._modules[name] = replacement
        else:
            _replace_cpu_rmsnorms(child)


@dataclass
class LoadedE97Checkpoint:
    """A strictly reconstructed E97 checkpoint ready for eval or generation."""

    model: LadderLM
    config: dict[str, Any]
    checkpoint_path: Path
    step: int | None
    loss: float | None
    checkpoint_metadata: dict[str, Any]
    weight_mode: str
    schedulefree_train_weight_swap: bool

    @property
    def tokenizer_name(self) -> str | None:
        return self.config.get("tokenizer")


def load_e97_checkpoint(
    path: str | Path,
    *,
    args_json: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | str | None = None,
    weight_mode: str = "train",
    use_triton: bool | None = None,
    mmap: bool = True,
) -> LoadedE97Checkpoint:
    """Strictly reconstruct and load an E97 training checkpoint.

    ``weight_mode='train'`` is the generation default for schedule-free training
    runs, including the completed 150B checkpoint.  It uses the saved optimizer
    state to recover the y/train weights from the checkpoint's averaged x-mode.
    Use ``weight_mode='saved'`` only when the stored averaged weights are desired.
    """

    if weight_mode not in {"train", "saved"}:
        raise ValueError("weight_mode must be 'train' or 'saved'")
    checkpoint_path = resolve_e97_checkpoint(path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=mmap,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a train.py checkpoint")
    state_dict = checkpoint["model_state_dict"]
    config = e97_checkpoint_config(checkpoint_path, checkpoint, args_json)
    resolved_device = torch.device(device)
    effective_use_triton = use_triton
    if resolved_device.type != "cuda" and effective_use_triton is None:
        effective_use_triton = False
    model = build_e97_model(
        config,
        vocab_size=_vocab_size(config, state_dict),
        use_triton=effective_use_triton,
    )
    model.load_state_dict(state_dict, strict=True)
    swapped = (
        _apply_schedulefree_train_weights(model, checkpoint, config)
        if weight_mode == "train"
        else False
    )
    step = int(checkpoint["step"]) if checkpoint.get("step") is not None else None
    loss = float(checkpoint["loss"]) if checkpoint.get("loss") is not None else None
    raw_metadata = checkpoint.get("checkpoint_metadata")
    checkpoint_metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    # Drop references to the multi-gigabyte mmap-backed payload before moving the
    # reconstructed model to its generation device.
    del state_dict
    if isinstance(checkpoint, dict):
        checkpoint.pop("model_state_dict", None)
        checkpoint.pop("optimizer_state_dict", None)

    resolved_dtype = _resolve_dtype(dtype)
    if resolved_device.type == "cpu" and hasattr(model, "fused_add_norm"):
        model.fused_add_norm = False
        _replace_cpu_rmsnorms(model)
    if resolved_dtype is None:
        model = model.to(device=resolved_device)
    else:
        model = model.to(device=resolved_device, dtype=resolved_dtype)
    for module in model.modules():
        if isinstance(module, E97SplitEditLayer):
            module.fused_inference = bool(module.use_triton and resolved_device.type == "cuda")
    model.eval()

    loaded = LoadedE97Checkpoint(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        step=step,
        loss=loss,
        checkpoint_metadata=checkpoint_metadata,
        weight_mode=weight_mode,
        schedulefree_train_weight_swap=swapped,
    )
    del checkpoint
    return loaded


def _tokenizer(config: Mapping[str, Any]):
    tokenizer_name = config.get("tokenizer")
    if tokenizer_name:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "E97 text generation requires the eval extra: pip install 'ndm[eval]'"
            ) from exc
        encoding = tiktoken.get_encoding(tokenizer_name)
        return (
            lambda text: encoding.encode(text, disallowed_special=()),
            encoding.decode,
        )
    return (
        lambda text: list(text.encode("utf-8")),
        lambda ids: bytes(token for token in ids if 0 <= token < 256).decode(
            "utf-8", errors="replace"
        ),
    )


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    scores = logits.float()
    if temperature <= 0:
        return int(scores.argmax().item())
    scores = scores / temperature
    if 0 < top_k < scores.numel():
        threshold = torch.topk(scores, top_k).values[-1]
        scores = torch.where(scores < threshold, -torch.inf, scores)
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = F.softmax(sorted_scores, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_scores[remove] = -torch.inf
        scores = torch.empty_like(scores).scatter_(0, sorted_indices, sorted_scores)
    return int(torch.multinomial(F.softmax(scores, dim=-1), 1).item())


def _uses_triton(model: torch.nn.Module) -> bool:
    try:
        on_cuda = next(model.parameters()).is_cuda
    except StopIteration:
        on_cuda = False
    return on_cuda and any(
        isinstance(module, E97SplitEditLayer) and bool(module.use_triton)
        for module in model.modules()
    )


@torch.no_grad()
def generate_e97(
    loaded: LoadedE97Checkpoint,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.0,
    max_context: int = 2048,
    mode: str = "auto",
    stop_token_ids: Iterable[int] = (),
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate text from a loaded E97 checkpoint.

    ``auto`` selects fused full-context generation when Triton is enabled and
    exact stateful generation when it is disabled.  The current shared
    sequential Triton engine pads unaligned lengths for sparse checkpoints; its
    real-token outputs are causal and correct, but its padded final state must
    not be carried token by token.  Consequently, stateful generation is
    deliberately rejected unless the model was loaded with ``use_triton=False``.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_context <= 0:
        raise ValueError("max_context must be positive")
    if mode not in {"auto", "full-context", "stateful"}:
        raise ValueError("mode must be auto, full-context, or stateful")
    encode, decode = _tokenizer(loaded.config)
    prompt_tokens = encode(prompt)
    if not prompt_tokens:
        raise ValueError("prompt must encode to at least one token")
    if seed is not None:
        torch.manual_seed(seed)

    fused = _uses_triton(loaded.model)
    selected_mode = "full-context" if mode == "auto" and fused else mode
    if selected_mode == "auto":
        selected_mode = "stateful"
    if selected_mode == "stateful" and fused:
        raise ValueError(
            "stateful E97 generation requires load_e97_checkpoint(..., use_triton=False); "
            "use mode='full-context' with the fused sequential kernel"
        )

    model_device = next(loaded.model.parameters()).device
    generated = list(prompt_tokens)
    stop_tokens = {int(token) for token in stop_token_ids}
    hidden = None

    if selected_mode == "stateful" and max_new_tokens:
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=model_device)
        logits, (hidden, _) = loaded.model(
            prompt_tensor,
            return_loss=False,
            return_prev_hiddens=True,
            prev_hiddens=None,
        )

    for index in range(max_new_tokens):
        if selected_mode == "full-context":
            context = generated[-max_context:]
            tokens = torch.tensor([context], dtype=torch.long, device=model_device)
            logits = loaded.model(tokens, return_loss=False)
        elif index > 0:
            tokens = torch.tensor([[generated[-1]]], dtype=torch.long, device=model_device)
            logits, (hidden, _) = loaded.model(
                tokens,
                return_loss=False,
                return_prev_hiddens=True,
                prev_hiddens=hidden,
            )

        next_token = _sample_token(
            logits[0, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(next_token)
        if next_token in stop_tokens:
            break

    new_tokens = generated[len(prompt_tokens):]
    return {
        "text": decode(generated),
        "new_text": decode(new_tokens),
        "token_ids": generated,
        "new_token_ids": new_tokens,
        "mode": selected_mode,
        "model": "emender/nonlin",
        "historical_level": str(loaded.config["level"]),
        "kernel_api": (
            "e97-sequential-split-edit-triton" if fused else "e97-split-edit-eager"
        ),
        "kernel_core": "e88-shared-triton" if fused else "python-reference",
        "checkpoint": str(loaded.checkpoint_path),
        "step": loaded.step,
        "weight_mode": loaded.weight_mode,
    }


__all__ = [
    "LoadedE97Checkpoint",
    "build_e97_model",
    "e97_checkpoint_config",
    "e97_model_kwargs_from_config",
    "generate_e97",
    "is_e97_level",
    "load_e97_checkpoint",
    "resolve_e97_checkpoint",
]
