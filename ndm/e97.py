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
from typing import Any, Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class E97RecurrentCache:
    """Committed recurrent state after an exact token prefix.

    Cache values are immutable from the caller's perspective. Advancing a
    cache returns a new value, which lets serving code generate into a shadow
    cache and commit it only after a complete assistant turn is accepted.
    """

    token_ids: tuple[int, ...]
    hidden: Any
    next_logits: torch.Tensor
    checkpoint: str

    @property
    def state_bytes(self) -> int:
        """Bytes occupied by recurrent tensors and boundary logits."""

        def tensor_bytes(value: Any) -> int:
            if torch.is_tensor(value):
                return value.numel() * value.element_size()
            if isinstance(value, (list, tuple)):
                return sum(tensor_bytes(item) for item in value)
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            return 0

        return tensor_bytes(self.hidden) + tensor_bytes(self.next_logits)


def _cache_checkpoint_identity(loaded: LoadedE97Checkpoint) -> str:
    return str(loaded.checkpoint_path)


def e97_cache_suffix(
    cache: E97RecurrentCache,
    requested_token_ids: Sequence[int],
) -> tuple[int, ...] | None:
    """Return an append-only suffix, or ``None`` for an incompatible prefix."""

    requested = tuple(int(token) for token in requested_token_ids)
    prefix_length = len(cache.token_ids)
    if len(requested) < prefix_length:
        return None
    if requested[:prefix_length] != cache.token_ids:
        return None
    return requested[prefix_length:]


def _validate_e97_cache_advance(
    loaded: LoadedE97Checkpoint,
    consumed: tuple[int, ...],
    cache: E97RecurrentCache | None,
) -> str:
    if not consumed:
        if cache is None:
            raise ValueError("an initial E97 cache requires at least one token")
        return _cache_checkpoint_identity(loaded)
    checkpoint = _cache_checkpoint_identity(loaded)
    if cache is not None and cache.checkpoint != checkpoint:
        raise ValueError("E97 cache belongs to a different checkpoint")
    if any(
        isinstance(module, E97SplitEditLayer) and bool(module.use_conv)
        for module in loaded.model.modules()
    ):
        raise NotImplementedError(
            "cached E97 inference does not yet carry convolution buffers"
        )
    return checkpoint


@torch.no_grad()
def advance_e97_cache_segment(
    loaded: LoadedE97Checkpoint,
    token_ids: Sequence[int],
    cache: E97RecurrentCache | None = None,
) -> E97RecurrentCache:
    """Consume one explicit logical segment with a fused variable-length call.

    This compatibility path preserves the full-prefix behavior used to
    evaluate dense-agent v1, but BF16 projection rounding can depend on segment
    boundaries. New serving authorities should use :func:`advance_e97_cache`.
    """

    consumed = tuple(int(token) for token in token_ids)
    checkpoint = _validate_e97_cache_advance(loaded, consumed, cache)
    if not consumed:
        assert cache is not None
        return cache
    model_device = next(loaded.model.parameters()).device
    tokens = torch.tensor([consumed], dtype=torch.long, device=model_device)
    logits, (hidden, _) = loaded.model(
        tokens,
        return_loss=False,
        return_prev_hiddens=True,
        prev_hiddens=None if cache is None else cache.hidden,
    )
    return E97RecurrentCache(
        token_ids=(cache.token_ids if cache is not None else ()) + consumed,
        hidden=hidden,
        next_logits=logits[0, -1].detach(),
        checkpoint=checkpoint,
    )


@torch.no_grad()
def advance_e97_cache(
    loaded: LoadedE97Checkpoint,
    token_ids: Sequence[int],
    cache: E97RecurrentCache | None = None,
) -> E97RecurrentCache:
    """Consume ``token_ids`` and return the resulting recurrent boundary.

    The fused E97 recurrence pads unaligned inference chunks internally, but
    returns the state at the valid-token boundary. Inference states remain
    FP32, so arbitrary HTTP/turn chunking does not repeatedly narrow them.
    """

    consumed = tuple(int(token) for token in token_ids)
    checkpoint = _validate_e97_cache_advance(loaded, consumed, cache)
    if not consumed:
        assert cache is not None
        return cache

    # Canonicalize ingestion to one-token model calls. BF16 projection GEMMs
    # can round differently when their time dimension changes; even small
    # boundary differences can eventually change greedy decoding. Tokenwise
    # ingestion makes replay, HTTP turn boundaries, and arbitrary caller
    # chunking execute the same shapes. This is the correctness authority;
    # faster chunk-invariant projection kernels may replace it after separate
    # qualification.
    model_device = next(loaded.model.parameters()).device
    hidden = None if cache is None else cache.hidden
    next_logits = None
    for token in consumed:
        tokens = torch.tensor([[token]], dtype=torch.long, device=model_device)
        logits, (hidden, _) = loaded.model(
            tokens,
            return_loss=False,
            return_prev_hiddens=True,
            prev_hiddens=hidden,
        )
        next_logits = logits[0, -1].detach()
    assert next_logits is not None
    return E97RecurrentCache(
        token_ids=(cache.token_ids if cache is not None else ()) + consumed,
        hidden=hidden,
        next_logits=next_logits,
        checkpoint=checkpoint,
    )


@torch.no_grad()
def generate_e97_from_cache(
    loaded: LoadedE97Checkpoint,
    cache: E97RecurrentCache,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.0,
    stop_token_ids: Iterable[int] = (),
) -> tuple[list[int], E97RecurrentCache]:
    """Generate into a new cache without mutating the committed input cache."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cache.checkpoint != _cache_checkpoint_identity(loaded):
        raise ValueError("E97 cache belongs to a different checkpoint")

    stop_tokens = {int(token) for token in stop_token_ids}
    shadow = cache
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token = _sample_token(
            shadow.next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(token)
        # Consume every emitted token, including RS/stop, so a committed cache
        # represents exactly the prefix reconstructed by the next Pi request.
        shadow = advance_e97_cache(loaded, [token], shadow)
        if token in stop_tokens:
            break
    return generated, shadow


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

    ``auto`` retains the historical full-context default for fused checkpoints
    until cached production serving is separately qualified. Explicit
    ``stateful`` mode uses the valid-length fused final state and FP32 recurrent
    caches when Triton is enabled.
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

    model_device = next(loaded.model.parameters()).device
    generated = list(prompt_tokens)
    stop_tokens = {int(token) for token in stop_token_ids}

    if selected_mode == "stateful":
        cache = advance_e97_cache(loaded, prompt_tokens)
        new_tokens, _ = generate_e97_from_cache(
            loaded,
            cache,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_ids=stop_tokens,
        )
        generated.extend(new_tokens)
    else:
        for _ in range(max_new_tokens):
            context = generated[-max_context:]
            tokens = torch.tensor([context], dtype=torch.long, device=model_device)
            logits = loaded.model(tokens, return_loss=False)
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
    "E97RecurrentCache",
    "LoadedE97Checkpoint",
    "advance_e97_cache",
    "advance_e97_cache_segment",
    "build_e97_model",
    "e97_cache_suffix",
    "e97_checkpoint_config",
    "e97_model_kwargs_from_config",
    "generate_e97",
    "generate_e97_from_cache",
    "is_e97_level",
    "load_e97_checkpoint",
    "resolve_e97_checkpoint",
]
