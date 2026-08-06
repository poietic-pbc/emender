#!/usr/bin/env python3
"""Generate readable samples from current racer checkpoints.

This is intentionally tokenization-aware. The older batch generation helpers in
the repo are byte-oriented and are not appropriate for p50k_base checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ndm" / "cuda"))


def resolve_checkpoint(path: str) -> Path:
    ckpt = Path(path).resolve()
    if ckpt.is_symlink():
        ckpt = ckpt.parent / os.readlink(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return ckpt


def load_args(ckpt: Path) -> dict:
    args_path = ckpt.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found beside checkpoint: {args_path}")
    return json.loads(args_path.read_text())


def build_model(args: dict, vocab_size: int) -> torch.nn.Module:
    level = args["level"]
    if str(level) in {"97", "E97", "E97-M2"}:
        from ndm.e97 import build_e97_model

        return build_e97_model(args, vocab_size=vocab_size)

    if level == "mamba2":
        from ndm.models.mamba2_baseline import Mamba2LM

        return Mamba2LM(
            vocab_size=vocab_size,
            dim=args["dim"],
            depth=args["depth"],
            d_state=args.get("mamba_d_state", 64),
            expand=args.get("mamba_expand", 2),
            headdim=64,
            loss_chunk_size=0,
        )

    if level == "m2rnn":
        from ndm.models.m2rnn_baseline import M2RNNLM

        return M2RNNLM(
            vocab_size=vocab_size,
            dim=args["dim"],
            depth=args["depth"],
            n_heads=args.get("n_heads"),
            n_state=args.get("n_state", 64),
            expansion=args.get("expansion", 1.0),
            paper_shape=bool(args.get("m2rnn_paper_shape", False)),
            k_head_dim=args.get("m2rnn_k_head_dim"),
            v_head_dim=args.get("m2rnn_v_head_dim"),
            num_q_heads=args.get("m2rnn_q_heads"),
            num_k_heads=args.get("m2rnn_k_heads"),
            num_v_heads=args.get("m2rnn_v_heads"),
            num_f_heads=args.get("m2rnn_f_heads"),
            num_g_heads=args.get("m2rnn_g_heads"),
            num_weight_heads=args.get("m2rnn_weight_heads"),
            use_gate=bool(args.get("use_gate", 1)),
            use_residual=bool(args.get("m2rnn_use_residual", 1)),
            state_weight_trainable=not bool(args.get("m2rnn_freeze_state_weight", 0)),
            use_conv=bool(args.get("use_conv", 0)),
            d_conv=args.get("d_conv", 4),
            output_norm=bool(args.get("m2rnn_output_norm", 0)),
            normalize_qk=bool(args.get("m2rnn_normalize_qk", 0)),
            dropout=args.get("dropout", 0.0),
            gradient_clipping=args.get("m2rnn_state_grad_clip"),
            gradient_checkpointing=False,
            loss_chunk_size=0,
        )

    if level.lower() == "e88_fused":
        from ndm.models.e88_fused import E88FusedLM

        return E88FusedLM(
            vocab_size=vocab_size,
            dim=args["dim"],
            depth=args["depth"],
            n_heads=args.get("n_heads"),
            n_state=args.get("n_state", 64),
            expansion=args.get("expansion", 1.0),
            use_gate=bool(args.get("use_gate", 1)),
            checkpoint_interval=args.get("checkpoint_interval", 16),
        )

    from ndm.models import LadderLM

    return LadderLM(
        vocab_size=vocab_size,
        dim=args["dim"],
        depth=args["depth"],
        level=level,
        expansion=args.get("expansion", 1.0),
        n_groups=args.get("n_groups", 32),
        n_state=args.get("n_state", 64),
        n_slots=args.get("n_slots", 64),
        n_heads=args.get("n_heads"),
        top_k=args.get("top_k"),
        k_fast=args.get("k_fast"),
        k_slow=args.get("k_slow"),
        use_gate=bool(args.get("use_gate", 1)),
        gate_activation=args.get("gate_activation", "sigmoid"),
        linear_state=bool(args.get("linear_state", 0)),
        use_write_gate=bool(args.get("use_write_gate", 0)),
        e88_decay_mode=args.get("e88_decay_mode", "mamba"),
        e88_value_residual=bool(args.get("e88_value_residual", 0)),
        state_expansion=args.get("state_expansion", 2),
        r_h_mode=args.get("r_h_mode", "none"),
        use_conv=bool(args.get("use_conv", 0)),
        d_conv=args.get("d_conv", 4),
        dropout=args.get("dropout", 0.0),
        checkpoint_interval=args.get("checkpoint_interval", 16),
        gradient_checkpointing=False,
        projection_chunk_size=0,
        loss_chunk_size=0,
        use_triton=bool(args.get("use_triton", 0)),
    )


def apply_schedulefree_train_weights(model: torch.nn.Module, ckpt: dict, args: dict) -> bool:
    if args.get("optimizer", "adamw") != "schedulefree":
        return False
    if "optimizer_state_dict" not in ckpt:
        return False
    import schedulefree

    opt = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.get("lr", 3e-4),
        weight_decay=args.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
        warmup_steps=args.get("warmup_steps", 0),
    )
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    opt.train()
    return True


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, history: list[int],
           rep_penalty: float) -> int:
    logits = logits.float().clone()
    if rep_penalty != 1.0 and history:
        for tok in set(history):
            v = logits[tok]
            logits[tok] = v / rep_penalty if v > 0 else v * rep_penalty
    if temperature <= 0:
        return int(logits.argmax().item())
    logits /= temperature
    if top_k > 0 and top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < values[-1], torch.full_like(logits, float("-inf")), logits)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        mask = torch.cumsum(probs, dim=-1) > top_p
        mask[0] = False
        sorted_logits[mask] = float("-inf")
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def should_stop(tok: int, stop_tokens: set[int]) -> bool:
    return tok in stop_tokens


@torch.no_grad()
def generate_stateful(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new: int,
    args,
    stop_tokens: set[int],
) -> list[int]:
    generated = list(prompt_tokens)
    hiddens = None
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)
    logits, (hiddens, _) = model(prompt, return_loss=False, return_prev_hiddens=True, prev_hiddens=None)
    next_tok = sample(
        logits[0, -1],
        args.temperature,
        args.top_k,
        args.top_p,
        generated[-args.rep_window :],
        args.rep_penalty,
    )
    generated.append(next_tok)
    if should_stop(next_tok, stop_tokens):
        return generated
    while len(generated) < len(prompt_tokens) + max_new:
        token = torch.tensor([[generated[-1]]], dtype=torch.long, device=args.device)
        logits, (hiddens, _) = model(token, return_loss=False, return_prev_hiddens=True, prev_hiddens=hiddens)
        next_tok = sample(
            logits[0, -1],
            args.temperature,
            args.top_k,
            args.top_p,
            generated[-args.rep_window :],
            args.rep_penalty,
        )
        generated.append(next_tok)
        if should_stop(next_tok, stop_tokens):
            break
    return generated


@torch.no_grad()
def generate_full_context(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new: int,
    args,
    stop_tokens: set[int],
) -> list[int]:
    generated = list(prompt_tokens)
    while len(generated) < len(prompt_tokens) + max_new:
        ctx = generated[-args.max_context :]
        x = torch.tensor([ctx], dtype=torch.long, device=args.device)
        logits = model(x, return_loss=False)
        next_tok = sample(
            logits[0, -1],
            args.temperature,
            args.top_k,
            args.top_p,
            generated[-args.rep_window :],
            args.rep_penalty,
        )
        generated.append(next_tok)
        if should_stop(next_tok, stop_tokens):
            break
    return generated


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out_dir", default="/tmp/racer_generations")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--rep_penalty", type=float, default=1.0)
    p.add_argument("--rep_window", type=int, default=64)
    p.add_argument("--max_context", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    p.add_argument("--full_context", action="store_true")
    p.add_argument("--prefix_rs", action="store_true",
                   help="Prepend the raw record-separator delimiter before the prompt.")
    p.add_argument("--stop_on_rs", action="store_true",
                   help="Stop generation when the raw record-separator delimiter is sampled.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    model_args = load_args(ckpt_path)

    tokenizer_name = model_args.get("tokenizer")
    if tokenizer_name:
        import tiktoken

        enc = tiktoken.get_encoding(tokenizer_name)
        prompt_tokens = enc.encode(args.prompt, disallowed_special=())
        decode = enc.decode
        vocab_size = enc.n_vocab
        rs_tokens = enc.encode("\x1e", disallowed_special=())
    else:
        enc = None
        prompt_tokens = list(args.prompt.encode("utf-8"))
        decode = lambda toks: bytes([t for t in toks if 0 <= t < 256]).decode("utf-8", errors="replace")
        vocab_size = 256
        rs_tokens = [0x1e]

    if args.prefix_rs:
        prompt_tokens = rs_tokens + prompt_tokens
    stop_tokens = set(rs_tokens) if args.stop_on_rs else set()

    model = build_model(model_args, vocab_size)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    used_sf_swap = apply_schedulefree_train_weights(model, ckpt, model_args)
    model = model.to(args.device).bfloat16().eval()

    is_e97 = str(model_args["level"]) in {"97", "E97", "E97-M2"}
    fused_e97_generation = (
        is_e97
        and bool(model_args.get("use_triton", 0))
        and str(args.device).startswith("cuda")
    )
    if is_e97:
        from ndm.models import E97SplitEditLayer

        for module in model.modules():
            if isinstance(module, E97SplitEditLayer):
                module.fused_inference = fused_e97_generation
    # The shared sequential Triton kernel right-pads unaligned lengths. Its
    # real-position outputs are causal, but the padded final state is not safe to
    # carry token-by-token. Fused E97 generation therefore defaults to the exact
    # full-context path. Use scripts/generate_e97.py --mode stateful for the eager
    # state-carrying path.
    use_full_context = args.full_context or model_args["level"] == "mamba2" or (
        fused_e97_generation
    )
    t0 = time.time()
    if use_full_context:
        generated = generate_full_context(model, prompt_tokens, args.max_new_tokens, args, stop_tokens)
        mode = "full_context"
    else:
        generated = generate_stateful(model, prompt_tokens, args.max_new_tokens, args, stop_tokens)
        mode = "stateful"
    elapsed = time.time() - t0

    text = decode(generated)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.txt"
    meta = {
        "label": args.label,
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_loss": ckpt.get("loss"),
        "level": model_args["level"],
        "model": "emender/nonlin" if is_e97 else str(model_args["level"]),
        "kernel_api": (
            "e97-sequential-split-edit-triton"
            if fused_e97_generation
            else ("e97-split-edit-eager" if is_e97 else None)
        ),
        "kernel_core": (
            "e88-shared-triton"
            if fused_e97_generation
            else ("python-reference" if is_e97 else None)
        ),
        "mode": mode,
        "tokenizer": tokenizer_name or "byte",
        "prompt_tokens": len(prompt_tokens),
        "max_new_tokens": args.max_new_tokens,
        "generated_new_tokens": max(0, len(generated) - len(prompt_tokens)),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "rep_penalty": args.rep_penalty,
        "seed": args.seed,
        "prefix_rs": args.prefix_rs,
        "stop_on_rs": args.stop_on_rs,
        "rs_tokens": rs_tokens,
        "schedulefree_train_weight_swap": used_sf_swap,
        "elapsed_s": elapsed,
        "tokens_per_s": max(0, len(generated) - len(prompt_tokens)) / max(elapsed, 1e-9),
    }
    out_path.write_text(
        json.dumps(meta, indent=2)
        + "\n\n--- PROMPT ---\n"
        + args.prompt
        + "\n\n--- SAMPLE ---\n"
        + text,
        errors="replace",
    )
    print(json.dumps(meta, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
