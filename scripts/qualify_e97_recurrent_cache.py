#!/usr/bin/env python3
"""Qualify dense E97 fused recurrent caching against uninterrupted replay."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from ndm.e97 import (
    E97RecurrentCache,
    advance_e97_cache,
    e97_cache_suffix,
    generate_e97_from_cache,
    load_e97_checkpoint,
)

RS = "\x1e"
PROMPTS = (
    'System:\nYou are a precise tool-using agent. Respond with either "Action:" and one JSON '
    '"Arguments:" object, or "Final:". Never invent tool results.\n\n'
    'User:\nCalculate 308 * 641.\n\nAssistant:\nAction: calculator\n'
    'Arguments: {"expression":"308 * 641"}' + RS + '\n\n'
    'Tool:\n{"expression":"308 * 641","value":"197428"}\n\nAssistant:\n',
    'System:\nYou are a precise tool-using agent. Use registered tools and cite the value returned.\n\n'
    'User:\nWho owns Project-004211?\n\nAssistant:\nAction: lookup\n'
    'Arguments: {"field":"owner","project":"Project-004211"}' + RS + '\n\n'
    'Tool:\n{"field":"owner","value":"Devika"}\n\nAssistant:\n',
    'System:\nYou are a bounded repository assistant. Never invent file contents.\n\n'
    'User:\nRead the project configuration and report the tokenizer.\n\n'
    'Assistant:\nAction: read\nArguments: {"path":"args.json","offset":1,"limit":80}' + RS + '\n\n'
    'Tool:\n{"content":"{\\"tokenizer\\":\\"p50k_base\\"}"}\n\nAssistant:\n',
)
CHUNK_PATTERNS = (
    (1, 7, 16, 31),
    (3, 5, 11, 23),
    (15, 1, 17, 9),
    (2, 29, 4, 13),
    (8, 8, 8, 8),
    (31, 16, 7, 1),
    (5, 19, 2, 27),
    (9, 3, 25, 6),
)


def tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def max_abs_difference(left: Any, right: Any) -> float:
    left_tensors = list(tensors(left))
    right_tensors = list(tensors(right))
    if len(left_tensors) != len(right_tensors):
        raise AssertionError("hidden tensor counts differ")
    maximum = 0.0
    for left_tensor, right_tensor in zip(left_tensors, right_tensors):
        if left_tensor.shape != right_tensor.shape:
            raise AssertionError("hidden tensor shapes differ")
        difference = (left_tensor.float() - right_tensor.float()).abs().max().item()
        maximum = max(maximum, float(difference))
    return maximum


def clone_hidden(cache: E97RecurrentCache) -> list[torch.Tensor]:
    return [tensor.detach().clone() for tensor in tensors(cache.hidden)]


def hidden_unchanged(before: list[torch.Tensor], cache: E97RecurrentCache) -> bool:
    after = list(tensors(cache.hidden))
    return len(before) == len(after) and all(
        torch.equal(left, right) for left, right in zip(before, after)
    )


def segmented_cache(loaded, token_ids: list[int], pattern: tuple[int, ...]) -> E97RecurrentCache:
    cache = None
    cursor = 0
    chunk_index = 0
    while cursor < len(token_ids):
        width = pattern[chunk_index % len(pattern)]
        stop = min(len(token_ids), cursor + width)
        cache = advance_e97_cache(loaded, token_ids[cursor:stop], cache)
        cursor = stop
        chunk_index += 1
    assert cache is not None
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--generation-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.world <= 0 or not 0 <= args.rank < args.world:
        raise ValueError("rank must be in [0, world)")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device="cuda",
        dtype=torch.bfloat16,
        weight_mode="saved",
        use_triton=True,
        mmap=True,
    )

    import tiktoken

    tokenizer = tiktoken.get_encoding(str(loaded.config["tokenizer"]))
    prompt = PROMPTS[args.rank % len(PROMPTS)]
    token_ids = tokenizer.encode(prompt, disallowed_special=())
    pattern = CHUNK_PATTERNS[args.rank % len(CHUNK_PATTERNS)]

    full_started = time.monotonic()
    full = advance_e97_cache(loaded, token_ids)
    full_seconds = time.monotonic() - full_started

    split_started = time.monotonic()
    split = segmented_cache(loaded, token_ids, pattern)
    split_seconds = time.monotonic() - split_started

    reset = advance_e97_cache(loaded, token_ids)
    hidden_diff = max_abs_difference(full.hidden, split.hidden)
    reset_hidden_diff = max_abs_difference(full.hidden, reset.hidden)
    logit_diff = float((full.next_logits.float() - split.next_logits.float()).abs().max().item())
    reset_logit_diff = float((full.next_logits.float() - reset.next_logits.float()).abs().max().item())
    full_argmax = int(full.next_logits.argmax().item())
    split_argmax = int(split.next_logits.argmax().item())

    committed_hidden = clone_hidden(split)
    committed_tokens = split.token_ids
    generated_full, shadow_full = generate_e97_from_cache(
        loaded,
        full,
        max_new_tokens=args.generation_tokens,
        temperature=0,
        top_k=0,
    )
    generated_split, shadow_split = generate_e97_from_cache(
        loaded,
        split,
        max_new_tokens=args.generation_tokens,
        temperature=0,
        top_k=0,
    )
    synthetic_stop = int(split.next_logits.argmax().item())
    stopped_tokens, stopped_shadow = generate_e97_from_cache(
        loaded,
        split,
        max_new_tokens=4,
        temperature=0,
        top_k=0,
        stop_token_ids=(synthetic_stop,),
    )

    post_generation_hidden_diff = max_abs_difference(
        shadow_full.hidden, shadow_split.hidden
    )
    post_generation_logit_diff = float(
        (shadow_full.next_logits.float() - shadow_split.next_logits.float())
        .abs().max().item()
    )
    post_generation_argmax_equal = int(shadow_full.next_logits.argmax().item()) == int(
        shadow_split.next_logits.argmax().item()
    )

    state_tensors = list(tensors(split.hidden))
    state_dtypes = sorted({str(tensor.dtype) for tensor in state_tensors})
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in state_tensors)
    append_suffix = e97_cache_suffix(split, [*token_ids, 17, 18])
    branch_rejected = e97_cache_suffix(split, [*token_ids[:-1], token_ids[-1] ^ 1]) is None
    truncation_rejected = e97_cache_suffix(split, token_ids[:-1]) is None
    transaction_preserved = (
        split.token_ids == committed_tokens and hidden_unchanged(committed_hidden, split)
    )
    stop_consumed = (
        stopped_tokens == [synthetic_stop]
        and stopped_shadow.token_ids[-1] == synthetic_stop
    )

    checks = {
        "token_prefix_equal": full.token_ids == split.token_ids == tuple(token_ids),
        "boundary_greedy_equal": full_argmax == split_argmax,
        "greedy_continuation_equal": generated_full == generated_split,
        "post_generation_argmax_equal": post_generation_argmax_equal,
        "reset_greedy_equal": int(reset.next_logits.argmax().item()) == full_argmax,
        "state_fp32": state_dtypes == ["torch.float32"],
        "state_finite": finite,
        "transaction_preserved": transaction_preserved,
        "stop_token_consumed": stop_consumed,
        "append_suffix_exact": append_suffix == (17, 18),
        "branch_rejected": branch_rejected,
        "truncation_rejected": truncation_rejected,
    }
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "schema": "emender-e97-recurrent-cache-qualification-v1",
        "status": status,
        "rank": args.rank,
        "world": args.world,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": loaded.step,
        "prompt_tokens": len(token_ids),
        "chunk_pattern": list(pattern),
        "generation_tokens": generated_split,
        "generation_text": tokenizer.decode(generated_split),
        "checks": checks,
        "measurements": {
            "hidden_max_abs_difference": hidden_diff,
            "next_logits_max_abs_difference": logit_diff,
            "post_generation_hidden_max_abs_difference": post_generation_hidden_diff,
            "post_generation_logits_max_abs_difference": post_generation_logit_diff,
            "reset_hidden_max_abs_difference": reset_hidden_diff,
            "reset_logits_max_abs_difference": reset_logit_diff,
            "full_boundary_argmax": full_argmax,
            "split_boundary_argmax": split_argmax,
            "recurrent_tensor_count": len(state_tensors),
            "recurrent_state_dtypes": state_dtypes,
            "cache_state_bytes_including_boundary_logits": split.state_bytes,
            "full_prefill_seconds": full_seconds,
            "segmented_prefill_seconds": split_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    if not all(math.isfinite(value) for value in (
        hidden_diff,
        logit_diff,
        post_generation_hidden_diff,
        post_generation_logit_diff,
        reset_hidden_diff,
        reset_logit_diff,
    )):
        result["status"] = "fail"
        result["checks"]["differences_finite"] = False
    else:
        result["checks"]["differences_finite"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
