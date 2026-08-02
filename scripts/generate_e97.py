#!/usr/bin/env python3
"""Load an E97/Emender training checkpoint and generate text.

The checkpoint may be a ``.pt`` file, ``latest.pt``, or a run directory.  Its
training ``args.json`` is discovered beside the checkpoint unless supplied with
``--args-json``.  Schedule-free checkpoints default to their recovered y/train
weights, which are the usable generation weights for the 150B-token run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ndm.e97 import generate_e97, load_e97_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="E97 checkpoint .pt, latest.pt symlink, or run directory",
    )
    parser.add_argument(
        "--args-json",
        default=None,
        help="Training args.json when it is not beside the checkpoint",
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Prompt text")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--stop-token-id", type=int, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["train", "saved"],
        default="train",
        help="Recover schedule-free y/train weights (default) or use stored x weights",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full-context", "stateful"],
        default="auto",
        help=(
            "auto uses fused full-context generation for Triton checkpoints; "
            "stateful loads the exact eager recurrence"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text()

    # Stateful generation carries the exact recurrent state one token at a time.
    # Until the E97 single-token fused kernel is parity-cleared, that mode uses
    # the reference recurrence. Full-context mode safely keeps Triton enabled.
    use_triton = False if args.mode == "stateful" else None
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device=args.device,
        dtype=args.dtype,
        weight_mode=args.weight_mode,
        use_triton=use_triton,
    )
    print(
        f"loaded E97 step={loaded.step} loss={loaded.loss} "
        f"weight_mode={loaded.weight_mode} checkpoint={loaded.checkpoint_path}",
        file=sys.stderr,
        flush=True,
    )
    result = generate_e97(
        loaded,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_context=args.max_context,
        mode=args.mode,
        stop_token_ids=args.stop_token_id,
        seed=args.seed,
    )
    print(result["text"])
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
