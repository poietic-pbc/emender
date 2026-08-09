#!/usr/bin/env python3
"""Render one frozen paper arm into a train.py argv stream."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "e97-mlp": "e97_mlp.json",
    "e97-linear-mlp": "e97_linear_mlp.json",
    "gdn2-mlp": "gdn2_mlp.json",
}
VALUE_KEYS = (
    "level", "dim", "depth", "n_heads", "n_state", "n_slots", "n_groups",
    "state_expansion", "expansion", "mlp_ratio", "mlp_multiple", "use_gate",
    "gate_activation", "linear_state", "e88_raw_write", "e88_decay_mode",
    "use_triton", "use_chunked_e97", "e97_chunk_size", "use_conv", "d_conv",
    "gdn2_mlp_ratio", "optimizer", "lr", "weight_decay", "warmup_steps",
    "grad_clip", "batch_size", "chunk_size", "tokenizer", "seed", "diloco_k",
    "diloco_island_size", "diloco_outer_optimizer", "diloco_outer_lr",
    "diloco_outer_beta", "diloco_export_basis", "sampler_schema",
    "sampler_corpus_sha256", "sampler_tokenizer_sha256", "sampler_key",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=FILES, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.world_size <= 0 or args.steps <= 0:
        raise SystemExit("world-size and steps must be positive")
    config = json.loads((ROOT / "configs/frontier/e97_gdn2_paper" / FILES[args.arm]).read_text())
    result = ["--params", "100m", "--data", args.data,
              "--exact_output_dir", args.output, "--steps", str(args.steps),
              "--train_minutes", "0", "--save_every", str(args.save_every),
              "--keep_checkpoints", "3", "--bf16",
              "--sampler_data_world_size", str(args.world_size)]
    if args.world_size > 1:
        result += ["--diloco", "--diloco_merge_topology", "hierarchical",
                   "--diloco_merge_group_size", "8"]
    if args.resume:
        result += ["--resume", args.resume]
    for key in VALUE_KEYS:
        if key == "sampler_data_world_size" or key not in config:
            continue
        value = config[key]
        if key == "bf16":
            continue
        result += [f"--{key}", str(int(value) if isinstance(value, bool) else value)]
    sys.stdout.buffer.write(b"\0".join(item.encode() for item in result) + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
