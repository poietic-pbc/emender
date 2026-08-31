#!/usr/bin/env python3
"""Validate one raw E97 4B Pi SFT checkpoint without materializing tensor storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ndm.data.masked_sft_dataset import sha256
from scripts.train_e97_4b_pi_sft import EXPECTED_PARAMETERS, SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    digest = sha256(args.checkpoint)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise SystemExit("checkpoint SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise SystemExit("checkpoint schema mismatch")
    required = {
        "model_state_dict", "optimizer_state_dict", "sft_updates", "sft_total_tokens",
        "assistant_target_tokens", "parent_checkpoint_sha256", "authority_manifest_sha256",
        "pack_manifest_sha256", "data_world_size", "context_size", "island_size",
        "diloco_k", "source_commit", "weight_mode",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise SystemExit(f"checkpoint missing fields: {missing}")
    parameters = sum(tensor.numel() for tensor in checkpoint["model_state_dict"].values())
    if parameters != EXPECTED_PARAMETERS:
        raise SystemExit(f"parameter count mismatch: {parameters}")
    if checkpoint["weight_mode"] != "saved-eval-x":
        raise SystemExit("checkpoint weight mode mismatch")
    if int(checkpoint["assistant_target_tokens"]) <= 0:
        raise SystemExit("checkpoint has no assistant targets")
    receipt = {
        "schema": "emender-e97-4b-pi-sft-reload-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": digest,
        "mmap_load": "passed",
        "parameters": parameters,
        "sft_updates": int(checkpoint["sft_updates"]),
        "sft_total_tokens": int(checkpoint["sft_total_tokens"]),
        "assistant_target_tokens": int(checkpoint["assistant_target_tokens"]),
        "world_size": int(checkpoint["data_world_size"]),
        "context_size": int(checkpoint["context_size"]),
        "parent_checkpoint_sha256": checkpoint["parent_checkpoint_sha256"],
        "authority_manifest_sha256": checkpoint["authority_manifest_sha256"],
        "pack_manifest_sha256": checkpoint["pack_manifest_sha256"],
        "source_commit": checkpoint["source_commit"],
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
