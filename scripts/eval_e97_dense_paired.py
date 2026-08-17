#!/usr/bin/env python3
"""Read-only paired diagnostics for the immutable dense E97 final seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import tiktoken

from ndm.e97 import load_e97_checkpoint
from scripts.eval_e97_moe_paired import (
    aggregate,
    event,
    generate,
    score_assistant_responses,
    score_hellaswag,
    score_mmlu,
    score_retrieval,
    score_wikitext,
    sha256,
)

SCHEMA = "emender-e97-dense-paired-eval-result-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="dense-513b")
    parser.add_argument("--weight-mode", choices=("train", "saved"), default="train")
    parser.add_argument("--generation-tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        if dist.get_world_size() != 8:
            raise RuntimeError("dense paired evaluation requires eight ranks")
        if sha256(args.checkpoint) != args.checkpoint_sha256:
            raise RuntimeError("dense checkpoint SHA-256 mismatch")
        if sha256(args.panel) != args.panel_sha256:
            raise RuntimeError("paired panel SHA-256 mismatch")
        panel = torch.load(args.panel, map_location="cpu", weights_only=False)
        if panel.get("schema") != "emender-e97-moe-paired-eval-panel-v2":
            raise RuntimeError("unexpected paired panel schema")
        event("load_start", label=args.label, checkpoint=str(args.checkpoint))
        loaded = load_e97_checkpoint(
            args.checkpoint, device="cuda", dtype=torch.bfloat16,
            weight_mode=args.weight_mode, use_triton=True, mmap=True,
            args_json=args.args_json)
        if (loaded.step != 2322520
                or int(loaded.config.get("dim", -1)) != 1792
                or int(loaded.config.get("depth", -1)) != 11):
            raise RuntimeError("dense graph is not the bound final E97 authority")
        model = loaded.model
        model.gradient_checkpointing = False
        model.loss_chunk_size = 2048
        for module in model.modules():
            if hasattr(module, "checkpoint_interval"):
                module.checkpoint_interval = 16
                module.projection_chunk_size = 2048
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        encoding = tiktoken.get_encoding(panel["tokenizer"])
        routing = []
        result = {
            "schema": SCHEMA,
            "label": args.label,
            "panel_sha256": args.panel_sha256,
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": args.checkpoint_sha256,
                "step": loaded.step,
                "weight_mode": loaded.weight_mode,
                "schedulefree_train_weight_swap": loaded.schedulefree_train_weight_swap,
                "tokens": int(loaded.checkpoint_metadata.get(
                    "total_tokens", 513013841920)),
            },
            "wikitext": score_wikitext(model, panel, torch.device("cuda"), routing),
            "mmlu": score_mmlu(model, panel, encoding, torch.device("cuda"), routing),
            "hellaswag": score_hellaswag(model, panel, encoding, torch.device("cuda"), routing),
            "assistant_responses": score_assistant_responses(
                model, panel, encoding, torch.device("cuda"), routing),
            "retrieval": score_retrieval(model, panel, encoding, torch.device("cuda"), routing),
            "generations": (
                generate(model, panel, encoding, torch.device("cuda"), routing,
                         args.generation_tokens, mode="greedy")
                + generate(model, panel, encoding, torch.device("cuda"), routing,
                           args.generation_tokens, mode="sample")),
        }
        if dist.get_rank() == 0:
            result["peak_hbm_bytes"] = int(torch.cuda.max_memory_allocated())
            aggregate(result)
            predictions = [0, 0, 0, 0]
            answers = [0, 0, 0, 0]
            for row in result["mmlu"]:
                predictions[int(row["prediction"])] += 1
                answers[int(row["answer"])] += 1
            result["mmlu_diagnostics"] = {
                "prediction_counts": predictions,
                "answer_counts": answers,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            event("evaluation_complete", output=str(args.output),
                  summary=result["summary"],
                  mmlu_diagnostics=result["mmlu_diagnostics"])
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
