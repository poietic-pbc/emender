#!/usr/bin/env python3
"""Real-checkpoint cached recurrent decoding versus full-prefix recomputation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tiktoken

from ndm.e97 import load_e97_checkpoint
from ndm.e97_moe_checkpoint import load_node_sharded_model
from ndm.e97_moe_ep import assert_node_local_ep_group, create_moe_process_groups
from ndm.models.e97_moe import E97MoEConfig, convert_e97_ffns_to_node_local_moe


SCHEMA = "emender-e97-moe-decode-parity-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-args-json", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--expert-backend", choices=("rocblas", "triton", "grouped"), default="rocblas")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_tokens(values, device):
    return torch.tensor(values, device=device, dtype=torch.long).unsqueeze(0)


def compare_logits(cached, recomputed, one_shot, target):
    cached = cached.float()
    recomputed = recomputed.float()
    one_shot = one_shot.float()
    cached_logp = F.log_softmax(cached, dim=-1)[target]
    recomputed_logp = F.log_softmax(recomputed, dim=-1)[target]
    return {
        "cached_recompute_max_abs": float((cached - recomputed).abs().max().item()),
        "cached_oneshot_max_abs": float((cached - one_shot).abs().max().item()),
        "cached_recompute_top1_equal": bool(cached.argmax() == recomputed.argmax()),
        "cached_oneshot_top1_equal": bool(cached.argmax() == one_shot.argmax()),
        "target_logp_abs": float((cached_logp - recomputed_logp).abs().item()),
    }


def teacher_forced_parity(model, prompt, response, device):
    sequence = prompt + response
    with torch.inference_mode():
        one_shot, _state = model(local_tokens(sequence, device), return_prev_hiddens=True)
        prompt_logits, (hiddens, _conv) = model(
            local_tokens(prompt, device), return_prev_hiddens=True)
    cached = prompt_logits[0, -1]
    rows = []
    for index, token in enumerate(response):
        prefix = prompt + response[:index]
        with torch.inference_mode():
            recomputed, _state = model(
                local_tokens(prefix, device), return_prev_hiddens=True)
        row = compare_logits(
            cached, recomputed[0, -1], one_shot[0, len(prompt) - 1 + index], token)
        row["index"] = index
        rows.append(row)
        if index + 1 < len(response):
            with torch.inference_mode():
                next_logits, (hiddens, _conv) = model(
                    local_tokens([token], device), return_prev_hiddens=True,
                    prev_hiddens=hiddens)
            cached = next_logits[0, -1]
    return rows


def greedy_cached(model, prompt, count, device):
    with torch.inference_mode():
        logits, (hiddens, _conv) = model(
            local_tokens(prompt, device), return_prev_hiddens=True)
    result = []
    token = int(logits[0, -1].argmax().item())
    for index in range(count):
        result.append(token)
        if index + 1 < count:
            with torch.inference_mode():
                logits, (hiddens, _conv) = model(
                    local_tokens([token], device), return_prev_hiddens=True,
                    prev_hiddens=hiddens)
            token = int(logits[0, -1].argmax().item())
    return result


def greedy_recomputed(model, prompt, count, device):
    result = []
    for _ in range(count):
        with torch.inference_mode():
            logits, _state = model(
                local_tokens(prompt + result, device), return_prev_hiddens=True)
        result.append(int(logits[0, -1].argmax().item()))
    return result


def main():
    args = parse_args()
    if args.tokens <= 0:
        raise SystemExit("tokens must be positive")
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        if dist.get_world_size() != 8:
            raise RuntimeError("decode parity requires one complete eight-rank node")
        groups = create_moe_process_groups()
        if groups.local_rank != local_rank or groups.node_count != 1:
            raise RuntimeError("decode parity world is not one contiguous node")
        topology = assert_node_local_ep_group(groups.node_group)
        panel_sha = sha256(args.panel)
        panel = torch.load(args.panel, map_location="cpu", weights_only=False)
        if panel.get("schema") != "emender-e97-moe-paired-eval-panel-v2":
            raise RuntimeError("unexpected paired panel schema")
        loaded = load_e97_checkpoint(
            args.seed_checkpoint, device="cuda", dtype=torch.bfloat16,
            weight_mode="train", use_triton=True, mmap=True,
            args_json=args.seed_args_json)
        if loaded.step != 2322520 or int(loaded.config.get("dim", -1)) != 1792:
            raise RuntimeError("dense seed is not the bound E97 authority")
        model = loaded.model
        model.gradient_checkpointing = False
        model.loss_chunk_size = 0
        for module in model.modules():
            if hasattr(module, "checkpoint_interval"):
                module.checkpoint_interval = 16
                module.projection_chunk_size = 2048
        convert_e97_ffns_to_node_local_moe(
            model,
            E97MoEConfig(hidden_dim=8832, routed_experts=64, shared_experts=1,
                         top_k=3, expert_parallel_size=8,
                         expert_backend=args.expert_backend),
            local_expert_rank=local_rank, expert_group=groups.node_group)
        model.eval()
        manifest = load_node_sharded_model(
            args.generation, model, node_group=groups.node_group,
            verify_sha256=True)
        encoding = tiktoken.get_encoding(panel["tokenizer"])
        rank = dist.get_rank()
        likelihood = panel["assistant_response_likelihood"][rank]
        prompt = encoding.encode_ordinary(likelihood["prompt"])
        response = encoding.encode_ordinary(likelihood["response"])[:args.tokens]
        if not prompt or not response:
            raise RuntimeError("decode parity example tokenization is empty")
        rows = teacher_forced_parity(model, prompt, response, torch.device("cuda"))
        generation_prompt = encoding.encode_ordinary(panel["generation_prompts"][rank]["prompt"])
        cached_generation = greedy_cached(model, generation_prompt, args.tokens, torch.device("cuda"))
        recomputed_generation = greedy_recomputed(
            model, generation_prompt, args.tokens, torch.device("cuda"))
        local = {
            "rank": rank, "example_id": likelihood["id"], "tokens": len(response),
            "teacher_forced": rows,
            "greedy_equal": cached_generation == recomputed_generation,
            "cached_generation": cached_generation,
            "recomputed_generation": recomputed_generation,
        }
        gathered = [None] * 8
        dist.all_gather_object(gathered, local)
        if rank == 0:
            comparisons = [row for example in gathered for row in example["teacher_forced"]]
            result = {
                "schema": SCHEMA,
                "panel_sha256": panel_sha,
                "checkpoint": {
                    "generation": str(args.generation.resolve()),
                    "step": int(manifest["step"]),
                    "accepted_tokens": int(manifest["accepted_tokens"]),
                    "source_commit": manifest["source_commit"],
                },
                "examples": gathered,
                "summary": {
                    "comparisons": len(comparisons),
                    "cached_recompute_max_abs": max(row["cached_recompute_max_abs"] for row in comparisons),
                    "cached_oneshot_max_abs": max(row["cached_oneshot_max_abs"] for row in comparisons),
                    "target_logp_max_abs": max(row["target_logp_abs"] for row in comparisons),
                    "cached_recompute_top1_fraction": sum(row["cached_recompute_top1_equal"] for row in comparisons) / len(comparisons),
                    "cached_oneshot_top1_fraction": sum(row["cached_oneshot_top1_equal"] for row in comparisons) / len(comparisons),
                    "greedy_exact_fraction": sum(example["greedy_equal"] for example in gathered) / len(gathered),
                },
                "peak_hbm_bytes": int(torch.cuda.max_memory_allocated()),
                "hostname": topology.hostname,
            }
            summary = result["summary"]
            result["pass"] = bool(
                summary["cached_recompute_top1_fraction"] == 1.0
                and summary["cached_oneshot_top1_fraction"] == 1.0
                and summary["greedy_exact_fraction"] == 1.0
                and summary["target_logp_max_abs"] <= 0.02)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"event": "decode_parity_complete", "time_unix": time.time(),
                              "output": str(args.output), "pass": result["pass"],
                              **summary}, sort_keys=True), flush=True)
        passed = torch.zeros(1, device="cuda", dtype=torch.int32)
        if rank == 0:
            passed.fill_(int(result["pass"]))
        dist.broadcast(passed, src=0)
        if int(passed.item()) != 1:
            raise RuntimeError("cached recurrent decoding parity failed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
