#!/usr/bin/env python3
"""Fail-closed E97 35B MoE training runner for one eight-GCD node island."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist

from ndm.data.tokenized_dataset import TokenizedStreamDataset
from ndm.e97 import load_e97_checkpoint
from ndm.e97_moe_checkpoint import (
    load_node_sharded_checkpoint,
    save_node_sharded_checkpoint,
)
from ndm.e97_moe_ep import (
    assert_node_local_ep_group,
    average_replicated_gradients_,
    create_moe_process_groups,
    diloco_average_schedulefree_,
    node_replicated_parameters,
)
from ndm.e97_moe_optimizer import FusedScheduleFreeAdamW
from ndm.models.e97_moe import (
    E97MoEConfig,
    convert_e97_ffns_to_node_local_moe,
    e97_moe_auxiliary_loss,
)


DEFAULT_SEED = Path(
    "/lustre/orion/bif148/proj-shared/emender/frontier_runs/"
    "final-seed-production-256n/milestones/"
    "step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt")
DEFAULT_DATA = Path(
    "/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.007e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--diloco-k", type=int, default=40)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument(
        "--profile-phases", action="store_true",
        help="synchronize GPU events and emit forward/backward/reduction/optimizer timings")
    return parser.parse_args()


def emit(path: Path, event: str, **fields) -> None:
    record = {"event": event, "time_unix": time.time(), **fields}
    line = json.dumps(record, sort_keys=True)
    if dist.get_rank() == 0:
        print(line, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    if not args.seed_checkpoint.is_file() or not args.data.is_file():
        raise SystemExit("seed checkpoint or training data is unavailable")
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        groups = create_moe_process_groups()
        if groups.local_rank != local_rank:
            raise RuntimeError("Slurm rank ordering does not match contiguous node islands")
        topology = assert_node_local_ep_group(groups.node_group)
        torch.manual_seed(970035)
        torch.cuda.manual_seed_all(970035)
        source_commit = os.environ.get("EMENDER_SOURCE_COMMIT")
        if not source_commit:
            source_commit = os.popen("git rev-parse HEAD").read().strip()
        emit(args.log_jsonl, "load_start", commit=source_commit,
             seed_checkpoint=str(args.seed_checkpoint), hostname=topology.hostname)
        loaded = load_e97_checkpoint(
            args.seed_checkpoint, device="cuda", dtype=torch.bfloat16,
            weight_mode="train", use_triton=True, mmap=True)
        if loaded.step != 2322520 or int(loaded.config.get("dim", -1)) != 1792:
            raise RuntimeError("loaded checkpoint is not the bound final 513B E97 seed")
        model = loaded.model
        convert_e97_ffns_to_node_local_moe(
            model,
            E97MoEConfig(hidden_dim=8832, routed_experts=64, shared_experts=1,
                         top_k=3, expert_parallel_size=8),
            local_expert_rank=local_rank, expert_group=groups.node_group)
        model.train()
        parameter_count_local = sum(parameter.numel() for parameter in model.parameters())
        local_expert_count = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if name.endswith(("local_gate_weight", "local_up_weight", "local_down_weight")))
        if parameter_count_local != 5_750_016_656 or local_expert_count != 4_178_313_216:
            raise RuntimeError(
                f"packed shard count mismatch: total={parameter_count_local}, local={local_expert_count}")
        emit(args.log_jsonl, "model_ready", local_parameter_count=parameter_count_local,
             local_expert_parameter_count=local_expert_count,
             hbm_allocated=torch.cuda.memory_allocated(),
             hbm_reserved=torch.cuda.memory_reserved())

        dataset = TokenizedStreamDataset(
            data_path=str(args.data), chunk_size=args.chunk_size + 1,
            rank=dist.get_rank(), world_size=dist.get_world_size(),
            seed=42, tokenizer_name="p50k_base")
        optimizer = FusedScheduleFreeAdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), warmup_steps=0)
        if args.resume_root is not None:
            manifest = load_node_sharded_checkpoint(
                args.resume_root / f"node-{groups.node_index}",
                model, optimizer, node_group=groups.node_group)
            emit(args.log_jsonl, "restart_loaded", checkpoint_step=manifest["step"],
                 checkpoint_tokens=manifest["accepted_tokens"])
        optimizer.train()
        replicated = tuple(node_replicated_parameters(model))
        start = None
        step = 0
        accepted_tokens = 0
        while True:
            if step >= args.max_steps and args.max_steps > 0:
                break
            if start is not None and args.minutes > 0 and time.monotonic() - start >= args.minutes * 60:
                break
            optimizer.zero_grad(set_to_none=True)
            chunks, _, actual_lengths = dataset.get_batch(
                args.batch_size, device=torch.device("cuda"))
            step_start = time.monotonic()
            phase_events = None
            if args.profile_phases:
                phase_events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in ("start", "forward", "backward", "reduce", "optimizer", "merge")
                }
                phase_events["start"].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(chunks, return_loss=True)
            auxiliary = e97_moe_auxiliary_loss(model)
            if phase_events is not None:
                phase_events["forward"].record()
            objective = loss + auxiliary
            if not torch.isfinite(objective):
                raise FloatingPointError(f"nonfinite objective at step {step}")
            objective.backward()
            if phase_events is not None:
                phase_events["backward"].record()
            replicated_ids = {id(parameter) for parameter in replicated}
            missing_replicated = [
                name for name, parameter in model.named_parameters()
                if id(parameter) in replicated_ids and parameter.grad is None]
            if missing_replicated:
                emit(args.log_jsonl, "missing_replicated_gradients",
                     names=missing_replicated)
                raise RuntimeError(
                    f"replicated parameters missing gradients: {missing_replicated}")
            average_replicated_gradients_(
                replicated, group=groups.node_group, topology=topology)
            if phase_events is not None:
                phase_events["reduce"].record()
            optimizer.step()
            optimizer.assert_no_master_weights()
            if phase_events is not None:
                phase_events["optimizer"].record()
            merge_seconds = 0.0
            if groups.node_count > 1 and (step + 1) % args.diloco_k == 0:
                # Release inactive variable-routing blocks before RCCL allocates its
                # cross-node collective workspace.  At 32 nodes, one lane rank can
                # otherwise retain enough allocator cache to starve RCCL despite
                # live tensors fitting comfortably in HBM.
                torch.cuda.empty_cache()
                dist.barrier(group=groups.diloco_lane_group)
                merge_start = time.monotonic()
                diloco_average_schedulefree_(
                    model, optimizer, lane_group=groups.diloco_lane_group)
                merge_seconds = time.monotonic() - merge_start
            if phase_events is not None:
                phase_events["merge"].record()
                phase_events["merge"].synchronize()
                emit(args.log_jsonl, "phase_profile", step=step,
                     forward_ms=phase_events["start"].elapsed_time(phase_events["forward"]),
                     backward_ms=phase_events["forward"].elapsed_time(phase_events["backward"]),
                     replicated_reduce_ms=phase_events["backward"].elapsed_time(phase_events["reduce"]),
                     optimizer_ms=phase_events["reduce"].elapsed_time(phase_events["optimizer"]),
                     merge_ms=phase_events["optimizer"].elapsed_time(phase_events["merge"]))
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=groups.node_group)
            dist.all_reduce(auxiliary, op=dist.ReduceOp.AVG, group=groups.node_group)
            step_seconds = time.monotonic() - step_start
            tokens = int(actual_lengths.sum().item()) - args.batch_size
            accepted_tokens += tokens * dist.get_world_size()
            emit(args.log_jsonl, "step", step=step, loss=float(loss.item()),
                 auxiliary_loss=float(auxiliary.item()), step_seconds=step_seconds,
                 diloco_merge_seconds=merge_seconds, node_count=groups.node_count,
                 accepted_tokens=accepted_tokens,
                 tokens_per_second=(tokens * dist.get_world_size() / step_seconds),
                 hbm_allocated=torch.cuda.memory_allocated(),
                 hbm_reserved=torch.cuda.memory_reserved(),
                 max_hbm_allocated=torch.cuda.max_memory_allocated())
            if start is None:
                start = time.monotonic()
            step += 1
        measured_seconds = 0.0 if start is None else time.monotonic() - start
        emit(args.log_jsonl, "complete", steps=step, accepted_tokens=accepted_tokens,
             measured_training_seconds=measured_seconds,
             hbm_allocated=torch.cuda.memory_allocated(),
             max_hbm_allocated=torch.cuda.max_memory_allocated())
        if args.checkpoint_root is not None:
            checkpoint = save_node_sharded_checkpoint(
                args.checkpoint_root / f"node-{groups.node_index}",
                model, optimizer, step=step, accepted_tokens=accepted_tokens,
                source_commit=source_commit, node_group=groups.node_group)
            emit(args.log_jsonl, "checkpoint_complete", checkpoint_path=str(checkpoint), step=step,
                 accepted_tokens=accepted_tokens)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
