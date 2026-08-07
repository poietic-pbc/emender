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
    parser.add_argument(
        "--expert-backend", choices=("triton", "rocblas", "grouped"), default="rocblas")
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--empty-cache-interval", type=int, default=0)
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


def _canonical_checkpoint(args, groups, model, optimizer, *, step, accepted_tokens,
                          source_commit):
    """Publish one canonical eight-rank island, coordinated by global rank zero."""
    dist.barrier()
    if groups.node_index == 0:
        save_node_sharded_checkpoint(
            args.checkpoint_root, model, optimizer, step=step,
            accepted_tokens=accepted_tokens, source_commit=source_commit,
            node_group=groups.node_group, keep_generations=args.keep_checkpoints)
    dist.barrier()
    generation = (args.checkpoint_root / "latest").resolve(strict=True)
    manifest = json.loads((generation / "manifest.json").read_text())
    if (manifest.get("complete") is not True or int(manifest.get("step", -1)) != step
            or int(manifest.get("accepted_tokens", -1)) != accepted_tokens):
        raise RuntimeError("canonical checkpoint authority does not match the completed step")
    return generation


def main() -> None:
    args = parse_args()
    if not args.seed_checkpoint.is_file() or not args.data.is_file():
        raise SystemExit("seed checkpoint or training data is unavailable")
    if args.save_every < 0 or args.keep_checkpoints < 1:
        raise SystemExit("save-every must be nonnegative and keep-checkpoints must be positive")
    if args.save_every and args.checkpoint_root is None:
        raise SystemExit("save-every requires checkpoint-root")
    if args.save_every and args.save_every % args.diloco_k:
        raise SystemExit("save-every must be aligned to completed DiLoCo K boundaries")
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
                         top_k=3, expert_parallel_size=8,
                         expert_backend=args.expert_backend),
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

        optimizer = FusedScheduleFreeAdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), warmup_steps=0)
        starting_step = int(loaded.step)
        accepted_tokens = 0
        if args.resume_root is not None:
            manifest = load_node_sharded_checkpoint(
                args.resume_root, model, optimizer, node_group=groups.node_group)
            starting_step = int(manifest["step"])
            accepted_tokens = int(manifest["accepted_tokens"])
            emit(args.log_jsonl, "restart_loaded", checkpoint_step=starting_step,
                 checkpoint_tokens=accepted_tokens)
        # A fresh/resumed execution gets a different deterministic random stream:
        # requested offset = source/checkpoint starting step + 42 + global rank.
        # TokenizedStreamDataset adds rank internally to this base seed.
        data_seed_base = 42 + starting_step
        dataset = TokenizedStreamDataset(
            data_path=str(args.data), chunk_size=args.chunk_size + 1,
            rank=dist.get_rank(), world_size=dist.get_world_size(),
            seed=data_seed_base, tokenizer_name="p50k_base")
        emit(args.log_jsonl, "data_stream_ready", starting_step=starting_step,
             seed_base=data_seed_base,
             rank_seed=data_seed_base + dist.get_rank(), replay_previous_launch=False)
        optimizer.train()
        replicated = tuple(node_replicated_parameters(model))
        start = None
        step = starting_step
        completed_steps = 0
        last_checkpoint_step = None
        while True:
            if completed_steps >= args.max_steps and args.max_steps > 0:
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
            if (args.empty_cache_interval > 0 and
                    (step + 1) % args.empty_cache_interval == 0):
                # Variable dropless routing changes ragged GEMM workspace sizes.
                # Trim only at a completed step boundary where activations are
                # dead; this prevents inactive slabs from starving the next
                # step without changing model or optimizer state.
                torch.cuda.empty_cache()
            if start is None:
                start = time.monotonic()
            step += 1
            completed_steps += 1
            if (args.checkpoint_root is not None and args.save_every > 0
                    and step % args.save_every == 0):
                checkpoint_start = time.monotonic()
                checkpoint = _canonical_checkpoint(
                    args, groups, model, optimizer, step=step,
                    accepted_tokens=accepted_tokens, source_commit=source_commit)
                last_checkpoint_step = step
                emit(args.log_jsonl, "checkpoint_complete",
                     checkpoint_path=str(checkpoint), step=step,
                     accepted_tokens=accepted_tokens,
                     checkpoint_seconds=time.monotonic() - checkpoint_start,
                     canonical_node=0)
        measured_seconds = 0.0 if start is None else time.monotonic() - start
        emit(args.log_jsonl, "complete", step=step, steps_completed=completed_steps,
             accepted_tokens=accepted_tokens,
             measured_training_seconds=measured_seconds,
             hbm_allocated=torch.cuda.memory_allocated(),
             max_hbm_allocated=torch.cuda.max_memory_allocated())
        if args.checkpoint_root is not None and last_checkpoint_step != step:
            checkpoint_start = time.monotonic()
            checkpoint = _canonical_checkpoint(
                args, groups, model, optimizer, step=step,
                accepted_tokens=accepted_tokens, source_commit=source_commit)
            emit(args.log_jsonl, "checkpoint_complete",
                 checkpoint_path=str(checkpoint), step=step,
                 accepted_tokens=accepted_tokens,
                 checkpoint_seconds=time.monotonic() - checkpoint_start,
                 canonical_node=0)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
