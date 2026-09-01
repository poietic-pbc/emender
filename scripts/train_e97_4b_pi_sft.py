#!/usr/bin/env python3
"""Fixed-world full-parameter masked SFT for the E97 4B Pi agent lineage."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from schedulefree import AdamWScheduleFree

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity, sha256
from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree
from ndm.e97 import load_e97_checkpoint
from train import diloco_merge

SCHEMA = "emender-e97-4b-pi-masked-sft-v1"
EXPECTED_PARAMETERS = 4_045_972_080


def emit(path: Path, event: str, rank: int, **values) -> None:
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"event": event, "time_unix": time.time(), **values}, sort_keys=True) + "\n")


def make_island_group(world_size: int, rank: int, island_size: int):
    if world_size % island_size:
        raise RuntimeError("world size must be divisible by island size")
    selected = None
    for first in range(0, world_size, island_size):
        ranks = list(range(first, first + island_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            selected = group
        if rank in ranks:
            warmup = torch.zeros(1, device="cuda")
            dist.all_reduce(warmup, group=group)
        dist.barrier()
    if selected is None:
        raise RuntimeError("rank has no DDP island")
    return selected


def objective(model, tokens, masks, length: int, spans, island_targets: torch.Tensor,
              island_size: int) -> tuple[torch.Tensor, int]:
    total = torch.zeros((), device=tokens.device, dtype=torch.float32)
    observed = 0
    for span_index, (start, stop) in enumerate(spans):
        real = stop - start
        if real < 2:
            raise RuntimeError("masked-SFT record is too short")
        padded = ((real - 2) // 16 + 1) * 16 + 1
        inputs = torch.zeros((1, padded), device=tokens.device, dtype=torch.long)
        inputs[:, :real] = tokens[:, start:stop]
        target_mask = torch.zeros((1, padded - 1), device=tokens.device, dtype=torch.bool)
        target_mask[:, :real - 1] = masks[:, start + 1:stop]
        observed += int(target_mask.sum())
        sync_context = model.no_sync() if span_index + 1 < len(spans) else nullcontext()
        with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            part = model(
                inputs,
                return_loss=True,
                actual_length=torch.tensor([real], device=inputs.device),
                loss_mask=target_mask,
                loss_reduction="sum",
            )
            # DDP averages gradients over the island. Multiplying by island_size
            # yields the exact target-token-normalized B8 island gradient.
            scaled = part * (island_size / island_targets.to(torch.float32))
        scaled.backward()
        total += part.detach().float()
    expected = int(masks[:, 1:length].sum())
    if observed != expected:
        raise RuntimeError(f"assistant target accounting mismatch: {observed} != {expected}")
    return total, observed


def merge_args(bucket_numel: int):
    return SimpleNamespace(
        optimizer="schedulefree",
        diloco_outer_optimizer="avg",
        diloco_export_basis="x",
        diloco_outer_lr=1.0,
        diloco_outer_beta=0.0,
        diloco_merge_bucket_numel=bucket_numel,
        diloco_merge_topology="global",
        diloco_merge_completion_barrier=1,
        _diloco_merge_groups=None,
        diloco_merge_debug=0,
        diloco_merge_debug_ranks="0",
    )


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    latest = path.parent / "latest.pt"
    temporary_link = path.parent / f".latest.pt.{os.getpid()}.tmp"
    try:
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(path.name)
        os.replace(temporary_link, latest)
    finally:
        temporary_link.unlink(missing_ok=True)


def build_optimizer(parameters, args):
    common = dict(
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps)
    if args.offload_schedulefree_state:
        return CPUOffloadAdamWScheduleFree(
            parameters, **common,
            pin_memory=bool(args.schedulefree_offload_pin_memory),
            release_gradients=bool(args.schedulefree_offload_release_gradients),
            bucket_numel=args.schedulefree_offload_bucket_numel)
    return AdamWScheduleFree(parameters, **common)


def load_resume_optimizer(path: Path, optimizer, expected: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("schema") != SCHEMA or "optimizer_state_dict" not in checkpoint:
        raise RuntimeError("resume is not an E97 4B Pi SFT optimizer checkpoint")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume identity mismatch: {key}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    metadata = {
        "updates": int(checkpoint["sft_updates"]),
        "total_tokens": int(checkpoint["sft_total_tokens"]),
        "assistant_target_tokens": int(checkpoint["assistant_target_tokens"]),
    }
    del checkpoint
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--source-args-json", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    lineage = parser.add_mutually_exclusive_group()
    lineage.add_argument("--resume", type=Path)
    lineage.add_argument("--new-stage-from", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--island-size", type=int, default=8)
    parser.add_argument("--diloco-k", type=int, default=8)
    parser.add_argument("--merge-bucket-numel", type=int, default=67_108_864)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--sampler-key", type=int, default=974003)
    parser.add_argument("--offload-schedulefree-state", action="store_true")
    parser.add_argument("--schedulefree-offload-bucket-numel", type=int, default=67_108_864)
    parser.add_argument("--schedulefree-offload-pin-memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--schedulefree-offload-release-gradients", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if (args.new_stage_from is not None
            and args.new_stage_from.resolve() != args.parent_checkpoint.resolve()):
        raise SystemExit("new-stage-from must equal the hash-bound parent checkpoint")
    if args.steps <= 0 or args.save_every <= 0 or args.diloco_k <= 0:
        raise SystemExit("steps/save-every/diloco-k must be positive")
    if args.save_every % args.diloco_k or args.steps % args.diloco_k:
        raise SystemExit("save and terminal steps must be K-aligned")
    if (args.keep_checkpoints <= 0 or args.merge_bucket_numel <= 0
            or args.schedulefree_offload_bucket_numel <= 0):
        raise SystemExit("checkpoint retention and optimizer/merge buckets must be positive")

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world not in {8, 64} or args.island_size != 8:
        raise RuntimeError("qualified Pi SFT worlds are 8 or 64 ranks in eight-rank islands")
    if sha256(args.authority_root / "manifest.json") != args.authority_sha256:
        raise RuntimeError("masked-SFT authority manifest mismatch")
    if sha256(args.pack_root / "manifest.json") != args.pack_sha256:
        raise RuntimeError("masked-SFT pack manifest mismatch")

    load_path = args.resume or args.new_stage_from or args.parent_checkpoint
    weight_mode = "saved" if (args.resume is not None or args.new_stage_from is not None) else "train"
    loaded = load_e97_checkpoint(
        load_path, args_json=args.source_args_json, device=device,
        dtype=torch.bfloat16, weight_mode=weight_mode, use_triton=True, mmap=True)
    core_model = loaded.model.train()
    core_model.gradient_checkpointing = True
    core_model.gradient_checkpoint_group_size = 1
    parameter_count = sum(parameter.numel() for parameter in core_model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"E97 4B parameter mismatch: {parameter_count}")

    island_group = make_island_group(world, rank, args.island_size)
    model = DDP(
        core_model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False, gradient_as_bucket_view=True,
        process_group=island_group)
    optimizer = build_optimizer(core_model.parameters(), args)

    optimizer_state_storage = (
        CPUOffloadAdamWScheduleFree.state_storage
        if args.offload_schedulefree_state else "accelerator")
    expected_resume = {
        "parent_checkpoint_sha256": args.parent_sha256,
        "authority_manifest_sha256": args.authority_sha256,
        "pack_manifest_sha256": args.pack_sha256,
        "data_world_size": world,
        "context_size": args.context_size,
        "island_size": args.island_size,
        "diloco_k": args.diloco_k,
        "optimizer_state_storage": optimizer_state_storage,
    }
    if args.resume is not None:
        clocks = load_resume_optimizer(args.resume, optimizer, expected_resume)
        start_update = clocks["updates"]
        total_tokens = clocks["total_tokens"]
        total_targets = clocks["assistant_target_tokens"]
    else:
        start_update = total_tokens = total_targets = 0
        if args.offload_schedulefree_state:
            initialization = optimizer.initialize_state_()
            emit(args.log_jsonl, "optimizer_state_initialized", rank,
                 storage=optimizer_state_storage, **initialization)
    if args.offload_schedulefree_state:
        optimizer.assert_state_offloaded()
    if start_update > args.steps or start_update % args.diloco_k:
        raise RuntimeError("resume update clock is invalid")
    optimizer.train()

    identity = SFTSamplerIdentity(
        authority_manifest_sha256=args.authority_sha256,
        pack_manifest_sha256=args.pack_sha256,
        sampler_key=args.sampler_key,
        data_world_size=world,
        context_size=args.context_size,
    )
    data = MaskedSFTPackedDataset(
        args.authority_root, args.pack_root, identity=identity, rank=rank,
        initial_absolute_rank_sample_index=start_update,
        verify_payload_hashes=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    emit(args.log_jsonl, "start", rank,
         parent_checkpoint=str(args.parent_checkpoint), parent_sha256=args.parent_sha256,
         source_weight_mode=("resume-saved-plus-optimizer" if args.resume else
                             "new-stage-saved-x" if args.new_stage_from else
                             "parent-train-y"),
         source_commit=args.source_commit, world_size=world, island_size=args.island_size,
         diloco_k=args.diloco_k, context_size=args.context_size, lr=args.lr,
         warmup_steps=args.warmup_steps, total_parameters=parameter_count,
         optimizer_state_storage=optimizer_state_storage,
         optimizer_state_bucket_numel=(args.schedulefree_offload_bucket_numel
                                        if args.offload_schedulefree_state else None),
         start_update=start_update)

    merge_configuration = merge_args(args.merge_bucket_numel)
    recent_losses: list[float] = []
    for update in range(start_update + 1, args.steps + 1):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        tokens, masks, lengths, target_counts, spans = data.get_batch_with_record_spans(1, device=device)
        island_targets = target_counts.sum().to(torch.int64)
        dist.all_reduce(island_targets, op=dist.ReduceOp.SUM, group=island_group)
        if int(island_targets) <= 0:
            raise RuntimeError("island sampled no assistant targets")
        local_loss, _ = objective(
            model, tokens, masks, int(lengths[0]), spans[0], island_targets,
            args.island_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("nonfinite SFT gradient norm")
        optimizer.step()

        merge_seconds = 0.0
        if update % args.diloco_k == 0:
            merge_seconds = diloco_merge(
                core_model, optimizer, merge_configuration, world, None,
                step=update, merge_index=update // args.diloco_k)

        counts = torch.tensor([int(lengths.sum()), int(target_counts.sum())], device=device, dtype=torch.int64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        loss_sum = local_loss.clone()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        step_loss = float(loss_sum / counts[1])
        recent_losses.append(step_loss)
        recent_losses = recent_losses[-100:]
        total_tokens += int(counts[0])
        total_targets += int(counts[1])
        emit(args.log_jsonl, "step", rank, update=update, loss=step_loss,
             global_tokens=int(counts[0]), global_targets=int(counts[1]),
             total_tokens=total_tokens, total_targets=total_targets,
             grad_norm=float(grad_norm), merge_seconds=merge_seconds,
             step_seconds=time.monotonic() - begin,
             max_hbm_allocated=torch.cuda.max_memory_allocated(),
             max_hbm_reserved=torch.cuda.max_memory_reserved())

        if update % args.save_every == 0:
            dist.barrier()
            if rank == 0:
                optimizer.eval()
                loss_value = sum(recent_losses) / len(recent_losses)
                checkpoint = args.output_root / f"checkpoint_agent_sft_u{update:06d}_loss_{loss_value:.4f}.pt"
                payload = {
                    "schema": SCHEMA,
                    "model_state_dict": core_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sft_updates": update,
                    "sft_total_tokens": total_tokens,
                    "assistant_target_tokens": total_targets,
                    "loss": loss_value,
                    "weight_mode": "saved-eval-x",
                    "parent_checkpoint": str(args.parent_checkpoint),
                    **expected_resume,
                    "source_commit": args.source_commit,
                    "sampler_key": args.sampler_key,
                    "sampler_cursor": update,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_steps": args.warmup_steps,
                    "merge_bucket_numel": args.merge_bucket_numel,
                }
                atomic_save(checkpoint, payload)
                digest = sha256(checkpoint)
                emit(args.log_jsonl, "checkpoint", rank, update=update,
                     checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                     checkpoint_sha256=digest, loss=loss_value)
                optimizer.train()
                checkpoints = sorted(args.output_root.glob("checkpoint_agent_sft_u*.pt"))
                for old in checkpoints[:-args.keep_checkpoints]:
                    if old.resolve() != checkpoint.resolve():
                        old.unlink()
            dist.barrier()

    emit(args.log_jsonl, "complete", rank, updates=args.steps,
         total_tokens=total_tokens, assistant_target_tokens=total_targets,
         final_loss=(sum(recent_losses) / len(recent_losses)))
    data.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
