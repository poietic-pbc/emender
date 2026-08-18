#!/usr/bin/env python3
"""Fail-closed E97 35B MoE training runner for one eight-GCD node island."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist

from ndm.data.masked_sft_dataset import (
    MaskedSFTPackedDataset,
    SFTSamplerIdentity,
    restore_sft_checkpoint_metadata,
    sft_checkpoint_metadata,
    sha256,
)
from ndm.data.tokenized_dataset import (
    BOUNDARY_COUNTER_SAMPLER_SCHEMA,
    COUNTER_SAMPLER_SCHEMA,
    COUNTER_SAMPLER_SCHEMAS,
    LEGACY_SAMPLER_SCHEMA,
    CounterSamplerIdentity,
    TokenizedStreamDataset,
    sampler_checkpoint_metadata,
)
from ndm.e97 import load_e97_checkpoint
from ndm.e97_moe_checkpoint import (
    load_node_sharded_checkpoint,
    load_node_sharded_model,
    save_node_sharded_checkpoint,
)
from ndm.e97_moe_ep import (
    assert_node_local_ep_group,
    average_replicated_gradients_,
    sum_replicated_gradients_,
    synchronize_sharded_gradients_,
    create_moe_process_groups,
    diloco_average_schedulefree_,
    node_replicated_parameters,
)
from ndm.e97_moe_optimizer import FusedScheduleFreeAdamW
from ndm.models.e97_moe import (
    E97MoEConfig,
    convert_e97_ffns_to_node_local_moe,
    e97_moe_auxiliary_loss,
    iter_e97_moe_layers,
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
    parser.add_argument("--seed-args-json", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--loss-chunk-size", type=int, default=0)
    parser.add_argument(
        "--checkpoint-loss-chunks", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--projection-chunk-size", type=int, default=0)
    parser.add_argument("--sequence-chunk-size", type=int, default=0)
    parser.add_argument("--full-bptt-segments", action="store_true")
    parser.add_argument("--checkpoint-group-size", type=int, default=1)
    parser.add_argument("--moe-token-chunk-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.007e-3)
    parser.add_argument("--resume-lr-override", type=float)
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
    parser.add_argument("--empty-cache-before-backward", action="store_true")
    parser.add_argument("--offload-schedulefree-z", action="store_true")
    parser.add_argument("--offload-schedulefree-state", action="store_true")
    parser.add_argument("--sampler-schema")
    parser.add_argument("--sampler-corpus-sha256")
    parser.add_argument("--sampler-tokenizer-sha256")
    parser.add_argument("--sampler-key", type=int)
    parser.add_argument("--sampler-data-world-size", type=int)
    parser.add_argument("--sampler-stream-origin-accepted-tokens", type=int)
    parser.add_argument("--sft-authority-root", type=Path)
    parser.add_argument("--sft-authority-manifest-sha256")
    parser.add_argument("--sft-pack-root", type=Path)
    parser.add_argument("--sft-pack-manifest-sha256")
    parser.add_argument("--sft-sampler-key", type=int)
    parser.add_argument("--sft-parent-root", type=Path)
    parser.add_argument("--sft-parent-manifest-sha256")
    parser.add_argument(
        "--sft-resume-parent-optimizer", action="store_true",
        help="On the initial SFT step, restore the exact parent ScheduleFree state "
             "and group clocks instead of constructing fresh optimizer state.")
    parser.add_argument(
        "--sft-parent-optimizer-split",
        choices=("router-preserved", "nonrouter-preserved"),
        help="Restore mature state only for the selected parameter cohort and use "
             "fresh ScheduleFree state for the complementary trainable cohort.")
    parser.add_argument(
        "--sft-transition-data-world-size", action="store_true",
        help="Explicitly continue a K-aligned SFT checkpoint with only its fixed "
             "data-world size changed; all token/target/cursor clocks are retained.")
    parser.add_argument("--sft-validation-batches", type=int, default=0)
    parser.add_argument(
        "--sft-reset-state-between-records", action="store_true",
        help="Run every packed conversation from a clean recurrent state.")
    parser.add_argument(
        "--sft-transition-record-reset", action="store_true",
        help="Explicitly fork a complete K-aligned legacy packed-state SFT checkpoint "
             "into the record-reset objective lineage.")
    parser.add_argument(
        "--sft-cross-node-gradient-sync", action="store_true",
        help="Synchronize corresponding parameter-shard gradients before every update.")
    parser.add_argument(
        "--sft-transition-cross-node-gradient-sync", action="store_true",
        help="Explicitly fork a complete SFT checkpoint from model averaging to DDP.")
    parser.add_argument(
        "--sft-validation-exhaustive", action="store_true",
        help="Enumerate every validation pack exactly once; requires one eight-rank node.")
    parser.add_argument(
        "--sft-validation-only", action="store_true",
        help="Load the bound SFT parent, run exhaustive validation, and exit without updates.")
    parser.add_argument("--final-checkpoint-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--sampler-transition-from-legacy", action="store_true",
        help="Explicitly transition one complete K-aligned legacy authority to the "
             "counter sampler; never relabels historical samples.")
    parser.add_argument(
        "--sampler-transition-from-counter", action="store_true",
        help="Start a new boundary-relative counter phase (for example a new context "
             "or data world) from an exact counter-sampled checkpoint boundary.")
    parser.add_argument(
        "--profile-phases", action="store_true",
        help="synchronize GPU events and emit forward/backward/reduction/optimizer timings")
    return parser.parse_args()


def _sampler_identity(args, *, world_size: int) -> CounterSamplerIdentity | None:
    values = {
        "corpus_sha256": args.sampler_corpus_sha256,
        "tokenizer_sha256": args.sampler_tokenizer_sha256,
        "sampler_key": args.sampler_key,
        "data_world_size": args.sampler_data_world_size,
    }
    stream_origin = getattr(
        args, "sampler_stream_origin_accepted_tokens", None)
    if args.sampler_schema is None:
        provided = [name for name, value in values.items() if value is not None]
        if stream_origin is not None:
            provided.append("stream_origin_accepted_tokens")
        if (provided or args.sampler_transition_from_legacy
                or args.sampler_transition_from_counter):
            raise RuntimeError(
                "sampler fields/transition require --sampler-schema: "
                + ", ".join(provided))
        return None
    if args.sampler_schema not in COUNTER_SAMPLER_SCHEMAS:
        raise RuntimeError(
            f"unsupported sampler schema {args.sampler_schema!r}; expected one of "
            f"{sorted(COUNTER_SAMPLER_SCHEMAS)!r}")
    missing = [name for name, value in values.items() if value is None]
    if (args.sampler_schema == BOUNDARY_COUNTER_SAMPLER_SCHEMA
            and stream_origin is None):
        missing.append("stream_origin_accepted_tokens")
    if args.sampler_schema == COUNTER_SAMPLER_SCHEMA and stream_origin not in (None, 0):
        raise RuntimeError("counter-v1 cannot use a nonzero stream origin")
    if missing:
        raise RuntimeError("counter sampler identity missing: " + ", ".join(missing))
    if int(args.sampler_data_world_size) != int(world_size):
        raise RuntimeError(
            f"sampler world {args.sampler_data_world_size} != launched world {world_size}")
    return CounterSamplerIdentity(
        schema=args.sampler_schema,
        corpus_sha256=args.sampler_corpus_sha256,
        tokenizer_sha256=args.sampler_tokenizer_sha256,
        sampler_key=args.sampler_key,
        data_world_size=args.sampler_data_world_size,
        context_size=args.chunk_size,
        stream_origin_accepted_tokens=int(stream_origin or 0),
    )


def _sampler_metadata(identity, *, accepted_tokens: int):
    if identity is None:
        return {"schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"}
    return sampler_checkpoint_metadata(
        identity, total_accepted_tokens=accepted_tokens)


def _sft_identity(args, *, world_size: int) -> SFTSamplerIdentity | None:
    fields = (
        args.sft_authority_root, args.sft_authority_manifest_sha256,
        args.sft_pack_root, args.sft_pack_manifest_sha256,
        args.sft_sampler_key, args.sft_parent_root,
        args.sft_parent_manifest_sha256,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise RuntimeError("masked SFT requires a complete authority/pack/parent identity")
    if args.sampler_schema is not None:
        raise RuntimeError("byte-window and record-pack samplers are mutually exclusive")
    return SFTSamplerIdentity(
        authority_manifest_sha256=args.sft_authority_manifest_sha256,
        pack_manifest_sha256=args.sft_pack_manifest_sha256,
        sampler_key=args.sft_sampler_key, data_world_size=world_size,
        context_size=args.chunk_size, split="train")


def _parent_authority(args) -> dict:
    generation = args.sft_parent_root
    if not (generation / "manifest.json").is_file():
        generation = (generation / "latest").resolve(strict=True)
    manifest_path = generation / "manifest.json"
    if sha256(manifest_path) != args.sft_parent_manifest_sha256:
        raise RuntimeError("SFT parent manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True:
        raise RuntimeError("SFT parent is not a complete checkpoint authority")
    return {
        "manifest_sha256": args.sft_parent_manifest_sha256,
        "step": int(manifest["step"]),
        "accepted_tokens": int(manifest["accepted_tokens"]),
        "generation": str(generation.resolve()),
    }


def _sft_optimizer_parameter_groups(model, split: str | None):
    if split is None:
        return model.parameters()
    routers = []
    nonrouters = []
    for name, parameter in model.named_parameters():
        (routers if name.endswith(".mlp.router.weight") else nonrouters).append(parameter)
    if len(routers) != model.depth or not nonrouters:
        raise RuntimeError("SFT optimizer router/non-router partition is incomplete")
    preserved, fresh = (
        (routers, nonrouters) if split == "router-preserved"
        else (nonrouters, routers))
    return [{"params": preserved}, {"params": fresh}]


def _sft_metadata(identity, parent, *, total_tokens, target_tokens, cursor):
    return sft_checkpoint_metadata(
        identity, parent=parent, total_tokens=total_tokens,
        assistant_target_tokens=target_tokens,
        absolute_rank_sample_index=cursor)


def _sft_optimizer_split_policy(transition):
    """Recover the immutable split policy through objective/world transitions."""
    seen = set()
    while isinstance(transition, dict) and id(transition) not in seen:
        seen.add(id(transition))
        policy = transition.get("optimizer_state")
        if policy in {"router-preserved", "nonrouter-preserved"}:
            return policy
        transition = transition.get("previous_sampler_transition")
    return None


def _sft_transition_has_policy(transition, key, value):
    seen = set()
    while isinstance(transition, dict) and id(transition) not in seen:
        seen.add(id(transition))
        if transition.get(key) == value:
            return True
        transition = transition.get("previous_sampler_transition")
    return False


def _masked_sft_record_reset_objective(
        model, chunks, masks, actual_lengths, record_spans, *, node_group):
    """Score each complete conversation from a clean recurrent state.

    Every rank executes the node-wide maximum record count so node-local expert
    collectives stay ordered. Missing records use a fully masked two-token
    dummy; they contribute neither language-model nor router-auxiliary weight.
    """
    if chunks.shape[0] != 1 or len(record_spans) != 1:
        raise RuntimeError("record-reset SFT requires one pack per rank")
    spans = record_spans[0]
    if not spans or spans[-1][1] != int(actual_lengths[0].item()):
        raise RuntimeError("record-reset spans do not cover the exact pack")
    target_mask = masks[:, 1:].contiguous()
    positions = torch.arange(target_mask.shape[1], device=chunks.device).unsqueeze(0)
    target_mask = target_mask & (positions < (actual_lengths.unsqueeze(1) - 1))
    local_targets = target_mask.sum(dtype=torch.int64)
    node_targets = local_targets.clone()
    dist.all_reduce(node_targets, op=dist.ReduceOp.SUM, group=node_group)
    if int(node_targets.item()) <= 0:
        raise RuntimeError("masked SFT batch contains zero node-wide assistant targets")

    local_records = torch.tensor(len(spans), device=chunks.device, dtype=torch.int64)
    dist.all_reduce(local_records, op=dist.ReduceOp.MAX, group=node_group)
    local_prediction_rows = torch.tensor(
        sum(max(0, stop - start - 1) for start, stop in spans),
        device=chunks.device, dtype=torch.int64)
    node_prediction_rows = local_prediction_rows.clone()
    dist.all_reduce(node_prediction_rows, op=dist.ReduceOp.SUM, group=node_group)
    loss_parts = []
    auxiliary = chunks.new_zeros((), dtype=torch.float32)
    observed_targets = 0
    for record_index in range(int(local_records.item())):
        if record_index < len(spans):
            start, stop = spans[record_index]
            if stop - start < 2:
                raise RuntimeError("SFT record is too short for causal training")
            real_length = stop - start
            # The sparse-checkpoint recurrent training kernel requires the
            # number of prediction/input rows to be divisible by 16. Tail
            # padding is causally after every real token and is fully masked;
            # the resulting final state is discarded at this record boundary.
            padded_length = ((real_length - 2) // 16 + 1) * 16 + 1
            record_tokens = torch.zeros(
                (1, padded_length), device=chunks.device, dtype=torch.long)
            record_tokens[:, :real_length] = chunks[:, start:stop]
            record_mask = torch.zeros(
                (1, padded_length - 1), device=chunks.device, dtype=torch.bool)
            record_mask[:, :real_length - 1] = masks[:, start + 1:stop]
            record_length = torch.tensor(
                [real_length], device=chunks.device, dtype=torch.long)
            weight = (real_length - 1) / int(node_prediction_rows.item())
            observed_targets += int(record_mask.sum().item())
        else:
            record_tokens = torch.zeros((1, 17), device=chunks.device, dtype=torch.long)
            record_mask = torch.zeros((1, 16), device=chunks.device, dtype=torch.bool)
            record_length = torch.ones(1, device=chunks.device, dtype=torch.long)
            weight = 0.0
        part = model(
            record_tokens, return_loss=True, actual_length=record_length,
            loss_mask=record_mask, loss_reduction="sum")
        loss_parts.append(part)
        auxiliary = auxiliary + e97_moe_auxiliary_loss(model).float() * weight
    if observed_targets != int(local_targets.item()):
        raise RuntimeError("record-reset target accounting mismatch")
    loss_sum = torch.stack(loss_parts).sum()
    return (loss_sum / node_targets.to(dtype=loss_sum.dtype), loss_sum.detach(),
            local_targets, node_targets, auxiliary)


def _masked_sft_objective(model, chunks, masks, actual_lengths, *, node_group):
    """Exact node-wide assistant-target normalization for node-local EP.

    Expert gradients already sum contributions arriving through differentiable
    all-to-all. Replicated gradients are therefore reduced with SUM rather than
    AVG by the caller, while every source loss uses the shared node denominator.
    """
    target_mask = masks[:, 1:].contiguous()
    positions = torch.arange(target_mask.shape[1], device=chunks.device).unsqueeze(0)
    target_mask = target_mask & (positions < (actual_lengths.unsqueeze(1) - 1))
    local_targets = target_mask.sum(dtype=torch.int64)
    node_targets = local_targets.clone()
    dist.all_reduce(node_targets, op=dist.ReduceOp.SUM, group=node_group)
    if int(node_targets.item()) <= 0:
        raise RuntimeError("masked SFT batch contains zero node-wide assistant targets")
    loss_sum = model(
        chunks, return_loss=True, actual_length=actual_lengths,
        loss_mask=target_mask, loss_reduction="sum")
    loss = loss_sum / node_targets.to(dtype=loss_sum.dtype)
    return loss, loss_sum.detach(), local_targets, node_targets


def _run_sft_validation(args, model, optimizer, groups, train_identity):
    if args.sft_validation_batches <= 0 and not args.sft_validation_exhaustive:
        return None
    validation_identity = SFTSamplerIdentity(
        authority_manifest_sha256=train_identity.authority_manifest_sha256,
        pack_manifest_sha256=train_identity.pack_manifest_sha256,
        sampler_key=train_identity.sampler_key,
        data_world_size=train_identity.data_world_size,
        context_size=train_identity.context_size, split="validation")
    dataset = MaskedSFTPackedDataset(
        args.sft_authority_root, args.sft_pack_root,
        identity=validation_identity, rank=dist.get_rank())
    optimizer.eval()
    model.eval()
    totals = torch.zeros(2, device="cuda", dtype=torch.float64)
    routing_layers = list(iter_e97_moe_layers(model))
    routing_counts = torch.zeros(
        (len(routing_layers), 64), device="cuda", dtype=torch.int64)

    def capture_routing() -> None:
        for layer_index, layer in enumerate(routing_layers):
            counts = layer.last_metrics.get("expert_token_counts")
            if counts is None or counts.numel() != 64:
                raise RuntimeError("SFT validation routing metrics are incomplete")
            routing_counts[layer_index] += counts.to(
                device="cuda", dtype=torch.int64)

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.sft_validation_exhaustive:
                if dist.get_world_size() != 8 or groups.node_count != 1:
                    raise RuntimeError(
                        "exhaustive SFT validation requires one complete eight-rank node")
                pack_count = len(dataset.packs)
                pack_losses = torch.zeros(pack_count, device="cuda", dtype=torch.float64)
                pack_targets = torch.zeros(pack_count, device="cuda", dtype=torch.int64)
                rank = dist.get_rank()
                iterations = (pack_count + 7) // 8
                for iteration in range(iterations):
                    pack_id = iteration * 8 + rank
                    if pack_id < pack_count:
                        token, mask, length, target_count, _sample_id = dataset.pack_at(pack_id)
                        spans = (dataset.record_spans_at(pack_id),)
                    else:
                        token = torch.zeros(dataset.sequence_tokens, dtype=torch.long)
                        mask = torch.zeros(dataset.sequence_tokens, dtype=torch.bool)
                        length, target_count, spans = 2, 0, (((0, 2),),)
                    chunks = token.unsqueeze(0).to("cuda", non_blocking=True)
                    masks = mask.unsqueeze(0).to("cuda", non_blocking=True)
                    lengths = torch.tensor([length], device="cuda", dtype=torch.long)
                    if args.sft_reset_state_between_records:
                        (_loss, local_sum, local_targets, _node_targets,
                         _auxiliary) = _masked_sft_record_reset_objective(
                            model, chunks, masks, lengths, spans,
                            node_group=groups.node_group)
                    else:
                        _loss, local_sum, local_targets, _node_targets = _masked_sft_objective(
                            model, chunks, masks, lengths, node_group=groups.node_group)
                    capture_routing()
                    if pack_id < pack_count:
                        if int(local_targets.item()) != target_count:
                            raise RuntimeError("exhaustive validation target accounting drift")
                        pack_losses[pack_id] = local_sum.double()
                        pack_targets[pack_id] = local_targets
                dist.all_reduce(pack_losses, op=dist.ReduceOp.SUM)
                dist.all_reduce(pack_targets, op=dist.ReduceOp.SUM)
                if bool((pack_targets <= 0).any().item()):
                    raise RuntimeError("exhaustive validation contains an unmeasured pack")
                totals[0] = pack_losses.sum()
                totals[1] = pack_targets.sum()
                losses_cpu = pack_losses.cpu().numpy()
                targets_cpu = pack_targets.cpu().numpy()
                rng = np.random.default_rng(970035)
                bootstrap = np.empty(2000, dtype=np.float64)
                for sample in range(len(bootstrap)):
                    indices = rng.integers(0, pack_count, size=pack_count)
                    bootstrap[sample] = (
                        losses_cpu[indices].sum() / targets_cpu[indices].sum())
                result = {
                    "loss": float(totals[0].item() / totals[1].item()),
                    "target_tokens": int(totals[1].item()),
                    "packs": pack_count,
                    "batches": iterations,
                    "sampling": "exact-pack-enumeration",
                    "bootstrap_p025": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_p975": float(np.quantile(bootstrap, 0.975)),
                    "pack_mean_nll": float(np.mean(losses_cpu / targets_cpu)),
                }
            else:
                for _ in range(args.sft_validation_batches):
                    if args.sft_reset_state_between_records:
                        (chunks, masks, lengths, _targets, spans) = (
                            dataset.get_batch_with_record_spans(
                                args.batch_size, device=torch.device("cuda")))
                        (_loss, local_sum, local_targets, _node_targets,
                         _auxiliary) = _masked_sft_record_reset_objective(
                            model, chunks, masks, lengths, spans,
                            node_group=groups.node_group)
                    else:
                        chunks, masks, lengths, _targets = dataset.get_batch(
                            args.batch_size, device=torch.device("cuda"))
                        _loss, local_sum, local_targets, _node_targets = _masked_sft_objective(
                            model, chunks, masks, lengths, node_group=groups.node_group)
                    capture_routing()
                    totals[0] += local_sum.double()
                    totals[1] += local_targets.double()
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                result = {"loss": float((totals[0] / totals[1]).item()),
                          "target_tokens": int(totals[1].item()),
                          "batches": args.sft_validation_batches,
                          "sampling": "counter-with-replacement"}
        if totals[1].item() <= 0:
            raise RuntimeError("SFT validation has zero assistant targets")
        dist.all_reduce(routing_counts, op=dist.ReduceOp.SUM)
        routing = []
        for layer_index, counts in enumerate(routing_counts.double()):
            total = float(counts.sum().item())
            mean = total / counts.numel()
            probabilities = counts / total
            positive = probabilities > 0
            entropy = float((-(probabilities[positive] * probabilities[positive].log()).sum()
                             / np.log(counts.numel())).item())
            routing.append({
                "layer": layer_index,
                "max_over_mean": float(counts.max().item() / mean),
                "coefficient_of_variation": float(
                    (counts.std(unbiased=False) / mean).item()),
                "normalized_entropy": entropy,
                "unused_experts": int((counts == 0).sum().item()),
            })
        result["routing"] = routing
        return result
    finally:
        dataset.close()
        model.train()
        optimizer.train()


def emit(path: Path, event: str, **fields) -> None:
    record = {"event": event, "time_unix": time.time(), **fields}
    line = json.dumps(record, sort_keys=True)
    if dist.get_rank() == 0:
        print(line, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _canonical_checkpoint(args, groups, model, optimizer, *, step, accepted_tokens,
                          source_commit, sampler_metadata, sampler_transition):
    """Publish one canonical eight-rank island, coordinated by global rank zero."""
    dist.barrier()
    if groups.node_index == 0:
        save_node_sharded_checkpoint(
            args.checkpoint_root, model, optimizer, step=step,
            accepted_tokens=accepted_tokens, source_commit=source_commit,
            sampler=sampler_metadata,
            sampler_transition=sampler_transition,
            node_group=groups.node_group, keep_generations=args.keep_checkpoints)
    dist.barrier()
    # Only global rank zero reads the just-published Lustre symlink/manifest.
    # Other nodes can retain stale metadata briefly even after the RCCL barrier;
    # broadcasting the verified authority prevents a false post-publication
    # failure while preserving rank-zero checkpoint ownership.
    authority = torch.zeros(3, device="cuda", dtype=torch.int64)
    generation = args.checkpoint_root / (
        f"step-{step:08d}-tokens-{accepted_tokens:016d}")
    if dist.get_rank() == 0:
        try:
            resolved = (args.checkpoint_root / "latest").resolve(strict=True)
            manifest = json.loads((resolved / "manifest.json").read_text())
            valid = (resolved == generation.resolve(strict=True)
                     and manifest.get("complete") is True
                     and int(manifest.get("step", -1)) == step
                     and int(manifest.get("accepted_tokens", -1)) == accepted_tokens
                     and manifest.get("sampler") == sampler_metadata
                     and manifest.get("sampler_transition") == sampler_transition)
            authority.copy_(torch.tensor(
                [int(valid), int(manifest.get("step", -1)),
                 int(manifest.get("accepted_tokens", -1))],
                device="cuda", dtype=torch.int64))
        except Exception:
            authority.zero_()
    dist.broadcast(authority, src=0)
    observed = tuple(int(value) for value in authority.cpu().tolist())
    if observed != (1, step, accepted_tokens):
        raise RuntimeError(
            f"canonical checkpoint authority mismatch: observed={observed}, "
            f"expected={(1, step, accepted_tokens)}")
    return generation


def _detach_recurrent_hiddens(hiddens):
    return [
        [head.detach() for head in layer] if isinstance(layer, (list, tuple))
        else layer.detach()
        for layer in hiddens
    ]


def _segmented_full_bptt_objective(
        model, chunks: torch.Tensor, *, sequence_chunk_size: int):
    """Build one full-BPTT graph from bounded forward segments.

    Recurrent states remain attached across segment boundaries. Per-segment
    block/loss checkpoints bound active materialization; one final backward
    traverses the entire context in reverse.
    """
    prediction_tokens = chunks.shape[1] - 1
    if prediction_tokens <= 0 or prediction_tokens % sequence_chunk_size:
        raise RuntimeError("full-BPTT segment must exactly partition prediction tokens")
    previous_hiddens = None
    losses = []
    auxiliaries = []
    for start in range(0, prediction_tokens, sequence_chunk_size):
        token_segment = chunks[:, start:start + sequence_chunk_size + 1]
        segment_loss, (previous_hiddens, _conv) = model(
            token_segment, return_loss=True, return_prev_hiddens=True,
            prev_hiddens=previous_hiddens)
        losses.append(segment_loss)
        auxiliaries.append(e97_moe_auxiliary_loss(model))
    return torch.stack(losses).mean(), torch.stack(auxiliaries).mean()


def _tbptt_objective_backward(model, chunks: torch.Tensor, *, sequence_chunk_size: int):
    """Backpropagate one long window in bounded, state-continuous TBPTT segments.

    Segment C consumes tokens [start:start+C] and predicts exactly the next C
    tokens. Recurrent state is carried forward, then detached after each segment;
    gradients are accumulated and normalized before one optimizer update.
    """
    prediction_tokens = chunks.shape[1] - 1
    if prediction_tokens <= 0 or prediction_tokens % sequence_chunk_size:
        raise RuntimeError("TBPTT chunk must exactly partition prediction tokens")
    segment_count = prediction_tokens // sequence_chunk_size
    previous_hiddens = None
    loss_total = chunks.new_zeros((), dtype=torch.float32)
    auxiliary_total = chunks.new_zeros((), dtype=torch.float32)

    for start in range(0, prediction_tokens, sequence_chunk_size):
        token_segment = chunks[:, start:start + sequence_chunk_size + 1]
        segment_loss, (new_hiddens, _conv) = model(
            token_segment, return_loss=True, return_prev_hiddens=True,
            prev_hiddens=previous_hiddens)
        segment_auxiliary = e97_moe_auxiliary_loss(model)
        ((segment_loss + segment_auxiliary) / segment_count).backward()
        loss_total = loss_total + segment_loss.detach().float() / segment_count
        auxiliary_total = (
            auxiliary_total + segment_auxiliary.detach().float() / segment_count)
        previous_hiddens = _detach_recurrent_hiddens(new_hiddens)

    return loss_total, auxiliary_total


def main() -> None:
    args = parse_args()
    if not args.seed_checkpoint.is_file() or not args.data.is_file():
        raise SystemExit("seed checkpoint or training data is unavailable")
    if (args.max_steps < 0 or args.minutes < 0
            or (args.max_steps == 0 and args.minutes == 0
                and not args.sft_validation_only)):
        raise SystemExit("set positive max-steps and/or minutes; zero means no limit")
    if args.save_every < 0 or args.keep_checkpoints < 1:
        raise SystemExit("save-every must be nonnegative and keep-checkpoints must be positive")
    if (args.chunk_size <= 0 or args.loss_chunk_size < 0
            or args.checkpoint_interval <= 0 or args.projection_chunk_size < 0
            or args.sequence_chunk_size < 0 or args.checkpoint_group_size <= 0
            or args.moe_token_chunk_size < 0):
        raise SystemExit(
            "chunk-size/checkpoint-interval must be positive and chunk controls nonnegative")
    if args.chunk_size % args.checkpoint_interval:
        raise SystemExit("chunk-size must be divisible by checkpoint-interval")
    if (args.projection_chunk_size > 0
            and (args.chunk_size % args.projection_chunk_size
                 or args.projection_chunk_size % args.checkpoint_interval)):
        raise SystemExit(
            "projection-chunk-size must divide context and be divisible by checkpoint-interval")
    if (args.sequence_chunk_size > 0
            and (args.chunk_size % args.sequence_chunk_size
                 or args.sequence_chunk_size % args.checkpoint_interval)):
        raise SystemExit(
            "sequence-chunk-size must divide context and be divisible by checkpoint-interval")
    if args.full_bptt_segments and args.sequence_chunk_size <= 0:
        raise SystemExit("full-bptt-segments requires a positive sequence-chunk-size")
    effective_rows = args.sequence_chunk_size or args.chunk_size
    if (args.moe_token_chunk_size > 0
            and (args.moe_token_chunk_size > effective_rows
                 or effective_rows % args.moe_token_chunk_size)):
        raise SystemExit(
            "training moe-token-chunk-size must divide the effective sequence segment")
    if args.resume_lr_override is not None:
        if ((args.resume_root is None and not args.sft_resume_parent_optimizer
             and args.sft_parent_optimizer_split is None)
                or args.resume_lr_override <= 0):
            raise SystemExit(
                "positive resume-lr-override requires resume-root or parent-optimizer SFT")
    if args.save_every and args.checkpoint_root is None:
        raise SystemExit("save-every requires checkpoint-root")
    if args.save_every and args.save_every % args.diloco_k:
        raise SystemExit("save-every must be aligned to completed DiLoCo K boundaries")
    if args.sft_validation_batches < 0 or args.final_checkpoint_delay_seconds < 0:
        raise SystemExit("SFT validation batches and checkpoint delay must be nonnegative")
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        groups = create_moe_process_groups()
        sampler_identity = _sampler_identity(
            args, world_size=dist.get_world_size())
        sft_identity = _sft_identity(args, world_size=dist.get_world_size())
        sft_parent = _parent_authority(args) if sft_identity is not None else None
        if (args.sft_resume_parent_optimizer and args.sft_parent_optimizer_split):
            raise RuntimeError("full and split parent-optimizer transitions are exclusive")
        if ((args.sft_resume_parent_optimizer or args.sft_parent_optimizer_split)
                and sft_identity is None):
            raise RuntimeError("parent-optimizer transition is valid only for masked SFT")
        if args.sft_transition_record_reset and (
                not args.sft_reset_state_between_records or args.resume_root is None):
            raise RuntimeError(
                "record-reset transition requires reset mode and --resume-root")
        if args.sft_reset_state_between_records and (
                sft_identity is None or args.batch_size != 1):
            raise RuntimeError("record-reset SFT requires masked SFT batch_size=1")
        if args.sft_transition_cross_node_gradient_sync and (
                not args.sft_cross_node_gradient_sync or args.resume_root is None):
            raise RuntimeError(
                "DDP transition requires cross-node gradient sync and --resume-root")
        if args.sft_cross_node_gradient_sync and (
                sft_identity is None or groups.node_count <= 1
                or args.diloco_k != 1):
            raise RuntimeError(
                "SFT cross-node gradient sync requires multinode masked SFT and diloco-k=1")
        if args.sft_validation_only and (
                sft_identity is None or not args.sft_validation_exhaustive
                or args.resume_root is not None or args.sft_resume_parent_optimizer
                or args.sft_parent_optimizer_split is not None):
            raise RuntimeError(
                "validation-only requires fresh model-only SFT parent exhaustive evaluation")
        if sft_identity is not None and (args.sequence_chunk_size > 0
                                         or args.full_bptt_segments):
            raise RuntimeError("initial masked SFT requires one complete unsegmented context")
        if (args.sampler_transition_from_legacy
                and args.sampler_transition_from_counter):
            raise RuntimeError("sampler transition modes are mutually exclusive")
        if ((args.sampler_transition_from_legacy
                or args.sampler_transition_from_counter)
                and args.resume_root is None):
            raise RuntimeError("sampler transition requires --resume-root")
        if ((args.sampler_transition_from_legacy
                or args.sampler_transition_from_counter)
                and sampler_identity.schema != BOUNDARY_COUNTER_SAMPLER_SCHEMA):
            raise RuntimeError("sampler transition requires boundary-relative counter-v2")
        if (sampler_identity is not None
                and sampler_identity.stream_origin_accepted_tokens > 0
                and args.resume_root is None):
            raise RuntimeError(
                "a positive counter-v2 stream origin requires resume authority")
        if groups.node_count > 1 and args.minutes > 0:
            raise RuntimeError(
                "multinode production requires an exact max-steps boundary; "
                "independent wall-clock stopping can desynchronize node islands")
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
            weight_mode="train", use_triton=True, mmap=True,
            args_json=args.seed_args_json)
        if loaded.step != 2322520 or int(loaded.config.get("dim", -1)) != 1792:
            raise RuntimeError("loaded checkpoint is not the bound final 513B E97 seed")
        model = loaded.model
        model.gradient_checkpointing = bool(args.gradient_checkpointing)
        model.gradient_checkpoint_group_size = int(args.checkpoint_group_size)
        model.loss_chunk_size = int(args.loss_chunk_size)
        model.checkpoint_loss_chunks = bool(args.checkpoint_loss_chunks)
        recurrent_mixers = []
        for module in model.modules():
            if hasattr(module, "checkpoint_interval"):
                module.checkpoint_interval = int(args.checkpoint_interval)
                module.projection_chunk_size = int(args.projection_chunk_size)
                recurrent_mixers.append(module)
        if len(recurrent_mixers) != model.depth:
            raise RuntimeError(
                f"expected {model.depth} recurrent mixers, found {len(recurrent_mixers)}")
        convert_e97_ffns_to_node_local_moe(
            model,
            E97MoEConfig(hidden_dim=8832, routed_experts=64, shared_experts=1,
                         top_k=3, expert_parallel_size=8,
                         expert_backend=args.expert_backend),
            local_expert_rank=local_rank, expert_group=groups.node_group)
        for moe_layer in iter_e97_moe_layers(model):
            moe_layer.token_chunk_size = int(args.moe_token_chunk_size)
        parent_manifest = None
        if (sft_identity is not None and args.resume_root is None
                and not args.sft_resume_parent_optimizer
                and args.sft_parent_optimizer_split is None):
            parent_manifest = load_node_sharded_model(
                sft_parent["generation"], model, node_group=groups.node_group)
            if (int(parent_manifest["step"]) != sft_parent["step"]
                    or int(parent_manifest["accepted_tokens"])
                    != sft_parent["accepted_tokens"]):
                raise RuntimeError("loaded SFT parent clock mismatch")
            emit(args.log_jsonl, "sft_parent_loaded", parent=sft_parent,
                 optimizer_state="fresh")
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
             gradient_checkpointing=model.gradient_checkpointing,
             gradient_checkpointing_requested=args.gradient_checkpointing,
             sequence_chunk_size=args.sequence_chunk_size,
             full_bptt_segments=args.full_bptt_segments,
             tbptt_truncated=(
                 args.sequence_chunk_size > 0 and not args.full_bptt_segments),
             checkpoint_group_size=args.checkpoint_group_size,
             moe_token_chunk_size=args.moe_token_chunk_size,
             offload_schedulefree_z=args.offload_schedulefree_z,
             offload_schedulefree_state=args.offload_schedulefree_state,
             loss_chunk_size=model.loss_chunk_size,
             checkpoint_loss_chunks=model.checkpoint_loss_chunks,
             checkpoint_interval=args.checkpoint_interval,
             projection_chunk_size=args.projection_chunk_size,
             context_size=args.chunk_size,
             sft_state_policy=(
                 "reset-at-record-boundaries-v1"
                 if args.sft_reset_state_between_records else "continuous-pack-v1"),
             sft_optimizer_sync_policy=(
                 "corresponding-lane-gradient-sum-v1"
                 if args.sft_cross_node_gradient_sync else "diloco-model-average-v1"),
             hbm_allocated=torch.cuda.memory_allocated(),
             hbm_reserved=torch.cuda.memory_reserved())

        optimizer = FusedScheduleFreeAdamW(
            _sft_optimizer_parameter_groups(model, args.sft_parent_optimizer_split),
            lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), warmup_steps=0)
        starting_step = int(sft_parent["step"] if sft_identity is not None else loaded.step)
        accepted_tokens = int(sft_parent["accepted_tokens"] if sft_identity is not None else 0)
        sft_total_tokens = 0
        sft_target_tokens = 0
        sft_cursor = 0
        sft_steps = 0
        sampler_transition = None
        restore_root = args.resume_root
        parent_optimizer_transition = (
            sft_identity is not None and args.resume_root is None
            and (args.sft_resume_parent_optimizer
                 or args.sft_parent_optimizer_split is not None))
        if parent_optimizer_transition:
            restore_root = Path(sft_parent["generation"])
        if restore_root is not None:
            manifest = load_node_sharded_checkpoint(
                restore_root, model, optimizer, node_group=groups.node_group,
                expected_sampler_identity=sampler_identity,
                expected_sft_identity=sft_identity,
                expected_sft_parent=sft_parent,
                allow_sft_parent_optimizer_transition=parent_optimizer_transition,
                allow_sft_world_size_transition=args.sft_transition_data_world_size,
                allow_legacy_sampler_transition=args.sampler_transition_from_legacy,
                allow_counter_sampler_transition=args.sampler_transition_from_counter,
                diloco_k=args.diloco_k)
            starting_step = int(manifest["step"])
            accepted_tokens = int(manifest["accepted_tokens"])
            if (args.resume_root is not None
                    and args.sft_parent_optimizer_split is not None
                    and _sft_optimizer_split_policy(manifest.get("sampler_transition"))
                    != args.sft_parent_optimizer_split):
                raise RuntimeError("SFT split optimizer policy mismatch on resume")
            if sft_identity is not None:
                sft_total_tokens, sft_target_tokens, sft_cursor = manifest["sft_restore_clocks"]
                sft_steps = starting_step - int(sft_parent["step"])
                if sft_steps < 0 or sft_cursor != sft_steps * args.batch_size:
                    raise RuntimeError("SFT step and per-rank sample cursor disagree")
            if parent_optimizer_transition and args.sft_parent_optimizer_split is not None:
                fresh_group = optimizer.param_groups[1]
                for parameter in fresh_group["params"]:
                    optimizer.state[parameter].clear()
                emit(args.log_jsonl, "sft_optimizer_state_split",
                     policy=args.sft_parent_optimizer_split,
                     preserved_parameters=len(optimizer.param_groups[0]["params"]),
                     fresh_parameters=len(fresh_group["params"]),
                     preserved_k=optimizer.param_groups[0]["k"],
                     fresh_k=fresh_group["k"])
            if args.resume_lr_override is not None:
                for optimizer_group in optimizer.param_groups:
                    optimizer_group["lr"] = float(args.resume_lr_override)
            restore_status = manifest["sampler_restore_status"]
            if restore_status == "sft-parent-optimizer-transition":
                sampler_transition = {
                    "status": "counter-to-sft-preserve-optimizer",
                    "boundary_step": starting_step,
                    "boundary_accepted_tokens": accepted_tokens,
                    "previous_sampler": manifest.get("sampler"),
                    "new_sampler_identity": sft_identity.to_metadata(),
                    "optimizer_state": (
                        args.sft_parent_optimizer_split or "preserved"),
                }
            elif restore_status in {"legacy-transition", "counter-transition"}:
                previous_sampler = manifest.get("sampler")
                if previous_sampler is None:
                    previous_sampler = {
                        "schema": LEGACY_SAMPLER_SCHEMA,
                        "status": "legacy-metadata-absent",
                    }
                sampler_transition = {
                    "status": (
                        "legacy-to-counter" if restore_status == "legacy-transition"
                        else "counter-to-counter"),
                    "boundary_step": starting_step,
                    "boundary_accepted_tokens": accepted_tokens,
                    "previous_sampler": previous_sampler,
                    "new_sampler_identity": sampler_identity.to_metadata(),
                }
            elif restore_status == "sft-world-size-transition":
                sampler_transition = {
                    "status": "sft-data-world-size-transition",
                    "boundary_step": starting_step,
                    "boundary_accepted_tokens": accepted_tokens,
                    "boundary_cursor": sft_cursor,
                    "previous_sampler_identity": manifest["previous_sft_identity"],
                    "new_sampler_identity": sft_identity.to_metadata(),
                    "optimizer_state": "preserved-exact",
                }
            else:
                sampler_transition = manifest.get("sampler_transition")
            prior_reset = _sft_transition_has_policy(
                manifest.get("sampler_transition"), "objective_state_policy",
                "reset-at-record-boundaries-v1")
            if args.sft_transition_record_reset:
                if prior_reset or sft_cursor % args.diloco_k:
                    raise RuntimeError(
                        "record-reset objective transition requires a legacy K boundary")
                sampler_transition = {
                    "status": "sft-record-reset-objective-transition",
                    "boundary_step": starting_step,
                    "boundary_accepted_tokens": accepted_tokens,
                    "boundary_cursor": sft_cursor,
                    "objective_state_policy": "reset-at-record-boundaries-v1",
                    "previous_sampler_transition": manifest.get("sampler_transition"),
                    "optimizer_state": "preserved-exact",
                }
            elif args.sft_reset_state_between_records != prior_reset:
                raise RuntimeError(
                    "SFT recurrent-state objective policy mismatches checkpoint lineage")
            prior_ddp = _sft_transition_has_policy(
                manifest.get("sampler_transition"), "optimizer_sync_policy",
                "corresponding-lane-gradient-sum-v1")
            if args.sft_transition_cross_node_gradient_sync:
                if prior_ddp or sft_cursor % args.diloco_k:
                    raise RuntimeError("DDP transition requires a complete checkpoint boundary")
                sampler_transition = {
                    "status": "sft-cross-node-gradient-sync-transition",
                    "boundary_step": starting_step,
                    "boundary_accepted_tokens": accepted_tokens,
                    "boundary_cursor": sft_cursor,
                    "objective_state_policy": (
                        "reset-at-record-boundaries-v1" if prior_reset
                        else "continuous-pack-v1"),
                    "optimizer_sync_policy": "corresponding-lane-gradient-sum-v1",
                    "previous_sampler_transition": manifest.get("sampler_transition"),
                    "optimizer_state": "preserved-exact",
                }
            elif args.sft_cross_node_gradient_sync != prior_ddp:
                raise RuntimeError("SFT optimizer synchronization policy mismatches lineage")
            emit(args.log_jsonl, "restart_loaded", checkpoint_step=starting_step,
                 checkpoint_tokens=accepted_tokens,
                 learning_rates=[group["lr"] for group in optimizer.param_groups],
                 resume_lr_override=args.resume_lr_override,
                 sampler_restore_status=restore_status,
                 sampler_transition=sampler_transition,
                 sft_total_tokens=sft_total_tokens,
                 sft_target_tokens=sft_target_tokens, sft_cursor=sft_cursor)
        data_seed_base = 42 + starting_step
        dataset = (MaskedSFTPackedDataset(
            args.sft_authority_root, args.sft_pack_root,
            identity=sft_identity, rank=dist.get_rank(),
            initial_absolute_rank_sample_index=sft_cursor)
            if sft_identity is not None else TokenizedStreamDataset(
            data_path=str(args.data), chunk_size=args.chunk_size + 1,
            rank=dist.get_rank(), world_size=dist.get_world_size(),
            seed=data_seed_base, tokenizer_name="p50k_base",
            sampler_identity=sampler_identity,
            total_accepted_tokens=(
                accepted_tokens if sampler_identity is not None else None),
            accepted_tokens_per_sample=args.chunk_size))
        if sft_identity is not None:
            emit(args.log_jsonl, "sft_data_ready", starting_step=starting_step,
                 sampler_identity=sft_identity.to_metadata(), parent=sft_parent,
                 sft_total_tokens=sft_total_tokens,
                 sft_target_tokens=sft_target_tokens,
                 absolute_rank_sample_index=sft_cursor,
                 pack_counts=dataset.pack_manifest["splits"])
        elif sampler_identity is None:
            emit(args.log_jsonl, "data_stream_ready", starting_step=starting_step,
                 sampler_schema=LEGACY_SAMPLER_SCHEMA,
                 seed_base=data_seed_base,
                 rank_seed=data_seed_base + dist.get_rank(),
                 replay_previous_launch=False)
        else:
            emit(args.log_jsonl, "data_stream_ready", starting_step=starting_step,
                 sampler_schema=sampler_identity.schema,
                 sampler_identity=sampler_identity.to_metadata(),
                 accepted_tokens=accepted_tokens,
                 absolute_rank_sample_index=(
                     dataset.initial_absolute_rank_sample_index),
                 sampler_transition=sampler_transition)
        optimizer.train()
        if args.offload_schedulefree_state:
            optimizer.offload_state_()
            emit(args.log_jsonl, "optimizer_state_offloaded",
                 hbm_allocated=torch.cuda.memory_allocated(),
                 hbm_reserved=torch.cuda.memory_reserved())
        elif args.offload_schedulefree_z:
            optimizer.offload_z_()
            emit(args.log_jsonl, "optimizer_z_offloaded",
                 hbm_allocated=torch.cuda.memory_allocated(),
                 hbm_reserved=torch.cuda.memory_reserved())
        if (sft_identity is not None
                and (args.sft_validation_batches > 0
                     or args.sft_validation_exhaustive)):
            initial_validation = _run_sft_validation(
                args, model, optimizer, groups, sft_identity)
            emit(args.log_jsonl, "sft_validation", phase="initial", **initial_validation)
        if args.sft_validation_only:
            emit(args.log_jsonl, "validation_only_complete",
                 step=starting_step, accepted_tokens=accepted_tokens)
            return
        replicated = tuple(node_replicated_parameters(model))
        start = None
        step = starting_step
        completed_steps = 0
        last_checkpoint_step = None
        interval_node_target_tokens = 0
        while True:
            if completed_steps >= args.max_steps and args.max_steps > 0:
                break
            if start is not None and args.minutes > 0 and time.monotonic() - start >= args.minutes * 60:
                break
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            record_spans = None
            if sft_identity is not None:
                if args.sft_reset_state_between_records:
                    (chunks, assistant_masks, actual_lengths, batch_target_counts,
                     record_spans) = dataset.get_batch_with_record_spans(
                        args.batch_size, device=torch.device("cuda"))
                else:
                    chunks, assistant_masks, actual_lengths, batch_target_counts = (
                        dataset.get_batch(args.batch_size, device=torch.device("cuda")))
            else:
                chunks, _, actual_lengths = dataset.get_batch(
                    args.batch_size, device=torch.device("cuda"))
                assistant_masks = None
                batch_target_counts = None
            step_start = time.monotonic()
            phase_events = None
            if args.profile_phases:
                phase_events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in ("start", "forward", "backward", "reduce", "optimizer", "merge")
                }
                phase_events["start"].record()
            local_loss_sum = None
            local_target_count = None
            node_target_count = None
            if sft_identity is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if args.sft_reset_state_between_records:
                        (loss, local_loss_sum, local_target_count, node_target_count,
                         auxiliary) = _masked_sft_record_reset_objective(
                            model, chunks, assistant_masks, actual_lengths, record_spans,
                            node_group=groups.node_group)
                        objective = loss + auxiliary
                    else:
                        loss, local_loss_sum, local_target_count, node_target_count = (
                            _masked_sft_objective(
                                model, chunks, assistant_masks, actual_lengths,
                                node_group=groups.node_group))
                        auxiliary = e97_moe_auxiliary_loss(model)
                        objective = loss + auxiliary / dist.get_world_size(groups.node_group)
                gradients_ready = False
            elif args.sequence_chunk_size > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if args.full_bptt_segments:
                        loss, auxiliary = _segmented_full_bptt_objective(
                            model, chunks,
                            sequence_chunk_size=args.sequence_chunk_size)
                        gradients_ready = False
                    else:
                        loss, auxiliary = _tbptt_objective_backward(
                            model, chunks,
                            sequence_chunk_size=args.sequence_chunk_size)
                        gradients_ready = True
                objective = loss + auxiliary
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(chunks, return_loss=True)
                auxiliary = e97_moe_auxiliary_loss(model)
                objective = loss + auxiliary
                gradients_ready = False
            forward_hbm_allocated = torch.cuda.memory_allocated()
            forward_hbm_reserved = torch.cuda.memory_reserved()
            forward_max_hbm_allocated = torch.cuda.max_memory_allocated()
            if phase_events is not None:
                phase_events["forward"].record()
            if not torch.isfinite(objective):
                raise FloatingPointError(f"nonfinite objective at step {step}")
            if not gradients_ready:
                if args.empty_cache_before_backward:
                    torch.cuda.empty_cache()
                objective.backward()
            backward_hbm_allocated = torch.cuda.memory_allocated()
            backward_hbm_reserved = torch.cuda.memory_reserved()
            backward_max_hbm_allocated = torch.cuda.max_memory_allocated()
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
            if sft_identity is not None:
                sum_replicated_gradients_(
                    replicated, group=groups.node_group, topology=topology)
            else:
                average_replicated_gradients_(
                    replicated, group=groups.node_group, topology=topology)
            if args.sft_cross_node_gradient_sync:
                synchronize_sharded_gradients_(
                    model.parameters(), lane_group=groups.diloco_lane_group,
                    local_weight=int(node_target_count.item()))
            if phase_events is not None:
                phase_events["reduce"].record()
            optimizer.step()
            optimizer.assert_no_master_weights()
            if phase_events is not None:
                phase_events["optimizer"].record()
            merge_seconds = 0.0
            if sft_identity is not None and not args.sft_cross_node_gradient_sync:
                interval_node_target_tokens += int(node_target_count.item())
            merge_boundary = (
                (sft_steps + 1) % args.diloco_k == 0
                if sft_identity is not None else (step + 1) % args.diloco_k == 0)
            if (groups.node_count > 1 and merge_boundary
                    and not args.sft_cross_node_gradient_sync):
                # Release inactive variable-routing blocks before RCCL allocates its
                # cross-node collective workspace.  At 32 nodes, one lane rank can
                # otherwise retain enough allocator cache to starve RCCL despite
                # live tensors fitting comfortably in HBM.
                torch.cuda.empty_cache()
                dist.barrier(group=groups.diloco_lane_group)
                merge_start = time.monotonic()
                diloco_average_schedulefree_(
                    model, optimizer, lane_group=groups.diloco_lane_group,
                    weight=(interval_node_target_tokens
                            if sft_identity is not None else None))
                merge_seconds = time.monotonic() - merge_start
                if sft_identity is not None:
                    interval_node_target_tokens = 0
            if phase_events is not None:
                phase_events["merge"].record()
                phase_events["merge"].synchronize()
                emit(args.log_jsonl, "phase_profile", step=step,
                     forward_ms=phase_events["start"].elapsed_time(phase_events["forward"]),
                     backward_ms=phase_events["forward"].elapsed_time(phase_events["backward"]),
                     replicated_reduce_ms=phase_events["backward"].elapsed_time(phase_events["reduce"]),
                     optimizer_ms=phase_events["reduce"].elapsed_time(phase_events["optimizer"]),
                     merge_ms=phase_events["optimizer"].elapsed_time(phase_events["merge"]),
                     forward_hbm_allocated=forward_hbm_allocated,
                     forward_hbm_reserved=forward_hbm_reserved,
                     forward_max_hbm_allocated=forward_max_hbm_allocated,
                     backward_hbm_allocated=backward_hbm_allocated,
                     backward_hbm_reserved=backward_hbm_reserved,
                     backward_max_hbm_allocated=backward_max_hbm_allocated)
            if sft_identity is not None:
                node_loss_sum = local_loss_sum.clone()
                dist.all_reduce(node_loss_sum, op=dist.ReduceOp.SUM, group=groups.node_group)
                if args.sft_cross_node_gradient_sync:
                    global_loss_sum = node_loss_sum.clone()
                    global_target_count = node_target_count.clone()
                    dist.all_reduce(
                        global_loss_sum, op=dist.ReduceOp.SUM,
                        group=groups.diloco_lane_group)
                    dist.all_reduce(
                        global_target_count, op=dist.ReduceOp.SUM,
                        group=groups.diloco_lane_group)
                    reported_loss = global_loss_sum / global_target_count.to(
                        global_loss_sum.dtype)
                else:
                    reported_loss = node_loss_sum / node_target_count.to(node_loss_sum.dtype)
            else:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=groups.node_group)
                reported_loss = loss
            dist.all_reduce(auxiliary, op=dist.ReduceOp.AVG, group=groups.node_group)
            step_seconds = time.monotonic() - step_start
            tokens = int(actual_lengths.sum().item()) - args.batch_size
            global_counts = torch.tensor(
                [tokens, int(batch_target_counts.sum().item()) if sft_identity is not None else 0],
                device="cuda", dtype=torch.int64)
            if sft_identity is not None:
                dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
                step_global_tokens, step_global_targets = (
                    int(global_counts[0].item()), int(global_counts[1].item()))
                sft_total_tokens += step_global_tokens
                sft_target_tokens += step_global_targets
                accepted_tokens = sft_parent["accepted_tokens"] + sft_total_tokens
            else:
                step_global_tokens = tokens * dist.get_world_size()
                step_global_targets = 0
                accepted_tokens += step_global_tokens
            emit(args.log_jsonl, "step", step=step, loss=float(reported_loss.item()),
                 auxiliary_loss=float(auxiliary.item()), step_seconds=step_seconds,
                 diloco_merge_seconds=merge_seconds, node_count=groups.node_count,
                 accepted_tokens=accepted_tokens,
                 sft_total_tokens=(sft_total_tokens if sft_identity is not None else None),
                 sft_target_tokens=(sft_target_tokens if sft_identity is not None else None),
                 step_target_tokens=step_global_targets,
                 tokens_per_second=(step_global_tokens / step_seconds),
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
            if sft_identity is not None:
                sft_steps += 1
                sft_cursor += args.batch_size
                if dataset.next_absolute_rank_sample_index != sft_cursor:
                    raise RuntimeError("SFT in-process sampler cursor drift")
            save_boundary = (
                sft_steps % args.save_every == 0
                if sft_identity is not None and args.save_every > 0
                else step % args.save_every == 0 if args.save_every > 0 else False)
            if args.checkpoint_root is not None and save_boundary:
                checkpoint_start = time.monotonic()
                checkpoint = _canonical_checkpoint(
                    args, groups, model, optimizer, step=step,
                    accepted_tokens=accepted_tokens, source_commit=source_commit,
                    sampler_metadata=(
                        _sft_metadata(
                            sft_identity, sft_parent, total_tokens=sft_total_tokens,
                            target_tokens=sft_target_tokens, cursor=sft_cursor)
                        if sft_identity is not None else _sampler_metadata(
                            sampler_identity, accepted_tokens=accepted_tokens)),
                    sampler_transition=sampler_transition)
                last_checkpoint_step = step
                emit(args.log_jsonl, "checkpoint_complete",
                     checkpoint_path=str(checkpoint), step=step,
                     accepted_tokens=accepted_tokens,
                     checkpoint_seconds=time.monotonic() - checkpoint_start,
                     canonical_node=0)
        measured_seconds = 0.0 if start is None else time.monotonic() - start
        validation = _run_sft_validation(
            args, model, optimizer, groups, sft_identity) if sft_identity is not None else None
        if validation is not None:
            emit(args.log_jsonl, "sft_validation", phase="final", **validation)
        emit(args.log_jsonl, "complete", step=step, steps_completed=completed_steps,
             accepted_tokens=accepted_tokens,
             sft_total_tokens=(sft_total_tokens if sft_identity is not None else None),
             sft_target_tokens=(sft_target_tokens if sft_identity is not None else None),
             sft_cursor=(sft_cursor if sft_identity is not None else None),
             measured_training_seconds=measured_seconds,
             hbm_allocated=torch.cuda.memory_allocated(),
             max_hbm_allocated=torch.cuda.max_memory_allocated())
        if args.checkpoint_root is not None and last_checkpoint_step != step:
            if args.final_checkpoint_delay_seconds:
                emit(args.log_jsonl, "final_checkpoint_delay",
                     seconds=args.final_checkpoint_delay_seconds)
                time.sleep(args.final_checkpoint_delay_seconds)
            checkpoint_start = time.monotonic()
            checkpoint = _canonical_checkpoint(
                args, groups, model, optimizer, step=step,
                accepted_tokens=accepted_tokens, source_commit=source_commit,
                sampler_metadata=(
                    _sft_metadata(
                        sft_identity, sft_parent, total_tokens=sft_total_tokens,
                        target_tokens=sft_target_tokens, cursor=sft_cursor)
                    if sft_identity is not None else _sampler_metadata(
                        sampler_identity, accepted_tokens=accepted_tokens)),
                sampler_transition=sampler_transition)
            emit(args.log_jsonl, "checkpoint_complete",
                 checkpoint_path=str(checkpoint), step=step,
                 accepted_tokens=accepted_tokens,
                 checkpoint_seconds=time.monotonic() - checkpoint_start,
                 canonical_node=0)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
