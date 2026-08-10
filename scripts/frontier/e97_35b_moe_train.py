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


def emit(path: Path, event: str, **fields) -> None:
    record = {"event": event, "time_unix": time.time(), **fields}
    line = json.dumps(record, sort_keys=True)
    if dist.get_rank() == 0:
        print(line, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _canonical_checkpoint(args, groups, model, optimizer, *, step, accepted_tokens,
                          source_commit, sampler_identity, sampler_transition):
    """Publish one canonical eight-rank island, coordinated by global rank zero."""
    dist.barrier()
    if groups.node_index == 0:
        save_node_sharded_checkpoint(
            args.checkpoint_root, model, optimizer, step=step,
            accepted_tokens=accepted_tokens, source_commit=source_commit,
            sampler=_sampler_metadata(
                sampler_identity, accepted_tokens=accepted_tokens),
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
                     and manifest.get("sampler") == _sampler_metadata(
                         sampler_identity, accepted_tokens=accepted_tokens)
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
    if args.max_steps < 0 or args.minutes < 0 or (args.max_steps == 0 and args.minutes == 0):
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
            and args.moe_token_chunk_size != effective_rows):
        raise SystemExit(
            "training moe-token-chunk-size must equal the effective sequence segment; "
            "smaller chunks change the router balance objective")
    if args.resume_lr_override is not None:
        if args.resume_root is None or args.resume_lr_override <= 0:
            raise SystemExit("positive resume-lr-override requires resume-root")
    if args.save_every and args.checkpoint_root is None:
        raise SystemExit("save-every requires checkpoint-root")
    if args.save_every and args.save_every % args.diloco_k:
        raise SystemExit("save-every must be aligned to completed DiLoCo K boundaries")
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        groups = create_moe_process_groups()
        sampler_identity = _sampler_identity(
            args, world_size=dist.get_world_size())
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
             hbm_allocated=torch.cuda.memory_allocated(),
             hbm_reserved=torch.cuda.memory_reserved())

        optimizer = FusedScheduleFreeAdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), warmup_steps=0)
        starting_step = int(loaded.step)
        accepted_tokens = 0
        sampler_transition = None
        if args.resume_root is not None:
            manifest = load_node_sharded_checkpoint(
                args.resume_root, model, optimizer, node_group=groups.node_group,
                expected_sampler_identity=sampler_identity,
                allow_legacy_sampler_transition=args.sampler_transition_from_legacy,
                allow_counter_sampler_transition=args.sampler_transition_from_counter,
                diloco_k=args.diloco_k)
            starting_step = int(manifest["step"])
            accepted_tokens = int(manifest["accepted_tokens"])
            if args.resume_lr_override is not None:
                for optimizer_group in optimizer.param_groups:
                    optimizer_group["lr"] = float(args.resume_lr_override)
            restore_status = manifest["sampler_restore_status"]
            if restore_status in {"legacy-transition", "counter-transition"}:
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
            else:
                sampler_transition = manifest.get("sampler_transition")
            emit(args.log_jsonl, "restart_loaded", checkpoint_step=starting_step,
                 checkpoint_tokens=accepted_tokens,
                 learning_rates=[group["lr"] for group in optimizer.param_groups],
                 resume_lr_override=args.resume_lr_override,
                 sampler_restore_status=restore_status,
                 sampler_transition=sampler_transition)
        data_seed_base = 42 + starting_step
        dataset = TokenizedStreamDataset(
            data_path=str(args.data), chunk_size=args.chunk_size + 1,
            rank=dist.get_rank(), world_size=dist.get_world_size(),
            seed=data_seed_base, tokenizer_name="p50k_base",
            sampler_identity=sampler_identity,
            total_accepted_tokens=(
                accepted_tokens if sampler_identity is not None else None),
            accepted_tokens_per_sample=args.chunk_size)
        if sampler_identity is None:
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
            torch.cuda.reset_peak_memory_stats()
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
            if args.sequence_chunk_size > 0:
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
                     merge_ms=phase_events["optimizer"].elapsed_time(phase_events["merge"]),
                     forward_hbm_allocated=forward_hbm_allocated,
                     forward_hbm_reserved=forward_hbm_reserved,
                     forward_max_hbm_allocated=forward_max_hbm_allocated,
                     backward_hbm_allocated=backward_hbm_allocated,
                     backward_hbm_reserved=backward_hbm_reserved,
                     backward_max_hbm_allocated=backward_max_hbm_allocated)
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
                    accepted_tokens=accepted_tokens, source_commit=source_commit,
                    sampler_identity=sampler_identity,
                    sampler_transition=sampler_transition)
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
                accepted_tokens=accepted_tokens, source_commit=source_commit,
                sampler_identity=sampler_identity,
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
