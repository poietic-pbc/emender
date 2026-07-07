"""Real train.py-backed async DiLoCo orchestration helpers.

This module keeps transport out of scope but runs real model/optimizer steps
through train.py's import-safe helpers.  It is the production-training
counterpart to the synthetic protocol prototype in :mod:`ndm.async_diloco`.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field, replace
import copy
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from ndm.async_diloco import (
    AsyncDiLoCoCheckpointCadence,
    AsyncDiLoCoCheckpointManager,
    AsyncDiLoCoGenerationMetrics,
    AsyncDiLoCoMergeResult,
    AsyncDiLoCoMetricsSummary,
    AsyncDiLoCoUpdate,
    GLOBAL_MERGER_ROLE,
    build_metrics_summary,
    compute_dense_delta,
    default_global_quorum,
    default_local_quorum,
    quorum_merge,
    stable_json_dumps,
    state_num_bytes,
)

import train


_FILE_QUORUM_NO_GO_REASON = (
    "actual_multinode_file_quorum_debug records local training and metadata quorum only; "
    "dense cross-node delta exchange and merge are not implemented"
)


@dataclass(frozen=True)
class RealAsyncWorkerSpec:
    """One local trainer worker attached to a GPU island."""

    worker_id: str
    node_id: str = "node-0"
    device: str = "cpu"
    local_steps: int = 1
    seed_offset: int = 0
    fail_before_submit: bool = False
    timed_out: bool = False
    stale_generation: int | None = None


@dataclass(frozen=True)
class RealAsyncDiLoCoConfig:
    """Configuration for a run-local real async DiLoCo training attempt."""

    run_id: str
    run_dir: str | Path
    train_args: Namespace
    worker_specs: Sequence[RealAsyncWorkerSpec]
    generations: int = 1
    local_quorum: int | None = None
    global_quorum: int | None = None
    global_node_count: int | None = None
    eta_outer: float = 1.0
    weight_by: str = "tokens"
    timeout_s: float = 900.0
    initial_generation: int = 0
    initial_checkpoint: str | Path | None = None
    synthetic_token_stream: bool = False
    synthetic_vocab_size: int = 256
    metrics_json: str | Path | None = None
    walltime_remaining_s: float | None = None
    estimated_finalization_duration_s: float | None = None
    checkpoint_cadence: AsyncDiLoCoCheckpointCadence = field(
        default_factory=lambda: AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=None,
            recovery_every_seconds=None,
            export_every_generations=None,
            export_every_seconds=None,
        )
    )


@dataclass(frozen=True)
class RealAsyncWorkerReport:
    worker_id: str
    node_id: str
    base_generation: int
    update: AsyncDiLoCoUpdate | None
    elapsed_s: float
    tokens: int
    losses: tuple[float, ...] = ()
    failed: bool = False
    timed_out: bool = False
    invalid: bool = False
    error: str | None = None


@dataclass(frozen=True)
class RealAsyncNodeResult:
    node_id: str
    generation: int
    node_update: AsyncDiLoCoUpdate | None
    worker_reports: tuple[RealAsyncWorkerReport, ...]
    metrics: AsyncDiLoCoGenerationMetrics


@dataclass(frozen=True)
class RealAsyncGlobalResult:
    generation: int
    state: Mapping[str, torch.Tensor]
    metrics: AsyncDiLoCoGenerationMetrics
    publish_paths: tuple[str, ...]


@dataclass(frozen=True)
class RealAsyncDiLoCoRunResult:
    run_id: str
    generations: tuple[RealAsyncGlobalResult, ...]
    node_results: tuple[RealAsyncNodeResult, ...]
    metrics_summary: AsyncDiLoCoMetricsSummary
    metrics_json: str | None
    latest_path: str

    @property
    def latest_generation(self) -> int:
        advanced = [
            result.generation
            for result in self.generations
            if result.metrics.latest_advanced
        ]
        return max(advanced) if advanced else -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "latest_generation": self.latest_generation,
            "latest_path": self.latest_path,
            "metrics_summary": self.metrics_summary.to_dict(),
            "global_generations": [
                {
                    "generation": result.generation,
                    "metrics": result.metrics.to_dict(),
                    "publish_paths": list(result.publish_paths),
                }
                for result in self.generations
            ],
            "node_generations": [
                {
                    "node_id": result.node_id,
                    "generation": result.generation,
                    "metrics": result.metrics.to_dict(),
                    "node_update_submitted": result.node_update is not None,
                    "worker_reports": [
                        {
                            "worker_id": report.worker_id,
                            "node_id": report.node_id,
                            "base_generation": report.base_generation,
                            "tokens": report.tokens,
                            "losses": list(report.losses),
                            "failed": report.failed,
                            "timed_out": report.timed_out,
                            "invalid": report.invalid,
                            "error": report.error,
                        }
                        for report in result.worker_reports
                    ],
                }
                for result in self.node_results
            ],
        }


@dataclass(frozen=True)
class RealAsyncFileRankConfig:
    """Configuration for one Slurm-launched node rank in the debug file path."""

    run_id: str
    run_dir: str | Path
    metrics_json: str | Path | None
    train_args: Namespace
    node_rank: int
    node_count: int
    global_quorum: int
    local_steps: int
    local_worker_count: int = 1
    local_quorum: int = 1
    timeout_s: float = 900.0
    eta_outer: float = 1.0
    weight_by: str = "tokens"
    initial_checkpoint: str | Path | None = None
    synthetic_token_stream: bool = False
    synthetic_vocab_size: int = 256
    device: str = "cpu"
    walltime_remaining_s: float | None = None
    estimated_finalization_duration_s: float | None = None
    checkpoint_cadence: AsyncDiLoCoCheckpointCadence = field(default_factory=AsyncDiLoCoCheckpointCadence)


def default_tiny_e97_train_args(**overrides: Any) -> Namespace:
    """Return a CPU-friendly E97 Namespace compatible with train.py helpers."""

    values = {
        "level": "E97",
        "params": "100m",
        "dim": 8,
        "depth": 1,
        "data": None,
        "tokenizer": None,
        "seed": 42,
        "use_triton": 0,
        "bf16": False,
        "e88_raw_write": 0,
        "expansion": 1.0,
        "n_groups": 2,
        "n_state": 4,
        "n_slots": 4,
        "n_heads": 2,
        "top_k": None,
        "k_fast": None,
        "k_slow": None,
        "use_gate": 1,
        "gate_activation": "sigmoid",
        "linear_state": 1,
        "use_write_gate": 0,
        "e88_decay_mode": "mamba",
        "e88_value_residual": 0,
        "use_chunked_e97": 0,
        "e97_chunk_size": 4,
        "state_expansion": 2,
        "r_h_mode": "none",
        "use_conv": 0,
        "d_conv": 4,
        "gdn2_mlp_ratio": 0.0,
        "dropout": 0.0,
        "checkpoint_interval": 16,
        "gradient_checkpointing": False,
        "projection_chunk_size": 0,
        "loss_chunk_size": 0,
        "mlp_ratio": 0.0,
        "mlp_multiple": 8,
        "head_type_logits": None,
        "corner_mixture": None,
        "lam_max": None,
        "beta_max": None,
        "igain_max": None,
        "layer_kwargs": None,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "optimizer": "adamw",
        "knob_lr_mult": 1.0,
        "grad_accum": 1,
        "grad_clip": 1.0,
        "steps": 8,
        "min_lr_frac": 0.1,
        "tbptt": False,
        "batch_size": 2,
        "chunk_size": 8,
    }
    values.update(overrides)
    return Namespace(**values)


def run_real_async_diloco_file_rank(config: RealAsyncFileRankConfig) -> dict[str, Any]:
    """Run one actual node process and let rank 0 publish a quorum record.

    This path is intentionally bounded for Frontier debug validation: every
    Slurm task runs real local token training from the same seed checkpoint and
    writes durable per-node artifacts, while rank 0 merges node metadata through
    shared storage. It avoids claiming a dense tensor allreduce/storage exchange
    that this debug path does not implement.
    """

    if config.node_count <= 0:
        raise ValueError("node_count must be positive")
    if config.node_rank < 0 or config.node_rank >= config.node_count:
        raise ValueError("node_rank must be in [0, node_count)")
    if config.global_quorum <= 0 or config.global_quorum > config.node_count:
        raise ValueError("global_quorum must be in [1, node_count]")
    if config.local_worker_count <= 0:
        raise ValueError("local_worker_count must be positive")
    if config.local_quorum <= 0 or config.local_quorum > config.local_worker_count:
        raise ValueError("local_quorum must be in [1, local_worker_count]")
    if config.synthetic_token_stream:
        raise ValueError("synthetic_token_stream is disabled for actual multinode validation")

    run_dir = Path(config.run_dir)
    progress_dir = run_dir / "progress"
    nodes_dir = run_dir / "node_updates"
    progress_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir.mkdir(parents=True, exist_ok=True)

    node_id = f"node-{int(config.node_rank):05d}"
    generation = 0
    start_s = time.monotonic()
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="starting",
        generation=generation,
        extra={
            "node_count": config.node_count,
            "global_quorum": config.global_quorum,
            "local_worker_count": config.local_worker_count,
            "local_quorum": config.local_quorum,
            "device": config.device,
        },
    )

    train_args = _copy_train_args(config.train_args)
    train.normalize_training_args(train_args)
    torch.manual_seed(int(getattr(train_args, "seed", 42)))
    global_model = train.build_training_model(train_args)
    if config.initial_checkpoint is not None:
        train.load_checkpoint(str(config.initial_checkpoint), global_model)
    base_state = _floating_state_dict(global_model)
    del global_model

    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="checkpoint_loaded",
        generation=generation,
        extra={"base_state_bytes": state_num_bytes(base_state)},
    )

    worker_specs = tuple(
        RealAsyncWorkerSpec(
            worker_id=f"{node_id}/worker-{worker_idx:05d}",
            node_id=node_id,
            device=_rank_worker_device(config.device, worker_idx, config.local_worker_count),
            local_steps=config.local_steps,
            seed_offset=int(config.node_rank) * int(config.local_worker_count) + worker_idx,
        )
        for worker_idx in range(int(config.local_worker_count))
    )
    node_result = _run_real_node_supervisor(
        run_id=config.run_id,
        node_id=node_id,
        generation=generation,
        base_state=base_state,
        train_args=train_args,
        worker_specs=worker_specs,
        local_quorum=config.local_quorum,
        eta_outer=config.eta_outer,
        weight_by=config.weight_by,
        timeout_s=config.timeout_s,
        synthetic_token_stream=False,
        synthetic_vocab_size=config.synthetic_vocab_size,
        progress_dir=progress_dir,
        node_rank=config.node_rank,
    )
    node_payload = _node_result_payload(
        config,
        node_result,
        elapsed_s=max(0.0, time.monotonic() - start_s),
    )
    _atomic_write_json(nodes_dir / f"{node_id}.json", node_payload)
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="node_update_written",
        generation=generation,
        extra={
            "node_update_submitted": node_result.node_update is not None,
            "tokens": node_result.metrics.tokens_per_generation,
            "loss": node_result.metrics.loss_moving_average.get("loss"),
        },
    )

    root_payload: dict[str, Any] | None = None
    if int(config.node_rank) == 0:
        root_payload = _coordinate_file_rank_quorum(
            config=config,
            start_s=start_s,
            nodes_dir=nodes_dir,
            progress_dir=progress_dir,
            generation=generation,
        )

    return {
        "run_id": config.run_id,
        "node_rank": int(config.node_rank),
        "node_id": node_id,
        "node_update_path": str(nodes_dir / f"{node_id}.json"),
        "node_update_submitted": node_result.node_update is not None,
        "coordinator": int(config.node_rank) == 0,
        "global_result": root_payload,
    }


def run_real_async_diloco(config: RealAsyncDiLoCoConfig) -> RealAsyncDiLoCoRunResult:
    """Run one or more async DiLoCo generations using real train.py steps."""

    if config.generations <= 0:
        raise ValueError("generations must be positive")
    if not config.worker_specs:
        raise ValueError("at least one worker spec is required")

    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_args = _copy_train_args(config.train_args)
    train.normalize_training_args(train_args)
    worker_specs = tuple(config.worker_specs)
    local_quorum = (
        default_local_quorum(len(worker_specs))
        if config.local_quorum is None
        else int(config.local_quorum)
    )
    if local_quorum <= 0 or local_quorum > len(worker_specs):
        raise ValueError("local_quorum must be in [1, worker_count]")

    node_ids = tuple(sorted({spec.node_id for spec in worker_specs}))
    requested_nodes = (
        len(node_ids)
        if config.global_node_count is None
        else int(config.global_node_count)
    )
    global_quorum = (
        default_global_quorum(requested_nodes)
        if config.global_quorum is None
        else int(config.global_quorum)
    )
    if global_quorum <= 0 or global_quorum > requested_nodes:
        raise ValueError("global_quorum must be in [1, global_node_count]")

    torch.manual_seed(int(getattr(train_args, "seed", 42)))
    global_model = train.build_training_model(train_args)
    if config.initial_checkpoint is not None:
        train.load_checkpoint(str(config.initial_checkpoint), global_model)
    base_state = _floating_state_dict(global_model)
    latest_generation = int(config.initial_generation)
    manager = AsyncDiLoCoCheckpointManager(
        run_dir,
        run_id=config.run_id,
        role=GLOBAL_MERGER_ROLE,
        cadence=config.checkpoint_cadence,
    )

    node_results: list[RealAsyncNodeResult] = []
    global_results: list[RealAsyncGlobalResult] = []
    summary_metrics: list[AsyncDiLoCoGenerationMetrics] = []
    for offset in range(config.generations):
        generation = latest_generation + offset
        generation_start_s = time.monotonic()
        per_node_results: list[RealAsyncNodeResult] = []
        for node_id in node_ids:
            node_specs = tuple(spec for spec in worker_specs if spec.node_id == node_id)
            node_result = _run_real_node_supervisor(
                run_id=config.run_id,
                node_id=node_id,
                generation=generation,
                base_state=base_state,
                train_args=train_args,
                worker_specs=node_specs,
                local_quorum=min(local_quorum, len(node_specs)),
                eta_outer=config.eta_outer,
                weight_by=config.weight_by,
                timeout_s=config.timeout_s,
                synthetic_token_stream=config.synthetic_token_stream,
                synthetic_vocab_size=config.synthetic_vocab_size,
            )
            per_node_results.append(node_result)
            node_results.append(node_result)
            summary_metrics.append(node_result.metrics)

        global_result = _run_real_global_supervisor(
            run_id=config.run_id,
            generation=generation,
            base_state=base_state,
            node_results=per_node_results,
            requested_nodes=requested_nodes,
            global_quorum=global_quorum,
            eta_outer=config.eta_outer,
            weight_by=config.weight_by,
            generation_duration_s=max(0.0, time.monotonic() - generation_start_s),
            manager=manager,
            walltime_remaining_s=config.walltime_remaining_s,
            estimated_finalization_duration_s=config.estimated_finalization_duration_s,
        )
        global_results.append(global_result)
        summary_metrics.append(global_result.metrics)
        if global_result.metrics.latest_advanced:
            base_state = {name: tensor.detach().clone() for name, tensor in global_result.state.items()}

    summary = build_metrics_summary(
        run_id=config.run_id,
        requested_workers=len(worker_specs),
        participating_workers=sum(
            result.metrics.participating_workers for result in node_results
        ),
        generations=tuple(summary_metrics),
    )
    metrics_json = config.metrics_json
    if metrics_json is None:
        metrics_json = run_dir / "real_async_metrics.json"
    result = RealAsyncDiLoCoRunResult(
        run_id=config.run_id,
        generations=tuple(global_results),
        node_results=tuple(node_results),
        metrics_summary=summary,
        metrics_json=str(metrics_json),
        latest_path=str(manager.latest_path),
    )
    _write_json(metrics_json, result.to_dict())
    return result


def _run_real_node_supervisor(
    *,
    run_id: str,
    node_id: str,
    generation: int,
    base_state: Mapping[str, torch.Tensor],
    train_args: Namespace,
    worker_specs: Sequence[RealAsyncWorkerSpec],
    local_quorum: int,
    eta_outer: float,
    weight_by: str,
    timeout_s: float,
    synthetic_token_stream: bool,
    synthetic_vocab_size: int,
    progress_dir: str | Path | None = None,
    node_rank: int | None = None,
) -> RealAsyncNodeResult:
    start_s = time.monotonic()
    deadline_s = start_s + float(timeout_s)
    reports: list[RealAsyncWorkerReport] = []
    for spec in worker_specs:
        if _accepted_report_count(reports, generation) >= local_quorum:
            break
        if time.monotonic() >= deadline_s:
            break
        reports.append(_run_real_worker(
            run_id=run_id,
            generation=generation,
            base_state=base_state,
            train_args=train_args,
            spec=spec,
            synthetic_token_stream=synthetic_token_stream,
            synthetic_vocab_size=synthetic_vocab_size,
            progress_dir=progress_dir,
            node_rank=node_rank,
        ))

    reported = {report.worker_id for report in reports}
    for spec in worker_specs:
        if spec.worker_id in reported:
            continue
        reports.append(_status_worker_report(spec, generation, base_state, timed_out=True))

    reports = [_sanitize_report_for_quorum(report, base_state, generation) for report in reports]
    updates = [_report_update_or_status(report, base_state, generation) for report in reports]
    merge_start_s = time.monotonic()
    merge_result = quorum_merge(
        base_state,
        tuple(updates),
        run_id=run_id,
        generation=generation,
        requested_workers=len(worker_specs),
        quorum_threshold=local_quorum,
        eta_outer=eta_outer,
        weight_by=weight_by,
        generation_duration_s=max(0.0, time.monotonic() - start_s),
    )
    node_delta = compute_dense_delta(base_state, merge_result.state)
    metrics = _metrics_with_update_bytes(
        merge_result,
        worker_update_bytes=sum(
            state_num_bytes(update.delta)
            for update in merge_result.accepted_updates
        ),
        node_update_bytes=state_num_bytes(node_delta) if merge_result.advanced else 0,
        merge_start_s=merge_start_s,
    )
    node_update = None
    if merge_result.advanced:
        node_update = AsyncDiLoCoUpdate(
            worker_id=node_id,
            base_generation=generation,
            delta=node_delta,
            tokens=metrics.tokens_per_generation,
            local_steps=sum(update.local_steps for update in merge_result.accepted_updates),
            loss_moving_average=dict(metrics.loss_moving_average),
        )
    return RealAsyncNodeResult(
        node_id=node_id,
        generation=generation,
        node_update=node_update,
        worker_reports=tuple(sorted(reports, key=lambda report: report.worker_id)),
        metrics=metrics,
    )


def _run_real_global_supervisor(
    *,
    run_id: str,
    generation: int,
    base_state: Mapping[str, torch.Tensor],
    node_results: Sequence[RealAsyncNodeResult],
    requested_nodes: int,
    global_quorum: int,
    eta_outer: float,
    weight_by: str,
    generation_duration_s: float,
    manager: AsyncDiLoCoCheckpointManager,
    walltime_remaining_s: float | None,
    estimated_finalization_duration_s: float | None,
) -> RealAsyncGlobalResult:
    node_updates = [
        result.node_update for result in node_results
        if result.node_update is not None
    ]
    for idx in range(max(0, requested_nodes - len(node_updates))):
        node_updates.append(_status_update(f"missing-node-{idx}", generation, base_state, timed_out=True))
    merge_start_s = time.monotonic()
    merge_result = quorum_merge(
        base_state,
        tuple(node_updates),
        run_id=run_id,
        generation=generation,
        requested_workers=requested_nodes,
        quorum_threshold=global_quorum,
        eta_outer=eta_outer,
        weight_by=weight_by,
        generation_duration_s=generation_duration_s,
    )
    metrics = _metrics_with_update_bytes(
        merge_result,
        node_update_bytes=sum(
            state_num_bytes(update.delta)
            for update in merge_result.accepted_updates
        ),
        global_state_bytes=state_num_bytes(merge_result.state) if merge_result.advanced else 0,
        merge_start_s=merge_start_s,
    )
    publish_paths: tuple[str, ...] = ()
    if merge_result.advanced:
        publish_result = manager.publish_global_generation(
            metrics,
            walltime_remaining_s=walltime_remaining_s,
            estimated_finalization_duration_s=estimated_finalization_duration_s,
        )
        metrics = publish_result.metrics
        publish_paths = tuple(metrics.checkpoint_paths)
    return RealAsyncGlobalResult(
        generation=generation,
        state=merge_result.state,
        metrics=metrics,
        publish_paths=publish_paths,
    )


def _run_real_worker(
    *,
    run_id: str,
    generation: int,
    base_state: Mapping[str, torch.Tensor],
    train_args: Namespace,
    spec: RealAsyncWorkerSpec,
    synthetic_token_stream: bool,
    synthetic_vocab_size: int,
    progress_dir: str | Path | None = None,
    node_rank: int | None = None,
) -> RealAsyncWorkerReport:
    del run_id
    start_s = time.monotonic()
    if spec.fail_before_submit:
        return _status_worker_report(spec, generation, base_state, failed=True, error="configured failure before submit")
    if spec.timed_out:
        return _status_worker_report(spec, generation, base_state, timed_out=True, error="configured timeout before submit")

    try:
        args = _copy_train_args(train_args)
        torch.manual_seed(int(getattr(args, "seed", 42)) + int(spec.seed_offset))
        device = torch.device(spec.device)
        model = train.build_training_model(args).to(device)
        model.load_state_dict(base_state, strict=False)
        if bool(getattr(args, "bf16", False)):
            model = model.bfloat16()
        optimizer = train.build_training_optimizer(model, args)
        batch_iter = _build_batch_iter(
            args,
            rank=spec.seed_offset,
            device=device,
            synthetic=synthetic_token_stream,
            synthetic_vocab_size=synthetic_vocab_size,
        )
        losses: list[float] = []
        tokens = 0
        hidden_state = None
        current_step = -1
        for step in range(max(1, int(spec.local_steps))):
            current_step = step
            metrics = train.train_one_optimizer_step(
                model,
                optimizer,
                args,
                batch_iter=batch_iter,
                device=device,
                step=step,
                hidden_state=hidden_state,
            )
            hidden_state = metrics.get("hidden_state")
            loss = float(metrics["loss"])
            step_tokens = int(metrics["tokens_processed"])
            tokens += step_tokens
            if not math.isfinite(loss):
                error = f"non-finite loss at local_step={step}: {loss}"
                _write_worker_step_progress(
                    progress_dir,
                    node_rank=node_rank,
                    node_id=spec.node_id,
                    worker_id=spec.worker_id,
                    generation=generation,
                    local_step=step,
                    tokens=tokens,
                    elapsed_s=max(0.0, time.monotonic() - start_s),
                    loss=loss,
                    error=error,
                )
                return RealAsyncWorkerReport(
                    worker_id=spec.worker_id,
                    node_id=spec.node_id,
                    base_generation=generation,
                    update=None,
                    elapsed_s=max(0.0, time.monotonic() - start_s),
                    tokens=tokens,
                    losses=tuple(losses),
                    invalid=True,
                    error=error,
                )
            losses.append(loss)
            _write_worker_step_progress(
                progress_dir,
                node_rank=node_rank,
                node_id=spec.node_id,
                worker_id=spec.worker_id,
                generation=generation,
                local_step=step,
                tokens=tokens,
                elapsed_s=max(0.0, time.monotonic() - start_s),
                loss=loss,
            )
        worker_delta = _floating_delta_from_model(base_state, model)
        base_generation = generation if spec.stale_generation is None else int(spec.stale_generation)
        update = AsyncDiLoCoUpdate(
            worker_id=spec.worker_id,
            base_generation=base_generation,
            delta=worker_delta,
            tokens=tokens,
            local_steps=max(1, int(spec.local_steps)),
            loss_moving_average={
                "loss": _mean(losses),
                "loss_100": _mean(losses),
            },
        )
        nonfinite_reason = _update_nonfinite_reason(update)
        if nonfinite_reason is not None:
            return RealAsyncWorkerReport(
                worker_id=spec.worker_id,
                node_id=spec.node_id,
                base_generation=base_generation,
                update=None,
                elapsed_s=max(0.0, time.monotonic() - start_s),
                tokens=tokens,
                losses=tuple(losses),
                invalid=True,
                error=nonfinite_reason,
            )
        return RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=spec.node_id,
            base_generation=base_generation,
            update=update,
            elapsed_s=max(0.0, time.monotonic() - start_s),
            tokens=tokens,
            losses=tuple(losses),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_worker_step_progress(
            progress_dir,
            node_rank=node_rank,
            node_id=spec.node_id,
            worker_id=spec.worker_id,
            generation=generation,
            local_step=max(0, locals().get("current_step", 0)),
            tokens=locals().get("tokens", 0),
            elapsed_s=max(0.0, time.monotonic() - start_s),
            error=error,
        )
        return RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=spec.node_id,
            base_generation=generation,
            update=None,
            elapsed_s=max(0.0, time.monotonic() - start_s),
            tokens=locals().get("tokens", 0),
            failed=True,
            error=error,
        )


def _build_batch_iter(
    args: Namespace,
    *,
    rank: int,
    device: torch.device,
    synthetic: bool,
    synthetic_vocab_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if synthetic:
        return _synthetic_batches(args, rank=rank, device=device, vocab_size=synthetic_vocab_size)
    dataset = train.build_training_dataset(args, rank=rank, dist_enabled=True)

    def _iter() -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        while True:
            yield train.get_training_batch(dataset, args, device)

    return _iter()


def _rank_worker_device(base_device: str, worker_idx: int, local_worker_count: int) -> str:
    """Map local file-rank workers onto per-node devices when a GPU base is used."""

    device = str(base_device)
    if local_worker_count <= 1:
        return device
    if device in {"cuda", "hip"}:
        return f"cuda:{int(worker_idx)}"
    return device


def _synthetic_batches(
    args: Namespace,
    *,
    rank: int,
    device: torch.device,
    vocab_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(getattr(args, "seed", 42)) + 1009 * int(rank))
    batch_size = int(getattr(args, "batch_size", 1))
    seq_len = int(getattr(args, "chunk_size", 8)) + 1
    while True:
        chunks = torch.randint(
            low=0,
            high=max(2, int(vocab_size)),
            size=(batch_size, seq_len),
            generator=generator,
            dtype=torch.long,
        ).to(device)
        doc_end = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        yield chunks, doc_end, lengths


def _floating_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().cpu()
        for name, tensor in model.state_dict().items()
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
    }


def _floating_delta_from_model(
    base_state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Return a CPU dense delta without first cloning the full worker state."""

    worker_state = model.state_dict()
    missing = sorted(set(base_state) - set(worker_state))
    if missing:
        raise ValueError(f"worker state missing base tensors: {missing}")

    delta: dict[str, torch.Tensor] = {}
    for name, base_tensor in base_state.items():
        worker_tensor = worker_state[name]
        if not torch.is_tensor(worker_tensor) or not torch.is_floating_point(worker_tensor):
            raise ValueError(f"worker tensor {name!r} is not a floating tensor")
        if base_tensor.shape != worker_tensor.shape:
            raise ValueError(
                f"tensor {name!r} shape differs: {tuple(base_tensor.shape)} "
                f"!= {tuple(worker_tensor.shape)}")
        worker_cpu = worker_tensor.detach().to(device="cpu", dtype=base_tensor.dtype)
        delta[name] = worker_cpu - base_tensor.detach()
    return delta


def _status_worker_report(
    spec: RealAsyncWorkerSpec,
    generation: int,
    base_state: Mapping[str, torch.Tensor],
    *,
    failed: bool = False,
    timed_out: bool = False,
    invalid: bool = False,
    error: str | None = None,
) -> RealAsyncWorkerReport:
    return RealAsyncWorkerReport(
        worker_id=spec.worker_id,
        node_id=spec.node_id,
        base_generation=generation,
        update=_status_update(
            spec.worker_id,
            generation,
            base_state,
            failed=failed,
            timed_out=timed_out,
            invalid=invalid,
        ),
        elapsed_s=0.0,
        tokens=0,
        failed=failed,
        timed_out=timed_out,
        invalid=invalid,
        error=error,
    )


def _status_update(
    worker_id: str,
    generation: int,
    base_state: Mapping[str, torch.Tensor],
    *,
    failed: bool = False,
    timed_out: bool = False,
    invalid: bool = False,
) -> AsyncDiLoCoUpdate:
    return AsyncDiLoCoUpdate(
        worker_id=worker_id,
        base_generation=generation,
        delta={name: torch.zeros_like(tensor) for name, tensor in base_state.items()},
        tokens=1,
        local_steps=1,
        failed=failed,
        timed_out=timed_out,
        invalid=invalid,
    )


def _sanitize_report_for_quorum(
    report: RealAsyncWorkerReport,
    base_state: Mapping[str, torch.Tensor],
    generation: int,
) -> RealAsyncWorkerReport:
    reason = _update_nonfinite_reason(report.update)
    if reason is None:
        return report
    return RealAsyncWorkerReport(
        worker_id=report.worker_id,
        node_id=report.node_id,
        base_generation=report.base_generation,
        update=_status_update(report.worker_id, generation, base_state, invalid=True),
        elapsed_s=report.elapsed_s,
        tokens=report.tokens,
        losses=tuple(loss for loss in report.losses if math.isfinite(float(loss))),
        invalid=True,
        error=reason,
    )


def _report_update_or_status(
    report: RealAsyncWorkerReport,
    base_state: Mapping[str, torch.Tensor],
    generation: int,
) -> AsyncDiLoCoUpdate:
    if report.update is not None:
        return report.update
    return _status_update(
        report.worker_id,
        generation,
        base_state,
        failed=report.failed,
        timed_out=report.timed_out,
        invalid=report.invalid,
    )


def _metrics_with_update_bytes(
    merge_result: AsyncDiLoCoMergeResult,
    *,
    merge_start_s: float,
    worker_update_bytes: int | None = None,
    node_update_bytes: int | None = None,
    global_state_bytes: int | None = None,
) -> AsyncDiLoCoGenerationMetrics:
    update_bytes: dict[str, int] = {}
    if worker_update_bytes is not None:
        update_bytes["worker"] = int(worker_update_bytes)
    if node_update_bytes is not None:
        update_bytes["node"] = int(node_update_bytes)
    if global_state_bytes is not None:
        update_bytes["global_state"] = int(global_state_bytes)
    return dataclass_replace_metrics(
        merge_result.metrics,
        merge_duration_s=max(0.0, time.monotonic() - merge_start_s),
        update_bytes=update_bytes,
    )


def dataclass_replace_metrics(
    metrics: AsyncDiLoCoGenerationMetrics,
    **changes: Any,
) -> AsyncDiLoCoGenerationMetrics:
    return replace(metrics, **changes)


def _accepted_report_count(
    reports: Sequence[RealAsyncWorkerReport],
    generation: int,
) -> int:
    return sum(
        1 for report in reports
        if report.update is not None
        and not report.failed
        and not report.timed_out
        and not report.invalid
        and report.update.base_generation == generation
    )


def _copy_train_args(args: Namespace) -> Namespace:
    return copy.deepcopy(args)


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def _finite_json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _update_nonfinite_reason(update: AsyncDiLoCoUpdate | None) -> str | None:
    if update is None:
        return None
    for key, value in update.loss_moving_average.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return f"loss_moving_average[{key!r}] is not numeric: {value!r}"
        if not math.isfinite(scalar):
            return f"loss_moving_average[{key!r}] is non-finite: {scalar}"
    for name, tensor in update.delta.items():
        if not torch.is_tensor(tensor):
            return f"delta tensor {name!r} is not a tensor"
        if not torch.isfinite(tensor.detach()).all().item():
            return f"delta tensor {name!r} contains non-finite values"
    return None


def _write_worker_step_progress(
    progress_dir: str | Path | None,
    *,
    node_rank: int | None,
    node_id: str,
    worker_id: str,
    generation: int,
    local_step: int,
    tokens: int,
    elapsed_s: float,
    loss: Any = None,
    error: str | None = None,
) -> None:
    if progress_dir is None:
        return
    safe_loss = _finite_json_float(loss)
    payload = {
        "schema_version": 1,
        "node_rank": (None if node_rank is None else int(node_rank)),
        "node_id": node_id,
        "worker_id": worker_id,
        "generation": int(generation),
        "local_step": int(local_step),
        "tokens": int(tokens),
        "elapsed_s": float(max(0.0, elapsed_s)),
        "loss": safe_loss,
        "loss_finite": safe_loss is not None,
        "tokens_finite": math.isfinite(float(tokens)),
        "elapsed_finite": math.isfinite(float(elapsed_s)),
        "failed": error is not None,
        "invalid": error is not None,
        "error": error,
    }
    name = (
        f"{node_id}.{worker_id.replace('/', '_')}"
        f".gen{int(generation):06d}.step{int(local_step):06d}.json"
    )
    _atomic_write_json(Path(progress_dir) / "steps" / name, payload)


def _node_result_payload(
    config: RealAsyncFileRankConfig,
    node_result: RealAsyncNodeResult,
    *,
    elapsed_s: float,
) -> dict[str, Any]:
    losses = [
        loss
        for report in node_result.worker_reports
        for loss in report.losses
        if math.isfinite(float(loss))
    ]
    invalid_reasons = [
        report.error
        for report in node_result.worker_reports
        if report.invalid and report.error
    ]
    failed_reasons = [
        report.error
        for report in node_result.worker_reports
        if report.failed and report.error
    ]
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "node_rank": int(config.node_rank),
        "node_id": node_result.node_id,
        "generation": int(node_result.generation),
        "elapsed_s": float(elapsed_s),
        "node_update_submitted": node_result.node_update is not None,
        "bounded_debug_update_kind": "metadata_quorum_no_dense_delta_storage",
        "metrics": node_result.metrics.to_dict(),
        "tokens": int(node_result.metrics.tokens_per_generation),
        "loss": (_finite_json_float(_mean(losses)) if losses else None),
        "loss_finite": bool(losses),
        "losses": [float(loss) for loss in losses],
        "invalid_reasons": invalid_reasons,
        "failed_reasons": failed_reasons,
        "worker_reports": [
            {
                "worker_id": report.worker_id,
                "node_id": report.node_id,
                "base_generation": report.base_generation,
                "tokens": report.tokens,
                "losses": [
                    float(loss)
                    for loss in report.losses
                    if math.isfinite(float(loss))
                ],
                "failed": report.failed,
                "timed_out": report.timed_out,
                "invalid": report.invalid,
                "error": report.error,
            }
            for report in node_result.worker_reports
        ],
    }


def _coordinate_file_rank_quorum(
    *,
    config: RealAsyncFileRankConfig,
    start_s: float,
    nodes_dir: Path,
    progress_dir: Path,
    generation: int,
) -> dict[str, Any]:
    deadline_s = start_s + float(config.timeout_s)
    metrics_path = Path(config.metrics_json) if config.metrics_json is not None else Path(config.run_dir) / "real_async_metrics.json"
    accepted: list[dict[str, Any]] = []
    all_payloads: list[dict[str, Any]] = []
    last_partial_write_s = 0.0

    while time.monotonic() < deadline_s:
        all_payloads = _read_node_payloads(nodes_dir)
        accepted = [
            payload for payload in all_payloads
            if payload.get("node_update_submitted") is True
            and int(payload.get("generation", -1)) == generation
        ]
        partial = _file_quorum_payload(
            config=config,
            generation=generation,
            accepted=accepted,
            all_payloads=all_payloads,
            start_s=start_s,
            latest_advanced=False,
            checkpoint_paths=(),
            checkpoint_sizes={},
        )
        now_s = time.monotonic()
        if now_s - last_partial_write_s >= 5.0 or len(accepted) >= config.global_quorum:
            _atomic_write_json(metrics_path, partial)
            _write_rank_heartbeat(
                progress_dir,
                node_rank=config.node_rank,
                node_id="node-00000",
                stage="coordinator_waiting",
                generation=generation,
                extra={
                    "accepted_nodes": len(accepted),
                    "seen_nodes": len(all_payloads),
                    "global_quorum": config.global_quorum,
                    "partial_metrics_json": str(metrics_path),
                },
            )
            last_partial_write_s = now_s
        if len(accepted) >= config.global_quorum:
            break
        time.sleep(1.0)

    timed_out = max(0, int(config.node_count) - len(all_payloads))
    failed = sum(1 for payload in all_payloads if payload.get("node_update_submitted") is not True)
    invalid = sum(1 for payload in all_payloads if payload.get("invalid_reasons"))
    metrics = _file_quorum_metrics(
        config=config,
        generation=generation,
        accepted=accepted,
        all_payloads=all_payloads,
        start_s=start_s,
        timed_out=timed_out,
        failed=failed,
        invalid=invalid,
    )

    checkpoint_paths: tuple[str, ...] = ()
    checkpoint_sizes: dict[str, int] = {}
    metadata_quorum_reached = len(accepted) >= config.global_quorum
    latest_advanced = False

    final_payload = _file_quorum_payload(
        config=config,
        generation=generation,
        accepted=accepted,
        all_payloads=all_payloads,
        start_s=start_s,
        latest_advanced=latest_advanced,
        checkpoint_paths=checkpoint_paths,
        checkpoint_sizes=checkpoint_sizes,
        metrics=metrics,
        metadata_quorum_reached=metadata_quorum_reached,
    )
    _atomic_write_json(metrics_path, final_payload)
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id="node-00000",
        stage="coordinator_finalized" if latest_advanced else "coordinator_deferred",
        generation=generation,
        extra={
            "accepted_nodes": len(accepted),
            "seen_nodes": len(all_payloads),
            "global_quorum": config.global_quorum,
            "latest_advanced": latest_advanced,
            "metadata_quorum_reached": metadata_quorum_reached,
            "no_go_reason": _FILE_QUORUM_NO_GO_REASON,
        },
    )
    return final_payload


def _read_node_payloads(nodes_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(nodes_dir.glob("node-*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _file_quorum_metrics(
    *,
    config: RealAsyncFileRankConfig,
    generation: int,
    accepted: Sequence[Mapping[str, Any]],
    all_payloads: Sequence[Mapping[str, Any]],
    start_s: float,
    timed_out: int,
    failed: int,
    invalid: int,
) -> AsyncDiLoCoGenerationMetrics:
    duration_s = max(0.0, time.monotonic() - start_s)
    tokens = sum(int(payload.get("tokens", 0)) for payload in accepted)
    losses = [
        float(payload["loss"])
        for payload in accepted
        if payload.get("loss") is not None and math.isfinite(float(payload["loss"]))
    ]
    metadata_quorum_reached = len(accepted) >= int(config.global_quorum)
    return AsyncDiLoCoGenerationMetrics(
        run_id=config.run_id,
        generation=int(generation),
        requested_workers=int(config.node_count),
        participating_workers=len(all_payloads),
        quorum_threshold=int(config.global_quorum),
        quorum_size=len(accepted),
        accepted_updates=len(accepted),
        stale_updates=0,
        timed_out_updates=int(timed_out),
        failed_updates=int(failed),
        invalid_updates=int(invalid),
        generation_duration_s=duration_s,
        merge_duration_s=0.0,
        rebase_duration_s=0.0,
        checkpoint_duration_s=0.0,
        tokens_per_sec=(float(tokens) / duration_s if duration_s > 0.0 else 0.0),
        tokens_per_generation=int(tokens),
        update_bytes={"node_metadata": sum(len(stable_json_dumps(payload)) for payload in accepted)},
        loss_moving_average={"loss": _mean(losses), "loss_100": _mean(losses)} if losses else {},
        update_norms={},
        latest_advanced=False,
        quorum_status=("metadata_quorum_no_dense_delta" if metadata_quorum_reached else "deferred"),
    )


def _file_quorum_payload(
    *,
    config: RealAsyncFileRankConfig,
    generation: int,
    accepted: Sequence[Mapping[str, Any]],
    all_payloads: Sequence[Mapping[str, Any]],
    start_s: float,
    latest_advanced: bool,
    checkpoint_paths: Sequence[str],
    checkpoint_sizes: Mapping[str, int],
    metrics: AsyncDiLoCoGenerationMetrics | None = None,
    metadata_quorum_reached: bool | None = None,
) -> dict[str, Any]:
    timed_out = max(0, int(config.node_count) - len(all_payloads))
    failed = sum(1 for payload in all_payloads if payload.get("node_update_submitted") is not True)
    invalid = sum(1 for payload in all_payloads if payload.get("invalid_reasons"))
    if metrics is None:
        metrics = _file_quorum_metrics(
            config=config,
            generation=generation,
            accepted=accepted,
            all_payloads=all_payloads,
            start_s=start_s,
            timed_out=timed_out,
            failed=failed,
            invalid=invalid,
        )
    if metadata_quorum_reached is None:
        metadata_quorum_reached = len(accepted) >= int(config.global_quorum)
    metrics_dict = metrics.to_dict()
    if checkpoint_paths:
        metrics_dict["checkpoint_paths"] = list(checkpoint_paths)
        metrics_dict["checkpoint_sizes"] = dict(checkpoint_sizes)
        metrics_dict["latest_advanced"] = bool(latest_advanced)
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "mode": "actual_multinode_file_quorum_debug",
        "bounded_debug_alternative": {
            "dense_delta_exchange": "not_implemented_for_debug_shared_storage",
            "proof": "one Slurm-launched process per node runs real local token training and rank 0 merges node metadata quorum",
            "no_go_for_async_diloco_claims": True,
            "no_go_reason": _FILE_QUORUM_NO_GO_REASON,
        },
        "dense_delta_exchange": "not_implemented_for_debug_shared_storage",
        "metadata_quorum_reached": bool(metadata_quorum_reached),
        "no_go_for_async_diloco_claims": True,
        "no_go_reason": _FILE_QUORUM_NO_GO_REASON,
        "synthetic_token_stream": False,
        "node_count": int(config.node_count),
        "global_quorum": int(config.global_quorum),
        "generation": int(generation),
        "latest_generation": (int(generation) if latest_advanced else -1),
        "latest_path": str(Path(config.run_dir) / "latest.json"),
        "metrics_summary": build_metrics_summary(
            run_id=config.run_id,
            requested_workers=int(config.node_count),
            participating_workers=len(all_payloads),
            generations=(metrics,),
        ).to_dict(),
        "global_generations": [{
            "generation": int(generation),
            "metrics": metrics_dict,
            "publish_paths": list(checkpoint_paths),
        }],
        "node_generations": list(all_payloads),
        "accepted_node_ids": [str(payload.get("node_id")) for payload in accepted],
        "partial": not latest_advanced,
    }


def _write_rank_heartbeat(
    progress_dir: Path,
    *,
    node_rank: int,
    node_id: str,
    stage: str,
    generation: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "node_rank": int(node_rank),
        "node_id": node_id,
        "stage": stage,
        "generation": int(generation),
        "monotonic_s": time.monotonic(),
    }
    if extra:
        payload.update(dict(extra))
    _atomic_write_json(progress_dir / f"{node_id}.heartbeat.json", payload)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    tmp.replace(target)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")


__all__ = [
    "RealAsyncDiLoCoConfig",
    "RealAsyncDiLoCoRunResult",
    "RealAsyncFileRankConfig",
    "RealAsyncGlobalResult",
    "RealAsyncNodeResult",
    "RealAsyncWorkerReport",
    "RealAsyncWorkerSpec",
    "default_tiny_e97_train_args",
    "run_real_async_diloco",
    "run_real_async_diloco_file_rank",
]
