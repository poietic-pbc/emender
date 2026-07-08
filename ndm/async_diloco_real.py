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
import socket
import struct
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
    RESILIENT_QUORUM_DILOCO_MODE,
    STRICT_COLLECTIVE_DILOCO_MODE,
    build_metrics_summary,
    compute_dense_delta,
    default_global_quorum,
    default_local_quorum,
    quorum_merge,
    stable_json_dumps,
    state_num_bytes,
)
from ndm.async_diloco_compiled_mpich import (
    COMPILED_MPICH_TRANSPORT,
    CompiledMpichHelperConfig,
    run_compiled_mpich_dense_quorum,
)
from ndm.async_diloco_mpi import run_mpi_dense_quorum

import train


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
    quorum_mode: str = RESILIENT_QUORUM_DILOCO_MODE
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
    """Configuration for one Slurm-launched rank in the TCP quorum path.

    The class name is kept for compatibility with existing callers. The live
    quorum path no longer scans shared-storage update files; workers submit
    per-rank metadata to a coordinator TCP socket.
    """

    run_id: str
    run_dir: str | Path
    metrics_json: str | Path | None
    train_args: Namespace
    node_rank: int
    node_count: int
    global_quorum: int
    local_steps: int
    timeout_s: float = 900.0
    quorum_mode: str = RESILIENT_QUORUM_DILOCO_MODE
    eta_outer: float = 1.0
    weight_by: str = "tokens"
    initial_checkpoint: str | Path | None = None
    synthetic_token_stream: bool = False
    allow_synthetic_token_stream: bool = False
    synthetic_vocab_size: int = 256
    device: str = "cpu"
    coordinator_host: str = "127.0.0.1"
    coordinator_bind_host: str = "0.0.0.0"
    coordinator_port: int = 29497
    connect_retry_interval_s: float = 0.2
    transport: str = "tcp"
    transport_selector: str = ""
    transport_approval_class: str = ""
    production_approval_eligible: bool = False
    allow_tcp_scale_debug: bool = False
    mpi_bucket_bytes: int = 64 * 1024 * 1024
    compiled_mpich_helper_bin: str | Path | None = None
    compiled_mpich_ipc_dir: str | Path | None = None
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
    """Run one actual process and let rank 0 publish a quorum record.

    The TCP mode is intentionally bounded for Frontier debug validation: every
    Slurm task runs real local token training from the same seed checkpoint and
    submits per-rank metadata to a rank-0 TCP coordinator. Durable files are
    written only as metrics/checkpoint/post-run artifacts, not as the live
    quorum/update collection path.

    The MPI mode is the dense data-plane path: every rank packs its node delta
    into checksummed buckets and sends those bytes over nonblocking MPI
    point-to-point. Shared storage remains limited to metrics/checkpoints.
    """

    if config.node_count <= 0:
        raise ValueError("node_count must be positive")
    if config.node_rank < 0 or config.node_rank >= config.node_count:
        raise ValueError("node_rank must be in [0, node_count)")
    if config.global_quorum <= 0 or config.global_quorum > config.node_count:
        raise ValueError("global_quorum must be in [1, node_count]")
    if config.synthetic_token_stream and not config.allow_synthetic_token_stream:
        raise ValueError("synthetic_token_stream is disabled for actual multinode validation")

    run_dir = Path(config.run_dir)
    transport = str(config.transport).strip().lower()
    if transport not in {"tcp", "mpi-dense", COMPILED_MPICH_TRANSPORT}:
        raise ValueError(
            "transport must be 'tcp', 'mpi-dense', or "
            f"'{COMPILED_MPICH_TRANSPORT}'"
        )
    quorum_mode = str(config.quorum_mode).strip().lower()
    if quorum_mode not in {RESILIENT_QUORUM_DILOCO_MODE, STRICT_COLLECTIVE_DILOCO_MODE}:
        raise ValueError("quorum_mode must be 'resilient_quorum' or 'strict_collective'")
    if quorum_mode == STRICT_COLLECTIVE_DILOCO_MODE and transport != COMPILED_MPICH_TRANSPORT:
        raise ValueError("strict_collective quorum_mode requires compiled MPICH transport")
    if transport == "tcp" and int(config.node_count) > 8:
        if not config.allow_tcp_scale_debug:
            raise ValueError(
                "TCP async quorum transport is local/debug-only; pass the explicit "
                "small-debug override for nonlocal TCP runs"
            )

    progress_dir = run_dir / "progress"
    nodes_dir = run_dir / "node_update_artifacts"
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
        extra={"node_count": config.node_count, "global_quorum": config.global_quorum},
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

    spec = RealAsyncWorkerSpec(
        worker_id=f"{node_id}/worker-00000",
        node_id=node_id,
        device=config.device,
        local_steps=config.local_steps,
        seed_offset=config.node_rank,
    )
    node_result = _run_real_node_supervisor(
        run_id=config.run_id,
        node_id=node_id,
        generation=generation,
        base_state=base_state,
        train_args=train_args,
        worker_specs=(spec,),
        local_quorum=1,
        eta_outer=config.eta_outer,
        weight_by=config.weight_by,
        timeout_s=config.timeout_s,
        quorum_mode=config.quorum_mode,
        synthetic_token_stream=config.synthetic_token_stream,
        synthetic_vocab_size=config.synthetic_vocab_size,
    )
    node_payload = _node_result_payload(
        config,
        node_result,
        elapsed_s=max(0.0, time.monotonic() - start_s),
    )
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="node_update_ready",
        generation=generation,
        extra={
            "node_update_submitted": node_result.node_update is not None,
            "tokens": node_result.metrics.tokens_per_generation,
            "loss": node_result.metrics.loss_moving_average.get("loss"),
            "transport": transport,
        },
    )

    root_payload: dict[str, Any] | None = None
    if transport == COMPILED_MPICH_TRANSPORT:
        root_payload = _coordinate_compiled_mpich_dense_rank(
            config=config,
            start_s=start_s,
            base_state=base_state,
            node_result=node_result,
            artifact_dir=nodes_dir,
            progress_dir=progress_dir,
            generation=generation,
        )
    elif transport == "mpi-dense":
        root_payload = _coordinate_mpi_dense_rank(
            config=config,
            start_s=start_s,
            base_state=base_state,
            node_result=node_result,
            artifact_dir=nodes_dir,
            progress_dir=progress_dir,
            generation=generation,
        )
    elif int(config.node_rank) == 0:
        root_payload = _coordinate_network_rank_quorum(
            config=config,
            start_s=start_s,
            own_payload=node_payload,
            artifact_dir=nodes_dir,
            progress_dir=progress_dir,
            generation=generation,
        )
    else:
        submit = _submit_network_rank_payload(config, node_payload, start_s=start_s)
        node_payload = {
            **node_payload,
            "transport_submit_latency_s": submit["submit_latency_s"],
            "transport_bytes_sent": submit["bytes_sent"],
        }
        _atomic_write_json(nodes_dir / f"{node_id}.json", node_payload)
        _write_rank_heartbeat(
            progress_dir,
            node_rank=config.node_rank,
            node_id=node_id,
            stage="node_update_submitted",
            generation=generation,
            extra=submit,
        )

    return {
        "run_id": config.run_id,
        "node_rank": int(config.node_rank),
        "node_id": node_id,
        "node_update_path": str(nodes_dir / f"{node_id}.json"),
        "node_update_submitted": node_result.node_update is not None,
        "coordinator": int(config.node_rank) == 0,
        "transport": transport,
        "transport_selector": config.transport_selector or transport,
        "transport_approval_class": (
            config.transport_approval_class
            or ("frontier-production-candidate" if transport == COMPILED_MPICH_TRANSPORT else "tcp-debug-only")
        ),
        "production_approval_eligible": bool(config.production_approval_eligible),
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
                quorum_mode=config.quorum_mode,
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
            quorum_mode=config.quorum_mode,
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
    quorum_mode: str,
    synthetic_token_stream: bool,
    synthetic_vocab_size: int,
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
        ))

    reported = {report.worker_id for report in reports}
    for spec in worker_specs:
        if spec.worker_id in reported:
            continue
        reports.append(_status_worker_report(spec, generation, base_state, timed_out=True))

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
        mode=quorum_mode,
        checkpoint_state_id=f"{run_id}:{node_id}:gen{int(generation):06d}",
        missing_worker_ids=tuple(
            report.worker_id for report in reports if report.timed_out
        ),
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
            update_id=f"{node_id}:gen{int(generation):06d}",
            global_generation=generation,
            checkpoint_state_id=metrics.checkpoint_state_id,
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
    quorum_mode: str,
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
        mode=quorum_mode,
        checkpoint_state_id=f"{run_id}:global:gen{int(generation):06d}",
        missing_worker_ids=tuple(
            update.worker_id for update in node_updates if update.timed_out
        ),
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
        for step in range(max(1, int(spec.local_steps))):
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
            losses.append(float(metrics["loss"]))
            tokens += int(metrics["tokens_processed"])
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
        return RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=spec.node_id,
            base_generation=generation,
            update=None,
            elapsed_s=max(0.0, time.monotonic() - start_s),
            tokens=0,
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
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
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "avg": float(sum(ordered) / len(ordered)),
    }


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
        "loss": (_mean(losses) if losses else None),
        "losses": [float(loss) for loss in losses],
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
            for report in node_result.worker_reports
        ],
    }


def _submit_network_rank_payload(
    config: RealAsyncFileRankConfig,
    payload: Mapping[str, Any],
    *,
    start_s: float,
) -> dict[str, Any]:
    deadline_s = start_s + float(config.timeout_s)
    retry_s = max(0.05, float(config.connect_retry_interval_s))
    last_error: str | None = None
    submit_start_s = time.monotonic()
    wire_payload = {
        **dict(payload),
        "transport": "tcp",
        "transport_submit_wall_s": time.time(),
    }
    payload_bytes = stable_json_dumps(wire_payload).encode("utf-8")
    frame = struct.pack("!Q", len(payload_bytes)) + payload_bytes
    while time.monotonic() < deadline_s:
        try:
            with socket.create_connection(
                (str(config.coordinator_host), int(config.coordinator_port)),
                timeout=min(5.0, max(0.1, deadline_s - time.monotonic())),
            ) as sock:
                sock.sendall(frame)
                ack_size = _recv_exact(sock, 8)
                ack_len = struct.unpack("!Q", ack_size)[0]
                ack = json.loads(_recv_exact(sock, ack_len).decode("utf-8"))
                if ack.get("ok") is not True:
                    raise RuntimeError(str(ack.get("error", "coordinator rejected update")))
                return {
                    "coordinator_host": str(config.coordinator_host),
                    "coordinator_port": int(config.coordinator_port),
                    "submit_latency_s": max(0.0, time.monotonic() - submit_start_s),
                    "bytes_sent": len(frame),
                    "ack": ack,
                }
        except (OSError, RuntimeError, json.JSONDecodeError, struct.error) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(retry_s, max(0.0, deadline_s - time.monotonic())))
    raise TimeoutError(
        f"timed out submitting rank payload to tcp coordinator "
        f"{config.coordinator_host}:{config.coordinator_port}: {last_error}"
    )


def _coordinate_network_rank_quorum(
    *,
    config: RealAsyncFileRankConfig,
    start_s: float,
    own_payload: Mapping[str, Any],
    artifact_dir: Path,
    progress_dir: Path,
    generation: int,
) -> dict[str, Any]:
    deadline_s = start_s + float(config.timeout_s)
    metrics_path = Path(config.metrics_json) if config.metrics_json is not None else Path(config.run_dir) / "real_async_metrics.json"
    received_by_node: dict[str, dict[str, Any]] = {}
    own_wire_bytes = len(stable_json_dumps(own_payload).encode("utf-8")) + 8
    own = {
        **dict(own_payload),
        "transport": "tcp",
        "transport_submit_latency_s": 0.0,
        "transport_bytes_sent": own_wire_bytes,
        "transport_submit_wall_s": time.time(),
    }
    received_by_node[str(own.get("node_id"))] = own
    _atomic_write_json(artifact_dir / f"{own.get('node_id')}.json", own)
    last_partial_write_s = 0.0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((str(config.coordinator_bind_host), int(config.coordinator_port)))
        server.listen(max(1, int(config.node_count)))
        server.settimeout(0.2)
        _write_rank_heartbeat(
            progress_dir,
            node_rank=config.node_rank,
            node_id="node-00000",
            stage="coordinator_listening",
            generation=generation,
            extra={
                "transport": "tcp",
                "bind_host": str(config.coordinator_bind_host),
                "coordinator_port": int(config.coordinator_port),
                "global_quorum": int(config.global_quorum),
            },
        )
        while time.monotonic() < deadline_s:
            all_payloads = _sorted_node_payloads(received_by_node.values())
            accepted = _accepted_network_payloads(all_payloads, generation)
            partial = _network_quorum_payload(
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
                        "transport": "tcp",
                        "accepted_nodes": len(accepted),
                        "seen_nodes": len(all_payloads),
                        "global_quorum": config.global_quorum,
                        "partial_metrics_json": str(metrics_path),
                    },
                )
                last_partial_write_s = now_s
            if len(accepted) >= config.global_quorum:
                break
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    payload = _recv_framed_json(conn)
                    node_id = str(payload.get("node_id", ""))
                    if not node_id:
                        raise ValueError("payload missing node_id")
                    received = {
                        **payload,
                        "transport": "tcp",
                        "transport_receive_wall_s": time.time(),
                        "transport_receive_latency_s": max(
                            0.0,
                            time.time() - float(payload.get("transport_submit_wall_s", time.time())),
                        ),
                        "transport_bytes_sent": int(
                            payload.get("transport_bytes_sent")
                            or (len(stable_json_dumps(payload).encode("utf-8")) + 8)
                        ),
                    }
                    received.setdefault(
                        "transport_submit_latency_s",
                        received["transport_receive_latency_s"],
                    )
                    received_by_node[node_id] = received
                    _atomic_write_json(artifact_dir / f"{node_id}.json", received)
                    _send_framed_json(conn, {"ok": True, "node_id": node_id})
                except Exception as exc:
                    _send_framed_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    all_payloads = _sorted_node_payloads(received_by_node.values())
    accepted = _accepted_network_payloads(all_payloads, generation)
    timed_out = max(0, int(config.node_count) - len(all_payloads))
    failed = sum(1 for payload in all_payloads if payload.get("node_update_submitted") is not True)
    metrics = _network_quorum_metrics(
        config=config,
        generation=generation,
        accepted=accepted,
        all_payloads=all_payloads,
        start_s=start_s,
        timed_out=timed_out,
        failed=failed,
    )

    checkpoint_paths: tuple[str, ...] = ()
    checkpoint_sizes: dict[str, int] = {}
    latest_advanced = False
    if len(accepted) >= config.global_quorum:
        manager = AsyncDiLoCoCheckpointManager(
            config.run_dir,
            run_id=config.run_id,
            role=GLOBAL_MERGER_ROLE,
            cadence=config.checkpoint_cadence,
        )
        publish = manager.publish_global_generation(
            metrics,
            walltime_remaining_s=config.walltime_remaining_s,
            estimated_finalization_duration_s=config.estimated_finalization_duration_s,
        )
        metrics = publish.metrics
        checkpoint_paths = tuple(metrics.checkpoint_paths)
        checkpoint_sizes = dict(metrics.checkpoint_sizes)
        latest_advanced = True

    final_payload = _network_quorum_payload(
        config=config,
        generation=generation,
        accepted=accepted,
        all_payloads=all_payloads,
        start_s=start_s,
        latest_advanced=latest_advanced,
        checkpoint_paths=checkpoint_paths,
        checkpoint_sizes=checkpoint_sizes,
        metrics=metrics,
    )
    _atomic_write_json(metrics_path, final_payload)
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id="node-00000",
        stage="coordinator_finalized" if latest_advanced else "coordinator_deferred",
        generation=generation,
        extra={
            "transport": "tcp",
            "accepted_nodes": len(accepted),
            "seen_nodes": len(all_payloads),
            "global_quorum": config.global_quorum,
            "latest_advanced": latest_advanced,
        },
    )
    return final_payload


def _coordinate_mpi_dense_rank(
    *,
    config: RealAsyncFileRankConfig,
    start_s: float,
    base_state: Mapping[str, torch.Tensor],
    node_result: RealAsyncNodeResult,
    artifact_dir: Path,
    progress_dir: Path,
    generation: int,
) -> dict[str, Any] | None:
    metrics_path = Path(config.metrics_json) if config.metrics_json is not None else Path(config.run_dir) / "real_async_metrics.json"
    node_id = f"node-{int(config.node_rank):05d}"
    if node_result.node_update is None:
        local_update = _status_update(node_id, generation, base_state, failed=True)
    else:
        local_update = node_result.node_update
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="mpi_dense_send_starting",
        generation=generation,
        extra={
            "transport": "mpi-dense",
            "global_quorum": int(config.global_quorum),
            "mpi_bucket_bytes": int(config.mpi_bucket_bytes),
        },
    )
    payload = run_mpi_dense_quorum(
        base_state=base_state,
        local_update=local_update,
        run_id=config.run_id,
        generation=generation,
        requested_ranks=config.node_count,
        quorum=config.global_quorum,
        timeout_s=config.timeout_s,
        bucket_bytes=config.mpi_bucket_bytes,
        base_checkpoint=(None if config.initial_checkpoint is None else str(config.initial_checkpoint)),
    )
    _atomic_write_json(artifact_dir / f"{node_id}.json", {
        "schema_version": 1,
        "run_id": config.run_id,
        "node_rank": int(config.node_rank),
        "node_id": node_id,
        "generation": int(generation),
        "node_update_submitted": node_result.node_update is not None,
        "transport": "mpi-dense",
        "dense_delta_bytes": state_num_bytes(local_update.delta),
        "metrics": node_result.metrics.to_dict(),
    })
    if int(config.node_rank) != 0:
        _write_rank_heartbeat(
            progress_dir,
            node_rank=config.node_rank,
            node_id=node_id,
            stage="mpi_dense_result_received",
            generation=generation,
            extra={"transport": "mpi-dense"},
        )
        return None

    if payload is None:
        raise RuntimeError("MPI dense root did not produce a quorum payload")
    global_metrics_payload = ((payload.get("global_generations") or [{}])[0].get("metrics") or {})
    metrics = AsyncDiLoCoGenerationMetrics.from_dict(global_metrics_payload)
    latest_advanced = False
    checkpoint_paths: tuple[str, ...] = ()
    if metrics.quorum_status == "advanced":
        manager = AsyncDiLoCoCheckpointManager(
            config.run_dir,
            run_id=config.run_id,
            role=GLOBAL_MERGER_ROLE,
            cadence=config.checkpoint_cadence,
        )
        publish = manager.publish_global_generation(
            metrics,
            walltime_remaining_s=config.walltime_remaining_s,
            estimated_finalization_duration_s=config.estimated_finalization_duration_s,
        )
        metrics = publish.metrics
        checkpoint_paths = tuple(metrics.checkpoint_paths)
        latest_advanced = True
    final_payload = {
        **payload,
        "mode": "actual_multinode_mpi_dense_quorum",
        "transport": {
            **dict(payload.get("transport") or {}),
            "name": "mpi-dense",
            "selector": config.transport_selector or "mpi-dense",
            "actual": "mpi-dense",
            "approval_class": config.transport_approval_class or "legacy-comparison-only",
            "production_approval_eligible": False,
        },
        "production_approval_eligible": False,
        "node_count": int(config.node_count),
        "global_quorum": int(config.global_quorum),
        "generation": int(generation),
        "latest_generation": (int(generation) if latest_advanced else -1),
        "latest_path": str(Path(config.run_dir) / "latest.json"),
        "partial": not latest_advanced,
        "global_generations": [{
            "generation": int(generation),
            "metrics": metrics.to_dict(),
            "publish_paths": list(checkpoint_paths),
        }],
        "metrics_summary": build_metrics_summary(
            run_id=config.run_id,
            requested_workers=int(config.node_count),
            participating_workers=int(metrics.participating_workers),
            generations=(metrics,),
        ).to_dict(),
    }
    _atomic_write_json(metrics_path, final_payload)
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="mpi_dense_coordinator_finalized" if latest_advanced else "mpi_dense_coordinator_deferred",
        generation=generation,
        extra={
            "transport": "mpi-dense",
            "accepted_nodes": int(metrics.accepted_updates),
            "global_quorum": int(config.global_quorum),
            "latest_advanced": latest_advanced,
            "metrics_json": str(metrics_path),
        },
    )
    return final_payload


def _coordinate_compiled_mpich_dense_rank(
    *,
    config: RealAsyncFileRankConfig,
    start_s: float,
    base_state: Mapping[str, torch.Tensor],
    node_result: RealAsyncNodeResult,
    artifact_dir: Path,
    progress_dir: Path,
    generation: int,
) -> dict[str, Any] | None:
    metrics_path = Path(config.metrics_json) if config.metrics_json is not None else Path(config.run_dir) / "real_async_metrics.json"
    node_id = f"node-{int(config.node_rank):05d}"
    if config.compiled_mpich_helper_bin is None:
        raise ValueError("compiled_mpich_helper_bin is required for compiled MPICH transport")
    ipc_dir = Path(config.compiled_mpich_ipc_dir) if config.compiled_mpich_ipc_dir is not None else Path(config.run_dir) / "ipc"
    if node_result.node_update is None:
        local_update = _status_update(node_id, generation, base_state, failed=True)
    else:
        local_update = node_result.node_update
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="compiled_mpich_helper_send_starting",
        generation=generation,
        extra={
            "transport": COMPILED_MPICH_TRANSPORT,
            "global_quorum": int(config.global_quorum),
            "bucket_bytes": int(config.mpi_bucket_bytes),
            "helper_bin": str(config.compiled_mpich_helper_bin),
            "ipc_dir": str(ipc_dir),
        },
    )
    payload = run_compiled_mpich_dense_quorum(
        base_state=base_state,
        local_update=local_update,
        run_id=config.run_id,
        generation=generation,
        requested_ranks=config.node_count,
        quorum=config.global_quorum,
        rank=int(config.node_rank),
        helper=CompiledMpichHelperConfig(
            helper_bin=config.compiled_mpich_helper_bin,
            ipc_dir=ipc_dir,
            bucket_bytes=int(config.mpi_bucket_bytes),
            timeout_s=float(config.timeout_s),
        ),
        base_checkpoint=(None if config.initial_checkpoint is None else str(config.initial_checkpoint)),
        quorum_mode=str(config.quorum_mode),
    )
    _atomic_write_json(artifact_dir / f"{node_id}.json", {
        "schema_version": 1,
        "run_id": config.run_id,
        "node_rank": int(config.node_rank),
        "node_id": node_id,
        "generation": int(generation),
        "node_update_submitted": node_result.node_update is not None,
        "transport": COMPILED_MPICH_TRANSPORT,
        "dense_delta_bytes": state_num_bytes(local_update.delta),
        "helper_bin": str(config.compiled_mpich_helper_bin),
        "ipc_dir": str(ipc_dir),
        "metrics": node_result.metrics.to_dict(),
    })
    if int(config.node_rank) != 0:
        _write_rank_heartbeat(
            progress_dir,
            node_rank=config.node_rank,
            node_id=node_id,
            stage="compiled_mpich_helper_result_received",
            generation=generation,
            extra={"transport": COMPILED_MPICH_TRANSPORT},
        )
        return None

    if payload is None:
        raise RuntimeError("compiled MPICH helper root did not produce a quorum payload")
    global_metrics_payload = ((payload.get("global_generations") or [{}])[0].get("metrics") or {})
    metrics = AsyncDiLoCoGenerationMetrics.from_dict(global_metrics_payload)
    latest_advanced = False
    checkpoint_paths: tuple[str, ...] = ()
    if metrics.quorum_status == "advanced":
        manager = AsyncDiLoCoCheckpointManager(
            config.run_dir,
            run_id=config.run_id,
            role=GLOBAL_MERGER_ROLE,
            cadence=config.checkpoint_cadence,
        )
        publish = manager.publish_global_generation(
            metrics,
            walltime_remaining_s=config.walltime_remaining_s,
            estimated_finalization_duration_s=config.estimated_finalization_duration_s,
        )
        metrics = publish.metrics
        checkpoint_paths = tuple(metrics.checkpoint_paths)
        latest_advanced = True
    final_payload = {
        **payload,
        "mode": "actual_multinode_compiled_mpich_quorum",
        "transport": {
            **dict(payload.get("transport") or {}),
            "name": COMPILED_MPICH_TRANSPORT,
            "selector": config.transport_selector or "compiled-cray-mpich-helper-p2p",
            "actual": COMPILED_MPICH_TRANSPORT,
            "approval_class": config.transport_approval_class or "frontier-production-candidate",
            "production_approval_eligible": bool(config.production_approval_eligible),
            "filesystem_live_quorum": False,
            "tcp_dense_data_plane": False,
            "mpi4py": False,
        },
        "production_approval_eligible": bool(config.production_approval_eligible),
        "node_count": int(config.node_count),
        "global_quorum": int(config.global_quorum),
        "generation": int(generation),
        "latest_generation": (int(generation) if latest_advanced else -1),
        "latest_path": str(Path(config.run_dir) / "latest.json"),
        "partial": not latest_advanced,
        "global_generations": [{
            "generation": int(generation),
            "metrics": metrics.to_dict(),
            "publish_paths": list(checkpoint_paths),
        }],
        "metrics_summary": build_metrics_summary(
            run_id=config.run_id,
            requested_workers=int(config.node_count),
            participating_workers=int(metrics.participating_workers),
            generations=(metrics,),
        ).to_dict(),
    }
    _atomic_write_json(metrics_path, final_payload)
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage=(
            "compiled_mpich_helper_coordinator_finalized"
            if latest_advanced else
            "compiled_mpich_helper_coordinator_deferred"
        ),
        generation=generation,
        extra={
            "transport": COMPILED_MPICH_TRANSPORT,
            "accepted_nodes": int(metrics.accepted_updates),
            "global_quorum": int(config.global_quorum),
            "latest_advanced": latest_advanced,
            "metrics_json": str(metrics_path),
        },
    )
    return final_payload


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed before frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed_json(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if size <= 0 or size > 64 * 1024 * 1024:
        raise ValueError(f"invalid frame size: {size}")
    payload = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("framed payload must be a JSON object")
    return payload


def _send_framed_json(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    data = stable_json_dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(data)) + data)


def _accepted_network_payloads(
    payloads: Sequence[Mapping[str, Any]],
    generation: int,
) -> list[dict[str, Any]]:
    return [
        dict(payload) for payload in payloads
        if payload.get("node_update_submitted") is True
        and int(payload.get("generation", -1)) == int(generation)
    ]


def _sorted_node_payloads(payloads: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(payload) for payload in payloads), key=lambda item: str(item.get("node_id", "")))


def _network_quorum_metrics(
    *,
    config: RealAsyncFileRankConfig,
    generation: int,
    accepted: Sequence[Mapping[str, Any]],
    all_payloads: Sequence[Mapping[str, Any]],
    start_s: float,
    timed_out: int,
    failed: int,
) -> AsyncDiLoCoGenerationMetrics:
    duration_s = max(0.0, time.monotonic() - start_s)
    tokens = sum(int(payload.get("tokens", 0)) for payload in accepted)
    payload_bytes = sum(int(payload.get("transport_bytes_sent", 0) or 0) for payload in accepted)
    if payload_bytes <= 0:
        payload_bytes = sum(len(stable_json_dumps(payload)) for payload in accepted)
    losses = [
        float(payload["loss"])
        for payload in accepted
        if payload.get("loss") is not None and math.isfinite(float(payload["loss"]))
    ]
    advanced = len(accepted) >= int(config.global_quorum)
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
        invalid_updates=0,
        generation_duration_s=duration_s,
        merge_duration_s=0.0,
        rebase_duration_s=0.0,
        checkpoint_duration_s=0.0,
        tokens_per_sec=(float(tokens) / duration_s if duration_s > 0.0 else 0.0),
        tokens_per_generation=int(tokens),
        update_bytes={
            "tcp_payload": int(payload_bytes),
            "node_metadata": sum(len(stable_json_dumps(payload)) for payload in accepted),
        },
        loss_moving_average={"loss": _mean(losses), "loss_100": _mean(losses)} if losses else {},
        update_norms={},
        latest_advanced=False,
        quorum_status=("advanced" if advanced else "deferred"),
    )


def _network_quorum_payload(
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
) -> dict[str, Any]:
    timed_out = max(0, int(config.node_count) - len(all_payloads))
    failed = sum(1 for payload in all_payloads if payload.get("node_update_submitted") is not True)
    if metrics is None:
        metrics = _network_quorum_metrics(
            config=config,
            generation=generation,
            accepted=accepted,
            all_payloads=all_payloads,
            start_s=start_s,
            timed_out=timed_out,
            failed=failed,
        )
    metrics_dict = metrics.to_dict()
    if checkpoint_paths:
        metrics_dict["checkpoint_paths"] = list(checkpoint_paths)
        metrics_dict["checkpoint_sizes"] = dict(checkpoint_sizes)
        metrics_dict["latest_advanced"] = bool(latest_advanced)
    submit_latencies = [
        float(payload["transport_submit_latency_s"])
        for payload in all_payloads
        if payload.get("transport_submit_latency_s") is not None
    ]
    receive_latencies = [
        float(payload["transport_receive_latency_s"])
        for payload in all_payloads
        if payload.get("transport_receive_latency_s") is not None
    ]
    transport_bytes = sum(int(payload.get("transport_bytes_sent", 0) or 0) for payload in all_payloads)
    seen_node_ids = {str(payload.get("node_id")) for payload in all_payloads}
    timed_out_node_ids = [
        f"node-{idx:05d}"
        for idx in range(int(config.node_count))
        if f"node-{idx:05d}" not in seen_node_ids
    ]
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "mode": "actual_multinode_tcp_quorum_debug",
        "transport": {
            "name": "tcp",
            "selector": config.transport_selector or "tcp",
            "actual": "tcp",
            "approval_class": config.transport_approval_class or "tcp-debug-only",
            "production_approval_eligible": False,
            "coordinator_host": str(config.coordinator_host),
            "coordinator_bind_host": str(config.coordinator_bind_host),
            "coordinator_port": int(config.coordinator_port),
            "filesystem_live_quorum": False,
            "bytes_sent": int(transport_bytes),
            "submit_latency_s": _distribution(submit_latencies),
            "receive_latency_s": _distribution(receive_latencies),
            "timed_out_node_ids": timed_out_node_ids,
        },
        "bounded_debug_alternative": {
            "dense_delta_exchange": "mpi_p2p_target_not_python_debug_payload",
            "proof": "one Slurm-launched process per GPU runs real local token training and rank 0 merges TCP-submitted rank metadata quorum",
        },
        "production_approval_eligible": False,
        "synthetic_token_stream": bool(config.synthetic_token_stream),
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
