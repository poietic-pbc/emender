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
import os
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

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
from ndm.resilient_node_quorum import GenerationFence
from ndm.resilient_node_transport import (
    RESILIENT_NODE_TRANSPORT,
    DiskBucketSpool,
    NodeManagerClient,
    QuorumTransportServer,
    TransportConfig,
    apply_aggregate_delta,
    pack_dense_delta,
)

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
    optimizer_state_dict: Mapping[str, Any] | None = None


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
    generations: int = 1
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
    resilient_spool_dir: str | Path | None = None
    resilient_coordinator_epoch: int = 1
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
    if config.generations <= 0:
        raise ValueError("generations must be positive")

    run_dir = Path(config.run_dir)
    transport = str(config.transport).strip().lower()
    if transport not in {"tcp", "mpi-dense", COMPILED_MPICH_TRANSPORT, RESILIENT_NODE_TRANSPORT}:
        raise ValueError(
            "transport must be 'tcp', 'mpi-dense', or "
            f"'{COMPILED_MPICH_TRANSPORT}', or '{RESILIENT_NODE_TRANSPORT}'"
        )
    quorum_mode = str(config.quorum_mode).strip().lower()
    if quorum_mode not in {RESILIENT_QUORUM_DILOCO_MODE, STRICT_COLLECTIVE_DILOCO_MODE}:
        raise ValueError("quorum_mode must be 'resilient_quorum' or 'strict_collective'")
    if quorum_mode == STRICT_COLLECTIVE_DILOCO_MODE and transport != COMPILED_MPICH_TRANSPORT:
        raise ValueError("strict_collective quorum_mode requires compiled MPICH transport")
    # The generic rank-coordinator TCP path is deliberately a small-debug path.
    # The resilient node-manager transport is the scale path and has its own
    # bounded framing, spool, fencing, and deadline controls.
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
    start_s = time.monotonic()
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="starting",
        generation=0,
        extra={"node_count": config.node_count, "global_quorum": config.global_quorum},
    )

    train_args = _copy_train_args(config.train_args)
    train.normalize_training_args(train_args)
    train_args._diloco_local_steps = int(config.local_steps)
    torch.manual_seed(int(getattr(train_args, "seed", 42)))
    global_model = train.build_training_model(train_args)
    optimizer_state_dict: Mapping[str, Any] | None = None
    initial_checkpoint_payload: Mapping[str, Any] | None = None
    if config.initial_checkpoint is not None:
        _, _, ckpt = train.load_checkpoint(
            str(config.initial_checkpoint),
            global_model,
            return_checkpoint=True,
        )
        optimizer_state_dict = ckpt.get("optimizer_state_dict")
        # Only step/path metadata is used when emitting chain checkpoints.
        # Retaining ``ckpt`` also retained a second complete CPU model beside
        # ``base_state`` for every rank throughout every generation.
        initial_checkpoint_payload = {
            "checkpoint_path": str(config.initial_checkpoint),
            "step": int(ckpt.get("step", 0) or 0),
        }
        del ckpt
    base_state = _floating_state_dict(global_model)
    del global_model

    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="checkpoint_loaded",
        generation=0,
        extra={
            "base_state_bytes": state_num_bytes(base_state),
            "optimizer_state_loaded": optimizer_state_dict is not None,
        },
    )

    last_result: dict[str, Any] | None = None
    for generation in range(config.generations):
        last_result = _run_real_async_diloco_file_rank_generation(
            config=config,
            generation=generation,
            start_s=start_s,
            train_args=train_args,
            base_state=base_state,
            optimizer_state_dict=optimizer_state_dict,
            initial_checkpoint_payload=initial_checkpoint_payload,
            progress_dir=progress_dir,
            nodes_dir=nodes_dir,
        )
        next_state = last_result.pop("_private_global_state", None)
        next_optimizer = last_result.pop("_private_optimizer_state", None)
        if next_state is not None:
            base_state = next_state
        if next_optimizer is not None:
            optimizer_state_dict = next_optimizer

    assert last_result is not None
    return last_result


def _run_real_async_diloco_file_rank_generation(
    *,
    config: RealAsyncFileRankConfig,
    generation: int,
    start_s: float,
    train_args: Namespace,
    base_state: Mapping[str, torch.Tensor],
    optimizer_state_dict: Mapping[str, Any] | None,
    initial_checkpoint_payload: Mapping[str, Any] | None,
    progress_dir: Path,
    nodes_dir: Path,
) -> dict[str, Any]:
    """Run one merge interval using state initialized by the outer process loop."""

    node_id = f"node-{int(config.node_rank):05d}"
    transport = str(config.transport).strip().lower()
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
        optimizer_state_dict=optimizer_state_dict,
        consume_optimizer_state=True,
    )
    generation_optimizer_state = (
        node_result.worker_reports[0].optimizer_state_dict
        if node_result.worker_reports else optimizer_state_dict
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
    if transport == RESILIENT_NODE_TRANSPORT:
        root_payload = _coordinate_resilient_node_rank(
            config=config,
            start_s=start_s,
            base_state=base_state,
            node_result=node_result,
            artifact_dir=nodes_dir,
            progress_dir=progress_dir,
            generation=generation,
            train_args=train_args,
            optimizer_state_dict=generation_optimizer_state,
            initial_checkpoint_payload=initial_checkpoint_payload,
        )
    elif transport == COMPILED_MPICH_TRANSPORT:
        root_payload = _coordinate_compiled_mpich_dense_rank(
            config=config,
            start_s=start_s,
            base_state=base_state,
            node_result=node_result,
            artifact_dir=nodes_dir,
            progress_dir=progress_dir,
            generation=generation,
            train_args=train_args,
            optimizer_state_dict=generation_optimizer_state,
            initial_checkpoint_payload=initial_checkpoint_payload,
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
            train_args=train_args,
            optimizer_state_dict=generation_optimizer_state,
            initial_checkpoint_payload=initial_checkpoint_payload,
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

    next_global_state = None
    if root_payload is not None:
        next_global_state = root_payload.pop("_private_global_state", None)
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
        "_private_global_state": next_global_state,
        "_private_optimizer_state": generation_optimizer_state,
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
    optimizer_state_dict: Mapping[str, Any] | None = None
    initial_checkpoint_payload: Mapping[str, Any] | None = None
    if config.initial_checkpoint is not None:
        _, _, ckpt = train.load_checkpoint(
            str(config.initial_checkpoint),
            global_model,
            return_checkpoint=True,
        )
        optimizer_state_dict = ckpt.get("optimizer_state_dict")
        initial_checkpoint_payload = {
            "checkpoint_path": str(config.initial_checkpoint),
            "step": int(ckpt.get("step", 0) or 0),
        }
        del ckpt
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
                optimizer_state_dict=optimizer_state_dict,
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
            train_args=train_args,
            optimizer_state_dict=optimizer_state_dict,
            initial_checkpoint_payload=initial_checkpoint_payload,
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
    optimizer_state_dict: Mapping[str, Any] | None = None,
    consume_optimizer_state: bool = False,
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
            optimizer_state_dict=optimizer_state_dict,
            consume_optimizer_state=consume_optimizer_state,
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
    train_args: Namespace | None = None,
    optimizer_state_dict: Mapping[str, Any] | None = None,
    initial_checkpoint_payload: Mapping[str, Any] | None = None,
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
        chain_checkpoint_path = _write_verified_chain_checkpoint(
            run_dir=manager.root,
            run_id=run_id,
            generation=generation,
            state=merge_result.state,
            metrics=metrics,
            train_args=train_args,
            optimizer_state_dict=optimizer_state_dict,
            initial_checkpoint_payload=initial_checkpoint_payload,
        )
        publish_result = manager.publish_global_generation(
            metrics,
            walltime_remaining_s=walltime_remaining_s,
            estimated_finalization_duration_s=estimated_finalization_duration_s,
            extra_checkpoint_paths=(chain_checkpoint_path,),
        )
        metrics = publish_result.metrics
        publish_paths = tuple(metrics.checkpoint_paths)
    return RealAsyncGlobalResult(
        generation=generation,
        state=merge_result.state,
        metrics=metrics,
        publish_paths=publish_paths,
    )


def _write_verified_chain_checkpoint(
    *,
    run_dir: str | Path,
    run_id: str,
    generation: int,
    state: Mapping[str, torch.Tensor],
    metrics: AsyncDiLoCoGenerationMetrics,
    train_args: Namespace | None,
    optimizer_state_dict: Mapping[str, Any] | None,
    initial_checkpoint_payload: Mapping[str, Any] | None,
) -> Path:
    if train_args is None:
        raise ValueError("train_args are required to write a chain checkpoint")
    args = _copy_train_args(train_args)
    train.normalize_training_args(args)
    model = train.build_training_model(args)
    model.load_state_dict(dict(state), strict=False)
    optimizer = train.build_training_optimizer(model, args)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
    if str(getattr(args, "optimizer", "")) == "schedulefree":
        _align_schedulefree_optimizer_state_to_model(model, optimizer)

    source_step = 0
    if initial_checkpoint_payload is not None:
        source_step = int(initial_checkpoint_payload.get("step", 0) or 0)
    local_steps = max(1, int(getattr(args, "_diloco_local_steps", getattr(args, "steps", 1))))
    step = source_step + (int(generation) + 1) * local_steps
    loss = float(metrics.loss_moving_average.get("loss") or metrics.loss_moving_average.get("loss_100") or 0.0)
    output_dir = Path(run_dir) / "checkpoints" / _chain_checkpoint_label(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_step_{step:06d}_loss_{loss:.4f}.pt"
    metadata = {
        "kind": "async_diloco_chain",
        "run_id": run_id,
        "generation": int(generation),
        "source_checkpoint": (
            None if initial_checkpoint_payload is None else initial_checkpoint_payload.get("checkpoint_path")
        ),
        "source_checkpoint_step": source_step,
        "model": getattr(args, "_model_metadata", None),
        "tokenizer": getattr(args, "tokenizer", None),
        "optimizer": getattr(args, "optimizer", None),
        "local_steps": local_steps,
        "tokens_per_generation": int(metrics.tokens_per_generation),
        "accepted_updates": int(metrics.accepted_updates),
        "stale_updates": int(metrics.stale_updates),
        "failed_updates": int(metrics.failed_updates),
        "invalid_updates": int(metrics.invalid_updates),
        "timed_out_updates": int(metrics.timed_out_updates),
    }
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "checkpoint_metadata": metadata,
    }
    tmp_path = output_dir / f".{ckpt_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    try:
        torch.save(payload, tmp_path)
        # Verify structure through mmap so checkpoint validation does not
        # reconstruct another complete model+optimizer beside the writer.
        loaded = torch.load(tmp_path, map_location="cpu", mmap=True, weights_only=True)
        if "model_state_dict" not in loaded or "optimizer_state_dict" not in loaded:
            raise ValueError("chain checkpoint verification failed: missing model or optimizer state")
        if int(loaded.get("step", -1)) != step:
            raise ValueError("chain checkpoint verification failed: step mismatch")
        del loaded
        os.replace(tmp_path, ckpt_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    latest_path = Path(run_dir) / "latest.pt"
    tmp_latest = latest_path.with_name(f".latest.pt.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        if tmp_latest.exists() or tmp_latest.is_symlink():
            tmp_latest.unlink()
        tmp_latest.symlink_to(ckpt_path.relative_to(latest_path.parent))
        os.replace(tmp_latest, latest_path)
    finally:
        if tmp_latest.exists() or tmp_latest.is_symlink():
            tmp_latest.unlink()
    return ckpt_path


def _align_schedulefree_optimizer_state_to_model(
    model: torch.nn.Module,
    optimizer: Any,
    *,
    preserve_existing: bool = False,
) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.setdefault(param, {})
            if not preserve_existing or "z" not in state:
                state["z"] = param.detach().clone()
            state.setdefault("exp_avg_sq", torch.zeros_like(param, memory_format=torch.preserve_format))
        if not preserve_existing:
            group["train_mode"] = False


def _chain_checkpoint_label(args: Namespace) -> str:
    level = str(getattr(args, "level", "model"))
    params = str(getattr(args, "params", "unknown"))
    date = time.strftime("%Y%m%d", time.gmtime())
    return f"emender_{level}_{params}_{date}"


@dataclass(frozen=True)
class PersistentWindowReport:
    generation: int
    tokens: int
    losses: tuple[float, ...]
    elapsed_s: float


@dataclass(frozen=True)
class CoalescedWindowReport:
    """One bounded mutable interval produced beside native completion."""

    local_window_start: int
    local_window_end: int
    exact_tokens: int
    losses: tuple[float, ...]
    elapsed_s: float
    reached_hard_bound: bool
    snapshot_deferred: bool = False
    translation_elapsed_s: float = 0.0

    @property
    def window_count(self) -> int:
        return (
            0 if self.snapshot_deferred
            else self.local_window_end - self.local_window_start
        )


class PersistentRealWorkerSession:
    """One resident model/inner-optimizer/data lane across exact K windows.

    Production bounded-lag v2 bootstraps this object once per incarnation.
    `run_window` exposes the only mutation boundary; the caller may then seal
    a cumulative interval or translate x/z before invoking the next window.
    """

    def __init__(
        self,
        *,
        base_state: Mapping[str, torch.Tensor],
        train_args: Namespace,
        spec: RealAsyncWorkerSpec,
        synthetic_token_stream: bool,
        synthetic_vocab_size: int,
        optimizer_state_dict: Mapping[str, Any] | None = None,
        consume_optimizer_state: bool = False,
        bootstrap_phase_callback: Callable[
            [str, Mapping[str, Any]], None
        ] | None = None,
    ):
        self.args = _copy_train_args(train_args)
        self.spec = spec
        self.base_keys = tuple(sorted(base_state))
        self.hidden_state: Any = None
        self.closed = False
        self.windows_completed = 0
        self.bootstrap_counts = {
            "model_build": 0,
            "optimizer_build": 0,
            "data_iterator_build": 0,
        }
        self._snapshot_slots: tuple[
            dict[str, torch.Tensor], dict[str, torch.Tensor]
        ] | None = None
        self._snapshot_copy_ready: dict[tuple[int, ...], Any] = {}
        self._next_snapshot_slot = 0
        torch.manual_seed(
            int(getattr(self.args, "seed", 42)) + int(spec.seed_offset))
        self.device = torch.device(spec.device)

        def phase(name: str) -> None:
            if bootstrap_phase_callback is not None:
                bootstrap_phase_callback(name, {"step": 0})

        phase("model_build_start")
        self.model = train.build_training_model(self.args).to(self.device)
        self.bootstrap_counts["model_build"] += 1
        phase("model_device_ready")
        self.model.load_state_dict(base_state, strict=False)
        phase("model_state_loaded")
        if bool(getattr(self.args, "bf16", False)):
            self.model = self.model.bfloat16()
        phase("model_dtype_ready")
        self.optimizer = train.build_training_optimizer(
            self.model, self.args)
        self.bootstrap_counts["optimizer_build"] += 1
        phase("optimizer_built")
        if optimizer_state_dict:
            self.optimizer.load_state_dict(optimizer_state_dict)
            if consume_optimizer_state:
                _release_consumed_optimizer_state(optimizer_state_dict)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.args.lr
        if str(getattr(self.args, "optimizer", "")) == "schedulefree":
            # ScheduleFree creates per-parameter state lazily. A sparse first
            # E97 window can reach global translation before every resident
            # parameter has acquired the required z point.
            _align_schedulefree_optimizer_state_to_model(
                self.model, self.optimizer, preserve_existing=True)
        phase("optimizer_state_loaded")
        self.batch_iter = _build_batch_iter(
            self.args,
            rank=spec.seed_offset,
            device=self.device,
            synthetic=synthetic_token_stream,
            synthetic_vocab_size=synthetic_vocab_size,
        )
        self.bootstrap_counts["data_iterator_build"] += 1
        phase("data_iterator_ready")
        name_by_parameter = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
        }
        self.optimizer_parameter_names = tuple(
            name_by_parameter[id(parameter)]
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        # Allocation belongs to bootstrap, never the bounded K-boundary
        # foreground pause.  The slots contain no valid snapshot until
        # `snapshot()` fills one at a coherent optimizer boundary.
        model_values = self.model.state_dict()
        self._snapshot_slots = tuple({
            name: torch.empty_like(
                model_values[name],
                device="cpu",
                pin_memory=(self.device.type == "cuda"),
            )
            for name in self.base_keys
        } for _ in range(2))
        phase("snapshot_slots_preallocated")

    def run_window(
        self,
        generation: int,
        *,
        progress_callback: Callable[
            [int, Mapping[str, Any]], None
        ] | None = None,
        phase_callback: Callable[
            [str, Mapping[str, Any]], None
        ] | None = None,
    ) -> PersistentWindowReport:
        if self.closed:
            raise RuntimeError("persistent real-worker session is closed")
        started = time.monotonic()
        losses: list[float] = []
        tokens = 0
        local_steps = max(1, int(self.spec.local_steps))
        for local_step in range(local_steps):
            metrics = train.train_one_optimizer_step(
                self.model,
                self.optimizer,
                self.args,
                batch_iter=self.batch_iter,
                device=self.device,
                step=int(generation) * local_steps + local_step,
                hidden_state=self.hidden_state,
                phase_callback=phase_callback,
            )
            self.hidden_state = metrics.get("hidden_state")
            losses.append(float(metrics["loss"]))
            tokens += int(metrics["tokens_processed"])
            if progress_callback is not None:
                progress_callback(local_step + 1, metrics)
        self.windows_completed += 1
        return PersistentWindowReport(
            generation=int(generation),
            tokens=tokens,
            losses=tuple(losses),
            elapsed_s=max(0.0, time.monotonic() - started),
        )

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Capture a coherent endpoint into one of two bounded CPU slots.

        Callers invoke this only between exact K windows, when the resident
        optimizer is not mutating the model.  Alternating preallocated slots
        let the prior immutable endpoint remain available to the background
        publisher/correction ledger while the live model resumes in the next
        window; no background path rereads the concurrently mutating model.

        A device session enqueues the copies into pinned CPU storage on the
        current device stream.  Subsequent model work on that stream is
        ordered after the immutable copy, so local ownership can transfer
        without waiting for all device-to-host traffic.  The background
        consumer must call :meth:`wait_snapshot_ready` before reading it.
        """
        if self.closed:
            raise RuntimeError("persistent real-worker session is closed")
        values = self.model.state_dict()
        if set(self.base_keys) - set(values):
            raise ValueError("persistent model layout lost base tensors")
        if self._snapshot_slots is None:
            raise RuntimeError("persistent snapshot slots were not preallocated")
        slot = self._snapshot_slots[self._next_snapshot_slot]
        self._next_snapshot_slot = (self._next_snapshot_slot + 1) % len(
            self._snapshot_slots)
        snapshot_key = tuple(
            int(slot[name].data_ptr()) for name in self.base_keys)
        prior_copy = self._snapshot_copy_ready.get(snapshot_key)
        if prior_copy is not None:
            if not prior_copy.query():
                raise RuntimeError(
                    "persistent snapshot slot reused before copy completion")
            self._snapshot_copy_ready.pop(snapshot_key, None)
        with torch.no_grad():
            for name in self.base_keys:
                source = values[name].detach()
                target = slot[name]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError("persistent snapshot slot layout changed")
                target.copy_(source, non_blocking=True)
        if self.device.type == "cuda":
            snapshot_copy_ready = torch.cuda.Event(
                enable_timing=False, blocking=False)
            snapshot_copy_ready.record(torch.cuda.current_stream(self.device))
            self._snapshot_copy_ready[snapshot_key] = snapshot_copy_ready
        return dict(slot)

    def order_after_snapshot(
        self,
        snapshot: Mapping[str, torch.Tensor],
    ) -> None:
        """Order this thread's device stream after an admitted snapshot copy."""
        if tuple(sorted(snapshot)) != self.base_keys:
            raise ValueError("persistent snapshot ordering layout changed")
        snapshot_key = tuple(
            int(snapshot[name].data_ptr()) for name in self.base_keys)
        snapshot_copy_ready = self._snapshot_copy_ready.get(snapshot_key)
        if snapshot_copy_ready is not None:
            torch.cuda.current_stream(self.device).wait_event(
                snapshot_copy_ready)

    def wait_snapshot_ready(
        self,
        snapshot: Mapping[str, torch.Tensor],
        *,
        deadline: float,
    ) -> float:
        """Wait on the background path until an admitted snapshot is readable."""
        if tuple(sorted(snapshot)) != self.base_keys:
            raise ValueError("persistent snapshot readiness layout changed")
        snapshot_key = tuple(
            int(snapshot[name].data_ptr()) for name in self.base_keys)
        snapshot_copy_ready = self._snapshot_copy_ready.get(snapshot_key)
        if snapshot_copy_ready is None:
            return time.monotonic()
        while not snapshot_copy_ready.query():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "persistent snapshot copy completion deadline expired")
            time.sleep(min(0.001, remaining))
        self._snapshot_copy_ready.pop(snapshot_key, None)
        return time.monotonic()

    @property
    def snapshot_slot_count(self) -> int:
        return 0 if self._snapshot_slots is None else len(self._snapshot_slots)

    def translate(self, corrections: Mapping[str, torch.Tensor]) -> None:
        """Translate resident model x and audited ScheduleFree z at a boundary."""
        if self.closed or tuple(sorted(corrections)) != self.base_keys:
            raise ValueError("persistent correction layout differs from model")
        model_state = self.model.state_dict()
        for name in self.base_keys:
            correction = corrections[name]
            target = model_state[name]
            if (correction.shape != target.shape
                    or not torch.isfinite(correction).all()):
                raise ValueError("persistent correction is malformed/nonfinite")

        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(parameters) != len(self.optimizer_parameter_names):
            raise ValueError("persistent optimizer parameter order changed")
        targets: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for name, parameter in zip(
                self.optimizer_parameter_names, parameters):
            record = self.optimizer.state.get(parameter)
            if not isinstance(record, dict):
                raise ValueError("ScheduleFree per-parameter state is missing")
            z = record.get("z")
            if not isinstance(z, torch.Tensor) or z.shape != parameter.shape:
                raise ValueError("ScheduleFree z point is missing or malformed")
            for key, value in record.items():
                if isinstance(value, torch.Tensor) and value.shape == z.shape:
                    if key not in {"z", "exp_avg_sq"}:
                        raise ValueError(
                            f"unknown parameter-valued optimizer buffer: {key}")
            targets.append((model_state[name], z, corrections[name]))

        # Validation of every x/z destination completes before the first
        # mutation.  The training lane is quiescent at this point, so this
        # short no-grad loop is the atomic resident boundary swap.
        with torch.no_grad():
            for target, z, correction in targets:
                target.add_(correction.to(target))
                z.add_(correction.to(z))

    def close(self) -> None:
        if any(not event.query() for event in self._snapshot_copy_ready.values()):
            raise RuntimeError(
                "persistent real-worker closed with snapshot copy in flight")
        self._snapshot_copy_ready.clear()
        self.closed = True


class PersistentAsyncTrainingLane:
    """Run adjacent K windows while a prior native descriptor is owned.

    The model, optimizer, iterator, and hidden state remain in the supplied
    :class:`PersistentRealWorkerSession`.  This helper owns only a short-lived
    control thread for one mutable cumulative interval.  Native completion can
    wait for transport/reduction/publication on the caller thread while this
    lane keeps the GPU model moving.  ``finish_at_boundary`` is the sole join:
    it stops before another forward pass, optionally translates ScheduleFree
    x/z, and returns one bounded interval rather than a FIFO of per-K deltas.
    """

    def __init__(
        self,
        session: PersistentRealWorkerSession,
        *,
        max_windows: int,
        progress_callback_factory: Callable[
            [int], Callable[[int, Mapping[str, Any]], None] | None
        ] | None = None,
        phase_callback_factory: Callable[
            [int], Callable[[str, Mapping[str, Any]], None] | None
        ] | None = None,
        window_start_callback: Callable[[int], None] | None = None,
    ):
        if isinstance(max_windows, bool) or not 1 <= int(max_windows) <= 8:
            raise ValueError(
                "persistent async lane requires a finite window bound in [1,8]")
        self.session = session
        self.max_windows = int(max_windows)
        self.progress_callback_factory = progress_callback_factory
        self.phase_callback_factory = phase_callback_factory
        self.window_start_callback = window_start_callback
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_requested = False
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._window_start = 0
        self._window_end = 0
        self._tokens = 0
        self._losses: list[float] = []
        self._started_s = 0.0
        self._start_state: dict[str, torch.Tensor] | None = None
        self._snapshot_deferred = False
        self._boundary_report: CoalescedWindowReport | None = None
        self._prepared_correction_identity: tuple[
            tuple[str, tuple[int, ...], str, str, int], ...
        ] | None = None
        self._correction_applied = False

    def start(
        self,
        *,
        local_window_start: int,
        start_state: Mapping[str, torch.Tensor],
        admission_deadline: float,
    ) -> None:
        if (self._thread is not None or local_window_start < 0
                or not start_state):
            raise RuntimeError("persistent async lane cannot be started")
        if time.monotonic() >= admission_deadline:
            raise TimeoutError("persistent async lane admission deadline expired")
        self._window_start = int(local_window_start)
        self._window_end = int(local_window_start)
        self._start_state = {
            name: value
            for name, value in start_state.items()
        }
        self._started_s = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name=f"async-v2-k-window-{local_window_start}",
            daemon=False,
        )
        self._thread.start()
        remaining = admission_deadline - time.monotonic()
        if remaining <= 0 or not self._started.wait(remaining):
            self.abort()
            raise TimeoutError("persistent async lane did not start before deadline")

    def _run(self) -> None:
        try:
            order_after_snapshot = getattr(
                self.session, "order_after_snapshot", None)
            if order_after_snapshot is not None:
                if self._start_state is None:
                    raise RuntimeError(
                        "persistent async lane lost its ordered interval start")
                order_after_snapshot(self._start_state)
            while True:
                with self._condition:
                    if self._stop_requested:
                        self._condition.notify_all()
                        return
                    local_window = self._window_end
                    self._running = True
                    self._started.set()
                if self.window_start_callback is not None:
                    self.window_start_callback(local_window)
                progress = (
                    None if self.progress_callback_factory is None
                    else self.progress_callback_factory(local_window)
                )
                phase = (
                    None if self.phase_callback_factory is None
                    else self.phase_callback_factory(local_window)
                )
                report = self.session.run_window(
                    local_window,
                    progress_callback=progress,
                    phase_callback=phase,
                )
                with self._condition:
                    if report.generation != local_window:
                        raise ValueError(
                            "persistent async lane window identity changed")
                    self._window_end = local_window + 1
                    completed = self._window_end - self._window_start
                    if completed <= self.max_windows:
                        self._tokens += int(report.tokens)
                        self._losses.extend(
                            float(value) for value in report.losses)
                    else:
                        # The trainer continues on its live mutable state, but
                        # no third speculative snapshot/interval is retained.
                        # This work is disposable until a verified result is
                        # atomically applied at a later safe boundary.
                        self._snapshot_deferred = True
                        self._tokens = 0
                        self._losses = [
                            float(value) for value in report.losses[-1:]]
                    self._running = False
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._running = False
                self._stop_requested = True
                self._started.set()
                self._condition.notify_all()

    def finish_at_boundary(
        self,
        *,
        deadline: float,
        corrections: Mapping[str, torch.Tensor] | None = None,
    ) -> CoalescedWindowReport:
        """Stop after the active K and optionally translate resident x/z.

        Calling without ``corrections`` is the first half of the v2.1
        safe-boundary rendezvous: the lane is quiescent, but no model or
        optimizer point has changed.  :meth:`apply_at_boundary` performs the
        already-prepared translation only after the manager releases the exact
        all-eight transaction.
        """
        if self._boundary_report is not None:
            if corrections is None:
                return self._boundary_report
            return self.apply_at_boundary(corrections)
        thread = self._thread
        if thread is None:
            raise RuntimeError("persistent async lane was not started")
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
            while self._running and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "persistent async lane did not reach a safe K boundary")
                self._condition.wait(remaining)
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise TimeoutError("persistent async lane did not stop at boundary")
        if self._error is not None:
            raise self._error
        if (
            self._window_end <= self._window_start
            or (self._tokens <= 0 and not self._snapshot_deferred)
        ):
            raise ValueError("persistent async lane produced an empty interval")
        self._boundary_report = CoalescedWindowReport(
            local_window_start=self._window_start,
            local_window_end=self._window_end,
            exact_tokens=self._tokens,
            losses=tuple(self._losses),
            elapsed_s=max(0.0, time.monotonic() - self._started_s),
            reached_hard_bound=(
                self._window_end - self._window_start >= self.max_windows),
            snapshot_deferred=self._snapshot_deferred,
            translation_elapsed_s=0.0,
        )
        if corrections is None:
            return self._boundary_report
        self.prepare_at_boundary(corrections, deadline=deadline)
        return self.apply_at_boundary(corrections)

    @staticmethod
    def _correction_identity(
        corrections: Mapping[str, torch.Tensor],
    ) -> tuple[tuple[str, tuple[int, ...], str, str, int], ...]:
        return tuple(
            (
                name,
                tuple(int(size) for size in value.shape),
                str(value.dtype),
                str(value.device),
                int(value.data_ptr()),
            )
            for name, value in sorted(corrections.items())
        )

    def prepare_at_boundary(
        self,
        corrections: Mapping[str, torch.Tensor],
        *,
        deadline: float,
    ) -> CoalescedWindowReport:
        """Prepare the dense host interval rebase before all-eight release.

        The lane is already quiescent, so the retained interval start can be
        rebased without touching live model or ScheduleFree z state.  This
        keeps host snapshot preparation inside the bounded boundary-ready
        phase and leaves only the atomic resident x/z translation after the
        manager releases the exact transaction.
        """
        if self._boundary_report is None:
            raise RuntimeError(
                "persistent async lane has not reached its safe boundary")
        if self._prepared_correction_identity is not None:
            raise RuntimeError(
                "persistent async lane correction was already prepared")
        if self._correction_applied:
            raise RuntimeError(
                "persistent async lane correction was already applied")
        if not corrections or self._start_state is None:
            raise ValueError(
                "persistent async lane correction preparation is empty")
        if self._snapshot_deferred:
            # The over-age contribution is discarded, but the next interval
            # must still begin from the corrected quiescent live boundary.
            self._start_state = self.session.snapshot()
        wait_snapshot_ready = getattr(
            self.session, "wait_snapshot_ready", None)
        if wait_snapshot_ready is not None:
            wait_snapshot_ready(self._start_state, deadline=deadline)
        for name, start in self._start_state.items():
            correction = corrections.get(name)
            if (
                correction is None
                or correction.shape != start.shape
                or not torch.isfinite(correction).all()
            ):
                raise ValueError(
                    "persistent async interval correction is malformed")
        with torch.no_grad():
            for name, start in self._start_state.items():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "persistent interval rebase preparation expired")
                start.add_(corrections[name].to(start))
        if time.monotonic() > deadline:
            raise TimeoutError(
                "persistent interval rebase preparation expired")
        self._prepared_correction_identity = self._correction_identity(
            corrections)
        return self._boundary_report

    def apply_at_boundary(
        self,
        corrections: Mapping[str, torch.Tensor],
    ) -> CoalescedWindowReport:
        """Translate a quiescent lane once after all-eight manager release."""
        if self._boundary_report is None:
            raise RuntimeError(
                "persistent async lane has not reached its safe boundary")
        if self._correction_applied:
            raise RuntimeError(
                "persistent async lane correction was already applied")
        if not corrections:
            raise ValueError("persistent async lane correction is empty")
        if (
            self._prepared_correction_identity is None
            or self._prepared_correction_identity
            != self._correction_identity(corrections)
        ):
            raise RuntimeError(
                "persistent async lane correction was not exactly prepared")
        translation_elapsed_s = 0.0
        if self._start_state is None:
            raise RuntimeError("persistent async lane lost its interval start")
        translation_started = time.monotonic()
        self.session.translate(corrections)
        translation_elapsed_s = max(
            0.0, time.monotonic() - translation_started)
        self._correction_applied = True
        self._boundary_report = replace(
            self._boundary_report,
            elapsed_s=max(0.0, time.monotonic() - self._started_s),
            translation_elapsed_s=translation_elapsed_s,
        )
        return self._boundary_report

    @property
    def start_state(self) -> Mapping[str, torch.Tensor]:
        if self._start_state is None:
            raise RuntimeError("persistent async lane has no interval start")
        return self._start_state

    def abort(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(1.0)


def _run_real_worker(
    *,
    run_id: str,
    generation: int,
    base_state: Mapping[str, torch.Tensor],
    train_args: Namespace,
    spec: RealAsyncWorkerSpec,
    synthetic_token_stream: bool,
    synthetic_vocab_size: int,
    optimizer_state_dict: Mapping[str, Any] | None = None,
    consume_optimizer_state: bool = False,
    progress_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    boundary_callback: Callable[
        [Mapping[str, torch.Tensor], torch.nn.Module, Any, int], None
    ] | None = None,
    delta_consumer: Callable[[Mapping[str, torch.Tensor], torch.nn.Module, int], None] | None = None,
    model_state_consumer: Callable[[torch.nn.Module], None] | None = None,
    optimizer_parameter_name_consumer: Callable[[tuple[str, ...]], None] | None = None,
    phase_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
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
        def bootstrap_phase(name: str) -> None:
            if phase_callback is not None:
                phase_callback(name, {"step": int(generation) * max(1, int(spec.local_steps))})

        bootstrap_phase("model_build_start")
        model = train.build_training_model(args).to(device)
        bootstrap_phase("model_device_ready")
        model.load_state_dict(base_state, strict=False)
        bootstrap_phase("model_state_loaded")
        if bool(getattr(args, "bf16", False)):
            model = model.bfloat16()
        bootstrap_phase("model_dtype_ready")
        optimizer = train.build_training_optimizer(model, args)
        bootstrap_phase("optimizer_built")
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)
            if consume_optimizer_state:
                _release_consumed_optimizer_state(optimizer_state_dict)
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
        bootstrap_phase("optimizer_state_loaded")
        batch_iter = _build_batch_iter(
            args,
            rank=spec.seed_offset,
            device=device,
            synthetic=synthetic_token_stream,
            synthetic_vocab_size=synthetic_vocab_size,
        )
        bootstrap_phase("data_iterator_ready")
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
                step=int(generation) * max(1, int(spec.local_steps)) + step,
                hidden_state=hidden_state,
                phase_callback=phase_callback,
            )
            hidden_state = metrics.get("hidden_state")
            losses.append(float(metrics["loss"]))
            tokens += int(metrics["tokens_processed"])
            if progress_callback is not None:
                progress_callback(step + 1, metrics)
        # The bounded-lag production path uses this exact post-K/pre-seal
        # boundary to admit a verified prior result.  It is deliberately
        # inside the model-owning lane: a result can translate live x/z here,
        # but can never mutate a partially executed optimizer window.
        if boundary_callback is not None:
            boundary_callback(base_state, model, optimizer, tokens)
        if delta_consumer is None:
            worker_delta = _floating_delta_from_model(base_state, model)
        else:
            # The live split-role path streams directly from the trained model
            # into its bounded node-local spool.  Do not first materialize a
            # second full CPU model-sized delta merely to serialize it.
            delta_consumer(base_state, model, tokens)
            worker_delta = {}
        if model_state_consumer is not None:
            model_state_consumer(model)
        if optimizer_parameter_name_consumer is not None:
            name_by_parameter = {
                id(parameter): name
                for name, parameter in model.named_parameters()
            }
            ordered_names = tuple(
                name_by_parameter[id(parameter)]
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            optimizer_parameter_name_consumer(ordered_names)
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
            optimizer_state_dict=(optimizer.state_dict() if hasattr(optimizer, "state_dict") else None),
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


def _release_consumed_optimizer_state(state: Mapping[str, Any]) -> None:
    """Release a mutable checkpoint mapping once the optimizer owns its state.

    ``Optimizer.load_state_dict`` casts/copies tensor state to the live
    parameters. Retaining the source mapping pins another model-sized CPU state
    throughout local training. File-rank workers receive ownership of this
    mutable mapping and replace it with the returned live state after training.
    """

    if isinstance(state, dict):
        state.clear()


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


def _coordinate_resilient_node_rank(
    *,
    config: RealAsyncFileRankConfig,
    start_s: float,
    base_state: Mapping[str, torch.Tensor],
    node_result: RealAsyncNodeResult,
    artifact_dir: Path,
    progress_dir: Path,
    generation: int,
    train_args: Namespace,
    optimizer_state_dict: Mapping[str, Any] | None,
    initial_checkpoint_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Exchange a real E97 delta through the MPI-independent node quorum."""

    node_id = f"node-{int(config.node_rank):05d}"
    if node_result.node_update is None:
        raise RuntimeError("local node quorum failed before resilient transport submission")
    layout, buckets = pack_dense_delta(
        node_result.node_update.delta, bucket_bytes=int(config.mpi_bucket_bytes)
    )
    server: QuorumTransportServer | None = None
    server_thread: threading.Thread | None = None
    if int(config.node_rank) == 0:
        server = QuorumTransportServer(
            (str(config.coordinator_bind_host), int(config.coordinator_port)),
            TransportConfig(
                run_id=config.run_id,
                quorum=int(config.global_quorum),
                expected_buckets=len(buckets),
                max_bucket_bytes=int(config.mpi_bucket_bytes),
                generation_deadline_s=float(config.timeout_s),
            ),
            Path(config.run_dir) / "resilient_metadata",
            coordinator_epoch=int(config.resilient_coordinator_epoch),
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

    spool_root = (
        Path(config.resilient_spool_dir)
        if config.resilient_spool_dir is not None
        else Path(os.environ.get("TMPDIR", "/tmp")) / "emender-resilient" / config.run_id / node_id
    )
    spool_limit = max(int(config.mpi_bucket_bytes), sum(len(item) for item in buckets))
    client = NodeManagerClient(
        str(config.coordinator_host),
        int(config.coordinator_port),
        node_id,
        DiskBucketSpool(spool_root, spool_limit),
        timeout_s=float(config.timeout_s),
        max_bucket_bytes=int(config.mpi_bucket_bytes),
    )
    fence = GenerationFence(
        config.run_id, int(generation), 0, int(config.resilient_coordinator_epoch)
    )
    try:
        commit, aggregate_buckets = client.exchange(
            fence, buckets, weight=max(1, int(node_result.node_update.tokens))
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)

    global_state = apply_aggregate_delta(
        base_state, layout, aggregate_buckets, eta_outer=float(config.eta_outer)
    )
    accepted_nodes = tuple(str(item) for item in commit["accepted_nodes"])
    metrics = dataclass_replace_metrics(
        node_result.metrics,
        requested_workers=int(config.node_count),
        participating_workers=len(accepted_nodes),
        quorum_threshold=int(config.global_quorum),
        quorum_size=len(accepted_nodes),
        accepted_updates=len(accepted_nodes),
        timed_out_updates=max(0, int(config.node_count) - len(accepted_nodes)),
        checkpoint_state_id=f"{config.run_id}:global:gen{int(generation):06d}",
        update_bytes={
            "node": sum(len(item) for item in buckets),
            "global_state": state_num_bytes(global_state),
        },
    )
    checkpoint_paths: tuple[str, ...] = ()
    if int(config.node_rank) == 0:
        chain_path = _write_verified_chain_checkpoint(
            run_dir=config.run_dir,
            run_id=config.run_id,
            generation=generation,
            state=global_state,
            metrics=metrics,
            train_args=train_args,
            optimizer_state_dict=optimizer_state_dict,
            initial_checkpoint_payload=initial_checkpoint_payload,
        )
        manager = AsyncDiLoCoCheckpointManager(
            config.run_dir,
            run_id=config.run_id,
            role=GLOBAL_MERGER_ROLE,
            cadence=config.checkpoint_cadence,
        )
        metrics = manager.publish_global_generation(
            metrics,
            walltime_remaining_s=config.walltime_remaining_s,
            estimated_finalization_duration_s=config.estimated_finalization_duration_s,
            extra_checkpoint_paths=(chain_path,),
        ).metrics
        checkpoint_paths = tuple(metrics.checkpoint_paths)

    payload = {
        "schema_version": 1,
        "mode": RESILIENT_NODE_TRANSPORT,
        "run_id": config.run_id,
        "latest_generation": int(generation),
        "accepted_node_ids": list(accepted_nodes),
        "global_generations": [{"generation": int(generation), "metrics": metrics.to_dict()}],
        "transport": {
            "name": RESILIENT_NODE_TRANSPORT,
            "mpi_world_collective": False,
            "dense_data_plane": True,
            "bucket_count": len(buckets),
            "bucket_bytes": int(config.mpi_bucket_bytes),
            "spool_root": str(spool_root),
        },
        "publish_paths": list(checkpoint_paths),
        "elapsed_s": max(0.0, time.monotonic() - start_s),
        "_private_global_state": global_state,
    }
    _atomic_write_json(artifact_dir / f"{node_id}.json", {
        key: value for key, value in payload.items() if not key.startswith("_private")
    })
    if int(config.node_rank) == 0:
        metrics_path = Path(config.metrics_json) if config.metrics_json else Path(config.run_dir) / "real_async_metrics.json"
        _atomic_write_json(metrics_path, {
            key: value for key, value in payload.items() if not key.startswith("_private")
        })
    _write_rank_heartbeat(
        progress_dir,
        node_rank=config.node_rank,
        node_id=node_id,
        stage="resilient_node_commit_applied",
        generation=generation,
        extra={"accepted_nodes": list(accepted_nodes), "checkpoint_paths": list(checkpoint_paths)},
    )
    return payload


def _coordinate_mpi_dense_rank(
    *,
    config: RealAsyncFileRankConfig,
    start_s: float,
    base_state: Mapping[str, torch.Tensor],
    node_result: RealAsyncNodeResult,
    artifact_dir: Path,
    progress_dir: Path,
    generation: int,
    train_args: Namespace,
    optimizer_state_dict: Mapping[str, Any] | None,
    initial_checkpoint_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    metrics_path = Path(config.metrics_json) if config.metrics_json is not None else Path(config.run_dir) / "real_async_metrics.json"
    node_id = f"node-{int(config.node_rank):05d}"
    if node_result.node_update is None:
        local_update = _status_update(node_id, generation, base_state, failed=True)
    else:
        local_update = node_result.node_update
    local_update_bytes = state_num_bytes(local_update.delta)
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
        "dense_delta_bytes": local_update_bytes,
        "metrics": node_result.metrics.to_dict(),
    })
    private_global_state = payload.pop("_private_global_state", None)
    if int(config.node_rank) != 0:
        _write_rank_heartbeat(
            progress_dir,
            node_rank=config.node_rank,
            node_id=node_id,
            stage="mpi_dense_result_received",
            generation=generation,
            extra={"transport": "mpi-dense"},
        )
        return {"_private_global_state": private_global_state}

    if payload is None:
        raise RuntimeError("MPI dense root did not produce a quorum payload")
    global_metrics_payload = ((payload.get("global_generations") or [{}])[0].get("metrics") or {})
    metrics = AsyncDiLoCoGenerationMetrics.from_dict(global_metrics_payload)
    latest_advanced = False
    checkpoint_paths: tuple[str, ...] = ()
    if metrics.quorum_status == "advanced":
        if private_global_state is None:
            raise RuntimeError("MPI dense root did not return a merged state for checkpoint chaining")
        chain_checkpoint_path = _write_verified_chain_checkpoint(
            run_dir=config.run_dir,
            run_id=config.run_id,
            generation=generation,
            state=private_global_state,
            metrics=metrics,
            train_args=train_args,
            optimizer_state_dict=optimizer_state_dict,
            initial_checkpoint_payload=initial_checkpoint_payload,
        )
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
            extra_checkpoint_paths=(chain_checkpoint_path,),
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
    final_payload["_private_global_state"] = private_global_state
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
    train_args: Namespace,
    optimizer_state_dict: Mapping[str, Any] | None,
    initial_checkpoint_payload: Mapping[str, Any] | None,
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
    local_update_bytes = state_num_bytes(local_update.delta)
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
    if payload is None:
        raise RuntimeError("compiled MPICH helper did not produce a quorum payload")
    private_global_state = payload.pop("_private_global_state", None)
    _atomic_write_json(artifact_dir / f"{node_id}.json", {
        "schema_version": 1,
        "run_id": config.run_id,
        "node_rank": int(config.node_rank),
        "node_id": node_id,
        "generation": int(generation),
        "node_update_submitted": node_result.node_update is not None,
        "transport": COMPILED_MPICH_TRANSPORT,
        "dense_delta_bytes": local_update_bytes,
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
        return {"_private_global_state": private_global_state}

    global_metrics_payload = ((payload.get("global_generations") or [{}])[0].get("metrics") or {})
    metrics = AsyncDiLoCoGenerationMetrics.from_dict(global_metrics_payload)
    latest_advanced = False
    checkpoint_paths: tuple[str, ...] = ()
    if metrics.quorum_status == "advanced":
        if private_global_state is None:
            raise RuntimeError("compiled MPICH root did not return a merged state for checkpoint chaining")
        chain_checkpoint_path = _write_verified_chain_checkpoint(
            run_dir=config.run_dir,
            run_id=config.run_id,
            generation=generation,
            state=private_global_state,
            metrics=metrics,
            train_args=train_args,
            optimizer_state_dict=optimizer_state_dict,
            initial_checkpoint_payload=initial_checkpoint_payload,
        )
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
            extra_checkpoint_paths=(chain_checkpoint_path,),
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
    "CoalescedWindowReport",
    "PersistentAsyncTrainingLane",
    "PersistentRealWorkerSession",
    "PersistentWindowReport",
    "default_tiny_e97_train_args",
    "run_real_async_diloco",
    "run_real_async_diloco_file_rank",
]
