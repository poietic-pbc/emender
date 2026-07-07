"""Pure async quorum DiLoCo state math and metrics utilities.

This module is intentionally orchestration- and transport-free.  It gives unit
tests, local simulations, and Frontier debug jobs one stable representation for
the core async DiLoCo generation summary.

Metrics schema version 1
------------------------
``AsyncDiLoCoGenerationMetrics`` serializes as a JSON object with these required
fields:

``schema_version``
    Integer schema version.  The initial version is ``1``.
``run_id``
    Stable run/debug-job identifier.
``generation``
    Global generation advanced or attempted by this record.
``requested_workers`` / ``participating_workers``
    Worker counts requested by the run and observed in this generation.
``quorum_threshold`` / ``quorum_size``
    Configured quorum threshold and effective accepted quorum size.
``accepted_updates`` / ``stale_updates`` / ``timed_out_updates`` /
``failed_updates`` / ``invalid_updates``
    Per-generation update outcome counts.
``generation_duration_s`` / ``merge_duration_s`` / ``rebase_duration_s`` /
``checkpoint_duration_s``
    Wall-clock timing buckets in seconds.
``tokens_per_sec`` / ``tokens_per_generation``
    Aggregate throughput and accepted token count.
``update_bytes``
    Dense or compressed update byte volume by level, for example
    ``{"worker": 123, "node": 456}``.
``loss_moving_average``
    Loss moving-average values by window name.
``update_norms``
    Norm summaries by tensor/basis name.
``checkpoint_paths`` / ``checkpoint_sizes``
    Durable checkpoint paths and byte sizes.
``latest_advanced``
    Whether the authoritative latest pointer advanced for this generation.
``resume_source_generation``
    Source generation used to resume, or ``null`` for a fresh run.

``AsyncDiLoCoMetricsSummary`` serializes as a JSON object containing the same
schema/run identifiers, all generation records, aggregate update counters, and a
``quorum_distribution`` block with average, min, max, p50, p90, p95, and p99.
All writers use sorted-key compact JSON so byte-for-byte serialization is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch


ASYNC_DILOCO_METRICS_SCHEMA_VERSION = 1
ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION = 1

GLOBAL_MERGER_ROLE = "global_merger"
GENERATION_MANIFEST_KIND = "generation"
RECOVERY_CHECKPOINT_KIND = "recovery"
EXPORT_CHECKPOINT_KIND = "export"

GENERATION_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "generation",
    "requested_workers",
    "participating_workers",
    "quorum_threshold",
    "quorum_size",
    "accepted_updates",
    "stale_updates",
    "timed_out_updates",
    "failed_updates",
    "invalid_updates",
    "generation_duration_s",
    "merge_duration_s",
    "rebase_duration_s",
    "checkpoint_duration_s",
    "tokens_per_sec",
    "tokens_per_generation",
    "update_bytes",
    "loss_moving_average",
    "update_norms",
    "checkpoint_paths",
    "checkpoint_sizes",
    "latest_advanced",
    "resume_source_generation",
)

SUMMARY_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "requested_workers",
    "participating_workers",
    "quorum_distribution",
    "totals",
    "generations",
    "latest_advancement",
    "resume_source_generation",
)

CHECKPOINT_LATEST_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "generation",
    "manifest_path",
    "published_by",
    "published_at_s",
)


TensorState = Mapping[str, torch.Tensor]
DEFAULT_LOCAL_WORKERS_PER_NODE = 8
DEFAULT_LOCAL_QUORUM_TARGET = 6
DEFAULT_GLOBAL_QUORUM_FRACTION = 2.0 / 3.0


def _clone_state(state: TensorState) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def compute_dense_delta(base: TensorState, worker: TensorState) -> dict[str, torch.Tensor]:
    """Return ``worker - base`` for every tensor in ``base``.

    The function is strict about key equality so async manifests cannot silently
    merge an incomplete or differently-shaped state.
    """

    if set(base) != set(worker):
        missing = sorted(set(base) - set(worker))
        extra = sorted(set(worker) - set(base))
        raise ValueError(f"state keys differ: missing={missing} extra={extra}")
    delta: dict[str, torch.Tensor] = {}
    for name, base_tensor in base.items():
        worker_tensor = worker[name]
        if base_tensor.shape != worker_tensor.shape:
            raise ValueError(
                f"tensor {name!r} shape differs: {tuple(base_tensor.shape)} "
                f"!= {tuple(worker_tensor.shape)}")
        delta[name] = worker_tensor.detach() - base_tensor.detach()
    return delta


def apply_dense_delta(
    base: TensorState,
    delta: TensorState,
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Apply ``base + scale * delta`` and return a new tensor state."""

    if set(base) != set(delta):
        missing = sorted(set(base) - set(delta))
        extra = sorted(set(delta) - set(base))
        raise ValueError(f"delta keys differ: missing={missing} extra={extra}")
    return {
        name: tensor.detach() + float(scale) * delta[name].detach()
        for name, tensor in base.items()
    }


def weighted_mean_deltas(
    updates: Sequence["AsyncDiLoCoUpdate"],
    *,
    weight_by: str = "tokens",
) -> dict[str, torch.Tensor]:
    """Compute a weighted mean over update deltas."""

    if not updates:
        raise ValueError("at least one update is required")

    keys = set(updates[0].delta)
    for update in updates:
        if set(update.delta) != keys:
            raise ValueError(f"update {update.worker_id!r} delta keys differ")

    weights = []
    for update in updates:
        if weight_by == "tokens":
            weight = update.tokens
        elif weight_by == "local_steps":
            weight = update.local_steps
        elif weight_by == "equal":
            weight = 1
        else:
            raise ValueError(f"unknown weight_by mode: {weight_by!r}")
        if weight <= 0:
            raise ValueError(f"update {update.worker_id!r} has non-positive weight {weight}")
        weights.append(float(weight))

    total_weight = sum(weights)
    out: dict[str, torch.Tensor] = {}
    for name in sorted(keys):
        acc = torch.zeros_like(updates[0].delta[name], dtype=updates[0].delta[name].dtype)
        for update, weight in zip(updates, weights):
            acc = acc + update.delta[name] * (weight / total_weight)
        out[name] = acc
    return out


def rebase_state(
    local_state: TensorState,
    old_base: TensorState,
    new_base: TensorState,
) -> dict[str, torch.Tensor]:
    """Shift local state from ``old_base`` to ``new_base`` preserving displacement."""

    if set(local_state) != set(old_base) or set(local_state) != set(new_base):
        raise ValueError("local_state, old_base, and new_base must have identical keys")
    return {
        name: local_state[name].detach() + (new_base[name].detach() - old_base[name].detach())
        for name in local_state
    }


def state_num_bytes(state: TensorState) -> int:
    """Return total tensor payload bytes for a state or delta mapping."""

    return int(sum(tensor.numel() * tensor.element_size() for tensor in state.values()))


def state_norms(state: TensorState) -> dict[str, float]:
    """Return deterministic L2 norms for a state or delta mapping."""

    return {
        name: float(torch.linalg.vector_norm(tensor.detach().float()).item())
        for name, tensor in sorted(state.items())
    }


def _validate_tensor_state(state: TensorState, label: str) -> None:
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label} tensor {name!r} is not a torch.Tensor")
        if not torch.isfinite(tensor.detach()).all().item():
            raise ValueError(f"{label} tensor {name!r} contains non-finite values")


def _validate_update_delta(base_state: TensorState, update: "AsyncDiLoCoUpdate") -> None:
    if set(base_state) != set(update.delta):
        missing = sorted(set(base_state) - set(update.delta))
        extra = sorted(set(update.delta) - set(base_state))
        raise ValueError(
            f"update {update.worker_id!r} delta keys differ: missing={missing} extra={extra}"
        )
    for name, base_tensor in base_state.items():
        delta_tensor = update.delta[name]
        if base_tensor.shape != delta_tensor.shape:
            raise ValueError(
                f"update {update.worker_id!r} tensor {name!r} shape differs: "
                f"{tuple(delta_tensor.shape)} != {tuple(base_tensor.shape)}"
            )
        if not torch.isfinite(delta_tensor.detach()).all().item():
            raise ValueError(
                f"update {update.worker_id!r} tensor {name!r} contains non-finite values"
            )


def default_local_quorum(worker_count: int = DEFAULT_LOCAL_WORKERS_PER_NODE) -> int:
    """Return the production default local quorum target.

    Frontier nodes expose eight GPU workers for this path; the production
    default is six accepted worker updates per node.  Smaller synthetic tests
    keep a feasible default by capping at the worker count.
    """

    worker_count = int(worker_count)
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    return min(DEFAULT_LOCAL_QUORUM_TARGET, worker_count)


def default_global_quorum(node_count: int) -> int:
    """Return the default global quorum, ceil(2/3 * node_count)."""

    node_count = int(node_count)
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    return int(math.ceil(DEFAULT_GLOBAL_QUORUM_FRACTION * node_count))


@dataclass(frozen=True)
class AsyncDiLoCoUpdate:
    """One worker/node delta submitted to an async quorum DiLoCo generation."""

    worker_id: str
    base_generation: int
    delta: Mapping[str, torch.Tensor]
    tokens: int
    local_steps: int
    loss_moving_average: Mapping[str, float] = field(default_factory=dict)
    failed: bool = False
    timed_out: bool = False
    invalid: bool = False


@dataclass(frozen=True)
class AsyncDiLoCoGenerationMetrics:
    """Machine-readable metrics for one async DiLoCo generation."""

    run_id: str
    generation: int
    requested_workers: int
    participating_workers: int
    quorum_threshold: int
    quorum_size: int
    accepted_updates: int
    stale_updates: int
    timed_out_updates: int
    failed_updates: int
    invalid_updates: int
    generation_duration_s: float
    merge_duration_s: float
    rebase_duration_s: float
    checkpoint_duration_s: float
    tokens_per_sec: float
    tokens_per_generation: int
    update_bytes: Mapping[str, int] = field(default_factory=dict)
    loss_moving_average: Mapping[str, float] = field(default_factory=dict)
    update_norms: Mapping[str, float] = field(default_factory=dict)
    checkpoint_paths: Sequence[str] = field(default_factory=tuple)
    checkpoint_sizes: Mapping[str, int] = field(default_factory=dict)
    latest_advanced: bool = False
    resume_source_generation: int | None = None
    quorum_status: str = "advanced"
    schema_version: int = ASYNC_DILOCO_METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize({
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "requested_workers": self.requested_workers,
            "participating_workers": self.participating_workers,
            "quorum_threshold": self.quorum_threshold,
            "quorum_size": self.quorum_size,
            "accepted_updates": self.accepted_updates,
            "stale_updates": self.stale_updates,
            "timed_out_updates": self.timed_out_updates,
            "failed_updates": self.failed_updates,
            "invalid_updates": self.invalid_updates,
            "generation_duration_s": self.generation_duration_s,
            "merge_duration_s": self.merge_duration_s,
            "rebase_duration_s": self.rebase_duration_s,
            "checkpoint_duration_s": self.checkpoint_duration_s,
            "tokens_per_sec": self.tokens_per_sec,
            "tokens_per_generation": self.tokens_per_generation,
            "update_bytes": dict(self.update_bytes),
            "loss_moving_average": dict(self.loss_moving_average),
            "update_norms": dict(self.update_norms),
            "checkpoint_paths": list(self.checkpoint_paths),
            "checkpoint_sizes": dict(self.checkpoint_sizes),
            "latest_advanced": self.latest_advanced,
            "resume_source_generation": self.resume_source_generation,
            "quorum_status": self.quorum_status,
        })

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AsyncDiLoCoGenerationMetrics":
        _require_fields(payload, GENERATION_REQUIRED_FIELDS, "generation metrics")
        return cls(
            schema_version=int(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            generation=int(payload["generation"]),
            requested_workers=int(payload["requested_workers"]),
            participating_workers=int(payload["participating_workers"]),
            quorum_threshold=int(payload["quorum_threshold"]),
            quorum_size=int(payload["quorum_size"]),
            accepted_updates=int(payload["accepted_updates"]),
            stale_updates=int(payload["stale_updates"]),
            timed_out_updates=int(payload["timed_out_updates"]),
            failed_updates=int(payload["failed_updates"]),
            invalid_updates=int(payload["invalid_updates"]),
            generation_duration_s=float(payload["generation_duration_s"]),
            merge_duration_s=float(payload["merge_duration_s"]),
            rebase_duration_s=float(payload["rebase_duration_s"]),
            checkpoint_duration_s=float(payload["checkpoint_duration_s"]),
            tokens_per_sec=float(payload["tokens_per_sec"]),
            tokens_per_generation=int(payload["tokens_per_generation"]),
            update_bytes={str(k): int(v) for k, v in payload["update_bytes"].items()},
            loss_moving_average={
                str(k): float(v) for k, v in payload["loss_moving_average"].items()
            },
            update_norms={str(k): float(v) for k, v in payload["update_norms"].items()},
            checkpoint_paths=tuple(str(p) for p in payload["checkpoint_paths"]),
            checkpoint_sizes={str(k): int(v) for k, v in payload["checkpoint_sizes"].items()},
            latest_advanced=bool(payload["latest_advanced"]),
            resume_source_generation=(
                None if payload["resume_source_generation"] is None
                else int(payload["resume_source_generation"])
            ),
            quorum_status=str(
                payload.get(
                    "quorum_status",
                    "advanced" if bool(payload["latest_advanced"]) else "deferred",
                )
            ),
        )


@dataclass(frozen=True)
class AsyncDiLoCoMetricsSummary:
    """Run-level metrics summary for async quorum DiLoCo."""

    run_id: str
    requested_workers: int
    participating_workers: int
    generations: Sequence[AsyncDiLoCoGenerationMetrics]
    schema_version: int = ASYNC_DILOCO_METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        generation_dicts = [metric.to_dict() for metric in self.generations]
        latest_generation = None
        for metric in self.generations:
            if metric.latest_advanced:
                latest_generation = metric.generation
        health_counters = sustained_health_counters(self.generations)
        resume_sources = sorted({
            metric.resume_source_generation
            for metric in self.generations
            if metric.resume_source_generation is not None
        })
        return _canonicalize({
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "requested_workers": self.requested_workers,
            "participating_workers": self.participating_workers,
            "quorum_distribution": quorum_distribution(
                metric.quorum_size for metric in self.generations),
            "totals": {
                "accepted_updates": sum(m.accepted_updates for m in self.generations),
                "stale_updates": sum(m.stale_updates for m in self.generations),
                "timed_out_updates": sum(m.timed_out_updates for m in self.generations),
                "failed_updates": sum(m.failed_updates for m in self.generations),
                "invalid_updates": sum(m.invalid_updates for m in self.generations),
                "tokens_per_generation": sum(m.tokens_per_generation for m in self.generations),
                "update_bytes": _sum_mapping(m.update_bytes for m in self.generations),
            },
            "latest_advancement": {
                "advanced": latest_generation is not None,
                "generation": latest_generation,
            },
            "quorum_status": {
                "advanced": sum(1 for m in self.generations if m.quorum_status == "advanced"),
                "deferred": sum(1 for m in self.generations if m.quorum_status == "deferred"),
            },
            "sustained_health": health_counters,
            "resume_source_generation": resume_sources[0] if len(resume_sources) == 1 else None,
            "generations": generation_dicts,
        })

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AsyncDiLoCoMetricsSummary":
        _require_fields(payload, SUMMARY_REQUIRED_FIELDS, "metrics summary")
        generations = tuple(
            AsyncDiLoCoGenerationMetrics.from_dict(item)
            for item in payload["generations"]
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            requested_workers=int(payload["requested_workers"]),
            participating_workers=int(payload["participating_workers"]),
            generations=generations,
        )


@dataclass(frozen=True)
class AsyncDiLoCoMergeResult:
    """Result from one pure quorum merge."""

    state: Mapping[str, torch.Tensor]
    accepted_updates: Sequence[AsyncDiLoCoUpdate]
    stale_updates: Sequence[AsyncDiLoCoUpdate]
    timed_out_updates: Sequence[AsyncDiLoCoUpdate]
    failed_updates: Sequence[AsyncDiLoCoUpdate]
    invalid_updates: Sequence[AsyncDiLoCoUpdate]
    metrics: AsyncDiLoCoGenerationMetrics

    @property
    def advanced(self) -> bool:
        return self.metrics.quorum_status == "advanced"


@dataclass(frozen=True)
class AsyncDiLoCoCheckpointCadence:
    """Policy knobs for async DiLoCo durable checkpoint publication."""

    recovery_every_generations: int | None = None
    recovery_every_seconds: float | None = 600.0
    export_every_generations: int | None = None
    export_every_seconds: float | None = 3600.0
    finalization_reserve_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in ("recovery_every_generations", "export_every_generations"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")
        for name in (
            "recovery_every_seconds",
            "export_every_seconds",
            "finalization_reserve_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when configured")


@dataclass(frozen=True)
class AsyncDiLoCoCheckpointRecord:
    """Metadata for a generated manifest, recovery checkpoint, or export."""

    run_id: str
    generation: int
    kind: str
    path: str
    duration_s: float
    size_bytes: int
    overhead_percent: float
    latest_advanced: bool
    published_by: str
    finalized: bool
    reason: str | None = None
    schema_version: int = ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize({
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "kind": self.kind,
            "path": self.path,
            "duration_s": self.duration_s,
            "size_bytes": self.size_bytes,
            "overhead_percent": self.overhead_percent,
            "latest_advanced": self.latest_advanced,
            "published_by": self.published_by,
            "finalized": self.finalized,
            "reason": self.reason,
        })

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AsyncDiLoCoCheckpointRecord":
        return cls(
            schema_version=int(payload.get("schema_version", ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION)),
            run_id=str(payload["run_id"]),
            generation=int(payload["generation"]),
            kind=str(payload["kind"]),
            path=str(payload["path"]),
            duration_s=float(payload["duration_s"]),
            size_bytes=int(payload["size_bytes"]),
            overhead_percent=float(payload["overhead_percent"]),
            latest_advanced=bool(payload["latest_advanced"]),
            published_by=str(payload["published_by"]),
            finalized=bool(payload["finalized"]),
            reason=None if payload.get("reason") is None else str(payload["reason"]),
        )


@dataclass(frozen=True)
class AsyncDiLoCoPublishResult:
    """Result of publishing one finalized global generation."""

    generation_manifest: AsyncDiLoCoCheckpointRecord
    recovery_checkpoint: AsyncDiLoCoCheckpointRecord | None
    export_checkpoint: AsyncDiLoCoCheckpointRecord | None
    finalization_checkpoint: AsyncDiLoCoCheckpointRecord | None
    metrics: AsyncDiLoCoGenerationMetrics
    latest_path: str
    latest_advanced: bool

    @property
    def checkpoint_records(self) -> tuple[AsyncDiLoCoCheckpointRecord, ...]:
        records = [self.generation_manifest]
        for record in (
            self.recovery_checkpoint,
            self.export_checkpoint,
            self.finalization_checkpoint,
        ):
            if record is not None:
                records.append(record)
        return tuple(records)


@dataclass(frozen=True)
class AsyncDiLoCoResumeSource:
    """Restart source selected from finalized global generation manifests."""

    run_id: str
    generation: int
    manifest_path: str
    checkpoint_paths: tuple[str, ...]
    latest_path: str | None = None


class AsyncDiLoCoCheckpointManager:
    """Durable filesystem publisher for async DiLoCo global generations.

    Only ``GLOBAL_MERGER_ROLE`` may call ``publish_global_generation``. Other
    roles can write cache manifests with ``emit_cached_manifest`` but those
    files are explicitly non-authoritative and are ignored by resume selection.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        role: str,
        cadence: AsyncDiLoCoCheckpointCadence | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.role = role
        self.cadence = cadence or AsyncDiLoCoCheckpointCadence()
        self._time_source = time_source or time.monotonic
        self._last_recovery_generation: int | None = None
        self._last_recovery_time_s: float | None = None
        self._last_export_generation: int | None = None
        self._last_export_time_s: float | None = None
        self._finalization_generation: int | None = None

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    @property
    def latest_pt_path(self) -> Path:
        return self.root / "latest.pt"

    def emit_cached_manifest(
        self,
        *,
        generation: int,
        payload: Mapping[str, Any],
    ) -> AsyncDiLoCoCheckpointRecord:
        """Write a non-authoritative worker/supervisor cache manifest."""

        now_s = self._time_source()
        path = (
            self.root
            / "cache"
            / self.role
            / f"gen_{int(generation):06d}"
            / "manifest.json"
        )
        record = self._write_manifest_record(
            path=path,
            generation=int(generation),
            kind="cache",
            payload={
                "payload": dict(payload),
                "authoritative": False,
                "published_at_s": now_s,
            },
            duration_s=0.0,
            generation_duration_s=0.0,
            latest_advanced=False,
            finalized=False,
            reason=None,
        )
        return record

    def publish_global_generation(
        self,
        metrics: AsyncDiLoCoGenerationMetrics,
        *,
        checkpoint_path: str | Path | None = None,
        walltime_remaining_s: float | None = None,
        estimated_finalization_duration_s: float | None = None,
    ) -> AsyncDiLoCoPublishResult:
        """Publish a finalized global generation and atomically advance latest."""

        if self.role != GLOBAL_MERGER_ROLE:
            raise PermissionError("only the global merger may advance async DiLoCo latest")
        if metrics.run_id != self.run_id:
            raise ValueError(f"metrics run_id {metrics.run_id!r} does not match {self.run_id!r}")
        self._assert_can_advance_latest(metrics.generation)

        now_s = self._time_source()
        generation_manifest = self._write_generation_manifest(
            metrics,
            now_s=now_s,
            checkpoint_path=checkpoint_path,
        )
        recovery_checkpoint = None
        export_checkpoint = None
        finalization_checkpoint = None

        recovery_reason = self._recovery_reason(metrics.generation, now_s)
        if recovery_reason is not None:
            recovery_checkpoint = self._write_checkpoint_record(
                metrics,
                kind=RECOVERY_CHECKPOINT_KIND,
                reason=recovery_reason,
            )
            self._last_recovery_generation = metrics.generation
            self._last_recovery_time_s = now_s

        export_reason = self._export_reason(metrics.generation, now_s)
        if export_reason is not None:
            export_checkpoint = self._write_checkpoint_record(
                metrics,
                kind=EXPORT_CHECKPOINT_KIND,
                reason=export_reason,
            )
            self._last_export_generation = metrics.generation
            self._last_export_time_s = now_s

        if self.should_finalize(
            walltime_remaining_s=walltime_remaining_s,
            estimated_checkpoint_duration_s=estimated_finalization_duration_s,
        ):
            finalization_checkpoint = self._write_checkpoint_record(
                metrics,
                kind=RECOVERY_CHECKPOINT_KIND,
                reason="walltime_finalization",
            )
            self._finalization_generation = metrics.generation
            self._last_recovery_generation = metrics.generation
            self._last_recovery_time_s = now_s

        checkpoint_paths = [generation_manifest.path]
        checkpoint_sizes = {generation_manifest.path: generation_manifest.size_bytes}
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_paths.append(str(checkpoint_path))
            checkpoint_sizes[str(checkpoint_path)] = _path_size_bytes(checkpoint_path)
        for record in (recovery_checkpoint, export_checkpoint, finalization_checkpoint):
            if record is not None:
                checkpoint_paths.append(record.path)
                checkpoint_sizes[record.path] = record.size_bytes

        latest_payload = {
            "schema_version": ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": metrics.generation,
            "manifest_path": generation_manifest.path,
            "checkpoint_paths": checkpoint_paths,
            "published_by": self.role,
            "published_at_s": now_s,
        }
        if checkpoint_path is not None:
            latest_payload["checkpoint_path"] = str(checkpoint_path)
            _atomic_replace_symlink(self.latest_pt_path, checkpoint_path)
        _atomic_write_json(self.latest_path, latest_payload)

        updated_metrics = _replace_generation_checkpoint_metrics(
            metrics,
            checkpoint_paths=tuple(checkpoint_paths),
            checkpoint_sizes=checkpoint_sizes,
            checkpoint_duration_s=sum(record.duration_s for record in (
                generation_manifest,
                recovery_checkpoint,
                export_checkpoint,
                finalization_checkpoint,
            ) if record is not None),
            latest_advanced=True,
        )
        return AsyncDiLoCoPublishResult(
            generation_manifest=generation_manifest,
            recovery_checkpoint=recovery_checkpoint,
            export_checkpoint=export_checkpoint,
            finalization_checkpoint=finalization_checkpoint,
            metrics=updated_metrics,
            latest_path=str(self.latest_path),
            latest_advanced=True,
        )

    def should_finalize(
        self,
        *,
        walltime_remaining_s: float | None,
        estimated_checkpoint_duration_s: float | None = None,
    ) -> bool:
        if walltime_remaining_s is None:
            return False
        if self._finalization_generation is not None:
            return False
        estimated = 0.0 if estimated_checkpoint_duration_s is None else estimated_checkpoint_duration_s
        if walltime_remaining_s < estimated:
            return False
        return walltime_remaining_s <= self.cadence.finalization_reserve_seconds + estimated

    def select_resume_source(self) -> AsyncDiLoCoResumeSource | None:
        """Return the newest finalized global generation, ignoring partial data."""

        candidates = list(self._iter_finalized_global_manifests())
        if not candidates:
            return None
        newest_payload, newest_path = max(
            candidates,
            key=lambda item: int(item[0]["generation"]),
        )
        latest_path = None
        checkpoint_paths = tuple(str(p) for p in newest_payload.get("checkpoint_paths", ()))
        if newest_payload.get("checkpoint_path"):
            checkpoint_paths = (str(newest_payload["checkpoint_path"]), *checkpoint_paths)
        if self.latest_path.exists():
            try:
                latest_payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                latest_payload = {}
            if (
                latest_payload.get("run_id") == self.run_id
                and int(latest_payload.get("generation", -1)) == int(newest_payload["generation"])
                and latest_payload.get("published_by") == GLOBAL_MERGER_ROLE
            ):
                latest_path = str(self.latest_path)
                checkpoint_paths = tuple(str(p) for p in latest_payload.get("checkpoint_paths", ()))
                if latest_payload.get("checkpoint_path"):
                    checkpoint_paths = (str(latest_payload["checkpoint_path"]), *checkpoint_paths)
        return AsyncDiLoCoResumeSource(
            run_id=str(newest_payload["run_id"]),
            generation=int(newest_payload["generation"]),
            manifest_path=str(newest_path),
            checkpoint_paths=checkpoint_paths,
            latest_path=latest_path,
        )

    def _write_generation_manifest(
        self,
        metrics: AsyncDiLoCoGenerationMetrics,
        *,
        now_s: float,
        checkpoint_path: str | Path | None = None,
    ) -> AsyncDiLoCoCheckpointRecord:
        path = self._generation_manifest_path(metrics.generation)
        checkpoint_paths = [str(path)]
        if checkpoint_path is not None:
            checkpoint_paths.append(str(checkpoint_path))
        payload = {
            "metrics": metrics.to_dict(),
            "checkpoint_paths": checkpoint_paths,
            "authoritative": True,
            "published_at_s": now_s,
        }
        if checkpoint_path is not None:
            payload["checkpoint_path"] = str(checkpoint_path)
        return self._write_manifest_record(
            path=path,
            generation=metrics.generation,
            kind=GENERATION_MANIFEST_KIND,
            payload=payload,
            duration_s=0.0,
            generation_duration_s=metrics.generation_duration_s,
            latest_advanced=False,
            finalized=True,
            reason="every_generation",
        )

    def _write_checkpoint_record(
        self,
        metrics: AsyncDiLoCoGenerationMetrics,
        *,
        kind: str,
        reason: str,
    ) -> AsyncDiLoCoCheckpointRecord:
        start_s = self._time_source()
        path = self._checkpoint_manifest_path(kind, metrics.generation, reason)
        payload = {
            "metrics": metrics.to_dict(),
            "authoritative": True,
            "reason": reason,
            "published_at_s": start_s,
        }
        duration_s = max(0.0, self._time_source() - start_s)
        return self._write_manifest_record(
            path=path,
            generation=metrics.generation,
            kind=kind,
            payload=payload,
            duration_s=duration_s,
            generation_duration_s=metrics.generation_duration_s,
            latest_advanced=False,
            finalized=True,
            reason=reason,
        )

    def _write_manifest_record(
        self,
        *,
        path: Path,
        generation: int,
        kind: str,
        payload: Mapping[str, Any],
        duration_s: float,
        generation_duration_s: float,
        latest_advanced: bool,
        finalized: bool,
        reason: str | None,
    ) -> AsyncDiLoCoCheckpointRecord:
        record_payload = {
            "schema_version": ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": generation,
            "kind": kind,
            "published_by": self.role,
            "finalized": finalized,
            "latest_advanced": latest_advanced,
            "duration_s": duration_s,
            "size_bytes": 0,
            "overhead_percent": _checkpoint_overhead_percent(
                duration_s,
                generation_duration_s,
            ),
            "path": str(path),
            "reason": reason,
            **dict(payload),
        }
        _atomic_write_json(path, record_payload)
        size_bytes = _path_size_bytes(path)
        record_payload["size_bytes"] = size_bytes
        _atomic_write_json(path, record_payload)
        return AsyncDiLoCoCheckpointRecord(
            run_id=self.run_id,
            generation=generation,
            kind=kind,
            path=str(path),
            duration_s=duration_s,
            size_bytes=size_bytes,
            overhead_percent=float(record_payload["overhead_percent"]),
            latest_advanced=latest_advanced,
            published_by=self.role,
            finalized=finalized,
            reason=reason,
        )

    def _recovery_reason(self, generation: int, now_s: float) -> str | None:
        if (
            self.cadence.recovery_every_generations is None
            and self.cadence.recovery_every_seconds is None
        ):
            return None
        reasons = []
        if self._last_recovery_generation is None:
            reasons.append("initial")
        elif (
            self.cadence.recovery_every_generations is not None
            and generation - self._last_recovery_generation
            >= self.cadence.recovery_every_generations
        ):
            reasons.append("generation_interval")
        if (
            self._last_recovery_time_s is not None
            and self.cadence.recovery_every_seconds is not None
            and now_s - self._last_recovery_time_s >= self.cadence.recovery_every_seconds
        ):
            reasons.append("wall_clock_interval")
        return "+".join(reasons) if reasons else None

    def _export_reason(self, generation: int, now_s: float) -> str | None:
        if (
            self.cadence.export_every_generations is None
            and self.cadence.export_every_seconds is None
        ):
            return None
        reasons = []
        if self._last_export_generation is None:
            reasons.append("initial")
        elif (
            self.cadence.export_every_generations is not None
            and generation - self._last_export_generation
            >= self.cadence.export_every_generations
        ):
            reasons.append("generation_interval")
        if (
            self._last_export_time_s is not None
            and self.cadence.export_every_seconds is not None
            and now_s - self._last_export_time_s >= self.cadence.export_every_seconds
        ):
            reasons.append("wall_clock_interval")
        return "+".join(reasons) if reasons else None

    def _iter_finalized_global_manifests(
        self,
    ) -> Iterable[tuple[Mapping[str, Any], Path]]:
        for path in sorted((self.root / "generations").glob("gen_*/manifest.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("run_id") != self.run_id:
                continue
            if payload.get("published_by") != GLOBAL_MERGER_ROLE:
                continue
            if payload.get("kind") != GENERATION_MANIFEST_KIND:
                continue
            if payload.get("finalized") is not True:
                continue
            yield payload, path

    def _assert_can_advance_latest(self, generation: int) -> None:
        current_generation = self._current_authoritative_generation()
        if current_generation is None:
            return
        if int(generation) <= current_generation:
            raise ValueError(
                f"generation {generation} would not advance async DiLoCo latest "
                f"beyond current generation {current_generation}"
            )

    def _current_authoritative_generation(self) -> int | None:
        candidates = [
            int(payload["generation"])
            for payload, _path in self._iter_finalized_global_manifests()
        ]
        if self.latest_path.exists():
            try:
                payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            if (
                payload.get("run_id") == self.run_id
                and payload.get("published_by") == GLOBAL_MERGER_ROLE
            ):
                try:
                    candidates.append(int(payload["generation"]))
                except (KeyError, TypeError, ValueError):
                    pass
        if not candidates:
            return None
        return max(candidates)

    def _generation_manifest_path(self, generation: int) -> Path:
        return self.root / "generations" / f"gen_{generation:06d}" / "manifest.json"

    def _checkpoint_manifest_path(self, kind: str, generation: int, reason: str) -> Path:
        return (
            self.root
            / f"{kind}_checkpoints"
            / f"gen_{generation:06d}"
            / f"{reason}.json"
        )


def quorum_merge(
    base_state: TensorState,
    updates: Sequence[AsyncDiLoCoUpdate],
    *,
    run_id: str,
    generation: int,
    requested_workers: int,
    quorum_threshold: int,
    eta_outer: float = 1.0,
    weight_by: str = "tokens",
    generation_duration_s: float = 0.0,
    merge_duration_s: float = 0.0,
    rebase_duration_s: float = 0.0,
    checkpoint_duration_s: float = 0.0,
    checkpoint_paths: Sequence[str] = (),
    checkpoint_sizes: Mapping[str, int] | None = None,
    latest_advanced: bool = False,
    resume_source_generation: int | None = None,
) -> AsyncDiLoCoMergeResult:
    """Apply one reject-stale quorum merge over synthetic tensor states."""

    if requested_workers <= 0:
        raise ValueError("requested_workers must be positive")
    if quorum_threshold <= 0 or quorum_threshold > requested_workers:
        raise ValueError("quorum_threshold must be in [1, requested_workers]")
    _validate_tensor_state(base_state, "base_state")

    accepted: list[AsyncDiLoCoUpdate] = []
    stale: list[AsyncDiLoCoUpdate] = []
    timed_out: list[AsyncDiLoCoUpdate] = []
    failed: list[AsyncDiLoCoUpdate] = []
    invalid: list[AsyncDiLoCoUpdate] = []

    for update in updates:
        if update.failed:
            failed.append(update)
        elif update.timed_out:
            timed_out.append(update)
        elif update.invalid:
            invalid.append(update)
        elif update.base_generation != generation:
            stale.append(update)
        else:
            _validate_update_delta(base_state, update)
            accepted.append(update)

    checkpoint_sizes_dict = {} if checkpoint_sizes is None else dict(checkpoint_sizes)
    metrics_kwargs = {
        "run_id": run_id,
        "generation": generation,
        "requested_workers": requested_workers,
        "participating_workers": len(updates),
        "quorum_threshold": quorum_threshold,
        "quorum_size": len(accepted),
        "accepted_updates": len(accepted),
        "stale_updates": len(stale),
        "timed_out_updates": len(timed_out),
        "failed_updates": len(failed),
        "invalid_updates": len(invalid),
        "generation_duration_s": generation_duration_s,
        "merge_duration_s": merge_duration_s,
        "rebase_duration_s": rebase_duration_s,
        "checkpoint_duration_s": checkpoint_duration_s,
        "checkpoint_paths": tuple(checkpoint_paths),
        "checkpoint_sizes": checkpoint_sizes_dict,
        "resume_source_generation": resume_source_generation,
    }

    if len(accepted) < quorum_threshold:
        metrics = AsyncDiLoCoGenerationMetrics(
            **metrics_kwargs,
            tokens_per_sec=0.0,
            tokens_per_generation=0,
            update_bytes={
                "accepted": sum(state_num_bytes(update.delta) for update in accepted),
            },
            loss_moving_average={},
            update_norms={},
            latest_advanced=False,
            quorum_status="deferred",
        )
        return AsyncDiLoCoMergeResult(
            state=_clone_state(base_state),
            accepted_updates=tuple(accepted),
            stale_updates=tuple(stale),
            timed_out_updates=tuple(timed_out),
            failed_updates=tuple(failed),
            invalid_updates=tuple(invalid),
            metrics=metrics,
        )

    mean_delta = weighted_mean_deltas(accepted, weight_by=weight_by)
    merged_state = apply_dense_delta(base_state, mean_delta, scale=eta_outer)
    total_tokens = int(sum(update.tokens for update in accepted))
    tokens_per_sec = (
        total_tokens / generation_duration_s if generation_duration_s > 0.0 else 0.0
    )
    loss_ma = _weighted_scalar_mean(
        [update.loss_moving_average for update in accepted],
        [float(update.tokens) for update in accepted],
    )
    metrics = AsyncDiLoCoGenerationMetrics(
        **metrics_kwargs,
        tokens_per_sec=tokens_per_sec,
        tokens_per_generation=total_tokens,
        update_bytes={"accepted": sum(state_num_bytes(update.delta) for update in accepted)},
        loss_moving_average=loss_ma,
        update_norms=state_norms(mean_delta),
        latest_advanced=latest_advanced,
        quorum_status="advanced",
    )
    return AsyncDiLoCoMergeResult(
        state=merged_state,
        accepted_updates=tuple(accepted),
        stale_updates=tuple(stale),
        timed_out_updates=tuple(timed_out),
        failed_updates=tuple(failed),
        invalid_updates=tuple(invalid),
        metrics=metrics,
    )


def build_metrics_summary(
    *,
    run_id: str,
    requested_workers: int,
    participating_workers: int,
    generations: Sequence[AsyncDiLoCoGenerationMetrics],
) -> AsyncDiLoCoMetricsSummary:
    return AsyncDiLoCoMetricsSummary(
        run_id=run_id,
        requested_workers=requested_workers,
        participating_workers=participating_workers,
        generations=tuple(generations),
    )


def quorum_distribution(values: Iterable[int]) -> dict[str, float | int]:
    """Return average/min/max and percentile quorum-size statistics."""

    xs = sorted(int(value) for value in values)
    if not xs:
        return {
            "average": 0.0,
            "min": 0,
            "max": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "average": sum(xs) / len(xs),
        "min": xs[0],
        "max": xs[-1],
        "p50": percentile(xs, 50.0),
        "p90": percentile(xs, 90.0),
        "p95": percentile(xs, 95.0),
        "p99": percentile(xs, 99.0),
    }


def percentile(values: Sequence[int | float], q: float) -> float:
    """Linear-interpolated percentile over sorted or unsorted values."""

    if not values:
        return 0.0
    xs = sorted(float(value) for value in values)
    if len(xs) == 1:
        return xs[0]
    q = min(100.0, max(0.0, float(q)))
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def sustained_health_counters(
    generations: Sequence[AsyncDiLoCoGenerationMetrics],
) -> dict[str, int | bool]:
    """Return run-level counters suitable for sustained-health policy hooks."""

    advanced = 0
    deferred = 0
    current_deferred_streak = 0
    max_deferred_streak = 0
    current_nonadvanced_latest_streak = 0
    max_nonadvanced_latest_streak = 0
    for metric in generations:
        if metric.quorum_status == "advanced":
            advanced += 1
            current_deferred_streak = 0
        else:
            deferred += 1
            current_deferred_streak += 1
            max_deferred_streak = max(max_deferred_streak, current_deferred_streak)
        if metric.latest_advanced:
            current_nonadvanced_latest_streak = 0
        else:
            current_nonadvanced_latest_streak += 1
            max_nonadvanced_latest_streak = max(
                max_nonadvanced_latest_streak,
                current_nonadvanced_latest_streak,
            )
    return {
        "advanced_generations": advanced,
        "deferred_generations": deferred,
        "current_deferred_streak": current_deferred_streak,
        "max_deferred_streak": max_deferred_streak,
        "current_nonadvanced_latest_streak": current_nonadvanced_latest_streak,
        "max_nonadvanced_latest_streak": max_nonadvanced_latest_streak,
        "healthy": deferred == 0 and current_nonadvanced_latest_streak == 0,
    }


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize a metrics payload deterministically."""

    return json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"))


def write_metrics_json(path: str | Path, summary: AsyncDiLoCoMetricsSummary) -> None:
    Path(path).write_text(stable_json_dumps(summary.to_dict()) + "\n", encoding="utf-8")


def read_metrics_json(path: str | Path) -> AsyncDiLoCoMetricsSummary:
    return AsyncDiLoCoMetricsSummary.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_generation_metrics_jsonl(
    path: str | Path,
    generations: Sequence[AsyncDiLoCoGenerationMetrics],
) -> None:
    lines = [stable_json_dumps(metric.to_dict()) for metric in generations]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_generation_metrics_jsonl(path: str | Path) -> tuple[AsyncDiLoCoGenerationMetrics, ...]:
    metrics = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            metrics.append(AsyncDiLoCoGenerationMetrics.from_dict(json.loads(line)))
    return tuple(metrics)


def validate_generation_metrics(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, GENERATION_REQUIRED_FIELDS, "generation metrics")


def validate_metrics_summary(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, SUMMARY_REQUIRED_FIELDS, "metrics summary")


def validate_checkpoint_latest(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, CHECKPOINT_LATEST_REQUIRED_FIELDS, "checkpoint latest")


def _require_fields(payload: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def _replace_generation_checkpoint_metrics(
    metrics: AsyncDiLoCoGenerationMetrics,
    *,
    checkpoint_paths: Sequence[str],
    checkpoint_sizes: Mapping[str, int],
    checkpoint_duration_s: float,
    latest_advanced: bool,
) -> AsyncDiLoCoGenerationMetrics:
    return replace(
        metrics,
        checkpoint_paths=tuple(checkpoint_paths),
        checkpoint_sizes=dict(checkpoint_sizes),
        checkpoint_duration_s=float(checkpoint_duration_s),
        latest_advanced=bool(latest_advanced),
    )


def _checkpoint_overhead_percent(duration_s: float, generation_duration_s: float) -> float:
    if generation_duration_s <= 0.0:
        return 0.0
    return 100.0 * float(duration_s) / float(generation_duration_s)


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if not path.exists():
        return 0
    return int(sum(child.stat().st_size for child in path.rglob("*") if child.is_file()))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(stable_json_dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_replace_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"checkpoint target does not exist: {target_path}")
    tmp = link_path.with_name(f".{link_path.name}.{os.getpid()}.tmp")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        if target_path.parent.resolve() == link_path.parent.resolve():
            tmp.symlink_to(target_path.name)
        else:
            tmp.symlink_to(str(target_path))
        os.replace(tmp, link_path)
    finally:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("only scalar tensors can be serialized in metrics payloads")
        return _canonicalize(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric value: {value}")
        return value
    return value


def _sum_mapping(items: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: MutableMapping[str, int] = {}
    for item in items:
        for key, value in item.items():
            total[str(key)] = total.get(str(key), 0) + int(value)
    return dict(sorted(total.items()))


def _weighted_scalar_mean(
    rows: Sequence[Mapping[str, float]],
    weights: Sequence[float],
) -> dict[str, float]:
    sums: dict[str, float] = {}
    weight_sums: dict[str, float] = {}
    for row, weight in zip(rows, weights):
        for key, value in row.items():
            sums[str(key)] = sums.get(str(key), 0.0) + float(value) * weight
            weight_sums[str(key)] = weight_sums.get(str(key), 0.0) + weight
    return {
        key: sums[key] / weight_sums[key]
        for key in sorted(sums)
        if weight_sums[key] > 0.0
    }


@dataclass(frozen=True)
class AsyncDiLoCoWorkerSpec:
    worker_id: str
    gpu_id: int
    local_steps: int = 1
    tokens_per_step: int = 1024
    delay_s: float = 0.0
    fail_before_submit: bool = False
    delta_scale: float = 0.01

    @property
    def tokens(self) -> int:
        return int(self.local_steps * self.tokens_per_step)


@dataclass(frozen=True)
class AsyncDiLoCoWorkerReport:
    worker_id: str
    gpu_id: int
    base_generation: int
    update: AsyncDiLoCoUpdate | None
    elapsed_s: float
    tokens: int = 0
    failed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class AsyncDiLoCoSupervisorPrototypeResult:
    node_id: str
    generation: int
    node_update: AsyncDiLoCoUpdate | None
    worker_reports: Sequence[AsyncDiLoCoWorkerReport]
    metrics: AsyncDiLoCoGenerationMetrics


@dataclass(frozen=True)
class AsyncDiLoCoGlobalPrototypeResult:
    generation: int
    state: Mapping[str, torch.Tensor]
    metrics: AsyncDiLoCoGenerationMetrics
    checkpoint_behavior: Mapping[str, Any]
    publish_result: AsyncDiLoCoPublishResult | None = None


@dataclass(frozen=True)
class AsyncDiLoCoPrototypeResult:
    run_id: str
    generation: int
    initial_state_path: str | None
    supervisor: AsyncDiLoCoSupervisorPrototypeResult
    global_merger: AsyncDiLoCoGlobalPrototypeResult
    metrics_summary: AsyncDiLoCoMetricsSummary
    metrics_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize({
            "run_id": self.run_id,
            "generation": self.generation,
            "initial_state_path": self.initial_state_path,
            "effective_quorum": self.supervisor.metrics.quorum_size,
            "tokens_per_sec": self.global_merger.metrics.tokens_per_sec,
            "update_bytes": dict(self.global_merger.metrics.update_bytes),
            "generation_latency_s": self.global_merger.metrics.generation_duration_s,
            "checkpoint_behavior": dict(self.global_merger.checkpoint_behavior),
            "metrics_path": self.metrics_path,
            "metrics_summary": self.metrics_summary.to_dict(),
            "worker_reports": [
                {
                    "worker_id": report.worker_id,
                    "gpu_id": report.gpu_id,
                    "base_generation": report.base_generation,
                    "tokens": report.tokens,
                    "elapsed_s": report.elapsed_s,
                    "failed": report.failed,
                    "error": report.error,
                }
                for report in self.supervisor.worker_reports
            ],
        })


@dataclass(frozen=True)
class AsyncDiLoCoPrototypeConfig:
    """One local worker/supervisor/global-merger generation.

    ``initial_state_path`` is read-only. Durable ``latest`` publication happens
    only inside ``run_dir`` so debug loads cannot update production latest.
    """

    run_id: str
    generation: int = 0
    node_id: str = "node-0"
    worker_specs: Sequence[AsyncDiLoCoWorkerSpec] = field(default_factory=tuple)
    local_quorum: int | None = None
    global_quorum: int | None = None
    global_node_count: int = 1
    timeout_s: float = 30.0
    eta_outer: float = 1.0
    weight_by: str = "tokens"
    initial_state_path: str | Path | None = None
    initial_state: TensorState | None = None
    run_dir: str | Path | None = None
    use_processes: bool = False
    include_group_merger: bool = True


def load_async_diloco_readonly_state(path: str | Path) -> dict[str, torch.Tensor]:
    source = Path(path)
    if source.is_dir():
        source = source / "state.pt"
    if not source.exists():
        raise FileNotFoundError(f"async DiLoCo initial state not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    return _checkpoint_payload_to_tensor_state(payload)


def run_async_diloco_worker_supervisor_prototype(
    config: AsyncDiLoCoPrototypeConfig,
) -> AsyncDiLoCoPrototypeResult:
    worker_specs = tuple(config.worker_specs) or tuple(
        AsyncDiLoCoWorkerSpec(worker_id=f"worker-{idx}", gpu_id=idx)
        for idx in range(8)
    )
    local_quorum = (
        config.local_quorum
        if config.local_quorum is not None
        else default_local_quorum(len(worker_specs))
    )
    if local_quorum <= 0 or local_quorum > len(worker_specs):
        raise ValueError("local_quorum must be in [1, worker_count]")
    global_node_count = int(config.global_node_count)
    if global_node_count <= 0:
        raise ValueError("global_node_count must be positive")
    global_quorum = (
        config.global_quorum
        if config.global_quorum is not None
        else default_global_quorum(global_node_count)
    )
    if global_quorum <= 0 or global_quorum > global_node_count:
        raise ValueError("global_quorum must be in [1, global_node_count]")

    base_state = _prototype_base_state(config)
    supervisor = _run_node_supervisor_prototype(
        run_id=config.run_id,
        node_id=config.node_id,
        generation=config.generation,
        base_state=base_state,
        worker_specs=worker_specs,
        local_quorum=local_quorum,
        timeout_s=config.timeout_s,
        eta_outer=config.eta_outer,
        weight_by=config.weight_by,
        use_processes=config.use_processes,
    )
    node_update = supervisor.node_update
    if node_update is not None and config.include_group_merger:
        node_update = replace(node_update, worker_id=f"group-0/{node_update.worker_id}")
    global_result = _run_global_merger_prototype(
        run_id=config.run_id,
        generation=config.generation,
        base_state=base_state,
        node_update=node_update,
        requested_nodes=global_node_count,
        global_quorum=global_quorum,
        eta_outer=config.eta_outer,
        weight_by=config.weight_by,
        generation_duration_s=supervisor.metrics.generation_duration_s,
        run_dir=config.run_dir,
    )
    summary = build_metrics_summary(
        run_id=config.run_id,
        requested_workers=len(worker_specs),
        participating_workers=supervisor.metrics.participating_workers,
        generations=(supervisor.metrics, global_result.metrics),
    )
    metrics_path = None
    if config.run_dir is not None:
        metrics_path = str(Path(config.run_dir) / "prototype_metrics.json")
    result = AsyncDiLoCoPrototypeResult(
        run_id=config.run_id,
        generation=config.generation,
        initial_state_path=None if config.initial_state_path is None else str(config.initial_state_path),
        supervisor=supervisor,
        global_merger=global_result,
        metrics_summary=summary,
        metrics_path=metrics_path,
    )
    if metrics_path is not None:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(stable_json_dumps(result.to_dict()) + "\n", encoding="utf-8")
    return result


def _prototype_base_state(config: AsyncDiLoCoPrototypeConfig) -> dict[str, torch.Tensor]:
    if config.initial_state is not None:
        return _clone_state(config.initial_state)
    if config.initial_state_path is not None:
        return load_async_diloco_readonly_state(config.initial_state_path)
    return {
        "x": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "z": torch.tensor([2.0, 3.0], dtype=torch.float32),
    }


def _checkpoint_payload_to_tensor_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "x" in payload and "z" in payload:
        x_payload = payload["x"]
        z_payload = payload["z"]
        if torch.is_tensor(x_payload) and torch.is_tensor(z_payload):
            return {"x": x_payload.detach().clone().cpu(), "z": z_payload.detach().clone().cpu()}
        if isinstance(x_payload, Sequence) and isinstance(z_payload, Sequence):
            out = {
                f"x.{idx}": tensor.detach().clone().cpu()
                for idx, tensor in enumerate(x_payload)
                if torch.is_tensor(tensor)
            }
            out.update({
                f"z.{idx}": tensor.detach().clone().cpu()
                for idx, tensor in enumerate(z_payload)
                if torch.is_tensor(tensor)
            })
            if out:
                return out
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            state_dict = payload.get(key)
            if isinstance(state_dict, Mapping):
                tensors = {
                    str(name): tensor.detach().clone().cpu()
                    for name, tensor in sorted(state_dict.items())
                    if torch.is_tensor(tensor)
                }
                if tensors:
                    return {f"x.{name}": tensor for name, tensor in tensors.items()}
    raise ValueError("checkpoint payload does not contain x/z or model tensor state")


def _run_node_supervisor_prototype(
    *,
    run_id: str,
    node_id: str,
    generation: int,
    base_state: TensorState,
    worker_specs: Sequence[AsyncDiLoCoWorkerSpec],
    local_quorum: int,
    timeout_s: float,
    eta_outer: float,
    weight_by: str,
    use_processes: bool,
) -> AsyncDiLoCoSupervisorPrototypeResult:
    start_s = time.monotonic()
    reports = (
        _collect_process_worker_reports(base_state, generation, worker_specs, local_quorum, timeout_s)
        if use_processes
        else _collect_inline_worker_reports(base_state, generation, worker_specs, local_quorum, timeout_s)
    )
    elapsed_s = max(0.0, time.monotonic() - start_s)
    accepted_updates = [
        report.update for report in reports
        if report.update is not None and not report.failed
    ]
    missing_ids = sorted(set(spec.worker_id for spec in worker_specs) - set(report.worker_id for report in reports))
    status_updates = [
        _prototype_status_update(spec.worker_id, generation, base_state, timed_out=True)
        for spec in worker_specs
        if spec.worker_id in missing_ids
    ]
    status_updates.extend(
        _prototype_status_update(report.worker_id, generation, base_state, failed=True)
        for report in reports
        if report.failed
    )
    status_updates.extend(
        _prototype_status_update(report.worker_id, generation, base_state, timed_out=True)
        for report in reports
        if report.update is None and not report.failed
    )
    merge_start_s = time.monotonic()
    merge_result = quorum_merge(
        base_state,
        tuple(accepted_updates + status_updates),
        run_id=run_id,
        generation=generation,
        requested_workers=len(worker_specs),
        quorum_threshold=local_quorum,
        eta_outer=eta_outer,
        weight_by=weight_by,
        generation_duration_s=elapsed_s,
    )
    node_delta = compute_dense_delta(base_state, merge_result.state)
    update_bytes = {
        "worker": sum(state_num_bytes(update.delta) for update in accepted_updates),
        "node": state_num_bytes(node_delta) if merge_result.advanced else 0,
    }
    metrics = replace(
        merge_result.metrics,
        merge_duration_s=max(0.0, time.monotonic() - merge_start_s),
        update_bytes=update_bytes,
    )
    node_update = None
    if not merge_result.advanced:
        return AsyncDiLoCoSupervisorPrototypeResult(
            node_id=node_id,
            generation=generation,
            node_update=node_update,
            worker_reports=tuple(sorted(reports, key=lambda report: report.worker_id)),
            metrics=metrics,
        )
    node_update = AsyncDiLoCoUpdate(
        worker_id=node_id,
        base_generation=generation,
        delta=node_delta,
        tokens=metrics.tokens_per_generation,
        local_steps=sum(update.local_steps for update in accepted_updates),
        loss_moving_average=dict(metrics.loss_moving_average),
    )
    return AsyncDiLoCoSupervisorPrototypeResult(
        node_id=node_id,
        generation=generation,
        node_update=node_update,
        worker_reports=tuple(sorted(reports, key=lambda report: report.worker_id)),
        metrics=metrics,
    )


def _collect_inline_worker_reports(
    base_state: TensorState,
    generation: int,
    worker_specs: Sequence[AsyncDiLoCoWorkerSpec],
    local_quorum: int,
    timeout_s: float,
) -> tuple[AsyncDiLoCoWorkerReport, ...]:
    reports: list[AsyncDiLoCoWorkerReport] = []
    deadline_s = time.monotonic() + timeout_s
    for spec in worker_specs:
        if sum(1 for report in reports if report.update is not None and not report.failed) >= local_quorum:
            break
        if time.monotonic() >= deadline_s:
            break
        reports.append(_run_prototype_worker(base_state, generation, spec))
    return tuple(reports)


def _collect_process_worker_reports(
    base_state: TensorState,
    generation: int,
    worker_specs: Sequence[AsyncDiLoCoWorkerSpec],
    local_quorum: int,
    timeout_s: float,
) -> tuple[AsyncDiLoCoWorkerReport, ...]:
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    result_queue = ctx.Queue()
    processes: list[tuple[AsyncDiLoCoWorkerSpec, mp.Process]] = []
    for spec in worker_specs:
        process = ctx.Process(
            target=_prototype_worker_process_entry,
            args=(_clone_state(base_state), generation, spec, result_queue),
        )
        process.start()
        processes.append((spec, process))

    reports: list[AsyncDiLoCoWorkerReport] = []
    deadline_s = time.monotonic() + timeout_s
    while time.monotonic() < deadline_s:
        if sum(1 for report in reports if report.update is not None and not report.failed) >= local_quorum:
            break
        try:
            payload = result_queue.get(timeout=min(0.05, max(0.0, deadline_s - time.monotonic())))
            reports.append(_prototype_worker_report_from_payload(payload, base_state))
        except queue.Empty:
            continue

    reported = {report.worker_id for report in reports}
    for spec, process in processes:
        process.join(timeout=0.05)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(timeout=1.0)
        if spec.worker_id in reported:
            continue
        failed = not timed_out and process.exitcode not in (0, None)
        reports.append(AsyncDiLoCoWorkerReport(
            worker_id=spec.worker_id,
            gpu_id=spec.gpu_id,
            base_generation=generation,
            update=None,
            elapsed_s=0.0,
            failed=failed,
            error=f"exitcode={process.exitcode}" if failed else "timed out before submit",
        ))
    return tuple(reports)


def _prototype_worker_process_entry(
    base_state: TensorState,
    generation: int,
    spec: AsyncDiLoCoWorkerSpec,
    result_queue: Any,
) -> None:
    result_queue.put(_prototype_worker_report_to_payload(
        _run_prototype_worker(base_state, generation, spec)
    ))


def _prototype_worker_report_to_payload(report: AsyncDiLoCoWorkerReport) -> dict[str, Any]:
    update_payload = None
    if report.update is not None:
        update_payload = {
            "worker_id": report.update.worker_id,
            "base_generation": report.update.base_generation,
            "delta": {
                name: tensor.detach().cpu().tolist()
                for name, tensor in report.update.delta.items()
            },
            "tokens": report.update.tokens,
            "local_steps": report.update.local_steps,
            "loss_moving_average": dict(report.update.loss_moving_average),
            "failed": report.update.failed,
            "timed_out": report.update.timed_out,
            "invalid": report.update.invalid,
        }
    return {
        "worker_id": report.worker_id,
        "gpu_id": report.gpu_id,
        "base_generation": report.base_generation,
        "update": update_payload,
        "elapsed_s": report.elapsed_s,
        "tokens": report.tokens,
        "failed": report.failed,
        "error": report.error,
    }


def _prototype_worker_report_from_payload(
    payload: Mapping[str, Any],
    base_state: TensorState,
) -> AsyncDiLoCoWorkerReport:
    update = None
    update_payload = payload.get("update")
    if isinstance(update_payload, Mapping):
        update = AsyncDiLoCoUpdate(
            worker_id=str(update_payload["worker_id"]),
            base_generation=int(update_payload["base_generation"]),
            delta={
                str(name): torch.tensor(values, dtype=base_state[str(name)].dtype)
                for name, values in update_payload["delta"].items()
            },
            tokens=int(update_payload["tokens"]),
            local_steps=int(update_payload["local_steps"]),
            loss_moving_average={
                str(name): float(value)
                for name, value in update_payload["loss_moving_average"].items()
            },
            failed=bool(update_payload["failed"]),
            timed_out=bool(update_payload["timed_out"]),
            invalid=bool(update_payload["invalid"]),
        )
    return AsyncDiLoCoWorkerReport(
        worker_id=str(payload["worker_id"]),
        gpu_id=int(payload["gpu_id"]),
        base_generation=int(payload["base_generation"]),
        update=update,
        elapsed_s=float(payload["elapsed_s"]),
        tokens=int(payload["tokens"]),
        failed=bool(payload["failed"]),
        error=None if payload.get("error") is None else str(payload["error"]),
    )


def _run_prototype_worker(
    base_state: TensorState,
    generation: int,
    spec: AsyncDiLoCoWorkerSpec,
) -> AsyncDiLoCoWorkerReport:
    start_s = time.monotonic()
    if spec.delay_s > 0.0:
        time.sleep(spec.delay_s)
    if spec.fail_before_submit:
        return AsyncDiLoCoWorkerReport(
            worker_id=spec.worker_id,
            gpu_id=spec.gpu_id,
            base_generation=generation,
            update=None,
            elapsed_s=max(0.0, time.monotonic() - start_s),
            failed=True,
            error="configured failure before submit",
        )
    shift = float(spec.delta_scale * (spec.gpu_id + 1) * max(1, spec.local_steps))
    worker_state = {name: tensor + shift for name, tensor in base_state.items()}
    update = AsyncDiLoCoUpdate(
        worker_id=spec.worker_id,
        base_generation=generation,
        delta=compute_dense_delta(base_state, worker_state),
        tokens=spec.tokens,
        local_steps=spec.local_steps,
        loss_moving_average={"loss_100": max(0.0, 1.0 - 0.01 * spec.local_steps)},
    )
    return AsyncDiLoCoWorkerReport(
        worker_id=spec.worker_id,
        gpu_id=spec.gpu_id,
        base_generation=generation,
        update=update,
        elapsed_s=max(0.0, time.monotonic() - start_s),
        tokens=update.tokens,
    )


def _prototype_status_update(
    worker_id: str,
    generation: int,
    base_state: TensorState,
    *,
    timed_out: bool = False,
    failed: bool = False,
) -> AsyncDiLoCoUpdate:
    return AsyncDiLoCoUpdate(
        worker_id=worker_id,
        base_generation=generation,
        delta={name: torch.zeros_like(tensor) for name, tensor in base_state.items()},
        tokens=1,
        local_steps=1,
        failed=failed,
        timed_out=timed_out,
    )


def _run_global_merger_prototype(
    *,
    run_id: str,
    generation: int,
    base_state: TensorState,
    node_update: AsyncDiLoCoUpdate | None,
    requested_nodes: int,
    global_quorum: int,
    eta_outer: float,
    weight_by: str,
    generation_duration_s: float,
    run_dir: str | Path | None,
) -> AsyncDiLoCoGlobalPrototypeResult:
    merge_start_s = time.monotonic()
    node_updates: list[AsyncDiLoCoUpdate] = []
    if node_update is not None:
        node_updates.append(node_update)
    missing_nodes = requested_nodes - len(node_updates)
    for idx in range(max(0, missing_nodes)):
        node_updates.append(
            _prototype_status_update(
                f"missing-node-{idx}",
                generation,
                base_state,
                timed_out=True,
            )
        )
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
    metrics = replace(
        merge_result.metrics,
        merge_duration_s=max(0.0, time.monotonic() - merge_start_s),
        update_bytes={
            "node": 0 if node_update is None else state_num_bytes(node_update.delta),
            "global_state": state_num_bytes(merge_result.state) if merge_result.advanced else 0,
        },
    )
    publish_result = None
    checkpoint_behavior: dict[str, Any] = {
        "enabled": False,
        "latest_advanced": False,
        "paths": [],
        "sizes": {},
    }
    if run_dir is not None and merge_result.advanced:
        manager = AsyncDiLoCoCheckpointManager(
            run_dir,
            run_id=run_id,
            role=GLOBAL_MERGER_ROLE,
            cadence=AsyncDiLoCoCheckpointCadence(
                recovery_every_generations=None,
                recovery_every_seconds=None,
                export_every_generations=None,
                export_every_seconds=None,
            ),
        )
        publish_result = manager.publish_global_generation(metrics)
        metrics = publish_result.metrics
        checkpoint_behavior = {
            "enabled": True,
            "latest_advanced": publish_result.latest_advanced,
            "latest_path": publish_result.latest_path,
            "paths": list(metrics.checkpoint_paths),
            "sizes": dict(metrics.checkpoint_sizes),
        }
    return AsyncDiLoCoGlobalPrototypeResult(
        generation=generation,
        state=merge_result.state,
        metrics=metrics,
        checkpoint_behavior=checkpoint_behavior,
        publish_result=publish_result,
    )


__all__ = [
    "ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION",
    "ASYNC_DILOCO_METRICS_SCHEMA_VERSION",
    "CHECKPOINT_LATEST_REQUIRED_FIELDS",
    "DEFAULT_GLOBAL_QUORUM_FRACTION",
    "DEFAULT_LOCAL_QUORUM_TARGET",
    "DEFAULT_LOCAL_WORKERS_PER_NODE",
    "GENERATION_REQUIRED_FIELDS",
    "EXPORT_CHECKPOINT_KIND",
    "GENERATION_MANIFEST_KIND",
    "GLOBAL_MERGER_ROLE",
    "RECOVERY_CHECKPOINT_KIND",
    "SUMMARY_REQUIRED_FIELDS",
    "AsyncDiLoCoCheckpointCadence",
    "AsyncDiLoCoCheckpointManager",
    "AsyncDiLoCoCheckpointRecord",
    "AsyncDiLoCoGenerationMetrics",
    "AsyncDiLoCoMergeResult",
    "AsyncDiLoCoMetricsSummary",
    "AsyncDiLoCoGlobalPrototypeResult",
    "AsyncDiLoCoPrototypeConfig",
    "AsyncDiLoCoPrototypeResult",
    "AsyncDiLoCoPublishResult",
    "AsyncDiLoCoResumeSource",
    "AsyncDiLoCoSupervisorPrototypeResult",
    "AsyncDiLoCoUpdate",
    "AsyncDiLoCoWorkerReport",
    "AsyncDiLoCoWorkerSpec",
    "apply_dense_delta",
    "build_metrics_summary",
    "compute_dense_delta",
    "default_global_quorum",
    "default_local_quorum",
    "load_async_diloco_readonly_state",
    "quorum_distribution",
    "quorum_merge",
    "read_generation_metrics_jsonl",
    "read_metrics_json",
    "rebase_state",
    "run_async_diloco_worker_supervisor_prototype",
    "stable_json_dumps",
    "state_norms",
    "state_num_bytes",
    "sustained_health_counters",
    "validate_generation_metrics",
    "validate_checkpoint_latest",
    "validate_metrics_summary",
    "weighted_mean_deltas",
    "write_generation_metrics_jsonl",
    "write_metrics_json",
]
