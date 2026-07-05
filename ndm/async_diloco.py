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
import os
from pathlib import Path
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
        walltime_remaining_s: float | None = None,
        estimated_finalization_duration_s: float | None = None,
    ) -> AsyncDiLoCoPublishResult:
        """Publish a finalized global generation and atomically advance latest."""

        if self.role != GLOBAL_MERGER_ROLE:
            raise PermissionError("only the global merger may advance async DiLoCo latest")
        if metrics.run_id != self.run_id:
            raise ValueError(f"metrics run_id {metrics.run_id!r} does not match {self.run_id!r}")

        now_s = self._time_source()
        generation_manifest = self._write_generation_manifest(metrics, now_s=now_s)
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
        latest_path = str(self.latest_path) if self.latest_path.exists() else None
        checkpoint_paths = tuple(str(p) for p in newest_payload.get("checkpoint_paths", ()))
        if self.latest_path.exists():
            try:
                latest_payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                latest_payload = {}
            if (
                latest_payload.get("run_id") == self.run_id
                and int(latest_payload.get("generation", -1)) == int(newest_payload["generation"])
            ):
                checkpoint_paths = tuple(str(p) for p in latest_payload.get("checkpoint_paths", ()))
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
    ) -> AsyncDiLoCoCheckpointRecord:
        path = self._generation_manifest_path(metrics.generation)
        payload = {
            "metrics": metrics.to_dict(),
            "checkpoint_paths": [str(path)],
            "authoritative": True,
            "published_at_s": now_s,
        }
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
            accepted.append(update)

    if len(accepted) < quorum_threshold:
        raise RuntimeError(
            f"quorum not reached for generation {generation}: "
            f"accepted={len(accepted)} threshold={quorum_threshold}")

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
        run_id=run_id,
        generation=generation,
        requested_workers=requested_workers,
        participating_workers=len(updates),
        quorum_threshold=quorum_threshold,
        quorum_size=len(accepted),
        accepted_updates=len(accepted),
        stale_updates=len(stale),
        timed_out_updates=len(timed_out),
        failed_updates=len(failed),
        invalid_updates=len(invalid),
        generation_duration_s=generation_duration_s,
        merge_duration_s=merge_duration_s,
        rebase_duration_s=rebase_duration_s,
        checkpoint_duration_s=checkpoint_duration_s,
        tokens_per_sec=tokens_per_sec,
        tokens_per_generation=total_tokens,
        update_bytes={"accepted": sum(state_num_bytes(update.delta) for update in accepted)},
        loss_moving_average=loss_ma,
        update_norms=state_norms(mean_delta),
        checkpoint_paths=tuple(checkpoint_paths),
        checkpoint_sizes={} if checkpoint_sizes is None else dict(checkpoint_sizes),
        latest_advanced=latest_advanced,
        resume_source_generation=resume_source_generation,
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


__all__ = [
    "ASYNC_DILOCO_CHECKPOINT_SCHEMA_VERSION",
    "ASYNC_DILOCO_METRICS_SCHEMA_VERSION",
    "CHECKPOINT_LATEST_REQUIRED_FIELDS",
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
    "AsyncDiLoCoPublishResult",
    "AsyncDiLoCoResumeSource",
    "AsyncDiLoCoUpdate",
    "apply_dense_delta",
    "build_metrics_summary",
    "compute_dense_delta",
    "quorum_distribution",
    "quorum_merge",
    "read_generation_metrics_jsonl",
    "read_metrics_json",
    "rebase_state",
    "stable_json_dumps",
    "state_norms",
    "state_num_bytes",
    "validate_generation_metrics",
    "validate_checkpoint_latest",
    "validate_metrics_summary",
    "weighted_mean_deltas",
    "write_generation_metrics_jsonl",
    "write_metrics_json",
]
