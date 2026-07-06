"""Real train.py-backed async DiLoCo orchestration helpers.

This module keeps transport out of scope but runs real model/optimizer steps
through train.py's import-safe helpers.  It is the production-training
counterpart to the synthetic protocol prototype in :mod:`ndm.async_diloco`.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field, replace
import copy
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
        publish_result = manager.publish_global_generation(metrics)
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
        worker_state = _floating_state_dict(model)
        base_generation = generation if spec.stale_generation is None else int(spec.stale_generation)
        update = AsyncDiLoCoUpdate(
            worker_id=spec.worker_id,
            base_generation=base_generation,
            delta=compute_dense_delta(base_state, worker_state),
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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")


__all__ = [
    "RealAsyncDiLoCoConfig",
    "RealAsyncDiLoCoRunResult",
    "RealAsyncGlobalResult",
    "RealAsyncNodeResult",
    "RealAsyncWorkerReport",
    "RealAsyncWorkerSpec",
    "default_tiny_e97_train_args",
    "run_real_async_diloco",
]
