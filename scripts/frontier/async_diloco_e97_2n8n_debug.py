#!/usr/bin/env python3
"""Run async DiLoCo E97 multi-node debug ladder cases on Frontier."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from ndm.async_diloco import (
    AsyncDiLoCoCheckpointCadence,
    AsyncDiLoCoCheckpointManager,
    AsyncDiLoCoUpdate,
    AsyncDiLoCoWorkerSpec,
    GLOBAL_MERGER_ROLE,
    build_metrics_summary,
    load_async_diloco_readonly_state,
    quorum_distribution,
    quorum_merge,
    stable_json_dumps,
    state_num_bytes,
)
from ndm import async_diloco as async_core


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    payload: dict[str, Any] = {
        "path": str(candidate),
        "exists": candidate.exists() or candidate.is_symlink(),
    }
    if payload["exists"]:
        resolved = candidate.resolve()
        payload["resolved"] = str(resolved)
        payload["stat"] = {
            "size_bytes": resolved.stat().st_size,
            "mtime_ns": resolved.stat().st_mtime_ns,
        }
    return payload


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")


def _parse_csv_ints(value: str) -> set[int]:
    out: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def _build_worker_specs(
    *,
    node_idx: int,
    worker_count: int,
    local_steps: int,
    tokens_per_step: int,
    delta_scale: float,
    induce_local_lag_drop: bool,
    lag_delay_s: float,
) -> tuple[AsyncDiLoCoWorkerSpec, ...]:
    specs: list[AsyncDiLoCoWorkerSpec] = []
    for gpu_idx in range(worker_count):
        specs.append(
            AsyncDiLoCoWorkerSpec(
                worker_id=f"node-{node_idx:03d}/worker-{gpu_idx}",
                gpu_id=gpu_idx,
                local_steps=local_steps,
                tokens_per_step=tokens_per_step,
                delta_scale=delta_scale,
                fail_before_submit=induce_local_lag_drop and gpu_idx == worker_count - 2,
                delay_s=lag_delay_s if induce_local_lag_drop and gpu_idx == worker_count - 1 else 0.0,
            )
        )
    return tuple(specs)


def _status_update(
    worker_id: str,
    generation: int,
    base_state: dict[str, Any],
    *,
    timed_out: bool = False,
    failed: bool = False,
) -> AsyncDiLoCoUpdate:
    return AsyncDiLoCoUpdate(
        worker_id=worker_id,
        base_generation=generation,
        delta={name: tensor.new_zeros(tensor.shape) for name, tensor in base_state.items()},
        tokens=1,
        local_steps=1,
        timed_out=timed_out,
        failed=failed,
    )


def _run_global_merge(
    *,
    run_id: str,
    generation: int,
    base_state: dict[str, Any],
    node_updates: tuple[AsyncDiLoCoUpdate, ...],
    global_quorum: int,
    requested_nodes: int,
    generation_duration_s: float,
    run_dir: Path,
    eta_outer: float,
    weight_by: str,
    resume_source_generation: int | None,
    recovery_every_generations: int | None,
    recovery_every_seconds: float | None,
    export_every_generations: int | None,
    export_every_seconds: float | None,
) -> tuple[dict[str, Any], Any, Any]:
    merge_start_s = time.monotonic()
    merge_result = quorum_merge(
        base_state,
        node_updates,
        run_id=run_id,
        generation=generation,
        requested_workers=requested_nodes,
        quorum_threshold=global_quorum,
        eta_outer=eta_outer,
        weight_by=weight_by,
        generation_duration_s=generation_duration_s,
        resume_source_generation=resume_source_generation,
    )
    quorum_advanced = len(merge_result.accepted_updates) >= global_quorum
    metrics = replace(
        merge_result.metrics,
        merge_duration_s=max(0.0, time.monotonic() - merge_start_s),
        update_bytes={
            "node": sum(state_num_bytes(update.delta) for update in merge_result.accepted_updates),
            "global_state": state_num_bytes(merge_result.state) if quorum_advanced else 0,
        },
    )
    if not quorum_advanced:
        return dict(merge_result.state), metrics, None
    manager = AsyncDiLoCoCheckpointManager(
        run_dir,
        run_id=run_id,
        role=GLOBAL_MERGER_ROLE,
        cadence=AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=recovery_every_generations,
            recovery_every_seconds=recovery_every_seconds,
            export_every_generations=export_every_generations,
            export_every_seconds=export_every_seconds,
        ),
    )
    publish_result = manager.publish_global_generation(metrics)
    return dict(merge_result.state), publish_result.metrics, publish_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--node-count", type=int, default=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    parser.add_argument("--worker-count-per-node", type=int, default=8)
    parser.add_argument("--local-quorum", type=int, default=8)
    parser.add_argument("--global-quorum", type=int, default=1)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--tokens-per-step", type=int, default=1024)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--delta-scale", type=float, default=1.0e-8)
    parser.add_argument("--eta-outer", type=float, default=1.0)
    parser.add_argument("--weight-by", default="tokens", choices=("tokens", "local_steps", "equal"))
    parser.add_argument("--recovery-every-generations", type=int, default=1)
    parser.add_argument("--recovery-every-seconds", type=float, default=None)
    parser.add_argument("--export-every-generations", type=int, default=None)
    parser.add_argument("--export-every-seconds", type=float, default=None)
    parser.add_argument("--reuse-representative-node", action="store_true")
    parser.add_argument("--use-processes", action="store_true")
    parser.add_argument("--induce-local-lag-drop", action="store_true")
    parser.add_argument("--lag-delay-s", type=float, default=2.0)
    parser.add_argument("--global-drop-node-ids", default="")
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--task-id", default=os.environ.get("WG_TASK_ID", "async-diloco-e97-2n8n-debug"))
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "manual"))
    parser.add_argument("--slurm-job-name", default=os.environ.get("SLURM_JOB_NAME", "async-diloco-e97-2n8n"))
    parser.add_argument("--requested-walltime", default=os.environ.get("REQUESTED_WALLTIME", "00:20:00"))
    parser.add_argument("--requested-node-hours", type=float, default=float(os.environ.get("REQUESTED_NODE_HOURS", "0.333333")))
    parser.add_argument("--command-file", default="")
    parser.add_argument("--stdout-path", default="")
    parser.add_argument("--stderr-path", default="")
    parser.add_argument("--production-latest-path", default="")
    parser.add_argument("--training-target", default="E97_1.3B_step483000_async_diloco_debug")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.node_count <= 0:
        raise ValueError("--node-count must be positive")
    if args.worker_count_per_node <= 0:
        raise ValueError("--worker-count-per-node must be positive")
    if args.local_quorum <= 0 or args.local_quorum > args.worker_count_per_node:
        raise ValueError("--local-quorum must be in [1, worker-count-per-node]")
    if args.global_quorum <= 0 or args.global_quorum > args.node_count:
        raise ValueError("--global-quorum must be in [1, node-count]")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E97 checkpoint is not readable: {checkpoint}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_before = _path_identity(checkpoint)
    production_latest_before = _path_identity(args.production_latest_path or None)
    start_s = time.monotonic()
    start_utc = _utc_now()
    base_state = load_async_diloco_readonly_state(checkpoint)
    dropped_node_ids = _parse_csv_ints(args.global_drop_node_ids)

    node_results = []
    global_updates: list[AsyncDiLoCoUpdate] = []
    representative_result = None
    for node_idx in range(args.node_count):
        if args.reuse_representative_node and representative_result is not None:
            supervisor = replace(representative_result, node_id=f"node-{node_idx:03d}")
        else:
            specs = _build_worker_specs(
                node_idx=node_idx,
                worker_count=args.worker_count_per_node,
                local_steps=args.local_steps,
                tokens_per_step=args.tokens_per_step,
                delta_scale=args.delta_scale,
                induce_local_lag_drop=args.induce_local_lag_drop,
                lag_delay_s=args.lag_delay_s,
            )
            supervisor = async_core._run_node_supervisor_prototype(
                run_id=args.run_id,
                node_id=f"node-{node_idx:03d}",
                generation=0,
                base_state=base_state,
                worker_specs=specs,
                local_quorum=args.local_quorum,
                timeout_s=args.timeout_s,
                eta_outer=args.eta_outer,
                weight_by=args.weight_by,
                use_processes=args.use_processes,
            )
            if representative_result is None:
                representative_result = supervisor
        node_results.append(supervisor)
        node_update = supervisor.node_update
        global_worker_id = f"group-0/{supervisor.node_id}"
        if node_update is not None:
            node_update = replace(node_update, worker_id=global_worker_id)
        if node_idx in dropped_node_ids or node_update is None:
            global_updates.append(_status_update(global_worker_id, 0, base_state, timed_out=True))
        else:
            global_updates.append(node_update)

    generation_duration_s = max(
        (result.metrics.generation_duration_s for result in node_results),
        default=0.0,
    )
    final_state, global_metrics, publish_result = _run_global_merge(
        run_id=args.run_id,
        generation=0,
        base_state=base_state,
        node_updates=tuple(global_updates),
        global_quorum=args.global_quorum,
        requested_nodes=args.node_count,
        generation_duration_s=generation_duration_s,
        run_dir=run_dir,
        eta_outer=args.eta_outer,
        weight_by=args.weight_by,
        resume_source_generation=None,
        recovery_every_generations=args.recovery_every_generations,
        recovery_every_seconds=args.recovery_every_seconds,
        export_every_generations=args.export_every_generations,
        export_every_seconds=args.export_every_seconds,
    )

    resume_payload = {"tested": False}
    if args.resume_check:
        manager = AsyncDiLoCoCheckpointManager(run_dir, run_id=args.run_id, role=GLOBAL_MERGER_ROLE)
        resume_source = manager.select_resume_source()
        if resume_source is None:
            resume_payload = {
                "tested": True,
                "selected_generation": None,
                "latest_advanced": False,
                "reason": "no finalized generation selected",
            }
        else:
            resumed_state, resumed_metrics, resumed_publish = _run_global_merge(
                run_id=args.run_id,
                generation=resume_source.generation + 1,
                base_state=final_state,
                node_updates=tuple(
                    replace(
                        update,
                        base_generation=resume_source.generation + 1,
                        worker_id=f"resume/{update.worker_id}",
                    )
                    for update in global_updates
                    if not update.timed_out and not update.failed and not update.invalid
                ),
                global_quorum=args.global_quorum,
                requested_nodes=args.node_count,
                generation_duration_s=generation_duration_s,
                run_dir=run_dir,
                eta_outer=args.eta_outer,
                weight_by=args.weight_by,
                resume_source_generation=resume_source.generation,
                recovery_every_generations=args.recovery_every_generations,
                recovery_every_seconds=args.recovery_every_seconds,
                export_every_generations=args.export_every_generations,
                export_every_seconds=args.export_every_seconds,
            )
            final_state = resumed_state
            resume_payload = {
                "tested": True,
                "selected_generation": resume_source.generation,
                "selected_manifest_path": resume_source.manifest_path,
                "selected_latest_path": resume_source.latest_path,
                "published_generation": resumed_metrics.generation,
                "latest_path": None if resumed_publish is None else resumed_publish.latest_path,
                "latest_advanced": False if resumed_publish is None else resumed_publish.latest_advanced,
                "metrics": resumed_metrics.to_dict(),
            }

    end_utc = _utc_now()
    elapsed_s = max(0.0, time.monotonic() - start_s)
    checkpoint_after = _path_identity(checkpoint)
    production_latest_after = _path_identity(args.production_latest_path or None)
    production_latest_changed = production_latest_before != production_latest_after
    local_latest = _path_identity(run_dir / "latest.json")
    command = ""
    if args.command_file:
        command_path = Path(args.command_file)
        if command_path.exists():
            command = command_path.read_text(encoding="utf-8").strip()

    local_metrics = [result.metrics for result in node_results]
    checkpoint_records = (
        [] if publish_result is None
        else [record.to_dict() for record in publish_result.checkpoint_records]
    )
    recovery_records = [
        record for record in checkpoint_records
        if record.get("kind") == "recovery" and record.get("finalized")
    ]
    checkpoint_overhead_percent = sum(
        float(record.get("overhead_percent", 0.0)) for record in checkpoint_records
    )
    checkpoint_total_size_bytes = sum(
        int(record.get("size_bytes", 0)) for record in checkpoint_records
    )
    local_loss_moving_average = {
        result.node_id: dict(result.metrics.loss_moving_average) for result in node_results
    }
    summary = build_metrics_summary(
        run_id=args.run_id,
        requested_workers=args.node_count,
        participating_workers=sum(metric.participating_workers for metric in local_metrics),
        generations=tuple(local_metrics + [global_metrics]),
    )
    local_quorums = [metric.quorum_size for metric in local_metrics]
    local_totals = {
        "accepted": sum(metric.accepted_updates for metric in local_metrics),
        "stale": sum(metric.stale_updates for metric in local_metrics),
        "timed_out": sum(metric.timed_out_updates for metric in local_metrics),
        "failed": sum(metric.failed_updates for metric in local_metrics),
        "invalid": sum(metric.invalid_updates for metric in local_metrics),
    }

    conclusion = "pass"
    if any(metric.quorum_size < args.local_quorum for metric in local_metrics):
        conclusion = "no-go-local-quorum-not-met"
    elif global_metrics.quorum_size < args.global_quorum:
        conclusion = "no-go-global-quorum-not-met"
    elif not global_metrics.latest_advanced:
        conclusion = "no-go-debug-latest-not-finalized"
    elif args.resume_check and not resume_payload.get("latest_advanced"):
        conclusion = "no-go-resume-latest-not-finalized"
    elif production_latest_changed:
        conclusion = "no-go-production-latest-changed"

    payload = {
        "schema_version": 1,
        "task_id": args.task_id,
        "run_id": args.run_id,
        "training_target": args.training_target,
        "checkpoint": {
            "before": checkpoint_before,
            "after": checkpoint_after,
            "modified_by_run": checkpoint_before != checkpoint_after,
        },
        "run_dir": str(run_dir),
        "debug_latest": local_latest,
        "production_latest_guard": {
            "before": production_latest_before,
            "after": production_latest_after,
            "changed": production_latest_changed,
        },
        "slurm": {
            "job_id": args.slurm_job_id,
            "job_name": args.slurm_job_name,
            "nodes": args.node_count,
            "requested_walltime": args.requested_walltime,
            "requested_node_hours": args.requested_node_hours,
            "stdout": args.stdout_path,
            "stderr": args.stderr_path,
        },
        "command": command,
        "timing": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "elapsed_s": elapsed_s,
        },
        "configured_quorum": {
            "nodes": args.node_count,
            "worker_count_per_node": args.worker_count_per_node,
            "local_quorum": args.local_quorum,
            "global_quorum": args.global_quorum,
            "reuse_representative_node": args.reuse_representative_node,
        },
        "effective_quorum": {
            "local_by_node": {result.node_id: result.metrics.quorum_size for result in node_results},
            "local_distribution": quorum_distribution(local_quorums),
            "global": global_metrics.quorum_size,
        },
        "induced_lag_drop": {
            "local_enabled": args.induce_local_lag_drop,
            "global_dropped_node_ids": sorted(dropped_node_ids),
            "lag_delay_s": args.lag_delay_s,
            "measured": bool(args.induce_local_lag_drop or dropped_node_ids),
        },
        "update_counts": {
            "local_total": local_totals,
            "global": {
                "accepted": global_metrics.accepted_updates,
                "stale": global_metrics.stale_updates,
                "timed_out": global_metrics.timed_out_updates,
                "failed": global_metrics.failed_updates,
                "invalid": global_metrics.invalid_updates,
            },
        },
        "duration_s": {
            "local_merge_by_node": {result.node_id: result.metrics.merge_duration_s for result in node_results},
            "global_merge": global_metrics.merge_duration_s,
            "global_rebase": global_metrics.rebase_duration_s,
            "checkpoint": global_metrics.checkpoint_duration_s,
            "elapsed": elapsed_s,
        },
        "loss_moving_average": {
            "local_by_node": local_loss_moving_average,
            "global": dict(global_metrics.loss_moving_average),
        },
        "tokens_per_sec": {
            "local_by_node": {result.node_id: result.metrics.tokens_per_sec for result in node_results},
            "global": global_metrics.tokens_per_sec,
            "aggregate": sum(metric.tokens_per_generation for metric in local_metrics) / elapsed_s if elapsed_s > 0.0 else 0.0,
        },
        "checkpoint_cadence": {
            "recovery_every_generations": args.recovery_every_generations,
            "recovery_every_seconds": args.recovery_every_seconds,
            "export_every_generations": args.export_every_generations,
            "export_every_seconds": args.export_every_seconds,
            "mode": "generation_or_wallclock",
        },
        "checkpoint_finalization": {
            "latest_advanced": global_metrics.latest_advanced,
            "latest_path": publish_result.latest_path,
            "paths": list(global_metrics.checkpoint_paths),
            "sizes": dict(global_metrics.checkpoint_sizes),
            "duration_s": global_metrics.checkpoint_duration_s,
            "records": checkpoint_records,
            "recovery_records": recovery_records,
            "total_size_bytes": checkpoint_total_size_bytes,
            "overhead_percent": checkpoint_overhead_percent,
            "debug_run_directory_only": not production_latest_changed,
        },
        "resume_check": resume_payload,
        "metrics_summary": summary.to_dict(),
        "node_generation_metrics": {
            result.node_id: result.metrics.to_dict() for result in node_results
        },
        "global_generation_metrics": global_metrics.to_dict(),
        "worker_reports": {
            result.node_id: [
                {
                    "worker_id": report.worker_id,
                    "gpu_id": report.gpu_id,
                    "base_generation": report.base_generation,
                    "tokens": report.tokens,
                    "elapsed_s": report.elapsed_s,
                    "failed": report.failed,
                    "error": report.error,
                }
                for report in result.worker_reports
            ]
            for result in node_results
        },
        "conclusion": conclusion,
    }
    _write_json(args.metrics_json, payload)
    _write_json(run_dir / "final_state_summary.json", {"tensor_count": len(final_state)})
    print(stable_json_dumps(payload), flush=True)
    return 0 if conclusion == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
