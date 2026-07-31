#!/usr/bin/env python3
"""Run one async DiLoCo prototype generation for a Frontier E97 debug job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from ndm.async_diloco import (
    AsyncDiLoCoPrototypeConfig,
    AsyncDiLoCoWorkerSpec,
    run_async_diloco_worker_supervisor_prototype,
    stable_json_dumps,
)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local async DiLoCo worker/supervisor prototype generation."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--local-quorum", type=int, default=8)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--tokens-per-step", type=int, default=1024)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--delta-scale", type=float, default=1.0e-8)
    parser.add_argument("--use-processes", action="store_true")
    parser.add_argument("--task-id", default=os.environ.get("WG_TASK_ID", "async-diloco-e97-1n-debug"))
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "manual"))
    parser.add_argument("--slurm-job-name", default=os.environ.get("SLURM_JOB_NAME", "async-diloco-e97-1n-debug"))
    parser.add_argument("--slurm-node-count", type=int, default=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
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
    if args.worker_count <= 0:
        raise ValueError("--worker-count must be positive")
    if args.local_quorum <= 0 or args.local_quorum > args.worker_count:
        raise ValueError("--local-quorum must be in [1, worker-count]")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E97 checkpoint is not readable: {checkpoint}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_before = _path_identity(checkpoint)
    production_latest_before = _path_identity(args.production_latest_path or None)
    start = time.monotonic()
    start_utc = _utc_now()

    worker_specs = tuple(
        AsyncDiLoCoWorkerSpec(
            worker_id=f"worker-{idx}",
            gpu_id=idx,
            local_steps=args.local_steps,
            tokens_per_step=args.tokens_per_step,
            delta_scale=args.delta_scale,
        )
        for idx in range(args.worker_count)
    )

    result = run_async_diloco_worker_supervisor_prototype(
        AsyncDiLoCoPrototypeConfig(
            run_id=args.run_id,
            generation=0,
            node_id="frontier-node-0",
            worker_specs=worker_specs,
            local_quorum=args.local_quorum,
            global_quorum=1,
            timeout_s=args.timeout_s,
            initial_state_path=checkpoint,
            run_dir=run_dir,
            use_processes=args.use_processes,
            include_group_merger=True,
        )
    )

    end_utc = _utc_now()
    elapsed_s = max(0.0, time.monotonic() - start)
    checkpoint_after = _path_identity(checkpoint)
    production_latest_after = _path_identity(args.production_latest_path or None)
    production_latest_changed = production_latest_before != production_latest_after
    local_latest = _path_identity(run_dir / "latest.json")
    supervisor_metrics = result.supervisor.metrics.to_dict()
    global_metrics = result.global_merger.metrics.to_dict()
    command = ""
    if args.command_file:
        command_path = Path(args.command_file)
        if command_path.exists():
            command = command_path.read_text(encoding="utf-8").strip()

    conclusion = "pass"
    if result.supervisor.metrics.quorum_size < args.local_quorum:
        conclusion = "no-go-local-quorum-not-met"
    elif not result.global_merger.metrics.latest_advanced:
        conclusion = "no-go-debug-latest-not-finalized"
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
            "nodes": args.slurm_node_count,
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
            "worker_count": args.worker_count,
            "local_quorum": args.local_quorum,
            "global_quorum": 1,
        },
        "effective_quorum": {
            "local": result.supervisor.metrics.quorum_size,
            "global": result.global_merger.metrics.quorum_size,
        },
        "update_counts": {
            "local": {
                "accepted": result.supervisor.metrics.accepted_updates,
                "stale": result.supervisor.metrics.stale_updates,
                "timed_out": result.supervisor.metrics.timed_out_updates,
                "failed": result.supervisor.metrics.failed_updates,
                "invalid": result.supervisor.metrics.invalid_updates,
            },
            "global": {
                "accepted": result.global_merger.metrics.accepted_updates,
                "stale": result.global_merger.metrics.stale_updates,
                "timed_out": result.global_merger.metrics.timed_out_updates,
                "failed": result.global_merger.metrics.failed_updates,
                "invalid": result.global_merger.metrics.invalid_updates,
            },
        },
        "generation_duration_s": {
            "local": result.supervisor.metrics.generation_duration_s,
            "global": result.global_merger.metrics.generation_duration_s,
            "elapsed": elapsed_s,
        },
        "tokens_per_sec": {
            "local": result.supervisor.metrics.tokens_per_sec,
            "global": result.global_merger.metrics.tokens_per_sec,
        },
        "checkpoint_finalization": {
            "latest_advanced": result.global_merger.metrics.latest_advanced,
            "latest_path": result.global_merger.checkpoint_behavior.get("latest_path"),
            "paths": list(result.global_merger.metrics.checkpoint_paths),
            "sizes": dict(result.global_merger.metrics.checkpoint_sizes),
            "duration_s": result.global_merger.metrics.checkpoint_duration_s,
            "debug_run_directory_only": not production_latest_changed,
        },
        "metrics_summary": result.metrics_summary.to_dict(),
        "prototype_metrics_path": result.metrics_path,
        "supervisor_generation_metrics": supervisor_metrics,
        "global_generation_metrics": global_metrics,
        "worker_reports": result.to_dict()["worker_reports"],
        "conclusion": conclusion,
    }
    _write_json(args.metrics_json, payload)
    print(stable_json_dumps(payload), flush=True)
    return 0 if conclusion == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
