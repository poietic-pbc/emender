#!/usr/bin/env python3
"""Run real train.py-backed async DiLoCo E97 training.

This entrypoint is intentionally small: orchestration logic lives in
``ndm.async_diloco_real`` so tests can import it without shelling out.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ndm.async_diloco import stable_json_dumps
from ndm.async_diloco import AsyncDiLoCoCheckpointCadence
from ndm.async_diloco_real import (
    RealAsyncDiLoCoConfig,
    RealAsyncFileRankConfig,
    RealAsyncWorkerSpec,
    default_tiny_e97_train_args,
    run_real_async_diloco,
    run_real_async_diloco_file_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real async DiLoCo E97 trainer using train.py helper steps."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--checkpoint", default="",
                        help="Optional train.py checkpoint used as the initial global state.")
    parser.add_argument("--data", default="")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--synthetic-token-stream", action="store_true",
                        help="Use deterministic local token batches for smoke tests.")
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--local-quorum", type=int, default=0,
                        help="Per-node quorum. Defaults to async DiLoCo local default.")
    parser.add_argument("--global-quorum", type=int, default=0,
                        help="Global node quorum. Defaults to ceil(2/3 * node-count).")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--eta-outer", type=float, default=1.0)
    parser.add_argument("--level", default="E97")
    parser.add_argument("--params", default="100m")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-state", type=int, default=None)
    parser.add_argument("--n-groups", type=int, default=None)
    parser.add_argument("--n-slots", type=int, default=None)
    parser.add_argument("--expansion", type=float, default=None)
    parser.add_argument("--state-expansion", type=int, default=None)
    parser.add_argument("--gate-activation", default=None)
    parser.add_argument("--linear-state", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--mlp-multiple", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true",
                        help="Store/train the worker model in bf16, matching train.py production memory mode.")
    parser.add_argument("--use-chunked-e97", action="store_true",
                        help="Use the chunked E97 kernel path exposed by train.py.")
    parser.add_argument("--e97-chunk-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--projection-chunk-size", type=int, default=0)
    parser.add_argument("--loss-chunk-size", type=int, default=0)
    parser.add_argument("--recovery-every-generations", type=int, default=-1,
                        help="Recovery checkpoint generation cadence; -1 disables generation cadence.")
    parser.add_argument("--recovery-every-seconds", type=float, default=-1.0,
                        help="Recovery checkpoint wall-clock cadence; -1 disables wall-clock cadence.")
    parser.add_argument("--export-every-generations", type=int, default=-1,
                        help="Export checkpoint generation cadence; -1 disables generation cadence.")
    parser.add_argument("--export-every-seconds", type=float, default=-1.0,
                        help="Export checkpoint wall-clock cadence; -1 disables wall-clock cadence.")
    parser.add_argument("--finalization-reserve-seconds", type=float, default=1200.0)
    parser.add_argument("--walltime-remaining-s", type=float, default=-1.0,
                        help="If set, publish a finalization checkpoint when inside the reserve window.")
    parser.add_argument("--estimated-finalization-duration-s", type=float, default=-1.0)
    parser.add_argument("--device", default=os.environ.get("ASYNC_DILOCO_DEVICE", "cpu"))
    parser.add_argument("--actual-multinode-file-quorum", action="store_true",
                        help="Run one Slurm-launched node rank and use rank 0 as a file quorum coordinator.")
    parser.add_argument("--node-rank", type=int, default=None,
                        help="Actual node rank for --actual-multinode-file-quorum; defaults to SLURM_PROCID.")
    return parser.parse_args()


def _optional_positive_int(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def _optional_positive_float(value: float) -> float | None:
    return None if float(value) < 0.0 else float(value)


def main() -> int:
    args = parse_args()
    if args.worker_count <= 0:
        raise ValueError("--worker-count must be positive")
    if args.node_count <= 0:
        raise ValueError("--node-count must be positive")
    if not args.synthetic_token_stream and not args.data:
        raise ValueError("--data is required unless --synthetic-token-stream is set")
    if args.actual_multinode_file_quorum and args.synthetic_token_stream:
        raise ValueError("--actual-multinode-file-quorum requires real data; synthetic token stream is disabled")

    train_overrides = {
        key: value
        for key, value in {
            "data": args.data or None,
            "tokenizer": args.tokenizer,
            "level": args.level,
            "params": args.params,
            "dim": args.dim,
            "depth": args.depth,
            "n_heads": args.n_heads,
            "n_state": args.n_state,
            "n_groups": args.n_groups,
            "n_slots": args.n_slots,
            "expansion": args.expansion,
            "state_expansion": args.state_expansion,
            "gate_activation": args.gate_activation,
            "linear_state": args.linear_state,
            "mlp_ratio": args.mlp_ratio,
            "mlp_multiple": args.mlp_multiple,
        }.items()
        if value is not None
    }
    train_args = default_tiny_e97_train_args(
        **train_overrides,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
        bf16=args.bf16,
        use_triton=(None if args.bf16 else 0),
        use_chunked_e97=1 if args.use_chunked_e97 else 0,
        e97_chunk_size=args.e97_chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        gradient_checkpointing=args.gradient_checkpointing,
        projection_chunk_size=args.projection_chunk_size,
        loss_chunk_size=args.loss_chunk_size,
    )
    if args.actual_multinode_file_quorum:
        node_rank = args.node_rank
        if node_rank is None:
            node_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
        global_quorum = None if args.global_quorum <= 0 else args.global_quorum
        if global_quorum is None:
            global_quorum = int((2 * args.node_count + 2) // 3)
        result = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
            run_id=args.run_id,
            run_dir=Path(args.run_dir),
            metrics_json=(args.metrics_json or None),
            train_args=train_args,
            node_rank=int(node_rank),
            node_count=args.node_count,
            global_quorum=int(global_quorum),
            local_steps=args.local_steps,
            timeout_s=args.timeout_s,
            eta_outer=args.eta_outer,
            initial_checkpoint=(Path(args.checkpoint) if args.checkpoint else None),
            synthetic_token_stream=False,
            device=args.device,
            walltime_remaining_s=_optional_positive_float(args.walltime_remaining_s),
            estimated_finalization_duration_s=_optional_positive_float(
                args.estimated_finalization_duration_s
            ),
            checkpoint_cadence=AsyncDiLoCoCheckpointCadence(
                recovery_every_generations=_optional_positive_int(args.recovery_every_generations),
                recovery_every_seconds=_optional_positive_float(args.recovery_every_seconds),
                export_every_generations=_optional_positive_int(args.export_every_generations),
                export_every_seconds=_optional_positive_float(args.export_every_seconds),
                finalization_reserve_seconds=float(args.finalization_reserve_seconds),
            ),
        ))
        print(stable_json_dumps(result), flush=True)
        return 0

    worker_specs = []
    for idx in range(args.worker_count):
        node_id = f"node-{idx % args.node_count}"
        worker_specs.append(RealAsyncWorkerSpec(
            worker_id=f"{node_id}/worker-{idx}",
            node_id=node_id,
            device=args.device,
            local_steps=args.local_steps,
            seed_offset=idx,
        ))

    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id=args.run_id,
        run_dir=Path(args.run_dir),
        metrics_json=(args.metrics_json or None),
        train_args=train_args,
        worker_specs=tuple(worker_specs),
        generations=args.generations,
        local_quorum=(None if args.local_quorum <= 0 else args.local_quorum),
        global_quorum=(None if args.global_quorum <= 0 else args.global_quorum),
        global_node_count=args.node_count,
        eta_outer=args.eta_outer,
        timeout_s=args.timeout_s,
        synthetic_token_stream=args.synthetic_token_stream,
        initial_checkpoint=(Path(args.checkpoint) if args.checkpoint else None),
        walltime_remaining_s=_optional_positive_float(args.walltime_remaining_s),
        estimated_finalization_duration_s=_optional_positive_float(
            args.estimated_finalization_duration_s
        ),
        checkpoint_cadence=AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=_optional_positive_int(args.recovery_every_generations),
            recovery_every_seconds=_optional_positive_float(args.recovery_every_seconds),
            export_every_generations=_optional_positive_int(args.export_every_generations),
            export_every_seconds=_optional_positive_float(args.export_every_seconds),
            finalization_reserve_seconds=float(args.finalization_reserve_seconds),
        ),
    ))
    print(stable_json_dumps({
        "run_id": result.run_id,
        "latest_generation": result.latest_generation,
        "latest_path": result.latest_path,
        "metrics_json": result.metrics_json,
        "global_quorum_status": [
            generation.metrics.quorum_status for generation in result.generations
        ],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
