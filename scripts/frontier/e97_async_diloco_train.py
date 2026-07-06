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
from ndm.async_diloco_real import (
    RealAsyncDiLoCoConfig,
    RealAsyncWorkerSpec,
    default_tiny_e97_train_args,
    run_real_async_diloco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real async DiLoCo E97 trainer using train.py helper steps."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--data", default="")
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
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("ASYNC_DILOCO_DEVICE", "cpu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_count <= 0:
        raise ValueError("--worker-count must be positive")
    if args.node_count <= 0:
        raise ValueError("--node-count must be positive")
    if not args.synthetic_token_stream and not args.data:
        raise ValueError("--data is required unless --synthetic-token-stream is set")

    train_args = default_tiny_e97_train_args(
        data=args.data or None,
        level=args.level,
        params=args.params,
        dim=args.dim,
        depth=args.depth,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
    )
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
