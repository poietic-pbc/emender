#!/usr/bin/env python3
"""Exec a torchrun worker with CPU and memory bound to its GPU's NUMA node."""

from __future__ import annotations

import json
import os
import shutil
import sys


def _physical_gpu(local_rank: int) -> int:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible:
        return local_rank
    if local_rank >= len(visible):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds CUDA_VISIBLE_DEVICES={visible!r}")
    try:
        return int(visible[local_rank])
    except ValueError as exc:
        raise RuntimeError(
            "NUMA binding requires numeric CUDA_VISIBLE_DEVICES entries; "
            f"got {visible[local_rank]!r}") from exc


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: numa_local_rank_exec.py [--] TRAIN_SCRIPT [ARGS...]")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    physical_gpu = _physical_gpu(local_rank)
    # Host authority: GPUs 0-3 attach to NUMA node 0; GPUs 4-7 to node 1.
    if not 0 <= physical_gpu <= 7:
        raise RuntimeError(f"unsupported physical GPU id for this host: {physical_gpu}")
    numa_node = 0 if physical_gpu <= 3 else 1
    triton_prefix = os.environ.get("NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX")
    if triton_prefix:
        triton_cache = f"{triton_prefix}-rank{local_rank}"
        os.environ["TRITON_CACHE_DIR"] = triton_cache
        os.makedirs(triton_cache, exist_ok=True)
    else:
        triton_cache = os.environ.get("TRITON_CACHE_DIR")
    numactl = shutil.which("numactl")
    if numactl is None:
        raise RuntimeError("numactl is required for the 8-GPU CPU-offload run")

    command = [
        numactl,
        f"--cpunodebind={numa_node}",
        f"--membind={numa_node}",
        sys.executable,
        *argv,
    ]
    evidence = {
        "local_rank": local_rank,
        "physical_gpu": physical_gpu,
        "numa_node": numa_node,
        "triton_cache_dir": triton_cache,
        "command": command,
    }
    print("[numa-rank] " + json.dumps(evidence, sort_keys=True), flush=True)
    if os.environ.get("NUMA_LOCAL_RANK_EXEC_DRY_RUN") == "1":
        return 0
    os.execv(numactl, command)
    raise AssertionError("os.execv returned")


if __name__ == "__main__":
    raise SystemExit(main())
