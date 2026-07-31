#!/usr/bin/env python3
"""Minimal torch.distributed allreduce diagnostic for Frontier RCCL probes."""

import argparse
import json
import os
import socket
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


DTYPES = {
    "float32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def parse_sizes(raw: str) -> List[Tuple[str, int]]:
    sizes: List[Tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, value = item.split("=", 1)
        else:
            name, value = item, item
        sizes.append((name.strip(), int(value.strip())))
    if not sizes:
        raise ValueError("no tensor sizes were provided")
    return sizes


def env_snapshot() -> Dict[str, str]:
    prefixes = (
        "SLURM",
        "MPICH",
        "NCCL",
        "RCCL",
        "FI_",
        "MASTER",
        "ROCR",
        "HIP",
        "HSA",
        "OMP",
        "LD_LIBRARY_PATH",
        "CRAY",
        "ROCM",
    )
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def one_allreduce(name: str, numel: int, dtype: torch.dtype, device: torch.device) -> Dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    result: Dict[str, Any] = {
        "name": name,
        "numel": numel,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes_per_rank": numel * torch.empty((), dtype=dtype).element_size(),
    }

    try:
        tensor = torch.empty(numel, device=device, dtype=dtype)
        tensor.fill_(float(rank + 1))
        torch.cuda.synchronize(device)
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        expected = float(world_size * (world_size + 1) // 2)
        first = float(tensor[0].item())
        last = float(tensor[-1].item())
        ok = abs(first - expected) <= max(1.0e-3, abs(expected) * 1.0e-3) and abs(
            last - expected
        ) <= max(1.0e-3, abs(expected) * 1.0e-3)
        result.update(
            {
                "status": "pass" if ok else "fail",
                "elapsed_s": elapsed,
                "expected": expected,
                "first": first,
                "last": last,
                "bandwidth_gib_s_per_rank": (result["bytes_per_rank"] / elapsed / (1024**3))
                if elapsed > 0
                else None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must report any failure.
        result.update({"status": "error", "error": repr(exc)})
    finally:
        try:
            del tensor
            torch.cuda.empty_cache()
        except Exception:
            pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=os.environ.get(
            "RCCL_DIAG_SIZES", "scalar=1,medium=1048576,model=1286589072"
        ),
        help="Comma-separated NAME=NUMEL entries.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("RCCL_DIAG_DTYPE", "float32"),
        choices=sorted(DTYPES),
    )
    parser.add_argument("--output-json", default=os.environ.get("RCCL_DIAG_JSON", ""))
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    rank_info = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }
    if rank == 0:
        print("RCCL_DIAG_ENV " + json.dumps(env_snapshot(), sort_keys=True), flush=True)
    print("RCCL_DIAG_RANK " + json.dumps(rank_info, sort_keys=True), flush=True)

    dtype = DTYPES[args.dtype]
    results = []
    for name, numel in parse_sizes(args.sizes):
        result = one_allreduce(name, numel, dtype, device)
        results.append(result)
        print(
            "RCCL_DIAG_RESULT "
            + json.dumps({"rank": rank, "hostname": socket.gethostname(), **result}, sort_keys=True),
            flush=True,
        )
        if result["status"] != "pass":
            break

    dist.barrier()
    dist.destroy_process_group()
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if rank == 0 and args.output_json:
        payload = {
            "started_utc": started,
            "finished_utc": finished,
            "world_size": world_size,
            "dtype": args.dtype,
            "sizes": [{"name": name, "numel": numel} for name, numel in parse_sizes(args.sizes)],
            "rank0_results": results,
            "env": env_snapshot(),
        }
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    return 0 if all(result["status"] == "pass" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
