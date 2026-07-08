"""Compiled Cray MPICH dense helper bridge for train.py async DiLoCo.

This module owns only the Python-to-helper IPC contract.  Dense movement is
delegated to ``scripts/frontier/compiled_mpich_dense_helper.cpp`` so the
multinode data plane does not import ``mpi4py``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from ndm.async_diloco import AsyncDiLoCoUpdate, stable_json_dumps
from ndm.async_diloco_mpi import (
    DenseBucket,
    DenseTransportQuorumConfig,
    DenseUpdateEnvelope,
    collect_dense_quorum_from_envelopes,
    pack_dense_update,
)


COMPILED_MPICH_TRANSPORT = "compiled-cray-mpich-helper-p2p"
COMPILED_MPICH_REQUEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompiledMpichHelperConfig:
    helper_bin: str | Path
    ipc_dir: str | Path
    helper_lib: str | Path | None = None
    bucket_bytes: int = 64 * 1024 * 1024
    timeout_s: float = 900.0
    root_rank: int = 0


def run_compiled_mpich_dense_quorum(
    *,
    base_state: Mapping[str, torch.Tensor],
    local_update: AsyncDiLoCoUpdate,
    run_id: str,
    generation: int,
    requested_ranks: int,
    quorum: int,
    rank: int,
    helper: CompiledMpichHelperConfig,
    base_checkpoint: str | None = None,
) -> dict[str, Any] | None:
    """Pack one dense update, invoke the compiled helper, and merge on root."""

    if requested_ranks <= 0:
        raise ValueError("requested_ranks must be positive")
    if rank < 0 or rank >= requested_ranks:
        raise ValueError("rank must be in [0, requested_ranks)")
    helper_bin = Path(helper.helper_bin)
    if not helper_bin.is_file():
        raise FileNotFoundError(f"compiled MPICH helper is not readable: {helper_bin}")
    helper_lib = resolve_compiled_mpich_helper_library(helper)
    if not helper_lib.is_file():
        raise FileNotFoundError(
            "compiled MPICH helper shared library is not readable: "
            f"{helper_lib}; rebuild with scripts/frontier/build_compiled_mpich_dense_helper.sh"
        )
    ipc_dir = Path(helper.ipc_dir)
    envelope = pack_dense_update(
        local_update,
        run_id=run_id,
        rank=rank,
        generation=generation,
        base_checkpoint=base_checkpoint,
        bucket_bytes=int(helper.bucket_bytes),
    )
    envelope = DenseUpdateEnvelope(
        header={**envelope.header, "transport": COMPILED_MPICH_TRANSPORT},
        buckets=envelope.buckets,
    )
    request_path = write_compiled_mpich_request(
        envelope,
        ipc_dir=ipc_dir,
        run_id=run_id,
        rank=rank,
        world_size=requested_ranks,
        generation=generation,
        base_generation=local_update.base_generation,
        quorum=quorum,
        timeout_s=helper.timeout_s,
        bucket_bytes_target=helper.bucket_bytes,
        base_checkpoint=base_checkpoint,
    )
    rc = _call_compiled_mpich_shared_library(helper_lib, ipc_dir, request_path)
    if rc != 0:
        raise RuntimeError(f"compiled MPICH helper shared library failed rc={rc}")
    result_path = request_path.with_name(f"result.gen{int(generation):06d}.json")
    if not result_path.is_file():
        raise FileNotFoundError(f"compiled MPICH helper did not write result: {result_path}")
    helper_result = json.loads(result_path.read_text(encoding="utf-8"))
    if helper_result.get("status") == "error":
        raise RuntimeError(str(helper_result.get("error", "compiled MPICH helper error")))
    if int(rank) != int(helper.root_rank):
        return None

    envelopes = load_received_payloads(ipc_dir, helper_result.get("received_payloads") or [])
    quorum_result = collect_dense_quorum_from_envelopes(
        base_state,
        envelopes,
        config=DenseTransportQuorumConfig(
            run_id=run_id,
            generation=generation,
            base_generation=local_update.base_generation,
            requested_ranks=requested_ranks,
            quorum=quorum,
            timeout_s=helper.timeout_s,
        ),
        timed_out_ranks=tuple(int(rank) for rank in helper_result.get("timed_out_ranks") or ()),
    )
    payload = quorum_result.to_payload()
    payload["transport"]["name"] = COMPILED_MPICH_TRANSPORT
    payload["transport"]["helper_result"] = helper_result
    payload["latest_generation"] = generation if quorum_result.metrics.latest_advanced else -1
    return payload


def resolve_compiled_mpich_helper_library(helper: CompiledMpichHelperConfig) -> Path:
    """Resolve the shared-library bridge path for a helper binary config."""

    if helper.helper_lib is not None:
        return Path(helper.helper_lib)
    helper_bin = Path(helper.helper_bin)
    if helper_bin.suffix == ".so":
        return helper_bin
    return helper_bin.with_name(f"{helper_bin.name}.so")


def _call_compiled_mpich_shared_library(helper_lib: Path, ipc_dir: Path, request_path: Path) -> int:
    """Call the in-process MPICH bridge inside the current Slurm rank world."""

    library = ctypes.CDLL(str(helper_lib))
    run_once = library.compiled_mpich_dense_helper_run_once
    run_once.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    run_once.restype = ctypes.c_int
    return int(run_once(str(ipc_dir).encode("utf-8"), str(request_path).encode("utf-8")))


def write_compiled_mpich_request(
    envelope: DenseUpdateEnvelope,
    *,
    ipc_dir: Path,
    run_id: str,
    rank: int,
    world_size: int,
    generation: int,
    base_generation: int,
    quorum: int,
    timeout_s: float,
    bucket_bytes_target: int,
    base_checkpoint: str | None,
) -> Path:
    """Write header, buckets, and helper request using atomic JSON renames."""

    rank_dir = ipc_dir / f"rank_{int(rank):05d}"
    gen_dir = rank_dir / f"gen{int(generation):06d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    header_rel = Path(f"rank_{int(rank):05d}") / f"gen{int(generation):06d}" / "header.json"
    _atomic_write_text(ipc_dir / header_rel, stable_json_dumps(envelope.header) + "\n")

    descriptors: list[dict[str, Any]] = []
    for bucket in envelope.buckets:
        rel = Path(f"rank_{int(rank):05d}") / f"gen{int(generation):06d}" / f"bucket{bucket.index:05d}.bin"
        bucket_path = ipc_dir / rel
        tmp = bucket_path.with_name(f".{bucket_path.name}.tmp")
        tmp.write_bytes(bucket.payload)
        tmp.replace(bucket_path)
        descriptors.append({
            "index": int(bucket.index),
            "nbytes": len(bucket.payload),
            "checksum_sha256": bucket.checksum_sha256,
            "ipc": {"kind": "file", "path": rel.as_posix(), "offset": 0},
        })

    request = {
        "schema_version": COMPILED_MPICH_REQUEST_SCHEMA_VERSION,
        "command": "dense_quorum_generation",
        "transport": COMPILED_MPICH_TRANSPORT,
        "run_id": str(run_id),
        "rank": int(rank),
        "world_size": int(world_size),
        "generation": int(generation),
        "base_generation": int(base_generation),
        "base_checkpoint": base_checkpoint,
        "quorum": int(quorum),
        "timeout_s": float(timeout_s),
        "bucket_bytes_target": int(bucket_bytes_target),
        "header_bytes": len(stable_json_dumps(envelope.header).encode("utf-8")),
        "payload_bytes": envelope.payload_bytes,
        "header_path": header_rel.as_posix(),
        "bucket_descriptors": descriptors,
    }
    request_path = rank_dir / f"request.gen{int(generation):06d}.json"
    _atomic_write_text(request_path, stable_json_dumps(request) + "\n")
    return request_path


def load_received_payloads(
    ipc_dir: str | Path,
    received_payloads: Sequence[Mapping[str, Any]],
) -> tuple[DenseUpdateEnvelope, ...]:
    """Load root-local helper output back into dense envelopes."""

    root = Path(ipc_dir)
    envelopes: list[DenseUpdateEnvelope] = []
    for item in received_payloads:
        header = json.loads((root / str(item["header_path"])).read_text(encoding="utf-8"))
        buckets: list[DenseBucket] = []
        for idx, rel in enumerate(item.get("bucket_paths") or ()):
            payload = (root / str(rel)).read_bytes()
            meta = (header.get("buckets") or [])[idx]
            buckets.append(DenseBucket(
                index=int(meta["index"]),
                offset=int(meta["offset"]),
                payload=payload,
                checksum_sha256=str(meta["checksum_sha256"]),
                tensor_entries=tuple(dict(entry) for entry in meta.get("tensors") or ()),
            ))
        envelopes.append(DenseUpdateEnvelope(header=header, buckets=tuple(buckets)))
    return tuple(envelopes)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.monotonic_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


__all__ = [
    "COMPILED_MPICH_TRANSPORT",
    "CompiledMpichHelperConfig",
    "load_received_payloads",
    "resolve_compiled_mpich_helper_library",
    "run_compiled_mpich_dense_quorum",
    "write_compiled_mpich_request",
]
