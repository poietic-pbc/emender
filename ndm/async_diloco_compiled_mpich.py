"""Compiled Cray MPICH dense helper bridge for train.py async DiLoCo.

This module owns only the Python-to-helper IPC contract.  Dense movement is
delegated to ``scripts/frontier/compiled_mpich_dense_helper.cpp`` so the
multinode data plane does not import ``mpi4py``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import torch

from ndm.async_diloco import (
    RESILIENT_QUORUM_DILOCO_MODE,
    AsyncDiLoCoUpdate,
    quorum_merge,
    stable_json_dumps,
    state_num_bytes,
)
from ndm.async_diloco_mpi import (
    DenseBucket,
    DenseTransportMetrics,
    DenseTransportQuorumConfig,
    DenseTransportQuorumResult,
    DenseUpdateEnvelope,
    collect_dense_quorum_from_envelopes,
    pack_dense_update,
)


COMPILED_MPICH_TRANSPORT = "compiled-cray-mpich-helper-collective-reduce"
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
    quorum_mode: str = RESILIENT_QUORUM_DILOCO_MODE,
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
    request_path = write_compiled_mpich_update_request(
        local_update,
        ipc_dir=ipc_dir,
        run_id=run_id,
        rank=rank,
        world_size=requested_ranks,
        generation=generation,
        base_generation=local_update.base_generation,
        quorum=quorum,
        timeout_s=helper.timeout_s,
        bucket_bytes_target=int(helper.bucket_bytes),
        base_checkpoint=base_checkpoint,
    )
    # The request files are now authoritative.  Release the model-sized local
    # CPU delta before MPICH allocates its bucket work buffers.
    if isinstance(local_update.delta, dict):
        local_update.delta.clear()
    rc = _call_compiled_mpich_shared_library(helper_lib, ipc_dir, request_path)
    if rc != 0:
        raise RuntimeError(f"compiled MPICH helper shared library failed rc={rc}")
    result_path = request_path.with_name(f"result.gen{int(generation):06d}.json")
    if not result_path.is_file():
        raise FileNotFoundError(f"compiled MPICH helper did not write result: {result_path}")
    helper_result = json.loads(result_path.read_text(encoding="utf-8"))
    if helper_result.get("status") == "error":
        raise RuntimeError(str(helper_result.get("error", "compiled MPICH helper error")))
    quorum_config = DenseTransportQuorumConfig(
        run_id=run_id,
        generation=generation,
        base_generation=local_update.base_generation,
        requested_ranks=requested_ranks,
        quorum=quorum,
        timeout_s=helper.timeout_s,
        quorum_mode=quorum_mode,
    )
    if helper_result.get("aggregate_payload"):
        quorum_result = collect_compiled_mpich_aggregate_result(
            base_state,
            helper_result=helper_result,
            ipc_dir=ipc_dir,
            config=quorum_config,
        )
    else:
        envelopes = load_received_payloads(ipc_dir, helper_result.get("received_payloads") or [])
        quorum_result = collect_dense_quorum_from_envelopes(
            base_state,
            envelopes,
            config=quorum_config,
            timed_out_ranks=tuple(int(rank) for rank in helper_result.get("timed_out_ranks") or ()),
        )
    payload = quorum_result.to_payload(include_private_state=True)
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
    # The previous helper result has been consumed before the next generation
    # is submitted. Replace this rank's node-local workspace so request,
    # input, and aggregate files cannot accumulate throughout a long run.
    if rank_dir.exists():
        shutil.rmtree(rank_dir)
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


def write_compiled_mpich_update_request(
    update: AsyncDiLoCoUpdate,
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
    """Pack an update directly to bounded bucket files.

    Unlike :func:`pack_dense_update`, this never retains the full serialized
    payload in an envelope.  Peak serialization storage is one target bucket
    (or one oversized tensor, matching the established no-tensor-split wire
    contract).
    """

    if bucket_bytes_target <= 0:
        raise ValueError("bucket_bytes_target must be positive")
    rank_dir = ipc_dir / f"rank_{int(rank):05d}"
    if rank_dir.exists():
        shutil.rmtree(rank_dir)
    gen_dir = rank_dir / f"gen{int(generation):06d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    dtype_names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    descriptors: list[dict[str, Any]] = []
    bucket_headers: list[dict[str, Any]] = []
    tensor_headers: list[dict[str, Any]] = []
    current = bytearray()
    current_entries: list[dict[str, Any]] = []
    stream_offset = 0
    bucket_offset = 0
    payload_hasher = hashlib.sha256()

    def flush_bucket() -> None:
        nonlocal current, current_entries, bucket_offset
        index = len(descriptors)
        payload = bytes(current)
        checksum = hashlib.sha256(payload).hexdigest()
        rel = Path(f"rank_{int(rank):05d}") / f"gen{int(generation):06d}" / f"bucket{index:05d}.bin"
        _atomic_write_bytes(ipc_dir / rel, payload)
        descriptors.append({
            "index": index,
            "nbytes": len(payload),
            "checksum_sha256": checksum,
            "ipc": {"kind": "file", "path": rel.as_posix(), "offset": 0},
        })
        bucket_headers.append({
            "index": index,
            "offset": bucket_offset,
            "nbytes": len(payload),
            "checksum_sha256": checksum,
            "tensors": list(current_entries),
        })
        payload_hasher.update(payload)
        bucket_offset += len(payload)
        current = bytearray()
        current_entries = []

    for name, tensor in sorted(update.delta.items()):
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            raise ValueError(f"delta entry {name!r} is not a floating tensor")
        contiguous = tensor.detach().cpu().contiguous()
        if contiguous.dtype not in dtype_names:
            raise ValueError(f"unsupported dense tensor dtype: {contiguous.dtype}")
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        if current and len(current) + len(raw) > bucket_bytes_target:
            flush_bucket()
        entry = {
            "name": str(name),
            "shape": list(contiguous.shape),
            "dtype": dtype_names[contiguous.dtype],
            "numel": int(contiguous.numel()),
            "offset": int(stream_offset),
            "nbytes": len(raw),
            "checksum_sha256": hashlib.sha256(raw).hexdigest(),
        }
        current.extend(raw)
        current_entries.append(entry)
        tensor_headers.append(dict(entry))
        stream_offset += len(raw)
        del raw, contiguous
    if current or not descriptors:
        flush_bucket()

    staleness = max(0, int(generation) - int(base_generation))
    header = {
        "schema_version": 1,
        "transport": COMPILED_MPICH_TRANSPORT,
        "run_id": str(run_id),
        "rank": int(rank),
        "worker_id": update.worker_id,
        "generation": int(generation),
        "global_generation": int(generation if update.global_generation is None else update.global_generation),
        "base_generation": int(base_generation),
        "base_checkpoint": base_checkpoint,
        "update_id": update.update_id or f"{update.worker_id}:g{int(generation):06d}:base{int(base_generation):06d}",
        "checkpoint_state_id": update.checkpoint_state_id,
        "submitted_at_s": update.submitted_at_s,
        "tokens": int(update.tokens),
        "local_steps": int(update.local_steps),
        "loss_window": {str(k): float(v) for k, v in update.loss_moving_average.items()},
        "staleness": staleness,
        "failed": bool(update.failed),
        "timed_out": bool(update.timed_out),
        "invalid": bool(update.invalid),
        "payload_bytes": int(stream_offset),
        "bucket_bytes_target": int(bucket_bytes_target),
        "bucket_count": len(descriptors),
        "buckets": bucket_headers,
        "tensors": tensor_headers,
        "payload_checksum_sha256": payload_hasher.hexdigest(),
    }
    header_rel = Path(f"rank_{int(rank):05d}") / f"gen{int(generation):06d}" / "header.json"
    header_text = stable_json_dumps(header)
    _atomic_write_text(ipc_dir / header_rel, header_text + "\n")
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
        "header_bytes": len(header_text.encode("utf-8")),
        "payload_bytes": int(stream_offset),
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


def load_aggregate_payload(
    ipc_dir: str | Path,
    aggregate_payload: Mapping[str, Any],
    helper_result: Mapping[str, Any],
) -> DenseUpdateEnvelope:
    """Load the single helper-materialized aggregate update envelope.

    The C++ helper intentionally writes only reduced aggregate bucket bytes on
    root.  It reuses the deterministic local bucket layout but does not compute
    SHA256 checksums, so the Python side rebuilds checksum metadata before using
    the normal dense unpacker.
    """

    root = Path(ipc_dir)
    header = json.loads((root / str(aggregate_payload["source_header_path"])).read_text(encoding="utf-8"))
    bucket_paths = [str(path) for path in aggregate_payload.get("bucket_paths") or ()]
    source_buckets = list(header.get("buckets") or ())
    if len(bucket_paths) != len(source_buckets):
        raise ValueError("compiled MPICH aggregate bucket count mismatch")

    buckets: list[DenseBucket] = []
    rebuilt_header = {
        **header,
        "transport": COMPILED_MPICH_TRANSPORT,
        "rank": int(aggregate_payload.get("rank", 0)),
        "worker_id": "compiled_mpich_aggregate",
        "tokens": int(helper_result.get("accepted_tokens", 0)),
        "local_steps": int(helper_result.get("accepted_local_steps", 0)),
        "loss_window": {
            str(k): float(v)
            for k, v in dict(helper_result.get("aggregate_loss_window") or {}).items()
        },
        "failed": False,
        "timed_out": False,
        "invalid": False,
        "staleness": 0,
    }
    rebuilt_bucket_headers: list[dict[str, Any]] = []
    rebuilt_tensor_headers: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    for idx, rel in enumerate(bucket_paths):
        payload = (root / rel).read_bytes()
        source_bucket = dict(source_buckets[idx])
        tensor_entries: list[dict[str, Any]] = []
        for entry in source_bucket.get("tensors") or ():
            rebuilt_entry = dict(entry)
            local_start = int(rebuilt_entry["offset"]) - int(source_bucket["offset"])
            local_end = local_start + int(rebuilt_entry["nbytes"])
            raw = payload[local_start:local_end]
            if len(raw) != int(rebuilt_entry["nbytes"]):
                raise ValueError("compiled MPICH aggregate tensor metadata crosses bucket boundary")
            rebuilt_entry["checksum_sha256"] = hashlib.sha256(raw).hexdigest()
            tensor_entries.append(rebuilt_entry)
            rebuilt_tensor_headers.append(dict(rebuilt_entry))
        bucket_checksum = hashlib.sha256(payload).hexdigest()
        rebuilt_bucket = {
            **source_bucket,
            "nbytes": len(payload),
            "checksum_sha256": bucket_checksum,
            "tensors": tensor_entries,
        }
        rebuilt_bucket_headers.append(rebuilt_bucket)
        payload_parts.append(payload)
        buckets.append(DenseBucket(
            index=int(rebuilt_bucket["index"]),
            offset=int(rebuilt_bucket["offset"]),
            payload=payload,
            checksum_sha256=bucket_checksum,
            tensor_entries=tuple(tensor_entries),
        ))
    rebuilt_header["payload_bytes"] = int(sum(len(part) for part in payload_parts))
    rebuilt_header["bucket_count"] = len(buckets)
    rebuilt_header["buckets"] = rebuilt_bucket_headers
    rebuilt_header["tensors"] = rebuilt_tensor_headers
    rebuilt_header["payload_checksum_sha256"] = hashlib.sha256(b"".join(payload_parts)).hexdigest()
    return DenseUpdateEnvelope(header=rebuilt_header, buckets=tuple(buckets))


def collect_compiled_mpich_aggregate_result(
    base_state: Mapping[str, torch.Tensor],
    *,
    helper_result: Mapping[str, Any],
    ipc_dir: str | Path,
    config: DenseTransportQuorumConfig,
) -> DenseTransportQuorumResult:
    """Apply one reduced aggregate update while preserving quorum metrics."""

    threshold = config.quorum_threshold()
    accepted_count = int(helper_result.get("accepted_count", len(helper_result.get("accepted_ranks") or ())))
    stale_count = int(helper_result.get("stale_count", len(helper_result.get("stale_ranks") or ())))
    failed_count = int(helper_result.get("failed_count", len(helper_result.get("failed_ranks") or ())))
    timed_out_count = int(helper_result.get("timed_out_count", len(helper_result.get("timed_out_ranks") or ())))
    invalid_count = int(helper_result.get("invalid_count", len(helper_result.get("invalid_ranks") or ())))
    aggregate_bytes = int(helper_result.get("aggregate_update_bytes", helper_result.get("bytes_received", 0)))
    reduce_metrics = dict(helper_result.get("reduce_metrics") or {})
    reduce_duration_s = float(reduce_metrics.get("reduce_duration_s", 0.0))
    bucket_timings = {
        f"bucket_{int(item.get('index', idx)):05d}": float(item.get("reduce_latency_s", 0.0))
        for idx, item in enumerate(reduce_metrics.get("per_bucket") or ())
    }
    transport_metrics = DenseTransportMetrics(
        quorum_size=accepted_count,
        timed_out_ranks=tuple(int(rank) for rank in helper_result.get("timed_out_ranks") or ()),
        stale_ranks=tuple(int(rank) for rank in helper_result.get("stale_ranks") or ()),
        failed_ranks=tuple(int(rank) for rank in helper_result.get("failed_ranks") or ()),
        bytes_sent=int(helper_result.get("bytes_sent", 0)),
        bytes_received=aggregate_bytes,
        bucket_timings_s=bucket_timings,
        merge_latency_s=0.0,
    )

    aggregate_payload = helper_result.get("aggregate_payload")
    if accepted_count < threshold or not aggregate_payload:
        merge_result = quorum_merge(
            base_state,
            (),
            run_id=config.run_id,
            generation=config.generation,
            requested_workers=config.requested_ranks,
            quorum_threshold=threshold,
            generation_duration_s=reduce_duration_s,
            mode=config.quorum_mode,
            checkpoint_state_id=f"{config.run_id}:gen{int(config.generation):06d}",
            missing_worker_ids=tuple(
                f"rank-{rank}" for rank in helper_result.get("timed_out_ranks") or ()
            ),
        )
        metrics = replace(
            merge_result.metrics,
            participating_workers=accepted_count + stale_count + failed_count + timed_out_count + invalid_count,
            stale_updates=stale_count,
            timed_out_updates=timed_out_count,
            failed_updates=failed_count,
            invalid_updates=invalid_count,
            update_bytes={
                **dict(merge_result.metrics.update_bytes),
                "mpi_reduce_payload_sent": int(helper_result.get("bytes_sent", 0)),
                "mpi_reduce_aggregate": aggregate_bytes,
            },
        )
        return DenseTransportQuorumResult(
            state=merge_result.state,
            accepted_updates=(),
            stale_updates=(),
            metrics=metrics,
            transport_metrics=transport_metrics,
        )

    merge_start = time.monotonic()
    # Applying the aggregate through ``load_aggregate_payload`` used to retain
    # every bucket as Python ``bytes``, join a second full-payload copy, unpack
    # a complete delta, and finally allocate a complete replacement state.
    # For E97 that is several simultaneous ~5.5 GiB objects *per rank*.  The
    # helper's node leader owns the aggregate files, so consume those files one
    # at a time and update the existing CPU base state in place instead.
    aggregate_bytes_applied = apply_aggregate_payload_in_place(
        base_state,
        ipc_dir=ipc_dir,
        aggregate_payload=aggregate_payload,
        scale=float(config.eta_outer),
    )
    aggregate_update = AsyncDiLoCoUpdate(
        worker_id="compiled_mpich_aggregate",
        base_generation=int(config.base_generation),
        delta={},
        tokens=int(helper_result.get("accepted_tokens", 0)),
        local_steps=int(helper_result.get("accepted_local_steps", 0)),
        loss_moving_average={
            str(k): float(v)
            for k, v in dict(helper_result.get("aggregate_loss_window") or {}).items()
        },
    )
    # Use the normal quorum accounting on an empty state.  The real state was
    # already advanced bucket-by-bucket above, avoiding another model-sized
    # allocation while preserving the established metrics schema.
    merge_result = quorum_merge(
        {},
        (aggregate_update,),
        run_id=config.run_id,
        generation=config.generation,
        requested_workers=config.requested_ranks,
        quorum_threshold=1,
        eta_outer=1.0,
        weight_by="tokens",
        generation_duration_s=reduce_duration_s,
        mode=config.quorum_mode,
        checkpoint_state_id=f"{config.run_id}:gen{int(config.generation):06d}",
    )
    merge_latency_s = max(0.0, time.monotonic() - merge_start)
    total_tokens = int(helper_result.get("accepted_tokens", aggregate_update.tokens))
    tokens_per_sec = total_tokens / reduce_duration_s if reduce_duration_s > 0.0 else 0.0
    metrics = replace(
        merge_result.metrics,
        requested_workers=config.requested_ranks,
        participating_workers=accepted_count + stale_count + failed_count + timed_out_count + invalid_count,
        quorum_threshold=threshold,
        quorum_size=accepted_count,
        accepted_updates=accepted_count,
        stale_updates=stale_count,
        timed_out_updates=timed_out_count,
        failed_updates=failed_count,
        invalid_updates=invalid_count,
        generation_duration_s=reduce_duration_s,
        merge_duration_s=merge_latency_s,
        tokens_per_sec=tokens_per_sec,
        tokens_per_generation=total_tokens,
        update_bytes={
            **dict(merge_result.metrics.update_bytes),
            "accepted_dense_delta": aggregate_bytes_applied,
            "mpi_reduce_payload_sent": int(helper_result.get("bytes_sent", 0)),
            "mpi_reduce_aggregate": aggregate_bytes,
        },
        loss_moving_average=dict(aggregate_update.loss_moving_average),
        latest_advanced=False,
        quorum_status="advanced",
    )
    transport_metrics = replace(transport_metrics, merge_latency_s=merge_latency_s)
    return DenseTransportQuorumResult(
        state=base_state,
        # The aggregate metadata object deliberately has no model-sized delta.
        # Consumers use this tuple only for accounting on this transport path.
        accepted_updates=(aggregate_update,),
        stale_updates=(),
        metrics=metrics,
        transport_metrics=transport_metrics,
    )


def apply_aggregate_payload_in_place(
    base_state: Mapping[str, torch.Tensor],
    *,
    ipc_dir: str | Path,
    aggregate_payload: Mapping[str, Any],
    scale: float = 1.0,
) -> int:
    """Stream a node-leader aggregate into ``base_state`` without full copies.

    Tensor packing guarantees that a tensor is wholly contained in one bucket.
    At most one bucket byte buffer and one zero-copy tensor view are therefore
    live while the existing CPU state is updated in place.
    """

    root = Path(ipc_dir)
    header = json.loads(
        (root / str(aggregate_payload["source_header_path"])).read_text(encoding="utf-8")
    )
    bucket_paths = [str(path) for path in aggregate_payload.get("bucket_paths") or ()]
    source_buckets = list(header.get("buckets") or ())
    if len(bucket_paths) != len(source_buckets):
        raise ValueError("compiled MPICH aggregate bucket count mismatch")

    dtype_by_name = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    seen: set[str] = set()
    total_bytes = 0
    with torch.no_grad():
        for rel, source_bucket_raw in zip(bucket_paths, source_buckets):
            source_bucket = dict(source_bucket_raw)
            payload = bytearray((root / rel).read_bytes())
            expected_nbytes = int(source_bucket["nbytes"])
            if len(payload) != expected_nbytes:
                raise ValueError("compiled MPICH aggregate bucket length mismatch")
            total_bytes += len(payload)
            bucket_offset = int(source_bucket["offset"])
            for entry_raw in source_bucket.get("tensors") or ():
                entry = dict(entry_raw)
                name = str(entry["name"])
                if name in seen:
                    raise ValueError(f"compiled MPICH aggregate repeats tensor {name!r}")
                if name not in base_state:
                    raise ValueError(f"compiled MPICH aggregate has unknown tensor {name!r}")
                dtype_name = str(entry["dtype"])
                if dtype_name not in dtype_by_name:
                    raise ValueError(f"unsupported aggregate tensor dtype: {dtype_name}")
                start = int(entry["offset"]) - bucket_offset
                end = start + int(entry["nbytes"])
                if start < 0 or end > len(payload):
                    raise ValueError("compiled MPICH aggregate tensor metadata crosses bucket boundary")
                view = torch.frombuffer(
                    memoryview(payload)[start:end],
                    dtype=dtype_by_name[dtype_name],
                    count=int(entry["numel"]),
                ).reshape(tuple(int(dim) for dim in entry["shape"]))
                target = base_state[name]
                if tuple(target.shape) != tuple(view.shape):
                    raise ValueError(f"compiled MPICH aggregate tensor {name!r} shape mismatch")
                target.add_(view.to(dtype=target.dtype), alpha=float(scale))
                seen.add(name)
            del payload
    missing = sorted(set(base_state) - seen)
    if missing:
        raise ValueError(f"compiled MPICH aggregate is missing base tensors: {missing}")
    return total_bytes


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.monotonic_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.monotonic_ns()}.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


__all__ = [
    "COMPILED_MPICH_TRANSPORT",
    "CompiledMpichHelperConfig",
    "apply_aggregate_payload_in_place",
    "collect_compiled_mpich_aggregate_result",
    "load_aggregate_payload",
    "load_received_payloads",
    "resolve_compiled_mpich_helper_library",
    "run_compiled_mpich_dense_quorum",
    "write_compiled_mpich_request",
    "write_compiled_mpich_update_request",
]
