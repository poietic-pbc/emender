"""Dense async DiLoCo update transport helpers for Frontier MPI runs.

The production data plane for train.py-native async DiLoCo moves tensor deltas,
not endpoint checkpoints or live shared-storage files.  This module keeps the
wire format and quorum accounting importable without MPI, while the optional
``mpi4py`` runtime path uses nonblocking Cray MPICH point-to-point calls when it
is available on Frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import site
import time
from typing import Any, Mapping, Sequence

import torch

from ndm.async_diloco import (
    AsyncDiLoCoGenerationMetrics,
    AsyncDiLoCoUpdate,
    build_metrics_summary,
    quorum_merge,
    stable_json_dumps,
    state_num_bytes,
)


MPI_DENSE_UPDATE_SCHEMA_VERSION = 1
MPI_DENSE_HEADER_TAG = 62010
MPI_DENSE_HEADER_BYTES_TAG = 62011
MPI_DENSE_BUCKET_TAG_BASE = 62100
MPI_DENSE_RESULT_TAG = 62200


@dataclass(frozen=True)
class DenseBucket:
    index: int
    offset: int
    payload: bytes
    checksum_sha256: str
    tensor_entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DenseUpdateEnvelope:
    header: dict[str, Any]
    buckets: tuple[DenseBucket, ...]

    @property
    def payload_bytes(self) -> int:
        return int(sum(len(bucket.payload) for bucket in self.buckets))


@dataclass(frozen=True)
class DenseTransportQuorumConfig:
    run_id: str
    generation: int
    base_generation: int
    requested_ranks: int
    quorum: int | None = None
    quorum_fraction: float | None = None
    timeout_s: float = 900.0
    stale_policy: str = "reject"
    weight_by: str = "tokens"
    eta_outer: float = 1.0

    def quorum_threshold(self) -> int:
        if self.requested_ranks <= 0:
            raise ValueError("requested_ranks must be positive")
        if self.quorum is not None:
            threshold = int(self.quorum)
        elif self.quorum_fraction is not None:
            threshold = int(math.ceil(float(self.quorum_fraction) * self.requested_ranks))
        else:
            threshold = self.requested_ranks
        if threshold <= 0 or threshold > self.requested_ranks:
            raise ValueError("quorum threshold must be in [1, requested_ranks]")
        return threshold


@dataclass(frozen=True)
class DenseTransportMetrics:
    quorum_size: int
    timed_out_ranks: tuple[int, ...]
    stale_ranks: tuple[int, ...]
    failed_ranks: tuple[int, ...]
    bytes_sent: int
    bytes_received: int
    accepted_ranks: tuple[int, ...] = ()
    missing_ranks: tuple[int, ...] = ()
    late_ranks: tuple[int, ...] = ()
    invalid_ranks: tuple[int, ...] = ()
    rejected_ranks: tuple[int, ...] = ()
    bucket_timings_s: Mapping[str, float] = field(default_factory=dict)
    merge_latency_s: float = 0.0
    rebase_latency_s: float = 0.0
    checkpoint_latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quorum_size": int(self.quorum_size),
            "timed_out_ranks": list(self.timed_out_ranks),
            "stale_ranks": list(self.stale_ranks),
            "late_ranks": list(self.late_ranks),
            "failed_ranks": list(self.failed_ranks),
            "invalid_ranks": list(self.invalid_ranks),
            "missing_ranks": list(self.missing_ranks),
            "accepted_ranks": list(self.accepted_ranks),
            "rejected_ranks": list(self.rejected_ranks),
            "bytes_sent": int(self.bytes_sent),
            "bytes_received": int(self.bytes_received),
            "bucket_timings_s": dict(self.bucket_timings_s),
            "merge_latency_s": float(self.merge_latency_s),
            "rebase_latency_s": float(self.rebase_latency_s),
            "checkpoint_latency_s": float(self.checkpoint_latency_s),
        }


@dataclass(frozen=True)
class DenseTransportQuorumResult:
    state: Mapping[str, torch.Tensor]
    accepted_updates: tuple[AsyncDiLoCoUpdate, ...]
    stale_updates: tuple[AsyncDiLoCoUpdate, ...]
    metrics: AsyncDiLoCoGenerationMetrics
    transport_metrics: DenseTransportMetrics

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "transport": {
                "name": "cray-mpich-gpu-aware-p2p",
                "dense_data_plane": True,
                "filesystem_live_quorum": False,
                "tcp_dense_data_plane": False,
                "metrics": self.transport_metrics.to_dict(),
            },
            "metrics_summary": build_metrics_summary(
                run_id=self.metrics.run_id,
                requested_workers=self.metrics.requested_workers,
                participating_workers=self.metrics.participating_workers,
                generations=(self.metrics,),
            ).to_dict(),
            "global_generations": [{
                "generation": self.metrics.generation,
                "metrics": self.metrics.to_dict(),
            }],
        }


def pack_dense_update(
    update: AsyncDiLoCoUpdate,
    *,
    run_id: str,
    rank: int,
    generation: int,
    base_checkpoint: str | None = None,
    bucket_bytes: int = 64 * 1024 * 1024,
    staleness: int = 0,
) -> DenseUpdateEnvelope:
    """Pack one delta update into a checksummed bucketed wire envelope."""

    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    tensor_entries: list[dict[str, Any]] = []
    buckets: list[DenseBucket] = []
    current = bytearray()
    current_entries: list[dict[str, Any]] = []
    stream_offset = 0
    bucket_offset = 0

    for name, tensor in sorted(update.delta.items()):
        if not torch.is_tensor(tensor):
            raise ValueError(f"delta entry {name!r} is not a tensor")
        raw = _tensor_to_bytes(tensor.detach().cpu().contiguous())
        if current and len(current) + len(raw) > bucket_bytes:
            buckets.append(_finalize_bucket(len(buckets), bucket_offset, current, current_entries))
            bucket_offset += len(current)
            current = bytearray()
            current_entries = []
        entry = {
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": _dtype_name(tensor.dtype),
            "numel": int(tensor.numel()),
            "offset": int(stream_offset),
            "nbytes": int(len(raw)),
            "checksum_sha256": hashlib.sha256(raw).hexdigest(),
        }
        current.extend(raw)
        current_entries.append(entry)
        tensor_entries.append(dict(entry))
        stream_offset += len(raw)

    if current or not buckets:
        buckets.append(_finalize_bucket(len(buckets), bucket_offset, current, current_entries))

    header = {
        "schema_version": MPI_DENSE_UPDATE_SCHEMA_VERSION,
        "transport": "cray-mpich-gpu-aware-p2p",
        "run_id": str(run_id),
        "rank": int(rank),
        "worker_id": update.worker_id,
        "generation": int(generation),
        "global_generation": int(
            generation if update.global_generation is None else update.global_generation
        ),
        "base_generation": int(update.base_generation),
        "base_checkpoint": base_checkpoint,
        "update_id": (
            update.update_id
            or f"{update.worker_id}:g{int(generation):06d}:base{int(update.base_generation):06d}"
        ),
        "checkpoint_state_id": update.checkpoint_state_id,
        "submitted_at_s": update.submitted_at_s,
        "tokens": int(update.tokens),
        "local_steps": int(update.local_steps),
        "loss_window": {str(k): float(v) for k, v in update.loss_moving_average.items()},
        "staleness": int(staleness),
        "failed": bool(update.failed),
        "timed_out": bool(update.timed_out),
        "invalid": bool(update.invalid),
        "payload_bytes": int(stream_offset),
        "bucket_bytes_target": int(bucket_bytes),
        "bucket_count": len(buckets),
        "buckets": [
            {
                "index": bucket.index,
                "offset": bucket.offset,
                "nbytes": len(bucket.payload),
                "checksum_sha256": bucket.checksum_sha256,
                "tensors": list(bucket.tensor_entries),
            }
            for bucket in buckets
        ],
        "tensors": tensor_entries,
        "payload_checksum_sha256": hashlib.sha256(
            b"".join(bucket.payload for bucket in buckets)
        ).hexdigest(),
    }
    return DenseUpdateEnvelope(header=header, buckets=tuple(buckets))


def unpack_dense_update(envelope: DenseUpdateEnvelope) -> AsyncDiLoCoUpdate:
    """Validate checksums and reconstruct an ``AsyncDiLoCoUpdate``."""

    header = envelope.header
    if int(header.get("schema_version", -1)) != MPI_DENSE_UPDATE_SCHEMA_VERSION:
        raise ValueError("unsupported dense update schema_version")
    payload = b"".join(bucket.payload for bucket in envelope.buckets)
    expected_payload = str(header.get("payload_checksum_sha256", ""))
    actual_payload = hashlib.sha256(payload).hexdigest()
    if expected_payload != actual_payload:
        raise ValueError("dense update payload checksum mismatch")
    expected_buckets = list(header.get("buckets") or [])
    if len(expected_buckets) != len(envelope.buckets):
        raise ValueError("dense update bucket count mismatch")
    for expected, bucket in zip(expected_buckets, envelope.buckets):
        if int(expected["index"]) != bucket.index:
            raise ValueError("dense update bucket index mismatch")
        if str(expected["checksum_sha256"]) != hashlib.sha256(bucket.payload).hexdigest():
            raise ValueError(f"dense update bucket {bucket.index} checksum mismatch")

    delta: dict[str, torch.Tensor] = {}
    for entry in header.get("tensors") or []:
        start = int(entry["offset"])
        end = start + int(entry["nbytes"])
        raw = payload[start:end]
        if hashlib.sha256(raw).hexdigest() != str(entry["checksum_sha256"]):
            raise ValueError(f"dense update tensor {entry['name']!r} checksum mismatch")
        delta[str(entry["name"])] = _tensor_from_bytes(
            raw,
            dtype=_dtype_from_name(str(entry["dtype"])),
            shape=tuple(int(dim) for dim in entry["shape"]),
        )
    return AsyncDiLoCoUpdate(
        worker_id=str(header["worker_id"]),
        base_generation=int(header["base_generation"]),
        delta=delta,
        tokens=int(header["tokens"]),
        local_steps=int(header["local_steps"]),
        loss_moving_average={
            str(k): float(v) for k, v in dict(header.get("loss_window") or {}).items()
        },
        failed=bool(header.get("failed", False)),
        timed_out=bool(header.get("timed_out", False)),
        invalid=bool(header.get("invalid", False)),
        update_id=(None if header.get("update_id") is None else str(header.get("update_id"))),
        global_generation=int(header.get("global_generation", header.get("generation", 0))),
        checkpoint_state_id=(
            None if header.get("checkpoint_state_id") is None
            else str(header.get("checkpoint_state_id"))
        ),
        submitted_at_s=(
            None if header.get("submitted_at_s") is None
            else float(header.get("submitted_at_s"))
        ),
    )


def collect_dense_quorum_from_envelopes(
    base_state: Mapping[str, torch.Tensor],
    envelopes: Sequence[DenseUpdateEnvelope],
    *,
    config: DenseTransportQuorumConfig,
    timed_out_ranks: Sequence[int] = (),
    receive_started_s: float | None = None,
) -> DenseTransportQuorumResult:
    """Merge any fresh envelopes that reach quorum without requiring unanimity."""

    start_s = time.monotonic() if receive_started_s is None else float(receive_started_s)
    threshold = config.quorum_threshold()
    accepted: list[AsyncDiLoCoUpdate] = []
    stale: list[AsyncDiLoCoUpdate] = []
    accepted_ranks: list[int] = []
    stale_ranks: list[int] = []
    late_ranks: list[int] = []
    failed_ranks: list[int] = []
    invalid_ranks: list[int] = []
    bytes_received = 0
    bucket_timings: dict[str, float] = {}

    for envelope in envelopes:
        bucket_start = time.monotonic()
        try:
            update = unpack_dense_update(envelope)
            rank = int(envelope.header.get("rank", -1))
            bytes_received += int(envelope.header.get("payload_bytes", envelope.payload_bytes))
            bucket_timings[f"rank_{rank:05d}"] = max(0.0, time.monotonic() - bucket_start)
            header_generation = int(envelope.header.get("generation", -1))
            header_base_generation = int(envelope.header.get("base_generation", -1))
            is_late = (
                header_generation > int(config.generation)
                or update.base_generation > int(config.base_generation)
            )
            is_stale = (
                header_generation != int(config.generation)
                or update.base_generation != int(config.base_generation)
                or int(envelope.header.get("staleness", 0)) > 0
            )
            if is_late:
                late_ranks.append(rank)
                stale.append(update)
                continue
            if is_stale:
                stale_ranks.append(rank)
                stale.append(update)
                continue
            if update.invalid:
                invalid_ranks.append(rank)
                continue
            if update.failed or update.timed_out:
                failed_ranks.append(rank)
                continue
            accepted.append(update)
            accepted_ranks.append(rank)
            if len(accepted) >= threshold:
                break
        except Exception:
            failed_ranks.append(int(envelope.header.get("rank", -1)))

    missing = set(int(rank) for rank in timed_out_ranks)
    seen = {int(envelope.header.get("rank", -1)) for envelope in envelopes}
    for rank in range(int(config.requested_ranks)):
        if rank not in seen and len(accepted) < threshold:
            missing.add(rank)

    merge_start = time.monotonic()
    merge_result = quorum_merge(
        base_state,
        tuple(accepted) + tuple(stale),
        run_id=config.run_id,
        generation=config.generation,
        requested_workers=config.requested_ranks,
        quorum_threshold=threshold,
        eta_outer=config.eta_outer,
        weight_by=config.weight_by,
        generation_duration_s=max(0.0, time.monotonic() - start_s),
        checkpoint_state_id=f"{config.run_id}:gen{int(config.generation):06d}",
        missing_worker_ids=tuple(f"rank-{rank}" for rank in sorted(missing)),
    )
    merge_latency = max(0.0, time.monotonic() - merge_start)
    transport_metrics = DenseTransportMetrics(
        quorum_size=len(merge_result.accepted_updates),
        timed_out_ranks=tuple(sorted(missing)),
        stale_ranks=tuple(sorted(rank for rank in stale_ranks if rank >= 0)),
        failed_ranks=tuple(sorted(rank for rank in failed_ranks if rank >= 0)),
        bytes_sent=sum(int(envelope.header.get("payload_bytes", envelope.payload_bytes)) for envelope in envelopes),
        bytes_received=bytes_received,
        accepted_ranks=tuple(sorted(rank for rank in accepted_ranks if rank >= 0)),
        missing_ranks=tuple(sorted(missing)),
        late_ranks=tuple(sorted(rank for rank in late_ranks if rank >= 0)),
        invalid_ranks=tuple(sorted(rank for rank in invalid_ranks if rank >= 0)),
        rejected_ranks=tuple(sorted({
            rank for rank in (*stale_ranks, *late_ranks, *invalid_ranks)
            if rank >= 0
        })),
        bucket_timings_s=bucket_timings,
        merge_latency_s=merge_latency,
    )
    metrics = AsyncDiLoCoGenerationMetrics(
        **{
            **merge_result.metrics.to_dict(),
            "stale_updates": len(stale),
            "timed_out_updates": len(transport_metrics.timed_out_ranks),
            "failed_updates": len(transport_metrics.failed_ranks),
            "late_updates": len(transport_metrics.late_ranks),
            "rejected_updates": len(transport_metrics.rejected_ranks),
            "missing_updates": len(transport_metrics.missing_ranks),
            "merge_duration_s": merge_latency,
            "update_bytes": {
                **dict(merge_result.metrics.update_bytes),
                "mpi_dense_payload_received": int(bytes_received),
                "mpi_dense_payload_sent": int(transport_metrics.bytes_sent),
                "accepted_dense_delta": int(
                    sum(state_num_bytes(update.delta) for update in merge_result.accepted_updates)
                ),
            },
        }
    )
    return DenseTransportQuorumResult(
        state=merge_result.state,
        accepted_updates=tuple(merge_result.accepted_updates),
        stale_updates=tuple(stale),
        metrics=metrics,
        transport_metrics=transport_metrics,
    )


def run_mpi_dense_quorum(
    *,
    base_state: Mapping[str, torch.Tensor],
    local_update: AsyncDiLoCoUpdate,
    run_id: str,
    generation: int,
    requested_ranks: int,
    quorum: int,
    timeout_s: float,
    bucket_bytes: int = 64 * 1024 * 1024,
    base_checkpoint: str | None = None,
    root_rank: int = 0,
    comm: Any | None = None,
) -> dict[str, Any] | None:
    """Run a nonblocking mpi4py dense update quorum.

    Non-root ranks send a compact JSON header followed by bucket payloads and
    then wait for the root decision.  The root posts nonblocking receives and
    advances once quorum arrives or the timeout expires.  ``None`` is returned
    on non-root ranks until a final decision is received.
    """

    if comm is None:
        def _import_mpi4py() -> Any:
            import mpi4py  # type: ignore

            mpi4py.rc.initialize = True
            mpi4py.rc.threads = True
            mpi4py.rc.thread_level = "serialized"
            from mpi4py import MPI  # type: ignore

            return MPI

        try:
            MPI = _import_mpi4py()
        except Exception:
            cray_mpi4py_site = os.environ.get(
                "CRAY_MPI4PY_SITE",
                "/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages",
            )
            if os.path.isdir(os.path.join(cray_mpi4py_site, "mpi4py")):
                site.addsitedir(cray_mpi4py_site)
            try:
                MPI = _import_mpi4py()
            except Exception as exc:  # pragma: no cover - depends on Frontier env
                raise RuntimeError(
                    "mpi4py is required for MPI dense transport; set CRAY_MPI4PY_SITE "
                    "to a compatible Cray mpi4py site-packages directory"
                ) from exc
        comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    world = int(comm.Get_size())
    if requested_ranks > world:
        raise ValueError("requested_ranks cannot exceed MPI world size")

    envelope = pack_dense_update(
        local_update,
        run_id=run_id,
        rank=rank,
        generation=generation,
        base_checkpoint=base_checkpoint,
        bucket_bytes=bucket_bytes,
    )
    header_bytes = stable_json_dumps(envelope.header).encode("utf-8")
    deadline = time.monotonic() + float(timeout_s)
    if rank != root_rank:
        requests = [comm.isend(len(header_bytes), dest=root_rank, tag=MPI_DENSE_HEADER_TAG)]
        requests.append(
            comm.Isend(
                [memoryview(header_bytes), MPI.BYTE],
                dest=root_rank,
                tag=MPI_DENSE_HEADER_BYTES_TAG,
            )
        )
        for bucket in envelope.buckets:
            requests.append(
                comm.Isend(
                    [memoryview(bucket.payload), MPI.BYTE],
                    dest=root_rank,
                    tag=MPI_DENSE_BUCKET_TAG_BASE + bucket.index,
                )
            )
        while requests and time.monotonic() < deadline:
            remaining = []
            for req in requests:
                if not req.Test():
                    remaining.append(req)
            requests = remaining
            time.sleep(0.001)
        if requests:
            raise TimeoutError(f"rank {rank} timed out sending dense MPI update")
        result = comm.recv(source=root_rank, tag=MPI_DENSE_RESULT_TAG)
        return result if isinstance(result, dict) else None

    envelopes = [envelope]
    received_ranks = {root_rank}
    pending_header_lengths = {
        peer: comm.irecv(source=peer, tag=MPI_DENSE_HEADER_TAG)
        for peer in range(world)
        if peer != root_rank and peer < requested_ranks
    }
    pending_headers: dict[int, Any] = {}
    pending_header_buffers: dict[int, bytearray] = {}
    pending_buckets: dict[tuple[int, int], Any] = {}
    pending_bucket_buffers: dict[tuple[int, int], bytearray] = {}
    buckets_by_peer: dict[int, dict[int, DenseBucket]] = {}
    headers: dict[int, dict[str, Any]] = {}
    while time.monotonic() < deadline and len(envelopes) < quorum:
        for peer, req in list(pending_header_lengths.items()):
            ready, data = req.test()
            if not ready:
                continue
            header_len = int(data)
            del pending_header_lengths[peer]
            pending_header_buffers[peer] = bytearray(header_len)
            pending_headers[peer] = comm.Irecv(
                [memoryview(pending_header_buffers[peer]), MPI.BYTE],
                source=peer,
                tag=MPI_DENSE_HEADER_BYTES_TAG,
            )
        for peer, req in list(pending_headers.items()):
            ready = req.Test()
            if not ready:
                continue
            header = json.loads(bytes(pending_header_buffers[peer]).decode("utf-8"))
            headers[peer] = header
            del pending_headers[peer]
            del pending_header_buffers[peer]
            for bucket in header.get("buckets") or []:
                idx = int(bucket["index"])
                key = (peer, idx)
                pending_bucket_buffers[key] = bytearray(int(bucket["nbytes"]))
                pending_buckets[key] = comm.Irecv(
                    [memoryview(pending_bucket_buffers[key]), MPI.BYTE],
                    source=peer,
                    tag=MPI_DENSE_BUCKET_TAG_BASE + idx,
                )
        for key, req in list(pending_buckets.items()):
            ready = req.Test()
            if not ready:
                continue
            peer, idx = key
            header = headers[peer]
            header_bucket = (header.get("buckets") or [])[idx]
            bucket = DenseBucket(
                index=idx,
                offset=int(header_bucket["offset"]),
                payload=bytes(pending_bucket_buffers[key]),
                checksum_sha256=str(header_bucket["checksum_sha256"]),
                tensor_entries=tuple(dict(item) for item in header_bucket.get("tensors") or ()),
            )
            del pending_buckets[key]
            del pending_bucket_buffers[key]
            buckets_by_peer.setdefault(peer, {})[idx] = bucket
            if len(buckets_by_peer[peer]) == int(header["bucket_count"]):
                ordered = tuple(
                    buckets_by_peer[peer][int(bucket_meta["index"])]
                    for bucket_meta in header.get("buckets") or []
                )
                envelopes.append(DenseUpdateEnvelope(header=header, buckets=ordered))
                received_ranks.add(peer)
        time.sleep(0.001)

    timed_out = tuple(rank for rank in range(requested_ranks) if rank not in received_ranks)
    result = collect_dense_quorum_from_envelopes(
        base_state,
        envelopes,
        config=DenseTransportQuorumConfig(
            run_id=run_id,
            generation=generation,
            base_generation=local_update.base_generation,
            requested_ranks=requested_ranks,
            quorum=quorum,
            timeout_s=timeout_s,
        ),
        timed_out_ranks=timed_out,
    )
    payload = result.to_payload()
    payload["latest_generation"] = generation if result.metrics.latest_advanced else -1
    for peer in range(world):
        if peer != root_rank and peer < requested_ranks:
            comm.isend(payload, dest=peer, tag=MPI_DENSE_RESULT_TAG)
    return payload


def _finalize_bucket(
    index: int,
    offset: int,
    payload: bytearray,
    tensor_entries: Sequence[Mapping[str, Any]],
) -> DenseBucket:
    raw = bytes(payload)
    return DenseBucket(
        index=int(index),
        offset=int(offset),
        payload=raw,
        checksum_sha256=hashlib.sha256(raw).hexdigest(),
        tensor_entries=tuple(dict(entry) for entry in tensor_entries),
    )


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype is torch.bfloat16:
        return tensor.view(torch.int16).numpy().tobytes(order="C")
    return tensor.numpy().tobytes(order="C")


def _tensor_from_bytes(raw: bytes, *, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype is torch.bfloat16:
        storage = torch.frombuffer(bytearray(raw), dtype=torch.int16).clone()
        return storage.view(torch.bfloat16).reshape(shape)
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if name not in aliases:
        raise ValueError(f"unsupported dense update dtype: {name}")
    return aliases[name]


__all__ = [
    "DenseBucket",
    "DenseTransportMetrics",
    "DenseTransportQuorumConfig",
    "DenseTransportQuorumResult",
    "DenseUpdateEnvelope",
    "collect_dense_quorum_from_envelopes",
    "pack_dense_update",
    "run_mpi_dense_quorum",
    "unpack_dense_update",
]
