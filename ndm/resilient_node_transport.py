"""Framed point-to-point transport for resilient node-quorum generations.

This is deliberately independent of MPI.  One node manager per physical node
submits locally aggregated, fixed-size float64 buckets and retains them on
node-local storage until the coordinator returns a fenced commit.  The server
streams committed buckets back to clients; model payloads never enter the
metadata directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from ndm.resilient_node_quorum import GenerationFence, MetadataCoordinator


_U32 = struct.Struct("!I")
_F64 = struct.Struct("!d")
MAX_HEADER_BYTES = 64 * 1024
RESILIENT_NODE_TRANSPORT = "resilient-node-quorum-sharded-p2p"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def encode_f64(values: Sequence[float]) -> bytes:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().to(torch.float64).contiguous().numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    return array.astype(">f8", copy=False).tobytes()


def decode_f64(payload: bytes) -> tuple[float, ...]:
    if len(payload) % _F64.size:
        raise ValueError("float64 bucket has a partial element")
    return tuple(np.frombuffer(payload, dtype=">f8").astype(np.float64))


def decode_f64_tensor(payload: bytes) -> torch.Tensor:
    """Vectorized dense decode for the live path (no Python scalar materialization)."""
    if len(payload) % _F64.size:
        raise ValueError("float64 bucket has a partial element")
    return torch.from_numpy(
        np.frombuffer(payload, dtype=">f8").astype(np.float64)).clone()


def _recv_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            raise EOFError("peer closed framed transport")
        data.extend(part)
    return bytes(data)


def _write_all(stream, payload: bytes) -> None:
    """Write a complete frame segment through buffered or raw streams.

    ``socket.makefile(..., buffering=0)`` exposes ``SocketIO.write``, whose
    contract permits a successful partial write.  Dense owner frames are far
    larger than a socket send buffer, so treating one ``write`` call as
    complete truncates the frame and leaves the receiver waiting for bytes
    that will never arrive.  Retain the bounded segment as a memoryview and
    advance it until every byte has been accepted.
    """
    remaining = memoryview(payload)
    while remaining:
        written = stream.write(remaining)
        if written is None or int(written) <= 0:
            raise BrokenPipeError("framed transport write made no progress")
        written = int(written)
        if written > len(remaining):
            raise OSError("framed transport write exceeded remaining payload")
        remaining = remaining[written:]


def send_frame(stream, header: Mapping[str, object], payload: bytes = b"") -> None:
    wire = dict(header)
    wire["payload_bytes"] = len(payload)
    wire["payload_sha256"] = _sha(payload)
    encoded = json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("frame header too large")
    _write_all(stream, _U32.pack(len(encoded)))
    _write_all(stream, encoded)
    _write_all(stream, payload)
    stream.flush()


def recv_frame(stream, *, max_payload_bytes: int) -> tuple[dict[str, object], bytes]:
    header_size = _U32.unpack(_recv_exact(stream, _U32.size))[0]
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ValueError("invalid frame header size")
    header = json.loads(_recv_exact(stream, header_size))
    payload_size = int(header["payload_bytes"])
    if payload_size < 0 or payload_size > max_payload_bytes:
        raise ValueError("frame payload exceeds configured bound")
    payload = _recv_exact(stream, payload_size)
    if _sha(payload) != header["payload_sha256"]:
        raise ValueError("frame payload checksum mismatch")
    return header, payload


class DiskBucketSpool:
    """Crash-surviving sender spool with a hard aggregate byte limit."""

    def __init__(self, root: str | Path, max_bytes: int):
        self.root, self.max_bytes = Path(root), int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    @property
    def bytes_used(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("*.bucket"))

    def retain(self, fence: GenerationFence, bucket: int, payload: bytes) -> Path:
        name = f"g{fence.generation}-a{fence.attempt}-e{fence.coordinator_epoch}-b{bucket}.bucket"
        target = self.root / name
        prior = target.stat().st_size if target.exists() else 0
        if self.bytes_used - prior + len(payload) > self.max_bytes:
            raise BufferError("disk bucket retention limit exceeded")
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target

    def release(self, fence: GenerationFence) -> None:
        for path in self.root.glob(
            f"g{fence.generation}-a{fence.attempt}-e{fence.coordinator_epoch}-b*.bucket"
        ):
            path.unlink()

    def prune(self, *, keep_generations: int) -> None:
        """Bound replay history by generation as well as by bytes."""
        if keep_generations <= 0:
            raise ValueError("keep_generations must be positive")
        generations = sorted({int(path.name.split("-", 1)[0][1:])
                              for path in self.root.glob("g*-a*-e*-b*.bucket")})
        for generation in generations[:-keep_generations]:
            for path in self.root.glob(f"g{generation}-a*-e*-b*.bucket"):
                path.unlink()


@dataclass(frozen=True)
class TransportConfig:
    run_id: str
    quorum: int
    expected_buckets: int
    max_bucket_bytes: int
    generation_deadline_s: float
    heartbeat_timeout_s: float = 30.0
    apply_deadline_s: float = 30.0


class _Generation:
    def __init__(self, fence: GenerationFence, deadline: float):
        self.fence, self.deadline = fence, deadline
        self.updates: dict[str, dict[int, tuple[int, bytes]]] = {}
        self.receipts: dict[tuple[str, int], tuple[int, str]] = {}
        self.accepted: tuple[str, ...] | None = None
        self.accepted_tokens = 0
        self.aggregates: dict[int, bytes] = {}
        self.condition = threading.Condition()
        self.last_heartbeat: dict[str, float] = {}
        self.applied: set[str] = set()
        self.delivered: set[str] = set()


class QuorumTransportServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, config: TransportConfig, metadata_root: str | Path,
                 coordinator_epoch: int = 1):
        self.config = config
        self.coordinator = MetadataCoordinator(metadata_root, config.run_id, coordinator_epoch)
        self._generations: dict[tuple[int, int], _Generation] = {}
        self._lock = threading.Lock()
        super().__init__(address, _RequestHandler)

    def generation(self, generation: int, attempt: int) -> _Generation:
        key = (generation, attempt)
        with self._lock:
            if key not in self._generations:
                fence = self.coordinator.fence(generation, attempt)
                self._generations[key] = _Generation(
                    fence, time.monotonic() + self.config.generation_deadline_s
                )
            return self._generations[key]

    def submit(self, header: Mapping[str, object], payload: bytes) -> _Generation:
        if header.get("run_id") != self.config.run_id:
            raise ValueError("run fence mismatch")
        state = self.generation(int(header["generation"]), int(header["attempt"]))
        if int(header["coordinator_epoch"]) != state.fence.coordinator_epoch:
            raise ValueError("coordinator epoch fence mismatch")
        bucket, node = int(header["bucket"]), str(header["node_id"])
        if bucket not in range(self.config.expected_buckets):
            raise ValueError("bucket outside generation schema")
        weight = int(header["weight"])
        if weight <= 0:
            raise ValueError("weight must be positive")
        values = decode_f64(payload)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("nonfinite bucket rejected")
        with state.condition:
            if time.monotonic() >= state.deadline:
                raise TimeoutError("generation deadline expired")
            if state.accepted is not None and node not in state.accepted:
                raise ValueError("late node rejected after accepted set was frozen")
            receipt = state.receipts.get((node, bucket))
            if receipt is not None:
                if receipt != (weight, _sha(payload)):
                    raise ValueError("conflicting duplicate bucket rejected")
                return state  # idempotent reconnect/retry of the same chunk identity
            state.updates.setdefault(node, {})[bucket] = (weight, payload)
            state.receipts[(node, bucket)] = (weight, _sha(payload))
            state.last_heartbeat[node] = time.monotonic()
            complete = sorted(
                n for n, buckets in state.updates.items()
                if set(buckets) == set(range(self.config.expected_buckets))
            )
            if state.accepted is None and len(complete) >= self.config.quorum:
                state.accepted = tuple(complete[: self.config.quorum])
                for index in range(self.config.expected_buckets):
                    items = [state.updates[n][index] for n in state.accepted]
                    vectors = [decode_f64(item[1]) for item in items]
                    if len({len(v) for v in vectors}) != 1:
                        raise ValueError("bucket vector widths differ")
                    total = sum(item[0] for item in items)
                    state.aggregates[index] = encode_f64([
                        sum(vector[i] * item[0] for vector, item in zip(vectors, items)) / total
                        for i in range(len(vectors[0]))
                    ])
                state.accepted_tokens = sum(state.updates[node][0][0]
                                            for node in state.accepted)
                self._publish_manifest(state)
                state.updates.clear()  # retain only compact receipts after incremental reduction
                state.condition.notify_all()
            return state

    def _publish_manifest(self, state: _Generation) -> None:
        # Metadata contains checksums and membership only, never model payloads.
        root = self.coordinator.root / "network-generations"
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "transport": "resilient-node-quorum-sharded-p2p",
            "run_id": state.fence.run_id, "generation": state.fence.generation,
            "attempt": state.fence.attempt,
            "coordinator_epoch": state.fence.coordinator_epoch,
            "accepted_nodes": list(state.accepted or ()),
            "buckets": {str(k): {"bytes": len(v), "sha256": _sha(v)}
                        for k, v in state.aggregates.items()},
        }
        target = root / f"{state.fence.generation:08d}.json"
        _atomic_bytes(target, (json.dumps(payload, sort_keys=True) + "\n").encode())

    def evict_expired(self, generation: int, attempt: int) -> tuple[str, ...]:
        """Evict incomplete members whose heartbeat deadline elapsed."""
        state = self.generation(generation, attempt)
        now = time.monotonic()
        with state.condition:
            evicted = tuple(sorted(node for node, seen in state.last_heartbeat.items()
                                   if now - seen > self.config.heartbeat_timeout_s
                                   and state.accepted is None))
            for node in evicted:
                state.updates.pop(node, None)
                state.last_heartbeat.pop(node, None)
                for key in [key for key in state.receipts if key[0] == node]:
                    state.receipts.pop(key, None)
            return evicted

    def acknowledge_apply(self, fence: GenerationFence, node_id: str,
                          payload_sha256: str) -> None:
        state = self.generation(fence.generation, fence.attempt)
        if state.fence != fence or state.accepted is None or node_id not in state.accepted:
            raise ValueError("apply acknowledgement fence or membership mismatch")
        expected = _sha(b"".join(state.aggregates[index]
                                 for index in sorted(state.aggregates)))
        if payload_sha256 != expected:
            raise ValueError("applied payload identity mismatch")
        with state.condition:
            state.applied.add(node_id)
            state.condition.notify_all()

    def wait_for_apply(self, fence: GenerationFence) -> tuple[str, ...]:
        state = self.generation(fence.generation, fence.attempt)
        deadline = time.monotonic() + self.config.apply_deadline_s
        with state.condition:
            while state.accepted is not None and not set(state.accepted) <= state.applied:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("aggregate apply deadline expired")
                state.condition.wait(remaining)
            return tuple(sorted(state.applied))

    def acknowledge_delivery(self, state: _Generation, node_id: str) -> None:
        """Release a reduced bulk chunk once every accepted stream received it."""
        with state.condition:
            state.delivered.add(node_id)
            if state.accepted is not None and set(state.accepted) <= state.delivered:
                state.aggregates.clear()


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        persistent = True
        while persistent:
            try:
                header, payload = recv_frame(
                    self.rfile, max_payload_bytes=self.server.config.max_bucket_bytes
                )
                persistent = header.get("op") == "stream_submit"
                if header.get("op") not in {"submit", "stream_submit"}:
                    raise ValueError("unsupported operation")
                state = self.server.submit(header, payload)
                with state.condition:
                    while state.accepted is None:
                        remaining = state.deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("node quorum lost before generation deadline")
                        state.condition.wait(remaining)
                send_frame(self.wfile, {
                    "op": "commit", "accepted_nodes": list(state.accepted),
                    "accepted_tokens": state.accepted_tokens,
                    "bucket_count": len(state.aggregates),
                    "generation": state.fence.generation,
                    "attempt": state.fence.attempt,
                    "coordinator_epoch": state.fence.coordinator_epoch,
                })
                for bucket in sorted(state.aggregates):
                    send_frame(self.wfile, {"op": "aggregate", "bucket": bucket},
                               state.aggregates[bucket])
                self.server.acknowledge_delivery(state, str(header["node_id"]))
            except EOFError:
                return
            except Exception as error:
                send_frame(self.wfile, {"op": "error", "error": str(error)})
                return


class NodeManagerClient:
    def __init__(self, host: str, port: int, node_id: str, spool: DiskBucketSpool,
                 *, timeout_s: float = 30.0, max_bucket_bytes: int = 64 << 20):
        self.host, self.port, self.node_id = host, port, node_id
        self.spool, self.timeout_s = spool, float(timeout_s)
        self.max_bucket_bytes = int(max_bucket_bytes)

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + self.timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                return socket.create_connection((self.host, self.port),
                                                min(1.0, self.timeout_s))
            except OSError as error:
                last_error = error
                time.sleep(min(.1, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"manager connect deadline expired: {last_error}")

    def exchange(self, fence: GenerationFence, buckets: Sequence[bytes], *, weight: int
                 ) -> tuple[dict[str, object], tuple[bytes, ...]]:
        paths = [self.spool.retain(fence, i, payload) for i, payload in enumerate(buckets)]
        replies = []
        # One independent connection per bucket allows bucket sharding/replay.
        for index, path in enumerate(paths):
            sock = self._connect()
            sock.settimeout(self.timeout_s)
            stream = sock.makefile("rwb", buffering=0)
            send_frame(stream, {
                "op": "submit", "run_id": fence.run_id, "node_id": self.node_id,
                "generation": fence.generation, "attempt": fence.attempt,
                "coordinator_epoch": fence.coordinator_epoch, "bucket": index,
                "weight": int(weight),
            }, path.read_bytes())
            replies.append((sock, stream))
        try:
            header, _ = recv_frame(replies[0][1], max_payload_bytes=0)
            if header["op"] == "error":
                raise RuntimeError(str(header["error"]))
            aggregates = []
            for _ in range(int(header["bucket_count"])):
                aggregate_header, payload = recv_frame(
                    replies[0][1], max_payload_bytes=self.max_bucket_bytes
                )
                if aggregate_header["op"] != "aggregate":
                    raise RuntimeError("expected aggregate frame")
                aggregates.append((int(aggregate_header["bucket"]), payload))
            self.spool.release(fence)
            return header, tuple(payload for _, payload in sorted(aggregates))
        finally:
            for sock, stream in replies:
                stream.close()
                sock.close()


class BulkChunkStream(Protocol):
    """Bulk plane; control headers never contain update tensor bytes."""

    def exchange_chunks(self, fence: GenerationFence, chunks: Sequence[bytes], *,
                        weight: int) -> tuple[dict[str, object], tuple[bytes, ...]]: ...


class BoundedNodeManagerBulkStream:
    """Checksummed one-chunk window with reconnect/replay and shard identities.

    Each chunk uses a distinct fenced attempt, so the coordinator incrementally
    reduces and releases participant bytes before the next chunk is admitted.
    The window is deliberately one: peak manager ownership is O(quorum*chunk),
    never O(participants*full_update).
    """

    def __init__(self, client: NodeManagerClient, *, max_chunk_bytes: int):
        self.client, self.max_chunk_bytes = client, int(max_chunk_bytes)
        if self.max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        self.high_water_bytes = 0

    def exchange_chunks(self, fence: GenerationFence, chunks: Sequence[bytes], *,
                        weight: int) -> tuple[dict[str, object], tuple[bytes, ...]]:
        aggregates, last_header = [], None
        sock = self.client._connect(); sock.settimeout(self.client.timeout_s)
        stream = sock.makefile("rwb", buffering=0)
        try:
            for shard, payload in enumerate(chunks):
                if len(payload) > self.max_chunk_bytes:
                    raise BufferError("bulk chunk exceeds bounded in-flight window")
                self.high_water_bytes = max(self.high_water_bytes, len(payload))
                chunk_fence = GenerationFence(fence.run_id, fence.generation, shard,
                                              fence.coordinator_epoch)
                deadline = time.monotonic() + self.client.timeout_s
                while True:
                    try:
                        send_frame(stream, {
                            "op": "stream_submit", "run_id": chunk_fence.run_id,
                            "node_id": self.client.node_id,
                            "generation": chunk_fence.generation,
                            "attempt": chunk_fence.attempt,
                            "coordinator_epoch": chunk_fence.coordinator_epoch,
                            "bucket": 0, "weight": int(weight), "shard": shard,
                        }, payload)
                        last_header, _ = recv_frame(stream, max_payload_bytes=0)
                        if last_header["op"] == "error":
                            raise RuntimeError(str(last_header["error"]))
                        aggregate_header, aggregate = recv_frame(
                            stream, max_payload_bytes=self.max_chunk_bytes)
                        if aggregate_header["op"] != "aggregate":
                            raise RuntimeError("expected bulk aggregate chunk")
                        aggregates.append(aggregate)
                        break
                    except (OSError, EOFError, ConnectionError):
                        stream.close(); sock.close()
                        if time.monotonic() >= deadline:
                            raise TimeoutError("bulk chunk reconnect deadline expired")
                        sock = self.client._connect(); sock.settimeout(self.client.timeout_s)
                        stream = sock.makefile("rwb", buffering=0)
        finally:
            stream.close(); sock.close()
        if last_header is None:
            raise ValueError("bulk stream requires at least one chunk")
        return last_header, tuple(aggregates)


@dataclass(frozen=True)
class DenseBucketLayout:
    """Deterministic tensor layout for streamed node deltas.

    Values are transported as network-order float64 so shard owners can compute
    an exact weighted mean without understanding PyTorch serialization.  The
    returned aggregate is cast back to each base tensor's dtype and applied with
    the same ``base + eta_outer * mean_delta`` rule as async DiLoCo.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]
    bucket_elements: int


def pack_dense_delta(delta: Mapping[str, torch.Tensor], *, bucket_bytes: int
                     ) -> tuple[DenseBucketLayout, tuple[bytes, ...]]:
    if bucket_bytes < _F64.size:
        raise ValueError("bucket_bytes must hold at least one float64 value")
    names = tuple(sorted(delta))
    tensors = [delta[name].detach().cpu() for name in names]
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("resilient dense transport accepts floating tensors only")
    elements = max(1, bucket_bytes // _F64.size)
    buckets = []
    pending = torch.empty(0, dtype=torch.float64)
    for tensor in tensors:
        flat = tensor.to(torch.float64).reshape(-1)
        offset = 0
        while offset < flat.numel():
            take = min(elements - pending.numel(), flat.numel() - offset)
            piece = flat[offset:offset + take]
            pending = torch.cat((pending, piece)) if pending.numel() else piece
            offset += take
            if pending.numel() == elements:
                buckets.append(encode_f64(pending))
                pending = torch.empty(0, dtype=torch.float64)
    if pending.numel():
        buckets.append(encode_f64(pending))
    buckets = tuple(buckets)
    if not buckets:
        buckets = (b"",)
    return DenseBucketLayout(
        names=names,
        shapes=tuple(tuple(tensor.shape) for tensor in tensors),
        counts=tuple(tensor.numel() for tensor in tensors),
        bucket_elements=elements,
    ), buckets


def apply_aggregate_delta(base_state: Mapping[str, torch.Tensor], layout: DenseBucketLayout,
                          buckets: Sequence[bytes], *, eta_outer: float = 1.0
                          ) -> dict[str, torch.Tensor]:
    total_values = sum(len(payload) // _F64.size for payload in buckets)
    if total_values != sum(layout.counts):
        raise ValueError("aggregate tensor element count mismatch")
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, count in zip(layout.names, layout.shapes, layout.counts):
        if name not in base_state:
            raise ValueError(f"aggregate has unknown base tensor {name!r}")
        base = base_state[name]
        if tuple(base.shape) != shape:
            raise ValueError(f"aggregate tensor {name!r} shape mismatch")
        mean = torch.empty(count, dtype=torch.float64)
        copied = 0
        source_offset = offset
        bucket_index = source_offset // layout.bucket_elements
        bucket_offset = source_offset % layout.bucket_elements
        while copied < count:
            vector = decode_f64_tensor(buckets[bucket_index])
            take = min(count - copied, vector.numel() - bucket_offset)
            mean[copied:copied + take].copy_(vector[bucket_offset:bucket_offset + take])
            copied += take
            bucket_index += 1
            bucket_offset = 0
        mean = mean.reshape(shape)
        result[name] = base + mean.to(device=base.device, dtype=base.dtype) * float(eta_outer)
        offset += count
    missing = sorted(set(base_state) - set(layout.names))
    if missing:
        raise ValueError(f"aggregate is missing base tensors: {missing}")
    return result


def exchange_dense_delta(client: NodeManagerClient, fence: GenerationFence,
                         base_state: Mapping[str, torch.Tensor],
                         local_delta: Mapping[str, torch.Tensor], *, weight: int,
                         bucket_bytes: int, eta_outer: float = 1.0
                         ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Submit a real E97 node delta and return the redistributed global state."""
    layout, buckets = pack_dense_delta(local_delta, bucket_bytes=bucket_bytes)
    header, aggregate = client.exchange(fence, buckets, weight=weight)
    return header, apply_aggregate_delta(base_state, layout, aggregate, eta_outer=eta_outer)


class NodeStepSupervisor:
    """Deadline supervisor that terminates only the unhealthy local step."""

    def __init__(self, deadline_s: float):
        self.deadline_s = float(deadline_s)

    def run(self, command: Sequence[str], *, env: Mapping[str, str] | None = None
            ) -> subprocess.CompletedProcess[bytes]:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=None if env is None else dict(env), start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=self.deadline_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise TimeoutError(f"node step exceeded {self.deadline_s}s deadline")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
