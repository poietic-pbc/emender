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
import os
from pathlib import Path
import socket
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
from typing import Callable, Mapping, Sequence

from ndm.resilient_node_quorum import GenerationFence, MetadataCoordinator


_U32 = struct.Struct("!I")
_F64 = struct.Struct("!d")
MAX_HEADER_BYTES = 64 * 1024


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode_f64(values: Sequence[float]) -> bytes:
    return b"".join(_F64.pack(float(value)) for value in values)


def decode_f64(payload: bytes) -> tuple[float, ...]:
    if len(payload) % _F64.size:
        raise ValueError("float64 bucket has a partial element")
    return tuple(value[0] for value in struct.iter_unpack("!d", payload))


def _recv_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            raise EOFError("peer closed framed transport")
        data.extend(part)
    return bytes(data)


def send_frame(stream, header: Mapping[str, object], payload: bytes = b"") -> None:
    wire = dict(header)
    wire["payload_bytes"] = len(payload)
    wire["payload_sha256"] = _sha(payload)
    encoded = json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("frame header too large")
    stream.write(_U32.pack(len(encoded)))
    stream.write(encoded)
    stream.write(payload)
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


@dataclass(frozen=True)
class TransportConfig:
    run_id: str
    quorum: int
    expected_buckets: int
    max_bucket_bytes: int
    generation_deadline_s: float


class _Generation:
    def __init__(self, fence: GenerationFence, deadline: float):
        self.fence, self.deadline = fence, deadline
        self.updates: dict[str, dict[int, tuple[int, bytes]]] = {}
        self.accepted: tuple[str, ...] | None = None
        self.aggregates: dict[int, bytes] = {}
        self.condition = threading.Condition()


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
        with state.condition:
            if state.accepted is not None and node not in state.accepted:
                raise ValueError("late node rejected after accepted set was frozen")
            state.updates.setdefault(node, {})[bucket] = (weight, payload)
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
                self._publish_manifest(state)
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
        target.write_text(json.dumps(payload, sort_keys=True) + "\n")


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            header, payload = recv_frame(
                self.rfile, max_payload_bytes=self.server.config.max_bucket_bytes
            )
            if header.get("op") != "submit":
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
                "bucket_count": len(state.aggregates),
                "generation": state.fence.generation,
                "attempt": state.fence.attempt,
                "coordinator_epoch": state.fence.coordinator_epoch,
            })
            for bucket in sorted(state.aggregates):
                send_frame(self.wfile, {"op": "aggregate", "bucket": bucket},
                           state.aggregates[bucket])
        except Exception as error:
            send_frame(self.wfile, {"op": "error", "error": str(error)})


class NodeManagerClient:
    def __init__(self, host: str, port: int, node_id: str, spool: DiskBucketSpool,
                 *, timeout_s: float = 30.0, max_bucket_bytes: int = 64 << 20):
        self.host, self.port, self.node_id = host, port, node_id
        self.spool, self.timeout_s = spool, float(timeout_s)
        self.max_bucket_bytes = int(max_bucket_bytes)

    def exchange(self, fence: GenerationFence, buckets: Sequence[bytes], *, weight: int
                 ) -> tuple[dict[str, object], tuple[bytes, ...]]:
        paths = [self.spool.retain(fence, i, payload) for i, payload in enumerate(buckets)]
        replies = []
        # One independent connection per bucket allows bucket sharding/replay.
        for index, path in enumerate(paths):
            sock = socket.create_connection((self.host, self.port), self.timeout_s)
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
