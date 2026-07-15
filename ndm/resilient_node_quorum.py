"""Node-level, non-collective resilient quorum protocol primitives.

This module deliberately has no MPI dependency.  A deployment gives each node
manager an independent point-to-point connection to shard owners.  The durable
metadata directory contains only small fenced manifests; bucket payloads remain
in bounded node/shard stores and are replayable after owner reassignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import socket
import struct
import threading
import time
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
TRANSPORT_MODE = "resilient-node-quorum-sharded-p2p"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass(frozen=True)
class GenerationFence:
    run_id: str
    generation: int
    attempt: int
    coordinator_epoch: int

    def key(self) -> str:
        return f"{self.run_id}:g{self.generation}:a{self.attempt}:e{self.coordinator_epoch}"


@dataclass(frozen=True)
class BucketUpdate:
    fence: GenerationFence
    node_id: str
    bucket: int
    weight: int
    payload: bytes
    checksum: str

    @classmethod
    def create(cls, fence: GenerationFence, node_id: str, bucket: int,
               weight: int, payload: bytes) -> "BucketUpdate":
        if weight <= 0:
            raise ValueError("weight must be positive")
        return cls(fence, node_id, bucket, weight, bytes(payload), _digest(payload))

    def validate(self) -> None:
        if _digest(self.payload) != self.checksum:
            raise ValueError("bucket checksum mismatch")


class BoundedBucketStore:
    """In-memory stand-in for a node-local spool with strict byte accounting."""

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._items: dict[tuple[str, str, int], BucketUpdate] = {}
        self.bytes_used = 0

    def put(self, update: BucketUpdate) -> None:
        update.validate()
        key = (update.fence.key(), update.node_id, update.bucket)
        previous = self._items.get(key)
        needed = self.bytes_used - (len(previous.payload) if previous else 0) + len(update.payload)
        if needed > self.max_bytes:
            raise BufferError("bucket retention limit exceeded")
        self._items[key] = update
        self.bytes_used = needed

    def generation(self, fence: GenerationFence) -> tuple[BucketUpdate, ...]:
        return tuple(v for (key, _, _), v in self._items.items() if key == fence.key())

    def release(self, fence: GenerationFence) -> None:
        for key in [k for k in self._items if k[0] == fence.key()]:
            self.bytes_used -= len(self._items.pop(key).payload)


class MetadataCoordinator:
    """Fenced coordinator for small durable membership/commit records."""

    def __init__(self, root: str | Path, run_id: str, epoch: int):
        self.root, self.run_id, self.epoch = Path(root), run_id, epoch
        epoch_path = self.root / "coordinator.json"
        if epoch_path.exists():
            current = json.loads(epoch_path.read_text())["epoch"]
            if epoch <= current:
                raise RuntimeError("coordinator epoch is not newer")
        _atomic_json(epoch_path, {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                                  "epoch": epoch})

    def fence(self, generation: int, attempt: int = 0) -> GenerationFence:
        return GenerationFence(self.run_id, generation, attempt, self.epoch)

    def assert_current(self, fence: GenerationFence) -> None:
        current = json.loads((self.root / "coordinator.json").read_text())
        if (fence.run_id != self.run_id or fence.coordinator_epoch != current["epoch"]
                or self.epoch != current["epoch"]):
            raise RuntimeError("stale coordinator or generation fence")

    def commit(self, fence: GenerationFence, updates: Sequence[BucketUpdate], *,
               quorum: int, expected_buckets: int) -> dict[str, object]:
        self.assert_current(fence)
        by_node: dict[str, dict[int, BucketUpdate]] = {}
        for update in updates:
            update.validate()
            if update.fence != fence:
                continue
            by_node.setdefault(update.node_id, {})[update.bucket] = update
        accepted = sorted(node for node, buckets in by_node.items()
                          if set(buckets) == set(range(expected_buckets)))
        if len(accepted) < quorum:
            raise TimeoutError("node quorum lost")
        # Freeze the exact accepted set. Extra/late nodes cannot alter this commit.
        accepted = accepted[:quorum]
        aggregates: dict[str, dict[str, object]] = {}
        for bucket in range(expected_buckets):
            members = [by_node[node][bucket] for node in accepted]
            width = len(members[0].payload)
            if any(len(item.payload) != width for item in members):
                raise ValueError("bucket widths differ")
            total_weight = sum(item.weight for item in members)
            # Byte vectors provide deterministic protocol math tests; tensor codecs
            # use the same weighted sum before serializing a committed bucket.
            values = bytes(round(sum(item.payload[i] * item.weight for item in members)
                                 / total_weight) for i in range(width))
            aggregates[str(bucket)] = {"checksum": _digest(values), "payload_hex": values.hex(),
                                       "weight": total_weight}
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION, "transport": TRANSPORT_MODE,
            "run_id": fence.run_id, "generation": fence.generation,
            "attempt": fence.attempt, "coordinator_epoch": fence.coordinator_epoch,
            "accepted_nodes": accepted, "quorum": quorum, "buckets": aggregates,
        }
        generation_path = self.root / "generations" / f"{fence.generation:08d}.json"
        _atomic_json(generation_path, manifest)
        _atomic_json(self.root / "latest.json", {
            "schema_version": SCHEMA_VERSION, "generation": fence.generation,
            "manifest": str(generation_path), "manifest_sha256": _digest(generation_path.read_bytes()),
        })
        return manifest

    def catch_up(self, local_generation: int) -> dict[str, object] | None:
        latest_path = self.root / "latest.json"
        if not latest_path.exists():
            return None
        latest = json.loads(latest_path.read_text())
        if latest["generation"] <= local_generation:
            return None
        manifest_path = Path(latest["manifest"])
        if _digest(manifest_path.read_bytes()) != latest["manifest_sha256"]:
            raise ValueError("latest manifest checksum mismatch")
        return json.loads(manifest_path.read_text())


def reassign_and_replay(fence: GenerationFence, retained: Iterable[BoundedBucketStore],
                        replacement: BoundedBucketStore) -> int:
    """Replay retained sender buckets to a replacement shard owner."""
    count = 0
    for store in retained:
        for update in store.generation(fence):
            replacement.put(update)
            count += 1
    return count


def _wire_update(update: BucketUpdate) -> bytes:
    """Encode one bucket without pickle or implicit Python object framing."""
    value = {
        "schema_version": SCHEMA_VERSION, "run_id": update.fence.run_id,
        "generation": update.fence.generation, "attempt": update.fence.attempt,
        "coordinator_epoch": update.fence.coordinator_epoch,
        "node_id": update.node_id, "bucket": update.bucket, "weight": update.weight,
        "checksum": update.checksum, "payload_hex": update.payload.hex(),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read_exact(stream: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("truncated shard-owner frame")
        chunks.extend(chunk)
    return bytes(chunks)


class ShardOwnerServer:
    """Bounded point-to-point bucket receiver; no MPI communicator is involved.

    The server accepts one structured, length-prefixed bucket per connection.
    Payload bytes are admitted only after schema, size, checksum, and current-fence
    checks.  This deliberately small primitive can run in an independent Slurm
    node-manager step and be replaced without poisoning healthy peers.
    """

    def __init__(self, address: tuple[str, int], store: BoundedBucketStore,
                 fence: GenerationFence, *, max_frame_bytes: int):
        self.store, self.fence, self.max_frame_bytes = store, fence, max_frame_bytes
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(address)
        self._socket.listen()
        self._socket.settimeout(.1)
        self.address = self._socket.getsockname()
        self._closed = threading.Event()

    def serve(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._closed.is_set():
                    return
                raise
            with connection:
                try:
                    size = struct.unpack("!I", _read_exact(connection, 4))[0]
                    if size <= 0 or size > self.max_frame_bytes:
                        raise ValueError("invalid shard-owner frame size")
                    value = json.loads(_read_exact(connection, size))
                    if value.get("schema_version") != SCHEMA_VERSION:
                        raise ValueError("unsupported bucket schema")
                    fence = GenerationFence(value["run_id"], value["generation"],
                                            value["attempt"], value["coordinator_epoch"])
                    if fence != self.fence:
                        raise ValueError("stale or late generation fence")
                    payload = bytes.fromhex(value["payload_hex"])
                    update = BucketUpdate(fence, value["node_id"], value["bucket"],
                                          value["weight"], payload, value["checksum"])
                    self.store.put(update)
                    reply = {"ok": True, "checksum": update.checksum}
                except Exception as error:
                    reply = {"ok": False, "error": str(error)}
                encoded = json.dumps(reply, sort_keys=True).encode()
                connection.sendall(struct.pack("!I", len(encoded)) + encoded)

    def close(self) -> None:
        self._closed.set()
        self._socket.close()


def send_bucket(address: tuple[str, int], update: BucketUpdate, *, timeout: float,
                max_frame_bytes: int) -> None:
    """Send a retained bucket to one owner and require a checksummed receipt."""
    encoded = _wire_update(update)
    if len(encoded) > max_frame_bytes:
        raise ValueError("bucket frame exceeds configured maximum")
    with socket.create_connection(address, timeout=timeout) as stream:
        stream.settimeout(timeout)
        stream.sendall(struct.pack("!I", len(encoded)) + encoded)
        size = struct.unpack("!I", _read_exact(stream, 4))[0]
        reply = json.loads(_read_exact(stream, size))
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "shard owner rejected bucket"))
    if reply.get("checksum") != update.checksum:
        raise ValueError("shard-owner receipt checksum mismatch")


def supervise_until_quorum(processes: Mapping[str, object], store: BoundedBucketStore,
                           fence: GenerationFence, *, quorum: int,
                           expected_buckets: int, deadline: float) -> tuple[str, ...]:
    """Wait to a monotonic deadline, terminating only incomplete node steps."""
    while time.monotonic() < deadline:
        complete = _complete_nodes(store.generation(fence), expected_buckets)
        if len(complete) >= quorum:
            accepted = tuple(sorted(complete)[:quorum])
            for node_id, process in processes.items():
                if node_id not in accepted and getattr(process, "is_alive")():
                    getattr(process, "terminate")()
                    getattr(process, "join")(5)
            return accepted
        time.sleep(.01)
    complete = _complete_nodes(store.generation(fence), expected_buckets)
    for node_id, process in processes.items():
        if node_id not in complete and getattr(process, "is_alive")():
            getattr(process, "terminate")()
            getattr(process, "join")(5)
    if len(complete) < quorum:
        raise TimeoutError("node quorum lost at progress deadline")
    return tuple(sorted(complete)[:quorum])


def _complete_nodes(updates: Iterable[BucketUpdate], expected_buckets: int) -> set[str]:
    by_node: dict[str, set[int]] = {}
    for update in updates:
        by_node.setdefault(update.node_id, set()).add(update.bucket)
    wanted = set(range(expected_buckets))
    return {node for node, buckets in by_node.items() if buckets == wanted}
