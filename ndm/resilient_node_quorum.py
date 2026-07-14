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

