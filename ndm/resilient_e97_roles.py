"""Local split-role protocol for resilient E97 trainers and node managers.

Trainers are the only processes allowed to own a model or optimizer.  They
publish checksummed tensor shards into a bounded node-local spool.  The CPU
manager validates and aggregates those shards without importing a model,
optimizer, dataset, distributed process group, or collective runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import torch


SCHEMA = 1


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class LocalFence:
    run_id: str
    generation: int
    attempt: int
    coordinator_epoch: int
    payload_id: str

    @property
    def key(self) -> str:
        return (f"{self.run_id}-g{self.generation}-a{self.attempt}-"
                f"e{self.coordinator_epoch}-{self.payload_id}")


class LocalTrainerSpool:
    """Bounded atomic trainer-to-manager spool.

    A shard is a flat CPU tensor serialized without pickle.  Its manifest is
    published last, so a manager never observes a partial contribution.
    """

    def __init__(self, root: str | Path, max_bytes: int):
        self.root, self.max_bytes = Path(root), int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    @property
    def bytes_used(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def publish(self, fence: LocalFence, trainer_id: int, shards: Sequence[torch.Tensor],
                *, weight: int, source_id: str) -> Path:
        if not 0 <= trainer_id < 8:
            raise ValueError("trainer_id must identify one of eight real trainers")
        if weight <= 0 or not source_id:
            raise ValueError("weight and source identity are required")
        directory = self.root / fence.key / f"trainer-{trainer_id}"
        if (directory / "manifest.json").exists():
            raise ValueError("duplicate trainer contribution")
        records, encoded = [], []
        for index, tensor in enumerate(shards):
            value = tensor.detach().cpu().contiguous()
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("trainer shard must be finite floating-point data")
            raw = value.to(torch.float64).numpy().tobytes()
            records.append({"index": index, "elements": value.numel(),
                            "sha256": hashlib.sha256(raw).hexdigest()})
            encoded.append(raw)
        manifest = {"schema": SCHEMA, "fence": fence.__dict__, "trainer_id": trainer_id,
                    "weight": weight, "source_id": source_id, "shards": records}
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        needed = sum(map(len, encoded)) + len(manifest_bytes)
        if self.bytes_used + needed > self.max_bytes:
            raise BufferError("local trainer spool byte limit exceeded")
        for record, raw in zip(records, encoded):
            _atomic(directory / f"shard-{record['index']:05d}.f64", raw)
        _atomic(directory / "manifest.json", manifest_bytes)
        return directory / "manifest.json"

    def release_generation(self, fence: LocalFence) -> None:
        root = self.root / fence.key
        if root.exists():
            for path in sorted(root.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            root.rmdir()


class CpuNodeManager:
    """Validate and exactly weight trainer shards without model ownership."""

    def __init__(self, spool: LocalTrainerSpool, *, quorum: int = 6):
        if not 1 <= quorum <= 8:
            raise ValueError("local quorum must be in [1, 8]")
        self.spool, self.quorum = spool, quorum

    def collect(self, fence: LocalFence, *, deadline: float,
                expected_source_id: str) -> tuple[tuple[int, ...], int, tuple[torch.Tensor, ...]]:
        accepted = []
        while time.monotonic() < deadline:
            accepted = self._validated(fence, expected_source_id)
            if len(accepted) >= self.quorum:
                break
            time.sleep(.02)
        if len(accepted) < self.quorum:
            raise TimeoutError("local trainer quorum lost at aggregation deadline")
        accepted = accepted[:self.quorum]
        widths = {tuple(item[2]) for item in accepted}
        if len(widths) != 1:
            raise ValueError("incompatible trainer shard layout")
        total_weight = sum(item[1] for item in accepted)
        aggregates = []
        for shard_index in range(len(accepted[0][3])):
            aggregate = sum(item[3][shard_index] * item[1] for item in accepted) / total_weight
            if not torch.isfinite(aggregate).all():
                raise ValueError("nonfinite aggregate rejected")
            aggregates.append(aggregate)
        return tuple(item[0] for item in accepted), total_weight, tuple(aggregates)

    def _validated(self, fence: LocalFence, source_id: str):
        values = []
        for manifest_path in sorted((self.spool.root / fence.key).glob("trainer-*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != SCHEMA or manifest.get("fence") != fence.__dict__:
                raise ValueError("stale or incompatible trainer fence")
            if manifest.get("source_id") != source_id:
                raise ValueError("trainer payload/source identity mismatch")
            shards = []
            for record in manifest["shards"]:
                raw = (manifest_path.parent / f"shard-{record['index']:05d}.f64").read_bytes()
                if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                    raise ValueError("corrupt trainer shard")
                tensor = torch.frombuffer(bytearray(raw), dtype=torch.float64).clone()
                if tensor.numel() != record["elements"] or not torch.isfinite(tensor).all():
                    raise ValueError("invalid trainer shard")
                shards.append(tensor)
            values.append((int(manifest["trainer_id"]), int(manifest["weight"]),
                           [int(r["elements"]) for r in manifest["shards"]], shards))
        return values
