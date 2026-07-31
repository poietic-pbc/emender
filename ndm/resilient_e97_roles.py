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
import fcntl
from pathlib import Path
import tempfile
import time
from typing import Iterable, Mapping, Sequence

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
    """Bounded atomic trainer-to-manager stream spool.

    Every trainer contribution is one large sequential data file plus one small
    manifest.  Shard offsets retain the bounded tensor-stream interface without
    creating/fsyncing a file per microchunk.  The manifest is published last,
    so a manager never observes a partial contribution.
    """

    def __init__(self, root: str | Path, max_bytes: int):
        self.root, self.max_bytes = Path(root), int(max_bytes)
        self.high_water_bytes = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.files_published = 0
        self.root.mkdir(parents=True, exist_ok=True)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._usage_lock = self.root / ".usage.lock"
        with self._usage_lock.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0, os.SEEK_END)
            if stream.tell() < 8:
                actual = self._scan_payload_bytes()
                stream.seek(0); stream.write(int(actual).to_bytes(8, "little"))
                stream.truncate(8); stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _scan_payload_bytes(self) -> int:
        total = 0
        for directory, _, names in os.walk(self.root):
            for name in names:
                path = Path(directory) / name
                if path == self._usage_lock:
                    continue
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
        return total

    def _reserve(self, delta: int) -> int:
        """Atomically reserve/release bytes across all local role processes."""
        with self._usage_lock.open("r+b", buffering=0) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            current = int.from_bytes(stream.read(8), "little")
            updated = current + int(delta)
            if updated < 0:
                updated = 0
            if updated > self.max_bytes:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                raise BufferError("local trainer spool byte limit exceeded")
            stream.seek(0); stream.write(updated.to_bytes(8, "little"))
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self.high_water_bytes = max(self.high_water_bytes, updated)
        return updated

    def _remove_payloads(self, paths: Iterable[Path]) -> None:
        removed = 0
        for path in paths:
            try:
                removed += path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
        if removed:
            self._reserve(-removed)

    @property
    def bytes_used(self) -> int:
        with self._usage_lock.open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            total = int.from_bytes(stream.read(8), "little")
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return total

    @property
    def file_count(self) -> int:
        return sum(sum(name != ".usage.lock" for name in names)
                   for _, _, names in os.walk(self.root))

    def publish(self, fence: LocalFence, trainer_id: int, shards: Iterable[torch.Tensor],
                *, weight: int, source_id: str) -> Path:
        if not 0 <= trainer_id < 8:
            raise ValueError("trainer_id must identify one of eight real trainers")
        if weight <= 0 or not source_id:
            raise ValueError("weight and source identity are required")
        directory = self.root / fence.key / f"trainer-{trainer_id}"
        if (directory / "manifest.json").exists():
            raise ValueError("duplicate trainer contribution")
        records = []
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / "contribution.data"
        fd, temporary_name = tempfile.mkstemp(prefix=".contribution.", dir=directory)
        temporary = Path(temporary_name)
        written = 0
        reserved = 0
        try:
            with os.fdopen(fd, "wb", buffering=8 << 20) as stream:
                for index, tensor in enumerate(shards):
                    value = tensor.detach().cpu().contiguous()
                    if not value.is_floating_point() or not torch.isfinite(value).all():
                        raise ValueError("trainer shard must be finite floating-point data")
                    # Trainer deltas originate in model precision. Preserve their
                    # established f32/f64 wire contract, but append into one stream.
                    dtype = "float64" if value.dtype == torch.float64 else "float32"
                    encoded_value = value.to(
                        torch.float64 if dtype == "float64" else torch.float32)
                    raw = encoded_value.numpy().tobytes()
                    self._reserve(len(raw)); reserved += len(raw)
                    stream.write(raw)
                    records.append({"index": index, "offset": written,
                                    "nbytes": len(raw), "elements": value.numel(),
                                    "dtype": dtype,
                                    "sha256": hashlib.sha256(raw).hexdigest()})
                    written += len(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if not records:
                raise ValueError("trainer contribution must contain at least one shard")
            os.replace(temporary, data_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
            if reserved:
                self._reserve(-reserved)
            raise
        manifest = {"schema": SCHEMA, "fence": fence.__dict__, "trainer_id": trainer_id,
                    "weight": weight, "source_id": source_id,
                    "data_file": data_path.name, "data_bytes": written,
                    "shards": records}
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        try:
            self._reserve(len(manifest_bytes)); reserved += len(manifest_bytes)
        except BufferError:
            data_path.unlink(missing_ok=True)
            self._reserve(-reserved)
            raise
        try:
            _atomic(directory / "manifest.json", manifest_bytes)
        except BaseException:
            data_path.unlink(missing_ok=True)
            self._reserve(-reserved)
            raise
        self.bytes_written += written
        self.files_published += 2
        self.high_water_bytes = max(self.high_water_bytes, self.bytes_used)
        return directory / "manifest.json"

    def release_generation(self, fence: LocalFence) -> None:
        root = self.root / fence.key
        if root.exists():
            payloads = []
            for directory, names, files in os.walk(root, topdown=False):
                for name in files:
                    payloads.append(Path(directory) / name)
            self._remove_payloads(payloads)
            for directory, names, _ in os.walk(root, topdown=False):
                for name in names:
                    try:
                        (Path(directory) / name).rmdir()
                    except FileNotFoundError:
                        pass
            try:
                root.rmdir()
            except FileNotFoundError:
                pass

    def release_trainer(self, fence: LocalFence, trainer_id: int) -> None:
        directory = self.root / fence.key / f"trainer-{trainer_id}"
        if directory.exists():
            try:
                self._remove_payloads(tuple(directory.iterdir()))
                directory.rmdir()
            except FileNotFoundError:
                pass  # manager and owning trainer may release concurrently

    def publish_aggregate(self, fence: LocalFence, members: Sequence[int],
                          shards: Sequence[torch.Tensor], *, weight: int,
                          source_id: str,
                          accepted_peers: Sequence[str] = (),
                          storage_dtype: torch.dtype = torch.float64) -> Path:
        """Atomically publish the manager result after every shard is durable."""
        if weight <= 0 or not members or not source_id:
            raise ValueError("aggregate membership, weight, and source are required")
        if storage_dtype not in {torch.float32, torch.float64}:
            raise ValueError("aggregate storage dtype must be float32 or float64")
        dtype_name = "float32" if storage_dtype == torch.float32 else "float64"
        directory = self.root / "aggregates" / fence.key
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            current = json.loads(manifest_path.read_text())
            metadata_matches = (
                current.get("schema") == SCHEMA
                and current.get("fence") == fence.__dict__
                and current.get("members") == sorted(int(member) for member in members)
                and int(current.get("weight", -1)) == int(weight)
                and current.get("source_id") == source_id
                and current.get("accepted_peers") == sorted(set(map(str, accepted_peers)))
                and current.get("storage_dtype", "float64") == dtype_name
                and len(current.get("shards", ())) == len(shards)
            )
            payload_matches = metadata_matches
            if payload_matches:
                for tensor, record in zip(shards, current["shards"]):
                    value = tensor.detach().cpu().to(storage_dtype).contiguous()
                    raw = value.numpy().tobytes()
                    if (record.get("dtype", "float64") != dtype_name
                            or int(record.get("elements", -1)) != value.numel()
                            or int(record.get("nbytes", -1)) != len(raw)
                            or record.get("sha256") != hashlib.sha256(raw).hexdigest()):
                        payload_matches = False
                        break
            if payload_matches:
                return manifest_path
            raise ValueError("conflicting aggregate publication")
        records = []
        offset = 0
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / "aggregate.data"
        fd, temporary_name = tempfile.mkstemp(prefix=".aggregate.", dir=directory)
        temporary = Path(temporary_name)
        reserved = 0
        try:
            with os.fdopen(fd, "wb", buffering=8 << 20) as stream:
                for index, tensor in enumerate(shards):
                    value = tensor.detach().cpu().to(storage_dtype).contiguous()
                    if not torch.isfinite(value).all():
                        raise ValueError("nonfinite aggregate rejected")
                    raw = value.numpy().tobytes()
                    self._reserve(len(raw)); reserved += len(raw)
                    stream.write(raw)
                    records.append({"index": index, "offset": offset, "nbytes": len(raw),
                                    "elements": value.numel(),
                                    "dtype": dtype_name,
                                    "sha256": hashlib.sha256(raw).hexdigest()})
                    offset += len(raw)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, data_path)
        except BaseException:
            temporary.unlink(missing_ok=True); data_path.unlink(missing_ok=True)
            if reserved: self._reserve(-reserved)
            raise
        manifest = {"schema": SCHEMA, "fence": fence.__dict__,
                    "members": sorted(int(member) for member in members),
                    "weight": int(weight), "source_id": source_id,
                    "accepted_peers": sorted(set(map(str, accepted_peers))),
                    "storage_dtype": dtype_name,
                    "data_file": "aggregate.data", "data_bytes": offset,
                    "shards": records}
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        try:
            self._reserve(len(manifest_bytes)); reserved += len(manifest_bytes)
            _atomic(manifest_path, manifest_bytes)
        except BaseException:
            data_path.unlink(missing_ok=True); manifest_path.unlink(missing_ok=True)
            self._reserve(-reserved)
            raise
        self.bytes_written += offset
        self.files_published += 2
        self.high_water_bytes = max(self.high_water_bytes, self.bytes_used)
        return manifest_path

    def prune_aggregates(self, *, keep_generations: int = 2) -> None:
        if keep_generations <= 0:
            raise ValueError("keep_generations must be positive")
        root = self.root / "aggregates"
        generations = sorted((path for path in root.glob("*") if path.is_dir()),
                             key=lambda path: path.stat().st_mtime_ns)
        for directory in generations[:-keep_generations]:
            self._remove_payloads(tuple(directory.iterdir()))
            directory.rmdir()

    def stream_aggregate(self, fence: LocalFence, *, deadline: float,
                         expected_source_id: str) -> tuple[dict[str, object], Iterable[torch.Tensor]]:
        """Return one validated bounded shard stream without a dense aggregate copy."""
        path = self.root / "aggregates" / fence.key / "manifest.json"
        while time.monotonic() < deadline and not path.exists():
            time.sleep(.02)
        if not path.exists():
            raise TimeoutError("aggregate apply deadline expired")
        manifest = json.loads(path.read_text())
        if (manifest.get("schema") != SCHEMA or manifest.get("fence") != fence.__dict__
                or manifest.get("source_id") != expected_source_id):
            raise ValueError("aggregate identity/fence mismatch")
        data_path = path.parent / str(manifest["data_file"])
        if not data_path.is_file() or data_path.stat().st_size != int(manifest["data_bytes"]):
            raise ValueError("corrupt aggregate data stream")

        def shards() -> Iterable[torch.Tensor]:
            expected_offset = 0
            with data_path.open("rb", buffering=8 << 20) as stream:
                for record in manifest["shards"]:
                    dtype_name = record.get("dtype", "float64")
                    if dtype_name not in {"float32", "float64"}:
                        raise ValueError("invalid aggregate shard dtype")
                    width = 4 if dtype_name == "float32" else 8
                    if (int(record.get("offset", -1)) != expected_offset
                            or int(record.get("nbytes", -1))
                            != int(record.get("elements", -1)) * width):
                        raise ValueError("invalid aggregate shard layout")
                    stream.seek(expected_offset)
                    raw = stream.read(int(record["nbytes"]))
                    self.bytes_read += len(raw)
                    if (len(raw) != int(record["nbytes"])
                            or hashlib.sha256(raw).hexdigest() != record["sha256"]):
                        raise ValueError("corrupt aggregate shard")
                    dtype = torch.float32 if dtype_name == "float32" else torch.float64
                    value = torch.frombuffer(bytearray(raw), dtype=dtype)
                    if value.numel() != record["elements"] or not torch.isfinite(value).all():
                        raise ValueError("invalid aggregate shard")
                    expected_offset += len(raw)
                    yield value
            if expected_offset != int(manifest["data_bytes"]):
                raise ValueError("invalid aggregate byte count")

        return manifest, shards()

    def wait_aggregate(self, fence: LocalFence, *, deadline: float,
                       expected_source_id: str) -> tuple[dict[str, object], tuple[torch.Tensor, ...]]:
        manifest, shards = self.stream_aggregate(
            fence, deadline=deadline, expected_source_id=expected_source_id)
        return manifest, tuple(shards)


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
            accepted = self._validated_manifests(fence, expected_source_id)
            if len(accepted) >= self.quorum:
                break
            time.sleep(.02)
        if len(accepted) < self.quorum:
            raise TimeoutError("local trainer quorum lost at aggregation deadline")
        self.spool.high_water_bytes = max(self.spool.high_water_bytes, self.spool.bytes_used)
        accepted = accepted[:self.quorum]
        widths = {tuple(record["elements"] for record in item[3]["shards"])
                  for item in accepted}
        if len(widths) != 1:
            raise ValueError("incompatible trainer shard layout")
        total_weight = sum(item[1] for item in accepted)
        aggregates = []
        streams = [((manifest_path.parent / str(manifest["data_file"])).open(
                    "rb", buffering=8 << 20)) for _, _, manifest_path, manifest in accepted]
        try:
            for shard_index in range(len(accepted[0][3]["shards"])):
                weighted = None
                for stream, (_, weight, _, manifest) in zip(streams, accepted):
                    record = manifest["shards"][shard_index]
                    stream.seek(int(record["offset"]))
                    raw = stream.read(int(record["nbytes"]))
                    self.spool.bytes_read += len(raw)
                    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                        raise ValueError("corrupt trainer shard")
                    dtype = torch.float64 if record.get("dtype") == "float64" else torch.float32
                    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).to(torch.float64)
                    if tensor.numel() != record["elements"] or not torch.isfinite(tensor).all():
                        raise ValueError("invalid trainer shard")
                    weighted = (tensor.mul(weight) if weighted is None
                                else weighted.add_(tensor, alpha=weight))
                aggregate = weighted.div_(total_weight)
                if not torch.isfinite(aggregate).all():
                    raise ValueError("nonfinite aggregate rejected")
                aggregates.append(aggregate)
        finally:
            for stream in streams:
                stream.close()
        return tuple(item[0] for item in accepted), total_weight, tuple(aggregates)

    def _validated_manifests(self, fence: LocalFence, source_id: str):
        values = []
        for manifest_path in sorted((self.spool.root / fence.key).glob("trainer-*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != SCHEMA or manifest.get("fence") != fence.__dict__:
                raise ValueError("stale or incompatible trainer fence")
            if manifest.get("source_id") != source_id:
                raise ValueError("trainer payload/source identity mismatch")
            data_path = manifest_path.parent / str(manifest.get("data_file", ""))
            if not data_path.is_file() or data_path.stat().st_size != int(
                    manifest.get("data_bytes", -1)):
                raise ValueError("corrupt trainer data stream")
            expected_offset = 0
            for record in manifest["shards"]:
                width = 8 if record.get("dtype") == "float64" else 4
                if (int(record.get("offset", -1)) != expected_offset
                        or int(record.get("nbytes", -1)) != record["elements"] * width):
                    raise ValueError("corrupt trainer shard descriptor")
                expected_offset += int(record["nbytes"])
            values.append((int(manifest["trainer_id"]), int(manifest["weight"]),
                           manifest_path, manifest))
        return values
