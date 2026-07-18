"""Deterministic, bounded tensor sharding for resilient E97 DiLoCo.

This is the tensor-plane implementation of Compute Pool v1 requirements R05,
R08, and R15.  A reducer instance owns one shard only; callers distribute
``ShardChunk`` objects according to ``TensorLayout.owner``.  No object in this
module assembles or brokers a full model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Mapping, Sequence

import torch


_F64 = struct.Struct("!d")
SCHEMA = "resilient-e97-tensor-layout-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    elements: int


@dataclass(frozen=True)
class ShardChunk:
    layout_digest: str
    shard_id: int
    element_offset: int
    elements: int
    payload: bytes
    checksum_sha256: str

    @property
    def nbytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class TensorLayout:
    records: tuple[TensorRecord, ...]
    chunk_elements: int
    total_elements: int
    digest: str

    @classmethod
    def from_state(cls, state: Mapping[str, torch.Tensor], *, max_chunk_bytes: int) -> "TensorLayout":
        if max_chunk_bytes < _F64.size:
            raise ValueError("max_chunk_bytes must hold a float64 element")
        records, offset = [], 0
        for name in sorted(state):
            tensor = state[name]
            if not tensor.is_floating_point():
                raise ValueError(f"E97 state tensor {name!r} is not floating point")
            record = TensorRecord(name, tuple(tensor.shape), str(tensor.dtype), offset, tensor.numel())
            records.append(record)
            offset += tensor.numel()
        if not records:
            raise ValueError("E97 tensor layout cannot be empty")
        chunk_elements = max_chunk_bytes // _F64.size
        identity = {
            "schema": SCHEMA,
            "chunk_elements": chunk_elements,
            "records": [record.__dict__ for record in records],
        }
        return cls(tuple(records), chunk_elements, offset, _digest(identity))

    @property
    def shard_count(self) -> int:
        return math.ceil(self.total_elements / self.chunk_elements)

    def owner(self, shard_id: int, owners: Sequence[str], *, run_id: str,
              generation: int, attempt: int) -> str:
        if not owners or shard_id not in range(self.shard_count):
            raise ValueError("valid shard and at least one owner are required")
        ordered = tuple(sorted(set(owners)))
        key = f"{run_id}\0{generation}\0{attempt}\0{shard_id}".encode()
        return ordered[int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(ordered)]

    def pack(self, state: Mapping[str, torch.Tensor]) -> tuple[ShardChunk, ...]:
        self._validate_state(state)
        flat = torch.cat([state[record.name].detach().cpu().to(torch.float64).reshape(-1)
                          for record in self.records])
        chunks = []
        for shard_id, offset in enumerate(range(0, flat.numel(), self.chunk_elements)):
            raw = b"".join(_F64.pack(float(value))
                           for value in flat[offset:offset + self.chunk_elements].tolist())
            chunks.append(ShardChunk(self.digest, shard_id, offset, len(raw) // _F64.size,
                                     raw, hashlib.sha256(raw).hexdigest()))
        return tuple(chunks)

    def unpack(self, chunks: Sequence[ShardChunk]) -> dict[str, torch.Tensor]:
        ordered = sorted(chunks, key=lambda chunk: chunk.shard_id)
        if [chunk.shard_id for chunk in ordered] != list(range(self.shard_count)):
            raise ValueError("aggregate shard set is incomplete or duplicated")
        values = []
        for chunk in ordered:
            _validate_chunk(self, chunk)
            values.extend(value[0] for value in struct.iter_unpack("!d", chunk.payload))
        if len(values) != self.total_elements:
            raise ValueError("aggregate element count differs from layout")
        flat = torch.tensor(values, dtype=torch.float64)
        result = {}
        for record in self.records:
            dtype = getattr(torch, record.dtype.removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"unsupported recorded dtype {record.dtype!r}")
            result[record.name] = flat[record.offset:record.offset + record.elements].reshape(
                record.shape).to(dtype=dtype)
        return result

    def _validate_state(self, state: Mapping[str, torch.Tensor]) -> None:
        actual = tuple((name, tuple(state[name].shape), str(state[name].dtype), state[name].numel())
                       for name in sorted(state))
        expected = tuple((r.name, r.shape, r.dtype, r.elements) for r in self.records)
        if actual != expected:
            raise ValueError("E97 state does not match bound tensor layout")
        if any(not torch.isfinite(state[r.name]).all() for r in self.records):
            raise ValueError("nonfinite E97 state rejected")


def _validate_chunk(layout: TensorLayout, chunk: ShardChunk) -> None:
    if chunk.layout_digest != layout.digest or chunk.shard_id not in range(layout.shard_count):
        raise ValueError("chunk layout or shard identity mismatch")
    expected_offset = chunk.shard_id * layout.chunk_elements
    expected_elements = min(layout.chunk_elements, layout.total_elements - expected_offset)
    if (chunk.element_offset, chunk.elements, len(chunk.payload)) != (
            expected_offset, expected_elements, expected_elements * _F64.size):
        raise ValueError("chunk byte bounds or offsets mismatch")
    if hashlib.sha256(chunk.payload).hexdigest() != chunk.checksum_sha256:
        raise ValueError("chunk checksum mismatch")
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("!d", chunk.payload)):
        raise ValueError("nonfinite chunk rejected")


class ExactWeightedShardReducer:
    """Bounded owner-local reducer with deterministic float64 reference math.

    Contributions remain isolated until ``finalize`` supplies the frozen,
    complete accepted identities.  ``math.fsum`` over identity-sorted inputs
    makes the result stable across network arrival orders.  Payload buffers are
    released immediately after finalization; compact receipt hashes remain for
    idempotent replay acknowledgement.
    """

    def __init__(self, layout: TensorLayout, shard_id: int, *, max_inflight_bytes: int):
        if shard_id not in range(layout.shard_count) or max_inflight_bytes <= 0:
            raise ValueError("valid shard and positive byte bound are required")
        self.layout, self.shard_id = layout, shard_id
        self.max_inflight_bytes = int(max_inflight_bytes)
        self._pending: dict[str, tuple[int, ShardChunk]] = {}
        self._receipts: dict[str, tuple[int, str]] = {}
        self.high_water_bytes = 0

    @property
    def inflight_bytes(self) -> int:
        return sum(chunk.nbytes for _, chunk in self._pending.values())

    def submit(self, contribution_id: str, *, weight: int, chunk: ShardChunk) -> bool:
        if not contribution_id or weight <= 0:
            raise ValueError("contribution identity and positive token weight are required")
        _validate_chunk(self.layout, chunk)
        if chunk.shard_id != self.shard_id:
            raise ValueError("chunk routed to wrong shard owner")
        receipt = (int(weight), chunk.checksum_sha256)
        prior = self._receipts.get(contribution_id)
        if prior is not None:
            if prior != receipt:
                raise ValueError("conflicting contribution replay")
            return False
        if self.inflight_bytes + chunk.nbytes > self.max_inflight_bytes:
            raise BufferError("shard owner backpressure byte bound exceeded")
        self._pending[contribution_id] = (int(weight), chunk)
        self._receipts[contribution_id] = receipt
        self.high_water_bytes = max(self.high_water_bytes, self.inflight_bytes)
        return True

    def finalize(self, accepted_ids: Sequence[str]) -> ShardChunk:
        accepted = tuple(sorted(accepted_ids))
        if not accepted or len(set(accepted)) != len(accepted):
            raise ValueError("frozen accepted identities must be nonempty and unique")
        missing = [identity for identity in accepted if identity not in self._pending]
        if missing:
            raise ValueError(f"accepted contribution missing shard: {missing}")
        total_weight = sum(self._pending[identity][0] for identity in accepted)
        decoded = {
            identity: tuple(value[0] for value in struct.iter_unpack(
                "!d", self._pending[identity][1].payload)) for identity in accepted
        }
        width = len(next(iter(decoded.values())))
        values = [math.fsum(self._pending[identity][0] * decoded[identity][element]
                            for identity in accepted) / total_weight
                  for element in range(width)]
        payload = b"".join(_F64.pack(value) for value in values)
        template = self._pending[accepted[0]][1]
        self._pending.clear()  # prompt payload release after the frozen result exists
        return ShardChunk(self.layout.digest, self.shard_id, template.element_offset,
                          template.elements, payload, hashlib.sha256(payload).hexdigest())
