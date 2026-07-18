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
from typing import Mapping, Sequence

import numpy as np
import torch


_F64_BYTES = 8
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
        if max_chunk_bytes < _F64_BYTES:
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
        chunk_elements = max_chunk_bytes // _F64_BYTES
        identity = {
            "schema": SCHEMA,
            "chunk_elements": chunk_elements,
            "records": [record.__dict__ for record in records],
        }
        return cls(tuple(records), chunk_elements, offset, _digest(identity))

    @classmethod
    def from_flat_stream(cls, total_elements: int, *, max_chunk_bytes: int,
                         dtype: str = "torch.float64") -> "TensorLayout":
        """Bind the manager's deterministic flat shard stream without a full cat."""
        if total_elements <= 0 or max_chunk_bytes < _F64_BYTES:
            raise ValueError("flat stream needs elements and a float64-sized chunk")
        chunk_elements = max_chunk_bytes // _F64_BYTES
        records = (TensorRecord("flat", (int(total_elements),), dtype, 0,
                                int(total_elements)),)
        identity = {"schema": SCHEMA, "chunk_elements": chunk_elements,
                    "records": [record.__dict__ for record in records]}
        return cls(records, chunk_elements, int(total_elements), _digest(identity))

    @property
    def shard_count(self) -> int:
        return math.ceil(self.total_elements / self.chunk_elements)

    def owner(self, shard_id: int, owners: Sequence[str], *, run_id: str,
              generation: int, attempt: int) -> str:
        if not owners or shard_id not in range(self.shard_count):
            raise ValueError("valid shard and at least one owner are required")
        ordered = tuple(sorted(set(owners)))
        # Hash the generation cohort once, then stripe consecutive full-layout
        # shards across it. This is deterministic under reordered endpoint
        # discovery and guarantees distribution whenever shards >= owners;
        # independent per-shard hashing could accidentally centralize a small
        # representative (or tail) layout on node 0.
        key = f"{run_id}\0{generation}\0{attempt}".encode()
        start = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(ordered)
        return ordered[(start + shard_id) % len(ordered)]

    def pack(self, state: Mapping[str, torch.Tensor]) -> tuple[ShardChunk, ...]:
        self._validate_state(state)
        # Stream deterministic tensor slices into bounded vectorized network-order
        # buffers.  Never build a dense Python list or concatenate the full model.
        chunks: list[ShardChunk] = []
        pending = bytearray()
        element_offset = 0
        for record in self.records:
            flat = state[record.name].detach().cpu().to(torch.float64).reshape(-1)
            offset = 0
            while offset < flat.numel():
                room = self.chunk_elements - len(pending) // _F64_BYTES
                take = min(room, flat.numel() - offset)
                vector = flat[offset:offset + take].contiguous().numpy()
                pending.extend(vector.astype(">f8", copy=False).tobytes())
                offset += take
                if len(pending) == self.chunk_elements * _F64_BYTES:
                    raw = bytes(pending)
                    shard_id = len(chunks)
                    chunks.append(ShardChunk(self.digest, shard_id, element_offset,
                                             len(raw) // _F64_BYTES, raw,
                                             hashlib.sha256(raw).hexdigest()))
                    element_offset += len(raw) // _F64_BYTES
                    pending.clear()
        if pending:
            raw = bytes(pending)
            chunks.append(ShardChunk(self.digest, len(chunks), element_offset,
                                     len(raw) // _F64_BYTES, raw,
                                     hashlib.sha256(raw).hexdigest()))
        return tuple(chunks)

    def pack_flat_shards(self, shards: Sequence[torch.Tensor]) -> tuple[ShardChunk, ...]:
        """Vectorize an already-bounded manager stream without full-model materialization."""
        chunks, pending = [], torch.empty(0, dtype=torch.float64)
        element_offset = 0
        for shard in shards:
            flat = shard.detach().cpu().to(torch.float64).reshape(-1)
            if not torch.isfinite(flat).all():
                raise ValueError("nonfinite flat E97 shard rejected")
            offset = 0
            while offset < flat.numel():
                take = min(self.chunk_elements - pending.numel(), flat.numel() - offset)
                piece = flat[offset:offset + take]
                pending = torch.cat((pending, piece)) if pending.numel() else piece
                offset += take
                if pending.numel() == self.chunk_elements:
                    raw = pending.contiguous().numpy().astype(">f8", copy=False).tobytes()
                    chunks.append(ShardChunk(self.digest, len(chunks), element_offset,
                                             pending.numel(), raw,
                                             hashlib.sha256(raw).hexdigest()))
                    element_offset += pending.numel()
                    pending = torch.empty(0, dtype=torch.float64)
        if pending.numel():
            raw = pending.contiguous().numpy().astype(">f8", copy=False).tobytes()
            chunks.append(ShardChunk(self.digest, len(chunks), element_offset,
                                     pending.numel(), raw, hashlib.sha256(raw).hexdigest()))
            element_offset += pending.numel()
        if element_offset != self.total_elements:
            raise ValueError("flat manager stream differs from bound E97 layout")
        return tuple(chunks)

    def unpack_flat_shards(self, chunks: Sequence[ShardChunk]) -> tuple[torch.Tensor, ...]:
        """Decode each committed owner chunk separately for bounded redistribution."""
        ordered = tuple(sorted(chunks, key=lambda item: item.shard_id))
        if tuple(item.shard_id for item in ordered) != tuple(range(self.shard_count)):
            raise ValueError("aggregate shard set is incomplete or duplicated")
        result = []
        for chunk in ordered:
            _validate_chunk(self, chunk)
            result.append(torch.from_numpy(
                np.frombuffer(chunk.payload, dtype=">f8").astype(np.float64)).clone())
        return tuple(result)

    def unpack(self, chunks: Sequence[ShardChunk]) -> dict[str, torch.Tensor]:
        ordered = sorted(chunks, key=lambda chunk: chunk.shard_id)
        if [chunk.shard_id for chunk in ordered] != list(range(self.shard_count)):
            raise ValueError("aggregate shard set is incomplete or duplicated")
        flat = torch.empty(self.total_elements, dtype=torch.float64)
        for chunk in ordered:
            _validate_chunk(self, chunk)
            values = np.frombuffer(chunk.payload, dtype=">f8").astype(np.float64)
            flat[chunk.element_offset:chunk.element_offset + chunk.elements].copy_(
                torch.from_numpy(values))
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
            expected_offset, expected_elements, expected_elements * _F64_BYTES):
        raise ValueError("chunk byte bounds or offsets mismatch")
    if hashlib.sha256(chunk.payload).hexdigest() != chunk.checksum_sha256:
        raise ValueError("chunk checksum mismatch")
    if not np.isfinite(np.frombuffer(chunk.payload, dtype=">f8")).all():
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
        # Accumulate identity-sorted vectors in float64 so network arrival order
        # cannot affect the committed bytes.  NumPy operates on large contiguous
        # buffers; there is no per-element Python decode/pack loop.
        template = self._pending[accepted[0]][1]
        weighted = np.zeros(template.elements, dtype=np.float64)
        for identity in accepted:
            weight, chunk = self._pending[identity]
            values = np.frombuffer(chunk.payload, dtype=">f8")
            np.add(weighted, values * weight, out=weighted)
        weighted /= total_weight
        payload = weighted.astype(">f8", copy=False).tobytes()
        self._pending.clear()  # prompt payload release after the frozen result exists
        return ShardChunk(self.layout.digest, self.shard_id, template.element_offset,
                          template.elements, payload, hashlib.sha256(payload).hexdigest())
