"""Real E97 bindings for the persistent native node service.

This module contains no owner TCP server and writes no dense filesystem spool.
Python publishes only bounded JSON identities.  A trainer fills a service-owned
memfd in final f32 wire form, submits its immutable descriptor through the
metadata-only AF_UNIX RPC, and later maps the service's one shared result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator, Mapping, Sequence
import struct

import numpy as np
import torch

from ndm.native_dataplane import (
    Buffer, Client, DType, NativeLibrary, Operation, Role,
    copy_fd_range, create_memfd, encode_flat_layout, seal_memfd,
)


SCHEMA = "emender-native-e97-generation-v1"
ASYNC_V2_SCHEMA = "emender-native-e97-generation-v2"


def atomic_metadata(path: str | Path, value: Mapping[str, object]) -> Path:
    """Durably replace one small node-local metadata record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def wait_metadata(path: str | Path, *, deadline: float,
                  expected: Mapping[str, object] | None = None) -> dict[str, object]:
    target = Path(path)
    last: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("native metadata must be a JSON object")
            if expected and any(value.get(key) != item for key, item in expected.items()):
                raise ValueError("native metadata identity/fence mismatch")
            return value
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            last = error
            time.sleep(.02)
    raise TimeoutError(f"native metadata deadline expired for {target}: {last}")


def artifact_path(build_manifest: str | Path, name: str) -> Path:
    manifest_path = Path(build_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("artifacts", {}).get(name)
    if not isinstance(record, dict):
        raise ValueError(f"native build manifest has no {name!r}")
    path = (manifest_path.parent / str(record.get("path", ""))).resolve()
    path.relative_to(manifest_path.parent)
    if (not path.is_file() or path.stat().st_size != int(record.get("bytes", -1))
            or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256")):
        raise ValueError(f"native artifact {name!r} failed digest/size verification")
    return path


def runtime_digests(*, build_manifest: str | Path, config_path: str | Path,
                    provider: str, attestation: Mapping[str, object]) -> dict[str, object]:
    """Bind provider, build bundle, immutable E97 config, and source identity."""
    manifest_path, config = Path(build_manifest).resolve(), Path(config_path).resolve()
    manifest_bytes, config_bytes = manifest_path.read_bytes(), config.read_bytes()
    manifest = json.loads(manifest_bytes)
    bundle = str(attestation.get("bundle_sha256", ""))
    if len(bundle) != 64 or manifest.get("bundle_sha256") != bundle:
        raise ValueError("native runtime build bundle differs from launch attestation")
    if provider not in {"cxi", "tcp;ofi_rxm"}:
        raise ValueError("native runtime provider is not an approved exact provider")
    return {
        "schema": "emender-native-e97-runtime-digests-v1",
        "provider": provider,
        "provider_sha256": hashlib.sha256(
            ("emender-native-provider-v1\0" + provider).encode()).hexdigest(),
        "build_bundle_sha256": bundle,
        "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_commit": str(attestation.get("source_commit", "")),
        "artifacts": {name: str(record["sha256"])
                      for name, record in sorted(manifest["artifacts"].items())},
    }


def state_elements(state: Mapping[str, torch.Tensor]) -> int:
    if not state or any(not tensor.is_floating_point() for tensor in state.values()):
        raise ValueError("native E97 state must contain floating tensors")
    return sum(int(state[name].numel()) for name in sorted(state))


def layout_identity(total_elements: int, *, payload_max: int) -> bytes:
    return encode_flat_layout(
        total_elements, source_dtype=DType.F32, payload_max=payload_max)[1]


def state_digest(state: Mapping[str, torch.Tensor]) -> bytes:
    """Hash the exact sorted trainer base without a second model allocation."""
    digest = hashlib.sha256(b"emender-native-e97-base-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "little")); digest.update(encoded_name)
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        # NumPy has no stable bfloat16 dtype across the approved runtimes.
        # Reinterpret the already-contiguous CPU storage as bytes so the
        # digest covers the exact source bits without dtype conversion or a
        # second model-sized allocation.
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.digest()


def _crc32c(data: bytes | bytearray | memoryview) -> int:
    value = 0xFFFFFFFF
    for byte in data:
        value ^= int(byte)
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value ^ 0xFFFFFFFF


def _hash_fd(fd: int, *, offset: int, length: int) -> bytes:
    digest = hashlib.sha256()
    consumed = 0
    while consumed < length:
        chunk = os.pread(fd, min(1 << 20, length - consumed), offset + consumed)
        if not chunk:
            raise ValueError("native frame payload ended before its admitted extent")
        digest.update(chunk); consumed += len(chunk)
    return digest.digest()


def fd_sha256(fd: int, *, length: int) -> bytes:
    """Hash one exact memfd extent through a read-only zero-copy mapping."""
    if fd < 0 or length <= 0 or os.fstat(fd).st_size != length:
        raise ValueError("native digest extent does not match its memfd")
    with mmap.mmap(fd, length, flags=mmap.MAP_PRIVATE,
                   prot=mmap.PROT_READ) as mapping:
        return hashlib.sha256(mapping).digest()


def encode_owner_frame_fd(*, source_fd: int, payload_offset: int, payload_bytes: int,
                          payload_max: int, run_id: str, fence_epoch: int,
                          generation: int, attempt: int, owner_epoch: int,
                          worker_id: str, incarnation: str, layout_digest: bytes,
                          base_digest: bytes, result_root: bytes, weight: int,
                          chunk_index: int, chunk_count: int,
                          deadline_unix_ns: int,
                          message_seq: int | None = None,
                          source_offset: int | None = None) -> tuple[int, int]:
    """Encode one normative native result-data frame around a memfd slice."""
    local_offset = payload_offset if source_offset is None else int(source_offset)
    if (source_fd < 0 or payload_bytes <= 0 or payload_bytes > payload_max
            or payload_offset < 0 or local_offset < 0 or weight <= 0
            or generation < 0 or generation >= (1 << 32)
            or chunk_index not in range(chunk_count) or chunk_count <= 0):
        raise ValueError("native owner frame bounds are invalid")
    if os.fstat(source_fd).st_size < local_offset + payload_bytes:
        raise ValueError("native owner payload slice exceeds its memfd")
    layout = bytes(layout_digest); base = bytes(base_digest); root = bytes(result_root)
    if any(len(item) != 32 for item in (layout, base, root)):
        raise ValueError("native owner frame digest width is invalid")
    payload_digest = _hash_fd(
        source_fd, offset=local_offset, length=payload_bytes)
    sequence = (((generation + 1) << 32) | (chunk_index + 1)
                if message_seq is None else int(message_seq))
    if sequence <= 0:
        raise ValueError("native owner frame message sequence is invalid")
    run_key = hashlib.sha256(run_id.encode()).digest()[:16]
    worker_key = hashlib.sha256(worker_id.encode()).digest()[:16]
    boot_key = hashlib.sha256(incarnation.encode()).digest()[:16]
    header = bytearray()
    header.extend(b"EMNDP1\0\0")
    header.extend(struct.pack("<HHHHII", 1, 0, 8, 0, 320, 0))  # result_data
    header.extend(run_key)
    header.extend(struct.pack("<QQIIQQ", fence_epoch, generation, attempt,
                              chunk_index, owner_epoch, generation))
    header.extend(worker_key); header.extend(boot_key)
    header.extend(layout); header.extend(base)
    header.extend(payload_digest); header.extend(root)
    header.extend(struct.pack(
        "<QQQQQQQIIII", payload_offset, payload_bytes, payload_bytes, weight,
        sequence, deadline_unix_ns, 0, chunk_index, chunk_count, 0, 0))
    if len(header) != 312:
        raise AssertionError(f"native owner header prefix is {len(header)}, expected 312")
    header.extend(struct.pack("<II", _crc32c(header), 0))
    frame_bytes = 320 + payload_bytes
    fd = create_memfd("emender-ndp-owner-frame", allow_sealing=True)
    try:
        os.ftruncate(fd, frame_bytes)
        os.pwrite(fd, header, 0)
        copy_fd_range(
            source_fd, fd, payload_bytes, source_offset=local_offset,
            destination_offset=320)
        seal_memfd(fd)
        return fd, frame_bytes
    except BaseException:
        os.close(fd); raise


def encode_credit_frame_fd(*, payload_offset: int, payload_bytes: int,
                           payload_max: int, run_id: str, fence_epoch: int,
                           generation: int, attempt: int, owner_epoch: int,
                           worker_id: str, incarnation: str,
                           layout_digest: bytes, base_digest: bytes,
                           permitted_root: bytes, weight: int,
                           chunk_index: int, chunk_count: int,
                           deadline_unix_ns: int, message_seq: int) -> int:
    """Grant one exact frozen result chunk independently of CQ completion."""
    if (payload_bytes <= 0 or payload_bytes > payload_max or payload_offset < 0
            or generation < 0 or generation >= (1 << 32) or weight <= 0
            or message_seq <= 0 or chunk_index not in range(chunk_count)
            or chunk_count <= 0):
        raise ValueError("native credit frame bounds are invalid")
    layout, base, root = (bytes(layout_digest), bytes(base_digest),
                          bytes(permitted_root))
    if any(len(item) != 32 for item in (layout, base, root)) or root == bytes(32):
        raise ValueError("native credit frame digest width is invalid")
    header = bytearray(b"EMNDP1\0\0")
    header.extend(struct.pack("<HHHHII", 1, 0, 3, 0, 320, 0))
    header.extend(hashlib.sha256(run_id.encode()).digest()[:16])
    header.extend(struct.pack("<QQIIQQ", fence_epoch, generation, attempt,
                              chunk_index, owner_epoch, generation))
    header.extend(hashlib.sha256(worker_id.encode()).digest()[:16])
    header.extend(hashlib.sha256(incarnation.encode()).digest()[:16])
    header.extend(layout); header.extend(base)
    header.extend(bytes(32)); header.extend(root)
    header.extend(struct.pack(
        "<QQQQQQQIIII", payload_offset, payload_bytes, payload_bytes, weight,
        message_seq, deadline_unix_ns, payload_bytes,
        chunk_index, chunk_count, 0, 0))
    if len(header) != 312:
        raise AssertionError(f"native credit header prefix is {len(header)}, expected 312")
    header.extend(struct.pack("<II", _crc32c(header), 0))
    fd = create_memfd("emender-ndp-owner-credit", allow_sealing=True)
    try:
        os.ftruncate(fd, 320); os.pwrite(fd, header, 0); seal_memfd(fd)
        return fd
    except BaseException:
        os.close(fd); raise


def decode_credit_frame_fd(fd: int, *, payload_max: int,
                           expected: Mapping[str, object]) -> dict[str, object]:
    """Validate one exact byte-credit grant before sending its result chunk."""
    if fd < 0 or os.fstat(fd).st_size < 320:
        raise ValueError("received native credit frame extent is invalid")
    header = os.pread(fd, 320, 0)
    if len(header) != 320 or header[:8] != b"EMNDP1\0\0":
        raise ValueError("received native credit frame magic is invalid")
    if struct.unpack_from("<HHHHII", header, 8) != (1, 0, 3, 0, 320, 0):
        raise ValueError("received native credit frame version/type is invalid")
    if _crc32c(header[:312]) != struct.unpack_from("<I", header, 312)[0]:
        raise ValueError("received native credit frame checksum mismatch")
    cursor = 24
    run_key = header[cursor:cursor + 16]; cursor += 16
    fence, generation, attempt, shard, owner_epoch, sequence = struct.unpack_from(
        "<QQIIQQ", header, cursor); cursor += 40
    worker_key, boot_key = header[cursor:cursor + 16], header[cursor + 16:cursor + 32]
    cursor += 32
    layout, base = header[cursor:cursor + 32], header[cursor + 32:cursor + 64]
    cursor += 64
    payload_digest, result_root = header[cursor:cursor + 32], header[cursor + 32:cursor + 64]
    cursor += 64
    (payload_offset, payload_bytes, shard_bytes, weight, message_seq,
     deadline_unix_ns, credit, chunk_index, chunk_count, status, reason) = \
        struct.unpack_from("<QQQQQQQIIII", header, cursor)
    actual = {
        "run_key": run_key, "fence_epoch": fence, "generation": generation,
        "attempt": attempt, "shard_id": shard, "owner_epoch": owner_epoch,
        "sequence": sequence, "worker_key": worker_key, "incarnation": boot_key,
        "layout_digest": layout, "base_digest": base, "result_root": result_root,
        "payload_offset": payload_offset, "payload_bytes": payload_bytes,
        "shard_bytes": shard_bytes, "weight": weight, "message_seq": message_seq,
        "deadline_unix_ns": deadline_unix_ns, "credit": credit,
        "chunk_index": chunk_index, "chunk_count": chunk_count,
        "status": status, "reason": reason,
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("received native credit identity/fence mismatch")
    if (payload_digest != bytes(32) or result_root == bytes(32)
            or payload_bytes <= 0 or payload_bytes > payload_max
            or shard_bytes != payload_bytes or credit != payload_bytes
            or chunk_index not in range(chunk_count) or status != 0 or reason != 0):
        raise ValueError("received native credit bounds are invalid")
    return actual


def decode_owner_frame_fd(fd: int, *, frame_bytes: int, payload_max: int,
                          expected: Mapping[str, object]) -> dict[str, object]:
    """Independently validate a frame already authenticated by native fabric."""
    if fd < 0 or not 320 < frame_bytes <= payload_max + 320:
        raise ValueError("received native owner frame extent is invalid")
    header = os.pread(fd, 320, 0)
    if len(header) != 320 or header[:8] != b"EMNDP1\0\0":
        raise ValueError("received native owner frame magic is invalid")
    if struct.unpack_from("<HHHHII", header, 8) != (1, 0, 8, 0, 320, 0):
        raise ValueError("received native owner frame version/type is invalid")
    if _crc32c(header[:312]) != struct.unpack_from("<I", header, 312)[0]:
        raise ValueError("received native owner frame header checksum mismatch")
    cursor = 24
    run_key = header[cursor:cursor + 16]; cursor += 16
    fence, generation, attempt, shard, owner_epoch, sequence = struct.unpack_from(
        "<QQIIQQ", header, cursor); cursor += 40
    worker_key, boot_key = header[cursor:cursor + 16], header[cursor + 16:cursor + 32]
    cursor += 32
    layout, base = header[cursor:cursor + 32], header[cursor + 32:cursor + 64]
    cursor += 64
    payload_digest, result_root = header[cursor:cursor + 32], header[cursor + 32:cursor + 64]
    cursor += 64
    (payload_offset, payload_bytes, shard_bytes, weight, message_seq,
     deadline_unix_ns, credit, chunk_index, chunk_count, status, reason) = \
        struct.unpack_from("<QQQQQQQIIII", header, cursor)
    actual = {
        "run_key": run_key, "fence_epoch": fence, "generation": generation,
        "attempt": attempt, "shard_id": shard, "owner_epoch": owner_epoch,
        "sequence": sequence, "worker_key": worker_key, "incarnation": boot_key,
        "layout_digest": layout, "base_digest": base, "result_root": result_root,
        "payload_offset": payload_offset, "payload_bytes": payload_bytes,
        "shard_bytes": shard_bytes, "weight": weight, "message_seq": message_seq,
        "deadline_unix_ns": deadline_unix_ns, "credit": credit,
        "chunk_index": chunk_index, "chunk_count": chunk_count,
        "status": status, "reason": reason,
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("received native owner frame identity/fence mismatch")
    if (payload_bytes <= 0 or payload_bytes > payload_max
            or frame_bytes != 320 + payload_bytes or shard_bytes != payload_bytes
            or credit != 0 or status != 0 or reason != 0
            or chunk_index not in range(chunk_count)):
        raise ValueError("received native owner frame payload bounds are invalid")
    if _hash_fd(fd, offset=320, length=payload_bytes) != payload_digest:
        raise ValueError("received native owner frame payload checksum mismatch")
    return actual


@dataclass(frozen=True)
class GenerationMetadata:
    run_id: str
    fence_epoch: int
    generation: int
    attempt: int
    owner_epoch: int
    total_elements: int
    layout_digest: str
    base_digest: str
    plan_digest: str
    deadline_unix_ns: int
    runtime_digests: Mapping[str, object]
    policy_id: str = ""
    policy_digest: str = ""
    code_digest: str = ""
    base_global_version: int = 0
    local_window_start: int = 0
    local_window_end: int = 1

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "GenerationMetadata":
        schema = value.get("schema")
        if schema not in {SCHEMA, ASYNC_V2_SCHEMA}:
            raise ValueError("native E97 generation schema mismatch")
        integer_fields = ("fence_epoch", "generation", "attempt", "owner_epoch",
                          "total_elements", "deadline_unix_ns")
        if any(isinstance(value.get(field), bool)
               or not isinstance(value.get(field), int) for field in integer_fields):
            raise ValueError("native E97 generation identity is invalid")
        result = cls(
            str(value["run_id"]), int(value["fence_epoch"]),
            int(value["generation"]), int(value["attempt"]),
            int(value["owner_epoch"]), int(value["total_elements"]),
            str(value["layout_digest"]), str(value["base_digest"]),
            str(value["plan_digest"]), int(value["deadline_unix_ns"]),
            dict(value["runtime_digests"]),
            str(value.get("policy_id", "")),
            str(value.get("policy_digest", "")),
            str(value.get("code_digest", "")),
            int(value.get("base_global_version", value["generation"])),
            int(value.get("local_window_start", value["generation"])),
            int(value.get("local_window_end", int(value["generation"]) + 1)),
        )
        if (not result.run_id or result.fence_epoch <= 0 or result.generation < 0
                or result.attempt <= 0 or result.owner_epoch <= 0
                or result.total_elements <= 0 or result.deadline_unix_ns <= time.time_ns()
                or not isinstance(value.get("runtime_digests"), Mapping)
                or any(len(item) != 64 or any(character not in "0123456789abcdef"
                                              for character in item)
                       for item in (result.layout_digest, result.base_digest,
                                    result.plan_digest))):
            raise ValueError("native E97 generation identity is invalid")
        if result.policy_id:
            if (schema != ASYNC_V2_SCHEMA
                    or result.policy_id != "async-decoupled-v2.0-exp"
                    or len(result.policy_digest) != 64
                    or len(result.code_digest) != 64
                    or result.base_global_version < 0
                    or result.local_window_start < 0
                    or result.local_window_end <= result.local_window_start
                    or result.local_window_end - result.local_window_start > 8):
                raise ValueError("native E97 async-v2 metadata identity is invalid")
        elif schema != SCHEMA:
            raise ValueError("native E97 v2 schema requires explicit policy identity")
        return result

    def as_json(self) -> dict[str, object]:
        return {"schema": (ASYNC_V2_SCHEMA if self.policy_id else SCHEMA),
                **self.__dict__,
                "runtime_digests": dict(self.runtime_digests)}


class NativeTrainerDataPlane:
    """One GPU trainer's direct producer connection to the node service."""

    def __init__(self, client: Client, metadata: GenerationMetadata, *, rank: int,
                 identity: str, incarnation: str, control_root: Path):
        self.client, self.metadata = client, metadata
        self.rank, self.identity, self.incarnation = int(rank), identity, incarnation
        self.control_root = control_root
        self.buffer: Buffer | None = None
        self.submission: Operation | None = None

    @classmethod
    def connect(cls, *, build_manifest: str | Path, socket_path: str,
                run_id: str, fence_epoch: int, generation: int, rank: int,
                identity: str, incarnation: str, control_root: str | Path,
                deadline: float) -> "NativeTrainerDataPlane":
        root = Path(control_root)
        value = wait_metadata(
            root / f"native-generation-{generation:08d}.json", deadline=deadline,
            expected={"run_id": run_id, "fence_epoch": fence_epoch,
                      "generation": generation})
        metadata = GenerationMetadata.from_json(value)
        client = Client.open(
            library=NativeLibrary(artifact_path(build_manifest, "local_library")),
            role=Role.TRAINER, run_key=run_id, fence_epoch=fence_epoch,
            worker_key=identity, incarnation=incarnation, socket_path=socket_path,
            deadline_s=max(.001, min(10.0, deadline - time.monotonic())))
        client.attach_generation(
            total_elements=metadata.total_elements,
            layout_digest=bytes.fromhex(metadata.layout_digest),
            generation=generation, attempt=metadata.attempt,
            owner_epoch=metadata.owner_epoch, source_dtype=DType.F32,
            base_digest=bytes.fromhex(metadata.base_digest),
            plan_digest=bytes.fromhex(metadata.plan_digest),
            deadline_s=max(.001, (metadata.deadline_unix_ns - time.time_ns()) / 1e9),
            deadline_unix_ns=metadata.deadline_unix_ns)
        return cls(client, metadata, rank=rank, identity=identity,
                   incarnation=incarnation, control_root=root)

    def allocate_delta(self, *, deadline_s: float) -> Buffer:
        if self.buffer is not None:
            raise RuntimeError("native trainer already owns a generation buffer")
        self.buffer = self.client.allocate(
            elements=self.metadata.total_elements, dtype=DType.F32,
            deadline_s=deadline_s)
        return self.buffer

    def publish_model_delta(self, base_state: Mapping[str, torch.Tensor], model,
                            tokens: int, *, chunk_elements: int,
                            deadline_s: float,
                            aggregation_weight: int | None = None,
                            contribution_identity: Mapping[str, object] | None = None,
                            ) -> dict[str, object]:
        """Fill final service-owned f32 storage without a trainer-sized spool."""
        if self.buffer is None or tokens <= 0 or chunk_elements <= 0:
            raise RuntimeError("native trainer delta buffer/tokens are not ready")
        worker_state = model.state_dict()
        cursor = 0
        with self.buffer.mapped(DType.F32, write=True) as target:
            for name in sorted(base_state):
                worker = worker_state[name].detach().reshape(-1)
                base = base_state[name].detach().reshape(-1)
                if worker.numel() != base.numel():
                    raise ValueError(f"trainer state layout changed for {name}")
                for offset in range(0, worker.numel(), chunk_elements):
                    end = min(offset + chunk_elements, worker.numel())
                    # This is the established K40 delta contract: cast the
                    # trained tensor to base precision, subtract the unchanged
                    # base, then project exactly once to f32 wire storage.
                    piece = worker[offset:end].to(
                        device="cpu", dtype=base_state[name].dtype).sub(base[offset:end])
                    encoded = piece.to(torch.float32).contiguous().numpy()
                    target[cursor:cursor + encoded.size] = encoded
                    cursor += encoded.size
        if cursor != self.metadata.total_elements:
            raise ValueError("native trainer wrote an incomplete flat layout")
        return self._seal_submit(
            tokens=tokens, aggregation_weight=aggregation_weight,
            contribution_identity=contribution_identity,
            deadline_s=deadline_s)

    def publish_state_delta(
        self,
        base_state: Mapping[str, torch.Tensor],
        endpoint_state: Mapping[str, torch.Tensor],
        tokens: int,
        *,
        chunk_elements: int,
        deadline_s: float,
        aggregation_weight: int | None = None,
        contribution_identity: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Seal a post-K endpoint after a prior result was applied at boundary.

        The v2 trainer deliberately retains its mutable interval endpoint until
        the preceding immutable descriptor is released.  This variant streams
        that endpoint directly into the same service-owned memfd and therefore
        does not require rebuilding a model or materializing a Python delta.
        """
        if self.buffer is None or tokens <= 0 or chunk_elements <= 0:
            raise RuntimeError("native trainer endpoint/tokens are not ready")
        if tuple(sorted(base_state)) != tuple(sorted(endpoint_state)):
            raise ValueError("native trainer endpoint layout changed")
        cursor = 0
        with self.buffer.mapped(DType.F32, write=True) as target:
            for name in sorted(base_state):
                endpoint = endpoint_state[name].detach().reshape(-1)
                base = base_state[name].detach().reshape(-1)
                if endpoint.numel() != base.numel():
                    raise ValueError(f"trainer state layout changed for {name}")
                for offset in range(0, endpoint.numel(), chunk_elements):
                    end = min(offset + chunk_elements, endpoint.numel())
                    piece = endpoint[offset:end].to(
                        device="cpu", dtype=base_state[name].dtype).sub(
                            base[offset:end])
                    encoded = piece.to(torch.float32).contiguous().numpy()
                    target[cursor:cursor + encoded.size] = encoded
                    cursor += encoded.size
        if cursor != self.metadata.total_elements:
            raise ValueError("native trainer wrote an incomplete flat layout")
        return self._seal_submit(
            tokens=tokens, aggregation_weight=aggregation_weight,
            contribution_identity=contribution_identity,
            deadline_s=deadline_s)

    def publish_flat_shards(self, shards, *, tokens: int, deadline_s: float,
                            aggregation_weight: int | None = None,
                            contribution_identity: Mapping[str, object] | None = None,
                            ) -> dict[str, object]:
        """Control-fixture companion using the identical direct memfd admission."""
        if self.buffer is None or tokens <= 0:
            raise RuntimeError("native trainer delta buffer/tokens are not ready")
        cursor = 0
        with self.buffer.mapped(DType.F32, write=True) as target:
            for shard in shards:
                encoded = shard.detach().cpu().to(torch.float32).reshape(-1).numpy()
                target[cursor:cursor + encoded.size] = encoded
                cursor += encoded.size
        if cursor != self.metadata.total_elements:
            raise ValueError("native trainer wrote an incomplete flat layout")
        return self._seal_submit(
            tokens=tokens, aggregation_weight=aggregation_weight,
            contribution_identity=contribution_identity,
            deadline_s=deadline_s)

    def _seal_submit(self, *, tokens: int, deadline_s: float,
                     aggregation_weight: int | None = None,
                     contribution_identity: Mapping[str, object] | None = None,
                     ) -> dict[str, object]:
        assert self.buffer is not None
        weight = int(tokens if aggregation_weight is None else aggregation_weight)
        if tokens <= 0 or weight <= 0:
            raise ValueError("native exact tokens and aggregation weight must be positive")
        identity = dict(contribution_identity or {})
        if identity:
            required = {
                "policy_id", "policy_digest", "code_digest",
                "base_global_version", "base_global_digest",
                "base_lag_at_seal", "local_window_start", "local_window_end",
                "window_count", "contribution_sequence",
            }
            if not required <= identity.keys():
                raise ValueError("native async-v2 contribution identity is incomplete")
            lag = int(identity["base_lag_at_seal"])
            if (identity["policy_id"] != "async-decoupled-v2.0-exp"
                    or not 0 <= lag <= 6
                    or weight != int(tokens) * (7 - lag)
                    or int(identity["window_count"])
                    != int(identity["local_window_end"])
                    - int(identity["local_window_start"])):
                raise ValueError("native async-v2 token/lag/window identity is invalid")
        digest = self.buffer.sha256()
        if identity:
            identity.setdefault("payload_digest", digest.hex())
            # The endpoint identity is reconstructable from the verified base
            # and immutable cumulative delta without a second full-model read.
            identity.setdefault(
                "interval_endpoint_digest",
                hashlib.sha256(
                    bytes.fromhex(self.metadata.base_digest) + digest).hexdigest())
        self.buffer.seal()
        owned_started = time.monotonic()
        self.submission = self.client.submit(
            self.buffer, trainer_key=self.identity,
            trainer_incarnation=self.incarnation,
            submission_seq=self.rank, weight=weight, source_dtype=DType.F32,
            source_sha256=digest, deadline_s=deadline_s)
        # Releasing the producer's public handle is safe: the persistent service
        # retained the immutable memfd before acknowledging submission.
        self.buffer.close(); self.buffer = None
        marker = {
            "schema": (
                "emender-native-e97-submission-v2"
                if identity else "emender-native-e97-submission-v1"),
            "run_id": self.metadata.run_id,
            "fence_epoch": self.metadata.fence_epoch,
            "generation": self.metadata.generation,
            "attempt": self.metadata.attempt, "rank": self.rank,
            "trainer": self.identity, "incarnation": self.incarnation,
            "submission_seq": self.rank, "tokens": int(tokens),
            "aggregation_weight": weight,
            "owned_ack_seconds": time.monotonic() - owned_started,
            "source_sha256": digest.hex(),
            "layout_digest": self.metadata.layout_digest,
            "dense_files_written": 0, "trainer_spool_bytes": 0,
            **identity,
        }
        if identity:
            descriptor_identity = {
                "run_id": marker["run_id"],
                "fence_epoch": marker["fence_epoch"],
                "trainer": marker["trainer"],
                "incarnation": marker["incarnation"],
                "contribution_sequence": marker["contribution_sequence"],
                "local_window_start": marker["local_window_start"],
                "local_window_end": marker["local_window_end"],
                "window_count": marker["window_count"],
                "base_global_version": marker["base_global_version"],
                "base_global_digest": marker["base_global_digest"],
                "policy_digest": marker["policy_digest"],
                "code_digest": marker["code_digest"],
                "exact_tokens": marker["tokens"],
                "base_lag_at_seal": marker["base_lag_at_seal"],
                "payload_digest": marker["payload_digest"],
            }
            marker["descriptor_digest"] = hashlib.sha256(json.dumps(
                descriptor_identity, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
        atomic_metadata(
            self.control_root /
            f"native-submit-{self.metadata.generation:08d}-{self.rank:02d}.json",
            marker)
        return marker

    @contextmanager
    def result_shards(self, *, deadline: float, chunk_elements: int
                      ) -> Iterator[tuple[dict[str, object], Iterator[torch.Tensor]]]:
        value = wait_metadata(
            self.control_root /
            f"native-result-{self.metadata.generation:08d}.json", deadline=deadline,
            expected={"run_id": self.metadata.run_id,
                      "fence_epoch": self.metadata.fence_epoch,
                      "generation": self.metadata.generation})
        attempt = int(value.get("attempt", 0))
        owner_epoch = int(value.get("owner_epoch", 0))
        deadline_unix_ns = int(value.get("deadline_unix_ns", 0))
        source_dtype = DType(int(value.get("source_dtype", 0)))
        if (value.get("layout_digest") != self.metadata.layout_digest
                or value.get("base_digest") != self.metadata.base_digest
                or value.get("plan_digest") != self.metadata.plan_digest
                or attempt < self.metadata.attempt or owner_epoch <= 0
                or deadline_unix_ns <= time.time_ns()
                or int(value.get("global_weight", 0)) <= 0
                or len(str(value.get("result_root", ""))) != 64):
            raise ValueError("native result marker identity/root is invalid")
        if self.metadata.policy_id and (
                value.get("policy_id") != self.metadata.policy_id
                or value.get("policy_digest") != self.metadata.policy_digest
                or int(value.get("commit_global_version", -1))
                != self.metadata.base_global_version
                or not 0 <= int(value.get("commit_lag", -1)) <= 6
                or int(value.get("base_global_version", -1))
                != int(value.get("commit_global_version", -1))
                   - int(value.get("commit_lag", -1))
                or int(value.get("exact_tokens", 0)) <= 0
                or int(value.get("aggregation_weight", 0))
                != int(value.get("exact_tokens", 0))
                   * (7 - int(value.get("commit_lag", -1)))):
            raise ValueError("native async-v2 result identity/weight is invalid")
        if self.metadata.policy_id:
            accepted = value.get("accepted_local_contributions")
            if not isinstance(accepted, list) or not accepted:
                raise ValueError(
                    "native async-v2 result lacks accepted correction identities")
            ranks: set[int] = set()
            for record in accepted:
                if not isinstance(record, Mapping):
                    raise ValueError(
                        "native async-v2 accepted correction identity is invalid")
                accepted_rank = int(record.get("rank", -1))
                window_start = int(record.get("local_window_start", -1))
                window_end = int(record.get("local_window_end", -1))
                if (accepted_rank < 0 or accepted_rank in ranks
                        or window_start < 0 or window_end <= window_start
                        or int(record.get("window_count", -1))
                           != window_end - window_start
                        or int(record.get("base_global_version", -1)) < 0
                        or any(
                            len(str(record.get(name, ""))) != 64
                            or any(
                                character not in "0123456789abcdef"
                                for character in str(record.get(name, "")))
                            for name in (
                                "payload_digest", "descriptor_digest"))):
                    raise ValueError(
                        "native async-v2 accepted correction identity is invalid")
                ranks.add(accepted_rank)
        # The two-node owner plane replaces local attempt 1 with the exact
        # global attempt 2.  Refresh the persistent trainer connection from
        # the controller-published fenced marker before requesting its view;
        # otherwise the server correctly rejects the stale attempt-1 header.
        self.client.refresh_generation(
            total_elements=self.metadata.total_elements,
            layout_digest=bytes.fromhex(self.metadata.layout_digest),
            generation=self.metadata.generation, attempt=attempt,
            owner_epoch=owner_epoch, source_dtype=source_dtype,
            deadline_s=max(.001, deadline - time.monotonic()),
            deadline_unix_ns=deadline_unix_ns,
            base_digest=bytes.fromhex(self.metadata.base_digest),
            plan_digest=bytes.fromhex(self.metadata.plan_digest))
        view = self.client.result_view_handle(int(value["operation_handle"]))
        try:
            if (view.fence_epoch != self.metadata.fence_epoch
                    or view.generation != self.metadata.generation
                    or view.attempt != attempt
                    or view.layout_digest.hex() != self.metadata.layout_digest
                    or view.base_digest.hex() != self.metadata.base_digest
                    or view.result_root.hex() != value["result_root"]
                    or view.global_weight != int(value["global_weight"])
                    or view.dtype is not DType.F32):
                raise ValueError("native shared result differs from fenced marker")

            def shards() -> Iterator[torch.Tensor]:
                with view.mapped(DType.F32) as flat:
                    for offset in range(0, flat.size, chunk_elements):
                        # A bounded clone decouples torch from the read-only mmap
                        # before the next slice or independent trainer closes it.
                        yield torch.from_numpy(
                            np.array(flat[offset:offset + chunk_elements], copy=True))

            yield value, shards()
        finally:
            view.close()

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.close(); self.buffer = None
        if self.submission is not None:
            self.submission.close(); self.submission = None
        self.client.close()


def exact_weighted_reference(contributions: Sequence[np.ndarray],
                             weights: Sequence[int]) -> np.ndarray:
    """Independent rank-ordered reference for K40/native parity tests."""
    if (not contributions or len(contributions) != len(weights)
            or any(weight <= 0 for weight in weights)):
        raise ValueError("reference contributions and positive weights are required")
    shape = contributions[0].shape
    if any(value.shape != shape or value.dtype != np.dtype("float32")
           or not np.isfinite(value).all() for value in contributions):
        raise ValueError("reference inputs must be finite shape-identical f32 arrays")
    weighted = contributions[0].astype(np.float64) * int(weights[0])
    for value, weight in zip(contributions[1:], weights[1:]):
        weighted += value.astype(np.float64) * int(weight)
    weighted /= sum(int(weight) for weight in weights)
    return weighted.astype(np.float32)


__all__ = [
    "ASYNC_V2_SCHEMA", "GenerationMetadata", "NativeTrainerDataPlane",
    "SCHEMA", "artifact_path",
    "atomic_metadata", "decode_credit_frame_fd", "decode_owner_frame_fd",
    "encode_credit_frame_fd", "encode_owner_frame_fd",
    "exact_weighted_reference", "fd_sha256", "layout_identity", "runtime_digests",
    "state_digest", "state_elements", "wait_metadata",
]
