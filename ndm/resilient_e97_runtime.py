"""End-to-end split-role generation and checkpoint protocol for resilient E97.

This module intentionally imports no model implementation.  Trainers own tensor
state and serialize checkpoints; managers only validate float64 shards and small
JSON manifests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable, Iterator, Mapping, Sequence

import torch

from ndm.resilient_e97_roles import CpuNodeManager, LocalFence, LocalTrainerSpool
from ndm.fenced_admission import AllocationLease, SQLiteFencedControlStore

PINNED_STEP_1525000_SHA256 = "1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9"


def assert_node_local_path(path: str | Path, shared_root: str | Path) -> Path:
    """Reject project-shared or Lustre-backed live-path configuration."""
    target, shared = Path(path).resolve(), Path(shared_root).resolve()
    if target == shared or target.is_relative_to(shared):
        raise ValueError("live resilient path must not be inside the shared run directory")
    mounts = []
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 3:
                mount = Path(fields[1].replace("\\040", " ")).resolve()
                if target == mount or target.is_relative_to(mount):
                    mounts.append((len(str(mount)), fields[2]))
    except OSError as error:
        raise RuntimeError(f"cannot verify node-local mount: {error}") from error
    if not mounts:
        raise ValueError("live resilient path has no verifiable mount")
    if max(mounts)[1].lower() in {"lustre", "gpfs", "nfs", "nfs4", "cifs"}:
        raise ValueError("live resilient path is on a shared filesystem")
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def heartbeat(run_dir: Path, identity: str, *, generation: int, step: int,
              loss: float | None, stage: str) -> None:
    atomic_json(run_dir / "supervision" / f"{identity}.json", {
        "identity": identity, "heartbeat_time": time.time(), "progress_time": time.time(),
        "generation": generation, "step": step, "loss": loss, "stage": stage,
    })


def tensor_layout(state: Mapping[str, torch.Tensor]) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    return tuple((name, tuple(state[name].shape), state[name].numel()) for name in sorted(state))


def flatten_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], *,
                  chunk_elements: int = 131072) -> tuple[torch.Tensor, ...]:
    if tensor_layout(before) != tensor_layout(after):
        raise ValueError("trainer state layout changed during local generation")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    return flatten_tensors(
        {name: after[name].detach() - before[name].detach() for name in before},
        chunk_elements=chunk_elements)


def flatten_tensors(tensors: Mapping[str, torch.Tensor], *,
                    chunk_elements: int = 131072) -> Iterator[torch.Tensor]:
    """Serialize a tensor mapping without a whole-model flatten/cat allocation.

    Each returned model-precision CPU shard is independently bounded.  Parameter
    boundaries are deliberately not encoded in the wire layout: ``apply_delta``
    consumes the same deterministic sorted element stream.
    """
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    pending = []
    pending_elements = 0

    for name in sorted(tensors):
        flat = tensors[name].detach().reshape(-1)
        offset = 0
        while offset < flat.numel():
            take = min(chunk_elements - pending_elements, flat.numel() - offset)
            pending.append(flat[offset:offset + take].to(device="cpu").clone())
            pending_elements += take
            offset += take
            if pending_elements == chunk_elements:
                yield torch.cat(pending) if len(pending) > 1 else pending[0]
                pending = []
                pending_elements = 0
    if pending:
        yield torch.cat(pending) if len(pending) > 1 else pending[0]


def apply_delta(base: Mapping[str, torch.Tensor], shards: Sequence[torch.Tensor], *,
                eta_outer: float) -> dict[str, torch.Tensor]:
    if sum(shard.numel() for shard in shards) != sum(value.numel() for value in base.values()):
        raise ValueError("aggregate shard count does not match trainer state")
    result = {}
    shard_index = shard_offset = 0
    for name, value in sorted(base.items()):
        updated = value.clone().reshape(-1)
        value_offset = 0
        while value_offset < value.numel():
            shard = shards[shard_index].reshape(-1)
            take = min(value.numel() - value_offset, shard.numel() - shard_offset)
            piece = shard[shard_offset:shard_offset + take]
            if not torch.isfinite(piece).all():
                raise ValueError(f"aggregate shard layout invalid for {name}")
            updated[value_offset:value_offset + take].add_(
                piece.to(updated), alpha=float(eta_outer))
            value_offset += take
            shard_offset += take
            if shard_offset == shard.numel():
                shard_index += 1
                shard_offset = 0
        result[name] = updated.reshape(value.shape)
    return result


class SplitManagerLoop:
    """Model-free local manager which releases contributions after publication."""
    def __init__(self, spool: LocalTrainerSpool, *, quorum: int, source_id: str,
                 aggregation_deadline_s: float):
        self.spool, self.source_id = spool, source_id
        self.manager = CpuNodeManager(spool, quorum=quorum)
        self.aggregation_deadline_s = float(aggregation_deadline_s)

    def generation(self, fence: LocalFence) -> dict[str, object]:
        members, weight, shards = self.manager.collect(
            fence, deadline=time.monotonic() + self.aggregation_deadline_s,
            expected_source_id=self.source_id)
        path = self.spool.publish_aggregate(fence, members, shards, weight=weight,
                                            source_id=self.source_id,
                                            accepted_peers=("node-0",))
        for trainer_id in members:
            self.spool.release_trainer(fence, trainer_id)
        return {"members": members, "weight": weight, "manifest": str(path)}


def outer_state_migration(seed: Mapping[str, object], *, policy: str,
                          approved_config: Mapping[str, object] | None = None) -> dict[str, object]:
    """Restore new-harness outer state or explicitly initialize the pinned cold start."""
    outer = seed.get("outer_update_state")
    if outer is not None:
        return {"status": "restored", "state": outer, "policy": "checkpoint"}
    if policy != "initialize-from-approved-config":
        raise ValueError("cold-start seed has no outer state; approved initialization is required")
    if seed.get("sha256") != PINNED_STEP_1525000_SHA256 or int(seed.get("step", -1)) != 1525000:
        raise ValueError("outer initialization is restricted to the pinned step-1525000 seed")
    state = dict(approved_config or {})
    if not state:
        raise ValueError("approved outer-update configuration is required")
    return {"status": "initialized_not_restored", "state": state, "policy": policy,
            "source_step": 1525000, "source_generation": 0,
            "source_sha256": PINNED_STEP_1525000_SHA256}


def finalize_checkpoint(run_dir: str | Path, checkpoint: str | Path, *,
                        run_id: str, generation: int, step: int,
                        async_chain: Sequence[str], membership: Sequence[object],
                        fence: LocalFence, source_id: str, code_id: str,
                        outer_update_state: Mapping[str, object],
                        migration: Mapping[str, object],
                        accepted_tokens: int = 0,
                        generation_identity: Mapping[str, object] | None = None,
                        digests: Mapping[str, object] | None = None,
                        control_store: SQLiteFencedControlStore | None = None,
                        allocation_lease: AllocationLease | None = None) -> Path:
    """Validate a designated trainer checkpoint and immutably publish its manifest."""
    root, checkpoint = Path(run_dir), Path(checkpoint)
    if (control_store is None) != (allocation_lease is None):
        raise ValueError("fenced checkpoint publication requires store and lease together")
    if control_store is not None:
        if allocation_lease.run_id != run_id or allocation_lease.fence != fence.coordinator_epoch:
            raise ValueError("checkpoint allocation lease differs from generation fence")
        control_store.assert_current(allocation_lease)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("trainer checkpoint is missing or empty")
    loaded = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "outer_update_state", "step",
                "generation", "run_id", "source_id", "payload_id", "coordinator_epoch"}
    if (not required <= set(loaded) or int(loaded["step"]) != int(step)
            or int(loaded["generation"]) != int(generation)
            or loaded["outer_update_state"] != dict(outer_update_state)
            or loaded["run_id"] != run_id or loaded["source_id"] != source_id
            or loaded["payload_id"] != fence.payload_id
            or int(loaded["coordinator_epoch"]) != fence.coordinator_epoch):
        raise ValueError("checkpoint complete-state/fencing validation failed")
    del loaded
    raw_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    target_name = f"generation-{generation:08d}"
    if control_store is not None:
        # A superseded writer may finish an already-open filesystem write. A
        # fence-specific immutable name prevents that orphan from overwriting
        # the current allocation's handoff before the authoritative CAS.
        target_name += f"-fence-{allocation_lease.fence:08d}"
    target = root / "handoff" / f"{target_name}.json"
    outer_digest = hashlib.sha256(json.dumps(
        dict(outer_update_state), sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()
    payload = {
        "schema": 1, "run_id": run_id, "generation": generation, "step": step,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": raw_sha, "contains": ["model", "inner_optimizer"],
        "outer_update_state": dict(outer_update_state), "outer_state_migration": dict(migration),
        "async_chain": list(async_chain),
        "membership": sorted(membership, key=lambda item: json.dumps(item, sort_keys=True)),
        "fence": fence.__dict__, "source_id": source_id, "code_id": code_id,
        "payload_id": fence.payload_id, "finalized": True,
        "accepted_tokens": int(accepted_tokens),
        "generation_identity": dict(generation_identity or fence.__dict__),
        "digests": {"checkpoint_sha256": raw_sha, "outer_state_sha256": outer_digest,
                    **dict(digests or {})},
    }
    if target.exists():
        if json.loads(target.read_text()) != payload:
            raise FileExistsError("immutable handoff generation already exists")
    else:
        atomic_json(target, payload)
    manifest_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    latest = {
        "generation": generation, "manifest": str(target.resolve()),
        "manifest_sha256": manifest_sha, "fence": fence.coordinator_epoch,
        "accepted_tokens": int(accepted_tokens),
    }
    if control_store is not None:
        name = f"generation-{generation:08d}"
        commit = {
            "run_id": run_id, "generation": generation,
            "generation_identity": payload["generation_identity"],
            "membership": payload["membership"],
            "accepted_tokens": int(accepted_tokens),
            "outer_state_sha256": outer_digest,
            "checkpoint_sha256": raw_sha, "manifest_sha256": manifest_sha,
            "digests": payload["digests"], "fence": allocation_lease.fence,
        }
        control_store.publish_bundle(allocation_lease, (
            ("commit", name, commit),
            ("checkpoint", name, {"checkpoint_sha256": raw_sha,
                                  "manifest_sha256": manifest_sha,
                                  "generation": generation}),
            ("latest", "authoritative", latest),
        ))
        control_store.assert_current(allocation_lease)
    atomic_json(root / "handoff" / "latest.json", latest)
    if control_store is not None:
        control_store.assert_current(allocation_lease)
    return target
