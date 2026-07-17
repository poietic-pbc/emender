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
from typing import Mapping, Sequence

import torch

from ndm.resilient_e97_roles import CpuNodeManager, LocalFence, LocalTrainerSpool

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
    flat = torch.cat(tuple((after[name].detach().cpu().to(torch.float64) -
                            before[name].detach().cpu().to(torch.float64)).reshape(-1)
                           for name in sorted(before)))
    return tuple(flat[offset:offset + chunk_elements].clone()
                 for offset in range(0, flat.numel(), chunk_elements))


def apply_delta(base: Mapping[str, torch.Tensor], shards: Sequence[torch.Tensor], *,
                eta_outer: float) -> dict[str, torch.Tensor]:
    flat = torch.cat(tuple(shard.reshape(-1) for shard in shards)) if shards else torch.empty(0)
    if flat.numel() != sum(value.numel() for value in base.values()):
        raise ValueError("aggregate shard count does not match trainer state")
    result = {}
    offset = 0
    for name, value in sorted(base.items()):
        shard = flat[offset:offset + value.numel()]
        if shard.numel() != value.numel() or not torch.isfinite(shard).all():
            raise ValueError(f"aggregate shard layout invalid for {name}")
        result[name] = value + shard.reshape(value.shape).to(value) * float(eta_outer)
        offset += value.numel()
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
                                            source_id=self.source_id)
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
                        async_chain: Sequence[str], membership: Sequence[int],
                        fence: LocalFence, source_id: str, code_id: str,
                        outer_update_state: Mapping[str, object],
                        migration: Mapping[str, object]) -> Path:
    """Validate a designated trainer checkpoint and immutably publish its manifest."""
    root, checkpoint = Path(run_dir), Path(checkpoint)
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
    target = root / "handoff" / f"generation-{generation:08d}.json"
    if target.exists():
        raise FileExistsError("immutable handoff generation already exists")
    payload = {
        "schema": 1, "run_id": run_id, "generation": generation, "step": step,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": raw_sha, "contains": ["model", "inner_optimizer"],
        "outer_update_state": dict(outer_update_state), "outer_state_migration": dict(migration),
        "async_chain": list(async_chain), "membership": sorted(map(int, membership)),
        "fence": fence.__dict__, "source_id": source_id, "code_id": code_id,
        "payload_id": fence.payload_id, "finalized": True,
    }
    atomic_json(target, payload)
    atomic_json(root / "handoff" / "latest.json", {
        "generation": generation, "manifest": str(target.resolve()),
        "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return target
