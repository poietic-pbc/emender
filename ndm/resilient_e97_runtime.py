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
from ndm.manifest_peer_control import AllocationClaim, ManifestPeerAuthority


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
              loss: float | None, stage: str, **evidence: object) -> None:
    atomic_json(run_dir / "supervision" / f"{identity}.json", {
        "identity": identity, "heartbeat_time": time.time(), "progress_time": time.time(),
        "generation": generation, "step": step, "loss": loss, "stage": stage,
        "process_incarnation":
            os.environ.get("RESILIENT_E97_PROCESS_INCARNATION", ""),
        "node_incarnation":
            os.environ.get("RESILIENT_E97_NODE_INCARNATION", ""),
        "cohort_restart_sequence": int(
            os.environ.get("RESILIENT_E97_COHORT_RESTART_SEQUENCE", "0")),
        **evidence,
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


def apply_delta(base: Mapping[str, torch.Tensor], shards: Iterable[torch.Tensor], *,
                eta_outer: float, in_place: bool = False) -> dict[str, torch.Tensor]:
    """Apply a bounded shard iterable, optionally reusing trainer-owned state."""
    result = {}
    iterator = iter(shards)
    shard = None
    shard_offset = 0
    for name, value in sorted(base.items()):
        updated = (value if in_place else value.clone()).reshape(-1)
        value_offset = 0
        while value_offset < value.numel():
            if shard is None:
                try:
                    shard = next(iterator).reshape(-1)
                except StopIteration as error:
                    raise ValueError(
                        "aggregate shard count does not match trainer state") from error
                if shard.numel() <= 0:
                    raise ValueError("aggregate shard count does not match trainer state")
            take = min(value.numel() - value_offset, shard.numel() - shard_offset)
            piece = shard[shard_offset:shard_offset + take]
            if not torch.isfinite(piece).all():
                raise ValueError(f"aggregate shard layout invalid for {name}")
            updated[value_offset:value_offset + take].add_(
                piece.to(updated), alpha=float(eta_outer))
            value_offset += take
            shard_offset += take
            if shard_offset == shard.numel():
                shard = None
                shard_offset = 0
        result[name] = updated.reshape(value.shape)
    if shard is not None or next(iterator, None) is not None:
        raise ValueError("aggregate shard count does not match trainer state")
    return result


def apply_delta_with_correction_ledger(
        base: Mapping[str, torch.Tensor],
        shards: Iterable[torch.Tensor],
        *,
        eta_outer: float,
        interval_start: Mapping[str, torch.Tensor],
        interval_endpoint: Mapping[str, torch.Tensor],
        accepted_own_interval: bool,
        in_place: bool = False,
        ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply a global result and construct its safe-boundary correction.

    The correction is computed from the same bounded native result shards that
    advance the worker's global anchor:

    ``(S_h - S_a) - C_i(a,h)``.

    For the currently sealed local interval, ``C_i`` is the exact interval
    endpoint minus its mutable start only when the fenced accepted-set identity
    names that descriptor; it is zero for unaccepted work.  Constructing the
    global shift directly from the result shards avoids retaining another
    model-sized copy of ``S_a`` after the in-place anchor update.  The returned
    correction is the one bounded cohort consumed by the resident
    ScheduleFree model/``z`` translation at a verified K boundary.
    """
    names = tuple(sorted(base))
    if (names != tuple(sorted(interval_start))
            or names != tuple(sorted(interval_endpoint))):
        raise ValueError(
            "correction-ledger tensor layout differs from global state")
    if not isinstance(accepted_own_interval, bool):
        raise TypeError("accepted_own_interval must be a bool")

    result: dict[str, torch.Tensor] = {}
    corrections: dict[str, torch.Tensor] = {}
    iterator = iter(shards)
    shard = None
    shard_offset = 0
    for name in names:
        value = base[name]
        start = interval_start[name]
        endpoint = interval_endpoint[name]
        if (start.shape != value.shape or endpoint.shape != value.shape
                or not start.is_floating_point()
                or not endpoint.is_floating_point()):
            raise ValueError(
                f"correction-ledger interval layout invalid for {name}")
        updated = (value if in_place else value.clone()).reshape(-1)
        start_flat = start.detach().reshape(-1)
        endpoint_flat = endpoint.detach().reshape(-1)
        correction = torch.empty_like(updated)
        value_offset = 0
        while value_offset < value.numel():
            if shard is None:
                try:
                    shard = next(iterator).reshape(-1)
                except StopIteration as error:
                    raise ValueError(
                        "aggregate shard count does not match trainer state") from error
                if shard.numel() <= 0:
                    raise ValueError(
                        "aggregate shard count does not match trainer state")
            take = min(
                value.numel() - value_offset,
                shard.numel() - shard_offset,
            )
            aggregate_piece = shard[shard_offset:shard_offset + take]
            if not torch.isfinite(aggregate_piece).all():
                raise ValueError(f"aggregate shard layout invalid for {name}")
            target_piece = aggregate_piece.to(updated)
            correction_piece = correction[
                value_offset:value_offset + take]
            correction_piece.copy_(target_piece).mul_(float(eta_outer))
            if accepted_own_interval:
                local_delta = endpoint_flat[
                    value_offset:value_offset + take].to(updated).sub(
                        start_flat[
                            value_offset:value_offset + take].to(updated))
                if not torch.isfinite(local_delta).all():
                    raise ValueError(
                        f"correction-ledger local delta invalid for {name}")
                correction_piece.sub_(local_delta)
            updated[value_offset:value_offset + take].add_(
                target_piece, alpha=float(eta_outer))
            value_offset += take
            shard_offset += take
            if shard_offset == shard.numel():
                shard = None
                shard_offset = 0
        result[name] = updated.reshape(value.shape)
        corrections[name] = correction.reshape(value.shape)
    if shard is not None or next(iterator, None) is not None:
        raise ValueError("aggregate shard count does not match trainer state")
    return result, corrections


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


def outer_state_migration(
        seed: Mapping[str, object], *, policy: str,
        approved_config: Mapping[str, object] | None = None,
        approved_seed: Mapping[str, object] | None = None) -> dict[str, object]:
    """Restore new-harness outer state or explicitly initialize the pinned cold start."""
    outer = seed.get("outer_update_state")
    if outer is not None:
        return {"status": "restored", "state": outer, "policy": "checkpoint"}
    if policy != "initialize-from-approved-config":
        raise ValueError("cold-start seed has no outer state; approved initialization is required")
    if approved_seed is None:
        raise ValueError("cold-start outer initialization requires an approved seed identity")
    expected = dict(approved_seed)
    expected_step = int(expected.get("step", -1))
    expected_sha256 = expected.get("sha256")
    if (
        seed.get("sha256") != expected_sha256
        or int(seed.get("step", -1)) != expected_step
    ):
        raise ValueError(
            "outer initialization is restricted to the approved immutable seed")
    state = dict(approved_config or {})
    if not state:
        raise ValueError("approved outer-update configuration is required")
    return {"status": "initialized_not_restored", "state": state, "policy": policy,
            "source_step": expected_step, "source_generation": 0,
            "source_sha256": expected_sha256}


def finalize_checkpoint(run_dir: str | Path, checkpoint: str | Path, *,
                        run_id: str, generation: int, step: int,
                        async_chain: Sequence[str], membership: Sequence[object],
                        fence: LocalFence, source_id: str, code_id: str,
                        outer_update_state: Mapping[str, object],
                        migration: Mapping[str, object],
                        accepted_tokens: int = 0,
                        generation_identity: Mapping[str, object] | None = None,
                        digests: Mapping[str, object] | None = None,
                        peer_authority: ManifestPeerAuthority | None = None,
                        allocation_claim: AllocationClaim | None = None) -> Path:
    """Validate a designated trainer checkpoint and immutably publish its manifest."""
    root, checkpoint = Path(run_dir), Path(checkpoint)
    if (peer_authority is None) != (allocation_claim is None):
        raise ValueError(
            "fenced checkpoint publication requires authority and claim together")
    if peer_authority is not None:
        if (
            allocation_claim.run_id != run_id
            or allocation_claim.fence != fence.coordinator_epoch
        ):
            raise ValueError(
                "checkpoint allocation claim differs from generation fence")
        peer_authority.assert_current(allocation_claim)
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
    checkpoint_digest = hashlib.sha256()
    with checkpoint.open("rb") as checkpoint_stream:
        for block in iter(
                lambda: checkpoint_stream.read(8 * 1024 * 1024), b""):
            checkpoint_digest.update(block)
    raw_sha = checkpoint_digest.hexdigest()
    target_name = f"generation-{generation:08d}"
    if peer_authority is not None:
        # A superseded writer may finish an already-open filesystem write. A
        # fence-specific immutable name prevents that orphan from becoming an
        # ancestor of the current allocation's commit receipt chain.
        target_name += f"-fence-{allocation_claim.fence:08d}"
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
        "generation": generation,
        "manifest": str(target.resolve()),
        "manifest_sha256": manifest_sha,
        "fence": fence.coordinator_epoch,
        "accepted_tokens": int(accepted_tokens),
        "authoritative_source": "local_fixture_manifest",
    }
    if peer_authority is not None:
        # The complete checkpoint was streamed and verified immediately above;
        # pass that digest into the immutable receipt publisher so production
        # does not read a multi-gigabyte E97 checkpoint twice.
        receipt = peer_authority.publish_checkpoint(
            allocation_claim, target,
            verified_checkpoint_sha256=raw_sha)
        latest = receipt.pointer()
    atomic_json(root / "handoff" / "latest.json", latest)
    if peer_authority is not None:
        peer_authority.assert_current(allocation_claim)
    return target
