"""Pure async quorum DiLoCo state and delta helpers.

This module deliberately contains no job-launch or transport code.  It operates
on dense CPU/GPU tensors and small metadata records so the same math can be used
by unit tests, local simulators, and future Frontier merger plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch


TensorSeq = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class AsyncDilocoState:
    """ScheduleFree tensor state for one global or worker generation."""

    generation: int
    x: TensorSeq
    z: TensorSeq
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_xz(self.x, self.z, label="state")

    def clone(self, *, generation: int | None = None) -> "AsyncDilocoState":
        return AsyncDilocoState(
            generation=self.generation if generation is None else int(generation),
            x=_clone_tensors(self.x),
            z=_clone_tensors(self.z),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class AsyncDilocoDelta:
    """Dense x/z displacement produced by one worker from a named base."""

    worker_id: str
    base_generation: int
    dx: TensorSeq
    dz: TensorSeq
    tokens: float | None = None
    local_steps: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_xz(self.dx, self.dz, label="delta")


@dataclass(frozen=True)
class RejectedUpdate:
    worker_id: str
    base_generation: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "base_generation": self.base_generation,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QuorumMergeMetrics:
    configured_quorum: int
    effective_quorum: int
    accepted_count: int
    rejected_count: int
    merge_time_s: float
    current_generation: int
    next_generation: int
    advanced: bool
    eta_outer: float
    weight_mode: str
    accepted_worker_ids: tuple[str, ...]
    rejected_updates: tuple[RejectedUpdate, ...]
    missing_worker_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_quorum": self.configured_quorum,
            "effective_quorum": self.effective_quorum,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "merge_time_s": self.merge_time_s,
            "current_generation": self.current_generation,
            "next_generation": self.next_generation,
            "advanced": self.advanced,
            "eta_outer": self.eta_outer,
            "weight_mode": self.weight_mode,
            "accepted_worker_ids": list(self.accepted_worker_ids),
            "rejected_updates": [r.to_dict() for r in self.rejected_updates],
            "missing_worker_ids": list(self.missing_worker_ids),
        }


@dataclass(frozen=True)
class QuorumMergeResult:
    state: AsyncDilocoState
    accepted: tuple[AsyncDilocoDelta, ...]
    rejected: tuple[RejectedUpdate, ...]
    metrics: QuorumMergeMetrics

    @property
    def advanced(self) -> bool:
        return self.metrics.advanced


def _clone_tensors(tensors: Iterable[torch.Tensor]) -> TensorSeq:
    return tuple(t.detach().clone() for t in tensors)


def _validate_xz(x: Sequence[torch.Tensor], z: Sequence[torch.Tensor], *, label: str) -> None:
    if len(x) != len(z):
        raise ValueError(f"{label} x/z length mismatch: {len(x)} != {len(z)}")
    if not x:
        raise ValueError(f"{label} must contain at least one tensor")
    for i, (xt, zt) in enumerate(zip(x, z)):
        if not isinstance(xt, torch.Tensor) or not isinstance(zt, torch.Tensor):
            raise TypeError(f"{label} x/z entries must be torch.Tensor objects")
        if xt.shape != zt.shape:
            raise ValueError(f"{label} tensor {i} shape mismatch: {xt.shape} != {zt.shape}")
        if xt.device != zt.device:
            raise ValueError(f"{label} tensor {i} device mismatch: {xt.device} != {zt.device}")


def _validate_compatible_state(base: AsyncDilocoState, other: AsyncDilocoState, *, label: str) -> None:
    _validate_compatible_tensors(base.x, other.x, label=f"{label}.x")
    _validate_compatible_tensors(base.z, other.z, label=f"{label}.z")


def _validate_compatible_delta(base: AsyncDilocoState, delta: AsyncDilocoDelta) -> None:
    _validate_compatible_tensors(base.x, delta.dx, label="delta.dx")
    _validate_compatible_tensors(base.z, delta.dz, label="delta.dz")


def _validate_compatible_tensors(
    reference: Sequence[torch.Tensor],
    candidate: Sequence[torch.Tensor],
    *,
    label: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} length mismatch: {len(reference)} != {len(candidate)}")
    for i, (rt, ct) in enumerate(zip(reference, candidate)):
        if rt.shape != ct.shape:
            raise ValueError(f"{label} tensor {i} shape mismatch: {rt.shape} != {ct.shape}")
        if rt.device != ct.device:
            raise ValueError(f"{label} tensor {i} device mismatch: {rt.device} != {ct.device}")


def extract_schedulefree_xz(
    model: torch.nn.Module,
    optimizer: Any,
    *,
    generation: int,
    metadata: Mapping[str, Any] | None = None,
) -> AsyncDilocoState:
    """Clone ScheduleFree eval-weight x and base-iterate z from a live optimizer.

    ScheduleFree stores x in ``p.data`` while the optimizer is in eval mode.  This
    helper performs that mode swap, clones x/z, and restores train mode when the
    optimizer was training before extraction.  Scalar clocks stay local and are
    copied only into metadata for observability.
    """

    params = tuple(model.parameters())
    if not params:
        raise ValueError("cannot extract ScheduleFree state from a parameterless model")
    train_modes = tuple(bool(group.get("train_mode", False)) for group in optimizer.param_groups)
    was_training = any(train_modes)
    if was_training:
        optimizer.eval()
    try:
        x = _clone_tensors(p.data for p in params)
        z = []
        for i, p in enumerate(params):
            param_state = optimizer.state.get(p, {})
            if "z" not in param_state:
                raise KeyError(f"ScheduleFree optimizer state for parameter {i} has no 'z' tensor")
            z.append(param_state["z"].detach().clone())
        meta = dict(metadata or {})
        meta.setdefault(
            "schedulefree_param_groups",
            [
                {
                    "k": group.get("k"),
                    "weight_sum": group.get("weight_sum"),
                    "lr_max": group.get("lr_max"),
                    "train_mode": group.get("train_mode"),
                }
                for group in optimizer.param_groups
            ],
        )
        return AsyncDilocoState(generation=int(generation), x=x, z=tuple(z), metadata=meta)
    finally:
        if was_training:
            optimizer.train()


def install_schedulefree_xz_state(
    model: torch.nn.Module,
    optimizer: Any,
    state: AsyncDilocoState,
) -> None:
    """Install x/z tensors into a live ScheduleFree model/optimizer pair."""

    params = tuple(model.parameters())
    if len(params) != len(state.x):
        raise ValueError(f"model parameter count {len(params)} does not match state count {len(state.x)}")
    train_modes = tuple(bool(group.get("train_mode", False)) for group in optimizer.param_groups)
    was_training = any(train_modes)
    if was_training:
        optimizer.eval()
    try:
        with torch.no_grad():
            for i, (p, x, z) in enumerate(zip(params, state.x, state.z)):
                if p.data.shape != x.shape or p.data.shape != z.shape:
                    raise ValueError(f"state tensor {i} shape does not match parameter shape {p.data.shape}")
                p.data.copy_(x)
                optimizer.state.setdefault(p, {})["z"] = z.detach().clone().to(device=p.device, dtype=p.dtype)
    finally:
        if was_training:
            optimizer.train()


def compute_dense_delta(
    base: AsyncDilocoState,
    worker: AsyncDilocoState,
    *,
    worker_id: str,
    tokens: float | None = None,
    local_steps: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AsyncDilocoDelta:
    """Return ``worker - base`` as dense x/z DiLoCo deltas."""

    _validate_compatible_state(base, worker, label="worker")
    return AsyncDilocoDelta(
        worker_id=str(worker_id),
        base_generation=base.generation,
        dx=tuple(wx.detach().clone() - bx for bx, wx in zip(base.x, worker.x)),
        dz=tuple(wz.detach().clone() - bz for bz, wz in zip(base.z, worker.z)),
        tokens=tokens,
        local_steps=local_steps,
        metadata=dict(metadata or {}),
    )


def apply_dense_delta(
    base: AsyncDilocoState,
    delta: AsyncDilocoDelta,
    *,
    eta_outer: float = 1.0,
    generation: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AsyncDilocoState:
    """Apply one dense x/z delta to a base state."""

    _validate_compatible_delta(base, delta)
    eta = float(eta_outer)
    meta = dict(base.metadata)
    meta.update(metadata or {})
    return AsyncDilocoState(
        generation=base.generation + 1 if generation is None else int(generation),
        x=tuple(bx + eta * dx.to(device=bx.device, dtype=bx.dtype) for bx, dx in zip(base.x, delta.dx)),
        z=tuple(bz + eta * dz.to(device=bz.device, dtype=bz.dtype) for bz, dz in zip(base.z, delta.dz)),
        metadata=meta,
    )


def rebase_local_state(
    local: AsyncDilocoState,
    old_base: AsyncDilocoState,
    new_base: AsyncDilocoState,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> AsyncDilocoState:
    """Shift local x/z by the base movement while preserving local displacement."""

    _validate_compatible_state(old_base, local, label="local")
    _validate_compatible_state(old_base, new_base, label="new_base")
    meta = dict(local.metadata)
    meta.update(metadata or {})
    return AsyncDilocoState(
        generation=new_base.generation,
        x=tuple(lx + (nx - ox) for lx, ox, nx in zip(local.x, old_base.x, new_base.x)),
        z=tuple(lz + (nz - oz) for lz, oz, nz in zip(local.z, old_base.z, new_base.z)),
        metadata=meta,
    )


def _update_weight(update: AsyncDilocoDelta, weight_mode: str) -> float:
    if weight_mode == "equal":
        return 1.0
    if weight_mode == "tokens":
        value = update.tokens if update.tokens is not None else update.local_steps
        return 1.0 if value is None else float(value)
    if weight_mode == "local_steps":
        value = update.local_steps if update.local_steps is not None else update.tokens
        return 1.0 if value is None else float(value)
    raise ValueError(f"unknown weight_mode {weight_mode!r}; expected equal, tokens, or local_steps")


def weighted_mean_delta(
    updates: Sequence[AsyncDilocoDelta],
    *,
    weight_mode: str = "tokens",
) -> AsyncDilocoDelta:
    """Compute the dense weighted mean of accepted deltas."""

    if not updates:
        raise ValueError("cannot average an empty update set")
    base_generation = updates[0].base_generation
    for update in updates:
        if update.base_generation != base_generation:
            raise ValueError("cannot average deltas from different base generations")
        _validate_compatible_tensors(updates[0].dx, update.dx, label="updates.dx")
        _validate_compatible_tensors(updates[0].dz, update.dz, label="updates.dz")

    weights = [_update_weight(update, weight_mode) for update in updates]
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"{weight_mode} weights must be non-negative")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError(f"{weight_mode} weights must sum to a positive value")

    ref = updates[0]
    dx_acc = [torch.zeros_like(t, dtype=torch.promote_types(t.dtype, torch.float32)) for t in ref.dx]
    dz_acc = [torch.zeros_like(t, dtype=torch.promote_types(t.dtype, torch.float32)) for t in ref.dz]
    for update, weight in zip(updates, weights):
        scale = float(weight) / total
        for acc, tensor in zip(dx_acc, update.dx):
            acc.add_(tensor.to(device=acc.device, dtype=acc.dtype), alpha=scale)
        for acc, tensor in zip(dz_acc, update.dz):
            acc.add_(tensor.to(device=acc.device, dtype=acc.dtype), alpha=scale)

    return AsyncDilocoDelta(
        worker_id="weighted_mean",
        base_generation=base_generation,
        dx=tuple(dx_acc),
        dz=tuple(dz_acc),
        tokens=sum(update.tokens or 0.0 for update in updates),
        local_steps=sum(update.local_steps or 0.0 for update in updates),
        metadata={"weight_mode": weight_mode, "weights": weights},
    )


def quorum_merge(
    base: AsyncDilocoState,
    updates: Sequence[AsyncDilocoDelta],
    *,
    quorum: int,
    eta_outer: float = 1.0,
    weight_mode: str = "tokens",
    expected_worker_ids: Iterable[str] | None = None,
    max_staleness: int = 0,
    metrics_path: str | Path | None = None,
) -> QuorumMergeResult:
    """Reject stale updates and merge once the accepted quorum is satisfied."""

    if quorum <= 0:
        raise ValueError("quorum must be positive")
    if max_staleness < 0:
        raise ValueError("max_staleness must be non-negative")

    t0 = time.perf_counter()
    accepted: list[AsyncDilocoDelta] = []
    rejected: list[RejectedUpdate] = []
    seen_workers: set[str] = set()
    min_allowed_generation = base.generation - int(max_staleness)
    for update in updates:
        if update.worker_id in seen_workers:
            rejected.append(RejectedUpdate(update.worker_id, update.base_generation, "duplicate_worker"))
            continue
        seen_workers.add(update.worker_id)
        if update.base_generation != base.generation:
            if update.base_generation < min_allowed_generation or max_staleness == 0:
                reason = "stale_generation"
            else:
                reason = "non_mainline_staleness_policy"
            rejected.append(RejectedUpdate(update.worker_id, update.base_generation, reason))
            continue
        _validate_compatible_delta(base, update)
        accepted.append(update)

    advanced = len(accepted) >= quorum
    if advanced:
        mean = weighted_mean_delta(accepted, weight_mode=weight_mode)
        next_state = apply_dense_delta(
            base,
            mean,
            eta_outer=eta_outer,
            generation=base.generation + 1,
            metadata={"merged_from_generation": base.generation},
        )
    else:
        next_state = base.clone(generation=base.generation)

    expected = set(expected_worker_ids or ())
    missing = tuple(sorted(expected.difference({update.worker_id for update in accepted})))
    merge_time_s = time.perf_counter() - t0
    metrics = QuorumMergeMetrics(
        configured_quorum=int(quorum),
        effective_quorum=len(accepted),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        merge_time_s=merge_time_s,
        current_generation=base.generation,
        next_generation=next_state.generation,
        advanced=advanced,
        eta_outer=float(eta_outer),
        weight_mode=weight_mode,
        accepted_worker_ids=tuple(update.worker_id for update in accepted),
        rejected_updates=tuple(rejected),
        missing_worker_ids=missing,
    )
    if metrics_path is not None:
        write_merge_metrics(metrics, metrics_path)
    return QuorumMergeResult(
        state=next_state,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        metrics=metrics,
    )


def write_merge_metrics(metrics: QuorumMergeMetrics, path: str | Path) -> None:
    """Write merge metrics as stable JSON for downstream harnesses."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
