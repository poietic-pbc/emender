"""Local-process async quorum DiLoCo simulation with ScheduleFree state."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


@dataclass
class LocalScheduleFreeState:
    """Dense ScheduleFree x/z tensor state for local simulations."""

    x: tuple[torch.Tensor, ...]
    z: tuple[torch.Tensor, ...]

    def clone(self) -> "LocalScheduleFreeState":
        return LocalScheduleFreeState(
            x=tuple(t.detach().clone() for t in self.x),
            z=tuple(t.detach().clone() for t in self.z),
        )


@dataclass(frozen=True)
class LocalScheduleFreeDelta:
    dx: tuple[torch.Tensor, ...]
    dz: tuple[torch.Tensor, ...]


@dataclass
class LocalAsyncUpdate:
    worker_id: int
    base_generation: int
    delta: LocalScheduleFreeDelta
    tokens: int
    local_steps: int
    loss_before: float
    loss_after: float
    submitted_at: float = 0.0


@dataclass
class LocalAsyncGenerationRecord:
    generation: int
    next_generation: int
    cause: str
    effective_quorum: int
    accepted_workers: tuple[int, ...]
    stale_workers: tuple[int, ...]
    missing_workers: tuple[int, ...]
    token_weight_sum: float
    loss_before_mean: float
    loss_after_mean: float
    advanced_at: float
    checkpoint_path: str | None = None
    latest_advanced: bool = False


@dataclass
class LocalAsyncDilocoConfig:
    num_workers: int = 4
    quorum: int = 4
    timeout_s: float = 1.0
    timeout_min_updates: int = 1
    max_generations: int = 4
    local_steps: int = 2
    lr: float = 0.03
    outer_lr: float = 1.0
    token_weighted: bool = True
    checkpoint_every_generations: int = 1
    checkpoint_wall_interval_s: float = math.inf
    seed: int = 0
    worker_delay: Callable[[int, int], float] | None = None
    worker_drop: Callable[[int, int], bool] | None = None
    token_count: Callable[[int, int], int] | None = None


@dataclass
class LocalAsyncSimulationMetrics:
    generation_records: list[LocalAsyncGenerationRecord] = field(default_factory=list)
    accepted_updates: int = 0
    stale_rejected: int = 0
    dropped_updates: int = 0
    quorum_advances: int = 0
    timeout_advances: int = 0
    checkpoint_writes: int = 0
    resumed_from_generation: int | None = None
    loss_initial: float | None = None
    loss_final: float | None = None

    @property
    def effective_quorums(self) -> list[int]:
        return [record.effective_quorum for record in self.generation_records]

    def summary(self) -> dict[str, Any]:
        quorums = self.effective_quorums
        causes: dict[str, int] = {}
        for record in self.generation_records:
            causes[record.cause] = causes.get(record.cause, 0) + 1
        return {
            "generations": len(self.generation_records),
            "accepted_updates": self.accepted_updates,
            "stale_rejected": self.stale_rejected,
            "dropped_updates": self.dropped_updates,
            "quorum_advances": self.quorum_advances,
            "timeout_advances": self.timeout_advances,
            "effective_quorum_distribution": {
                str(q): quorums.count(q) for q in sorted(set(quorums))
            },
            "effective_quorum_min": min(quorums) if quorums else 0,
            "effective_quorum_max": max(quorums) if quorums else 0,
            "effective_quorum_mean": sum(quorums) / len(quorums) if quorums else 0.0,
            "timeout_causes": causes,
            "checkpoint_writes": self.checkpoint_writes,
            "resumed_from_generation": self.resumed_from_generation,
            "loss_initial": self.loss_initial,
            "loss_final": self.loss_final,
            "loss_delta": (
                None
                if self.loss_initial is None or self.loss_final is None
                else self.loss_final - self.loss_initial
            ),
        }


def _local_schedulefree_train_modes(optimizer: Any) -> list[bool]:
    return [bool(group.get("train_mode", False)) for group in optimizer.param_groups]


def _local_restore_schedulefree_train_modes(optimizer: Any, modes: Sequence[bool]) -> None:
    for mode in modes:
        if mode:
            optimizer.train()
        else:
            optimizer.eval()
        break


def extract_local_schedulefree_state(model: Any, optimizer: Any) -> LocalScheduleFreeState:
    """Extract ScheduleFree eval x and base z tensors from a real optimizer."""

    modes = _local_schedulefree_train_modes(optimizer)
    optimizer.eval()
    try:
        params = tuple(model.parameters())
        x = tuple(p.data.detach().clone() for p in params)
        z = tuple(
            optimizer.state.get(p, {}).get("z", p.data).detach().clone()
            for p in params
        )
        return LocalScheduleFreeState(x=x, z=z)
    finally:
        _local_restore_schedulefree_train_modes(optimizer, modes)


def load_local_schedulefree_state(
    model: Any,
    optimizer: Any,
    state: LocalScheduleFreeState,
    *,
    train_mode: bool = True,
) -> None:
    params = tuple(model.parameters())
    if len(params) != len(state.x) or len(params) != len(state.z):
        raise ValueError("state tensor count does not match model")
    optimizer.eval()
    for p, x, z in zip(params, state.x, state.z):
        p.data.copy_(x)
        param_state = optimizer.state.setdefault(p, {})
        param_state["z"] = z.detach().clone()
        # Fresh AdamWScheduleFree optimizers lazily create the second moment on
        # the first step. Loading only x/z would make the state non-empty and can
        # bypass that lazy branch, so seed the expected zero buffer here.
        param_state.setdefault("exp_avg_sq", torch.zeros_like(p.data))
    if train_mode:
        optimizer.train()


def local_schedulefree_delta(
    after: LocalScheduleFreeState,
    before: LocalScheduleFreeState,
) -> LocalScheduleFreeDelta:
    return LocalScheduleFreeDelta(
        dx=tuple(a - b for a, b in zip(after.x, before.x)),
        dz=tuple(a - b for a, b in zip(after.z, before.z)),
    )


def merge_local_async_updates(
    base: LocalScheduleFreeState,
    updates: Sequence[LocalAsyncUpdate],
    *,
    token_weighted: bool = True,
    outer_lr: float = 1.0,
) -> LocalScheduleFreeState:
    if not updates:
        raise ValueError("cannot merge an empty update set")
    weights = [float(update.tokens if token_weighted else 1.0) for update in updates]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("merge weights must sum to a positive value")
    merged_dx = []
    merged_dz = []
    for idx in range(len(base.x)):
        dx = torch.zeros_like(base.x[idx])
        dz = torch.zeros_like(base.z[idx])
        for weight, update in zip(weights, updates):
            dx.add_(update.delta.dx[idx], alpha=weight / weight_sum)
            dz.add_(update.delta.dz[idx], alpha=weight / weight_sum)
        merged_dx.append(dx)
        merged_dz.append(dz)
    return LocalScheduleFreeState(
        x=tuple(x + float(outer_lr) * dx for x, dx in zip(base.x, merged_dx)),
        z=tuple(z + float(outer_lr) * dz for z, dz in zip(base.z, merged_dz)),
    )


def rebase_local_schedulefree_state(
    local: LocalScheduleFreeState,
    old_base: LocalScheduleFreeState,
    new_base: LocalScheduleFreeState,
) -> LocalScheduleFreeState:
    return LocalScheduleFreeState(
        x=tuple(lx + (nx - ox) for lx, ox, nx in zip(local.x, old_base.x, new_base.x)),
        z=tuple(lz + (nz - oz) for lz, oz, nz in zip(local.z, old_base.z, new_base.z)),
    )


class LocalAsyncCheckpointManager:
    """Small local state checkpoint manager used by the simulator."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.generations_dir = self.root / "generations"
        self.generations_dir.mkdir(parents=True, exist_ok=True)

    def generation_dir(self, generation: int) -> Path:
        return self.generations_dir / f"gen_{generation:06d}"

    def latest_generation(self) -> int | None:
        latest = self.root / "latest"
        if not latest.exists() and not latest.is_symlink():
            return None
        target = latest.resolve()
        if not target.name.startswith("gen_"):
            raise ValueError(f"latest points outside generation directories: {target}")
        return int(target.name.split("_", 1)[1])

    def save_generation(
        self,
        generation: int,
        state: LocalScheduleFreeState,
        manifest: Mapping[str, Any],
    ) -> Path:
        gen_dir = self.generation_dir(generation)
        gen_dir.mkdir(parents=True, exist_ok=False)
        torch.save(
            {"generation": generation, "x": state.x, "z": state.z},
            gen_dir / "state.pt",
        )
        payload = dict(manifest)
        payload.setdefault("generation", generation)
        _atomic_write_json(gen_dir / "manifest.json", payload)
        tmp = self.root / f".latest.{os.getpid()}.tmp"
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        tmp.symlink_to(gen_dir.relative_to(self.root))
        os.replace(tmp, self.root / "latest")
        return gen_dir

    def load_generation(self, generation: int) -> tuple[int, LocalScheduleFreeState, dict[str, Any]]:
        gen_dir = self.generation_dir(generation)
        data = torch.load(gen_dir / "state.pt", map_location="cpu", weights_only=False)
        manifest = json.loads((gen_dir / "manifest.json").read_text(encoding="utf-8"))
        return (
            int(data["generation"]),
            LocalScheduleFreeState(x=tuple(data["x"]), z=tuple(data["z"])),
            manifest,
        )

    def load_latest(self) -> tuple[int, LocalScheduleFreeState, dict[str, Any]]:
        generation = self.latest_generation()
        if generation is None:
            raise FileNotFoundError("no local async latest generation")
        return self.load_generation(generation)


class LocalAsyncQuorumMerger:
    def __init__(
        self,
        state: LocalScheduleFreeState,
        config: LocalAsyncDilocoConfig,
        checkpoint_manager: LocalAsyncCheckpointManager | None = None,
        *,
        start_generation: int = 0,
    ):
        if config.quorum <= 0 or config.quorum > config.num_workers:
            raise ValueError("quorum must be in [1, num_workers]")
        self.state = state.clone()
        self.config = config
        self.current_generation = int(start_generation)
        self.opened_at = 0.0
        self.accepted: dict[int, LocalAsyncUpdate] = {}
        self.stale_workers: list[int] = []
        self.metrics = LocalAsyncSimulationMetrics()
        self.checkpoint_manager = checkpoint_manager
        self._last_checkpoint_time = -math.inf
        if checkpoint_manager is not None and checkpoint_manager.latest_generation() is None:
            self._write_checkpoint(self.current_generation, "initial", 0.0)

    def submit(self, update: LocalAsyncUpdate) -> bool:
        if update.base_generation != self.current_generation:
            self.metrics.stale_rejected += 1
            self.stale_workers.append(update.worker_id)
            return False
        if update.worker_id in self.accepted:
            return False
        self.accepted[update.worker_id] = update
        self.metrics.accepted_updates += 1
        if len(self.accepted) >= self.config.quorum:
            self._advance(update.submitted_at, "quorum")
        return True

    def maybe_timeout(self, now: float) -> bool:
        if now - self.opened_at < self.config.timeout_s:
            return False
        if len(self.accepted) < self.config.timeout_min_updates:
            self.opened_at = now
            return False
        self._advance(now, "timeout")
        return True

    def _advance(self, now: float, cause: str) -> None:
        old_generation = self.current_generation
        updates = list(self.accepted.values())
        self.state = merge_local_async_updates(
            self.state,
            updates,
            token_weighted=self.config.token_weighted,
            outer_lr=self.config.outer_lr,
        )
        self.current_generation += 1
        accepted_workers = tuple(sorted(update.worker_id for update in updates))
        missing_workers = tuple(
            worker for worker in range(self.config.num_workers)
            if worker not in accepted_workers
        )
        weight_sum = sum(
            float(update.tokens if self.config.token_weighted else 1.0)
            for update in updates
        )
        record = LocalAsyncGenerationRecord(
            generation=old_generation,
            next_generation=self.current_generation,
            cause=cause,
            effective_quorum=len(updates),
            accepted_workers=accepted_workers,
            stale_workers=tuple(sorted(set(self.stale_workers))),
            missing_workers=missing_workers,
            token_weight_sum=weight_sum,
            loss_before_mean=sum(update.loss_before for update in updates) / len(updates),
            loss_after_mean=sum(update.loss_after for update in updates) / len(updates),
            advanced_at=now,
        )
        self.metrics.generation_records.append(record)
        if cause == "timeout":
            self.metrics.timeout_advances += 1
        else:
            self.metrics.quorum_advances += 1
        self.accepted = {}
        self.stale_workers = []
        self.opened_at = now
        if self._checkpoint_due(now):
            path = self._write_checkpoint(self.current_generation, cause, now)
            record.checkpoint_path = None if path is None else str(path)
            record.latest_advanced = path is not None

    def _checkpoint_due(self, now: float) -> bool:
        generation_interval = int(self.config.checkpoint_every_generations or 0)
        by_generation = (
            generation_interval > 0
            and self.current_generation % generation_interval == 0
        )
        by_wall = now - self._last_checkpoint_time >= self.config.checkpoint_wall_interval_s
        return by_generation or by_wall

    def _write_checkpoint(self, generation: int, cause: str, now: float) -> Path | None:
        if self.checkpoint_manager is None:
            return None
        path = self.checkpoint_manager.save_generation(
            generation,
            self.state,
            {"cause": cause, "time": now, "current_open_generation": self.current_generation},
        )
        self.metrics.checkpoint_writes += 1
        self._last_checkpoint_time = now
        return path


def _build_local_toy_model(seed: int, lr: float) -> tuple[Any, Any]:
    import schedulefree

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=lr,
        warmup_steps=0,
        weight_decay=0.0,
    )
    return model, optimizer


def _local_toy_batch(seed: int, n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    y = 2.0 * x - 0.5 + 0.03 * torch.randn(x.shape, generator=generator)
    return x, y


def _local_toy_loss(model: Any, seed: int = 999) -> float:
    x, y = _local_toy_batch(seed, n=128)
    model.eval()
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean().item())


def _local_worker_update(
    base_state: LocalScheduleFreeState,
    worker_id: int,
    generation: int,
    config: LocalAsyncDilocoConfig,
) -> LocalAsyncUpdate:
    model, optimizer = _build_local_toy_model(config.seed, config.lr)
    load_local_schedulefree_state(model, optimizer, base_state, train_mode=True)
    loss_before = _local_toy_loss(model)
    optimizer.train()
    for step in range(config.local_steps):
        x, y = _local_toy_batch(config.seed * 100_000 + generation * 1000 + worker_id * 17 + step)
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    loss_after = _local_toy_loss(model)
    after = extract_local_schedulefree_state(model, optimizer)
    tokens = (
        int(config.token_count(worker_id, generation))
        if config.token_count is not None
        else 100 + 7 * worker_id
    )
    return LocalAsyncUpdate(
        worker_id=worker_id,
        base_generation=generation,
        delta=local_schedulefree_delta(after, base_state),
        tokens=tokens,
        local_steps=config.local_steps,
        loss_before=loss_before,
        loss_after=loss_after,
    )


def run_local_synchronous_diloco_reference(
    config: LocalAsyncDilocoConfig,
) -> tuple[LocalScheduleFreeState, list[float]]:
    model, optimizer = _build_local_toy_model(config.seed, config.lr)
    state = extract_local_schedulefree_state(model, optimizer)
    losses = [_local_toy_loss(model)]
    for generation in range(config.max_generations):
        updates = tuple(
            _local_worker_update(state, worker, generation, config)
            for worker in range(config.num_workers)
        )
        state = merge_local_async_updates(
            state,
            updates,
            token_weighted=config.token_weighted,
            outer_lr=config.outer_lr,
        )
        load_local_schedulefree_state(model, optimizer, state, train_mode=True)
        losses.append(_local_toy_loss(model))
    return state, losses


def run_local_async_diloco_simulation(
    config: LocalAsyncDilocoConfig,
    checkpoint_dir: str | Path | None = None,
    *,
    resume: bool = False,
) -> tuple[LocalScheduleFreeState, LocalAsyncSimulationMetrics]:
    import heapq

    checkpoint_manager = LocalAsyncCheckpointManager(checkpoint_dir) if checkpoint_dir else None
    model, optimizer = _build_local_toy_model(config.seed, config.lr)
    if resume and checkpoint_manager is not None and checkpoint_manager.latest_generation() is not None:
        start_generation, state, _manifest = checkpoint_manager.load_latest()
        load_local_schedulefree_state(model, optimizer, state, train_mode=True)
    else:
        start_generation = 0
        state = extract_local_schedulefree_state(model, optimizer)

    merger = LocalAsyncQuorumMerger(
        state,
        config,
        checkpoint_manager,
        start_generation=start_generation,
    )
    merger.metrics.resumed_from_generation = start_generation if resume else None
    merger.metrics.loss_initial = _local_toy_loss(model)
    events: list[tuple[float, int, LocalAsyncUpdate]] = []
    event_seq = 0

    def schedule_generation(generation: int, now: float) -> None:
        nonlocal event_seq
        base = merger.state.clone()
        for worker in range(config.num_workers):
            if config.worker_drop is not None and config.worker_drop(worker, generation):
                merger.metrics.dropped_updates += 1
                continue
            update = _local_worker_update(base, worker, generation, config)
            delay = 0.0 if config.worker_delay is None else float(config.worker_delay(worker, generation))
            update.submitted_at = now + delay
            heapq.heappush(events, (update.submitted_at, event_seq, update))
            event_seq += 1

    now = 0.0
    last_scheduled_generation = start_generation
    schedule_generation(start_generation, now)
    while merger.current_generation < start_generation + config.max_generations:
        next_timeout = merger.opened_at + config.timeout_s
        next_event_time = events[0][0] if events else math.inf
        if next_timeout <= next_event_time:
            now = next_timeout
            before = merger.current_generation
            merger.maybe_timeout(now)
            if (
                merger.current_generation != before
                and merger.current_generation != last_scheduled_generation
            ):
                schedule_generation(merger.current_generation, now)
                last_scheduled_generation = merger.current_generation
            continue
        if not events:
            break
        now, _seq, update = heapq.heappop(events)
        before = merger.current_generation
        merger.submit(update)
        if (
            merger.current_generation != before
            and merger.current_generation != last_scheduled_generation
        ):
            schedule_generation(merger.current_generation, now)
            last_scheduled_generation = merger.current_generation

    load_local_schedulefree_state(model, optimizer, merger.state, train_mode=True)
    merger.metrics.loss_final = _local_toy_loss(model)
    return merger.state, merger.metrics


def write_local_async_simulation_summary(
    path: str | Path,
    metrics: LocalAsyncSimulationMetrics,
) -> None:
    _atomic_write_json(Path(path), metrics.summary())
