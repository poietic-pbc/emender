"""Process topology and supervision for true resilient E97 nodes.

Managers are CPU-only control/data-plane processes.  Trainers are the only
model-bearing processes and each owns exactly one physical GPU.  Deliberately
there is no MPI process group: every child is a separate process group and can
be detected, killed, and restarted without terminating its siblings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence


TRAINERS_PER_FRONTIER_NODE = 8


@dataclass(frozen=True)
class ChildSpec:
    role: str
    node_rank: int
    local_rank: int | None
    command: tuple[str, ...]
    env: Mapping[str, str]

    @property
    def identity(self) -> str:
        suffix = "manager" if self.local_rank is None else f"trainer-{self.local_rank}"
        return f"node-{self.node_rank}/{suffix}"


def true_frontier_topology(node_count: int, manager_command: Sequence[str],
                           trainer_command: Sequence[str]) -> tuple[ChildSpec, ...]:
    """Return exactly one manager and eight real trainers per physical node."""
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if not manager_command or not trainer_command:
        raise ValueError("manager and trainer commands are required")
    result: list[ChildSpec] = []
    for node in range(node_count):
        result.append(ChildSpec("manager", node, None, tuple(manager_command), {
            "RESILIENT_E97_ROLE": "manager", "RESILIENT_E97_NODE_RANK": str(node),
            "CUDA_VISIBLE_DEVICES": "",
        }))
        for local_rank in range(TRAINERS_PER_FRONTIER_NODE):
            result.append(ChildSpec("trainer", node, local_rank, tuple(trainer_command), {
                "RESILIENT_E97_ROLE": "trainer", "RESILIENT_E97_NODE_RANK": str(node),
                "RESILIENT_E97_LOCAL_RANK": str(local_rank),
                "CUDA_VISIBLE_DEVICES": str(local_rank),
            }))
    validate_true_topology(result, node_count=node_count)
    return tuple(result)


def validate_true_topology(specs: Sequence[ChildSpec], *, node_count: int) -> None:
    managers = [item for item in specs if item.role == "manager"]
    trainers = [item for item in specs if item.role == "trainer"]
    unknown = [item.role for item in specs if item.role not in {"manager", "trainer"}]
    if unknown:
        raise ValueError(f"sentinel/unknown roles are forbidden: {unknown}")
    if len(managers) != node_count:
        raise ValueError(f"expected {node_count} managers, got {len(managers)}")
    if len(trainers) != node_count * TRAINERS_PER_FRONTIER_NODE:
        raise ValueError(
            f"expected {node_count * TRAINERS_PER_FRONTIER_NODE} real trainers, got {len(trainers)}"
        )
    for node in range(node_count):
        node_managers = [item for item in managers if item.node_rank == node]
        ranks = sorted(item.local_rank for item in trainers if item.node_rank == node)
        if len(node_managers) != 1 or ranks != list(range(TRAINERS_PER_FRONTIER_NODE)):
            raise ValueError(f"node {node} does not have one manager and GPU trainers 0..7")
        if node_managers[0].env.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("manager must be CPU-only")


class IndependentProcessSupervisor:
    """Supervise children with per-role heartbeat and progress deadlines."""

    def __init__(self, run_dir: str | Path, *, heartbeat_deadline_s: float,
                 progress_deadline_s: float):
        self.run_dir = Path(run_dir)
        self.heartbeat_deadline_s = float(heartbeat_deadline_s)
        self.progress_deadline_s = float(progress_deadline_s)
        if min(self.heartbeat_deadline_s, self.progress_deadline_s) <= 0:
            raise ValueError("deadlines must be positive")
        self.processes: dict[str, subprocess.Popen] = {}
        self.events: list[dict[str, object]] = []

    def start(self, spec: ChildSpec) -> subprocess.Popen:
        env = os.environ.copy(); env.update(spec.env)
        process = subprocess.Popen(spec.command, env=env, start_new_session=True)
        self.processes[spec.identity] = process
        self.events.append({"event": "started", "identity": spec.identity,
                            "pid": process.pid, "time": time.time()})
        return process

    def check(self, now: float | None = None) -> tuple[str, ...]:
        now = time.time() if now is None else float(now)
        evicted: list[str] = []
        for identity, process in list(self.processes.items()):
            state_path = self.run_dir / "supervision" / f"{identity.replace('/', '-')}.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            heartbeat_age = now - float(state.get("heartbeat_time", 0))
            progress_age = now - float(state.get("progress_time", 0))
            exited = process.poll() is not None
            if exited or heartbeat_age > self.heartbeat_deadline_s or progress_age > self.progress_deadline_s:
                if not exited:
                    os.killpg(process.pid, signal.SIGTERM)
                reason = "exit" if exited else ("heartbeat_deadline" if heartbeat_age > self.heartbeat_deadline_s else "progress_deadline")
                self.events.append({"event": "evicted", "identity": identity,
                                    "reason": reason, "time": now})
                evicted.append(identity)
                del self.processes[identity]
        return tuple(evicted)

