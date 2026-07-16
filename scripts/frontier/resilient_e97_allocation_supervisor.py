#!/usr/bin/env python3
"""Supervise the 2-manager/16-trainer Frontier allocation without a job-wide step.

Each child owns an independent ``srun --no-kill`` step.  A failed child is
restarted without waiting for, signalling, or recreating healthy siblings.
Role programs publish heartbeat/progress JSON through the shared run directory;
missing deadlines cause only that role's process group to be evicted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


TRAINERS_PER_NODE = 8


@dataclass
class Child:
    role: str
    node_rank: int
    node: str
    local_rank: int | None
    command: str
    process: subprocess.Popen | None = None
    restarts: int = 0

    @property
    def identity(self) -> str:
        suffix = "manager" if self.local_rank is None else f"trainer-{self.local_rank}"
        return f"node-{self.node_rank}-{suffix}"


class AllocationSupervisor:
    def __init__(self, run_dir: Path, children: list[Child], *, heartbeat_s: float,
                 progress_s: float, max_restarts: int, poll_s: float = 2.0):
        self.run_dir, self.children = run_dir, children
        self.heartbeat_s, self.progress_s = heartbeat_s, progress_s
        self.max_restarts, self.poll_s = max_restarts, poll_s
        self.stopping = False
        (run_dir / "supervision").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def _event(self, event: str, child: Child, **extra: object) -> None:
        record = {"time": time.time(), "event": event, "identity": child.identity,
                  "role": child.role, "node_rank": child.node_rank,
                  "local_rank": child.local_rank, **extra}
        with (self.run_dir / "supervision" / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def start(self, child: Child) -> None:
        role_env = [f"RESILIENT_E97_ROLE={child.role}",
                    f"RESILIENT_E97_NODE_RANK={child.node_rank}",
                    f"RUN_DIR={self.run_dir}"]
        resources = ["-c8"] if child.role == "manager" else [
            "-c7", "--gpus-per-task=1", f"--gpu-bind=map_gpu:{child.local_rank}"
        ]
        if child.role == "manager":
            role_env.append("CUDA_VISIBLE_DEVICES=")
        else:
            role_env += [f"RESILIENT_E97_LOCAL_RANK={child.local_rank}",
                         "ASYNC_LOCAL_STEPS=40"]
        argv = ["srun", "--overlap", "--no-kill", "--exact", "-N1", "-n1",
                "-w", child.node, *resources, "env", *role_env, *shlex.split(child.command)]
        log = self.run_dir / "logs" / child.identity
        stdout = (log.with_suffix(".out")).open("ab", buffering=0)
        stderr = (log.with_suffix(".err")).open("ab", buffering=0)
        child.process = subprocess.Popen(argv, stdout=stdout, stderr=stderr,
                                         start_new_session=True)
        self._event("started", child, pid=child.process.pid, restart=child.restarts)

    def _deadline_reason(self, child: Child, now: float) -> str | None:
        assert child.process is not None
        if child.process.poll() is not None:
            return f"exit:{child.process.returncode}"
        state_path = self.run_dir / "supervision" / f"{child.identity}.json"
        if not state_path.exists():
            return None  # connect deadline is enforced by role command itself
        state = json.loads(state_path.read_text())
        if now - float(state.get("heartbeat_time", 0)) > self.heartbeat_s:
            return "heartbeat_deadline"
        if now - float(state.get("progress_time", 0)) > self.progress_s:
            return "progress_deadline"
        return None

    def stop_child(self, child: Child, reason: str) -> None:
        assert child.process is not None
        if child.process.poll() is None:
            os.killpg(child.process.pid, signal.SIGTERM)
            try:
                child.process.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(child.process.pid, signal.SIGKILL)
                child.process.wait()
        self._event("evicted", child, reason=reason, exit_code=child.process.returncode)

    def run(self) -> int:
        for child in self.children:
            self.start(child)
        while not self.stopping:
            now = time.time()
            for child in self.children:
                reason = self._deadline_reason(child, now)
                if reason is None:
                    continue
                self.stop_child(child, reason)
                if child.restarts >= self.max_restarts:
                    self._event("restart_exhausted", child, reason=reason)
                    return 1
                child.restarts += 1
                self.start(child)
            time.sleep(self.poll_s)
        for child in self.children:
            if child.process is not None and child.process.poll() is None:
                self.stop_child(child, "allocation_term_handoff")
        return 0


def main() -> int:
    run_dir = Path(os.environ["RUN_DIR"])
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).splitlines()
    if len(nodes) != 2:
        raise SystemExit("true resilient gate requires exactly two physical nodes")
    manager = os.environ["RESILIENT_E97_MANAGER_COMMAND"]
    trainer = os.environ["RESILIENT_E97_TRAINER_COMMAND"]
    children: list[Child] = []
    for node_rank, node in enumerate(nodes):
        children.append(Child("manager", node_rank, node, None, manager))
        children.extend(Child("trainer", node_rank, node, rank, trainer)
                        for rank in range(TRAINERS_PER_NODE))
    supervisor = AllocationSupervisor(
        run_dir, children,
        heartbeat_s=float(os.environ.get("RESILIENT_E97_HEARTBEAT_DEADLINE_S", "60")),
        progress_s=float(os.environ.get("RESILIENT_E97_PROGRESS_DEADLINE_S", "900")),
        max_restarts=int(os.environ.get("RESILIENT_E97_MAX_RESTARTS", "2")),
    )
    signal.signal(signal.SIGTERM, lambda *_: setattr(supervisor, "stopping", True))
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
