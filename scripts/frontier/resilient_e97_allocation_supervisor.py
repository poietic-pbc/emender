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
    started_at: float | None = None

    @property
    def identity(self) -> str:
        suffix = (self.role if self.local_rank is None
                  else f"trainer-{self.local_rank}")
        return f"node-{self.node_rank}-{suffix}"


class AllocationSupervisor:
    def __init__(self, run_dir: Path, children: list[Child], *, heartbeat_s: float,
                 progress_s: float, max_restarts: int, startup_s: float = 120.0,
                 poll_s: float = 2.0,
                 launch_backend: str = "independent-step"):
        self.run_dir, self.children = run_dir, children
        self.heartbeat_s, self.progress_s = heartbeat_s, progress_s
        self.max_restarts, self.poll_s = max_restarts, poll_s
        self.startup_s = startup_s
        if launch_backend not in {"independent-step", "node-local-child"}:
            raise ValueError("unsupported supervisor launch backend")
        self.launch_backend = launch_backend
        self.stopping = False
        self.injected: set[str] = set()
        (run_dir / "supervision").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def _event(self, event: str, child: Child, **extra: object) -> None:
        record = {"time": time.time(), "event": event, "identity": child.identity,
                  "role": child.role, "node_rank": child.node_rank,
                  "local_rank": child.local_rank, **extra}
        with (self.run_dir / "supervision" / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def start(self, child: Child) -> None:
        role_values = {"RESILIENT_E97_ROLE": child.role,
                       "RESILIENT_E97_NODE_RANK": str(child.node_rank),
                       "RUN_DIR": str(self.run_dir)}
        role_env = [f"{key}={value}" for key, value in role_values.items()]
        if child.role == "manager":
            resources = ["-c8"]
            role_env.append("CUDA_VISIBLE_DEVICES=")
            role_values["CUDA_VISIBLE_DEVICES"] = ""
        elif child.role == "trainer":
            # Keep the complete node allocation visible to each overlapping
            # one-task step so map_gpu can select its stable local rank.
            resources = ["-c7", "--gpus-per-node=8",
                         f"--gpu-bind=map_gpu:{child.local_rank}"]
            role_env += [f"RESILIENT_E97_LOCAL_RANK={child.local_rank}",
                         "ASYNC_LOCAL_STEPS=40"]
            role_values.update({"RESILIENT_E97_LOCAL_RANK": str(child.local_rank),
                                "ASYNC_LOCAL_STEPS": "40",
                                "ROCR_VISIBLE_DEVICES": str(child.local_rank)})
        else:
            # The batch allocation already owns all eight GCDs on each node.
            # Frontier GRES cannot be allocated again to overlapping steps:
            # doing so leaves both node-supervisor steps pending with
            # "Requested nodes are busy".  Reuse the allocation's device
            # cgroup; direct children bind ROCR_VISIBLE_DEVICES=0..7.
            resources = ["-c64"]
        if self.launch_backend == "node-local-child":
            env = os.environ.copy(); env.update(role_values)
            argv = shlex.split(child.command)
        else:
            env = None
            argv = ["srun", "--overlap", "--no-kill", "--exact", "-N1", "-n1",
                    "-w", child.node, *resources, "env", *role_env, *shlex.split(child.command)]
        log = self.run_dir / "logs" / child.identity
        stdout = (log.with_suffix(".out")).open("ab", buffering=0)
        stderr = (log.with_suffix(".err")).open("ab", buffering=0)
        child.process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env,
                                         start_new_session=True)
        child.started_at = time.time()
        self._event("started", child, pid=child.process.pid, restart=child.restarts)

    def _deadline_reason(self, child: Child, now: float) -> str | None:
        assert child.process is not None
        if child.process.poll() is not None:
            return f"exit:{child.process.returncode}"
        bulk_root = os.environ.get("RESILIENT_E97_BULK_ROOT")
        run_id = os.environ.get("RESILIENT_E97_RUN_ID")
        state_root = (Path(bulk_root) / run_id / f"node-{child.node_rank}"
                      if bulk_root and run_id else self.run_dir)
        state_path = state_root / "supervision" / f"{child.identity}.json"
        # A node supervisor deliberately owns no training loop.  Its progress is
        # the model-free manager that it supervises, so use that fenced heartbeat
        # for generation-gated whole-node-step injection.
        if child.role == "node-supervisor":
            state_path = (state_root / "supervision" /
                          f"node-{child.node_rank}-manager.json")
        if not state_path.exists():
            if child.started_at is not None and now - child.started_at > self.startup_s:
                return "startup_deadline"
            return None
        state = json.loads(state_path.read_text())
        liveness_path = state_path.with_name(f"{child.identity}.liveness.json")
        if liveness_path.exists():
            liveness = json.loads(liveness_path.read_text())
            if liveness.get("identity") == child.identity:
                state["heartbeat_time"] = max(
                    float(state.get("heartbeat_time", 0)),
                    float(liveness.get("heartbeat_time", 0)))
        injection_variable = {
            "trainer": "RESILIENT_E97_INJECT_TRAINER",
            "manager": "RESILIENT_E97_INJECT_MANAGER",
            "node-supervisor": "RESILIENT_E97_INJECT_NODE_STEP",
        }[child.role]
        injection = os.environ.get(injection_variable, "")
        if injection and child.identity not in self.injected:
            fields = injection.split(":")
            if len(fields) != 3:
                raise ValueError("injection must be NODE:RANK:FINALIZED_GENERATION")
            node, rank, generation = map(int, fields)
            matches = (child.node_rank == node and
                       ((child.role == "trainer" and child.local_rank == rank) or
                        (child.role in {"manager", "node-supervisor"} and rank == -1)))
            if matches and int(state.get("generation", -1)) >= generation:
                self.injected.add(child.identity)
                self._event("generation_gated_injection", child, generation=generation,
                            injection_class=child.role)
                return "injected_generation_gate"
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
        completed: set[str] = set()
        while not self.stopping:
            now = time.time()
            for child in self.children:
                if child.identity in completed:
                    continue
                reason = self._deadline_reason(child, now)
                if reason is None:
                    continue
                if reason == "exit:0":
                    completed.add(child.identity)
                    self._event("completed", child, exit_code=0)
                    continue
                self.stop_child(child, reason)
                if child.restarts >= self.max_restarts:
                    self._event("restart_exhausted", child, reason=reason)
                    return 1
                child.restarts += 1
                self.start(child)
            if len(completed) == len(self.children):
                return 0
            time.sleep(self.poll_s)
        for child in self.children:
            if child.process is not None and child.process.poll() is None:
                self.stop_child(child, "allocation_term_handoff")
        return 0


def _node_local_main() -> int:
    run_dir = Path(os.environ["RUN_DIR"])
    node_rank = int(os.environ["RESILIENT_E97_NODE_RANK"])
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    manager = os.environ["RESILIENT_E97_MANAGER_COMMAND"]
    trainer = os.environ["RESILIENT_E97_TRAINER_COMMAND"]
    children = [Child("manager", node_rank, node, None, manager)]
    children.extend(Child("trainer", node_rank, node, rank, trainer)
                    for rank in range(TRAINERS_PER_NODE))
    supervisor = AllocationSupervisor(
        run_dir, children,
        heartbeat_s=float(os.environ.get("RESILIENT_E97_HEARTBEAT_DEADLINE_S", "60")),
        progress_s=float(os.environ.get("RESILIENT_E97_PROGRESS_DEADLINE_S", "900")),
        max_restarts=int(os.environ.get("RESILIENT_E97_MAX_RESTARTS", "2")),
        startup_s=float(os.environ.get("RESILIENT_E97_STARTUP_DEADLINE_S", "120")),
        launch_backend="node-local-child")
    signal.signal(signal.SIGTERM, lambda *_: setattr(supervisor, "stopping", True))
    return supervisor.run()


def main() -> int:
    if "--node-local" in sys.argv:
        return _node_local_main()
    run_dir = Path(os.environ["RUN_DIR"])
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).splitlines()
    if len(nodes) != 2:
        raise SystemExit("true resilient gate requires exactly two physical nodes")
    manager = os.environ["RESILIENT_E97_MANAGER_COMMAND"]
    trainer = os.environ["RESILIENT_E97_TRAINER_COMMAND"]
    children: list[Child] = []
    # The supervision state is deliberately node-local.  Keep its reader on
    # the same physical node as its writers and launch role processes directly
    # beneath one allocation step per node.  The children are subprocesses,
    # not nested sruns, so this does not depend on nested-step scheduling.
    launch_mode = os.environ.get("RESILIENT_E97_LAUNCH_MODE", "node-local")
    if launch_mode == "node-local":
        # The batch allocation already owns all eight GPUs on each node.  A
        # single two-node step inherits that device cgroup; asking Slurm for
        # ``--gpus-per-node`` again attempts a second GRES allocation and
        # leaves the step pending with "Requested nodes are busy".
        argv = ["srun", "--overlap", "--no-kill", "--exact", "-N2", "-n2",
                "--ntasks-per-node=1", "-c64",
                sys.executable, __file__, "--node-local"]
        return subprocess.call(argv)
    elif launch_mode == "independent-step":
        for node_rank, node in enumerate(nodes):
            children.append(Child("manager", node_rank, node, None, manager))
            children.extend(Child("trainer", node_rank, node, rank, trainer)
                            for rank in range(TRAINERS_PER_NODE))
    else:
        raise ValueError("RESILIENT_E97_LAUNCH_MODE must be node-local or independent-step")
    supervisor = AllocationSupervisor(
        run_dir, children,
        heartbeat_s=float(os.environ.get("RESILIENT_E97_HEARTBEAT_DEADLINE_S", "60")),
        progress_s=float(os.environ.get("RESILIENT_E97_PROGRESS_DEADLINE_S", "900")),
        max_restarts=int(os.environ.get("RESILIENT_E97_MAX_RESTARTS", "2")),
        startup_s=float(os.environ.get("RESILIENT_E97_STARTUP_DEADLINE_S", "120")),
    )
    signal.signal(signal.SIGTERM, lambda *_: setattr(supervisor, "stopping", True))
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
