#!/usr/bin/env python3
"""Supervise the 2-manager/16-trainer Frontier allocation without a job-wide step.

Each child owns an independent ``srun --no-kill`` step.  A failed child is
restarted without waiting for, signalling, or recreating healthy siblings.
Role programs publish heartbeat/progress JSON through the shared run directory;
missing deadlines cause only that role's process group to be evicted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.fenced_admission import AllocationLease, FenceRejected, SQLiteFencedControlStore
from ndm.native_dataplane import create_memfd, seal_memfd


TRAINERS_PER_NODE = 8
POOL_PROTOCOL_ID = "resilient-diloco-compute-pool-v1"
GENERATION_BASELINE_S = (212.0, 215.0)
READY_HARD_S = 180.0
K40_HARD_S = 420.0
EXCHANGE_COMMIT_HARD_S = 180.0
FIRST_COMMIT_HARD_S = 720.0


class AllocationLeaseGuard:
    """Renew the sole allocation fence while roles run beneath this process."""

    def __init__(self, store: SQLiteFencedControlStore, lease: AllocationLease, *,
                 ttl_s: float, renew_s: float):
        if not 0 < renew_s < ttl_s:
            raise ValueError("lease renewal interval must be positive and below TTL")
        self.store, self.lease = store, lease
        self.ttl_s, self.renew_s = float(ttl_s), float(renew_s)
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._renew, name="allocation-fence-renewal",
                                       daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _renew(self) -> None:
        while not self.stop.wait(self.renew_s):
            try:
                self.lease = self.store.renew(self.lease, ttl_s=self.ttl_s)
            except BaseException as error:
                self.error = error
                self.stop.set()
                return

    def close(self) -> None:
        self.stop.set()
        self.thread.join(self.renew_s + 1)
        if self.error is None:
            try:
                self.store.release(self.lease)
            except FenceRejected:
                pass


def _allocation_admission(run_dir: Path) -> AllocationLeaseGuard | None | bool:
    """Acquire before any manager/trainer can load a model.

    ``False`` means another live allocation owns the run and the caller must
    exit successfully without spawning roles. ``None`` is retained for unit
    fixtures that do not configure a logical run.
    """
    run_id = os.environ.get("RESILIENT_E97_RUN_ID")
    if not run_id:
        return None
    database = Path(os.environ.get(
        "RESILIENT_E97_FENCE_DB", str(run_dir / "control" / "pool-v1.sqlite3")))
    store = SQLiteFencedControlStore(
        database, timeout_s=float(os.environ.get("RESILIENT_E97_CONTROL_TIMEOUT_S", "5")))
    allocation_id = os.environ.get(
        "RESILIENT_E97_ALLOCATION_ID", os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}"))
    incarnation = os.environ.get("RESILIENT_E97_ALLOCATION_INCARNATION", uuid.uuid4().hex)
    config_material = {
        "source": os.environ.get("RESILIENT_E97_SOURCE_ID", "unknown"),
        "payload": os.environ.get("RESILIENT_E97_PAYLOAD_ID", "unknown"),
        "code": os.environ.get("RESILIENT_E97_CODE_ID", "unknown"),
        "manager": os.environ.get("RESILIENT_E97_MANAGER_COMMAND", ""),
        "trainer": os.environ.get("RESILIENT_E97_TRAINER_COMMAND", ""),
        "dataplane_backend": os.environ.get("DILOCO_DATAPLANE", "python-tcp-debug"),
        "native_launch_attestation": os.environ.get("NDP_LAUNCH_ATTESTATION", ""),
    }
    config_id = hashlib.sha256(json.dumps(
        config_material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ttl_s = float(os.environ.get("RESILIENT_E97_LEASE_TTL_S", "60"))
    lease = store.acquire(
        run_id=run_id, allocation_id=allocation_id, incarnation=incarnation,
        protocol_id=POOL_PROTOCOL_ID, config_id=config_id, ttl_s=ttl_s)
    if lease is None:
        print(f"resilient_pool_admission=lost run_id={run_id}; exiting before model load")
        return False
    os.environ["RESILIENT_E97_FENCE_DB"] = str(database)
    os.environ["RESILIENT_E97_ALLOCATION_LEASE"] = json.dumps(lease.__dict__, sort_keys=True)
    os.environ["RESILIENT_E97_FENCE_EPOCH"] = str(lease.fence)
    os.environ["RESILIENT_E97_ALLOCATION_ADMITTED_AT"] = str(lease.acquired_at)
    telemetry = run_dir / "supervision" / "allocation-lease.json"
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    temporary = telemetry.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "stage": "admitted_before_model_load", "run_id": run_id,
        "allocation_id": allocation_id, "incarnation": incarnation,
        "fence": lease.fence, "protocol": POOL_PROTOCOL_ID,
        "generation_baseline_s": list(GENERATION_BASELINE_S),
        "ready_hard_s": READY_HARD_S, "k40_hard_s": K40_HARD_S,
        "exchange_commit_hard_s": EXCHANGE_COMMIT_HARD_S,
        "first_commit_hard_s": FIRST_COMMIT_HARD_S,
        "lease_ttl_s": ttl_s,
    }, sort_keys=True) + "\n")
    os.replace(temporary, telemetry)
    guard = AllocationLeaseGuard(
        store, lease, ttl_s=ttl_s,
        renew_s=float(os.environ.get("RESILIENT_E97_LEASE_RENEW_S", "10")))
    guard.start()
    return guard


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


def _retain_node_evidence(run_dir: Path, *, bulk_root: Path, run_id: str,
                          node_rank: int) -> Path:
    """Atomically retain only small node-local control evidence after roles stop.

    Tensor mailboxes, recovery checkpoints, and caches remain excluded from the
    shared filesystem. JSON/JSONL supervision, stage telemetry, and compact
    generation/control manifests are sufficient to audit deadlines, bytes,
    high-water/release, membership, and failure/rejoin behavior.
    """
    source_root = bulk_root / run_id / f"node-{node_rank}"
    retained = run_dir / "retained-evidence" / f"node-{node_rank}"
    for tree in ("supervision", "telemetry", "control"):
        source_tree = source_root / tree
        if not source_tree.is_dir():
            continue
        for source in source_tree.rglob("*"):
            if not source.is_file() or source.suffix not in {".json", ".jsonl"}:
                continue
            target = retained / tree / source.relative_to(source_tree)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
    retained.mkdir(parents=True, exist_ok=True)
    snapshot = retained / "snapshot.json"
    temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "schema": 1, "run_id": run_id, "node_rank": node_rank,
        "source": str(source_root), "retained_at": time.time(),
        "included": ["supervision/**/*.json", "telemetry/**/*.jsonl",
                     "control/**/*.json", "control/**/*.jsonl"],
        "excluded": ["mailbox", "*.data", "*.pt", "kernel-cache"],
    }, sort_keys=True) + "\n")
    os.replace(temporary, snapshot)
    return retained


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
        self._last_shared_stage: dict[str, tuple[int, int, str]] = {}
        self.native_token_fd = -1
        if any(child.role == "native-service" for child in children):
            self.native_token_fd = create_memfd(
                "emender-ndp-admission", allow_sealing=True)
            os.write(self.native_token_fd, os.urandom(32))
            os.lseek(self.native_token_fd, 0, os.SEEK_SET)
            seal_memfd(self.native_token_fd)
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
        if child.role in {"manager", "native-service"}:
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
            # Eight cold E97 trainers sharing the default home-directory
            # kernel caches serialize on Triton/Inductor locks. On Frontier
            # that kept every trainer inside its first optimizer step until the
            # bounded generation deadline. Match the established production
            # launchers: isolate caches per local trainer and keep kernel-build
            # traffic on node-local storage.
            cache_root = Path(os.environ.get(
                "RESILIENT_E97_KERNEL_CACHE_ROOT", "/tmp/resilient-e97-kernel-cache"))
            cache_identity = (
                f"{os.environ.get('RESILIENT_E97_RUN_ID', 'unknown')}-rank-{child.local_rank}")
            triton_cache = cache_root / cache_identity / "triton"
            inductor_cache = cache_root / cache_identity / "inductor"
            triton_cache.mkdir(parents=True, exist_ok=True)
            inductor_cache.mkdir(parents=True, exist_ok=True)
            role_values.update({"TRITON_CACHE_DIR": str(triton_cache),
                                "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache)})
            role_env.extend([f"TRITON_CACHE_DIR={triton_cache}",
                             f"TORCHINDUCTOR_CACHE_DIR={inductor_cache}"])
        else:
            # The batch allocation already owns all eight GCDs on each node.
            # Frontier GRES cannot be allocated again to overlapping steps:
            # doing so leaves both node-supervisor steps pending with
            # "Requested nodes are busy".  Reuse the allocation's device
            # cgroup; direct children bind ROCR_VISIBLE_DEVICES=0..7.
            # Frontier exposes 56 allocatable CPU cores per node to Slurm.
            # A 64-CPU step is unsatisfiable and remains pending as
            # "Requested nodes are busy" until the allocation expires.
            resources = ["-c56"]
        if self.launch_backend == "node-local-child":
            env = os.environ.copy(); env.update(role_values)
            if self.native_token_fd >= 0:
                env["EMENDER_NDP_ADMISSION_TOKEN_FD"] = str(self.native_token_fd)
            argv = shlex.split(child.command)
            if child.role == "native-service":
                telemetry = self.run_dir / "logs" / f"{child.identity}-transport.jsonl"
                argv.extend([
                    "--bind-node", child.node,
                    "--telemetry", str(telemetry),
                    "--admission-token-fd", str(self.native_token_fd),
                ])
        else:
            env = None
            argv = ["srun", "--overlap", "--no-kill", "--exact", "-N1", "-n1",
                    "-w", child.node, *resources, "env", *role_env, *shlex.split(child.command)]
        log = self.run_dir / "logs" / child.identity
        stdout = (log.with_suffix(".out")).open("ab", buffering=0)
        stderr = (log.with_suffix(".err")).open("ab", buffering=0)
        pass_fds = ((self.native_token_fd,) if self.launch_backend == "node-local-child"
                    and self.native_token_fd >= 0 else ())
        child.process = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr, env=env,
            start_new_session=True, pass_fds=pass_fds)
        child.started_at = time.time()
        self._event("started", child, pid=child.process.pid, restart=child.restarts)

    def _deadline_reason(self, child: Child, now: float) -> str | None:
        assert child.process is not None
        if child.process.poll() is not None:
            return f"exit:{child.process.returncode}"
        if child.role == "native-service":
            return None
        state_path = self._state_path(child)
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
        stage = str(state.get("stage", "unknown"))
        stage_budget = {
            "runtime_import": READY_HARD_S,
            "collecting": K40_HARD_S + EXCHANGE_COMMIT_HARD_S,
            "training": K40_HARD_S,
            "streaming_delta": EXCHANGE_COMMIT_HARD_S,
            "local_reduce_wait": K40_HARD_S,
            "leader_apply_wait": EXCHANGE_COMMIT_HARD_S,
            "peer_apply": EXCHANGE_COMMIT_HARD_S,
            "submitted": EXCHANGE_COMMIT_HARD_S,
            "owner_transport": EXCHANGE_COMMIT_HARD_S,
            "freeze": EXCHANGE_COMMIT_HARD_S,
            "redistribution": EXCHANGE_COMMIT_HARD_S,
            "checkpoint_commit": EXCHANGE_COMMIT_HARD_S,
        }.get(stage, self.progress_s)
        if now - float(state.get("progress_time", 0)) > min(self.progress_s, stage_budget):
            return "progress_deadline"
        admitted_at = float(os.environ.get("RESILIENT_E97_ALLOCATION_ADMITTED_AT", now))
        initial_generation = int(os.environ.get("RESILIENT_E97_INITIAL_GENERATION", "0"))
        if (int(state.get("generation", initial_generation)) <= initial_generation
                and now - admitted_at > FIRST_COMMIT_HARD_S):
            return "first_atomic_generation_deadline"
        return None

    def _state_path(self, child: Child) -> Path:
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
        return state_path

    def _manager_ready_for_trainers(self, child: Child) -> bool:
        """A manager is READY after import and fenced pool admission."""
        state_path = self._state_path(child)
        try:
            state = json.loads(state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        return (state.get("identity") == child.identity
                and str(state.get("stage", "runtime_import")) != "runtime_import")

    def _share_stage_transition(self, child: Child) -> None:
        """Retain one small shared record per role/generation stage.

        Heartbeats and tensor traffic stay node local.  This deliberately
        excludes step and heartbeat changes so the live SLO surface is bounded
        by protocol transitions rather than polling frequency or tensor count.
        """
        try:
            state = json.loads(self._state_path(child).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if state.get("identity") != child.identity:
            return
        stage = str(state.get("stage", "unknown"))
        generation = int(state.get("generation", 0))
        transition = (child.restarts, generation, stage)
        if self._last_shared_stage.get(child.identity) == transition:
            return
        self._last_shared_stage[child.identity] = transition
        self._event(
            "role_stage", child, restart=child.restarts, stage=stage,
            generation=generation,
            progress_time=float(state.get("progress_time", 0)),
            heartbeat_time=float(state.get("heartbeat_time", 0)),
        )

    def stop_children(self, children: list[Child], reason: str, *,
                      grace_s: float = 15.0, kill_grace_s: float = 2.0) -> None:
        """Stop every role under one shared grace bound, independent of count."""
        active = [child for child in children
                  if child.process is not None and child.process.poll() is None]
        for child in active:
            try:
                os.killpg(child.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + max(0.0, grace_s)
        while active and time.monotonic() < deadline:
            active = [child for child in active if child.process.poll() is None]
            if active:
                time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))
        for child in active:
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_deadline = time.monotonic() + max(0.0, kill_grace_s)
        while active and time.monotonic() < kill_deadline:
            active = [child for child in active if child.process.poll() is None]
            if active:
                time.sleep(min(self.poll_s,
                               max(0.0, kill_deadline - time.monotonic())))
        for child in children:
            if child.process is not None:
                self._event("evicted", child, reason=reason,
                            exit_code=child.process.poll())

    def stop_child(self, child: Child, reason: str) -> None:
        assert child.process is not None
        self.stop_children([child], reason)

    def run(self) -> int:
        if self.launch_backend == "node-local-child":
            services = [child for child in self.children
                        if child.role == "native-service"]
            managers = [child for child in self.children if child.role == "manager"]
            deferred = [child for child in self.children if child.role == "trainer"]
            for child in services:
                self.start(child)
            service_deadline = time.monotonic() + self.startup_s
            socket_path = Path(os.environ.get("EMENDER_NDP_SOCKET", ""))
            while services and not self.stopping:
                if all(child.process is not None and child.process.poll() is None
                       for child in services) and socket_path.is_socket():
                    break
                if any(child.process is not None and child.process.poll() is not None
                       for child in services):
                    self.stop_children(services, "native_service_startup_failure")
                    return 1
                if time.monotonic() >= service_deadline:
                    self.stop_children(services, "native_service_startup_deadline")
                    return 1
                time.sleep(self.poll_s)
            for child in managers:
                self.start(child)
            while not self.stopping:
                for child in managers:
                    self._share_stage_transition(child)
                if all(self._manager_ready_for_trainers(child) for child in managers):
                    break
                now = time.time()
                for child in managers:
                    if self._manager_ready_for_trainers(child):
                        continue
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
            if self.stopping:
                self.stop_children([*managers, *services], "allocation_term_handoff")
                return 0
            # Node-local role heartbeats deliberately stay off Lustre's hot
            # path.  Publish one compact shared attestation after every local
            # manager has crossed fenced pool READY and before cold trainers
            # are admitted.  Operators can then enforce the allocation-wide
            # READY SLO without probing compute-node /tmp or inferring it from
            # buffered role stdout.
            for child in managers:
                state = json.loads(self._state_path(child).read_text())
                self._event(
                    "manager_ready", child,
                    stage=str(state.get("stage", "unknown")),
                    generation=int(state.get("generation", 0)),
                    ready_progress_time=float(state.get("progress_time", 0)),
                    ready_heartbeat_time=float(state.get("heartbeat_time", 0)),
                )
            for child in deferred:
                self.start(child)
        else:
            for child in self.children:
                self.start(child)
        monitored = [child for child in self.children if child.role != "native-service"]
        services = [child for child in self.children if child.role == "native-service"]
        completed: set[str] = set()
        while not self.stopping:
            now = time.time()
            if any(child.process is not None and child.process.poll() is not None
                   for child in services):
                self.stop_children(
                    [child for child in monitored if child.identity not in completed],
                    "native_service_lost")
                return 1
            for child in monitored:
                if child.identity in completed:
                    continue
                self._share_stage_transition(child)
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
            if len(completed) == len(monitored):
                self.stop_children(services, "allocation_complete")
                return 0
            time.sleep(self.poll_s)
        self.stop_children(
            [child for child in self.children if child.identity not in completed],
            "allocation_term_handoff")
        return 0


def _node_local_main() -> int:
    run_dir = Path(os.environ["RUN_DIR"])
    # The one-task-per-node srun supplies SLURM_NODEID.  The explicit role
    # variable is present only when this entrypoint is launched as a Child by
    # an outer supervisor, so do not require it in the default live mode.
    node_rank_text = os.environ.get("RESILIENT_E97_NODE_RANK")
    if node_rank_text is None:
        node_rank_text = os.environ["SLURM_NODEID"]
    node_rank = int(node_rank_text)
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    manager = os.environ["RESILIENT_E97_MANAGER_COMMAND"]
    trainer = os.environ["RESILIENT_E97_TRAINER_COMMAND"]
    service = os.environ.get("NDP_SERVICE_COMMAND")
    bulk_root = Path(os.environ.get("RESILIENT_E97_BULK_ROOT", "/tmp/resilient-e97"))
    children = []
    if service:
        run_id = os.environ["RESILIENT_E97_RUN_ID"]
        socket_path = bulk_root / run_id / f"node-{node_rank}" / "control" / "ndp.sock"
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["EMENDER_NDP_SOCKET"] = str(socket_path)
        service = f"{service} --socket {shlex.quote(str(socket_path))}"
        children.append(Child("native-service", node_rank, node, None, service))
    children.append(Child("manager", node_rank, node, None, manager))
    children.extend(Child("trainer", node_rank, node, rank, trainer)
                    for rank in range(TRAINERS_PER_NODE))
    supervisor = AllocationSupervisor(
        run_dir, children,
        heartbeat_s=float(os.environ.get("RESILIENT_E97_HEARTBEAT_DEADLINE_S", "60")),
        progress_s=float(os.environ.get("RESILIENT_E97_PROGRESS_DEADLINE_S", "600")),
        max_restarts=int(os.environ.get("RESILIENT_E97_MAX_RESTARTS", "2")),
        startup_s=float(os.environ.get("RESILIENT_E97_STARTUP_DEADLINE_S", "120")),
        launch_backend="node-local-child")
    signal.signal(signal.SIGTERM, lambda *_: setattr(supervisor, "stopping", True))
    try:
        return supervisor.run()
    finally:
        run_id = os.environ.get("RESILIENT_E97_RUN_ID")
        bulk_root = os.environ.get("RESILIENT_E97_BULK_ROOT")
        if run_id and bulk_root:
            _retain_node_evidence(
                run_dir, bulk_root=Path(bulk_root), run_id=run_id,
                node_rank=node_rank)


def _allocation_main() -> int:
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
                "--ntasks-per-node=1", "-c56",
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
        progress_s=float(os.environ.get("RESILIENT_E97_PROGRESS_DEADLINE_S", "600")),
        max_restarts=int(os.environ.get("RESILIENT_E97_MAX_RESTARTS", "2")),
        startup_s=float(os.environ.get("RESILIENT_E97_STARTUP_DEADLINE_S", "120")),
    )
    signal.signal(signal.SIGTERM, lambda *_: setattr(supervisor, "stopping", True))
    return supervisor.run()


def main() -> int:
    if "--node-local" in sys.argv:
        return _node_local_main()
    run_dir = Path(os.environ["RUN_DIR"])
    admission = _allocation_admission(run_dir)
    if admission is False:
        return 0
    try:
        result = _allocation_main()
        if isinstance(admission, AllocationLeaseGuard) and admission.error is not None:
            raise FenceRejected(f"allocation lease renewal failed: {admission.error}")
        return result
    finally:
        if isinstance(admission, AllocationLeaseGuard):
            admission.close()


if __name__ == "__main__":
    sys.exit(main())
