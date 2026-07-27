#!/usr/bin/env python3
"""Supervise a bounded 2/4/8-node Frontier allocation without a job-wide step.

Each child owns an independent ``srun --no-kill`` step.  Failures remain
node-local, but the async-v2.1 model-owning lanes form one atomic recovery
cohort: a manager, service, or trainer failure fences and reconstructs all
eight local trainers together.  Role programs publish heartbeat/progress JSON
through the shared run directory; missing deadlines never turn a partial node
apply into next-generation READY.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.manifest_peer_control import (
    AllocationClaim,
    FenceRejected,
    ManifestPeerAuthority,
)
from ndm.native_dataplane import create_memfd, seal_memfd


TRAINERS_PER_NODE = 8
QUALIFICATION_NODE_LADDER = (2, 4, 8, 16, 32, 64, 256)
POOL_PROTOCOL_ID = "resilient-diloco-compute-pool-v1"
GENERATION_BASELINE_S = (212.0, 215.0)
READY_HARD_S = 180.0
K40_HARD_S = 420.0
EXCHANGE_COMMIT_HARD_S = 180.0
RESULT_PREPARATION_HARD_S = 420.0
BOUNDARY_RENDEZVOUS_HARD_S = 420.0
ALL_EIGHT_APPLY_HARD_S = 60.0
FIRST_COMMIT_HARD_S = 720.0


class AllocationFenceGuard:
    """Hold one immutable scheduler claim while native peers own live state."""

    def __init__(self, authority: ManifestPeerAuthority, claim: AllocationClaim):
        self.authority, self.claim = authority, claim
        self.error: BaseException | None = None

    def start(self) -> None:
        # Scheduler lifetime plus the persistent native peer protocol owns
        # liveness.  No filesystem heartbeat or renewal transaction exists.
        self.authority.assert_current(self.claim)

    def close(self) -> None:
        # Claims and commit receipts are immutable restart evidence.  A later
        # allocation supersedes this claim with a strictly larger scheduler
        # fence; deleting/releasing it would permit fence reuse.
        return None


def _scheduler_fence() -> int:
    """Return the monotonic scheduler token used by native peer identities."""
    explicit = os.environ.get("RESILIENT_E97_ALLOCATION_FENCE")
    if explicit:
        fence = int(explicit)
    else:
        scheduler_id = os.environ.get("SLURM_JOB_ID", "")
        match = re.match(r"^([0-9]+)", scheduler_id)
        if match is not None:
            fence = int(match.group(1))
        else:
            # Direct local protocol fixtures have no scheduler.  Their
            # explicit coordinator epoch is already part of every native
            # command/frame and remains deterministic.
            fence = int(os.environ.get("RESILIENT_E97_COORDINATOR_EPOCH", "1"))
    if fence <= 0:
        raise ValueError("allocation scheduler fence must be positive")
    return fence


def _allocation_admission(run_dir: Path) -> AllocationFenceGuard | None | bool:
    """Acquire before any manager/trainer can load a model.

    ``False`` means another live allocation owns the run and the caller must
    exit successfully without spawning roles. ``None`` is retained for unit
    fixtures that do not configure a logical run.
    """
    run_id = os.environ.get("RESILIENT_E97_RUN_ID")
    if not run_id:
        return None
    authority = ManifestPeerAuthority(run_dir)
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
    claim = authority.claim(
        run_id=run_id,
        allocation_id=allocation_id,
        incarnation=incarnation,
        fence=_scheduler_fence(),
        protocol_id=POOL_PROTOCOL_ID,
        config_id=config_id,
    )
    if claim is None:
        print(f"resilient_pool_admission=lost run_id={run_id}; exiting before model load")
        return False
    os.environ["RESILIENT_E97_ALLOCATION_CLAIM"] = claim.encode()
    os.environ["RESILIENT_E97_FENCE_EPOCH"] = str(claim.fence)
    admitted_at = time.time()
    os.environ["RESILIENT_E97_ALLOCATION_ADMITTED_AT"] = str(admitted_at)
    telemetry = run_dir / "supervision" / "allocation-fence.json"
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    temporary = telemetry.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "stage": "admitted_before_model_load", "run_id": run_id,
        "allocation_id": allocation_id, "incarnation": incarnation,
        "fence": claim.fence, "protocol": POOL_PROTOCOL_ID,
        "claim_digest": claim.claim_digest,
        "base_generation": claim.base_generation,
        "base_commit_digest": claim.base_commit_digest,
        "live_authority": "native_peer_memory",
        "restart_authority": "immutable_commit_receipt_chain",
        "generation_baseline_s": list(GENERATION_BASELINE_S),
        "ready_hard_s": READY_HARD_S, "k40_hard_s": K40_HARD_S,
        "exchange_commit_hard_s": EXCHANGE_COMMIT_HARD_S,
        "boundary_rendezvous_hard_s": BOUNDARY_RENDEZVOUS_HARD_S,
        "all_eight_apply_hard_s": ALL_EIGHT_APPLY_HARD_S,
        "first_commit_hard_s": FIRST_COMMIT_HARD_S,
    }, sort_keys=True) + "\n")
    os.replace(temporary, telemetry)
    guard = AllocationFenceGuard(authority, claim)
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
    incarnation: str = ""

    @property
    def identity(self) -> str:
        suffix = (self.role if self.local_rank is None
                  else f"trainer-{self.local_rank}")
        return f"node-{self.node_rank}-{suffix}"

    @property
    def global_rank(self) -> int | None:
        return (None if self.role != "trainer" or self.local_rank is None else
                self.node_rank * TRAINERS_PER_NODE + self.local_rank)


def _node_local_child_cpu_set(
    child: Child,
    available: tuple[int, ...] | list[int] | set[int] | None = None,
) -> set[int]:
    """Partition the existing 56-CPU node step across eight trainers.

    Node-local children are ordinary subprocesses, so without an explicit
    partition every trainer inherits the complete supervisor cpuset. Eight
    concurrent PyTorch host result-materialization passes would then each
    create a full-size thread pool and oversubscribe the same cores. Give every
    trainer seven disjoint CPUs; managers and the native service retain the
    complete cpuset so their compact work can progress beside the trainers.
    """
    cpus = tuple(sorted(
        os.sched_getaffinity(0) if available is None else available))
    if child.role != "trainer":
        return set(cpus)
    if child.local_rank is None or not 0 <= child.local_rank < TRAINERS_PER_NODE:
        raise ValueError("trainer CPU partition requires a valid local rank")
    required = TRAINERS_PER_NODE * 7
    if len(cpus) < required:
        raise RuntimeError(
            f"node-local trainer CPU partition requires {required} CPUs; "
            f"only {len(cpus)} are available")
    offset = child.local_rank * 7
    return set(cpus[offset:offset + 7])


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
        self._node_incarnations = {
            child.node_rank: uuid.uuid4().hex for child in children
        }
        self._cohort_restart_sequence = {
            node_rank: 0 for node_rank in self._node_incarnations
        }
        self.native_token_fd = -1
        if any(child.role == "native-service" for child in children):
            self.native_token_fd = create_memfd(
                "emender-ndp-admission", allow_sealing=True)
            os.write(self.native_token_fd, os.urandom(32))
            os.lseek(self.native_token_fd, 0, os.SEEK_SET)
            seal_memfd(self.native_token_fd)
        (run_dir / "supervision").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def node_incarnation(self, node_rank: int) -> str:
        """Return the one incarnation shared by a node's atomic role cohort."""
        try:
            return self._node_incarnations[int(node_rank)]
        except KeyError as error:
            raise ValueError("unknown node rank for atomic cohort") from error

    def _node_control_root(self, node_rank: int) -> Path:
        bulk_root = os.environ.get("RESILIENT_E97_BULK_ROOT")
        run_id = os.environ.get("RESILIENT_E97_RUN_ID")
        if bulk_root and run_id:
            return (Path(bulk_root) / run_id / f"node-{node_rank}" /
                    "control")
        return self.run_dir / "control"

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temporary, path)

    def _durable_generation(self) -> int | None:
        """Return the generation selected by the immutable receipt lineage."""
        try:
            authority = ManifestPeerAuthority(self.run_dir)
            encoded_claim = os.environ.get("RESILIENT_E97_ALLOCATION_CLAIM")
            claim = (
                AllocationClaim.decode(encoded_claim)
                if encoded_claim
                else authority.current_claim(os.environ.get("RESILIENT_E97_RUN_ID"))
            )
            if claim is None:
                return None
            commit = authority.current_commit(claim)
            if commit is None:
                return None
            return commit.generation
        except (FenceRejected, FileNotFoundError, KeyError, TypeError,
                ValueError, json.JSONDecodeError, OSError):
            return None

    def _apply_progress_time(self, child: Child, generation: int) -> float:
        """Observe immutable per-trainer apply receipts as manager progress.

        Redistribution/apply is intentionally pipelined across independent
        trainers.  The manager stage record changes only at stage boundaries,
        while each completed trainer publishes a checksum-validated receipt.
        Those monotonically accumulating receipts are real forward progress.
        """
        if child.role not in {"manager", "node-supervisor"}:
            return 0.0
        control = self._state_path(child).parent.parent / "control"
        newest = 0.0
        for receipt in control.glob(f"native-applied-{generation:08d}-*.json"):
            try:
                payload = json.loads(receipt.read_text())
                if (int(payload.get("generation", -1)) == generation
                        and (not os.environ.get("RESILIENT_E97_RUN_ID")
                             or payload.get("run_id")
                                == os.environ["RESILIENT_E97_RUN_ID"])
                        and (not child.incarnation
                             or payload.get("node_incarnation")
                                == self.node_incarnation(child.node_rank))):
                    newest = max(newest, receipt.stat().st_mtime)
            except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
        return newest

    def _event(self, event: str, child: Child, **extra: object) -> None:
        record = {"time": time.time(), "event": event, "identity": child.identity,
                  "role": child.role, "node_rank": child.node_rank,
                  "local_rank": child.local_rank, "global_rank": child.global_rank,
                  "process_incarnation": child.incarnation, **extra}
        with (self.run_dir / "supervision" / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def start(self, child: Child) -> None:
        child.incarnation = uuid.uuid4().hex
        role_values = {"RESILIENT_E97_ROLE": child.role,
                       "RESILIENT_E97_NODE_RANK": str(child.node_rank),
                       "RESILIENT_E97_PROCESS_INCARNATION": child.incarnation,
                       "RESILIENT_E97_NODE_INCARNATION":
                           self.node_incarnation(child.node_rank),
                       "RESILIENT_E97_COHORT_RESTART_SEQUENCE": str(
                           self._cohort_restart_sequence[child.node_rank]),
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
                                "RESILIENT_E97_GLOBAL_RANK": str(child.global_rank),
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
                                "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
                                "OMP_NUM_THREADS": "7",
                                "MKL_NUM_THREADS": "7",
                                "OPENBLAS_NUM_THREADS": "7",
                                "NUMEXPR_NUM_THREADS": "7"})
            role_env.extend([f"TRITON_CACHE_DIR={triton_cache}",
                             f"TORCHINDUCTOR_CACHE_DIR={inductor_cache}",
                             "OMP_NUM_THREADS=7",
                             "MKL_NUM_THREADS=7",
                             "OPENBLAS_NUM_THREADS=7",
                             "NUMEXPR_NUM_THREADS=7"])
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
        child_cpu_set: set[int] | None = None
        if self.launch_backend == "node-local-child":
            env = os.environ.copy(); env.update(role_values)
            if self.native_token_fd >= 0:
                env["EMENDER_NDP_ADMISSION_TOKEN_FD"] = str(self.native_token_fd)
            child_cpu_set = _node_local_child_cpu_set(child)
            env["RESILIENT_E97_CPUSET"] = ",".join(
                str(cpu) for cpu in sorted(child_cpu_set))
            argv = shlex.split(child.command)
            if child.role == "native-service":
                telemetry = self.run_dir / "logs" / f"{child.identity}-transport.jsonl"
                # CXI is selected by its explicit cxi0 domain. Passing the
                # Slurm hostname to fi_getinfo(FI_SOURCE) is a TCP-style bind
                # request and makes native CXI provider resolution fail before
                # endpoint creation. Test-only socket providers retain their
                # explicit local bind.
                if "--production" not in argv:
                    argv.extend(["--bind-node", child.node])
                argv.extend([
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
        if child.role == "native-service" and self.native_token_fd >= 0:
            # The service consumes the sealed admission token with read(2).
            # A replacement service inherits the same open-file description,
            # including its advanced offset, so rewind before every spawn.
            os.lseek(self.native_token_fd, 0, os.SEEK_SET)
        child.process = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr, env=env,
            start_new_session=True, pass_fds=pass_fds,
            preexec_fn=(
                None
                if child_cpu_set is None
                else lambda: os.sched_setaffinity(0, child_cpu_set)
            ))
        child.started_at = time.time()
        self._event(
            "started", child, pid=child.process.pid, restart=child.restarts,
            cpu_set=(
                None if child_cpu_set is None else sorted(child_cpu_set)))

    def _deadline_reason(self, child: Child, now: float) -> str | None:
        assert child.process is not None
        if child.process.poll() is not None:
            return f"exit:{child.process.returncode}"
        state_path = self._state_path(child)
        if not state_path.exists():
            if child.role == "native-service":
                return None
            if child.started_at is not None and now - child.started_at > self.startup_s:
                return "startup_deadline"
            return None
        state = json.loads(state_path.read_text())
        if (child.incarnation
                and state.get("process_incarnation") != child.incarnation):
            if (child.started_at is not None
                    and now - child.started_at > self.startup_s):
                return "startup_deadline"
            return None
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
            "native-service": "RESILIENT_E97_INJECT_NATIVE_SERVICE",
        }[child.role]
        injection = os.environ.get(injection_variable, "")
        if injection and child.identity not in self.injected:
            fields = injection.split(":")
            if len(fields) not in {3, 4}:
                raise ValueError(
                    "injection must be NODE:RANK:GENERATION[:STAGE]")
            node, rank, generation = map(int, fields[:3])
            requested_stage = fields[3] if len(fields) == 4 else None
            observed_stage = str(state.get("stage", "unknown"))
            finalized_stage = {
                "trainer": "applied", "manager": "published",
                "node-supervisor": "published",
            }.get(child.role)
            stage_matches = (observed_stage == requested_stage
                             if requested_stage is not None
                             else observed_stage == finalized_stage)
            matches = (child.node_rank == node and
                       ((child.role == "trainer" and child.local_rank == rank) or
                        (child.role in {"manager", "node-supervisor", "native-service"}
                         and rank == -1)))
            if (child.role == "native-service" and requested_stage is None):
                raise ValueError("native-service injection requires an exact owner stage")
            if (matches and stage_matches
                    and int(state.get("generation", -1)) >= generation):
                self.injected.add(child.identity)
                self._event("generation_gated_injection", child, generation=generation,
                            injection_class=child.role,
                            observed_generation=int(state.get("generation", -1)),
                            observed_stage=observed_stage)
                return ("injected_native_service_stage"
                        if child.role == "native-service"
                        else "injected_generation_gate")
        if child.role == "native-service":
            return None
        if now - float(state.get("heartbeat_time", 0)) > self.heartbeat_s:
            return "heartbeat_deadline"
        stage = str(state.get("stage", "unknown"))
        stage_budget = {
            "runtime_import": READY_HARD_S,
            "collecting": K40_HARD_S + EXCHANGE_COMMIT_HARD_S,
            "training": K40_HARD_S,
            "streaming_delta": EXCHANGE_COMMIT_HARD_S,
            "local_reduce_wait": K40_HARD_S,
            # This wait is the enclosing background path across result
            # readiness, rank-0 materialization, and its immutable checkpoint.
            # The constituent readiness/background-preparation/foreground
            # apply/checkpoint stages retain their 180/420/60/180-second
            # bounds.
            "leader_apply_wait": RESULT_PREPARATION_HARD_S,
            "result_preparation": RESULT_PREPARATION_HARD_S,
            "boundary_rendezvous": BOUNDARY_RENDEZVOUS_HARD_S,
            "peer_apply": ALL_EIGHT_APPLY_HARD_S,
            "submitted": EXCHANGE_COMMIT_HARD_S,
            "owner_transport": EXCHANGE_COMMIT_HARD_S,
            "freeze": EXCHANGE_COMMIT_HARD_S,
            "redistribution": EXCHANGE_COMMIT_HARD_S,
            "checkpoint_commit": EXCHANGE_COMMIT_HARD_S,
        }.get(stage, self.progress_s)
        progress_time = max(
            float(state.get("progress_time", 0)),
            self._apply_progress_time(child, int(state.get("generation", 0))),
        )
        if now - progress_time > min(self.progress_s, stage_budget):
            return "progress_deadline"
        admitted_at = float(os.environ.get("RESILIENT_E97_ALLOCATION_ADMITTED_AT", now))
        initial_generation = int(os.environ.get("RESILIENT_E97_INITIAL_GENERATION", "0"))
        # This is a hot diagnostic/liveness loop.  The manager publishes this
        # evidence only after native peer control agrees with the immutable
        # commit receipt.  Do not poll immutable manifests (and never a shared
        # control store) here. Receipt-chain reads are reserved for an actual
        # cohort/fresh-allocation recovery.
        peer_committed_generation = (
            int(state.get("authoritative_generation", initial_generation))
            if (
                child.role in {"manager", "node-supervisor"}
                and len(str(state.get("commit_receipt_digest", ""))) == 64
            )
            else initial_generation
        )
        if (child.role in {"manager", "node-supervisor"}
                and peer_committed_generation <= initial_generation
                and int(state.get("generation", initial_generation)) <= initial_generation
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
        if child.role in {"node-supervisor", "native-service"}:
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
                and (not child.incarnation
                     or (state.get("process_incarnation") == child.incarnation
                         and state.get("node_incarnation")
                            == self.node_incarnation(child.node_rank)))
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

    def _archive_incomplete_cohort(
            self, node_rank: int, failed_incarnation: str) -> Path:
        """Retain and remove volatile state from an incomplete node apply.

        The stopped cohort can no longer write these files.  Moving them under
        its fenced incarnation preserves causal evidence while ensuring a new
        manager/trainer cohort cannot consume a stale generation, submission,
        candidate result, apply marker, or per-rank recovery manifest.
        """
        control = self._node_control_root(node_rank)
        failed = control / "failed-cohorts" / failed_incarnation
        failed.mkdir(parents=True, exist_ok=True)
        patterns = (
            "native-generation-*.json",
            "native-layout-*.json",
            "native-submit-*.json",
            "native-result-*.json",
            "native-applied-*.json",
            "native-checkpoint-*.json",
            "native-manager-sync-*.json",
            "trainer-applied-*.json",
            "trainer-proposal-*.json",
            "leader-apply-release-*.json",
            "node-applied-*.json",
            "atomic-cohort-recovery.json",
        )
        for pattern in patterns:
            for source in sorted(control.glob(pattern)):
                if source.is_file():
                    os.replace(source, failed / source.name)
        recovery = control / "recovery"
        if recovery.is_dir():
            failed_recovery = failed / "recovery"
            failed_recovery.mkdir(parents=True, exist_ok=True)
            for source in sorted(recovery.glob("*.json")):
                if source.is_file():
                    os.replace(source, failed_recovery / source.name)
        return failed

    def _wait_restarted_service(self, service: Child) -> bool:
        socket_value = os.environ.get("EMENDER_NDP_SOCKET", "")
        if not socket_value:
            raise RuntimeError("native service restart requires its socket path")
        socket_path = Path(socket_value)
        socket_path.unlink(missing_ok=True)
        self.start(service)
        deadline = time.monotonic() + self.startup_s
        while time.monotonic() < deadline:
            if service.process is None or service.process.poll() is not None:
                return False
            if socket_path.is_socket():
                return True
            time.sleep(self.poll_s)
        return False

    def _restart_native_node_cohort(
            self, services: list[Child], manager: Child,
            trainers: list[Child], reason: str) -> bool:
        """Fail and reconstruct one fenced all-eight-trainer node cohort."""
        node_rank = manager.node_rank
        matches = [
            service for service in services if service.node_rank == node_rank
        ]
        local_trainers = sorted(
            (trainer for trainer in trainers
             if trainer.node_rank == node_rank),
            key=lambda trainer: int(trainer.local_rank),
        )
        if len(matches) != 1 or len(local_trainers) != TRAINERS_PER_NODE:
            raise RuntimeError(
                "atomic native restart requires one service and eight trainers")
        service = matches[0]
        cohort = [service, manager, *local_trainers]
        if any(child.restarts >= self.max_restarts for child in cohort):
            self._event(
                "cohort_restart_exhausted", manager, reason=reason,
                required_trainers=list(range(TRAINERS_PER_NODE)))
            return False

        authoritative_generation = self._durable_generation()
        if authoritative_generation is None:
            self._event(
                "cohort_restart_rejected", manager, reason=reason,
                rejection="authoritative_latest_unavailable")
            return False
        failed_incarnation = self.node_incarnation(node_rank)
        self.stop_children(cohort, f"atomic_node_apply:{reason}")
        failed_root = self._archive_incomplete_cohort(
            node_rank, failed_incarnation)

        self._cohort_restart_sequence[node_rank] += 1
        new_incarnation = uuid.uuid4().hex
        self._node_incarnations[node_rank] = new_incarnation
        for child in cohort:
            child.restarts += 1
        control = self._node_control_root(node_rank)
        recovery_path = control / "atomic-cohort-recovery.json"
        recovery = {
            "schema": "emender-async-v21-atomic-cohort-recovery-v1",
            "run_id": os.environ.get("RESILIENT_E97_RUN_ID", ""),
            "allocation_fence": int(os.environ.get(
                "RESILIENT_E97_FENCE_EPOCH", "0")),
            "node_rank": node_rank,
            "failed_incarnation": failed_incarnation,
            "node_incarnation": new_incarnation,
            "restart_sequence": self._cohort_restart_sequence[node_rank],
            "authoritative_generation": authoritative_generation,
            "required_trainers": list(range(TRAINERS_PER_NODE)),
            "reason": reason,
            "failed_evidence": str(failed_root),
            "status": "reconstructing",
        }
        self._atomic_json(recovery_path, recovery)
        self._event(
            "atomic_cohort_failed", manager, reason=reason,
            failed_incarnation=failed_incarnation,
            node_incarnation=new_incarnation,
            authoritative_generation=authoritative_generation,
            required_trainers=list(range(TRAINERS_PER_NODE)))

        if not self._wait_restarted_service(service):
            return False
        self.start(manager)
        deadline = time.monotonic() + self.startup_s
        while time.monotonic() < deadline:
            if manager.process is None or manager.process.poll() is not None:
                return False
            if self._manager_ready_for_trainers(manager):
                break
            time.sleep(self.poll_s)
        else:
            return False

        recovery["status"] = "reconstructed"
        self._atomic_json(recovery_path, recovery)
        for trainer in local_trainers:
            self.start(trainer)
        self._event(
            "atomic_cohort_reconstructed", manager, reason=reason,
            failed_incarnation=failed_incarnation,
            node_incarnation=new_incarnation,
            authoritative_generation=authoritative_generation,
            trainer_incarnations={
                str(trainer.local_rank): trainer.incarnation
                for trainer in local_trainers
            })
        return True

    def _restart_native_service_for_manager(
            self, services: list[Child], manager: Child, reason: str) -> bool:
        """Compatibility helper for startup-only manager rejoin fixtures."""
        matches = [service for service in services
                   if service.node_rank == manager.node_rank]
        if not matches:
            return True
        if len(matches) != 1:
            raise RuntimeError("manager must own exactly one native service")
        service = matches[0]
        self.stop_child(service, f"manager_rejoin:{reason}")
        service.restarts += 1
        self._event(
            "native_service_rejoin", service,
            manager_identity=manager.identity, manager_restart=manager.restarts + 1,
            reason=reason, service_restart=service.restarts)
        return self._wait_restarted_service(service)

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
        managers = [child for child in monitored if child.role == "manager"]
        trainers = [child for child in monitored if child.role == "trainer"]
        completed: set[str] = set()
        while not self.stopping:
            now = time.time()
            for child in services:
                reason = self._deadline_reason(child, now)
                if reason == "injected_native_service_stage":
                    self.stop_child(child, reason)
            failed_services = [
                child for child in services
                if child.process is not None and child.process.poll() is not None
            ]
            for service in failed_services:
                local_managers = [
                    manager for manager in managers
                    if manager.node_rank == service.node_rank
                ]
                if (self.launch_backend != "node-local-child"
                        or len(local_managers) != 1
                        or local_managers[0].identity in completed
                        or not self._restart_native_node_cohort(
                            services, local_managers[0], trainers,
                            "native_service_lost")):
                    self.stop_children(
                        [child for child in monitored
                         if child.identity not in completed],
                        "native_service_lost")
                    return 1
                for child in (
                        local_managers
                        + [trainer for trainer in trainers
                           if trainer.node_rank == service.node_rank]):
                    completed.discard(child.identity)
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
                if (self.launch_backend == "node-local-child"
                        and services
                        and child.role in {"manager", "trainer"}):
                    local_managers = [
                        manager for manager in managers
                        if manager.node_rank == child.node_rank
                    ]
                    if (len(local_managers) != 1
                            or not self._restart_native_node_cohort(
                                services, local_managers[0], trainers, reason)):
                        self._event(
                            "restart_exhausted", child, reason=reason,
                            restart_scope="atomic_eight_trainer_node")
                        return 1
                    for local in (
                            local_managers
                            + [trainer for trainer in trainers
                               if trainer.node_rank == child.node_rank]):
                        completed.discard(local.identity)
                    continue
                self.stop_child(child, reason)
                if child.restarts >= self.max_restarts:
                    self._event("restart_exhausted", child, reason=reason)
                    if child.role == "trainer":
                        # Rank retirement is contribution membership only.  It
                        # must not poison its model-free manager, native
                        # service, seven sibling ranks, or the Slurm node step.
                        completed.add(child.identity)
                        self._event("rank_retired", child, reason=reason,
                                    decision="ineligible",
                                    revocation="lease_incarnation_revoked")
                        continue
                    return 1
                if (child.role == "manager"
                        and not self._restart_native_service_for_manager(
                            services, child, reason)):
                    self._event("native_service_rejoin_failed", child, reason=reason)
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
    node_count = int(os.environ.get("RESILIENT_E97_NODE_COUNT", "2"))
    if node_count not in QUALIFICATION_NODE_LADDER:
        raise SystemExit(
            "async-decoupled-v2.1 qualification requires an exact serial-ladder "
            "node count")
    if node_count >= 4:
        if os.environ.get("ASYNC_V21_GATE") != "scale":
            raise SystemExit(
                "async-decoupled-v2.1 scale allocation requires the scale gate")
        required_scale_bindings = (
            "ASYNC_V21_SCALE_AUTHORIZATION",
            "ASYNC_V21_SCALE_AUTHORIZATION_DIGEST",
            "ASYNC_V21_PRIOR_RUNG_PASS",
            "ASYNC_V21_PRIOR_RUNG_PASS_DIGEST",
            "ASYNC_V21_SCALE_CLOSURE_DIGEST",
            "ASYNC_V21_SCALE_CLOSE_OFFSET_NS",
            "ASYNC_V21_SCALE_STABLE_DIVERSITY_FLOOR",
            "ASYNC_V21_SCALE_PER_READY_WORKER_TOKEN_FLOOR",
        )
        missing = tuple(
            name for name in required_scale_bindings if not os.environ.get(name))
        if missing:
            raise SystemExit(
                "async-decoupled-v2.1 scale allocation is missing immutable "
                f"authorization/closure bindings: {', '.join(missing)}")
    if len(nodes) != node_count:
        raise SystemExit("Slurm allocation differs from explicit resilient-pool capacity")
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
        # single bounded multi-node step inherits that device cgroup; asking Slurm for
        # ``--gpus-per-node`` again attempts a second GRES allocation and
        # leaves the step pending with "Requested nodes are busy".
        argv = ["srun", "--overlap", "--no-kill", "--exact",
                f"-N{node_count}", f"-n{node_count}",
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
        if isinstance(admission, AllocationFenceGuard) and admission.error is not None:
            raise FenceRejected(f"allocation fence failed: {admission.error}")
        return result
    finally:
        if isinstance(admission, AllocationFenceGuard):
            admission.close()


if __name__ == "__main__":
    sys.exit(main())
