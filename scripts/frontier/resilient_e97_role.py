#!/usr/bin/env python3
"""Real manager/trainer entrypoints for the split resilient E97 launcher."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Mapping
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _role_import_heartbeat() -> tuple[threading.Event, threading.Thread] | None:
    """Publish liveness, but no generation progress, during heavy imports."""
    if len(sys.argv) < 2 or sys.argv[1] not in {"manager", "trainer"}:
        return None
    run_id = os.environ.get("RESILIENT_E97_RUN_ID")
    bulk_root = os.environ.get("RESILIENT_E97_BULK_ROOT")
    node_rank = os.environ.get("RESILIENT_E97_NODE_RANK", "0")
    if not run_id or not bulk_root:
        return None
    role = sys.argv[1]
    local_rank = os.environ.get("RESILIENT_E97_LOCAL_RANK")
    if role == "trainer" and local_rank is None:
        return None
    identity = (f"node-{node_rank}-manager" if role == "manager"
                else f"node-{node_rank}-trainer-{local_rank}")
    state = (Path(bulk_root) / run_id / f"node-{node_rank}" / "supervision" /
             f"{identity}.json")
    stop = threading.Event()
    progress_started = time.time()

    def publish() -> None:
        state.parent.mkdir(parents=True, exist_ok=True)
        while not stop.is_set():
            now = time.time()
            temporary = state.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({
                "identity": identity, "heartbeat_time": now,
                "progress_time": progress_started,
                "generation": 0, "step": 0, "loss": None,
                "stage": "runtime_import", "bootstrap_pid": os.getpid(),
                "process_incarnation": os.environ.get(
                    "RESILIENT_E97_PROCESS_INCARNATION", ""),
                "node_incarnation": os.environ.get(
                    "RESILIENT_E97_NODE_INCARNATION", ""),
                "cohort_restart_sequence": int(os.environ.get(
                    "RESILIENT_E97_COHORT_RESTART_SEQUENCE", "0")),
            }, sort_keys=True))
            os.replace(temporary, state)
            stop.wait(5)

    thread = threading.Thread(target=publish, name=f"{role}-import-heartbeat", daemon=True)
    thread.start()
    return stop, thread


_IMPORT_HEARTBEAT = _role_import_heartbeat()

import numpy as np
import torch

from ndm.resilient_e97_roles import LocalFence, LocalTrainerSpool
from ndm.native_artifacts import (
    NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG, attest_launch, validate_backend,
)
from ndm.native_dataplane import DType, copy_fd_range, create_memfd, seal_memfd
from ndm.native_e97_runtime import (
    GenerationMetadata, NativeTrainerDataPlane, atomic_metadata,
    decode_credit_frame_fd, decode_owner_frame_fd, encode_credit_frame_fd,
    encode_owner_frame_fd, fd_sha256, layout_identity, runtime_digests, state_digest,
    state_elements, wait_metadata,
)
from ndm.native_pool_runtime import NativeManagerSession
from ndm.async_diloco_v2 import (
    ASYNC_DECOUPLED_V21, AsyncV21DescriptorService, AsyncV21WorkerLane,
    AtomicEightTrainerApply, SafeBoundaryRendezvous,
    OuterState, ResultEnvelope, ScheduleFreeLocalState,
)
from ndm.resilient_e97_reducer import TensorLayout
from ndm.manifest_peer_control import (
    AllocationClaim,
    CommitReceipt,
    ManifestPeerAuthority,
)
from ndm.resilient_e97_runtime import (
    SplitManagerLoop, apply_delta, apply_delta_with_correction_ledger,
    atomic_json, assert_node_local_path, finalize_checkpoint, flatten_tensors,
    heartbeat, outer_state_migration,
)
from ndm.resilient_pool_runtime import (
    DistributedOwnerServer, OwnerEndpoint, PoolControlClient, PoolControlConfig,
    PoolControlServer, PoolStageSLO, chunk_manifest_digest, contribution_id,
    fetch_owned_shards, live_owner_endpoints, submit_owned_shards,
)


ASYNC_V21_E97_NATIVE_RESIDENT_BYTES = 64_001_671_648
ASYNC_V21_SNAPSHOT_ADMISSION_S = 1.0
ASYNC_V21_BOUNDARY_RENDEZVOUS_S = 420.0
ASYNC_V21_ALL_EIGHT_APPLY_S = 60.0
ASYNC_V21_APPLY_RECEIPT_PUBLICATION_S = 10.0

_CAUSAL_PHASE_BY_STAGE = {
    "async_v21_endpoint_snapshot": "freeze_snapshot",
    "async_v21_snapshot_admission": "snapshot_admission",
    "native_direct_memfd": "publish_network",
    "native_owner_contribution": "publish_network",
    "native_owner_redistribution": "publish_network",
    "native_local_reduction": "aggregation",
    "async_v21_checkpoint_write": "checkpoint",
    "async_v21_checkpoint_hash": "checkpoint",
    "checkpoint_publication": "checkpoint",
    "async_v21_result_readiness": "result_wait",
    "native_trainer_apply": "apply_swap",
    "native_node_apply_swap": "apply_swap",
}


def _causal_work_id(identity: str, generation: int) -> str:
    return hashlib.sha256(
        f"{identity}|{int(generation)}".encode("utf-8")).hexdigest()


def production_overlap_probe(*, background_release: threading.Event,
                             background_started: threading.Event,
                             timeout_s: float = 2.0) -> list[object]:
    """Exercise the rendered role's actual v2 ownership/continuation edge."""
    digest = "0" * 64
    lane = AsyncV21WorkerLane(
        run_id="production-overlap-probe", fence=1,
        worker_id="node-0", incarnation="production-overlap-probe",
        local=ScheduleFreeLocalState(
            x=np.asarray([0.0]), parameter_points={"z": np.asarray([0.0])}),
        anchor_version=0, anchor_state=np.asarray([0.0]),
        anchor_digest=digest, layout_digest="1" * 64, code_digest="2" * 64,
        policy=ASYNC_DECOUPLED_V21)
    lane.finish_window(
        np.asarray([1.0]), exact_tokens=1, begin_ns=1, end_ns=2)
    contribution = lane.seal()
    events: list[object] = []
    service = AsyncV21DescriptorService(lane=lane, telemetry=events.append)

    def certify(_contribution, phase):
        phase("discovery_membership_quorum_start")
        background_started.set()
        if not background_release.wait(timeout_s):
            raise TimeoutError("probe background release timed out")
        phase("checkpoint_publication")
        return ResultEnvelope.create(
            run_id="production-overlap-probe", allocation_fence=1,
            version=1, base_version=0, base_digest=digest,
            state=np.asarray([0.5]), outer=OuterState(step=1, accepted_tokens=1),
            policy_digest=ASYNC_DECOUPLED_V21.digest,
            layout_digest="1" * 64, code_digest="2" * 64,
            manifest_digest="3" * 64,
            selected_contribution_digests=(contribution.digest,),
            reload_verified=True, latest_cas_verified=True)

    if service.handoff(
            contribution, certify, deadline=time.monotonic() + timeout_s) != "OWNED":
        raise RuntimeError("production v2 service rejected local ownership")
    if not background_started.wait(timeout_s):
        raise TimeoutError("production background did not start")
    # This is a real lane transition, not a synthetic event: q=1 -> q=2
    # completes while q=[0,1) remains owned by the delayed service, and its
    # applied global anchor honestly remains S_0.
    lane.finish_window(
        np.asarray([2.0]), exact_tokens=1, begin_ns=3, end_ns=4)
    service.event(
        contribution, "k40_start",
        applied_anchor_version=lane.applied_anchor_version,
        local_model_basis="worker_local")
    return [service, events]


def _owner_endpoint_from_snapshot(peer: dict[str, object]) -> OwnerEndpoint:
    """Decode endpoint fields while excluding control-only snapshot metadata."""
    endpoint_fields = OwnerEndpoint.__dataclass_fields__
    return OwnerEndpoint(**{
        name: peer[name] for name in endpoint_fields if name in peer
    })


def _native_remote_endpoints(
        endpoints: tuple[OwnerEndpoint, ...], *, local_worker_id: str,
        minimum_contributions: int) -> tuple[OwnerEndpoint, ...]:
    """Validate a frozen owner set and return its deterministic peer order.

    Every manager derives the same round-robin tournament from the frozen
    worker identities. Each round contains disjoint bounded point-to-point
    pairs, so four peers finish in three transfer waves without a launched-rank
    collective. The explicit contribution floor remains control-plane policy
    and can therefore be lower than allocation capacity in later rungs.
    """
    ordered = tuple(sorted(endpoints, key=lambda item: item.worker_id))
    workers = [item.worker_id for item in ordered]
    if len(set(workers)) != len(workers):
        raise ValueError("native frozen endpoints require a unique stable worker identity")
    if local_worker_id not in workers:
        raise ValueError("native frozen endpoints exclude the local worker")
    if len(ordered) < minimum_contributions:
        raise ValueError("native frozen endpoint set is below explicit contribution floor")
    if len(ordered) % 2:
        raise ValueError("native v1 round-robin owner schedule requires an even peer set")
    by_worker = {item.worker_id: item for item in ordered}
    rotation = list(workers)
    peers: list[OwnerEndpoint] = []
    for _ in range(len(rotation) - 1):
        pairs = tuple(zip(rotation[:len(rotation) // 2],
                          reversed(rotation[len(rotation) // 2:])))
        peer = next((right if left == local_worker_id else left
                     for left, right in pairs
                     if local_worker_id in {left, right}), None)
        if peer is None:
            raise RuntimeError("native round-robin schedule omitted local worker")
        peers.append(by_worker[peer])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return tuple(peers)


def _native_owner_ranges(
        worker_ids: tuple[str, ...], *, run_id: str, fence: int,
        generation: int, attempt: int, owner_epoch: int,
        f64_layout_bytes: int, payload_max: int, itemsize: int
        ) -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Map canonical f64 shards to balanced owners and typed byte extents.

    Placement follows NDP11: owners are sorted by their stable 128-bit worker
    keys and the first shard is rotated by the fenced generation digest.  The
    same shard IDs therefore select identical owners for the f64 contribution
    plane and the f32 result plane even though their byte extents differ.
    """
    workers = tuple(worker_ids)
    if (not workers or len(set(workers)) != len(workers)
            or any(not worker for worker in workers)):
        raise ValueError("native owner placement requires unique stable workers")
    if (f64_layout_bytes <= 0 or payload_max <= 0 or payload_max % 8
            or f64_layout_bytes % 8 or itemsize not in {4, 8}
            or min(fence, generation, attempt, owner_epoch) < 0):
        raise ValueError("native owner placement extent or identity is invalid")
    keyed = tuple(sorted(
        workers, key=lambda worker: hashlib.sha256(worker.encode()).digest()[:16]))
    run_key = hashlib.sha256(run_id.encode()).digest()[:16]
    material = b"".join((
        run_key, int(fence).to_bytes(8, "little"),
        int(generation).to_bytes(8, "little"),
        int(attempt).to_bytes(4, "little"),
        int(owner_epoch).to_bytes(8, "little"),
    ))
    rotation = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    ranges: dict[str, list[tuple[int, int, int]]] = {
        worker: [] for worker in workers
    }
    shard_count = (f64_layout_bytes + payload_max - 1) // payload_max
    for shard in range(shard_count):
        f64_offset = shard * payload_max
        f64_extent = min(payload_max, f64_layout_bytes - f64_offset)
        elements = f64_extent // 8
        owner = keyed[(rotation + shard) % len(keyed)]
        ranges[owner].append(
            (shard, (f64_offset // 8) * itemsize, elements * itemsize))
    return {worker: tuple(value) for worker, value in ranges.items()}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("role", choices=("manager", "trainer"))
    value.add_argument("--run-dir", required=True); value.add_argument("--run-id", required=True)
    value.add_argument("--generations", type=int, default=1)
    value.add_argument("--local-steps", type=int, default=40)
    value.add_argument("--local-quorum", type=int, default=6)
    value.add_argument("--node-count", type=int, default=1)
    value.add_argument("--global-quorum", type=int, default=1)
    value.add_argument("--global-token-min", type=int, default=1)
    value.add_argument("--ready-fraction", type=float, default=None)
    value.add_argument("--coordinator-host", default="127.0.0.1")
    value.add_argument("--coordinator-port", type=int, default=29571)
    value.add_argument("--deadline-s", type=float, default=120.0)
    value.add_argument("--source-id", required=True); value.add_argument("--payload-id", required=True)
    value.add_argument("--code-id", default="unknown")
    value.add_argument("--coordinator-epoch", type=int, default=1)
    value.add_argument("--seed", default=""); value.add_argument("--train-args-json", default="")
    value.add_argument("--data", default=""); value.add_argument("--device", default="cuda:0")
    value.add_argument("--control", action="store_true")
    value.add_argument("--eta-outer", type=float, default=1.0)
    value.add_argument(
        "--diloco-policy",
        default=os.environ.get("RESILIENT_E97_DILOCO_POLICY",
                               ASYNC_DECOUPLED_V21.policy_id))
    value.add_argument(
        "--max-commit-lag", type=int,
        default=ASYNC_DECOUPLED_V21.max_commit_lag)
    value.add_argument(
        "--max-anchor-lag", type=int,
        default=ASYNC_DECOUPLED_V21.max_anchor_lag)
    value.add_argument(
        "--max-result-lag", type=int,
        default=ASYNC_DECOUPLED_V21.max_result_lag)
    value.add_argument(
        "--max-speculative-windows", type=int,
        default=ASYNC_DECOUPLED_V21.max_speculative_windows)
    value.add_argument("--migration-policy", default="")
    value.add_argument("--bulk-root", default=os.environ.get("RESILIENT_E97_BULK_ROOT", "/tmp/resilient-e97"))
    # Eight f32 trainer contributions plus the bounded f64 numerator/result
    # working set fit under this hard 64-GiB node-local ledger.
    value.add_argument("--max-spool-bytes", type=int, default=64 << 30)
    value.add_argument("--initial-generation", type=int, default=0)
    value.add_argument("--resume-handoff", default="")
    value.add_argument("--bulk-chunk-bytes", type=int, default=64 << 20)
    value.add_argument("--local-spool-chunk-bytes", type=int, default=64 << 20)
    value.add_argument("--dataplane-backend", default=os.environ.get("DILOCO_DATAPLANE", ""))
    value.add_argument("--native-build-manifest", default=os.environ.get("NDP_BUILD_MANIFEST", ""))
    value.add_argument("--native-gate-json", default=os.environ.get("NDP_FULL_LAYOUT_GATE_JSON", ""))
    return value


def _async_v21_policy(args):
    """Fail closed before model load on any v2.1 identity/policy drift."""
    if bool(args.control):
        # The scalar/control fixture may shrink Q/T/deadlines, but it still
        # exercises and labels the exact v2 lag/coalescing/outer policy.  No
        # control fixture is production-promotion eligible.
        if (str(args.diloco_policy) != ASYNC_DECOUPLED_V21.policy_id
                or int(args.local_steps) != ASYNC_DECOUPLED_V21.k_local_steps
                or int(args.max_commit_lag)
                != ASYNC_DECOUPLED_V21.max_commit_lag
                or int(args.max_anchor_lag)
                != ASYNC_DECOUPLED_V21.max_anchor_lag
                or int(args.max_result_lag)
                != ASYNC_DECOUPLED_V21.max_result_lag
                or int(args.max_speculative_windows)
                != ASYNC_DECOUPLED_V21.max_speculative_windows
                or float(args.eta_outer) != ASYNC_DECOUPLED_V21.eta_outer):
            raise ValueError(
                "control fixture mislabeled or changed async-v2.1 policy")
        return ASYNC_DECOUPLED_V21
    actual = {
        "policy_id": str(args.diloco_policy),
        "policy_schema": ASYNC_DECOUPLED_V21.policy_schema,
        "contribution_schema": ASYNC_DECOUPLED_V21.contribution_schema,
        "manifest_schema": ASYNC_DECOUPLED_V21.manifest_schema,
        "checkpoint_schema": ASYNC_DECOUPLED_V21.checkpoint_schema,
        "native_abi": ASYNC_DECOUPLED_V21.native_abi,
        "wire_protocol_major": ASYNC_DECOUPLED_V21.wire_protocol_major,
        "wire_protocol_minor": ASYNC_DECOUPLED_V21.wire_protocol_minor,
        "k_local_steps": int(args.local_steps),
        "max_commit_lag": int(args.max_commit_lag),
        "max_anchor_lag": int(args.max_anchor_lag),
        "max_result_lag": int(args.max_result_lag),
        "max_speculative_windows": int(args.max_speculative_windows),
        "q_min": int(args.global_quorum),
        "t_min": int(args.global_token_min),
        "active_membership_fraction": None,
        "group_deadline_s": min(float(args.deadline_s), 420.0),
        "ready_deadline_s": 180.0,
        "owned_deadline_s": 1.0,
        "catch_up_deadline_s": 420.0,
        "first_commit_deadline_s": 720.0,
        "generation_attempt_retries": 0,
        "eta_outer": float(args.eta_outer),
        "outer_mode": "delta_sgd",
        "owned_descriptor_capacity": 1,
        "mutable_interval_capacity": 1,
        "result_mailbox_capacity": 1,
        "result_staging_capacity": 1,
        "owner_reassignments": 2,
    }
    reviewed = ASYNC_DECOUPLED_V21.manifest()
    reviewed.pop("policy_digest")
    if actual != reviewed:
        if actual["policy_id"] != ASYNC_DECOUPLED_V21.policy_id:
            raise ValueError("historical/unknown async policy is not v2.1")
        raise ValueError(
            "rendered async-v2.1 policy differs from reviewed constants")
    if args.ready_fraction is not None:
        raise ValueError(
            "async-v2.1 initial profile disables active-membership fraction")
    return ASYNC_DECOUPLED_V21


def _dataplane_policy(args) -> tuple[str, bool, bool]:
    """Resolve once before model load; Frontier/real E97 defaults native."""
    backend = getattr(args, "dataplane_backend", "") or os.environ.get(
        "DILOCO_DATAPLANE", "")
    production = bool(os.environ.get("SLURM_JOB_ID")) or not bool(args.control)
    full_layout = not bool(args.control)
    if not backend:
        backend = NATIVE_CXI if production or full_layout else PYTHON_TCP_DEBUG
    validate_backend(backend, production=production, full_layout=full_layout)
    if backend == PYTHON_TCP_DEBUG and (not args.control or args.node_count > 2):
        raise ValueError("Python TCP is restricted to the explicit <=2-node control fixture")
    return backend, production, full_layout


def _attest_dataplane(args) -> dict[str, object]:
    backend, production, full_layout = _dataplane_policy(args)
    required_gate = os.environ.get("NDP_REQUIRED_GATE", "G2")
    attestation = attest_launch(
        backend=backend, production=production, full_layout=full_layout,
        build_manifest=getattr(args, "native_build_manifest", "") or None,
        gate_json=getattr(args, "native_gate_json", "") or None,
        source_root=ROOT if backend != PYTHON_TCP_DEBUG else None,
        required_gate=required_gate,
    )
    _require_wired_dense_runtime(backend)
    return attestation


def _require_wired_dense_runtime(backend: str) -> None:
    """Reject unknown selectors; native branches are structurally wired below."""
    if backend not in {PYTHON_TCP_DEBUG, NATIVE_TEST, NATIVE_CXI}:
        raise ValueError("unsupported split-role dense backend")


def _fence(args, generation: int) -> LocalFence:
    return LocalFence(args.run_id, generation, 0, _fence_epoch(args), args.payload_id)


def _fence_epoch(args) -> int:
    return int(os.environ.get("RESILIENT_E97_FENCE_EPOCH", args.coordinator_epoch))


def _native_runtime_resume_compatible(
        recorded: object, current: object) -> bool:
    """Allow a control-only source advance without weakening native identity.

    The fresh-allocation fix necessarily changes the repository commit and the
    digest of the build manifest that records it.  Every substantive native
    runtime field remains exact: schema, bundle, artifacts, provider, and
    training configuration.  Unknown fields are compared too, so this narrow
    provenance exception cannot silently authorize a new runtime capability.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    provenance = {"source_commit", "build_manifest_sha256"}
    if not provenance <= recorded.keys() or not provenance <= current.keys():
        return False
    return ({key: value for key, value in recorded.items()
             if key not in provenance}
            == {key: value for key, value in current.items()
                if key not in provenance})


def _resume_handoff_identity_matches(
        handoff: object, args, *,
        recorded_runtime: object, native: bool) -> bool:
    """Bind a fresh trainer to the stable execution identity and newer fence."""
    if not isinstance(handoff, dict):
        return False
    fence = handoff.get("fence")
    return bool(
        isinstance(fence, dict)
        and handoff.get("run_id") == args.run_id
        and handoff.get("payload_id") == args.payload_id
        and handoff.get("source_id") == args.source_id
        and (not native or handoff.get("code_id") == args.code_id)
        and (
            not native
            or isinstance(recorded_runtime, dict)
            and len(str(recorded_runtime.get("source_commit", ""))) == 40
        )
        and int(fence.get("coordinator_epoch", -1)) <= _fence_epoch(args)
        and handoff.get("finalized") is True
    )


def _peer_authority(
        args,
        ) -> tuple[ManifestPeerAuthority, AllocationClaim] | None:
    encoded = os.environ.get("RESILIENT_E97_ALLOCATION_CLAIM")
    if not encoded:
        return None  # direct local protocol fixtures do not own an allocation
    claim = AllocationClaim.decode(encoded)
    if claim.run_id != args.run_id or claim.fence != _fence_epoch(args):
        raise ValueError("role allocation claim does not match run/fence")
    authority = ManifestPeerAuthority(args.run_dir)
    authority.assert_current(claim)
    return authority, claim


def _latest_role_generation(control: Path, identity: str, args) -> int:
    path = control / "recovery" / f"{identity}.json"
    if not path.exists():
        return args.initial_generation
    value = json.loads(path.read_text())
    if (value.get("identity") != identity or value.get("run_id") != args.run_id
            or value.get("payload_id") != args.payload_id
            or value.get("source_id") != args.source_id):
        raise ValueError("role recovery identity/fence mismatch")
    prior_epoch = int(value.get("coordinator_epoch", -1))
    if prior_epoch > _fence_epoch(args):
        raise ValueError("role recovery was written by a newer allocation fence")
    if prior_epoch < _fence_epoch(args):
        return args.initial_generation  # old node-local work is disposable
    return max(args.initial_generation, int(value["generation"]))


def _publish_role_recovery(control: Path, identity: str, args, generation: int,
                           **extra: object) -> None:
    atomic_json(control / "recovery" / f"{identity}.json", {
        "schema": 1, "identity": identity, "run_id": args.run_id,
        "payload_id": args.payload_id, "source_id": args.source_id,
        "coordinator_epoch": _fence_epoch(args), "generation": generation, **extra})


def _cohort_restart_sequence() -> int:
    value = int(os.environ.get("RESILIENT_E97_COHORT_RESTART_SEQUENCE", "0"))
    if value < 0:
        raise ValueError("atomic cohort restart sequence cannot be negative")
    return value


def _validate_atomic_cohort_recovery(
        control: Path, args, *, node: int, node_incarnation: str,
        generation: int, admitted_statuses: tuple[str, ...],
        ) -> dict[str, object] | None:
    """Validate the supervisor's fenced all-eight reconstruction record."""
    sequence = _cohort_restart_sequence()
    if sequence == 0:
        return None
    path = control / "atomic-cohort-recovery.json"
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(
            "atomic cohort restart lacks valid recovery authority") from error
    if (
        value.get("schema")
        != "emender-async-v21-atomic-cohort-recovery-v1"
        or value.get("run_id") != args.run_id
        or int(value.get("allocation_fence", -1)) != _fence_epoch(args)
        or int(value.get("node_rank", -1)) != node
        or value.get("node_incarnation") != node_incarnation
        or int(value.get("restart_sequence", -1)) != sequence
        or int(value.get("authoritative_generation", -1)) != generation
        or value.get("required_trainers") != list(range(8))
        or value.get("status") not in admitted_statuses
    ):
        raise ValueError("atomic cohort recovery identity/fence mismatch")
    return value


def _native_manager_resume_point(
        run: Path, args,
        fenced: tuple[ManifestPeerAuthority, AllocationClaim] | None,
        *, native_runtime: dict[str, object] | None = None,
        ) -> tuple[int, dict[str, object]]:
    """Resolve a restarted model-free manager against fenced committed state.

    A manager owns no model or optimizer, so synchronizing it means validating
    the immutable handoff chain and advertising the resulting base generation.
    Node-local native handles are deliberately recreated under a new
    incarnation.  Unfinished transport state is never inferred from old local
    markers or replayed as a committed generation.
    """
    initial = int(args.initial_generation)
    target = initial + int(args.generations)
    commit = None if fenced is None else fenced[0].current_commit(fenced[1])
    latest_path = run / "handoff" / "latest.json"
    if commit is None and not latest_path.exists():
        return initial, {"status": "cold_start", "generation": initial,
                         "fence": _fence_epoch(args)}
    latest = (
        json.loads(latest_path.read_text())
        if commit is None else commit.pointer())
    generation = int(
        latest.get("generation", -1) if commit is None else commit.generation)
    source_fence = int(
        latest.get("fence", -1)
        if commit is None else commit.allocation_fence)
    current_fence = _fence_epoch(args)
    if generation < initial or generation > target:
        raise ValueError("authoritative latest generation is outside this run bound")
    if source_fence <= 0 or source_fence > current_fence:
        raise ValueError("manager rejoin latest fence is invalid or newer than allocation")
    manifest = Path(
        str(latest.get("manifest", ""))
        if commit is None else commit.manifest_path).resolve()
    try:
        manifest.relative_to((run / "handoff").resolve())
    except ValueError as error:
        raise ValueError("manager rejoin manifest escapes the handoff root") from error
    encoded = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    expected_manifest_sha = (
        latest.get("manifest_sha256")
        if commit is None else commit.manifest_sha256)
    if manifest_sha256 != expected_manifest_sha:
        raise ValueError("manager rejoin manifest checksum mismatch")
    value = json.loads(encoded)
    source_code_id = str(value.get("code_id", ""))
    recorded_runtime = dict(value.get("digests", {})).get("native_runtime")
    code_compatible = source_code_id == args.code_id
    if not code_compatible:
        code_compatible = (native_runtime is not None
                           and _native_runtime_resume_compatible(
                               recorded_runtime, native_runtime))
    if (not value.get("finalized") or value.get("run_id") != args.run_id
            or value.get("payload_id") != args.payload_id
            or value.get("source_id") != args.source_id
            or not code_compatible
            or int(value.get("generation", -1)) != generation
            or int(value.get("fence", {}).get("coordinator_epoch", -1))
            != source_fence):
        raise ValueError("manager rejoin manifest identity mismatch")
    if fenced is not None and (
        commit is None
        or commit.generation != generation
        or commit.manifest_sha256 != manifest_sha256
    ):
        raise ValueError("manager rejoin handoff is not authoritative")
    evidence = {
        "status": "synchronized", "generation": generation,
        "fence": current_fence, "source_fence": source_fence,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
    }
    if commit is not None:
        apply_receipts = fenced[0].node_apply_receipts(commit)
        evidence.update({
            "commit_receipt_digest": commit.receipt_digest,
            "accepted_tokens": commit.accepted_tokens,
            "result_root": commit.result_root,
            "apply_receipts": [
                {"worker_id": item.node_id,
                 "receipt_digest": item.receipt_digest}
                for item in apply_receipts
            ],
        })
    if source_code_id != args.code_id:
        evidence["source_code_id"] = source_code_id
    return generation, evidence


def _normalized_apply_receipts(
        value: Mapping[str, object]) -> list[dict[str, str]]:
    receipts = []
    seen: set[str] = set()
    for item in value.get("apply_receipts", []):
        if not isinstance(item, Mapping):
            raise ValueError("native recovery apply receipt is not a mapping")
        worker_id = str(item.get("worker_id", ""))
        receipt_digest = str(item.get("receipt_digest", ""))
        if (
            not worker_id
            or worker_id in seen
            or len(receipt_digest) != 64
        ):
            raise ValueError("native recovery apply receipt identity is invalid")
        seen.add(worker_id)
        receipts.append({
            "worker_id": worker_id,
            "receipt_digest": receipt_digest,
        })
    return sorted(receipts, key=lambda item: item["worker_id"])


def _validate_native_recovery_handshake(
        handshake: Mapping[str, object],
        sync_evidence: Mapping[str, object], *, generation: int) -> None:
    expected_receipts = _normalized_apply_receipts(sync_evidence)
    observed_receipts = _normalized_apply_receipts(handshake)
    if (
        handshake.get("status") != "recover"
        or int(handshake.get("generation", -1)) != generation
        or str(handshake.get("receipt_digest", ""))
        != str(sync_evidence.get("commit_receipt_digest", ""))
        or not isinstance(handshake.get("requires_node_apply"), bool)
        or (
            generation > 0
            and (
                handshake.get("manifest_digest")
                != sync_evidence.get("manifest_sha256")
                or handshake.get("result_root")
                != sync_evidence.get("result_root")
                or int(handshake.get("accepted_tokens", -1))
                != int(sync_evidence.get("accepted_tokens", -2))
                or observed_receipts != expected_receipts
            )
        )
    ):
        raise ValueError("native peer recovery handshake disagrees with manifest")


def _validate_native_rejoin_instruction(
        instruction: Mapping[str, object],
        sync_evidence: Mapping[str, object], *, generation: int) -> None:
    if (
        instruction.get("status") != "rejoin"
        or int(instruction.get("generation", -1)) != generation
        or int(instruction.get("authoritative_generation", -1)) != generation
        or instruction.get("requires_rejoin") is not True
        or instruction.get("requires_reload") is not False
        or str(instruction.get("receipt_digest", ""))
        != str(sync_evidence.get("commit_receipt_digest", ""))
        or instruction.get("manifest_digest")
        != sync_evidence.get("manifest_sha256")
        or instruction.get("result_root")
        != sync_evidence.get("result_root")
        or int(instruction.get("accepted_tokens", -1))
        != int(sync_evidence.get("accepted_tokens", -2))
        or _normalized_apply_receipts(instruction)
        != _normalized_apply_receipts(sync_evidence)
    ):
        raise ValueError(
            "native peer-control rejoin instruction disagrees with authority")


def _node_apply_receipt_digest(
        sync_evidence: Mapping[str, object], *, worker_id: str) -> str:
    matches = [
        item["receipt_digest"]
        for item in _normalized_apply_receipts(sync_evidence)
        if item["worker_id"] == worker_id
    ]
    if not matches:
        return ""
    if len(matches) != 1:
        raise ValueError("native recovery has duplicate node-apply authority")
    return matches[0]


def _ready_recovered_peer(
        pool_client: PoolControlClient, endpoint: OwnerEndpoint, *,
        generation: int, run_id: str, fence: int,
        apply_receipt_digest: str, deadline: float,
        ) -> dict[str, object]:
    """Advertise one recovered peer after any in-flight node apply closes.

    A manager can restart after its own node applied but before the other
    frozen node published its all-eight receipt.  Peer control retains the
    global atomic-apply gate until every frozen worker receipt is present.
    Waiting here is a bounded recovery lifecycle state; no trainer foreground
    work or one-node generation authority is admitted during the wait.
    """
    last: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return pool_client.ready(
                endpoint, generation,
                run_id=run_id, fence=fence,
                apply_receipt_digest=apply_receipt_digest)
        except RuntimeError as error:
            last = error
            if "READY requires this incarnation's atomic node-apply receipt" \
                    not in str(error):
                raise
            time.sleep(min(.02, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"recovered peer READY deadline expired: {last}")


def _authoritative_trainer_resume_handoff(
        run: Path, args,
        fenced: tuple[ManifestPeerAuthority, AllocationClaim] | None,
        ) -> Path:
    """Resolve a trainer restart to the latest exact committed handoff.

    The submit-time handoff remains the cold-start anchor.  Once this
    allocation commits a generation, however, a supervised trainer restart
    must reload that newer publication instead of replaying the anchor.  The
    fenced publication and immutable latest pointer must agree exactly before
    the newer handoff is accepted.
    """
    configured = Path(args.resume_handoff).resolve()
    if fenced is None:
        return configured
    authority, claim = fenced
    authoritative = authority.current_commit(claim)
    if authoritative is None:
        raise ValueError("resume handoff has no authoritative immutable commit")
    generation = authoritative.generation
    initial = int(args.initial_generation)
    target = initial + int(args.generations)
    if generation < initial or generation > target:
        raise ValueError("authoritative trainer generation is outside this run bound")
    if generation == initial:
        return configured

    manifest = authoritative.manifest_path.resolve()
    try:
        manifest.relative_to((run / "handoff").resolve())
    except ValueError as error:
        raise ValueError("trainer resume manifest escapes the handoff root") from error
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if authoritative.manifest_sha256 != manifest_sha256:
        raise ValueError("trainer resume manifest receipt is not authoritative")
    return manifest


def _reload_verified_async_v2_latest(
        run: Path, args,
        fenced: tuple[ManifestPeerAuthority, AllocationClaim] | None,
        *, generation: int, deadline: float,
        ) -> tuple[dict[str, object], dict[str, object], str]:
    """Reload and fence-check the only result eligible for a K-boundary apply.

    A native result handle is a candidate, not global authority.  Eligibility
    requires the immutable handoff to reload under the expected run/fence and,
    when an allocation lease is active, exact agreement with the durable
    ``latest/authoritative`` CAS record.  This check deliberately lives in the
    real trainer entrypoint instead of a probe or post-run validator.
    """
    commit: CommitReceipt | None = None
    if fenced is None:
        latest = wait_metadata(
            run / "handoff/latest.json", deadline=deadline,
            expected={"generation": int(generation),
                      "fence": _fence_epoch(args)})
    else:
        authority, claim = fenced
        node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
        node_control = (
            Path(args.bulk_root) / args.run_id / f"node-{node}" / "control")
        peer_commit = wait_metadata(
            node_control / f"peer-commit-{generation:08d}.json",
            deadline=deadline,
            expected={
                "generation": int(generation),
                "fence": _fence_epoch(args),
            },
        )
        # The manager discovers commit through peer memory and names the exact
        # receipt in a node-local handoff.  Trainers read immutable authority
        # once for reload verification; they never poll a shared directory.
        commit = authority.current_commit(claim)
        if (
            commit is None
            or commit.generation != int(generation)
            or commit.receipt_digest
            != str(peer_commit.get("commit_receipt_digest", ""))
        ):
            raise ValueError(
                "node-local peer commit differs from immutable authority")
        latest = commit.pointer()
    published_manifest = Path(str(latest.get("manifest", ""))).resolve()
    try:
        published_manifest.relative_to((run / "handoff").resolve())
    except ValueError as error:
        raise ValueError(
            "async v2 authoritative latest escapes handoff root") from error
    manifest_bytes = published_manifest.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != latest.get("manifest_sha256"):
        raise ValueError(
            "async v2 authoritative latest manifest digest mismatch")
    published = json.loads(manifest_bytes)
    if (not isinstance(published, dict)
            or not published.get("finalized")
            or published.get("run_id") != args.run_id
            or int(published.get("generation", -1)) != int(generation)
            or int(dict(published.get("fence", {})).get(
                "coordinator_epoch", -1)) != _fence_epoch(args)):
        raise ValueError(
            "async v2 authoritative latest was not reload verified")
    if fenced is not None and (
        commit is None
        or commit.generation != int(generation)
        or commit.allocation_fence != _fence_epoch(args)
        or commit.manifest_sha256 != manifest_digest
    ):
        raise ValueError(
            "async v2 latest does not match immutable commit authority")
    return latest, published, manifest_digest


def _native_ready_delay(*, node: int, generation: int) -> float:
    """Return an exact, bounded test-only late-READY delay for one peer."""
    encoded = os.environ.get("RESILIENT_E97_DELAY_READY", "")
    if not encoded:
        return 0.0
    fields = encoded.split(":")
    if len(fields) != 3:
        raise ValueError("delayed READY injection must be NODE:GENERATION:SECONDS")
    target_node, target_generation = map(int, fields[:2])
    delay = float(fields[2])
    if not 0.0 <= delay <= 180.0:
        raise ValueError("delayed READY injection must be bounded by 180 seconds")
    return delay if (node, generation) == (target_node, target_generation) else 0.0


def _wait_native_ready_delay(control: Path, args, *, node: int, generation: int,
                             incarnation: str,
                             term_requested: dict[str, bool]) -> bool:
    """Apply and attest the bounded late-READY injection for one incarnation."""
    ready_delay = _native_ready_delay(node=node, generation=generation)
    if not ready_delay:
        return not term_requested["value"]
    delay_marker = (control /
                    f"native-delayed-ready-{generation:08d}-{incarnation}.json")
    marker = {
        "schema": "emender-native-delayed-ready-v1",
        "run_id": args.run_id, "worker_id": f"node-{node}",
        "incarnation": incarnation, "fence_epoch": _fence_epoch(args),
        "generation": generation, "delay_seconds": ready_delay,
    }
    atomic_metadata(delay_marker, {**marker, "status": "delaying"})
    delay_deadline = time.monotonic() + ready_delay
    while (not term_requested["value"] and time.monotonic() < delay_deadline):
        time.sleep(min(.1, delay_deadline - time.monotonic()))
    atomic_metadata(
        delay_marker,
        {**marker, "status": ("cancelled" if term_requested["value"]
                              else "completed")})
    return not term_requested["value"]


def _terminal_native_checkpoint(run: Path, args, *, completed: int,
                                deadline: float) -> Path:
    """Resolve the already-fenced final checkpoint for a disposable follower.

    Intermediate generations still require identity-specific node-local
    recovery checkpoints.  Once the terminal generation is atomically
    published, however, writing fifteen additional 7.9 GiB copies only delays
    apply acknowledgement and cannot improve recovery: the fenced handoff is
    the authoritative continuation state.
    """
    latest = wait_metadata(
        run / "handoff/latest.json", deadline=deadline,
        expected={"generation": completed, "fence": _fence_epoch(args)})
    manifest = Path(str(latest.get("manifest", ""))).resolve()
    manifest.relative_to((run / "handoff").resolve())
    encoded = manifest.read_bytes()
    if (__import__("hashlib").sha256(encoded).hexdigest()
            != latest.get("manifest_sha256")):
        raise ValueError("terminal native handoff manifest digest mismatch")
    value = json.loads(encoded)
    if (not isinstance(value, dict) or not value.get("finalized")
            or value.get("run_id") != args.run_id
            or value.get("payload_id") != args.payload_id
            or value.get("source_id") != args.source_id
            or int(value.get("generation", -1)) != completed
            or int(dict(value.get("fence", {})).get("coordinator_epoch", -1))
            != _fence_epoch(args)):
        raise ValueError("terminal native handoff identity mismatch")
    checkpoint = Path(str(value.get("checkpoint", ""))).resolve()
    checkpoint.relative_to((run / "checkpoints").resolve())
    if (not checkpoint.is_file()
            or checkpoint.stat().st_size != int(value.get("checkpoint_bytes", -1))
            or len(str(value.get("checkpoint_sha256", ""))) != 64):
        raise ValueError("terminal native checkpoint extent/digest is invalid")
    return checkpoint


def _native_safe_boundary_transaction(
        args, *, node: int, generation: int, result_root: str,
        node_incarnation: str,
        ) -> SafeBoundaryRendezvous:
    """Construct the one exact candidate/boundary/apply transaction."""
    return SafeBoundaryRendezvous(
        run_id=args.run_id,
        fence=_fence_epoch(args),
        node_id=f"node-{node}",
        node_incarnation=node_incarnation,
        result_version=generation + 1,
        result_digest=result_root,
        trainer_count=8,
        rendezvous_timeout_s=ASYNC_V21_BOUNDARY_RENDEZVOUS_S,
        apply_timeout_s=ASYNC_V21_ALL_EIGHT_APPLY_S,
    )


def _wait_for_native_boundary_control(
        control: Path, args, *, generation: int, transaction_digest: str,
        marker_name: str, deadline: float,
        ) -> dict[str, object]:
    """Wait for rendezvous/release or fail immediately on its fenced abort."""
    marker = control / f"{marker_name}-{generation:08d}.json"
    abort = control / f"native-boundary-abort-{generation:08d}.json"
    expected = {
        "run_id": args.run_id,
        "fence_epoch": _fence_epoch(args),
        "generation": generation,
        "transaction_digest": transaction_digest,
    }
    while True:
        if abort.exists():
            value = wait_metadata(
                abort, deadline=deadline, expected=expected)
            raise RuntimeError(
                "safe-boundary rendezvous aborted before apply release: "
                f"{value.get('reason', 'unknown')}")
        if marker.exists():
            return wait_metadata(
                marker, deadline=deadline, expected=expected)
        if time.monotonic() >= deadline:
            break
        time.sleep(.02)
    raise TimeoutError(
        f"safe-boundary control marker did not arrive: {marker_name}")


def _coordinate_native_safe_boundary(
        control: Path, args, *, bulk: Path, identity: str,
        node: int, generation: int,
        result_root: str, node_incarnation: str,
        preparation_deadline: float,
        committed_evidence: Mapping[str, object],
        ) -> tuple[SafeBoundaryRendezvous, dict[str, object]]:
    """Gather candidate receipts, rendezvous eight boundaries, then release.

    Preparation receipts never satisfy the second loop.  Any exception before
    release publishes a fenced abort marker while the transaction still has
    zero applied lanes; trainers observe it before translating live x/z.
    """
    transaction = _native_safe_boundary_transaction(
        args,
        node=node,
        generation=generation,
        result_root=result_root,
        node_incarnation=node_incarnation,
    )
    try:
        for rank in range(args.local_quorum):
            prepared = wait_metadata(
                control
                / f"native-candidate-prepared-{generation:08d}-{rank:02d}.json",
                deadline=preparation_deadline,
                expected={
                    "schema":
                        "emender-native-e97-candidate-prepared-v2.1",
                    "run_id": args.run_id,
                    "fence_epoch": _fence_epoch(args),
                    "generation": generation,
                    "result_root": result_root,
                    "rank": rank,
                    "node_incarnation": node_incarnation,
                    "transaction_digest": transaction.transaction_digest,
                },
            )
            transaction.record_candidate_prepared(
                rank=rank,
                trainer_incarnation=str(
                    prepared["trainer_incarnation"]),
                candidate_digest=str(prepared["candidate_digest"]),
                preparation_started_monotonic_s=float(
                    prepared["preparation_started_monotonic_s"]),
                prepared_monotonic_s=float(
                    prepared["candidate_prepared_monotonic_s"]),
            )

        rendezvous_opened = time.monotonic()
        rendezvous = transaction.open_boundary_rendezvous(
            opened_monotonic_s=rendezvous_opened)
        atomic_metadata(
            control / f"native-boundary-rendezvous-{generation:08d}.json",
            {
                **rendezvous,
                "schema":
                    "emender-native-e97-boundary-rendezvous-v2.1",
                "run_id": args.run_id,
                "fence_epoch": _fence_epoch(args),
                "generation": generation,
                "result_root": result_root,
                "node_incarnation": node_incarnation,
                "transaction_digest": transaction.transaction_digest,
            },
        )
        heartbeat(
            bulk,
            identity,
            generation=generation,
            step=generation * args.local_steps,
            loss=None,
            stage="boundary_rendezvous",
            **committed_evidence,
        )
        assert transaction.boundary_deadline_monotonic_s is not None
        for rank in range(args.local_quorum):
            boundary = wait_metadata(
                control
                / f"native-boundary-ready-{generation:08d}-{rank:02d}.json",
                deadline=transaction.boundary_deadline_monotonic_s,
                expected={
                    "schema": "emender-native-e97-boundary-ready-v2.1",
                    "run_id": args.run_id,
                    "fence_epoch": _fence_epoch(args),
                    "generation": generation,
                    "result_root": result_root,
                    "rank": rank,
                    "node_incarnation": node_incarnation,
                    "transaction_digest": transaction.transaction_digest,
                },
            )
            transaction.record_boundary_ready(
                rank=rank,
                trainer_incarnation=str(
                    boundary["trainer_incarnation"]),
                candidate_digest=str(boundary["candidate_digest"]),
                boundary_monotonic_s=float(
                    boundary["boundary_ready_monotonic_s"]),
                local_window=int(boundary["local_window"]),
            )

        released = time.monotonic()
        release = dict(transaction.release_apply(
            released_monotonic_s=released))
        atomic_metadata(
            control / f"native-apply-release-{generation:08d}.json",
            {
                **release,
                "schema": "emender-native-e97-apply-release-v2.1",
                "run_id": args.run_id,
                "fence_epoch": _fence_epoch(args),
                "generation": generation,
                "result_root": result_root,
                "node_incarnation": node_incarnation,
                "transaction_digest": transaction.transaction_digest,
                "release_monotonic_s": released,
            },
        )
        return transaction, release
    except BaseException as error:
        if not transaction.released:
            abort = transaction.abort_before_release(
                aborted_monotonic_s=time.monotonic(),
                reason=f"{type(error).__name__}: {error}",
            )
            abort_metrics = transaction.telemetry()
            atomic_metadata(
                control
                / f"native-boundary-abort-{generation:08d}.json",
                {
                    "run_id": args.run_id,
                    "fence_epoch": _fence_epoch(args),
                    "generation": generation,
                    "result_root": result_root,
                    "node_incarnation": node_incarnation,
                    "transaction_digest":
                        transaction.transaction_digest,
                    **abort,
                    "metrics": abort_metrics,
                },
            )
        raise


def _liveness_heartbeat(bulk: Path, identity: str, interval_s: float = 5.0):
    """Refresh liveness without disguising stalled generation progress."""
    state = bulk / "supervision" / f"{identity}.liveness.json"
    stop = threading.Event()

    def publish() -> None:
        while not stop.wait(interval_s):
            atomic_json(state, {"identity": identity, "heartbeat_time": time.time()})

    thread = threading.Thread(target=publish, name=f"{identity}-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _stage_telemetry(bulk: Path, identity: str, generation: int, stage: str,
                     started: float, hard_s: float, *,
                     ended: float | None = None,
                     **metrics: object) -> None:
    ended = time.monotonic() if ended is None else float(ended)
    if ended < started:
        raise ValueError(f"{stage} telemetry end precedes its start")
    elapsed = ended - started
    record = {"timestamp": time.time(), "identity": identity,
              "generation": generation, "stage": stage,
              "monotonic_start_s": started, "monotonic_end_s": ended,
              "elapsed_s": elapsed, "hard_s": hard_s,
              "within_slo": elapsed <= hard_s, **metrics}
    causal_phase = _CAUSAL_PHASE_BY_STAGE.get(stage)
    if causal_phase is not None:
        record["causal_phase"] = causal_phase
        record["causal_id"] = _causal_work_id(identity, generation)
        record.setdefault(
            "foreground_component_s",
            elapsed if causal_phase in {
                "freeze_snapshot", "snapshot_admission", "apply_swap",
            } else 0.0,
        )
    path = bulk / "telemetry" / f"{identity}-pool.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    if elapsed > hard_s:
        raise TimeoutError(f"{stage} exceeded {hard_s}s stage SLO")


_MANAGER_EXCHANGE_STAGES = frozenset({
    "freeze", "owner_transport", "redistribution", "checkpoint_commit",
    "result_preparation", "published",
})


def _wait_for_manager_exchange_window(bulk: Path, *, node: int, generation: int,
                                      deadline: float) -> str:
    """Do not spend a trainer's apply bound while its manager reduces locally.

    The manager's generation deadline covers K40 contribution collection and
    exact local reduction.  Only its fenced transition to ``freeze`` opens the
    distinct 180-second owner-transport/apply window.  The compact heartbeat is
    node-local, atomically replaced, and contains no tensor or model data.
    """
    state_path = bulk / "supervision" / f"node-{node}-manager.json"
    last_stage = "missing"
    while time.monotonic() < deadline:
        try:
            state = json.loads(state_path.read_text())
        except FileNotFoundError:
            time.sleep(.02)
            continue
        last_stage = str(state.get("stage", "unknown"))
        if (int(state.get("generation", -1)) >= generation
                and last_stage in _MANAGER_EXCHANGE_STAGES):
            return last_stage
        time.sleep(.02)
    raise TimeoutError(
        f"manager local reduction deadline expired before exchange: {last_stage}")


def _wait_for_leader_apply_release(bulk: Path, *, generation: int,
                                   fence: LocalFence, deadline: float) -> None:
    """Keep node-0 peers off the aggregate until its leader prepares a checkpoint.

    Job 5028835 left only 31 seconds in the fixed commit window after the
    global aggregate became visible. Eight simultaneous full-file readers
    prevented the designated trainer from reaching checkpoint creation.  The
    generation-scoped marker contains no tensor data and opens the bounded
    background result-preparation cohort only after the leader has materialized
    the result and proposed its immutable checkpoint.  The other seven readers
    may then overlap; live model swap remains gated by the distinct all-eight
    preapply barrier below.
    """
    marker = bulk / "control" / f"leader-apply-release-{generation:08d}.json"
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(.02)
    if not marker.exists():
        raise TimeoutError("checkpoint leader apply deadline expired")
    value = json.loads(marker.read_text())
    if (int(value.get("generation", -1)) != generation
            or value.get("fence") != fence.__dict__):
        raise ValueError("checkpoint leader apply marker fence mismatch")


def _pool_hosts(args) -> tuple[str, ...]:
    configured = os.environ.get("RESILIENT_E97_PEER_HOSTS", "")
    if configured:
        hosts = tuple(item.strip() for item in configured.split(",") if item.strip())
    elif os.environ.get("SLURM_JOB_NODELIST"):
        hosts = tuple(subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            text=True).splitlines())
    else:
        # Deterministic local multi-process integration fixture.
        hosts = tuple(args.coordinator_host for _ in range(args.node_count))
    if len(hosts) != args.node_count:
        raise ValueError("active manager host discovery differs from configured capacity")
    return hosts


def _pool_config(
        args, *, committed_generation: int = 0,
        committed_receipt_digest: str = "",
        committed_accepted_tokens: int = 0,
        committed_manifest_digest: str = "",
        committed_result_root: str = "",
        committed_apply_receipts: tuple[tuple[str, str], ...] = (),
        ) -> PoolControlConfig:
    policy = _async_v21_policy(args).digest
    backend, production, full_layout = _dataplane_policy(args)
    attestation = getattr(args, "_dataplane_attestation", {})
    scale = int(args.node_count) >= 4
    if scale and os.environ.get("ASYNC_V21_GATE") != "scale":
        raise ValueError(
            "4+ nodes require the authorized v2.1 scale controller")
    scale_values = {}
    if scale:
        required = (
            "ASYNC_V21_SCALE_CLOSURE_DIGEST",
            "ASYNC_V21_SCALE_CLOSE_OFFSET_NS",
            "ASYNC_V21_SCALE_STABLE_DIVERSITY_FLOOR",
            "ASYNC_V21_SCALE_PER_READY_WORKER_TOKEN_FLOOR",
        )
        if any(not os.environ.get(name) for name in required):
            raise ValueError(
                "4+ nodes reject the two-node Q_min early-close path; "
                "a complete V21S17 closure is required")
        close_offset_ns = int(
            os.environ["ASYNC_V21_SCALE_CLOSE_OFFSET_NS"])
        if close_offset_ns <= 0:
            raise ValueError("V21S17 close offset must be positive")
        scale_values = {
            "scale_close_offset_s": close_offset_ns / 1_000_000_000,
            "scale_stable_diversity_floor": int(
                os.environ[
                    "ASYNC_V21_SCALE_STABLE_DIVERSITY_FLOOR"]),
            "scale_per_ready_worker_token_floor": int(
                os.environ[
                    "ASYNC_V21_SCALE_PER_READY_WORKER_TOKEN_FLOOR"]),
            "scale_closure_digest":
                os.environ["ASYNC_V21_SCALE_CLOSURE_DIGEST"],
        }
    return PoolControlConfig(
        args.run_id, _fence_epoch(args), args.global_quorum, args.global_token_min,
        args.ready_fraction, args.source_id, policy, args.payload_id, args.code_id,
        PoolStageSLO.production(), backend, production, full_layout,
        str(attestation.get("bundle_sha256", ""))
        if backend != PYTHON_TCP_DEBUG else "",
        **scale_values,
        committed_generation=committed_generation,
        committed_receipt_digest=committed_receipt_digest,
        committed_accepted_tokens=committed_accepted_tokens,
        committed_manifest_digest=committed_manifest_digest,
        committed_result_root=committed_result_root,
        committed_apply_receipts=committed_apply_receipts)


def _copy_frame_payload(frame_fd: int, destination_fd: int, *,
                        payload_bytes: int, payload_offset: int) -> None:
    copy_fd_range(
        frame_fd, destination_fd, payload_bytes, source_offset=320,
        destination_offset=payload_offset)


class _NativePeerInbox:
    """Demultiplex one bounded native RX queue across concurrent peers.

    The compiled endpoint owns four fixed receive slots.  One Python receiver
    drains those slots and hands descriptors to peer workers through queues
    capped at the same four-frame credit bound.  Dense bytes stay in memfds;
    Python handles only descriptors and frame metadata.
    """

    def __init__(self, session: NativeManagerSession, *, peer_ids: tuple[str, ...],
                 capacity: int, frames_per_peer: int | dict[str, int], deadline: float,
                 queue_slots: int = 4):
        if isinstance(frames_per_peer, int):
            frame_budgets = {peer: frames_per_peer for peer in peer_ids}
        else:
            frame_budgets = {str(peer): int(value)
                             for peer, value in frames_per_peer.items()}
        if (not peer_ids or len(set(peer_ids)) != len(peer_ids)
                or set(frame_budgets) != set(peer_ids)
                or any(value <= 0 for value in frame_budgets.values())
                or capacity <= 320 or queue_slots <= 0):
            raise ValueError("native peer inbox bounds are invalid")
        self.session = session
        self.capacity = int(capacity)
        self.frame_budgets = frame_budgets
        self.deadline = float(deadline)
        self.queues = {
            peer: queue.Queue(maxsize=queue_slots) for peer in peer_ids
        }
        self.received = {peer: 0 for peer in peer_ids}
        self._queue_lock = threading.Lock()
        self.queued_frames = 0
        self.high_water_frames = 0
        self.stop = threading.Event()
        self.done = threading.Event()
        self.failure: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="native-owner-rx-demux", daemon=True)

    def __enter__(self) -> "_NativePeerInbox":
        self.thread.start()
        return self

    def _put(self, peer: str, item: tuple[int, int]) -> bool:
        while not self.stop.is_set() and time.monotonic() < self.deadline:
            try:
                self.queues[peer].put(item, timeout=.01)
                with self._queue_lock:
                    self.queued_frames += 1
                    self.high_water_frames = max(
                        self.high_water_frames, self.queued_frames)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        expected = sum(self.frame_budgets.values())
        total = 0
        try:
            while (total < expected and not self.stop.is_set()
                   and time.monotonic() < self.deadline):
                frame_fd = create_memfd("emender-ndp-rx-demux", allow_sealing=True)
                os.ftruncate(frame_fd, self.capacity)
                keep = False
                try:
                    received = self.session.receive_owner_fd(
                        frame_fd, capacity=self.capacity)
                    if received is None:
                        time.sleep(.001)
                        continue
                    peer, frame_bytes = received
                    if peer not in self.queues:
                        raise ValueError("native owner frame arrived from an unfenced peer")
                    if self.received[peer] >= self.frame_budgets[peer]:
                        raise ValueError("native owner peer exceeded its fixed frame budget")
                    keep = self._put(peer, (frame_fd, int(frame_bytes)))
                    if not keep:
                        raise TimeoutError("native peer inbox backpressure deadline expired")
                    self.received[peer] += 1
                    total += 1
                finally:
                    if not keep:
                        os.close(frame_fd)
            if not self.stop.is_set() and total != expected:
                raise TimeoutError("native peer inbox receive deadline expired")
        except BaseException as error:
            self.failure = error
        finally:
            self.done.set()

    def receive(self, peer_id: str, *, deadline: float) -> tuple[int, int]:
        if peer_id not in self.queues:
            raise KeyError(f"no native inbox for {peer_id}")
        inbox = self.queues[peer_id]
        while time.monotonic() < deadline:
            if self.failure is not None:
                raise self.failure
            try:
                item = inbox.get(
                    timeout=min(.01, max(.001, deadline - time.monotonic())))
                with self._queue_lock:
                    self.queued_frames -= 1
                return item
            except queue.Empty:
                if self.done.is_set() and inbox.empty():
                    break
        if self.failure is not None:
            raise self.failure
        raise TimeoutError(f"native peer inbox deadline expired for {peer_id}")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(2)
        for inbox in self.queues.values():
            while True:
                try:
                    frame_fd, _frame_bytes = inbox.get_nowait()
                except queue.Empty:
                    break
                with self._queue_lock:
                    self.queued_frames -= 1
                os.close(frame_fd)
        if self.thread.is_alive():
            raise RuntimeError("native peer inbox did not stop within its bound")

    def __exit__(self, exception_type, _exception, _traceback) -> None:
        self.close()
        if exception_type is None and self.failure is not None:
            raise self.failure


def _native_peer_exchange(session: NativeManagerSession, local_result, *, args,
                          node: int, peer_id: str, peer_incarnation: str,
                          peer_root: bytes, peer_weight: int,
                          peer_layout_digest: bytes | None = None,
                          deadline: float, inbox: _NativePeerInbox | None = None,
                          local_digest_future=None, local_digest_value: bytes | None = None,
                          send_ranges: tuple[tuple[int, ...], ...] | None = None,
                          receive_ranges: tuple[tuple[int, ...], ...] | None = None,
                          receive_length: int | None = None,
                          wire_chunk_count: int | None = None,
                          ) -> tuple[int, bytes, bytes]:
    """Symmetric memfd/libfabric exchange of exact deterministic shard ranges."""
    payload_max = args.bulk_chunk_bytes
    natural_chunk_count = (local_result.length + payload_max - 1) // payload_max
    chunk_count = natural_chunk_count if wire_chunk_count is None else int(wire_chunk_count)
    if chunk_count <= 0:
        raise ValueError("native peer wire shard count is invalid")
    if send_ranges is None:
        send_ranges = tuple(
            (chunk, chunk * payload_max,
             min(payload_max, local_result.length - chunk * payload_max))
            for chunk in range(natural_chunk_count))
    if receive_ranges is None:
        receive_ranges = tuple(
            (chunk, chunk * payload_max,
             min(payload_max, local_result.length - chunk * payload_max))
            for chunk in range(natural_chunk_count))

    def normalized(ranges: tuple[tuple[int, ...], ...]
                   ) -> tuple[tuple[int, int, int, int], ...]:
        values = []
        for item in ranges:
            if len(item) == 3:
                chunk, wire_offset, extent = item
                local_offset = wire_offset
            elif len(item) == 4:
                chunk, local_offset, wire_offset, extent = item
            else:
                raise ValueError("native peer shard range width is invalid")
            values.append((int(chunk), int(local_offset),
                           int(wire_offset), int(extent)))
        return tuple(values)

    send_ranges = normalized(send_ranges)
    receive_ranges = normalized(receive_ranges)
    remote_length = int(local_result.length if receive_length is None else receive_length)
    if (not send_ranges or not receive_ranges or remote_length <= 0
            or any(chunk not in range(chunk_count) or local_offset < 0
                   or wire_offset < 0 or extent <= 0 or extent > payload_max
                   or local_offset + extent > local_result.length
                   for chunk, local_offset, wire_offset, extent in send_ranges)
            or any(chunk not in range(chunk_count) or local_offset < 0
                   or wire_offset < 0 or extent <= 0 or extent > payload_max
                   or local_offset + extent > remote_length
                   for chunk, local_offset, wire_offset, extent in receive_ranges)):
        raise ValueError("native peer shard ranges are invalid")
    send_by_chunk = {chunk: (local_offset, wire_offset, extent)
                     for chunk, local_offset, wire_offset, extent in send_ranges}
    receive_by_chunk = {chunk: (local_offset, wire_offset, extent)
                        for chunk, local_offset, wire_offset, extent in receive_ranges}
    if (len(send_by_chunk) != len(send_ranges)
            or len(receive_by_chunk) != len(receive_ranges)):
        raise ValueError("native peer shard ranges contain duplicate IDs")
    remote_fd = create_memfd("emender-ndp-remote-result", allow_sealing=True)
    os.ftruncate(remote_fd, remote_length)
    peer_layout = (local_result.layout_digest if peer_layout_digest is None
                   else bytes(peer_layout_digest))
    if len(peer_layout) != 32:
        os.close(remote_fd)
        raise ValueError("native peer layout digest metadata is invalid")

    def frame_deadline() -> int:
        return min(session.transport.deadline_unix_ns,
                   time.time_ns() + max(1, int(
                       (deadline - time.monotonic()) * 1e9)))

    def send_all(*, sequence_base: int,
                 selected=send_ranges) -> None:
        for chunk, local_offset, wire_offset, extent in selected:
            frame_fd, frame_bytes = encode_owner_frame_fd(
                source_fd=local_result.fd, source_offset=local_offset,
                payload_offset=wire_offset,
                payload_bytes=extent, payload_max=payload_max,
                run_id=args.run_id, fence_epoch=_fence_epoch(args),
                generation=local_result.generation, attempt=local_result.attempt,
                owner_epoch=local_result.client.owner_epoch,
                worker_id=f"node-{node}",
                incarnation=session.owner_endpoint.incarnation,
                layout_digest=local_result.layout_digest,
                base_digest=local_result.base_digest,
                result_root=local_result.result_root,
                weight=local_result.global_weight, chunk_index=chunk,
                chunk_count=chunk_count,
                deadline_unix_ns=frame_deadline(),
                protocol_major=args._async_v21_policy.wire_protocol_major,
                protocol_minor=args._async_v21_policy.wire_protocol_minor,
                message_seq=((local_result.generation + 1) << 32)
                + sequence_base + chunk)
            try:
                session.transfer_frozen_fd(
                    peer_id, frame_fd, frame_bytes=frame_bytes,
                    result_root=local_result.result_root,
                    replay_identity=chunk.to_bytes(4, "little"),
                    deadline_unix_ns=frame_deadline())
            finally:
                os.close(frame_fd)

    def send_credits(*, sequence_base: int,
                     selected=receive_ranges) -> None:
        for chunk, _local_offset, wire_offset, extent in selected:
            credit_fd = encode_credit_frame_fd(
                payload_offset=wire_offset, payload_bytes=extent,
                payload_max=payload_max, run_id=args.run_id,
                fence_epoch=_fence_epoch(args), generation=local_result.generation,
                attempt=local_result.attempt,
                owner_epoch=local_result.client.owner_epoch,
                worker_id=f"node-{node}",
                incarnation=session.owner_endpoint.incarnation,
                layout_digest=peer_layout,
                base_digest=local_result.base_digest,
                permitted_root=peer_root, weight=peer_weight,
                chunk_index=chunk, chunk_count=chunk_count,
                deadline_unix_ns=frame_deadline(),
                protocol_major=args._async_v21_policy.wire_protocol_major,
                protocol_minor=args._async_v21_policy.wire_protocol_minor,
                message_seq=((local_result.generation + 1) << 32)
                + sequence_base + chunk)
            try:
                session.transfer_frozen_fd(
                    peer_id, credit_fd, frame_bytes=320,
                    result_root=peer_root,
                    replay_identity=b"credit" + chunk.to_bytes(4, "little"),
                    deadline_unix_ns=frame_deadline())
            finally:
                os.close(credit_fd)

    def receive_credits(selected=send_ranges) -> None:
        expected_chunks = {
            chunk: (local_offset, wire_offset, extent)
            for chunk, local_offset, wire_offset, extent in selected
        }
        seen: set[int] = set()
        while len(seen) != len(expected_chunks):
            if time.monotonic() >= deadline:
                raise TimeoutError("native owner credit deadline expired")
            if inbox is None:
                credit_fd = create_memfd("emender-ndp-rx-credit")
                os.ftruncate(credit_fd, 320)
                received = session.receive_owner_fd(credit_fd, capacity=320)
                if received is None:
                    os.close(credit_fd)
                    time.sleep(.001)
                    continue
                worker, frame_bytes = received
            else:
                credit_fd, frame_bytes = inbox.receive(peer_id, deadline=deadline)
                worker = peer_id
            try:
                if worker != peer_id or frame_bytes != 320:
                    raise ValueError("native owner credit arrived from wrong frozen peer")
                value = decode_credit_frame_fd(
                    credit_fd, payload_max=payload_max,
                    protocol_major=
                        args._async_v21_policy.wire_protocol_major,
                    protocol_minor=
                        args._async_v21_policy.wire_protocol_minor,
                    expected={
                        "run_key": __import__("hashlib").sha256(
                            args.run_id.encode()).digest()[:16],
                        "fence_epoch": _fence_epoch(args),
                        "generation": local_result.generation,
                        "attempt": local_result.attempt,
                        "owner_epoch": local_result.client.owner_epoch,
                        "worker_key": __import__("hashlib").sha256(
                            peer_id.encode()).digest()[:16],
                        "incarnation": __import__("hashlib").sha256(
                            peer_incarnation.encode()).digest()[:16],
                        "layout_digest": local_result.layout_digest,
                        "base_digest": local_result.base_digest,
                        "result_root": local_result.result_root,
                        "weight": local_result.global_weight,
                        "chunk_count": chunk_count,
                    })
                chunk = int(value["chunk_index"])
                if chunk not in expected_chunks:
                    raise ValueError("native owner credit named an unassigned shard")
                _local_offset, expected_wire_offset, expected_extent = expected_chunks[chunk]
                if (int(value["payload_offset"]) != expected_wire_offset
                        or int(value["credit"]) != expected_extent):
                    raise ValueError("native owner credit extent mismatch")
                seen.add(chunk)
            finally:
                os.close(credit_fd)

    def receive_all(selected=receive_ranges) -> None:
        expected_chunks = {
            chunk: (local_offset, wire_offset, extent)
            for chunk, local_offset, wire_offset, extent in selected
        }
        seen: set[int] = set()
        capacity = payload_max + 320
        while len(seen) != len(expected_chunks):
            if time.monotonic() >= deadline:
                raise TimeoutError("native owner redistribution deadline expired")
            if inbox is None:
                frame_fd = create_memfd("emender-ndp-rx-frame", allow_sealing=True)
                os.ftruncate(frame_fd, capacity)
                received = session.receive_owner_fd(frame_fd, capacity=capacity)
                if received is None:
                    os.close(frame_fd)
                    time.sleep(.001)
                    continue
                worker, frame_bytes = received
            else:
                frame_fd, frame_bytes = inbox.receive(peer_id, deadline=deadline)
                worker = peer_id
            try:
                if worker != peer_id:
                    raise ValueError("native owner frame arrived from wrong frozen peer")
                value = decode_owner_frame_fd(
                    frame_fd, frame_bytes=frame_bytes, payload_max=payload_max,
                    protocol_major=
                        args._async_v21_policy.wire_protocol_major,
                    protocol_minor=
                        args._async_v21_policy.wire_protocol_minor,
                    expected={
                        "run_key": __import__("hashlib").sha256(
                            args.run_id.encode()).digest()[:16],
                        "fence_epoch": _fence_epoch(args),
                        "generation": local_result.generation,
                        "attempt": local_result.attempt,
                        "owner_epoch": local_result.client.owner_epoch,
                        "worker_key": __import__("hashlib").sha256(
                            peer_id.encode()).digest()[:16],
                        "incarnation": __import__("hashlib").sha256(
                            peer_incarnation.encode()).digest()[:16],
                        "layout_digest": peer_layout,
                        "base_digest": local_result.base_digest,
                        "result_root": peer_root,
                        "weight": peer_weight,
                        "chunk_count": chunk_count,
                    })
                chunk = int(value["chunk_index"])
                if chunk not in expected_chunks:
                    raise ValueError("native owner frame named an unassigned shard")
                if chunk in seen:
                    continue  # authenticated idempotent replay
                local_offset, expected_wire_offset, expected_extent = expected_chunks[chunk]
                if (int(value["payload_offset"]) != expected_wire_offset
                        or int(value["payload_bytes"]) != expected_extent):
                    raise ValueError("native redistribution chunk offset mismatch")
                _copy_frame_payload(
                    frame_fd, remote_fd, payload_bytes=int(value["payload_bytes"]),
                    payload_offset=local_offset)
                seen.add(chunk)
            finally:
                os.close(frame_fd)

    try:
        # Hash the immutable local numerator while CXI is consuming it. The
        # remote raw digest follows the transfer while those pages are still
        # resident. Both become metadata for native admission; Python never
        # carries a dense value or sends one through a socket.
        with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="native-local-result-digest"
                ) as digest_executor:
            if local_digest_value is not None:
                if len(bytes(local_digest_value)) != 32:
                    raise ValueError("native local digest metadata is invalid")
                local_digest = None
            else:
                local_digest = (local_digest_future
                                if local_digest_future is not None
                                else digest_executor.submit(local_result.sha256))
            # Static byte credits are distinct from CQ completion and bound
            # each exact frozen chunk.  Handshake one chunk at a time on each
            # peer route: FI_EP_RDM does not promise message ordering, so a
            # batched CREDIT followed by batched DATA can otherwise expose a
            # later DATA before an earlier CREDIT.  The per-peer handshake
            # keeps at most one logical frame outstanding without introducing
            # a fixed-world rendezvous or reducing concurrency across peers.
            if f"node-{node}" < peer_id:
                for item in send_ranges:
                    receive_credits((item,))
                    send_all(sequence_base=1, selected=(item,))
                # The higher worker echoes the last granted credit only after
                # consuming the last DATA.  Waiting for that authenticated,
                # idempotent replay closes the first direction before a
                # reverse CREDIT can overtake its final DATA on FI_EP_RDM.
                receive_credits((send_ranges[-1],))
                for item in receive_ranges:
                    send_credits(sequence_base=chunk_count + 1,
                                 selected=(item,))
                    receive_all((item,))
            else:
                for item in receive_ranges:
                    send_credits(sequence_base=1, selected=(item,))
                    receive_all((item,))
                send_credits(sequence_base=1,
                             selected=(receive_ranges[-1],))
                for item in send_ranges:
                    receive_credits((item,))
                    send_all(sequence_base=chunk_count + 1,
                             selected=(item,))
            seal_memfd(remote_fd)
            remote_digest = fd_sha256(remote_fd, length=remote_length)
            return remote_fd, (bytes(local_digest_value) if local_digest is None
                               else local_digest.result()), remote_digest
    except BaseException:
        os.close(remote_fd)
        raise


def _packed_range_map(
        ranges: tuple[tuple[int, int, int], ...]
        ) -> tuple[tuple[int, int, int, int], ...]:
    """Map canonical disjoint wire ranges onto one compact local extent."""
    packed: list[tuple[int, int, int, int]] = []
    cursor = 0
    prior_end = 0
    seen: set[int] = set()
    for shard, wire_offset, extent in ranges:
        shard, wire_offset, extent = int(shard), int(wire_offset), int(extent)
        if (shard < 0 or shard in seen or wire_offset < prior_end
                or extent <= 0):
            raise ValueError("native owner ranges are invalid or overlapping")
        seen.add(shard)
        packed.append((shard, cursor, wire_offset, extent))
        cursor += extent
        prior_end = wire_offset + extent
    if not packed:
        raise ValueError("native owner ranges are empty")
    return tuple(packed)


def _packed_owner_fd(source_fd: int, *,
                     ranges: tuple[tuple[int, int, int], ...]
                     ) -> tuple[int, tuple[tuple[int, int, int, int], ...], bytes]:
    """Retain assigned round-robin shards in one compact sealed replay memfd."""
    if source_fd < 0:
        raise ValueError("native packed owner source is invalid")
    packed_ranges = _packed_range_map(ranges)
    source_length = os.fstat(source_fd).st_size
    packed_length = sum(extent for _shard, _local, _wire, extent in packed_ranges)
    target = create_memfd("emender-ndp-packed-owner", allow_sealing=True)
    try:
        os.ftruncate(target, packed_length)
        for _shard, local_offset, wire_offset, extent in packed_ranges:
            if wire_offset + extent > source_length:
                raise ValueError("native packed owner range exceeds its source")
            copy_fd_range(
                source_fd, target, extent, source_offset=wire_offset,
                destination_offset=local_offset)
        seal_memfd(target)
        return (target, packed_ranges,
                fd_sha256(target, length=packed_length))
    except BaseException:
        os.close(target)
        raise


def _native_sharded_owner_reduce(
        session: NativeManagerSession, local_operation, local_result, freeze, *,
        args, node: int, incarnation: str,
        remote_endpoints: tuple[OwnerEndpoint, ...], workers: list[str],
        weights: dict[str, int], roots: dict[str, bytes], base_digest: bytes,
        plan_digest: bytes, pool_client: PoolControlClient, bulk: Path,
        identity: str, generation: int, exchange_deadline: float,
        semantic_identity: dict[str, object],
        ):
    """Reduce by balanced shard owners and redistribute one exact native view.

    Every sender moves each f64 numerator shard exactly once to its deterministic
    owner. Owners pack their disjoint round-robin shards into compact native
    layouts, reduce those layouts, and send each compact f32 owner result to
    every node using the canonical wire offsets. Two bounded local native
    conversions install the assembled result as the ordinary read-only service
    view without carrying dense values through Python or a Python socket.
    """
    def remaining_s() -> float:
        remaining = exchange_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native sharded owner phase deadline expired")
        return max(.001, min(float(args.deadline_s), remaining))

    local_worker = f"node-{node}"
    peer_by_worker = {item.worker_id: item for item in remote_endpoints}
    worker_ids = tuple(workers)
    f64_layout_bytes = int(local_result.length)
    full_layout_digest = bytes(local_result.layout_digest)
    total_elements = f64_layout_bytes // 8
    contribution_ranges = _native_owner_ranges(
        worker_ids, run_id=args.run_id, fence=_fence_epoch(args),
        generation=generation, attempt=1, owner_epoch=1,
        f64_layout_bytes=f64_layout_bytes, payload_max=args.bulk_chunk_bytes,
        itemsize=8)
    result_ranges = _native_owner_ranges(
        worker_ids, run_id=args.run_id, fence=_fence_epoch(args),
        generation=generation, attempt=1, owner_epoch=1,
        f64_layout_bytes=f64_layout_bytes, payload_max=args.bulk_chunk_bytes,
        itemsize=4)
    shard_count = sum(len(value) for value in contribution_ranges.values())
    owner_bytes = {
        worker: sum(extent for _shard, _offset, extent in ranges)
        for worker, ranges in contribution_ranges.items()
    }
    if (set(worker_ids) != set(contribution_ranges)
            or max(owner_bytes.values()) - min(owner_bytes.values())
               > args.bulk_chunk_bytes):
        raise RuntimeError("native shard-owner placement is not balanced")

    local_packed_fd, local_packed_ranges, local_packed_digest = _packed_owner_fd(
        local_result.fd, ranges=contribution_ranges[local_worker])
    remote_packed_fds: dict[str, int] = {}
    remote_packed_digests: dict[str, bytes] = {}
    contribution_started = time.monotonic()
    try:
        frame_budgets = {
            peer: (len(contribution_ranges[peer])
                   + len(contribution_ranges[local_worker])
                   + (1 if local_worker < peer else 0))
            for peer in peer_by_worker
        }
        with _NativePeerInbox(
                session, peer_ids=tuple(peer_by_worker),
                capacity=args.bulk_chunk_bytes + 320,
                frames_per_peer=frame_budgets, deadline=exchange_deadline,
                queue_slots=4) as inbox:
            with ThreadPoolExecutor(
                    max_workers=len(peer_by_worker),
                    thread_name_prefix="native-shard-contribution") as executor:
                futures = {
                    peer: executor.submit(
                        _native_peer_exchange, session, local_result, args=args,
                        node=node, peer_id=peer,
                        peer_incarnation=endpoint.incarnation,
                        peer_root=roots[peer], peer_weight=weights[peer],
                        deadline=exchange_deadline, inbox=inbox,
                        local_digest_value=local_result.result_root,
                        send_ranges=contribution_ranges[peer],
                        receive_ranges=local_packed_ranges,
                        receive_length=owner_bytes[local_worker],
                        wire_chunk_count=shard_count)
                    for peer, endpoint in peer_by_worker.items()
                }
                failures = []
                for peer, future in futures.items():
                    try:
                        remote_fd, _unused_local_digest, remote_digest = future.result()
                        remote_packed_fds[peer] = remote_fd
                        remote_packed_digests[peer] = remote_digest
                    except BaseException as error:
                        failures.append(error)
                if failures:
                    raise failures[0]
        _stage_telemetry(
            bulk, identity, generation, "native_owner_contribution",
            contribution_started, pool_client.timeout_s,
            sent_bytes=sum(owner_bytes[peer] for peer in peer_by_worker),
            received_bytes=owner_bytes[local_worker] * len(peer_by_worker),
            owner_count=len(worker_ids), shard_count=shard_count,
            maximum_owner_bytes=max(owner_bytes.values()),
            minimum_owner_bytes=min(owner_bytes.values()),
            balanced_owner_placement=True, python_dense_socket_bytes=0,
            receive_queue_high_water_frames=inbox.high_water_frames,
            **semantic_identity)

        # Attempt 1 produced the retained full-layout f64 numerator. Attempt 2
        # installs only this owner's compact element count, imports one compact
        # source per frozen worker, and finalizes its assigned f32 result shards
        # with the exact accepted-token weight.
        session.abort(deadline_s=5)
        local_result.close(); local_operation.close(); freeze.close()
        compact_layout_digest = session.install_generation(
            total_elements=owner_bytes[local_worker] // 8,
            generation=generation, attempt=2, owner_epoch=1,
            source_dtype=DType.F64, base_digest=base_digest,
            plan_digest=plan_digest, payload_max=args.bulk_chunk_bytes,
            deadline_s=remaining_s())
        imported_sources = tuple(
            ((local_packed_fd if worker == local_worker else remote_packed_fds[worker]),
             worker,
             incarnation if worker == local_worker else peer_by_worker[worker].incarnation,
             sequence, weights[worker],
             local_packed_digest if worker == local_worker
             else remote_packed_digests[worker])
            for sequence, worker in enumerate(workers))
        with session.import_reduction_sources(
                imported_sources, source_dtype=DType.F64,
                deadline_s=remaining_s()):
            owner_freeze = session.freeze(deadline_s=remaining_s())
            owner_operation, owner_result = session.finalize_redistribution(
                deadline_s=remaining_s())
    finally:
        os.close(local_packed_fd)
        for remote_fd in remote_packed_fds.values():
            os.close(remote_fd)

    if (owner_result.length * 2 != owner_bytes[local_worker]
            or owner_result.layout_digest != compact_layout_digest):
        owner_result.close(); owner_operation.close(); owner_freeze.close()
        session.abort(deadline_s=1)
        raise RuntimeError("native compact owner result extent/layout is invalid")
    owner_map = pool_client.announce_owner_result(
        generation=generation, attempt=1, worker_id=local_worker,
        incarnation=incarnation, result_root=owner_result.result_root.hex(),
        layout_digest=owner_result.layout_digest.hex(),
        global_weight=owner_result.global_weight, result_bytes=owner_result.length,
        deadline=exchange_deadline)
    owner_records = dict(owner_map["owners"])
    if set(owner_records) != set(worker_ids):
        raise RuntimeError("native compact owner map differs from frozen membership")
    parsed_owner_records = {}
    for worker, record in owner_records.items():
        value = dict(record)
        root = bytes.fromhex(str(value["result_root"]))
        layout = bytes.fromhex(str(value["layout_digest"]))
        result_bytes = int(value["result_bytes"])
        expected_bytes = sum(extent for _shard, _offset, extent
                             in result_ranges[worker])
        if (len(root) != 32 or len(layout) != 32
                or result_bytes != expected_bytes):
            raise RuntimeError("native compact owner metadata is invalid")
        parsed_owner_records[worker] = (root, layout, result_bytes)

    result_packed_ranges = {
        worker: _packed_range_map(result_ranges[worker])
        for worker in worker_ids
    }
    full_result_bytes = f64_layout_bytes // 2
    aggregate_fd = create_memfd("emender-ndp-assembled-result", allow_sealing=True)
    remote_result_fds: dict[str, int] = {}
    redistribution_started = time.monotonic()
    try:
        os.ftruncate(aggregate_fd, full_result_bytes)
        for _shard, local_offset, wire_offset, extent in result_packed_ranges[local_worker]:
            copy_fd_range(
                owner_result.fd, aggregate_fd, extent, source_offset=local_offset,
                destination_offset=wire_offset)
        frame_budgets = {
            peer: (len(result_ranges[local_worker]) + len(result_ranges[peer])
                   + (1 if local_worker < peer else 0))
            for peer in peer_by_worker
        }
        with _NativePeerInbox(
                session, peer_ids=tuple(peer_by_worker),
                capacity=args.bulk_chunk_bytes + 320,
                frames_per_peer=frame_budgets, deadline=exchange_deadline,
                queue_slots=4) as inbox:
            with ThreadPoolExecutor(
                    max_workers=len(peer_by_worker),
                    thread_name_prefix="native-shard-redistribution") as executor:
                futures = {
                    peer: executor.submit(
                        _native_peer_exchange, session, owner_result, args=args,
                        node=node, peer_id=peer,
                        peer_incarnation=endpoint.incarnation,
                        peer_root=parsed_owner_records[peer][0],
                        peer_weight=owner_result.global_weight,
                        peer_layout_digest=parsed_owner_records[peer][1],
                        deadline=exchange_deadline, inbox=inbox,
                        local_digest_value=owner_result.result_root,
                        send_ranges=result_packed_ranges[local_worker],
                        receive_ranges=result_packed_ranges[peer],
                        receive_length=parsed_owner_records[peer][2],
                        wire_chunk_count=shard_count)
                    for peer, endpoint in peer_by_worker.items()
                }
                failures = []
                for peer, future in futures.items():
                    try:
                        remote_fd, _unused_local_digest, _remote_digest = future.result()
                        remote_result_fds[peer] = remote_fd
                    except BaseException as error:
                        failures.append(error)
                if failures:
                    raise failures[0]
        for peer, remote_fd in remote_result_fds.items():
            for _shard, local_offset, wire_offset, extent in result_packed_ranges[peer]:
                copy_fd_range(
                    remote_fd, aggregate_fd, extent, source_offset=local_offset,
                    destination_offset=wire_offset)
        seal_memfd(aggregate_fd)
        aggregate_digest = fd_sha256(aggregate_fd, length=full_result_bytes)
        result_owner_bytes = {
            worker: sum(extent for _shard, _offset, extent in ranges)
            for worker, ranges in result_ranges.items()
        }
        _stage_telemetry(
            bulk, identity, generation, "native_owner_redistribution",
            redistribution_started, pool_client.timeout_s,
            sent_bytes=result_owner_bytes[local_worker] * len(peer_by_worker),
            received_bytes=sum(result_owner_bytes[peer] for peer in peer_by_worker),
            owner_count=len(worker_ids), shard_count=shard_count,
            maximum_owner_bytes=max(result_owner_bytes.values()),
            minimum_owner_bytes=min(result_owner_bytes.values()),
            balanced_owner_placement=True, python_dense_socket_bytes=0,
            receive_queue_high_water_frames=inbox.high_water_frames,
            **semantic_identity)
    except BaseException:
        os.close(aggregate_fd)
        raise
    finally:
        for remote_fd in remote_result_fds.values():
            os.close(remote_fd)

    # Install the complete f32 aggregate into the ordinary native result
    # lifecycle. F32->f64 retains the accepted-token numerator; f64->f32 then
    # divides once by that same exact weight. Both passes are native and local.
    session.abort(deadline_s=5)
    owner_result.close(); owner_operation.close(); owner_freeze.close()
    assembled_identity = f"assembled:{_fence_epoch(args)}:{generation}"
    accepted_tokens = sum(weights.values())
    restored_layout_digest = session.install_generation(
        total_elements=total_elements, generation=generation,
        attempt=1, owner_epoch=1,
        source_dtype=DType.F32, base_digest=base_digest,
        plan_digest=plan_digest, payload_max=args.bulk_chunk_bytes,
        deadline_s=remaining_s())
    if restored_layout_digest != full_layout_digest:
        os.close(aggregate_fd)
        session.abort(deadline_s=1)
        raise RuntimeError("native restored full layout differs from attempt 1")
    try:
        with session.import_reduction_sources(
                ((aggregate_fd, "assembled-global", assembled_identity, 0,
                  accepted_tokens, aggregate_digest),),
                source_dtype=DType.F32, deadline_s=remaining_s()):
            bridge_freeze = session.freeze(deadline_s=remaining_s())
            bridge_operation, bridge_result = session.finalize_redistribution(
                deadline_s=remaining_s())
    finally:
        os.close(aggregate_fd)
    bridge_fd = os.dup(bridge_result.fd)
    bridge_digest = bridge_result.sha256()
    if (bridge_result.length != f64_layout_bytes
            or bridge_result.layout_digest != full_layout_digest):
        os.close(bridge_fd)
        bridge_result.close(); bridge_operation.close(); bridge_freeze.close()
        session.abort(deadline_s=1)
        raise RuntimeError("native assembled bridge result is invalid")
    session.abort(deadline_s=5)
    bridge_result.close(); bridge_operation.close(); bridge_freeze.close()
    # The final install RPC remains inside the owner-exchange deadline, while
    # the installed native result must remain valid through the distinct
    # bounded recovery-receipt/commit phase.  Keeping those deadlines separate
    # prevents a healthy durable publication from reaching COMMIT with an
    # already-expired service generation after slow full-model checkpoint I/O.
    final_operation_deadline_s = remaining_s()
    session.install_reduction_attempt(
        generation=generation, attempt=2, owner_epoch=1,
        source_dtype=DType.F64, base_digest=base_digest,
        plan_digest=plan_digest, deadline_s=final_operation_deadline_s,
        generation_deadline_s=(
            final_operation_deadline_s
            + _native_post_result_lifetime_s(args)))
    try:
        with session.import_reduction_sources(
                ((bridge_fd, "assembled-global", assembled_identity, 0,
                  accepted_tokens, bridge_digest),),
                source_dtype=DType.F64, deadline_s=remaining_s()):
            final_freeze = session.freeze(deadline_s=remaining_s())
            final_operation, final_result = session.finalize_redistribution(
                deadline_s=remaining_s())
    finally:
        os.close(bridge_fd)
    if (final_result.length != full_result_bytes
            or final_result.layout_digest != full_layout_digest
            or final_result.attempt != 2):
        final_result.close(); final_operation.close(); final_freeze.close()
        session.abort(deadline_s=1)
        raise RuntimeError("native final assembled result is invalid")
    _stage_telemetry(
        bulk, identity, generation, "native_redistribution",
        redistribution_started, pool_client.timeout_s,
        contributions=len(imported_sources), input_dtype="float64",
        result_dtype="float32", result_bytes=final_result.length,
        owner_count=len(worker_ids), owner_shards=shard_count,
        python_dense_socket_bytes=0)
    return final_operation, final_result, final_freeze


def _native_manager_session_lifetime_s(args) -> float:
    """Bound one endpoint identity across this finite generation sequence."""
    # ``deadline_s`` remains the per-generation/per-operation ceiling at every
    # call site below.  The transport open deadline and signed endpoint expiry
    # are session lifetime bounds, so they must cover the whole allocation
    # rather than expiring while a healthy manager advertises READY for the
    # next generation.  Candidate publication, safe-boundary rendezvous, and
    # atomic apply each have distinct post-result clocks, so generation count
    # times the admission ceiling alone is not a sufficient session lifetime.
    configured_s = float(args.deadline_s) * max(1, int(args.generations))
    requested = os.environ.get("RESILIENT_E97_REQUESTED_WALLTIME")
    if not requested:
        return configured_s
    try:
        hours, minutes, seconds = (int(part) for part in requested.split(":"))
    except (TypeError, ValueError):
        raise ValueError(
            "RESILIENT_E97_REQUESTED_WALLTIME must be HH:MM:SS") from None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(
            "RESILIENT_E97_REQUESTED_WALLTIME must be HH:MM:SS")
    # The session starts after allocation admission, so a lease this long
    # necessarily outlives Slurm's remaining wall clock.  Slurm is still the
    # authoritative outer hard stop.
    requested_s = float(hours * 3600 + minutes * 60 + seconds)
    return max(configured_s, requested_s)


def _native_post_result_lifetime_s(args) -> float:
    """Keep a finalized result valid through each distinct bounded phase."""
    checkpoint_candidate_s = min(float(args.deadline_s), 420.0)
    return (
        checkpoint_candidate_s
        + ASYNC_V21_BOUNDARY_RENDEZVOUS_S
        + ASYNC_V21_ALL_EIGHT_APPLY_S
    )


def _native_manager(args) -> int:
    """Model-free controller for direct memfd admission and native ownership."""
    if args.local_quorum != 8 and not args.control:
        raise ValueError("native E97 production requires all eight local trainers")
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    identity = f"node-{node}-manager"
    bulk = assert_node_local_path(
        Path(args.bulk_root) / args.run_id / f"node-{node}", run)
    fenced = _peer_authority(args)
    backend, production, full_layout = _dataplane_policy(args)
    provider = "cxi" if backend == NATIVE_CXI else os.environ.get(
        "NDP_TEST_PROVIDER", "tcp;ofi_rxm")
    config_path = ROOT / "configs/frontier/e97_async_256.yaml"
    digests = runtime_digests(
        build_manifest=args.native_build_manifest, config_path=config_path,
        provider=provider, attestation=args._dataplane_attestation)
    incarnation = (
        os.environ.get("RESILIENT_E97_NODE_INCARNATION") or uuid.uuid4().hex)
    session = NativeManagerSession.start(
        backend=backend, run_id=args.run_id, fence_epoch=_fence_epoch(args),
        worker_id=f"node-{node}", incarnation=incarnation,
        host=_pool_hosts(args)[node], build_manifest=args.native_build_manifest,
        gate_json=args.native_gate_json or None, source_root=ROOT,
        production=production, full_layout=full_layout,
        required_gate=os.environ.get("NDP_REQUIRED_GATE", "G2"),
        deadline_s=_native_manager_session_lifetime_s(args),
        telemetry_path=bulk / "telemetry" / f"{identity}-native.jsonl",
        payload_max=args.bulk_chunk_bytes, resident_limit_bytes=args.max_spool_bytes)
    control = bulk / "control"
    session.write_readiness(control / "native-service-ready.json")
    start_generation, sync_evidence = _native_manager_resume_point(
        run, args, fenced, native_runtime=digests)
    _validate_atomic_cohort_recovery(
        control, args, node=node, node_incarnation=incarnation,
        generation=start_generation,
        admitted_statuses=("reconstructing", "reconstructed"))
    atomic_metadata(
        control / f"native-manager-sync-{incarnation}.json", {
            "schema": "emender-native-manager-sync-v1",
            "run_id": args.run_id, "worker_id": f"node-{node}",
            "incarnation": incarnation, "fence_epoch": _fence_epoch(args),
            **sync_evidence,
        })
    heartbeat(bulk, identity, generation=start_generation,
              step=start_generation * args.local_steps, loss=None,
              stage="native_service_ready")
    term_requested = {"value": False}

    def request_term(*_ignored) -> None:
        term_requested["value"] = True

    signal.signal(signal.SIGTERM, request_term)
    pool_config = _pool_config(
        args,
        committed_generation=start_generation,
        committed_receipt_digest=str(
            sync_evidence.get("commit_receipt_digest", "")),
        committed_accepted_tokens=int(
            sync_evidence.get("accepted_tokens", 0)),
        committed_manifest_digest=str(
            sync_evidence.get("manifest_sha256", ""))
        if start_generation > 0 else "",
        committed_result_root=str(
            sync_evidence.get("result_root", ""))
        if start_generation > 0 else "",
        committed_apply_receipts=tuple(
            (str(item["worker_id"]), str(item["receipt_digest"]))
            for item in sync_evidence.get("apply_receipts", [])
        ),
    )
    control_server = control_thread = None
    pool_client = None
    if args.node_count > 1:
        if args.node_count not in (2, 4, 8, 16, 32, 64, 256):
            raise ValueError(
                "async-decoupled-v2.1 qualification requires an exact "
                "serial-ladder node count")
        if node == 0:
            control_server = PoolControlServer(
                ("0.0.0.0", args.coordinator_port), pool_config,
                evidence_root=run / "retained-evidence" / "pool-control")
            control_thread = threading.Thread(target=control_server.serve_forever, daemon=True)
            control_thread.start()
        pool_client = PoolControlClient(
            (args.coordinator_host, args.coordinator_port),
            timeout_s=min(args.deadline_s, pool_config.slo.sync_s)).bind(
                args.run_id, _fence_epoch(args))
        recovery_handshake = pool_client.recover(
            worker_id=f"node-{node}",
            incarnation=incarnation,
            known_generation=start_generation,
            known_receipt_digest=str(
                sync_evidence.get("commit_receipt_digest", "")),
        )
        _validate_native_recovery_handshake(
            recovery_handshake, sync_evidence,
            generation=start_generation)
        if _wait_native_ready_delay(
                control, args, node=node, generation=start_generation,
                incarnation=incarnation, term_requested=term_requested):
            _ready_recovered_peer(
                pool_client, session.owner_endpoint,
                generation=start_generation,
                run_id=args.run_id, fence=_fence_epoch(args),
                apply_receipt_digest=_node_apply_receipt_digest(
                    sync_evidence, worker_id=f"node-{node}"),
                deadline=time.monotonic() + pool_config.slo.sync_s)
    liveness_stop, liveness_thread = _liveness_heartbeat(bulk, identity)
    terminal_published = False
    try:
        target_generation = args.initial_generation + args.generations
        for generation in range(start_generation, target_generation):
            if term_requested["value"]:
                break
            native_deadline = time.monotonic() + min(args.deadline_s, 420.0)
            snapshot = (pool_client.open_generation(
                generation, 1, deadline=time.monotonic() + pool_config.slo.sync_s)
                if pool_client is not None else None)
            request = wait_metadata(
                control / f"native-layout-{generation:08d}.json",
                deadline=native_deadline,
                expected={"run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                          "generation": generation, "rank": 0,
                          **({"node_incarnation": incarnation}
                             if os.environ.get(
                                 "RESILIENT_E97_NODE_INCARNATION")
                             else {})})
            elements = int(request["total_elements"])
            expected_layout = layout_identity(
                elements, payload_max=args.bulk_chunk_bytes)
            if request.get("layout_digest") != expected_layout.hex():
                raise ValueError("trainer layout differs from native flat-layout ABI")
            base_digest = bytes.fromhex(str(request["base_digest"]))
            plan_digest = __import__("hashlib").sha256(json.dumps(
                {"generation": generation, "runtime_digests": digests,
                 "policy_digest": args._async_v21_policy.digest},
                sort_keys=True, separators=(",", ":")).encode()).digest()
            session.install_generation(
                total_elements=elements, generation=generation, attempt=1,
                owner_epoch=1, source_dtype=DType.F32,
                payload_max=args.bulk_chunk_bytes, base_digest=base_digest,
                plan_digest=plan_digest,
                deadline_s=max(.001, native_deadline - time.monotonic()))
            metadata = GenerationMetadata(
                args.run_id, _fence_epoch(args), generation, 1, 1, elements,
                expected_layout.hex(), base_digest.hex(), plan_digest.hex(),
                session.local.generation_deadline_ns, digests,
                policy_id=args._async_v21_policy.policy_id,
                policy_digest=args._async_v21_policy.digest,
                code_digest=hashlib.sha256(args.code_id.encode()).hexdigest(),
                base_global_version=generation,
                local_window_start=generation,
                local_window_end=generation + 1,
                policy_schema=args._async_v21_policy.policy_schema,
                contribution_schema=args._async_v21_policy.contribution_schema,
                native_abi=args._async_v21_policy.native_abi,
                wire_protocol_major=args._async_v21_policy.wire_protocol_major,
                wire_protocol_minor=args._async_v21_policy.wire_protocol_minor,
                stable_worker_id=f"node-{node}",
                worker_incarnation=incarnation)
            atomic_metadata(control / f"native-generation-{generation:08d}.json",
                            metadata.as_json())
            heartbeat(bulk, identity, generation=generation,
                      step=generation * args.local_steps, loss=None, stage="training_wait")
            submissions = []
            for rank in range(args.local_quorum):
                submission = wait_metadata(
                    control / f"native-submit-{generation:08d}-{rank:02d}.json",
                    deadline=native_deadline,
                    expected={"run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                              "generation": generation, "rank": rank,
                              "layout_digest": expected_layout.hex(),
                              "policy_id": args._async_v21_policy.policy_id,
                              "policy_digest": args._async_v21_policy.digest,
                              "worker_incarnation": incarnation})
                base_version = int(submission.get("base_global_version", -1))
                lag = generation - base_version
                if (lag != int(submission.get("base_lag_at_seal", -1))
                        or not 0 <= lag
                        <= args._async_v21_policy.max_commit_lag
                        or int(submission.get("local_window_end", -1))
                        <= int(submission.get("local_window_start", -1))
                        or len(str(submission.get(
                            "descriptor_digest", ""))) != 64
                        or int(submission.get("exact_tokens", 0)) <= 0
                        or "aggregation_weight" in submission):
                    raise ValueError(
                        "native submission has unverifiable v2.1 "
                        "base/lag/exact-token identity")
                submissions.append(submission)
                # An accepted rank is monotonic lifecycle progress.  Refresh
                # the manager deadline at each acceptance instead of expiring
                # an advancing generation from its initial training_wait time.
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps + len(submissions),
                          loss=None, stage="training_wait")
            local_weight = sum(
                int(item["exact_tokens"]) for item in submissions)
            contribution_base_versions = {
                int(item["base_global_version"]) for item in submissions
            }
            if len(contribution_base_versions) != 1:
                raise ValueError(
                    "node-local v2 trainer lanes disagree on original base")
            contribution_base_version = contribution_base_versions.pop()
            commit_lag = generation - contribution_base_version
            contribution_digest = hashlib.sha256(json.dumps(
                sorted(str(item["source_sha256"]) for item in submissions),
                separators=(",", ":")).encode()).hexdigest()
            local_semantic_identity = {
                "policy_id": args._async_v21_policy.policy_id,
                "policy_schema": args._async_v21_policy.policy_schema,
                "allocation_fence": _fence_epoch(args),
                "base_global_version": contribution_base_version,
                "commit_global_version": generation,
                "commit_lag": commit_lag,
                "exact_tokens": local_weight,
                "contribution_digest": contribution_digest,
            }
            heartbeat(bulk, identity, generation=generation,
                      step=generation * args.local_steps, loss=None, stage="freeze")
            local_reduce_started = time.monotonic()
            freeze = session.freeze(
                deadline_s=max(.001, native_deadline - time.monotonic()))
            local_operation, local_result = session.finalize_redistribution(
                deadline_s=max(.001, native_deadline - time.monotonic()))
            _stage_telemetry(
                bulk, identity, generation, "native_local_reduction",
                local_reduce_started, pool_config.slo.sync_s,
                contributions=len(submissions), input_dtype="float32",
                result_dtype="float64", result_bytes=local_result.length,
                **local_semantic_identity)
            final_operation, final_result = local_operation, local_result
            if pool_client is not None:
                while True:
                    close = pool_client.contribute_and_freeze(
                        generation=generation, attempt=1,
                        worker_id=f"node-{node}",
                        incarnation=incarnation,
                        contribution_seq=generation,
                        accepted_tokens=local_weight,
                        payload_digest=local_result.result_root.hex(),
                        # ADR-002 gives the complete open-group-to-freeze phase
                        # one absolute 420-second generation clock. The
                        # 15-second native freeze SLO cannot replace that Q/T
                        # close window: harmless K40 skew can make one node
                        # contribute later.
                        deadline=native_deadline)
                    if close.get("status") != "rejoin":
                        break
                    rejoin_generation, rejoin_evidence = (
                        _native_manager_resume_point(
                            run, args, fenced, native_runtime=digests))
                    if rejoin_generation != generation:
                        raise ValueError(
                            "peer-control rejoin advanced beyond local generation")
                    _validate_native_rejoin_instruction(
                        close, rejoin_evidence, generation=generation)
                    rejoin_handshake = pool_client.recover(
                        worker_id=f"node-{node}",
                        incarnation=incarnation,
                        known_generation=rejoin_generation,
                        known_receipt_digest=str(
                            rejoin_evidence.get(
                                "commit_receipt_digest", "")),
                    )
                    _validate_native_recovery_handshake(
                        rejoin_handshake, rejoin_evidence,
                        generation=rejoin_generation)
                    _ready_recovered_peer(
                        pool_client, session.owner_endpoint,
                        generation=rejoin_generation,
                        run_id=args.run_id, fence=_fence_epoch(args),
                        apply_receipt_digest=_node_apply_receipt_digest(
                            rejoin_evidence,
                            worker_id=f"node-{node}"),
                        deadline=native_deadline)
                    snapshot = pool_client.open_generation(
                        generation, 1, deadline=native_deadline)
                    atomic_metadata(
                        control
                        / f"native-peer-control-rejoin-{generation:08d}.json", {
                            "schema":
                                "emender-native-peer-control-rejoin-v1",
                            "run_id": args.run_id,
                            "fence_epoch": _fence_epoch(args),
                            "worker_id": f"node-{node}",
                            "incarnation": incarnation,
                            "generation": generation,
                            "commit_receipt_digest":
                                rejoin_evidence[
                                    "commit_receipt_digest"],
                            "manifest_sha256":
                                rejoin_evidence["manifest_sha256"],
                            "result_root": rejoin_evidence["result_root"],
                            "accepted_tokens":
                                rejoin_evidence["accepted_tokens"],
                            "apply_receipts":
                                rejoin_evidence["apply_receipts"],
                        })
                    heartbeat(
                        bulk, identity, generation=generation,
                        step=generation * args.local_steps,
                        loss=None, stage="peer_control_rejoined",
                        commit_receipt_digest=
                            rejoin_evidence["commit_receipt_digest"])
                if close.get("status") == "catch_up":
                    authoritative_generation = int(
                        close.get("authoritative_generation", -1))
                    if (
                        authoritative_generation <= generation
                        or len(str(close.get("receipt_digest", ""))) != 64
                        or len(str(close.get("manifest_digest", ""))) != 64
                        or len(str(close.get("result_root", ""))) != 64
                        or int(close.get("accepted_tokens", -1)) < 0
                        or close.get("requires_reload") is not True
                    ):
                        raise ValueError(
                            "generation catch-up receipt lacks immutable authority")
                    atomic_metadata(
                        control
                        / f"native-generation-catch-up-{generation:08d}.json", {
                            "schema":
                                "emender-native-generation-catch-up-v1",
                            "run_id": args.run_id,
                            "fence_epoch": _fence_epoch(args),
                            "worker_id": f"node-{node}",
                            "incarnation": incarnation,
                            **close,
                        })
                    heartbeat(
                        bulk, identity, generation=authoritative_generation,
                        step=authoritative_generation * args.local_steps,
                        loss=None, stage="generation_catch_up",
                        source_generation=generation,
                        authoritative_generation=authoritative_generation,
                        commit_receipt_digest=close["receipt_digest"],
                        requires_reload=True)
                    # The model-free manager owns no model/optimizer state.
                    # Its successful exit leaves the supervisor restart budget
                    # untouched while the immutable handoff directs the local
                    # trainers' bounded authoritative reload.
                    final_result.close()
                    final_operation.close()
                    freeze.close()
                    terminal_published = True
                    return 0
                if close.get("status") != "commit_ready":
                    raise TimeoutError(f"native global freeze failed: {close}")
                frozen = tuple(close["frozen_identities"])
                workers = sorted(str(item["worker_id"]) for item in frozen)
                endpoints = tuple(_owner_endpoint_from_snapshot(peer)
                                  for peer in snapshot["peers"]
                                  if str(peer["worker_id"]) in workers)
                remote_endpoints = _native_remote_endpoints(
                    endpoints, local_worker_id=f"node-{node}",
                    minimum_contributions=args.global_quorum)
                session.install_routes(endpoints)
                weights = {str(key): int(value) for key, value in
                           dict(close["exact_tokens_by_worker"]).items()}
                roots = {str(key): bytes.fromhex(str(value)) for key, value in
                         dict(close["accepted_payloads"]).items()}
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps, loss=None,
                          stage="owner_transport")
                # The complete owner-transfer/finalization/redistribution phase
                # shares the normative 180-second absolute bound. Individual
                # pair readiness remains capped by the 15-second freeze SLO.
                owner_phase_deadline = time.monotonic() + min(args.deadline_s, 180.0)

                def await_route(peer_endpoint: OwnerEndpoint):
                    peer_id = peer_endpoint.worker_id
                    route_ready_started = time.monotonic()
                    route_ready = pool_client.await_peer_route_ready(
                        generation=generation, attempt=1,
                        worker_id=f"node-{node}", incarnation=incarnation,
                        peer_worker_id=peer_id,
                        peer_incarnation=peer_endpoint.incarnation,
                        deadline=min(owner_phase_deadline, time.monotonic()
                                     + pool_config.slo.freeze_s))
                    return peer_endpoint, route_ready, (
                        time.monotonic() - route_ready_started)

                with ThreadPoolExecutor(
                        max_workers=len(remote_endpoints),
                        thread_name_prefix="native-route-ready") as route_executor:
                    route_futures = [
                        route_executor.submit(await_route, peer_endpoint)
                        for peer_endpoint in remote_endpoints]
                    routes = [future.result() for future in route_futures]
                for peer_endpoint, route_ready, route_ready_elapsed in routes:
                    _stage_telemetry(
                        bulk, identity, generation, "native_route_readiness",
                        time.monotonic() - route_ready_elapsed,
                        pool_config.slo.freeze_s,
                        peer_id=peer_endpoint.worker_id,
                        participants=route_ready["workers"], pairwise=True,
                        python_dense_socket_bytes=0)
                final_operation, final_result, freeze = _native_sharded_owner_reduce(
                    session, local_operation, local_result, freeze,
                    args=args, node=node, incarnation=incarnation,
                    remote_endpoints=remote_endpoints, workers=workers,
                    weights=weights, roots=roots, base_digest=base_digest,
                    plan_digest=plan_digest, pool_client=pool_client,
                    bulk=bulk, identity=identity, generation=generation,
                    exchange_deadline=owner_phase_deadline,
                    semantic_identity=local_semantic_identity)
                validated = pool_client.validate_result_root(
                    generation=generation, attempt=1, worker_id=f"node-{node}",
                    incarnation=incarnation, result_root=final_result.result_root.hex(),
                    global_weight=final_result.global_weight,
                    result_bytes=final_result.length,
                    deadline=time.monotonic() + pool_config.slo.apply_s)
                if validated["status"] != "validated":
                    raise RuntimeError("native result root did not validate")
                # Owner transport is complete once the native service returns
                # and validates the exact result root.  Checkpoint publication
                # has its own supervised progress interval; do not leave the
                # manager falsely charged to the completed transport interval
                # while it waits for the trainer proposal and immutable commit.
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps, loss=None,
                          stage="checkpoint_commit")
            accepted_tokens = (
                local_weight if pool_client is None
                else int(close["accepted_tokens"]))
            accepted_local_contributions = [{
                "rank": int(item["rank"]),
                "trainer": str(item["trainer"]),
                "incarnation": str(item["incarnation"]),
                "contribution_sequence": int(
                    item["contribution_sequence"]),
                "local_window_start": int(item["local_window_start"]),
                "local_window_end": int(item["local_window_end"]),
                "window_count": int(item["window_count"]),
                "base_global_version": int(item["base_global_version"]),
                "payload_digest": str(item["payload_digest"]),
                "descriptor_digest": str(item["descriptor_digest"]),
            } for item in submissions]
            result_marker = {
                "schema": "emender-native-e97-result-v2.1",
                "run_id": args.run_id,
                "fence_epoch": _fence_epoch(args), "generation": generation,
                "attempt": final_result.attempt,
                "owner_epoch": session.local.owner_epoch,
                "source_dtype": int(session.local.source_dtype),
                "deadline_unix_ns": session.local.generation_deadline_ns,
                "operation_handle": final_operation.handle,
                "layout_digest": final_result.layout_digest.hex(),
                "base_digest": final_result.base_digest.hex(),
                "plan_digest": session.local.plan_digest.hex(),
                "result_root": final_result.result_root.hex(),
                "global_weight": accepted_tokens,
                "exact_tokens": accepted_tokens,
                "policy_id": args._async_v21_policy.policy_id,
                "policy_schema": args._async_v21_policy.policy_schema,
                "policy_digest": args._async_v21_policy.digest,
                "native_abi": args._async_v21_policy.native_abi,
                "wire_protocol_major":
                    args._async_v21_policy.wire_protocol_major,
                "wire_protocol_minor":
                    args._async_v21_policy.wire_protocol_minor,
                "base_global_version": contribution_base_version,
                "commit_global_version": generation,
                "commit_lag": commit_lag,
                "result_bytes": final_result.length,
                "members": [int(item["rank"]) for item in submissions],
                "accepted_local_contributions":
                    accepted_local_contributions,
                "accepted_peers": ([f"node-{node}"] if snapshot is None else
                                   sorted(str(item["worker_id"])
                                          for item in close["frozen_identities"])),
                "runtime_digests": digests, "trainer_spool_bytes": 0,
                "python_dense_socket_bytes": 0,
            }
            session.checkpoint_proposal(
                control / f"native-checkpoint-{generation:08d}.json", final_result,
                publisher=identity,
                metadata={"publication_generation": generation + 1,
                          "runtime_digests": digests})
            atomic_metadata(control / f"native-result-{generation:08d}.json",
                            result_marker)
            atomic_commit_started = time.monotonic()
            if node == 0:
                proposal = wait_metadata(
                    control / f"trainer-proposal-{generation:08d}.json",
                    deadline=time.monotonic() + pool_config.slo.apply_s,
                    expected={"generation": generation + 1})
                publication = finalize_checkpoint(
                    run, proposal["checkpoint"], run_id=args.run_id,
                    generation=int(proposal["generation"]), step=int(proposal["step"]),
                    async_chain=proposal["async_chain"], membership=proposal["membership"],
                    fence=LocalFence(**proposal["fence"]), source_id=args.source_id,
                    code_id=args.code_id,
                    outer_update_state=proposal["outer_update_state"],
                    migration=proposal["migration"],
                    accepted_tokens=int(proposal["accepted_tokens"]),
                    generation_identity=proposal["generation_identity"],
                    digests=proposal["digests"],
                    peer_authority=None if fenced is None else fenced[0],
                    allocation_claim=None if fenced is None else fenced[1])
            if fenced is None:
                latest = wait_metadata(
                    run / "handoff" / "latest.json",
                    deadline=time.monotonic() + pool_config.slo.apply_s,
                    expected={"generation": generation + 1,
                              "fence": _fence_epoch(args)})
                publication = Path(str(latest["manifest"]))
                commit_receipt = None
            else:
                if pool_client is None:
                    committed_peer_state = None
                    commit_receipt = fenced[0].current_commit(fenced[1])
                elif node == 0:
                    commit_receipt = fenced[0].current_commit(fenced[1])
                    if (
                        commit_receipt is None
                        or commit_receipt.generation != generation + 1
                    ):
                        raise RuntimeError(
                            "publisher did not create the expected commit receipt")
                    committed_peer_state = pool_client.commit_authority(
                        result_generation=generation + 1,
                        attempt=1,
                        receipt_digest=commit_receipt.receipt_digest,
                        previous_receipt_digest=(
                            commit_receipt.previous_receipt_digest),
                        manifest_digest=commit_receipt.manifest_sha256,
                        result_root=commit_receipt.result_root,
                        accepted_tokens=commit_receipt.accepted_tokens,
                    )
                else:
                    committed_peer_state = pool_client.wait_for_commit(
                        result_generation=generation + 1,
                        deadline=time.monotonic() + pool_config.slo.apply_s,
                    )
                    commit_receipt = fenced[0].current_commit(fenced[1])
                if (
                    commit_receipt is None
                    or commit_receipt.generation != generation + 1
                    or (
                        committed_peer_state is not None
                        and (
                            committed_peer_state.get("status") != "committed"
                            or committed_peer_state.get("receipt_digest")
                            != commit_receipt.receipt_digest
                            or committed_peer_state.get("manifest_digest")
                            != commit_receipt.manifest_sha256
                            or committed_peer_state.get("result_root")
                            != commit_receipt.result_root
                            or int(committed_peer_state.get(
                                "accepted_tokens", -1))
                            != commit_receipt.accepted_tokens
                        )
                    )
                ):
                    raise RuntimeError(
                        "native peer commit differs from immutable authority")
                latest = commit_receipt.pointer()
                publication = commit_receipt.manifest_path
                atomic_metadata(
                    control / f"peer-commit-{generation + 1:08d}.json", {
                        "schema": "emender-native-peer-commit-handoff-v1",
                        "run_id": args.run_id,
                        "fence": _fence_epoch(args),
                        "generation": generation + 1,
                        "commit_receipt_digest":
                            commit_receipt.receipt_digest,
                        "manifest_sha256": commit_receipt.manifest_sha256,
                        "result_root": commit_receipt.result_root,
                        "accepted_tokens": commit_receipt.accepted_tokens,
                        "source": "native_peer_memory",
                    })
            committed_evidence = (
                {
                    "authoritative_generation": generation + 1,
                    "commit_receipt_digest": commit_receipt.receipt_digest,
                }
                if commit_receipt is not None else {}
            )
            if node == 0:
                _stage_telemetry(
                    bulk, identity, generation, "fenced_atomic_commit",
                    atomic_commit_started, min(args.deadline_s, 420.0),
                    policy_id=args._async_v21_policy.policy_id,
                    allocation_fence=_fence_epoch(args),
                    base_global_version=generation,
                    commit_global_version=generation + 1,
                    commit_lag=1,
                    exact_tokens=accepted_tokens,
                    contribution_digest=final_result.result_root.hex(),
                    accepted_tokens=int(proposal["accepted_tokens"]),
                    membership=len(proposal["membership"]))
            # Shared-result reads, checkpoint I/O, hashing, and reload
            # verification are background preparation.  The one service-owned
            # aggregate has a capacity-one node-local reader credit; durable
            # checkpoint work may overlap after each reader releases it.  Do
            # not begin the 60-second all-eight foreground transaction until
            # every trainer has retained its fenced preparation receipt.
            heartbeat(bulk, identity, generation=generation,
                      step=generation * args.local_steps, loss=None,
                      stage="result_preparation", **committed_evidence)
            preparation_deadline = (
                time.monotonic() + min(args.deadline_s, 420.0))
            boundary_transaction, apply_release = (
                _coordinate_native_safe_boundary(
                    control,
                    args,
                    bulk=bulk,
                    identity=identity,
                    node=node,
                    generation=generation,
                    result_root=final_result.result_root.hex(),
                    node_incarnation=incarnation,
                    preparation_deadline=preparation_deadline,
                    committed_evidence=committed_evidence,
                )
            )
            apply_release_started = float(
                apply_release["released_monotonic_s"])
            heartbeat(bulk, identity, generation=generation,
                      step=generation * args.local_steps, loss=None,
                      stage="peer_apply", **committed_evidence)
            atomic_apply_deadline = float(
                apply_release["apply_deadline_monotonic_s"])
            # The released 60-second clock bounds the atomic model apply, not
            # the subsequent telemetry flush and create-once receipt rename.
            # Keep receipt discovery independently bounded, then fail closed
            # against the immutable apply-finish timestamp below.
            apply_receipt_deadline = (
                atomic_apply_deadline
                + ASYNC_V21_APPLY_RECEIPT_PUBLICATION_S)
            node_apply = AtomicEightTrainerApply(
                root=control,
                run_id=args.run_id,
                fence=_fence_epoch(args),
                node_id=f"node-{node}",
                node_incarnation=incarnation,
                result_version=generation + 1,
                result_digest=final_result.result_root.hex(),
                trainer_count=8,
                transaction_digest=(
                    boundary_transaction.transaction_digest),
            )
            durable_trainer_receipts = []
            trainer_apply_intervals: list[tuple[float, float]] = []
            for rank in range(args.local_quorum):
                applied = wait_metadata(
                    control / f"native-applied-{generation:08d}-{rank:02d}.json",
                    deadline=apply_receipt_deadline,
                    expected={"run_id": args.run_id,
                              "fence_epoch": _fence_epoch(args),
                              "generation": generation,
                              "result_root": final_result.result_root.hex(),
                              "rank": rank,
                              "node_incarnation": incarnation,
                              "transaction_digest":
                                  boundary_transaction.transaction_digest})
                apply_started = float(applied["apply_started_monotonic_s"])
                apply_finished = float(applied["apply_finished_monotonic_s"])
                if (
                    not math.isfinite(apply_started)
                    or not math.isfinite(apply_finished)
                    or apply_started < apply_release_started
                    or apply_finished < apply_started
                    or apply_finished > atomic_apply_deadline
                ):
                    raise ValueError(
                        "trainer apply receipt has an invalid monotonic interval")
                trainer_apply_intervals.append(
                    (apply_started, apply_finished))
                boundary_transaction.record_applied(
                    rank=rank,
                    trainer_incarnation=str(
                        applied["trainer_incarnation"]),
                    apply_started_monotonic_s=apply_started,
                    apply_finished_monotonic_s=apply_finished,
                )
                node_apply.record_trainer(
                    rank=rank,
                    trainer_incarnation=str(applied["trainer_incarnation"]),
                    recovery_digest=str(applied["recovery_digest"]),
                )
                durable_trainer_receipts.append((
                    rank,
                    str(applied["trainer_incarnation"]),
                    str(applied["recovery_digest"]),
                ))
            node_marker = node_apply.commit_node()
            apply_receipt_digest = ""
            if fenced is not None:
                if commit_receipt is None:
                    raise RuntimeError(
                        "node apply lacks immutable commit receipt")
                durable_node_apply = fenced[0].record_node_apply(
                    fenced[1],
                    commit_receipt,
                    node_id=f"node-{node}",
                    node_incarnation=incarnation,
                    trainer_receipts=durable_trainer_receipts,
                )
                apply_receipt_digest = durable_node_apply.receipt_digest
                if pool_client is not None:
                    applied_peer_state = pool_client.node_applied(
                        generation=generation + 1,
                        worker_id=f"node-{node}",
                        incarnation=incarnation,
                        receipt_digest=apply_receipt_digest,
                        commit_receipt_digest=commit_receipt.receipt_digest,
                    )
                    if applied_peer_state.get("status") != "node_applied":
                        raise RuntimeError(
                            "native peer node-apply handshake was not admitted")
            session.commit(
                publication_manifest=publication, authoritative_latest=latest,
                deadline_s=pool_config.slo.apply_s)
            if (
                len(trainer_apply_intervals) != 8
                or max(finish for _start, finish in trainer_apply_intervals)
                - apply_release_started
                > ASYNC_V21_ALL_EIGHT_APPLY_S
            ):
                raise TimeoutError(
                    "complete all-eight verified result apply exceeded 60 seconds")
            safe_boundary_metrics = boundary_transaction.telemetry()
            atomic_metadata(
                control
                / (
                    f"native-rendezvous-summary-"
                    f"{generation:08d}.json"
                ), {
                    "schema":
                        "emender-native-e97-rendezvous-summary-v2.1",
                    "run_id": args.run_id,
                    "fence_epoch": _fence_epoch(args),
                    "generation": generation,
                    "result_root": final_result.result_root.hex(),
                    "node_incarnation": incarnation,
                    "transaction_digest":
                        boundary_transaction.transaction_digest,
                    "metrics": safe_boundary_metrics,
                })
            assert boundary_transaction.opened_monotonic_s is not None
            _stage_telemetry(
                bulk, identity, generation,
                "native_node_boundary_rendezvous",
                boundary_transaction.opened_monotonic_s,
                ASYNC_V21_BOUNDARY_RENDEZVOUS_S,
                ended=apply_release_started,
                policy_id=args._async_v21_policy.policy_id,
                allocation_fence=_fence_epoch(args),
                base_global_version=generation,
                commit_global_version=generation + 1,
                contribution_digest=final_result.result_root.hex(),
                transaction_digest=(
                    boundary_transaction.transaction_digest),
                phase_scope="node_all_eight",
                foreground_interruption="safe_boundary_rendezvous",
                foreground_blocking=True,
                trainer_count=8,
                candidate_prepared_count=(
                    safe_boundary_metrics["candidate_prepared_count"]),
                boundary_ready_count=(
                    safe_boundary_metrics["boundary_ready_count"]),
                candidate_preparation=(
                    safe_boundary_metrics["candidate_preparation"]),
                boundary_rendezvous=(
                    safe_boundary_metrics["boundary_rendezvous"]),
                release_to_apply=(
                    safe_boundary_metrics["release_to_apply"]),
                total_foreground_idle=(
                    safe_boundary_metrics["total_foreground_idle"]),
                policy_bound_s=ASYNC_V21_BOUNDARY_RENDEZVOUS_S)
            _stage_telemetry(
                bulk, identity, generation, "native_node_apply_swap",
                apply_release_started, ASYNC_V21_ALL_EIGHT_APPLY_S,
                ended=max(
                    finish for _start, finish in trainer_apply_intervals),
                policy_id=args._async_v21_policy.policy_id,
                allocation_fence=_fence_epoch(args),
                base_global_version=generation,
                commit_global_version=generation + 1,
                contribution_digest=final_result.result_root.hex(),
                phase_scope="node_all_eight",
                foreground_interruption="verified_result_apply",
                foreground_blocking=True,
                atomic_live_model_swap=True,
                trainer_count=len(trainer_apply_intervals),
                transaction_digest=(
                    boundary_transaction.transaction_digest),
                candidate_preparation=(
                    safe_boundary_metrics["candidate_preparation"]),
                boundary_rendezvous=(
                    safe_boundary_metrics["boundary_rendezvous"]),
                release_to_apply=(
                    safe_boundary_metrics["release_to_apply"]),
                total_foreground_idle=(
                    safe_boundary_metrics["total_foreground_idle"]),
                policy_bound_s=ASYNC_V21_ALL_EIGHT_APPLY_S)
            final_result.close(); final_operation.close(); freeze.close()
            heartbeat(bulk, identity, generation=generation + 1,
                      step=(generation + 1) * args.local_steps, loss=None,
                      stage="published_node_applied",
                      authoritative_generation=generation + 1,
                      commit_receipt_digest=(
                          "" if commit_receipt is None
                          else commit_receipt.receipt_digest),
                      node_apply_receipt_digest=apply_receipt_digest)
            has_next_generation = generation + 1 < target_generation
            if pool_client is not None and has_next_generation:
                next_generation = generation + 1
                if _wait_native_ready_delay(
                        control, args, node=node, generation=next_generation,
                        incarnation=incarnation, term_requested=term_requested):
                    pool_client.ready(session.owner_endpoint, next_generation,
                                      run_id=args.run_id,
                                      fence=_fence_epoch(args),
                                      apply_receipt_digest=apply_receipt_digest)
            elif not has_next_generation:
                terminal_published = True
    except BaseException:
        try:
            session.abort(deadline_s=1)
        except Exception:
            pass
        raise
    finally:
        liveness_stop.set(); liveness_thread.join(10)
        if pool_client is not None and not terminal_published:
            try:
                pool_client.drain(session.owner_endpoint.worker_id,
                                  session.owner_endpoint.incarnation)
            except Exception:
                pass
        session.close("allocation_term_handoff" if term_requested["value"] else "normal")
        if control_server is not None:
            control_server.shutdown(); control_server.server_close()
        if control_thread is not None:
            control_thread.join(2)
    return 0


def manager(args) -> int:
    if _IMPORT_HEARTBEAT is not None:
        stop, thread = _IMPORT_HEARTBEAT
        stop.set(); thread.join(10)
    backend, _, _ = _dataplane_policy(args)
    return (_python_debug_manager(args) if backend == PYTHON_TCP_DEBUG
            else _native_manager(args))


def _python_debug_manager(args) -> int:
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    identity = f"node-{node}-manager"
    bulk = Path(args.bulk_root) / args.run_id / f"node-{node}"
    bulk = assert_node_local_path(bulk, run)
    fenced = _peer_authority(args)
    spool = LocalTrainerSpool(bulk / "mailbox", args.max_spool_bytes)
    loop = SplitManagerLoop(spool, quorum=args.local_quorum, source_id=args.source_id,
                            aggregation_deadline_s=min(args.deadline_s, 420.0))
    control_server = control_thread = owner_server = owner_thread = None
    pool_client = endpoint = None
    peer_incarnation = uuid.uuid4().hex
    pool_config = _pool_config(args)
    if args.node_count > 1:
        hosts = _pool_hosts(args)
        owner_server = DistributedOwnerServer(
            ("0.0.0.0", args.coordinator_port + 1 + node), f"node-{node}",
            max_owner_bytes=args.max_spool_bytes)
        owner_thread = threading.Thread(target=owner_server.serve_forever, daemon=True)
        owner_thread.start()
        endpoint = OwnerEndpoint(f"node-{node}", peer_incarnation, hosts[node],
                                 args.coordinator_port + 1 + node)
        if node == 0:
            control_server = PoolControlServer(
                ("0.0.0.0", args.coordinator_port), pool_config,
                evidence_root=run / "retained-evidence" / "pool-control")
            control_thread = threading.Thread(target=control_server.serve_forever, daemon=True)
            control_thread.start()
        pool_client = PoolControlClient(
            (args.coordinator_host, args.coordinator_port),
            timeout_s=min(args.deadline_s, pool_config.slo.sync_s)).bind(
                args.run_id, _fence_epoch(args))
    control = bulk / "control"
    target_generation = args.initial_generation + args.generations
    start_generation = _latest_role_generation(control, identity, args)
    recovery_path = control / "recovery" / f"{identity}.json"
    recovery_value = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    accepted_token_clock = (int(recovery_value.get("accepted_tokens", 0))
                            if int(recovery_value.get("coordinator_epoch", -1))
                            == _fence_epoch(args) else 0)
    last_commit_deadline = None
    if pool_client is not None:
        ready_started = time.monotonic()
        pool_client.ready(endpoint, start_generation, run_id=args.run_id,
                          fence=_fence_epoch(args))
        _stage_telemetry(bulk, identity, start_generation, "ready", ready_started,
                         pool_config.slo.first_heartbeat_s)
    liveness_stop, liveness_thread = _liveness_heartbeat(bulk, identity)
    try:
        for generation in range(start_generation, target_generation):
            fence = _fence(args, generation)
            heartbeat(bulk, identity, generation=generation, step=generation * args.local_steps,
                      loss=None, stage="collecting")
            if args.node_count == 1:
                result = loop.generation(fence)
            else:
                generation_started = time.monotonic()
                snapshot = pool_client.open_generation(
                    generation, 0,
                    deadline=time.monotonic() + min(args.deadline_s, pool_config.slo.sync_s))
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps, loss=None,
                          stage="training_wait")
                members, local_weight, local_shards = loop.manager.collect(
                    fence, deadline=time.monotonic() + min(
                        args.deadline_s, pool_config.slo.training_hard_s),
                    expected_source_id=args.source_id)
                _stage_telemetry(bulk, identity, generation, "k40_and_local_reduce",
                                 generation_started, pool_config.slo.training_hard_s,
                                 local_members=len(members), local_tokens=local_weight)
                last_commit_deadline = time.monotonic() + min(args.deadline_s, 180.0)
                layout = TensorLayout.from_flat_stream(
                    sum(shard.numel() for shard in local_shards),
                    max_chunk_bytes=args.bulk_chunk_bytes)
                chunks = layout.pack_flat_shards(local_shards)
                retained_bytes = sum(chunk.nbytes for chunk in chunks)
                if retained_bytes > args.max_spool_bytes:
                    raise BufferError("sender replay retention exceeds node-local byte bound")
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps, loss=None, stage="freeze")
                freeze_started = time.monotonic()
                close = pool_client.contribute_and_freeze(
                    generation=generation, attempt=0, worker_id=f"node-{node}",
                    incarnation=peer_incarnation, contribution_seq=generation,
                    accepted_tokens=local_weight,
                    payload_digest=chunk_manifest_digest(chunks),
                    deadline=min(last_commit_deadline, time.monotonic()
                                 + min(args.deadline_s, pool_config.slo.freeze_s)))
                if close.get("status") != "commit_ready":
                    raise TimeoutError(f"global contribution floor did not commit: {close}")
                _stage_telemetry(bulk, identity, generation, "freeze", freeze_started,
                                 pool_config.slo.freeze_s,
                                 accepted_tokens=int(close["accepted_tokens"]),
                                 accepted_contributions=len(close["frozen_identities"]))
                frozen = tuple(close["frozen_identities"])
                accepted_ids = tuple(sorted(contribution_id(item) for item in frozen))
                accepted_workers = {str(item["worker_id"]) for item in frozen}
                frozen_endpoints = tuple(OwnerEndpoint(
                    str(peer["worker_id"]), str(peer["incarnation"]),
                    str(peer["host"]), int(peer["port"])) for peer in snapshot["peers"]
                    if str(peer["worker_id"]) in accepted_workers)
                endpoints = live_owner_endpoints(
                    frozen_endpoints, deadline=min(last_commit_deadline,
                        time.monotonic() + min(args.deadline_s, pool_config.slo.transport_s)))
                if not endpoints:
                    raise TimeoutError("frozen generation has no live distributed shard owners")
                local_identity = f"node-{node}:{peer_incarnation}:{generation}"
                transport_metrics = {"p2p_bytes_sent": 0, "peak_retained_bytes": 0,
                                     "released_bytes": 0}
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps, loss=None,
                          stage="owner_transport")
                owner_started = time.monotonic()
                last_error = None
                for replay in range(2):
                    endpoints = live_owner_endpoints(endpoints, deadline=min(
                        last_commit_deadline, time.monotonic() + 5.0))
                    if not endpoints:
                        raise TimeoutError("owner loss left no replay target")
                    owner_server.install(
                        layout, run_id=args.run_id, fence=_fence_epoch(args),
                        generation=generation, attempt=0, owners=endpoints,
                        accepted_ids=accepted_ids)
                    try:
                        if local_identity in accepted_ids:
                            transport_metrics = submit_owned_shards(
                                layout=layout, chunks=chunks,
                                contribution_id=local_identity, weight=local_weight,
                                endpoints=endpoints, run_id=args.run_id,
                                fence=_fence_epoch(args), generation=generation, attempt=0,
                                deadline=last_commit_deadline)
                            if replay:
                                transport_metrics["replay_bytes_sent"] = \
                                    transport_metrics["p2p_bytes_sent"]
                        committed_chunks, redistribution = fetch_owned_shards(
                            layout=layout, endpoints=endpoints, run_id=args.run_id,
                            fence=_fence_epoch(args), generation=generation, attempt=0,
                            deadline=last_commit_deadline)
                        last_error = None
                        break
                    except (OSError, RuntimeError, TimeoutError) as error:
                        last_error = error
                        if time.monotonic() >= last_commit_deadline:
                            break
                if last_error is not None:
                    raise TimeoutError(f"owner reassignment/replay failed: {last_error}")
                owner_elapsed = max(time.monotonic() - owner_started, 1e-9)
                _stage_telemetry(
                    bulk, identity, generation, "owner_transport_redistribution",
                    owner_started, pool_config.slo.transport_s + pool_config.slo.apply_s,
                    p2p_bytes=int(transport_metrics["p2p_bytes_sent"]),
                    replay_bytes=int(transport_metrics.get("replay_bytes_sent", 0)),
                    redistribution_bytes=int(redistribution["redistribution_bytes"]),
                    bytes_per_second=(int(transport_metrics["p2p_bytes_sent"])
                                      + int(redistribution["redistribution_bytes"])) / owner_elapsed,
                    high_water_bytes=max(transport_metrics["peak_retained_bytes"],
                                         owner_server.high_water_bytes),
                    released_bytes=int(transport_metrics["released_bytes"]),
                    owner_count=len(endpoints))
                global_shards = layout.unpack_flat_shards(committed_chunks)
                aggregate_started = time.monotonic()
                spool.publish_aggregate(
                    fence, members, global_shards,
                    weight=int(close["accepted_tokens"]), source_id=args.source_id,
                    accepted_peers=tuple(sorted(accepted_workers)),
                    # Owners reduce exactly in float64. Trainers ultimately
                    # apply to f32 model state, so project once here to the
                    # identical apply dtype and halve the node-local stream.
                    storage_dtype=torch.float32)
                _stage_telemetry(
                    bulk, identity, generation, "global_aggregate_publish",
                    aggregate_started, pool_config.slo.apply_s,
                    aggregate_bytes=sum(shard.numel() * 4 for shard in global_shards),
                    storage_dtype="float32")
                for trainer_id in members:
                    spool.release_trainer(fence, trainer_id)
                result = {"members": members, "weight": int(close["accepted_tokens"]),
                          "accepted_nodes": tuple(sorted(accepted_workers)),
                          "network_high_water_bytes": max(
                              transport_metrics["peak_retained_bytes"],
                              owner_server.high_water_bytes),
                          "p2p_bytes_sent": transport_metrics["p2p_bytes_sent"],
                          "replay_bytes_sent": transport_metrics.get("replay_bytes_sent", 0),
                          "redistribution_bytes": redistribution["redistribution_bytes"],
                          "owner_count": len(endpoints), "layout_digest": layout.digest,
                          "frozen_identities": frozen}
                del chunks, committed_chunks, global_shards, local_shards
            accepted_token_clock += int(result["weight"])
            atomic_json(bulk / "control" / f"node-{node}-bulk-ownership.json", {
                "backend": "bounded-node-local-filesystem", "bulk_root": str(bulk),
                "shared_run_dir_is_bulk_path": bulk.is_relative_to(run),
                "max_bytes": args.max_spool_bytes,
                "high_water_bytes": spool.high_water_bytes,
                "post_release_bytes": spool.bytes_used,
                "bytes_written": spool.bytes_written, "bytes_read": spool.bytes_read,
                "published_files": spool.files_published,
                "transport_bytes": int(result.get("network_high_water_bytes", 0))})
            atomic_json(bulk / "control" / f"node-{node}-generation-{generation:08d}.json", {
                "generation": generation, "members": list(result["members"]),
                "weight": int(result["weight"]), "source_id": args.source_id,
                "payload_id": args.payload_id,
                "accepted_tokens": accepted_token_clock,
                "accepted_peers": list(result.get("accepted_nodes", (f"node-{node}",))),
                "p2p_bytes_sent": int(result.get("p2p_bytes_sent", 0)),
                "replay_bytes_sent": int(result.get("replay_bytes_sent", 0)),
                "redistribution_bytes": int(result.get("redistribution_bytes", 0)),
                "owner_count": int(result.get("owner_count", 1)),
                "central_full_model_broker": False,
                "layout_digest": result.get("layout_digest", args.payload_id),
                "frozen_identities": result.get("frozen_identities", [])})
            _publish_role_recovery(control, identity, args, generation + 1,
                                   step=(generation + 1) * args.local_steps,
                                   membership=list(result["members"]),
                                   accepted_tokens=accepted_token_clock)
            spool.prune_aggregates(keep_generations=2)
            heartbeat(bulk, identity, generation=generation + 1,
                      step=(generation + 1) * args.local_steps, loss=None, stage="published")
            if pool_client is not None:
                pool_client.ready(endpoint, generation + 1, run_id=args.run_id,
                                  fence=_fence_epoch(args))
        if node == 0:
            proposal = bulk / "control" / "trainer-proposal.json"
            deadline = (last_commit_deadline if last_commit_deadline is not None
                        else time.monotonic() + min(args.deadline_s, 60.0))
            while time.monotonic() < deadline and not proposal.exists():
                time.sleep(.02)
            if not proposal.exists():
                raise TimeoutError("checkpoint proposal deadline expired")
            value = json.loads(proposal.read_text())
            proposal_fence = LocalFence(**value["fence"])
            commit_started = time.monotonic()
            heartbeat(bulk, identity, generation=int(value["generation"]),
                      step=int(value["step"]), loss=None, stage="checkpoint_commit")
            finalize_checkpoint(
                run, value["checkpoint"], run_id=args.run_id,
                generation=int(value["generation"]), step=int(value["step"]),
                async_chain=value["async_chain"], membership=value["membership"],
                fence=proposal_fence, source_id=args.source_id, code_id=args.code_id,
                outer_update_state=value["outer_update_state"], migration=value["migration"],
                accepted_tokens=int(value["accepted_tokens"]),
                generation_identity=value["generation_identity"],
                digests=value.get("digests", {}),
                peer_authority=None if fenced is None else fenced[0],
                allocation_claim=None if fenced is None else fenced[1])
            _stage_telemetry(bulk, identity, int(value["generation"]),
                             "fenced_atomic_commit", commit_started, 60.0,
                             accepted_tokens=int(value["accepted_tokens"]),
                             membership=len(value["membership"]))
            proposal.unlink()
    finally:
        liveness_stop.set(); liveness_thread.join(10)
        if pool_client is not None and endpoint is not None:
            try:
                pool_client.drain(endpoint.worker_id, endpoint.incarnation)
            except Exception:
                pass
        if owner_server is not None:
            owner_server.shutdown(); owner_server.server_close()
        if owner_thread is not None: owner_thread.join(2)
        if control_server is not None:
            control_server.shutdown(); control_server.server_close()
        if control_thread is not None: control_thread.join(2)
    return 0


def _load_real(args):
    from ndm.async_diloco_real import default_tiny_e97_train_args
    if not args.seed or not args.train_args_json:
        raise ValueError("real E97 trainer requires --seed and --train-args-json")
    expected_sha = os.environ.get("RESILIENT_E97_SEED_SHA256", "")
    expected_step = int(os.environ.get("RESILIENT_E97_SEED_STEP", "-1"))
    expected_size = int(os.environ.get("RESILIENT_E97_SEED_SIZE", "-1"))
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or expected_step <= 0
        or expected_size <= 0
    ):
        raise ValueError("canonical immutable E97 seed identity is incomplete")
    seed_path = Path(args.seed).resolve()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    expected_parent = Path(f"/tmp/emender-e97-seed-{job_id}").resolve()
    if not job_id or seed_path.parent != expected_parent:
        raise ValueError("real E97 seed must be materialized in the current job-local directory")
    if seed_path.stat().st_size != expected_size:
        raise ValueError("job-local E97 seed size differs from canonical identity")
    seed_digest = __import__("hashlib").sha256()
    with seed_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            seed_digest.update(block)
    seed_sha = seed_digest.hexdigest()
    if seed_sha != expected_sha:
        raise ValueError("job-local E97 seed SHA256 differs from canonical identity")
    overrides = json.loads(Path(args.train_args_json).read_text())
    overrides.update({"data": args.data, "optimizer": "schedulefree"})
    train_args = default_tiny_e97_train_args(**overrides)
    payload = torch.load(seed_path, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError("verified seed lacks model or ScheduleFree inner optimizer state")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError("verified seed payload step differs from canonical identity")
    state = {name: value.clone() for name, value in payload["model_state_dict"].items()
             if value.is_floating_point()}
    seed_meta = {"step": int(payload.get("step", -1)), "sha256": seed_sha,
                 "outer_update_state": payload.get("outer_update_state")}
    migration = outer_state_migration(
        seed_meta, policy=args.migration_policy,
        approved_config={
            "mode": "delta_sgd",
            "eta_outer": args.eta_outer,
            "step": 0,
            "accepted_tokens": int(os.environ.get(
                "RESILIENT_E97_SEED_ACCEPTED_TOKENS", "150793748480")),
        },
        approved_seed={"step": expected_step, "sha256": expected_sha})
    return train_args, state, payload["optimizer_state_dict"], int(payload.get("step", 0)), migration


def trainer(args) -> int:
    if args.local_steps != 40 and not args.control:
        raise ValueError("approved E97 runtime requires local_steps=40")
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    rank = int(os.environ.get("RESILIENT_E97_LOCAL_RANK", "0")); identity = f"node-{node}-trainer-{rank}"
    bulk = Path(args.bulk_root) / args.run_id / f"node-{node}"
    bulk = assert_node_local_path(bulk, run)
    fenced = _peer_authority(args)
    backend, _, _ = _dataplane_policy(args)
    native = backend != PYTHON_TCP_DEBUG
    native_runtime = (runtime_digests(
        build_manifest=args.native_build_manifest,
        config_path=ROOT / "configs/frontier/e97_async_256.yaml",
        provider="cxi" if backend == NATIVE_CXI else os.environ.get(
            "NDP_TEST_PROVIDER", "tcp;ofi_rxm"),
        attestation=args._dataplane_attestation) if native else None)
    # Loading and cloning the real E97 checkpoint can exceed the steady-state
    # heartbeat deadline when all eight local trainers start together. Keep
    # liveness independent from generation progress throughout bootstrap.
    _liveness_heartbeat(bulk, identity)
    if _IMPORT_HEARTBEAT is not None:
        stop, thread = _IMPORT_HEARTBEAT
        stop.set(); thread.join(10)
    spool = (LocalTrainerSpool(bulk / "mailbox", args.max_spool_bytes)
             if not native else None)
    control = bulk / "control"
    target_generation = args.initial_generation + args.generations
    if args.control:
        state, optimizer_state, step, migration = {"weight": torch.tensor([0.0])}, {}, 0, {
            "status": "control_initialized",
            "policy": "control",
            "state": {
                "mode": "delta_sgd", "eta_outer": 1.0,
                "step": 0, "accepted_tokens": 0,
            }}
        train_args = None
    else:
        if native and _cohort_restart_sequence() and not args.resume_handoff:
            raise ValueError(
                "atomic trainer reconstruction requires authoritative handoff")
        train_args, state, optimizer_state, step, migration = _load_real(args)
    async_chain = [args.seed] if args.seed else []
    accepted_token_clock = int(
        dict(migration.get("state", {})).get("accepted_tokens", 0))
    resume_generation = int(args.initial_generation)
    if args.resume_handoff:
        resume_handoff = _authoritative_trainer_resume_handoff(run, args, fenced)
        handoff = json.loads(resume_handoff.read_text())
        resume_generation = int(handoff["generation"])
        checkpoint_path = Path(handoff["checkpoint"])
        if __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest() != handoff["checkpoint_sha256"]:
            raise ValueError("resume checkpoint checksum mismatch")
        resumed = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        recorded_runtime = dict(
            (resumed.get("native_runtime_digests") or {}) if native else {})
        if (native and not _native_runtime_resume_compatible(
                recorded_runtime, native_runtime)):
            raise ValueError("resume checkpoint native runtime digest mismatch")
        if (int(resumed["generation"]) != resume_generation
                or resumed["outer_update_state"] != handoff["outer_update_state"]):
            raise ValueError("resume generation/outer state does not match handoff")
        state = {name: value.clone() for name, value in resumed["model_state_dict"].items()}
        optimizer_state, step = resumed["optimizer_state_dict"], int(resumed["step"])
        migration = {"status": "restored", "state": resumed["outer_update_state"],
                     "policy": "new-harness-handoff"}
        accepted_token_clock = int(handoff.get("accepted_tokens", 0))
        async_chain = list(handoff.get("async_chain", ())) + [str(resume_handoff)]
        if not _resume_handoff_identity_matches(
                handoff, args,
                recorded_runtime=recorded_runtime,
                native=native):
            raise ValueError("resume handoff membership/identity/fence mismatch")
    recovery_manifest = control / "recovery" / f"{identity}.json"
    recovery = (
        None if _cohort_restart_sequence()
        else (json.loads(recovery_manifest.read_text())
              if recovery_manifest.exists() else None))
    if (recovery is not None and int(recovery.get("coordinator_epoch", -1))
            == _fence_epoch(args)):
        start_generation = _latest_role_generation(control, identity, args)
        checkpoint_path = Path(recovery["checkpoint"])
        if __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest() != recovery["checkpoint_sha256"]:
            raise ValueError("role recovery checkpoint checksum mismatch")
        saved = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        if (native and dict(saved.get("native_runtime_digests", {}))
                != native_runtime):
            raise ValueError("role recovery native runtime digest mismatch")
        if (saved["identity"] != identity or saved["run_id"] != args.run_id
                or saved["payload_id"] != args.payload_id
                or int(saved["coordinator_epoch"]) != _fence_epoch(args)
                or int(saved["generation"]) != start_generation):
            raise ValueError("role recovery checkpoint fence mismatch")
        state = {name: value.clone() for name, value in saved["model_state_dict"].items()}
        optimizer_state, migration = saved["optimizer_state_dict"], saved["migration"]
        step, async_chain = int(saved["step"]), list(saved["async_chain"])
        accepted_token_clock = int(saved.get("accepted_tokens", 0))
    else:
        start_generation = resume_generation
    node_incarnation = (
        os.environ.get("RESILIENT_E97_NODE_INCARNATION") or "")
    _validate_atomic_cohort_recovery(
        control, args, node=node, node_incarnation=node_incarnation,
        generation=start_generation, admitted_statuses=("reconstructed",))
    losses = []
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("requested", True))
    completed = start_generation
    leader_checkpoint: Path | None = None
    # Exactly one mutable cumulative interval may run beside the immutable
    # native descriptor.  It is retained until the prior result is
    # reload-verified and applied at a real K boundary; this is never a FIFO
    # of per-window dense objects.
    prefetched_interval: dict[str, object] | None = None
    deferred_interval_start: dict[str, torch.Tensor] | None = None
    v2_owned_seconds_max = 0.0
    v2_mutable_high_water = 0
    v2_defer_count = 0
    persistent_worker = None
    async_training_lane = None
    next_local_window = start_generation
    trainer_incarnation = (
        os.environ.get("RESILIENT_E97_PROCESS_INCARNATION") or uuid.uuid4().hex)
    v2_policy = args._async_v21_policy
    if native:
        # Emitted by the real model-owning role after native attestation and
        # authoritative restore.  This is the sole production policy marker;
        # the older serial generation scheduler is a v1 compatibility fixture
        # and is intentionally not constructed here.
        atomic_metadata(control / f"production-pipeline-{identity}.json", {
            "schema": "emender-production-async-decoupled-v2.1",
            "stage": "async_v21_policy",
            "implementation":
                "ndm.async_diloco_real.PersistentAsyncTrainingLane",
            "policy_reference":
                "ndm.async_diloco_v2.AsyncV21WorkerLane",
            **v2_policy.manifest(),
            "owned_descriptor_capacity": 1,
            "mutable_interval_capacity": 1,
            "result_mailbox_capacity": 1,
            "result_staging_capacity": 1,
            "role_source": str(Path(__file__).resolve()),
            "code_id": args.code_id,
            "run_id": args.run_id,
            "allocation_fence": _fence_epoch(args),
            "identity": identity,
        })
    for generation in range(start_generation, target_generation):
        # This timestamp belongs to the trainer generation lifecycle, not to
        # any individual data-plane stage.  Initialize it at the sole loop
        # entry so fresh, resumed, and supervisor-restarted trainers all have
        # a valid monotonic origin before a result is delayed, rejected, or
        # admitted at the safe boundary.  Job 5037971 reached commit_ready and
        # then crashed in native_generation_pipeline telemetry because this
        # origin previously existed only in the manager loop.
        generation_started = time.monotonic()
        if stop["requested"]:
            break
        # One deadline covers the complete generation, including real local
        # training, publication, quorum aggregation, and apply.  Starting a
        # fresh timeout only after the expensive 40-step train can let a live
        # but too-slow generation run until Slurm TERM without ever failing the
        # configured generation bound.
        # Stage-specific downstream gate: K40 is bounded at 420s (known live
        # evidence is ~2.5m; measured baseline 212-215s), then exchange and
        # atomic commit receive a distinct <=180s bound. No 900s silent wait.
        generation_deadline = time.monotonic() + min(args.deadline_s, 420.0)
        native_plane = None
        owned_marker: dict[str, object] | None = None

        def admit_native_generation():
            elements = state_elements(state)
            layout = layout_identity(elements, payload_max=args.bulk_chunk_bytes)
            if rank == 0:
                atomic_metadata(control / f"native-layout-{generation:08d}.json", {
                    "schema": "emender-native-e97-layout-request-v1",
                    "run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                    "generation": generation, "rank": rank,
                    "node_incarnation": node_incarnation,
                    "total_elements": elements, "layout_digest": layout.hex(),
                    "base_digest": state_digest(state).hex(),
                    "runtime_digests": native_runtime,
                })
            admitted = NativeTrainerDataPlane.connect(
                build_manifest=args.native_build_manifest,
                socket_path=os.environ["EMENDER_NDP_SOCKET"], run_id=args.run_id,
                fence_epoch=_fence_epoch(args), generation=generation, rank=rank,
                identity=identity, incarnation=trainer_incarnation,
                worker_incarnation=node_incarnation or None,
                control_root=control, deadline=generation_deadline)
            if dict(admitted.metadata.runtime_digests) != native_runtime:
                admitted.close()
                raise ValueError("manager/trainer native runtime digest mismatch")
            admitted.allocate_delta(
                deadline_s=max(.001, generation_deadline - time.monotonic()))
            return admitted

        # A capacity-deferred interval already owns a corrected, coherent
        # boundary.  Resume its K40 before joining the next native-generation
        # admission barrier; descriptor admission then runs while the adjacent
        # mutable lane is active instead of extending foreground idle.
        defer_native_admission = bool(
            native and not args.control and deferred_interval_start is not None)
        if native and not defer_native_admission:
            native_plane = admit_native_generation()
        if args.control:
            loss = 1.0 / (step + args.local_steps + rank + 1)
            delta = {"weight": torch.full_like(state["weight"], float(rank + 1))}
            tokens = rank + 1
        else:
            from ndm.async_diloco_real import (
                PersistentAsyncTrainingLane, PersistentRealWorkerSession,
                RealAsyncWorkerSpec,
                _run_real_worker,
            )

            fence = _fence(args, generation)

            phase_log = bulk / "telemetry" / f"{identity}.jsonl"
            phase_log.parent.mkdir(parents=True, exist_ok=True)

            def make_training_phase(local_window, *, overlap_scope):
                def training_phase(phase, details):
                    record = {
                        "timestamp": time.time(),
                        "monotonic_s": time.monotonic(),
                        "identity": identity,
                        "generation": generation,
                        "local_window": local_window,
                        "overlap_scope": overlap_scope,
                        "policy_id": v2_policy.policy_id,
                        "applied_anchor_version": generation,
                        "local_model_basis": "worker_local",
                        "phase": phase,
                        **details,
                    }
                    with phase_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                        stream.flush()
                    heartbeat(
                        bulk, identity, generation=generation,
                        step=int(details.get("step", step)),
                        loss=details.get("loss"), stage=phase)

                return training_phase

            def publish_trained_delta(base_state, model, tokens):
                if native:
                    # Native v2 seals below, after an optional prefetched K
                    # interval has been rebased at this exact safe boundary.
                    return
                local_spool_started = time.monotonic()
                bytes_before = spool.bytes_written
                heartbeat(bulk, identity, generation=generation, step=step, loss=None,
                          stage="streaming_delta")
                worker_state = model.state_dict()

                def shards():
                    # Local trainer-to-manager files are already bounded by the
                    # shared byte ledger and contain one sequential data file.
                    # Use a coarse record here to avoid tens of thousands of
                    # finite-check/hash/ledger operations per E97 contribution.
                    # The distributed owner transport repacks these shards to
                    # the independent ``bulk_chunk_bytes`` network bound.
                    chunk_elements = max(1, args.local_spool_chunk_bytes // 8)
                    for name, base_tensor in sorted(base_state.items()):
                        worker_tensor = worker_state[name].detach().reshape(-1)
                        base_flat = base_tensor.detach().reshape(-1)
                        if worker_tensor.numel() != base_flat.numel():
                            raise ValueError(f"trainer state layout changed for {name}")
                        for offset in range(0, worker_tensor.numel(), chunk_elements):
                            end = min(offset + chunk_elements, worker_tensor.numel())
                            worker_chunk = worker_tensor[offset:end].to(
                                device="cpu", dtype=base_tensor.dtype)
                            yield worker_chunk.sub(base_flat[offset:end])

                spool.publish(fence, rank, shards(), weight=tokens,
                              source_id=args.source_id)
                _stage_telemetry(
                    bulk, identity, generation, "local_delta_spool",
                    local_spool_started, 180.0,
                    spool_bytes=spool.bytes_written - bytes_before,
                    local_spool_chunk_bytes=args.local_spool_chunk_bytes,
                    network_chunk_bytes=args.bulk_chunk_bytes,
                    files_published=2)
                heartbeat(bulk, identity, generation=generation, step=step, loss=None,
                          stage="delta_spooled")

            def training_progress(local_step, metrics):
                if stop["requested"]:
                    raise InterruptedError("allocation TERM requested during local training")
                if time.monotonic() >= generation_deadline:
                    raise TimeoutError(
                        f"generation {generation} deadline exceeded during local training "
                        f"at step {local_step}/{args.local_steps}")
                heartbeat(
                    bulk, identity, generation=generation,
                    step=step + local_step, loss=float(metrics["loss"]),
                    stage="training")

            retained_endpoint: dict[str, torch.Tensor] = {}

            if native:
                if persistent_worker is None:
                    persistent_worker = PersistentRealWorkerSession(
                        base_state=state,
                        train_args=train_args,
                        spec=RealAsyncWorkerSpec(
                            identity, f"node-{node}", args.device,
                            args.local_steps, rank),
                        synthetic_token_stream=False,
                        synthetic_vocab_size=256,
                        optimizer_state_dict=optimizer_state,
                        consume_optimizer_state=True,
                        bootstrap_phase_callback=make_training_phase(
                            next_local_window,
                            overlap_scope=(
                                "steady_state"
                                if generation + 1 < target_generation
                                else "terminal_drain")))
                    optimizer_state = {}
                    _stage_telemetry(
                        bulk, identity, generation,
                        "async_v21_persistent_bootstrap",
                        generation_started, 420.0,
                        policy_id=v2_policy.policy_id,
                        **persistent_worker.bootstrap_counts)
                if prefetched_interval is not None:
                    if int(prefetched_interval["generation"]) != generation:
                        raise ValueError(
                            "v2 mutable interval generation is not adjacent")
                    interval_start = prefetched_interval["start"]
                    tokens = int(prefetched_interval["tokens"])
                    loss = float(prefetched_interval["loss"])
                    interval_anchor_version = int(
                        prefetched_interval["anchor_version"])
                    interval_anchor_digest = str(
                        prefetched_interval["anchor_digest"])
                    interval_window_start = int(
                        prefetched_interval["local_window_start"])
                    interval_window_end = int(
                        prefetched_interval["local_window_end"])
                    interval_window_count = int(
                        prefetched_interval["window_count"])
                    if (interval_window_end != next_local_window
                            or interval_window_count
                            != interval_window_end - interval_window_start):
                        raise ValueError(
                            "v2 coalesced interval window identity changed")
                    prefetched_interval = None
                else:
                    interval_start_was_deferred = (
                        deferred_interval_start is not None)
                    interval_start = (
                        state if deferred_interval_start is None
                        else deferred_interval_start)
                    deferred_interval_start = None
                    interval_anchor_version = generation
                    interval_anchor_digest = state_digest(state).hex()
                    interval_window_start = next_local_window
                    window = persistent_worker.run_window(
                        interval_window_start,
                        progress_callback=training_progress,
                        phase_callback=make_training_phase(
                            interval_window_start,
                            overlap_scope=(
                                "steady_state"
                                if generation + 1 < target_generation
                                else "terminal_drain")))
                    if interval_start_was_deferred:
                        # The corrected snapshot copy and this K execute on
                        # the same ordered device stream.  Verify host
                        # readiness after the K, before the first background
                        # delta read, without charging that copy to the prior
                        # all-eight apply clock.
                        persistent_worker.wait_snapshot_ready(
                            interval_start,
                            deadline=generation_deadline,
                        )
                    interval_window_end = interval_window_start + 1
                    interval_window_count = 1
                    next_local_window = interval_window_end
                    tokens = window.tokens
                    loss = float(window.losses[-1])
                endpoint_snapshot_started = time.monotonic()
                retained_endpoint = persistent_worker.snapshot()
                endpoint_snapshot_completed = time.monotonic()
                lane_admission_started = endpoint_snapshot_completed
                lane_admission_completed = endpoint_snapshot_completed
                # The coherent endpoint is admitted to the bounded local
                # background owner before the next mutable K window starts.
                # Native publication below consumes only that immutable
                # snapshot while the foreground lane advances.
                if generation + 1 < target_generation:
                    def make_lookahead_phase(local_window):
                        def lookahead_phase(phase, details):
                            record = {
                                "timestamp": time.time(),
                                "monotonic_s": time.monotonic(),
                                "identity": identity,
                                "generation": generation,
                                "local_window": local_window,
                                "overlap_scope": "steady_state",
                                "policy_id": v2_policy.policy_id,
                                "applied_anchor_version": generation,
                                "local_model_basis": "worker_local",
                                "phase": phase,
                                **details,
                            }
                            with phase_log.open("a", encoding="utf-8") as stream:
                                stream.write(
                                    json.dumps(record, sort_keys=True) + "\n")
                                stream.flush()
                            heartbeat(
                                bulk, identity, generation=generation,
                                step=int(details.get(
                                    "step", local_window * args.local_steps)),
                                loss=details.get("loss"), stage=phase)
                        return lookahead_phase

                    def make_lookahead_progress(local_window):
                        window_deadline = time.monotonic() + 420.0

                        def lookahead_progress(local_step, metrics):
                            if stop["requested"]:
                                raise InterruptedError(
                                    "allocation TERM requested during async K40")
                            if time.monotonic() >= window_deadline:
                                raise TimeoutError(
                                    f"local window {local_window} exceeded "
                                    "420 seconds")
                            heartbeat(
                                bulk, identity, generation=generation,
                                step=(
                                    local_window * args.local_steps
                                    + local_step),
                                loss=float(metrics["loss"]), stage="training")
                        return lookahead_progress

                    def async_window_start(local_window):
                        _stage_telemetry(
                            bulk, identity, generation,
                            "async_v21_k40_start",
                            time.monotonic(), 420.0,
                            policy_id=v2_policy.policy_id,
                            local_window=local_window,
                            base_global_version=generation,
                            applied_anchor_version=generation,
                            local_model_basis="worker_local",
                            prior_contribution_generation=generation,
                            prior_stage="snapshot_owned")

                    if async_training_lane is not None:
                        raise RuntimeError(
                            "v2 training lane overlapped two descriptors")
                    async_training_lane = PersistentAsyncTrainingLane(
                        persistent_worker,
                        max_windows=v2_policy.max_speculative_windows,
                        progress_callback_factory=make_lookahead_progress,
                        phase_callback_factory=make_lookahead_phase,
                        window_start_callback=async_window_start,
                    )
                    # This is local OWNED: the immutable slot and its lifetime
                    # have transferred to the background publisher.  The next
                    # K starts only after this point.
                    async_training_lane.start(
                        local_window_start=next_local_window,
                        start_state=retained_endpoint,
                        admission_deadline=(
                            endpoint_snapshot_started
                            + ASYNC_V21_SNAPSHOT_ADMISSION_S),
                    )
                    lane_admission_completed = time.monotonic()
                    v2_owned_seconds_max = max(
                        v2_owned_seconds_max,
                        lane_admission_completed - endpoint_snapshot_started)
                    if native_plane is None:
                        native_plane = admit_native_generation()
                    # The current anchor digest is already verified and bound
                    # by the native generation metadata.  Rehashing the full
                    # host state here would re-enter the bounded foreground
                    # pause after coherent capture.
                    lookahead_anchor_digest = str(
                        native_plane.metadata.base_digest)
                if native_plane is None:
                    # Terminal drain has no next mutable K to overlap, but the
                    # deferred interval must still publish through the exact
                    # native generation admitted here.
                    native_plane = admit_native_generation()
                # Device sessions transfer the immutable slot locally when
                # the ordered pinned-memory copy is enqueued.  Completion is
                # awaited only here, after the next mutable K lane owns state,
                # and before the background publisher can read the snapshot.
                snapshot_copy_completed = (
                    persistent_worker.wait_snapshot_ready(
                        retained_endpoint, deadline=generation_deadline))
                # Persist the already-timestamped causal records only after
                # the next mutable lane owns state.  Telemetry serialization
                # and filesystem latency are background work, not part of the
                # snapshot/admission interruption they describe.
                _stage_telemetry(
                    bulk, identity, generation,
                    "async_v21_endpoint_snapshot",
                    endpoint_snapshot_started,
                    ASYNC_V21_SNAPSHOT_ADMISSION_S,
                    ended=endpoint_snapshot_completed,
                    endpoint_bytes=sum(
                        value.numel() * value.element_size()
                        for value in retained_endpoint.values()),
                    foreground_interruption="snapshot_capture",
                    phase_scope="trainer_snapshot",
                    snapshot_coherent=True,
                    snapshot_slots=persistent_worker.snapshot_slot_count,
                    snapshot_copy_completion_deferred=(
                        async_training_lane is not None),
                    snapshot_copy_completion_s=(
                        snapshot_copy_completed - endpoint_snapshot_started),
                    source_mutation_ordered_after_copy=True,
                    live_model_read_after_snapshot=False,
                    python_dense_socket_bytes=0,
                    lustre_dense_hot_path_bytes=0)
                if generation + 1 < target_generation:
                    _stage_telemetry(
                        bulk, identity, generation,
                        "async_v21_snapshot_admission",
                        lane_admission_started,
                        ASYNC_V21_SNAPSHOT_ADMISSION_S,
                        ended=lane_admission_completed,
                        policy_id=v2_policy.policy_id,
                        local_window=next_local_window,
                        foreground_interruption="snapshot_admission",
                        phase_scope="snapshot_owned",
                        owned=True,
                        ownership_scope="local_immutable_snapshot",
                        immutable_snapshot=True,
                        mutable_training_resumed=True,
                        foreground_pause_s=(
                            lane_admission_completed
                            - endpoint_snapshot_started),
                        policy_bound_s=ASYNC_V21_SNAPSHOT_ADMISSION_S)
                delta = {}
            else:
                interval_start = state
                interval_anchor_version = generation
                interval_anchor_digest = state_digest(state).hex()
                report = _run_real_worker(
                    run_id=args.run_id, generation=generation, base_state=state,
                    train_args=train_args,
                    spec=RealAsyncWorkerSpec(
                        identity, f"node-{node}", args.device,
                        args.local_steps, rank),
                    synthetic_token_stream=False, synthetic_vocab_size=256,
                    optimizer_state_dict=optimizer_state,
                    consume_optimizer_state=True,
                    progress_callback=training_progress,
                    delta_consumer=publish_trained_delta,
                    phase_callback=make_training_phase(
                        generation,
                        overlap_scope=(
                            "steady_state"
                            if generation + 1 < target_generation
                            else "terminal_drain")))
                if report.update is None:
                    raise RuntimeError(
                        report.error or "real E97 trainer produced no update")
                delta = report.update.delta
                optimizer_state = report.optimizer_state_dict or {}
                tokens, loss = report.tokens, float(report.losses[-1])

            if native:
                if set(retained_endpoint) != set(interval_start):
                    raise ValueError("v2 retained endpoint layout differs from interval")
                lag = generation - interval_anchor_version
                heartbeat(bulk, identity, generation=generation, step=step,
                          loss=None, stage="streaming_delta")
                descriptor_started = time.monotonic()
                marker = native_plane.publish_state_delta(
                    interval_start, retained_endpoint, tokens,
                    chunk_elements=max(
                        1, args.local_spool_chunk_bytes // 4),
                    deadline_s=max(
                        .001, generation_deadline - time.monotonic()),
                    contribution_identity={
                        "policy_id": v2_policy.policy_id,
                        "policy_schema": v2_policy.policy_schema,
                        "contribution_schema":
                            v2_policy.contribution_schema,
                        "policy_digest": v2_policy.digest,
                        "native_abi": v2_policy.native_abi,
                        "wire_protocol_major":
                            v2_policy.wire_protocol_major,
                        "wire_protocol_minor":
                            v2_policy.wire_protocol_minor,
                        "code_digest": hashlib.sha256(
                            args.code_id.encode()).hexdigest(),
                        "worker_id":
                            native_plane.metadata.stable_worker_id,
                        "worker_incarnation":
                            native_plane.metadata.worker_incarnation,
                        "base_global_version": interval_anchor_version,
                        "base_global_digest": interval_anchor_digest,
                        "base_lag_at_seal": lag,
                        "local_window_start": interval_window_start,
                        "local_window_end": interval_window_end,
                        "window_count": interval_window_count,
                        "contribution_sequence": generation,
                        "local_trainer_set_digest": hashlib.sha256(
                            f"node-{node}:rank-{rank}".encode()).hexdigest(),
                        "endpoint_digest": hashlib.sha256(
                            (native_plane.metadata.stable_worker_id + ":"
                             + native_plane.metadata.worker_incarnation
                             ).encode()).hexdigest(),
                        "anchor_lag": lag,
                        "result_lag": 0,
                        "speculative_window_lag": min(
                            v2_policy.max_speculative_windows,
                            interval_window_end - generation),
                    })
                owned_marker = marker
                _stage_telemetry(
                    bulk, identity, generation, "native_direct_memfd",
                    descriptor_started, 180.0, trainer_spool_bytes=0,
                    python_dense_socket_bytes=0, producer_direct=True,
                    local_owned=True, fabric_receipt_waited=False,
                    phase_class="snapshot_queue_admission",
                    immutable_snapshot=True,
                    mutable_training_already_resumed=(
                        async_training_lane is not None),
                    foreground_blocking=False,
                    foreground_component_s=0.0,
                    base_global_version=interval_anchor_version,
                    base_lag_at_seal=lag,
                    local_window_start=interval_window_start,
                    local_window_end=interval_window_end,
                    window_count=interval_window_count,
                    exact_tokens=tokens,
                    storage_dtype="float32")
                heartbeat(bulk, identity, generation=generation, step=step,
                          loss=None, stage="submitted")
        fence = _fence(args, generation)
        if args.control:
            heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                      stage="streaming_delta")
            if native:
                marker = native_plane.publish_flat_shards(
                    flatten_tensors(
                        delta, chunk_elements=max(1, args.bulk_chunk_bytes // 4)),
                    tokens=tokens,
                    deadline_s=max(.001, generation_deadline - time.monotonic()),
                    contribution_identity={
                        "policy_id": v2_policy.policy_id,
                        "policy_schema": v2_policy.policy_schema,
                        "contribution_schema":
                            v2_policy.contribution_schema,
                        "policy_digest": v2_policy.digest,
                        "native_abi": v2_policy.native_abi,
                        "wire_protocol_major":
                            v2_policy.wire_protocol_major,
                        "wire_protocol_minor":
                            v2_policy.wire_protocol_minor,
                        "code_digest": hashlib.sha256(
                            args.code_id.encode()).hexdigest(),
                        "worker_id":
                            native_plane.metadata.stable_worker_id,
                        "worker_incarnation":
                            native_plane.metadata.worker_incarnation,
                        "base_global_version": generation,
                        "base_global_digest": native_plane.metadata.base_digest,
                        "base_lag_at_seal": 0,
                        "local_window_start": generation,
                        "local_window_end": generation + 1,
                        "window_count": 1,
                        "contribution_sequence": generation,
                        "local_trainer_set_digest": hashlib.sha256(
                            f"node-{node}:rank-{rank}".encode()).hexdigest(),
                        "endpoint_digest": hashlib.sha256(
                            (native_plane.metadata.stable_worker_id + ":"
                             + native_plane.metadata.worker_incarnation
                             ).encode()).hexdigest(),
                        "anchor_lag": 0,
                        "result_lag": 0,
                        "speculative_window_lag": 1,
                    })
            else:
                spool.publish(fence, rank, flatten_tensors(
                    delta, chunk_elements=max(1, args.bulk_chunk_bytes // 8)),
                    weight=tokens, source_id=args.source_id)
        del delta
        if args.node_count > 1:
            heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                      stage="local_reduce_wait")
            _wait_for_manager_exchange_window(
                bulk, node=node, generation=generation, deadline=generation_deadline)
        heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                  stage="submitted")
        exchange_deadline = time.monotonic() + min(args.deadline_s, 180.0)
        if args.node_count > 1 and node == 0 and rank != 0:
            heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                      stage="leader_apply_wait")
            # This composite wait includes the leader's independently bounded
            # result-readiness, materialization/apply, and immutable checkpoint
            # stages.  Job 5080070 proved that charging all three to the single
            # 180-second result window can expire before a valid release exists
            # (153.614s readiness plus 42.853s materialization).  Keep their
            # inner bounds unchanged while enforcing ADR-002's enclosing
            # 420-second freeze-to-verified-result deadline.
            leader_release_deadline = (
                time.monotonic() + min(args.deadline_s, 420.0))
            _wait_for_leader_apply_release(
                bulk, generation=generation, fence=fence,
                deadline=leader_release_deadline)
            exchange_deadline = time.monotonic() + min(args.deadline_s, 180.0)
        if native:
            result_wait_started = time.monotonic()
            # Owner transport keeps its 180-second inner bound above, while
            # ADR-002 separately gives freeze-to-reload-verified latest a
            # 420-second enclosing clock.  Start that clock at result
            # readiness so harmless inter-node K/checkpoint skew cannot make
            # one node abandon an already committed global transaction.
            result_readiness_deadline = (
                result_wait_started + min(args.deadline_s, 420.0))
            native_context = native_plane.result_shards(
                deadline=result_readiness_deadline,
                chunk_elements=max(1, args.bulk_chunk_bytes // 4))
            manifest, aggregate = native_context.__enter__()
            _stage_telemetry(
                bulk, identity, generation, "async_v21_result_readiness",
                result_wait_started, min(args.deadline_s, 420.0),
                policy_id=v2_policy.policy_id,
                allocation_fence=_fence_epoch(args),
                base_global_version=int(manifest["base_global_version"]),
                commit_global_version=int(manifest["commit_global_version"]),
                commit_lag=int(manifest["commit_lag"]),
                exact_tokens=int(manifest["exact_tokens"]),
                contribution_digest=str(manifest["result_root"]),
                dense_result_owned_by_native_service=True,
                foreground_blocking=False,
                foreground_component_s=0.0,
                mutable_training_active=(
                    async_training_lane is not None),
                safe_boundary_pending=True)
        else:
            manifest, aggregate = spool.stream_aggregate(
                fence, deadline=exchange_deadline,
                expected_source_id=args.source_id)
        # Waiting for distributed ownership (and, on node 0 peers, the leader
        # checkpoint marker) has its own bounded window.  Once the complete
        # node-local aggregate is visible, every eligible trainer enters the
        # bounded background preparation stage. The node supervisor gives
        # every trainer a disjoint seven-CPU partition, so all eight read-only
        # views may materialize concurrently without the 56-thread-per-rank
        # oversubscription observed in job 5080730. There is still exactly one
        # service-owned immutable result and one per-trainer bounded
        # correction cohort. The separate all-eight release below starts the
        # finite foreground apply window only after every candidate has been
        # reload-verified.
        heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                  stage="result_preparation")
        trainer_apply_started = time.monotonic()
        if not native:
            spool.release_trainer(fence, rank)
        pending_corrections = None
        accepted_own_interval = False
        if native and not args.control:
            if owned_marker is None:
                raise RuntimeError(
                    "native async v2 result has no owned descriptor identity")
            accepted_records = manifest.get(
                "accepted_local_contributions")
            if not isinstance(accepted_records, list):
                raise ValueError(
                    "native async v2 result lacks accepted correction ledger")
            rank_records = [
                item for item in accepted_records
                if isinstance(item, dict)
                and int(item.get("rank", -1)) == rank
            ]
            if rank_records:
                if len(rank_records) != 1:
                    raise ValueError(
                        "native async v2 correction identity is duplicated")
                accepted = rank_records[0]
                expected_correction_identity = {
                    "descriptor_digest": str(
                        owned_marker.get("descriptor_digest", "")),
                    "payload_digest": str(
                        owned_marker.get("payload_digest", "")),
                    "contribution_sequence": int(
                        owned_marker.get("contribution_sequence", -1)),
                    "local_window_start": int(
                        owned_marker.get("local_window_start", -1)),
                    "local_window_end": int(
                        owned_marker.get("local_window_end", -1)),
                    "base_global_version": int(
                        owned_marker.get("base_global_version", -1)),
                }
                if any(
                        accepted.get(name) != value
                        for name, value in
                        expected_correction_identity.items()):
                    raise ValueError(
                        "native async v2 accepted correction identity conflicts "
                        "with the owned descriptor")
                accepted_own_interval = True
            state, pending_corrections = (
                apply_delta_with_correction_ledger(
                    state, aggregate, eta_outer=args.eta_outer,
                    interval_start=interval_start,
                    interval_endpoint=retained_endpoint,
                    accepted_own_interval=accepted_own_interval,
                    in_place=True,
                ))
        else:
            state = apply_delta(
                state, aggregate, eta_outer=args.eta_outer, in_place=True)
        if native:
            # Materialize the candidate global result and release the bounded
            # native read view.  It is not yet eligible for the worker:
            # ScheduleFree x/z translation occurs only after immutable
            # checkpoint reload verification and fenced latest CAS below.
            native_context.__exit__(None, None, None)
            _stage_telemetry(
                bulk, identity, generation, "native_result_materialize",
                trainer_apply_started, PoolStageSLO.production().apply_s,
                lane_rank=rank, result_bytes=int(manifest["result_bytes"]),
                python_dense_socket_bytes=0,
                policy_id=v2_policy.policy_id,
                allocation_fence=_fence_epoch(args),
                base_global_version=int(manifest["base_global_version"]),
                commit_global_version=int(manifest["commit_global_version"]),
                commit_lag=int(manifest["commit_lag"]),
                anchor_lag_before_apply=1 if prefetched_interval is not None else 0,
                result_version_lag_at_apply=0,
                safe_k_boundary=True,
                accepted_own_interval=accepted_own_interval,
                accepted_descriptor_digest=(
                    str(owned_marker["descriptor_digest"])
                    if accepted_own_interval else None),
                exact_tokens=int(manifest["exact_tokens"]),
                contribution_digest=str(manifest["result_root"]))
            # Release this independently checked read-only native view
            # immediately.  The result completion path continues with durable
            # verification while the persistent model-owning lane is still
            # executing K windows.
            atomic_metadata(
                control / f"native-result-applied-{generation:08d}-{rank:02d}.json", {
                    "schema": "emender-native-e97-result-applied-lane-v1",
                    "run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                    "generation": generation, "result_root": manifest["result_root"],
                    "rank": rank,
                    "node_incarnation":
                        native_plane.metadata.worker_incarnation,
                })
        accepted_token_clock += int(
            manifest["exact_tokens"] if native else manifest["weight"])
        current_outer = dict(migration.get("state", {}))
        if (
            current_outer.get("mode") != "delta_sgd"
            or float(current_outer.get("eta_outer", -1)) != 1.0
            or int(current_outer.get("accepted_tokens", -1))
            > accepted_token_clock
        ):
            raise ValueError("v2.1 outer state is unavailable or incompatible")
        migration = {
            **migration,
            "state": {
                "mode": "delta_sgd",
                "eta_outer": 1.0,
                "step": int(current_outer.get("step", 0)) + 1,
                "accepted_tokens": accepted_token_clock,
            },
        }
        step += args.local_steps; losses.append(loss)
        completed = generation + 1
        heartbeat(bulk, identity, generation=completed, step=step, loss=loss,
                  stage="checkpoint_commit" if node == 0 and rank == 0
                  else "redistribution")
        # Authoritative v2 recovery is global S/O only.  The active local
        # ScheduleFree state may already contain one disposable lookahead
        # interval, so serializing it beside S_(g+1) would create a false
        # model/inner-state pairing.  A restarted incarnation constructs fresh
        # inner state from the verified latest global checkpoint.
        checkpoint_optimizer_state = {} if native else optimizer_state
        checkpoint_publication_started = time.monotonic()

        # The authoritative leader proposal is on the first-commit critical
        # path. Its exclusive streamed apply must flow directly into one
        # complete checkpoint and proposal before same-node peer reads begin.
        # The same immutable file is also valid leader recovery state, so do
        # not serialize the model twice.
        checkpoint_write_started = time.monotonic()
        if node == 0 and rank == 0 and (native or completed == target_generation):
            if fenced is not None:
                fenced[0].assert_current(fenced[1])
            checkpoint_name = f"generation-{completed:08d}"
            if fenced is not None:
                checkpoint_name += f"-fence-{fenced[1].fence:08d}"
            leader_checkpoint = run / "checkpoints" / f"{checkpoint_name}.pt"
            leader_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = leader_checkpoint.with_suffix(".tmp")
            torch.save({"identity": identity, "model_state_dict": state,
                        "optimizer_state_dict": checkpoint_optimizer_state,
                        "inner_optimizer_restart": "fresh_from_global",
                        "outer_update_state": migration.get("state", {}),
                        "migration": migration, "step": step,
                        "generation": completed, "run_id": args.run_id,
                        "source_id": args.source_id, "payload_id": args.payload_id,
                        "coordinator_epoch": _fence_epoch(args),
                        "membership": manifest["members"], "fence": fence.__dict__,
                        "accepted_peers": manifest.get("accepted_peers", []),
                        "accepted_tokens": accepted_token_clock,
                        "native_runtime_digests": native_runtime,
                        "async_chain": async_chain, "loss": losses[-1]}, temporary)
            os.replace(temporary, leader_checkpoint)
            if fenced is not None:
                fenced[0].assert_current(fenced[1])
            proposal_path = (bulk / "control" /
                             (f"trainer-proposal-{generation:08d}.json"
                              if native else "trainer-proposal.json"))
            native_result = ({
                "attempt": int(manifest["attempt"]),
                "layout_digest": str(manifest["layout_digest"]),
                "base_digest": str(manifest["base_digest"]),
                "result_root": str(manifest["result_root"]),
                "global_weight": int(manifest["global_weight"]),
                "result_bytes": int(manifest["result_bytes"]),
            } if native else None)
            proposal_value = {
                "checkpoint": str(leader_checkpoint.resolve()),
                "generation": completed, "step": step,
                "async_chain": async_chain,
                "membership": manifest.get("accepted_peers") or manifest["members"],
                "accepted_tokens": accepted_token_clock,
                "generation_identity": {**fence.__dict__,
                                        "result_generation": completed},
                "digests": {"source_id": args.source_id,
                            "payload_id": args.payload_id,
                            "code_id": args.code_id,
                            **({"native_runtime": native_runtime,
                                "native_result": native_result} if native else {})},
                "fence": fence.__dict__,
                "outer_update_state": migration.get("state", {}),
                "migration": migration}
            if native:
                atomic_json(proposal_path, proposal_value)
            else:
                atomic_json(bulk / "control" / "trainer-proposal.json",
                            proposal_value)

        if node == 0 and rank == 0:
            atomic_json(
                bulk / "control" / f"leader-apply-release-{generation:08d}.json",
                {"generation": generation, "result_generation": completed,
                 "fence": fence.__dict__})

        terminal_native_follower = (
            native and leader_checkpoint is None and completed == target_generation)
        if terminal_native_follower:
            recovery_checkpoint = _terminal_native_checkpoint(
                run, args, completed=completed,
                deadline=time.monotonic() + min(args.deadline_s, 60.0))
        elif leader_checkpoint is None:
            recovery_checkpoint = (bulk / "recovery" / identity /
                                   f"generation-{completed:08d}.pt")
            recovery_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = recovery_checkpoint.with_suffix(".tmp")
            torch.save({"identity": identity, "model_state_dict": state,
                        "optimizer_state_dict": checkpoint_optimizer_state,
                        "inner_optimizer_restart": "fresh_from_global",
                        "migration": migration,
                        "outer_update_state": migration.get("state", {}), "step": step,
                        "generation": completed, "run_id": args.run_id,
                        "source_id": args.source_id, "payload_id": args.payload_id,
                        "coordinator_epoch": _fence_epoch(args),
                        "membership": manifest["members"], "fence": fence.__dict__,
                        "accepted_peers": manifest.get("accepted_peers", []),
                        "accepted_tokens": accepted_token_clock,
                        "native_runtime_digests": native_runtime,
                        "async_chain": async_chain}, temporary)
            os.replace(temporary, recovery_checkpoint)
        else:
            recovery_checkpoint = leader_checkpoint
        if not terminal_native_follower:
            _stage_telemetry(
                bulk, identity, generation,
                "async_v21_checkpoint_write",
                checkpoint_write_started, 420.0,
                checkpoint_bytes=recovery_checkpoint.stat().st_size,
                foreground_blocking=False,
                foreground_component_s=0.0,
                immutable_snapshot=True,
                mutable_training_active=(
                    async_training_lane is not None))
        checkpoint_hash_started = time.monotonic()
        recovery_checkpoint_sha256 = hashlib.sha256(
            recovery_checkpoint.read_bytes()).hexdigest()
        _stage_telemetry(
            bulk, identity, generation,
            "async_v21_checkpoint_hash",
            checkpoint_hash_started, 420.0,
            checkpoint_bytes=recovery_checkpoint.stat().st_size,
            checkpoint_sha256=recovery_checkpoint_sha256,
            foreground_blocking=False,
            foreground_component_s=0.0,
            mutable_training_active=(
                async_training_lane is not None))
        if not terminal_native_follower:
            _publish_role_recovery(
                control, identity, args, completed, step=step,
                checkpoint=str(recovery_checkpoint),
                checkpoint_sha256=recovery_checkpoint_sha256,
                membership=manifest["members"], fence=fence.__dict__,
                accepted_tokens=accepted_token_clock,
                **({"native_runtime_digests": native_runtime} if native else {}))
        if native:
            # A native result becomes an applied anchor only after the current
            # fenced publisher has reload-verified the immutable handoff and
            # advanced authoritative latest.  This is candidate preparation,
            # not permission to touch the live x/z state.  The independent
            # model lane continues old-anchor K windows until the manager opens
            # the separately fenced boundary rendezvous below.
            _latest, published, manifest_digest = (
                _reload_verified_async_v2_latest(
                    run, args, fenced, generation=completed,
                    deadline=(
                        time.monotonic()
                        + min(args.deadline_s, 180.0)),
                ))
            semantic_result = {
                "policy_id": v2_policy.policy_id,
                "allocation_fence": _fence_epoch(args),
                "base_global_version": int(manifest["base_global_version"]),
                "commit_global_version": int(
                    manifest["commit_global_version"]),
                "commit_lag": int(manifest["commit_lag"]),
                "exact_tokens": int(manifest["exact_tokens"]),
                "contribution_digest": str(manifest["result_root"]),
            }
            # Close the independently bounded background checkpoint clock as
            # soon as publication and reload/CAS verification finish.  The
            # later K-boundary rendezvous and released all-eight apply have
            # their own 420s/60s clocks and must never be charged here.
            _stage_telemetry(
                bulk, identity, generation, "checkpoint_publication",
                checkpoint_publication_started, 420.0,
                checkpoint_bytes=int(published["checkpoint_bytes"]),
                phase_class="checkpoint_publish_reload",
                foreground_blocking=False,
                foreground_component_s=0.0,
                mutable_training_active=(
                    async_training_lane is not None),
                reload_verified=True, latest_cas_verified=True,
                **semantic_result)
            if node == 0 and rank == 0:
                correctness = {
                    "timestamp": time.time(),
                    "identity": identity,
                    "generation": generation,
                    "stage": "async_v21_correctness",
                    "policy_id": v2_policy.policy_id,
                    "allocation_fence": _fence_epoch(args),
                    "freeze_to_latest_s": (
                        time.monotonic() - checkpoint_publication_started),
                    "checkpoint_bytes": int(published["checkpoint_bytes"]),
                    "manifest_digest": manifest_digest,
                    "reload_verified": True,
                    "latest_cas_verified": True,
                }
                with (bulk / "telemetry" /
                      f"{identity}-pool.jsonl").open(
                          "a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(correctness, sort_keys=True) + "\n")
                    stream.flush()
            if not args.control and pending_corrections is None:
                raise RuntimeError(
                    "verified async v2 result lacks correction ledger")
            boundary_transaction = _native_safe_boundary_transaction(
                args,
                node=node,
                generation=generation,
                result_root=str(manifest["result_root"]),
                node_incarnation=native_plane.metadata.worker_incarnation,
            )
            candidate_prepared_monotonic_s = time.monotonic()
            atomic_metadata(
                control
                / (
                    f"native-candidate-prepared-{generation:08d}-"
                    f"{rank:02d}.json"
                ), {
                    "schema":
                        "emender-native-e97-candidate-prepared-v2.1",
                    "run_id": args.run_id,
                    "fence_epoch": _fence_epoch(args),
                    "generation": generation,
                    "result_root": manifest["result_root"],
                    "rank": rank,
                    "node_incarnation":
                        native_plane.metadata.worker_incarnation,
                    "trainer_incarnation": trainer_incarnation,
                    "transaction_digest":
                        boundary_transaction.transaction_digest,
                    "candidate_digest": recovery_checkpoint_sha256,
                    "recovery_digest": recovery_checkpoint_sha256,
                    "preparation_started_monotonic_s":
                        trainer_apply_started,
                    "candidate_prepared_monotonic_s":
                        candidate_prepared_monotonic_s,
                    "reload_verified": True,
                    "latest_cas_verified": True,
                })
            _stage_telemetry(
                bulk, identity, generation, "native_candidate_preparation",
                trainer_apply_started, ASYNC_V21_BOUNDARY_RENDEZVOUS_S,
                ended=candidate_prepared_monotonic_s,
                lane_rank=rank,
                transaction_digest=boundary_transaction.transaction_digest,
                candidate_digest=recovery_checkpoint_sha256,
                result_bytes=int(manifest["result_bytes"]),
                reload_verified=True,
                latest_cas_verified=True,
                foreground_blocking=False,
                foreground_component_s=0.0,
                mutable_training_active=(async_training_lane is not None),
                python_dense_socket_bytes=0,
                lustre_dense_hot_path_bytes=0,
                phase_scope="trainer_candidate_preparation")

            # The manager opens this rendezvous only after all eight immutable
            # candidates exist.  Reaching it does not release apply: each
            # trainer must first finish its current K window and publish the
            # exact transaction/fence-bound boundary receipt.
            rendezvous = _wait_for_native_boundary_control(
                control,
                args,
                generation=generation,
                transaction_digest=boundary_transaction.transaction_digest,
                marker_name="native-boundary-rendezvous",
                deadline=(
                    time.monotonic()
                    + ASYNC_V21_BOUNDARY_RENDEZVOUS_S),
            )
            boundary_deadline = float(
                rendezvous["boundary_deadline_monotonic_s"])
            if (
                boundary_deadline
                != float(rendezvous["opened_monotonic_s"])
                + ASYNC_V21_BOUNDARY_RENDEZVOUS_S
            ):
                raise ValueError(
                    "safe-boundary rendezvous changed the reviewed deadline")
            heartbeat(bulk, identity, generation=generation, step=step,
                      loss=loss, stage="boundary_rendezvous")
            mutable_report = None
            boundary_report = None
            if async_training_lane is not None:
                # The interval basis is a detached immutable CPU snapshot.
                # Rebase it while the mutable GPU lane finishes its current K
                # window; boundary-ready is still published only after that
                # lane drains.  This keeps dense host preparation inside the
                # 420-second rendezvous and reserves the released 60-second
                # clock for the atomic resident x/z translation.
                async_training_lane.prepare_at_boundary(
                    pending_corrections,
                    deadline=boundary_deadline,
                )
                boundary_report = async_training_lane.finish_at_boundary(
                    deadline=boundary_deadline,
                    corrections=None,
                )
                boundary_window = boundary_report.local_window_end
            else:
                # Control lanes and terminal followers are already stopped at
                # the completed K boundary.  They still publish a distinct
                # receipt and participate in the same all-eight fence.
                boundary_window = (
                    generation + 1 if args.control else interval_window_end)
            boundary_ready_monotonic_s = time.monotonic()
            if boundary_ready_monotonic_s > boundary_deadline:
                raise TimeoutError(
                    "trainer missed the bounded safe-boundary rendezvous")
            atomic_metadata(
                control
                / (
                    f"native-boundary-ready-{generation:08d}-"
                    f"{rank:02d}.json"
                ), {
                    "schema": "emender-native-e97-boundary-ready-v2.1",
                    "run_id": args.run_id,
                    "fence_epoch": _fence_epoch(args),
                    "generation": generation,
                    "result_root": manifest["result_root"],
                    "rank": rank,
                    "node_incarnation":
                        native_plane.metadata.worker_incarnation,
                    "trainer_incarnation": trainer_incarnation,
                    "transaction_digest":
                        boundary_transaction.transaction_digest,
                    "candidate_digest": recovery_checkpoint_sha256,
                    "local_window": boundary_window,
                    "boundary_ready_monotonic_s":
                        boundary_ready_monotonic_s,
                })
            try:
                release = _wait_for_native_boundary_control(
                    control,
                    args,
                    generation=generation,
                    transaction_digest=(
                        boundary_transaction.transaction_digest),
                    marker_name="native-apply-release",
                    deadline=boundary_deadline,
                )
            except BaseException:
                rendezvous_failed_monotonic_s = time.monotonic()
                _stage_telemetry(
                    bulk, identity, generation,
                    "native_boundary_rendezvous",
                    boundary_ready_monotonic_s,
                    ASYNC_V21_BOUNDARY_RENDEZVOUS_S,
                    ended=rendezvous_failed_monotonic_s,
                    lane_rank=rank,
                    transaction_digest=(
                        boundary_transaction.transaction_digest),
                    candidate_digest=recovery_checkpoint_sha256,
                    local_window=boundary_window,
                    foreground_interruption="safe_boundary_rendezvous",
                    foreground_blocking=True,
                    foreground_component_s=(
                        rendezvous_failed_monotonic_s
                        - boundary_ready_monotonic_s),
                    phase_scope="trainer_lane",
                    policy_bound_s=(
                        ASYNC_V21_BOUNDARY_RENDEZVOUS_S),
                    candidate_prepared_monotonic_s=(
                        candidate_prepared_monotonic_s),
                    boundary_ready_monotonic_s=(
                        boundary_ready_monotonic_s),
                    rendezvous_aborted=True)
                raise
            release_monotonic_s = float(release["released_monotonic_s"])
            apply_deadline = float(
                release["apply_deadline_monotonic_s"])
            if (
                release_monotonic_s < boundary_ready_monotonic_s
                or apply_deadline
                != release_monotonic_s + ASYNC_V21_ALL_EIGHT_APPLY_S
                or time.monotonic() > apply_deadline
            ):
                raise ValueError(
                    "all-eight release has an invalid absolute apply clock")
            _stage_telemetry(
                bulk, identity, generation, "native_boundary_rendezvous",
                boundary_ready_monotonic_s,
                ASYNC_V21_BOUNDARY_RENDEZVOUS_S,
                ended=release_monotonic_s,
                lane_rank=rank,
                transaction_digest=boundary_transaction.transaction_digest,
                candidate_digest=recovery_checkpoint_sha256,
                local_window=boundary_window,
                foreground_interruption="safe_boundary_rendezvous",
                foreground_blocking=True,
                foreground_component_s=(
                    release_monotonic_s
                    - boundary_ready_monotonic_s),
                phase_scope="trainer_lane",
                policy_bound_s=ASYNC_V21_BOUNDARY_RENDEZVOUS_S,
                candidate_prepared_monotonic_s=(
                    candidate_prepared_monotonic_s),
                boundary_ready_monotonic_s=(
                    boundary_ready_monotonic_s),
                release_monotonic_s=release_monotonic_s)
            heartbeat(bulk, identity, generation=generation, step=step,
                      loss=loss, stage="peer_apply")
            safe_apply_started = time.monotonic()
            if not args.control:
                if async_training_lane is not None:
                    mutable_report = async_training_lane.apply_at_boundary(
                        pending_corrections)
                    if mutable_report.snapshot_deferred:
                        # Capacity exhaustion defers the speculative snapshot,
                        # never the trainer.  The corrected live boundary is
                        # the start of the next admissible local interval.
                        deferred_interval_start = dict(
                            async_training_lane.start_state)
                        prefetched_interval = None
                        v2_defer_count += 1
                    else:
                        prefetched_interval = {
                            "generation": generation + 1,
                            "start": dict(async_training_lane.start_state),
                            "tokens": mutable_report.exact_tokens,
                            "loss": float(mutable_report.losses[-1]),
                            # Rebase translates the interval start and endpoint
                            # but never relabels the original global anchor.
                            "anchor_version": generation,
                            "anchor_digest": lookahead_anchor_digest,
                            "local_window_start":
                                mutable_report.local_window_start,
                            "local_window_end": mutable_report.local_window_end,
                            "window_count": mutable_report.window_count,
                        }
                    next_local_window = mutable_report.local_window_end
                    v2_mutable_high_water = max(
                        v2_mutable_high_water,
                        mutable_report.window_count)
                    async_training_lane = None
                else:
                    # Terminal generation: there is no speculative interval,
                    # but applying the verified result still occurs at the
                    # completed K boundary and preserves audited inner state.
                    persistent_worker.translate(pending_corrections)
            apply_finished_monotonic_s = time.monotonic()
            if apply_finished_monotonic_s > apply_deadline:
                raise TimeoutError(
                    "trainer atomic x/z apply exceeded the released 60s clock")
            _stage_telemetry(
                bulk, identity, generation, "native_trainer_apply",
                safe_apply_started, ASYNC_V21_ALL_EIGHT_APPLY_S,
                ended=apply_finished_monotonic_s,
                lane_rank=rank, result_bytes=int(manifest["result_bytes"]),
                python_dense_socket_bytes=0,
                foreground_interruption="verified_result_apply",
                foreground_blocking=True,
                atomic_live_model_swap=True,
                phase_scope="trainer_lane",
                policy_bound_s=ASYNC_V21_ALL_EIGHT_APPLY_S,
                anchor_lag_before_apply=completed - generation,
                result_version_lag_at_apply=0,
                speculative_window_lag=(
                    0 if mutable_report is None
                    else mutable_report.window_count),
                local_window_start=(
                    generation + 1 if args.control
                    else interval_window_end if mutable_report is None
                    else mutable_report.local_window_start),
                local_window_end=(
                    generation + 1 if args.control
                    else interval_window_end if mutable_report is None
                    else mutable_report.local_window_end),
                safe_k_boundary=True,
                transaction_digest=(
                    boundary_transaction.transaction_digest),
                candidate_digest=recovery_checkpoint_sha256,
                boundary_ready_monotonic_s=(
                    boundary_ready_monotonic_s),
                release_monotonic_s=release_monotonic_s,
                result_reload_verified=True,
                latest_cas_verified=True,
                **semantic_result)
            control_integrity_started = time.monotonic()
            _stage_telemetry(
                bulk, identity, generation, "control_handoff_integrity",
                control_integrity_started, 420.0,
                manifest_digest=manifest_digest,
                reload_verified=True, latest_cas_verified=True,
                **semantic_result)
            application_window = (
                generation + 1 if args.control
                else interval_window_end if mutable_report is None
                else mutable_report.local_window_end)
            apply_receipt = {
                "timestamp": time.time(),
                "identity": identity,
                "generation": generation,
                "local_window": application_window,
                "stage": "safe_boundary_apply",
                "policy_id": v2_policy.policy_id,
                "allocation_fence": _fence_epoch(args),
                "known_global_version": completed,
                "result_version": completed,
                "applied_anchor_version": generation,
                "anchor_lag_before_apply": 1,
                "result_version_lag_at_apply": 0,
                "speculative_window_lag": (
                    0 if mutable_report is None
                    else mutable_report.window_count),
                "result_digest": str(manifest["result_root"]),
                "manifest_digest": manifest_digest,
                "transaction_digest":
                    boundary_transaction.transaction_digest,
                "candidate_digest": recovery_checkpoint_sha256,
                "boundary_ready_monotonic_s":
                    boundary_ready_monotonic_s,
                "release_monotonic_s": release_monotonic_s,
                "reload_verified": True,
                "latest_cas_verified": True,
                "accepted_contribution_digest": str(
                    owned_marker["descriptor_digest"])
                    if accepted_own_interval else None,
            }
            with (bulk / "telemetry" /
                  f"{identity}-pool.jsonl").open(
                      "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(apply_receipt, sort_keys=True) + "\n")
                stream.flush()
            atomic_metadata(
                control / f"native-applied-{generation:08d}-{rank:02d}.json", {
                    "schema": "emender-native-e97-applied-v2.1",
                    "run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                    "generation": generation, "result_root": manifest["result_root"],
                    "rank": rank, "checkpoint": str(recovery_checkpoint),
                    "accepted_tokens": accepted_token_clock,
                    "node_incarnation":
                        native_plane.metadata.worker_incarnation,
                    "trainer_incarnation": trainer_incarnation,
                    "transaction_digest":
                        boundary_transaction.transaction_digest,
                    "candidate_digest": recovery_checkpoint_sha256,
                    "recovery_digest": recovery_checkpoint_sha256,
                    "boundary_ready_monotonic_s":
                        boundary_ready_monotonic_s,
                    "release_monotonic_s": release_monotonic_s,
                    "apply_started_monotonic_s": safe_apply_started,
                    "apply_finished_monotonic_s":
                        apply_finished_monotonic_s,
                })
            native_plane.close()
        heartbeat(bulk, identity, generation=generation + 1, step=step, loss=loss, stage="applied")
    if native and not args.control:
        bounds = {
            "timestamp": time.time(),
            "identity": identity,
            "generation": completed,
            "stage": "async_v21_bounds",
            "policy_id": v2_policy.policy_id,
            "allocation_fence": _fence_epoch(args),
            "sealed_descriptor_capacity": 1,
            "mutable_interval_capacity": 1,
            "result_mailbox_capacity": 1,
            "result_staging_capacity": 1,
            "sealed_descriptor_high_water": 1,
            "mutable_interval_high_water": (
                1 if v2_mutable_high_water else 0),
            "mutable_window_high_water": v2_mutable_high_water,
            "result_mailbox_high_water": 1,
            "result_staging_high_water": 0,
            "native_owned_seconds_max": v2_owned_seconds_max,
            "resident_admission_bytes":
                ASYNC_V21_E97_NATIVE_RESIDENT_BYTES,
            "resident_limit_bytes": args.max_spool_bytes,
            "resident_headroom_bytes": (
                args.max_spool_bytes
                - ASYNC_V21_E97_NATIVE_RESIDENT_BYTES),
            "python_dense_socket_bytes": 0,
            "lustre_dense_hot_path_bytes": 0,
            "pause_count": 0,
            "defer_count": v2_defer_count,
            "drop_count": 0,
            "windows_completed": persistent_worker.windows_completed,
            **persistent_worker.bootstrap_counts,
        }
        with (bulk / "telemetry" /
              f"{identity}-pool.jsonl").open(
                  "a", encoding="utf-8") as stream:
            stream.write(json.dumps(bounds, sort_keys=True) + "\n")
            stream.flush()
    if persistent_worker is not None:
        persistent_worker.close()
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.generations <= 0 or args.deadline_s <= 0:
        raise ValueError("generations and deadlines must be positive")
    if not 0 < args.local_spool_chunk_bytes <= args.max_spool_bytes:
        raise ValueError("local spool chunk bound must be positive and within the byte ledger")
    if not 0 < args.bulk_chunk_bytes <= args.max_spool_bytes:
        raise ValueError("owner transport chunk bound must be positive and within the byte ledger")
    args._async_v21_policy = _async_v21_policy(args)
    backend, _production, _full_layout = _dataplane_policy(args)
    if (not args.control and backend != PYTHON_TCP_DEBUG
            and args.max_spool_bytes
            < ASYNC_V21_E97_NATIVE_RESIDENT_BYTES):
        raise ValueError(
            "async-v2 E97 native resident formula exceeds configured cap")
    args._dataplane_attestation = _attest_dataplane(
        args)  # hard gate before trainer calls ``_load_real``
    return manager(args) if args.role == "manager" else trainer(args)


if __name__ == "__main__":
    raise SystemExit(main())
