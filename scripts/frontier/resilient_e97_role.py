#!/usr/bin/env python3
"""Real manager/trainer entrypoints for the split resilient E97 launcher."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
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
            }, sort_keys=True))
            os.replace(temporary, state)
            stop.wait(5)

    thread = threading.Thread(target=publish, name=f"{role}-import-heartbeat", daemon=True)
    thread.start()
    return stop, thread


_IMPORT_HEARTBEAT = _role_import_heartbeat()

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
from ndm.native_pipeline import (
    CommittedResult, GenerationIdentity,
    LiveNativeGenerationScheduler, NativeGenerationPipeline,
    finite_result_verifier,
)
from ndm.resilient_e97_reducer import TensorLayout
from ndm.fenced_admission import AllocationLease, SQLiteFencedControlStore
from ndm.resilient_e97_runtime import (SplitManagerLoop, apply_delta, atomic_json,
                                       PINNED_STEP_1525000_SHA256, assert_node_local_path,
                                       finalize_checkpoint, flatten_tensors, heartbeat,
                                       outer_state_migration)
from ndm.resilient_pool_runtime import (
    DistributedOwnerServer, OwnerEndpoint, PoolControlClient, PoolControlConfig,
    PoolControlServer, PoolStageSLO, chunk_manifest_digest, contribution_id,
    fetch_owned_shards, live_owner_endpoints, submit_owned_shards,
)


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
    attestation = attest_launch(
        backend=backend, production=production, full_layout=full_layout,
        build_manifest=getattr(args, "native_build_manifest", "") or None,
        gate_json=getattr(args, "native_gate_json", "") or None,
        source_root=ROOT if backend != PYTHON_TCP_DEBUG else None,
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


def _fenced_control(args) -> tuple[SQLiteFencedControlStore, AllocationLease] | None:
    database = os.environ.get("RESILIENT_E97_FENCE_DB")
    encoded = os.environ.get("RESILIENT_E97_ALLOCATION_LEASE")
    if not database and not encoded:
        return None  # direct local protocol fixtures do not own an allocation
    if not database or not encoded:
        raise ValueError("allocation fence database and lease must be supplied together")
    lease = AllocationLease(**json.loads(encoded))
    if lease.run_id != args.run_id or lease.fence != _fence_epoch(args):
        raise ValueError("role allocation lease does not match run/fence")
    store = SQLiteFencedControlStore(database)
    store.assert_current(lease)
    return store, lease


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


def _native_manager_resume_point(
        run: Path, args,
        fenced: tuple[SQLiteFencedControlStore, AllocationLease] | None,
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
    latest_path = run / "handoff" / "latest.json"
    if not latest_path.exists():
        return initial, {"status": "cold_start", "generation": initial,
                         "fence": _fence_epoch(args)}
    latest = json.loads(latest_path.read_text())
    generation = int(latest.get("generation", -1))
    source_fence = int(latest.get("fence", -1))
    current_fence = _fence_epoch(args)
    if generation < initial or generation > target:
        raise ValueError("authoritative latest generation is outside this run bound")
    if source_fence <= 0 or source_fence > current_fence:
        raise ValueError("manager rejoin latest fence is invalid or newer than allocation")
    manifest = Path(str(latest.get("manifest", ""))).resolve()
    try:
        manifest.relative_to((run / "handoff").resolve())
    except ValueError as error:
        raise ValueError("manager rejoin manifest escapes the handoff root") from error
    encoded = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    if manifest_sha256 != latest.get("manifest_sha256"):
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
    if fenced is not None:
        fenced[0].assert_current(fenced[1])
        authoritative = fenced[0].read_publication(
            args.run_id, "latest", "authoritative")
        if (authoritative is None
                or int(authoritative.get("generation", -1)) != generation
                or int(authoritative.get("fence", -1)) != source_fence
                or authoritative.get("manifest_sha256") != manifest_sha256):
            raise ValueError("manager rejoin handoff is not authoritative")
    evidence = {
        "status": "synchronized", "generation": generation,
        "fence": current_fence, "source_fence": source_fence,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
    }
    if source_code_id != args.code_id:
        evidence["source_code_id"] = source_code_id
    return generation, evidence


def _authoritative_trainer_resume_handoff(
        run: Path, args,
        fenced: tuple[SQLiteFencedControlStore, AllocationLease] | None,
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
    store, lease = fenced
    store.assert_current(lease)
    authoritative = store.read_publication(
        args.run_id, "latest", "authoritative")
    if authoritative is None:
        raise ValueError("resume handoff has no authoritative fenced latest")
    generation = int(authoritative.get("generation", -1))
    initial = int(args.initial_generation)
    target = initial + int(args.generations)
    if generation < initial or generation > target:
        raise ValueError("authoritative trainer generation is outside this run bound")
    if generation == initial:
        return configured

    latest_path = run / "handoff" / "latest.json"
    latest = json.loads(latest_path.read_text())
    manifest = Path(str(latest.get("manifest", ""))).resolve()
    try:
        manifest.relative_to((run / "handoff").resolve())
    except ValueError as error:
        raise ValueError("trainer resume manifest escapes the handoff root") from error
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source_fence = int(latest.get("fence", -1))
    if (int(latest.get("generation", -1)) != generation
            or int(authoritative.get("fence", -1)) != source_fence
            or latest.get("manifest_sha256") != manifest_sha256
            or authoritative.get("manifest_sha256") != manifest_sha256):
        raise ValueError("trainer resume latest pointer is not authoritative")
    return manifest


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


def _wait_for_native_apply_lane(control: Path, args, *, generation: int,
                                rank: int, result_root: str,
                                deadline: float) -> dict[str, object] | None:
    """Serialize readers of the one shared node aggregate by local rank.

    The service intentionally exposes one read-only result instead of eight
    trainer-sized copies.  Full E97 showed that letting all eight trainers
    stream that 5.5 GiB mapping concurrently turns a roughly 2.5-second apply
    into an apply-deadline failure.  The preceding rank's authenticated apply
    marker is a node-local, metadata-only credit: every trainer still maps and
    applies the canonical aggregate, but only one reader per node is active.
    """
    if rank <= 0:
        return None
    return wait_metadata(
        control / f"native-result-applied-{generation:08d}-{rank - 1:02d}.json",
        deadline=deadline,
        expected={"run_id": args.run_id,
                  "fence_epoch": _fence_epoch(args),
                  "generation": generation,
                  "result_root": result_root,
                  "rank": rank - 1})


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
                     started: float, hard_s: float, **metrics: object) -> None:
    elapsed = time.monotonic() - started
    record = {"timestamp": time.time(), "identity": identity,
              "generation": generation, "stage": stage,
              "elapsed_s": elapsed, "hard_s": hard_s,
              "within_slo": elapsed <= hard_s, **metrics}
    path = bulk / "telemetry" / f"{identity}-pool.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    if elapsed > hard_s:
        raise TimeoutError(f"{stage} exceeded {hard_s}s stage SLO")


_MANAGER_EXCHANGE_STAGES = frozenset({
    "freeze", "owner_transport", "redistribution", "checkpoint_commit", "published",
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
    """Keep node-0 peers off the aggregate until its checkpoint leader applies.

    Job 5028835 left only 31 seconds in the fixed commit window after the
    global aggregate became visible. Eight simultaneous full-file readers
    prevented the designated trainer from reaching checkpoint creation. The
    generation-scoped marker contains no tensor data and opens a fresh bounded
    peer-apply window only after the leader has applied (and, on the terminal
    generation, proposed) its checkpoint.
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


def _pool_config(args) -> PoolControlConfig:
    policy = __import__("hashlib").sha256(json.dumps({
        "q_min": args.global_quorum, "t_min": args.global_token_min,
        "ready_fraction": args.ready_fraction,
    }, sort_keys=True).encode()).hexdigest()
    backend, production, full_layout = _dataplane_policy(args)
    attestation = getattr(args, "_dataplane_attestation", {})
    return PoolControlConfig(
        args.run_id, _fence_epoch(args), args.global_quorum, args.global_token_min,
        args.ready_fraction, args.source_id, policy, args.payload_id, args.code_id,
        PoolStageSLO.production(), backend, production, full_layout,
        str(attestation.get("bundle_sha256", "")) if backend != PYTHON_TCP_DEBUG else "")


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
        identity: str, generation: int, exchange_deadline: float
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
            receive_queue_high_water_frames=inbox.high_water_frames)

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
            receive_queue_high_water_frames=inbox.high_water_frames)
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
            final_operation_deadline_s + min(float(args.deadline_s), 180.0)))
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
    # are session lifetime bounds, so they must cover every requested
    # generation rather than expiring while a healthy manager advertises READY
    # for the next one.  Slurm's requested walltime remains the outer hard cap.
    return float(args.deadline_s) * max(1, int(args.generations))


def _native_manager(args) -> int:
    """Model-free controller for direct memfd admission and native ownership."""
    if args.local_quorum != 8 and not args.control:
        raise ValueError("native E97 production requires all eight local trainers")
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    identity = f"node-{node}-manager"
    bulk = assert_node_local_path(
        Path(args.bulk_root) / args.run_id / f"node-{node}", run)
    fenced = _fenced_control(args)
    backend, production, full_layout = _dataplane_policy(args)
    provider = "cxi" if backend == NATIVE_CXI else os.environ.get(
        "NDP_TEST_PROVIDER", "tcp;ofi_rxm")
    config_path = ROOT / "configs/frontier/e97_resilient_split_role_flat.json"
    digests = runtime_digests(
        build_manifest=args.native_build_manifest, config_path=config_path,
        provider=provider, attestation=args._dataplane_attestation)
    incarnation = uuid.uuid4().hex
    session = NativeManagerSession.start(
        backend=backend, run_id=args.run_id, fence_epoch=_fence_epoch(args),
        worker_id=f"node-{node}", incarnation=incarnation,
        host=_pool_hosts(args)[node], build_manifest=args.native_build_manifest,
        gate_json=args.native_gate_json or None, source_root=ROOT,
        production=production, full_layout=full_layout,
        deadline_s=_native_manager_session_lifetime_s(args),
        telemetry_path=bulk / "telemetry" / f"{identity}-native.jsonl",
        payload_max=args.bulk_chunk_bytes, resident_limit_bytes=args.max_spool_bytes)
    control = bulk / "control"
    session.write_readiness(control / "native-service-ready.json")
    start_generation, sync_evidence = _native_manager_resume_point(
        run, args, fenced, native_runtime=digests)
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
    pool_config = _pool_config(args)
    control_server = control_thread = None
    pool_client = None
    if args.node_count > 1:
        if args.node_count not in {2, 4, 8}:
            raise ValueError(
                "ordered native E97 scale runtime permits only 2, 4, or 8 nodes")
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
        if _wait_native_ready_delay(
                control, args, node=node, generation=start_generation,
                incarnation=incarnation, term_requested=term_requested):
            pool_client.ready(session.owner_endpoint, start_generation,
                              run_id=args.run_id, fence=_fence_epoch(args))
    liveness_stop, liveness_thread = _liveness_heartbeat(bulk, identity)
    terminal_published = False
    try:
        target_generation = args.initial_generation + args.generations
        for generation in range(start_generation, target_generation):
            if term_requested["value"]:
                break
            if fenced is not None:
                fenced[0].assert_current(fenced[1])
            native_deadline = time.monotonic() + min(args.deadline_s, 420.0)
            snapshot = (pool_client.open_generation(
                generation, 1, deadline=time.monotonic() + pool_config.slo.sync_s)
                if pool_client is not None else None)
            request = wait_metadata(
                control / f"native-layout-{generation:08d}.json",
                deadline=native_deadline,
                expected={"run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                          "generation": generation, "rank": 0})
            elements = int(request["total_elements"])
            expected_layout = layout_identity(
                elements, payload_max=args.bulk_chunk_bytes)
            if request.get("layout_digest") != expected_layout.hex():
                raise ValueError("trainer layout differs from native flat-layout ABI")
            base_digest = bytes.fromhex(str(request["base_digest"]))
            plan_digest = __import__("hashlib").sha256(json.dumps(
                {"generation": generation, "runtime_digests": digests},
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
                session.local.generation_deadline_ns, digests)
            atomic_metadata(control / f"native-generation-{generation:08d}.json",
                            metadata.as_json())
            heartbeat(bulk, identity, generation=generation,
                      step=generation * args.local_steps, loss=None, stage="training_wait")
            submissions = [wait_metadata(
                control / f"native-submit-{generation:08d}-{rank:02d}.json",
                deadline=native_deadline,
                expected={"run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                          "generation": generation, "rank": rank,
                          "layout_digest": expected_layout.hex()})
                for rank in range(args.local_quorum)]
            local_weight = sum(int(item["tokens"]) for item in submissions)
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
                result_dtype="float64", result_bytes=local_result.length)
            final_operation, final_result = local_operation, local_result
            if pool_client is not None:
                close = pool_client.contribute_and_freeze(
                    generation=generation, attempt=1, worker_id=f"node-{node}",
                    incarnation=incarnation, contribution_seq=generation,
                    accepted_tokens=local_weight,
                    payload_digest=local_result.result_root.hex(),
                    deadline=time.monotonic() + pool_config.slo.freeze_s)
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
                           dict(close["accepted_weights"]).items()}
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
                    exchange_deadline=owner_phase_deadline)
                validated = pool_client.validate_result_root(
                    generation=generation, attempt=1, worker_id=f"node-{node}",
                    incarnation=incarnation, result_root=final_result.result_root.hex(),
                    global_weight=final_result.global_weight,
                    result_bytes=final_result.length,
                    deadline=time.monotonic() + pool_config.slo.apply_s)
                if validated["status"] != "validated":
                    raise RuntimeError("native result root did not validate")
            result_marker = {
                "schema": "emender-native-e97-result-v1", "run_id": args.run_id,
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
                "global_weight": final_result.global_weight,
                "weight": final_result.global_weight,
                "result_bytes": final_result.length,
                "members": [int(item["rank"]) for item in submissions],
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
                    control_store=None if fenced is None else fenced[0],
                    allocation_lease=None if fenced is None else fenced[1])
            else:
                latest_wait = wait_metadata(
                    run / "handoff" / "latest.json",
                    deadline=time.monotonic() + pool_config.slo.apply_s,
                    expected={"generation": generation + 1})
                publication = Path(str(latest_wait["manifest"]))
            latest = wait_metadata(
                run / "handoff" / "latest.json",
                deadline=time.monotonic() + pool_config.slo.apply_s,
                expected={"generation": generation + 1,
                          "fence": _fence_epoch(args)})
            # Each trainer's native result apply remains bounded by APPLY.
            # Waiting for all eight independent durable recovery receipts is
            # the aggregate exchange/commit phase: live E97 checkpoint I/O can
            # legitimately outlast one reader's 60 second apply SLO, while the
            # allocation contract still caps this complete phase at 180s.
            recovery_deadline = time.monotonic() + min(args.deadline_s, 180.0)
            for rank in range(args.local_quorum):
                wait_metadata(
                    control / f"native-applied-{generation:08d}-{rank:02d}.json",
                    deadline=recovery_deadline,
                    expected={"run_id": args.run_id,
                              "fence_epoch": _fence_epoch(args),
                              "generation": generation,
                              "result_root": final_result.result_root.hex(),
                              "rank": rank})
            session.commit(
                publication_manifest=publication, authoritative_latest=latest,
                deadline_s=pool_config.slo.apply_s)
            final_result.close(); final_operation.close(); freeze.close()
            heartbeat(bulk, identity, generation=generation + 1,
                      step=(generation + 1) * args.local_steps, loss=None,
                      stage="published")
            has_next_generation = generation + 1 < target_generation
            if pool_client is not None and has_next_generation:
                next_generation = generation + 1
                if _wait_native_ready_delay(
                        control, args, node=node, generation=next_generation,
                        incarnation=incarnation, term_requested=term_requested):
                    pool_client.ready(session.owner_endpoint, next_generation,
                                      run_id=args.run_id,
                                      fence=_fence_epoch(args))
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
    fenced = _fenced_control(args)
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
            if fenced is not None:
                fenced[0].assert_current(fenced[1])
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
                control_store=None if fenced is None else fenced[0],
                allocation_lease=None if fenced is None else fenced[1])
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
    overrides = json.loads(Path(args.train_args_json).read_text())
    overrides.update({"data": args.data, "optimizer": "schedulefree"})
    train_args = default_tiny_e97_train_args(**overrides)
    seed_sha = __import__("hashlib").sha256(Path(args.seed).read_bytes()).hexdigest()
    if seed_sha != PINNED_STEP_1525000_SHA256:
        raise ValueError("seed SHA256 does not match pinned step-1525000 checkpoint")
    payload = torch.load(args.seed, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError("verified seed lacks model or ScheduleFree inner optimizer state")
    state = {name: value.clone() for name, value in payload["model_state_dict"].items()
             if value.is_floating_point()}
    seed_meta = {"step": int(payload.get("step", -1)), "sha256": seed_sha,
                 "outer_update_state": payload.get("outer_update_state")}
    migration = outer_state_migration(
        seed_meta, policy=args.migration_policy,
        approved_config={"algorithm": "weighted-mean", "eta_outer": args.eta_outer})
    return train_args, state, payload["optimizer_state_dict"], int(payload.get("step", 0)), migration


def trainer(args) -> int:
    if args.local_steps != 40 and not args.control:
        raise ValueError("approved E97 runtime requires local_steps=40")
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    rank = int(os.environ.get("RESILIENT_E97_LOCAL_RANK", "0")); identity = f"node-{node}-trainer-{rank}"
    bulk = Path(args.bulk_root) / args.run_id / f"node-{node}"
    bulk = assert_node_local_path(bulk, run)
    fenced = _fenced_control(args)
    backend, _, _ = _dataplane_policy(args)
    native = backend != PYTHON_TCP_DEBUG
    native_runtime = (runtime_digests(
        build_manifest=args.native_build_manifest,
        config_path=ROOT / "configs/frontier/e97_resilient_split_role_flat.json",
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
            "status": "control_initialized", "policy": "control"}
        train_args = None
    else:
        train_args, state, optimizer_state, step, migration = _load_real(args)
    async_chain = [args.seed] if args.seed else []
    accepted_token_clock = 0
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
        if (handoff.get("run_id") != args.run_id or handoff.get("payload_id") != args.payload_id
                or handoff.get("source_id") != args.source_id
                or (native and handoff.get("code_id")
                    != recorded_runtime.get("source_commit"))
                or int(handoff["fence"]["coordinator_epoch"]) > _fence_epoch(args)
                or not handoff.get("finalized")):
            raise ValueError("resume handoff membership/identity/fence mismatch")
        if fenced is not None:
            latest = fenced[0].read_publication(args.run_id, "latest", "authoritative")
            if (latest is None or int(latest["generation"]) != resume_generation
                    or latest["manifest_sha256"] != __import__("hashlib").sha256(
                        resume_handoff.read_bytes()).hexdigest()):
                raise ValueError("resume handoff is not the authoritative fenced latest")
    recovery_manifest = control / "recovery" / f"{identity}.json"
    recovery = json.loads(recovery_manifest.read_text()) if recovery_manifest.exists() else None
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
    losses = []
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("requested", True))
    completed = start_generation
    leader_checkpoint: Path | None = None
    trainer_incarnation = uuid.uuid4().hex
    pipeline = (NativeGenerationPipeline(
        run_id=args.run_id, fence=_fence_epoch(args),
        incarnation=trainer_incarnation) if native else None)
    scheduler_events = []
    scheduler = (LiveNativeGenerationScheduler(
        pipeline, telemetry=scheduler_events.append) if native else None)
    if native:
        # This marker is deliberately emitted by the real trainer entrypoint,
        # after native attestation and state restoration.  Renderer regressions
        # assert it so a test cannot accidentally exercise only the policy
        # abstraction in ndm.native_pipeline.
        atomic_metadata(control / f"production-pipeline-{identity}.json", {
            "schema": "emender-production-delayed-pipeline-v1",
            "implementation": (
                "ndm.native_pipeline.LiveNativeGenerationScheduler"),
            "role_source": str(Path(__file__).resolve()),
            "code_id": args.code_id,
            "run_id": args.run_id,
            "fence_epoch": _fence_epoch(args),
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
        if fenced is not None:
            fenced[0].assert_current(fenced[1])
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
        if native:
            elements = state_elements(state)
            layout = layout_identity(elements, payload_max=args.bulk_chunk_bytes)
            if rank == 0:
                atomic_metadata(control / f"native-layout-{generation:08d}.json", {
                    "schema": "emender-native-e97-layout-request-v1",
                    "run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                    "generation": generation, "rank": rank,
                    "total_elements": elements, "layout_digest": layout.hex(),
                    "base_digest": state_digest(state).hex(),
                    "runtime_digests": native_runtime,
                })
            native_plane = NativeTrainerDataPlane.connect(
                build_manifest=args.native_build_manifest,
                socket_path=os.environ["EMENDER_NDP_SOCKET"], run_id=args.run_id,
                fence_epoch=_fence_epoch(args), generation=generation, rank=rank,
                identity=identity, incarnation=trainer_incarnation,
                control_root=control, deadline=generation_deadline)
            if dict(native_plane.metadata.runtime_digests) != native_runtime:
                native_plane.close()
                raise ValueError("manager/trainer native runtime digest mismatch")
            native_plane.allocate_delta(
                deadline_s=max(.001, generation_deadline - time.monotonic()))
        if args.control:
            loss = 1.0 / (step + args.local_steps + rank + 1)
            delta = {"weight": torch.full_like(state["weight"], float(rank + 1))}
            tokens = rank + 1
        else:
            from ndm.async_diloco_real import RealAsyncWorkerSpec, _run_real_worker

            fence = _fence(args, generation)

            phase_log = bulk / "telemetry" / f"{identity}.jsonl"
            phase_log.parent.mkdir(parents=True, exist_ok=True)

            def training_phase(phase, details):
                record = {
                    "timestamp": time.time(), "monotonic_s": time.monotonic(),
                    "identity": identity, "generation": generation,
                    "phase": phase, **details,
                }
                with phase_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                if scheduler is not None and phase in {"training_start", "training_end"}:
                    scheduler.event(GenerationIdentity(
                        args.run_id, _fence_epoch(args), generation,
                        native_plane.metadata.attempt, trainer_incarnation,
                        native_plane.metadata.layout_digest,
                        native_plane.metadata.base_digest),
                        "k40_start" if phase == "training_start" else "k40_end")
                heartbeat(
                    bulk, identity, generation=generation,
                    step=int(details.get("step", step)),
                    loss=details.get("loss"), stage=phase)

            def publish_trained_delta(base_state, model, tokens):
                if native:
                    heartbeat(bulk, identity, generation=generation, step=step,
                              loss=None, stage="streaming_delta")
                    marker = native_plane.publish_model_delta(
                        base_state, model, tokens,
                        chunk_elements=max(1, args.local_spool_chunk_bytes // 4),
                        deadline_s=max(.001, generation_deadline - time.monotonic()))
                    slot = pipeline.reserve(deadline=generation_deadline)
                    token = pipeline.handoff(
                        slot, GenerationIdentity(
                            args.run_id, _fence_epoch(args), generation,
                            native_plane.metadata.attempt, trainer_incarnation,
                            native_plane.metadata.layout_digest,
                            native_plane.metadata.base_digest),
                        marker, weight=tokens, digest=marker["source_sha256"])
                    # submit() acknowledged that the persistent service retained
                    # the sealed memfd, so the producer ownership token is now
                    # safe to release while outer work continues independently.
                    pipeline.release(token)
                    _stage_telemetry(
                        bulk, identity, generation, "native_direct_memfd",
                        time.monotonic(), 180.0, trainer_spool_bytes=0,
                        python_dense_socket_bytes=0,
                        producer_direct=True, storage_dtype="float32")
                    heartbeat(bulk, identity, generation=generation, step=step,
                              loss=None, stage="submitted")
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

            report = _run_real_worker(
                run_id=args.run_id, generation=generation, base_state=state,
                train_args=train_args,
                spec=RealAsyncWorkerSpec(identity, f"node-{node}", args.device,
                                         args.local_steps, rank),
                synthetic_token_stream=False, synthetic_vocab_size=256,
                optimizer_state_dict=optimizer_state, consume_optimizer_state=True,
                progress_callback=training_progress,
                delta_consumer=publish_trained_delta,
                phase_callback=training_phase)
            if report.update is None:
                raise RuntimeError(report.error or "real E97 trainer produced no update")
            delta = report.update.delta
            optimizer_state = report.optimizer_state_dict or {}
            tokens, loss = report.tokens, float(report.losses[-1])
        fence = _fence(args, generation)
        if args.control:
            heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                      stage="streaming_delta")
            if native:
                marker = native_plane.publish_flat_shards(
                    flatten_tensors(
                        delta, chunk_elements=max(1, args.bulk_chunk_bytes // 4)),
                    tokens=tokens,
                    deadline_s=max(.001, generation_deadline - time.monotonic()))
                slot = pipeline.reserve(deadline=generation_deadline)
                token = pipeline.handoff(
                    slot, GenerationIdentity(
                        args.run_id, _fence_epoch(args), generation,
                        native_plane.metadata.attempt, trainer_incarnation,
                        native_plane.metadata.layout_digest,
                        native_plane.metadata.base_digest),
                    marker, weight=tokens, digest=marker["source_sha256"])
                pipeline.release(token)
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
            _wait_for_leader_apply_release(
                bulk, generation=generation, fence=fence, deadline=exchange_deadline)
            exchange_deadline = time.monotonic() + min(args.deadline_s, 180.0)
        if native:
            native_context = native_plane.result_shards(
                deadline=exchange_deadline,
                chunk_elements=max(1, args.bulk_chunk_bytes // 4))
            manifest, aggregate = native_context.__enter__()
            committed = CommittedResult(
                GenerationIdentity(
                    args.run_id, _fence_epoch(args), generation,
                    int(manifest["attempt"]), trainer_incarnation,
                    str(manifest["layout_digest"]), str(manifest["base_digest"])),
                manifest, str(manifest["result_root"]),
                str(manifest.get("membership_root", manifest["result_root"])),
                int(manifest["global_weight"]), time.monotonic_ns())
            if not pipeline.publish_committed(
                    committed, verify=finite_result_verifier):
                native_context.__exit__(None, None, None)
                raise ValueError("native pipeline rejected committed result")
            admitted = pipeline.take_at_boundary(
                trainer_generation=generation, fence=_fence_epoch(args),
                incarnation=trainer_incarnation,
                base_digest=str(manifest["base_digest"]))
            if admitted is None or admitted.payload is not manifest:
                native_context.__exit__(None, None, None)
                raise ValueError("native committed result missed safe boundary")
            _stage_telemetry(
                bulk, identity, generation, "native_generation_pipeline",
                generation_started, args.deadline_s,
                **pipeline.metrics.__dict__)
        else:
            manifest, aggregate = spool.stream_aggregate(
                fence, deadline=exchange_deadline,
                expected_source_id=args.source_id)
        # Waiting for distributed ownership (and, on node 0 peers, the leader
        # checkpoint marker) has its own bounded window.  Once the complete
        # node-local aggregate is visible, begin a fresh supervised apply
        # window for every trainer; liveness alone must not disguise progress.
        heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                  stage="peer_apply")
        if native:
            apply_lane_deadline = (
                time.monotonic() + min(args.deadline_s, 180.0))
            _wait_for_native_apply_lane(
                control, args, generation=generation, rank=rank,
                result_root=str(manifest["result_root"]),
                deadline=apply_lane_deadline)
        # Waiting for the preceding local reader is bounded by the complete
        # exchange window, but is not itself APPLY work.  Start the per-reader
        # APPLY SLO only after this trainer owns the shared-result read lane.
        trainer_apply_started = time.monotonic()
        if not native:
            spool.release_trainer(fence, rank)
        state = apply_delta(state, aggregate, eta_outer=args.eta_outer, in_place=True)
        if native:
            native_context.__exit__(None, None, None)
            _stage_telemetry(
                bulk, identity, generation, "native_trainer_apply",
                trainer_apply_started, PoolStageSLO.production().apply_s,
                lane_rank=rank, result_bytes=int(manifest["result_bytes"]),
                python_dense_socket_bytes=0)
            # Release the one-reader lane as soon as native result mapping and
            # model apply finish.  The durable per-trainer recovery checkpoint
            # below is independent local I/O; serializing it behind this
            # credit made later ranks exceed APPLY despite bounded apply work.
            atomic_metadata(
                control / f"native-result-applied-{generation:08d}-{rank:02d}.json", {
                    "schema": "emender-native-e97-result-applied-lane-v1",
                    "run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                    "generation": generation, "result_root": manifest["result_root"],
                    "rank": rank,
                })
        accepted_token_clock += int(manifest["weight"])
        step += args.local_steps; losses.append(loss)
        completed = generation + 1
        heartbeat(bulk, identity, generation=completed, step=step, loss=loss,
                  stage="checkpoint_commit" if node == 0 and rank == 0
                  else "redistribution")

        # The authoritative leader proposal is on the first-commit critical
        # path. Its exclusive streamed apply must flow directly into one
        # complete checkpoint and proposal before same-node peer reads begin.
        # The same immutable file is also valid leader recovery state, so do
        # not serialize the model twice.
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
                        "optimizer_state_dict": optimizer_state,
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
                        "optimizer_state_dict": optimizer_state, "migration": migration,
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
            _publish_role_recovery(
                control, identity, args, completed, step=step,
                checkpoint=str(recovery_checkpoint),
                checkpoint_sha256=__import__("hashlib").sha256(
                    recovery_checkpoint.read_bytes()).hexdigest(),
                membership=manifest["members"], fence=fence.__dict__,
                accepted_tokens=accepted_token_clock,
                **({"native_runtime_digests": native_runtime} if native else {}))
        if native:
            native_plane.close()
            atomic_metadata(
                control / f"native-applied-{generation:08d}-{rank:02d}.json", {
                    "schema": "emender-native-e97-applied-v1",
                    "run_id": args.run_id, "fence_epoch": _fence_epoch(args),
                    "generation": generation, "result_root": manifest["result_root"],
                    "rank": rank, "checkpoint": str(recovery_checkpoint),
                    "accepted_tokens": accepted_token_clock,
                })
        heartbeat(bulk, identity, generation=generation + 1, step=step, loss=loss, stage="applied")
    if scheduler is not None:
        scheduler.close()
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.generations <= 0 or args.deadline_s <= 0:
        raise ValueError("generations and deadlines must be positive")
    if not 0 < args.local_spool_chunk_bytes <= args.max_spool_bytes:
        raise ValueError("local spool chunk bound must be positive and within the byte ledger")
    if not 0 < args.bulk_chunk_bytes <= args.max_spool_bytes:
        raise ValueError("owner transport chunk bound must be positive and within the byte ledger")
    args._dataplane_attestation = _attest_dataplane(
        args)  # hard gate before trainer calls ``_load_real``
    return manager(args) if args.role == "manager" else trainer(args)


if __name__ == "__main__":
    raise SystemExit(main())
