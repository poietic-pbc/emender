#!/usr/bin/env python3
"""Real manager/trainer entrypoints for the split resilient E97 launcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    encode_owner_frame_fd, layout_identity, runtime_digests, state_digest,
    state_elements, wait_metadata,
)
from ndm.native_pool_runtime import NativeManagerSession
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


def _native_peer_exchange(session: NativeManagerSession, local_result, *, args,
                          node: int, peer_id: str, peer_incarnation: str,
                          peer_root: bytes, peer_weight: int,
                          deadline: float) -> int:
    """Symmetric two-node memfd/libfabric exchange with bounded chunk replay."""
    payload_max = args.bulk_chunk_bytes
    chunk_count = (local_result.length + payload_max - 1) // payload_max
    remote_fd = create_memfd("emender-ndp-remote-result", allow_sealing=True)
    os.ftruncate(remote_fd, local_result.length)

    def frame_deadline() -> int:
        return min(session.transport.deadline_unix_ns,
                   time.time_ns() + max(1, int(
                       (deadline - time.monotonic()) * 1e9)))

    def send_all(*, sequence_base: int) -> None:
        for chunk in range(chunk_count):
            offset = chunk * payload_max
            extent = min(payload_max, local_result.length - offset)
            frame_fd, frame_bytes = encode_owner_frame_fd(
                source_fd=local_result.fd, payload_offset=offset,
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

    def send_credits(*, sequence_base: int) -> None:
        for chunk in range(chunk_count):
            offset = chunk * payload_max
            extent = min(payload_max, local_result.length - offset)
            credit_fd = encode_credit_frame_fd(
                payload_offset=offset, payload_bytes=extent,
                payload_max=payload_max, run_id=args.run_id,
                fence_epoch=_fence_epoch(args), generation=local_result.generation,
                attempt=local_result.attempt,
                owner_epoch=local_result.client.owner_epoch,
                worker_id=f"node-{node}",
                incarnation=session.owner_endpoint.incarnation,
                layout_digest=local_result.layout_digest,
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

    def receive_credits() -> None:
        seen: set[int] = set()
        while len(seen) != chunk_count:
            if time.monotonic() >= deadline:
                raise TimeoutError("native owner credit deadline expired")
            credit_fd = create_memfd("emender-ndp-rx-credit")
            os.ftruncate(credit_fd, 320)
            try:
                received = session.receive_owner_fd(credit_fd, capacity=320)
                if received is None:
                    time.sleep(.001); continue
                worker, frame_bytes = received
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
                expected_offset = chunk * payload_max
                expected_extent = min(payload_max, local_result.length - expected_offset)
                if (int(value["payload_offset"]) != expected_offset
                        or int(value["credit"]) != expected_extent):
                    raise ValueError("native owner credit extent mismatch")
                seen.add(chunk)
            finally:
                os.close(credit_fd)

    def receive_all() -> None:
        seen: set[int] = set()
        capacity = payload_max + 320
        while len(seen) != chunk_count:
            if time.monotonic() >= deadline:
                raise TimeoutError("native owner redistribution deadline expired")
            frame_fd = create_memfd("emender-ndp-rx-frame", allow_sealing=True)
            os.ftruncate(frame_fd, capacity)
            try:
                received = session.receive_owner_fd(frame_fd, capacity=capacity)
                if received is None:
                    time.sleep(.001); continue
                worker, frame_bytes = received
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
                        "layout_digest": local_result.layout_digest,
                        "base_digest": local_result.base_digest,
                        "result_root": peer_root,
                        "weight": peer_weight,
                        "chunk_count": chunk_count,
                    })
                chunk = int(value["chunk_index"])
                if chunk in seen:
                    continue  # authenticated idempotent replay
                expected_offset = chunk * payload_max
                if int(value["payload_offset"]) != expected_offset:
                    raise ValueError("native redistribution chunk offset mismatch")
                _copy_frame_payload(
                    frame_fd, remote_fd, payload_bytes=int(value["payload_bytes"]),
                    payload_offset=expected_offset)
                seen.add(chunk)
            finally:
                os.close(frame_fd)

    # Static byte credits are granted before each direction; they are distinct
    # from CQ completion and bound each exact frozen chunk. The lower worker's
    # data direction runs first, without a fixed-world rendezvous.
    if f"node-{node}" < peer_id:
        receive_credits(); send_all(sequence_base=1)
        send_credits(sequence_base=chunk_count + 1); receive_all()
    else:
        send_credits(sequence_base=1); receive_all()
        receive_credits(); send_all(sequence_base=chunk_count + 1)
    seal_memfd(remote_fd)
    return remote_fd


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
        production=production, full_layout=full_layout, deadline_s=args.deadline_s,
        telemetry_path=bulk / "telemetry" / f"{identity}-native.jsonl",
        payload_max=args.bulk_chunk_bytes, resident_limit_bytes=args.max_spool_bytes)
    control = bulk / "control"
    session.write_readiness(control / "native-service-ready.json")
    heartbeat(bulk, identity, generation=args.initial_generation,
              step=args.initial_generation * args.local_steps, loss=None,
              stage="native_service_ready")
    pool_config = _pool_config(args)
    control_server = control_thread = None
    pool_client = None
    if args.node_count > 1:
        if args.node_count != 2:
            raise ValueError("native E97 v1 owner exchange currently requires exactly two nodes")
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
        pool_client.ready(session.owner_endpoint, args.initial_generation,
                          run_id=args.run_id, fence=_fence_epoch(args))
    term_requested = {"value": False}

    def request_term(*_ignored) -> None:
        term_requested["value"] = True

    signal.signal(signal.SIGTERM, request_term)
    liveness_stop, liveness_thread = _liveness_heartbeat(bulk, identity)
    try:
        for generation in range(args.initial_generation,
                                args.initial_generation + args.generations):
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
            freeze = session.freeze(
                deadline_s=max(.001, native_deadline - time.monotonic()))
            local_operation, local_result = session.finalize_redistribution(
                deadline_s=max(.001, native_deadline - time.monotonic()))
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
                if len(workers) != 2 or f"node-{node}" not in workers:
                    raise RuntimeError("native two-node accepted set is incomplete")
                endpoints = tuple(_owner_endpoint_from_snapshot(peer)
                                  for peer in snapshot["peers"]
                                  if str(peer["worker_id"]) in workers)
                session.install_routes(endpoints)
                peer_id = next(item for item in workers if item != f"node-{node}")
                peer_endpoint = next(item for item in endpoints
                                     if item.worker_id == peer_id)
                weights = {str(key): int(value) for key, value in
                           dict(close["accepted_weights"]).items()}
                roots = {str(key): bytes.fromhex(str(value)) for key, value in
                         dict(close["accepted_payloads"]).items()}
                heartbeat(bulk, identity, generation=generation,
                          step=generation * args.local_steps, loss=None,
                          stage="owner_transport")
                exchange_deadline = time.monotonic() + pool_config.slo.transport_s
                remote_fd = _native_peer_exchange(
                    session, local_result, args=args, node=node, peer_id=peer_id,
                    peer_incarnation=peer_endpoint.incarnation,
                    peer_root=roots[peer_id], peer_weight=weights[peer_id],
                    deadline=exchange_deadline)
                local_fd = os.dup(local_result.fd)
                session.abort(deadline_s=5)
                local_result.close(); local_operation.close(); freeze.close()
                session.install_reduction_attempt(
                    generation=generation, attempt=2, owner_epoch=1,
                    source_dtype=DType.F64, base_digest=base_digest,
                    plan_digest=plan_digest,
                    deadline_s=max(.001, args.deadline_s))
                registered = []
                for sequence, worker in enumerate(workers):
                    fd = local_fd if worker == f"node-{node}" else remote_fd
                    buffer = session.local.register_memfd(
                        fd, length=elements * 8, handle_generation=generation)
                    operation = session.local.submit(
                        buffer, trainer_key=worker,
                        trainer_incarnation=(incarnation if worker == f"node-{node}"
                                             else peer_endpoint.incarnation),
                        submission_seq=sequence, weight=weights[worker],
                        source_dtype=DType.F64, deadline_s=args.deadline_s)
                    buffer.close(); registered.append(operation)
                os.close(local_fd); os.close(remote_fd)
                freeze = session.freeze(deadline_s=args.deadline_s)
                final_operation, final_result = session.finalize_redistribution(
                    deadline_s=args.deadline_s)
                for operation in registered:
                    operation.close()
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
                "operation_handle": final_operation.handle,
                "layout_digest": final_result.layout_digest.hex(),
                "base_digest": final_result.base_digest.hex(),
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
            apply_deadline = time.monotonic() + pool_config.slo.apply_s
            for rank in range(args.local_quorum):
                wait_metadata(
                    control / f"native-applied-{generation:08d}-{rank:02d}.json",
                    deadline=apply_deadline,
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
            if pool_client is not None:
                pool_client.ready(session.owner_endpoint, generation + 1,
                                  run_id=args.run_id, fence=_fence_epoch(args))
    except BaseException:
        try:
            session.abort(deadline_s=1)
        except Exception:
            pass
        raise
    finally:
        liveness_stop.set(); liveness_thread.join(10)
        if pool_client is not None:
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
    if args.resume_handoff:
        handoff = json.loads(Path(args.resume_handoff).read_text())
        checkpoint_path = Path(handoff["checkpoint"])
        if __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest() != handoff["checkpoint_sha256"]:
            raise ValueError("resume checkpoint checksum mismatch")
        resumed = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        if (native and dict(resumed.get("native_runtime_digests", {}))
                != native_runtime):
            raise ValueError("resume checkpoint native runtime digest mismatch")
        if (int(resumed["generation"]) != args.initial_generation
                or resumed["outer_update_state"] != handoff["outer_update_state"]):
            raise ValueError("resume generation/outer state does not match handoff")
        state = {name: value.clone() for name, value in resumed["model_state_dict"].items()}
        optimizer_state, step = resumed["optimizer_state_dict"], int(resumed["step"])
        migration = {"status": "restored", "state": resumed["outer_update_state"],
                     "policy": "new-harness-handoff"}
        accepted_token_clock = int(handoff.get("accepted_tokens", 0))
        async_chain = list(handoff.get("async_chain", ())) + [str(Path(args.resume_handoff).resolve())]
        if (handoff.get("run_id") != args.run_id or handoff.get("payload_id") != args.payload_id
                or handoff.get("source_id") != args.source_id
                or int(handoff["fence"]["coordinator_epoch"]) > _fence_epoch(args)
                or not handoff.get("finalized")):
            raise ValueError("resume handoff membership/identity/fence mismatch")
        if fenced is not None:
            latest = fenced[0].read_publication(args.run_id, "latest", "authoritative")
            if (latest is None or int(latest["generation"]) != args.initial_generation
                    or latest["manifest_sha256"] != __import__("hashlib").sha256(
                        Path(args.resume_handoff).read_bytes()).hexdigest()):
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
        start_generation = args.initial_generation
    losses = []
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("requested", True))
    completed = start_generation
    leader_checkpoint: Path | None = None
    trainer_incarnation = uuid.uuid4().hex
    for generation in range(start_generation, target_generation):
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
                heartbeat(
                    bulk, identity, generation=generation,
                    step=int(details.get("step", step)),
                    loss=details.get("loss"), stage=phase)

            def publish_trained_delta(base_state, model, tokens):
                if native:
                    heartbeat(bulk, identity, generation=generation, step=step,
                              loss=None, stage="streaming_delta")
                    native_plane.publish_model_delta(
                        base_state, model, tokens,
                        chunk_elements=max(1, args.local_spool_chunk_bytes // 4),
                        deadline_s=max(.001, generation_deadline - time.monotonic()))
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
                native_plane.publish_flat_shards(
                    flatten_tensors(
                        delta, chunk_elements=max(1, args.bulk_chunk_bytes // 4)),
                    tokens=tokens,
                    deadline_s=max(.001, generation_deadline - time.monotonic()))
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
        if not native:
            spool.release_trainer(fence, rank)
        state = apply_delta(state, aggregate, eta_outer=args.eta_outer, in_place=True)
        if native:
            native_context.__exit__(None, None, None)
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

        if leader_checkpoint is None:
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
