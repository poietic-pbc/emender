import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from ndm.resilient_e97_roles import LocalFence, LocalTrainerSpool
from ndm.resilient_e97_runtime import (
    apply_delta, apply_delta_with_correction_ledger, assert_node_local_path,
    finalize_checkpoint, flatten_tensors, outer_state_migration,
)
from ndm.manifest_peer_control import FenceRejected
from ndm.native_e97_runtime import GenerationMetadata, SCHEMA, state_digest


# Exact node-0 native-generation-00000001.json identity retained from Frontier
# job 5039258.  The runtime digests are intentionally preserved: this catches
# producer/consumer drift rather than testing a hand-written minimal payload.
JOB_5039258_GENERATION_1 = {
    "attempt": 1,
    "base_digest": "d9486e29c02b22a453ad70d2205cf801c05222f9454e89dcd79a4326b740e53e",
    "deadline_unix_ns": 1784585715282347119,
    "fence_epoch": 1,
    "generation": 1,
    "layout_digest": "e3fb15da10a151dbd33d6f66a3a2f8723be69bbaf7b34a6b3652bee0f5a352e2",
    "owner_epoch": 1,
    "plan_digest": "4cfa18b8601f69b77a069f6459648823e46db0a9ac113dbbebf425e3334ca663",
    "run_id": "exact-2n-clean-overlap-3cfed722beb0",
    "runtime_digests": {
        "artifacts": {
            "local_library": "e049cf7a31d38be642c113a1d633401359d9fe14caf4991b1e1cc68aec0a2e4c",
            "service_binary": "a0b5ea04695886685acf32054f4ca09132853e4e97240350a9fbbeb2e43e0a5b",
            "synthetic_gate_binary": "b23fb009ecb84d184c0c7549fa5913dc462a1a31941074d745630a83dd6d6fac",
            "transport_library": "132554f1ecdeedfb8033f468715aa74aecfc8c3d6ef5ce1dad2aed74f641f97a",
        },
        "build_bundle_sha256": "59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d",
        "build_manifest_sha256": "af148a0a064a3398088f445a7237d32aa4b3d77994c2a2dcc17a1c125b3bbfa2",
        "config_sha256": "afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c",
        "provider": "cxi",
        "provider_sha256": "9f78e6ad1221d69097239d1f29df35d87d62a2ab7f1051875c561c3f7b4bc6a4",
        "schema": "emender-native-e97-runtime-digests-v1",
        "source_commit": "3cfed722beb086d015cff254f473af2a63eaa492",
    },
    "schema": SCHEMA,
    "total_elements": 1376692624,
}


ROLE = Path(__file__).parents[1] / "scripts/frontier/resilient_e97_role.py"


def test_job_5039258_generation_1_reconnect_identity_is_valid(monkeypatch):
    """The emitted next-generation record was valid before cleanup ate its TTL."""
    import ndm.native_e97_runtime as native_runtime

    monkeypatch.setattr(native_runtime.time, "time_ns", lambda: 1784585346000000000)
    monkeypatch.setattr(native_runtime, "wait_metadata",
                        lambda *_args, **_kwargs: JOB_5039258_GENERATION_1)
    monkeypatch.setattr(native_runtime, "artifact_path",
                        lambda *_args, **_kwargs: Path("unused-native-library"))
    monkeypatch.setattr(native_runtime, "NativeLibrary",
                        lambda *_args, **_kwargs: object())

    class ReconnectedClient:
        attached = None

        @classmethod
        def open(cls, **_kwargs):
            return cls()

        def attach_generation(self, **kwargs):
            self.attached = kwargs

    monkeypatch.setattr(native_runtime, "Client", ReconnectedClient)
    plane = native_runtime.NativeTrainerDataPlane.connect(
        build_manifest="unused-manifest", socket_path="unused-socket",
        run_id=JOB_5039258_GENERATION_1["run_id"], fence_epoch=1,
        generation=1, rank=7, identity="node-0-trainer-7",
        incarnation="recovery-incarnation", control_root="unused-control",
        deadline=time.monotonic() + 10)
    assert plane.metadata.generation == 1
    assert plane.client.attached["deadline_unix_ns"] == 1784585715282347119
    assert plane.metadata.deadline_unix_ns - native_runtime.time.time_ns() > 360_000_000_000


@pytest.mark.parametrize("mutation", [
    {"deadline_unix_ns": 1784585345999999999},  # stale
    {"owner_epoch": 0},                         # partial identity
    {"layout_digest": "g" * 64},               # corrupt digest
    {"total_elements": float("nan")},           # non-finite/coercible input
    {"fence_epoch": 0},                         # obsolete fence
])
def test_generation_identity_rejects_stale_partial_corrupt_nonfinite_and_obsolete(
        monkeypatch, mutation):
    import ndm.native_e97_runtime as native_runtime

    monkeypatch.setattr(native_runtime.time, "time_ns", lambda: 1784585346000000000)
    value = {**JOB_5039258_GENERATION_1, **mutation}
    with pytest.raises((TypeError, ValueError), match="native E97 generation"):
        GenerationMetadata.from_json(value)


def test_reconstructed_trainer_rejects_stale_manager_metadata_before_native_open(
        monkeypatch):
    """A prior node incarnation cannot attach or mutate the fresh service."""
    import ndm.native_e97_runtime as native_runtime

    metadata = GenerationMetadata(
        run_id="run", fence_epoch=7, generation=1, attempt=1,
        owner_epoch=1, total_elements=1,
        layout_digest="1" * 64, base_digest="2" * 64,
        plan_digest="3" * 64, deadline_unix_ns=time.time_ns() + 10**12,
        runtime_digests={"schema": "runtime"},
        policy_id="async-decoupled-v2.1-simple",
        policy_digest="4" * 64, code_digest="5" * 64,
        base_global_version=1, local_window_start=1, local_window_end=2,
        policy_schema="emender-async-policy-v2.1",
        contribution_schema="emender-native-e97-submission-v2.1",
        native_abi=0x00020001, wire_protocol_major=2,
        wire_protocol_minor=1, stable_worker_id="node-1",
        worker_incarnation="failed-node-incarnation",
    )
    monkeypatch.setattr(
        native_runtime, "wait_metadata",
        lambda *_args, **_kwargs: metadata.as_json())

    class MustNotOpen:
        @classmethod
        def open(cls, **_kwargs):
            raise AssertionError("stale metadata reached native mutation")

    monkeypatch.setattr(native_runtime, "Client", MustNotOpen)
    with pytest.raises(ValueError, match="stale node incarnation"):
        native_runtime.NativeTrainerDataPlane.connect(
            build_manifest="unused", socket_path="unused",
            run_id="run", fence_epoch=7, generation=1, rank=4,
            identity="node-1-trainer-4", incarnation="fresh-trainer",
            worker_incarnation="fresh-node-incarnation",
            control_root="unused", deadline=time.monotonic() + 1)


def test_native_state_digest_hashes_exact_bfloat16_storage_bits():
    state = {"bf": torch.tensor([1.0, -2.0], dtype=torch.bfloat16)}
    expected = hashlib.sha256(b"emender-native-e97-base-v1\0")
    expected.update((2).to_bytes(4, "little")); expected.update(b"bf")
    expected.update(b"torch.bfloat16\0")
    expected.update((1).to_bytes(4, "little"))
    expected.update((2).to_bytes(8, "little"))
    expected.update(b"\x80\x3f\x00\xc0")

    assert state_digest(state) == expected.digest()


def test_manager_publishes_heartbeat_before_heavy_runtime_imports():
    text = ROLE.read_text()
    bootstrap = text.index("_IMPORT_HEARTBEAT = _role_import_heartbeat()")
    assert bootstrap < text.index("import torch")
    assert bootstrap < text.index("from ndm.resilient_e97_runtime import")
    assert '"stage": "runtime_import"' in text
    assert "os.replace(temporary, state)" in text


def test_live_native_selection_is_wired_and_python_debug_remains_explicit():
    from ndm.native_artifacts import NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG
    from scripts.frontier import resilient_e97_role as role

    for backend in (PYTHON_TCP_DEBUG, NATIVE_TEST, NATIVE_CXI):
        role._require_wired_dense_runtime(backend)

    source = ROLE.read_text()
    native_manager = source[source.index("def _native_manager(args)"):
                            source.index("def manager(args)")]
    assert "LocalTrainerSpool(" not in native_manager
    assert "DistributedOwnerServer(" not in native_manager
    assert "NativeManagerSession.start(" in native_manager
    assert "spool = (LocalTrainerSpool" in source
    assert "if not native else None" in source
    assert "manager/trainer native runtime digest mismatch" in source
    assert "resume checkpoint native runtime digest mismatch" in source
    assert "role recovery native runtime digest mismatch" in source


def test_native_manager_endpoint_lifetime_spans_all_configured_generations():
    from scripts.frontier import resilient_e97_role as role

    args = SimpleNamespace(deadline_s=600.0, generations=3)
    assert role._native_manager_session_lifetime_s(args) == 1800.0
    manager = ROLE.read_text()[ROLE.read_text().index("def _native_manager(args)"):]
    assert "deadline_s=_native_manager_session_lifetime_s(args)" in manager


def test_native_generation_lifetime_spans_all_post_result_phase_clocks():
    """Frontier job 5083440 reached two complete node applies before expiry.

    A finalized native result must remain valid across the independently
    bounded checkpoint/candidate, safe-boundary rendezvous, and all-eight
    apply/commit phases.  Treating only 180 seconds as the post-result lifetime
    made the native COMMIT reject both otherwise complete node transactions.
    """
    from scripts.frontier import resilient_e97_role as role

    args = SimpleNamespace(deadline_s=420.0)
    assert role._native_post_result_lifetime_s(args) == 420.0 + 420.0 + 60.0


def test_restarted_trainer_resolves_newer_authoritative_handoff(tmp_path):
    from scripts.frontier import resilient_e97_role as role

    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    configured = handoff_root / "generation-00000005-fence-00000002.json"
    configured.write_text('{"generation":5}\n')
    committed = handoff_root / "generation-00000006-fence-00000004.json"
    committed.write_text('{"generation":6}\n')
    digest = hashlib.sha256(committed.read_bytes()).hexdigest()
    (handoff_root / "latest.json").write_text(json.dumps({
        "generation": 6, "fence": 4, "manifest": str(committed),
        "manifest_sha256": digest,
    }))

    class Authority:
        def current_commit(self, claim):
            assert claim == "claim"
            return SimpleNamespace(
                generation=6,
                manifest_path=committed.resolve(),
                manifest_sha256=digest,
            )

    args = SimpleNamespace(
        resume_handoff=str(configured), run_id="run",
        initial_generation=5, generations=3,
    )
    assert role._authoritative_trainer_resume_handoff(
        tmp_path, args, (Authority(), "claim")) == committed.resolve()

    committed.write_text('{"generation":6,"corrupt":true}\n')
    with pytest.raises(ValueError, match="not authoritative"):
        role._authoritative_trainer_resume_handoff(
            tmp_path, args, (Authority(), "claim"))


def test_atomic_cohort_recovery_rejects_stale_fence_incarnation_and_generation(
        tmp_path, monkeypatch):
    from scripts.frontier import resilient_e97_role as role

    control = tmp_path / "control"
    control.mkdir()
    path = control / "atomic-cohort-recovery.json"
    value = {
        "schema": "emender-async-v21-atomic-cohort-recovery-v1",
        "run_id": "run", "allocation_fence": 7, "node_rank": 1,
        "failed_incarnation": "failed-node",
        "node_incarnation": "fresh-node", "restart_sequence": 2,
        "authoritative_generation": 1,
        "required_trainers": list(range(8)),
        "status": "reconstructed",
    }
    path.write_text(json.dumps(value))
    monkeypatch.setenv("RESILIENT_E97_COHORT_RESTART_SEQUENCE", "2")
    monkeypatch.setenv("RESILIENT_E97_FENCE_EPOCH", "7")
    args = SimpleNamespace(run_id="run", coordinator_epoch=7)

    assert role._validate_atomic_cohort_recovery(
        control, args, node=1, node_incarnation="fresh-node",
        generation=1, admitted_statuses=("reconstructed",)) == value
    for mutation in (
        {"allocation_fence": 6},
        {"node_incarnation": "failed-node"},
        {"authoritative_generation": 0},
    ):
        path.write_text(json.dumps({**value, **mutation}))
        with pytest.raises(ValueError, match="identity/fence"):
            role._validate_atomic_cohort_recovery(
                control, args, node=1, node_incarnation="fresh-node",
                generation=1, admitted_statuses=("reconstructed",))


def test_async_v2_boundary_rejects_latest_that_differs_from_commit_receipt(
        tmp_path):
    from scripts.frontier import resilient_e97_role as role

    handoff = tmp_path / "handoff"
    handoff.mkdir()
    manifest = handoff / "generation-00000001-fence-00000001.json"
    manifest.write_text(json.dumps({
        "run_id": "run",
        "generation": 1,
        "fence": {"coordinator_epoch": 1},
        "finalized": True,
    }))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (handoff / "latest.json").write_text(json.dumps({
        "generation": 1,
        "fence": 1,
        "manifest": str(manifest),
        "manifest_sha256": digest,
    }))

    bulk_root = tmp_path / "bulk"
    control = bulk_root / "run" / "node-0" / "control"
    control.mkdir(parents=True)
    (control / "peer-commit-00000001.json").write_text(json.dumps({
        "generation": 1,
        "fence": 1,
        "commit_receipt_digest": "1" * 64,
    }))

    class Authority:
        def current_commit(self, claim):
            assert claim == "claim"
            return SimpleNamespace(
                generation=1,
                allocation_fence=1,
                manifest_sha256="0" * 64,
                receipt_digest="1" * 64,
                pointer=lambda: {
                "generation": 1,
                "fence": 1,
                "manifest": str(manifest),
                "manifest_sha256": digest,
                },
            )

    with pytest.raises(ValueError, match="immutable commit authority"):
        role._reload_verified_async_v2_latest(
            tmp_path,
            SimpleNamespace(
                run_id="run", coordinator_epoch=1,
                bulk_root=str(bulk_root)),
            (Authority(), "claim"),
            generation=1,
            deadline=time.monotonic() + 1,
        )


def test_native_result_preparation_and_foreground_apply_have_separate_deadlines():
    source = ROLE.read_text()
    trainer = source[source.index("def trainer(args)"):]
    preparation = trainer.index('stage="result_preparation"')
    prepared = trainer.index("native-candidate-prepared-", preparation)
    rendezvous = trainer.index(
        'marker_name="native-boundary-rendezvous"', prepared)
    boundary = trainer.index("native-boundary-ready-", rendezvous)
    release = trainer.index(
        'marker_name="native-apply-release"', boundary)
    foreground = trainer.index('stage="peer_apply"', release)

    assert "ASYNC_V21_BOUNDARY_RENDEZVOUS_S" in trainer[prepared:foreground]
    assert "ASYNC_V21_ALL_EIGHT_APPLY_S" in trainer[release:foreground]
    assert (
        preparation < prepared < rendezvous < boundary < release < foreground
    )


def test_owner_endpoint_snapshot_filters_control_only_lease_metadata():
    from scripts.frontier import resilient_e97_role as role

    endpoint = role._owner_endpoint_from_snapshot({
        "worker_id": "node-0", "incarnation": "node-0-boot",
        "host": "127.0.0.1", "port": 29571,
        "backend": "python-tcp-debug", "lease_expiry": 1234.5,
    })

    assert endpoint.worker_id == "node-0"
    assert endpoint.incarnation == "node-0-boot"
    assert not hasattr(endpoint, "lease_expiry")


def test_native_owner_credits_follow_reciprocal_pair_route_readiness():
    source = ROLE.read_text()
    manager = source[source.index("def _native_manager(args)"):
                     source.index("def manager(args)")]
    install = manager.index("session.install_routes(endpoints)")
    reciprocal_ready = manager.index("pool_client.await_peer_route_ready(")
    exchange = manager.index("_native_sharded_owner_reduce(")

    assert install < reciprocal_ready < exchange
    assert '"native_route_readiness"' in manager
    assert "pairwise=True" in manager
    assert 'thread_name_prefix="native-route-ready"' in manager
    assert 'thread_name_prefix="native-shard-contribution"' in source
    assert 'thread_name_prefix="native-shard-redistribution"' in source
    assert "_NativePeerInbox(" in source


def test_native_peer_inbox_demultiplexes_with_fixed_queue_bound():
    from scripts.frontier import resilient_e97_role as role

    class FakeSession:
        def __init__(self):
            self.frames = [
                ("node-2", 320), ("node-1", 640), ("node-3", 320),
                ("node-2", 640), ("node-3", 640), ("node-1", 320),
            ]
            self.lock = threading.Lock()

        def receive_owner_fd(self, _fd, *, capacity):
            assert capacity == 960
            with self.lock:
                return self.frames.pop(0) if self.frames else None

    session = FakeSession()
    observed = {}
    with role._NativePeerInbox(
            session, peer_ids=("node-1", "node-2", "node-3"),
            capacity=960, frames_per_peer=2,
            deadline=time.monotonic() + 2, queue_slots=1) as inbox:
        def consume(peer):
            values = []
            for _ in range(2):
                frame_fd, frame_bytes = inbox.receive(
                    peer, deadline=time.monotonic() + 2)
                try:
                    values.append(frame_bytes)
                finally:
                    os.close(frame_fd)
            observed[peer] = values

        consumers = [threading.Thread(target=consume, args=(peer,))
                     for peer in ("node-1", "node-2", "node-3")]
        for thread in consumers:
            thread.start()
        for thread in consumers:
            thread.join()

    assert session.frames == []
    assert inbox.queued_frames == 0
    assert 1 <= inbox.high_water_frames <= 3
    assert observed == {
        "node-1": [640, 320],
        "node-2": [320, 640],
        "node-3": [320, 640],
    }


def test_native_four_node_peer_schedule_is_bounded_and_deterministic():
    from ndm.resilient_pool_runtime import OwnerEndpoint
    from scripts.frontier import resilient_e97_role as role

    endpoints = tuple(
        OwnerEndpoint(f"node-{index}", f"inc-{index}", "host", 29571 + index)
        for index in (3, 0, 2, 1)
    )
    peers = role._native_remote_endpoints(
        endpoints, local_worker_id="node-2", minimum_contributions=4)

    assert [peer.worker_id for peer in peers] == ["node-1", "node-0", "node-3"]
    schedules = {
        worker: [peer.worker_id for peer in role._native_remote_endpoints(
            endpoints, local_worker_id=worker, minimum_contributions=4)]
        for worker in ("node-0", "node-1", "node-2", "node-3")
    }
    all_pairs = set()
    for round_index in range(3):
        pairs = {
            tuple(sorted((worker, schedule[round_index])))
            for worker, schedule in schedules.items()
        }
        assert len(pairs) == 2
        all_pairs.update(pairs)
    assert len(all_pairs) == 6
    with pytest.raises(ValueError, match="unique stable worker"):
        role._native_remote_endpoints(
            endpoints + (endpoints[0],), local_worker_id="node-2",
            minimum_contributions=4)
    with pytest.raises(ValueError, match="below explicit contribution floor"):
        role._native_remote_endpoints(
            endpoints[:3], local_worker_id="node-2", minimum_contributions=4)

    manager = ROLE.read_text()[ROLE.read_text().index("def _native_manager(args)"):]
    assert "native E97 v1 owner exchange currently requires exactly two nodes" not in manager
    assert "async-decoupled-v2 qualification permits exactly two nodes" not in manager
    assert "args.node_count not in (2, 4, 8, 16, 32, 64, 256)" in manager
    assert "for peer_endpoint in remote_endpoints" in manager
    assert "_native_sharded_owner_reduce(" in manager


def test_native_eight_node_peer_schedule_balances_every_pair_once():
    from ndm.resilient_pool_runtime import OwnerEndpoint
    from scripts.frontier import resilient_e97_role as role

    workers = tuple(f"node-{index}" for index in range(8))
    endpoints = tuple(
        OwnerEndpoint(worker, f"inc-{worker}", "host", 29571 + index)
        for index, worker in enumerate(reversed(workers))
    )
    schedules = {
        worker: [peer.worker_id for peer in role._native_remote_endpoints(
            endpoints, local_worker_id=worker, minimum_contributions=8)]
        for worker in workers
    }

    all_pairs = set()
    for round_index in range(7):
        pairs = {
            tuple(sorted((worker, schedule[round_index])))
            for worker, schedule in schedules.items()
        }
        assert len(pairs) == 4
        assert len({worker for pair in pairs for worker in pair}) == 8
        all_pairs.update(pairs)
    assert len(all_pairs) == 28


def test_native_eight_node_shard_ranges_are_balanced_and_byte_bounded():
    from scripts.frontier import resilient_e97_role as role

    workers = tuple(f"node-{index}" for index in range(8))
    f64_layout_bytes = 173 * (64 << 20)
    contribution = role._native_owner_ranges(
        workers, run_id="run", fence=11, generation=12, attempt=1,
        owner_epoch=1, f64_layout_bytes=f64_layout_bytes,
        payload_max=64 << 20, itemsize=8)
    redistribution = role._native_owner_ranges(
        workers, run_id="run", fence=11, generation=12, attempt=1,
        owner_epoch=1, f64_layout_bytes=f64_layout_bytes,
        payload_max=64 << 20, itemsize=4)

    contribution_counts = [len(contribution[worker]) for worker in workers]
    assert max(contribution_counts) - min(contribution_counts) == 1
    assert sum(contribution_counts) == 173
    assert {shard for ranges in contribution.values()
            for shard, _offset, _extent in ranges} == set(range(173))
    assert sum(extent for ranges in contribution.values()
               for _shard, _offset, extent in ranges) == f64_layout_bytes
    assert sum(extent for ranges in redistribution.values()
               for _shard, _offset, extent in ranges) == f64_layout_bytes // 2
    for worker in workers:
        assert [item[0] for item in contribution[worker]] == [
            item[0] for item in redistribution[worker]]
        assert all(right[0] - left[0] == len(workers)
                   for left, right in zip(contribution[worker],
                                          contribution[worker][1:]))
        assert sum(item[2] for item in contribution[worker]) <= (
            f64_layout_bytes + len(workers) - 1) // len(workers) + (64 << 20)


def test_native_packed_owner_fd_retains_only_assigned_round_robin_shards():
    from ndm.native_dataplane import create_memfd
    from scripts.frontier import resilient_e97_role as role

    source = create_memfd("owner-pack-source", allow_sealing=True)
    payload = bytes(range(96))
    assert os.pwrite(source, payload, 0) == len(payload)
    ranges = ((0, 0, 32), (2, 64, 32))
    packed_fd, packed_ranges, digest = role._packed_owner_fd(
        source, ranges=ranges)
    try:
        assert os.fstat(packed_fd).st_size == 64
        assert os.pread(packed_fd, 64, 0) == payload[:32] + payload[64:]
        assert packed_ranges == ((0, 0, 0, 32), (2, 32, 64, 32))
        assert digest == hashlib.sha256(payload[:32] + payload[64:]).digest()
    finally:
        os.close(source)
        os.close(packed_fd)


def test_native_peer_inbox_accepts_exact_per_peer_frame_budgets():
    from scripts.frontier import resilient_e97_role as role

    class FakeSession:
        def __init__(self):
            self.frames = [
                ("node-2", 320), ("node-1", 640), ("node-2", 640),
                ("node-1", 320), ("node-2", 320),
            ]
            self.lock = threading.Lock()

        def receive_owner_fd(self, _fd, *, capacity):
            assert capacity == 960
            with self.lock:
                return self.frames.pop(0) if self.frames else None

    session = FakeSession()
    observed = {}
    budgets = {"node-1": 2, "node-2": 3}
    with role._NativePeerInbox(
            session, peer_ids=tuple(budgets), capacity=960,
            frames_per_peer=budgets, deadline=time.monotonic() + 2,
            queue_slots=2) as inbox:
        def consume(peer):
            values = []
            for _ in range(budgets[peer]):
                frame_fd, frame_bytes = inbox.receive(
                    peer, deadline=time.monotonic() + 2)
                try:
                    values.append(frame_bytes)
                finally:
                    os.close(frame_fd)
            observed[peer] = values

        consumers = [threading.Thread(target=consume, args=(peer,))
                     for peer in budgets]
        for thread in consumers:
            thread.start()
        for thread in consumers:
            thread.join()

    assert session.frames == []
    assert inbox.queued_frames == 0
    assert observed == {"node-1": [640, 320], "node-2": [320, 640, 320]}


def test_native_peer_exchange_moves_only_reciprocal_sparse_ranges():
    from ndm.native_dataplane import create_memfd, seal_memfd
    from scripts.frontier import resilient_e97_role as role

    class Fabric:
        def __init__(self):
            self.queues = {worker: queue.Queue() for worker in ("node-0", "node-1")}
            self.history = []
            self.lock = threading.Lock()

    class Session:
        def __init__(self, fabric, worker):
            self.fabric, self.worker = fabric, worker
            self.owner_endpoint = SimpleNamespace(incarnation=f"{worker}-boot")
            self.transport = SimpleNamespace(deadline_unix_ns=time.time_ns() + 5_000_000_000)

        def transfer_frozen_fd(self, peer_id, fd, *, frame_bytes, **_metadata):
            frame = os.pread(fd, frame_bytes, 0)
            assert len(frame) == frame_bytes
            with self.fabric.lock:
                self.fabric.history.append(
                    (self.worker, peer_id,
                     int.from_bytes(frame[12:14], "little")))
            self.fabric.queues[peer_id].put((self.worker, frame))

        def receive_owner_fd(self, fd, *, capacity):
            try:
                worker, frame = self.fabric.queues[self.worker].get_nowait()
            except queue.Empty:
                return None
            assert len(frame) <= capacity
            assert os.pwrite(fd, frame, 0) == len(frame)
            return worker, len(frame)

    def result(worker, payload, weight):
        fd = create_memfd(f"sparse-range-{worker}", allow_sealing=True)
        assert os.pwrite(fd, payload, 0) == len(payload)
        seal_memfd(fd)
        root = hashlib.sha256(payload).digest()
        return SimpleNamespace(
            fd=fd, length=len(payload), generation=7, attempt=1,
            client=SimpleNamespace(owner_epoch=1),
            layout_digest=bytes.fromhex("11" * 32),
            base_digest=bytes.fromhex("22" * 32),
            result_root=root, global_weight=weight,
            sha256=lambda: root,
        )

    fabric = Fabric()
    sessions = [Session(fabric, f"node-{index}") for index in range(2)]
    results = [result("node-0", bytes(range(96)), 17),
               result("node-1", bytes(reversed(range(96))), 23)]
    args = SimpleNamespace(
        run_id="selective-exchange", coordinator_epoch=9,
        bulk_chunk_bytes=32,
        _async_v21_policy=role.ASYNC_DECOUPLED_V21,
    )
    deadline = time.monotonic() + 3
    exchanges = (
        dict(session=sessions[0], local_result=results[0], args=args, node=0,
             peer_id="node-1", peer_incarnation="node-1-boot",
             peer_root=results[1].result_root, peer_weight=23,
             send_ranges=((0, 0, 32), (2, 64, 32)),
             receive_ranges=((1, 0, 32, 32),),
             receive_length=32),
        dict(session=sessions[1], local_result=results[1], args=args, node=1,
             peer_id="node-0", peer_incarnation="node-0-boot",
             peer_root=results[0].result_root, peer_weight=17,
             send_ranges=((1, 32, 32),),
             receive_ranges=((0, 0, 0, 32), (2, 32, 64, 32)),
             receive_length=64),
    )
    returned = [None, None]
    failures = []

    def exchange(index):
        try:
            returned[index] = role._native_peer_exchange(
                **exchanges[index], deadline=deadline, wire_chunk_count=3)
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=exchange, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert failures == []
        expected_payloads = (
            os.pread(results[1].fd, 32, 32),
            os.pread(results[0].fd, 32, 0) + os.pread(results[0].fd, 32, 64),
        )
        for index, expected in enumerate(expected_payloads):
            remote_fd, local_digest, remote_digest = returned[index]
            assert os.fstat(remote_fd).st_size == len(expected)
            assert os.pread(remote_fd, len(expected), 0) == expected
            assert local_digest == results[index].result_root
            assert remote_digest == hashlib.sha256(expected).digest()
            os.close(remote_fd)
        assert [frame_type for _source, _peer, frame_type in fabric.history] == [
            3, 8, 3, 8, 3, 3, 8,
        ]
    finally:
        for value in results:
            os.close(value.fd)


def test_terminal_native_follower_reuses_fenced_authoritative_checkpoint(tmp_path):
    from scripts.frontier import resilient_e97_role as role

    checkpoint = tmp_path / "checkpoints/generation-00000001-fence-00000001.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authoritative-generation-one")
    manifest = tmp_path / "handoff/generation-00000001-fence-00000001.json"
    manifest.parent.mkdir(parents=True)
    value = {
        "schema": 1, "finalized": True, "run_id": "run-a",
        "payload_id": "payload-a", "source_id": "source-a",
        "generation": 1, "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "fence": {"coordinator_epoch": 1},
    }
    manifest.write_text(json.dumps(value, sort_keys=True))
    latest = tmp_path / "handoff/latest.json"
    latest.write_text(json.dumps({
        "generation": 1, "fence": 1, "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    args = SimpleNamespace(
        run_id="run-a", payload_id="payload-a", source_id="source-a",
        coordinator_epoch=1)

    assert role._terminal_native_checkpoint(
        tmp_path, args, completed=1, deadline=time.monotonic() + 1) == checkpoint
    value["payload_id"] = "stale-payload"
    manifest.write_text(json.dumps(value, sort_keys=True))
    latest.write_text(json.dumps({
        "generation": 1, "fence": 1, "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    with pytest.raises(ValueError, match="identity"):
        role._terminal_native_checkpoint(
            tmp_path, args, completed=1, deadline=time.monotonic() + 1)


def test_native_trainer_background_preparation_is_serialized_by_local_rank(
        tmp_path):
    """One reader avoids node contention before the all-eight release."""
    from scripts.frontier import resilient_e97_role as role

    control = tmp_path / "control"
    control.mkdir()
    args = SimpleNamespace(run_id="run-a", coordinator_epoch=4)
    observed = {}

    def wait_for_rank_one():
        observed["marker"] = role._wait_for_native_apply_lane(
            control, args, generation=2, rank=1,
            result_root="ab" * 32, deadline=time.monotonic() + 2)

    waiter = threading.Thread(target=wait_for_rank_one)
    waiter.start()
    time.sleep(.05)
    assert waiter.is_alive(), "rank one must not contend with rank zero's result view"
    role.atomic_metadata(control / "native-result-applied-00000002-00.json", {
        "run_id": "run-a", "fence_epoch": 4, "generation": 2,
        "result_root": "ab" * 32, "rank": 0,
    })
    waiter.join(2)

    assert not waiter.is_alive()
    assert observed["marker"]["rank"] == 0
    assert role._wait_for_native_apply_lane(
        control, args, generation=2, rank=0,
        result_root="ab" * 32, deadline=time.monotonic() + .1) is None
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args)"):]
    visible = trainer.index("manifest, aggregate = native_context.__enter__()")
    lane = trainer.index("_wait_for_native_apply_lane(", visible)
    apply = trainer.index("state = apply_delta(", lane)
    assert visible < lane < apply
    assert "min(args.deadline_s, 420.0)" in trainer[visible:apply]
    assert "deadline=apply_lane_deadline" in trainer[lane:apply]


def test_native_result_preparation_excludes_foreground_apply_interval():
    """Checkpoint work finishes before the finite all-eight foreground swap.

    Trainers prepare independently verified candidates from the one read-only
    service result through a capacity-one reader credit. Durable checkpoint
    and reload verification remain background work. Each trainer then reaches
    a distinct K boundary; the manager release begins the 60-second x/z
    translation interval only after all eight boundary receipts exist.
    """
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args)"):]
    visible = trainer.index("manifest, aggregate = native_context.__enter__()")
    lane = trainer.index("_wait_for_native_apply_lane(", visible)
    apply = trainer.index("state = apply_delta(", lane)
    lane_credit = trainer.index("native-result-applied-", apply)
    recovery_save = trainer.index("torch.save(", lane_credit)
    prepared = trainer.index("native-candidate-prepared-", recovery_save)
    rendezvous = trainer.index(
        'marker_name="native-boundary-rendezvous"', prepared)
    boundary_stop = trainer.index(
        "async_training_lane.finish_at_boundary(", rendezvous)
    boundary_ready = trainer.index("native-boundary-ready-", boundary_stop)
    release = trainer.index(
        'marker_name="native-apply-release"', boundary_ready)
    live_swap = trainer.index("safe_apply_started = time.monotonic()", release)
    durable_receipt = trainer.index("native-applied-", live_swap)

    assert (
        visible < lane < apply < lane_credit < recovery_save < prepared
        < rendezvous < boundary_stop < boundary_ready < release < live_swap
        < durable_receipt
    )


def test_checkpoint_publication_clock_closes_before_safe_boundary_wait():
    """The background checkpoint SLO must not include boundary rendezvous.

    Frontier clean job 5082866 completed immutable checkpoint publication and
    candidate preparation, then legitimately waited for the next K40 boundary.
    Keeping ``checkpoint_publication`` open across that independent wait made
    rank zero raise the 420-second background SLO after live apply and before
    its durable native-applied receipt.
    """
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args) -> int:"):]
    checkpoint_start = trainer.index(
        "checkpoint_publication_started = time.monotonic()")
    reload_verified = trainer.index(
        "_reload_verified_async_v2_latest(", checkpoint_start)
    checkpoint_complete = trainer.index(
        'bulk, identity, generation, "checkpoint_publication"',
        reload_verified,
    )
    candidate = trainer.index(
        "native-candidate-prepared-", reload_verified)
    rendezvous = trainer.index(
        'marker_name="native-boundary-rendezvous"', candidate)
    release = trainer.index(
        'marker_name="native-apply-release"', rendezvous)
    live_swap = trainer.index(
        "safe_apply_started = time.monotonic()", release)
    durable_receipt = trainer.index(
        "native-applied-", live_swap)

    assert (
        checkpoint_start < reload_verified < checkpoint_complete < candidate
        < rendezvous < release < live_swap < durable_receipt
    )
    assert (
        '"checkpoint_publication"'
        not in trainer[live_swap:durable_receipt]
    )


def test_async_v21_publishes_the_retained_endpoint_without_a_second_model_read():
    """The immutable post-K snapshot is the only dense descriptor source."""
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args) -> int:"):]
    snapshot = trainer.index("retained_endpoint = persistent_worker.snapshot()")
    publish = trainer.index("marker = native_plane.publish_state_delta(", snapshot)
    owned = trainer.index("owned_marker = marker", publish)
    descriptor_path = trainer[
        snapshot:trainer.index("fence = _fence(args, generation)", owned)]

    assert "interval_start, retained_endpoint, tokens" in descriptor_path
    assert "publish_model_delta(" not in descriptor_path
    assert '"async_v21_endpoint_snapshot"' in descriptor_path
    assert '"native_direct_memfd"' in descriptor_path


def test_production_async_lane_keeps_result_and_checkpoint_off_next_k_path():
    """The rendered trainer resumes immediately after its coherent snapshot."""
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args) -> int:"):]
    snapshot = trainer.index("retained_endpoint = persistent_worker.snapshot()")
    lane_start = trainer.index("async_training_lane.start(", snapshot)
    publish = trainer.index("native_plane.publish_state_delta(", snapshot)
    result_wait = trainer.index("native_plane.result_shards(", publish)
    candidate_apply = trainer.index("state = apply_delta(", result_wait)
    checkpoint = trainer.index("torch.save(", candidate_apply)
    verified_latest = trainer.index(
        "_reload_verified_async_v2_latest(", checkpoint)
    safe_boundary = trainer.index(
        "async_training_lane.finish_at_boundary(", verified_latest)
    verified_apply = trainer.index(
        '"native_trainer_apply"', safe_boundary)

    assert (snapshot < lane_start < publish < result_wait
            < candidate_apply < checkpoint
            < verified_latest < safe_boundary < verified_apply)
    background_path = trainer[publish:safe_boundary]
    assert '"async_v21_snapshot_admission"' in trainer[snapshot:publish]
    assert '"async_v21_result_readiness"' in trainer[publish:candidate_apply]
    completion_path = trainer[result_wait:safe_boundary]
    assert "persistent_worker.run_window(" not in completion_path
    assert "persistent_worker.translate(" not in completion_path
    assert "publish_model_delta(" not in background_path

    real_source = Path("ndm/async_diloco_real.py").read_text()
    lane = real_source[
        real_source.index("class PersistentAsyncTrainingLane"):
        real_source.index("def _run_real_worker(")
    ]
    assert "self.session.run_window(" in lane
    assert ">= self.max_windows" in lane
    assert "self.session.translate(corrections)" in lane
    assert "result_shards(" not in lane
    assert "apply_delta(" not in lane
    assert "torch.save(" not in lane
    assert "wait_metadata(" not in lane


def test_production_trainer_entrypoint_overlaps_blocked_native_result_and_applies_at_boundary(
        tmp_path, monkeypatch):
    """Drive the real role entrypoint through two commits with a blocked q result.

    The fake endpoints replace only the model math and native service.  The
    production ``trainer`` orchestration still owns bootstrap, descriptor
    sealing, the persistent K lane, checkpoint creation, fenced-latest reload,
    correction-ledger application, telemetry, and recovery receipts.
    """
    import ndm.async_diloco_real as real
    from ndm.async_diloco_v2 import ASYNC_DECOUPLED_V2
    from ndm.native_artifacts import NATIVE_TEST
    from scripts.frontier import resilient_e97_role as role

    run = tmp_path / "run"
    bulk_root = tmp_path / "bulk"
    run.mkdir()
    runtime_identity = {
        "schema": "fake-native-runtime",
        "provider": "tcp;ofi_rxm",
        "source_commit": "test-source",
    }
    result_pending = threading.Event()
    release_background = threading.Event()
    next_k_started = threading.Event()
    next_k_completed = threading.Event()
    following_k_started = threading.Event()
    allow_following_k_finish = threading.Event()
    publication_verified = threading.Event()
    first_apply = threading.Event()
    releaser_errors = []
    stage_trace = []
    model_builds = []
    optimizer_builds = []
    iterator_builds = []
    train_calls = {"count": 0}
    published_intervals = []
    translated = []

    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class ScheduleFreeFixture:
        def __init__(self, parameters):
            parameter = tuple(parameters)[0]
            self.param_groups = [{"params": [parameter], "lr": 0.1}]
            self.state = {
                parameter: {
                    "z": parameter.detach().clone(),
                    "exp_avg_sq": torch.tensor([7.0]),
                    "step": 0,
                },
            }

    def build_model(_args):
        model = OneParamModel()
        model_builds.append(model)
        return model

    def build_optimizer(model, _args):
        optimizer = ScheduleFreeFixture(model.parameters())
        optimizer_builds.append(optimizer)
        return optimizer

    def build_iterator(*_args, **_kwargs):
        iterator_builds.append(object())
        return iterator_builds[-1]

    def train_step(model, optimizer, _args, **kwargs):
        call = train_calls["count"] + 1
        train_calls["count"] = call
        if call == 41:
            next_k_started.set()
        if call == 81:
            following_k_started.set()
            assert allow_following_k_finish.wait(2)
        with torch.no_grad():
            model.weight.add_(0.025)
            optimizer.state[model.weight]["z"].add_(0.025)
        optimizer.state[model.weight]["step"] += 1
        if call == 80:
            next_k_completed.set()
        return {
            "loss": 1.0,
            "tokens_processed": 1,
            "hidden_state": call,
        }

    monkeypatch.setattr(real.train, "build_training_model", build_model)
    monkeypatch.setattr(real.train, "build_training_optimizer", build_optimizer)
    monkeypatch.setattr(real, "_build_batch_iter", build_iterator)
    monkeypatch.setattr(real.train, "train_one_optimizer_step", train_step)

    original_translate = real.PersistentRealWorkerSession.translate

    def traced_translate(self, corrections):
        before = (
            float(self.model.weight.detach()),
            float(self.optimizer.state[self.model.weight]["z"]),
        )
        original_translate(self, corrections)
        after = (
            float(self.model.weight.detach()),
            float(self.optimizer.state[self.model.weight]["z"]),
        )
        translated.append({
            "before": before,
            "after": after,
            "correction": float(corrections["weight"]),
            "publication_verified": publication_verified.is_set(),
            "windows_completed": self.windows_completed,
        })
        first_apply.set()

    monkeypatch.setattr(
        real.PersistentRealWorkerSession, "translate", traced_translate)

    class FencedStore:
        authoritative = None
        checks = 0

        def assert_current(self, lease):
            assert lease.fence == 1
            self.checks += 1

        def read_publication(self, run_id, namespace, key):
            assert (run_id, namespace, key) == (
                "entrypoint-overlap", "latest", "authoritative")
            return self.authoritative

    store = FencedStore()
    lease = SimpleNamespace(fence=1)

    class ResultContext:
        def __init__(self, plane):
            self.plane = plane

        def __enter__(self):
            generation = self.plane.generation
            if generation == 0:
                stage_trace.extend([
                    "result_shards_pending",
                    "native_reduce_pending",
                    "outer_commit_pending",
                    "checkpoint_publication_pending",
                ])
                result_pending.set()
                assert release_background.wait(2)
            marker = self.plane.marker
            local_delta = float(self.plane.local_delta)
            manifest = {
                "attempt": 2,
                "layout_digest": "1" * 64,
                "base_digest": "2" * 64,
                "result_root": f"{generation + 3:x}" * 64,
                "global_weight": int(marker["exact_tokens"]),
                "exact_tokens": int(marker["exact_tokens"]),
                "base_global_version": int(
                    marker["base_global_version"]),
                "commit_global_version": generation,
                "commit_lag": (
                    generation
                    - int(marker["base_global_version"])),
                "result_bytes": 4,
                "members": [0],
                "accepted_peers": ["node-0"],
                "accepted_local_contributions": [{
                    name: marker[name] for name in (
                        "rank", "trainer", "incarnation",
                        "contribution_sequence", "local_window_start",
                        "local_window_end", "window_count",
                        "base_global_version", "payload_digest",
                        "descriptor_digest",
                    )
                }],
            }
            return manifest, iter((torch.tensor([local_delta]),))

        def __exit__(self, *_args):
            stage_trace.append(
                f"result_view_released_g{self.plane.generation}")

    class FakeNativePlane:
        instances = []

        def __init__(self, generation, rank, identity, incarnation):
            self.generation = generation
            self.rank = rank
            self.identity = identity
            self.incarnation = incarnation
            self.marker = None
            self.local_delta = None
            self.closed = False
            self.__class__.instances.append(self)
            self.metadata = SimpleNamespace(
                runtime_digests=runtime_identity,
                stable_worker_id="node-0",
                worker_incarnation=incarnation,
                base_digest="2" * 64)

        @classmethod
        def connect(cls, **kwargs):
            return cls(
                kwargs["generation"], kwargs["rank"],
                kwargs["identity"], kwargs["incarnation"])

        def allocate_delta(self, **_kwargs):
            return object()

        def publish_state_delta(
                self, base_state, endpoint_state, tokens, *,
                contribution_identity, **_kwargs):
            base_value = float(base_state["weight"])
            endpoint_value = float(endpoint_state["weight"])
            self.local_delta = endpoint_value - base_value
            payload_digest = hashlib.sha256(
                f"payload:{self.generation}:{self.local_delta}".encode()
            ).hexdigest()
            descriptor_digest = hashlib.sha256(
                f"descriptor:{self.generation}:"
                f"{contribution_identity['local_window_start']}:"
                f"{contribution_identity['local_window_end']}".encode()
            ).hexdigest()
            self.marker = {
                **contribution_identity,
                "rank": self.rank,
                "trainer": self.identity,
                "incarnation": self.incarnation,
                "exact_tokens": int(tokens),
                "payload_digest": payload_digest,
                "descriptor_digest": descriptor_digest,
                "owned_ack_seconds": 0.001,
            }
            published_intervals.append({
                **self.marker,
                "base_value": base_value,
                "endpoint_value": endpoint_value,
                "local_delta": self.local_delta,
            })
            stage_trace.append(f"owned_g{self.generation}")
            return self.marker

        def result_shards(self, **_kwargs):
            return ResultContext(self)

        def close(self):
            self.closed = True

    def local_path(path, _shared):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fake_latest(path, *, expected, **_kwargs):
        path = Path(path)
        assert path.name == "latest.json"
        generation = int(expected["generation"])
        checkpoint = next(
            item for item in (run / "checkpoints").iterdir()
            if item.name.startswith(f"generation-{generation:08d}"))
        manifest = run / "handoff" / (
            f"generation-{generation:08d}-fence-00000001.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "schema": 1,
            "run_id": "entrypoint-overlap",
            "generation": generation,
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()).hexdigest(),
            "payload_id": "payload",
            "source_id": "source",
            "fence": {"coordinator_epoch": 1},
            "finalized": True,
        }, sort_keys=True))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        latest = {
            "generation": generation,
            "fence": 1,
            "manifest": str(manifest),
            "manifest_sha256": digest,
        }
        store.authoritative = {
            "generation": generation,
            "fence": 1,
            "manifest_sha256": digest,
        }
        path.write_text(json.dumps(latest, sort_keys=True))
        publication_verified.set()
        stage_trace.append(f"latest_cas_g{generation}")
        return latest

    def fake_boundary_control(
            _control, _args, *, generation, transaction_digest,
            marker_name, **_kwargs):
        now = time.monotonic()
        stage_trace.append(f"{marker_name}_g{generation}")
        if marker_name == "native-boundary-rendezvous":
            return {
                "run_id": "entrypoint-overlap",
                "fence_epoch": 1,
                "generation": generation,
                "transaction_digest": transaction_digest,
                "opened_monotonic_s": now,
                "boundary_deadline_monotonic_s": now + 420.0,
            }
        assert marker_name == "native-apply-release"
        return {
            "run_id": "entrypoint-overlap",
            "fence_epoch": 1,
            "generation": generation,
            "transaction_digest": transaction_digest,
            "released_monotonic_s": now,
            "apply_deadline_monotonic_s": now + 60.0,
        }

    monkeypatch.setattr(role, "_peer_authority", lambda _args: None)
    monkeypatch.setattr(
        role, "_dataplane_policy",
        lambda _args: (NATIVE_TEST, False, False))
    monkeypatch.setattr(role, "runtime_digests",
                        lambda **_kwargs: runtime_identity)
    monkeypatch.setattr(role, "assert_node_local_path", local_path)
    monkeypatch.setattr(role, "NativeTrainerDataPlane", FakeNativePlane)
    monkeypatch.setattr(role, "wait_metadata", fake_latest)
    monkeypatch.setattr(
        role, "_wait_for_native_boundary_control", fake_boundary_control)
    monkeypatch.setattr(role.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(role, "heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        role, "_liveness_heartbeat",
        lambda *_args, **_kwargs: (
            SimpleNamespace(set=lambda: None),
            SimpleNamespace(join=lambda *_args: None),
        ))
    monkeypatch.setattr(role, "_load_real", lambda _args: (
        SimpleNamespace(seed=3, bf16=False, lr=0.1),
        {"weight": torch.zeros(1)},
        {},
        0,
        {"status": "test", "state": {
            "mode": "delta_sgd", "eta_outer": 1.0,
            "step": 0, "accepted_tokens": 0}},
    ))
    monkeypatch.setenv("RESILIENT_E97_NODE_RANK", "0")
    monkeypatch.setenv("RESILIENT_E97_LOCAL_RANK", "0")
    monkeypatch.setenv("EMENDER_NDP_SOCKET", str(tmp_path / "native.sock"))
    monkeypatch.setenv("RESILIENT_E97_FENCE_EPOCH", "1")

    def release_when_next_window_is_honest_and_complete():
        try:
            assert result_pending.wait(2)
            assert next_k_started.wait(2)
            assert next_k_completed.wait(2)
            assert not first_apply.is_set()
            assert len(model_builds) == len(optimizer_builds) == 1
            assert len(iterator_builds) == 1
            torch.testing.assert_close(
                model_builds[0].weight.detach(), torch.tensor([2.0]))
            torch.testing.assert_close(
                optimizer_builds[0].state[
                    model_builds[0].weight]["z"],
                torch.tensor([2.0]))
            release_background.set()
            assert following_k_started.wait(2)
            assert publication_verified.wait(2)
            assert not first_apply.is_set()
            allow_following_k_finish.set()
        except BaseException as error:
            releaser_errors.append(error)
            release_background.set()
            allow_following_k_finish.set()

    releaser = threading.Thread(
        target=release_when_next_window_is_honest_and_complete)
    releaser.start()
    args = SimpleNamespace(
        local_steps=40,
        control=False,
        run_dir=str(run),
        bulk_root=str(bulk_root),
        run_id="entrypoint-overlap",
        initial_generation=0,
        generations=2,
        max_spool_bytes=65_000_000_000,
        native_build_manifest="unused-native-manifest",
        bulk_chunk_bytes=16,
        local_spool_chunk_bytes=16,
        source_id="source",
        payload_id="payload",
        code_id="code",
        coordinator_epoch=1,
        deadline_s=5.0,
        node_count=1,
        eta_outer=1.0,
        resume_handoff="",
        seed="unused",
        train_args_json="unused",
        data="unused",
        device="cpu",
        _async_v21_policy=ASYNC_DECOUPLED_V2,
        _dataplane_attestation={},
    )

    assert role.trainer(args) == 0
    releaser.join(2)
    assert not releaser.is_alive()
    if releaser_errors:
        raise releaser_errors[0]

    assert len(model_builds) == len(optimizer_builds) == 1
    assert len(iterator_builds) == 1
    assert train_calls["count"] == 120
    assert [(item["local_window_start"], item["local_window_end"])
            for item in published_intervals] == [(0, 1), (1, 3)]
    assert [item["base_global_version"]
            for item in published_intervals] == [0, 0]
    assert [item["local_delta"] for item in published_intervals] == \
        pytest.approx([1.0, 2.0], abs=1e-5)
    assert translated[0]["windows_completed"] == 3
    assert translated[0]["publication_verified"] is True
    assert translated[0]["before"] == pytest.approx(
        (3.0, 3.0), abs=1e-5)
    assert translated[0]["correction"] == pytest.approx(0.0, abs=1e-5)
    assert translated[0]["after"] == pytest.approx(
        (3.0, 3.0), abs=1e-5)
    assert translated[1]["correction"] == pytest.approx(0.0, abs=1e-5)
    assert translated[1]["after"] == pytest.approx(
        (3.0, 3.0), abs=1e-5)
    # The trainer lane performs no shared-store liveness polling.  Production
    # fencing is fixed in its native peer identity; only the checkpoint
    # publisher touches immutable restart authority.
    assert store.checks == 0

    records = [
        json.loads(line)
        for line in (
            bulk_root / "entrypoint-overlap" / "node-0" /
            "telemetry" / "node-0-trainer-0-pool.jsonl"
        ).read_text().splitlines()
    ]
    k_starts = [
        item for item in records
        if item["stage"] == "async_v21_k40_start"]
    apply_receipts = [
        item for item in records
        if item["stage"] == "safe_boundary_apply"]
    assert k_starts[0]["local_window"] == 1
    assert k_starts[0]["applied_anchor_version"] == 0
    assert apply_receipts[0]["local_window"] == 3
    assert apply_receipts[0]["result_version"] == 1
    assert apply_receipts[0]["anchor_lag_before_apply"] == 1
    assert apply_receipts[0]["speculative_window_lag"] == 2
    assert apply_receipts[0]["latest_cas_verified"] is True
    assert apply_receipts[0]["accepted_contribution_digest"] == \
        published_intervals[0]["descriptor_digest"]
    assert stage_trace.index("owned_g0") < stage_trace.index(
        "result_shards_pending")
    assert stage_trace.index("result_shards_pending") < stage_trace.index(
        "latest_cas_g1")


def test_native_trainer_generation_timer_covers_every_result_lifecycle_path():
    """Regression for job 5037971's post-commit_ready NameError.

    Generation entry is shared by fresh, handoff-resumed, local-recovery, and
    supervisor-restarted trainers.  Result delay/rejection and successful
    commit-ready admission all occur below this initialization.
    """
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args) -> int:"):]
    loop = trainer.index("for generation in range(start_generation, target_generation):")
    started = trainer.index("generation_started = time.monotonic()", loop)
    stop = trainer.index('if stop["requested"]:', loop)
    publish = trainer.index("native_plane.publish_flat_shards(", loop)
    result = trainer.index("native_plane.result_shards(", publish)
    candidate = trainer.index("state = apply_delta(", result)
    verified = trainer.index(
        "_reload_verified_async_v2_latest(", candidate)
    telemetry = trainer.index('"native_trainer_apply"', verified)

    assert loop < started < stop < publish < result < candidate < verified < telemetry
    assert trainer[started:telemetry].count("generation_started =") == 1


def test_native_pipeline_commit_ready_advances_without_foreground_blocking():
    """Deterministic generation-0 commit/apply then generation-1 handoff path."""
    from ndm.native_pipeline import (
        CommittedResult, GenerationIdentity, NativeGenerationPipeline,
        finite_result_verifier)

    digest = "a" * 64
    root = "b" * 64
    pipe = NativeGenerationPipeline(run_id="production-path", fence=9,
                                    incarnation="restart-a")

    def identity(generation):
        return GenerationIdentity(
            "production-path", 9, generation, 1, "restart-a", digest, digest)

    generation_0 = identity(0)
    token_0 = pipe.handoff(pipe.reserve(), generation_0, "sealed-g0",
                           weight=5_245_440, digest=digest)
    pipe.release(token_0)
    committed_0 = CommittedResult(
        generation_0, {"status": "commit_ready"}, root, root, 5_245_440,
        time.monotonic_ns())
    assert pipe.publish_committed(committed_0, verify=finite_result_verifier)
    assert pipe.take_at_boundary(
        trainer_generation=0, fence=9, incarnation="restart-a",
        base_digest=digest) is committed_0

    started = time.monotonic()
    token_1 = pipe.handoff(pipe.reserve(deadline=started + .05), identity(1),
                           "sealed-g1", weight=1, digest=digest)
    assert time.monotonic() - started < .05
    assert token_1.identity.generation == 1
    assert pipe.metrics.applied_results == 1
    pipe.release(token_1)


def test_manager_bounds_background_readiness_then_all_eight_apply_separately():
    """Unready results are background; begun all-eight apply has a 60s bound."""
    source = ROLE.read_text()
    manager = source[source.index("def _native_manager(args)"):
                     source.index("def manager(args)")]
    receipt_loop = manager.index("for rank in range(args.local_quorum):")
    node_apply_stage = manager.index('"native_node_apply_swap"', receipt_loop)
    window = manager[receipt_loop - 300:node_apply_stage + 128]

    assert "preparation_deadline" in window
    assert "min(args.deadline_s, 420.0)" in window
    assert "_coordinate_native_safe_boundary(" in window
    assert "apply_release_started = float(" in window
    assert 'apply_release["released_monotonic_s"]' in window
    assert "atomic_apply_deadline" in window
    assert "deadline=atomic_apply_deadline" in window
    assert "ASYNC_V21_ALL_EIGHT_APPLY_S" in window
    assert '"native_node_apply_swap"' in window

    coordinator = source[
        source.index("def _coordinate_native_safe_boundary("):
        source.index("def _liveness_heartbeat(")
    ]
    boundaries = coordinator.index("native-boundary-ready-")
    release = coordinator.index("transaction.release_apply(", boundaries)
    assert boundaries < release


def test_final_native_result_lifetime_covers_bounded_recovery_commit_phase():
    source = ROLE.read_text()
    owner = source[source.index("def _native_sharded_owner_reduce("):
                   source.index("def _native_manager_session_lifetime_s(")]

    assert "final_operation_deadline_s = remaining_s()" in owner
    assert "deadline_s=final_operation_deadline_s" in owner
    assert "generation_deadline_s=(" in owner
    assert "_native_post_result_lifetime_s(args)" in owner


def test_import_liveness_does_not_refresh_runtime_import_progress_deadline():
    text = ROLE.read_text()
    bootstrap = text[text.index("def _role_import_heartbeat"):
                     text.index("_IMPORT_HEARTBEAT = _role_import_heartbeat()")]
    assert "progress_started = time.time()" in bootstrap
    assert '"progress_time": progress_started' in bootstrap
    assert '"progress_time": now' not in bootstrap


def _control_processes(run, bulk, *, run_id, generations, initial=0, resume="", epoch=1):
    common = ["--run-dir", str(run), "--run-id", run_id, "--generations", str(generations),
              "--initial-generation", str(initial), "--local-steps", "40", "--deadline-s", "15",
              "--source-id", "seed", "--payload-id", "layout", "--code-id", "code",
              "--coordinator-epoch", str(epoch), "--control", "--bulk-root", str(bulk)]
    if resume:
        common += ["--resume-handoff", str(resume)]
    processes = [subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                                  env={**os.environ, "RESILIENT_E97_NODE_RANK": "0"})]
    for rank in range(6):
        processes.append(subprocess.Popen(
            [sys.executable, str(ROLE), "trainer", *common],
            env={**os.environ, "RESILIENT_E97_NODE_RANK": "0",
                 "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    # Seven cold Python/Torch processes can spend well over 40 seconds merely
    # importing on a contended Frontier login node.  The role's own 15-second
    # protocol deadline starts after import, so keep this subprocess harness
    # timeout distinct from (and comfortably outside) the protocol bound.
    assert [item.wait(timeout=90) for item in processes] == [0] * len(processes)


def test_eight_independent_trainers_advance_three_exact_generations(tmp_path):
    bulk_root = tmp_path.with_name(tmp_path.name + "-bulk")
    common = ["--run-dir", str(tmp_path), "--run-id", "control", "--generations", "3",
              "--local-steps", "40", "--deadline-s", "30", "--source-id", "seed-sha",
              "--payload-id", "layout-sha", "--code-id", "code-sha", "--control",
              "--bulk-root", str(bulk_root), "--local-quorum", "8"]
    manager = subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                               env={**os.environ, "RESILIENT_E97_NODE_RANK": "0"})
    trainers = []
    for rank in range(8):
        trainers.append(subprocess.Popen(
            [sys.executable, str(ROLE), "trainer", *common],
            env={**os.environ, "RESILIENT_E97_NODE_RANK": "0",
                 "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    assert manager.wait(timeout=60) == 0
    assert [item.wait(timeout=60) for item in trainers] == [0] * 8
    for rank in range(8):
        state = json.loads((bulk_root / "control/node-0/supervision" /
                            f"node-0-trainer-{rank}.json").read_text())
        assert state["generation"] == 3 and state["step"] == 120
        assert state["loss"] > 0
    checkpoint = torch.load(tmp_path / "checkpoints/generation-00000003.pt",
                            weights_only=True)
    reference = 0.0
    mailbox = bulk_root / "control/node-0/mailbox"
    for manifest_path in sorted((bulk_root / "control/node-0/control").glob(
            "node-0-generation-*.json")):
        members = json.loads(manifest_path.read_text())["members"]
        reference += sum((rank + 1) ** 2 for rank in members) / sum(rank + 1 for rank in members)
    # async-decoupled-v2.1-simple applies the exact eta_outer=1 update.
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(
        reference)
    handoff = json.loads((tmp_path / "handoff/generation-00000003.json").read_text())
    assert handoff["membership"] == ["node-0"]
    assert handoff["checkpoint_sha256"] == hashlib.sha256(
        Path(handoff["checkpoint"]).read_bytes()).hexdigest()
    assert not list(mailbox.glob("control-g*/trainer-*"))
    ownership = json.loads((bulk_root / "control/node-0/control/node-0-bulk-ownership.json").read_text())
    assert ownership["shared_run_dir_is_bulk_path"] is False
    assert 0 < ownership["high_water_bytes"] <= ownership["max_bytes"]
    assert ownership["post_release_bytes"] < ownership["high_water_bytes"]
    assert ownership["published_files"] <= 2 * 3
    assert len(list((mailbox / "aggregates").glob("*/manifest.json"))) <= 2
    assert {path.relative_to(tmp_path).parts[0] for path in tmp_path.rglob("*") if path.is_file()} \
        <= {"checkpoints", "handoff"}


def test_two_model_free_managers_exchange_without_collective(tmp_path):
    bulk_root = tmp_path.with_name(tmp_path.name + "-bulk")
    # The live runtime derives two owner endpoints at coordinator_port+1/+2.
    # Reserving only the coordinator port is racy on shared login nodes: a
    # healthy manager can fail before READY because either adjacent port is in
    # use. Verify the complete local fixture block before launching processes.
    port = None
    # Do not allocate the fixture from Linux's ephemeral client-port range.
    # Closing a port-0 probe lets any of the many concurrent subprocesses on a
    # shared login node immediately reuse it for an outbound connection before
    # the manager binds. A PID-distributed scan below 32768 avoids that kernel
    # allocator race while retaining the complete three-port probe.
    for attempt in range(256):
        probes = []
        try:
            candidate = 20_000 + ((os.getpid() * 3 + attempt * 3) % 10_000)
            first = socket.socket()
            first.bind(("127.0.0.1", candidate))
            probes.append(first)
            for offset in (1, 2):
                probe = socket.socket()
                probe.bind(("127.0.0.1", candidate + offset))
                probes.append(probe)
            port = candidate
            break
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
    assert port is not None, "no free contiguous local three-port block"
    common = ["--run-dir", str(tmp_path), "--run-id", "network", "--generations", "1",
              "--local-steps", "40", "--deadline-s", "60", "--source-id", "seed",
              "--payload-id", "layout", "--code-id", "code", "--control",
              "--node-count", "2", "--global-quorum", "2", "--coordinator-host", "127.0.0.1",
              "--coordinator-port", str(port), "--bulk-root", str(bulk_root)]
    processes = []
    for node in range(2):
        processes.append(subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                                          env={**os.environ, "RESILIENT_E97_NODE_RANK": str(node)}))
    for node in range(2):
        for rank in range(6):
            processes.append(subprocess.Popen(
                [sys.executable, str(ROLE), "trainer", *common],
                env={**os.environ, "RESILIENT_E97_NODE_RANK": str(node),
                     "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    # Fourteen cold Python/torch processes contend on shared login nodes. Keep
    # this local fixture bounded without conflating login load with the live
    # Frontier READY/K40 stage SLOs exercised by the launcher tests and jobs.
    assert [item.wait(timeout=120) for item in processes] == [0] * len(processes)
    manifests = list((tmp_path / "retained-evidence/pool-control").glob("*.jsonl"))
    assert len(manifests) == 1
    closes = [json.loads(line) for line in manifests[0].read_text().splitlines()]
    assert {item["worker_id"] for item in closes[-1]["frozen_identities"]} \
        == {"node-0", "node-1"}
    role_source = ROLE.read_text()
    assert all(word not in role_source for word in ("mpi4py", "TCPStore", "RCCL", "all_reduce"))
    assert "QuorumTransportServer" not in role_source
    assert "central_full_model_broker\": False" in role_source
    for node in range(2):
        generation = json.loads((bulk_root / f"network/node-{node}/control" /
                                 f"node-{node}-generation-00000000.json").read_text())
        assert generation["central_full_model_broker"] is False
        assert generation["p2p_bytes_sent"] > 0
        assert generation["redistribution_bytes"] > 0
        telemetry = [json.loads(line) for line in (
            bulk_root / f"network/node-{node}/telemetry/node-{node}-manager-pool.jsonl"
        ).read_text().splitlines()]
        stages = {item["stage"] for item in telemetry}
        assert {"ready", "k40_and_local_reduce", "freeze",
                "owner_transport_redistribution"} <= stages
        transport = next(item for item in telemetry
                         if item["stage"] == "owner_transport_redistribution")
        assert transport["within_slo"] is True
        assert transport["bytes_per_second"] > 0
        assert transport["released_bytes"] > 0


def test_fresh_process_restart_matches_uninterrupted_continuation(tmp_path):
    uninterrupted, restarted = tmp_path / "uninterrupted", tmp_path / "restarted"
    bulk_a = tmp_path.with_name(tmp_path.name + "-bulk-a")
    bulk_b = tmp_path.with_name(tmp_path.name + "-bulk-b")
    _control_processes(uninterrupted, bulk_a, run_id="control-a", generations=3)
    _control_processes(restarted, bulk_b, run_id="control-b", generations=2)
    resume = restarted / "handoff/generation-00000002.json"
    saved = json.loads(resume.read_text())
    loaded = torch.load(saved["checkpoint"], weights_only=True)
    assert loaded["generation"] == saved["generation"] == 2
    assert loaded["outer_update_state"] == saved["outer_update_state"]
    _control_processes(restarted, bulk_b, run_id="control-b", generations=1,
                       initial=2, resume=resume, epoch=2)
    expected = torch.load(uninterrupted / "checkpoints/generation-00000003.pt",
                          weights_only=True)
    actual = torch.load(restarted / "checkpoints/generation-00000003.pt", weights_only=True)
    assert torch.equal(actual["model_state_dict"]["weight"],
                       expected["model_state_dict"]["weight"])
    assert actual["optimizer_state_dict"] == expected["optimizer_state_dict"]
    assert actual["outer_update_state"] == expected["outer_update_state"]
    assert actual["step"] == expected["step"] == 120
    assert actual["coordinator_epoch"] == 2
    recovery_record = json.loads((
        bulk_b / "control-b/node-0/control/recovery/node-0-trainer-0.json"
    ).read_text())
    assert Path(recovery_record["checkpoint"]) == (
        restarted / "checkpoints/generation-00000003.pt").resolve()
    assert not (bulk_b /
        "control-b/node-0/recovery/node-0-trainer-0/generation-00000003.pt").exists()
    recovery = torch.load(recovery_record["checkpoint"], weights_only=True)
    assert recovery["identity"] == "node-0-trainer-0"
    assert recovery["generation"] == 3 and recovery["step"] == 120
    assert {"model_state_dict", "optimizer_state_dict", "outer_update_state",
            "membership", "fence", "async_chain"} <= recovery.keys()


def test_apply_identity_deadline_and_corruption_fail_closed(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 4096)
    fence = LocalFence("run", 0, 0, 1, "layout")
    with pytest.raises(TimeoutError, match="deadline"):
        spool.wait_aggregate(fence, deadline=0, expected_source_id="source")
    spool.publish_aggregate(fence, [0], [torch.tensor([2.])], weight=1, source_id="source")
    with pytest.raises(ValueError, match="identity"):
        spool.wait_aggregate(fence, deadline=1e20, expected_source_id="other")
    shard = next((tmp_path / "aggregates").rglob("*.data")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        spool.wait_aggregate(fence, deadline=1e20, expected_source_id="source")
    with pytest.raises(ValueError, match="count"):
        apply_delta({"x": torch.ones(1)}, (), eta_outer=1)
    with pytest.raises(ValueError, match="shared run"):
        assert_node_local_path(tmp_path / "live", tmp_path)


def test_delta_publication_and_apply_are_bounded_across_parameter_boundaries():
    delta = {"z": torch.arange(11, dtype=torch.float32),
             "a": torch.arange(5, dtype=torch.float32) + 20}
    shards = tuple(flatten_tensors(delta, chunk_elements=6))
    assert [shard.numel() for shard in shards] == [6, 6, 4]
    assert all(shard.dtype == torch.float32 and shard.device.type == "cpu" for shard in shards)
    base = {name: torch.zeros_like(value) for name, value in delta.items()}
    applied = apply_delta(base, shards, eta_outer=.5)
    for name in delta:
        assert torch.equal(applied[name], delta[name] * .5)


def test_safe_boundary_correction_ledger_subtracts_accepted_interval_once():
    anchor = {
        "a": torch.tensor([10.0, 1.0]),
        "z": torch.tensor([2.0]),
    }
    interval_start = {
        "a": torch.tensor([13.0, 0.0]),
        "z": torch.tensor([5.0]),
    }
    endpoint = {
        "a": torch.tensor([16.0, -1.0]),
        "z": torch.tensor([9.0]),
    }
    # eta * aggregate is the global anchor shift: [10,4] and [3].
    shards = (
        torch.tensor([20.0, 8.0, 6.0]),
    )

    updated, correction = apply_delta_with_correction_ledger(
        anchor, shards, eta_outer=.5,
        interval_start=interval_start,
        interval_endpoint=endpoint,
        accepted_own_interval=True,
        in_place=True,
    )

    torch.testing.assert_close(updated["a"], torch.tensor([20.0, 5.0]))
    torch.testing.assert_close(updated["z"], torch.tensor([5.0]))
    # ([10,4] - [3,-1]) and ([3] - [4]); the own interval is removed once.
    torch.testing.assert_close(correction["a"], torch.tensor([7.0, 5.0]))
    torch.testing.assert_close(correction["z"], torch.tensor([-1.0]))


def test_safe_boundary_correction_ledger_preserves_unaccepted_displacement():
    anchor = {"weight": torch.tensor([2.0])}
    interval_start = {"weight": torch.tensor([9.0])}
    endpoint = {"weight": torch.tensor([14.0])}

    updated, correction = apply_delta_with_correction_ledger(
        anchor, (torch.tensor([8.0]),), eta_outer=.5,
        interval_start=interval_start,
        interval_endpoint=endpoint,
        accepted_own_interval=False,
        in_place=True,
    )

    torch.testing.assert_close(updated["weight"], torch.tensor([6.0]))
    # No accepted identity means C_i=0; all speculative local work survives.
    torch.testing.assert_close(correction["weight"], torch.tensor([4.0]))
    torch.testing.assert_close(
        endpoint["weight"] + correction["weight"], torch.tensor([18.0]))


def test_canonical_cold_start_outer_policy_and_immutable_reloadable_handoff(tmp_path):
    canonical = json.loads(
        (Path(__file__).parents[1] / "configs/frontier/e97_async_256.yaml").read_text()
    )["seed"]
    seed = {"step": canonical["step"], "sha256": canonical["sha256"]}
    with pytest.raises(ValueError, match="approved initialization"):
        outer_state_migration(seed, policy="")
    with pytest.raises(ValueError, match="approved seed identity"):
        outer_state_migration(
            seed, policy="initialize-from-approved-config",
            approved_config={"algorithm": "weighted-mean", "eta_outer": 1.0})
    migration = outer_state_migration(
        seed, policy="initialize-from-approved-config",
        approved_config={"algorithm": "weighted-mean", "eta_outer": 1.0},
        approved_seed=seed)
    assert migration["status"] == "initialized_not_restored"
    checkpoint = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": {"x": torch.ones(1)},
                "optimizer_state_dict": {"state": {}},
                "outer_update_state": migration["state"], "step": 40,
                "generation": 10, "run_id": "run", "source_id": "source",
                "payload_id": "payload", "coordinator_epoch": 2}, checkpoint)
    fence = LocalFence("run", 9, 0, 2, "payload")
    manifest = finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=10, step=40,
        async_chain=["canonical-step-2300930"], membership=range(6), fence=fence,
        source_id="source", code_id="code", outer_update_state=migration["state"],
        migration=migration)
    payload = json.loads(manifest.read_text())
    assert payload["contains"] == ["model", "inner_optimizer"]
    assert payload["outer_state_migration"]["status"] == "initialized_not_restored"
    assert torch.load(payload["checkpoint"], weights_only=True)["step"] == 40
    with pytest.raises(FileExistsError, match="immutable"):
        finalize_checkpoint(tmp_path, checkpoint, run_id="run", generation=10, step=40,
                            async_chain=[], membership=range(6), fence=fence,
                            source_id="source", code_id="code",
                            outer_update_state=migration["state"],
                            migration=migration)


def test_final_canonical_seed_can_initialize_approved_outer_policy():
    canonical = json.loads(
        (Path(__file__).parents[1] / "configs/frontier/e97_async_256.yaml").read_text()
    )["seed"]
    migration = outer_state_migration(
        {
            "step": canonical["step"],
            "sha256": canonical["sha256"],
            "outer_update_state": None,
        },
        policy="initialize-from-approved-config",
        approved_config={"algorithm": "weighted-mean", "eta_outer": 1.0},
        approved_seed={
            "step": canonical["step"],
            "sha256": canonical["sha256"],
        },
    )
    assert migration["source_step"] == 2300930
    assert migration["source_sha256"] == (
        "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2")


def test_fenced_atomic_global_commit_and_newer_allocation_restart(tmp_path):
    from ndm.manifest_peer_control import ManifestPeerAuthority

    authority = ManifestPeerAuthority(tmp_path)
    old = authority.claim(
        run_id="run", allocation_id="job-a", incarnation="a", fence=1,
        protocol_id="pool-v1", config_id="cfg")
    outer_one = {
        "mode": "delta_sgd", "eta_outer": 1.0,
        "step": 1, "accepted_tokens": 17,
    }
    checkpoint = tmp_path / "checkpoints/generation-1.pt"
    checkpoint.parent.mkdir()
    torch.save({"model_state_dict": {"x": torch.tensor([3.0])},
                "optimizer_state_dict": {"state": {}},
                "outer_update_state": outer_one,
                "step": 40, "generation": 1, "run_id": "run", "source_id": "seed",
                "payload_id": "layout", "coordinator_epoch": old.fence,
                "accepted_tokens": 17}, checkpoint)
    manifest = finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=1, step=40,
        async_chain=["seed"], membership=["node-a:inc-a"],
        fence=LocalFence("run", 0, 0, old.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state=outer_one,
        migration={"status": "restored"}, accepted_tokens=17,
        generation_identity={"run_id": "run", "generation": 0,
                             "attempt": 0, "fence": old.fence},
        digests={
            "layout": "layout", "code": "code",
            "result_root": "11" * 32,
            "previous_result_root": "00" * 32,
        },
        peer_authority=authority, allocation_claim=old)
    latest = authority.current_commit(old)
    assert latest.generation == 1 and latest.accepted_tokens == 17
    assert manifest.name == "generation-00000001-fence-00000001.json"
    assert json.loads(manifest.read_text())["membership"] == ["node-a:inc-a"]
    assert finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=1, step=40,
        async_chain=["seed"], membership=["node-a:inc-a"],
        fence=LocalFence("run", 0, 0, old.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state=outer_one,
        migration={"status": "restored"}, accepted_tokens=17,
        generation_identity={"run_id": "run", "generation": 0,
                             "attempt": 0, "fence": old.fence},
        digests={
            "layout": "layout", "code": "code",
            "result_root": "11" * 32,
            "previous_result_root": "00" * 32,
        },
        peer_authority=authority, allocation_claim=old) == manifest

    new = authority.claim(
        run_id="run", allocation_id="job-b", incarnation="b", fence=2,
        protocol_id="pool-v1", config_id="cfg")
    assert new.fence == old.fence + 1
    assert new.base_commit_digest == latest.receipt_digest
    loaded = torch.load(json.loads(manifest.read_text())["checkpoint"], weights_only=True)
    assert loaded["accepted_tokens"] == 17
    stale_checkpoint = tmp_path / "checkpoints/generation-2-stale.pt"
    torch.save({**loaded, "generation": 2, "step": 80}, stale_checkpoint)
    with pytest.raises(FenceRejected):
        finalize_checkpoint(
            tmp_path, stale_checkpoint, run_id="run", generation=2, step=80,
            async_chain=[], membership=[], fence=LocalFence("run", 1, 0, old.fence, "layout"),
            source_id="seed", code_id="code",
            outer_update_state=outer_one, migration={},
            accepted_tokens=17,
            peer_authority=authority, allocation_claim=old)
    assert not (tmp_path / "handoff/generation-00000002.json").exists()

    outer_two = {
        "mode": "delta_sgd", "eta_outer": 1.0,
        "step": 2, "accepted_tokens": 25,
    }
    fresh_checkpoint = tmp_path / "checkpoints/generation-2-fresh.pt"
    torch.save({**loaded, "generation": 2, "step": 80,
                "coordinator_epoch": new.fence, "accepted_tokens": 25,
                "outer_update_state": outer_two},
               fresh_checkpoint)
    continued = finalize_checkpoint(
        tmp_path, fresh_checkpoint, run_id="run", generation=2, step=80,
        async_chain=[str(manifest)], membership=["node-a:inc-b"],
        fence=LocalFence("run", 1, 0, new.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state=outer_two,
        migration={"status": "restored"}, accepted_tokens=25,
        generation_identity={"run_id": "run", "generation": 1,
                             "attempt": 0, "fence": new.fence},
        digests={
            "result_root": "22" * 32,
            "previous_result_root": "11" * 32,
        },
        peer_authority=authority, allocation_claim=new)
    assert continued.name == "generation-00000002-fence-00000002.json"
    recovered = authority.current_commit(new, verify_checkpoint=True)
    assert recovered.generation == 2
    assert recovered.accepted_tokens == 25
    assert recovered.previous_receipt_digest == latest.receipt_digest
    assert recovered.manifest_path == continued.resolve()
