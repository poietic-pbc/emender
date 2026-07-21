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
from ndm.resilient_e97_runtime import (apply_delta, finalize_checkpoint,
                                       assert_node_local_path, flatten_tensors,
                                       outer_state_migration)
from ndm.fenced_admission import FenceRejected, SQLiteFencedControlStore
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

    class Store:
        def assert_current(self, lease):
            assert lease == "lease"

        def read_publication(self, run_id, namespace, key):
            assert (run_id, namespace, key) == ("run", "latest", "authoritative")
            return {"generation": 6, "fence": 4,
                    "manifest_sha256": digest}

    args = SimpleNamespace(
        resume_handoff=str(configured), run_id="run",
        initial_generation=5, generations=3,
    )
    assert role._authoritative_trainer_resume_handoff(
        tmp_path, args, (Store(), "lease")) == committed.resolve()

    (handoff_root / "latest.json").write_text(json.dumps({
        "generation": 6, "fence": 4, "manifest": str(committed),
        "manifest_sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="not authoritative"):
        role._authoritative_trainer_resume_handoff(
            tmp_path, args, (Store(), "lease"))


def test_native_apply_lanes_receive_fresh_post_exchange_deadline():
    source = ROLE.read_text()
    trainer = source[source.index("def trainer(args)"):]
    reset = "apply_lane_deadline = (\n                time.monotonic() + min(args.deadline_s, 180.0))"
    assert reset in trainer
    assert "deadline=apply_lane_deadline" in trainer


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


def test_native_trainer_apply_lanes_are_serialized_by_local_rank(tmp_path):
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
    assert "deadline=apply_lane_deadline" in trainer[lane:apply]


def test_native_apply_lane_excludes_durable_recovery_checkpoint_io():
    """A slow local checkpoint must not hold the shared-result read lane.

    Intermediate generations persist one trainer recovery checkpoint per GPU.
    Charging that disk write to the next rank's native result-view lane makes
    eight otherwise bounded applies exceed the 60 second APPLY budget.
    """
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args)"):]
    wait = trainer.index("_wait_for_native_apply_lane(")
    apply = trainer.index("state = apply_delta(", wait)
    lane_credit = trainer.index("native-result-applied-", apply)
    recovery_save = trainer.index("torch.save(", lane_credit)
    durable_receipt = trainer.index("native-applied-", recovery_save)

    assert wait < apply < lane_credit < recovery_save < durable_receipt
    timer_reset = trainer.rfind("trainer_apply_started = time.monotonic()", wait, apply)
    assert timer_reset > wait, "the APPLY SLO must measure apply, not lane waiting"


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
    rejected = trainer.index("if not pipeline.publish_committed(", publish)
    safe_boundary = trainer.index("pipeline.take_at_boundary(", rejected)
    telemetry = trainer.index('"native_generation_pipeline"', safe_boundary)

    assert loop < started < stop < publish < rejected < safe_boundary < telemetry
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


def test_manager_uses_exchange_commit_bound_for_all_recovery_receipts():
    """Eight durable checkpoints are an aggregate commit phase, not one apply."""
    source = ROLE.read_text()
    manager = source[source.index("def _native_manager(args)"):
                     source.index("def manager(args)")]
    receipt_loop = manager.index("for rank in range(args.local_quorum):")
    commit = manager.index("session.commit(", receipt_loop)
    window = manager[receipt_loop - 300:commit]

    assert "recovery_deadline" in window
    assert "min(args.deadline_s, 180.0)" in window
    assert "deadline=recovery_deadline" in window
    assert "apply_deadline" not in window


def test_final_native_result_lifetime_covers_bounded_recovery_commit_phase():
    source = ROLE.read_text()
    owner = source[source.index("def _native_sharded_owner_reduce("):
                   source.index("def _native_manager_session_lifetime_s(")]

    assert "final_operation_deadline_s = remaining_s()" in owner
    assert "deadline_s=final_operation_deadline_s" in owner
    assert "generation_deadline_s=(" in owner
    assert "final_operation_deadline_s + min(float(args.deadline_s), 180.0)" in owner


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
    assert [item.wait(timeout=40) for item in processes] == [0] * len(processes)


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
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(reference)
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


def test_pinned_cold_start_outer_policy_and_immutable_reloadable_handoff(tmp_path):
    from ndm.resilient_e97_runtime import PINNED_STEP_1525000_SHA256
    seed = {"step": 1525000, "sha256": PINNED_STEP_1525000_SHA256}
    with pytest.raises(ValueError, match="approved initialization"):
        outer_state_migration(seed, policy="")
    migration = outer_state_migration(
        seed, policy="initialize-from-approved-config",
        approved_config={"algorithm": "weighted-mean", "eta_outer": 1.0})
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
        async_chain=["pinned-step-1525000"], membership=range(6), fence=fence,
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


def test_fenced_atomic_global_commit_and_newer_allocation_restart(tmp_path):
    now = [100.0]
    store = SQLiteFencedControlStore(tmp_path / "pool.sqlite", clock=lambda: now[0])
    old = store.acquire(run_id="run", allocation_id="job-a", incarnation="a",
                        protocol_id="pool-v1", config_id="cfg", ttl_s=10)
    checkpoint = tmp_path / "generation-1.pt"
    torch.save({"model_state_dict": {"x": torch.tensor([3.0])},
                "optimizer_state_dict": {"state": {}},
                "outer_update_state": {"algorithm": "weighted-mean"},
                "step": 40, "generation": 1, "run_id": "run", "source_id": "seed",
                "payload_id": "layout", "coordinator_epoch": old.fence,
                "accepted_tokens": 17}, checkpoint)
    manifest = finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=1, step=40,
        async_chain=["seed"], membership=["node-a:inc-a"],
        fence=LocalFence("run", 0, 0, old.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state={"algorithm": "weighted-mean"},
        migration={"status": "restored"}, accepted_tokens=17,
        generation_identity={"run_id": "run", "generation": 0,
                             "attempt": 0, "fence": old.fence},
        digests={"layout": "layout", "code": "code"},
        control_store=store, allocation_lease=old)
    latest = store.read_publication("run", "latest", "authoritative")
    assert latest["generation"] == 1 and latest["accepted_tokens"] == 17
    assert manifest.name == "generation-00000001-fence-00000001.json"
    assert json.loads(manifest.read_text())["membership"] == ["node-a:inc-a"]
    assert finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=1, step=40,
        async_chain=["seed"], membership=["node-a:inc-a"],
        fence=LocalFence("run", 0, 0, old.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state={"algorithm": "weighted-mean"},
        migration={"status": "restored"}, accepted_tokens=17,
        generation_identity={"run_id": "run", "generation": 0,
                             "attempt": 0, "fence": old.fence},
        digests={"layout": "layout", "code": "code"},
        control_store=store, allocation_lease=old) == manifest

    now[0] = old.expires_at
    new = store.acquire(run_id="run", allocation_id="job-b", incarnation="b",
                        protocol_id="pool-v1", config_id="cfg", ttl_s=10)
    assert new.fence == old.fence + 1
    loaded = torch.load(json.loads(manifest.read_text())["checkpoint"], weights_only=True)
    assert loaded["accepted_tokens"] == 17
    stale_checkpoint = tmp_path / "generation-2.pt"
    torch.save({**loaded, "generation": 2, "step": 80}, stale_checkpoint)
    with pytest.raises(FenceRejected):
        finalize_checkpoint(
            tmp_path, stale_checkpoint, run_id="run", generation=2, step=80,
            async_chain=[], membership=[], fence=LocalFence("run", 1, 0, old.fence, "layout"),
            source_id="seed", code_id="code",
            outer_update_state={"algorithm": "weighted-mean"}, migration={},
            accepted_tokens=17, control_store=store, allocation_lease=old)
    assert not (tmp_path / "handoff/generation-00000002.json").exists()

    fresh_checkpoint = tmp_path / "generation-2-fresh.pt"
    torch.save({**loaded, "generation": 2, "step": 80,
                "coordinator_epoch": new.fence, "accepted_tokens": 25},
               fresh_checkpoint)
    continued = finalize_checkpoint(
        tmp_path, fresh_checkpoint, run_id="run", generation=2, step=80,
        async_chain=[str(manifest)], membership=["node-a:inc-b"],
        fence=LocalFence("run", 1, 0, new.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state={"algorithm": "weighted-mean"},
        migration={"status": "restored"}, accepted_tokens=25,
        generation_identity={"run_id": "run", "generation": 1,
                             "attempt": 0, "fence": new.fence},
        control_store=store, allocation_lease=new)
    assert continued.name == "generation-00000002-fence-00000002.json"
    assert store.read_publication("run", "latest", "authoritative") == {
        "accepted_tokens": 25, "fence": new.fence, "generation": 2,
        "manifest": str(continued.resolve()),
        "manifest_sha256": hashlib.sha256(continued.read_bytes()).hexdigest(),
    }
