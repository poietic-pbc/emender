from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import threading
import time

import numpy as np
import pytest
import torch

from ndm.native_artifacts import NATIVE_TEST
from ndm.native_dataplane import (
    Client, Command, DType, NativeLibrary, Role, create_memfd, seal_memfd,
)
from ndm.native_e97_runtime import (
    GenerationMetadata, NativeTrainerDataPlane, atomic_metadata,
    exact_weighted_reference, wait_metadata,
)
from ndm.native_pool_runtime import NativeManagerSession, NativeTrainerHandoff
from ndm.native_transport import NativeTransport, NativeTransportLibrary
from ndm.resilient_pool_runtime import OwnerEndpoint


ROOT = Path(__file__).resolve().parents[1]
BUILD_MANIFEST = ROOT / "build/native-resilient-dataplane/native-artifacts.json"


def test_wait_metadata_tolerates_stale_atomic_generation_until_publication(tmp_path):
    latest = tmp_path / "latest.json"
    atomic_metadata(latest, {"generation": 1, "fence": 9})

    def publish_next_generation():
        time.sleep(.05)
        atomic_metadata(latest, {"generation": 2, "fence": 9})

    publisher = threading.Thread(target=publish_next_generation)
    publisher.start()
    try:
        assert wait_metadata(
            latest, deadline=time.monotonic() + 1,
            expected={"generation": 2, "fence": 9},
        )["generation"] == 2
    finally:
        publisher.join()


def test_native_service_validates_dense_submissions_without_global_serialization():
    """Large sealed buffers may be scanned concurrently across trainer RPCs."""
    source = (ROOT / "src/native_resilient_dataplane/src/ndp.cpp").read_text()
    submit = source[source.index("int Service::submit("):
                    source.index("void Service::release_submissions()")]
    unlock = submit.index("lock.unlock();")
    checksum = submit.index("Sha256::digest")
    relock = submit.index("lock.lock();")
    assert "std::unique_lock<std::mutex> lock(mutex_);" in submit
    assert unlock < checksum < relock
    assert "buffer_found->second != validation_buffer" in submit
    assert "generation_ != validation_generation" in submit


def test_native_service_reduces_once_validated_sealed_sources_in_parallel():
    """Admission is the sole hash boundary; element ranges reduce in parallel."""
    source = (ROOT / "src/native_resilient_dataplane/src/ndp.cpp").read_text()
    submit = source[source.index("int Service::submit("):
                    source.index("void Service::release_submissions()")]
    reduce_local = source[source.index("int Service::reduce_local("):
                          source.index("bool local_spool_path(")]
    assert "validation_workers" in submit
    assert "finite_workers" in submit
    assert "Sha256::digest(mapping.data, mapping.bytes)" in submit
    assert "actual != receipt.digest" in submit
    assert "nonfinite.load(std::memory_order_relaxed)" in submit
    assert "Re-hashing here would add a redundant full-layout pass" in reduce_local
    assert "Sha256::digest(mapping.data, mapping.bytes)" not in reduce_local
    assert "std::thread::hardware_concurrency()" in reduce_local
    assert "parallel_reduction_workers" in reduce_local
    assert "remainder_elements = elements % parallel_reduction_workers" in reduce_local
    assert "index < end" in reduce_local
    assert "index != end" not in reduce_local
    parallel = reduce_local[reduce_local.index("const std::size_t parallel_reduction_workers"):]
    assert "for (std::size_t source_index = 0;" in parallel
    assert parallel.index("for (std::uint64_t index = begin;") < parallel.index(
        "for (std::size_t source_index = 0;")
    assert "std::atomic<bool> nonfinite" in reduce_local


def test_native_service_dense_sha_uses_optimized_crypto_provider():
    """Full-layout admission/root hashes must not use the scalar fallback."""
    sha = (ROOT / "src/native_resilient_dataplane/src/sha256.hpp").read_text()
    cmake = (ROOT / "src/native_resilient_dataplane/CMakeLists.txt").read_text()

    assert "#include <openssl/evp.h>" in sha
    assert "EVP_DigestUpdate" in sha
    assert "EVP_sha256()" in sha
    assert "find_package(OpenSSL REQUIRED COMPONENTS Crypto)" in cmake
    assert cmake.count("OpenSSL::Crypto") >= 2


def test_native_final_projection_fuses_divide_and_f32_write_in_parallel():
    """Final apply keeps exact element semantics without two serial passes."""
    source = (ROOT / "src/native_resilient_dataplane/src/ndp.cpp").read_text()
    project = source[source.index("int Service::project_result("):
                     source.index("int Service::control(")]
    assert "projection_workers" in project
    assert "numerator_[index] = divided" in project
    assert "projected[index] = static_cast<float>(divided)" in project
    assert project.index("numerator_[index] = divided") < project.index(
        "projected[index] = static_cast<float>(divided)")
    assert "for (double& value : numerator_)" not in project


def test_native_manager_imports_owner_results_on_independent_rpc_sessions(monkeypatch):
    """Attempt-2 source validation overlaps and owns clients through freeze."""
    session = object.__new__(NativeManagerSession)
    session._generation_installed = True
    session._frozen = False
    session.run_id = "run"
    session.fence_epoch = 19

    class Local:
        native = object()
        total_elements = 8
        layout_digest = bytes.fromhex("11" * 32)
        generation = 5
        attempt = 2
        owner_epoch = 1
        generation_deadline_ns = time.time_ns() + 10_000_000_000
        base_digest = bytes.fromhex("22" * 32)
        plan_digest = bytes.fromhex("33" * 32)

    session.local = Local()
    validation_barrier = threading.Barrier(2, timeout=2)
    opened, closed = [], []

    class FakeBuffer:
        def __enter__(self):
            return self

        def __exit__(self, *_ignored):
            return None

    class FakeOperation:
        def __init__(self, worker):
            self.worker, self.closed = worker, False

        def close(self):
            self.closed = True

    class FakeClient:
        def __init__(self, worker):
            self.worker = worker
            opened.append(worker)

        def attach_generation(self, **metadata):
            assert metadata["attempt"] == 2
            assert metadata["source_dtype"] is DType.F64

        def register_memfd(self, fd, *, length, handle_generation):
            assert fd in {41, 42}
            assert length == 64
            assert handle_generation == 5
            return FakeBuffer()

        def submit(self, _buffer, *, trainer_key, trainer_incarnation,
                   submission_seq, weight, source_dtype, source_sha256,
                   deadline_s):
            assert trainer_key == self.worker
            assert trainer_incarnation.endswith("-boot")
            assert submission_seq in {0, 1} and weight > 0
            assert source_sha256 in {bytes.fromhex("44" * 32),
                                     bytes.fromhex("55" * 32)}
            assert source_dtype is DType.F64 and deadline_s == 10
            validation_barrier.wait()
            return FakeOperation(self.worker)

        def close(self):
            closed.append(self.worker)

    monkeypatch.setattr(
        Client, "open",
        lambda **values: FakeClient(values["worker_key"].split(":")[-2]))
    sources = (
        (41, "node-0", "node-0-boot", 0, 17, bytes.fromhex("44" * 32)),
        (42, "node-1", "node-1-boot", 1, 23, bytes.fromhex("55" * 32)),
    )
    with session.import_reduction_sources(
            sources, source_dtype=DType.F64, deadline_s=10) as operations:
        assert [operation.worker for operation in operations] == ["node-0", "node-1"]
        assert not any(operation.closed for operation in operations)
        assert closed == []
    assert sorted(opened) == ["node-0", "node-1"]
    assert closed == ["node-0", "node-1"]
    assert all(operation.closed for operation in operations)


def test_native_manager_imports_thirty_two_frozen_owner_results(monkeypatch):
    """The ordered 32-node rung must not inherit the old 16-source gate."""
    session = object.__new__(NativeManagerSession)
    session._generation_installed = True
    session._frozen = False
    session.run_id = "run-32n"
    session.fence_epoch = 17

    class Local:
        native = object()
        total_elements = 1
        layout_digest = bytes.fromhex("11" * 32)
        generation = 14
        attempt = 2
        owner_epoch = 1
        generation_deadline_ns = time.time_ns() + 10_000_000_000
        base_digest = bytes.fromhex("22" * 32)
        plan_digest = bytes.fromhex("33" * 32)

    session.local = Local()
    opened, closed = [], []

    class FakeBuffer:
        def __enter__(self):
            return self

        def __exit__(self, *_ignored):
            return None

    class FakeOperation:
        def __init__(self, worker):
            self.worker = worker

        def close(self):
            return None

    class FakeClient:
        def __init__(self, worker):
            self.worker = worker
            opened.append(worker)

        def attach_generation(self, **metadata):
            assert metadata["attempt"] == 2

        def register_memfd(self, fd, *, length, handle_generation):
            assert length == 8 and handle_generation == 14
            return FakeBuffer()

        def submit(self, _buffer, **metadata):
            assert metadata["trainer_key"] == self.worker
            return FakeOperation(self.worker)

        def close(self):
            closed.append(self.worker)

    monkeypatch.setattr(
        Client, "open",
        lambda **values: FakeClient(values["worker_key"].split(":")[-2]))
    sources = tuple(
        (100 + index, f"node-{index}", f"node-{index}-boot", index, 1,
         bytes([index]) * 32)
        for index in range(32))
    with session.import_reduction_sources(
            sources, source_dtype=DType.F64, deadline_s=10) as operations:
        assert len(operations) == 32
        assert {operation.worker for operation in operations} == {
            f"node-{index}" for index in range(32)}
    assert sorted(opened) == sorted(f"node-{index}" for index in range(32))
    assert sorted(closed) == sorted(opened)


def test_native_manager_binds_both_abis_before_ready_installs_routes_and_drains(tmp_path):
    manifest = json.loads(BUILD_MANIFEST.read_text())
    transport_library = (
        BUILD_MANIFEST.parent / manifest["artifacts"]["transport_library"]["path"])
    session = NativeManagerSession.start(
        backend=NATIVE_TEST, run_id="run", fence_epoch=19,
        worker_id="node-0", incarnation="node-0-boot", host="127.0.0.1",
        build_manifest=BUILD_MANIFEST, gate_json=None, source_root=ROOT,
        production=False, full_layout=False, deadline_s=20,
        telemetry_path=tmp_path / "native.jsonl", payload_max=4096,
        resident_limit_bytes=1 << 20,
    )
    session.install_generation(
        total_elements=8, generation=4, payload_max=64, deadline_s=10)
    with session.allocate_trainer_buffer(deadline_s=10) as buffer:
        with buffer.mapped("float32", write=True) as target:
            target[:] = np.arange(8, dtype=np.float32)
        buffer.seal()
        submission = session.submit_local(
            buffer, trainer_key="trainer-0", trainer_incarnation="trainer-0-boot",
            submission_seq=1, weight=17, deadline_s=10)
    freeze = session.freeze(deadline_s=10)
    result_operation, result = session.finalize_redistribution(deadline_s=10)
    with result:
        with result.mapped("float32") as actual:
            assert np.array_equal(actual, np.arange(8, dtype=np.float32))
        with pytest.raises(RuntimeError, match="durable Python publication"):
            session.commit(
                publication_manifest=tmp_path / "missing.json",
                authoritative_latest={}, deadline_s=10)
        proposal = session.checkpoint_proposal(
            tmp_path / "checkpoint-proposal.json", result,
            publisher="node-0-trainer-0", metadata={"accepted": ["node-0"]})
        assert json.loads(proposal.read_text())["global_weight"] == 17
        original_fence = result.fence_epoch
        result.fence_epoch -= 1
        with pytest.raises(RuntimeError, match="result identity/fence"):
            session.checkpoint_proposal(
                tmp_path / "stale-checkpoint-proposal.json", result,
                publisher="node-0-trainer-0")
        result.fence_epoch = original_fence
        publication = tmp_path / "generation-00000004-fence-00000019.json"
        publication.write_text(json.dumps({
            "finalized": True, "run_id": "run", "generation": 4,
            "fence": {"coordinator_epoch": 19},
            "attempt": result.attempt,
            "layout_digest": result.layout_digest.hex(),
            "base_digest": result.base_digest.hex(),
            "result_root": result.result_root.hex(),
            "global_weight": result.global_weight,
            "result_bytes": result.length,
        }, sort_keys=True) + "\n")
        publication_digest = __import__("hashlib").sha256(
            publication.read_bytes()).hexdigest()
        with pytest.raises(RuntimeError, match="authoritative latest CAS"):
            session.commit(
                publication_manifest=publication,
                authoritative_latest={"generation": 4, "fence": 18}, deadline_s=10)
        mismatched_publication = tmp_path / "mismatched-result.json"
        mismatched = json.loads(publication.read_text())
        mismatched["result_root"] = "00" * 32
        mismatched_publication.write_text(json.dumps(mismatched, sort_keys=True) + "\n")
        mismatched_digest = __import__("hashlib").sha256(
            mismatched_publication.read_bytes()).hexdigest()
        with pytest.raises(RuntimeError, match="result identity"):
            session.commit(
                publication_manifest=mismatched_publication,
                authoritative_latest={
                    "generation": 4, "fence": 19,
                    "manifest": str(mismatched_publication.resolve()),
                    "manifest_sha256": mismatched_digest,
                }, deadline_s=10)
        session.commit(
            publication_manifest=publication,
            authoritative_latest={
                "generation": 4, "fence": 19,
                "manifest": str(publication.resolve()),
                "manifest_sha256": publication_digest,
            }, deadline_s=10)
    result_operation.close()
    freeze.close()
    submission.close()
    expiry = time.time_ns() + 10_000_000_000
    with NativeTransport.open(
            library=NativeTransportLibrary(transport_library),
            provider="tcp;ofi_rxm", production=False, bind_node="127.0.0.1",
            deadline_s=20, payload_max=4096, tx_slots=1, rx_slots=1,
            resident_limit_bytes=1 << 20) as peer:
        record = peer.bind(
            run_key="run", fence_epoch=19, worker_key="node-1",
            incarnation="node-1-boot", endpoint_epoch=1,
            expires_unix_ns=expiry,
        )
        endpoint = OwnerEndpoint(
            "node-1", "node-1-boot", "127.0.0.1", 0, NATIVE_TEST,
            record.encoded.hex(), record.provider, record.endpoint_epoch,
            record.expires_unix_ns, manifest["bundle_sha256"],
        )
        routes = session.install_routes((session.owner_endpoint, endpoint))
        assert routes.keys() == {"node-1"}
        assert session.telemetry().live_peers == 1

        ready = session.write_readiness(tmp_path / "ready.json")
        value = json.loads(ready.read_text())
        assert value["schema"] == "emender-native-manager-ready-v1"
        assert value["artifact_bundle_sha256"] == manifest["bundle_sha256"]
        assert value["provider"] == "tcp;ofi_rxm"
        assert value["python_dense_socket_bytes"] == value["trainer_spool_bytes"] == 0

    final = session.close("allocation_term_handoff")
    assert final.terminal_reason == "allocation_term_handoff"
    assert final.local["shared_bytes_current"] == 0
    assert final.transport["in_flight_bytes"] == 0
    assert final.transport["retained_bytes"] == 0

    # A manager process is disposable.  Its allocation-handoff close must
    # detach only this client/endpoint so a new incarnation can own the same
    # persistent service.  The node supervisor, not a manager child, owns the
    # eventual service-wide TERM/drain.
    replacement = NativeManagerSession.start(
        backend=NATIVE_TEST, run_id="run", fence_epoch=19,
        worker_id="node-0", incarnation="node-0-rejoined", host="127.0.0.1",
        build_manifest=BUILD_MANIFEST, gate_json=None, source_root=ROOT,
        production=False, full_layout=False, deadline_s=20,
        telemetry_path=tmp_path / "native-rejoined.jsonl", payload_max=4096,
        resident_limit_bytes=1 << 20,
    )
    replacement.install_generation(
        total_elements=8, generation=5, payload_max=64, deadline_s=10)
    replacement.abort(deadline_s=10)
    replacement.close("normal")


def test_native_manager_switches_compact_owner_layout_then_restores_full_attempt(tmp_path):
    """The 8-node owner path may compact, redistribute, and restore in one service."""
    import hashlib

    session = NativeManagerSession.start(
        backend=NATIVE_TEST, run_id="compact-owner", fence_epoch=29,
        worker_id="node-0", incarnation="node-0-boot", host="127.0.0.1",
        build_manifest=BUILD_MANIFEST, gate_json=None, source_root=ROOT,
        production=False, full_layout=False, deadline_s=20,
        telemetry_path=tmp_path / "native-compact.jsonl", payload_max=4096,
        resident_limit_bytes=1 << 20,
    )
    base_digest = bytes.fromhex("22" * 32)
    plan_digest = bytes.fromhex("33" * 32)

    def sealed_source(name: str, values: np.ndarray) -> tuple[int, bytes]:
        payload = values.tobytes()
        fd = create_memfd(name, allow_sealing=True)
        assert os.write(fd, payload) == len(payload)
        seal_memfd(fd)
        return fd, hashlib.sha256(payload).digest()

    try:
        full_layout = session.install_generation(
            total_elements=8, generation=6, attempt=1, source_dtype=DType.F32,
            payload_max=64, base_digest=base_digest, plan_digest=plan_digest,
            deadline_s=10)
        session.abort(deadline_s=10)

        compact_layout = session.install_generation(
            total_elements=3, generation=6, attempt=2, source_dtype=DType.F64,
            payload_max=64, base_digest=base_digest, plan_digest=plan_digest,
            deadline_s=10)
        assert compact_layout != full_layout
        compact_sources = (
            sealed_source("compact-a", np.array([3., 6., 9.], dtype=np.float64)),
            sealed_source("compact-b", np.array([7., 14., 21.], dtype=np.float64)),
        )
        try:
            sources = tuple(
                (fd, f"node-{index}", f"node-{index}-boot", index,
                 (3, 7)[index], digest)
                for index, (fd, digest) in enumerate(compact_sources))
            with session.import_reduction_sources(
                    sources, source_dtype=DType.F64, deadline_s=10):
                owner_freeze = session.freeze(deadline_s=10)
                owner_operation, owner_result = session.finalize_redistribution(
                    deadline_s=10)
            with owner_result.mapped("float32") as actual:
                assert np.array_equal(actual, np.array([1., 2., 3.], np.float32))
            assert owner_result.layout_digest == compact_layout
            assert owner_result.attempt == 2 and owner_result.global_weight == 10
        finally:
            for fd, _digest in compact_sources:
                os.close(fd)
        session.abort(deadline_s=10)
        owner_result.close(); owner_operation.close(); owner_freeze.close()

        assembled = np.arange(8, dtype=np.float32)
        aggregate_fd, aggregate_digest = sealed_source("assembled", assembled)
        try:
            restored_layout = session.install_generation(
                total_elements=8, generation=6, attempt=1,
                source_dtype=DType.F32, payload_max=64,
                base_digest=base_digest, plan_digest=plan_digest,
                deadline_s=10)
            assert restored_layout == full_layout
            with session.import_reduction_sources(
                    ((aggregate_fd, "assembled", "assembled-boot", 0, 10,
                      aggregate_digest),),
                    source_dtype=DType.F32, deadline_s=10):
                bridge_freeze = session.freeze(deadline_s=10)
                bridge_operation, bridge_result = session.finalize_redistribution(
                    deadline_s=10)
        finally:
            os.close(aggregate_fd)
        with bridge_result.mapped("float64") as actual:
            assert np.array_equal(actual, assembled.astype(np.float64) * 10)
        bridge_fd = os.dup(bridge_result.fd)
        bridge_digest = bridge_result.sha256()
        session.abort(deadline_s=10)
        bridge_result.close(); bridge_operation.close(); bridge_freeze.close()

        session.install_reduction_attempt(
            generation=6, attempt=2, owner_epoch=1, source_dtype=DType.F64,
            base_digest=base_digest, plan_digest=plan_digest, deadline_s=10)
        try:
            with session.import_reduction_sources(
                    ((bridge_fd, "assembled", "assembled-boot", 0, 10,
                      bridge_digest),),
                    source_dtype=DType.F64, deadline_s=10):
                final_freeze = session.freeze(deadline_s=10)
                final_operation, final_result = session.finalize_redistribution(
                    deadline_s=10)
        finally:
            os.close(bridge_fd)
        with final_result.mapped("float32") as actual:
            assert np.array_equal(actual, assembled)
        assert final_result.layout_digest == full_layout
        assert final_result.attempt == 2 and final_result.global_weight == 10
        session.abort(deadline_s=10)
        final_result.close(); final_operation.close(); final_freeze.close()
    finally:
        session.close("normal")


def test_transport_python_bridge_exposes_bounded_native_send_receive_abi():
    """Regression: live wiring must not silently omit the dense frame ABI."""
    manifest = json.loads(BUILD_MANIFEST.read_text())
    transport_library = (
        BUILD_MANIFEST.parent / manifest["artifacts"]["transport_library"]["path"])
    library = NativeTransportLibrary(transport_library)
    assert library.lib.ndp_transport_send_v1.argtypes is not None
    assert library.lib.ndp_transport_receive_v1.argtypes is not None

    # Bounds are rejected in Python before entering the provider.  This keeps
    # malformed live-role input from becoming an unbounded ctypes allocation.
    transport = object.__new__(NativeTransport)
    transport.payload_max = 4096
    transport.deadline_unix_ns = time.time_ns() + 1_000_000_000
    with pytest.raises(ValueError, match="frame byte bound"):
        transport.send(1, b"", deadline_unix_ns=time.time_ns() + 1)
    with pytest.raises(ValueError, match="receive capacity"):
        transport.receive(capacity=4096 + 321)


def test_frozen_owner_transfer_uses_native_fabric_and_bounds_replay(monkeypatch):
    """Regression: frozen dense bytes cannot escape through a Python socket."""
    session = object.__new__(NativeManagerSession)
    session._frozen = True
    session.routes = {"node-1": 7}
    session._owner_replays = {}
    sent, sent_fds = [], []

    class Transport:
        deadline_unix_ns = time.time_ns() + 1_000_000_000

        def send(self, peer_id, frame, *, deadline_unix_ns):
            sent.append((peer_id, frame, deadline_unix_ns))

        def send_fd(self, peer_id, fd, *, frame_bytes, deadline_unix_ns):
            sent_fds.append((peer_id, fd, frame_bytes, deadline_unix_ns))

        def receive(self, *, capacity=None):
            return (7, b"native-result")

    session.transport = Transport()
    root = bytes.fromhex("12" * 32)
    for _ in range(3):
        session.transfer_frozen_frame("node-1", b"dense-native-frame",
                                      result_root=root)
    assert [item[:2] for item in sent] == [
        (7, b"dense-native-frame"),
        (7, b"dense-native-frame"),
        (7, b"dense-native-frame"),
    ]
    with pytest.raises(RuntimeError, match="replay limit"):
        session.transfer_frozen_frame("node-1", b"dense-native-frame",
                                      result_root=root)
    for _ in range(3):
        session.transfer_frozen_fd(
            "node-1", 19, frame_bytes=832, result_root=root,
            replay_identity=b"chunk-7")
    assert [item[:3] for item in sent_fds] == [(7, 19, 832)] * 3
    with pytest.raises(RuntimeError, match="replay limit"):
        session.transfer_frozen_fd(
            "node-1", 19, frame_bytes=832, result_root=root,
            replay_identity=b"chunk-7")
    assert session.receive_owner_frame() == ("node-1", b"native-result")

    session._frozen = False
    with pytest.raises(RuntimeError, match="frozen accepted set"):
        session.transfer_frozen_frame("node-1", b"late", result_root=root)


def test_cross_process_trainer_handoff_passes_sealed_memfd_without_dense_socket(tmp_path):
    import fcntl
    import hashlib
    import multiprocessing
    import os
    from ndm.native_dataplane import _memfd_create

    path = tmp_path / "manager.seqpacket"
    identity = dict(run_id="run", fence_epoch=7, generation=11,
                    attempt=2, expected_bytes=4096, layout_digest="ab" * 32,
                    accepted_incarnations={"trainer-3": "boot-a"},
                    replay_ledger=tmp_path / "admitted.json")
    manager = NativeTrainerHandoff.listen(path, **identity)
    context = multiprocessing.get_context("fork")
    received = context.Queue()

    def manager_process():
        contribution = manager.receive_memfd()
        received.put({"trainer_id": contribution.trainer_id,
                      "generation": contribution.generation,
                      "attempt": contribution.attempt,
                      "weight": contribution.weight,
                      "layout_digest": contribution.layout_digest,
                      "payload": os.pread(contribution.fd, contribution.length, 0)})
        contribution.close()

    process = context.Process(target=manager_process)
    process.start()
    trainer = NativeTrainerHandoff.connect(path, **identity)
    fd = _memfd_create("trainer-delta", getattr(os, "MFD_ALLOW_SEALING", 2))
    payload = struct.pack("<1024f", *(float(index) for index in range(1024)))
    os.write(fd, payload)
    fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 0x0004 | 0x0002 | 0x0008)
    trainer.send_memfd(fd, trainer_id="trainer-3", incarnation="boot-a",
                       submission_seq=4, weight=99,
                       sha256=hashlib.sha256(payload).hexdigest())
    process.join(5)
    assert process.exitcode == 0
    contribution = received.get(timeout=1)
    assert contribution["trainer_id"] == "trainer-3"
    assert contribution["generation"] == 11 and contribution["attempt"] == 2
    assert contribution["weight"] == 99 and contribution["layout_digest"] == "ab" * 32
    assert contribution["payload"] == payload
    received.close(); received.join_thread()
    os.close(fd); trainer.close(); manager.close()


def test_descriptor_admission_rejects_stale_corrupt_nonfinite_and_restart_replay(tmp_path):
    import fcntl
    import hashlib
    import os
    import threading
    from ndm.native_dataplane import _memfd_create

    path = tmp_path / "service.seqpacket"
    ledger = tmp_path / "admitted.json"
    identity = dict(run_id="run", fence_epoch=7, generation=11, attempt=2,
                    expected_bytes=16, layout_digest="ab" * 32,
                    accepted_incarnations={"trainer": "boot-a"},
                    replay_ledger=ledger)

    def attempt(client_identity, payload, *, incarnation="boot-a", sequence=1,
                claimed=None, extent=None):
        manager = NativeTrainerHandoff.listen(path, **identity)
        outcome = []

        def receive():
            try:
                contribution = manager.receive_memfd()
                outcome.append(("accepted", contribution.sha256))
                contribution.close()
            except Exception as error:
                outcome.append(("rejected", str(error)))

        thread = threading.Thread(target=receive)
        thread.start()
        trainer = NativeTrainerHandoff.connect(path, **client_identity)
        fd = _memfd_create("adversarial-delta", getattr(os, "MFD_ALLOW_SEALING", 2))
        os.write(fd, payload if extent is None else payload[:extent])
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033),
                    0x0004 | 0x0002 | 0x0008)
        try:
            trainer.send_memfd(fd, trainer_id="trainer", incarnation=incarnation,
                               submission_seq=sequence, weight=3,
                               sha256=claimed or hashlib.sha256(payload).hexdigest())
        except Exception as error:
            outcome.append(("rejected", str(error)))
            # No packet was sent, so release the blocked accept.
            trainer.close()
            socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET).connect(str(path))
        thread.join(5)
        os.close(fd); trainer.close(); manager.close()
        return outcome[0]

    finite = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    assert attempt(identity, finite)[0] == "accepted"
    # The persisted identity ledger is loaded by a fresh service instance.
    assert "duplicate" in attempt(identity, finite)[1]
    wrong_fence = dict(identity, fence_epoch=8, replay_ledger=None)
    assert "identity" in attempt(wrong_fence, finite, sequence=2)[1]
    assert "incarnation" in attempt(identity, finite, incarnation="boot-b", sequence=3)[1]
    assert "digest" in attempt(identity, finite, sequence=4, claimed="00" * 32)[1]
    nonfinite = struct.pack("<4f", 1.0, float("nan"), 3.0, 4.0)
    assert "nonfinite" in attempt(identity, nonfinite, sequence=5)[1]
    assert "extent" in attempt(identity, finite, sequence=6, extent=12)[1]


def test_descriptor_admission_rejects_ancillary_fd_smuggling(tmp_path):
    """One metadata packet cannot hide a second producer-owned descriptor."""
    import fcntl
    import hashlib
    import multiprocessing
    import os
    from ndm.native_dataplane import _memfd_create

    path = tmp_path / "service.seqpacket"
    identity = dict(run_id="run", fence_epoch=7, generation=11, attempt=2,
                    expected_bytes=16, layout_digest="ab" * 32,
                    accepted_incarnations={"trainer": "boot-a"})
    manager = NativeTrainerHandoff.listen(path, **identity)
    context = multiprocessing.get_context("fork")
    outcome = context.Queue()

    def receive():
        try:
            manager.receive_memfd()
            outcome.put("accepted")
        except Exception as error:
            outcome.put(str(error))

    process = context.Process(target=receive)
    process.start()
    trainer = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    trainer.connect(str(path))
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    metadata = json.dumps({
        "run_id": "run", "fence_epoch": 7, "generation": 11, "attempt": 2,
        "layout_digest": "ab" * 32, "trainer_id": "trainer",
        "incarnation": "boot-a", "submission_seq": 1, "weight": 3,
        "length": 16, "sha256": hashlib.sha256(payload).hexdigest(),
    }).encode()
    descriptors = []
    for name in ("admitted", "smuggled"):
        fd = _memfd_create(name, getattr(os, "MFD_ALLOW_SEALING", 2))
        os.write(fd, payload)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033),
                    0x0004 | 0x0002 | 0x0008)
        descriptors.append(fd)
    trainer.sendmsg([metadata], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                  struct.pack("2i", *descriptors))])
    process.join(5)
    assert process.exitcode == 0
    assert "exactly one memfd" in outcome.get(timeout=1)
    for fd in descriptors:
        os.close(fd)
    trainer.close(); manager.close(); outcome.close(); outcome.join_thread()


def test_fenced_checkpoint_commit_releases_native_result_exactly_once(tmp_path):
    """The fenced publication approval has one native state transition."""
    import hashlib
    from types import SimpleNamespace

    publication = tmp_path / "generation.json"
    value = {
        "finalized": True, "run_id": "live-split", "generation": 8,
        "fence": {"coordinator_epoch": 19}, "attempt": 2,
        "layout_digest": "11" * 32, "base_digest": "22" * 32,
        "result_root": "33" * 32, "global_weight": 41,
        "result_bytes": 128,
    }
    publication.write_text(json.dumps(value, sort_keys=True))
    calls = []

    class Operation:
        def close(self):
            calls.append("close")

    session = object.__new__(NativeManagerSession)
    session.run_id, session.fence_epoch = "live-split", 19
    session.local = SimpleNamespace(control=lambda command, deadline_s: (
        calls.append((command, deadline_s)) or Operation()))
    session._checkpoint_proposed = True
    session._proposal_generation = 8
    session._checkpoint_identity = {
        key: value[key] for key in (
            "attempt", "layout_digest", "base_digest", "result_root",
            "global_weight", "result_bytes")}
    session._generation_installed = session._frozen = True
    session._owner_replays = {("node-1", bytes.fromhex("33" * 32)): 2}

    session.commit(
        publication_manifest=publication,
        authoritative_latest={
            "generation": 8, "fence": 19, "manifest": str(publication.resolve()),
            "manifest_sha256": hashlib.sha256(publication.read_bytes()).hexdigest(),
        }, deadline_s=7.0)

    assert calls == [(Command.COMMIT, 7.0), "close"]
    assert not session._generation_installed and not session._frozen
    assert session._checkpoint_identity is None and not session._owner_replays


def test_persistent_service_matches_k40_delta_tokens_for_all_eight_trainers(tmp_path):
    """Real node topology: eight K40 trainers produce directly into one service."""
    manifest = json.loads(BUILD_MANIFEST.read_text())
    library = NativeLibrary(
        BUILD_MANIFEST.parent / manifest["artifacts"]["local_library"]["path"])
    controller = Client.open(
        library=library, role=Role.CONTROLLER, run_key="e97-run", fence_epoch=23,
        worker_key="node-0-manager", incarnation="manager-boot", deadline_s=10)
    base_state = {
        "decoder.weight": torch.tensor(
            [1024.0, -3.25, 0.5, 7.0, -8192.0], dtype=torch.float64),
        "embedding.weight": torch.tensor(
            [1.0, -2.0, 4.0, 8.0], dtype=torch.float32),
    }
    total_elements = sum(value.numel() for value in base_state.values())
    digest = controller.install_flat_layout(
        total_elements, source_dtype=DType.F32, payload_max=64)
    base_digest, plan_digest = bytes.fromhex("31" * 32), bytes.fromhex("42" * 32)
    install = controller.install_generation(
        7, attempt=1, owner_epoch=1, base_digest=base_digest,
        plan_digest=plan_digest, deadline_s=10)
    generation = GenerationMetadata(
        "e97-run", 23, 7, 1, 1, total_elements, digest.hex(),
        base_digest.hex(), plan_digest.hex(), controller.generation_deadline_ns,
        {"provider": "tcp;ofi_rxm", "build_bundle_sha256": manifest["bundle_sha256"]},
    )
    atomic_metadata(tmp_path / "native-generation-00000007.json", generation.as_json())

    class Model:
        def __init__(self, state):
            self.state = state

        def state_dict(self):
            return self.state

    weights = [3, 5, 7, 11, 13, 17, 19, 23]
    trainers, contributions = [], []
    for rank, tokens in enumerate(weights):
        trainer = NativeTrainerDataPlane.connect(
            build_manifest=BUILD_MANIFEST,
            socket_path=os.environ["EMENDER_NDP_SOCKET"], run_id="e97-run",
            fence_epoch=23, generation=7, rank=rank,
            identity=f"node-0-trainer-{rank}", incarnation=f"trainer-boot-{rank}",
            control_root=tmp_path, deadline=time.monotonic() + 10)
        trainer.allocate_delta(deadline_s=10)
        worker_state = {
            name: base + torch.linspace(
                rank * 0.125 - 0.5, rank * 0.125 + 0.5, base.numel(),
                dtype=torch.float64).reshape(base.shape)
            for name, base in base_state.items()
        }
        expected = np.concatenate([
            (worker_state[name].to(dtype=base_state[name].dtype) - base_state[name])
            .to(torch.float32).reshape(-1).numpy()
            for name in sorted(base_state)
        ])
        contributions.append(expected)
        marker = trainer.publish_model_delta(
            base_state, Model(worker_state), tokens, chunk_elements=2, deadline_s=10)
        assert marker["tokens"] == tokens and marker["rank"] == rank
        assert marker["dense_files_written"] == marker["trainer_spool_bytes"] == 0
        trainers.append(trainer)
    freeze = controller.control(Command.FREEZE, deadline_s=10)
    result_operation = controller.control(Command.FINALIZE_OWNERS, deadline_s=10)
    with controller.result_view(result_operation) as manager_view:
        expected = exact_weighted_reference(contributions, weights)
        result_marker = {
            "schema": "emender-native-e97-result-v1", "run_id": "e97-run",
            "fence_epoch": 23, "generation": 7, "attempt": 1,
            "owner_epoch": 1, "source_dtype": int(DType.F32),
            "deadline_unix_ns": controller.generation_deadline_ns,
            "operation_handle": result_operation.handle,
            "layout_digest": digest.hex(),
            "base_digest": base_digest.hex(), "plan_digest": plan_digest.hex(),
            "result_root": manager_view.result_root.hex(),
            "global_weight": sum(weights),
        }
        atomic_metadata(tmp_path / "native-result-00000007.json", result_marker)
        for trainer in trainers:
            with trainer.result_shards(
                    deadline=time.monotonic() + 10, chunk_elements=3) as (marker, shards):
                actual = np.concatenate([item.numpy() for item in shards])
                assert np.array_equal(actual, expected)
                assert marker["global_weight"] == sum(weights)
    metrics = controller.metrics
    assert metrics.trainer_spool_bytes == metrics.python_dense_socket_bytes == 0
    assert metrics.handoff_full_copy_bytes == 0
    assert metrics.admitted_shared_bytes >= (len(trainers) + 1) * total_elements * 4
    controller.control(Command.ABORT, deadline_s=10).close()
    result_operation.close(); freeze.close(); install.close()
    for trainer in trainers:
        trainer.close()
    controller.close()


def test_persistent_service_preserves_exact_global_numerator(tmp_path):
    """Node exchange keeps f64 numerators; it never divides then reweights."""
    manifest = json.loads(BUILD_MANIFEST.read_text())
    library = NativeLibrary(
        BUILD_MANIFEST.parent / manifest["artifacts"]["local_library"]["path"])
    controller = Client.open(
        library=library, role=Role.CONTROLLER, run_key="global-run", fence_epoch=29,
        worker_key="node-0-manager", incarnation="manager-boot", deadline_s=10)
    elements = 19
    digest = controller.install_flat_layout(
        elements, source_dtype=DType.F32, payload_max=128)
    first_install = controller.install_generation(
        5, attempt=1, owner_epoch=1, deadline_s=10)
    generator = np.random.default_rng(971)
    node_arrays = [
        [generator.normal(0, 1e-5, elements).astype(np.float32),
         generator.normal(0, 1e5, elements).astype(np.float32),
         generator.normal(0, 1, elements).astype(np.float32)],
        [generator.normal(0, 1e3, elements).astype(np.float32),
         generator.normal(0, 1e-3, elements).astype(np.float32)],
    ]
    node_weights = [[3, 1_000_003, 29], [71, 101]]

    trainers, local_submissions = [], []
    for rank, (array, weight) in enumerate(zip(node_arrays[0], node_weights[0])):
        trainer = Client.open(
            library=library, role=Role.TRAINER, run_key="global-run", fence_epoch=29,
            worker_key=f"node-0-trainer-{rank}",
            incarnation=f"node-0-trainer-boot-{rank}", deadline_s=10)
        trainer.attach_generation(
            total_elements=elements, layout_digest=digest, generation=5, attempt=1,
            owner_epoch=1, source_dtype=DType.F32, deadline_s=10,
            deadline_unix_ns=controller.generation_deadline_ns,
            base_digest=controller.base_digest, plan_digest=controller.plan_digest)
        with trainer.allocate(deadline_s=10) as buffer:
            with buffer.mapped(DType.F32, write=True) as target:
                target[:] = array
            buffer.seal()
            local_submissions.append(trainer.submit(
                buffer, trainer_key=f"trainer-{rank}",
                trainer_incarnation=f"node-0-trainer-boot-{rank}",
                submission_seq=rank, weight=weight, deadline_s=10))
        trainers.append(trainer)
    local_freeze = controller.control(Command.FREEZE, deadline_s=10)
    local_operation = controller.control(Command.FINALIZE_OWNERS, deadline_s=10)
    local_view = controller.result_view(local_operation)
    local_numerator = node_arrays[0][0].astype(np.float64) * node_weights[0][0]
    for array, weight in zip(node_arrays[0][1:], node_weights[0][1:]):
        local_numerator += array.astype(np.float64) * weight
    with local_view.mapped(DType.F64) as actual:
        assert np.array_equal(actual, local_numerator)
    assert local_view.dtype is DType.F64
    assert local_view.global_weight == sum(node_weights[0])
    local_fd = os.dup(local_view.fd)
    reader_client = Client.open(
        library=library, role=Role.TRAINER, run_key="global-run", fence_epoch=29,
        worker_key="node-0-result-reader", incarnation="reader-boot", deadline_s=10)
    reader_client.attach_generation(
        total_elements=elements, layout_digest=digest, generation=5, attempt=1,
        owner_epoch=1, source_dtype=DType.F32, deadline_s=10,
        deadline_unix_ns=controller.generation_deadline_ns,
        base_digest=controller.base_digest, plan_digest=controller.plan_digest)
    reader = NativeTrainerDataPlane(
        reader_client,
        GenerationMetadata(
            "global-run", 29, 5, 1, 1, elements, digest.hex(),
            controller.base_digest.hex(), controller.plan_digest.hex(),
            controller.generation_deadline_ns, {}),
        rank=0, identity="node-0-result-reader", incarnation="reader-boot",
        control_root=tmp_path)
    controller.control(Command.ABORT, deadline_s=10).close()
    local_view.close(); local_operation.close(); local_freeze.close(); first_install.close()
    for operation in local_submissions:
        operation.close()
    for trainer in trainers:
        trainer.close()

    node1_numerator = node_arrays[1][0].astype(np.float64) * node_weights[1][0]
    for array, weight in zip(node_arrays[1][1:], node_weights[1][1:]):
        node1_numerator += array.astype(np.float64) * weight
    peer_fd = create_memfd("node-1-numerator", allow_sealing=True)
    os.ftruncate(peer_fd, node1_numerator.nbytes)
    os.pwrite(peer_fd, node1_numerator.tobytes(), 0); seal_memfd(peer_fd)
    controller.source_dtype = DType.F64
    second_install = controller.install_generation(
        5, attempt=2, owner_epoch=1, deadline_s=10)
    manager = object.__new__(NativeManagerSession)
    manager.local = controller
    manager.run_id, manager.fence_epoch = "global-run", 29
    manager._generation_installed, manager._frozen = True, False
    imported = (
        (local_fd, "node-0", "node-0-boot", 0, sum(node_weights[0]),
         __import__("hashlib").sha256(local_numerator.tobytes()).digest()),
        (peer_fd, "node-1", "node-1-boot", 1, sum(node_weights[1]),
         __import__("hashlib").sha256(node1_numerator.tobytes()).digest()),
    )
    with manager.import_reduction_sources(
            imported, source_dtype=DType.F64, deadline_s=10):
        global_freeze = manager.freeze(deadline_s=10)
        global_operation, global_result = manager.finalize_redistribution(
            deadline_s=10)
    global_result.close()
    with controller.result_view(global_operation) as global_view:
        expected = ((local_numerator + node1_numerator) /
                    sum(sum(item) for item in node_weights)).astype(np.float32)
        with global_view.mapped(DType.F32) as actual:
            assert np.array_equal(actual, expected)
        assert global_view.global_weight == sum(sum(item) for item in node_weights)
        assert global_view.dtype is DType.F32
        result_root = global_view.result_root.hex()

    # The live two-node path attaches trainers to local attempt 1, then the
    # managers publish the globally redistributed result from attempt 2.  A
    # trainer must adopt that fenced result attempt before sending ResultView;
    # otherwise the RPC server correctly rejects the stale request header.
    atomic_metadata(tmp_path / "native-result-00000005.json", {
        "schema": "emender-native-e97-result-v1", "run_id": "global-run",
        "fence_epoch": 29, "generation": 5, "attempt": 2, "owner_epoch": 1,
        "source_dtype": int(DType.F64),
        "operation_handle": global_operation.handle,
        "layout_digest": digest.hex(), "base_digest": controller.base_digest.hex(),
        "plan_digest": controller.plan_digest.hex(),
        "deadline_unix_ns": controller.generation_deadline_ns,
        "result_root": result_root,
        "global_weight": sum(sum(item) for item in node_weights),
    })
    with reader.result_shards(
            deadline=time.monotonic() + 10, chunk_elements=7) as (_marker, shards):
        assert np.array_equal(np.concatenate([item.numpy() for item in shards]), expected)
    reader.close()
    assert controller.metrics.projection_count == 2
    controller.control(Command.ABORT, deadline_s=10).close()
    global_operation.close(); global_freeze.close(); second_install.close()
    os.close(local_fd); os.close(peer_fd); controller.close()
