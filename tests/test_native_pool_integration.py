from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from ndm.native_artifacts import NATIVE_TEST
from ndm.native_pool_runtime import NativeManagerSession
from ndm.native_transport import NativeTransport, NativeTransportLibrary
from ndm.resilient_pool_runtime import OwnerEndpoint


ROOT = Path(__file__).resolve().parents[1]
BUILD_MANIFEST = ROOT / "build/native-resilient-dataplane/native-artifacts.json"


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
