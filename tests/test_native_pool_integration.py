from __future__ import annotations

import json
from pathlib import Path
import time

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

