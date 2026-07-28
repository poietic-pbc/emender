from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from ndm.native_artifacts import (
    BUILD_SCHEMA,
    GATE_SCHEMA,
    NATIVE_CXI,
    NATIVE_TEST,
    PYTHON_TCP_DEBUG,
    attest_launch,
    validate_backend,
    validate_build_manifest,
    validate_g2_gate,
)
from scripts.frontier import attest_native_dataplane as attestation_cli


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    paths = {
        "local_library": "lib/libemender_ndp.so.1",
        "transport_library": "lib/libemender_ndp_transport.so.1",
        "service_binary": "bin/ndp_cxi_service",
        "synthetic_gate_binary": "bin/ndp_frontier_2n_gate",
    }
    artifacts = {}
    for name, relative in paths.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile("/bin/true", target)
        raw = target.read_bytes()
        artifacts[name] = {
            "path": relative, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    bound = [(name, value["sha256"]) for name, value in sorted(artifacts.items())]
    value = {
        "schema": BUILD_SCHEMA, "source_commit": "1" * 40,
        "source_tree_dirty": False, "protocol_version": "1.0",
        "local_abi": 0x00010000, "transport_abi": 0x00010000,
        "build": {}, "artifacts": artifacts,
        "bundle_sha256": hashlib.sha256(_canonical(bound)).hexdigest(),
    }
    path = tmp_path / "native-artifacts.json"
    path.write_bytes(_canonical(value) + b"\n")
    return path, value


def _gate(tmp_path: Path, manifest: dict[str, object]) -> Path:
    value = {
        "schema": GATE_SCHEMA, "gate": "G2", "status": "passed",
        "source_commit": manifest["source_commit"],
        "bundle_sha256": manifest["bundle_sha256"],
        "provider": "cxi", "endpoint_type": "FI_EP_RDM",
        "production_provider": True, "layout_bytes": 5_506_770_496,
        "shard_count": 83, "trainers_per_node": 8, "nodes": 2,
        "logical_contribution_bytes": 11_013_540_992,
        "logical_redistribution_bytes": 11_013_540_992,
        "python_dense_socket_bytes": 0, "trainer_spool_bytes": 0,
        "disk_replay_bytes": 0, "handoff_full_copy_bytes": 0,
        "artifacts": {name: record["sha256"]
                      for name, record in manifest["artifacts"].items()},
    }
    path = tmp_path / "full-layout-gate.json"
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _fault_gate(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = _gate(tmp_path, manifest)
    value = json.loads(path.read_text())
    value["gate"] = "G2-fault-rejoin-replay"
    value["fault"] = {
        "peer_loss": True,
        "new_incarnation": True,
        "old_epoch_rejected": True,
        "partial_commit": False,
        "reassignment_count": 1,
        "replay_bytes": 134_217_728,
    }
    path = tmp_path / "failure-injection-gate.json"
    path.write_bytes(_canonical(value) + b"\n")
    return path


def test_production_attestation_binds_source_binaries_provider_and_full_layout(tmp_path):
    manifest_path, manifest = _manifest(tmp_path)
    gate_path = _gate(tmp_path, manifest)
    result = attest_launch(
        backend=NATIVE_CXI, production=True, full_layout=True,
        build_manifest=manifest_path, gate_json=gate_path,
    )
    assert result["status"] == "attested"
    assert result["bundle_sha256"] == manifest["bundle_sha256"]
    assert result["artifacts"] == {
        name: record["sha256"] for name, record in manifest["artifacts"].items()}


def test_fault_attestation_accepts_only_exact_passing_fault_gate(tmp_path):
    manifest_path, manifest = _manifest(tmp_path)
    gate_path = _fault_gate(tmp_path, manifest)

    result = attest_launch(
        backend=NATIVE_CXI,
        production=True,
        full_layout=True,
        build_manifest=manifest_path,
        gate_json=gate_path,
        required_gate="G2-fault-rejoin-replay",
    )
    assert result["status"] == "attested"

    with pytest.raises(ValueError, match="gate identity"):
        attest_launch(
            backend=NATIVE_CXI,
            production=True,
            full_layout=True,
            build_manifest=manifest_path,
            gate_json=gate_path,
        )

    value = json.loads(gate_path.read_text())
    value["fault"]["old_epoch_rejected"] = False
    gate_path.write_bytes(_canonical(value) + b"\n")
    with pytest.raises(ValueError, match="fault identity"):
        attest_launch(
            backend=NATIVE_CXI,
            production=True,
            full_layout=True,
            build_manifest=manifest_path,
            gate_json=gate_path,
            required_gate="G2-fault-rejoin-replay",
        )


def test_fault_attestation_cli_forwards_required_gate_kind(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_attest_launch(**kwargs):
        captured.update(kwargs)
        return {"status": "attested"}

    monkeypatch.setattr(attestation_cli, "attest_launch", fake_attest_launch)
    monkeypatch.setattr(sys, "argv", [
        "attest_native_dataplane.py",
        "verify",
        "--backend",
        NATIVE_CXI,
        "--production",
        "--full-layout",
        "--build-manifest",
        "/immutable/native-artifacts.json",
        "--gate-json",
        "/immutable/failure-injection-gate.json",
        "--required-gate",
        "G2-fault-rejoin-replay",
        "--source-root",
        "/immutable/source",
    ])

    assert attestation_cli.main() == 0
    assert captured["required_gate"] == "G2-fault-rejoin-replay"
    assert json.loads(capsys.readouterr().out) == {"status": "attested"}


def test_historical_component_manifest_is_valid_but_cannot_claim_g2(tmp_path):
    manifest_path, manifest = _manifest(tmp_path)
    manifest["artifacts"].pop("synthetic_gate_binary")
    bound = [
        (name, value["sha256"])
        for name, value in sorted(manifest["artifacts"].items())
    ]
    manifest["bundle_sha256"] = hashlib.sha256(_canonical(bound)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    build = validate_build_manifest(manifest_path)
    assert set(build.artifacts) == {
        "local_library", "transport_library", "service_binary"
    }
    with pytest.raises(ValueError, match="synthetic gate executable"):
        validate_g2_gate(_gate(tmp_path, manifest), build)


def test_production_and_full_layout_refuse_python_tcp_or_unattested_native(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    for production, full_layout in ((True, False), (False, True), (True, True)):
        with pytest.raises(ValueError, match="Python TCP"):
            validate_backend(PYTHON_TCP_DEBUG, production=production,
                             full_layout=full_layout)
    with pytest.raises(ValueError, match="exact G2"):
        attest_launch(
            backend=NATIVE_CXI, production=True, full_layout=True,
            build_manifest=manifest_path, gate_json=None,
        )
    with pytest.raises(ValueError, match="cannot be promoted"):
        attest_launch(
            backend=NATIVE_TEST, production=True, full_layout=False,
            build_manifest=manifest_path, gate_json=None,
        )


def test_artifact_tamper_and_gate_digest_mismatch_fail_closed(tmp_path):
    manifest_path, manifest = _manifest(tmp_path)
    gate_path = _gate(tmp_path, manifest)
    gate = json.loads(gate_path.read_text())
    gate["bundle_sha256"] = "0" * 64
    gate_path.write_bytes(_canonical(gate) + b"\n")
    with pytest.raises(ValueError, match="gate identity"):
        attest_launch(
            backend=NATIVE_CXI, production=True, full_layout=True,
            build_manifest=manifest_path, gate_json=gate_path,
        )

    artifact = tmp_path / manifest["artifacts"]["local_library"]["path"]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest/size"):
        attest_launch(
            backend=NATIVE_TEST, production=False, full_layout=False,
            build_manifest=manifest_path, gate_json=None,
        )
