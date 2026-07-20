import json
import os
from pathlib import Path
import subprocess
import importlib.util

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("exact_2n", ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def _native(root: Path) -> Path:
    (root / "bin").mkdir(parents=True); (root / "lib").mkdir()
    service = root / "bin/service"; service.write_text("#!/bin/sh\nexit 0\n"); service.chmod(0o755)
    (root / "lib/local.so").write_bytes(b"local"); (root / "lib/transport.so").write_bytes(b"transport")
    manifest = root / "native.json"
    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    manifest.write_text(json.dumps({"source_commit": commit, "bundle_sha256": "immutable-bundle", "artifacts": {
        "service_binary": {"path": "bin/service"}, "local_library": {"path": "lib/local.so"},
        "transport_library": {"path": "lib/transport.so"}}}))
    return manifest


def test_dry_run_renders_exact_real_k40_fenced_acceptance_without_submission(tmp_path):
    manifest = _native(tmp_path / "native"); gate = tmp_path / "g2.json"; gate.write_text("{}")
    output = tmp_path / "acceptance.json"
    result = subprocess.run([
        os.fspath(ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py"),
        "--repo", os.fspath(ROOT), "--native-build-manifest", os.fspath(manifest),
        "--full-layout-gate", os.fspath(gate), "--run-root", os.fspath(tmp_path / "runs"),
        "--output", os.fspath(output), "--allow-non-authoritative-dry-run"],
        text=True, capture_output=True, check=True)
    assert "sbatch" not in result.stdout
    plan = json.loads(output.read_text())
    assert plan["node_count"] == 2 and plan["k_local_steps"] == 40
    assert plan["authoritative_stage"] is None
    assert plan["forbidden_node_counts"] == [4, 8, 32, 64, 256]
    assert [p["name"] for p in plan["phases"]] == [
        "clean-overlap", "fault-rejoin", "invalid-result-rejection",
        "checkpoint-publication-failure", "fresh-restart"]
    assert plan["phases"][0]["generations"] == 5
    assert all(p["nodes"] == 2 and p["local_steps"] == 40 for p in plan["phases"])
    assert [p["fence_ordinal"] for p in plan["phases"]] == [1, 2, 3, 4, 5]
    assert plan["phases"][1]["injection"] == {"RESILIENT_E97_INJECT_NATIVE_SERVICE": "1:-1:6"}
    assert plan["phases"][2]["injection"] == {"RESILIENT_E97_INJECT_INVALID_RESULT": "1:9"}
    assert plan["phases"][3]["injection"] == {"RESILIENT_E97_INJECT_PUBLICATION_FAILURE": "10"}
    assert plan["phases"][4]["restart_from"] == "checkpoint-publication-failure"
    identities = {(p["source_commit"], p["native_bundle"]["bundle_sha256"]) for p in plan["phases"]}
    assert len(identities) == 1
    assert all(max(p["stage_deadlines"].values()) <= 420 for p in plan["phases"])


def test_submit_path_is_fail_closed_and_never_contains_4n_submission():
    source = (ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py").read_text()
    assert "origin/main" in source and "clean tracked source tree" in source
    assert "refusing to overlap another user allocation" in source
    assert 'command = ["sbatch", "--parsable", "-N", "2"' in source
    assert '"-N", "4"' not in source and '"--nodes=4"' not in source
    assert 'return advance(plan, output, state_path, repo)' in source
    assert "--dependency=" not in source


def _serial_plan(tmp_path):
    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    return {"source_commit": commit, "authoritative_stage": {"source": {"commit": commit},
            "build_manifest": "/stage/native.json"}, "phases": [{
        "name": name, "run_dir": str(tmp_path / name), "launcher": "two.sbatch",
        "generations": 5 if index == 0 else 1, "initial_generation": index,
        "injection": {}} for index, name in enumerate(("clean-overlap", "fault-rejoin"))]}


def test_one_job_qos_never_receives_concurrent_phase_submission(monkeypatch, tmp_path):
    calls = []
    plan = _serial_plan(tmp_path)
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(MODULE, "verify_source_identity", lambda *args: None)
    def output(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue": return ""
        if command[0] == "sbatch": return "5035685\n"
        raise AssertionError(command)
    monkeypatch.setattr(MODULE.subprocess, "check_output", output)
    state = tmp_path / "state.json"; output_path = tmp_path / "acceptance.json"
    assert MODULE.advance(plan, output_path, state, ROOT) == 75
    assert len([call for call in calls if call[0] == "sbatch"]) == 1
    saved = json.loads(state.read_text())
    assert saved["active"]["phase"] == "clean-overlap" and saved["next_phase"] == 0


def test_pending_phase_is_resumable_wait_and_does_not_submit(monkeypatch, tmp_path):
    plan = _serial_plan(tmp_path); state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema": "emender-exact-2n-serial-state-v1", "next_phase": 0,
        "active": {"phase": "clean-overlap", "job_id": "5035685", "run_dir": str(tmp_path / "clean-overlap")},
        "history": []}))
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "check_output", lambda command, **kwargs:
                        calls.append(command) or ("PENDING\n" if command[0] == "squeue" else ""))
    assert MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT) == 75
    saved = json.loads(state.read_text())
    assert saved["wait"] == {"kind": "slurm-terminal", "job_id": "5035685",
                             "observed_state": "PENDING", "resumable": True}
    assert not any(call[0] == "sbatch" for call in calls)


def test_stale_installed_bundle_source_is_rejected(tmp_path):
    manifest = _native(tmp_path / "native")
    value = json.loads(manifest.read_text()); value["source_commit"] = "1" * 40
    manifest.write_text(json.dumps(value))
    gate = tmp_path / "gate.json"; gate.write_text("{}")
    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(ValueError, match="does not match the launched source"):
        MODULE.build_plan(ROOT, commit, manifest, gate, tmp_path / "runs")
