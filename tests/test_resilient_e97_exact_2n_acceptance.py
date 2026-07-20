import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def _native(root: Path) -> Path:
    (root / "bin").mkdir(parents=True); (root / "lib").mkdir()
    service = root / "bin/service"; service.write_text("#!/bin/sh\nexit 0\n"); service.chmod(0o755)
    (root / "lib/local.so").write_bytes(b"local"); (root / "lib/transport.so").write_bytes(b"transport")
    manifest = root / "native.json"
    manifest.write_text(json.dumps({"bundle_sha256": "immutable-bundle", "artifacts": {
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
    assert 'dependency = "afternotok"' in source
