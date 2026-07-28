import json
import os
from pathlib import Path
import subprocess
import importlib.util
import sys

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("exact_2n", ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)
SEED_SPEC = importlib.util.spec_from_file_location(
    "seed_materializer", ROOT / "scripts/frontier/materialize_e97_s3_seed.py")
SEED_MODULE = importlib.util.module_from_spec(SEED_SPEC)
SEED_SPEC.loader.exec_module(SEED_MODULE)
CANONICAL_SEED = json.loads(
    (ROOT / "configs/frontier/e97_async_256.yaml").read_text())["seed"]


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
        sys.executable,
        os.fspath(ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py"),
        "--repo", os.fspath(ROOT), "--native-build-manifest", os.fspath(manifest),
        "--full-layout-gate", os.fspath(gate), "--run-root", os.fspath(tmp_path / "runs"),
        "--output", os.fspath(output), "--allow-non-authoritative-dry-run"],
        text=True, capture_output=True, check=True)
    assert "sbatch" not in result.stdout
    plan = json.loads(output.read_text())
    assert plan["node_count"] == 2 and plan["k_local_steps"] == 40
    assert plan["seed"] == CANONICAL_SEED
    assert plan["queue"] == {"partition": "batch", "qos": "debug"}
    assert plan["authoritative_stage"] is None
    assert plan["forbidden_node_counts"] == [4, 8, 32, 64, 256]
    assert [p["name"] for p in plan["phases"]] == [
        "clean-overlap", "fault-rejoin", "invalid-result-rejection",
        "checkpoint-publication-failure", "fresh-restart"]
    assert plan["phases"][0]["generations"] == 12
    assert all(p["nodes"] == 2 and p["local_steps"] == 40 for p in plan["phases"])
    assert all(p["seed"] == CANONICAL_SEED for p in plan["phases"])
    assert all(p["partition"] == "batch" and p["qos"] == "debug"
               for p in plan["phases"])
    assert [p["fence_ordinal"] for p in plan["phases"]] == [1, 2, 3, 4, 5]
    assert plan["phases"][1]["injection"] == {
        "RESILIENT_E97_INJECT_NATIVE_SERVICE": "1:-1:13"}
    assert plan["phases"][2]["injection"] == {
        "RESILIENT_E97_INJECT_INVALID_RESULT": "1:16"}
    assert plan["phases"][3]["injection"] == {
        "RESILIENT_E97_INJECT_PUBLICATION_FAILURE": "17"}
    assert plan["phases"][4]["restart_from"] == "checkpoint-publication-failure"
    identities = {(p["source_commit"], p["native_bundle"]["bundle_sha256"]) for p in plan["phases"]}
    assert len(identities) == 1
    assert all(max(p["stage_deadlines"].values()) <= 420 for p in plan["phases"])
    assert plan["phases"][0]["performance_gate"] == {
        "all_eight_apply_swap_seconds_max": 60,
        "causal_phase_classes": [
            "freeze_snapshot",
            "snapshot_admission",
            "publish_network",
            "aggregation",
            "checkpoint",
            "result_wait",
            "apply_swap",
        ],
        "foreground_idle_fraction_strict_max": 0.10,
        "foreground_gap_seconds_max": 60,
        "foreground_result_wait_seconds_max": 0,
        "steady_state_cadence_multiple_max": 1.25,
        "warmup_windows_per_trainer": 2,
        "measured_windows_per_trainer": 10,
        "requires_versioned_background_overlap": True,
        "commit_lag_p99_max": 2,
        "anchor_lag_p99_max": 2,
        "speculative_window_lag_p99_max": 2,
        "freeze_to_latest_seconds_max": 420,
        "pause_tail_statistics": ["maximum", "p99"],
        "snapshot_admission_seconds_max": 1,
    }
    assert plan["policy_id"] == "async-decoupled-v2.1-simple"
    assert plan["conformance"]["async_v21_requirements"] == [
        f"V21S{i:02d}" for i in range(1, 18)]
    assert plan["conformance"]["immutable_snapshot_requirements"] == [
        f"ISP{i:02d}" for i in range(1, 8)]


def test_submit_path_is_fail_closed_and_never_contains_4n_submission():
    source = (ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py").read_text()
    assert "origin/main" in source and "clean tracked source tree" in source
    assert "refusing to overlap another user allocation" in source
    assert 'command = ["sbatch", "--parsable", "-N", "2"' in source
    assert '"-p", "batch", "--qos=debug"' in source
    assert '"-N", "4"' not in source and '"--nodes=4"' not in source
    assert 'return advance(plan, output, state_path, repo)' in source
    assert "--dependency=" not in source
    assert "step_1525000" not in source and "step-1525000" not in source
    assert "1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9" not in source
    payload = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert "validate_pipelined_e97_performance.py" in payload
    assert "--telemetry-root" in payload and "pipelined-performance.json" in payload


def _serial_plan(tmp_path):
    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    return {"source_commit": commit, "authoritative_stage": {"source": {"commit": commit},
            "build_manifest": "/stage/native.json"}, "seed": CANONICAL_SEED,
            "queue": {"partition": "batch", "qos": "debug"}, "phases": [{
        "name": name, "run_dir": str(tmp_path / name),
        "launcher": "scripts/frontier/resilient_e97_true_2n.sbatch",
        "full_layout_gate": str(tmp_path / "g2.json"), "fence_ordinal": index + 1,
        "generations": 5 if index == 0 else 1, "initial_generation": index,
        "injection": {}} for index, name in enumerate(("clean-overlap", "fault-rejoin"))]}


def test_one_job_qos_never_receives_concurrent_phase_submission(monkeypatch, tmp_path):
    calls = []
    plan = _serial_plan(tmp_path)
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(MODULE, "verify_source_identity", lambda *args: None)
    def fake_prefetch(seed, cache_root, attestation):
        cache = tmp_path / f"sha256-{seed['sha256']}.pt"
        cache.write_bytes(b"fixture")
        attestation.write_text("{}")
        return cache, attestation
    monkeypatch.setattr(MODULE, "prefetch", fake_prefetch)
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
    assert saved["active"]["scheduler_request"] == {
        "partition": "batch", "qos": "debug"}
    sbatch = next(call for call in calls if call[0] == "sbatch")
    assert sbatch[sbatch.index("-p") + 1] == "batch"
    assert "--qos=debug" in sbatch
    assert sbatch[sbatch.index("--chdir") + 1] == os.fspath(ROOT.resolve())
    assert sbatch[sbatch.index("--output") + 1] == os.fspath(
        (tmp_path / "clean-overlap/slurm-%j.out").resolve())
    assert sbatch[sbatch.index("--error") + 1] == os.fspath(
        (tmp_path / "clean-overlap/slurm-%j.err").resolve())
    assert sbatch[-1] == os.fspath(
        (tmp_path / "clean-overlap/rendered.sbatch").resolve())
    exported = next(value for value in sbatch if value.startswith("--export=ALL,"))
    for required in ("REPO=", "NDP_FULL_LAYOUT_GATE_JSON=",
                     "NDP_REQUIRED_GATE=G2", "EMENDER_CONDA_ENV=",
                     "RESILIENT_E97_SEED_CONFIG=", "RESILIENT_E97_SEED_STEP=2300930",
                     "RESILIENT_E97_SEED_TOKENS=150793748480",
                     "RESILIENT_E97_SEED_SIZE=7719680116",
                     "RESILIENT_E97_SEED_SHA256=0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2",
                     "RESILIENT_E97_SEED_CACHE=",
                     "RESILIENT_E97_SEED_ATTESTATION=",
                     "RESILIENT_E97_SEED_ATTESTATION_SHA256=",
                     "RESILIENT_E97_DATA=",
                     "RESILIENT_E97_TIKTOKEN_CACHE_FILE=", "RESILIENT_E97_RUN_ID=",
                     "RESILIENT_E97_SOURCE_ID=", "RESILIENT_E97_PAYLOAD_ID="):
        assert required in exported
    assert "RESILIENT_E97_SEED=/lustre" not in exported


def test_batch_bootstrap_is_sbcast_only_and_offline_before_model_load():
    payload = (
        ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    bootstrap = payload[payload.index("SBCAST=$(command -v sbcast)"):
                        payload.index("export RESILIENT_E97_SEED")]
    assert '"$SBCAST" -f "$RESILIENT_E97_SEED_CACHE" "$RESILIENT_E97_SEED"' in bootstrap
    assert "--verify-local" in bootstrap
    assert "--expected-job-id" in bootstrap
    assert "materialize_e97_s3_seed.py" in bootstrap
    assert "--prefetch" not in bootstrap
    assert "s3://" not in bootstrap and "https://" not in bootstrap
    assert payload.index("--verify-local") < payload.index(
        "resilient_e97_allocation_supervisor.py")


def test_submit_shell_preserves_job_id_until_batch_runtime(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    rendered = MODULE.render_batch_script(
        ROOT, "scripts/frontier/resilient_e97_true_2n.sbatch",
        tmp_path / "rendered.sbatch")
    payload = rendered.read_text()
    deferred = "'/tmp/emender-e97-seed-${SLURM_JOB_ID}'"
    assert deferred in payload
    assert "/tmp/emender-e97-seed-/checkpoint-step-" not in payload

    result = subprocess.run(
        ["bash", "-c", r"""
set -euo pipefail
SLURM_JOB_ID=5059548
RESILIENT_E97_JOB_SEED_TEMPLATE='/tmp/emender-e97-seed-${SLURM_JOB_ID}'
RESILIENT_E97_JOB_SEED_DIR=${RESILIENT_E97_JOB_SEED_TEMPLATE/'${SLURM_JOB_ID}'/"$SLURM_JOB_ID"}
printf '%s\n' "$RESILIENT_E97_JOB_SEED_DIR/checkpoint-step-2300930.pt"
"""],
        text=True, capture_output=True, check=True, env={
            key: value for key, value in os.environ.items()
            if key != "SLURM_JOB_ID"
        })
    assert result.stdout.strip() == (
        "/tmp/emender-e97-seed-5059548/checkpoint-step-2300930.pt")
    monkeypatch.setenv("SLURM_JOB_ID", "5059548")
    destination = Path(result.stdout.strip())
    assert SEED_MODULE._validate_destination(destination) == destination


def test_scheduler_queries_and_retains_partition_and_qos_explicitly(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command, 0, stdout="PENDING|batch|debug\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    result = MODULE._scheduler_state("5059293")
    assert result == {
        "state": "PENDING", "exit_code": "", "partition": "batch", "qos": "debug"}
    assert calls == [[
        "squeue", "-h", "-j", "5059293", "-o", "%T|%P|%q"]]


def test_terminal_scheduler_evidence_queries_sacct_qos_explicitly(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sacct":
            return subprocess.CompletedProcess(
                command, 0, stdout="COMPLETED|0:0|batch|debug\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    assert MODULE._scheduler_state("5059293") == {
        "state": "COMPLETED",
        "exit_code": "0:0",
        "partition": "batch",
        "qos": "debug",
    }
    assert calls[-1] == [
        "sacct", "-n", "-X", "-j", "5059293",
        "--format=State,ExitCode,Partition,QOS", "-P",
    ]


def test_nonzero_squeue_invalid_job_falls_back_to_terminal_sacct_evidence(
        monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command, 1, stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n")
        if command[0] == "sacct":
            return subprocess.CompletedProcess(
                command, 0, stdout="FAILED|1:0|batch|debug\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    monkeypatch.setattr(
        MODULE.subprocess, "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy check_output path bypasses nonzero fallback")))
    assert MODULE._scheduler_state("5065388") == {
        "state": "FAILED",
        "exit_code": "1:0",
        "partition": "batch",
        "qos": "debug",
    }
    assert calls == [
        ["squeue", "-h", "-j", "5065388", "-o", "%T|%P|%q"],
        [
            "sacct", "-n", "-X", "-j", "5065388",
            "--format=State,ExitCode,Partition,QOS", "-P",
        ],
    ]


@pytest.mark.parametrize("partition,qos", [
    ("debug", ""),
    ("batch", "normal"),
    ("batch", ""),
])
def test_scheduler_queue_drift_fails_closed_before_wait(
        monkeypatch, tmp_path, partition, qos):
    plan = _serial_plan(tmp_path)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": "emender-exact-2n-serial-state-v1",
        "next_phase": 0,
        "active": {
            "phase": "clean-overlap",
            "job_id": "5059293",
            "run_dir": str(tmp_path / "clean-overlap"),
            "scheduler_request": {"partition": "batch", "qos": "debug"},
        },
        "history": [],
    }))
    monkeypatch.setattr(MODULE, "_scheduler_state", lambda _job: {
        "state": "PENDING", "exit_code": "", "partition": partition, "qos": qos})
    with pytest.raises(ValueError, match="Partition=batch and QOS=debug"):
        MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT)


def test_pending_phase_is_resumable_wait_and_does_not_submit(monkeypatch, tmp_path):
    plan = _serial_plan(tmp_path); state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema": "emender-exact-2n-serial-state-v1", "next_phase": 0,
        "active": {"phase": "clean-overlap", "job_id": "5035685", "run_dir": str(tmp_path / "clean-overlap")},
        "history": []}))
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or subprocess.CompletedProcess(
                            command, 0,
                            stdout=("PENDING|batch|debug\n"
                                    if command[0] == "squeue" else ""),
                            stderr=""))
    assert MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT) == 75
    saved = json.loads(state.read_text())
    assert saved["wait"] == {"kind": "slurm-terminal", "job_id": "5035685",
                             "observed_state": "PENDING", "resumable": True,
                             "scheduler_evidence": {
                                 "state": "PENDING", "exit_code": "",
                                 "partition": "batch", "qos": "debug"}}
    assert not any(call[0] == "sbatch" for call in calls)


@pytest.mark.parametrize("scheduler_state", ["PENDING", "RUNNING"])
def test_active_nonterminal_phase_never_submits_duplicate(
        monkeypatch, tmp_path, scheduler_state):
    plan = _serial_plan(tmp_path)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": "emender-exact-2n-serial-state-v1",
        "next_phase": 0,
        "active": {
            "phase": "clean-overlap",
            "job_id": "5065388",
            "run_dir": str(tmp_path / "clean-overlap"),
        },
        "history": [],
    }))
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or subprocess.CompletedProcess(
                            command, 0,
                            stdout=f"{scheduler_state}|batch|debug\n", stderr=""))
    assert MODULE.advance(
        plan, tmp_path / "acceptance.json", state, ROOT) == 75
    assert not any(call[0] == "sbatch" for call in calls)


def test_unexpected_clean_phase_failure_never_submits_next_phase(
        monkeypatch, tmp_path):
    plan = _serial_plan(tmp_path)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": "emender-exact-2n-serial-state-v1",
        "next_phase": 0,
        "active": {
            "phase": "clean-overlap",
            "job_id": "5065388",
            "run_dir": str(tmp_path / "clean-overlap"),
        },
        "history": [],
    }))
    calls = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or subprocess.CompletedProcess(
                            command, 0,
                            stdout=("" if command[0] == "squeue"
                                    else "FAILED|1:0|batch|debug\n"),
                            stderr=""))
    with pytest.raises(
            ValueError,
            match="clean-overlap had unexpected terminal state FAILED"):
        MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT)
    assert not any(call[0] == "sbatch" for call in calls)
    saved = json.loads(state.read_text())
    assert saved["active"]["job_id"] == "5065388"
    assert saved["next_phase"] == 0


def test_squeue_to_sacct_propagation_gap_is_retried_then_harvested(monkeypatch, tmp_path):
    plan = _serial_plan(tmp_path); plan["phases"] = plan["phases"][:1]
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema": "emender-exact-2n-serial-state-v1", "next_phase": 0,
        "active": {"phase": "clean-overlap", "job_id": "5053690",
                   "run_dir": str(tmp_path / "clean-overlap")}, "history": []}))
    monkeypatch.setattr(MODULE.time, "time", lambda: 1000.0)
    monkeypatch.setattr(MODULE, "_scheduler_state", lambda _job: {
        "state": "ACCOUNTING_PENDING", "exit_code": "",
        "partition": "", "qos": ""})
    assert MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT) == 75
    saved = json.loads(state.read_text())
    assert saved["accounting_pending_since"] == 1000.0

    monkeypatch.setattr(MODULE, "_scheduler_state", lambda _job: {
        "state": "COMPLETED", "exit_code": "0:0",
        "partition": "batch", "qos": "debug"})
    assert MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT) == 0
    assert "accounting_pending_since" not in json.loads(state.read_text())
    terminal = json.loads(
        (tmp_path / "clean-overlap/scheduler-terminal.json").read_text())
    assert terminal["partition"] == "batch" and terminal["qos"] == "debug"


def test_sacct_propagation_gap_has_a_bounded_failure_window(monkeypatch, tmp_path):
    plan = _serial_plan(tmp_path); state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema": "emender-exact-2n-serial-state-v1", "next_phase": 0,
        "active": {"phase": "clean-overlap", "job_id": "5053690",
                   "run_dir": str(tmp_path / "clean-overlap")}, "history": [],
        "accounting_pending_since": 1000.0}))
    monkeypatch.setattr(MODULE.time, "time", lambda: 1121.0)
    monkeypatch.setattr(MODULE, "_scheduler_state", lambda _job: {
        "state": "ACCOUNTING_PENDING", "exit_code": "",
        "partition": "", "qos": ""})
    with pytest.raises(TimeoutError, match="absent from squeue and sacct"):
        MODULE.advance(plan, tmp_path / "acceptance.json", state, ROOT)


def test_stale_installed_bundle_source_is_rejected(tmp_path):
    manifest = _native(tmp_path / "native")
    value = json.loads(manifest.read_text()); value["source_commit"] = "1" * 40
    manifest.write_text(json.dumps(value))
    gate = tmp_path / "gate.json"; gate.write_text("{}")
    commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(ValueError, match="does not match the launched source"):
        MODULE.build_plan(ROOT, commit, manifest, gate, tmp_path / "runs")


def test_exact_inputs_have_final_seed_and_guides_are_lock_step():
    expected = (
        "step-2300930 tokens=150793748480 size=7719680116 "
        "sha256:0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2"
    )
    for relative in (
        "configs/frontier/e97_resilient_debug_rendered.json",
        "configs/frontier/e97_resilient_production_rendered.json",
    ):
        rendered = json.loads((ROOT / relative).read_text())
        assert rendered["seed"] == expected
        assert "1525000" not in json.dumps(rendered)
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
