from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts.frontier.run_async_v21_qualification import (
    EVIDENCE_ONLY_PATH_PREFIXES,
    EXECUTION_SOURCE_SCHEMA,
    FAULT_PHASE_SPECS,
    PAYLOAD_SCHEMA,
    SEED_SHA256,
    TOKENIZER_SHA256,
    V21ScaleClosure,
    _arguments,
    _next_fault_phase,
    _source_digest,
    _verify_prior_clean_gate,
    build_plan,
    canonical_digest,
    submit_plan,
    validate_scale_evidence,
)


IDENTITIES = {
    "source_digest": "1" * 64,
    "policy_digest": "2" * 64,
    "bundle_digest": "3" * 64,
    "seed_digest": SEED_SHA256,
    "launcher_digest": "5" * 64,
}
SCALE_PARAMETERS = {
    "close_on_q_min": False,
    "uses_launched_ranks": False,
    "wait_for_all_ready": False,
}


def _write(path: Path, value: dict) -> Path:
    value = dict(value)
    value["manifest_digest"] = canonical_digest(value)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def test_native_g2_scheduler_logs_do_not_dirty_authoritative_source():
    repo = Path(__file__).resolve().parents[1]
    for suffix in ("out", "err"):
        completed = subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                f"logs/frontier/native-dataplane/"
                f"native-ndp-g2-clean-123456.{suffix}",
            ],
            cwd=repo,
            check=False,
        )
        assert completed.returncode == 0


def test_controller_direct_execution_bootstraps_repo_import_path(
    tmp_path: Path,
):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/frontier/run_async_v21_qualification.py"
    probe = """
import runpy
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
sys.path[:] = [
    entry for entry in sys.path
    if not entry or Path(entry).resolve() != repo
]
runpy.run_path(str(script), run_name="controller_import_probe")
assert str(repo) in sys.path
__import__("scripts.frontier.materialize_e97_s3_seed")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(repo), str(script)],
        cwd=tmp_path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("gate", ("clean", "faults", "convergence"))
def test_v21_two_node_dry_run_pins_scheduler_queue(tmp_path: Path, gate: str):
    plan = build_plan(
        gate=gate,
        nodes=2,
        state_path=tmp_path / "state.json",
        evidence_root=tmp_path,
        parameters={"seed": 7},
        **IDENTITIES,
    )
    assert plan["payload"]["schema"] == PAYLOAD_SCHEMA
    assert plan["scheduler"] == {
        "Nodes": 2,
        "Partition": "batch",
        "QOS": "debug",
    }
    assert "--nodes=2" in plan["command"]
    assert "--partition=batch" in plan["command"]
    assert "--qos=debug" in plan["command"]
    assert "--hold" in plan["command"]


def test_v21_clean_plan_binds_reviewed_full_acceptance_launch(tmp_path: Path):
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    plan = build_plan(
        gate="clean",
        nodes=2,
        state_path=tmp_path / "state.json",
        evidence_root=tmp_path,
        parameters={},
        clean_launch={
            "repo": str(repo),
            "source_commit": "9" * 40,
            "native_source_commit": "8" * 40,
            "execution_source_schema":
                "emender-async-v21-execution-source-v1",
            "execution_source_digest": IDENTITIES["source_digest"],
            "seed_config": str(repo / "configs/frontier/e97_async_256.yaml"),
            "native_build_manifest": str(tmp_path / "native-artifacts.json"),
            "native_build_manifest_sha256": "a" * 64,
            "full_layout_gate": str(tmp_path / "full-layout-gate.json"),
            "full_layout_gate_sha256": "b" * 64,
            "run_dir": str(run_dir),
            "acceptance_manifest": str(tmp_path / "clean-plan.json"),
            "seed_cache": str(tmp_path / f"sha256-{SEED_SHA256}.pt"),
            "seed_attestation": str(tmp_path / "seed-attestation.json"),
            "seed_attestation_sha256": "6" * 64,
            "train_args": str(
                repo / "configs/frontier/e97_resilient_split_role_flat.json"),
            "train_args_sha256": "c" * 64,
            "data": str(tmp_path / "commapile.txt"),
            "data_identity_digest": "7" * 64,
            "tokenizer": str(tmp_path / "p50k"),
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        **IDENTITIES,
    )

    assert plan["scheduler"] == {
        "Nodes": 2,
        "Partition": "batch",
        "QOS": "debug",
        "TimeLimit": "02:00:00",
    }
    assert plan["payload"]["parameters"] == {
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
        "freeze_to_latest_seconds_max": 420,
        "immutable_snapshot_requirements": [
            f"ISP{i:02d}" for i in range(1, 8)
        ],
        "local_owned_latency_seconds_max": 1,
        "local_steps": 40,
        "measured_windows_per_trainer": 10,
        "minimum_atomic_commits": 10,
        "pause_tail_statistics": ["maximum", "p99"],
        "progress_deadline_seconds": 2700,
        "real_trainers": 16,
        "snapshot_admission_seconds_max": 1,
        "steady_state_cadence_multiple_max": 1.25,
        "trainers_per_node": 8,
        "warmup_windows_per_trainer": 2,
    }
    assert plan["payload"]["training_inputs"] == {
        "data_identity_digest": "7" * 64,
        "execution_source_digest": IDENTITIES["source_digest"],
        "execution_source_schema": "emender-async-v21-execution-source-v1",
        "full_layout_gate_sha256": "b" * 64,
        "native_build_manifest_sha256": "a" * 64,
        "native_source_commit": "8" * 40,
        "seed_config": str(repo / "configs/frontier/e97_async_256.yaml"),
        "seed_attestation_sha256": "6" * 64,
        "source_commit": "9" * 40,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "train_args": str(
            repo / "configs/frontier/e97_resilient_split_role_flat.json"),
        "train_args_sha256": "c" * 64,
    }
    assert "--time=02:00:00" in plan["command"]
    assert "--network=job_vni" in plan["command"]
    assert f"--chdir={repo}" in plan["command"]
    assert f"--output={run_dir / 'slurm-%j.out'}" in plan["command"]
    assert f"--error={run_dir / 'slurm-%j.err'}" in plan["command"]
    exports = next(
        item.removeprefix("--export=")
        for item in plan["command"]
        if item.startswith("--export=")
    ).split(",")
    assert "RESILIENT_E97_ACCEPTANCE_PHASE=clean-overlap" in exports
    # Job 5084736 reached the tenth immutable checkpoint but Slurm delivered
    # the launcher's five-minute TERM signal before its tenth all-rank apply.
    # The clean controller must ask for exactly the ten accepted transactions
    # required by the gate and retain enough of the debug allocation to let
    # the final boundary/apply and in-job validator finish.
    assert "RESILIENT_E97_GENERATIONS=10" in exports
    assert "--signal=B:TERM@60" in plan["command"]
    assert "RESILIENT_E97_PROGRESS_DEADLINE_S=2700" in exports
    assert "RESILIENT_E97_GENERATION_DEADLINE_S=420" in exports
    assert "RESILIENT_E97_MAX_RESTARTS=0" in exports
    assert "RESILIENT_E97_STARTUP_SMOKE=0" in exports
    assert "RESILIENT_E97_REQUESTED_WALLTIME=02:00:00" in exports
    assert "RESILIENT_E97_COMPUTE_NODE_NETWORK_FETCHES=0" in exports
    assert "NDP_BUILD_MANIFEST=" + str(
        tmp_path / "native-artifacts.json") in exports
    assert "NDP_FULL_LAYOUT_GATE_JSON=" + str(
        tmp_path / "full-layout-gate.json") in exports
    assert "RESILIENT_E97_SEED_CACHE=" + str(
        tmp_path / f"sha256-{SEED_SHA256}.pt") in exports
    assert "RESILIENT_E97_TIKTOKEN_SHA256=" + TOKENIZER_SHA256 in exports


def _production_launch(tmp_path: Path) -> dict[str, str]:
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    return {
        "repo": str(repo),
        "source_commit": "9" * 40,
        "native_source_commit": "8" * 40,
        "execution_source_schema": EXECUTION_SOURCE_SCHEMA,
        "execution_source_digest": IDENTITIES["source_digest"],
        "seed_config": str(repo / "configs/frontier/e97_async_256.yaml"),
        "native_build_manifest": str(tmp_path / "native-artifacts.json"),
        "native_build_manifest_sha256": "a" * 64,
        "full_layout_gate": str(tmp_path / "full-layout-gate.json"),
        "full_layout_gate_sha256": "b" * 64,
        "run_dir": str(run_dir),
        "acceptance_manifest": str(tmp_path / "fault-plan.json"),
        "seed_cache": str(tmp_path / f"sha256-{SEED_SHA256}.pt"),
        "seed_attestation": str(tmp_path / "seed-attestation.json"),
        "seed_attestation_sha256": "6" * 64,
        "train_args": str(
            repo / "configs/frontier/e97_resilient_split_role_flat.json"),
        "train_args_sha256": "c" * 64,
        "data": str(tmp_path / "commapile.txt"),
        "data_identity_digest": "7" * 64,
        "tokenizer": str(tmp_path / "p50k"),
        "tokenizer_sha256": TOKENIZER_SHA256,
    }


def _passing_clean_terminal(
    tmp_path: Path, *, identities: dict[str, str] = IDENTITIES,
) -> Path:
    payload_input = tmp_path / "clean-payload-input.json"
    payload_input.write_text(json.dumps({
        "schema": "emender-async-v21-collector-input-v1",
        "payload_digest": "d" * 64,
        "payload": {
            "schema": PAYLOAD_SCHEMA,
            "gate": "clean",
            "nodes": 2,
            "identities": identities,
        },
        "scheduler": {
            "Nodes": 2, "Partition": "batch", "QOS": "debug",
            "TimeLimit": "02:00:00",
        },
        "model_command": ["sbatch", "--nodes=2"],
    }, sort_keys=True))
    terminal = {
        "schema": "emender-async-v21-terminal-verdict-v1",
        "payload_digest": "d" * 64,
        "payload_job_id": "5100201",
        "scheduler": {
            "job_id": "5100201",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "derived_exit_code": "0:0",
            "partition": "batch",
            "qos": "debug",
            "nodes": 2,
        },
        "validator_inputs": {
            "payload": {
                "path": str(payload_input),
                "bytes": payload_input.stat().st_size,
                "sha256": hashlib.sha256(payload_input.read_bytes()).hexdigest(),
            },
            "semantic_verdict": {"required": True, "passed": True},
        },
        "passed": True,
        "verdict": "passed",
    }
    terminal["manifest_digest"] = canonical_digest(terminal)
    path = tmp_path / "clean-terminal-verdict.json"
    path.write_text(json.dumps(terminal, sort_keys=True))
    return path


def test_prescribed_fault_cli_accepts_prior_gate_and_renders_serial_phases(
    tmp_path: Path,
):
    parsed = _arguments([
        "--gate", "faults", "--nodes", "2",
        "--repo", str(tmp_path / "snapshot"),
        "--seed-config", "configs/frontier/e97_async_256.yaml",
        "--native-build-manifest", str(tmp_path / "native.json"),
        "--full-layout-gate", str(tmp_path / "g2-fault.json"),
        "--prior-gate", str(tmp_path / "clean-terminal.json"),
        "--run-root", str(tmp_path / "runs"),
        "--state", str(tmp_path / "state.json"),
        "--output", str(tmp_path / "fault-manifest.json"),
        "--submit",
    ])
    assert parsed.prior_gate == str(tmp_path / "clean-terminal.json")
    assert [phase["name"] for phase in FAULT_PHASE_SPECS] == [
        "fault-baseline", "fault-rejoin", "fresh-recovery"]
    assert [phase["initial_generation"] for phase in FAULT_PHASE_SPECS] == [
        0, 2, 6]
    assert [phase["generations"] for phase in FAULT_PHASE_SPECS] == [2, 4, 5]


def test_fault_gate_binds_passing_clean_and_executable_injections(
    tmp_path: Path,
):
    prior_path = _passing_clean_terminal(tmp_path)
    prior = _verify_prior_clean_gate(prior_path, expected_identities=IDENTITIES)
    phase = _next_fault_phase(
        tmp_path / "state.json",
        campaign_digest="e" * 64,
        prior_gate=prior,
    )
    launch = _production_launch(tmp_path)
    launch.update({
        "fault_campaign_digest": "e" * 64,
        "fault_phase": phase["name"],
        "fault_phase_index": "0",
        "initial_generation": str(phase["initial_generation"]),
        "generations": str(phase["generations"]),
        "coordinator_epoch": "1",
        "resume_handoff": "",
        "prior_gate": str(prior_path),
        "prior_gate_sha256": prior["sha256"],
        "prior_payload_digest": prior["payload_digest"],
    })
    plan = build_plan(
        gate="faults",
        nodes=2,
        state_path=tmp_path / "state.json",
        evidence_root=tmp_path,
        parameters={},
        fault_launch=launch,
        **IDENTITIES,
    )

    assert plan["payload"]["parameters"]["fault_phase"] == "fault-baseline"
    assert plan["payload"]["prior_gate"] == {
        "path": str(prior_path.resolve()),
        "sha256": prior["sha256"],
        "payload_digest": "d" * 64,
    }
    exports = next(
        item.removeprefix("--export=")
        for item in plan["command"]
        if item.startswith("--export=")
    ).split(",")
    assert "RESILIENT_E97_ACCEPTANCE_PHASE=fault-baseline" in exports
    assert "RESILIENT_E97_GENERATIONS=2" in exports
    assert "RESILIENT_E97_INITIAL_GENERATION=0" in exports
    assert "RESILIENT_E97_MAX_RESTARTS=0" in exports
    assert "RESILIENT_E97_INJECT_TRAINER=" in exports
    assert "RESILIENT_E97_INJECT_MANAGER=" in exports
    assert "RESILIENT_E97_INJECT_NATIVE_SERVICE=" in exports
    assert plan["collector"]["semantic_verdict"].endswith(
        "fault-baseline-verdict.json")

    injection_phase = FAULT_PHASE_SPECS[1]
    assert injection_phase["injections"] == {
        "RESILIENT_E97_DELAY_READY": "1:2:45",
        "RESILIENT_E97_INJECT_TRAINER": "0:3:2",
        "RESILIENT_E97_INJECT_MANAGER":
            "1:-1:4:published_node_applied",
        "RESILIENT_E97_INJECT_NATIVE_SERVICE": "0:-1:4:owner_transport",
    }
    recovery = FAULT_PHASE_SPECS[2]
    assert recovery["generations"] >= 5
    assert recovery["minimum_commits"] >= 3
    assert recovery["fresh_allocation"] is True


def test_fault_campaign_stops_on_failed_phase_and_never_advances(
    tmp_path: Path,
):
    prior = _verify_prior_clean_gate(
        _passing_clean_terminal(tmp_path),
        expected_identities=IDENTITIES,
    )
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "schema": "emender-async-v21-qualification-state-v2",
        "payloads": {
            "a" * 64: {
                "status": "retired",
                "verdict": "failed",
                "payload_digest": "a" * 64,
                "campaign_digest": "e" * 64,
                "fault_phase": "fault-baseline",
                "fault_phase_index": 0,
                "prior_payload_digest": "d" * 64,
            },
        },
        "active_job": None,
    }))
    with pytest.raises(ValueError, match="fault-baseline.*failed"):
        _next_fault_phase(
            state,
            campaign_digest="e" * 64,
            prior_gate=prior,
        )


def test_v21_scale_rejects_missing_authorization_and_wrong_predecessor(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="authorization"):
        build_plan(
            gate="scale",
            nodes=4,
            state_path=tmp_path / "state.json",
            evidence_root=tmp_path,
            parameters=SCALE_PARAMETERS,
            **IDENTITIES,
        )

    authorization = _write(tmp_path / "authorization.json", {
        "schema": "emender-async-v21-scale-authorization-v1",
        "status": "passed",
        "authorized_nodes": 8,
        "identities": IDENTITIES,
        "review_signature": "signed-for-test",
        "closure": {"schema": "emender-v21s17-scale-closure-v1"},
    })
    predecessor = _write(tmp_path / "prior.json", {
        "schema": "emender-async-v21-rung-pass-v1",
        "status": "passed",
        "nodes": 2,
        "identities": IDENTITIES,
        "review_signature": "signed-for-test",
    })
    with pytest.raises(ValueError, match="predecessor.*4"):
        build_plan(
            gate="scale",
            nodes=8,
            state_path=tmp_path / "state.json",
            evidence_root=tmp_path,
            parameters=SCALE_PARAMETERS,
            authorization_path=authorization,
            predecessor_path=predecessor,
            allow_test_signatures=True,
            **IDENTITIES,
        )


def test_v21_unchanged_failed_payload_and_one_active_job_are_rejected(
    tmp_path: Path,
):
    state = tmp_path / "state.json"
    plan = build_plan(
        gate="faults",
        nodes=2,
        state_path=state,
        evidence_root=tmp_path,
        parameters={"scenario": "owner-loss"},
        **IDENTITIES,
    )
    state.write_text(json.dumps({
        "schema": "emender-async-v21-qualification-state-v1",
        "payloads": {plan["payload_digest"]: {"status": "failed"}},
        "active_job": None,
    }))
    with pytest.raises(ValueError, match="unchanged failed payload"):
        build_plan(
            gate="faults",
            nodes=2,
            state_path=state,
            evidence_root=tmp_path,
            parameters={"scenario": "owner-loss"},
            **IDENTITIES,
        )
    state.write_text(json.dumps({
        "schema": "emender-async-v21-qualification-state-v1",
        "payloads": {},
        "active_job": {"job_id": "123", "payload_digest": "a" * 64},
    }))
    with pytest.raises(ValueError, match="active job"):
        build_plan(
            gate="faults",
            nodes=2,
            state_path=state,
            evidence_root=tmp_path,
            parameters={"scenario": "new"},
            **IDENTITIES,
        )


def test_v21_scale_close_includes_all_preclose_arrivals_not_only_two(
    tmp_path: Path,
):
    arrival = _write(tmp_path / "arrival.json", {
        "schema": "emender-async-v21-two-node-arrivals-v1",
        "status": "passed",
        "nodes": 2,
        "samples_ns": [10, 20, 30, 40],
    })
    stage = _write(tmp_path / "stage.json", {
        "schema": "emender-async-v21-two-node-stages-v1",
        "status": "passed",
        "nodes": 2,
        "close_to_latest_ns": [50, 60, 70, 80],
        "cadence_ns": [15, 20, 25, 30],
    })
    closure = {
        "schema": "emender-v21s17-scale-closure-v1",
        "ready_snapshot_source": "leased-ready-at-group-open",
        "arrival_evidence": {
            "path": arrival.name,
            "digest": canonical_digest(
                {key: value for key, value in json.loads(arrival.read_text()).items()
                 if key != "manifest_digest"}),
        },
        "stage_evidence": {
            "path": stage.name,
            "digest": canonical_digest(
                {key: value for key, value in json.loads(stage.read_text()).items()
                 if key != "manifest_digest"}),
        },
        "quantile": {"numerator": 3, "denominator": 4},
        "margin": {"numerator": 3, "denominator": 2},
        "include_all_complete_admissible_preclose": True,
        "close_on_q_min": False,
        "uses_launched_ranks": False,
        "wait_for_all_ready": False,
        "stable_diversity_floor": 2,
        "per_ready_worker_token_floor": 1_967_040,
    }
    derived = validate_scale_evidence(
        closure, evidence_root=tmp_path, ready_count=8)
    finite = V21ScaleClosure(
        open_time_ns=1_000,
        ready_snapshot={f"node-{index}": f"boot-{index}" for index in range(8)},
        derived=derived,
    )
    assert not finite.can_close(1_001, complete_workers={"node-0", "node-1"})
    finite.record("node-0", "boot-0", 100, exact_tokens=6_000_000)
    finite.record("node-1", "boot-1", 101, exact_tokens=6_000_000)
    finite.record("node-2", "boot-2", finite.close_time_ns - 1,
                  exact_tokens=6_000_000)
    frozen = finite.freeze(finite.close_time_ns)
    assert tuple(item["worker_id"] for item in frozen) == (
        "node-0", "node-1", "node-2")
    assert finite.close_time_ns < 10_000


def test_v21_scale_closure_rejects_launched_rank_and_unexplained_constant(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="evidence"):
        validate_scale_evidence({
            "schema": "emender-v21s17-scale-closure-v1",
            "ready_snapshot_source": "launched-ranks",
            "close_offset_ns": 420_000_000_000,
            "uses_launched_ranks": True,
            "close_on_q_min": True,
        }, evidence_root=tmp_path, ready_count=64)


@pytest.mark.parametrize(
    ("nodes", "predecessor_nodes"),
    ((4, 2), (8, 4), (16, 8), (32, 16), (64, 32), (256, 64)),
)
def test_v21_each_scale_rung_accepts_only_its_exact_predecessor(
    tmp_path: Path, nodes: int, predecessor_nodes: int,
):
    arrival = _write(tmp_path / "arrival.json", {
        "schema": "emender-async-v21-two-node-arrivals-v1",
        "status": "passed",
        "nodes": 2,
        "samples_ns": [10, 20, 30],
    })
    stage = _write(tmp_path / "stage.json", {
        "schema": "emender-async-v21-two-node-stages-v1",
        "status": "passed",
        "nodes": 2,
        "close_to_latest_ns": [40, 50, 60],
        "cadence_ns": [15, 20, 25],
    })
    closure = {
        "schema": "emender-v21s17-scale-closure-v1",
        "ready_snapshot_source": "leased-ready-at-group-open",
        "arrival_evidence": {
            "path": arrival.name,
            "digest": json.loads(arrival.read_text())["manifest_digest"],
        },
        "stage_evidence": {
            "path": stage.name,
            "digest": json.loads(stage.read_text())["manifest_digest"],
        },
        "quantile": {"numerator": 1, "denominator": 1},
        "margin": {"numerator": 3, "denominator": 2},
        "include_all_complete_admissible_preclose": True,
        "close_on_q_min": False,
        "uses_launched_ranks": False,
        "wait_for_all_ready": False,
        "stable_diversity_floor": 2,
        "per_ready_worker_token_floor": 1,
    }
    authorization = _write(tmp_path / "authorization.json", {
        "schema": "emender-async-v21-scale-authorization-v1",
        "status": "passed",
        "authorized_nodes": nodes,
        "reviewed_ready_snapshot_size": nodes,
        "identities": IDENTITIES,
        "review_signature": "signed-for-test",
        "closure": closure,
    })
    predecessor = _write(tmp_path / "predecessor.json", {
        "schema": "emender-async-v21-rung-pass-v1",
        "status": "passed",
        "nodes": predecessor_nodes,
        "identities": IDENTITIES,
        "review_signature": "signed-for-test",
    })
    plan = build_plan(
        gate="scale",
        nodes=nodes,
        state_path=tmp_path / "state.json",
        evidence_root=tmp_path,
        parameters=SCALE_PARAMETERS,
        authorization_path=authorization,
        predecessor_path=predecessor,
        allow_test_signatures=True,
        **IDENTITIES,
    )
    assert plan["scheduler"]["Nodes"] == nodes
    exported = next(
        item for item in plan["command"] if item.startswith("--export="))
    assert "ASYNC_V21_SCALE_AUTHORIZATION_DIGEST=" in exported
    assert "ASYNC_V21_PRIOR_RUNG_PASS_DIGEST=" in exported
    assert "ASYNC_V21_SCALE_CLOSE_OFFSET_NS=" in exported


def test_v21_scale_rejects_two_node_early_close_parameters(tmp_path: Path):
    with pytest.raises(ValueError, match="Q_min early close"):
        build_plan(
            gate="scale",
            nodes=4,
            state_path=tmp_path / "state.json",
            evidence_root=tmp_path,
            parameters={
                "close_on_q_min": True,
                "uses_launched_ranks": False,
                "wait_for_all_ready": False,
            },
            **IDENTITIES,
        )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True,
    ).strip()


def _execution_identity_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "identity-repo"
    paths = {
        "scripts/frontier/controller.py": "controller-v1\n",
        "native/dataplane/protocol.cpp": "wire-v1\n",
        "src/runtime.cpp": "runtime-v1\n",
        "docs/RESILIENT_DILOCO_COMPUTE_POOL.md": "policy-v1\n",
        "configs/schema.json": '{"schema":"v1"}\n',
        "include/emender/ndp.h": "#define NDP_ABI 1\n",
        "data/training-index.json": '{"shard":"a"}\n',
        "tokenizers/p50k.sha256": "tokenizer-v1\n",
        "configs/frontier/e97_async_256.yaml": '{"seed":"v1"}\n',
        "docs/validation/pass.md": "evidence-v1\n",
        "reports/frontier/pass.json": '{"passed":true}\n',
        "logs/frontier/job.out": "job evidence\n",
    }
    for relative, content in paths.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "initial"],
        check=True,
    )
    return repo


def test_execution_source_identity_ignores_only_reviewed_evidence_paths(
    tmp_path: Path,
):
    repo = _execution_identity_repo(tmp_path)
    initial = _source_digest(repo)
    assert initial["schema"] == EXECUTION_SOURCE_SCHEMA
    assert tuple(initial["evidence_only_path_prefixes"]) == (
        EVIDENCE_ONLY_PATH_PREFIXES
    )

    for relative in (
        "docs/validation/pass.md",
        "reports/frontier/pass.json",
        "logs/frontier/job.out",
    ):
        target = repo / relative
        original = target.read_text()
        target.write_text(original + "retained evidence\n")
        assert _source_digest(repo)["digest"] == initial["digest"]
        target.write_text(original)

    for relative in (
        "scripts/frontier/controller.py",
        "native/dataplane/protocol.cpp",
        "src/runtime.cpp",
        "docs/RESILIENT_DILOCO_COMPUTE_POOL.md",
        "configs/schema.json",
        "include/emender/ndp.h",
        "data/training-index.json",
        "tokenizers/p50k.sha256",
        "configs/frontier/e97_async_256.yaml",
    ):
        target = repo / relative
        original = target.read_text()
        target.write_text(original + "executable drift\n")
        assert _source_digest(repo)["digest"] != initial["digest"], relative
        target.write_text(original)

    for relative in (
        "docs/validation/pass.md",
        "reports/frontier/pass.json",
        "logs/frontier/job.out",
    ):
        (repo / relative).write_text("new evidence commit\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "evidence only"],
        check=True,
    )
    assert _source_digest(repo, revision="HEAD")["digest"] == initial["digest"]
    assert _source_digest(repo, revision="HEAD~1")["digest"] == initial["digest"]


def _transaction_plan(tmp_path: Path) -> dict[str, object]:
    return build_plan(
        gate="faults",
        nodes=2,
        state_path=tmp_path / "state.json",
        evidence_root=tmp_path / "evidence",
        parameters={"scenario": "owner-loss"},
        **IDENTITIES,
    )


def test_collector_registration_failure_never_releases_held_payload(
    monkeypatch, tmp_path: Path,
):
    plan = _transaction_plan(tmp_path)
    calls: list[list[str]] = []
    collector_attempts = 0

    def run(command, **kwargs):
        nonlocal collector_attempts
        command = [str(item) for item in command]
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sacct":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sbatch" and "--hold" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="71001\n", stderr="")
        if command[0] == "sbatch":
            collector_attempts += 1
            if collector_attempts == 1:
                raise subprocess.CalledProcessError(
                    1, command, output="", stderr="collector rejected")
            return subprocess.CompletedProcess(
                command, 0, stdout="71002\n", stderr="")
        if command[:2] == ["scontrol", "release"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(
        "scripts.frontier.run_async_v21_qualification.subprocess.run", run)
    with pytest.raises(subprocess.CalledProcessError):
        submit_plan(plan)

    assert not any(command[:2] == ["scontrol", "release"] for command in calls)
    payload = json.loads((tmp_path / "state.json").read_text())["payloads"][
        plan["payload_digest"]
    ]
    assert payload["job_id"] == "71001"
    assert payload["status"] == "held"
    assert payload["collector"]["status"] == "registration-failed"
    held_submissions = [
        command for command in calls
        if command[0] == "sbatch" and "--hold" in command
    ]
    assert len(held_submissions) == 1

    # A retry reconciles the durable held job, registers one successful
    # collector, and only then releases.  It never submits the payload again.
    assert submit_plan(plan) == "71001"
    assert len([
        command for command in calls
        if command[0] == "sbatch" and "--hold" in command
    ]) == 1
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["payloads"][plan["payload_digest"]]["status"] == "released"
    assert saved["payloads"][plan["payload_digest"]]["collector"]["job_id"] == (
        "71002"
    )
    scheduler = saved["payloads"][plan["payload_digest"]]["collector"]["scheduler"]
    assert scheduler == {
        "Account": "bif148",
        "Nodes": 1,
        "Partition": "batch",
        "QOS": "normal",
    }


def test_repeated_reconciliation_submits_neither_payload_nor_collector_twice(
    monkeypatch, tmp_path: Path,
):
    plan = _transaction_plan(tmp_path)
    calls: list[list[str]] = []

    def run(command, **kwargs):
        command = [str(item) for item in command]
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sacct":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sbatch" and "--hold" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="72001\n", stderr="")
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout="72002\n", stderr="")
        if command[:2] == ["scontrol", "release"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(
        "scripts.frontier.run_async_v21_qualification.subprocess.run", run)
    assert submit_plan(plan) == "72001"
    assert submit_plan(plan) == "72001"
    assert len([
        command for command in calls
        if command[0] == "sbatch" and "--hold" in command
    ]) == 1
    assert len([
        command for command in calls
        if command[0] == "sbatch" and "--hold" not in command
    ]) == 1
    assert len([
        command for command in calls
        if command[:2] == ["scontrol", "release"]
    ]) == 1
    payload_submit = next(
        index for index, command in enumerate(calls)
        if command[0] == "sbatch" and "--hold" in command)
    collector_submit = next(
        index for index, command in enumerate(calls)
        if command[0] == "sbatch" and "--hold" not in command)
    release = next(
        index for index, command in enumerate(calls)
        if command[:2] == ["scontrol", "release"])
    assert payload_submit < collector_submit < release
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["payloads"][plan["payload_digest"]]["status"] == "released"
    assert saved["payloads"][plan["payload_digest"]]["collector"]["job_id"] == (
        "72002"
    )
    collector_command = calls[collector_submit]
    assert "--account=bif148" in collector_command
    assert "--qos=normal" in collector_command
    assert "--qos=debug" not in collector_command


@pytest.mark.parametrize("status", ("terminal", "retired"))
def test_terminal_and_retired_payloads_are_recognized_without_resubmission(
    monkeypatch, tmp_path: Path, status: str,
):
    plan = _transaction_plan(tmp_path)
    state = {
        "schema": "emender-async-v21-qualification-state-v2",
        "payloads": {
            plan["payload_digest"]: {
                "status": status,
                "job_id": "73001",
                "collector": {"job_id": "73002", "status": "completed"},
            },
        },
        "active_job": None,
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(
        "scripts.frontier.run_async_v21_qualification.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal/retired reconciliation touched scheduler")),
    )
    assert submit_plan(plan) == "73001"


@pytest.mark.parametrize("status", ("queued", "running"))
def test_queued_and_running_payloads_are_recognized_without_resubmission(
    monkeypatch, tmp_path: Path, status: str,
):
    plan = _transaction_plan(tmp_path)
    state = {
        "schema": "emender-async-v21-qualification-state-v2",
        "payloads": {
            plan["payload_digest"]: {
                "status": status,
                "job_id": "73101",
                "collector": {"job_id": "73102", "status": "registered"},
            },
        },
        "active_job": {
            "status": status,
            "job_id": "73101",
            "payload_digest": plan["payload_digest"],
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(
        "scripts.frontier.run_async_v21_qualification.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("queued/running reconciliation touched scheduler")),
    )
    assert submit_plan(plan) == "73101"


def test_fake_frontier_rejects_collector_without_account(tmp_path: Path):
    fake = Path(__file__).resolve().parent / "support/fake_async_v21_scheduler.py"
    sbatch = tmp_path / "sbatch"
    sbatch.symlink_to(fake)
    environment = {
        **os.environ,
        "FAKE_ASYNC_V21_SCHEDULER_STATE": str(tmp_path / "scheduler.json"),
    }
    rejected = subprocess.run(
        [
            str(sbatch), "--parsable", "--nodes=1", "--partition=batch",
            "--qos=normal", "--dependency=afterany:81000",
            "--job-name=collector-without-account",
        ],
        env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode != 0
    assert "requires an explicit account" in rejected.stderr
    assert not (tmp_path / "scheduler.json").exists()


def test_fake_frontier_rejects_second_submitted_debug_qos_job(
    tmp_path: Path,
):
    fake = Path(__file__).resolve().parent / "support/fake_async_v21_scheduler.py"
    sbatch = tmp_path / "sbatch"
    sbatch.symlink_to(fake)
    scheduler_state = tmp_path / "scheduler.json"
    environment = {
        **os.environ,
        "FAKE_ASYNC_V21_SCHEDULER_STATE": str(scheduler_state),
    }
    payload = subprocess.run(
        [
            str(sbatch), "--parsable", "--hold", "--nodes=2",
            "--partition=batch", "--qos=debug", "--job-name=held-payload",
        ],
        env=environment, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rejected = subprocess.run(
        [
            str(sbatch), "--parsable", "--account=bif148", "--nodes=1",
            "--partition=batch", "--qos=debug",
            f"--dependency=afterany:{payload.stdout.strip()}",
            "--job-name=debug-collector",
        ],
        env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode != 0
    assert "QOSMaxSubmitJobPerUserLimit" in rejected.stderr
    state = json.loads(scheduler_state.read_text())
    assert len(state["jobs"]) == 1
    held = next(iter(state["jobs"].values()))
    assert held["released"] is False
    assert held["qos"] == "debug"


def test_fake_scheduler_collector_survives_worker_death_and_is_exactly_once(
    tmp_path: Path,
):
    repo = Path(__file__).resolve().parents[1]
    fake_scheduler = (
        repo / "tests/support/fake_async_v21_scheduler.py").resolve()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name in ("sbatch", "scontrol", "squeue", "sacct"):
        (fake_bin / name).symlink_to(fake_scheduler)
    scheduler_state = tmp_path / "fake-slurm.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join([
            "import json",
            "import time",
            "from pathlib import Path",
            "from scripts.frontier.run_async_v21_qualification import (",
            "    SEED_SHA256, build_plan, submit_plan)",
            f"root = Path({str(tmp_path)!r})",
            "plan = build_plan(",
            "    gate='faults', nodes=2, state_path=root/'state.json',",
            "    evidence_root=root/'evidence',",
            "    source_digest='1'*64, policy_digest='2'*64,",
            "    bundle_digest='3'*64, seed_digest=SEED_SHA256,",
            "    launcher_digest='5'*64, parameters={'scenario':'death'})",
            "semantic = root/'semantic-verdict.json'",
            "semantic.write_text(json.dumps({'status':'passed'})+'\\n')",
            "plan['collector']['semantic_verdict'] = str(semantic)",
            "submit_plan(plan)",
            "while True: time.sleep(1)",
        ]) + "\n"
    )
    environment = dict(os.environ)
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "FAKE_ASYNC_V21_SCHEDULER_STATE": str(scheduler_state),
        "FAKE_ASYNC_V21_PYTHON": sys.executable,
        "PYTHONPATH": str(repo),
        "USER": environment.get("USER", "fake-user"),
    })
    monitor = subprocess.Popen(
        [sys.executable, str(worker)],
        cwd=repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    evidence = tmp_path / "evidence"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if scheduler_state.exists():
            scheduler = json.loads(scheduler_state.read_text())
            payloads = [
                item for item in scheduler.get("jobs", {}).values()
                if item.get("kind") == "payload"
            ]
            if payloads and payloads[0].get("released"):
                break
        time.sleep(0.05)
    else:
        monitor.kill()
        stdout, stderr = monitor.communicate()
        pytest.fail(f"payload was not released\nstdout={stdout}\nstderr={stderr}")
    os.kill(monitor.pid, signal.SIGKILL)
    monitor.wait(timeout=10)

    verdicts = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        verdicts = list(evidence.rglob("terminal-verdict.json"))
        if verdicts:
            break
        time.sleep(0.05)
    assert len(verdicts) == 1
    verdict_path = verdicts[0]
    verdict = json.loads(verdict_path.read_text())
    assert verdict["verdict"] == "passed"
    assert verdict["passed"] is True
    assert verdict["scheduler"]["partition"] == "batch"
    assert verdict["scheduler"]["qos"] == "debug"
    assert verdict["scheduler"]["exit_code"] == "0:0"
    assert verdict["scheduler"]["derived_exit_code"] == "0:0"
    assert verdict["validator_inputs"]["payload"]["sha256"]
    assert verdict["validator_inputs"]["semantic_verdict"]["sha256"]
    assert verdict["logs"]["stdout"]["sha256"]
    assert verdict["logs"]["stderr"]["sha256"]

    scheduler = json.loads(scheduler_state.read_text())
    assert len([
        item for item in scheduler["jobs"].values()
        if item["kind"] == "payload"
    ]) == 1
    collectors = [
        item for item in scheduler["jobs"].values()
        if item["kind"] == "collector"
    ]
    assert len(collectors) == 1
    assert collectors[0]["account"] == "bif148"
    assert collectors[0]["qos"] == "normal"
    durable_state = json.loads((tmp_path / "state.json").read_text())
    durable_payload = durable_state["payloads"][verdict["payload_digest"]]
    assert durable_state["active_job"] is None
    assert durable_payload["status"] == "terminal"
    assert durable_payload["collector"]["job_id"] == collectors[0]["job_id"]
    assert durable_payload["collector"]["status"] == "completed"
    assert durable_payload["collector"]["scheduler"] == {
        "Account": collectors[0]["account"],
        "Nodes": 1,
        "Partition": collectors[0]["partition"],
        "QOS": collectors[0]["qos"],
    }
    before = (
        verdict_path.stat().st_mtime_ns,
        hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
    )
    subprocess.run(
        collectors[0]["wrap_argv"],
        cwd=repo,
        env=environment,
        check=True,
    )
    after = (
        verdict_path.stat().st_mtime_ns,
        hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
    )
    assert after == before
