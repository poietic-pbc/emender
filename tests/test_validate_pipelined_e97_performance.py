from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/validate_pipelined_e97_performance.py"
SPEC = importlib.util.spec_from_file_location("pipelined_perf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _records(
    *,
    trainer_start: int = 0,
    trainer_count: int = 16,
    cadence: float = 101.0,
    background: bool = True,
):
    values = [{
        "stage": "async_v2_policy",
        "policy_id": "async-decoupled-v2.0-exp",
        "tau_hard": 6,
        "tau_target": 2,
        "sigma_hard": 8,
        "sigma_target": 2,
        "k_local_steps": 40,
        "sealed_descriptor_capacity": 1,
        "mutable_interval_capacity": 1,
        "result_mailbox_capacity": 1,
        "result_staging_capacity": 1,
        "allocation_fence": 7,
    }]
    for ordinal in range(trainer_start, trainer_start + trainer_count):
        identity = f"node-{ordinal // 8}-trainer-{ordinal % 8}"
        for window in range(12):  # two warm-up + ten measured
            start = window * cadence
            for step in range(40):
                began = start + step * 2.5
                values.extend([
                    {
                        "identity": identity,
                        "local_window": window,
                        "generation": window,
                        "phase": "optimizer_step_start",
                        "monotonic_s": began,
                        "timestamp": 1000.0 + began,
                        "policy_id": "async-decoupled-v2.0-exp",
                        "local_model_basis": "worker_local",
                        "applied_anchor_version": max(0, window - 1),
                    },
                    {
                        "identity": identity,
                        "local_window": window,
                        "generation": window,
                        "phase": "optimizer_step_end",
                        "monotonic_s": began + 2.4,
                        "timestamp": 1002.4 + began,
                        "policy_id": "async-decoupled-v2.0-exp",
                    },
                ])
            values.append({
                "identity": identity,
                "stage": "safe_boundary_apply",
                "local_window": window,
                "allocation_fence": 7,
                "applied_anchor_version": window,
                "known_global_version": window,
                "result_version": window,
                "anchor_lag_before_apply": 0,
                "result_version_lag_at_apply": 0,
                "speculative_window_lag": 1,
                "result_digest": f"{window + 1:064x}",
                "manifest_digest": f"{window + 2:064x}",
                "reload_verified": True,
                "latest_cas_verified": True,
            })
        values.append({
            "identity": identity,
            "stage": "async_v2_bounds",
            "sealed_descriptor_capacity": 1,
            "sealed_descriptor_high_water": 1,
            "mutable_interval_capacity": 1,
            "mutable_interval_high_water": 1,
            "mutable_window_high_water": 2,
            "result_mailbox_capacity": 1,
            "result_mailbox_high_water": 1,
            "result_staging_capacity": 1,
            "result_staging_high_water": 1,
            "native_owned_seconds_max": 0.02,
            "resident_admission_bytes": 64_001_671_648,
            "resident_limit_bytes": 68_719_476_736,
            "resident_headroom_bytes": 4_717_805_088,
            "pause_count": 0,
            "drop_count": 0,
            "python_dense_socket_bytes": 0,
            "lustre_dense_hot_path_bytes": 0,
            "model_build": 1,
            "optimizer_build": 1,
            "data_iterator_build": 1,
            "windows_completed": 12,
        })
    if background:
        for version in range(1, 12):
            for offset, stage in enumerate((
                "native_local_reduction",
                "native_owner_redistribution",
                "native_trainer_apply",
                "checkpoint_publication",
                "control_handoff_integrity",
            )):
                began = version * cadence + 10.0 + offset
                ended = began + 5.0
                values.append({
                    "identity": "node-0-manager",
                    "generation": version - 1,
                    "stage": stage,
                    "elapsed_s": ended - began,
                    "timestamp": 1000.0 + ended,
                    "within_slo": True,
                    "policy_id": "async-decoupled-v2.0-exp",
                    "allocation_fence": 7,
                    "base_global_version": max(0, version - 1),
                    "commit_global_version": version,
                    "commit_lag": min(1, version),
                    "contribution_digest": f"{version + 10:064x}",
                    "exact_tokens": 3_934_080,
                    "aggregation_weight": 3_934_080 * (7 - min(1, version)),
                })
            values.append({
                "identity": "node-0-manager",
                "stage": "async_v2_correctness",
                "policy_id": "async-decoupled-v2.0-exp",
                "allocation_fence": 7,
                "base_global_version": version - 1,
                "result_version": version,
                "freeze_to_latest_s": 50.0,
                "checkpoint_bytes": 7_719_680_116,
                "manifest_digest": f"{version + 30:064x}",
                "reload_verified": True,
                "latest_cas_verified": True,
            })
    return values


def test_semantic_validator_accepts_true_decoupling_and_bounded_lag():
    result = MODULE.validate(_records())
    assert result["status"] == "passed"
    assert result["schema"] == "emender-async-decoupled-e97-performance-v2"
    assert result["foreground_control_plane_idle_fraction"] < 0.10
    assert result["steady_state_cadence_multiple"] <= 1.25
    assert result["measured_trainers"] == 16
    assert result["measured_windows_per_trainer"] == 10
    assert result["lag"]["commit_max"] <= 2
    assert result["lag"]["anchor_max"] <= 2
    assert result["lag"]["speculative_max"] <= 2
    assert result["overlaps"]


def test_semantic_validator_accepts_latest_only_non_barrier_applications():
    records = [
        value for value in _records()
        if not (
            value.get("stage") == "safe_boundary_apply"
            and int(value["local_window"]) % 2
        )
    ]
    result = MODULE.validate(records)
    assert result["status"] == "passed"
    assert result["lag"]["speculative_max"] <= 2


def test_semantic_validator_rejects_false_tau_zero_labeling():
    records = _records()
    records[0]["policy_id"] = "strict-tau-zero-v1"
    records[0]["tau_hard"] = 0
    with pytest.raises(ValueError, match="false tau=0"):
        MODULE.validate(records)


def test_semantic_validator_rejects_missing_or_unbounded_queue_evidence():
    records = _records()
    for value in records:
        if value.get("stage") == "async_v2_bounds":
            value["sealed_descriptor_capacity"] = 2
            break
    with pytest.raises(ValueError, match="bounded queue"):
        MODULE.validate(records)

    records = _records()
    bounds = next(
        value for value in records
        if value.get("stage") == "async_v2_bounds")
    bounds["resident_limit_bytes"] = 64_001_671_647
    bounds["resident_headroom_bytes"] = -1
    with pytest.raises(ValueError, match="resident formula"):
        MODULE.validate(records)


def test_semantic_validator_rejects_per_window_model_rebootstrap():
    records = _records()
    bounds = next(
        value for value in records
        if value.get("stage") == "async_v2_bounds")
    bounds["model_build"] = 12
    with pytest.raises(ValueError, match="persistent resident session"):
        MODULE.validate(records)


def test_semantic_validator_rejects_training_lane_stall_even_when_background_is_slow():
    records = _records(cadence=130.0)
    with pytest.raises(ValueError, match="cadence|idle"):
        MODULE.validate(records)


def test_semantic_validator_rejects_missing_true_versioned_background_overlap():
    with pytest.raises(ValueError, match="versioned background"):
        MODULE.validate(_records(background=False))


def test_semantic_validator_rejects_unverifiable_or_stale_application():
    records = _records()
    apply = next(value for value in records
                 if value.get("stage") == "safe_boundary_apply"
                 and int(value["local_window"]) >= 2)
    apply["latest_cas_verified"] = False
    with pytest.raises(ValueError, match="unverifiable application"):
        MODULE.validate(records)

    records = _records()
    apply = next(value for value in records
                 if value.get("stage") == "safe_boundary_apply"
                 and int(value["local_window"]) >= 2)
    apply["anchor_lag_before_apply"] = 7
    with pytest.raises(ValueError, match="hard lag"):
        MODULE.validate(records)


def test_semantic_validator_requires_exact_k40_and_full_16_trainer_sample():
    records = _records()
    records.pop(next(
        index for index, value in enumerate(records)
        if value.get("phase") == "optimizer_step_start"
        and int(value["local_window"]) >= 2))
    with pytest.raises(ValueError, match="exact K40"):
        MODULE.validate(records)

    with pytest.raises(ValueError, match="16 trainers"):
        MODULE.validate(_records(trainer_count=15))


def test_post_supervisor_retained_evidence_recursively_harvests_both_nodes(tmp_path):
    retained = tmp_path / "run" / "retained-evidence"
    for node, values in (
        ("node-0", _records(trainer_start=0, trainer_count=8)),
        ("node-1", _records(trainer_start=8, trainer_count=8)),
    ):
        telemetry = retained / node / "telemetry"
        telemetry.mkdir(parents=True)
        (telemetry / f"{node}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )

    harvested = MODULE._records(retained)
    result = MODULE.validate(harvested)
    assert result["status"] == "passed"
    assert result["measured_trainers"] == 16
    assert {item["identity"] for item in result["overlaps"]} == {
        f"node-{ordinal // 8}-trainer-{ordinal % 8}" for ordinal in range(16)
    }
