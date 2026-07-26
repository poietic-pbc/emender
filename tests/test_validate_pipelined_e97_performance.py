from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/validate_pipelined_e97_performance.py"
SPEC = importlib.util.spec_from_file_location("pipelined_perf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _causal_id(identity: str, generation: int) -> str:
    return hashlib.sha256(
        f"{identity}|{generation}".encode("utf-8")).hexdigest()


def test_records_include_retained_control_json_policy(tmp_path):
    telemetry = tmp_path / "retained-evidence" / "node-0" / "telemetry"
    control = tmp_path / "retained-evidence" / "node-0" / "control"
    telemetry.mkdir(parents=True)
    control.mkdir(parents=True)
    (telemetry / "trainer.jsonl").write_text(
        '{"stage":"async_v21_bounds","identity":"node-0-trainer-0"}\n')
    (control / "production-pipeline-node-0-trainer-0.json").write_text(
        '{"stage":"async_v21_policy","policy_id":'
        '"async-decoupled-v2.1-simple"}\n')

    records = MODULE._records(tmp_path / "retained-evidence")

    assert {record["stage"] for record in records} == {
        "async_v21_bounds",
        "async_v21_policy",
    }


def _records(
    *,
    trainer_start: int = 0,
    trainer_count: int = 16,
    cadence: float = 101.0,
    background: bool = True,
):
    values = [{
        "stage": "async_v21_policy",
        "policy_id": "async-decoupled-v2.1-simple",
        "policy_schema": "emender-async-policy-v2.1",
        "contribution_schema": "emender-native-e97-submission-v2.1",
        "manifest_schema": "emender-native-e97-generation-v2.1",
        "checkpoint_schema": "emender-async-v21-reference-checkpoint-v1",
        "native_abi": 0x00020001,
        "wire_protocol_major": 2,
        "wire_protocol_minor": 1,
        "max_commit_lag": 2,
        "max_anchor_lag": 2,
        "max_result_lag": 2,
        "max_speculative_windows": 2,
        "eta_outer": 1.0,
        "k_local_steps": 40,
        "owned_descriptor_capacity": 1,
        "mutable_interval_capacity": 1,
        "result_mailbox_capacity": 1,
        "result_staging_capacity": 1,
        "allocation_fence": 7,
    }]
    fixture_node = trainer_start // 8
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
                        "policy_id": "async-decoupled-v2.1-simple",
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
                        "policy_id": "async-decoupled-v2.1-simple",
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
            "stage": "async_v21_bounds",
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
            generation = version - 1
            manager_identity = f"node-{fixture_node}-manager"
            for offset, stage in enumerate((
                "native_local_reduction",
                "native_owner_redistribution",
                "native_trainer_apply",
                "checkpoint_publication",
                "control_handoff_integrity",
                "fenced_atomic_commit",
            )):
                began = version * cadence + 10.0 + offset
                ended = began + 5.0
                values.append({
                    "identity": manager_identity,
                    "generation": generation,
                    "stage": stage,
                    "elapsed_s": ended - began,
                    "monotonic_start_s": began,
                    "monotonic_end_s": ended,
                    "timestamp": 1000.0 + ended,
                    "within_slo": True,
                    "policy_id": "async-decoupled-v2.1-simple",
                    "allocation_fence": 7,
                    "base_global_version": max(0, version - 1),
                    "commit_global_version": version,
                    "commit_lag": min(1, version),
                    "contribution_digest": f"{version + 10:064x}",
                    "exact_tokens": 3_934_080,
                    **({
                        "causal_phase": {
                            "native_local_reduction": "aggregation",
                            "native_owner_redistribution": "publish_network",
                            "native_trainer_apply": "apply_swap",
                            "checkpoint_publication": "checkpoint",
                        }[stage],
                        "causal_id": _causal_id(
                            manager_identity, generation),
                        "foreground_component_s": (
                            ended - began
                            if stage == "native_trainer_apply" else 0.0),
                    } if stage in {
                        "native_local_reduction",
                        "native_owner_redistribution",
                        "native_trainer_apply",
                        "checkpoint_publication",
                    } else {}),
                    **({
                        "foreground_interruption": "verified_result_apply",
                        "foreground_blocking": True,
                        "atomic_live_model_swap": True,
                        "phase_scope": "trainer_lane",
                        "policy_bound_s": 60.0,
                    } if stage == "native_trainer_apply" else {}),
                    **({
                        "foreground_blocking": False,
                    } if stage == "checkpoint_publication" else {}),
                })
            node_apply_started = version * cadence + 20.0
            node_apply_ended = node_apply_started + 5.0
            values.append({
                "identity": manager_identity,
                "generation": generation,
                "stage": "native_node_apply_swap",
                "elapsed_s": node_apply_ended - node_apply_started,
                "monotonic_start_s": node_apply_started,
                "monotonic_end_s": node_apply_ended,
                "timestamp": 1000.0 + node_apply_ended,
                "within_slo": True,
                "causal_phase": "apply_swap",
                "causal_id": _causal_id(manager_identity, generation),
                "foreground_component_s":
                    node_apply_ended - node_apply_started,
                "foreground_interruption": "verified_result_apply",
                "foreground_blocking": True,
                "atomic_live_model_swap": True,
                "phase_scope": "node_all_eight",
                "trainer_count": 8,
                "policy_bound_s": 60.0,
            })
            trainer_identity = f"node-{fixture_node}-trainer-0"
            for offset, stage in enumerate((
                "async_v21_endpoint_snapshot",
                "async_v21_snapshot_admission",
                "native_direct_memfd",
                "native_owner_contribution",
                "async_v21_checkpoint_write",
                "async_v21_checkpoint_hash",
                "async_v21_result_readiness",
            ), start=10):
                began = version * cadence + offset
                ended = began + 0.25
                values.append({
                    "identity": trainer_identity,
                    "generation": generation,
                    "stage": stage,
                    "elapsed_s": ended - began,
                    "monotonic_start_s": began,
                    "monotonic_end_s": ended,
                    "timestamp": 1000.0 + ended,
                    "within_slo": True,
                    "causal_phase": {
                        "async_v21_endpoint_snapshot": "freeze_snapshot",
                        "async_v21_snapshot_admission": "snapshot_admission",
                        "native_direct_memfd": "publish_network",
                        "native_owner_contribution": "publish_network",
                        "async_v21_checkpoint_write": "checkpoint",
                        "async_v21_checkpoint_hash": "checkpoint",
                        "async_v21_result_readiness": "result_wait",
                    }[stage],
                    "causal_id": _causal_id(trainer_identity, generation),
                    "foreground_component_s": (
                        ended - began
                        if stage in {
                            "async_v21_endpoint_snapshot",
                            "async_v21_snapshot_admission",
                        } else 0.0),
                    **({
                        "foreground_interruption": "snapshot_capture",
                        "phase_scope": "trainer_snapshot",
                        "snapshot_coherent": True,
                        "snapshot_slots": 2,
                        "live_model_read_after_snapshot": False,
                    } if stage == "async_v21_endpoint_snapshot" else {}),
                    **({
                        "foreground_interruption": "snapshot_admission",
                        "phase_scope": "snapshot_owned",
                        "owned": True,
                        "immutable_snapshot": True,
                        "mutable_training_resumed": True,
                        "foreground_pause_s": 0.5,
                        "policy_bound_s": 1.0,
                    } if stage == "async_v21_snapshot_admission" else {}),
                    **({
                        "immutable_snapshot": True,
                        "mutable_training_already_resumed": True,
                        "foreground_blocking": False,
                    } if stage == "native_direct_memfd" else {}),
                    **({
                        "foreground_blocking": False,
                    } if stage in {
                        "async_v21_checkpoint_write",
                        "async_v21_checkpoint_hash",
                    } else {}),
                    **({
                        "foreground_blocking": False,
                        "mutable_training_active": True,
                    } if stage == "async_v21_result_readiness" else {}),
                })
            values.append({
                "identity": "node-0-manager",
                "stage": "async_v21_correctness",
                "policy_id": "async-decoupled-v2.1-simple",
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
    assert result["schema"] == "emender-async-decoupled-e97-performance-v2.1"
    assert result["foreground_control_plane_idle_fraction"] < 0.10
    assert result["steady_state_cadence_multiple"] <= 1.25
    assert result["measured_trainers"] == 16
    assert result["measured_windows_per_trainer"] == 10
    assert result["lag"]["commit_max"] <= 2
    assert result["lag"]["anchor_max"] <= 2
    assert result["lag"]["speculative_max"] <= 2
    assert result["atomic_commits"] >= 10
    assert result["overlaps"]
    assert set(result["causal_phase_seconds"]) == MODULE.CAUSAL_PHASES


def test_semantic_validator_rejects_missing_causal_phase_timing():
    records = _records()
    for value in records:
        if value.get("causal_phase") == "checkpoint":
            value.pop("causal_phase")
            value.pop("causal_id")
            value.pop("foreground_component_s")
    with pytest.raises(ValueError, match="causal pipeline phases"):
        MODULE.validate(records)


def test_semantic_validator_rejects_mismatched_causal_identity():
    records = _records()
    causal = next(
        value for value in records
        if value.get("causal_phase") == "snapshot_admission")
    causal["causal_id"] = "f" * 64
    with pytest.raises(ValueError, match="mismatched causal identity"):
        MODULE.validate(records)


def test_semantic_validator_rejects_nonzero_foreground_result_wait():
    records = _records()
    result_wait = next(
        value for value in records
        if value.get("causal_phase") == "result_wait")
    result_wait["foreground_component_s"] = 0.01
    with pytest.raises(ValueError, match="result_wait"):
        MODULE.validate(records)


@pytest.mark.parametrize(
    ("phase_scope", "pause", "match"),
    [
        ("snapshot_owned", 1.01, "snapshot capture/admission"),
        ("node_all_eight", 60.01, "all-eight apply/swap"),
    ],
)
def test_semantic_validator_rejects_every_event_pause_bound(
    phase_scope, pause, match,
):
    records = _records()
    value = next(
        item for item in records if item.get("phase_scope") == phase_scope)
    if phase_scope == "snapshot_owned":
        value["foreground_pause_s"] = pause
    else:
        value["elapsed_s"] = pause
        value["monotonic_end_s"] = value["monotonic_start_s"] + pause
    with pytest.raises(ValueError, match=match):
        MODULE.validate(records)


def test_overlap_gate_rejects_200_second_bursty_alternation_despite_checkpoints_and_median():
    records = _records()
    identity = "node-0-trainer-0"
    final_window = 11
    for value in records:
        if (
            value.get("identity") == identity
            and int(value.get("local_window", -1)) == final_window
            and value.get("phase") in {
                "optimizer_step_start", "optimizer_step_end",
            }
        ):
            value["monotonic_s"] += 200.0
            value["timestamp"] += 200.0
    # Preserve real overlap for the delayed window so the failure is
    # specifically the every-event foreground tail, not missing background.
    begin = final_window * 101.0 + 205.0
    end = begin + 5.0
    records.append({
        "identity": "node-0-manager",
        "generation": final_window - 1,
        "stage": "native_local_reduction",
        "elapsed_s": end - begin,
        "monotonic_start_s": begin,
        "monotonic_end_s": end,
        "timestamp": 1000.0 + end,
        "within_slo": True,
        "policy_id": "async-decoupled-v2.1-simple",
        "allocation_fence": 7,
        "base_global_version": final_window - 1,
        "commit_global_version": final_window,
        "commit_lag": 1,
        "contribution_digest": f"{final_window + 10:064x}",
        "exact_tokens": 3_934_080,
        "causal_phase": "aggregation",
        "causal_id": _causal_id("node-0-manager", final_window - 1),
        "foreground_component_s": 0.0,
    })
    with pytest.raises(ValueError, match="every-event foreground tail"):
        MODULE.validate(records)


def test_semantic_validator_requires_ten_distinct_atomic_commits():
    records = _records()
    atomic = [
        value for value in records
        if value.get("stage") == "fenced_atomic_commit"
    ]
    for value in atomic[9:]:
        records.remove(value)
    with pytest.raises(ValueError, match="ten distinct atomic commits"):
        MODULE.validate(records)


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


def test_semantic_validator_rejects_v20_identity():
    records = _records()
    records[0]["policy_id"] = "async-decoupled-v2.0-exp"
    with pytest.raises(ValueError, match="historical"):
        MODULE.validate(records)


def test_semantic_validator_rejects_missing_or_unbounded_queue_evidence():
    records = _records()
    for value in records:
        if value.get("stage") == "async_v21_bounds":
            value["sealed_descriptor_capacity"] = 2
            break
    with pytest.raises(ValueError, match="bounded queue"):
        MODULE.validate(records)

    records = _records()
    bounds = next(
        value for value in records
        if value.get("stage") == "async_v21_bounds")
    bounds["resident_limit_bytes"] = 64_001_671_647
    bounds["resident_headroom_bytes"] = -1
    with pytest.raises(ValueError, match="resident formula"):
        MODULE.validate(records)


def test_semantic_validator_rejects_per_window_model_rebootstrap():
    records = _records()
    bounds = next(
        value for value in records
        if value.get("stage") == "async_v21_bounds")
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
