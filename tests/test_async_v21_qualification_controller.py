from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.frontier.run_async_v21_qualification import (
    PAYLOAD_SCHEMA,
    SEED_SHA256,
    V21ScaleClosure,
    build_plan,
    canonical_digest,
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
