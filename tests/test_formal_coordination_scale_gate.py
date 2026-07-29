from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from ndm.formal_coordination_gate import (
    CONFORMANCE_MANIFEST_SCHEMA,
    FORMAL_COORDINATION_GATE_SCHEMA,
    PROGRESS_ASSUMPTIONS,
    REQUIREMENTS,
    REQUIRED_NATIVE_SOURCE_PATHS,
    TRACE_SCHEMA,
    TRUST_BOUNDARY,
    _validate_proof_manifest,
    canonical_digest,
    file_sha256,
    manifest_digest,
    validate_formal_coordination_gate,
)
from scripts.frontier.run_async_v21_qualification import (
    AUTHORIZATION_SCHEMA,
    RUNG_PASS_SCHEMA,
    build_plan,
    scale_identity_contract,
    scale_identity_digest,
)


ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = {
    "source_digest": "a" * 64,
    "policy_digest": "b" * 64,
    "bundle_digest": "c" * 64,
    "seed_digest":
        "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2",
    "launcher_digest": "d" * 64,
}


def _write(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _repo_reference(relative: str) -> dict[str, object]:
    return {
        "root": "repository",
        "path": relative,
        "sha256": file_sha256(ROOT / relative),
    }


def _conformance_manifest(tmp_path: Path) -> Path:
    corpus_path = ROOT / "formal/resilient/corpus/native-v1/manifest.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    job = next(
        entry
        for entry in corpus["entries"]
        if entry["id"]
        == "native-job-5105811-generation-3-close-restart-rejoin"
    )
    source_sha256 = {
        relative: file_sha256(ROOT / relative)
        for relative in REQUIRED_NATIVE_SOURCE_PATHS
    }
    stress = json.loads(
        (
            ROOT / "reports/frontier/native-coordination-stress-v1.json"
        ).read_text(encoding="utf-8")
    )
    value: dict[str, object] = {
        "schema": CONFORMANCE_MANIFEST_SCHEMA,
        "status": "passed",
        "complete": True,
        "partial": False,
        "evaluator_only": False,
        "zero_divergences": 0,
        "traces_passed": len(corpus["entries"]),
        "events_compared": sum(
            int(entry["events"]) for entry in corpus["entries"]
        ),
        "production_transition_path": True,
        "native": {
            "source_commit": stress["identities"]["source_commit"],
            "source_tree_dirty": False,
            "bundle_sha256": IDENTITIES["bundle_digest"],
            "source_sha256": source_sha256,
            "trace_adapter_sha256":
                source_sha256["ndm/native_lean_conformance.py"],
            "artifact_sha256": {
                "service_binary": "1" * 64,
                "local_library": "2" * 64,
                "transport_library": "3" * 64,
            },
            "abi": {
                "coordination": "NDP_COORD_ABI_V1",
                "local": 65_536,
                "event_struct_bytes": 312,
                "member_struct_bytes": 200,
                "result_struct_bytes": 52_016,
            },
        },
        "lean": {
            "runner_sha256": "9" * 64,
            "kernel_sha256": file_sha256(
                ROOT
                / "formal/resilient/ResilientProtocol/Kernel.lean"
            ),
            "trace_schema_sha256": file_sha256(
                ROOT / "formal/resilient/trace-schema-v1.json"
            ),
        },
        "corpus": {
            "manifest_sha256": file_sha256(corpus_path),
            "job_5105811_trace_sha256": job["sha256"],
        },
    }
    value["manifest_digest"] = manifest_digest(value)
    return _write(tmp_path / "native-differential.json", value)


def _passed_8() -> dict[str, object]:
    stress = json.loads(
        (
            ROOT / "reports/frontier/native-coordination-stress-v1.json"
        ).read_text(encoding="utf-8")
    )
    value: dict[str, object] = {
        "schema": RUNG_PASS_SCHEMA,
        "status": "passed",
        "nodes": 8,
        "identities": IDENTITIES,
        "manifest_digest": "8" * 64,
        "native_coordination_lineage": {
            "hardening_manifest_sha256": file_sha256(
                ROOT
                / "docs/validation/"
                "harden-native-coordination-kernel-20260728.md"
            ),
            "schedule_stress_manifest_sha256": file_sha256(
                ROOT / "reports/frontier/native-coordination-stress-v1.json"
            ),
            "schedule_stress_source_commit":
                stress["identities"]["source_commit"],
            "schedule_stress_binary_sha256":
                stress["identities"]["native_binary_sha256"],
        },
    }
    return value


def _systems_evidence() -> dict[str, object]:
    return {
        "qualification_phases": ["clean", "fault", "fresh-recovery"],
        "evidence_digests": {
            "clean_terminal_verdict": "1" * 64,
            "fault_campaign_verdict": "2" * 64,
            "fresh_recovery_terminal_verdict": "3" * 64,
            "causal_telemetry": "4" * 64,
            "publication_receipt": "5" * 64,
            "checkpoint_manifest": "6" * 64,
        },
        "durable_afterany_collector": True,
        "collector_terminal_verdict": "passed",
        "causal_telemetry_complete": True,
        "publication_complete": True,
        "snapshot_admission_seconds_max": 1,
        "apply_swap_seconds_max": 60,
        "foreground_result_wait_seconds_max": 0,
        "forbidden_data_paths": [],
        "leased_ready_finite_closure": True,
        "immutable_safe_boundary_snapshots": True,
        "immediate_trainer_resume": True,
        "background_compiled_cxi": True,
        "later_atomic_apply": True,
        "checkpoint_recovery": True,
        "fencing_idempotency": True,
        "exact_token_eta_outer_one": True,
        "changed_payload_only_retry": True,
        "convergence_claim": False,
    }


def _scale_fields(nodes: int) -> dict[str, object]:
    contract = scale_identity_contract()
    return {
        "nodes": nodes,
        "identities": IDENTITIES,
        "identity_contract": contract,
        "execution_source": {
            "schema": "emender-async-v21-execution-source-v1",
            "digest": IDENTITIES["source_digest"],
            "commit": json.loads(
                (
                    ROOT
                    / "reports/frontier/native-coordination-stress-v1.json"
                ).read_text(encoding="utf-8")
            )["identities"]["source_commit"],
            "source_tree_dirty": False,
        },
        "identity_digest": scale_identity_digest(IDENTITIES, contract),
        "scheduler": {
            "Nodes": nodes,
            "Partition": "batch",
            "QOS": "debug",
        },
        "systems_evidence": _systems_evidence(),
    }


def _formal_gate(
    tmp_path: Path,
    differential: Path,
    passed_8: dict[str, object],
) -> Path:
    references = {
        "native_hardening": _repo_reference(
            "docs/validation/"
            "harden-native-coordination-kernel-20260728.md"
        ),
        "schedule_stress": _repo_reference(
            "reports/frontier/native-coordination-stress-v1.json"
        ),
        "proof_manifest": _repo_reference(
            "formal/resilient/proof-manifest-v1.json"
        ),
        "proof_coverage": _repo_reference(
            "formal/resilient/PROOF_COVERAGE.md"
        ),
        "trace_schema": _repo_reference(
            "formal/resilient/trace-schema-v1.json"
        ),
        "conformance_corpus": _repo_reference(
            "formal/resilient/corpus/native-v1/manifest.json"
        ),
        "native_differential": {
            "root": "evidence",
            "path": differential.name,
            "sha256": file_sha256(differential),
        },
    }
    contract = scale_identity_contract()
    value: dict[str, object] = {
        "schema": FORMAL_COORDINATION_GATE_SCHEMA,
        "status": "passed",
        "complete": True,
        "partial": False,
        "evaluator_only": False,
        "stale": False,
        "authorizes_nodes": [32],
        "identities": IDENTITIES,
        "identity_contract": contract,
        "execution_source": {
            "schema": "emender-async-v21-execution-source-v1",
            "digest": IDENTITIES["source_digest"],
            "commit": json.loads(
                (
                    ROOT
                    / "reports/frontier/native-coordination-stress-v1.json"
                ).read_text(encoding="utf-8")
            )["identities"]["source_commit"],
            "source_tree_dirty": False,
        },
        "identity_digest": scale_identity_digest(IDENTITIES, contract),
        "requirements": REQUIREMENTS,
        "trust_boundary": TRUST_BOUNDARY,
        "artifacts": references,
        "lineage": {
            f"{name}_sha256": reference["sha256"]
            for name, reference in references.items()
        }
        | {
            "scale_authorization_digest": "4" * 64,
            "passed_8_node_manifest_digest":
                passed_8["manifest_digest"],
            "passed_8_node_sha256": "5" * 64,
        },
        "review_signature": "signed-for-test",
    }
    value["manifest_digest"] = manifest_digest(value)
    return _write(tmp_path / "formal-gate.json", value)


def _validate(
    gate: Path,
    tmp_path: Path,
    passed_8: dict[str, object],
) -> dict[str, object]:
    return validate_formal_coordination_gate(
        gate,
        repository=ROOT,
        evidence_root=tmp_path,
        expected_nodes=32,
        expected_identities=IDENTITIES,
        expected_identity_contract=scale_identity_contract(),
        scale_authorization_digest="4" * 64,
        passed_8_node_manifest=passed_8,
        passed_8_node_sha256="5" * 64,
    )


def test_formal_gate_accepts_only_the_exact_proved_native_8_node_lineage(
    tmp_path: Path,
):
    passed_8 = _passed_8()
    gate = _formal_gate(
        tmp_path, _conformance_manifest(tmp_path), passed_8
    )
    assert _validate(gate, tmp_path, passed_8)["status"] == "passed"


def test_exact_32_node_controller_render_consumes_joined_formal_gate(
    tmp_path: Path,
):
    arrival = {
        "schema": "emender-async-v21-two-node-arrivals-v1",
        "status": "passed",
        "nodes": 2,
        "samples_ns": [10, 20, 30],
    }
    arrival["manifest_digest"] = canonical_digest(arrival)
    _write(tmp_path / "arrival.json", arrival)
    stage = {
        "schema": "emender-async-v21-two-node-stages-v1",
        "status": "passed",
        "nodes": 2,
        "close_to_latest_ns": [40, 50, 60],
        "cadence_ns": [15, 20, 25],
    }
    stage["manifest_digest"] = canonical_digest(stage)
    _write(tmp_path / "stage.json", stage)
    closure = {
        "schema": "emender-v21s17-scale-closure-v1",
        "ready_snapshot_source": "leased-ready-at-group-open",
        "arrival_evidence": {
            "path": "arrival.json",
            "digest": arrival["manifest_digest"],
        },
        "stage_evidence": {
            "path": "stage.json",
            "digest": stage["manifest_digest"],
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
    authorization_value: dict[str, object] = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "passed",
        "authorized_nodes": 32,
        "systems_scale_ladder": [2, 8, 32, 128],
        "review_only_nodes": [256],
        "convergence_required": False,
        "reviewed_ready_snapshot_size": 32,
        **_scale_fields(32),
        "review_signature": "signed-for-test",
        "closure": closure,
    }
    authorization_value["manifest_digest"] = canonical_digest(
        authorization_value
    )
    authorization = _write(
        tmp_path / "authorization.json", authorization_value
    )
    passed_8_value = {
        **_passed_8(),
        **_scale_fields(8),
        "review_signature": "signed-for-test",
    }
    passed_8_value["manifest_digest"] = manifest_digest(passed_8_value)
    predecessor = _write(tmp_path / "passed-8.json", passed_8_value)
    differential = _conformance_manifest(tmp_path)
    formal = _formal_gate(tmp_path, differential, passed_8_value)
    formal_value = json.loads(formal.read_text(encoding="utf-8"))
    formal_value["lineage"].update({
        "scale_authorization_digest":
            authorization_value["manifest_digest"],
        "passed_8_node_manifest_digest":
            passed_8_value["manifest_digest"],
        "passed_8_node_sha256": file_sha256(predecessor),
    })
    formal_value["manifest_digest"] = manifest_digest(formal_value)
    _write(formal, formal_value)

    plan = build_plan(
        gate="scale",
        nodes=32,
        state_path=tmp_path / "state.json",
        evidence_root=tmp_path,
        parameters={
            "close_on_q_min": False,
            "uses_launched_ranks": False,
            "wait_for_all_ready": False,
        },
        authorization_path=authorization,
        predecessor_path=predecessor,
        formal_coordination_path=formal,
        allow_test_signatures=True,
        **IDENTITIES,
    )
    formal_payload = plan["payload"]["formal_coordination_gate"]
    assert formal_payload["manifest_digest"] == formal_value["manifest_digest"]
    exported = next(
        item for item in plan["command"] if item.startswith("--export=")
    )
    assert "ASYNC_V21_FORMAL_COORDINATION_GATE=" in exported
    assert "ASYNC_V21_FORMAL_COORDINATION_GATE_DIGEST=" in exported
    assert "ASYNC_V21_FORMAL_COORDINATION_GATE_SHA256=" in exported
    assert plan["scheduler"] == {
        "Nodes": 32,
        "Partition": "batch",
        "QOS": "debug",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(status="failed"), "stale|partial"),
        (lambda value: value.update(complete=False), "stale|partial"),
        (lambda value: value.update(partial=True), "stale|partial"),
        (lambda value: value.update(evaluator_only=True), "evaluator"),
        (lambda value: value.update(stale=True), "stale"),
        (lambda value: value.update(authorizes_nodes=[]), "authorize"),
        (
            lambda value: value["requirements"]["compute_pool"].remove("R16"),
            "stale|partial",
        ),
        (
            lambda value: value["trust_boundary"].update(
                runtime_claim_from_lean=True
            ),
            "stale|partial",
        ),
        (
            lambda value: value["identities"].update(source_digest="f" * 64),
            "exact payload",
        ),
        (
            lambda value: value["execution_source"].update(
                source_tree_dirty=True
            ),
            "evidence/lineage",
        ),
        (
            lambda value: value["artifacts"].pop("proof_manifest"),
            "artifact set",
        ),
        (
            lambda value: value["artifacts"]["proof_manifest"].update(
                sha256="f" * 64
            ),
            "digest mismatch",
        ),
        (
            lambda value: value["lineage"].update(
                schedule_stress_sha256="f" * 64
            ),
            "lineage mismatch",
        ),
        (
            lambda value: value["lineage"].update(
                passed_8_node_manifest_digest="f" * 64
            ),
            "exact authorization",
        ),
    ),
)
def test_formal_gate_independently_rejects_missing_failed_stale_or_mismatch(
    tmp_path: Path, mutation, message: str,
):
    passed_8 = _passed_8()
    gate = _formal_gate(
        tmp_path, _conformance_manifest(tmp_path), passed_8
    )
    value = json.loads(gate.read_text(encoding="utf-8"))
    mutation(value)
    value["manifest_digest"] = manifest_digest(value)
    _write(gate, value)
    with pytest.raises(ValueError, match=message):
        _validate(gate, tmp_path, passed_8)


def test_formal_gate_rejects_wrong_8_node_hardening_or_stress_lineage(
    tmp_path: Path,
):
    passed_8 = _passed_8()
    gate = _formal_gate(
        tmp_path, _conformance_manifest(tmp_path), passed_8
    )
    passed_8["native_coordination_lineage"][
        "schedule_stress_binary_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="stress lineage"):
        _validate(gate, tmp_path, passed_8)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.update(zero_divergences=1),
            "differential evidence",
        ),
        (
            lambda value: value.update(complete=False),
            "differential evidence",
        ),
        (
            lambda value: value["native"].update(
                bundle_sha256="f" * 64
            ),
            "differential evidence",
        ),
        (
            lambda value: value["native"]["source_sha256"].update(
                {
                    "src/native_resilient_dataplane/src/"
                    "coordination_kernel.cpp": "f" * 64
                }
            ),
            "production-native source digest",
        ),
        (
            lambda value: value["native"].update(
                trace_adapter_sha256="f" * 64
            ),
            "trace-adapter",
        ),
        (
            lambda value: value["native"]["abi"].update(
                coordination="test-only"
            ),
            "differential evidence",
        ),
        (
            lambda value: value["lean"].update(
                kernel_sha256="f" * 64
            ),
            "proved kernel",
        ),
        (
            lambda value: value["corpus"].update(
                job_5105811_trace_sha256="f" * 64
            ),
            "proved kernel|differential evidence",
        ),
    ),
)
def test_formal_gate_rejects_nonzero_partial_or_wrong_native_differential(
    tmp_path: Path, mutation, message: str,
):
    passed_8 = _passed_8()
    differential = _conformance_manifest(tmp_path)
    gate = _formal_gate(tmp_path, differential, passed_8)
    differential_value = json.loads(
        differential.read_text(encoding="utf-8")
    )
    mutation(differential_value)
    differential_value["manifest_digest"] = manifest_digest(
        differential_value
    )
    _write(differential, differential_value)
    gate_value = json.loads(gate.read_text(encoding="utf-8"))
    digest = file_sha256(differential)
    gate_value["artifacts"]["native_differential"]["sha256"] = digest
    gate_value["lineage"]["native_differential_sha256"] = digest
    gate_value["manifest_digest"] = manifest_digest(gate_value)
    _write(gate, gate_value)
    with pytest.raises(ValueError, match=message):
        _validate(gate, tmp_path, passed_8)


def test_formal_gate_rejects_corrupt_json_and_manifest_digest(
    tmp_path: Path,
):
    passed_8 = _passed_8()
    differential = _conformance_manifest(tmp_path)
    gate = _formal_gate(tmp_path, differential, passed_8)
    value = json.loads(gate.read_text(encoding="utf-8"))
    value["status"] = "failed"
    _write(gate, value)
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        _validate(gate, tmp_path, passed_8)
    gate.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _validate(gate, tmp_path, passed_8)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda policy, progress: policy.update(
            runtime_boolean_is_not_a_theorem=False
        ),
        lambda policy, progress: policy.update(
            theorems_are_propositions_over_transition=False
        ),
        lambda policy, progress: policy.update(
            boolean_substitutes=["invariantHolds"]
        ),
        lambda policy, progress: policy["forbidden_lean_tokens"].remove(
            "unsafe"
        ),
        lambda policy, progress: progress.update(unconditional_claim=True),
        lambda policy, progress: progress.update(required_assumptions=[]),
    ),
)
def test_proof_manifest_rejects_boolean_substitutes_unsafe_or_hidden_liveness(
    tmp_path: Path, mutation,
):
    source = ROOT / "formal/resilient"
    package = tmp_path / "resilient"
    shutil.copytree(
        source,
        package,
        ignore=shutil.ignore_patterns(".lake", ".cache"),
    )
    manifest_path = package / "proof-manifest-v1.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(value["proof_policy"], value["progress"])
    _write(manifest_path, value)
    with pytest.raises(ValueError, match="unsafe|overclaims"):
        _validate_proof_manifest(
            manifest_path,
            package_root=package,
            expected_trace_schema_sha256=file_sha256(
                package / "trace-schema-v1.json"
            ),
        )


def test_progress_manifest_names_every_bounded_progress_assumption():
    assert PROGRESS_ASSUMPTIONS == [
        "finite_close_and_stage_deadlines",
        "surviving_eligible_stable_worker_quorum",
        "surviving_exact_token_floor",
        "bounded_permitted_failures",
        "bounded_owner_reassignments",
        "eventual_delivery_and_processing",
        "fair_scheduling_of_enabled_transitions",
    ]
    assert TRACE_SCHEMA == "emender-resilient-coordination-trace-v1"
