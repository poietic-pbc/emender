from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ndm.native_pool_production_policy import (
    CANDIDATE_SCHEMA,
    DEFAULT_POLICY,
    load_policy,
    main,
    validate_production_candidate,
)


SHA = "a" * 64
COMMIT = "b" * 40
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "local_library": "1" * 64,
    "transport_library": "2" * 64,
    "service_binary": "3" * 64,
    "synthetic_gate_binary": "4" * 64,
}


def _qualified_policy():
    policy = load_policy()
    policy["scale_admission"]["production_qualified_rungs"] = [2]
    return policy


def _candidate(policy, *, checkpoint_mode="resume"):
    runtime = deepcopy(policy["runtime_profile"])
    checkpoint = {
        "mode": "resume",
        "generation": 9,
        "authoritative_latest_generation": 9,
        "manifest_generation": 9,
        "accepted_tokens": 47_208_960,
        "manifest_accepted_tokens": 47_208_960,
        "fence": 7,
        "authoritative_latest_fence": 7,
        "checkpoint_sha256": "9" * 64,
        "manifest_sha256": "8" * 64,
        "checkpoint_digest_verified": True,
        "manifest_digest_verified": True,
    }
    if checkpoint_mode == "seed":
        checkpoint = {
            "mode": "seed",
            "generation": 0,
            "fence": None,
            "checkpoint_sha256": policy["checkpoint"]["seed_sha256"],
            "manifest_sha256": "8" * 64,
            "checkpoint_digest_verified": True,
            "manifest_digest_verified": True,
        }
    return {
        "schema": CANDIDATE_SCHEMA,
        "nodes": 2,
        "source_commit": COMMIT,
        "config_sha256": "c" * 64,
        "provider": "cxi",
        "endpoint_type": "FI_EP_RDM",
        "network": "job_vni",
        "native_attestation": {
            "status": "attested",
            "production": True,
            "full_layout": True,
            "backend": "native-cxi",
            "source_commit": COMMIT,
            "bundle_sha256": "d" * 64,
            "artifacts": deepcopy(ARTIFACTS),
        },
        "full_layout_gate": {
            "status": "passed",
            "source_commit": COMMIT,
            "bundle_sha256": "d" * 64,
            "provider": "cxi",
            "endpoint_type": "FI_EP_RDM",
            "production_provider": True,
            "artifacts": deepcopy(ARTIFACTS),
            "nodes": 2,
            "owners": 2,
            "layout_bytes": 5_506_770_496,
            "logical_contribution_bytes": 11_013_540_992,
            "logical_redistribution_bytes": 11_013_540_992,
            "global_weight": 3_934_080,
            "performance": {
                "logical_bytes_per_second":
                    policy["throughput"]["g2_minimum_logical_bytes_per_second"],
                "median_seconds": policy["throughput"]["g2_maximum_median_seconds"],
            },
        },
        "scale_qualification": {
            "status": "passed",
            "nodes": 2,
            "source_commit": COMMIT,
            "bundle_sha256": "d" * 64,
            "all_prior_rungs_passed": True,
            "provider": "cxi",
            "config_sha256": "c" * 64,
            "validated_payload_sha256": SHA,
        },
        "runtime_profile": runtime,
        "deadlines_seconds": deepcopy(policy["deadlines_seconds"]),
        "throughput_floors": {
            name: policy["throughput"][name]
            for name in (
                "g2_minimum_logical_bytes_per_second",
                "g2_maximum_median_seconds",
                "live_owner_exchange_floor_bytes_per_second",
                "minimum_committed_token_rate_at_floor",
            )
        },
        "checkpoint": checkpoint,
        "lease": {
            "acquired_before_model_load": True,
            "current": True,
            "renewal_healthy": True,
            "ttl_seconds": 60,
            "renew_interval_seconds": 10,
            "fence": 8,
        },
        "promotion": {
            "validated_payload_sha256": SHA,
            "production_payload_sha256": SHA,
            "changed_fields": ["qos", "walltime"],
            "validation_scheduler": {
                "partition": "batch", "qos": "debug", "walltime": "00:30:00",
            },
            "production_scheduler": {
                "partition": "batch", "qos": "normal", "walltime": "12:00:00",
            },
        },
    }


def test_reviewed_policy_is_bound_to_retained_measurements_and_closed_by_default():
    policy = load_policy(DEFAULT_POLICY)
    assert policy["status"] == "defined-no-submission-authority"
    assert policy["scale_admission"]["measured_policy_baseline_rungs"] == [2]
    assert policy["scale_admission"]["production_qualified_rungs"] == []
    assert policy["runtime_profile"]["q_min"] == 2
    assert policy["runtime_profile"]["t_min"] == 3_934_080
    assert policy["runtime_profile"]["owner_count"] == 2
    assert policy["runtime_profile"]["checkpoint_every_committed_generations"] == 1
    assert policy["throughput"]["g2_observed_median_seconds"] == 22.690315566
    for record in policy["measurement_evidence"].values():
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("checkpoint_mode", ["seed", "resume"])
def test_exact_qualified_candidate_is_admitted(checkpoint_mode):
    policy = _qualified_policy()
    admitted = validate_production_candidate(
        policy, _candidate(policy, checkpoint_mode=checkpoint_mode))
    assert admitted == {
        "schema": "emender-native-resilient-pool-production-admission-v1",
        "status": "admitted",
        "nodes": 2,
        "source_commit": COMMIT,
        "bundle_sha256": "d" * 64,
        "payload_sha256": SHA,
        "policy_version": "1.0.0",
        "checkpoint_generation": 0 if checkpoint_mode == "seed" else 9,
        "lease_fence": 8,
    }


def test_current_policy_refuses_unqualified_scale_rung():
    policy = load_policy()
    with pytest.raises(ValueError, match="unqualified production scale rung"):
        validate_production_candidate(policy, _candidate(policy))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["native_attestation"].__setitem__(
            "backend", "python-tcp-debug"), "production backend"),
        (lambda value: value.__setitem__("provider", "tcp;ofi_rxm"),
         "production provider"),
        (lambda value: value["native_attestation"]["artifacts"].pop(
            "service_binary"), "artifact digest set"),
        (lambda value: value["full_layout_gate"].__setitem__(
            "bundle_sha256", "e" * 64), "full-layout gate bundle"),
        (lambda value: value["runtime_profile"].__setitem__("t_min", 1),
         "runtime_profile.t_min"),
        (lambda value: value["promotion"].__setitem__(
            "production_payload_sha256", "f" * 64),
         "byte-identical production payload"),
        (lambda value: value["promotion"].__setitem__(
            "changed_fields", ["nodes", "qos", "walltime"]),
         "promotion difference allowlist"),
        (lambda value: (
            value["promotion"]["validation_scheduler"].__setitem__("nodes", 2),
            value["promotion"]["production_scheduler"].__setitem__("nodes", 4)),
         "actual validation/production scheduler differences"),
    ],
)
def test_prohibited_backend_digest_floor_and_promotion_drift_fail_closed(mutate, message):
    policy = _qualified_policy()
    candidate = _candidate(policy)
    mutate(candidate)
    with pytest.raises(ValueError, match=message):
        validate_production_candidate(policy, candidate)


def test_stale_checkpoint_or_fence_fails_closed():
    policy = _qualified_policy()
    candidate = _candidate(policy)
    candidate["checkpoint"]["generation"] = 8
    with pytest.raises(ValueError, match="checkpoint freshness generation"):
        validate_production_candidate(policy, candidate)

    candidate = _candidate(policy)
    candidate["lease"]["fence"] = candidate["checkpoint"]["fence"]
    with pytest.raises(ValueError, match="stale checkpoint/fence"):
        validate_production_candidate(policy, candidate)


def test_cli_emits_only_an_attested_record(tmp_path, capsys):
    policy = _qualified_policy()
    policy_path = tmp_path / "policy.json"
    candidate_path = tmp_path / "candidate.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    candidate_path.write_text(json.dumps(_candidate(policy)), encoding="utf-8")
    assert main([str(candidate_path), "--policy", str(policy_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "admitted"
