"""Fail-closed admission for the native resilient-pool v1 production profile.

The policy is intentionally separate from Slurm submission.  It validates a
fully materialized candidate record and returns an attested summary; it never
loads a model, acquires a lease, or submits work.  The packaging layer is
responsible for producing the candidate from immutable validation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping


POLICY_SCHEMA = "emender-native-resilient-pool-production-policy-v1"
CANDIDATE_SCHEMA = "emender-native-resilient-pool-production-candidate-v1"
DEFAULT_POLICY = (
    Path(__file__).resolve().parents[1]
    / "configs/frontier/native_resilient_pool_v1_production_policy.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_ARTIFACTS = frozenset(
    ("local_library", "transport_library", "service_binary", "synthetic_gate_binary")
)


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase full Git commit")
    return value


def _equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def load_policy(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
    """Load and minimally self-validate the reviewed policy artifact."""
    policy_path = Path(path)
    try:
        value = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"production policy cannot be read: {policy_path}") from error
    policy = dict(_object(value, "policy"))
    _equal(policy.get("schema"), POLICY_SCHEMA, "production policy schema")
    evidence = _object(policy.get("measurement_evidence"), "measurement_evidence")
    for name, record in evidence.items():
        _sha256(_object(record, f"measurement_evidence.{name}").get("sha256"),
                f"measurement_evidence.{name}.sha256")
    scale = _object(policy.get("scale_admission"), "scale_admission")
    ordered = scale.get("ordered_rungs")
    qualified = scale.get("production_qualified_rungs")
    if (not isinstance(ordered, list) or not ordered
            or any(not isinstance(item, int) or item < 2 for item in ordered)):
        raise ValueError("scale_admission.ordered_rungs must contain node counts")
    if (not isinstance(qualified, list)
            or any(item not in ordered for item in qualified)):
        raise ValueError("production-qualified scale rungs must be ordered rungs")
    return policy


def _validate_native_attestation(policy: Mapping[str, Any],
                                 candidate: Mapping[str, Any]) -> tuple[str, str]:
    required = _object(policy["native_admission"], "native_admission")
    attestation = _object(candidate.get("native_attestation"), "native_attestation")
    _equal(attestation.get("status"), "attested", "native attestation status")
    _equal(attestation.get("production"), True, "native attestation production flag")
    _equal(attestation.get("full_layout"), True, "native attestation full-layout flag")
    _equal(attestation.get("backend"), required["backend"], "production backend")
    _equal(candidate.get("provider"), required["provider"], "production provider")
    _equal(candidate.get("endpoint_type"), required["endpoint_type"],
           "production endpoint type")
    _equal(candidate.get("network"), required["network"], "production Slingshot network")

    source_commit = _git_commit(attestation.get("source_commit"),
                                "native_attestation.source_commit")
    bundle = _sha256(attestation.get("bundle_sha256"),
                     "native_attestation.bundle_sha256")
    _equal(candidate.get("source_commit"), source_commit, "candidate source commit")
    artifacts = _object(attestation.get("artifacts"), "native_attestation.artifacts")
    _equal(set(artifacts), set(_ARTIFACTS), "native artifact digest set")
    for name in _ARTIFACTS:
        _sha256(artifacts[name], f"native_attestation.artifacts.{name}")

    gate = _object(candidate.get("full_layout_gate"), "full_layout_gate")
    _equal(gate.get("status"), "passed", "full-layout gate status")
    _equal(gate.get("source_commit"), source_commit, "full-layout gate source commit")
    _equal(gate.get("bundle_sha256"), bundle, "full-layout gate bundle")
    _equal(gate.get("provider"), required["provider"], "full-layout gate provider")
    _equal(gate.get("endpoint_type"), required["endpoint_type"],
           "full-layout gate endpoint type")
    _equal(gate.get("production_provider"), True,
           "full-layout gate production-provider flag")
    _equal(gate.get("artifacts"), artifacts, "full-layout gate artifact digests")
    profile = _object(policy["runtime_profile"], "runtime_profile")
    _equal(gate.get("nodes"), 2, "full-layout gate node count")
    _equal(gate.get("owners"), 2, "full-layout gate owner count")
    _equal(gate.get("layout_bytes"), profile["layout_bytes"],
           "full-layout gate layout bytes")
    _equal(gate.get("logical_contribution_bytes"),
           profile["sender_replay_bytes_max"],
           "full-layout gate contribution bytes")
    _equal(gate.get("logical_redistribution_bytes"),
           profile["sender_replay_bytes_max"],
           "full-layout gate redistribution bytes")
    _equal(gate.get("global_weight"), profile["t_min"],
           "full-layout gate global weight")
    performance = _object(gate.get("performance"), "full_layout_gate.performance")
    throughput = _object(policy["throughput"], "throughput")
    floor = float(throughput["g2_minimum_logical_bytes_per_second"])
    measured = float(performance.get("logical_bytes_per_second", -1))
    if measured < floor:
        raise ValueError(
            f"full-layout gate throughput below production floor: {measured} < {floor}")
    median = float(performance.get("median_seconds", float("inf")))
    median_ceiling = float(throughput["g2_maximum_median_seconds"])
    if median > median_ceiling:
        raise ValueError(
            f"full-layout gate median above production ceiling: {median} > {median_ceiling}")
    return source_commit, bundle


def _validate_scale(policy: Mapping[str, Any], candidate: Mapping[str, Any],
                    source_commit: str, bundle: str) -> str:
    nodes = candidate.get("nodes")
    scale_policy = _object(policy["scale_admission"], "scale_admission")
    if nodes not in scale_policy["production_qualified_rungs"]:
        raise ValueError(f"unqualified production scale rung: {nodes!r} nodes")
    qualification = _object(candidate.get("scale_qualification"),
                            "scale_qualification")
    _equal(qualification.get("status"), "passed", "scale qualification status")
    _equal(qualification.get("nodes"), nodes, "scale qualification node count")
    _equal(qualification.get("source_commit"), source_commit,
           "scale qualification source commit")
    _equal(qualification.get("bundle_sha256"), bundle,
           "scale qualification native bundle")
    _equal(qualification.get("all_prior_rungs_passed"), True,
           "ordered scale qualification")
    _equal(qualification.get("provider"), policy["native_admission"]["provider"],
           "scale qualification provider")
    config_digest = _sha256(candidate.get("config_sha256"), "candidate.config_sha256")
    _equal(qualification.get("config_sha256"), config_digest,
           "scale qualification config")
    return _sha256(qualification.get("validated_payload_sha256"),
                   "scale_qualification.validated_payload_sha256")


def _validate_runtime(policy: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    expected = _object(policy["runtime_profile"], "runtime_profile")
    runtime = _object(candidate.get("runtime_profile"), "runtime_profile")
    for name, value in expected.items():
        _equal(runtime.get(name), value, f"runtime_profile.{name}")
    _equal(candidate.get("deadlines_seconds"), policy["deadlines_seconds"],
           "absolute deadline profile")
    expected_throughput = {
        name: policy["throughput"][name]
        for name in (
            "g2_minimum_logical_bytes_per_second",
            "g2_maximum_median_seconds",
            "live_owner_exchange_floor_bytes_per_second",
            "minimum_committed_token_rate_at_floor",
        )
    }
    _equal(candidate.get("throughput_floors"), expected_throughput,
           "production throughput floors")


def _validate_checkpoint(policy: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    checkpoint_policy = _object(policy["checkpoint"], "checkpoint")
    checkpoint = _object(candidate.get("checkpoint"), "checkpoint")
    lease = _object(candidate.get("lease"), "lease")
    for name in ("acquired_before_model_load", "current", "renewal_healthy"):
        _equal(lease.get(name), True, f"lease.{name}")
    _equal(lease.get("ttl_seconds"), policy["lease"]["ttl_seconds"],
           "lease.ttl_seconds")
    _equal(lease.get("renew_interval_seconds"),
           policy["lease"]["renew_interval_seconds"],
           "lease.renew_interval_seconds")
    lease_fence = lease.get("fence")
    if not isinstance(lease_fence, int) or lease_fence < 1:
        raise ValueError("lease.fence must be a positive current fence")
    _equal(checkpoint.get("checkpoint_digest_verified"), True,
           "checkpoint digest verification")
    _equal(checkpoint.get("manifest_digest_verified"), True,
           "checkpoint manifest verification")
    _sha256(checkpoint.get("checkpoint_sha256"), "checkpoint.checkpoint_sha256")
    _sha256(checkpoint.get("manifest_sha256"), "checkpoint.manifest_sha256")

    mode = checkpoint.get("mode")
    if mode == "seed":
        _equal(checkpoint.get("checkpoint_sha256"), checkpoint_policy["seed_sha256"],
               "approved cold-start seed")
        _equal(checkpoint.get("generation"), 0, "cold-start seed generation")
        _equal(checkpoint.get("fence"), None, "cold-start seed fence")
        return
    if mode != "resume":
        raise ValueError("checkpoint.mode must be 'seed' or 'resume'")
    latest_generation = checkpoint.get("authoritative_latest_generation")
    _equal(checkpoint.get("generation"), latest_generation,
           "checkpoint freshness generation")
    _equal(checkpoint.get("manifest_generation"), latest_generation,
           "checkpoint manifest generation")
    _equal(checkpoint.get("accepted_tokens"), checkpoint.get("manifest_accepted_tokens"),
           "checkpoint accepted-token clock")
    checkpoint_fence = checkpoint.get("fence")
    _equal(checkpoint.get("authoritative_latest_fence"), checkpoint_fence,
           "checkpoint authoritative fence")
    if (not isinstance(checkpoint_fence, int) or checkpoint_fence < 1
            or lease_fence <= checkpoint_fence):
        raise ValueError(
            "stale checkpoint/fence: resume allocation must hold a strictly newer fence")


def _validate_promotion(policy: Mapping[str, Any], candidate: Mapping[str, Any],
                        validated_payload: str) -> None:
    promotion_policy = _object(policy["promotion"], "promotion")
    promotion = _object(candidate.get("promotion"), "promotion")
    _equal(_sha256(promotion.get("validated_payload_sha256"),
                   "promotion.validated_payload_sha256"),
           validated_payload, "retained validation payload")
    _equal(_sha256(promotion.get("production_payload_sha256"),
                   "promotion.production_payload_sha256"),
           validated_payload, "byte-identical production payload")
    _equal(promotion.get("changed_fields"), promotion_policy["allowed_changed_fields"],
           "promotion difference allowlist")
    validation = _object(promotion.get("validation_scheduler"),
                         "promotion.validation_scheduler")
    production = _object(promotion.get("production_scheduler"),
                         "promotion.production_scheduler")
    all_scheduler_fields = set(validation) | set(production)
    allowed = list(promotion_policy["allowed_changed_fields"])
    actual_changes = [name for name in allowed
                      if validation.get(name) != production.get(name)]
    actual_changes.extend(sorted(
        name for name in all_scheduler_fields - set(allowed)
        if validation.get(name) != production.get(name)))
    _equal(actual_changes, allowed, "actual validation/production scheduler differences")
    _equal(validation.get("partition"), promotion_policy["partition"],
           "validation partition")
    _equal(production.get("partition"), promotion_policy["partition"],
           "production partition")
    _equal(validation.get("qos"), promotion_policy["validation_qos"],
           "validation QoS")
    _equal(production.get("qos"), promotion_policy["production_qos"],
           "production QoS")


def validate_production_candidate(policy: Mapping[str, Any],
                                  candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every production admission/promotion invariant.

    The returned mapping is suitable for inclusion in a launch manifest.  A
    mismatch raises ``ValueError`` and is deliberately non-recovering: callers
    must stop before model load rather than substitute a debug backend or an
    unqualified payload.
    """
    _equal(policy.get("schema"), POLICY_SCHEMA, "production policy schema")
    _equal(candidate.get("schema"), CANDIDATE_SCHEMA, "production candidate schema")
    source_commit, bundle = _validate_native_attestation(policy, candidate)
    validated_payload = _validate_scale(policy, candidate, source_commit, bundle)
    _validate_runtime(policy, candidate)
    _validate_checkpoint(policy, candidate)
    _validate_promotion(policy, candidate, validated_payload)
    return {
        "schema": "emender-native-resilient-pool-production-admission-v1",
        "status": "admitted",
        "nodes": candidate["nodes"],
        "source_commit": source_commit,
        "bundle_sha256": bundle,
        "payload_sha256": validated_payload,
        "policy_version": policy["version"],
        "checkpoint_generation": candidate["checkpoint"]["generation"],
        "lease_fence": candidate["lease"]["fence"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="validate a native resilient-pool v1 production candidate")
    parser.add_argument("candidate")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args(argv)
    try:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        admitted = validate_production_candidate(
            load_policy(args.policy), _object(candidate, "candidate"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(admitted, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
