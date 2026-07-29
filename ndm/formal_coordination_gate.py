"""Fail-closed formal/native coordination evidence for systems scale.

Lean proves properties of the pure coordination transition.  It is not a
runtime implementation and it does not discharge transport, byte-path,
snapshot, timing, scheduler, numerical, or scale evidence.  This module joins
the immutable proof and differential artifacts to the exact production-native
lineage that already passed at eight nodes.  The joined manifest is consumed
locally by the scale controller; compute nodes never run Lean.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


FORMAL_COORDINATION_GATE_SCHEMA = (
    "emender-formal-native-coordination-scale-gate-v1"
)
FORMAL_COORDINATION_LOCAL_SCHEMA = (
    "emender-formal-native-coordination-local-evidence-v1"
)
PROOF_MANIFEST_SCHEMA = "emender-resilient-lean-proof-manifest-v1"
CONFORMANCE_MANIFEST_SCHEMA = "emender-native-lean-conformance-manifest-v1"
STRESS_MANIFEST_SCHEMA = "emender-native-coordination-stress-manifest-v1"
CORPUS_MANIFEST_SCHEMA = "emender-native-lean-conformance-corpus-v1"
TRACE_SCHEMA = "emender-resilient-coordination-trace-v1"
JOB_5105811_TRACE_ID = (
    "native-job-5105811-generation-3-close-restart-rejoin"
)

REQUIREMENTS = {
    "compute_pool": [f"R{index:02d}" for index in range(1, 17)],
    "native": [f"NDP{index:02d}" for index in range(1, 18)],
    "async_v21": [f"V21S{index:02d}" for index in range(1, 18)],
    "immutable_snapshot": [f"ISP{index:02d}" for index in range(1, 8)],
}

PROGRESS_ASSUMPTIONS = [
    "finite_close_and_stage_deadlines",
    "surviving_eligible_stable_worker_quorum",
    "surviving_exact_token_floor",
    "bounded_permitted_failures",
    "bounded_owner_reassignments",
    "eventual_delivery_and_processing",
    "fair_scheduling_of_enabled_transitions",
]

FORBIDDEN_PROOF_TOKENS = {
    "sorry",
    "admit",
    "native_decide",
    "axiom",
    "opaque",
    "unsafe",
}

REQUIRED_PROOF_ROLES = {
    "pinned_toolchain",
    "build_definition",
    "empty_dependency_lock",
    "trace_schema",
    "authoritative_import_root",
    "model_types",
    "executable_transition",
    "safety_theorems",
    "conditional_progress_theorems",
    "job_5105811_regression_theorems",
    "invalid_variant_rejection_theorems",
    "transition_requirement_crosswalk",
    "digest_verifier",
}

REQUIRED_NATIVE_SOURCE_PATHS = {
    "ndm/native_lean_conformance.py",
    "ndm/native_coordination.py",
    "ndm/native_dataplane.py",
    "src/native_resilient_dataplane/src/client.cpp",
    "src/native_resilient_dataplane/src/rpc_server.cpp",
    "src/native_resilient_dataplane/src/service_core.hpp",
    "src/native_resilient_dataplane/src/ndp.cpp",
    "src/native_resilient_dataplane/src/coordination_kernel.cpp",
}

TRUST_BOUNDARY = {
    "lean_scope": "pure_coordination_only",
    "native_runtime_owns": [
        "networking",
        "libfabric",
        "timers",
        "bounded_buffers",
        "process_supervision",
        "runtime_effects",
    ],
    "production_traces_link_authorities": True,
    "lean_on_compute_nodes": False,
    "formal_replaces_runtime_evidence": False,
    "proof_claim_from_tests": False,
    "runtime_claim_from_lean": False,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_digest(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error
    return text


def manifest_digest(value: Mapping[str, object]) -> str:
    unsigned = {
        key: item for key, item in value.items() if key != "manifest_digest"
    }
    return canonical_digest(unsigned)


def _load_json(path: Path, *, schema: str, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is corrupt") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{label} schema mismatch")
    return value


def _resolve_reference(
    reference: object,
    *,
    repository: Path,
    evidence_root: Path,
    label: str,
) -> Path:
    if not isinstance(reference, Mapping):
        raise ValueError(f"{label} reference is missing")
    root_name = reference.get("root")
    root = (
        repository.resolve()
        if root_name == "repository"
        else evidence_root.resolve()
        if root_name == "evidence"
        else None
    )
    if root is None:
        raise ValueError(f"{label} reference root is invalid")
    relative = Path(str(reference.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} reference path must stay under its root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} reference escapes its root") from error
    expected = require_digest(reference.get("sha256"), f"{label} reference")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} reference digest mismatch")
    return path


def _validate_proof_manifest(
    path: Path,
    *,
    package_root: Path,
    expected_trace_schema_sha256: str,
) -> dict[str, object]:
    proof = _load_json(
        path, schema=PROOF_MANIFEST_SCHEMA, label="Lean proof manifest"
    )
    toolchain = proof.get("toolchain")
    policy = proof.get("proof_policy")
    progress = proof.get("progress")
    model = proof.get("model")
    artifacts = proof.get("artifacts")
    if (
        not isinstance(toolchain, Mapping)
        or toolchain.get("lean") != "leanprover/lean4:v4.26.0"
        or toolchain.get("upstream_commit")
        != "d8204c9fd894f91bbb2cdfec5912ec8196fd8562"
        or toolchain.get("lake") != "5.0.0-src+d8204c9"
        or toolchain.get("external_packages") != 0
        or not isinstance(policy, Mapping)
        or policy.get("runtime_boolean_is_not_a_theorem") is not True
        or policy.get("theorems_are_propositions_over_transition") is not True
        or policy.get("boolean_substitutes") != []
        or policy.get("safety_fairness_hypotheses") is not False
        or set(policy.get("forbidden_lean_tokens", []))
        != FORBIDDEN_PROOF_TOKENS
        or not isinstance(progress, Mapping)
        or progress.get("unconditional_claim") is not False
        or progress.get("required_assumptions") != PROGRESS_ASSUMPTIONS
        or not isinstance(model, Mapping)
        or model.get("authoritative_transition")
        != "ResilientProtocol.transition"
        or model.get("executable_trace_fold")
        != "ResilientProtocol.executeEvents"
        or model.get("trace_schema") != TRACE_SCHEMA
        or model.get("trace_schema_sha256")
        != expected_trace_schema_sha256
        or not isinstance(artifacts, list)
    ):
        raise ValueError(
            "Lean proof manifest is partial, unsafe, or overclaims progress"
        )
    roles: set[str] = set()
    paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("Lean proof artifact entry is invalid")
        relative_text = str(artifact.get("path", ""))
        relative = Path(relative_text)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in paths
        ):
            raise ValueError("Lean proof artifact path is invalid or duplicate")
        paths.add(relative_text)
        roles.add(str(artifact.get("role", "")))
        expected = require_digest(
            artifact.get("sha256"), f"Lean proof artifact {relative_text}"
        )
        target = (package_root / relative).resolve()
        try:
            target.relative_to(package_root.resolve())
        except ValueError as error:
            raise ValueError("Lean proof artifact escapes package") from error
        if not target.is_file() or file_sha256(target) != expected:
            raise ValueError(
                f"Lean proof artifact digest mismatch: {relative_text}"
            )
    if not REQUIRED_PROOF_ROLES.issubset(roles):
        raise ValueError("Lean theorem/proof coverage manifest is partial")
    if model.get("kernel_sha256") != next(
        artifact["sha256"]
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("role") == "executable_transition"
    ):
        raise ValueError("Lean executable kernel digest mismatch")
    return proof


def _validate_stress_manifest(
    path: Path,
    *,
    hardening_sha256: str,
) -> dict[str, object]:
    stress = _load_json(
        path, schema=STRESS_MANIFEST_SCHEMA, label="schedule-stress manifest"
    )
    hardening = stress.get("harden_binding")
    counts = stress.get("counts")
    generator = stress.get("generator")
    determinism = stress.get("determinism")
    execution = stress.get("execution")
    forbidden = stress.get("forbidden_facilities")
    if (
        stress.get("status") != "passed"
        or stress.get("authoritative_scope")
        != "native-transition-safety-only"
        or not isinstance(hardening, Mapping)
        or hardening.get("manifest_sha256") != hardening_sha256
        or not isinstance(counts, Mapping)
        or int(counts.get("systematic_schedules", 0)) <= 0
        or int(counts.get("random_schedules", 0)) <= 0
        or int(counts.get("permanent_corpus_schedules", 0)) < 3
        or counts.get("native_safety_failures") != 0
        or counts.get("known_bad_detected") != 2
        or counts.get("known_bad_minimized") != 2
        or not isinstance(generator, Mapping)
        or generator.get("prng") != "pcg-xsh-rr-64-32-v1"
        or not generator.get("causal_preconditions")
        or "ddmin" not in str(generator.get("shrink_order", ""))
        or not isinstance(determinism, Mapping)
        or determinism.get("byte_identical") is not True
        or int(determinism.get("full_campaign_repeats", 0)) < 2
        or not isinstance(stress.get("random_seed_partitions"), list)
        or not stress.get("random_seed_partitions")
        or not isinstance(execution, Mapping)
        or not execution.get("replay_random_template")
        or not execution.get("replay_corpus_template")
        or not isinstance(forbidden, Mapping)
        or any(value != 0 for value in forbidden.values())
    ):
        raise ValueError(
            "schedule-stress evidence is failed, partial, or non-replayable"
        )
    return stress


def _validate_corpus(
    path: Path,
    *,
    trace_schema_sha256: str,
) -> tuple[dict[str, object], Mapping[str, object]]:
    corpus = _load_json(
        path, schema=CORPUS_MANIFEST_SCHEMA, label="conformance corpus"
    )
    entries = corpus.get("entries")
    if (
        corpus.get("traceSchema") != TRACE_SCHEMA
        or corpus.get("traceSchemaDigest") != trace_schema_sha256
        or not isinstance(entries, list)
        or len(entries) < 15
    ):
        raise ValueError("permanent conformance corpus is partial or stale")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("id") == JOB_5105811_TRACE_ID
    ]
    if len(matches) != 1 or int(matches[0].get("events", 0)) != 63:
        raise ValueError("permanent job-5105811 corpus entry is missing")
    require_digest(matches[0].get("sha256"), "job-5105811 trace")
    return corpus, matches[0]


def _validate_conformance_manifest(
    path: Path,
    *,
    repository: Path,
    corpus_sha256: str,
    job_5105811_sha256: str,
    proof_kernel_sha256: str,
    trace_schema_sha256: str,
    expected_bundle_sha256: str,
) -> dict[str, object]:
    conformance = _load_json(
        path,
        schema=CONFORMANCE_MANIFEST_SCHEMA,
        label="production-native differential manifest",
    )
    native = conformance.get("native")
    lean = conformance.get("lean")
    corpus = conformance.get("corpus")
    source_digests = (
        native.get("source_sha256")
        if isinstance(native, Mapping)
        else None
    )
    artifact_digests = (
        native.get("artifact_sha256")
        if isinstance(native, Mapping)
        else None
    )
    if (
        conformance.get("manifest_digest")
        != manifest_digest(conformance)
        or
        conformance.get("status") != "passed"
        or conformance.get("complete") is not True
        or conformance.get("partial") is not False
        or conformance.get("evaluator_only") is not False
        or conformance.get("zero_divergences") != 0
        or int(conformance.get("traces_passed", 0)) < 15
        or int(conformance.get("events_compared", 0)) < 486
        or conformance.get("production_transition_path") is not True
        or not isinstance(native, Mapping)
        or native.get("source_tree_dirty") is not False
        or native.get("bundle_sha256") != expected_bundle_sha256
        or not isinstance(source_digests, Mapping)
        or not REQUIRED_NATIVE_SOURCE_PATHS.issubset(source_digests)
        or not isinstance(artifact_digests, Mapping)
        or not {"service_binary", "local_library", "transport_library"}
        .issubset(artifact_digests)
        or not all(
            require_digest(value, f"native artifact {name}")
            for name, value in artifact_digests.items()
        )
        or not isinstance(native.get("abi"), Mapping)
        or native["abi"].get("coordination") != "NDP_COORD_ABI_V1"
        or any(
            isinstance(native["abi"].get(field), bool)
            or not isinstance(native["abi"].get(field), int)
            or int(native["abi"][field]) <= 0
            for field in (
                "local",
                "event_struct_bytes",
                "member_struct_bytes",
                "result_struct_bytes",
            )
        )
        or not isinstance(lean, Mapping)
        or lean.get("kernel_sha256") != proof_kernel_sha256
        or lean.get("trace_schema_sha256") != trace_schema_sha256
        or require_digest(lean.get("runner_sha256"), "Lean runner")
        == "0" * 64
        or not isinstance(corpus, Mapping)
        or corpus.get("manifest_sha256") != corpus_sha256
        or corpus.get("job_5105811_trace_sha256") != job_5105811_sha256
    ):
        raise ValueError(
            "production-native differential evidence is stale, partial, "
            "failed, or does not bind the proved kernel"
        )
    for relative, expected in source_digests.items():
        if relative not in REQUIRED_NATIVE_SOURCE_PATHS:
            continue
        target = (repository / relative).resolve()
        try:
            target.relative_to(repository.resolve())
        except ValueError as error:
            raise ValueError("native source digest escapes repository") from error
        if (
            not target.is_file()
            or file_sha256(target)
            != require_digest(expected, f"native source {relative}")
        ):
            raise ValueError(f"production-native source digest mismatch: {relative}")
    adapter_digest = source_digests.get("ndm/native_lean_conformance.py")
    if native.get("trace_adapter_sha256") != adapter_digest:
        raise ValueError("production trace-adapter digest mismatch")
    return conformance


def validate_formal_coordination_gate(
    path: str | Path,
    *,
    repository: str | Path,
    evidence_root: str | Path,
    expected_nodes: int,
    expected_identities: Mapping[str, str],
    expected_identity_contract: Mapping[str, object],
    scale_authorization_digest: str,
    passed_8_node_manifest: Mapping[str, object],
    passed_8_node_sha256: str,
) -> dict[str, object]:
    """Validate the one immutable formal/native authorization for scale."""
    gate_path = Path(path).resolve()
    gate = _load_json(
        gate_path,
        schema=FORMAL_COORDINATION_GATE_SCHEMA,
        label="formal coordination scale gate",
    )
    encoded_digest = gate.get("manifest_digest")
    if encoded_digest != manifest_digest(gate):
        raise ValueError("formal coordination scale gate manifest digest mismatch")
    if (
        gate.get("status") != "passed"
        or gate.get("complete") is not True
        or gate.get("partial") is not False
        or gate.get("evaluator_only") is not False
        or gate.get("stale") is not False
        or expected_nodes not in gate.get("authorizes_nodes", [])
        or gate.get("requirements") != REQUIREMENTS
        or gate.get("trust_boundary") != TRUST_BOUNDARY
        or gate.get("identities") != dict(expected_identities)
        or gate.get("identity_contract") != dict(expected_identity_contract)
        or gate.get("identity_digest")
        != canonical_digest(
            {
                "identities": dict(expected_identities),
                "identity_contract": dict(expected_identity_contract),
            }
        )
    ):
        raise ValueError(
            "formal coordination scale gate is stale, partial, evaluator-only, "
            "or does not authorize this exact payload"
        )

    repository_path = Path(repository).resolve()
    evidence_path = Path(evidence_root).resolve()
    references = gate.get("artifacts")
    lineage = gate.get("lineage")
    if not isinstance(references, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError("formal coordination evidence/lineage is incomplete")
    expected_reference_names = {
        "native_hardening",
        "schedule_stress",
        "proof_manifest",
        "proof_coverage",
        "trace_schema",
        "conformance_corpus",
        "native_differential",
    }
    if set(references) != expected_reference_names:
        raise ValueError("formal coordination artifact set is partial")
    resolved = {
        name: _resolve_reference(
            references[name],
            repository=repository_path,
            evidence_root=evidence_path,
            label=name.replace("_", " "),
        )
        for name in expected_reference_names
    }

    hardening_sha = file_sha256(resolved["native_hardening"])
    stress_sha = file_sha256(resolved["schedule_stress"])
    proof_manifest_sha = file_sha256(resolved["proof_manifest"])
    proof_coverage_sha = file_sha256(resolved["proof_coverage"])
    trace_schema_sha = file_sha256(resolved["trace_schema"])
    corpus_sha = file_sha256(resolved["conformance_corpus"])
    conformance_sha = file_sha256(resolved["native_differential"])
    for name, actual in (
        ("native_hardening_sha256", hardening_sha),
        ("schedule_stress_sha256", stress_sha),
        ("proof_manifest_sha256", proof_manifest_sha),
        ("proof_coverage_sha256", proof_coverage_sha),
        ("trace_schema_sha256", trace_schema_sha),
        ("conformance_corpus_sha256", corpus_sha),
        ("native_differential_sha256", conformance_sha),
    ):
        if lineage.get(name) != actual:
            raise ValueError(f"formal coordination lineage mismatch: {name}")

    if (
        lineage.get("scale_authorization_digest")
        != require_digest(
            scale_authorization_digest, "scale authorization manifest"
        )
        or lineage.get("passed_8_node_manifest_digest")
        != passed_8_node_manifest.get("manifest_digest")
        or lineage.get("passed_8_node_sha256")
        != require_digest(passed_8_node_sha256, "passed 8-node manifest")
    ):
        raise ValueError(
            "formal coordination gate does not bind the exact authorization "
            "and passed 8-node verdict"
        )
    native_lineage = passed_8_node_manifest.get("native_coordination_lineage")
    if (
        passed_8_node_manifest.get("schema")
        != "emender-async-v21-direct-rung-pass-v2"
        or passed_8_node_manifest.get("status") != "passed"
        or passed_8_node_manifest.get("nodes") != 8
        or not isinstance(native_lineage, Mapping)
        or native_lineage.get("hardening_manifest_sha256") != hardening_sha
        or native_lineage.get("schedule_stress_manifest_sha256") != stress_sha
    ):
        raise ValueError(
            "passed 8-node verdict did not consume this exact native "
            "hardening/schedule-stress lineage"
        )

    try:
        trace_schema = json.loads(
            resolved["trace_schema"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("canonical trace schema is corrupt") from error
    if (
        not isinstance(trace_schema, Mapping)
        or not str(trace_schema.get("$id", "")).endswith(
            f"/{TRACE_SCHEMA}.json"
        )
        or trace_schema.get("additionalProperties") is not False
    ):
        raise ValueError("canonical trace schema identity is stale")
    stress = _validate_stress_manifest(
        resolved["schedule_stress"], hardening_sha256=hardening_sha
    )
    stress_identities = stress.get("identities")
    if (
        not isinstance(stress_identities, Mapping)
        or native_lineage.get("native_kernel_sha256")
        != stress_identities.get("native_binary_sha256")
    ):
        raise ValueError("passed 8-node native kernel differs from stress lineage")

    proof = _validate_proof_manifest(
        resolved["proof_manifest"],
        package_root=resolved["proof_manifest"].parent,
        expected_trace_schema_sha256=trace_schema_sha,
    )
    proof_artifacts = {
        str(item["role"]): str(item["sha256"])
        for item in proof["artifacts"]
        if isinstance(item, Mapping)
    }
    if (
        proof_artifacts.get("transition_requirement_crosswalk")
        != proof_coverage_sha
    ):
        raise ValueError("proof coverage/assumption manifest digest mismatch")
    corpus, job_entry = _validate_corpus(
        resolved["conformance_corpus"],
        trace_schema_sha256=trace_schema_sha,
    )
    del corpus
    _validate_conformance_manifest(
        resolved["native_differential"],
        repository=repository_path,
        corpus_sha256=corpus_sha,
        job_5105811_sha256=str(job_entry["sha256"]),
        proof_kernel_sha256=str(
            proof["model"]["kernel_sha256"]  # type: ignore[index]
        ),
        trace_schema_sha256=trace_schema_sha,
        expected_bundle_sha256=str(expected_identities["bundle_digest"]),
    )
    return gate
