#!/usr/bin/env python3
"""Build immutable local or joined formal/native coordination evidence.

The ``local`` subcommand records the proved-kernel, production-native
differential, permanent-corpus, and deterministic-stress evidence available
without a Slurm allocation.  Its output is deliberately non-authorizing.

The ``join`` subcommand binds that local evidence to the exact signed scale
authorization and collector-backed passed eight-node rung.  It emits an
unsigned review candidate; an Ed25519 reviewer must sign the candidate before
the 32-node controller accepts it.  Neither subcommand submits or queries
Slurm, and compute nodes do not execute Lean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.formal_coordination_gate import (  # noqa: E402
    FORMAL_COORDINATION_GATE_SCHEMA,
    FORMAL_COORDINATION_LOCAL_SCHEMA,
    REQUIREMENTS,
    TRUST_BOUNDARY,
    canonical_digest,
    file_sha256,
    manifest_digest,
)


ARTIFACT_PATHS = {
    "native_hardening": (
        "docs/validation/harden-native-coordination-kernel-20260728.md"
    ),
    "schedule_stress": "reports/frontier/native-coordination-stress-v1.json",
    "proof_manifest": "formal/resilient/proof-manifest-v1.json",
    "proof_coverage": "formal/resilient/PROOF_COVERAGE.md",
    "trace_schema": "formal/resilient/trace-schema-v1.json",
    "conformance_corpus": "formal/resilient/corpus/native-v1/manifest.json",
    "native_differential": (
        "reports/conformance/native-lean-v1/manifest.json"
    ),
}


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or corrupt: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _artifact_references() -> dict[str, object]:
    references: dict[str, object] = {}
    for name, relative in ARTIFACT_PATHS.items():
        target = ROOT / relative
        if not target.is_file():
            raise ValueError(f"required local artifact is missing: {relative}")
        references[name] = {
            "root": "repository",
            "path": relative,
            "sha256": file_sha256(target),
        }
    return references


def _local(arguments: argparse.Namespace) -> int:
    artifacts = _artifact_references()
    proof = _read_json(
        ROOT / ARTIFACT_PATHS["proof_manifest"], "proof manifest"
    )
    conformance = _read_json(
        ROOT / ARTIFACT_PATHS["native_differential"],
        "native differential manifest",
    )
    stress = _read_json(
        ROOT / ARTIFACT_PATHS["schedule_stress"],
        "schedule-stress manifest",
    )
    if (
        proof.get("schema") != "emender-resilient-lean-proof-manifest-v1"
        or conformance.get("status") != "passed"
        or stress.get("status") != "passed"
    ):
        raise ValueError("local formal/native evidence has not passed")
    value: dict[str, object] = {
        "schema": FORMAL_COORDINATION_LOCAL_SCHEMA,
        "status": "local_passed_pending_8_node_join",
        "complete_local_evidence": True,
        "authorizes_nodes": [],
        "execution_source": {
            "schema": "emender-async-v21-execution-source-v1",
            "digest": arguments.execution_source_digest,
            "commit": arguments.source_commit,
            "source_tree_dirty": False,
        },
        "requirements": REQUIREMENTS,
        "trust_boundary": TRUST_BOUNDARY,
        "artifacts": artifacts,
        "lineage": {
            f"{name}_sha256": reference["sha256"]
            for name, reference in artifacts.items()
        },
        "non_authorization_reason": (
            "the exact collector-backed passed 8-node manifest and its "
            "reviewed scale authorization are intentionally not available "
            "to this local non-submitting task"
        ),
    }
    value["manifest_digest"] = manifest_digest(value)
    _write(arguments.output.resolve(), value)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "sha256": file_sha256(arguments.output.resolve()),
                "manifest_digest": value["manifest_digest"],
                "authorizes_nodes": [],
            },
            sort_keys=True,
        )
    )
    return 0


def _join(arguments: argparse.Namespace) -> int:
    local = _read_json(arguments.local.resolve(), "local formal evidence")
    authorization = _read_json(
        arguments.authorization.resolve(), "scale authorization"
    )
    passed_8 = _read_json(arguments.passed_8.resolve(), "passed 8-node rung")
    if (
        local.get("schema") != FORMAL_COORDINATION_LOCAL_SCHEMA
        or local.get("status") != "local_passed_pending_8_node_join"
        or local.get("manifest_digest") != manifest_digest(local)
        or authorization.get("status") != "passed"
        or authorization.get("authorized_nodes") != 32
        or passed_8.get("status") != "passed"
        or passed_8.get("nodes") != 8
        or authorization.get("identities") != passed_8.get("identities")
        or authorization.get("identity_contract")
        != passed_8.get("identity_contract")
        or authorization.get("identity_digest")
        != passed_8.get("identity_digest")
    ):
        raise ValueError(
            "local evidence, 32-node authorization, and passed 8-node "
            "identity do not form one exact lineage"
        )
    native_lineage = passed_8.get("native_coordination_lineage")
    lineage = local.get("lineage")
    if (
        not isinstance(native_lineage, Mapping)
        or not isinstance(lineage, Mapping)
        or native_lineage.get("hardening_manifest_sha256")
        != lineage.get("native_hardening_sha256")
        or native_lineage.get("schedule_stress_manifest_sha256")
        != lineage.get("schedule_stress_sha256")
    ):
        raise ValueError(
            "passed 8-node verdict did not consume the local native lineage"
        )
    identity_contract = dict(authorization["identity_contract"])
    identities = dict(authorization["identities"])
    value: dict[str, object] = {
        "schema": FORMAL_COORDINATION_GATE_SCHEMA,
        "status": "passed",
        "complete": True,
        "partial": False,
        "evaluator_only": False,
        "stale": False,
        "authorizes_nodes": [32],
        "identities": identities,
        "identity_contract": identity_contract,
        "identity_digest": canonical_digest(
            {
                "identities": identities,
                "identity_contract": identity_contract,
            }
        ),
        "requirements": REQUIREMENTS,
        "trust_boundary": TRUST_BOUNDARY,
        "artifacts": dict(local["artifacts"]),
        "lineage": {
            **dict(lineage),
            "scale_authorization_digest": authorization["manifest_digest"],
            "passed_8_node_manifest_digest": passed_8["manifest_digest"],
            "passed_8_node_sha256": file_sha256(arguments.passed_8.resolve()),
        },
        "review_signature": {
            "status": "unsigned-review-candidate",
            "signing_domain": "emender-async-v21-review-v1",
        },
    }
    value["manifest_digest"] = manifest_digest(value)
    _write(arguments.output.resolve(), value)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "sha256": file_sha256(arguments.output.resolve()),
                "manifest_digest": value["manifest_digest"],
                "review_required": True,
                "authorizes_without_review": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    local = subcommands.add_parser("local")
    local.add_argument("--output", required=True, type=Path)
    local.add_argument("--execution-source-digest", required=True)
    local.add_argument("--source-commit", required=True)
    local.set_defaults(handler=_local)

    join = subcommands.add_parser("join")
    join.add_argument("--local", required=True, type=Path)
    join.add_argument("--authorization", required=True, type=Path)
    join.add_argument("--passed-8", required=True, type=Path)
    join.add_argument("--output", required=True, type=Path)
    join.set_defaults(handler=_join)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
