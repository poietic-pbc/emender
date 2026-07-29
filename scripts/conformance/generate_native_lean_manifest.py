#!/usr/bin/env python3
"""Generate the immutable zero-divergence production-native/Lean manifest."""

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
    JOB_5105811_TRACE_ID,
    canonical_digest,
    file_sha256,
)
from ndm.native_lean_conformance import (  # noqa: E402
    canonical_json,
    run_differential_trace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=ROOT
        / "build/native-resilient-dataplane/native-artifacts.json",
    )
    parser.add_argument(
        "--lean-runner",
        type=Path,
        default=ROOT
        / "formal/resilient/.lake/build/bin/resilient-conformance",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "formal/resilient/corpus/native-v1/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/conformance/native-lean-v1/manifest.json",
    )
    parser.add_argument(
        "--job-5105811-output",
        type=Path,
        default=ROOT
        / "reports/conformance/native-lean-v1/job-5105811-agreement.json",
    )
    return parser


def _artifact_digests(build: Mapping[str, object]) -> dict[str, str]:
    artifacts = build.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("native build manifest has no artifacts")
    required = ("service_binary", "local_library", "transport_library")
    result: dict[str, str] = {}
    for name in required:
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"native build manifest is missing {name}")
        digest = str(artifact.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError(f"native build {name} digest is invalid")
        result[name] = digest
    return result


def main() -> int:
    arguments = _parser().parse_args()
    build_path = arguments.build_manifest.resolve()
    runner = arguments.lean_runner.resolve()
    corpus_path = arguments.corpus_manifest.resolve()
    build = json.loads(build_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if (
        not isinstance(build, dict)
        or build.get("schema") != "emender-native-dataplane-build-v1"
        or build.get("source_tree_dirty") is not False
        or not isinstance(corpus, dict)
        or corpus.get("schema")
        != "emender-native-lean-conformance-corpus-v1"
        or not isinstance(corpus.get("entries"), list)
    ):
        raise ValueError("clean native build and canonical corpus are required")

    total_events = 0
    trace_results: list[dict[str, object]] = []
    common_identity: Mapping[str, object] | None = None
    for entry in corpus["entries"]:
        if not isinstance(entry, Mapping):
            raise ValueError("corpus entry is not an object")
        trace = ROOT / str(entry["path"])
        if (
            not trace.is_file()
            or file_sha256(trace) != entry.get("sha256")
        ):
            raise ValueError(f"corpus trace digest mismatch: {trace}")
        report = run_differential_trace(
            trace_path=trace,
            build_manifest=build_path,
            lean_runner=runner,
            repository=ROOT,
            divergence_directory=arguments.output.parent / "divergences",
        )
        if (
            report.get("verdict") != "agreement"
            or report.get("traceId") != entry.get("id")
            or report.get("traceSha256") != entry.get("sha256")
            or report.get("events") != entry.get("events")
        ):
            raise ValueError(f"differential trace did not agree: {entry['id']}")
        identity = report.get("identityManifest")
        if not isinstance(identity, Mapping):
            raise ValueError("differential report has no identity manifest")
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("differential corpus used inconsistent binaries")
        if entry.get("id") == JOB_5105811_TRACE_ID:
            arguments.job_5105811_output.parent.mkdir(
                parents=True, exist_ok=True
            )
            arguments.job_5105811_output.write_text(
                canonical_json(report) + "\n", encoding="utf-8"
            )
        total_events += int(report["events"])
        trace_results.append(
            {
                "id": entry["id"],
                "events": report["events"],
                "trace_sha256": report["traceSha256"],
                "final_state_digest": report["finalStateDigest"],
                "verdict": report["verdict"],
            }
        )

    if common_identity is None:
        raise ValueError("permanent conformance corpus is empty")
    native_identity = common_identity.get("native")
    lean_identity = common_identity.get("lean")
    source_identities = common_identity.get("sourceIdentities")
    call_path = common_identity.get("callPath")
    if (
        not isinstance(native_identity, Mapping)
        or not isinstance(lean_identity, Mapping)
        or not isinstance(source_identities, list)
        or not isinstance(call_path, list)
        or call_path[-1:] != ["coordination::step"]
    ):
        raise ValueError("production differential identity is incomplete")
    source_sha256 = {
        str(item["path"]): str(item["sha256"])
        for item in source_identities
        if isinstance(item, Mapping)
        and str(item.get("path", "")).startswith(
            ("ndm/", "src/native_resilient_dataplane/")
        )
    }
    trace_schema = ROOT / "formal/resilient/trace-schema-v1.json"
    job_entry = next(
        entry
        for entry in corpus["entries"]
        if entry.get("id") == JOB_5105811_TRACE_ID
    )
    value: dict[str, object] = {
        "schema": "emender-native-lean-conformance-manifest-v1",
        "status": "passed",
        "complete": True,
        "partial": False,
        "evaluator_only": False,
        "zero_divergences": 0,
        "traces_passed": len(trace_results),
        "events_compared": total_events,
        "production_transition_path": True,
        "call_path": call_path,
        "native": {
            "source_commit": native_identity.get("sourceCommit"),
            "source_tree_dirty": native_identity.get("sourceTreeDirty"),
            "bundle_sha256": native_identity.get("bundleSha256"),
            "source_sha256": source_sha256,
            "trace_adapter_sha256": source_sha256[
                "ndm/native_lean_conformance.py"
            ],
            "artifact_sha256": _artifact_digests(build),
            "abi": {
                "coordination": "NDP_COORD_ABI_V1",
                "local": native_identity.get("localAbi"),
                "event_struct_bytes": native_identity.get(
                    "eventStructBytes"
                ),
                "member_struct_bytes": native_identity.get(
                    "memberStructBytes"
                ),
                "result_struct_bytes": native_identity.get(
                    "resultStructBytes"
                ),
            },
        },
        "lean": {
            "toolchain": lean_identity.get("toolchain"),
            "runner_sha256": lean_identity.get("runnerSha256"),
            "kernel_sha256": file_sha256(
                ROOT
                / "formal/resilient/ResilientProtocol/Kernel.lean"
            ),
            "proof_manifest_sha256": file_sha256(
                ROOT / "formal/resilient/proof-manifest-v1.json"
            ),
            "trace_schema_sha256": file_sha256(trace_schema),
        },
        "corpus": {
            "manifest_sha256": file_sha256(corpus_path),
            "trace_schema": corpus.get("traceSchema"),
            "trace_schema_sha256": corpus.get("traceSchemaDigest"),
            "job_5105811_trace_id": JOB_5105811_TRACE_ID,
            "job_5105811_trace_sha256": job_entry["sha256"],
        },
        "traces": trace_results,
        "boundary_nonclaims": [
            "numerical_parity",
            "immutable_snapshot_ownership",
            "foreground_timing_and_tails",
            "transport_bytes_and_cxi",
            "g2",
            "two_node_qualification",
            "scheduler_queue",
            "eight_node_or_scale_policy",
        ],
        "exact_command": (
            f"{sys.executable} {Path(__file__).relative_to(ROOT)} "
            f"--build-manifest {build_path} --lean-runner {runner} "
            f"--corpus-manifest {corpus_path} "
            f"--output {arguments.output.resolve()}"
        ),
    }
    value["manifest_digest"] = canonical_digest(value)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "sha256": file_sha256(arguments.output.resolve()),
                "manifest_digest": value["manifest_digest"],
                "traces_passed": len(trace_results),
                "events_compared": total_events,
                "zero_divergences": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
