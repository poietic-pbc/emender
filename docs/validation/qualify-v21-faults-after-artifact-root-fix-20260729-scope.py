#!/usr/bin/env python3
"""Build the fail-closed v2.1 qualification change-scope certificate.

This generator deliberately reads both source revisions through Git instead of
the worktree.  The certificate therefore remains reproducible after the
certificate itself is committed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable


TASK_ID = "qualify-v21-faults-after-artifact-root-fix"
BASELINE = "b756c14fa3a7495e733c6498f69730c3e115f281"
CANDIDATE = "c9ba89a637426b68aa6f18204de776ab7ba669b1"
CANDIDATE_PARENT = "9523afde4d5a95103e25cd3bae4eaf5523dd2108"
OUTPUT = Path(
    "docs/validation/"
    "qualify-v21-faults-after-artifact-root-fix-20260729.json"
)
BASELINE_ROOT = Path(
    "/lustre/orion/bif148/scratch/erikgarrison/"
    "emender-qualification/qualify-simple-async-v21-2n-faults/"
    f"{BASELINE}"
)
PROOF_ROOT = Path(
    "/lustre/orion/bif148/scratch/erikgarrison/"
    "emender-qualification/qualify-v21-faults-after-artifact-root-fix/"
    f"{CANDIDATE}/local-ownership-proof"
)

# This is the complete eight-path delta of the artifact-root ownership fix
# itself (CANDIDATE_PARENT..CANDIDATE), not an inferred prefix allowlist.
OWNERSHIP_FIX_PATHS = {
    "docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md",
    "docs/validation/fix-g2-artifact-root-ownership-20260729.md",
    "reports/frontier/native-dataplane-reference-v1.json",
    "reports/frontier/native-dataplane-reference-v1.md",
    "scripts/frontier/native_dataplane_2n_gate.sbatch",
    "scripts/frontier/native_g2_artifact_namespace.py",
    "scripts/frontier/submit_native_dataplane_2n_gate.sh",
    "tests/test_native_g2_artifact_namespace.py",
}

R_IDS = [f"R{i:02d}" for i in range(1, 17)]
NDP_IDS = [f"NDP{i:02d}" for i in range(1, 18)]
V21S_IDS = [f"V21S{i:02d}" for i in range(1, 18)]
ISP_IDS = [f"ISP{i:02d}" for i in range(1, 8)]


def run_git(*args: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(revision: str, path: str) -> tuple[str | None, bytes | None]:
    spec = f"{revision}:{path}"
    probe = subprocess.run(
        ["git", "cat-file", "-e", spec],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        return None, None
    oid = run_git("rev-parse", spec).decode().strip()
    return oid, run_git("show", spec)


def tracked_paths(revision: str) -> set[str]:
    return set(
        run_git("ls-tree", "-r", "--name-only", revision)
        .decode()
        .splitlines()
    )


def tree_digest(revision: str, paths: Iterable[str]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    accumulator = hashlib.sha256()
    accumulator.update(b"emender-protected-source-set-v1\0")
    for path in sorted(set(paths)):
        oid, content = git_blob(revision, path)
        if oid is None or content is None:
            entry = {
                "path": path,
                "present": False,
                "git_blob_sha1": None,
                "sha256": None,
                "bytes": None,
            }
        else:
            entry = {
                "path": path,
                "present": True,
                "git_blob_sha1": oid,
                "sha256": sha256_bytes(content),
                "bytes": len(content),
            }
        encoded = canonical_bytes(entry)
        accumulator.update(len(encoded).to_bytes(8, "big"))
        accumulator.update(encoded)
        entries.append(entry)
    return {
        "algorithm": "sha256(domain || length || canonical-entry)",
        "domain": "emender-protected-source-set-v1",
        "digest": accumulator.hexdigest(),
        "path_count": len(entries),
        "entries": entries,
    }


def exact(*paths: str) -> Callable[[str], bool]:
    selected = set(paths)
    return lambda path: path in selected


def prefixes(*values: str) -> Callable[[str], bool]:
    return lambda path: any(path.startswith(value) for value in values)


def any_of(
    *selectors: Callable[[str], bool],
) -> Callable[[str], bool]:
    return lambda path: any(selector(path) for selector in selectors)


def protected_surface(
    name: str,
    selector: Callable[[str], bool],
    universe: set[str],
) -> dict[str, Any]:
    paths = sorted(path for path in universe if selector(path))
    baseline = tree_digest(BASELINE, paths)
    candidate = tree_digest(CANDIDATE, paths)
    return {
        "name": name,
        "paths": paths,
        "baseline": baseline,
        "candidate": candidate,
        "unchanged": baseline["digest"] == candidate["digest"],
    }


def artifact(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    exists = path.is_file()
    actual = sha256_file(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "matches_expected": (
            None
            if expected_sha256 is None
            else exists and actual == expected_sha256
        ),
    }


def changed_paths() -> list[dict[str, Any]]:
    raw = run_git(
        "diff", "--name-status", "--no-renames", BASELINE, CANDIDATE
    ).decode()
    result: list[dict[str, Any]] = []
    for line in raw.splitlines():
        status, path = line.split("\t", 1)
        base_oid, base_content = git_blob(BASELINE, path)
        parent_oid, parent_content = git_blob(CANDIDATE_PARENT, path)
        candidate_oid, candidate_content = git_blob(CANDIDATE, path)
        in_allowlist = path in OWNERSHIP_FIX_PATHS
        changed_before_fix = base_oid != parent_oid
        changed_by_fix = parent_oid != candidate_oid
        pure_fix_delta = (
            in_allowlist and not changed_before_fix and changed_by_fix
        )
        result.append(
            {
                "status": status,
                "path": path,
                "baseline": {
                    "git_blob_sha1": base_oid,
                    "sha256": (
                        sha256_bytes(base_content)
                        if base_content is not None
                        else None
                    ),
                    "bytes": (
                        len(base_content)
                        if base_content is not None
                        else None
                    ),
                },
                "candidate_parent": {
                    "git_blob_sha1": parent_oid,
                    "sha256": (
                        sha256_bytes(parent_content)
                        if parent_content is not None
                        else None
                    ),
                    "bytes": (
                        len(parent_content)
                        if parent_content is not None
                        else None
                    ),
                },
                "candidate": {
                    "git_blob_sha1": candidate_oid,
                    "sha256": (
                        sha256_bytes(candidate_content)
                        if candidate_content is not None
                        else None
                    ),
                    "bytes": (
                        len(candidate_content)
                        if candidate_content is not None
                        else None
                    ),
                },
                "in_exact_ownership_fix_allowlist": in_allowlist,
                "changed_before_ownership_fix_commit": changed_before_fix,
                "changed_by_ownership_fix_commit": changed_by_fix,
                "pure_ownership_fix_delta_from_clean_source": pure_fix_delta,
            }
        )
    return result


def requirement_mapping(
    identifiers: Iterable[str],
    authority: str,
) -> dict[str, Any]:
    return {
        identifier: {
            "authority": authority,
            "disposition": "blocked_at_change_scope_gate",
            "new_evidence": [
                str(OUTPUT),
                str(PROOF_ROOT / "ARTIFACT-OWNERSHIP.json"),
            ],
            "retained_clean_evidence": [
                str(
                    BASELINE_ROOT
                    / "clean/terminal-collector/"
                    "6f02204cdddc6512d05d8cbeaaeb8d9a08952ffb9837fe825"
                    "bea347d0269e0d7/terminal-verdict.json"
                ),
                str(
                    BASELINE_ROOT
                    / "clean/clean-overlap/pipelined-performance.json"
                ),
            ],
            "qualification_claim": False,
            "reason": (
                "The mandatory source-scope identity gate failed before "
                "sbatch; retained clean evidence is cited but not reusable "
                "for this candidate."
            ),
        }
        for identifier in identifiers
    }


def build() -> dict[str, Any]:
    for revision in (BASELINE, CANDIDATE_PARENT, CANDIDATE):
        run_git("cat-file", "-e", f"{revision}^{{commit}}")

    ownership_commit_paths = set(
        run_git(
            "diff",
            "--name-only",
            "--no-renames",
            CANDIDATE_PARENT,
            CANDIDATE,
        )
        .decode()
        .splitlines()
    )
    if ownership_commit_paths != OWNERSHIP_FIX_PATHS:
        raise RuntimeError(
            "ownership-fix allowlist does not exactly match the frozen "
            "candidate commit"
        )

    paths = changed_paths()
    universe = tracked_paths(BASELINE) | tracked_paths(CANDIDATE)

    protected = [
        protected_surface(
            "trainer_code",
            any_of(
                exact(
                    "scripts/frontier/resilient_e97_role.py",
                    "scripts/frontier/resilient_e97_rank_lane.py",
                    "scripts/frontier/resilient_e97_node_step_supervisor.py",
                    "scripts/frontier/resilient_e97_allocation_supervisor.py",
                    "ndm/resilient_e97_roles.py",
                    "ndm/resilient_e97_runtime.py",
                    "ndm/native_e97_runtime.py",
                ),
                prefixes("ndm/triton/"),
            ),
            universe,
        ),
        protected_surface(
            "manager_and_native_coordination_kernel",
            any_of(
                exact(
                    "ndm/resilient_pool_runtime.py",
                    "ndm/native_pool_runtime.py",
                    "ndm/native_coordination.py",
                    "src/native_resilient_dataplane/src/"
                    "coordination_kernel.cpp",
                    "src/native_resilient_dataplane/src/"
                    "coordination_kernel.hpp",
                    "src/native_resilient_dataplane/src/service_core.hpp",
                    "src/native_resilient_dataplane/src/rpc_server.cpp",
                    "src/native_resilient_dataplane/src/rpc_protocol.hpp",
                ),
            ),
            universe,
        ),
        protected_surface(
            "native_bundle_and_abi_sources",
            any_of(
                prefixes("src/native_resilient_dataplane/"),
                exact(
                    "ndm/native_dataplane.py",
                    "ndm/native_e97_runtime.py",
                    "scripts/frontier/"
                    "build_native_resilient_dataplane.sh",
                    "scripts/frontier/attest_native_dataplane.py",
                ),
            ),
            universe,
        ),
        protected_surface(
            "model_and_outer_math",
            exact(
                "ndm/async_diloco_v2.py",
                "ndm/resilient_e97_reducer.py",
                "ndm/resilient_e97_runtime.py",
            ),
            universe,
        ),
        protected_surface(
            "policy_and_schema",
            exact(
                "configs/frontier/"
                "native_resilient_pool_v1_production_policy.json",
                "docs/ASYNC_DECOUPLED_DILOCO_V2.md",
                "docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md",
                "ndm/async_diloco_v2.py",
                "ndm/native_pool_production_policy.py",
                "src/native_resilient_dataplane/include/emender/ndp.h",
            ),
            universe,
        ),
        protected_surface(
            "checkpoint_and_apply_path",
            exact(
                "ndm/manifest_peer_control.py",
                "ndm/native_pool_runtime.py",
                "ndm/resilient_pool_runtime.py",
                "ndm/resilient_e97_runtime.py",
                "scripts/frontier/resilient_e97_role.py",
            ),
            universe,
        ),
        protected_surface(
            "seed_data_tokenizer_bindings",
            exact(
                "configs/frontier/e97_resilient_split_role_flat.json",
                "scripts/frontier/materialize_e97_s3_seed.py",
                "scripts/frontier/"
                "render_resilient_e97_exact_2n_acceptance.py",
                "scripts/frontier/"
                "native_dataplane_2n_controller.py",
                "scripts/frontier/run_async_v21_qualification.py",
            ),
            universe,
        ),
        protected_surface(
            "rendered_model_execution_payload_surface",
            exact(
                "configs/frontier/e97_resilient_split_role_flat.json",
                "ndm/async_diloco_v2.py",
                "ndm/manifest_peer_control.py",
                "ndm/native_e97_runtime.py",
                "ndm/native_pool_runtime.py",
                "ndm/resilient_e97_runtime.py",
                "scripts/frontier/"
                "native_dataplane_2n_controller.py",
                "scripts/frontier/"
                "render_resilient_e97_exact_2n_acceptance.py",
                "scripts/frontier/resilient_e97_role.py",
                "scripts/frontier/resilient_e97_true_2n.sbatch",
                "scripts/frontier/run_async_v21_qualification.py",
            ),
            universe,
        ),
        protected_surface(
            "artifact_namespace_ownership",
            exact(
                "scripts/frontier/native_dataplane_2n_gate.sbatch",
                "scripts/frontier/native_g2_artifact_namespace.py",
                "scripts/frontier/submit_native_dataplane_2n_gate.sh",
            ),
            universe,
        ),
    ]
    protected_by_name = {item["name"]: item for item in protected}

    clean_payload = (
        "6f02204cdddc6512d05d8cbeaaeb8d9a08952ffb9837fe825"
        "bea347d0269e0d7"
    )
    clean_verdict = (
        BASELINE_ROOT
        / f"clean/terminal-collector/{clean_payload}/terminal-verdict.json"
    )
    ownership_manifest = PROOF_ROOT / "ARTIFACT-OWNERSHIP.json"
    proof_files = sorted(path for path in PROOF_ROOT.rglob("*") if path.is_file())

    scope_failures = [
        item["name"]
        for item in protected
        if item["name"] != "artifact_namespace_ownership"
        and not item["unchanged"]
    ]
    exact_out_of_allowlist = [
        entry["path"]
        for entry in paths
        if not entry["in_exact_ownership_fix_allowlist"]
    ]
    allowlisted_with_prior_drift = [
        entry["path"]
        for entry in paths
        if entry["in_exact_ownership_fix_allowlist"]
        and entry["changed_before_ownership_fix_commit"]
    ]
    reuse_disallowed = [
        entry["path"]
        for entry in paths
        if not entry["pure_ownership_fix_delta_from_clean_source"]
    ]
    certificate: dict[str, Any] = {
        "schema_version": 1,
        "certificate_type": "async-v2.1-change-scope-and-stop",
        "task_id": TASK_ID,
        "generated_at_utc": "2026-07-29T06:13:17Z",
        "source_freeze": {
            "baseline_clean_source_commit": BASELINE,
            "candidate_origin_main_commit": CANDIDATE,
            "candidate_origin_main_tree": run_git(
                "show", "-s", "--format=%T", CANDIDATE
            )
            .decode()
            .strip(),
            "candidate_parent_commit": CANDIDATE_PARENT,
            "candidate_commit_subject": run_git(
                "show", "-s", "--format=%s", CANDIDATE
            )
            .decode()
            .strip(),
            "freeze_rule": (
                "The exact candidate is the fetched origin/main release "
                "candidate. No later commit is absorbed into qualification."
            ),
            "remote_verification_at_freeze": {
                "observed_at_utc": "2026-07-29T05:59:00Z",
                "worktree_head": CANDIDATE,
                "fetched_origin_main": CANDIDATE,
                "git_ls_remote_refs_heads_main": CANDIDATE,
                "all_equal": True,
            },
        },
        "allowlist": {
            "rule": (
                "Reuse is allowed only when every baseline-to-candidate "
                "change is introduced by the one ownership-fix commit and is "
                "one of these exact eight paths, while every protected "
                "runtime surface remains byte-identical."
            ),
            "candidate_parent_to_candidate_paths": sorted(
                OWNERSHIP_FIX_PATHS
            ),
        },
        "change_inventory": {
            "comparison": f"{BASELINE}..{CANDIDATE}",
            "changed_path_count": len(paths),
            "pure_ownership_fix_path_count": sum(
                bool(entry["pure_ownership_fix_delta_from_clean_source"])
                for entry in paths
            ),
            "exact_out_of_allowlist_path_count": len(exact_out_of_allowlist),
            "exact_out_of_allowlist_paths": exact_out_of_allowlist,
            "allowlisted_path_with_prior_drift_count": len(
                allowlisted_with_prior_drift
            ),
            "allowlisted_paths_with_prior_drift": (
                allowlisted_with_prior_drift
            ),
            "reuse_disallowed_path_count": len(reuse_disallowed),
            "reuse_disallowed_paths": reuse_disallowed,
            "paths": paths,
        },
        "protected_surfaces": protected_by_name,
        "identity_checks": {
            "execution_source_digest": {
                "baseline": (
                    "95cc5018d948e844091bef0d96c5b07605385b0bcedff653"
                    "760b70b200964b8b"
                ),
                "candidate": (
                    "1ab716bf9853ed57cb1596eb09ee9f69bb6c0ccc45a65f739"
                    "0f1743a095b8931"
                ),
                "unchanged": False,
                "method": (
                    "scripts.frontier."
                    "render_resilient_e97_exact_2n_acceptance._source_digest"
                ),
            },
            "policy_digest": {
                "baseline": (
                    "fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710"
                    "fbf126a344e7d98"
                ),
                "candidate": (
                    "fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710"
                    "fbf126a344e7d98"
                ),
                "unchanged": True,
            },
            "train_args_sha256": {
                "baseline": (
                    "afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d1"
                    "8cc58bb93eb9fe9c"
                ),
                "candidate": (
                    "afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d1"
                    "8cc58bb93eb9fe9c"
                ),
                "unchanged": True,
            },
            "seed_config_sha256": {
                "baseline": (
                    "3f704e32bdfffd308eda13758b8a95b3989e4ecc7545a32a"
                    "673b5589c8085d24"
                ),
                "candidate": (
                    "3f704e32bdfffd308eda13758b8a95b3989e4ecc7545a32a"
                    "673b5589c8085d24"
                ),
                "unchanged": True,
            },
            "data_identity": {
                "baseline": (
                    "91321b2b90bb159f3aa73881455778f10e8df588edd526b10"
                    "66281fa72997962"
                ),
                "candidate": (
                    "91321b2b90bb159f3aa73881455778f10e8df588edd526b10"
                    "66281fa72997962"
                ),
                "bytes": 1000000725401,
                "unchanged": True,
            },
            "tokenizer_sha256": {
                "baseline": (
                    "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f9685"
                    "47f9f23eb70d2069"
                ),
                "candidate": (
                    "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f9685"
                    "47f9f23eb70d2069"
                ),
                "bytes": 836186,
                "unchanged": True,
            },
            "resilient_e97_role_sha256": {
                "baseline": (
                    "ebb9fe1f70fa78e77a9dc962c3732896538d033811b4ead1"
                    "980e03b3c3d82b7a"
                ),
                "candidate": (
                    "b81d9c95f84bbe3f097fac428651ce6bb2845cfe824ca8913"
                    "c50865f04ddc546"
                ),
                "unchanged": False,
            },
            "native_public_abi_header_sha256": {
                "baseline": (
                    "e84e0a6c61ce44a1a76d1be675cf9b19798cf87aa9fdd981"
                    "53bed14cde10fe24"
                ),
                "candidate": (
                    "e02caec4a725a60d737cfebf464da9a86f8e245b81f9515f"
                    "b11d56e8d6088c62"
                ),
                "unchanged": False,
            },
            "baseline_native_bundle": {
                "bundle_sha256": (
                    "f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d389"
                    "8fe562ce4182441"
                ),
                "native_manifest_sha256": (
                    "8cd0f836e9f798a48e72aef7c60591d6d939abe2b1b3cdb8"
                    "b06d4c30502fe00c"
                ),
                "candidate_validation": "rejected",
                "candidate_validation_error": (
                    "ValueError: native build does not match the launched "
                    "source commit"
                ),
            },
        },
        "exact_seed": {
            "model": "E97",
            "step": 2300930,
            "tokens": 150793748480,
            "bytes": 7719680116,
            "sha256": (
                "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40"
                "c1c74f7dd6a72b2"
            ),
            "seed_bootstrap_attestation": artifact(
                BASELINE_ROOT
                / "clean/clean-overlap/seed-bootstrap-attestation.json",
                "27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1"
                "af167776d1d710bb",
            ),
        },
        "retained_clean_evidence": {
            "reuse_allowed": False,
            "reason": (
                "The source-scope certificate fails, so model5109029 and "
                "collector5109030 remain immutable historical evidence but "
                "cannot qualify the frozen candidate."
            ),
            "model_job": 5109029,
            "collector_job": 5109030,
            "payload_digest": clean_payload,
            "clean_manifest": artifact(
                BASELINE_ROOT / "clean-manifest.json",
                "cda81bb450aff6fe34feaf1fbddbb0037da7278c4c9d0741f7"
                "15525211ca4747",
            ),
            "terminal_verdict": artifact(
                clean_verdict,
                "5cbd1091083e4fb8ddcc1b507e0b3e9f40e092dbd3408f48f"
                "1247acb033bb555",
            ),
            "performance_verdict": artifact(
                BASELINE_ROOT
                / "clean/clean-overlap/pipelined-performance.json",
                "e70ff5808c345fdcc35f2ffe72ce13c07a716f1a8a87513b5"
                "89bacaf0a7b7b64",
            ),
            "runtime_identity": artifact(
                BASELINE_ROOT
                / "clean/clean-overlap/runtime-identity.json",
                "5c0fa27593fa1b173a57512eaee13e8dcb8bfadfb1c1b3783"
                "fae4457e02c6805",
            ),
            "scheduler_terminal": artifact(
                BASELINE_ROOT
                / "clean/scheduler-evidence/"
                "sacct-5109029-5109030-terminal.txt",
                "c9ef0e9209b418409b558f2b14acefc5d5377657b623a57d1"
                "65c11755d64fc54",
            ),
        },
        "job5109414_ownership_regression": {
            "scheduler_simulated": False,
            "local_scheduler_invoked": False,
            "legacy_real_job": {
                "job_id": 5109414,
                "nodes": 2,
                "partition": "batch",
                "qos": "debug",
                "state": "FAILED",
                "exit_code": "73:0",
                "scheduler_evidence": [
                    artifact(
                        BASELINE_ROOT
                        / "g2/5109414/scheduler-evidence/"
                        "sacct-5109414-terminal.txt",
                        "e8600f5e9358451d78947b6a4567d98b414d17e2e02415fb7"
                        "a79c6d12b202079",
                    ),
                    artifact(
                        BASELINE_ROOT
                        / "g2/5109414/scheduler-evidence/"
                        "scontrol-5109414-terminal.txt",
                        "b74a013ade1594c5bd4de2b062be2d2f7703aeb5ebce8fabb"
                        "8e27b64c1e4e6ca",
                    ),
                ],
            },
            "proof_root": str(PROOF_ROOT),
            "proof_files": [artifact(path) for path in proof_files],
            "ownership_manifest": artifact(
                ownership_manifest,
                "97ab892f03a0a069aaca737c350f988727c5fd61d28d2a619"
                "302d1bf3cb296b1",
            ),
            "controller_root": str(
                PROOF_ROOT / "controller/5109414/scheduler-evidence"
            ),
            "batch_root": str(PROOF_ROOT / "5109414"),
            "collector_root": str(
                PROOF_ROOT / "collectors/5109415/payload-5109414"
            ),
            "namespaces_disjoint": True,
            "rendered_batch_guard_created_root": True,
            "focused_direct_tests": {
                "result": "8 passed in 0.44s",
                "fake_scheduler_test_selected": False,
                "selected": [
                    "test_job5109414_legacy_order_reproduces_exit_73",
                    (
                        "test_observation_and_collector_reconciliation_are_"
                        "idempotent_and_disjoint"
                    ),
                    (
                        "test_conflicting_authoritative_batch_artifacts_fail_"
                        "closed_without_overwrite"
                    ),
                    "test_evidence_writers_reject_cross_owner_symlink",
                    (
                        "test_batch_publication_is_single_winner_under_"
                        "duplicate_and_mkdir_races"
                    ),
                    (
                        "test_batch_publication_cannot_be_replaced_by_"
                        "concurrent_directory_rename"
                    ),
                    (
                        "test_cli_returns_73_for_an_existing_authoritative_"
                        "root"
                    ),
                ],
                "note": (
                    "The symlink test is parameterized over both non-batch "
                    "owner roots, producing eight collected cases."
                ),
            },
        },
        "scheduler_action": {
            "stopped_before_sbatch": True,
            "jobs_submitted_by_this_task": [],
            "active_jobs_observed_after_stop": [],
            "reason": (
                "Fail-closed source-scope gate rejected the shortcut before "
                "any scheduler submission."
            ),
        },
        "qualification_phases": {
            "scope_certificate": "failed",
            "reuse_model5109029_collector5109030": "blocked",
            "fresh_short_native_clean_g2": "not_run",
            "fresh_native_fault_g2": "not_run",
            "two_commit_fault_baseline": "not_run",
            "serialized_fault_rejoin": "not_run",
            "fresh_allocation_recovery": "not_run",
        },
        "requirements": {
            "checklist_authorities": [
                "docs/RESILIENT_DILOCO_COMPUTE_POOL.md",
                "docs/RESILIENT_DILOCO_GAP_MATRIX.md",
                "docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md",
                "docs/ASYNC_DECOUPLED_DILOCO_V2.md",
                "docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md",
            ],
            "R01_R16": requirement_mapping(
                R_IDS, "docs/RESILIENT_DILOCO_GAP_MATRIX.md"
            ),
            "NDP01_NDP17": requirement_mapping(
                NDP_IDS, "docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md"
            ),
            "V21S01_V21S17": requirement_mapping(
                V21S_IDS, "docs/ASYNC_DECOUPLED_DILOCO_V2.md"
            ),
            "ISP01_ISP07": requirement_mapping(
                ISP_IDS, "docs/RESILIENT_DILOCO_GAP_MATRIX.md"
            ),
        },
        "decision": {
            "scope_pass": False,
            "clean_evidence_reuse_allowed": False,
            "full_pass": False,
            "stop_phase": "change_scope_certificate",
            "protected_surface_failures": scope_failures,
            "exact_out_of_allowlist_path_count": len(exact_out_of_allowlist),
            "allowlisted_path_with_prior_drift_count": len(
                allowlisted_with_prior_drift
            ),
            "reuse_disallowed_path_count": len(reuse_disallowed),
            "required_action": (
                "Run an explicitly authorized full exact-source "
                "requalification; do not broaden this shortcut task."
            ),
        },
    }
    unsigned = dict(certificate)
    certificate["manifest_digest"] = {
        "algorithm": "sha256",
        "canonicalization": (
            "UTF-8 JSON, sorted keys, separators comma/colon, "
            "manifest_digest omitted"
        ),
        "sha256": sha256_bytes(canonical_bytes(unsigned)),
    }
    return certificate


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[2])
    certificate = build()
    OUTPUT.write_text(
        json.dumps(
            certificate,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "manifest_digest": certificate["manifest_digest"]["sha256"],
                "changed_path_count": certificate["change_inventory"][
                    "changed_path_count"
                ],
                "scope_pass": certificate["decision"]["scope_pass"],
                "full_pass": certificate["decision"]["full_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
