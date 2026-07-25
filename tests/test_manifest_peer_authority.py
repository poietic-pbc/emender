"""Negative SQLite tripwires and immutable peer-authority recovery tests.

Job 5072235 reached generation eight before a diagnostic SQLite constructor
raced the allocation renewal transaction on Lustre.  These tests deliberately
turn ``sqlite3.connect`` into a fatal operation and exercise the production
admission, diagnostic, commit, stale-incarnation, publication-failure, and
fresh-allocation recovery paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from ndm.manifest_peer_control import (
    FenceRejected,
    ManifestPeerAuthority,
)


ROOT = Path(__file__).parents[1]
COMPUTE_RUNTIME_CLOSURE = (
    "scripts/frontier/resilient_e97_true_2n.sbatch",
    "scripts/frontier/resilient_e97_allocation_supervisor.py",
    "scripts/frontier/resilient_e97_role.py",
    "ndm/manifest_peer_control.py",
    "ndm/resilient_e97_runtime.py",
    "ndm/resilient_pool_runtime.py",
    "ndm/native_e97_runtime.py",
    "ndm/async_diloco_v2.py",
)
FORBIDDEN_SQLITE_MARKERS = (
    "import sqlite3",
    "from sqlite3",
    "sqlite3.connect",
    "SQLiteFencedControlStore",
    "RESILIENT_E97_FENCE_DB",
    "pool-v1.sqlite",
    ".sqlite3",
)


def _claim(authority: ManifestPeerAuthority, fence: int, *,
           allocation: str | None = None, incarnation: str | None = None):
    return authority.claim(
        run_id="run",
        allocation_id=allocation or f"job-{fence}",
        incarnation=incarnation or f"allocation-{fence}",
        fence=fence,
        protocol_id="async-decoupled-v2.1-simple",
        config_id="config-sha256",
    )


def _checkpoint_manifest(root: Path, *, generation: int, fence: int,
                         previous_result_root: str = "00" * 32,
                         result_root: str | None = None) -> Path:
    checkpoint = root / "checkpoints" / (
        f"generation-{generation:08d}-fence-{fence:08d}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"model+outer+clock:{generation}:{fence}".encode())
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result_root = result_root or hashlib.sha256(
        f"result:{generation}:{fence}".encode()).hexdigest()
    manifest = root / "handoff" / (
        f"generation-{generation:08d}-fence-{fence:08d}.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema": 2,
        "finalized": True,
        "run_id": "run",
        "generation": generation,
        "step": generation * 40,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "outer_update_state": {
            "mode": "delta_sgd",
            "eta_outer": 1.0,
            "step": generation,
            "accepted_tokens": generation * 3_934_080,
        },
        "accepted_tokens": generation * 3_934_080,
        "fence": {"coordinator_epoch": fence},
        "generation_identity": {
            "run_id": "run",
            "coordinator_epoch": fence,
            "generation": generation - 1,
            "attempt": 1,
        },
        "membership": [
            {"worker_id": "node-0", "incarnation": f"n0-{fence}"},
            {"worker_id": "node-1", "incarnation": f"n1-{fence}"},
        ],
        "digests": {
            "checkpoint_sha256": checkpoint_sha256,
            "result_root": result_root,
            "previous_result_root": previous_result_root,
        },
    }, sort_keys=True) + "\n")
    return manifest


def test_rendered_compute_runtime_closure_has_no_sqlite_surface():
    for relative in COMPUTE_RUNTIME_CLOSURE:
        text = (ROOT / relative).read_text()
        for forbidden in FORBIDDEN_SQLITE_MARKERS:
            assert forbidden not in text, f"{relative} contains {forbidden!r}"
    # Commit discovery is live peer state.  Managers hand the exact digest to
    # trainers over node-local control; no compute role polls the immutable
    # shared receipt directory in the steady generation path.
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    assert ".wait_for_generation(" not in role


def test_sqlite_connect_is_fatal_across_admission_diagnostic_commit_and_restart(
        tmp_path, monkeypatch):
    """Every exercised compute path must stay operational with SQLite poisoned."""

    def fatal_sqlite(*_args, **_kwargs):
        raise AssertionError("compute-node runtime attempted sqlite3.connect")

    monkeypatch.setattr(sqlite3, "connect", fatal_sqlite)
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "run")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ID", "5072235")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_INCARNATION", "job-5072235")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_FENCE", "5072235")
    monkeypatch.setenv("RESILIENT_E97_SOURCE_ID", "seed")
    monkeypatch.setenv("RESILIENT_E97_PAYLOAD_ID", "layout")
    # Admission exports these values for its child roles.  Register their
    # original state with monkeypatch so later subprocess tests cannot inherit
    # a claim whose temporary authority tree has already been removed.
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_CLAIM", "")
    monkeypatch.setenv("RESILIENT_E97_FENCE_EPOCH", "0")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ADMITTED_AT", "0")

    from scripts.frontier import resilient_e97_allocation_supervisor as supervisor_module

    guard = supervisor_module._allocation_admission(tmp_path)
    assert isinstance(guard, supervisor_module.AllocationFenceGuard)
    manifest = _checkpoint_manifest(
        tmp_path, generation=1, fence=guard.claim.fence)
    receipt = guard.authority.publish_checkpoint(guard.claim, manifest)
    supervisor = supervisor_module.AllocationSupervisor(
        tmp_path, [], heartbeat_s=60, progress_s=60, max_restarts=0)
    assert supervisor._durable_generation() == 1
    assert guard.authority.current_commit(
        guard.claim, verify_checkpoint=True) == receipt


def test_stale_allocation_and_stale_incarnation_cannot_publish(tmp_path):
    authority = ManifestPeerAuthority(tmp_path)
    old = _claim(authority, 5072235)
    old_manifest = _checkpoint_manifest(tmp_path, generation=1, fence=old.fence)
    first = authority.publish_checkpoint(old, old_manifest)

    fresh = _claim(authority, 5072236)
    assert fresh is not None
    assert fresh.base_commit_digest == first.receipt_digest
    assert fresh.base_generation == 1

    stale_manifest = _checkpoint_manifest(
        tmp_path, generation=2, fence=old.fence)
    with pytest.raises(FenceRejected, match="newer allocation fence"):
        authority.publish_checkpoint(old, stale_manifest)

    # Reusing the scheduler fence with a different boot incarnation is a
    # split-brain conflict, never an idempotent admission.
    with pytest.raises(FenceRejected, match="conflicting allocation incarnation"):
        _claim(
            authority, 5072236,
            allocation="job-5072236",
            incarnation="different-boot")


def test_failed_publication_leaves_prior_commit_authoritative(
        tmp_path, monkeypatch):
    authority = ManifestPeerAuthority(tmp_path)
    claim = _claim(authority, 10)
    first = authority.publish_checkpoint(
        claim, _checkpoint_manifest(tmp_path, generation=1, fence=10))
    second_manifest = _checkpoint_manifest(
        tmp_path,
        generation=2,
        fence=10,
        previous_result_root=first.result_root,
    )

    original = authority._write_immutable

    def fail_commit(path, value):
        if path.parent.name == "commits":
            raise OSError("injected immutable publication failure")
        return original(path, value)

    monkeypatch.setattr(authority, "_write_immutable", fail_commit)
    with pytest.raises(OSError, match="publication failure"):
        authority.publish_checkpoint(claim, second_manifest)
    assert authority.current_commit(claim).receipt_digest == first.receipt_digest


def test_exact_once_commit_and_fresh_allocation_manifest_recovery(tmp_path):
    authority = ManifestPeerAuthority(tmp_path)
    first_claim = _claim(authority, 100)
    first_manifest = _checkpoint_manifest(
        tmp_path, generation=1, fence=first_claim.fence)
    first = authority.publish_checkpoint(first_claim, first_manifest)
    assert authority.publish_checkpoint(
        first_claim, first_manifest).receipt_digest == first.receipt_digest

    fresh_claim = _claim(authority, 101)
    recovered = authority.current_commit(fresh_claim, verify_checkpoint=True)
    assert recovered.receipt_digest == first.receipt_digest
    assert recovered.generation == 1
    assert recovered.accepted_tokens == 3_934_080
    assert recovered.outer_step == 1
    assert recovered.manifest_path == first_manifest.resolve()
    assert recovered.result_root == first.result_root

    second = authority.publish_checkpoint(
        fresh_claim,
        _checkpoint_manifest(
            tmp_path,
            generation=2,
            fence=fresh_claim.fence,
            previous_result_root=first.result_root,
        ),
    )
    assert second.previous_receipt_digest == first.receipt_digest
    assert second.generation == 2
    assert second.accepted_tokens == 7_868_160
    assert authority.current_commit(
        fresh_claim, verify_checkpoint=True).receipt_digest == second.receipt_digest
    apply = authority.record_node_apply(
        fresh_claim,
        second,
        node_id="node-0",
        node_incarnation="node-0-fresh",
        trainer_receipts=[
            (rank, f"trainer-{rank}-fresh", hashlib.sha256(
                f"recovery:{rank}".encode()).hexdigest())
            for rank in range(8)
        ],
    )
    assert apply.generation == 2
    assert authority.node_apply_receipts(second) == (apply,)
    with pytest.raises(FenceRejected, match="newer allocation fence"):
        authority.record_node_apply(
            first_claim,
            first,
            node_id="node-0",
            node_incarnation="stale",
            trainer_receipts=apply.trainer_receipts,
        )


def test_conflicting_exact_once_generation_fails_closed(tmp_path):
    authority = ManifestPeerAuthority(tmp_path)
    claim = _claim(authority, 77)
    accepted = _checkpoint_manifest(tmp_path, generation=1, fence=77)
    authority.publish_checkpoint(claim, accepted)

    value = json.loads(accepted.read_text())
    value["digests"]["result_root"] = "ab" * 32
    conflicting = accepted.with_name("conflicting-generation-1.json")
    conflicting.write_text(json.dumps(value, sort_keys=True) + "\n")
    with pytest.raises(FenceRejected, match="already committed"):
        authority.publish_checkpoint(claim, conflicting)
