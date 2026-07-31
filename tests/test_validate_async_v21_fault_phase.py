from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.frontier import validate_async_v21_fault_phase as MODULE


CAMPAIGN = "a" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_fixture(tmp_path: Path, phase: str) -> Path:
    initial, generations = MODULE.PHASES[phase]
    final = initial + generations
    run = tmp_path / phase
    handoff = run / "handoff"
    authority = handoff / "authority"
    checkpoint = run / "checkpoints" / f"generation-{final:08d}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{final}".encode())
    fence = 200 if phase == "fresh-recovery" else 100
    tokens = MODULE.SEED_TOKENS + final * MODULE.TOKENS_PER_COMMIT
    manifest = _write(
        handoff / f"generation-{final:08d}-fence-{fence:08d}.json",
        {
            "finalized": True,
            "generation": final,
            "run_id": "async-v21-faults-" + CAMPAIGN[:16],
            "payload_id": CAMPAIGN,
            "membership": ["node-0", "node-1"],
            "accepted_tokens": tokens,
            "outer_update_state": {
                "accepted_tokens": tokens,
                "eta_outer": 1.0,
                "mode": "delta_sgd",
                "step": final,
            },
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "fence": {"coordinator_epoch": fence},
        },
    )
    _write(handoff / "latest.json", {
        "generation": final,
        "fence": fence,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
    })
    for generation in range(initial + 1, final + 1):
        generation_tokens = (
            MODULE.SEED_TOKENS + generation * MODULE.TOKENS_PER_COMMIT)
        _write(
            authority / "commits"
            / f"commit-generation-{generation:08d}-fence-{fence:020d}-x.json",
            {
                "generation": generation,
                "allocation_fence": fence,
                "accepted_tokens": generation_tokens,
                "receipt_digest": f"{generation:064x}",
            },
        )
        for node in range(2):
            _write(
                authority / "applies"
                / (
                    f"apply-generation-{generation:08d}-"
                    f"fence-{fence:020d}-node-{node}.json"
                ),
                {
                    "generation": generation,
                    "allocation_fence": fence,
                    "node_id": f"node-{node}",
                    "trainer_receipts": [
                        {
                            "rank": rank,
                            "trainer_incarnation":
                                f"node-{node}-trainer-{rank}-g{generation}",
                        }
                        for rank in range(8)
                    ],
                },
            )
        record = {
            "status": "commit_ready",
            "generation": generation - 1,
            "required_contributions": 2,
            "recorded_at": 10.0 + generation,
            "frozen_identities": [
                {"worker_id": "node-0", "incarnation": f"n0-{generation}"},
                {"worker_id": "node-1", "incarnation": f"n1-{generation}"},
            ],
        }
        control = (
            run / "retained-evidence" / "pool-control"
            / f"generation-{generation - 1:08d}.jsonl")
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_text(json.dumps(record) + "\n")

    event_path = run / "supervision" / "events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    events = []
    if phase == "fault-rejoin":
        for generation, injection_class in (
            (2, "trainer"), (4, "manager"), (4, "native-service"),
        ):
            events.append({
                "event": "generation_gated_injection",
                "generation": generation,
                "injection_class": injection_class,
                "time": 1000.0 + generation,
            })
            events.append({
                "event": "atomic_cohort_reconstructed",
                "failed_incarnation": f"old-{generation}",
                "node_incarnation": f"new-{generation}",
                "trainer_incarnations": {
                    str(rank): f"trainer-{rank}-{generation}"
                    for rank in range(8)
                },
                "time": 1001.0 + generation,
            })
        _write(
            run / "retained-evidence" / "node-1" / "control"
            / "native-delayed-ready-00000002-fixture.json",
            {
                "generation": 2,
                "delay_seconds": 45,
                "delay_started_unix_ns": 1,
                "delay_finished_unix_ns": 45_000_000_001,
            },
        )
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events))

    if phase == "fresh-recovery":
        source_tokens = MODULE.SEED_TOKENS + 6 * MODULE.TOKENS_PER_COMMIT
        _write(
            handoff / "generation-00000006-fence-00000100.json",
            {
                "generation": 6,
                "accepted_tokens": source_tokens,
                "fence": {"coordinator_epoch": 100},
            },
        )
        for node in range(2):
            _write(
                run / "retained-evidence" / f"node-{node}" / "control"
                / f"native-manager-sync-node-{node}.json",
                {
                    "status": "synchronized",
                    "generation": 6,
                    "fence": fence,
                    "source_fence": 100,
                },
            )
    return run


def test_fault_semantic_probes_cover_rejection_and_bounded_progress():
    evidence = MODULE._semantic_probes()
    scenarios = {item["scenario"] for item in evidence}
    assert {
        "lag_0_1_2_admission",
        "lag_3_drop_and_catch_up",
        "duplicate_idempotence",
        "conflicting_identity",
        "checksum_corruption",
        "nonfinite_contribution",
        "wrong_fence",
        "failed_publication_invisibility",
        "mailbox_replacement",
        "local_owned_timeout",
    } <= scenarios
    assert all(item["passed"] and item["observed_unix_ns"] > 0
               for item in evidence)


@pytest.mark.parametrize(
    "phase", ("fault-baseline", "fault-rejoin", "fresh-recovery"))
def test_live_phase_requires_exact_commits_and_all_eight_apply_receipts(
    tmp_path: Path, phase: str,
):
    live = MODULE._live_phase(
        run_dir=_live_fixture(tmp_path, phase),
        phase=phase,
        campaign_digest=CAMPAIGN,
    )
    initial, generations = MODULE.PHASES[phase]
    assert live["initial_generation"] == initial
    assert live["final_generation"] == initial + generations
    assert len(live["commits"]) == generations
    assert len(live["node_applies"]) == generations
    assert all(
        all(node["trainer_receipts"] == 8 for node in generation["nodes"])
        for generation in live["node_applies"]
    )
    assert live["one_node_commit_authority"] is False
    if phase == "fault-rejoin":
        assert len(live["process_faults"]["injections"]) == 3
        assert len(live["process_faults"]["reconstructions"]) == 3
        assert live["process_faults"]["all_rank_abort"] is False
    if phase == "fresh-recovery":
        assert live["fresh_allocation"]["new_fence"] > (
            live["fresh_allocation"]["source_fence"])
        assert live["fresh_allocation"]["additional_k40_windows_per_trainer"] == 5
        assert live["fresh_allocation"]["additional_commits"] == 5


def test_live_phase_rejects_a_partial_node_apply(tmp_path: Path):
    run = _live_fixture(tmp_path, "fault-baseline")
    apply = next(
        (run / "handoff" / "authority" / "applies").glob(
            "apply-generation-00000002-*.json"))
    value = json.loads(apply.read_text())
    value["trainer_receipts"].pop()
    apply.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="partial"):
        MODULE._live_phase(
            run_dir=run,
            phase="fault-baseline",
            campaign_digest=CAMPAIGN,
        )
