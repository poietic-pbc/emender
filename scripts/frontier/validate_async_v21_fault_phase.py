#!/usr/bin/env python3
"""Fail-closed semantic validator for one production async-v2.1 fault phase."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from ndm.async_diloco_v2 import (
    ASYNC_DECOUPLED_V21,
    AsyncV21CommitAuthority,
    AsyncV21WorkerLane,
    Backpressure,
    ContributionEnvelope,
    LatestResultMailbox,
    OuterState,
    ResultEnvelope,
    ScheduleFreeLocalState,
    StaleContribution,
    build_contribution,
    digest_array,
    reference_aggregate,
)


SEED_TOKENS = 150_793_748_480
TOKENS_PER_COMMIT = 5_245_440
ZERO = "0" * 64
ONE = "1" * 64
TWO = "2" * 64
THREE = "3" * 64
PHASES = {
    "fault-baseline": (0, 2),
    "fault-rejoin": (2, 4),
    "fresh-recovery": (6, 5),
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    os.replace(temporary, path)


def _event(events: list[dict[str, object]], scenario: str, **evidence: object) -> None:
    events.append({
        "scenario": scenario,
        "observed_unix_ns": time.time_ns(),
        "passed": True,
        **evidence,
    })


def _contribution(
    *,
    worker: str,
    base: int = 0,
    current: int = 0,
    sequence: int = 0,
    fence: int = 7,
    tokens: int = 11,
    start: float = 0.0,
    end: float = 1.0,
) -> ContributionEnvelope:
    return build_contribution(
        run_id="run",
        allocation_fence=fence,
        worker_id=worker,
        worker_incarnation=f"{worker}-incarnation",
        contribution_sequence=sequence,
        local_window_start=sequence,
        local_window_end=sequence + 1,
        base_global_version=base,
        base_global_digest=f"{base:064x}",
        current_global_version=current,
        policy=ASYNC_DECOUPLED_V21,
        layout_digest=ONE,
        code_digest=TWO,
        exact_tokens=tokens,
        interval_start=np.asarray([start], dtype=np.float64),
        interval_end=np.asarray([end], dtype=np.float64),
        window_monotonic_ns=((sequence * 10 + 1, sequence * 10 + 9),),
        endpoint_digest=digest_array(np.asarray([end], dtype=np.float64)),
        local_trainer_set_digest=THREE,
        source_dtype="float32",
        shard_roots=(ONE,),
    )


def _result(
    *,
    version: int,
    state: float,
    fence: int = 7,
) -> ResultEnvelope:
    values = np.asarray([state], dtype=np.float64)
    return ResultEnvelope.create(
        run_id="run",
        allocation_fence=fence,
        version=version,
        base_version=max(0, version - 1),
        base_digest=ZERO,
        state=values,
        outer=OuterState(step=version, accepted_tokens=version * 10),
        policy_digest=ASYNC_DECOUPLED_V21.digest,
        layout_digest=ONE,
        code_digest=TWO,
        manifest_digest=THREE,
        selected_contribution_digests=(),
        reload_verified=True,
        latest_cas_verified=True,
    )


def _semantic_probes() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for lag in (0, 1, 2):
        record = _contribution(
            worker=f"node-{lag}", base=2 - lag, current=2)
        mean, tokens, admitted = reference_aggregate(
            2, (record,), policy=ASYNC_DECOUPLED_V21)
        if (
            tokens != 11
            or admitted[0].commit_lag != lag
            or mean.tolist() != [1.0]
        ):
            raise ValueError(f"lag-{lag} admission changed numerical state")
    _event(events, "lag_0_1_2_admission", admitted_lags=[0, 1, 2])
    try:
        reference_aggregate(
            3,
            (_contribution(worker="node-3", base=0, current=2),),
            policy=ASYNC_DECOUPLED_V21,
        )
    except StaleContribution:
        pass
    else:
        raise ValueError("lag-3 contribution entered numerical mutation")
    lane = AsyncV21WorkerLane.for_test()
    lane.newest_verified_version = 2
    for index in range(3):
        lane.finish_window(
            np.asarray([float(index + 1)]),
            exact_tokens=5,
            begin_ns=index * 2 + 1,
            end_ns=index * 2 + 2,
        )
    if (
        lane.local_window != 3
        or lane.paused_reason is not None
        or lane.snapshot_deferred_reason != "snapshot_admission_limit"
    ):
        raise ValueError("lag-3 catch-up did not retain bounded local progress")
    try:
        lane.seal()
    except Backpressure:
        pass
    else:
        raise ValueError("third speculative snapshot was admitted")
    _event(
        events,
        "lag_3_drop_and_catch_up",
        local_windows=lane.local_window,
        speculative_window_limit=2,
        foreground_paused=False,
    )

    authority = AsyncV21CommitAuthority.for_test()
    authority.install_membership({"node-a": "node-a-incarnation"})
    value = _contribution(worker="node-a")
    committed = authority.commit((value,))
    if authority.commit((value,)) is not committed or authority.version != 1:
        raise ValueError("identical contribution replay was not idempotent")
    _event(
        events,
        "duplicate_idempotence",
        result_digest=committed.result_digest,
        version=authority.version,
    )
    mutation_before = authority.state.copy()
    outer_before = authority.outer
    conflict = replace(
        value,
        identity=replace(
            value.identity, exact_tokens=value.identity.exact_tokens + 1),
    )
    for scenario, operation in (
        ("conflicting_identity", lambda: authority.replay_receipt(conflict)),
        (
            "checksum_corruption",
            lambda: AsyncV21CommitAuthority.for_test().commit(
                (replace(value, delta=np.asarray([99.0])),)),
        ),
        (
            "nonfinite_contribution",
            lambda: AsyncV21CommitAuthority.for_test().commit(
                (replace(value, delta=np.asarray([float("inf")])),)),
        ),
        (
            "wrong_fence",
            lambda: AsyncV21CommitAuthority.for_test().commit(
                (replace(
                    value,
                    identity=replace(
                        value.identity, allocation_fence=8)),)),
        ),
    ):
        try:
            operation()
        except (ValueError, StaleContribution):
            _event(events, scenario, accumulator_mutated=False)
        else:
            raise ValueError(f"{scenario} was accepted")
    if not np.array_equal(authority.state, mutation_before):
        raise ValueError("rejected input mutated the committed accumulator")
    if authority.outer != outer_before:
        raise ValueError("rejected input mutated the outer state")

    unpublished = AsyncV21CommitAuthority.for_test()
    unpublished.install_membership({"node-a": "node-a-incarnation"})
    before = unpublished.state.copy()
    try:
        unpublished.commit(
            (value,),
            publish=lambda _bundle: (_ for _ in ()).throw(
                OSError("injected publication failure")),
        )
    except OSError:
        pass
    else:
        raise ValueError("failed publication became visible")
    if (
        unpublished.version != 0
        or unpublished.outer.step != 0
        or not np.array_equal(unpublished.state, before)
    ):
        raise ValueError("failed publication mutated authority")
    _event(
        events,
        "failed_publication_invisibility",
        visible_version=unpublished.version,
        accumulator_mutated=False,
    )

    mailbox = LatestResultMailbox(
        run_id="run",
        fence=7,
        policy_digest=ASYNC_DECOUPLED_V21.digest,
        layout_digest=ONE,
        code_digest=TWO,
    )
    first = _result(version=1, state=1.0)
    if mailbox.publish(first) != "published":
        raise ValueError("first mailbox result was not published")
    lease = mailbox.take()
    if lease is None or mailbox.publish(
            _result(version=2, state=2.0)) != "staged":
        raise ValueError("held mailbox did not stage one replacement")
    try:
        mailbox.publish(_result(version=3, state=3.0))
    except Backpressure:
        pass
    else:
        raise ValueError("mailbox admitted a second staged replacement")
    lease.release()
    newest = mailbox.take()
    if newest is None or newest.result.version != 2:
        raise ValueError("mailbox replacement did not become latest")
    newest.release()
    try:
        mailbox.publish(_result(version=4, state=float("nan")))
    except ValueError:
        pass
    else:
        raise ValueError("nonfinite mailbox result became visible")
    _event(
        events,
        "mailbox_replacement",
        high_water=mailbox.high_water,
        latest_version=2,
    )

    bounded = AsyncV21WorkerLane.for_test(
        local=ScheduleFreeLocalState(
            x=np.asarray([0.0]),
            parameter_points={"z": np.asarray([0.0])},
        ))
    bounded.finish_window(
        np.asarray([1.0]), exact_tokens=1, begin_ns=1, end_ns=2)
    owned = bounded.seal()
    bounded.finish_window(
        np.asarray([2.0]), exact_tokens=1, begin_ns=3, end_ns=4)
    bounded.finish_window(
        np.asarray([3.0]), exact_tokens=1, begin_ns=5, end_ns=6)
    if (
        bounded.high_water["owned_descriptors"] != 1
        or bounded.high_water["mutable_windows"] > 2
        or bounded.paused_reason is not None
    ):
        raise ValueError("OWNED/mutable bounded-progress contract changed")
    bounded.release_sealed(owned.digest, outcome="owner_abort")
    _event(
        events,
        "local_owned_timeout",
        released=True,
        high_water=bounded.high_water,
        foreground_paused=False,
    )
    return events


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _live_phase(
    *,
    run_dir: Path,
    phase: str,
    campaign_digest: str,
) -> dict[str, object]:
    initial, generations = PHASES[phase]
    final_generation = initial + generations
    latest_path = run_dir / "handoff" / "latest.json"
    latest = _load_json(latest_path)
    manifest_path = Path(str(latest.get("manifest", ""))).resolve()
    manifest_path.relative_to((run_dir / "handoff").resolve())
    manifest = _load_json(manifest_path)
    checkpoint = Path(str(manifest.get("checkpoint", ""))).resolve()
    expected_run = "async-v21-faults-" + campaign_digest[:16]
    expected_tokens = SEED_TOKENS + final_generation * TOKENS_PER_COMMIT
    if (
        latest.get("generation") != final_generation
        or latest.get("manifest_sha256") != _sha256(manifest_path)
        or manifest.get("finalized") is not True
        or manifest.get("generation") != final_generation
        or manifest.get("run_id") != expected_run
        or manifest.get("payload_id") != campaign_digest
        or manifest.get("membership") != ["node-0", "node-1"]
        or manifest.get("accepted_tokens") != expected_tokens
        or manifest.get("outer_update_state") != {
            "accepted_tokens": expected_tokens,
            "eta_outer": 1.0,
            "mode": "delta_sgd",
            "step": final_generation,
        }
        or not checkpoint.is_file()
        or manifest.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise ValueError("terminal model/outer/token checkpoint is invalid")
    fence = int(latest.get("fence", 0))
    commits = []
    applies = []
    authority = run_dir / "handoff" / "authority"
    for generation in range(initial + 1, final_generation + 1):
        commit_matches = sorted(
            (authority / "commits").glob(
                f"commit-generation-{generation:08d}-*.json"))
        if len(commit_matches) != 1:
            raise ValueError(
                f"generation {generation} does not have one immutable commit")
        commit = _load_json(commit_matches[0])
        if (
            commit.get("generation") != generation
            or commit.get("allocation_fence") != fence
            or commit.get("accepted_tokens")
            != SEED_TOKENS + generation * TOKENS_PER_COMMIT
        ):
            raise ValueError("commit fence/token lineage is invalid")
        commits.append({
            "generation": generation,
            "path": str(commit_matches[0]),
            "sha256": _sha256(commit_matches[0]),
            "receipt_digest": commit.get("receipt_digest"),
        })
        apply_matches = sorted(
            (authority / "applies").glob(
                f"apply-generation-{generation:08d}-*.json"))
        if len(apply_matches) != 2:
            raise ValueError(
                f"generation {generation} lacks two node apply receipts")
        generation_applies = []
        for path in apply_matches:
            receipt = _load_json(path)
            trainer_receipts = receipt.get("trainer_receipts")
            ranks = (
                sorted(item.get("rank") for item in trainer_receipts)
                if isinstance(trainer_receipts, list) else []
            )
            if (
                receipt.get("generation") != generation
                or receipt.get("allocation_fence") != fence
                or ranks != list(range(8))
                or len({
                    item.get("trainer_incarnation")
                    for item in trainer_receipts
                }) != 8
            ):
                raise ValueError("node apply was partial or stale-incarnation")
            generation_applies.append({
                "node_id": receipt.get("node_id"),
                "path": str(path),
                "sha256": _sha256(path),
                "trainer_receipts": 8,
            })
        if sorted(item["node_id"] for item in generation_applies) != [
                "node-0", "node-1"]:
            raise ValueError("one-node apply authority was observed")
        applies.append({
            "generation": generation,
            "nodes": generation_applies,
        })

    control_records = []
    pool_control = run_dir / "retained-evidence" / "pool-control"
    for path in sorted(pool_control.glob("generation-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "commit_ready":
                continue
            frozen = record.get("frozen_identities")
            if (
                record.get("required_contributions") != 2
                or not isinstance(frozen, list)
                or len(frozen) != 2
                or sorted(item.get("worker_id") for item in frozen)
                != ["node-0", "node-1"]
            ):
                raise ValueError("one-node commit authority was observed")
            control_records.append({
                "generation": record.get("generation"),
                "recorded_at": record.get("recorded_at"),
                "frozen_identities": frozen,
            })
    if len(control_records) < generations:
        raise ValueError("missing READY/open/freeze/commit timing evidence")

    live: dict[str, object] = {
        "initial_generation": initial,
        "final_generation": final_generation,
        "commits": commits,
        "node_applies": applies,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "accepted_tokens": expected_tokens,
            "outer_update_state": manifest["outer_update_state"],
        },
        "allocation_fence": fence,
        "ready_freeze_commit_records": control_records,
        "one_node_commit_authority": False,
    }

    events_path = run_dir / "supervision" / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forbidden = [
        item for item in events
        if item.get("event") in {"restart_exhausted", "rank_retired"}
    ]
    if forbidden:
        raise ValueError("fault phase exhausted recovery or retired a trainer")
    if phase == "fault-rejoin":
        injected = [
            item for item in events
            if (
                item.get("event") == "generation_gated_injection"
                and item.get("generation") in {2, 3, 4}
            )
        ]
        if {
            item.get("injection_class") for item in injected
        } != {"trainer", "manager", "native-service"}:
            raise ValueError("trainer/manager/native-service injection is incomplete")
        reconstructed = [
            item for item in events
            if item.get("event") == "atomic_cohort_reconstructed"
        ]
        if len(reconstructed) < 3 or any(
            len(item.get("trainer_incarnations", {})) != 8
            or item.get("failed_incarnation") == item.get("node_incarnation")
            for item in reconstructed[-3:]
        ):
            raise ValueError("fault recovery did not reconstruct all eight trainers")
        delayed = list(
            (run_dir / "retained-evidence" / "node-1" / "control").glob(
                "native-delayed-ready-00000002-*.json"))
        if len(delayed) != 1:
            raise ValueError("delayed READY injection evidence is missing")
        live["process_faults"] = {
            "injections": injected,
            "reconstructions": reconstructed[-3:],
            "delayed_ready": {
                "path": str(delayed[0]),
                "sha256": _sha256(delayed[0]),
                "value": _load_json(delayed[0]),
            },
            "all_rank_abort": False,
        }
    if phase == "fresh-recovery":
        previous_matches = sorted(
            (run_dir / "handoff").glob(
                "generation-00000006-fence-*.json"))
        if len(previous_matches) != 1:
            raise ValueError("fresh allocation source checkpoint is missing")
        previous = _load_json(previous_matches[0])
        if (
            int(previous.get("fence", {}).get("coordinator_epoch", 0)) >= fence
            or previous.get("accepted_tokens")
            != SEED_TOKENS + 6 * TOKENS_PER_COMMIT
        ):
            raise ValueError("fresh allocation did not acquire a newer fence")
        syncs = []
        for node in range(2):
            paths = sorted(
                (run_dir / "retained-evidence" / f"node-{node}" / "control").glob(
                    "native-manager-sync-*.json"))
            synchronized = [
                _load_json(path) for path in paths
                if (
                    _load_json(path).get("status") == "synchronized"
                    and _load_json(path).get("generation") == 6
                    and _load_json(path).get("fence") == fence
                )
            ]
            if len(synchronized) != 1:
                raise ValueError("fresh manager did not synchronize generation 6")
            syncs.append(synchronized[0])
        live["fresh_allocation"] = {
            "source_manifest": str(previous_matches[0]),
            "source_manifest_sha256": _sha256(previous_matches[0]),
            "source_fence": int(
                previous["fence"]["coordinator_epoch"]),
            "new_fence": fence,
            "manager_sync": syncs,
            "additional_k40_windows_per_trainer": generations,
            "additional_commits": len(commits),
        }
        if len(commits) < 3 or generations < 5:
            raise ValueError("fresh allocation recovery is too short")
    return live


def validate(
    *,
    run_dir: Path,
    phase: str,
    full_layout_gate: Path,
    prior_gate: Path,
) -> dict[str, object]:
    if phase not in PHASES:
        raise ValueError("unsupported fault phase")
    campaign_digest = os.environ.get("ASYNC_V21_FAULT_CAMPAIGN_DIGEST", "")
    if len(campaign_digest) != 64:
        raise ValueError("fault campaign digest is missing")
    g2 = _load_json(full_layout_gate)
    if (
        g2.get("gate") != "G2-fault-rejoin-replay"
        or g2.get("status") != "passed"
        or g2.get("nodes") != 2
        or g2.get("provider") != "cxi"
        or not isinstance(g2.get("fault"), dict)
        or g2["fault"].get("peer_loss") is not True
        or g2["fault"].get("new_incarnation") is not True
        or g2["fault"].get("old_epoch_rejected") is not True
        or g2["fault"].get("partial_commit") is not False
    ):
        raise ValueError("exact-code native fault G2 is not a pass")
    prior = _load_json(prior_gate)
    if (
        prior.get("schema") != "emender-async-v21-terminal-verdict-v1"
        or prior.get("passed") is not True
        or prior.get("verdict") != "passed"
    ):
        raise ValueError("prior clean terminal verdict is not a pass")
    semantic = _semantic_probes()
    live = _live_phase(
        run_dir=run_dir.resolve(),
        phase=phase,
        campaign_digest=campaign_digest,
    )
    value: dict[str, object] = {
        "schema": "emender-async-v21-fault-phase-verdict-v1",
        "status": "passed",
        "passed": True,
        "phase": phase,
        "campaign_digest": campaign_digest,
        "nodes": 2,
        "partition": "batch",
        "qos": "debug",
        "q_min": 2,
        "maximum_speculative_windows": 2,
        "semantic_probes": semantic,
        "live": live,
        "native_fault_gate": {
            "path": str(full_layout_gate.resolve()),
            "sha256": _sha256(full_layout_gate),
            "payload_id": g2.get("payload_id"),
            "fault": g2["fault"],
            "bounds": g2.get("bounds"),
            "transport": g2.get("transport"),
        },
        "prior_clean_gate": {
            "path": str(prior_gate.resolve()),
            "sha256": _sha256(prior_gate),
            "payload_digest": prior.get("payload_digest"),
        },
        "requirements": {
            "compute_pool": [f"R{index:02d}" for index in range(1, 17)],
            "native": [f"NDP{index:02d}" for index in range(1, 18)],
            "async_v21": [f"V21S{index:02d}" for index in range(1, 18)],
            "immutable_snapshot":
                [f"ISP{index:02d}" for index in range(1, 8)],
        },
    }
    value["manifest_digest"] = _digest(value)
    return value


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--full-layout-gate", type=Path, required=True)
    parser.add_argument("--prior-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    verdict = validate(
        run_dir=args.run_dir,
        phase=args.phase,
        full_layout_gate=args.full_layout_gate.resolve(),
        prior_gate=args.prior_gate.resolve(),
    )
    _atomic_json(args.output.resolve(), verdict)
    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
