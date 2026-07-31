from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from ndm.async_diloco_v2 import (
    ASYNC_DECOUPLED_V21,
    AsyncV21CommitAuthority,
    AsyncV21WorkerLane,
    AtomicEightTrainerApply,
    Backpressure,
    OuterState,
    SafeBoundaryRendezvous,
    ScheduleFreeLocalState,
    StaleContribution,
    build_contribution,
    reference_aggregate,
)


def _contribution(
    *,
    worker: str,
    incarnation: str = "boot-a",
    sequence: int = 0,
    base: int = 0,
    current: int = 0,
    tokens: int = 11,
    start: float = 0.0,
    end: float = 1.0,
):
    return build_contribution(
        run_id="run",
        allocation_fence=7,
        worker_id=worker,
        worker_incarnation=incarnation,
        contribution_sequence=sequence,
        local_window_start=sequence,
        local_window_end=sequence + 1,
        base_global_version=base,
        base_global_digest=f"{base:064x}",
        current_global_version=current,
        policy=ASYNC_DECOUPLED_V21,
        layout_digest="1" * 64,
        code_digest="2" * 64,
        exact_tokens=tokens,
        interval_start=np.asarray([start]),
        interval_end=np.asarray([end]),
        window_monotonic_ns=((sequence * 10 + 1, sequence * 10 + 9),),
        endpoint_digest="3" * 64,
        local_trainer_set_digest="4" * 64,
        source_dtype="float32",
        shard_roots=("5" * 64,),
    )


def test_v21_lag_three_defers_snapshot_without_blocking_next_k_window():
    for lag in (0, 1, 2):
        record = _contribution(worker=f"node-{lag}", base=2 - lag, current=2)
        mean, tokens, admitted = reference_aggregate(
            2, (record,), policy=ASYNC_DECOUPLED_V21)
        np.testing.assert_array_equal(mean, np.asarray([1.0]))
        assert tokens == 11
        assert admitted[0].commit_lag == lag
        assert admitted[0].exact_tokens == 11
        assert not hasattr(admitted[0], "aggregation_weight")

    with pytest.raises(StaleContribution, match="lag 3"):
        reference_aggregate(
            3, (_contribution(worker="node-3", base=0, current=2),),
            policy=ASYNC_DECOUPLED_V21,
        )

    lane = AsyncV21WorkerLane.for_test()
    lane.newest_verified_version = 2
    lane.finish_window(np.asarray([1.0]), exact_tokens=5, begin_ns=1, end_ns=2)
    lane.finish_window(np.asarray([2.0]), exact_tokens=5, begin_ns=3, end_ns=4)
    lane.finish_window(
        np.asarray([3.0]), exact_tokens=5, begin_ns=5, end_ns=6)
    assert lane.local_window == 3
    assert lane.paused_reason is None
    assert lane.snapshot_deferred_reason == "snapshot_admission_limit"
    assert lane.speculative_window_lag == 0
    with pytest.raises(Backpressure, match="snapshot admission is deferred"):
        lane.seal()


def test_v21_exact_tokens_are_only_weight_and_eta_one():
    fresh = _contribution(
        worker="node-a", base=2, current=2, tokens=3, end=2.0)
    lagged = _contribution(
        worker="node-b", base=0, current=2, tokens=5, end=10.0)
    mean, accepted_tokens, admitted = reference_aggregate(
        2, (lagged, fresh), policy=ASYNC_DECOUPLED_V21)
    np.testing.assert_array_equal(mean, np.asarray([(3 * 2 + 5 * 10) / 8]))
    assert accepted_tokens == 8
    assert [item.exact_tokens for item in admitted] == [3, 5] or [
        item.exact_tokens for item in admitted] == [5, 3]

    authority = AsyncV21CommitAuthority(
        run_id="run",
        fence=7,
        state=np.asarray([100.0]),
        version=2,
        outer=OuterState(step=4, accepted_tokens=99),
        policy=ASYNC_DECOUPLED_V21,
        layout_digest="1" * 64,
        code_digest="2" * 64,
        version_digests={0: "0" * 64, 2: f"{2:064x}"},
        minimum_contributions=2,
        minimum_tokens=1,
    )
    authority.install_membership({"node-a": "boot-a", "node-b": "boot-a"})
    result = authority.commit((fresh, lagged))
    np.testing.assert_array_equal(
        result.state, np.asarray([100.0 + (3 * 2 + 5 * 10) / 8]))
    assert result.outer.eta_outer == 1.0
    assert result.outer.accepted_tokens == 107
    encoded = json.dumps(authority.last_manifest, sort_keys=True)
    assert "aggregation_weight" not in encoded
    assert '"exact_tokens"' in encoded


def test_v21_rejects_v20_policy_schema_and_digest():
    record = _contribution(worker="node-a")
    with pytest.raises(ValueError, match="policy"):
        replace(record.identity, policy_id="async-decoupled-v2.0-exp").validate()
    with pytest.raises(ValueError, match="schema"):
        replace(
            record.identity,
            contribution_schema="emender-native-e97-submission-v2",
        ).validate()
    with pytest.raises(ValueError, match="policy"):
        replace(record.identity, policy_digest="0" * 64).validate()


def test_v21_one_owned_one_mutable_and_no_third_cohort():
    lane = AsyncV21WorkerLane.for_test()
    lane.finish_window(np.asarray([1.0]), exact_tokens=7, begin_ns=1, end_ns=2)
    owned = lane.seal()
    lane.finish_window(np.asarray([2.0]), exact_tokens=7, begin_ns=3, end_ns=4)
    assert lane.high_water == {
        "owned_descriptors": 1,
        "mutable_intervals": 1,
        "mutable_windows": 1,
    }
    lane.finish_window(
        np.asarray([3.0]), exact_tokens=7, begin_ns=5, end_ns=6)
    assert lane.high_water["mutable_windows"] == 2
    lane.finish_window(
        np.asarray([4.0]), exact_tokens=7, begin_ns=7, end_ns=8)
    assert lane.local_window == 4
    assert lane.paused_reason is None
    assert lane.snapshot_deferred_reason == "snapshot_admission_limit"
    assert lane.mutable_window_count == 0
    with pytest.raises(Backpressure, match="owned"):
        lane.seal()
    lane.release_owned(owned.digest, outcome="not_selected")


def test_v21_node_ready_requires_all_eight_apply_markers(tmp_path: Path):
    transaction = AtomicEightTrainerApply(
        root=tmp_path,
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot-a",
        result_version=3,
        result_digest="a" * 64,
        trainer_count=8,
    )
    for rank in range(7):
        transaction.record_trainer(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}-boot-a",
            recovery_digest=f"{rank + 1:064x}",
        )
    assert not transaction.ready
    with pytest.raises(Backpressure, match="all eight"):
        transaction.commit_node()

    transaction.record_trainer(
        rank=7,
        trainer_incarnation="trainer-7-boot-a",
        recovery_digest="8" * 64,
    )
    marker = transaction.commit_node()
    assert transaction.ready
    assert marker["schema"] == "emender-async-v21-node-applied-v1"
    assert len(marker["trainers"]) == 8
    assert transaction.record_trainer(
        rank=0,
        trainer_incarnation="trainer-0-boot-a",
        recovery_digest=f"{1:064x}",
    ) == marker["trainers"][0]
    with pytest.raises(ValueError, match="conflicting"):
        transaction.record_trainer(
            rank=0,
            trainer_incarnation="trainer-0-delayed-conflict",
            recovery_digest=f"{1:064x}",
        )

    restarted = transaction.restart_from_latest(
        new_node_incarnation="node-boot-b",
        trainer_incarnations=[f"trainer-{rank}-boot-b" for rank in range(8)],
    )
    assert not restarted.ready
    assert restarted.node_incarnation == "node-boot-b"
    assert not tuple(tmp_path.glob("trainer-applied-*.json"))
    failed = tmp_path / "failed-cohorts/node-boot-a"
    assert len(tuple(failed.glob("trainer-applied-*.json"))) == 8
    assert len(tuple(failed.glob("node-applied-*.json"))) == 1


def test_job_5081295_candidate_preparation_cannot_release_partial_boundary_cohort():
    """Preparation-ready is not safe-boundary-ready (job 5081295)."""
    nodes = []
    for node in range(2):
        transaction = SafeBoundaryRendezvous(
            run_id="job-5081295",
            fence=5_081_295,
            node_id=f"node-{node}",
            node_incarnation=f"node-{node}-boot",
            result_version=1,
            result_digest="a" * 64,
            trainer_count=8,
            rendezvous_timeout_s=420.0,
            apply_timeout_s=60.0,
        )
        for rank in range(8):
            transaction.record_candidate_prepared(
                rank=rank,
                trainer_incarnation=f"node-{node}-trainer-{rank}",
                candidate_digest=f"{node * 8 + rank + 1:064x}",
                prepared_monotonic_s=42.176 + rank * 0.624,
            )
        transaction.open_boundary_rendezvous(opened_monotonic_s=50.0)
        nodes.append(transaction)

    # Mirrors the 9/16 job outcome: candidates existed everywhere while only
    # four trainers on node 0 and five on node 1 were at a K40 boundary.
    for node, boundary_count in zip(nodes, (4, 5), strict=True):
        node_index = int(node.node_id[-1])
        for rank in range(boundary_count):
            node.record_boundary_ready(
                rank=rank,
                trainer_incarnation=f"{node.node_id}-trainer-{rank}",
                candidate_digest=f"{node_index * 8 + rank + 1:064x}",
                boundary_monotonic_s=55.0 + rank,
                local_window=9,
            )
        with pytest.raises(Backpressure, match="boundary"):
            node.release_apply(released_monotonic_s=60.0)
        assert node.apply_deadline_monotonic_s is None
        assert node.applied_count == 0


def test_v21_apply_deadline_starts_at_all_eight_boundary_release():
    transaction = SafeBoundaryRendezvous(
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot",
        result_version=3,
        result_digest="b" * 64,
        trainer_count=8,
        rendezvous_timeout_s=420.0,
        apply_timeout_s=60.0,
    )
    for rank in range(8):
        transaction.record_candidate_prepared(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            candidate_digest=f"{rank + 1:064x}",
            prepared_monotonic_s=10.0 + rank,
        )
    transaction.open_boundary_rendezvous(opened_monotonic_s=30.0)
    for rank in range(8):
        transaction.record_boundary_ready(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            candidate_digest=f"{rank + 1:064x}",
            boundary_monotonic_s=40.0 + rank,
            local_window=4,
        )

    release = transaction.release_apply(released_monotonic_s=48.0)

    assert release["released_monotonic_s"] == 48.0
    assert release["apply_deadline_monotonic_s"] == 108.0
    assert transaction.apply_deadline_monotonic_s == 108.0
    assert transaction.boundary_ready_count == 8


def test_v21_boundary_timeout_aborts_before_any_apply_or_node_marker(
    tmp_path: Path,
):
    transaction = SafeBoundaryRendezvous(
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot-a",
        result_version=3,
        result_digest="c" * 64,
        trainer_count=8,
        rendezvous_timeout_s=420.0,
        apply_timeout_s=60.0,
    )
    for rank in range(8):
        transaction.record_candidate_prepared(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}-boot-a",
            candidate_digest=f"{rank + 1:064x}",
            preparation_started_monotonic_s=1.0,
            prepared_monotonic_s=2.0 + rank,
        )
    transaction.open_boundary_rendezvous(opened_monotonic_s=20.0)
    for rank in range(7):
        transaction.record_boundary_ready(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}-boot-a",
            candidate_digest=f"{rank + 1:064x}",
            boundary_monotonic_s=30.0 + rank,
            local_window=4,
        )
    with pytest.raises(TimeoutError, match="bounded"):
        transaction.record_boundary_ready(
            rank=7,
            trainer_incarnation="trainer-7-boot-a",
            candidate_digest=f"{8:064x}",
            boundary_monotonic_s=441.0,
            local_window=5,
        )
    abort = transaction.abort_before_release(
        aborted_monotonic_s=441.0,
        reason="trainer 7 missed its K boundary",
    )

    apply = AtomicEightTrainerApply(
        root=tmp_path,
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot-a",
        result_version=3,
        result_digest="c" * 64,
        trainer_count=8,
        transaction_digest=transaction.transaction_digest,
    )
    assert abort["applied_count"] == 0
    assert transaction.applied_count == 0
    abort_metrics = transaction.telemetry()
    assert abort_metrics["boundary_rendezvous"]["count"] == 7
    assert abort_metrics["boundary_rendezvous"]["maximum_s"] == 411.0
    assert abort_metrics["boundary_rendezvous"]["p99_s"] == 411.0
    assert abort_metrics["total_foreground_idle"]["count"] == 7
    assert abort_metrics["release_to_apply"]["count"] == 0
    assert not apply.ready
    assert not apply.node_marker_path.exists()
    with pytest.raises(ValueError, match="precedes"):
        transaction.record_applied(
            rank=0,
            trainer_incarnation="trainer-0-boot-a",
            apply_started_monotonic_s=442.0,
            apply_finished_monotonic_s=443.0,
        )
    with pytest.raises(ValueError, match="aborted"):
        transaction.release_apply(released_monotonic_s=442.0)

    # A supervisor retry gets a fresh node incarnation and transaction
    # identity.  The aborted incarnation cannot be replayed into it.
    retry = SafeBoundaryRendezvous(
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot-b",
        result_version=3,
        result_digest="c" * 64,
        trainer_count=8,
        rendezvous_timeout_s=420.0,
        apply_timeout_s=60.0,
    )
    assert retry.transaction_digest != transaction.transaction_digest


def test_v21_success_is_exactly_eight_applies_and_one_transaction_marker(
    tmp_path: Path,
):
    transaction = SafeBoundaryRendezvous(
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot",
        result_version=3,
        result_digest="d" * 64,
        trainer_count=8,
        rendezvous_timeout_s=420.0,
        apply_timeout_s=60.0,
    )
    for rank in range(8):
        transaction.record_candidate_prepared(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            candidate_digest=f"{rank + 1:064x}",
            preparation_started_monotonic_s=10.0,
            prepared_monotonic_s=11.0 + rank,
        )
    transaction.open_boundary_rendezvous(opened_monotonic_s=20.0)
    for rank in range(8):
        transaction.record_boundary_ready(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            candidate_digest=f"{rank + 1:064x}",
            boundary_monotonic_s=30.0 + rank,
            local_window=4 + rank,
        )
    transaction.release_apply(released_monotonic_s=38.0)

    durable = AtomicEightTrainerApply(
        root=tmp_path,
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-boot",
        result_version=3,
        result_digest="d" * 64,
        trainer_count=8,
        transaction_digest=transaction.transaction_digest,
    )
    for rank in range(8):
        transaction.record_applied(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            apply_started_monotonic_s=39.0 + rank,
            apply_finished_monotonic_s=40.0 + rank,
        )
        durable.record_trainer(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            recovery_digest=f"{rank + 17:064x}",
        )
    marker = durable.commit_node()
    telemetry = transaction.telemetry()

    assert transaction.applied_count == 8
    assert marker["apply_transaction_digest"] == (
        transaction.transaction_digest)
    assert len(marker["trainers"]) == 8
    assert len(tuple(tmp_path.glob("trainer-applied-*.json"))) == 8
    assert len(tuple(tmp_path.glob("node-applied-*.json"))) == 1
    assert telemetry["candidate_preparation"] == {
        "count": 8,
        "maximum_s": 8.0,
        "p99_s": 8.0,
        "events_s": [float(value) for value in range(1, 9)],
    }
    assert telemetry["boundary_rendezvous"]["count"] == 8
    assert telemetry["boundary_rendezvous"]["maximum_s"] == 8.0
    assert telemetry["boundary_rendezvous"]["p99_s"] == 8.0
    assert telemetry["release_to_apply"]["maximum_s"] == 9.0
    assert telemetry["release_to_apply"]["p99_s"] == 9.0
    assert telemetry["total_foreground_idle"]["maximum_s"] == 10.0
    assert telemetry["total_foreground_idle"]["p99_s"] == 10.0
    assert durable.commit_node() == marker
    with pytest.raises(ValueError, match="conflicting"):
        transaction.record_applied(
            rank=0,
            trainer_incarnation="trainer-0",
            apply_started_monotonic_s=41.0,
            apply_finished_monotonic_s=42.0,
        )


def test_v21_checkpoint_and_fresh_allocation_restore(tmp_path: Path):
    authority = AsyncV21CommitAuthority.for_test()
    record = _contribution(worker="node-a", tokens=9, end=4.0)
    authority.install_membership({"node-a": "boot-a"})
    result = authority.commit((record,))
    checkpoint = authority.checkpoint(tmp_path / "checkpoint.json")
    restored = AsyncV21CommitAuthority.restore(
        checkpoint,
        new_fence=8,
        expected_run_id="run",
        expected_policy=ASYNC_DECOUPLED_V21,
        expected_layout_digest="1" * 64,
        expected_code_digest="2" * 64,
    )
    assert restored.version == result.version
    assert restored.fence == 8
    assert restored.outer == result.outer
    np.testing.assert_array_equal(restored.state, result.state)

    historical = json.loads(checkpoint.read_text())
    historical["schema"] = "emender-async-decoupled-reference-checkpoint-v2"
    historical.pop("bundle_digest")
    checkpoint.write_text(json.dumps(historical))
    with pytest.raises(ValueError, match="digest|identity"):
        AsyncV21CommitAuthority.restore(
            checkpoint,
            new_fence=9,
            expected_run_id="run",
            expected_policy=ASYNC_DECOUPLED_V21,
            expected_layout_digest="1" * 64,
            expected_code_digest="2" * 64,
        )
