import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import (
    AsyncDiLoCoUpdate,
    RESILIENT_QUORUM_DILOCO_MODE,
    default_global_quorum,
    default_local_quorum,
    quorum_merge,
    rebase_state,
)
from ndm.async_diloco_local import (
    LocalAsyncCheckpointManager,
    LocalAsyncDilocoConfig,
    LocalAsyncQuorumMerger,
    LocalAsyncUpdate,
    LocalScheduleFreeDelta,
    LocalScheduleFreeState,
)


def _state(values):
    return {
        "x": torch.tensor([values[0], values[1]], dtype=torch.float32),
        "z": torch.tensor([values[2], values[3]], dtype=torch.float32),
    }


def _update(worker_id, generation, dx, dz, tokens=1, submitted_at=0.0):
    return AsyncDiLoCoUpdate(
        worker_id=str(worker_id),
        base_generation=generation,
        delta={
            "x": torch.tensor(dx, dtype=torch.float32),
            "z": torch.tensor(dz, dtype=torch.float32),
        },
        tokens=tokens,
        local_steps=2,
        loss_moving_average={"train": 0.5},
    )


def test_full_cohort_weighted_merge_matches_synchronous_average():
    base = _state([1.0, 2.0, 3.0, 4.0])
    updates = [
        _update(0, 0, [1.0, 3.0], [5.0, 7.0], tokens=2),
        _update(1, 0, [5.0, 7.0], [11.0, 13.0], tokens=6),
    ]

    result = quorum_merge(
        base,
        updates,
        run_id="test",
        generation=0,
        requested_workers=2,
        quorum_threshold=2,
        eta_outer=1.0,
        weight_by="tokens",
    )

    expected_dx = (2 * updates[0].delta["x"] + 6 * updates[1].delta["x"]) / 8
    expected_dz = (2 * updates[0].delta["z"] + 6 * updates[1].delta["z"]) / 8
    torch.testing.assert_close(result.state["x"], base["x"] + expected_dx)
    torch.testing.assert_close(result.state["z"], base["z"] + expected_dz)
    assert result.metrics.quorum_size == 2


def test_quorum_miss_defers_without_mutating_state_and_records_metrics():
    base = _state([1.0, 2.0, 3.0, 4.0])
    updates = [
        _update(0, 0, [1.0, 0.0], [0.0, 1.0], tokens=2),
        AsyncDiLoCoUpdate(
            worker_id="stale",
            base_generation=-1,
            delta={
                "x": torch.zeros(2, dtype=torch.float32),
                "z": torch.zeros(2, dtype=torch.float32),
            },
            tokens=1,
            local_steps=1,
        ),
        AsyncDiLoCoUpdate(
            worker_id="timed-out",
            base_generation=0,
            delta={
                "x": torch.zeros(2, dtype=torch.float32),
                "z": torch.zeros(2, dtype=torch.float32),
            },
            tokens=1,
            local_steps=1,
            timed_out=True,
        ),
        AsyncDiLoCoUpdate(
            worker_id="failed",
            base_generation=0,
            delta={
                "x": torch.zeros(2, dtype=torch.float32),
                "z": torch.zeros(2, dtype=torch.float32),
            },
            tokens=1,
            local_steps=1,
            failed=True,
        ),
        AsyncDiLoCoUpdate(
            worker_id="invalid",
            base_generation=0,
            delta={
                "x": torch.zeros(2, dtype=torch.float32),
                "z": torch.zeros(2, dtype=torch.float32),
            },
            tokens=1,
            local_steps=1,
            invalid=True,
        ),
    ]

    result = quorum_merge(
        base,
        updates,
        run_id="defer",
        generation=0,
        requested_workers=6,
        quorum_threshold=2,
    )

    assert not result.advanced
    assert result.metrics.quorum_status == "deferred"
    assert result.metrics.latest_advanced is False
    assert result.metrics.accepted_updates == 1
    assert result.metrics.stale_updates == 1
    assert result.metrics.timed_out_updates == 1
    assert result.metrics.failed_updates == 1
    assert result.metrics.invalid_updates == 1
    assert result.metrics.tokens_per_generation == 0
    torch.testing.assert_close(result.state["x"], base["x"])
    torch.testing.assert_close(result.state["z"], base["z"])


def test_resilient_quorum_advances_without_unanimity_and_records_generation_metadata():
    base = _state([0.0, 0.0, 1.0, 1.0])
    updates = [
        _update("rank-0", 4, [1.0, 0.0], [0.0, 1.0], tokens=2),
        _update("rank-1", 4, [3.0, 0.0], [0.0, 3.0], tokens=2),
        AsyncDiLoCoUpdate(
            worker_id="rank-2",
            base_generation=4,
            delta={"x": torch.zeros(2), "z": torch.zeros(2)},
            tokens=0,
            local_steps=0,
            timed_out=True,
        ),
    ]

    result = quorum_merge(
        base,
        updates,
        run_id="resilient",
        generation=4,
        requested_workers=3,
        quorum_threshold=2,
        weight_by="equal",
        missing_worker_ids=("rank-2",),
        checkpoint_state_id="state-gen-4",
    )

    assert result.advanced
    assert result.metrics.mode == RESILIENT_QUORUM_DILOCO_MODE
    assert result.metrics.global_generation == 4
    assert result.metrics.base_generation == 4
    assert result.metrics.quorum_size == 2
    assert result.metrics.timed_out_updates == 1
    assert result.metrics.missing_updates == 1
    assert result.metrics.catchup_events == (
        {
            "worker_id": "rank-2",
            "from_generation": None,
            "to_generation": 4,
            "checkpoint_state_id": "state-gen-4",
            "reason": "missing_or_timed_out",
        },
    )
    torch.testing.assert_close(result.state["x"], torch.tensor([2.0, 0.0]))


def test_stale_and_late_updates_are_rejected_and_get_catchup_state_reset():
    base = _state([0.0, 0.0, 0.0, 0.0])
    fresh = _update("fresh", 3, [2.0, 0.0], [0.0, 2.0], tokens=1)
    stale = _update("stale", 2, [100.0, 0.0], [0.0, 100.0], tokens=1)
    late = _update("late", 4, [200.0, 0.0], [0.0, 200.0], tokens=1)

    result = quorum_merge(
        base,
        (fresh, stale, late),
        run_id="catchup",
        generation=3,
        requested_workers=3,
        quorum_threshold=1,
        checkpoint_state_id="state-gen-3",
    )

    assert result.advanced
    assert result.metrics.accepted_updates == 1
    assert result.metrics.stale_updates == 1
    assert result.metrics.late_updates == 1
    assert result.metrics.rejected_updates == 2
    assert result.metrics.staleness_distribution == {"1": 1}
    assert result.metrics.outcome_rank_ids["accepted"] == ("fresh@base000003",)
    assert result.metrics.outcome_rank_ids["late"] == ("late@base000004",)
    assert result.metrics.catchup_events == (
        {
            "worker_id": "stale",
            "from_generation": 2,
            "to_generation": 3,
            "checkpoint_state_id": "state-gen-3",
            "reason": "stale_generation",
        },
    )
    torch.testing.assert_close(result.state["x"], torch.tensor([2.0, 0.0]))


def test_quorum_miss_still_rejects_corrupt_accepted_delta():
    base = _state([1.0, 2.0, 3.0, 4.0])
    bad = AsyncDiLoCoUpdate(
        worker_id="bad",
        base_generation=0,
        delta={
            "x": torch.tensor([float("nan"), 0.0], dtype=torch.float32),
            "z": torch.zeros(2, dtype=torch.float32),
        },
        tokens=1,
        local_steps=1,
    )

    with pytest.raises(ValueError, match="non-finite"):
        quorum_merge(
            base,
            (bad,),
            run_id="bad",
            generation=0,
            requested_workers=4,
            quorum_threshold=2,
        )


def test_default_quorum_helpers_expose_local_6_of_8_and_global_two_thirds():
    assert default_local_quorum() == 6
    assert default_local_quorum(8) == 6
    assert default_local_quorum(4) == 4
    assert default_global_quorum(1) == 1
    assert default_global_quorum(2) == 2
    assert default_global_quorum(3) == 2
    assert default_global_quorum(256) == 171


def test_rebase_preserves_local_displacement_and_xz_geometry():
    old_base = _state([1.0, 2.0, 10.0, 20.0])
    new_base = _state([3.0, 7.0, 13.0, 29.0])
    local = _state([1.5, 1.0, 11.5, 19.0])

    rebased = rebase_state(local, old_base, new_base)

    torch.testing.assert_close(rebased["x"] - new_base["x"], local["x"] - old_base["x"])
    torch.testing.assert_close(rebased["z"] - new_base["z"], local["z"] - old_base["z"])


def test_merger_rejects_stale_updates_and_records_metrics():
    base = LocalScheduleFreeState(
        x=(torch.zeros(2),),
        z=(torch.zeros(2),),
    )
    merger = LocalAsyncQuorumMerger(
        base,
        LocalAsyncDilocoConfig(num_workers=3, quorum=2, timeout_s=10.0, token_weighted=False),
    )

    assert merger.submit(_local_update(0, 0, [1.0, 0.0], [1.0, 0.0], submitted_at=0.1))
    assert merger.current_generation == 0
    assert merger.submit(_local_update(1, 0, [3.0, 0.0], [3.0, 0.0], submitted_at=0.2))
    assert merger.current_generation == 1

    assert not merger.submit(_local_update(2, 0, [100.0, 0.0], [100.0, 0.0], submitted_at=0.3))
    assert merger.metrics.stale_rejected == 1
    torch.testing.assert_close(merger.state.x[0], torch.tensor([2.0, 0.0]))
    assert merger.metrics.summary()["effective_quorum_distribution"] == {"2": 1}


def test_timeout_advances_with_partial_quorum_and_reports_missing_workers():
    merger = LocalAsyncQuorumMerger(
        LocalScheduleFreeState(x=(torch.zeros(2),), z=(torch.zeros(2),)),
        LocalAsyncDilocoConfig(
            num_workers=4,
            quorum=4,
            timeout_s=1.0,
            timeout_min_updates=1,
            token_weighted=False,
        ),
    )

    merger.submit(_local_update(0, 0, [4.0, 0.0], [8.0, 0.0], submitted_at=0.2))
    assert merger.current_generation == 0
    assert merger.maybe_timeout(1.0)

    record = merger.metrics.generation_records[-1]
    assert merger.current_generation == 1
    assert record.cause == "timeout"
    assert record.effective_quorum == 1
    assert record.missing_workers == (1, 2, 3)
    assert merger.metrics.summary()["timeout_causes"]["timeout"] == 1


def test_checkpoint_manager_latest_roundtrip(tmp_path):
    manager = LocalAsyncCheckpointManager(tmp_path)
    state0 = LocalScheduleFreeState(
        x=(torch.tensor([0.0, 1.0]),),
        z=(torch.tensor([2.0, 3.0]),),
    )
    state1 = LocalScheduleFreeState(
        x=(torch.tensor([4.0, 5.0]),),
        z=(torch.tensor([6.0, 7.0]),),
    )

    manager.save_generation(0, state0, {"kind": "initial"})
    manager.save_generation(1, state1, {"kind": "advanced", "effective_quorum": 2})

    generation, loaded, manifest = manager.load_latest()
    assert generation == 1
    assert manager.latest_generation() == 1
    assert manifest["kind"] == "advanced"
    torch.testing.assert_close(loaded.x[0], state1.x[0])
    assert (tmp_path / "latest").resolve() == manager.generation_dir(1).resolve()


def _local_update(worker_id, generation, dx, dz, tokens=1, submitted_at=0.0):
    return LocalAsyncUpdate(
        worker_id=worker_id,
        base_generation=generation,
        delta=LocalScheduleFreeDelta(
            dx=(torch.tensor(dx, dtype=torch.float32),),
            dz=(torch.tensor(dz, dtype=torch.float32),),
        ),
        tokens=tokens,
        local_steps=2,
        loss_before=1.0,
        loss_after=0.5,
        submitted_at=submitted_at,
    )
