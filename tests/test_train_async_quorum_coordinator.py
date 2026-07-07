import json

import pytest

torch = pytest.importorskip("torch")

import train


def _state(value):
    return {"w": torch.tensor([float(value), float(value) + 1.0])}


def _delta(value):
    return {"w": torch.tensor([float(value), float(value)])}


def _update(rank, generation, value, *, tokens=10, local_steps=2, loss=1.0):
    return train.TrainAsyncRankUpdate(
        rank=int(rank),
        base_generation=int(generation),
        delta=_delta(value),
        tokens=int(tokens),
        local_steps=int(local_steps),
        loss_window={"train": float(loss)},
        rank_step=int(local_steps),
    )


def test_train_async_quorum_advances_and_writes_metrics_jsonl(tmp_path):
    coordinator = train.TrainAsyncQuorumCoordinator(
        initial_state=_state(1.0),
        world_size=3,
        quorum=2,
        run_id="advance",
        metrics_path=tmp_path / "metrics.jsonl",
        now_fn=lambda: 10.0,
    )

    assert coordinator.open_generation() == 0
    first = coordinator.submit_update(_update(0, 0, 2.0, tokens=5, loss=0.7))
    assert not first.advanced
    second = coordinator.submit_update(_update(1, 0, 4.0, tokens=15, loss=0.3))

    assert second.advanced
    assert coordinator.current_generation == 1
    expected = _state(1.0)["w"] + (5 * _delta(2.0)["w"] + 15 * _delta(4.0)["w"]) / 20
    torch.testing.assert_close(coordinator.global_state["w"], expected)

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    for key in (
        "quorum_size",
        "accepted_updates",
        "rejected_updates",
        "stale_updates",
        "timed_out_updates",
        "staleness",
        "merge_latency_s",
        "tokens_per_accepted_update",
        "per_rank_progress",
        "loss_window",
    ):
        assert key in record
    assert record["quorum_size"] == 2
    assert record["accepted_updates"] == 2
    assert record["rejected_updates"] == 0
    assert record["tokens_per_accepted_update"] == pytest.approx(10.0)
    assert record["per_rank_progress"] == {"0": 2, "1": 2}
    assert record["loss_window"]["train"] == pytest.approx(0.4)


def test_train_async_quorum_defers_until_threshold(tmp_path):
    coordinator = train.TrainAsyncQuorumCoordinator(
        initial_state=_state(0.0),
        world_size=4,
        quorum=3,
        run_id="defer",
        metrics_path=tmp_path / "metrics.jsonl",
    )

    result = coordinator.submit_update(_update(0, 0, 10.0))

    assert not result.advanced
    assert result.status == "accepted"
    assert coordinator.current_generation == 0
    torch.testing.assert_close(coordinator.global_state["w"], _state(0.0)["w"])
    assert not (tmp_path / "metrics.jsonl").exists()


def test_train_async_quorum_rejects_stale_update_and_records_rejection(tmp_path):
    coordinator = train.TrainAsyncQuorumCoordinator(
        initial_state=_state(0.0),
        world_size=2,
        quorum=1,
        run_id="stale",
        metrics_path=tmp_path / "metrics.jsonl",
    )
    assert coordinator.submit_update(_update(0, 0, 1.0)).advanced

    stale = coordinator.submit_update(_update(1, 0, 100.0))

    assert not stale.advanced
    assert stale.status == "stale"
    assert stale.accepted is False
    assert coordinator.rejected_updates == 1
    assert coordinator.stale_updates == 1
    torch.testing.assert_close(coordinator.global_state["w"], _state(0.0)["w"] + _delta(1.0)["w"])


def test_train_async_rank_recovery_resync_rebases_local_state(tmp_path):
    coordinator = train.TrainAsyncQuorumCoordinator(
        initial_state=_state(5.0),
        world_size=2,
        quorum=1,
        run_id="resync",
        metrics_path=tmp_path / "metrics.jsonl",
    )
    local_state = _state(6.0)
    old_base = coordinator.global_state
    assert coordinator.submit_update(_update(0, 0, 3.0)).advanced

    resync = coordinator.resync_rank(rank=1, local_state=local_state, base_generation=0)

    assert resync.generation == 1
    assert resync.recovered is True
    torch.testing.assert_close(
        resync.state["w"] - coordinator.global_state["w"],
        local_state["w"] - old_base["w"],
    )


def test_train_async_avg_equivalence_when_all_ranks_participate(tmp_path):
    base = _state(2.0)
    updates = [
        _update(0, 0, 1.0, tokens=1),
        _update(1, 0, 3.0, tokens=1),
        _update(2, 0, 5.0, tokens=1),
    ]
    coordinator = train.TrainAsyncQuorumCoordinator(
        initial_state=base,
        world_size=3,
        quorum=3,
        run_id="equiv",
        metrics_path=tmp_path / "metrics.jsonl",
        weight_by="equal",
    )

    for update in updates:
        result = coordinator.submit_update(update)

    expected = base["w"] + sum((update.delta["w"] for update in updates), torch.zeros_like(base["w"])) / 3
    assert result.advanced
    torch.testing.assert_close(coordinator.global_state["w"], expected)
