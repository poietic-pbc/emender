import json

import pytest
import torch

from ndm.async_diloco import (
    GENERATION_REQUIRED_FIELDS,
    SUMMARY_REQUIRED_FIELDS,
    AsyncDiLoCoGenerationMetrics,
    AsyncDiLoCoUpdate,
    build_metrics_summary,
    compute_dense_delta,
    quorum_distribution,
    quorum_merge,
    read_generation_metrics_jsonl,
    read_metrics_json,
    rebase_state,
    stable_json_dumps,
    validate_generation_metrics,
    validate_metrics_summary,
    write_generation_metrics_jsonl,
    write_metrics_json,
)


def _state(x, z):
    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "z": torch.tensor(z, dtype=torch.float32),
    }


def _update(worker_id, base_generation, base, worker, tokens=10, local_steps=2, **kwargs):
    return AsyncDiLoCoUpdate(
        worker_id=worker_id,
        base_generation=base_generation,
        delta=compute_dense_delta(base, worker),
        tokens=tokens,
        local_steps=local_steps,
        loss_moving_average={"loss_100": float(tokens) / 100.0},
        **kwargs,
    )


def test_full_cohort_quorum_equals_synchronous_average():
    base = _state([1.0, 2.0], [4.0, 6.0])
    workers = [
        _state([2.0, 4.0], [6.0, 10.0]),
        _state([4.0, 6.0], [8.0, 12.0]),
        _state([6.0, 8.0], [10.0, 14.0]),
    ]
    updates = [
        _update(f"worker-{idx}", 7, base, worker, tokens=1)
        for idx, worker in enumerate(workers)
    ]

    result = quorum_merge(
        base,
        updates,
        run_id="unit",
        generation=7,
        requested_workers=3,
        quorum_threshold=3,
        weight_by="equal",
        latest_advanced=True,
    )

    expected_x = torch.stack([worker["x"] for worker in workers]).mean(dim=0)
    expected_z = torch.stack([worker["z"] for worker in workers]).mean(dim=0)
    assert torch.allclose(result.state["x"], expected_x)
    assert torch.allclose(result.state["z"], expected_z)
    assert result.metrics.quorum_size == 3
    assert result.metrics.accepted_updates == 3
    assert result.metrics.latest_advanced is True


def test_partial_quorum_token_weighted_merge_uses_accepted_updates_only():
    base = _state([10.0], [20.0])
    fast = _state([14.0], [28.0])
    slow = _state([22.0], [36.0])
    stale = _state([100.0], [200.0])
    updates = [
        _update("fast", 2, base, fast, tokens=1),
        _update("slow", 2, base, slow, tokens=3),
        _update("stale", 1, base, stale, tokens=100),
    ]

    result = quorum_merge(
        base,
        updates,
        run_id="unit",
        generation=2,
        requested_workers=4,
        quorum_threshold=2,
        weight_by="tokens",
        generation_duration_s=2.0,
    )

    expected_dx = ((fast["x"] - base["x"]) * 1.0 + (slow["x"] - base["x"]) * 3.0) / 4.0
    expected_dz = ((fast["z"] - base["z"]) * 1.0 + (slow["z"] - base["z"]) * 3.0) / 4.0
    assert torch.allclose(result.state["x"], base["x"] + expected_dx)
    assert torch.allclose(result.state["z"], base["z"] + expected_dz)
    assert result.metrics.accepted_updates == 2
    assert result.metrics.stale_updates == 1
    assert result.metrics.tokens_per_generation == 4
    assert result.metrics.tokens_per_sec == 2.0
    assert result.metrics.update_bytes["accepted"] == 16


def test_failed_timed_out_and_invalid_updates_are_counted_not_merged():
    base = _state([0.0], [0.0])
    accepted = _update("accepted", 0, base, _state([1.0], [2.0]))
    updates = [
        accepted,
        _update("failed", 0, base, _state([10.0], [20.0]), failed=True),
        _update("timeout", 0, base, _state([10.0], [20.0]), timed_out=True),
        _update("invalid", 0, base, _state([10.0], [20.0]), invalid=True),
    ]

    result = quorum_merge(
        base,
        updates,
        run_id="unit",
        generation=0,
        requested_workers=4,
        quorum_threshold=1,
    )

    assert torch.allclose(result.state["x"], torch.tensor([1.0]))
    assert result.metrics.failed_updates == 1
    assert result.metrics.timed_out_updates == 1
    assert result.metrics.invalid_updates == 1
    assert result.metrics.participating_workers == 4


def test_quorum_not_reached_raises_before_advancement():
    base = _state([0.0], [0.0])
    updates = [_update("stale", 4, base, _state([1.0], [1.0]))]

    with pytest.raises(RuntimeError, match="quorum not reached"):
        quorum_merge(
            base,
            updates,
            run_id="unit",
            generation=5,
            requested_workers=2,
            quorum_threshold=1,
        )


def test_rebase_preserves_local_displacement():
    old_base = _state([1.0, 2.0], [10.0, 20.0])
    local = _state([3.0, 5.0], [15.0, 35.0])
    new_base = _state([11.0, 12.0], [30.0, 40.0])

    rebased = rebase_state(local, old_base, new_base)

    assert torch.allclose(rebased["x"] - new_base["x"], local["x"] - old_base["x"])
    assert torch.allclose(rebased["z"] - new_base["z"], local["z"] - old_base["z"])


def test_quorum_distribution_reports_average_min_max_and_percentiles():
    stats = quorum_distribution([192, 220, 250, 256])

    assert stats["average"] == pytest.approx(229.5)
    assert stats["min"] == 192
    assert stats["max"] == 256
    assert stats["p50"] == pytest.approx(235.0)
    assert stats["p90"] == pytest.approx(254.2)
    assert stats["p95"] == pytest.approx(255.1)
    assert stats["p99"] == pytest.approx(255.82)


def test_metrics_required_fields_and_stable_serialization():
    metric = AsyncDiLoCoGenerationMetrics(
        run_id="run-a",
        generation=3,
        requested_workers=4,
        participating_workers=3,
        quorum_threshold=2,
        quorum_size=3,
        accepted_updates=3,
        stale_updates=1,
        timed_out_updates=0,
        failed_updates=0,
        invalid_updates=0,
        generation_duration_s=12.5,
        merge_duration_s=0.5,
        rebase_duration_s=0.25,
        checkpoint_duration_s=1.5,
        tokens_per_sec=1024.0,
        tokens_per_generation=12800,
        update_bytes={"node": 256, "worker": 512},
        loss_moving_average={"loss_100": 2.5},
        update_norms={"x": 1.25, "z": 2.5},
        checkpoint_paths=("gen_000003/state.pt",),
        checkpoint_sizes={"gen_000003/state.pt": 4096},
        latest_advanced=True,
        resume_source_generation=2,
    )
    summary = build_metrics_summary(
        run_id="run-a",
        requested_workers=4,
        participating_workers=3,
        generations=(metric,),
    )

    generation_payload = metric.to_dict()
    summary_payload = summary.to_dict()
    validate_generation_metrics(generation_payload)
    validate_metrics_summary(summary_payload)
    assert set(GENERATION_REQUIRED_FIELDS).issubset(generation_payload)
    assert set(SUMMARY_REQUIRED_FIELDS).issubset(summary_payload)

    first = stable_json_dumps(summary_payload)
    second = stable_json_dumps(json.loads(first))
    assert first == second
    assert list(json.loads(first)) == sorted(json.loads(first))
    assert '"quorum_distribution"' in first
    assert '"checkpoint_paths":["gen_000003/state.pt"]' in first


def test_core_async_metrics_artifact_round_trip(tmp_path):
    base = _state([1.0], [2.0])
    result0 = quorum_merge(
        base,
        [
            _update("node-0", 0, base, _state([2.0], [4.0]), tokens=5),
            _update("node-1", 0, base, _state([3.0], [6.0]), tokens=7),
        ],
        run_id="artifact-run",
        generation=0,
        requested_workers=2,
        quorum_threshold=2,
        generation_duration_s=1.0,
        merge_duration_s=0.1,
        checkpoint_paths=("generations/gen_000000/state.pt",),
        checkpoint_sizes={"generations/gen_000000/state.pt": 1234},
        latest_advanced=True,
        resume_source_generation=None,
    )
    result1 = quorum_merge(
        result0.state,
        [
            _update("node-0", 1, result0.state, _state([4.0], [8.0]), tokens=3),
            _update("node-1", 1, result0.state, _state([5.0], [10.0]), tokens=3),
        ],
        run_id="artifact-run",
        generation=1,
        requested_workers=2,
        quorum_threshold=2,
        generation_duration_s=2.0,
        merge_duration_s=0.2,
        checkpoint_paths=("generations/gen_000001/state.pt",),
        checkpoint_sizes={"generations/gen_000001/state.pt": 2345},
        latest_advanced=True,
        resume_source_generation=0,
    )
    summary = build_metrics_summary(
        run_id="artifact-run",
        requested_workers=2,
        participating_workers=2,
        generations=(result0.metrics, result1.metrics),
    )
    json_path = tmp_path / "async_diloco_metrics.json"
    jsonl_path = tmp_path / "async_diloco_generations.jsonl"

    write_metrics_json(json_path, summary)
    write_generation_metrics_jsonl(jsonl_path, summary.generations)

    loaded_summary = read_metrics_json(json_path)
    loaded_generations = read_generation_metrics_jsonl(jsonl_path)
    assert loaded_summary.to_dict() == summary.to_dict()
    assert [metric.to_dict() for metric in loaded_generations] == [
        metric.to_dict() for metric in summary.generations
    ]
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["quorum_distribution"]["average"] == 2.0
    assert payload["quorum_distribution"]["min"] == 2
    assert payload["quorum_distribution"]["max"] == 2
    assert payload["totals"]["accepted_updates"] == 4
    assert payload["latest_advancement"] == {"advanced": True, "generation": 1}
    assert payload["resume_source_generation"] == 0
