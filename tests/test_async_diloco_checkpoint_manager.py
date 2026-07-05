import json

import pytest

from ndm.async_diloco import (
    GLOBAL_MERGER_ROLE,
    AsyncDiLoCoCheckpointCadence,
    AsyncDiLoCoCheckpointManager,
    AsyncDiLoCoGenerationMetrics,
    validate_checkpoint_latest,
)


class ManualClock:
    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now


class StepClock:
    def __init__(self, now=0.0, step=0.25):
        self.now = float(now)
        self.step = float(step)

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def _metric(generation, *, generation_duration_s=60.0):
    return AsyncDiLoCoGenerationMetrics(
        run_id="run-a",
        generation=generation,
        requested_workers=4,
        participating_workers=4,
        quorum_threshold=3,
        quorum_size=4,
        accepted_updates=4,
        stale_updates=0,
        timed_out_updates=0,
        failed_updates=0,
        invalid_updates=0,
        generation_duration_s=generation_duration_s,
        merge_duration_s=1.0,
        rebase_duration_s=0.5,
        checkpoint_duration_s=0.0,
        tokens_per_sec=1024.0,
        tokens_per_generation=4096,
        update_bytes={"accepted": 128},
        loss_moving_average={"loss_100": 2.0},
        update_norms={"x": 1.0, "z": 2.0},
    )


def _manager(tmp_path, *, role=GLOBAL_MERGER_ROLE, cadence=None, clock=None):
    return AsyncDiLoCoCheckpointManager(
        tmp_path,
        run_id="run-a",
        role=role,
        cadence=cadence or AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=None,
            recovery_every_seconds=None,
            export_every_generations=None,
            export_every_seconds=None,
        ),
        time_source=clock or ManualClock(),
    )


def test_only_global_merger_can_advance_latest(tmp_path):
    worker = _manager(tmp_path, role="worker")

    worker_record = worker.emit_cached_manifest(
        generation=0,
        payload={"worker_id": "worker-0"},
    )
    assert worker_record.latest_advanced is False
    assert not (tmp_path / "latest.json").exists()
    with pytest.raises(PermissionError, match="only the global merger"):
        worker.publish_global_generation(_metric(0))

    result = _manager(tmp_path).publish_global_generation(_metric(0))
    latest_payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    validate_checkpoint_latest(latest_payload)
    assert latest_payload["generation"] == 0
    assert latest_payload["published_by"] == GLOBAL_MERGER_ROLE
    assert result.latest_advanced is True
    assert result.metrics.latest_advanced is True


def test_restart_resume_selects_newest_finalized_global_generation(tmp_path):
    manager = _manager(tmp_path)
    manager.publish_global_generation(_metric(0))
    manager.publish_global_generation(_metric(1))
    manager.publish_global_generation(_metric(2))

    resume = manager.select_resume_source()

    assert resume is not None
    assert resume.generation == 2
    assert "gen_000002" in resume.manifest_path
    assert resume.latest_path == str(tmp_path / "latest.json")


def test_partial_or_inflight_generation_data_does_not_become_latest(tmp_path):
    manager = _manager(tmp_path)
    manager.publish_global_generation(_metric(1))
    partial_path = tmp_path / "generations" / "gen_000009" / "manifest.json"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_text(
        json.dumps({
            "run_id": "run-a",
            "generation": 9,
            "kind": "generation",
            "published_by": GLOBAL_MERGER_ROLE,
            "finalized": False,
        }),
        encoding="utf-8",
    )

    resume = manager.select_resume_source()
    latest_payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))

    assert resume is not None
    assert resume.generation == 1
    assert latest_payload["generation"] == 1


def test_generation_manifest_is_written_for_every_generation(tmp_path):
    manager = _manager(tmp_path)

    for generation in range(4):
        manager.publish_global_generation(_metric(generation))

    for generation in range(4):
        manifest_path = (
            tmp_path
            / "generations"
            / f"gen_{generation:06d}"
            / "manifest.json"
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["kind"] == "generation"
        assert payload["finalized"] is True
        assert payload["metrics"]["generation"] == generation


def test_recovery_cadence_by_generation_interval(tmp_path):
    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_generations=2,
        recovery_every_seconds=None,
        export_every_seconds=None,
    )
    manager = _manager(tmp_path, cadence=cadence)

    results = [manager.publish_global_generation(_metric(gen)) for gen in range(5)]

    recovery_generations = [
        result.metrics.generation
        for result in results
        if result.recovery_checkpoint is not None
    ]
    assert recovery_generations == [0, 2, 4]
    assert results[2].recovery_checkpoint.reason == "generation_interval"


def test_recovery_cadence_by_wall_clock_interval(tmp_path):
    clock = ManualClock()
    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_generations=None,
        recovery_every_seconds=30.0,
        export_every_seconds=None,
    )
    manager = _manager(tmp_path, cadence=cadence, clock=clock)

    first = manager.publish_global_generation(_metric(0))
    clock.now = 10.0
    second = manager.publish_global_generation(_metric(1))
    clock.now = 31.0
    third = manager.publish_global_generation(_metric(2))

    assert first.recovery_checkpoint is not None
    assert second.recovery_checkpoint is None
    assert third.recovery_checkpoint is not None
    assert third.recovery_checkpoint.reason == "wall_clock_interval"


def test_recovery_cadence_uses_whichever_interval_fires_first(tmp_path):
    clock = ManualClock()
    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_generations=10,
        recovery_every_seconds=30.0,
        export_every_seconds=None,
    )
    manager = _manager(tmp_path, cadence=cadence, clock=clock)

    manager.publish_global_generation(_metric(0))
    clock.now = 31.0
    wall_result = manager.publish_global_generation(_metric(1))
    clock.now = 32.0
    no_result = manager.publish_global_generation(_metric(2))

    assert wall_result.recovery_checkpoint is not None
    assert wall_result.recovery_checkpoint.reason == "wall_clock_interval"
    assert no_result.recovery_checkpoint is None

    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_generations=2,
        recovery_every_seconds=300.0,
        export_every_seconds=None,
    )
    manager = _manager(tmp_path / "gen-first", cadence=cadence, clock=ManualClock())
    manager.publish_global_generation(_metric(0))
    manager.publish_global_generation(_metric(1))
    generation_result = manager.publish_global_generation(_metric(2))
    assert generation_result.recovery_checkpoint is not None
    assert generation_result.recovery_checkpoint.reason == "generation_interval"


def test_export_cadence_defaults_to_hourly_and_is_separate_from_recovery(tmp_path):
    clock = ManualClock()
    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_generations=None,
        recovery_every_seconds=None,
    )
    manager = _manager(tmp_path, cadence=cadence, clock=clock)

    first = manager.publish_global_generation(_metric(0))
    clock.now = 3599.0
    second = manager.publish_global_generation(_metric(1))
    clock.now = 3601.0
    third = manager.publish_global_generation(_metric(2))

    assert first.export_checkpoint is not None
    assert first.recovery_checkpoint is None
    assert second.export_checkpoint is None
    assert third.export_checkpoint is not None
    assert third.export_checkpoint.reason == "wall_clock_interval"


def test_finalization_triggers_with_enough_walltime_remaining_once(tmp_path):
    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_seconds=None,
        export_every_seconds=None,
        finalization_reserve_seconds=300.0,
    )
    manager = _manager(tmp_path, cadence=cadence)

    too_late = manager.publish_global_generation(
        _metric(0),
        walltime_remaining_s=20.0,
        estimated_finalization_duration_s=30.0,
    )
    final = manager.publish_global_generation(
        _metric(1),
        walltime_remaining_s=320.0,
        estimated_finalization_duration_s=30.0,
    )
    repeated = manager.publish_global_generation(
        _metric(2),
        walltime_remaining_s=200.0,
        estimated_finalization_duration_s=30.0,
    )

    assert too_late.finalization_checkpoint is None
    assert final.finalization_checkpoint is not None
    assert final.finalization_checkpoint.reason == "walltime_finalization"
    assert repeated.finalization_checkpoint is None


def test_checkpoint_metrics_include_duration_size_overhead_paths_and_latest_status(tmp_path):
    cadence = AsyncDiLoCoCheckpointCadence(
        recovery_every_generations=1,
        recovery_every_seconds=None,
        export_every_seconds=None,
    )
    manager = _manager(tmp_path, cadence=cadence, clock=StepClock(step=0.5))

    result = manager.publish_global_generation(_metric(0, generation_duration_s=10.0))

    assert result.metrics.latest_advanced is True
    assert result.metrics.checkpoint_duration_s > 0.0
    assert result.metrics.checkpoint_paths
    assert all(size > 0 for size in result.metrics.checkpoint_sizes.values())
    assert result.recovery_checkpoint is not None
    assert result.recovery_checkpoint.duration_s == pytest.approx(0.5)
    assert result.recovery_checkpoint.size_bytes > 0
    assert result.recovery_checkpoint.overhead_percent == pytest.approx(5.0)
    assert result.generation_manifest.path in result.metrics.checkpoint_paths
    assert result.recovery_checkpoint.path in result.metrics.checkpoint_paths
