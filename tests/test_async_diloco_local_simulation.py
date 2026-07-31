import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("schedulefree")

from ndm.async_diloco_local import (
    LocalAsyncCheckpointManager,
    LocalAsyncDilocoConfig,
    run_local_async_diloco_simulation,
    run_local_synchronous_diloco_reference,
)


def _assert_state_close(a, b, atol=1e-6, rtol=1e-6):
    assert len(a.x) == len(b.x)
    assert len(a.z) == len(b.z)
    for got, expected in zip(a.x, b.x):
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    for got, expected in zip(a.z, b.z):
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)


def test_zero_delay_full_cohort_matches_synchronous_diloco(tmp_path):
    config = LocalAsyncDilocoConfig(
        num_workers=4,
        quorum=4,
        timeout_s=100.0,
        max_generations=4,
        local_steps=3,
        lr=0.03,
        seed=123,
        token_weighted=True,
        worker_delay=lambda worker, gen: 0.0,
        token_count=lambda worker, gen: 100 + worker,
    )

    async_state, metrics = run_local_async_diloco_simulation(config, tmp_path / "async")
    sync_state, sync_losses = run_local_synchronous_diloco_reference(config)

    _assert_state_close(async_state, sync_state, atol=2e-6, rtol=2e-6)
    assert metrics.summary()["effective_quorum_distribution"] == {"4": 4}
    assert metrics.stale_rejected == 0
    assert metrics.loss_final == pytest.approx(sync_losses[-1], abs=1e-6)


def test_delay_drop_simulation_advances_without_every_worker_and_counts_stale(tmp_path):
    config = LocalAsyncDilocoConfig(
        num_workers=5,
        quorum=3,
        timeout_s=1.0,
        max_generations=5,
        local_steps=2,
        lr=0.03,
        seed=5,
        token_weighted=True,
        worker_delay=lambda worker, gen: 2.5 if worker == 4 else 0.05 * worker,
        worker_drop=lambda worker, gen: worker == 3 and gen in {1, 3},
        token_count=lambda worker, gen: 100 + 10 * worker,
    )

    _state, metrics = run_local_async_diloco_simulation(config, tmp_path / "delay_drop")
    summary = metrics.summary()

    assert summary["generations"] == 5
    assert metrics.dropped_updates == 2
    assert metrics.stale_rejected > 0
    assert summary["effective_quorum_max"] < config.num_workers
    assert summary["effective_quorum_min"] >= config.quorum
    assert summary["quorum_advances"] == 5
    assert summary["loss_final"] < summary["loss_initial"]


def test_timeout_metrics_report_partial_quorum_causes(tmp_path):
    config = LocalAsyncDilocoConfig(
        num_workers=4,
        quorum=4,
        timeout_s=0.5,
        timeout_min_updates=1,
        max_generations=3,
        local_steps=1,
        lr=0.02,
        seed=7,
        token_weighted=False,
        worker_delay=lambda worker, gen: 0.0 if worker == 0 else 2.0,
    )

    _state, metrics = run_local_async_diloco_simulation(config, tmp_path / "timeout")
    summary = metrics.summary()

    assert summary["generations"] == 3
    assert summary["timeout_advances"] == 3
    assert summary["timeout_causes"]["timeout"] == 3
    assert summary["effective_quorum_distribution"] == {"1": 3}
    assert all(record.missing_workers == (1, 2, 3) for record in metrics.generation_records)


def test_checkpoint_resume_continues_from_latest_finalized_generation(tmp_path):
    ckpt_dir = tmp_path / "resume"
    first = LocalAsyncDilocoConfig(
        num_workers=3,
        quorum=2,
        timeout_s=1.0,
        max_generations=3,
        local_steps=2,
        lr=0.03,
        seed=11,
        worker_delay=lambda worker, gen: 0.0,
    )
    _state0, metrics0 = run_local_async_diloco_simulation(first, ckpt_dir)
    manager = LocalAsyncCheckpointManager(ckpt_dir)

    assert metrics0.checkpoint_writes >= 4
    assert manager.latest_generation() == 3

    second = LocalAsyncDilocoConfig(
        num_workers=3,
        quorum=2,
        timeout_s=1.0,
        max_generations=2,
        local_steps=2,
        lr=0.03,
        seed=11,
        worker_delay=lambda worker, gen: 0.0,
    )
    _state1, metrics1 = run_local_async_diloco_simulation(second, ckpt_dir, resume=True)

    assert metrics1.resumed_from_generation == 3
    assert manager.latest_generation() == 5
    assert metrics1.summary()["generations"] == 2
    assert metrics1.loss_final < metrics1.loss_initial
