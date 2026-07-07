import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoCheckpointCadence
from ndm.async_diloco import AsyncDiLoCoGenerationMetrics, AsyncDiLoCoUpdate
from ndm.async_diloco_real import (
    RealAsyncDiLoCoConfig,
    RealAsyncFileRankConfig,
    RealAsyncNodeResult,
    RealAsyncWorkerSpec,
    RealAsyncWorkerReport,
    default_tiny_e97_train_args,
    run_real_async_diloco,
    run_real_async_diloco_file_rank,
    _run_real_worker,
)


def _args(**overrides):
    values = {
        "seed": 123,
        "batch_size": 2,
        "chunk_size": 8,
        "steps": 8,
        "lr": 1e-3,
    }
    values.update(overrides)
    return default_tiny_e97_train_args(**values)


def test_real_async_trainer_one_node_reduces_real_worker_updates(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-one-node",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=tuple(
            RealAsyncWorkerSpec(
                worker_id=f"worker-{idx}",
                node_id="node-0",
                local_steps=1,
                seed_offset=idx,
            )
            for idx in range(2)
        ),
        local_quorum=2,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    assert result.latest_generation == 0
    assert (tmp_path / "run" / "latest.json").exists()
    assert len(result.node_results) == 1
    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.metrics.quorum_status == "advanced"
    assert node.metrics.accepted_updates == 2
    assert global_generation.metrics.quorum_status == "advanced"
    assert global_generation.metrics.accepted_updates == 1
    assert global_generation.metrics.latest_advanced is True
    assert node.metrics.tokens_per_generation == 2 * 2 * 9
    assert node.metrics.tokens_per_sec > 0.0
    assert global_generation.metrics.tokens_per_sec > 0.0
    assert node.metrics.update_bytes["worker"] > 0
    assert global_generation.metrics.update_bytes["node"] > 0

    losses = [
        report.losses[0]
        for report in node.worker_reports
        if report.losses
    ]
    assert len(losses) == 2
    assert all(math.isfinite(loss) and loss > 1.0 for loss in losses)
    assert node.metrics.loss_moving_average["loss"] == pytest.approx(sum(losses) / 2)
    assert node.metrics.loss_moving_average["loss"] != pytest.approx(0.99)

    payload = json.loads(Path(result.metrics_json).read_text(encoding="utf-8"))
    assert payload["metrics_summary"]["quorum_status"]["advanced"] == 2
    assert payload["metrics_summary"]["totals"]["accepted_updates"] == 3
    assert payload["global_generations"][0]["metrics"]["checkpoint_paths"]


def test_real_async_trainer_local_quorum_defers_without_process_failure(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-local-defer",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="failed", fail_before_submit=True),
            RealAsyncWorkerSpec(worker_id="timed-out", timed_out=True),
        ),
        local_quorum=2,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.node_update is None
    assert node.metrics.quorum_status == "deferred"
    assert node.metrics.failed_updates == 1
    assert node.metrics.timed_out_updates == 1
    assert global_generation.metrics.quorum_status == "deferred"
    assert global_generation.metrics.latest_advanced is False
    assert not (tmp_path / "run" / "latest.json").exists()
    assert Path(result.metrics_json).exists()


def test_real_async_trainer_global_quorum_defers_without_process_failure(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-global-defer",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
            RealAsyncWorkerSpec(worker_id="worker-1", local_steps=1, seed_offset=1),
        ),
        local_quorum=2,
        global_quorum=2,
        global_node_count=2,
        synthetic_token_stream=True,
    ))

    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.metrics.quorum_status == "advanced"
    assert node.node_update is not None
    assert global_generation.metrics.quorum_status == "deferred"
    assert global_generation.metrics.accepted_updates == 1
    assert global_generation.metrics.timed_out_updates == 1
    assert global_generation.metrics.latest_advanced is False
    assert not (tmp_path / "run" / "latest.json").exists()


def test_real_async_trainer_rejects_stale_worker_but_collects_fresh_quorum(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-stale-accounting",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="stale", stale_generation=-1),
            RealAsyncWorkerSpec(worker_id="fresh-0", seed_offset=1),
            RealAsyncWorkerSpec(worker_id="fresh-1", seed_offset=2),
        ),
        local_quorum=2,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.metrics.quorum_status == "advanced"
    assert node.metrics.stale_updates == 1
    assert node.metrics.accepted_updates == 2
    assert node.node_update is not None
    assert global_generation.metrics.quorum_status == "advanced"
    assert global_generation.metrics.latest_advanced is True


def test_real_async_trainer_cli_smoke_runs_one_generation(tmp_path):
    metrics_json = tmp_path / "metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/frontier/e97_async_diloco_train.py",
            "--run-id",
            "cli-smoke",
            "--run-dir",
            str(tmp_path / "run"),
            "--metrics-json",
            str(metrics_json),
            "--synthetic-token-stream",
            "--worker-count",
            "1",
            "--local-quorum",
            "1",
            "--global-quorum",
            "1",
            "--generations",
            "1",
            "--local-steps",
            "1",
            "--dim",
            "8",
            "--depth",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    assert stdout["latest_generation"] == 0
    assert metrics_json.exists()
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload["global_generations"][0]["metrics"]["latest_advanced"] is True
    assert payload["node_generations"][0]["metrics"]["loss_moving_average"]["loss"] > 1.0


def test_real_async_trainer_accepts_initial_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed.pt"
    seed_args = _args(seed=999)
    model = __import__("train").build_training_model(seed_args)
    torch.save({"model_state_dict": model.state_dict(), "step": 1065000}, checkpoint)

    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-checkpoint-seed",
        run_dir=tmp_path / "run",
        train_args=seed_args,
        initial_checkpoint=checkpoint,
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
        ),
        local_quorum=1,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    assert result.latest_generation == 0
    assert Path(result.metrics_json).exists()


def test_real_async_worker_converts_model_to_bf16_before_training(monkeypatch):
    observed = {}

    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class DummyOptimizer:
        pass

    args = _args(bf16=True)

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_model",
        lambda _args: OneParamModel(),
    )
    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_optimizer",
        lambda model, _args: DummyOptimizer(),
    )

    def fake_train_one_optimizer_step(model, optimizer, _args, **kwargs):
        del optimizer, kwargs
        observed["dtype"] = next(model.parameters()).dtype
        with torch.no_grad():
            model.weight.add_(1.0)
        return {"loss": 2.0, "tokens_processed": 9, "hidden_state": None}

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.train_one_optimizer_step",
        fake_train_one_optimizer_step,
    )

    report = _run_real_worker(
        run_id="bf16-worker",
        generation=0,
        base_state={"weight": torch.zeros(1)},
        train_args=args,
        spec=RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
        synthetic_token_stream=True,
        synthetic_vocab_size=8,
    )

    assert observed["dtype"] is torch.bfloat16
    assert report.update is not None
    assert report.update.delta["weight"].dtype is torch.float32
    assert report.losses == (2.0,)


def test_real_async_trainer_checkpoint_cadence_records_recovery_and_finalization(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-cadence",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
        ),
        local_quorum=1,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
        walltime_remaining_s=0.0,
        estimated_finalization_duration_s=0.0,
        checkpoint_cadence=AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=1,
            recovery_every_seconds=None,
            export_every_generations=None,
            export_every_seconds=None,
            finalization_reserve_seconds=1200.0,
        ),
    ))

    paths = result.generations[0].metrics.checkpoint_paths
    assert len(paths) >= 3
    assert any("/recovery_checkpoints/" in path and "initial" in path for path in paths)
    assert any(
        "/recovery_checkpoints/" in path and "walltime_finalization" in path
        for path in paths
    )


def test_actual_multinode_file_rank_writes_progress_and_global_quorum(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr("ndm.async_diloco_real.train.load_checkpoint", lambda *_args, **_kwargs: None)

    def fake_node_supervisor(**kwargs):
        spec = kwargs["worker_specs"][0]
        generation = kwargs["generation"]
        delta = {"weight": torch.ones(1) * (int(spec.seed_offset) + 1)}
        update = AsyncDiLoCoUpdate(
            worker_id=kwargs["node_id"],
            base_generation=generation,
            delta=delta,
            tokens=8192,
            local_steps=2,
            loss_moving_average={"loss": 2.0 + int(spec.seed_offset), "loss_100": 2.0 + int(spec.seed_offset)},
        )
        worker = RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=kwargs["node_id"],
            base_generation=generation,
            update=update,
            elapsed_s=0.1,
            tokens=8192,
            losses=(2.0 + int(spec.seed_offset),),
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"],
            generation=generation,
            requested_workers=1,
            participating_workers=1,
            quorum_threshold=1,
            quorum_size=1,
            accepted_updates=1,
            stale_updates=0,
            timed_out_updates=0,
            failed_updates=0,
            invalid_updates=0,
            generation_duration_s=0.1,
            merge_duration_s=0.0,
            rebase_duration_s=0.0,
            checkpoint_duration_s=0.0,
            tokens_per_sec=81920.0,
            tokens_per_generation=8192,
            update_bytes={"worker": 4, "node": 4},
            loss_moving_average={"loss": 2.0 + int(spec.seed_offset), "loss_100": 2.0 + int(spec.seed_offset)},
        )
        return RealAsyncNodeResult(
            node_id=kwargs["node_id"],
            generation=generation,
            node_update=update,
            worker_reports=(worker,),
            metrics=metrics,
        )

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", fake_node_supervisor)

    base = {
        "run_id": "file-rank",
        "run_dir": tmp_path / "run",
        "metrics_json": tmp_path / "metrics.json",
        "train_args": _args(data=str(tmp_path / "real_tokens.txt")),
        "node_count": 2,
        "global_quorum": 2,
        "local_steps": 2,
        "timeout_s": 3.0,
        "synthetic_token_stream": False,
        "walltime_remaining_s": 0.0,
        "estimated_finalization_duration_s": 0.0,
        "checkpoint_cadence": AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=1,
            finalization_reserve_seconds=1200.0,
        ),
    }
    (tmp_path / "real_tokens.txt").write_text("real token source placeholder\n", encoding="utf-8")

    rank1 = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(node_rank=1, **base))
    rank0 = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(node_rank=0, **base))

    assert rank1["node_update_submitted"] is True
    assert rank0["coordinator"] is True
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "actual_multinode_file_quorum_debug"
    assert payload["synthetic_token_stream"] is False
    assert payload["latest_generation"] == -1
    assert payload["metadata_quorum_reached"] is True
    assert payload["dense_delta_exchange"] == "not_implemented_for_debug_shared_storage"
    assert payload["no_go_for_async_diloco_claims"] is True
    assert "dense cross-node delta exchange" in payload["no_go_reason"]
    assert payload["global_generations"][0]["metrics"]["quorum_status"] == "metadata_quorum_no_dense_delta"
    assert payload["global_generations"][0]["metrics"]["latest_advanced"] is False
    assert payload["global_generations"][0]["metrics"]["accepted_updates"] == 2
    assert payload["global_generations"][0]["metrics"]["tokens_per_generation"] == 16384
    assert payload["global_generations"][0]["metrics"]["tokens_per_sec"] > 0.0
    assert payload["accepted_node_ids"] == ["node-00000", "node-00001"]
    assert not (tmp_path / "run" / "latest.json").exists()
    assert (tmp_path / "run" / "progress" / "node-00000.heartbeat.json").exists()
    assert (tmp_path / "run" / "progress" / "node-00001.heartbeat.json").exists()
    assert (tmp_path / "run" / "node_updates" / "node-00000.json").exists()
    assert (tmp_path / "run" / "node_updates" / "node-00001.json").exists()


def test_actual_multinode_file_rank_uses_local_worker_topology_and_gpu_devices(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    observed = {}
    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())

    def fake_node_supervisor(**kwargs):
        specs = kwargs["worker_specs"]
        observed["worker_count"] = len(specs)
        observed["devices"] = [spec.device for spec in specs]
        observed["local_quorum"] = kwargs["local_quorum"]
        generation = kwargs["generation"]
        reports = []
        for idx, spec in enumerate(specs):
            update = AsyncDiLoCoUpdate(
                worker_id=spec.worker_id,
                base_generation=generation,
                delta={"weight": torch.ones(1) * (idx + 1)},
                tokens=2049,
                local_steps=1,
                loss_moving_average={"loss": 10.0 + idx, "loss_100": 10.0 + idx},
            )
            reports.append(RealAsyncWorkerReport(
                worker_id=spec.worker_id,
                node_id=kwargs["node_id"],
                base_generation=generation,
                update=update,
                elapsed_s=0.1,
                tokens=2049,
                losses=(10.0 + idx,),
            ))
        node_update = AsyncDiLoCoUpdate(
            worker_id=kwargs["node_id"],
            base_generation=generation,
            delta={"weight": torch.ones(1)},
            tokens=4098,
            local_steps=2,
            loss_moving_average={"loss": 10.5, "loss_100": 10.5},
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"],
            generation=generation,
            requested_workers=2,
            participating_workers=2,
            quorum_threshold=2,
            quorum_size=2,
            accepted_updates=2,
            stale_updates=0,
            timed_out_updates=0,
            failed_updates=0,
            invalid_updates=0,
            generation_duration_s=0.1,
            merge_duration_s=0.0,
            rebase_duration_s=0.0,
            checkpoint_duration_s=0.0,
            tokens_per_sec=40980.0,
            tokens_per_generation=4098,
            update_bytes={"worker": 8, "node": 4},
            loss_moving_average={"loss": 10.5, "loss_100": 10.5},
        )
        return RealAsyncNodeResult(
            node_id=kwargs["node_id"],
            generation=generation,
            node_update=node_update,
            worker_reports=tuple(reports),
            metrics=metrics,
        )

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", fake_node_supervisor)

    run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
        run_id="file-rank-topology",
        run_dir=tmp_path / "run",
        metrics_json=tmp_path / "metrics.json",
        train_args=_args(data=str(tmp_path / "real_tokens.txt")),
        node_rank=0,
        node_count=1,
        global_quorum=1,
        local_steps=1,
        local_worker_count=2,
        local_quorum=2,
        timeout_s=1.0,
        synthetic_token_stream=False,
        device="cuda",
    ))

    assert observed == {
        "worker_count": 2,
        "devices": ["cuda:0", "cuda:1"],
        "local_quorum": 2,
    }
    payload = json.loads((tmp_path / "run" / "node_updates" / "node-00000.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["accepted_updates"] == 2
    assert payload["worker_reports"][0]["worker_id"] == "node-00000/worker-00000"
    assert payload["worker_reports"][1]["worker_id"] == "node-00000/worker-00001"


def test_real_worker_nonfinite_loss_writes_progress_and_invalid_report(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class DummyOptimizer:
        pass

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_model",
        lambda _args: OneParamModel(),
    )
    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_optimizer",
        lambda model, _args: DummyOptimizer(),
    )
    monkeypatch.setattr(
        "ndm.async_diloco_real.train.train_one_optimizer_step",
        lambda *_args, **_kwargs: {
            "loss": float("nan"),
            "tokens_processed": 11,
            "hidden_state": None,
        },
    )

    report = _run_real_worker(
        run_id="nan-loss",
        generation=0,
        base_state={"weight": torch.zeros(1)},
        train_args=_args(),
        spec=RealAsyncWorkerSpec(
            worker_id="node-00000/worker-00000",
            node_id="node-00000",
            local_steps=2,
        ),
        synthetic_token_stream=True,
        synthetic_vocab_size=8,
        progress_dir=tmp_path / "progress",
        node_rank=0,
    )

    assert report.update is None
    assert report.invalid is True
    assert report.failed is False
    assert "non-finite loss at local_step=0" in str(report.error)
    progress_paths = sorted((tmp_path / "progress" / "steps").glob("*.json"))
    assert len(progress_paths) == 1
    payload = json.loads(progress_paths[0].read_text(encoding="utf-8"))
    assert payload["loss"] is None
    assert payload["loss_finite"] is False
    assert payload["tokens"] == 11
    assert payload["invalid"] is True


def test_real_worker_exception_writes_progress_error_state(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class DummyOptimizer:
        pass

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_model",
        lambda _args: OneParamModel(),
    )
    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_optimizer",
        lambda model, _args: DummyOptimizer(),
    )

    def raise_training_error(*_args, **_kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.train_one_optimizer_step",
        raise_training_error,
    )

    report = _run_real_worker(
        run_id="step-error",
        generation=0,
        base_state={"weight": torch.zeros(1)},
        train_args=_args(),
        spec=RealAsyncWorkerSpec(
            worker_id="node-00000/worker-00000",
            node_id="node-00000",
            local_steps=2,
        ),
        synthetic_token_stream=True,
        synthetic_vocab_size=8,
        progress_dir=tmp_path / "progress",
        node_rank=0,
    )

    assert report.update is None
    assert report.failed is True
    assert report.invalid is False
    assert "RuntimeError: synthetic training failure" == report.error
    progress_paths = sorted((tmp_path / "progress" / "steps").glob("*.json"))
    assert len(progress_paths) == 1
    payload = json.loads(progress_paths[0].read_text(encoding="utf-8"))
    assert payload["local_step"] == 0
    assert payload["failed"] is True
    assert payload["invalid"] is True
    assert payload["error"] == "RuntimeError: synthetic training failure"


def test_actual_multinode_file_rank_serializes_invalid_node_without_nan(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())

    def fake_node_supervisor(**kwargs):
        spec = kwargs["worker_specs"][0]
        worker = RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=kwargs["node_id"],
            base_generation=kwargs["generation"],
            update=None,
            elapsed_s=0.1,
            tokens=0,
            losses=(),
            invalid=True,
            error="delta tensor 'weight' contains non-finite values",
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"],
            generation=kwargs["generation"],
            requested_workers=1,
            participating_workers=1,
            quorum_threshold=1,
            quorum_size=0,
            accepted_updates=0,
            stale_updates=0,
            timed_out_updates=0,
            failed_updates=0,
            invalid_updates=1,
            generation_duration_s=0.1,
            merge_duration_s=0.0,
            rebase_duration_s=0.0,
            checkpoint_duration_s=0.0,
            tokens_per_sec=0.0,
            tokens_per_generation=0,
            update_bytes={"worker": 0, "node": 0},
            loss_moving_average={},
            quorum_status="deferred",
        )
        return RealAsyncNodeResult(
            node_id=kwargs["node_id"],
            generation=kwargs["generation"],
            node_update=None,
            worker_reports=(worker,),
            metrics=metrics,
        )

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", fake_node_supervisor)

    result = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
        run_id="invalid-node",
        run_dir=tmp_path / "run",
        metrics_json=tmp_path / "metrics.json",
        train_args=_args(data=str(tmp_path / "real_tokens.txt")),
        node_rank=0,
        node_count=1,
        global_quorum=1,
        local_steps=10,
        timeout_s=0.1,
        synthetic_token_stream=False,
    ))

    assert result["node_update_submitted"] is False
    node_payload = json.loads((tmp_path / "run" / "node_updates" / "node-00000.json").read_text(encoding="utf-8"))
    assert node_payload["loss"] is None
    assert node_payload["loss_finite"] is False
    assert node_payload["invalid_reasons"] == ["delta tensor 'weight' contains non-finite values"]
    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["partial"] is True
    assert metrics_payload["global_generations"][0]["metrics"]["quorum_status"] == "deferred"
    assert metrics_payload["global_generations"][0]["metrics"]["invalid_updates"] == 1


def test_actual_multinode_file_rank_rejects_synthetic_stream(tmp_path):
    with pytest.raises(ValueError, match="synthetic_token_stream is disabled"):
        run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
            run_id="reject-synthetic",
            run_dir=tmp_path / "run",
            metrics_json=tmp_path / "metrics.json",
            train_args=_args(),
            node_rank=0,
            node_count=1,
            global_quorum=1,
            local_steps=1,
            synthetic_token_stream=True,
        ))


def test_multinode_entrypoint_no_longer_imports_synthetic_debug_harness():
    source = Path("scripts/frontier/async_diloco_e97_multinode.py").read_text(encoding="utf-8")
    assert "from e97_async_diloco_train import main" in source
    assert "from async_diloco_e97_2n8n_debug import main" not in source
