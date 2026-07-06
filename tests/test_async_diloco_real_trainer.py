import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco_real import (
    RealAsyncDiLoCoConfig,
    RealAsyncWorkerSpec,
    default_tiny_e97_train_args,
    run_real_async_diloco,
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


def test_real_async_trainer_cli_accepts_model_geometry_overrides(tmp_path):
    checkpoint = tmp_path / "seed.pt"
    seed_args = _args(
        tokenizer="p50k_base",
        dim=16,
        depth=1,
        n_heads=2,
        n_state=4,
        n_slots=4,
        n_groups=2,
        linear_state=0,
        use_triton=0,
        use_chunked_e97=0,
        e97_chunk_size=4,
        gate_activation="silu",
        mlp_ratio=0.5,
        mlp_multiple=8,
    )
    model = __import__("train").build_training_model(seed_args)
    torch.save({"model_state_dict": model.state_dict(), "step": 1065000}, checkpoint)
    metrics_json = tmp_path / "metrics.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/frontier/e97_async_diloco_train.py",
            "--run-id",
            "cli-geometry",
            "--run-dir",
            str(tmp_path / "run"),
            "--metrics-json",
            str(metrics_json),
            "--checkpoint",
            str(checkpoint),
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
            "--tokenizer",
            "p50k_base",
            "--dim",
            "16",
            "--depth",
            "1",
            "--n-heads",
            "2",
            "--n-state",
            "4",
            "--n-slots",
            "4",
            "--n-groups",
            "2",
            "--linear-state",
            "0",
            "--use-triton",
            "0",
            "--use-chunked-e97",
            "0",
            "--e97-chunk-size",
            "4",
            "--gate-activation",
            "silu",
            "--mlp-ratio",
            "0.5",
            "--mlp-multiple",
            "8",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    assert stdout["latest_generation"] == 0
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload["global_generations"][0]["metrics"]["latest_advanced"] is True


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


def test_multinode_entrypoint_no_longer_imports_synthetic_debug_harness():
    source = Path("scripts/frontier/async_diloco_e97_multinode.py").read_text(encoding="utf-8")
    assert "from e97_async_diloco_train import main" in source
    assert "from async_diloco_e97_2n8n_debug import main" not in source
