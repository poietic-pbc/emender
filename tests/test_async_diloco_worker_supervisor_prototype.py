import json
import os
from pathlib import Path
import tempfile

import torch

from ndm.async_diloco import (
    AsyncDiLoCoPrototypeConfig,
    AsyncDiLoCoWorkerSpec,
    run_async_diloco_worker_supervisor_prototype,
)


def _write_readonly_state(path):
    torch.save({
        "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "z": torch.tensor([3.0, 4.0], dtype=torch.float32),
    }, path)
    before = path.stat().st_mtime_ns
    os.chmod(path, 0o444)
    return before


def test_single_process_dry_run_exercises_all_layers_and_checkpoint_manager(tmp_path):
    source = tmp_path / "e97_source_state.pt"
    before_mtime = _write_readonly_state(source)
    run_dir = tmp_path / "debug-run"
    specs = [
        AsyncDiLoCoWorkerSpec(worker_id=f"worker-{idx}", gpu_id=idx, local_steps=2)
        for idx in range(4)
    ]

    result = run_async_diloco_worker_supervisor_prototype(
        AsyncDiLoCoPrototypeConfig(
            run_id="dry-run",
            worker_specs=specs,
            local_quorum=4,
            initial_state_path=source,
            run_dir=run_dir,
            use_processes=False,
            include_group_merger=True,
        )
    )

    assert source.stat().st_mtime_ns == before_mtime
    assert result.supervisor.metrics.quorum_size == 4
    assert result.global_merger.metrics.quorum_size == 1
    assert result.global_merger.checkpoint_behavior["latest_advanced"] is True
    assert (run_dir / "latest.json").exists()
    assert result.metrics_path is not None
    payload = json.loads((run_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
    assert payload["effective_quorum"] == 4
    assert payload["checkpoint_behavior"]["latest_advanced"] is True
    assert payload["tokens_per_sec"] > 0.0
    assert payload["update_bytes"]["node"] > 0
    assert payload["generation_latency_s"] >= 0.0


def test_eight_local_worker_processes_complete_one_quorum_without_global_dist(tmp_path):
    specs = [
        AsyncDiLoCoWorkerSpec(worker_id=f"worker-{idx}", gpu_id=idx, local_steps=1)
        for idx in range(8)
    ]

    result = run_async_diloco_worker_supervisor_prototype(
        AsyncDiLoCoPrototypeConfig(
            run_id="eight-process",
            worker_specs=specs,
            local_quorum=8,
            timeout_s=10.0,
            run_dir=tmp_path / "run",
            use_processes=True,
        )
    )

    assert result.supervisor.metrics.quorum_size == 8
    assert result.supervisor.metrics.accepted_updates == 8
    assert result.supervisor.metrics.timed_out_updates == 0
    if torch.distributed.is_available():
        assert not torch.distributed.is_initialized()


def test_delayed_or_failed_worker_does_not_block_when_quorum_permits(tmp_path):
    specs = [
        AsyncDiLoCoWorkerSpec(worker_id=f"worker-{idx}", gpu_id=idx)
        for idx in range(6)
    ]
    specs.append(AsyncDiLoCoWorkerSpec(worker_id="failed", gpu_id=6, fail_before_submit=True))
    specs.append(AsyncDiLoCoWorkerSpec(worker_id="delayed", gpu_id=7, delay_s=2.0))

    result = run_async_diloco_worker_supervisor_prototype(
        AsyncDiLoCoPrototypeConfig(
            run_id="quorum-timeout",
            worker_specs=specs,
            local_quorum=6,
            timeout_s=0.5,
            run_dir=tmp_path / "run",
            use_processes=True,
        )
    )

    assert result.supervisor.metrics.quorum_size == 6
    assert result.global_merger.metrics.latest_advanced is True
    assert (
        result.supervisor.metrics.failed_updates
        + result.supervisor.metrics.timed_out_updates
    ) >= 1
    assert result.global_merger.metrics.tokens_per_sec > 0.0
    assert result.global_merger.metrics.update_bytes["node"] > 0


def test_local_quorum_miss_defers_generation_without_global_latest(tmp_path):
    specs = [
        AsyncDiLoCoWorkerSpec(worker_id="failed", gpu_id=0, fail_before_submit=True),
        AsyncDiLoCoWorkerSpec(worker_id="delayed", gpu_id=1, delay_s=1.0),
    ]
    run_dir = tmp_path / "local-defer"

    result = run_async_diloco_worker_supervisor_prototype(
        AsyncDiLoCoPrototypeConfig(
            run_id="local-defer",
            worker_specs=specs,
            local_quorum=2,
            timeout_s=0.01,
            run_dir=run_dir,
            use_processes=True,
        )
    )

    assert result.supervisor.node_update is None
    assert result.supervisor.metrics.quorum_status == "deferred"
    assert result.supervisor.metrics.accepted_updates == 0
    assert result.supervisor.metrics.timed_out_updates >= 1
    assert result.supervisor.metrics.failed_updates >= 1
    assert result.global_merger.metrics.quorum_status == "deferred"
    assert result.global_merger.metrics.latest_advanced is False
    assert result.global_merger.publish_result is None
    assert not (run_dir / "latest.json").exists()
    payload = json.loads((run_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics_summary"]["quorum_status"]["deferred"] == 2
    assert payload["metrics_summary"]["sustained_health"]["deferred_generations"] == 2
    assert payload["metrics_summary"]["sustained_health"]["max_deferred_streak"] == 2
    assert payload["metrics_summary"]["sustained_health"]["healthy"] is False
    assert payload["checkpoint_behavior"]["latest_advanced"] is False


def test_global_quorum_miss_defers_without_latest_advancement(tmp_path):
    specs = [
        AsyncDiLoCoWorkerSpec(worker_id=f"worker-{idx}", gpu_id=idx)
        for idx in range(6)
    ]
    run_dir = tmp_path / "global-defer"

    result = run_async_diloco_worker_supervisor_prototype(
        AsyncDiLoCoPrototypeConfig(
            run_id="global-defer",
            worker_specs=specs,
            local_quorum=6,
            global_node_count=3,
            global_quorum=2,
            timeout_s=1.0,
            run_dir=run_dir,
            use_processes=False,
        )
    )

    assert result.supervisor.node_update is not None
    assert result.supervisor.metrics.quorum_status == "advanced"
    assert result.global_merger.metrics.quorum_status == "deferred"
    assert result.global_merger.metrics.accepted_updates == 1
    assert result.global_merger.metrics.timed_out_updates == 2
    assert result.global_merger.metrics.quorum_threshold == 2
    assert result.global_merger.metrics.latest_advanced is False
    assert result.global_merger.publish_result is None
    assert not (run_dir / "latest.json").exists()
    payload = json.loads((run_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics_summary"]["quorum_status"] == {"advanced": 1, "deferred": 1}
    assert payload["metrics_summary"]["sustained_health"]["advanced_generations"] == 1
    assert payload["metrics_summary"]["sustained_health"]["deferred_generations"] == 1
    assert payload["metrics_summary"]["sustained_health"]["max_deferred_streak"] == 1
    assert payload["metrics_summary"]["latest_advancement"]["advanced"] is False


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dry"
        path.mkdir()
        test_single_process_dry_run_exercises_all_layers_and_checkpoint_manager(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eight"
        path.mkdir()
        test_eight_local_worker_processes_complete_one_quorum_without_global_dist(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "timeout"
        path.mkdir()
        test_delayed_or_failed_worker_does_not_block_when_quorum_permits(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "local_defer"
        path.mkdir()
        test_local_quorum_miss_defers_generation_without_global_latest(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "global_defer"
        path.mkdir()
        test_global_quorum_miss_defers_without_latest_advancement(path)
