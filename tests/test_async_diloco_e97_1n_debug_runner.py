import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_async_diloco_e97_1n_debug_runner_emits_required_ledger(tmp_path):
    checkpoint = tmp_path / "e97_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                "layers.0.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "layers.0.bias": torch.tensor([3.0], dtype=torch.float32),
            }
        },
        checkpoint,
    )
    production_latest = tmp_path / "production_latest.pt"
    production_latest.symlink_to(checkpoint)
    command_file = tmp_path / "command.txt"
    command_file.write_text("python async_diloco_e97_1n_debug.py --example\n", encoding="utf-8")
    metrics_json = tmp_path / "metrics.json"
    run_dir = tmp_path / "debug-run"

    subprocess.run(
        [
            sys.executable,
            "scripts/frontier/async_diloco_e97_1n_debug.py",
            "--run-id",
            "unit-run",
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            str(run_dir),
            "--metrics-json",
            str(metrics_json),
            "--worker-count",
            "4",
            "--local-quorum",
            "3",
            "--local-steps",
            "1",
            "--tokens-per-step",
            "128",
            "--timeout-s",
            "10",
            "--command-file",
            str(command_file),
            "--production-latest-path",
            str(production_latest),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "pass"
    assert payload["training_target"] == "E97_1.3B_step483000_async_diloco_debug"
    assert payload["configured_quorum"] == {
        "global_quorum": 1,
        "local_quorum": 3,
        "worker_count": 4,
    }
    assert payload["effective_quorum"]["local"] == 3
    assert payload["effective_quorum"]["global"] == 1
    assert payload["update_counts"]["local"]["accepted"] == 3
    assert payload["tokens_per_sec"]["global"] > 0.0
    assert payload["generation_duration_s"]["elapsed"] >= 0.0
    assert payload["checkpoint_finalization"]["latest_advanced"] is True
    assert payload["checkpoint_finalization"]["duration_s"] >= 0.0
    assert payload["checkpoint_finalization"]["paths"]
    assert payload["checkpoint_finalization"]["debug_run_directory_only"] is True
    assert payload["production_latest_guard"]["changed"] is False
    assert payload["checkpoint"]["modified_by_run"] is False
    assert (run_dir / "latest.json").exists()
