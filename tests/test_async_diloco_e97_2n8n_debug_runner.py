import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def test_async_diloco_e97_2n8n_runner_records_global_partial_quorum_and_resume(tmp_path):
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
    command_file.write_text("python async_diloco_e97_2n8n_debug.py --example\n", encoding="utf-8")
    metrics_json = tmp_path / "metrics.json"
    run_dir = tmp_path / "debug-run"

    subprocess.run(
        [
            sys.executable,
            "scripts/frontier/async_diloco_e97_2n8n_debug.py",
            "--run-id",
            "unit-run",
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            str(run_dir),
            "--metrics-json",
            str(metrics_json),
            "--node-count",
            "4",
            "--worker-count-per-node",
            "4",
            "--local-quorum",
            "3",
            "--global-quorum",
            "3",
            "--local-steps",
            "1",
            "--tokens-per-step",
            "128",
            "--timeout-s",
            "0.2",
            "--global-drop-node-ids",
            "3",
            "--reuse-representative-node",
            "--resume-check",
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
    assert payload["configured_quorum"] == {
        "global_quorum": 3,
        "local_quorum": 3,
        "nodes": 4,
        "reuse_representative_node": True,
        "worker_count_per_node": 4,
    }
    assert payload["effective_quorum"]["global"] == 3
    assert payload["effective_quorum"]["local_distribution"]["min"] == 3
    assert payload["update_counts"]["global"]["accepted"] == 3
    assert payload["update_counts"]["global"]["timed_out"] == 1
    assert payload["induced_lag_drop"]["global_dropped_node_ids"] == [3]
    assert payload["resume_check"]["tested"] is True
    assert payload["resume_check"]["selected_generation"] == 0
    assert payload["resume_check"]["published_generation"] == 1
    assert payload["checkpoint_finalization"]["latest_advanced"] is True
    assert payload["checkpoint_finalization"]["debug_run_directory_only"] is True
    assert payload["checkpoint_cadence"]["recovery_every_generations"] == 1
    assert payload["checkpoint_cadence"]["mode"] == "generation_or_wallclock"
    assert payload["checkpoint_finalization"]["recovery_records"]
    assert payload["checkpoint_finalization"]["total_size_bytes"] > 0
    assert payload["checkpoint_finalization"]["overhead_percent"] >= 0.0
    assert payload["loss_moving_average"]["global"]["loss_100"] > 0.0
    assert payload["production_latest_guard"]["changed"] is False
    assert payload["checkpoint"]["modified_by_run"] is False
    assert (run_dir / "latest.json").exists()


def test_async_diloco_multinode_entrypoint_and_wrappers_are_main_relative():
    entrypoint = ROOT / "scripts/frontier/async_diloco_e97_multinode.py"
    debug_wrapper = ROOT / "scripts/frontier/async_diloco_e97_2n8n_debug.sbatch"
    launch_wrapper = ROOT / "scripts/frontier/async_diloco_e97_256n12h_launch.sbatch"

    assert entrypoint.is_file()
    assert "from async_diloco_e97_2n8n_debug import main" in entrypoint.read_text(encoding="utf-8")

    debug_text = debug_wrapper.read_text(encoding="utf-8")
    launch_text = launch_wrapper.read_text(encoding="utf-8")
    wrapper_text = debug_text + "\n" + launch_text

    expected_entrypoint = "scripts/frontier/async_diloco_e97_multinode.py"
    assert f"ASYNC_ENTRYPOINT=${{ASYNC_ENTRYPOINT:-{expected_entrypoint}}}" in debug_text
    assert f"ASYNC_ENTRYPOINT=${{ASYNC_ENTRYPOINT:-{expected_entrypoint}}}" in launch_text
    assert 'python -u "$ASYNC_ENTRYPOINT"' in debug_text
    assert 'python -u "$ASYNC_ENTRYPOINT"' in launch_text
    assert ".wg-worktrees" not in wrapper_text


def test_async_diloco_launch_wrappers_expose_required_env_knobs():
    debug_text = (ROOT / "scripts/frontier/async_diloco_e97_2n8n_debug.sbatch").read_text(encoding="utf-8")
    launch_text = (ROOT / "scripts/frontier/async_diloco_e97_256n12h_launch.sbatch").read_text(encoding="utf-8")

    for token in (
        "E97_CHECKPOINT=${E97_CHECKPOINT:-",
        "OUTPUT_ROOT=${OUTPUT_ROOT:-",
        "ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-",
        "ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-",
        "REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-",
        "PRODUCTION_LATEST_GUARD=${PRODUCTION_LATEST_GUARD:-",
    ):
        assert token in debug_text

    for token in (
        "SEED_LATEST_PATH=${SEED_LATEST_PATH:-",
        "E97_CHECKPOINT=${E97_CHECKPOINT:-$SEED_LATEST_PATH}",
        "DEFAULT_E97_SEED_LATEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt",
        "TRAINING_TARGET=${TRAINING_TARGET:-E97_1.3B_step1065000_async_diloco_256n12h_20260706}",
        "SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1065000_async_quorum_b4_k40_256n12h}",
        "OUTPUT_ROOT=${OUTPUT_ROOT:-",
        "ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-",
        "ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-",
        "DILOCO_K=${DILOCO_K:-",
        "RECOVERY_EVERY_GENERATIONS=${RECOVERY_EVERY_GENERATIONS:-",
        "RECOVERY_EVERY_SECONDS=${RECOVERY_EVERY_SECONDS:-",
        "EXPORT_EVERY_GENERATIONS=${EXPORT_EVERY_GENERATIONS:-",
        "EXPORT_EVERY_SECONDS=${EXPORT_EVERY_SECONDS:-",
        "REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-",
        "PRODUCTION_LATEST_GUARD=${PRODUCTION_LATEST_GUARD:-",
        "PRODUCTION_LATEST_POLICY=${PRODUCTION_LATEST_POLICY:-run-local-latest-json-with-external-chain-latest-guard}",
    ):
        assert token in launch_text
