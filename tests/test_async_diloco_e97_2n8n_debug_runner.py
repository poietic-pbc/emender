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
    entrypoint_text = entrypoint.read_text(encoding="utf-8")
    assert "from e97_async_diloco_train import main" in entrypoint_text
    assert "from async_diloco_e97_2n8n_debug import main" not in entrypoint_text

    debug_text = debug_wrapper.read_text(encoding="utf-8")
    launch_text = launch_wrapper.read_text(encoding="utf-8")
    wrapper_text = debug_text + "\n" + launch_text

    expected_entrypoint = "scripts/frontier/async_diloco_e97_multinode.py"
    expected_production_entrypoint = "scripts/frontier/e97_async_diloco_train.py"
    assert f"ASYNC_ENTRYPOINT=${{ASYNC_ENTRYPOINT:-{expected_entrypoint}}}" in debug_text
    assert f"ASYNC_ENTRYPOINT=${{ASYNC_ENTRYPOINT:-{expected_production_entrypoint}}}" in launch_text
    assert 'python -u "$ASYNC_ENTRYPOINT"' in debug_text
    assert "frontier_activate_emender_conda_env" in launch_text
    assert "export TIKTOKEN_CACHE_DIR\nPYTHON_BIN=$(command -v python)" in launch_text
    assert '"$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"' in launch_text
    assert "ASYNC_ACTUAL_MULTINODE_TCP_QUORUM=${ASYNC_ACTUAL_MULTINODE_TCP_QUORUM:-$ASYNC_DILOCO_NON_PRODUCTION_DEBUG}" in launch_text
    assert "LAUNCH_CMD=(\n    srun\n    -N \"$ASYNC_NODE_COUNT\"\n    -n \"$ASYNC_NODE_COUNT\"\n    --ntasks-per-node=1" in launch_text
    assert "CMD+=(--actual-multinode-tcp-quorum)" in launch_text
    assert "Production validation rejects ASYNC_DILOCO_RUNTIME_PROBE_ONLY=1" in launch_text
    assert "ASYNC_REQUIRE_CORRECTED_B4_LADDER=${ASYNC_REQUIRE_CORRECTED_B4_LADDER:-1}" in launch_text
    assert "Corrected E97 B4 smoke ladder pre-submit gate failed" in launch_text
    assert "ASYNC_DILOCO_NON_PRODUCTION_DEBUG" in launch_text
    assert "export TIKTOKEN_CACHE_DIR" in launch_text
    assert "p50k_base tokenizer cache is missing under TIKTOKEN_CACHE_DIR" in launch_text
    assert "*async_diloco_e97_multinode.py|*async_diloco_e97_2n8n_debug.py" in launch_text
    assert "  python -u \"$ASYNC_ENTRYPOINT\"" not in launch_text
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
        "TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}",
        "DEFAULT_E97_SEED_LATEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt",
        "TRAINING_TARGET=${TRAINING_TARGET:-E97_1.3B_step1065000_async_diloco_256n12h_20260706}",
        "SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1065000_async_quorum_b4_k40_256n12h}",
        "OUTPUT_ROOT=${OUTPUT_ROOT:-",
        "ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-6}",
        "ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$(((2 * ASYNC_NODE_COUNT + 2) / 3))}",
        "DILOCO_K=${DILOCO_K:-",
        "BATCH_SIZE=${BATCH_SIZE:-4}",
        "CHUNK_SIZE=${CHUNK_SIZE:-2048}",
        "MODEL_TOKENIZER=${MODEL_TOKENIZER:-p50k_base}",
        "MODEL_DIM=${MODEL_DIM:-1792}",
        "MODEL_DEPTH=${MODEL_DEPTH:-11}",
        "MODEL_N_HEADS=${MODEL_N_HEADS:-216}",
        "MODEL_N_STATE=${MODEL_N_STATE:-32}",
        "MODEL_MLP_RATIO=${MODEL_MLP_RATIO:-2.2623}",
        "ASYNC_E97_BF16=${ASYNC_E97_BF16:-1}",
        "ASYNC_E97_USE_CHUNKED=${ASYNC_E97_USE_CHUNKED:-1}",
        "ASYNC_E97_CHECKPOINT_INTERVAL=${ASYNC_E97_CHECKPOINT_INTERVAL:-64}",
        "ASYNC_E97_GRADIENT_CHECKPOINTING=${ASYNC_E97_GRADIENT_CHECKPOINTING:-1}",
        "ASYNC_E97_PROJECTION_CHUNK_SIZE=${ASYNC_E97_PROJECTION_CHUNK_SIZE:-256}",
        "ASYNC_E97_LOSS_CHUNK_SIZE=${ASYNC_E97_LOSS_CHUNK_SIZE:-256}",
        "RECOVERY_EVERY_GENERATIONS=${RECOVERY_EVERY_GENERATIONS:-",
        "RECOVERY_EVERY_SECONDS=${RECOVERY_EVERY_SECONDS:-",
        "EXPORT_EVERY_GENERATIONS=${EXPORT_EVERY_GENERATIONS:-",
        "EXPORT_EVERY_SECONDS=${EXPORT_EVERY_SECONDS:-",
        "REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-",
        "PRODUCTION_LATEST_GUARD=${PRODUCTION_LATEST_GUARD:-",
        "PRODUCTION_LATEST_POLICY=${PRODUCTION_LATEST_POLICY:-run-local-latest-json-with-external-chain-latest-guard}",
        "SLURM_INTENDED_NTASKS_PER_NODE=${SLURM_INTENDED_NTASKS_PER_NODE:-1}",
        "SLURM_INTENDED_GPUS_PER_TASK=${SLURM_INTENDED_GPUS_PER_TASK:-0}",
        "SLURM_INTENDED_GPU_BIND=${SLURM_INTENDED_GPU_BIND:-unset}",
    ):
        assert token in launch_text


def test_production_async_launcher_records_artifacts_and_real_command_branch():
    launch_text = (ROOT / "scripts/frontier/async_diloco_e97_256n12h_launch.sbatch").read_text(encoding="utf-8")

    assert 'COMMAND_FILE="${ARTIFACT_DIR}/command.txt"' in launch_text
    assert 'ENV_FILE="${ARTIFACT_DIR}/env.txt"' in launch_text
    assert "printf '%q ' \"${LAUNCH_CMD[@]}\" > \"$COMMAND_FILE\"" in launch_text
    assert "} | tee \"$ENV_FILE\"" in launch_text
    assert 'echo "command_file=$COMMAND_FILE"' in launch_text
    assert 'echo "env_file=$ENV_FILE"' in launch_text
    assert 'echo "async_entrypoint=$ASYNC_ENTRYPOINT"' in launch_text
    assert 'echo "tiktoken_cache_dir=$TIKTOKEN_CACHE_DIR"' in launch_text
    assert 'echo "python_bin=$PYTHON_BIN"' in launch_text
    assert 'echo "async_launch_uses_srun=$([[ "$ASYNC_ACTUAL_MULTINODE_TCP_QUORUM" == "1" && "$ASYNC_NODE_COUNT" -gt 1 ]] && echo 1 || echo 0)"' in launch_text
    assert 'echo "presubmit_status=$PRESUBMIT_STATUS"' in launch_text
    assert 'echo "presubmit_failure_count=${#PRESUBMIT_FAILURES[@]}"' in launch_text
    assert 'echo "presubmit_failure_${idx}=${PRESUBMIT_FAILURES[$idx]}"' in launch_text
    assert 'echo "runtime_flag_drift_count=${#RUNTIME_FLAG_DRIFT[@]}"' in launch_text
    assert 'echo "runtime_flag_drift_${idx}=${RUNTIME_FLAG_DRIFT[$idx]}"' in launch_text
    assert 'echo "stable_b4_reference_ntasks_per_node=8"' in launch_text
    assert 'echo "slurm_intended_gpu_bind=$SLURM_INTENDED_GPU_BIND"' in launch_text
    assert 'CMD=(\n  "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"' in launch_text
    assert '--checkpoint "$E97_CHECKPOINT"' in launch_text
    assert '--data "$DATA"' in launch_text
    assert '--worker-count "$ASYNC_WORKER_COUNT"' in launch_text
    assert '--tokenizer "$MODEL_TOKENIZER"' in launch_text
    assert '--batch-size "$BATCH_SIZE"' in launch_text
    assert '--chunk-size "$CHUNK_SIZE"' in launch_text
    assert '--local-steps "$DILOCO_K"' in launch_text
    assert '--steps "$DILOCO_K"' in launch_text
    assert '--e97-chunk-size "$ASYNC_E97_CHUNK_SIZE"' in launch_text
    assert '--checkpoint-interval "$ASYNC_E97_CHECKPOINT_INTERVAL"' in launch_text
    assert '--projection-chunk-size "$ASYNC_E97_PROJECTION_CHUNK_SIZE"' in launch_text
    assert '--loss-chunk-size "$ASYNC_E97_LOSS_CHUNK_SIZE"' in launch_text
    assert '--recovery-every-generations "$RECOVERY_EVERY_GENERATIONS"' in launch_text
    assert '--finalization-reserve-seconds "$FINALIZATION_BUFFER_SECONDS"' in launch_text
    assert 'CMD+=(--dim "$MODEL_DIM")' in launch_text
    assert 'CMD+=(--depth "$MODEL_DEPTH")' in launch_text
    assert 'CMD+=(--n-heads "$MODEL_N_HEADS")' in launch_text
    assert 'CMD+=(--n-state "$MODEL_N_STATE")' in launch_text
    assert 'CMD+=(--mlp-ratio "$MODEL_MLP_RATIO")' in launch_text
    assert 'CMD+=(--bf16)' in launch_text
    assert 'CMD+=(--use-chunked-e97)' in launch_text
    assert 'CMD+=(--gradient-checkpointing)' in launch_text
    assert 'CMD+=(--walltime-remaining-s "$FINALIZATION_BUFFER_SECONDS")' in launch_text
    assert 'if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" ]]; then' in launch_text
    assert 'CMD+=(--synthetic-token-stream)' in launch_text
    assert 'exec "${LAUNCH_CMD[@]}"' in launch_text


def test_production_async_launcher_debug_multinode_uses_srun_not_bare_python():
    launch_text = (ROOT / "scripts/frontier/async_diloco_e97_256n12h_launch.sbatch").read_text(encoding="utf-8")

    assert "--actual-multinode-tcp-quorum" in launch_text
    assert "LAUNCH_CMD=(\n    srun" in launch_text
    assert '-N "$ASYNC_NODE_COUNT"' in launch_text
    assert '-n "$ASYNC_NODE_COUNT"' in launch_text
    assert "--ntasks-per-node=1" in launch_text
    assert "printf '%q ' \"${LAUNCH_CMD[@]}\" > \"$COMMAND_FILE\"" in launch_text
    assert 'exec "${LAUNCH_CMD[@]}"' in launch_text
    assert 'exec "${CMD[@]}"' not in launch_text


def test_production_async_launcher_refuses_non_comparable_corrected_ladder_by_default():
    launch_text = (ROOT / "scripts/frontier/async_diloco_e97_256n12h_launch.sbatch").read_text(encoding="utf-8")

    for token in (
        'if [[ "$ASYNC_REQUIRE_CORRECTED_B4_LADDER" == "1" && "$PRESUBMIT_STATUS" != "pass" ]]; then',
        'exit 70',
        "actual_multinode_tcp_quorum is metadata-only TCP control transport",
        "Slurm topology uses $SLURM_INTENDED_NTASKS_PER_NODE task(s) per node",
        "Slurm GPU request uses --gpus-per-task=$SLURM_INTENDED_GPUS_PER_TASK",
        "Slurm GPU binding is $SLURM_INTENDED_GPU_BIND",
        "runtime flag drift from stable train.py B4 path",
        "linear_state=$MODEL_LINEAR_STATE vs stable train.py B4 linear_state=0",
        "use_chunked_e97=$ASYNC_E97_USE_CHUNKED vs stable train.py B4 use_chunked_e97=0",
        "checkpoint_interval=$ASYNC_E97_CHECKPOINT_INTERVAL vs stable train.py B4 checkpoint_interval=16",
        "Set ASYNC_REQUIRE_CORRECTED_B4_LADDER=0 only for explicitly labeled diagnostic/non-comparable runs.",
    ):
        assert token in launch_text
