#!/bin/bash
# Multi-node async DiLoCo E97 debug job. This exercises the quorum prototype
# against the E97 checkpoint in an isolated non-production run directory.

#SBATCH -A bif148
#SBATCH -J async-diloco-e97-2n8n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH -o logs/frontier/async_diloco_e97/%x-%j.out
#SBATCH -e logs/frontier/async_diloco_e97/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-async-diloco-e97-2n8n-debug}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n}
DEFAULT_ENV_PREFIX="${REPO}/.envs/olcf-rocm711-torch210-py312"
if [[ ! -d "$DEFAULT_ENV_PREFIX" && -d /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 ]]; then
  DEFAULT_ENV_PREFIX=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
fi
ENV_PREFIX=${ENV_PREFIX:-$DEFAULT_ENV_PREFIX}
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-$ENV_PREFIX}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
SUMMARY_DIR="${RUN_ROOT}/summaries"
RUN_DIR="${RUN_ROOT}/async_run"
METRICS_JSON="${ARTIFACT_DIR}/async_diloco_e97_${SLURM_JOB_NUM_NODES:-manual}n_metrics.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
STDOUT_PATH="logs/frontier/async_diloco_e97/${SLURM_JOB_NAME:-async-diloco-e97-2n8n}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/async_diloco_e97/${SLURM_JOB_NAME:-async-diloco-e97-2n8n}-${SLURM_JOB_ID:-manual}.err"
ASYNC_ENTRYPOINT=${ASYNC_ENTRYPOINT:-scripts/frontier/async_diloco_e97_multinode.py}

E97_CHECKPOINT=${E97_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt}
TRAINING_TARGET=${TRAINING_TARGET:-E97_1.3B_step483000_async_diloco_debug}
ASYNC_NODE_COUNT=${ASYNC_NODE_COUNT:-${SLURM_JOB_NUM_NODES:-1}}
ASYNC_WORKER_COUNT_PER_NODE=${ASYNC_WORKER_COUNT_PER_NODE:-8}
ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-8}
ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$ASYNC_NODE_COUNT}
ASYNC_LOCAL_STEPS=${ASYNC_LOCAL_STEPS:-1}
ASYNC_TOKENS_PER_STEP=${ASYNC_TOKENS_PER_STEP:-1024}
ASYNC_TIMEOUT_S=${ASYNC_TIMEOUT_S:-900}
ASYNC_DELTA_SCALE=${ASYNC_DELTA_SCALE:-1.0e-8}
ASYNC_RECOVERY_EVERY_GENERATIONS=${ASYNC_RECOVERY_EVERY_GENERATIONS:-1}
ASYNC_RECOVERY_EVERY_SECONDS=${ASYNC_RECOVERY_EVERY_SECONDS:-}
ASYNC_EXPORT_EVERY_GENERATIONS=${ASYNC_EXPORT_EVERY_GENERATIONS:-}
ASYNC_EXPORT_EVERY_SECONDS=${ASYNC_EXPORT_EVERY_SECONDS:-}
ASYNC_REUSE_REPRESENTATIVE_NODE=${ASYNC_REUSE_REPRESENTATIVE_NODE:-0}
ASYNC_USE_PROCESSES=${ASYNC_USE_PROCESSES:-0}
ASYNC_INDUCE_LOCAL_LAG_DROP=${ASYNC_INDUCE_LOCAL_LAG_DROP:-0}
ASYNC_LAG_DELAY_S=${ASYNC_LAG_DELAY_S:-2.0}
ASYNC_GLOBAL_DROP_NODE_IDS=${ASYNC_GLOBAL_DROP_NODE_IDS:-}
ASYNC_RESUME_CHECK=${ASYNC_RESUME_CHECK:-0}
PRODUCTION_LATEST_GUARD=${PRODUCTION_LATEST_GUARD:-}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-00:20:00}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-$(python - <<'PY'
import os
nodes = float(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
wall = os.environ.get("REQUESTED_WALLTIME", "00:20:00")
parts = [float(p) for p in wall.split(":")]
seconds = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
print(f"{nodes * seconds / 3600:.6f}")
PY
)}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SUMMARY_DIR" logs/frontier/async_diloco_e97
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env

[[ -r "$E97_CHECKPOINT" ]] || { echo "E97_CHECKPOINT is not readable: $E97_CHECKPOINT" >&2; exit 4; }
[[ -f "$ASYNC_ENTRYPOINT" ]] || { echo "ASYNC_ENTRYPOINT is missing: $ASYNC_ENTRYPOINT" >&2; exit 65; }

RUN_ID="${TASK_ID}-${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
CMD=(
  python -u "$ASYNC_ENTRYPOINT"
  --run-id "$RUN_ID"
  --checkpoint "$E97_CHECKPOINT"
  --run-dir "$RUN_DIR"
  --metrics-json "$METRICS_JSON"
  --node-count "$ASYNC_NODE_COUNT"
  --worker-count-per-node "$ASYNC_WORKER_COUNT_PER_NODE"
  --local-quorum "$ASYNC_LOCAL_QUORUM"
  --global-quorum "$ASYNC_GLOBAL_QUORUM"
  --local-steps "$ASYNC_LOCAL_STEPS"
  --tokens-per-step "$ASYNC_TOKENS_PER_STEP"
  --timeout-s "$ASYNC_TIMEOUT_S"
  --delta-scale "$ASYNC_DELTA_SCALE"
  --recovery-every-generations "$ASYNC_RECOVERY_EVERY_GENERATIONS"
  --task-id "$TASK_ID"
  --slurm-job-id "${SLURM_JOB_ID:-manual}"
  --slurm-job-name "${SLURM_JOB_NAME:-async-diloco-e97-2n8n}"
  --requested-walltime "$REQUESTED_WALLTIME"
  --requested-node-hours "$REQUESTED_NODE_HOURS"
  --command-file "$COMMAND_FILE"
  --stdout-path "$STDOUT_PATH"
  --stderr-path "$STDERR_PATH"
  --training-target "$TRAINING_TARGET"
)
if [[ -n "$ASYNC_RECOVERY_EVERY_SECONDS" ]]; then
  CMD+=(--recovery-every-seconds "$ASYNC_RECOVERY_EVERY_SECONDS")
fi
if [[ -n "$ASYNC_EXPORT_EVERY_GENERATIONS" ]]; then
  CMD+=(--export-every-generations "$ASYNC_EXPORT_EVERY_GENERATIONS")
fi
if [[ -n "$ASYNC_EXPORT_EVERY_SECONDS" ]]; then
  CMD+=(--export-every-seconds "$ASYNC_EXPORT_EVERY_SECONDS")
fi
if [[ "$ASYNC_REUSE_REPRESENTATIVE_NODE" == "1" ]]; then
  CMD+=(--reuse-representative-node)
fi
if [[ "$ASYNC_USE_PROCESSES" == "1" ]]; then
  CMD+=(--use-processes)
fi
if [[ "$ASYNC_INDUCE_LOCAL_LAG_DROP" == "1" ]]; then
  CMD+=(--induce-local-lag-drop --lag-delay-s "$ASYNC_LAG_DELAY_S")
fi
if [[ -n "$ASYNC_GLOBAL_DROP_NODE_IDS" ]]; then
  CMD+=(--global-drop-node-ids "$ASYNC_GLOBAL_DROP_NODE_IDS")
fi
if [[ "$ASYNC_RESUME_CHECK" == "1" ]]; then
  CMD+=(--resume-check)
fi
if [[ -n "$PRODUCTION_LATEST_GUARD" ]]; then
  CMD+=(--production-latest-path "$PRODUCTION_LATEST_GUARD")
fi

printf '%q ' "${CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

{
  echo "task_id=$TASK_ID"
  echo "run_id=$RUN_ID"
  echo "training_target=$TRAINING_TARGET"
  echo "e97_checkpoint=$E97_CHECKPOINT"
  echo "run_root=$RUN_ROOT"
  echo "run_dir=$RUN_DIR"
  echo "metrics_json=$METRICS_JSON"
  echo "summary_file=$SUMMARY_FILE"
  echo "async_entrypoint=$ASYNC_ENTRYPOINT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "requested_walltime=$REQUESTED_WALLTIME"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "async_node_count=$ASYNC_NODE_COUNT"
  echo "async_worker_count_per_node=$ASYNC_WORKER_COUNT_PER_NODE"
  echo "async_local_quorum=$ASYNC_LOCAL_QUORUM"
  echo "async_global_quorum=$ASYNC_GLOBAL_QUORUM"
  echo "async_local_steps=$ASYNC_LOCAL_STEPS"
  echo "async_tokens_per_step=$ASYNC_TOKENS_PER_STEP"
  echo "async_timeout_s=$ASYNC_TIMEOUT_S"
  echo "async_delta_scale=$ASYNC_DELTA_SCALE"
  echo "async_recovery_every_generations=$ASYNC_RECOVERY_EVERY_GENERATIONS"
  echo "async_recovery_every_seconds=$ASYNC_RECOVERY_EVERY_SECONDS"
  echo "async_export_every_generations=$ASYNC_EXPORT_EVERY_GENERATIONS"
  echo "async_export_every_seconds=$ASYNC_EXPORT_EVERY_SECONDS"
  echo "async_reuse_representative_node=$ASYNC_REUSE_REPRESENTATIVE_NODE"
  echo "async_use_processes=$ASYNC_USE_PROCESSES"
  echo "async_induce_local_lag_drop=$ASYNC_INDUCE_LOCAL_LAG_DROP"
  echo "async_lag_delay_s=$ASYNC_LAG_DELAY_S"
  echo "async_global_drop_node_ids=$ASYNC_GLOBAL_DROP_NODE_IDS"
  echo "async_resume_check=$ASYNC_RESUME_CHECK"
  echo "production_latest_guard=$PRODUCTION_LATEST_GUARD"
  echo "git_commit=$(git rev-parse HEAD)"
  echo
  echo "=== modules ==="
  module -t list 2>&1 || true
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env
  echo
  echo "=== command ==="
  cat "$COMMAND_FILE"
} | tee "$ENV_FILE"

set +e
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/async_diloco_e97_2n8n.log"
STATUS=${PIPESTATUS[0]}
set -e

python - "$SUMMARY_FILE" "$STATUS" "$METRICS_JSON" "$ENV_FILE" "$COMMAND_FILE" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
status = int(sys.argv[2])
metrics_path = Path(sys.argv[3])
env_file = sys.argv[4]
command_file = sys.argv[5]
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
conclusion = metrics.get("conclusion", "no-go-missing-metrics")
summary.write_text(
    "# Async DiLoCo E97 2/8-Node Debug\n\n"
    f"Conclusion: `{conclusion}`\n\n"
    f"Exit status: `{status}`\n\n"
    f"Slurm job ID: `{metrics.get('slurm', {}).get('job_id', 'unknown')}`\n\n"
    f"Slurm nodes: `{metrics.get('slurm', {}).get('nodes', 'unknown')}`\n\n"
    f"Elapsed seconds: `{metrics.get('timing', {}).get('elapsed_s', 'unknown')}`\n\n"
    f"Node-hours: `{metrics.get('slurm', {}).get('requested_node_hours', 'unknown')}`\n\n"
    f"Metrics JSON: `{metrics_path}`\n\n"
    f"Env file: `{env_file}`\n\n"
    f"Command file: `{command_file}`\n\n"
    "## Metrics Excerpt\n\n"
    "```json\n"
    + json.dumps(
        {
            "training_target": metrics.get("training_target"),
            "configured_quorum": metrics.get("configured_quorum"),
            "effective_quorum": metrics.get("effective_quorum"),
            "induced_lag_drop": metrics.get("induced_lag_drop"),
            "update_counts": metrics.get("update_counts"),
            "duration_s": metrics.get("duration_s"),
            "loss_moving_average": metrics.get("loss_moving_average"),
            "tokens_per_sec": metrics.get("tokens_per_sec"),
            "checkpoint_cadence": metrics.get("checkpoint_cadence"),
            "checkpoint_finalization": metrics.get("checkpoint_finalization"),
            "resume_check": metrics.get("resume_check"),
            "production_latest_guard": metrics.get("production_latest_guard"),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n```\n",
    encoding="utf-8",
)
PY

exit "$STATUS"
