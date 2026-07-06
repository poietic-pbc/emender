#!/bin/bash
# Prepared 256-node 12-hour async quorum DiLoCo E97 launch wrapper.
#
# This production wrapper records the exact command and environment artifacts
# before any launch. It is approval-gated for production, and its debug mode
# still executes the real train.py-backed async trainer rather than the
# synthetic protocol harness.

#SBATCH -A bif148
#SBATCH -J async-diloco-e97-256n12h
#SBATCH -p batch
#SBATCH -N 256
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH -o logs/frontier/async_diloco_e97/%x-%j.out
#SBATCH -e logs/frontier/async_diloco_e97/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-submit-refreshed-e97-async-256n12h}
TRAINING_TARGET=${TRAINING_TARGET:-E97_1.3B_step1065000_async_diloco_256n12h_20260706}
DEFAULT_E97_SEED_LATEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
SEED_LATEST_PATH=${SEED_LATEST_PATH:-${E97_CHECKPOINT:-$DEFAULT_E97_SEED_LATEST}}
E97_CHECKPOINT=${E97_CHECKPOINT:-$SEED_LATEST_PATH}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco}
DEFAULT_ENV_PREFIX="${REPO}/.envs/olcf-rocm711-torch210-py312"
if [[ ! -d "$DEFAULT_ENV_PREFIX" && -d /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 ]]; then
  DEFAULT_ENV_PREFIX=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
fi
ENV_PREFIX=${ENV_PREFIX:-$DEFAULT_ENV_PREFIX}
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-$ENV_PREFIX}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1065000_async_quorum_b4_k40_256n12h}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${SCALEOUT_VARIANT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
RUN_DIR="${RUN_ROOT}/async_run"
METRICS_JSON="${ARTIFACT_DIR}/async_diloco_e97_256n_metrics.json"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
ASYNC_ENTRYPOINT=${ASYNC_ENTRYPOINT:-scripts/frontier/e97_async_diloco_train.py}
ASYNC_DILOCO_NON_PRODUCTION_DEBUG=${ASYNC_DILOCO_NON_PRODUCTION_DEBUG:-0}
ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=${ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM:-0}

# Async quorum recipe. Local quorum is six of eight workers per node; global
# quorum defaults to ceil(2/3 * nodes), including scaled debug node counts.
ASYNC_NODE_COUNT=${ASYNC_NODE_COUNT:-256}
ASYNC_WORKER_COUNT_PER_NODE=${ASYNC_WORKER_COUNT_PER_NODE:-8}
ASYNC_WORKER_COUNT=${ASYNC_WORKER_COUNT:-$((ASYNC_NODE_COUNT * ASYNC_WORKER_COUNT_PER_NODE))}
ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-6}
ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$(((2 * ASYNC_NODE_COUNT + 2) / 3))}
ASYNC_GLOBAL_TIMEOUT_S=${ASYNC_GLOBAL_TIMEOUT_S:-240}
ASYNC_STALENESS_POLICY=${ASYNC_STALENESS_POLICY:-reject_stale}
ASYNC_WEIGHTING=${ASYNC_WEIGHTING:-tokens}
ASYNC_TRANSPORT_MODE=${ASYNC_TRANSPORT_MODE:-cray-mpich-gpu-aware-p2p}
ASYNC_GROUP_AGGREGATOR_SIZE=${ASYNC_GROUP_AGGREGATOR_SIZE:-16}
ASYNC_DENSE_UPDATE_STORAGE=${ASYNC_DENSE_UPDATE_STORAGE:-0}

# E97 B4 K40 token geometry.
BATCH_SIZE=${BATCH_SIZE:-4}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
DILOCO_K=${DILOCO_K:-40}
GENERATIONS=${GENERATIONS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
MODEL_LEVEL=${MODEL_LEVEL:-E97}
MODEL_PARAMS=${MODEL_PARAMS:-1.3b}
MODEL_DIM=${MODEL_DIM:-}
MODEL_DEPTH=${MODEL_DEPTH:-}
TOKENS_PER_LOCAL_STEP=${TOKENS_PER_LOCAL_STEP:-16777216}
TOKENS_PER_DILOCO_GENERATION=${TOKENS_PER_DILOCO_GENERATION:-671088640}
TRAIN_MINUTES=${TRAIN_MINUTES:-700}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-12:00:00}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-3072.0}

# Durable-state policy recorded for production validation. The real trainer
# owns run-local latest.json/checkpoints; the external production latest path is
# recorded as a guard and is not mutated by this wrapper.
GENERATION_MANIFEST_EVERY=${GENERATION_MANIFEST_EVERY:-1}
RECOVERY_EVERY_GENERATIONS=${RECOVERY_EVERY_GENERATIONS:-5}
RECOVERY_EVERY_SECONDS=${RECOVERY_EVERY_SECONDS:-600}
EXPORT_EVERY_GENERATIONS=${EXPORT_EVERY_GENERATIONS:-45}
EXPORT_EVERY_SECONDS=${EXPORT_EVERY_SECONDS:-3600}
FINALIZATION_BUFFER_SECONDS=${FINALIZATION_BUFFER_SECONDS:-1200}
CHAIN_UPDATE_ON_FAILURE=${CHAIN_UPDATE_ON_FAILURE:-0}
PRODUCTION_LATEST_GUARD=${PRODUCTION_LATEST_GUARD:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt}
PRODUCTION_LATEST_POLICY=${PRODUCTION_LATEST_POLICY:-run-local-latest-json-with-external-chain-latest-guard}

HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-User 2026-07-06 conditional approval: refreshed seed registered, 4n20m validation passed, queue 256n12h.}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" logs/frontier/async_diloco_e97
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env
PYTHON_BIN=$(command -v python)
[[ -n "$PYTHON_BIN" ]] || { echo "Activated runtime did not provide python" >&2; exit 66; }

[[ -r "$E97_CHECKPOINT" ]] || { echo "E97_CHECKPOINT is not readable: $E97_CHECKPOINT" >&2; exit 4; }
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" != "1" ]]; then
  [[ -r "$DATA" ]] || { echo "DATA is not readable: $DATA" >&2; exit 4; }
fi
[[ -f "$ASYNC_ENTRYPOINT" ]] || {
  echo "Launch cannot start because ASYNC_ENTRYPOINT is missing: $ASYNC_ENTRYPOINT" >&2
  exit 65
}

if [[ "${ASYNC_DILOCO_RUNTIME_PROBE_ONLY:-0}" == "1" ]]; then
  echo "Production validation rejects ASYNC_DILOCO_RUNTIME_PROBE_ONLY=1; use ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1 to run the real trainer with debug-sized settings." >&2
  exit 67
fi
case "$ASYNC_ENTRYPOINT" in
  *async_diloco_e97_multinode.py|*async_diloco_e97_2n8n_debug.py)
    echo "Production validation rejects synthetic/protocol entrypoint: $ASYNC_ENTRYPOINT" >&2
    exit 68
    ;;
esac
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" && "$ASYNC_DILOCO_NON_PRODUCTION_DEBUG" != "1" ]]; then
  echo "Synthetic token streams are only allowed with ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1." >&2
  exit 69
fi

CMD=(
  "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"
  --run-id "${TASK_ID}-${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
  --checkpoint "$E97_CHECKPOINT"
  --run-dir "$RUN_DIR"
  --metrics-json "$METRICS_JSON"
  --data "$DATA"
  --node-count "$ASYNC_NODE_COUNT"
  --worker-count "$ASYNC_WORKER_COUNT"
  --local-quorum "$ASYNC_LOCAL_QUORUM"
  --global-quorum "$ASYNC_GLOBAL_QUORUM"
  --generations "$GENERATIONS"
  --local-steps "$DILOCO_K"
  --timeout-s "$ASYNC_GLOBAL_TIMEOUT_S"
  --level "$MODEL_LEVEL"
  --params "$MODEL_PARAMS"
  --batch-size "$BATCH_SIZE"
  --chunk-size "$CHUNK_SIZE"
  --lr "$LEARNING_RATE"
  --steps "$DILOCO_K"
)
if [[ -n "$MODEL_DIM" ]]; then
  CMD+=(--dim "$MODEL_DIM")
fi
if [[ -n "$MODEL_DEPTH" ]]; then
  CMD+=(--depth "$MODEL_DEPTH")
fi
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" ]]; then
  CMD+=(--synthetic-token-stream)
fi

printf '%q ' "${CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

{
  echo "task_id=$TASK_ID"
  echo "training_target=$TRAINING_TARGET"
  echo "seed_latest_path=$SEED_LATEST_PATH"
  echo "e97_checkpoint=$E97_CHECKPOINT"
  echo "data=$DATA"
  echo "async_entrypoint=$ASYNC_ENTRYPOINT"
  echo "scaleout_variant=$SCALEOUT_VARIANT"
  echo "run_root=$RUN_ROOT"
  echo "run_dir=$RUN_DIR"
  echo "metrics_json=$METRICS_JSON"
  echo "command_file=$COMMAND_FILE"
  echo "env_file=$ENV_FILE"
  echo "env_prefix=$ENV_PREFIX"
  echo "emender_conda_env=$EMENDER_CONDA_ENV"
  echo "python_bin=$PYTHON_BIN"
  echo "requested_walltime=$REQUESTED_WALLTIME"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "async_node_count=$ASYNC_NODE_COUNT"
  echo "async_worker_count_per_node=$ASYNC_WORKER_COUNT_PER_NODE"
  echo "async_worker_count=$ASYNC_WORKER_COUNT"
  echo "async_local_quorum=$ASYNC_LOCAL_QUORUM"
  echo "async_global_quorum=$ASYNC_GLOBAL_QUORUM"
  echo "async_global_timeout_s=$ASYNC_GLOBAL_TIMEOUT_S"
  echo "async_staleness_policy=$ASYNC_STALENESS_POLICY"
  echo "async_weighting=$ASYNC_WEIGHTING"
  echo "async_transport_mode=$ASYNC_TRANSPORT_MODE"
  echo "async_group_aggregator_size=$ASYNC_GROUP_AGGREGATOR_SIZE"
  echo "async_dense_update_storage=$ASYNC_DENSE_UPDATE_STORAGE"
  echo "batch_size=$BATCH_SIZE"
  echo "chunk_size=$CHUNK_SIZE"
  echo "diloco_k=$DILOCO_K"
  echo "generations=$GENERATIONS"
  echo "learning_rate=$LEARNING_RATE"
  echo "model_level=$MODEL_LEVEL"
  echo "model_params=$MODEL_PARAMS"
  echo "model_dim=$MODEL_DIM"
  echo "model_depth=$MODEL_DEPTH"
  echo "tokens_per_local_step=$TOKENS_PER_LOCAL_STEP"
  echo "tokens_per_diloco_generation=$TOKENS_PER_DILOCO_GENERATION"
  echo "generation_manifest_every=$GENERATION_MANIFEST_EVERY"
  echo "recovery_every_generations=$RECOVERY_EVERY_GENERATIONS"
  echo "recovery_every_seconds=$RECOVERY_EVERY_SECONDS"
  echo "export_every_generations=$EXPORT_EVERY_GENERATIONS"
  echo "export_every_seconds=$EXPORT_EVERY_SECONDS"
  echo "finalization_buffer_seconds=$FINALIZATION_BUFFER_SECONDS"
  echo "chain_update_on_failure=$CHAIN_UPDATE_ON_FAILURE"
  echo "production_latest_guard=$PRODUCTION_LATEST_GUARD"
  echo "production_latest_policy=$PRODUCTION_LATEST_POLICY"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "probe_only_allowed=0"
  echo "synthetic_runtime_allowed=0"
  echo "non_production_debug=$ASYNC_DILOCO_NON_PRODUCTION_DEBUG"
  echo "synthetic_token_stream=$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM"
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

if [[ "${ASYNC_DILOCO_HUMAN_APPROVED:-0}" != "1" && "$ASYNC_DILOCO_NON_PRODUCTION_DEBUG" != "1" ]]; then
  echo "Prepared launch package only. Refusing to start 256-node async DiLoCo without ASYNC_DILOCO_HUMAN_APPROVED=1 from WG human approval task." >&2
  exit 64
fi

exec "${CMD[@]}"
