#!/bin/bash
# Prepared 256-node 12-hour async quorum DiLoCo E97 launch wrapper.
#
# This file is intentionally approval-gated. It records the exact production
# candidate settings, but exits before training unless the downstream human
# approval task explicitly sets ASYNC_DILOCO_HUMAN_APPROVED=1.

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
TASK_ID=${WG_TASK_ID:-async-diloco-submit-256n12h-human-approval}
TRAINING_TARGET=${TRAINING_TARGET:-E97_1.3B_step483000_async_diloco_256n12h_20260705}
E97_CHECKPOINT=${E97_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step483000_async_quorum_b4_k40_256n12h}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${SCALEOUT_VARIANT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
RUN_DIR="${RUN_ROOT}/async_run"
METRICS_JSON="${ARTIFACT_DIR}/async_diloco_e97_256n_metrics.json"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
ASYNC_ENTRYPOINT=${ASYNC_ENTRYPOINT:-scripts/frontier/async_diloco_e97_2n8n_debug.py}

# Async quorum recipe.
ASYNC_NODE_COUNT=${ASYNC_NODE_COUNT:-256}
ASYNC_WORKER_COUNT_PER_NODE=${ASYNC_WORKER_COUNT_PER_NODE:-8}
ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-8}
ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-240}
ASYNC_LOCAL_TIMEOUT_S=${ASYNC_LOCAL_TIMEOUT_S:-120}
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
TOKENS_PER_LOCAL_STEP=${TOKENS_PER_LOCAL_STEP:-16777216}
TOKENS_PER_DILOCO_GENERATION=${TOKENS_PER_DILOCO_GENERATION:-671088640}
TRAIN_MINUTES=${TRAIN_MINUTES:-700}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-12:00:00}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-3072.0}

# Scale-adaptive durable-state policy. Recovery fires on whichever threshold
# arrives first; generation manifests are every generation.
GENERATION_MANIFEST_EVERY=${GENERATION_MANIFEST_EVERY:-1}
RECOVERY_EVERY_GENERATIONS=${RECOVERY_EVERY_GENERATIONS:-5}
RECOVERY_EVERY_SECONDS=${RECOVERY_EVERY_SECONDS:-600}
EXPORT_EVERY_GENERATIONS=${EXPORT_EVERY_GENERATIONS:-45}
EXPORT_EVERY_SECONDS=${EXPORT_EVERY_SECONDS:-3600}
FINALIZATION_BUFFER_SECONDS=${FINALIZATION_BUFFER_SECONDS:-1200}
CHAIN_UPDATE_ON_FAILURE=${CHAIN_UPDATE_ON_FAILURE:-0}

HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG async-diloco-submit-256n12h-human-approval must be resumed by the human before sbatch; prepared package only.}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" logs/frontier/async_diloco_e97
cd "$REPO"

{
  echo "task_id=$TASK_ID"
  echo "training_target=$TRAINING_TARGET"
  echo "e97_checkpoint=$E97_CHECKPOINT"
  echo "scaleout_variant=$SCALEOUT_VARIANT"
  echo "run_root=$RUN_ROOT"
  echo "run_dir=$RUN_DIR"
  echo "metrics_json=$METRICS_JSON"
  echo "requested_walltime=$REQUESTED_WALLTIME"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "async_node_count=$ASYNC_NODE_COUNT"
  echo "async_worker_count_per_node=$ASYNC_WORKER_COUNT_PER_NODE"
  echo "async_local_quorum=$ASYNC_LOCAL_QUORUM"
  echo "async_global_quorum=$ASYNC_GLOBAL_QUORUM"
  echo "async_local_timeout_s=$ASYNC_LOCAL_TIMEOUT_S"
  echo "async_global_timeout_s=$ASYNC_GLOBAL_TIMEOUT_S"
  echo "async_staleness_policy=$ASYNC_STALENESS_POLICY"
  echo "async_weighting=$ASYNC_WEIGHTING"
  echo "async_transport_mode=$ASYNC_TRANSPORT_MODE"
  echo "async_group_aggregator_size=$ASYNC_GROUP_AGGREGATOR_SIZE"
  echo "batch_size=$BATCH_SIZE"
  echo "chunk_size=$CHUNK_SIZE"
  echo "diloco_k=$DILOCO_K"
  echo "tokens_per_local_step=$TOKENS_PER_LOCAL_STEP"
  echo "tokens_per_diloco_generation=$TOKENS_PER_DILOCO_GENERATION"
  echo "generation_manifest_every=$GENERATION_MANIFEST_EVERY"
  echo "recovery_every_generations=$RECOVERY_EVERY_GENERATIONS"
  echo "recovery_every_seconds=$RECOVERY_EVERY_SECONDS"
  echo "export_every_generations=$EXPORT_EVERY_GENERATIONS"
  echo "export_every_seconds=$EXPORT_EVERY_SECONDS"
  echo "finalization_buffer_seconds=$FINALIZATION_BUFFER_SECONDS"
  echo "chain_update_on_failure=$CHAIN_UPDATE_ON_FAILURE"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "git_commit=$(git rev-parse HEAD)"
} | tee "$ENV_FILE"

cat > "$COMMAND_FILE" <<EOF
Prepared command. The prepare task does not execute it; the downstream
human-approval task must set ASYNC_DILOCO_HUMAN_APPROVED=1 before this wrapper
will attempt it:

python -u "$ASYNC_ENTRYPOINT" \\
  --run-id "${TASK_ID}-\${SLURM_JOB_ID:-manual}-${RUN_STAMP}" \\
  --checkpoint "$E97_CHECKPOINT" \\
  --run-dir "$RUN_DIR" \\
  --metrics-json "$METRICS_JSON" \\
  --node-count "$ASYNC_NODE_COUNT" \\
  --worker-count-per-node "$ASYNC_WORKER_COUNT_PER_NODE" \\
  --local-quorum "$ASYNC_LOCAL_QUORUM" \\
  --global-quorum "$ASYNC_GLOBAL_QUORUM" \\
  --local-steps "$DILOCO_K" \\
  --tokens-per-step "$TOKENS_PER_LOCAL_STEP" \\
  --timeout-s "$ASYNC_GLOBAL_TIMEOUT_S" \\
  --recovery-every-generations "$RECOVERY_EVERY_GENERATIONS" \\
  --recovery-every-seconds "$RECOVERY_EVERY_SECONDS" \\
  --export-every-generations "$EXPORT_EVERY_GENERATIONS" \\
  --export-every-seconds "$EXPORT_EVERY_SECONDS" \\
  --training-target "$TRAINING_TARGET"
EOF

[[ -r "$E97_CHECKPOINT" ]] || { echo "E97_CHECKPOINT is not readable: $E97_CHECKPOINT" >&2; exit 4; }

if [[ "${ASYNC_DILOCO_HUMAN_APPROVED:-0}" != "1" ]]; then
  echo "Prepared launch package only. Refusing to start 256-node async DiLoCo without ASYNC_DILOCO_HUMAN_APPROVED=1 from WG human approval task." >&2
  exit 64
fi

[[ -f "$ASYNC_ENTRYPOINT" ]] || {
  echo "Approved launch cannot start because ASYNC_ENTRYPOINT is missing: $ASYNC_ENTRYPOINT" >&2
  echo "The downstream approval task must first restore or replace the 32/64-node async multi-node entrypoint recorded in the evidence command files." >&2
  exit 65
}

CMD=(
  python -u "$ASYNC_ENTRYPOINT"
  --run-id "${TASK_ID}-${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
  --checkpoint "$E97_CHECKPOINT"
  --run-dir "$RUN_DIR"
  --metrics-json "$METRICS_JSON"
  --node-count "$ASYNC_NODE_COUNT"
  --worker-count-per-node "$ASYNC_WORKER_COUNT_PER_NODE"
  --local-quorum "$ASYNC_LOCAL_QUORUM"
  --global-quorum "$ASYNC_GLOBAL_QUORUM"
  --local-steps "$DILOCO_K"
  --tokens-per-step "$TOKENS_PER_LOCAL_STEP"
  --timeout-s "$ASYNC_GLOBAL_TIMEOUT_S"
  --recovery-every-generations "$RECOVERY_EVERY_GENERATIONS"
  --recovery-every-seconds "$RECOVERY_EVERY_SECONDS"
  --export-every-generations "$EXPORT_EVERY_GENERATIONS"
  --export-every-seconds "$EXPORT_EVERY_SECONDS"
  --training-target "$TRAINING_TARGET"
)

printf '%q ' "${CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"
exec "${CMD[@]}"
