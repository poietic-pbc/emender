#!/bin/bash
# Bounded 8-node E97 1.3B pretrained continuation probe for the K320 cadence.
#
# This is intentionally limited to singleton GPU islands, avg outer, no
# within-island DDP, and the staged pretrained architecture validated by
# validate-e97-1-3b.

#SBATCH -A bif148
#SBATCH -J e97-1p3b-8n-k320
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-run-e97-1-3b-3}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_pretrained_8n_k320}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt}
TRAIN_MINUTES=${TRAIN_MINUTES:-50}
DILOCO_K=${DILOCO_K:-320}
DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
SAVE_EVERY=${SAVE_EVERY:-320}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-4}
LOG_EVERY=${LOG_EVERY:-5}
VAL_EVERY=${VAL_EVERY:-10000}
COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG task validate-e97-1-3b explicitly authorized run-e97-1-3b-3 as one bounded 8-node pretrained GPU-island/no-DDP K320 probe; no 16/32/64-node continuation authorized by this task.}

[[ -r "$DATA" ]] || { echo "DATA is not readable: $DATA" >&2; exit 4; }
[[ -r "$VAL_DATA" ]] || { echo "VAL_DATA is not readable: $VAL_DATA" >&2; exit 4; }
[[ -r "$RESUME_CHECKPOINT" ]] || { echo "RESUME_CHECKPOINT is not readable: $RESUME_CHECKPOINT" >&2; exit 4; }

RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SCALEOUT_VARIANT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
STDOUT_PATH="logs/frontier/scaleout/${SLURM_JOB_NAME:-e97-1p3b-8n-k320}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/scaleout/${SLURM_JOB_NAME:-e97-1p3b-8n-k320}-${SLURM_JOB_ID:-manual}.err"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$SUMMARY_DIR" logs/frontier/scaleout
cd "$REPO"

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

if [[ -n "${EMENDER_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$EMENDER_CONDA_ENV"
fi

export TIKTOKEN_CACHE_DIR
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-3442}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}

REQUESTED_NODES=${SLURM_JOB_NUM_NODES:-8}
REQUESTED_SECONDS=3600
REQUESTED_NODE_HOURS=8.000000
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)

COMMON_ARGS=(
  --data "$DATA"
  --val_data "$VAL_DATA"
  --bf16
  --batch_size "$BATCH_SIZE"
  --chunk_size "$CHUNK_SIZE"
  --tokenizer p50k_base
  --optimizer schedulefree
  --seed 42
  --lr 0.001007
  --train_minutes "$TRAIN_MINUTES"
  --timer_after_compile_warmup
  --compile_warmup_steps "$COMPILE_WARMUP_STEPS"
  --log_every "$LOG_EVERY"
  --val_every "$VAL_EVERY"
  --save_every "$SAVE_EVERY"
  --keep_checkpoints "$KEEP_CHECKPOINTS"
  --output "${RUN_ROOT}/train"
  --resume "$RESUME_CHECKPOINT"
  --diloco
  --diloco_k "$DILOCO_K"
  --diloco_outer_optimizer "$DILOCO_OUTER_OPTIMIZER"
  --diloco_outer_lr "$DILOCO_OUTER_LR"
  --diloco_outer_beta "$DILOCO_OUTER_BETA"
  --diloco_export_basis "$DILOCO_EXPORT_BASIS"
  --diloco_island_size "$DILOCO_ISLAND_SIZE"
)

MODEL_ARGS=(
  --level E97
  --params 100m
  --dim 1792
  --depth 11
  --n_heads 216
  --n_state 32
  --n_groups 32
  --n_slots 64
  --state_expansion 2
  --expansion 1.0
  --use_gate 1
  --use_permutation 1
  --gate_activation silu
  --linear_state 0
  --use_write_gate 0
  --use_triton 1
  --use_chunked_e97 0
  --e97_chunk_size 32
  --mlp_ratio 2.2623
  --mlp_multiple 64
  --checkpoint_interval 16
  --weight_decay 0.01
  --grad_accum 1
  --grad_clip 1.0
)

{
  echo "=== e97 1.3B pretrained 8n K320 env ==="
  echo "task_id=$TASK_ID"
  echo "variant=$SCALEOUT_VARIANT"
  echo "start_utc=$START_UTC"
  echo "repo=$REPO"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "resume_checkpoint=$RESUME_CHECKPOINT"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "diloco_k=$DILOCO_K"
  echo "diloco_island_size=$DILOCO_ISLAND_SIZE"
  echo "diloco_outer_optimizer=$DILOCO_OUTER_OPTIMIZER"
  echo "save_every=$SAVE_EVERY"
  echo "train_minutes=$TRAIN_MINUTES"
  echo "batch_size=$BATCH_SIZE"
  echo "chunk_size=$CHUNK_SIZE"
  echo "git_commit=$GIT_COMMIT"
  echo
  echo "=== slurm ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|MASTER|OMP|ROCR|HIP|HSA|NDM|DATA|VAL_DATA|TRAIN_MINUTES|SCALEOUT|DILOCO|TIKTOKEN|HUMAN_APPROVAL|RESUME_CHECKPOINT)=' || true
  echo
  echo "=== command ==="
  printf '%q ' srun -N "$REQUESTED_NODES" -n "$((REQUESTED_NODES * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc \
    'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec python -u train.py "$@"' \
    bash "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}"
  echo
} | tee "$ENV_FILE"

set +e
srun -N "$REQUESTED_NODES" -n "$((REQUESTED_NODES * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest \
  bash -lc 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec python -u train.py "$@"' \
  bash "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python - <<PY
import json
manifest = {
    "task_id": "${TASK_ID}",
    "variant": "${SCALEOUT_VARIANT}",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "job_name": "${SLURM_JOB_NAME:-e97-1p3b-8n-k320}",
    "nodes": int("${REQUESTED_NODES}"),
    "requested_walltime": "${SLURM_TIMELIMIT:-01:00:00}",
    "requested_seconds": ${REQUESTED_SECONDS},
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "human_approval_record": "${HUMAN_APPROVAL_RECORD}",
    "resume_checkpoint": "${RESUME_CHECKPOINT}",
    "architecture_source": "staged args.json from E97_1.3B_diloco_20260623_103742_step260500",
    "diloco_k": int("${DILOCO_K}"),
    "diloco_island_size": int("${DILOCO_ISLAND_SIZE}"),
    "diloco_outer_optimizer": "${DILOCO_OUTER_OPTIMIZER}",
    "diloco_outer_lr": float("${DILOCO_OUTER_LR}"),
    "diloco_outer_beta": float("${DILOCO_OUTER_BETA}"),
    "diloco_export_basis": "${DILOCO_EXPORT_BASIS}",
    "batch_size": int("${BATCH_SIZE}"),
    "chunk_size": int("${CHUNK_SIZE}"),
    "save_every": int("${SAVE_EVERY}"),
    "train_minutes": int("${TRAIN_MINUTES}"),
    "start_utc": "${START_UTC}",
    "end_utc": "${END_UTC}",
    "exit_status": int("${TRAIN_STATUS}"),
    "git_commit": "${GIT_COMMIT}",
    "stdout": "${STDOUT_PATH}",
    "stderr": "${STDERR_PATH}",
    "run_root": "${RUN_ROOT}",
    "env_file": "${ENV_FILE}",
    "train_log": "${LOG_DIR}/train.log",
}
with open("${MANIFEST_FILE}", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

python - "$LOG_DIR/train.log" "$SUMMARY_FILE" "$TRAIN_STATUS" <<'PY'
import sys
from pathlib import Path

train_log = Path(sys.argv[1])
summary = Path(sys.argv[2])
train_status = sys.argv[3]
lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines() if train_log.exists() else []
step_lines = [ln for ln in lines if "loss" in ln and ("tok/s" in ln or "global_tok/s" in ln)]
diloco_lines = [ln for ln in lines if "DiLoCo" in ln or "DILOCO_" in ln]
checkpoint_lines = [ln for ln in lines if "checkpoint" in ln.lower() or "latest.pt" in ln]
error_lines = [
    ln for ln in lines
    if any(k in ln.lower() for k in ("error", "traceback", "exception", "non-finite", "nan", "nccl", "rccl", "timeout", "oom"))
]

with summary.open("w", encoding="utf-8") as out:
    out.write("# E97 1.3B Pretrained 8n K320 Summary\n\n")
    out.write(f"Train status: `{train_status}`\n\n")
    out.write("## Last Training Metric Lines\n\n")
    for ln in step_lines[-20:]:
        out.write(f"```text\n{ln}\n```\n")
    if not step_lines:
        out.write("No training metric lines found.\n")
    out.write("\n## Last DiLoCo Lines\n\n")
    for ln in diloco_lines[-20:]:
        out.write(f"```text\n{ln}\n```\n")
    if not diloco_lines:
        out.write("No DiLoCo lines found.\n")
    out.write("\n## Last Checkpoint Lines\n\n")
    for ln in checkpoint_lines[-20:]:
        out.write(f"```text\n{ln}\n```\n")
    if not checkpoint_lines:
        out.write("No checkpoint lines found.\n")
    out.write("\n## Error-Like Lines\n\n")
    for ln in error_lines[-40:]:
        out.write(f"```text\n{ln}\n```\n")
    if not error_lines:
        out.write("No error-like lines found.\n")
PY

exit "$TRAIN_STATUS"
