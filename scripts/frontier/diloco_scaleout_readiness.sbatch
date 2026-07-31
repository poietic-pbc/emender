#!/bin/bash
# Frontier DiLoCo scaleout readiness template.
# This file is a launch template only. Do not submit it until the corresponding
# WG task has recorded debug evidence, ledger impact, and human approval.

#SBATCH -A bif148
#SBATCH -J emender-diloco-scaleout
#SBATCH -p batch
#SBATCH -N 8
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-frontier-diloco-scaleout-readiness}
ACCOUNT=${SLURM_JOB_ACCOUNT:-bif148}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-gdn2-MLP}
SCALEOUT_NODES=${SCALEOUT_NODES:-${SLURM_JOB_NUM_NODES:-8}}
SCALEOUT_WALLTIME=${SCALEOUT_WALLTIME:-${SLURM_TIMELIMIT:-04:00:00}}
TRAIN_MINUTES=${TRAIN_MINUTES:-210}
HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:?Set HUMAN_APPROVAL_RECORD to the WG message or committed approval artifact before submission}
DATA=${DATA:?Set DATA to a Frontier-readable commapile training text or shard}
VAL_DATA=${VAL_DATA:-}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:?Set TIKTOKEN_CACHE_DIR to a pre-populated shared tokenizer cache}
DILOCO_K=${DILOCO_K:-250}
DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-8}
DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-global}
DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-8}
DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
LOG_EVERY=${LOG_EVERY:-10}
VAL_EVERY=${VAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-250}
COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    OUTPUT_ROOT="${MEMBERWORK}/emender/frontier_runs/scaleout"
  else
    OUTPUT_ROOT="${REPO}/frontier_runs/scaleout"
  fi
fi

RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SCALEOUT_VARIANT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
STDOUT_PATH="logs/frontier/scaleout/${SLURM_JOB_NAME:-emender-diloco-scaleout}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/scaleout/${SLURM_JOB_NAME:-emender-diloco-scaleout}-${SLURM_JOB_ID:-manual}.err"

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

if [[ -z "${GDN2_PATH:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    export GDN2_PATH="${MEMBERWORK}/emender/src/GatedDeltaNet-2"
  else
    export GDN2_PATH="${REPO}/../GatedDeltaNet-2"
  fi
else
  export GDN2_PATH
fi

COMMON_ARGS=(
  --data "$DATA"
  --bf16
  --batch_size "$BATCH_SIZE"
  --chunk_size "$CHUNK_SIZE"
  --tokenizer "${TOKENIZER:-p50k_base}"
  --optimizer "${OPTIMIZER:-schedulefree}"
  --seed "${SEED:-42}"
  --lr "${LR:-9e-4}"
  --train_minutes "$TRAIN_MINUTES"
  --timer_after_compile_warmup
  --compile_warmup_steps "$COMPILE_WARMUP_STEPS"
  --log_every "$LOG_EVERY"
  --val_every "$VAL_EVERY"
  --save_every "$SAVE_EVERY"
  --keep_checkpoints "${KEEP_CHECKPOINTS:-4}"
  --output "${RUN_ROOT}/train"
  --diloco
  --diloco_k "$DILOCO_K"
  --diloco_outer_optimizer "$DILOCO_OUTER_OPTIMIZER"
  --diloco_outer_lr "$DILOCO_OUTER_LR"
  --diloco_outer_beta "$DILOCO_OUTER_BETA"
  --diloco_export_basis "$DILOCO_EXPORT_BASIS"
  --diloco_island_size "$DILOCO_ISLAND_SIZE"
  --diloco_merge_topology "$DILOCO_MERGE_TOPOLOGY"
  --diloco_merge_group_size "$DILOCO_MERGE_GROUP_SIZE"
  --diloco_merge_bucket_numel "$DILOCO_MERGE_BUCKET_NUMEL"
)

if [[ -n "$VAL_DATA" ]]; then
  COMMON_ARGS+=(--val_data "$VAL_DATA")
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -r "$RESUME_CHECKPOINT" ]] || { echo "RESUME_CHECKPOINT is not readable: $RESUME_CHECKPOINT" >&2; exit 4; }
  COMMON_ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

if [[ -n "${DILOCO_BOOTSTRAP_OUTER_STATE:-}" ]]; then
  COMMON_ARGS+=(--diloco_bootstrap_outer_state "$DILOCO_BOOTSTRAP_OUTER_STATE")
fi

case "$SCALEOUT_VARIANT" in
  e97-MLP)
    MODEL_ARGS=(
      --level E97
      --linear_state 0
      --use_triton 1
      --n_state 32
      --n_heads 323
      --dim 1536
      --depth 10
      --expansion 1.0
      --use_gate 1
      --gate_activation silu
      --mlp_ratio 1.5
      --mlp_multiple 64
    )
    ;;
  e97-linear-MLP)
    MODEL_ARGS=(
      --level E97
      --linear_state 1
      --use_triton 1
      --use_chunked_e97 1
      --e97_chunk_size "${E97_CHUNK_SIZE:-32}"
      --n_state 32
      --n_heads 128
      --dim 1536
      --depth 21
      --expansion 1.0
      --use_gate 1
      --gate_activation silu
      --mlp_ratio 2.0
      --mlp_multiple 64
    )
    ;;
  gdn2-MLP)
    MODEL_ARGS=(
      --level gdn2-mlp
      --dim 2176
      --depth 12
      --n_heads 30
      --expansion 1.0
      --use_conv 1
      --d_conv 4
      --gdn2_mlp_ratio 3.258732449079677
    )
    ;;
  *)
    echo "Unknown SCALEOUT_VARIANT='$SCALEOUT_VARIANT' (expected e97-MLP, e97-linear-MLP, or gdn2-MLP)" >&2
    exit 2
    ;;
esac

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)
REQUESTED_NODES=${SLURM_JOB_NUM_NODES:-$SCALEOUT_NODES}
REQUESTED_SECONDS=$(python - <<'PY'
import os
t = os.environ.get("SLURM_TIMELIMIT") or os.environ.get("SCALEOUT_WALLTIME", "04:00:00")
if "-" in t:
    d, t = t.split("-", 1)
    day_s = int(d) * 86400
else:
    day_s = 0
parts = [int(p) for p in t.split(":")]
if len(parts) == 1:
    sec = parts[0] * 60
elif len(parts) == 2:
    sec = parts[0] * 3600 + parts[1] * 60
else:
    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
print(day_s + sec)
PY
)
REQUESTED_NODE_HOURS=$(python - <<PY
nodes = float("${REQUESTED_NODES}")
seconds = float("${REQUESTED_SECONDS}")
print(f"{nodes * seconds / 3600.0:.6f}")
PY
)

{
  echo "=== frontier diloco scaleout env ==="
  echo "task_id=$TASK_ID"
  echo "variant=$SCALEOUT_VARIANT"
  echo "start_utc=$START_UTC"
  echo "repo=$REPO"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "resume_checkpoint=${RESUME_CHECKPOINT:-}"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo
  echo "=== slurm ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|MASTER|OMP|ROCR|HIP|HSA|GDN2|NDM|DATA|VAL_DATA|TRAIN_MINUTES|SCALEOUT|DILOCO|TIKTOKEN|HUMAN_APPROVAL|RESUME_CHECKPOINT)=' || true
  echo
  echo "=== modules ==="
  module list 2>&1
  echo
  echo "=== rocm-smi ==="
  rocm-smi || true
  echo
  echo "=== python ==="
  python - <<'PY'
import sys
print(f"python={sys.executable}")
for mod in ("torch", "triton", "schedulefree", "tiktoken", "numpy"):
    try:
        m = __import__(mod)
        print(f"{mod}={getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{mod}_import_error={exc!r}")
try:
    import torch
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception as exc:
    print(f"torch_probe_error={exc!r}")
PY
  echo
  echo "=== git ==="
  git rev-parse --show-toplevel
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git status --short
  git diff --stat
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
    "job_name": "${SLURM_JOB_NAME:-emender-diloco-scaleout}",
    "account": "${ACCOUNT}",
    "partition": "${SLURM_JOB_PARTITION:-batch}",
    "qos": "${SLURM_JOB_QOS:-}",
    "nodes": int("${REQUESTED_NODES}"),
    "requested_walltime": "${SLURM_TIMELIMIT:-${SCALEOUT_WALLTIME}}",
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "human_approval_record": "${HUMAN_APPROVAL_RECORD}",
    "resume_checkpoint": "${RESUME_CHECKPOINT:-}",
    "diloco_outer_optimizer": "${DILOCO_OUTER_OPTIMIZER}",
    "diloco_outer_lr": float("${DILOCO_OUTER_LR}"),
    "diloco_outer_beta": float("${DILOCO_OUTER_BETA}"),
    "diloco_export_basis": "${DILOCO_EXPORT_BASIS}",
    "diloco_merge_topology": "${DILOCO_MERGE_TOPOLOGY}",
    "diloco_merge_group_size": int("${DILOCO_MERGE_GROUP_SIZE}"),
    "diloco_merge_bucket_numel": int("${DILOCO_MERGE_BUCKET_NUMEL}"),
    **({"diloco_bootstrap_outer_state": "${DILOCO_BOOTSTRAP_OUTER_STATE:-}"} if "${DILOCO_BOOTSTRAP_OUTER_STATE:-}" else {}),
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
    fh.write("\\n")
PY

python - "$LOG_DIR/train.log" \
  "$TASK_ID" \
  "$SCALEOUT_VARIANT" \
  "${SLURM_JOB_ID:-manual}" \
  "$REQUESTED_NODES" \
  "$REQUESTED_NODE_HOURS" \
  "$HUMAN_APPROVAL_RECORD" \
  "$TRAIN_STATUS" <<'PY' > "${SUMMARY_FILE}"
import sys
from pathlib import Path

(
    train_log_arg,
    task_id,
    scaleout_variant,
    job_id,
    requested_nodes,
    requested_node_hours,
    human_approval_record,
    train_status,
) = sys.argv[1:]

train_log = Path(train_log_arg)
lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines() if train_log.exists() else []
step_lines = [ln for ln in lines if "loss" in ln and ("tok/s" in ln or "global_tok/s" in ln)]
diloco_lines = [ln for ln in lines if "DiLoCo" in ln or "DILOCO_" in ln]
error_lines = [
    ln for ln in lines
    if any(k in ln.lower() for k in ("error", "traceback", "exception", "non-finite", "nan", "nccl", "rccl", "timeout", "oom"))
]

print("# Frontier DiLoCo Scaleout Summary")
print()
print(f"Task: `{task_id}`")
print(f"Variant: `{scaleout_variant}`")
print(f"Job ID: `{job_id}`")
print(f"Nodes: `{requested_nodes}`")
print(f"Requested node-hours: `{requested_node_hours}`")
print(f"Human approval record: `{human_approval_record}`")
print(f"Train status: `{train_status}`")
print()
print("## Last Training Metric Lines")
print()
if step_lines:
    for ln in step_lines[-20:]:
        print(f"- `{ln}`")
else:
    print("- No training loss/throughput line found.")
print()
print("## DiLoCo Lines")
print()
if diloco_lines:
    for ln in diloco_lines[-20:]:
        print(f"- `{ln}`")
else:
    print("- No DiLoCo line found.")
print()
print("## Error Lines")
print()
if error_lines:
    for ln in error_lines[-40:]:
        print(f"- `{ln}`")
else:
    print("- No obvious error line found by summary parser.")
PY

exit "$TRAIN_STATUS"
