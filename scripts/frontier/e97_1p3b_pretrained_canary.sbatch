#!/bin/bash
# One-node resume canary for the staged E97 1.3B pretrained continuation bundle.
#
# This wrapper intentionally uses the staged args.json architecture instead of
# the older generic e97-MLP template shape, while keeping the cheap canary
# envelope at one node, singleton GPU islands, avg outer, and K-aligned saves.

#SBATCH -A bif148
#SBATCH -J e97-1p3b-pretrained-canary
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-validate-e97-1-3b}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_pretrained_canary}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
DEFAULT_RESUME_CHECKPOINT=${DEFAULT_RESUME_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt}
CHAIN_LATEST_PATH=${CHAIN_LATEST_PATH:-}
CHAIN_SEED_CHECKPOINT=${CHAIN_SEED_CHECKPOINT:-$DEFAULT_RESUME_CHECKPOINT}
CHAIN_MANIFEST_PATH=${CHAIN_MANIFEST_PATH:-}
CHAIN_UPDATE_ON_FAILURE=${CHAIN_UPDATE_ON_FAILURE:-1}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
TRAIN_MINUTES=${TRAIN_MINUTES:-10}
DILOCO_K=${DILOCO_K:-40}
DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-global}
DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-8}
DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY:-8}
DILOCO_MERGE_COMPLETION_BARRIER=${DILOCO_MERGE_COMPLETION_BARRIER:-1}
DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-0}
DILOCO_MERGE_DEBUG=${DILOCO_MERGE_DEBUG:-0}
DILOCO_MERGE_DEBUG_RANKS=${DILOCO_MERGE_DEBUG_RANKS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
SAVE_EVERY=${SAVE_EVERY:-40}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-4}
LOG_EVERY=${LOG_EVERY:-5}
VAL_EVERY=${VAL_EVERY:-10000}
COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-600}
WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-1}
DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-1}
DISTRIBUTED_INIT_TIMEOUT_SECONDS=${DISTRIBUTED_INIT_TIMEOUT_SECONDS:-${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-}}
DISABLE_SCALAR_STATUS_COLLECTIVES=${DISABLE_SCALAR_STATUS_COLLECTIVES:-0}
STATUS_REQUEST_POLL_SECONDS=${STATUS_REQUEST_POLL_SECONDS:-1.0}
HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG task validate-e97-1-3b: one bounded 1-node avg E97 1.3B pretrained resume canary only; no 8+ node scaleout authorized by this submission.}
FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-current}
FRONTIER_PER_RANK_TRITON_CACHE=${FRONTIER_PER_RANK_TRITON_CACHE:-0}
FRONTIER_TRAIN_ENV_PREFLIGHT=${FRONTIER_TRAIN_ENV_PREFLIGHT:-0}
REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-0}

[[ -r "$DATA" ]] || { echo "DATA is not readable: $DATA" >&2; exit 4; }
[[ -r "$VAL_DATA" ]] || { echo "VAL_DATA is not readable: $VAL_DATA" >&2; exit 4; }
if [[ -z "$RESUME_CHECKPOINT" ]]; then
  if [[ -n "$CHAIN_LATEST_PATH" && -e "$CHAIN_LATEST_PATH" ]]; then
    RESUME_CHECKPOINT=$(readlink -f "$CHAIN_LATEST_PATH")
  elif [[ -n "$CHAIN_SEED_CHECKPOINT" ]]; then
    RESUME_CHECKPOINT="$CHAIN_SEED_CHECKPOINT"
  else
    RESUME_CHECKPOINT="$DEFAULT_RESUME_CHECKPOINT"
  fi
fi
if [[ -n "$CHAIN_LATEST_PATH" && -z "$CHAIN_MANIFEST_PATH" ]]; then
  CHAIN_MANIFEST_PATH="${CHAIN_LATEST_PATH}.manifest.json"
fi
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
RANK_START_LOG="${ARTIFACT_DIR}/rank-start.tsv"
STDOUT_PATH="logs/frontier/scaleout/${SLURM_JOB_NAME:-e97-1p3b-pretrained-canary}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/scaleout/${SLURM_JOB_NAME:-e97-1p3b-pretrained-canary}-${SLURM_JOB_ID:-manual}.err"
FRONTIER_TRITON_CACHE_BASE=${FRONTIER_TRITON_CACHE_BASE:-"${RUN_ROOT}/artifacts/triton-cache"}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$SUMMARY_DIR" logs/frontier/scaleout
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules

frontier_activate_emender_conda_env
TRAIN_PYTHON_BIN=$(command -v python)
export TRAIN_PYTHON_BIN
frontier_assert_emender_conda_env

export REPO
export EMENDER_CONDA_ENV
export TIKTOKEN_CACHE_DIR
export FRONTIER_PER_RANK_TRITON_CACHE
export FRONTIER_TRAIN_ENV_PREFLIGHT
export FRONTIER_TRITON_CACHE_BASE
export RANK_START_LOG
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
frontier_derive_master_port() {
  if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
    printf '%d\n' $((20000 + (10#${SLURM_JOB_ID} % 40000)))
  else
    printf '3442\n'
  fi
}
export MASTER_PORT=${MASTER_PORT:-$(frontier_derive_master_port)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
if [[ -n "$DISTRIBUTED_INIT_TIMEOUT_SECONDS" ]]; then
  export NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS="$DISTRIBUTED_INIT_TIMEOUT_SECONDS"
fi

if [[ "$FRONTIER_RCCL_ENV" == "recommended" ]]; then
  export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-kdreg2}
  export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
  export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-2048}
  export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
  export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
  export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-net.so}
  if [[ "${FRONTIER_RCCL_ALT_RDZV:-0}" == "1" ]]; then
    export FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO:-alt_read}
  fi
  if [[ -n "${AWS_OFI_RCCL_PLUGIN_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${AWS_OFI_RCCL_PLUGIN_DIR%/}/lib:${LD_LIBRARY_PATH:-}"
  fi
else
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
fi

RCCL_NET_PLUGIN_STATUS=$(frontier_resolve_librccl_net)
if [[ "$REQUIRE_RCCL_NET_PLUGIN" == "1" && "$RCCL_NET_PLUGIN_STATUS" == "not-found" ]]; then
  echo "librccl-net.so was not found after runtime setup; refusing to start ladder training" >&2
  exit 4
fi

REQUESTED_NODES=${SLURM_JOB_NUM_NODES:-1}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-${SLURM_TIMELIMIT:-00:20:00}}
REQUESTED_SECONDS=${REQUESTED_SECONDS:-1200}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-0.333333}
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
  --walltime_final_checkpoint_margin_seconds "$WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS"
  --walltime_check_every "$WALLTIME_CHECK_EVERY"
  --distributed_health_check_every "$DISTRIBUTED_HEALTH_CHECK_EVERY"
  --output "${RUN_ROOT}/train"
  --resume "$RESUME_CHECKPOINT"
  --diloco
  --diloco_k "$DILOCO_K"
  --diloco_outer_optimizer "$DILOCO_OUTER_OPTIMIZER"
  --diloco_outer_lr "$DILOCO_OUTER_LR"
  --diloco_outer_beta "$DILOCO_OUTER_BETA"
  --diloco_export_basis "$DILOCO_EXPORT_BASIS"
  --diloco_island_size "$DILOCO_ISLAND_SIZE"
  --diloco_merge_topology "$DILOCO_MERGE_TOPOLOGY"
  --diloco_merge_group_size "$DILOCO_MERGE_GROUP_SIZE"
  --diloco_merge_group_create_barrier_every "$DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY"
  --diloco_merge_completion_barrier "$DILOCO_MERGE_COMPLETION_BARRIER"
  --diloco_merge_bucket_numel "$DILOCO_MERGE_BUCKET_NUMEL"
  --diloco_merge_debug "$DILOCO_MERGE_DEBUG"
  --diloco_merge_debug_ranks "$DILOCO_MERGE_DEBUG_RANKS"
)

if [[ -n "$DISTRIBUTED_INIT_TIMEOUT_SECONDS" ]]; then
  COMMON_ARGS+=(--distributed_init_timeout_seconds "$DISTRIBUTED_INIT_TIMEOUT_SECONDS")
fi

if [[ "$DISABLE_SCALAR_STATUS_COLLECTIVES" == "1" ]]; then
  COMMON_ARGS+=(
    --disable_scalar_status_collectives
    --status_request_poll_seconds "$STATUS_REQUEST_POLL_SECONDS"
  )
fi

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
  echo "=== e97 1.3B pretrained canary env ==="
  echo "task_id=$TASK_ID"
  echo "variant=$SCALEOUT_VARIANT"
  echo "start_utc=$START_UTC"
  echo "repo=$REPO"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "resume_checkpoint=$RESUME_CHECKPOINT"
  echo "default_resume_checkpoint=$DEFAULT_RESUME_CHECKPOINT"
  echo "chain_latest_path=$CHAIN_LATEST_PATH"
  echo "chain_seed_checkpoint=$CHAIN_SEED_CHECKPOINT"
  echo "chain_manifest_path=$CHAIN_MANIFEST_PATH"
  echo "chain_update_on_failure=$CHAIN_UPDATE_ON_FAILURE"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "diloco_k=$DILOCO_K"
  echo "diloco_island_size=$DILOCO_ISLAND_SIZE"
  echo "diloco_merge_topology=$DILOCO_MERGE_TOPOLOGY"
  echo "diloco_merge_group_size=$DILOCO_MERGE_GROUP_SIZE"
  echo "diloco_merge_group_create_barrier_every=$DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY"
  echo "diloco_merge_completion_barrier=$DILOCO_MERGE_COMPLETION_BARRIER"
  echo "diloco_merge_bucket_numel=$DILOCO_MERGE_BUCKET_NUMEL"
  echo "diloco_merge_debug=$DILOCO_MERGE_DEBUG"
  echo "diloco_merge_debug_ranks=$DILOCO_MERGE_DEBUG_RANKS"
  echo "distributed_init_timeout_seconds=${DISTRIBUTED_INIT_TIMEOUT_SECONDS:-}"
  echo "disable_scalar_status_collectives=$DISABLE_SCALAR_STATUS_COLLECTIVES"
  echo "status_request_poll_seconds=$STATUS_REQUEST_POLL_SECONDS"
  echo "frontier_rccl_env=$FRONTIER_RCCL_ENV"
  echo "rccl_net_plugin_status=$RCCL_NET_PLUGIN_STATUS"
  echo "require_rccl_net_plugin=$REQUIRE_RCCL_NET_PLUGIN"
  echo "rank_start_log=$RANK_START_LOG"
  echo "frontier_per_rank_triton_cache=$FRONTIER_PER_RANK_TRITON_CACHE"
  echo "frontier_train_env_preflight=$FRONTIER_TRAIN_ENV_PREFLIGHT"
  echo "emender_conda_env=${EMENDER_CONDA_ENV:-}"
  echo "conda_prefix=${CONDA_PREFIX:-}"
  echo "train_python_bin=${TRAIN_PYTHON_BIN:-}"
  if [[ "$FRONTIER_PER_RANK_TRITON_CACHE" == "1" ]]; then
    echo "frontier_triton_cache_base=$FRONTIER_TRITON_CACHE_BASE"
  fi
  echo
  echo "=== slurm ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|RCCL|FI_|MASTER|OMP|ROCR|HIP|HSA|LD_LIBRARY_PATH|AWS_OFI_RCCL|FRONTIER|NDM|DATA|VAL_DATA|TRAIN_MINUTES|SCALEOUT|DILOCO|TIKTOKEN|TRITON|HUMAN_APPROVAL|RESUME_CHECKPOINT|DEFAULT_RESUME_CHECKPOINT|CHAIN_|DISABLE_SCALAR|STATUS_REQUEST|EMENDER_CONDA_ENV|CONDA_PREFIX)' || true
  echo
  echo "=== command ==="
  printf '%q ' srun -N "$REQUESTED_NODES" -n "$((REQUESTED_NODES * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc \
    'source "${REPO}/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; : "${TRAIN_PYTHON_BIN:?TRAIN_PYTHON_BIN unset}"; printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_PROCID:-}" "${SLURM_LOCALID:-}" "${SLURMD_NODENAME:-$(hostname)}" "${SLURM_NTASKS:-}" >> "$RANK_START_LOG"; if [[ "${FRONTIER_PER_RANK_TRITON_CACHE:-0}" == "1" ]]; then export TRITON_CACHE_DIR="${FRONTIER_TRITON_CACHE_BASE%/}/rank-${SLURM_PROCID}"; mkdir -p "$TRITON_CACHE_DIR"; fi; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec "$TRAIN_PYTHON_BIN" -u train.py "$@"' \
    bash "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}"
  echo
} | tee "$ENV_FILE"

if [[ "$FRONTIER_TRAIN_ENV_PREFLIGHT" == "1" ]]; then
  {
    echo
    echo "=== delegated srun training env preflight ==="
    srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
      bash -lc 'source "${REPO}/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; : "${TRAIN_PYTHON_BIN:?TRAIN_PYTHON_BIN unset}"; "$TRAIN_PYTHON_BIN" - <<'"'"'PY'"'"'
import platform
import sys
print(f"delegated_python_executable={sys.executable}")
print(f"delegated_python_version={platform.python_version()}")
import torch
print(f"delegated_torch_version={torch.__version__}")
print(f"delegated_torch_hip={getattr(torch.version, '"'"'hip'"'"', None)}")
import triton
print(f"delegated_triton_version={triton.__version__}")
PY
frontier_capture_runtime_env' \
      2>&1
  } | tee -a "$ENV_FILE"
fi

set +e
srun -N "$REQUESTED_NODES" -n "$((REQUESTED_NODES * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest \
  bash -lc 'source "${REPO}/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; : "${TRAIN_PYTHON_BIN:?TRAIN_PYTHON_BIN unset}"; printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_PROCID:-}" "${SLURM_LOCALID:-}" "${SLURMD_NODENAME:-$(hostname)}" "${SLURM_NTASKS:-}" >> "$RANK_START_LOG"; if [[ "${FRONTIER_PER_RANK_TRITON_CACHE:-0}" == "1" ]]; then export TRITON_CACHE_DIR="${FRONTIER_TRITON_CACHE_BASE%/}/rank-${SLURM_PROCID}"; mkdir -p "$TRITON_CACHE_DIR"; fi; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec "$TRAIN_PYTHON_BIN" -u train.py "$@"' \
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
    "job_name": "${SLURM_JOB_NAME:-e97-1p3b-pretrained-canary}",
    "nodes": int("${REQUESTED_NODES}"),
    "requested_walltime": "${REQUESTED_WALLTIME}",
    "requested_seconds": ${REQUESTED_SECONDS},
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "human_approval_record": "${HUMAN_APPROVAL_RECORD}",
    "resume_checkpoint": "${RESUME_CHECKPOINT}",
    "default_resume_checkpoint": "${DEFAULT_RESUME_CHECKPOINT}",
    "chain_latest_path": "${CHAIN_LATEST_PATH}",
    "chain_seed_checkpoint": "${CHAIN_SEED_CHECKPOINT}",
    "chain_manifest_path": "${CHAIN_MANIFEST_PATH}",
    "chain_update_on_failure": int("${CHAIN_UPDATE_ON_FAILURE}"),
    "architecture_source": "staged args.json from E97_1.3B_diloco_20260623_103742_step260500",
    "diloco_k": int("${DILOCO_K}"),
    "diloco_island_size": int("${DILOCO_ISLAND_SIZE}"),
    "diloco_outer_optimizer": "${DILOCO_OUTER_OPTIMIZER}",
    "diloco_outer_lr": float("${DILOCO_OUTER_LR}"),
    "diloco_outer_beta": float("${DILOCO_OUTER_BETA}"),
    "diloco_export_basis": "${DILOCO_EXPORT_BASIS}",
    "diloco_merge_topology": "${DILOCO_MERGE_TOPOLOGY}",
    "diloco_merge_group_size": int("${DILOCO_MERGE_GROUP_SIZE}"),
    "diloco_merge_group_create_barrier_every": int("${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY}"),
    "diloco_merge_completion_barrier": int("${DILOCO_MERGE_COMPLETION_BARRIER}"),
    "diloco_merge_bucket_numel": int("${DILOCO_MERGE_BUCKET_NUMEL}"),
    "diloco_merge_debug": int("${DILOCO_MERGE_DEBUG}"),
    "diloco_merge_debug_ranks": "${DILOCO_MERGE_DEBUG_RANKS}",
    "distributed_init_timeout_seconds": "${DISTRIBUTED_INIT_TIMEOUT_SECONDS}",
    "disable_scalar_status_collectives": int("${DISABLE_SCALAR_STATUS_COLLECTIVES}"),
    "status_request_poll_seconds": float("${STATUS_REQUEST_POLL_SECONDS}"),
    "require_rccl_net_plugin": int("${REQUIRE_RCCL_NET_PLUGIN}"),
    "rccl_net_plugin_status": "${RCCL_NET_PLUGIN_STATUS}",
    "frontier_per_rank_triton_cache": int("${FRONTIER_PER_RANK_TRITON_CACHE}"),
    "frontier_triton_cache_base": "${FRONTIER_TRITON_CACHE_BASE}",
    "rank_start_log": "${RANK_START_LOG}",
    "batch_size": int("${BATCH_SIZE}"),
    "chunk_size": int("${CHUNK_SIZE}"),
    "save_every": int("${SAVE_EVERY}"),
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
    out.write("# E97 1.3B Pretrained Resume Canary Summary\n\n")
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

if [[ -n "$CHAIN_LATEST_PATH" ]]; then
  RUN_LATEST=$(find "${RUN_ROOT}/train" -mindepth 2 -maxdepth 2 -name latest.pt -print 2>/dev/null | sort | tail -n 1 || true)
  RESOLVED_RUN_LATEST=""
  if [[ -n "$RUN_LATEST" ]]; then
    RESOLVED_RUN_LATEST=$(readlink -f "$RUN_LATEST" || true)
  fi

  if [[ -n "$RESOLVED_RUN_LATEST" && -r "$RESOLVED_RUN_LATEST" ]]; then
    if [[ "$TRAIN_STATUS" == "0" || "$CHAIN_UPDATE_ON_FAILURE" == "1" ]]; then
      mkdir -p "$(dirname "$CHAIN_LATEST_PATH")"
      TMP_CHAIN_LINK="${CHAIN_LATEST_PATH}.${SLURM_JOB_ID:-manual}.$$.tmp"
      ln -sfn "$RESOLVED_RUN_LATEST" "$TMP_CHAIN_LINK"
      mv -Tf "$TMP_CHAIN_LINK" "$CHAIN_LATEST_PATH"
      python - <<PY
import json
from pathlib import Path

manifest_path = Path("${CHAIN_MANIFEST_PATH}")
manifest_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "updated_at_utc": "${END_UTC}",
    "train_status": int("${TRAIN_STATUS}"),
    "job_id": "${SLURM_JOB_ID:-manual}",
    "job_name": "${SLURM_JOB_NAME:-e97-1p3b-pretrained-canary}",
    "variant": "${SCALEOUT_VARIANT}",
    "input_checkpoint": "${RESUME_CHECKPOINT}",
    "latest_symlink": "${CHAIN_LATEST_PATH}",
    "latest_checkpoint": "${RESOLVED_RUN_LATEST}",
    "run_latest": "${RUN_LATEST}",
    "run_root": "${RUN_ROOT}",
    "manifest": "${MANIFEST_FILE}",
    "summary": "${SUMMARY_FILE}",
    "git_commit": "${GIT_COMMIT}",
}
with manifest_path.open("w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
      echo "[chain] advanced ${CHAIN_LATEST_PATH} -> ${RESOLVED_RUN_LATEST}"
    else
      echo "[chain] not advancing ${CHAIN_LATEST_PATH}: train_status=${TRAIN_STATUS}, CHAIN_UPDATE_ON_FAILURE=${CHAIN_UPDATE_ON_FAILURE}"
    fi
  else
    echo "[chain] no readable run latest.pt found under ${RUN_ROOT}/train; chain pointer unchanged"
  fi
fi

exit "$TRAIN_STATUS"
