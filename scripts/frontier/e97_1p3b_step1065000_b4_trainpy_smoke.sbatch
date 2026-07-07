#!/bin/bash
# Bounded train.py-only E97 1.3B B4 smoke ladder from the refreshed
# step-1065000 seed. Submit directly for 1 node, or override -N/exports for
# the 2-node and 8-node debug-QOS ladder arms after the smaller arm is clean.

#SBATCH -A bif148
#SBATCH -J e97-s1065-b4-k40-trainpy-1n
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

export WG_TASK_ID=${WG_TASK_ID:-run-train-py}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1065000_b4_k40_trainpy_smoke}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt}
export DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
export VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
export TRAIN_MINUTES=${TRAIN_MINUTES:-10}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_K=${DILOCO_K:-40}
export DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
export DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
export DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
export DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
export DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-global}
export DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-8}
export DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY:-8}
export DILOCO_MERGE_COMPLETION_BARRIER=${DILOCO_MERGE_COMPLETION_BARRIER:-1}
export DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-0}
export DILOCO_MERGE_DEBUG=${DILOCO_MERGE_DEBUG:-1}
export DILOCO_MERGE_DEBUG_RANKS=${DILOCO_MERGE_DEBUG_RANKS:-0}
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_RCCL_ALT_RDZV=${FRONTIER_RCCL_ALT_RDZV:-1}
export FRONTIER_PER_RANK_TRITON_CACHE=${FRONTIER_PER_RANK_TRITON_CACHE:-1}
export FRONTIER_TRAIN_ENV_PREFLIGHT=${FRONTIER_TRAIN_ENV_PREFLIGHT:-1}
export BATCH_SIZE=${BATCH_SIZE:-4}
export CHUNK_SIZE=${CHUNK_SIZE:-2048}
export LOG_EVERY=${LOG_EVERY:-1}
export VAL_EVERY=${VAL_EVERY:-10000}
export SAVE_EVERY=${SAVE_EVERY:-40}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-3}
export COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-600}
export WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-10}
export DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-10}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-1200}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-00:20:00}
export REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-0.333333}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG run-train-py: bounded debug-QOS train.py smoke only; refreshed E97 step1065000 latest.pt; no async/quorum harness; no 64n/256n/production approval.}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
DEFAULT_ENV_PREFIX="${REPO}/.envs/olcf-rocm711-torch210-py312"
if [[ ! -d "$DEFAULT_ENV_PREFIX" && -d /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 ]]; then
  DEFAULT_ENV_PREFIX=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
fi
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-$DEFAULT_ENV_PREFIX}
exec bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
