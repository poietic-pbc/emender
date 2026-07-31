#!/bin/bash
# Reusable E97 1.3B pretrained K160 GPU-island scale ladder launcher.
#
# This wrapper keeps the validated pretrained architecture and delegates the
# actual train.py invocation to e97_1p3b_pretrained_canary.sbatch. Submit with
# command-line Slurm overrides for each rung, for example:
#
#   sbatch -N 16 -J e97-1p3b-k160-16n \
#     --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-16n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_16n,REQUESTED_NODE_HOURS=16.0 \
#     scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
#
# Command-line -N/-J/-t options override the defaults below. Keep
# DILOCO_ISLAND_SIZE=1 for no-DDP singleton GPU islands.

#SBATCH -A bif148
#SBATCH -J e97-1p3b-k160-scale
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 16
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

export WG_TASK_ID=${WG_TASK_ID:-run-e97-1p3b-k160-scale}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_pretrained_k160_${SLURM_JOB_NUM_NODES:-16}n}
export ENV_PREFIX=${ENV_PREFIX:-"${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/.envs/olcf-rocm711-torch210-py312"}
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-$ENV_PREFIX}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/train/emender_E97_1.3B_20260625_143545/latest.pt}
export TRAIN_MINUTES=${TRAIN_MINUTES:-50}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_K=${DILOCO_K:-160}
export DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
export DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
export DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
export DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
export DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-0}
export DILOCO_MERGE_DEBUG=${DILOCO_MERGE_DEBUG:-0}
export DILOCO_MERGE_DEBUG_RANKS=${DILOCO_MERGE_DEBUG_RANKS:-0}
export BATCH_SIZE=${BATCH_SIZE:-1}
export CHUNK_SIZE=${CHUNK_SIZE:-2048}
export SAVE_EVERY=${SAVE_EVERY:-160}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-4}
export LOG_EVERY=${LOG_EVERY:-5}
export VAL_EVERY=${VAL_EVERY:-10000}
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-1200}
export WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-160}
export DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-160}
export NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-1800}
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_RCCL_ALT_RDZV=${FRONTIER_RCCL_ALT_RDZV:-1}
export FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}
export FRONTIER_RUNTIME_PROFILE=${FRONTIER_RUNTIME_PROFILE:-olcf-rccl-debug}
export FRONTIER_RCCL_NET_PLUGIN_MODULE=${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}
export REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-3600}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:00:00}

if [[ -z "${REQUESTED_NODE_HOURS:-}" ]]; then
  REQUESTED_NODE_HOURS=$(awk -v nodes="${SLURM_JOB_NUM_NODES:-16}" -v seconds="$REQUESTED_SECONDS" 'BEGIN { printf "%.6f", nodes * seconds / 3600.0 }')
  export REQUESTED_NODE_HOURS
fi

export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG scale ladder requested by user on 2026-06-25: E97 1.3B pretrained K160 avg outer, singleton GPU islands, large trailing training-loss windows as primary gate; fixed eval is secondary smoke evidence.}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
exec bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
