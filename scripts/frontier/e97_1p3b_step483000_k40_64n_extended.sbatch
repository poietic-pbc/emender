#!/bin/bash
# Foundation-generation 64-node E97 1.3B continuation from the verified
# step-483000 checkpoint. This wrapper pins the agreed K40 hierarchical
# GPU-island/no-DDP recipe and delegates launch/manifest logic to the
# established E97 1.3B pretrained canary runner.
#
# Submit only after Slurm accepts the 64-node 24h extended estimate:
#
#   sbatch -p extended -N 64 -t 24:00:00 --network=disable_rdzv_get \
#     scripts/frontier/e97_1p3b_step483000_k40_64n_extended.sbatch

#SBATCH -A bif148
#SBATCH -J e97-s483-k40-hg4-64n24h
#SBATCH -p extended
#SBATCH -N 64
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

export WG_TASK_ID=${WG_TASK_ID:-submit-64n-k40}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step483000_k40_64n_hier_g4_bucket64m_avg_24h}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt}
export TRAIN_MINUTES=${TRAIN_MINUTES:-1410}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_K=${DILOCO_K:-40}
export DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
export DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
export DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
export DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
export DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-hierarchical}
export DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-4}
export DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY:-8}
export DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-67108864}
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_RCCL_ALT_RDZV=${FRONTIER_RCCL_ALT_RDZV:-1}
export BATCH_SIZE=${BATCH_SIZE:-1}
export CHUNK_SIZE=${CHUNK_SIZE:-2048}
export LOG_EVERY=${LOG_EVERY:-25}
export VAL_EVERY=${VAL_EVERY:-10000}
export SAVE_EVERY=${SAVE_EVERY:-160}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-24}
export COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-1800}
export WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-40}
export DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-40}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-86400}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-24:00:00}
export REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-1536.0}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG submit-64n-k40: user requested 64-node 24h extended foundation-generation run from verified step483000 checkpoint with K40 hierarchical g4 bucket64m avg GPU-island/no-DDP recipe.}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
exec bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
