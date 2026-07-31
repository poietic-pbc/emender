#!/bin/bash
# Two-hour chained 64-node E97 1.3B continuation.
#
# This uses the BATCH_SIZE=4, K80, hierarchical GPU-island/no-DDP recipe from
# the 64-node B4 scan, but advances a stable chain latest.pt pointer after each
# job so multiple regular-queue jobs can replace one fragile 24h allocation.
#
# Submit one job:
#
#   sbatch --network=disable_rdzv_get \
#     --export=ALL,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender \
#     scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch

#SBATCH -A bif148
#SBATCH -J e97-b4-k80-64n2h
#SBATCH -p batch
#SBATCH -N 64
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

CHAIN_ROOT=${CHAIN_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains}
CHAIN_NAME=${CHAIN_NAME:-E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg}
CHAIN_DIR="${CHAIN_ROOT%/}/${CHAIN_NAME}"

export WG_TASK_ID=${WG_TASK_ID:-chain-64n-b4-k80-2h}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-${CHAIN_NAME}_2h}
export CHAIN_LATEST_PATH=${CHAIN_LATEST_PATH:-${CHAIN_DIR}/latest.pt}
export CHAIN_MANIFEST_PATH=${CHAIN_MANIFEST_PATH:-${CHAIN_DIR}/latest.pt.manifest.json}
export CHAIN_SEED_CHECKPOINT=${CHAIN_SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt}
export CHAIN_UPDATE_ON_FAILURE=${CHAIN_UPDATE_ON_FAILURE:-1}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

export TRAIN_MINUTES=${TRAIN_MINUTES:-105}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_K=${DILOCO_K:-80}
export DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
export DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
export DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
export DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
export DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-hierarchical}
export DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-4}
export DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY:-8}
export DILOCO_MERGE_COMPLETION_BARRIER=${DILOCO_MERGE_COMPLETION_BARRIER:-1}
export DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-67108864}
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_RCCL_ALT_RDZV=${FRONTIER_RCCL_ALT_RDZV:-1}
export BATCH_SIZE=${BATCH_SIZE:-4}
export CHUNK_SIZE=${CHUNK_SIZE:-2048}
export LOG_EVERY=${LOG_EVERY:-25}
export VAL_EVERY=${VAL_EVERY:-10000}
export SAVE_EVERY=${SAVE_EVERY:-160}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-16}
export COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
export FRONTIER_PER_RANK_TRITON_CACHE=${FRONTIER_PER_RANK_TRITON_CACHE:-1}
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-900}
export WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-80}
export DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-80}
export DISABLE_SCALAR_STATUS_COLLECTIVES=${DISABLE_SCALAR_STATUS_COLLECTIVES:-1}
export STATUS_REQUEST_POLL_SECONDS=${STATUS_REQUEST_POLL_SECONDS:-1.0}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-7200}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-02:00:00}
export REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-128.0}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-64-node 2h E97 B4/K80 chained continuation from stable chain latest.pt; scalar status collectives disabled after 24h run failed on default_pg NumelIn=1 all-reduce.}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
exec bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
