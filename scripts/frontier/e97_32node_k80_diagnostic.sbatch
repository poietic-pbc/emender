#!/bin/bash
# Bounded E97-MLP 32-node cadence diagnostic for run-e97-32-node.
# Changes one training variable versus the K40 diagnostic: DILOCO_K=80.
# Outer optimizer remains stateless avg.

#SBATCH -A bif148
#SBATCH -J e97-32n-k80-diag
#SBATCH -p batch
#SBATCH -N 32
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/scaleout/%x-%j.out
#SBATCH -e logs/frontier/scaleout/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
export REPO
export WG_TASK_ID=run-e97-32-node
export TASK_ID=run-e97-32-node
export HUMAN_APPROVAL_RECORD="docs/FRONTIER_E97_32NODE_K80_DIAGNOSTIC_20260625.md"

export SCALEOUT_VARIANT=e97-MLP
export SCALEOUT_NODES=32
export SCALEOUT_WALLTIME=00:30:00
export TRAIN_MINUTES=20

export OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
export DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
export VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt

export DILOCO_K=80
export DILOCO_ISLAND_SIZE=8
export DILOCO_OUTER_OPTIMIZER=avg
export DILOCO_OUTER_LR=1.0
export DILOCO_OUTER_BETA=0.0
export DILOCO_EXPORT_BASIS=x

export BATCH_SIZE=1
export CHUNK_SIZE=2048
export LOG_EVERY=5
export VAL_EVERY=10000
export SAVE_EVERY=80
export KEEP_CHECKPOINTS=4
export COMPILE_WARMUP_STEPS=1

exec bash "${REPO}/scripts/frontier/diloco_scaleout_readiness.sbatch"
