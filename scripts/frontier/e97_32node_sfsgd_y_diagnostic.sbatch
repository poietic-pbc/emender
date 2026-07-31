#!/bin/bash
# Bounded from-scratch E97-MLP 32-node schedule-free sfsgd_y diagnostic.
# This is a separate scale probe, not a same-source avg-ladder rung.

#SBATCH -A bif148
#SBATCH -J e97-32n-sfsgd-y-diag
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
export WG_TASK_ID=run-e97-32-node-4
export TASK_ID=run-e97-32-node-4
export HUMAN_APPROVAL_RECORD="docs/FRONTIER_E97_SCHEDULEFREE_SCALE_SYNTHESIS_20260625.md"

export SCALEOUT_VARIANT=e97-MLP
export SCALEOUT_NODES=32
export SCALEOUT_WALLTIME=00:30:00
export TRAIN_MINUTES=20

export OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
export DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
export VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
unset RESUME_CHECKPOINT

export DILOCO_K=250
export DILOCO_ISLAND_SIZE=8
export DILOCO_OUTER_OPTIMIZER=sfsgd
export DILOCO_OUTER_LR=1.0
export DILOCO_OUTER_BETA=0.1
export DILOCO_EXPORT_BASIS=y

export BATCH_SIZE=1
export CHUNK_SIZE=2048
export LOG_EVERY=5
export VAL_EVERY=10000
export SAVE_EVERY=250
export KEEP_CHECKPOINTS=4
export COMPILE_WARMUP_STEPS=1

exec bash "${REPO}/scripts/frontier/diloco_scaleout_readiness.sbatch"
