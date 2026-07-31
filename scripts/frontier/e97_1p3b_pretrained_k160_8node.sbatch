#!/bin/bash
# Bounded 8-node E97 1.3B pretrained continuation probe for K160.
#
# Uses the validated staged checkpoint bundle and architecture args from
# validate-e97-1-3b. This wrapper intentionally changes only the scale/cadence
# envelope from the pretrained canary launcher: 8 nodes, K160, save_every 160,
# train_minutes about 50, singleton GPU islands, and avg outer.

#SBATCH -A bif148
#SBATCH -J e97-1p3b-k160-8n
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

export WG_TASK_ID=${WG_TASK_ID:-run-e97-1-3b-2}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_pretrained_k160_8node}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt}
export TRAIN_MINUTES=${TRAIN_MINUTES:-50}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_K=${DILOCO_K:-160}
export DILOCO_OUTER_OPTIMIZER=${DILOCO_OUTER_OPTIMIZER:-avg}
export DILOCO_OUTER_LR=${DILOCO_OUTER_LR:-1.0}
export DILOCO_OUTER_BETA=${DILOCO_OUTER_BETA:-0.0}
export DILOCO_EXPORT_BASIS=${DILOCO_EXPORT_BASIS:-x}
export BATCH_SIZE=${BATCH_SIZE:-1}
export CHUNK_SIZE=${CHUNK_SIZE:-2048}
export SAVE_EVERY=${SAVE_EVERY:-160}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-4}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-3600}
export REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-8.0}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:00:00}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG task validate-e97-1-3b explicitly authorized bounded 8-node pretrained GPU-island/no-DDP cadence probes K40/K160/K320; this task submits only the K160 8-node avg-outer probe.}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
exec bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
