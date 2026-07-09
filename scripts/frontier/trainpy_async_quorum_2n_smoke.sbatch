#!/bin/bash
# Two-node train.py-backed async quorum DiLoCo debug smoke.
#
# Launches 16 train.py-backed GPU ranks and configures one additional expected
# update. Quorum is 16/17, so the job must advance while recording a timed-out
# missing update path.

#SBATCH -A bif148
#SBATCH -J trainpy-async-quorum-2n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/trainpy_async_quorum/%x-%j.out
#SBATCH -e logs/frontier/trainpy_async_quorum/%x-%j.err

set -euo pipefail

export SMOKE_NAME=${SMOKE_NAME:-2n}
export SMOKE_NODE_COUNT=${SMOKE_NODE_COUNT:-${SLURM_JOB_NUM_NODES:-2}}
export WG_TASK_ID=${WG_TASK_ID:-add-train-py}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1282500_trainpy_async_quorum_2n}
export ASYNC_TRAINPY_RANKS=${ASYNC_TRAINPY_RANKS:-$((SMOKE_NODE_COUNT * 8))}
export ASYNC_EXPECTED_MISSING_UPDATES=${ASYNC_EXPECTED_MISSING_UPDATES:-1}
export ASYNC_EXPECTED_RANKS=${ASYNC_EXPECTED_RANKS:-$((ASYNC_TRAINPY_RANKS + ASYNC_EXPECTED_MISSING_UPDATES))}
export ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$ASYNC_TRAINPY_RANKS}
export ASYNC_TIMEOUT_S=${ASYNC_TIMEOUT_S:-120}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG add-train-py: 2-node train.py-native async quorum debug smoke; one train.py rank per GPU; 16/17 quorum validates missing-update recovery without killing the job.}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
exec bash "${REPO}/scripts/frontier/trainpy_async_quorum_smoke_common.sh"
