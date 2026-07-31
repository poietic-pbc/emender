#!/bin/bash
# Recommended-env RCCL diagnostic with Frontier alternative rendezvous enabled.

#SBATCH -A bif148
#SBATCH -J rccl-allreduce-alt-rdzv
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --network=disable_rdzv_get
#SBATCH -o logs/frontier/rccl_diag/%x-%j.out
#SBATCH -e logs/frontier/rccl_diag/%x-%j.err

export RCCL_DIAG_ENV_MODE=${RCCL_DIAG_ENV_MODE:-recommended}
export RCCL_DIAG_ALT_RDZV=1

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
exec bash "${REPO}/scripts/frontier/rccl_allreduce_diag.sbatch"
