#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-ep-roundtrip
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"
mkdir -p logs/frontier/e97_moe

echo "=== scheduler-live ==="
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N'
scontrol show job -dd "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(JobId|Partition|QOS|JobState|NodeList)=' || true
echo "commit=$(git rev-parse HEAD)"

source scripts/frontier/activate_emender_frontier.sh
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((24000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=WARN

srun --label --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  cd "'"$REPO"'"
  source scripts/frontier/activate_emender_frontier.sh >/dev/null
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-moe-ep-${SLURM_JOB_ID}-${SLURM_PROCID}"
  rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/frontier/e97_moe_ep_roundtrip.py
'
