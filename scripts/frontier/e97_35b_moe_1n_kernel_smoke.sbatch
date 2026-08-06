#!/bin/bash
# One-node/eight-GCD kernel qualification for the fail-closed E97 35B MoE path.
# This is not a production checkpoint parity or training run.

#SBATCH -A bif148
#SBATCH -J e97-35b-moe-kernel
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00
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

# Retain explicit live scheduler evidence for both required queue fields.
echo "=== scheduler-live ==="
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N'
scontrol show job -dd "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(JobId|Partition|QOS|JobState|NodeList)=' || true

echo "=== source ==="
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'branch=%s\n' "$(git branch --show-current)"

# Activate only the canonical Frontier environment. Each rank gets an isolated
# Triton cache and exactly one Slurm-bound GCD.
source scripts/frontier/activate_emender_frontier.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export NDM_PIN_TRITON_AUTOTUNE=1
export MPICH_GPU_SUPPORT_ENABLED=1

srun --label --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  cd "'"$REPO"'"
  source scripts/frontier/activate_emender_frontier.sh >/dev/null
  export TRITON_CACHE_DIR="/tmp/emender-e97-moe-${SLURM_JOB_ID}-rank-${SLURM_PROCID}"
  rm -rf "$TRITON_CACHE_DIR"
  echo "rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} host=$(hostname) rocr=${ROCR_VISIBLE_DEVICES:-unset}"
  "$EMENDER_PYTHON" -m pytest -q \
    tests/test_e97_moe_triton.py \
    tests/test_e97_moe.py \
    tests/test_e97_facade.py
'

echo "=== kernel-smoke-complete ==="
