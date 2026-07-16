#!/bin/bash
# True resilient E97 two-node gate: 2 CPU managers + 16 real GPU trainers.
# Managers and trainers are separate srun steps; --overlap and --no-kill keep a
# failed step from cancelling healthy siblings. There are no sentinel ranks;
# every GPU step is a real trainer. This file is debug-only.
#SBATCH -A bif148
#SBATCH -J resilient-e97-true-2n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/frontier/trainpy_async_quorum/%x-%j.out
#SBATCH -e logs/frontier/trainpy_async_quorum/%x-%j.err

set -euo pipefail

[[ ${SLURM_JOB_NUM_NODES:?} == 2 ]] || { echo "requires exactly two nodes" >&2; exit 64; }
[[ ${SLURM_JOB_QOS:?} == debug ]] || { echo "requires debug QoS" >&2; exit 64; }
[[ ${SLURM_TIMELIMIT:-02:00:00} == 02:00:00 || ${SLURM_TIMELIMIT:-} == 120 ]] || {
  echo "requires exactly 02:00:00" >&2; exit 64;
}
[[ ${ASYNC_LOCAL_STEPS:-40} == 40 ]] || { echo "E97 gate requires local_steps=40" >&2; exit 64; }
: "${RESILIENT_E97_MANAGER_COMMAND:?set CPU-only manager command}"
: "${RESILIENT_E97_TRAINER_COMMAND:?set approved E97 model/data/optimizer trainer command}"

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_DIR=${RUN_DIR:?set immutable run directory}
printf 'topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none\n'
exec python3 "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"
