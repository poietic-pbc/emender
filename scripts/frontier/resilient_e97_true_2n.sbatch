#!/bin/bash
# True resilient E97 two-node gate: 2 CPU managers + 16 real GPU trainers.
# Managers and trainers are separate srun steps; --overlap and --no-kill keep a
# failed step from cancelling healthy siblings.  This file is debug-only.
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
mkdir -p "$RUN_DIR/supervision" "$RUN_DIR/logs"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} == 2 ]] || exit 64

pids=()
for node_rank in 0 1; do
  node=${NODES[$node_rank]}
  # The manager has no GPU allocation and cannot accidentally load on a GPU.
  srun --overlap --no-kill --exact -N1 -n1 -w "$node" -c8 \
    env CUDA_VISIBLE_DEVICES= RESILIENT_E97_ROLE=manager \
      RESILIENT_E97_NODE_RANK="$node_rank" RUN_DIR="$RUN_DIR" \
      bash -lc "$RESILIENT_E97_MANAGER_COMMAND" \
      >"$RUN_DIR/logs/manager-${node_rank}.out" \
      2>"$RUN_DIR/logs/manager-${node_rank}.err" &
  pids+=("$!")
  for local_rank in 0 1 2 3 4 5 6 7; do
    # Each trainer is its own Slurm step/process group and executes the real
    # approved payload.  There are no sentinel ranks in this allocation.
    srun --overlap --no-kill --exact -N1 -n1 -w "$node" -c7 \
      --gpus-per-task=1 --gpu-bind="map_gpu:${local_rank}" \
      env RESILIENT_E97_ROLE=trainer RESILIENT_E97_NODE_RANK="$node_rank" \
        RESILIENT_E97_LOCAL_RANK="$local_rank" ASYNC_LOCAL_STEPS=40 RUN_DIR="$RUN_DIR" \
        bash -lc "$RESILIENT_E97_TRAINER_COMMAND" \
        >"$RUN_DIR/logs/trainer-${node_rank}-${local_rank}.out" \
        2>"$RUN_DIR/logs/trainer-${node_rank}-${local_rank}.err" &
    pids+=("$!")
  done
done

printf 'topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none\n'
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"

