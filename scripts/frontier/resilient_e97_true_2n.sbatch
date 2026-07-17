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
if [[ ${RESILIENT_E97_STARTUP_SMOKE:-0} == 1 ]]; then
  [[ ${SLURM_TIMELIMIT:-} == 00:20:00 || ${SLURM_TIMELIMIT:-} == 20 ]] || {
    echo "startup smoke requires exactly 00:20:00" >&2; exit 64;
  }
  [[ ${RESILIENT_E97_GENERATIONS:-1} == 1 ]] || {
    echo "startup smoke requires exactly one finalized generation" >&2; exit 64;
  }
  [[ -z ${RESILIENT_E97_INJECT_TRAINER:-}${RESILIENT_E97_INJECT_MANAGER:-}${RESILIENT_E97_INJECT_NODE_STEP:-} ]] || {
    echo "startup smoke forbids failure injection" >&2; exit 64;
  }
else
  [[ ${SLURM_TIMELIMIT:-02:00:00} == 02:00:00 || ${SLURM_TIMELIMIT:-} == 120 ]] || {
    echo "requires exactly 02:00:00" >&2; exit 64;
  }
fi
[[ ${ASYNC_LOCAL_STEPS:-40} == 40 ]] || { echo "E97 gate requires local_steps=40" >&2; exit 64; }
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_DIR=${RUN_DIR:?set immutable run directory}
: "${RESILIENT_E97_RUN_ID:?set fenced run identity}"
: "${RESILIENT_E97_SOURCE_ID:?set verified seed/source identity}"
: "${RESILIENT_E97_PAYLOAD_ID:?set tensor layout identity}"
: "${RESILIENT_E97_SEED:?set verified generation-9 seed checkpoint}"
: "${RESILIENT_E97_TRAIN_ARGS_JSON:?set approved E97 training arguments JSON}"
: "${RESILIENT_E97_DATA:?set approved token dataset}"
mapfile -t ALLOCATION_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#ALLOCATION_NODES[@]} == 2 ]] || { echo "allocation discovery did not resolve two nodes" >&2; exit 64; }
RESILIENT_E97_COORDINATOR_HOST=${RESILIENT_E97_COORDINATOR_HOST:-${ALLOCATION_NODES[0]}}
RESILIENT_E97_INITIAL_GENERATION=${RESILIENT_E97_INITIAL_GENERATION:-0}
RESILIENT_E97_RESUME_HANDOFF=${RESILIENT_E97_RESUME_HANDOFF:-}
ROLE="$REPO/scripts/frontier/resilient_e97_role.py"
COMMON="--run-dir $RUN_DIR --run-id $RESILIENT_E97_RUN_ID --generations ${RESILIENT_E97_GENERATIONS:-1} --initial-generation $RESILIENT_E97_INITIAL_GENERATION --local-steps 40 --node-count 2 --global-quorum ${RESILIENT_E97_GLOBAL_QUORUM:-2} --coordinator-host $RESILIENT_E97_COORDINATOR_HOST --coordinator-port ${RESILIENT_E97_COORDINATOR_PORT:-29571} --deadline-s ${RESILIENT_E97_GENERATION_DEADLINE_S:-900} --source-id $RESILIENT_E97_SOURCE_ID --payload-id $RESILIENT_E97_PAYLOAD_ID --code-id ${RESILIENT_E97_CODE_ID:-unknown} --coordinator-epoch ${RESILIENT_E97_COORDINATOR_EPOCH:-1} --bulk-root ${RESILIENT_E97_BULK_ROOT:-/tmp/resilient-e97} --max-spool-bytes ${RESILIENT_E97_MAX_SPOOL_BYTES:-34359738368} --bulk-chunk-bytes ${RESILIENT_E97_BULK_CHUNK_BYTES:-1048576}"
export RESILIENT_E97_MANAGER_COMMAND="python3 $ROLE manager $COMMON --local-quorum 6"
RESUME_ARGS=()
[[ -z $RESILIENT_E97_RESUME_HANDOFF ]] || RESUME_ARGS=(--resume-handoff "$RESILIENT_E97_RESUME_HANDOFF")
RESUME_COMMAND=
if ((${#RESUME_ARGS[@]})); then
  printf -v RESUME_COMMAND ' %q' "${RESUME_ARGS[@]}"
fi
export RESILIENT_E97_TRAINER_COMMAND="python3 $ROLE trainer $COMMON --seed $RESILIENT_E97_SEED --train-args-json $RESILIENT_E97_TRAIN_ARGS_JSON --data $RESILIENT_E97_DATA --migration-policy initialize-from-approved-config$RESUME_COMMAND"
printf 'topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none\n'
exec python3 "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"
