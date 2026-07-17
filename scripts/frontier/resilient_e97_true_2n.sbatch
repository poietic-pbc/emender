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
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_DIR=${RUN_DIR:?set immutable run directory}
: "${RESILIENT_E97_RUN_ID:?set fenced run identity}"
: "${RESILIENT_E97_SOURCE_ID:?set verified seed/source identity}"
: "${RESILIENT_E97_PAYLOAD_ID:?set tensor layout identity}"
: "${RESILIENT_E97_SEED:?set verified generation-9 seed checkpoint}"
: "${RESILIENT_E97_TRAIN_ARGS_JSON:?set approved E97 training arguments JSON}"
: "${RESILIENT_E97_DATA:?set approved token dataset}"
ROLE="$REPO/scripts/frontier/resilient_e97_role.py"
COMMON="--run-dir $RUN_DIR --run-id $RESILIENT_E97_RUN_ID --generations ${RESILIENT_E97_GENERATIONS:-1} --local-steps 40 --node-count 2 --global-quorum ${RESILIENT_E97_GLOBAL_QUORUM:-2} --coordinator-host ${RESILIENT_E97_COORDINATOR_HOST:?set node-0 manager host} --coordinator-port ${RESILIENT_E97_COORDINATOR_PORT:-29571} --deadline-s ${RESILIENT_E97_GENERATION_DEADLINE_S:-900} --source-id $RESILIENT_E97_SOURCE_ID --payload-id $RESILIENT_E97_PAYLOAD_ID --code-id ${RESILIENT_E97_CODE_ID:-unknown} --coordinator-epoch ${RESILIENT_E97_COORDINATOR_EPOCH:-1} --bulk-root ${RESILIENT_E97_BULK_ROOT:-/tmp/resilient-e97} --max-spool-bytes ${RESILIENT_E97_MAX_SPOOL_BYTES:-34359738368}"
export RESILIENT_E97_MANAGER_COMMAND="python3 $ROLE manager $COMMON --local-quorum 6"
export RESILIENT_E97_TRAINER_COMMAND="python3 $ROLE trainer $COMMON --seed $RESILIENT_E97_SEED --train-args-json $RESILIENT_E97_TRAIN_ARGS_JSON --data $RESILIENT_E97_DATA --migration-policy initialize-from-approved-config"
printf 'topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none\n'
exec python3 "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"
