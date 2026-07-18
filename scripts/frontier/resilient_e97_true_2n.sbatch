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
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/frontier/trainpy_async_quorum/%x-%j.out
#SBATCH -e logs/frontier/trainpy_async_quorum/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
APPROVED_EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
: "${EMENDER_CONDA_ENV:?immutable submit payload must export EMENDER_CONDA_ENV=$APPROVED_EMENDER_CONDA_ENV}"
[[ $(realpath "$EMENDER_CONDA_ENV") == $(realpath "$APPROVED_EMENDER_CONDA_ENV") ]] || {
  echo "EMENDER_CONDA_ENV must resolve to $APPROVED_EMENDER_CONDA_ENV" >&2; exit 64;
}
export EMENDER_CONDA_ENV
source "$REPO/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env
frontier_assert_emender_conda_env
TRAIN_PYTHON_BIN=$(command -v python || command -v python3)
: "${TRAIN_PYTHON_BIN:?Frontier runtime activation did not select TRAIN_PYTHON_BIN}"
[[ $(realpath "$TRAIN_PYTHON_BIN") == $(realpath "$APPROVED_EMENDER_CONDA_ENV/bin/python") ]] || {
  echo "activated Python is not the approved immutable runtime" >&2; exit 64;
}
RUNTIME_IDENTITY=$($TRAIN_PYTHON_BIN - <<'PY'
import json, platform, sys
import torch, triton
identity = {
    "executable": sys.executable,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "triton": triton.__version__,
}
expected = {
    "python": "3.12.13", "torch": "2.10.0+rocm7.1",
    "torch_hip": "7.1.25424", "triton": "3.6.0",
}
if any(identity[key] != value for key, value in expected.items()):
    raise SystemExit("runtime identity mismatch: " + json.dumps(identity, sort_keys=True))
print(json.dumps(identity, sort_keys=True))
PY
)
printf 'resilient_e97_runtime_identity=%s\n' "$RUNTIME_IDENTITY"
mkdir -p "${RUN_DIR:?set immutable run directory}"
printf '%s\n' "$RUNTIME_IDENTITY" > "$RUN_DIR/runtime-identity.json"

[[ ${SLURM_JOB_NUM_NODES:?} == 2 ]] || { echo "requires exactly two nodes" >&2; exit 64; }
[[ ${SLURM_JOB_QOS:?} == debug ]] || { echo "requires debug QoS" >&2; exit 64; }
if [[ ${RESILIENT_E97_STARTUP_SMOKE:-0} == 1 ]]; then
  # Frontier does not consistently export SLURM_TIMELIMIT into batch jobs.
  # The submitter must therefore bind the requested value into the immutable
  # payload; when Slurm does expose its value, require the two to agree.
  [[ ${RESILIENT_E97_REQUESTED_WALLTIME:-} == 00:50:00 || ${RESILIENT_E97_REQUESTED_WALLTIME:-} == 02:00:00 ]] || {
    echo "startup smoke requires exactly 00:50:00 or 02:00:00" >&2; exit 64;
  }
  requested_minutes=50
  [[ ${RESILIENT_E97_REQUESTED_WALLTIME} == 02:00:00 ]] && requested_minutes=120
  [[ -z ${SLURM_TIMELIMIT:-} || ${SLURM_TIMELIMIT} == ${RESILIENT_E97_REQUESTED_WALLTIME} || ${SLURM_TIMELIMIT} == ${requested_minutes} ]] || {
    echo "startup smoke Slurm walltime disagrees with requested ${RESILIENT_E97_REQUESTED_WALLTIME}" >&2; exit 64;
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
RUN_DIR=${RUN_DIR:?set immutable run directory}
: "${RESILIENT_E97_RUN_ID:?set fenced run identity}"
: "${RESILIENT_E97_SOURCE_ID:?set verified seed/source identity}"
: "${RESILIENT_E97_PAYLOAD_ID:?set tensor layout identity}"
: "${RESILIENT_E97_SEED:?set verified generation-9 seed checkpoint}"
: "${RESILIENT_E97_TRAIN_ARGS_JSON:?set approved E97 training arguments JSON}"
: "${RESILIENT_E97_DATA:?set approved token dataset}"
: "${RESILIENT_E97_TIKTOKEN_CACHE_FILE:?set verified p50k tokenizer cache file}"
RESILIENT_E97_TIKTOKEN_SHA256=${RESILIENT_E97_TIKTOKEN_SHA256:-94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069}
[[ $(sha256sum "$RESILIENT_E97_TIKTOKEN_CACHE_FILE" | awk '{print $1}') == "$RESILIENT_E97_TIKTOKEN_SHA256" ]] || {
  echo "p50k tokenizer cache checksum mismatch" >&2
  exit 64
}
APPROVED_TRAIN_ARGS="$REPO/configs/frontier/e97_resilient_split_role_flat.json"
[[ $(realpath "$RESILIENT_E97_TRAIN_ARGS_JSON") == $(realpath "$APPROVED_TRAIN_ARGS") ]] || {
  echo "training arguments must be the approved flat E97 split-role configuration: $APPROVED_TRAIN_ARGS" >&2
  exit 64
}
mapfile -t ALLOCATION_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#ALLOCATION_NODES[@]} == 2 ]] || { echo "allocation discovery did not resolve two nodes" >&2; exit 64; }
RESILIENT_E97_COORDINATOR_HOST=${RESILIENT_E97_COORDINATOR_HOST:-${ALLOCATION_NODES[0]}}
RESILIENT_E97_INITIAL_GENERATION=${RESILIENT_E97_INITIAL_GENERATION:-0}
RESILIENT_E97_RESUME_HANDOFF=${RESILIENT_E97_RESUME_HANDOFF:-}
# Compute nodes cannot fetch tiktoken vocabulary blobs from the public network.
# Stage the immutable, checksum-verified p50k input once per node before roles
# start; all tokenizer reads after this point are node-local.
RESILIENT_E97_NODE_TIKTOKEN_CACHE=${RESILIENT_E97_NODE_TIKTOKEN_CACHE:-/tmp/resilient-e97-tiktoken-$RESILIENT_E97_PAYLOAD_ID}
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=0 \
  bash -c 'set -euo pipefail; mkdir -p "$1"; cp "$2" "$1/ec7223a39ce59f226a68acc30dc1af2788490e15"; echo "$3  $1/ec7223a39ce59f226a68acc30dc1af2788490e15" | sha256sum --check --status' \
  bash "$RESILIENT_E97_NODE_TIKTOKEN_CACHE" "$RESILIENT_E97_TIKTOKEN_CACHE_FILE" "$RESILIENT_E97_TIKTOKEN_SHA256"
export TIKTOKEN_CACHE_DIR=$RESILIENT_E97_NODE_TIKTOKEN_CACHE
ROLE="$REPO/scripts/frontier/resilient_e97_role.py"
COMMON="--run-dir $RUN_DIR --run-id $RESILIENT_E97_RUN_ID --generations ${RESILIENT_E97_GENERATIONS:-1} --initial-generation $RESILIENT_E97_INITIAL_GENERATION --local-steps 40 --node-count 2 --global-quorum ${RESILIENT_E97_GLOBAL_QUORUM:-2} --coordinator-host $RESILIENT_E97_COORDINATOR_HOST --coordinator-port ${RESILIENT_E97_COORDINATOR_PORT:-29571} --deadline-s ${RESILIENT_E97_GENERATION_DEADLINE_S:-900} --source-id $RESILIENT_E97_SOURCE_ID --payload-id $RESILIENT_E97_PAYLOAD_ID --code-id ${RESILIENT_E97_CODE_ID:-unknown} --coordinator-epoch ${RESILIENT_E97_COORDINATOR_EPOCH:-1} --bulk-root ${RESILIENT_E97_BULK_ROOT:-/tmp/resilient-e97} --max-spool-bytes ${RESILIENT_E97_MAX_SPOOL_BYTES:-34359738368} --bulk-chunk-bytes ${RESILIENT_E97_BULK_CHUNK_BYTES:-1048576}"
export RESILIENT_E97_MANAGER_COMMAND="$TRAIN_PYTHON_BIN $ROLE manager $COMMON --local-quorum 6"
RESUME_ARGS=()
[[ -z $RESILIENT_E97_RESUME_HANDOFF ]] || RESUME_ARGS=(--resume-handoff "$RESILIENT_E97_RESUME_HANDOFF")
RESUME_COMMAND=
if ((${#RESUME_ARGS[@]})); then
  printf -v RESUME_COMMAND ' %q' "${RESUME_ARGS[@]}"
fi
export RESILIENT_E97_TRAINER_COMMAND="$TRAIN_PYTHON_BIN $ROLE trainer $COMMON --seed $RESILIENT_E97_SEED --train-args-json $RESILIENT_E97_TRAIN_ARGS_JSON --data $RESILIENT_E97_DATA --migration-policy initialize-from-approved-config$RESUME_COMMAND"
printf 'topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none\n'
exec "$TRAIN_PYTHON_BIN" "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"
