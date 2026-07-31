#!/bin/bash
set -euo pipefail

REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1263
RUN_ID=validate-resilient-pool-v1-2n-startup-r4-20260718T164450Z-4b9def7
RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/$RUN_ID
CODE_COMMIT=4b9def7cc9657d8648b67d1e385ea9bf161786fa

verify_sha256() {
  local expected=$1
  local path=$2
  local observed
  observed=$(sha256sum "$path")
  [[ ${observed%% *} == "$expected" ]]
}

git -C "$REPO" fetch origin
[[ $(git -C "$REPO" rev-parse origin/main) == "$CODE_COMMIT" || \
   $(git -C "$REPO" diff --name-only "$CODE_COMMIT"..origin/main -- ndm scripts/frontier configs/frontier | wc -l) == 0 ]]
git -C "$REPO" merge-base --is-ancestor ae2e6f26046fb7a6b348e845fb4615092a7c37e0 origin/main
git -C "$REPO" diff --quiet "$CODE_COMMIT"..HEAD -- ndm scripts/frontier configs/frontier
verify_sha256 6bad3f77e6a20834a13452c04de35299bd53f1d4304e4ab089a3524f3596d6ec "$REPO/scripts/frontier/resilient_e97_true_2n.sbatch"
verify_sha256 1893cdef5413945a0c49e6bc219dc0e9227eb5e510679429416851637b4c122e "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"
verify_sha256 c82025213f74ba85e6e4b1bf95f3f1a85cb0444ae921dda3fc37273366516833 "$REPO/scripts/frontier/resilient_e97_role.py"
verify_sha256 37de7776e5aee7d66e5e7b5737e6849dc5cd344cc283166beb88876a5f52264f "$REPO/ndm/resilient_pool_runtime.py"
[[ ! -e "$RUN_DIR" ]]
queue=$(squeue -u "$USER" -h -o '%i|%T|%j')
if [[ -n $queue ]]; then
  printf 'refusing submit while another user job is active or pending: %s\n' "$queue" >&2
  exit 69
fi

cd "$REPO"
exec sbatch --parsable \
  -A bif148 -p batch --qos=debug -N 2 --gpus-per-node=8 -t 00:20:00 \
  -J resilient-e97-pool-v1-startup-r4-2n --chdir="$REPO" \
  --export=ALL,REPO="$REPO",RUN_DIR="$RUN_DIR",RESILIENT_E97_RUN_ID="$RUN_ID",RESILIENT_E97_SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9,RESILIENT_E97_PAYLOAD_ID=4b9def7-20260718T164450Z-pool-v1-startup-r4-2n20m-k40-owner64m-io8mps15s,RESILIENT_E97_CODE_ID="$CODE_COMMIT",RESILIENT_E97_SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt,RESILIENT_E97_TRAIN_ARGS_JSON="$REPO/configs/frontier/e97_resilient_split_role_flat.json",RESILIENT_E97_DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,RESILIENT_E97_TIKTOKEN_CACHE_FILE=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15,RESILIENT_E97_TIKTOKEN_SHA256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069,RESILIENT_E97_GENERATIONS=1,RESILIENT_E97_INITIAL_GENERATION=0,RESILIENT_E97_COORDINATOR_EPOCH=1,RESILIENT_E97_GLOBAL_QUORUM=2,RESILIENT_E97_STARTUP_SMOKE=1,RESILIENT_E97_REQUESTED_WALLTIME=00:20:00,RESILIENT_E97_LAUNCH_MODE=node-local,RESILIENT_E97_STARTUP_DEADLINE_S=180,RESILIENT_E97_HEARTBEAT_DEADLINE_S=60,RESILIENT_E97_PROGRESS_DEADLINE_S=600,RESILIENT_E97_GENERATION_DEADLINE_S=600,RESILIENT_E97_BULK_ROOT=/tmp/resilient-e97,RESILIENT_E97_KERNEL_CACHE_ROOT=/tmp/resilient-e97-kernel-cache,RESILIENT_E97_MAX_SPOOL_BYTES=68719476736,RESILIENT_E97_BULK_CHUNK_BYTES=67108864,RESILIENT_E97_LOCAL_SPOOL_CHUNK_BYTES=67108864,RESILIENT_E97_MAX_RESTARTS=0,RESILIENT_E97_RESUME_HANDOFF=,RESILIENT_E97_INJECT_TRAINER=,RESILIENT_E97_INJECT_MANAGER=,RESILIENT_E97_INJECT_NODE_STEP=,EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 \
  scripts/frontier/resilient_e97_true_2n.sbatch
