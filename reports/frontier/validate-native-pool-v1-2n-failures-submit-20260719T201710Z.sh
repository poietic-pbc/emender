#!/bin/bash
set -euo pipefail

MODE=${1:?usage: validate-native-pool-v1-2n-failures-submit-20260719T201710Z.sh progress|owner-abort}
[[ $MODE == progress || $MODE == owner-abort ]] || {
  echo "mode must be progress or owner-abort" >&2
  exit 64
}

REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1318
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
readonly REPO
readonly SOURCE_COMMIT=f56e27a53c9985e95c4ab7d0fcfa28b8a2c513e6
readonly SOURCE_BRANCH=wg/agent-1318/validate-native-pool-v1-2n-failures
readonly BUILD_MANIFEST="$REPO/build/native-resilient-dataplane/native-artifacts.json"
readonly G2_GATE="$REPO/reports/frontier/native-dataplane/5033380/full-layout-gate.json"
readonly APPROVED_PYTHON="$EMENDER_PYTHON"
readonly APPROVED_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
readonly SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
readonly DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
readonly TIKTOKEN=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
readonly STAMP=$(date -u +%Y%m%dT%H%M%SZ)
readonly RUN_ID="npv1f-${MODE}-${STAMP}-${SOURCE_COMMIT:0:7}"
readonly RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/$RUN_ID

[[ $(git -C "$REPO" branch --show-current) == "$SOURCE_BRANCH" ]]
[[ $(git -C "$REPO" rev-parse HEAD) == "$SOURCE_COMMIT" ]]
[[ $(git -C "$REPO" rev-parse "origin/$SOURCE_BRANCH") == "$SOURCE_COMMIT" ]]
git -C "$REPO" diff --quiet --ignore-submodules --
git -C "$REPO" diff --cached --quiet --ignore-submodules --
[[ -z $(squeue -u "$USER" -h -o '%i') ]] || {
  echo "refusing to overlap another user allocation" >&2
  exit 69
}
[[ ! -e $RUN_DIR ]] || {
  echo "refusing to reuse run directory: $RUN_DIR" >&2
  exit 73
}
[[ -x $APPROVED_PYTHON && -r $SEED && -r $DATA && -r $TIKTOKEN && -r $G2_GATE ]]
[[ $(sha256sum "$SEED" | awk '{print $1}') == 1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9 ]]
[[ $(sha256sum "$TIKTOKEN" | awk '{print $1}') == 94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069 ]]
"$APPROVED_PYTHON" "$REPO/scripts/frontier/attest_native_dataplane.py" verify \
  --backend native-cxi --production --full-layout \
  --build-manifest "$BUILD_MANIFEST" --gate-json "$G2_GATE" \
  --source-root "$REPO" >/dev/null

GENERATIONS=3
MAX_RESTARTS=2
INJECT_TRAINER=
INJECT_MANAGER=
INJECT_NATIVE_SERVICE=
DELAY_READY=
if [[ $MODE == progress ]]; then
  INJECT_TRAINER=0:3:1
  INJECT_MANAGER=1:-1:1
  DELAY_READY=1:1:30
else
  MAX_RESTARTS=0
  INJECT_NATIVE_SERVICE=1:-1:1:owner_transport
fi

cd "$REPO"
exec sbatch --parsable \
  -A bif148 -p batch --qos=debug -N 2 --gpus-per-node=8 -t 00:30:00 \
  --network=job_vni \
  -J "validate-native-pool-fail-${MODE}" --chdir="$REPO" \
  --export=ALL,REPO="$REPO",RUN_DIR="$RUN_DIR",RESILIENT_E97_RUN_ID="$RUN_ID",RESILIENT_E97_SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9,RESILIENT_E97_PAYLOAD_ID="${SOURCE_COMMIT:0:7}-${STAMP}-native-pool-v1-failures-${MODE}-2n30m-k40",RESILIENT_E97_CODE_ID="$SOURCE_COMMIT",RESILIENT_E97_SEED="$SEED",RESILIENT_E97_TRAIN_ARGS_JSON="$REPO/configs/frontier/e97_resilient_split_role_flat.json",RESILIENT_E97_DATA="$DATA",RESILIENT_E97_TIKTOKEN_CACHE_FILE="$TIKTOKEN",RESILIENT_E97_TIKTOKEN_SHA256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069,RESILIENT_E97_GENERATIONS="$GENERATIONS",RESILIENT_E97_INITIAL_GENERATION=0,RESILIENT_E97_COORDINATOR_EPOCH=1,RESILIENT_E97_GLOBAL_QUORUM=2,RESILIENT_E97_STARTUP_SMOKE=0,RESILIENT_E97_REQUESTED_WALLTIME=00:30:00,RESILIENT_E97_LAUNCH_MODE=node-local,RESILIENT_E97_STARTUP_DEADLINE_S=180,RESILIENT_E97_HEARTBEAT_DEADLINE_S=60,RESILIENT_E97_PROGRESS_DEADLINE_S=600,RESILIENT_E97_GENERATION_DEADLINE_S=600,RESILIENT_E97_BULK_ROOT=/tmp/resilient-e97,RESILIENT_E97_KERNEL_CACHE_ROOT=/tmp/resilient-e97-kernel-cache,RESILIENT_E97_MAX_SPOOL_BYTES=68719476736,RESILIENT_E97_MAX_SHARED_BYTES=68719476736,RESILIENT_E97_BULK_CHUNK_BYTES=67108864,RESILIENT_E97_LOCAL_SPOOL_CHUNK_BYTES=67108864,RESILIENT_E97_MAX_RESTARTS="$MAX_RESTARTS",RESILIENT_E97_RESUME_HANDOFF=,RESILIENT_E97_INJECT_TRAINER="$INJECT_TRAINER",RESILIENT_E97_INJECT_MANAGER="$INJECT_MANAGER",RESILIENT_E97_INJECT_NODE_STEP=,RESILIENT_E97_INJECT_NATIVE_SERVICE="$INJECT_NATIVE_SERVICE",RESILIENT_E97_DELAY_READY="$DELAY_READY",DILOCO_DATAPLANE=native-cxi,FI_PROVIDER=cxi,FI_MR_CACHE_MONITOR=kdreg2,FI_CXI_ATS=0,NDP_BUILD_MANIFEST="$BUILD_MANIFEST",NDP_FULL_LAYOUT_GATE_JSON="$G2_GATE",EMENDER_CONDA_ENV="$APPROVED_ENV" \
  scripts/frontier/resilient_e97_true_2n.sbatch
