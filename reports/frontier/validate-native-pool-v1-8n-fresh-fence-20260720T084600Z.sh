#!/bin/bash
set -euo pipefail

readonly REPO=/lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-5121539b05
readonly SOURCE_COMMIT=5121539bcd7c9679e32b42bb2ebf1722672d9015
readonly SOURCE_BRANCH=wg/agent-1326/validate-native-pool-v1-8n
readonly BUILD_MANIFEST="$REPO/build/native-resilient-dataplane/native-artifacts.json"
readonly G2_GATE="$REPO/reports/frontier/native-dataplane/5034807/full-layout-gate.json"
readonly APPROVED_PYTHON=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
readonly APPROVED_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
readonly SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
readonly DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
readonly TIKTOKEN=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
readonly RUN_ID=npv1f-progress-20260719T223459Z-f56e27a
readonly RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/npv1f-progress-20260719T223459Z-f56e27a
readonly PAYLOAD_ID=f56e27a-20260719T223459Z-native-pool-v1-failures-progress-2n30m-k40
readonly SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9
readonly RESUME_HANDOFF="$RUN_DIR/handoff/generation-00000013-fence-00000014.json"
readonly FENCE_DB="$RUN_DIR/control/pool-v1.sqlite3"
readonly BULK_ROOT=/tmp/e8-f16-fresh-084600
readonly KERNEL_CACHE_ROOT=/tmp/e8-k-f16-fresh-084600
readonly NDP_SOCKET_PROBE="$BULK_ROOT/$RUN_ID/node-0/control/ndp.sock"

[[ -d $REPO/.git ]]
[[ $(git -C "$REPO" branch --show-current) == main ]]
[[ $(git -C "$REPO" rev-parse HEAD) == "$SOURCE_COMMIT" ]]
git -C "$REPO" diff --quiet --ignore-submodules --
git -C "$REPO" diff --cached --quiet --ignore-submodules --
git -C "$REPO" merge-base --is-ancestor "$SOURCE_COMMIT" "origin/$SOURCE_BRANCH"
[[ -x $APPROVED_PYTHON && -r $SEED && -r $DATA && -r $TIKTOKEN ]]
[[ -r $BUILD_MANIFEST && -r $G2_GATE && -r $RESUME_HANDOFF && -r $FENCE_DB ]]
[[ ${#NDP_SOCKET_PROBE} -lt 108 ]]
[[ $(sha256sum "$SEED" | awk '{print $1}') == 1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9 ]]
[[ $(sha256sum "$TIKTOKEN" | awk '{print $1}') == 94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069 ]]
[[ $(sha256sum "$REPO/configs/frontier/e97_resilient_split_role_flat.json" | awk '{print $1}') == afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c ]]
[[ $(sha256sum "$RESUME_HANDOFF" | awk '{print $1}') == 040ea79b0592ffae3f13e8d60cbd50c7fcc1e19341201cdd2e95fc959cbe4e5b ]]
[[ $(sqlite3 "$FENCE_DB" "select last_fence from lease_epochs where run_id='$RUN_ID';") == 15 ]]
[[ $(sqlite3 "$FENCE_DB" "select count(*) from leases where run_id='$RUN_ID';") == 0 ]]
[[ $(sqlite3 "$FENCE_DB" "select fence || ':' || json_extract(payload, '\$.generation') || ':' || json_extract(payload, '\$.accepted_tokens') from publications where run_id='$RUN_ID' and kind='latest' and name='authoritative';") == 14:13:131136000 ]]
[[ $(sqlite3 "$FENCE_DB" "select count(*) from publications where run_id='$RUN_ID' and kind='commit' and name='generation-00000014';") == 0 ]]
[[ -z $(squeue -u "$USER" -h -o '%i') ]] || {
  echo "refusing to overlap another user allocation" >&2
  exit 69
}

"$APPROVED_PYTHON" "$REPO/scripts/frontier/attest_native_dataplane.py" verify \
  --backend native-cxi --production --full-layout \
  --build-manifest "$BUILD_MANIFEST" --gate-json "$G2_GATE" \
  --source-root "$REPO" >/dev/null

cd "$REPO"
exec sbatch --parsable \
  -A bif148 -p batch --qos=debug -N 8 --gpus-per-node=8 -t 00:30:00 \
  --network=job_vni \
  -J validate-native-pool-8n-fresh --chdir="$REPO" \
  --export=ALL,REPO="$REPO",RUN_DIR="$RUN_DIR",RESILIENT_E97_RUN_ID="$RUN_ID",RESILIENT_E97_SOURCE_ID="$SOURCE_ID",RESILIENT_E97_PAYLOAD_ID="$PAYLOAD_ID",RESILIENT_E97_CODE_ID="$SOURCE_COMMIT",RESILIENT_E97_SEED="$SEED",RESILIENT_E97_TRAIN_ARGS_JSON="$REPO/configs/frontier/e97_resilient_split_role_flat.json",RESILIENT_E97_DATA="$DATA",RESILIENT_E97_TIKTOKEN_CACHE_FILE="$TIKTOKEN",RESILIENT_E97_TIKTOKEN_SHA256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069,RESILIENT_E97_NODE_COUNT=8,RESILIENT_E97_GENERATIONS=1,RESILIENT_E97_INITIAL_GENERATION=13,RESILIENT_E97_COORDINATOR_EPOCH=16,RESILIENT_E97_GLOBAL_QUORUM=8,RESILIENT_E97_GLOBAL_TOKEN_MIN=3934080,RESILIENT_E97_STARTUP_SMOKE=0,RESILIENT_E97_REQUESTED_WALLTIME=00:30:00,RESILIENT_E97_LAUNCH_MODE=node-local,RESILIENT_E97_STARTUP_DEADLINE_S=180,RESILIENT_E97_HEARTBEAT_DEADLINE_S=60,RESILIENT_E97_PROGRESS_DEADLINE_S=600,RESILIENT_E97_GENERATION_DEADLINE_S=600,RESILIENT_E97_FENCE_DB="$FENCE_DB",RESILIENT_E97_RESUME_HANDOFF="$RESUME_HANDOFF",RESILIENT_E97_BULK_ROOT="$BULK_ROOT",RESILIENT_E97_KERNEL_CACHE_ROOT="$KERNEL_CACHE_ROOT",RESILIENT_E97_MAX_SPOOL_BYTES=68719476736,RESILIENT_E97_MAX_SHARED_BYTES=68719476736,RESILIENT_E97_BULK_CHUNK_BYTES=67108864,RESILIENT_E97_LOCAL_SPOOL_CHUNK_BYTES=67108864,RESILIENT_E97_MAX_RESTARTS=2,RESILIENT_E97_INJECT_TRAINER=,RESILIENT_E97_INJECT_MANAGER=,RESILIENT_E97_INJECT_NODE_STEP=,RESILIENT_E97_INJECT_NATIVE_SERVICE=,RESILIENT_E97_DELAY_READY=,DILOCO_DATAPLANE=native-cxi,FI_PROVIDER=cxi,FI_MR_CACHE_MONITOR=kdreg2,FI_CXI_ATS=0,NDP_BUILD_MANIFEST="$BUILD_MANIFEST",NDP_FULL_LAYOUT_GATE_JSON="$G2_GATE",EMENDER_CONDA_ENV="$APPROVED_ENV" \
  scripts/frontier/resilient_e97_true_2n.sbatch
