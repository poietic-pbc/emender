#!/bin/bash
set -euo pipefail

readonly REPO=/lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-d57c7dbe10
readonly SOURCE_COMMIT=d57c7dbea7038e44afdf93ba9fab875474a7ac45
readonly SOURCE_BRANCH=wg/agent-1324/validate-native-pool-v1-4n
readonly BUILD_MANIFEST="$REPO/build/native-resilient-dataplane/native-artifacts.json"
readonly G2_GATE="$REPO/reports/frontier/native-dataplane/5034114/full-layout-gate.json"
readonly APPROVED_PYTHON=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
readonly APPROVED_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
readonly SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
readonly DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
readonly TIKTOKEN=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
readonly RUN_ID=npv1f-progress-20260719T223459Z-f56e27a
readonly RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/npv1f-progress-20260719T223459Z-f56e27a
readonly PAYLOAD_ID=f56e27a-20260719T223459Z-native-pool-v1-failures-progress-2n30m-k40
readonly SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9
readonly RESUME_HANDOFF="$RUN_DIR/handoff/generation-00000010-fence-00000007.json"
readonly FENCE_DB="$RUN_DIR/control/pool-v1.sqlite3"
readonly BULK_ROOT=/tmp/resilient-e97-4n-finalize-20260720T034732Z
readonly KERNEL_CACHE_ROOT=/tmp/resilient-e97-4n-finalize-kernel-cache-20260720T034732Z

[[ -d $REPO/.git ]]
[[ $(git -C "$REPO" branch --show-current) == main ]]
[[ $(git -C "$REPO" rev-parse HEAD) == "$SOURCE_COMMIT" ]]
git -C "$REPO" diff --quiet --ignore-submodules --
git -C "$REPO" diff --cached --quiet --ignore-submodules --
git -C "$REPO" merge-base --is-ancestor "$SOURCE_COMMIT" "origin/$SOURCE_BRANCH"
[[ -x $APPROVED_PYTHON && -r $SEED && -r $DATA && -r $TIKTOKEN ]]
[[ -r $BUILD_MANIFEST && -r $G2_GATE && -r $RESUME_HANDOFF && -r $FENCE_DB ]]
[[ $(sha256sum "$SEED" | awk '{print $1}') == 1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9 ]]
[[ $(sha256sum "$TIKTOKEN" | awk '{print $1}') == 94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069 ]]
[[ $(sha256sum "$RESUME_HANDOFF" | awk '{print $1}') == dc53a24b2b723463e8a5fab0640285d10543db93852989461ae5de339ecb88f7 ]]
[[ $(sqlite3 "$FENCE_DB" "select last_fence from lease_epochs where run_id='$RUN_ID';") == 7 ]]
[[ $(sqlite3 "$FENCE_DB" "select count(*) from leases where run_id='$RUN_ID';") == 0 ]]
[[ $(sqlite3 "$FENCE_DB" "select fence || ':' || json_extract(payload, '\$.generation') || ':' || json_extract(payload, '\$.accepted_tokens') from publications where run_id='$RUN_ID' and kind='latest' and name='authoritative';") == 7:10:78681600 ]]
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
  -A bif148 -p batch --qos=debug -N 4 --gpus-per-node=8 -t 00:30:00 \
  --network=job_vni \
  -J validate-native-pool-4n --chdir="$REPO" \
  --export=ALL,REPO="$REPO",RUN_DIR="$RUN_DIR",RESILIENT_E97_RUN_ID="$RUN_ID",RESILIENT_E97_SOURCE_ID="$SOURCE_ID",RESILIENT_E97_PAYLOAD_ID="$PAYLOAD_ID",RESILIENT_E97_CODE_ID="$SOURCE_COMMIT",RESILIENT_E97_SEED="$SEED",RESILIENT_E97_TRAIN_ARGS_JSON="$REPO/configs/frontier/e97_resilient_split_role_flat.json",RESILIENT_E97_DATA="$DATA",RESILIENT_E97_TIKTOKEN_CACHE_FILE="$TIKTOKEN",RESILIENT_E97_TIKTOKEN_SHA256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069,RESILIENT_E97_NODE_COUNT=4,RESILIENT_E97_GENERATIONS=1,RESILIENT_E97_INITIAL_GENERATION=10,RESILIENT_E97_COORDINATOR_EPOCH=8,RESILIENT_E97_GLOBAL_QUORUM=4,RESILIENT_E97_GLOBAL_TOKEN_MIN=3934080,RESILIENT_E97_STARTUP_SMOKE=0,RESILIENT_E97_REQUESTED_WALLTIME=00:30:00,RESILIENT_E97_LAUNCH_MODE=node-local,RESILIENT_E97_STARTUP_DEADLINE_S=180,RESILIENT_E97_HEARTBEAT_DEADLINE_S=60,RESILIENT_E97_PROGRESS_DEADLINE_S=600,RESILIENT_E97_GENERATION_DEADLINE_S=600,RESILIENT_E97_FENCE_DB="$FENCE_DB",RESILIENT_E97_RESUME_HANDOFF="$RESUME_HANDOFF",RESILIENT_E97_BULK_ROOT="$BULK_ROOT",RESILIENT_E97_KERNEL_CACHE_ROOT="$KERNEL_CACHE_ROOT",RESILIENT_E97_MAX_SPOOL_BYTES=68719476736,RESILIENT_E97_MAX_SHARED_BYTES=68719476736,RESILIENT_E97_BULK_CHUNK_BYTES=67108864,RESILIENT_E97_LOCAL_SPOOL_CHUNK_BYTES=67108864,RESILIENT_E97_MAX_RESTARTS=2,RESILIENT_E97_INJECT_TRAINER=,RESILIENT_E97_INJECT_MANAGER=,RESILIENT_E97_INJECT_NODE_STEP=,RESILIENT_E97_INJECT_NATIVE_SERVICE=,RESILIENT_E97_DELAY_READY=,DILOCO_DATAPLANE=native-cxi,FI_PROVIDER=cxi,FI_MR_CACHE_MONITOR=kdreg2,FI_CXI_ATS=0,NDP_BUILD_MANIFEST="$BUILD_MANIFEST",NDP_FULL_LAYOUT_GATE_JSON="$G2_GATE",EMENDER_CONDA_ENV="$APPROVED_ENV" \
  scripts/frontier/resilient_e97_true_2n.sbatch
