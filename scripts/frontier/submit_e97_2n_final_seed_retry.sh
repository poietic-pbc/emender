#!/bin/bash
# Submit exactly one held two-node final-seed same-set retry payload, register a
# durable afterany collector, verify Nodes/Partition/QOS, then release.
set -euo pipefail
CANONICAL_BASE=c625cede2b97ad43af6e1e47a5fd4d58e1dbafcb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}
BASE=${E97_FINAL_SEED_2N_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-same-set-retry-2n}
SEED_CACHE_ROOT=${E97_SEED_CACHE_ROOT:-/lustre/orion/bif148/proj-shared/emender/bootstrap/e97-seeds}
cd "$PROJECT_ROOT"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked source must be clean" >&2; exit 64; }
SOURCE_SHA=$(git rev-parse HEAD)
[[ $(git merge-base "$SOURCE_SHA" "$CANONICAL_BASE") == "$CANONICAL_BASE" ]] || {
  echo "source is not based on canonical origin/main $CANONICAL_BASE" >&2; exit 64;
}
[[ $(git cat-file -t "$CANONICAL_BASE") == commit ]] || { echo "canonical base unavailable" >&2; exit 64; }
mkdir -p "$BASE"/{attempts,payloads,runs,collectors}

config=$(cat <<EOF
CANONICAL_BASE=$CANONICAL_BASE
SOURCE_SHA=$SOURCE_SHA
NODES=2
TASKS_PER_NODE=8
WORLD_SIZE=16
PARTITION=batch
QOS=debug
DILOCO_K=40
SAVE_EVERY=200
KEEP_CHECKPOINTS=2
DILOCO_MERGE_TOPOLOGY=hierarchical
DILOCO_MERGE_BUCKET_NUMEL=67108864
DILOCO_MERGE_GROUP_SIZE=4
SEED_STEP=2300930
SEED_ACCEPTED_TOKENS=150793748480
SEED_BYTES=7719680116
SEED_SHA256=0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
SEED_URI=s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt
FAULT_RANK=1
FAULT_MERGE=3
FAULT_BUCKET=1
FAULT_LABEL=sf_x
FAULT_EXIT_CODE=86
EOF
)
payload_digest=$(
  { printf '%s\n' "$config"; sha256sum \
      "$SCRIPT_DIR/e97_same_allocation_restart.sbatch" \
      "$SCRIPT_DIR/e97_2n_final_seed_retry_srun_shim.sh" \
      "$SCRIPT_DIR/e97_2n_final_seed_retry_collector.sh" \
      "$SCRIPT_DIR/submit_e97_2n_final_seed_retry.sh" \
      "$PROJECT_ROOT/train.py" "$PROJECT_ROOT/configs/frontier/e97_async_256.yaml" \
      "$PROJECT_ROOT/scripts/frontier/materialize_e97_s3_seed.py";
  } | sha256sum | awk '{print $1}'
)
sentinel="$BASE/attempts/$payload_digest"
( set -o noclobber; printf '%s|%s|%s\n' "$SOURCE_SHA" "$(date -u +%FT%TZ)" "$payload_digest" > "$sentinel" ) 2>/dev/null || {
  echo "unchanged payload bytes already attempted: $payload_digest" >&2; exit 65;
}
payload_id=
collector_id=
cleanup_on_error() {
  rc=$?
  if (( rc != 0 )); then
    [[ -z $payload_id ]] || scancel "$payload_id" 2>/dev/null || true
    [[ -z $collector_id ]] || scancel "$collector_id" 2>/dev/null || true
    [[ -n $payload_id ]] || rm -f "$sentinel"
  fi
  exit "$rc"
}
trap cleanup_on_error EXIT

payload_root="$BASE/payloads/$payload_digest"
repo_exact="$payload_root/repo"
shim_bin="$payload_root/shim-bin"
mkdir -p "$payload_root" "$shim_bin"
printf '%s\n' "$config" > "$payload_root/config.env"
cp "$SCRIPT_DIR/e97_2n_final_seed_retry_srun_shim.sh" "$shim_bin/srun"
cp "$SCRIPT_DIR/e97_2n_final_seed_retry_collector.sh" "$payload_root/collector.sh"
cp "$SCRIPT_DIR/submit_e97_2n_final_seed_retry.sh" "$payload_root/submit.sh"
chmod 0555 "$shim_bin/srun" "$payload_root/collector.sh" "$payload_root/submit.sh"
git clone --shared --no-checkout "$PROJECT_ROOT" "$repo_exact"
git -C "$repo_exact" checkout --detach "$SOURCE_SHA"
[[ $(git -C "$repo_exact" rev-parse HEAD) == "$SOURCE_SHA" ]] || { echo "immutable checkout mismatch" >&2; exit 66; }
[[ -z $(git -C "$repo_exact" status --porcelain --untracked-files=no) ]] || { echo "immutable checkout dirty" >&2; exit 66; }
launcher="$repo_exact/scripts/frontier/e97_same_allocation_restart.sbatch"

# Submit/login-side only: resolve the immutable checkpoint and manifest plus the
# discovery authority, verify bytes, and atomically publish the SHA cache.
prefetch_json=$("$EMENDER_PYTHON" "$repo_exact/scripts/frontier/materialize_e97_s3_seed.py" \
  --seed-config "$repo_exact/configs/frontier/e97_async_256.yaml" --prefetch \
  --cache-root "$SEED_CACHE_ROOT" --attestation "$payload_root/seed-bootstrap-attestation.json")
read -r seed_cache seed_attestation seed_attestation_sha256 < <(
  "$EMENDER_PYTHON" -c 'import json,sys; v=json.load(sys.stdin); print(v["cache"],v["attestation"],v["attestation_sha256"])' <<< "$prefetch_json"
)
[[ $(basename "$seed_cache") == sha256-0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2.pt ]] || {
  echo "prefetch did not return the exact final seed cache" >&2; exit 66;
}
sha256sum "$payload_root/config.env" "$shim_bin/srun" "$payload_root"/{collector.sh,submit.sh,seed-bootstrap-attestation.json} \
  "$launcher" "$repo_exact/train.py" > "$payload_root/SHA256SUMS"

export EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
unset REPO
source "$repo_exact/scripts/frontier/activate_emender_frontier.sh"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN REPO="$repo_exact"
"$EMENDER_PYTHON" -m py_compile "$repo_exact/train.py" \
  "$repo_exact/scripts/frontier/materialize_e97_s3_seed.py"
bash -n "$launcher" "$shim_bin/srun" "$payload_root/collector.sh"

run_id="e97-final-seed-2n-$(date -u +%Y%m%dT%H%M%SZ)-${payload_digest:0:12}"
run_dir="$BASE/runs/$run_id"
state_dir="$run_dir/supervisor/final-seed-retry"
mkdir -p "$run_dir"/{identity,logs,supervisor} "$state_dir"
printf '%s\n' "$payload_digest" > "$run_dir/identity/payload-digest.txt"
cp "$payload_root/config.env" "$run_dir/identity/config.env"
cp "$payload_root/SHA256SUMS" "$run_dir/identity/payload-SHA256SUMS"
printf '%s  %s\n' "$seed_attestation_sha256" "$seed_attestation" > "$run_dir/identity/seed-attestation.sha256"

export PATH="$shim_bin:$PATH"
export RUN_ID="$run_id" RUN_DIR="$run_dir" FINAL_SEED_RETRY_STATE_DIR="$state_dir"
export REAL_SRUN=/usr/bin/srun TARGET_NODES=2 MIN_NODES=2 TASKS_PER_NODE=8 CPUS_PER_TASK=7
export MAX_CONSECUTIVE_NO_PROGRESS_FAILURES=2 REQUEUE_ON_EXHAUSTION=0
export EXECUTION_EPOCH_TIMEOUT_SECONDS=900 FAILED_STEP_WAIT_SECONDS=60 FAILED_STEP_KILL_GRACE_SECONDS=60
export DILOCO_K=40 SAVE_EVERY=200 KEEP_CHECKPOINTS=2
export DILOCO_MERGE_BUCKET_NUMEL=67108864 DILOCO_MERGE_TOPOLOGY=hierarchical
export DILOCO_MERGE_GROUP_SIZE=4 DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=8
export DILOCO_MERGE_COMPLETION_BARRIER=1 TRAIN_MINUTES=4 BATCH_SIZE=4 CHUNK_SIZE=2048
export LOG_EVERY=10 VAL_EVERY=10000 COMPILE_WARMUP_STEPS=1
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=180 WALLTIME_CHECK_EVERY=40 DISTRIBUTED_HEALTH_CHECK_EVERY=40
export E97_SEED_CONFIG="$repo_exact/configs/frontier/e97_async_256.yaml"
export E97_SEED_CACHE="$seed_cache" E97_SEED_ATTESTATION="$seed_attestation"
export E97_SEED_ATTESTATION_SHA256="$seed_attestation_sha256"
export EMENDER_DILOCO_EXIT_RANK=1 EMENDER_DILOCO_EXIT_MERGE=3 EMENDER_DILOCO_EXIT_BUCKET=1
export EMENDER_DILOCO_EXIT_LABEL=sf_x EMENDER_DILOCO_EXIT_CODE=86 EMENDER_DILOCO_EXIT_DELAY_SECONDS=0.25
export FRONTIER_RCCL_ENV=recommended FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
export FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug REQUIRE_RCCL_NET_PLUGIN=1

payload_id=$(sbatch --parsable --hold --no-kill --requeue --signal=B:TERM@300 \
  -A bif148 -p batch -q debug -J e97-final-seed-retry-2n -N2 -t 00:30:00 \
  --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest \
  --output="$run_dir/logs/batch-%j.out" --error="$run_dir/logs/batch-%j.err" \
  --export=ALL "$launcher")
submission="$run_dir/identity/submission.json"
"$EMENDER_PYTHON" - "$submission" <<PY
import json
v={'schema':'e97-final-seed-same-set-retry-submission-v1','payload_job_id':'$payload_id',
'source_sha':'$SOURCE_SHA','canonical_base':'$CANONICAL_BASE','payload_digest':'$payload_digest',
'run_id':'$run_id','nodes':2,'world_size':16,'partition':'batch','qos':'debug',
'held':True,'released':False,'run_dir':'$run_dir','payload_root':'$payload_root',
'seed_step':2300930,'seed_accepted_tokens':150793748480,'seed_bytes':7719680116,
'seed_sha256':'0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2',
'seed_cache':'$seed_cache','seed_attestation':'$seed_attestation',
'seed_attestation_sha256':'$seed_attestation_sha256','python_bin':'$EMENDER_PYTHON'}
open('$submission','w').write(json.dumps(v,sort_keys=True,indent=2)+'\n')
PY
collector_id=$(sbatch --parsable -A bif148 -p batch -q normal -N1 -t 00:10:00 \
  --dependency="afterany:$payload_id" -J collect-e97-final-seed-2n \
  --output="$BASE/collectors/%j.out" --error="$BASE/collectors/%j.err" \
  --export=ALL,PAYLOAD_JOB_ID="$payload_id",RUN_DIR="$run_dir",PAYLOAD_ROOT="$payload_root",REPO="$repo_exact",SUBMISSION_RECORD="$submission" \
  "$payload_root/collector.sh")
squeue -j "$payload_id,$collector_id" -h -o '%i|%T|%D|%N|%P|%q|%R' | tee "$run_dir/identity/squeue-held.txt"
job_record=$(scontrol show job "$payload_id" -o)
[[ $job_record == *"NumNodes=2"* && $job_record == *"Partition=batch"* && $job_record == *"QOS=debug"* ]] || {
  echo "fail closed: exact two-node scheduler binding mismatch: $job_record" >&2; exit 67;
}
"$EMENDER_PYTHON" - "$submission" "$collector_id" <<'PY'
import json,os,sys
p=sys.argv[1]; v=json.load(open(p)); v.update({'collector_job_id':sys.argv[2],
 'dependency':'afterany:'+v['payload_job_id'],'collector_registered_before_release':True})
t=p+'.tmp'; open(t,'w').write(json.dumps(v,sort_keys=True,indent=2)+'\n'); os.replace(t,p)
PY
scontrol release "$payload_id"
"$EMENDER_PYTHON" - "$submission" <<'PY'
import json,os,sys,time
p=sys.argv[1]; v=json.load(open(p)); v.update({'released':True,'released_epoch':time.time()})
t=p+'.tmp'; open(t,'w').write(json.dumps(v,sort_keys=True,indent=2)+'\n'); os.replace(t,p)
PY
trap - EXIT
printf 'payload_job_id=%s\ncollector_job_id=%s\nrun_id=%s\nrun_dir=%s\npayload_digest=%s\nsource_sha=%s\n' \
  "$payload_id" "$collector_id" "$run_id" "$run_dir" "$payload_digest" "$SOURCE_SHA"
