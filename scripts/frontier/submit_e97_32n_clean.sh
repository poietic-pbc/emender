#!/bin/bash
# Submit one immutable held 32-node clean payload, install its scheduler-owned
# afterany collector, verify Nodes/Partition/QOS explicitly, then release.
set -euo pipefail
SOURCE_SHA=ac0c90a91c4c8e68265e573cea9cb808e00987ac
PREDECESSOR_JOB=5126609
PREDECESSOR_PAYLOAD_DIGEST=875aa8d47f99f0ef9881f2db6ddfca240ced11d16580e032c2dba382fb7bc996
PROJECT_ROOT=${PROJECT_ROOT:-/lustre/orion/bif148/scratch/erikgarrison/emender}
BASE=${E97_32N_CLEAN_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-32n}
SEED=${INITIAL_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
CURRENT_REPO=$(cd "$SCRIPT_DIR/../.." && pwd -P)
source "$CURRENT_REPO/scripts/frontier/activate_emender_frontier.sh"
: "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN
[[ -L $SEED && -r $SEED && $(basename "$SEED") == latest.pt ]] || { echo "seed must be a readable latest.pt symlink" >&2; exit 64; }
[[ $(git -C "$PROJECT_ROOT" cat-file -t "$SOURCE_SHA") == commit ]] || { echo "exact source commit unavailable" >&2; exit 64; }
predecessor="$CURRENT_REPO/reports/frontier/e97-same-allocation-8n-${PREDECESSOR_JOB}-verdict.json"
"$EMENDER_PYTHON" - "$predecessor" "$SOURCE_SHA" "$PREDECESSOR_PAYLOAD_DIGEST" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
assert v.get('full_pass') is True, '8-node predecessor full_pass is not literal true'
assert v.get('source_sha') == sys.argv[2], '8-node predecessor source mismatch'
assert v.get('payload_digest') == sys.argv[3], '8-node predecessor payload mismatch'
PY
mkdir -p "$BASE"/{attempts,payloads,runs,collectors}

seed_resolved=$(readlink -f "$SEED")
seed_sha=$(sha256sum "$seed_resolved" | awk '{print $1}')
predecessor_sha=$(sha256sum "$predecessor" | awk '{print $1}')
config=$(cat <<EOF
SOURCE_SHA=$SOURCE_SHA
NODES=32
TASKS_PER_NODE=8
WORLD_SIZE=256
PARTITION=batch
QOS=normal
DILOCO_K=40
SAVE_EVERY=200
KEEP_CHECKPOINTS=2
DILOCO_MERGE_TOPOLOGY=hierarchical
DILOCO_MERGE_BUCKET_NUMEL=67108864
DILOCO_MERGE_GROUP_SIZE=4
TRAIN_MINUTES=14
MAX_EXECUTION_EPOCHS_PER_ATTEMPT=1
FAULT_INJECTION=none
SEED=$SEED
SEED_RESOLVED=$seed_resolved
SEED_SHA256=$seed_sha
PREDECESSOR_JOB=$PREDECESSOR_JOB
PREDECESSOR_PAYLOAD_DIGEST=$PREDECESSOR_PAYLOAD_DIGEST
PREDECESSOR_VERDICT_SHA256=$predecessor_sha
EOF
)
payload_digest=$(
  { printf '%s\n' "$config"; sha256sum \
      "$SCRIPT_DIR/e97_32n_clean_payload.sh" \
      "$SCRIPT_DIR/e97_32n_clean_collector.sh" \
      "$SCRIPT_DIR/submit_e97_32n_clean.sh" "$predecessor";
    git -C "$PROJECT_ROOT" show "$SOURCE_SHA:scripts/frontier/e97_same_allocation_restart.sbatch" | sha256sum;
    git -C "$PROJECT_ROOT" show "$SOURCE_SHA:train.py" | sha256sum;
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
mkdir -p "$payload_root"
printf '%s\n' "$config" > "$payload_root/config.env"
cp "$SCRIPT_DIR/e97_32n_clean_payload.sh" "$payload_root/payload.sh"
cp "$SCRIPT_DIR/e97_32n_clean_collector.sh" "$payload_root/collector.sh"
cp "$SCRIPT_DIR/submit_e97_32n_clean.sh" "$payload_root/submit.sh"
cp "$predecessor" "$payload_root/predecessor-verdict.json"
chmod 0555 "$payload_root"/{payload.sh,collector.sh,submit.sh}
if [[ ! -d $repo_exact/.git ]]; then
  git clone --shared --no-checkout "$PROJECT_ROOT" "$repo_exact"
  git -C "$repo_exact" checkout --detach "$SOURCE_SHA"
fi
[[ $(git -C "$repo_exact" rev-parse HEAD) == "$SOURCE_SHA" ]] || { echo "immutable checkout mismatch" >&2; exit 66; }
[[ -z $(git -C "$repo_exact" status --porcelain --untracked-files=no) ]] || { echo "immutable checkout is dirty" >&2; exit 66; }
launcher="$repo_exact/scripts/frontier/e97_same_allocation_restart.sbatch"
[[ -x $launcher ]] || { echo "production launcher missing or not executable" >&2; exit 66; }
sha256sum "$payload_root/config.env" "$payload_root"/{payload.sh,collector.sh,submit.sh,predecessor-verdict.json} \
  "$launcher" "$repo_exact/train.py" > "$payload_root/SHA256SUMS"

export EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
unset REPO
source "$repo_exact/scripts/frontier/activate_emender_frontier.sh"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN REPO="$repo_exact"
"$EMENDER_PYTHON" -m py_compile "$repo_exact/train.py"
bash -n "$launcher" "$payload_root"/{payload.sh,collector.sh}

run_id="e97-32n-clean-$(date -u +%Y%m%dT%H%M%SZ)-${payload_digest:0:12}"
run_dir="$BASE/runs/$run_id"
# ac0c90a9 writes launch.env before its in-loop mkdir; precreate the sole clean epoch.
mkdir -p "$run_dir"/{identity,logs,supervisor,monitor} "$run_dir/epochs/epoch-000001"
printf '%s\n' "$payload_digest" > "$run_dir/identity/payload-digest.txt"
cp "$payload_root/config.env" "$run_dir/identity/config.env"
cp "$payload_root/SHA256SUMS" "$run_dir/identity/payload-SHA256SUMS"
printf '%s  %s\n' "$seed_sha" "$seed_resolved" > "$run_dir/identity/seed.sha256"

export RUN_ID="$run_id" RUN_DIR="$run_dir" INITIAL_CHECKPOINT="$SEED"
export TARGET_NODES=32 MIN_NODES=32 TASKS_PER_NODE=8 CPUS_PER_TASK=7
export MAX_EXECUTION_EPOCHS_PER_ATTEMPT=1 REQUEUE_ON_EXHAUSTION=0
export EXECUTION_EPOCH_TIMEOUT_SECONDS=1800 FAILED_STEP_WAIT_SECONDS=60 FAILED_STEP_KILL_GRACE_SECONDS=60
export DILOCO_K=40 SAVE_EVERY=200 KEEP_CHECKPOINTS=2
export DILOCO_MERGE_BUCKET_NUMEL=67108864 DILOCO_MERGE_TOPOLOGY=hierarchical
export DILOCO_MERGE_GROUP_SIZE=4 DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=8
export DILOCO_MERGE_COMPLETION_BARRIER=1 TRAIN_MINUTES=14 BATCH_SIZE=4 CHUNK_SIZE=2048
export LOG_EVERY=10 VAL_EVERY=10000 COMPILE_WARMUP_STEPS=1
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=300 WALLTIME_CHECK_EVERY=40 DISTRIBUTED_HEALTH_CHECK_EVERY=40
export FRONTIER_RCCL_ENV=recommended FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
export FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug REQUIRE_RCCL_NET_PLUGIN=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
unset EMENDER_DILOCO_EXIT_RANK EMENDER_DILOCO_EXIT_MERGE EMENDER_DILOCO_EXIT_BUCKET
unset EMENDER_DILOCO_EXIT_LABEL EMENDER_DILOCO_EXIT_CODE EMENDER_DILOCO_EXIT_DELAY_SECONDS

payload_id=$(sbatch --parsable --hold --no-kill --requeue --signal=B:TERM@300 \
  -A bif148 -p batch -q normal -J e97-32n-clean -N32 -t 00:45:00 \
  --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest \
  --output="$run_dir/logs/batch-%j.out" --error="$run_dir/logs/batch-%j.err" \
  --export=ALL "$payload_root/payload.sh")
submission="$run_dir/identity/submission.json"
"$EMENDER_PYTHON" - "$submission" <<PY
import json
v={'schema':'e97-same-allocation-32n-clean-submission-v1','payload_job_id':'$payload_id',
'source_sha':'$SOURCE_SHA','payload_digest':'$payload_digest','run_id':'$run_id',
'nodes':32,'world_size':256,'partition':'batch','qos':'normal','held':True,'released':False,
'run_dir':'$run_dir','payload_root':'$payload_root','initial_checkpoint':'$SEED',
'seed_sha256':'$seed_sha','predecessor_job':'$PREDECESSOR_JOB',
'predecessor_payload_digest':'$PREDECESSOR_PAYLOAD_DIGEST','predecessor_verdict_sha256':'$predecessor_sha',
'python_bin':'$EMENDER_PYTHON','py_compile_passed':True,'fault_injection':'none'}
open('$submission','w').write(json.dumps(v,sort_keys=True,indent=2)+'\n')
PY
collector_id=$(sbatch --parsable -A bif148 -p batch -q normal -N1 -t 00:12:00 \
  --dependency="afterany:$payload_id" -J collect-e97-32n-clean \
  --output="$BASE/collectors/%j.out" --error="$BASE/collectors/%j.err" \
  --export=ALL,PAYLOAD_JOB_ID="$payload_id",RUN_DIR="$run_dir",PAYLOAD_ROOT="$payload_root",REPO="$repo_exact",SUBMISSION_RECORD="$submission" \
  "$payload_root/collector.sh")
squeue -j "$payload_id,$collector_id" -h -o '%i|%T|%D|%N|%P|%q|%R' | tee "$run_dir/identity/squeue-held.txt"
job_record=$(scontrol show job "$payload_id" -o)
[[ $job_record == *"NumNodes=32"* && $job_record == *"Partition=batch"* && $job_record == *"QOS=normal"* ]] || {
  echo "fail closed: payload scheduler binding mismatch: $job_record" >&2; exit 67;
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
