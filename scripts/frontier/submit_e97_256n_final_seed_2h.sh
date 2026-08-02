#!/bin/bash
# Submit exactly one immutable, unheld 256-node / 2,048-rank two-hour
# continuation and verify its live scheduler binding. The batch job writes all
# logs/checkpoint state directly to durable RUN_DIR; no collector job is used.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}
BASE=${E97_256N_FINAL_SEED_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n}
SEED_CACHE_ROOT=${E97_SEED_CACHE_ROOT:-/lustre/orion/bif148/proj-shared/emender/bootstrap/e97-seeds}
RUN_ID=${E97_256N_RUN_ID:-e97-final-seed-production-256n}
RUN_DIR=${E97_256N_RUN_DIR:-$BASE/runs/$RUN_ID}
SEED_STEP=2300930
SEED_ACCEPTED_TOKENS=150793748480
SEED_BYTES=7719680116
SEED_SHA256=0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
SEED_URI=s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt
TOKEN_MIGRATION_RECEIPT_REL=docs/validation/e97-total-token-migration-step2303840.json
TOKEN_MIGRATION_RECEIPT="$PROJECT_ROOT/$TOKEN_MIGRATION_RECEIPT_REL"

cd "$PROJECT_ROOT"
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312}
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN
[[ "$RUN_ID" != */* && "$RUN_ID" != . && "$RUN_ID" != .. ]] || {
  echo "RUN_ID must be one stable path component" >&2; exit 64;
}
[[ "$RUN_DIR" == /* ]] || { echo "RUN_DIR must be absolute" >&2; exit 64; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || {
  echo "tracked source must be clean" >&2; exit 64;
}
SOURCE_SHA=$(git rev-parse HEAD)
ORIGIN_MAIN_SHA=$(git rev-parse origin/main)
LOCAL_MAIN_SHA=$(git rev-parse main)
[[ "$SOURCE_SHA" == "$ORIGIN_MAIN_SHA" && "$SOURCE_SHA" == "$LOCAL_MAIN_SHA" ]] || {
  echo "submission requires clean HEAD == main == origin/main" >&2; exit 64;
}
if git cat-file -e "$SOURCE_SHA:.final_checkpoint_request" 2>/dev/null; then
  echo "forbidden .final_checkpoint_request is tracked in source" >&2; exit 64
fi
mkdir -p "$BASE"/{attempts,payloads,runs,attestations}

# Submit/login-side only. Resolve both S3 authorities, verify the complete
# 7,719,680,116-byte seed, and atomically publish/reuse its content-addressed
# cache before any scheduler mutation.
attestation_stage="$BASE/attestations/seed-bootstrap-$SOURCE_SHA.json"
prefetch_json=$("$EMENDER_PYTHON" scripts/frontier/materialize_e97_s3_seed.py \
  --seed-config configs/frontier/e97_async_256.yaml --prefetch \
  --cache-root "$SEED_CACHE_ROOT" --attestation "$attestation_stage")
read -r seed_cache seed_attestation seed_attestation_sha256 < <(
  "$EMENDER_PYTHON" -c 'import json,sys; v=json.load(sys.stdin); print(v["cache"],v["attestation"],v["attestation_sha256"])' \
    <<< "$prefetch_json"
)
[[ $(basename "$seed_cache") == "sha256-$SEED_SHA256.pt" \
   && -f "$seed_cache" && -f "$seed_attestation" ]] || {
  echo "prefetch did not return the exact content-addressed final seed" >&2; exit 66;
}
read -r verified_seed_bytes verified_seed_sha < <(
  "$EMENDER_PYTHON" - "$seed_cache" <<'PY'
import hashlib, pathlib, sys
path=pathlib.Path(sys.argv[1]); digest=hashlib.sha256(); size=0
with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(64*1024*1024), b''):
        digest.update(chunk); size += len(chunk)
print(size, digest.hexdigest())
PY
)
[[ "$verified_seed_bytes" == "$SEED_BYTES" && "$verified_seed_sha" == "$SEED_SHA256" ]] || {
  echo "submit-side seed size/SHA verification failed" >&2; exit 66;
}
[[ $(sha256sum "$seed_attestation" | awk '{print $1}') == "$seed_attestation_sha256" ]] || {
  echo "submit-side attestation digest verification failed" >&2; exit 66;
}

config=$(cat <<EOF
SOURCE_SHA=$SOURCE_SHA
ORIGIN_MAIN_SHA=$ORIGIN_MAIN_SHA
LOCAL_MAIN_SHA=$LOCAL_MAIN_SHA
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
NODES=256
TASKS_PER_NODE=8
WORLD_SIZE=2048
PARTITION=batch
QOS=debug
TIME_LIMIT=02:00:00
DILOCO_K=40
SAVE_EVERY=200
KEEP_CHECKPOINTS=2
DILOCO_MERGE_TOPOLOGY=hierarchical
DILOCO_MERGE_BUCKET_NUMEL=67108864
DILOCO_MERGE_GROUP_SIZE=4
SEED_STEP=$SEED_STEP
SEED_ACCEPTED_TOKENS=$SEED_ACCEPTED_TOKENS
SEED_BYTES=$SEED_BYTES
SEED_SHA256=$SEED_SHA256
SEED_URI=$SEED_URI
SEED_CACHE=$seed_cache
SEED_ATTESTATION_SHA256=$seed_attestation_sha256
TOKEN_MIGRATION_RECEIPT=$TOKEN_MIGRATION_RECEIPT_REL
TOKEN_MIGRATION_STEP=2303840
TOKEN_MIGRATION_TOTAL_TOKENS=199615447040
TOKEN_MIGRATION_SOURCE_JOB=5134243
TOKEN_MIGRATION_CHECKPOINT_BYTES=7719680116
FAULT_INJECTION=none
ACCEPTANCE_SHIM=none
REQUEUE=true
NO_KILL=true
EOF
)
asset_hashes=$(
  for asset in \
    scripts/frontier/submit_e97_256n_final_seed_2h.sh \
    scripts/frontier/e97_256n_final_seed_payload.sh \
    scripts/frontier/e97_same_allocation_restart.sbatch \
    scripts/frontier/materialize_e97_s3_seed.py \
    scripts/frontier/validate_total_token_migration_receipt.py \
    "$TOKEN_MIGRATION_RECEIPT_REL" \
    configs/frontier/e97_async_256.yaml train.py "$seed_attestation"; do
    printf '%s  %s\n' "$(sha256sum "$asset" | awk '{print $1}')" "${asset#$PROJECT_ROOT/}"
  done
)
payload_digest=$(
  { printf '%s\n' "$config"; printf '%s\n' "$asset_hashes"; } | sha256sum | awk '{print $1}'
)
sentinel="$BASE/attempts/$payload_digest"
( set -o noclobber; printf '%s|%s|%s\n' "$SOURCE_SHA" "$(date -u +%FT%TZ)" "$payload_digest" > "$sentinel" ) 2>/dev/null || {
  echo "unchanged payload bytes already attempted: $payload_digest" >&2; exit 65;
}
payload_id=
cleanup_on_error() {
  rc=$?
  if (( rc != 0 )); then
    # Never cancel a submitted training payload. Only a pre-submission failure
    # may remove the no-retry sentinel.
    [[ -n $payload_id ]] || rm -f "$sentinel"
  fi
  exit "$rc"
}
trap cleanup_on_error EXIT

payload_root="$BASE/payloads/$payload_digest"
repo_exact="$payload_root/repo"
mkdir -p "$payload_root"
printf '%s\n' "$config" > "$payload_root/config.env"
printf '%s\n' "$asset_hashes" > "$payload_root/source-assets.sha256"
cp scripts/frontier/e97_256n_final_seed_payload.sh "$payload_root/payload.sh"
cp scripts/frontier/submit_e97_256n_final_seed_2h.sh "$payload_root/submit.sh"
cp "$seed_attestation" "$payload_root/seed-bootstrap-attestation.json"
chmod 0555 "$payload_root"/{payload.sh,submit.sh}
git clone --shared --no-checkout "$PROJECT_ROOT" "$repo_exact"
git -C "$repo_exact" checkout --detach "$SOURCE_SHA"
[[ $(git -C "$repo_exact" rev-parse HEAD) == "$SOURCE_SHA" ]] || {
  echo "immutable checkout mismatch" >&2; exit 66;
}
[[ -z $(git -C "$repo_exact" status --porcelain --untracked-files=no) ]] || {
  echo "immutable checkout dirty" >&2; exit 66;
}
launcher="$repo_exact/scripts/frontier/e97_same_allocation_restart.sbatch"
sha256sum "$payload_root"/{config.env,source-assets.sha256,payload.sh,submit.sh,seed-bootstrap-attestation.json} \
  "$launcher" "$repo_exact/train.py" "$repo_exact/configs/frontier/e97_async_256.yaml" \
  "$repo_exact/scripts/frontier/materialize_e97_s3_seed.py" \
  "$repo_exact/scripts/frontier/validate_total_token_migration_receipt.py" \
  "$repo_exact/$TOKEN_MIGRATION_RECEIPT_REL" > "$payload_root/SHA256SUMS"

# Canonical Frontier environment is re-sourced from the immutable checkout for
# every preflight and is the exact environment exported to the payload.
export EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
unset REPO
source "$repo_exact/scripts/frontier/activate_emender_frontier.sh"
: "${EMENDER_PYTHON:?immutable activation did not set EMENDER_PYTHON}"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN REPO="$repo_exact"
"$EMENDER_PYTHON" -m py_compile "$repo_exact/train.py" \
  "$repo_exact/scripts/frontier/materialize_e97_s3_seed.py" \
  "$repo_exact/scripts/frontier/validate_total_token_migration_receipt.py"
bash -n "$launcher" "$payload_root"/{payload.sh,submit.sh}

# The stable run directory deliberately contains neither a timestamp nor a
# Slurm job ID. If a future changed payload uses this RUN_ID, the launcher
# preserves a readable newer train/latest.pt rather than resetting to the seed.
mkdir -p "$RUN_DIR"/{identity,logs,supervisor,monitor,terminal}
if [[ -e "$RUN_DIR/identity/run-id.txt" ]]; then
  [[ $(<"$RUN_DIR/identity/run-id.txt") == "$RUN_ID" ]] || {
    echo "stable run identity conflict" >&2; exit 66;
  }
fi
if [[ -e "$RUN_DIR/train/latest.pt" || -L "$RUN_DIR/train/latest.pt" ]]; then
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "existing stable checkpoint authority is not a readable atomic symlink" >&2; exit 66;
  }
  resume_name=$(basename "$(readlink -f "$RUN_DIR/train/latest.pt")")
  [[ $resume_name =~ ^checkpoint_step_([0-9]+)_ ]] || {
    echo "existing stable checkpoint authority has no exact step" >&2; exit 66;
  }
  resume_step=$((10#${BASH_REMATCH[1]}))
  (( resume_step > SEED_STEP )) || {
    echo "existing stable checkpoint authority is not newer than the cold seed" >&2; exit 66;
  }
  if (( resume_step == 2303840 )); then
    "$EMENDER_PYTHON" scripts/frontier/validate_total_token_migration_receipt.py \
      --receipt "$TOKEN_MIGRATION_RECEIPT" \
      --checkpoint "$RUN_DIR/train/latest.pt" --run-id "$RUN_ID" \
      --run-dir "$RUN_DIR" --latest "$RUN_DIR/train/latest.pt" \
      --expected-step 2303840 --expected-total-tokens 199615447040 \
      --expected-source-job-id 5134243 --expected-size-bytes 7719680116 >/dev/null
  fi
fi
printf '%s\n' "$payload_digest" > "$RUN_DIR/identity/payload-digest.txt"
printf '%s\n' "$config" > "$RUN_DIR/identity/config.env"
cp "$payload_root/SHA256SUMS" "$RUN_DIR/identity/payload-SHA256SUMS"
printf '%s  %s\n' "$SEED_SHA256" "$seed_cache" > "$RUN_DIR/identity/seed-cache.sha256"
printf '%s  %s\n' "$seed_attestation_sha256" "$seed_attestation" > "$RUN_DIR/identity/seed-attestation.sha256"
printf 'HEAD=%s\nmain=%s\norigin/main=%s\n' "$SOURCE_SHA" "$LOCAL_MAIN_SHA" "$ORIGIN_MAIN_SHA" \
  > "$RUN_DIR/identity/source-refs.txt"

export RUN_ID RUN_DIR
export TARGET_NODES=256 MIN_NODES=256 TASKS_PER_NODE=8 CPUS_PER_TASK=7
export MAX_CONSECUTIVE_NO_PROGRESS_FAILURES=2 REQUEUE_ON_EXHAUSTION=1
export EXECUTION_EPOCH_TIMEOUT_SECONDS=9000 FAILED_STEP_WAIT_SECONDS=60 FAILED_STEP_KILL_GRACE_SECONDS=60
export DILOCO_K=40 SAVE_EVERY=200 KEEP_CHECKPOINTS=2
export DILOCO_MERGE_BUCKET_NUMEL=67108864 DILOCO_MERGE_TOPOLOGY=hierarchical
export DILOCO_MERGE_GROUP_SIZE=4 DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=8
export DILOCO_MERGE_COMPLETION_BARRIER=1 TRAIN_MINUTES=180 BATCH_SIZE=4 CHUNK_SIZE=2048
export LOG_EVERY=10 VAL_EVERY=10000 COMPILE_WARMUP_STEPS=1
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=900 WALLTIME_CHECK_EVERY=40 DISTRIBUTED_HEALTH_CHECK_EVERY=40
export E97_SEED_CONFIG="$repo_exact/configs/frontier/e97_async_256.yaml"
export E97_SEED_CACHE="$seed_cache" E97_SEED_ATTESTATION="$seed_attestation"
export E97_SEED_ATTESTATION_SHA256="$seed_attestation_sha256"
export TOTAL_TOKEN_MIGRATION_RECEIPT="$repo_exact/$TOKEN_MIGRATION_RECEIPT_REL"
export FRONTIER_RCCL_ENV=recommended FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
export FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug REQUIRE_RCCL_NET_PLUGIN=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
unset EMENDER_DILOCO_EXIT_RANK EMENDER_DILOCO_EXIT_MERGE EMENDER_DILOCO_EXIT_BUCKET
unset EMENDER_DILOCO_EXIT_LABEL EMENDER_DILOCO_EXIT_CODE EMENDER_DILOCO_EXIT_DELAY_SECONDS

payload_id=$(sbatch --parsable --no-kill --requeue --signal=B:TERM@300 \
  -A bif148 -p batch -q debug -J e97-final-seed-256n -N256 -t 02:00:00 \
  --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest \
  --output="$RUN_DIR/logs/batch-%j.out" --error="$RUN_DIR/logs/batch-%j.err" \
  --export=ALL "$payload_root/payload.sh")
submission="$RUN_DIR/identity/submission.json"
"$EMENDER_PYTHON" - "$submission" <<PY
import json
v={'schema':'e97-final-seed-production-256n-submission-v2','payload_job_id':'$payload_id',
'source_sha':'$SOURCE_SHA','origin_main_sha':'$ORIGIN_MAIN_SHA','local_main_sha':'$LOCAL_MAIN_SHA',
'payload_digest':'$payload_digest','run_id':'$RUN_ID','run_dir':'$RUN_DIR','payload_root':'$payload_root',
'nodes':256,'world_size':2048,'tasks_per_node':8,'partition':'batch','qos':'debug','time_limit':'02:00:00',
'submitted_unheld':True,'collector_job_id':None,'seed_step':2300930,
'seed_accepted_tokens':150793748480,'seed_bytes':7719680116,'seed_sha256':'$SEED_SHA256',
'token_migration_receipt':'$TOKEN_MIGRATION_RECEIPT_REL','token_migration_step':2303840,
'token_migration_total_tokens':199615447040,'token_migration_source_job_id':5134243,
'seed_cache':'$seed_cache','seed_attestation':'$seed_attestation',
'seed_attestation_sha256':'$seed_attestation_sha256','python_bin':'$EMENDER_PYTHON',
'fault_injection':'none','acceptance_shim':'none','unchanged_payload_retried':False}
open('$submission','w').write(json.dumps(v,sort_keys=True,indent=2)+'\n')
PY
squeue -j "$payload_id" -h -o '%i|%T|%D|%N|%P|%q|%l|%R' \
  | tee "$RUN_DIR/identity/squeue-submitted.txt"
payload_record=$(scontrol show job "$payload_id" -o)
printf '%s\n' "$payload_record" > "$RUN_DIR/identity/scontrol-submitted.txt"
[[ $payload_record == *"NumNodes=256"* && $payload_record == *"NumTasks=2048"* \
   && $payload_record == *"Partition=batch"* && $payload_record == *"QOS=debug"* \
   && $payload_record == *"TimeLimit=02:00:00"* ]] || {
  # The user explicitly requires that a submitted training payload remain
  # intact even if post-submit inspection fails.
  echo "submitted payload binding inspection failed; payload left intact: $payload_record" >&2
  exit 67
}
trap - EXIT
printf 'payload_job_id=%s\ncollector_job_id=none\nrun_id=%s\nrun_dir=%s\npayload_digest=%s\nsource_sha=%s\nmonitor_root=%s\nterminal_evidence_command=%s\n' \
  "$payload_id" "$RUN_ID" "$RUN_DIR" "$payload_digest" "$SOURCE_SHA" "$RUN_DIR" \
  "sacct -j $payload_id -X -P --format=JobIDRaw,State,ExitCode,NNodes,NTasks,Partition,QOS,Timelimit,Start,End,Elapsed"
