#!/bin/bash
# Attended immutable submission for the fixed-world E97 4B Frontier campaign.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}
cd "$PROJECT_ROOT"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"

MODE=${MODE:-smoke}
BASE=${E97_4B_FRONTIER_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch}
CONFIG_REL=configs/frontier/e97_4b_from_scratch.json
PAYLOAD_REL=scripts/frontier/e97_4b_from_scratch.sbatch
COLLECTOR_REL=scripts/frontier/e97_4b_from_scratch_collector.sh
case "$MODE" in
  smoke)
    NODES=${NODES:-1}; QOS=debug; TIME_LIMIT=02:00:00
    RUN_ID=${RUN_ID:-e97-4b-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
    ;;
  bootstrap)
    [[ ${CONFIRM_BOOTSTRAP:-0} == 1 ]] || { echo "bootstrap requires CONFIRM_BOOTSTRAP=1" >&2; exit 64; }
    NODES=256; QOS=debug; TIME_LIMIT=00:30:00
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048}
    ;;
  production)
    [[ ${CONFIRM_PRODUCTION:-0} == 1 ]] || { echo "production requires CONFIRM_PRODUCTION=1" >&2; exit 64; }
    NODES=256; QOS=normal; TIME_LIMIT=06:05:00
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048}
    ;;
  *) echo "MODE must be smoke, bootstrap, or production" >&2; exit 64;;
esac
PARTITION=batch
WORLD_SIZE=$((NODES * 8))
RUN_DIR=${RUN_DIR:-$BASE/runs/$RUN_ID}
[[ "$RUN_ID" != */* && "$RUN_DIR" == /* ]] || { echo "invalid run identity" >&2; exit 64; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked source must be clean" >&2; exit 64; }
SOURCE_SHA=$(git rev-parse HEAD)
LOCAL_MAIN_SHA=$(git rev-parse main)
ORIGIN_MAIN_SHA=$(git rev-parse origin/main)
[[ "$SOURCE_SHA" == "$LOCAL_MAIN_SHA" && "$SOURCE_SHA" == "$ORIGIN_MAIN_SHA" ]] || {
  echo "submission requires HEAD == main == origin/main" >&2; exit 64;
}
if [[ -e "$RUN_DIR/train/latest.pt" || -L "$RUN_DIR/train/latest.pt" ]]; then
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "existing resume authority is not a readable symlink" >&2; exit 65;
  }
  [[ ${CONFIRM_RESUME:-0} == 1 ]] || { echo "existing run requires CONFIRM_RESUME=1" >&2; exit 64; }
fi

mkdir -p "$BASE"/{authority,payloads,runs} "$RUN_DIR"/{identity,logs,terminal}
CORPUS_RECEIPT="$BASE/authority/commapile-mainmix-v0.1-sha256.json"
asset_hashes=$(sha256sum "$CONFIG_REL" "$PAYLOAD_REL" "$COLLECTOR_REL" train.py \
  scripts/frontier/activate_emender_frontier.sh scripts/frontier/frontier_runtime_env.sh)
payload_digest=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n' \
  "$SOURCE_SHA" "$MODE" "$NODES" "$PARTITION" "$QOS" "$asset_hashes" \
  | sha256sum | awk '{print $1}')
payload_root="$BASE/payloads/$payload_digest"
repo_exact="$payload_root/repo"
if [[ ! -e "$repo_exact" ]]; then
  mkdir -p "$payload_root"
  printf '%s\n' "$asset_hashes" > "$payload_root/source-assets.sha256"
  git clone --shared --no-checkout "$PROJECT_ROOT" "$repo_exact"
  git -C "$repo_exact" checkout --detach "$SOURCE_SHA"
fi
[[ $(git -C "$repo_exact" rev-parse HEAD) == "$SOURCE_SHA" ]] || { echo "immutable checkout mismatch" >&2; exit 66; }
[[ -z $(git -C "$repo_exact" status --porcelain --untracked-files=no) ]] || { echo "immutable checkout dirty" >&2; exit 66; }

export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312}
source "$repo_exact/scripts/frontier/activate_emender_frontier.sh"
: "${EMENDER_PYTHON:?immutable activation did not set EMENDER_PYTHON}"
"$EMENDER_PYTHON" -m py_compile "$repo_exact/train.py"
bash -n "$repo_exact/$PAYLOAD_REL" "$repo_exact/$COLLECTOR_REL"

payload_id=$(sbatch --parsable --no-requeue --signal=B:TERM@300 \
  -A bif148 -p "$PARTITION" -q "$QOS" -J "e97-4b-${MODE}" -N"$NODES" -t "$TIME_LIMIT" \
  --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest \
  --output="$RUN_DIR/logs/sbatch-%j.out" --error="$RUN_DIR/logs/sbatch-%j.err" \
  --export=ALL,REPO="$repo_exact",RUN_ID="$RUN_ID",RUN_DIR="$RUN_DIR",RUN_MODE="$MODE",CONFIG="$repo_exact/$CONFIG_REL",CORPUS_RECEIPT="$CORPUS_RECEIPT",EXPECTED_NODES="$NODES",EXPECTED_WORLD_SIZE="$WORLD_SIZE",EXPECTED_PARTITION="$PARTITION",EXPECTED_QOS="$QOS",EXPECTED_TIME_LIMIT="$TIME_LIMIT" \
  "$repo_exact/$PAYLOAD_REL")

queued="$RUN_DIR/identity/squeue-${payload_id}-queued.txt"
for _ in $(seq 1 20); do
  squeue -j "$payload_id" -h -o '%i|%T|%N|%P|%q|%R' > "$queued" || true
  [[ -s "$queued" ]] && break
  sleep 1
done
cat "$queued"
grep -F "|$PARTITION|$QOS|" "$queued" >/dev/null || {
  echo "failed to retain queued/running Partition and QOS evidence" >&2; exit 66;
}
collector_id=$(sbatch --parsable --no-requeue -A bif148 -p batch -q normal -N1 -t 00:10:00 \
  --dependency="afterany:$payload_id" -J e97-4b-collector \
  --output="$RUN_DIR/logs/collector-%j.out" --error="$RUN_DIR/logs/collector-%j.err" \
  --export=ALL,PAYLOAD_JOB_ID="$payload_id",RUN_DIR="$RUN_DIR",EXPECTED_PARTITION="$PARTITION",EXPECTED_QOS="$QOS" \
  "$repo_exact/$COLLECTOR_REL")

record="$RUN_DIR/identity/submission-${payload_id}.json"
"$EMENDER_PYTHON" - "$record" <<PY
import json,os,sys,time
v={'schema':'emender-e97-4b-frontier-submission-v1','payload_job_id':'$payload_id',
'collector_job_id':'$collector_id','source_sha':'$SOURCE_SHA','payload_digest':'$payload_digest',
'run_id':'$RUN_ID','run_dir':'$RUN_DIR','mode':'$MODE','nodes':$NODES,'world_size':$WORLD_SIZE,
'partition':'$PARTITION','qos':'$QOS','time_limit':'$TIME_LIMIT','submitted_unheld':True,
'fixed_world':True,'from_scratch_if_latest_absent':True,'submitted_unix':time.time()}
p=sys.argv[1]+'.tmp'; open(p,'w').write(json.dumps(v,sort_keys=True)+'\n'); os.replace(p,sys.argv[1])
PY
printf 'payload_job_id=%s\ncollector_job_id=%s\nrun_dir=%s\nsource_sha=%s\npayload_digest=%s\n' \
  "$payload_id" "$collector_id" "$RUN_DIR" "$SOURCE_SHA" "$payload_digest"
