#!/bin/bash
# Submit one attended fixed-world E97 4B Pi SFT qualification or canary.
set -euo pipefail
usage() { echo "usage: $0 {qualification|canary} AUTHORITY_ROOT PACK_ROOT [RUN_ID]" >&2; exit 64; }
[[ $# -ge 3 && $# -le 4 ]] || usage
MODE=$1; AUTHORITY_ROOT=$(realpath "$2"); PACK_ROOT=$(realpath "$3")
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo "tracked source must be committed" >&2; exit 65; }
SOURCE_COMMIT=$(git rev-parse HEAD)
git merge-base --is-ancestor "$SOURCE_COMMIT" origin/main || { echo "source commit is not on origin/main" >&2; exit 65; }
AUTHORITY_SHA256=$(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}')
PACK_SHA256=$(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}')
case "$MODE" in
  qualification) NODES=1; WORLD=8; STEPS=${STEPS:-8}; SAVE_EVERY=${SAVE_EVERY:-8}; DILOCO_K=${DILOCO_K:-8} ;;
  canary) NODES=8; WORLD=64; STEPS=${STEPS:-64}; SAVE_EVERY=${SAVE_EVERY:-32}; DILOCO_K=${DILOCO_K:-8} ;;
  *) usage ;;
esac
(( STEPS % DILOCO_K == 0 && SAVE_EVERY % DILOCO_K == 0 )) || { echo "steps must be K-aligned" >&2; exit 64; }
RUN_ID=${4:-e97-4b-pi-sft-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$RUN_ID" != */* ]] || exit 64
RUN_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-pi-instruction/runs/$RUN_ID
[[ ! -e "$RUN_ROOT" ]] || { echo "run root already exists: $RUN_ROOT" >&2; exit 65; }
mkdir -p "$RUN_ROOT"/{identity,logs}
cat > "$RUN_ROOT/identity/submission.json" <<EOF
{"schema":"emender-e97-4b-pi-sft-submission-v1","mode":"$MODE","source_commit":"$SOURCE_COMMIT","nodes":$NODES,"world_size":$WORLD,"steps":$STEPS,"save_every":$SAVE_EVERY,"diloco_k":$DILOCO_K,"authority_manifest_sha256":"$AUTHORITY_SHA256","pack_manifest_sha256":"$PACK_SHA256"}
EOF
export REPO SOURCE_COMMIT RUN_ROOT EXPECTED_NODES=$NODES EXPECTED_WORLD_SIZE=$WORLD
export AUTHORITY_ROOT AUTHORITY_SHA256 PACK_ROOT PACK_SHA256 STEPS SAVE_EVERY DILOCO_K
export EXPECTED_PARTITION=batch EXPECTED_QOS=debug LR=${LR:-0.00001} WARMUP_STEPS=${WARMUP_STEPS:-8}
job=$(sbatch --parsable --nodes="$NODES" --time=02:00:00 --partition=batch --qos=debug --no-requeue \
  --export=ALL scripts/frontier/e97_4b_pi_sft.sbatch)
printf '%s\n' "$job" | tee "$RUN_ROOT/identity/job-id.txt"
printf 'RUN_ROOT=%s\nJOB_ID=%s\n' "$RUN_ROOT" "$job"
