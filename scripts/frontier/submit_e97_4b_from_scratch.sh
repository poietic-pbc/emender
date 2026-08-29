#!/bin/bash
# Attended immutable submission for the fixed-world E97 4B Frontier campaign.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}
cd "$PROJECT_ROOT"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"

MODE=${MODE:-smoke}
SEED_MODE=0
EXPECTED_TARGET_STEPS=0
BASE=${E97_4B_FRONTIER_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch}
CONFIG_REL=configs/frontier/e97_4b_from_scratch.json
PAYLOAD_REL=scripts/frontier/e97_4b_from_scratch.sbatch
COLLECTOR_REL=scripts/frontier/e97_4b_from_scratch_collector.sh
case "$MODE" in
  smoke)
    NODES=${NODES:-1}; QOS=debug; TIME_LIMIT=02:00:00
    RUN_ID=${RUN_ID:-e97-4b-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
    ;;
  rung)
    [[ ${CONFIRM_RUNG:-0} == 1 ]] || { echo "rung requires CONFIRM_RUNG=1" >&2; exit 64; }
    NODES=4; QOS=debug; TIME_LIMIT=00:20:00
    RUN_ID=${RUN_ID:-e97-4b-rung-4n-$(date -u +%Y%m%dT%H%M%SZ)}
    ;;
  probe_b2|probe_b4|probe_b5|probe_b6|probe_b8)
    [[ ${CONFIRM_BATCH_PROBE:-0} == 1 ]] || { echo "batch probe requires CONFIRM_BATCH_PROBE=1" >&2; exit 64; }
    NODES=2; QOS=debug; TIME_LIMIT=00:20:00
    batch=${MODE#probe_b}
    RUN_ID=${RUN_ID:-e97-4b-probe-b${batch}-2n-$(date -u +%Y%m%dT%H%M%SZ)}
    ;;
  seed_import_32n_canary)
    [[ ${CONFIRM_SEED_IMPORT:-0} == 1 ]] || { echo "seed-import canary requires CONFIRM_SEED_IMPORT=1" >&2; exit 64; }
    NODES=32; QOS=debug; TIME_LIMIT=01:00:00; SEED_MODE=1
    CONFIG_REL=configs/frontier/e97_4b_seed_import_32n.json
    RUN_ID=${RUN_ID:-e97-4b-seed-import-w256-b1k32-r1}
    SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-imported-seeds/hf-8bf6f0e9241a3eb869676fdf6b92578ced8a6f00/checkpoints/step_012800_tokens_6710886400/checkpoint_step_012800_loss_2.8143.pt}
    SEED_SHA256=81fcc932e93df59a478e43b31afc5f0b310f58b8a5deab91a73e5be1a4925ed9
    ;;
  seed_continuation_32n_debug)
    [[ ${CONFIRM_SEED_CONTINUATION:-0} == 1 ]] || { echo "seed continuation requires CONFIRM_SEED_CONTINUATION=1" >&2; exit 64; }
    NODES=32; QOS=debug; TIME_LIMIT=02:00:00; SEED_MODE=1
    CONFIG_REL=configs/frontier/e97_4b_seed_continuation_32n.json
    RUN_ID=${RUN_ID:-e97-4b-seed-cont-w256-b1k32-r2}
    SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-seed-import-w256-b1k32-r1/train/checkpoint_step_013312_loss_2.7593.pt}
    SEED_SHA256=fa7b53f8ea31ca177aac0bba6b1fd174a970d8d8db68314c96333efd80a50ade
    ;;
  seed_scale_64n_canary)
    [[ ${CONFIRM_SEED_SCALE:-0} == 1 ]] || { echo "seed-scale canary requires CONFIRM_SEED_SCALE=1" >&2; exit 64; }
    NODES=64; QOS=debug; TIME_LIMIT=01:00:00; SEED_MODE=1
    CONFIG_REL=configs/frontier/e97_4b_seed_scale_64n.json
    RUN_ID=${RUN_ID:-e97-4b-seed-scale-w512-b1k32-r3}
    SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-seed-cont-w256-b1k32-r2/train/checkpoint_step_015360_loss_2.8084.pt}
    SEED_SHA256=9da49a274934135de4b5ac4f9265c0e6eff5ec7672fb7d9c3fb6388b0da18f16
    ;;
  hybrid_ddp_8n_canary)
    [[ ${CONFIRM_HYBRID_DDP:-0} == 1 ]] || { echo "hybrid-DDP canary requires CONFIRM_HYBRID_DDP=1" >&2; exit 64; }
    NODES=8; QOS=debug; TIME_LIMIT=02:00:00; SEED_MODE=1
    CONFIG_REL=configs/frontier/e97_4b_hybrid_ddp_8n.json
    RUN_ID=${RUN_ID:-e97-4b-hybrid-ddp-8n-b4k32-r4}
    SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-seed-scale-w512-b1k32-r3/train/checkpoint_step_016384_loss_2.6330.pt}
    SEED_SHA256=bb1a1350c37126793ad2de2b4477014c09f671a0c731184a4c5496f21cfcd8bd
    ;;
  hybrid_ddp_96n_debug)
    [[ ${CONFIRM_96N_CAMPAIGN:-0} == 1 ]] || { echo "96-node campaign debug requires CONFIRM_96N_CAMPAIGN=1" >&2; exit 64; }
    NODES=96; QOS=debug; TIME_LIMIT=02:00:00; SEED_MODE=1; EXPECTED_TARGET_STEPS=18176
    CONFIG_REL=configs/frontier/e97_4b_hybrid_ddp_96n_campaign.json
    RUN_ID=${RUN_ID:-e97-4b-hybrid-ddp-96n-campaign-r5-debug}
    SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-hybrid-ddp-8n-b4k32-r4/train/checkpoint_step_017152_loss_2.6381.pt}
    SEED_SHA256=cfb25848725912da577452e1a23fc91c541c6bb891145732d3b9c08f7ba9cfc9
    ;;
  hybrid_ddp_96n_production)
    [[ ${CONFIRM_96N_CAMPAIGN:-0} == 1 ]] || { echo "96-node production requires CONFIRM_96N_CAMPAIGN=1" >&2; exit 64; }
    [[ ${CAMPAIGN_PHASE:-} =~ ^[1-4]$ ]] || { echo "production requires CAMPAIGN_PHASE=1..4" >&2; exit 64; }
    NODES=96; QOS=normal; TIME_LIMIT=06:00:00; SEED_MODE=1
    CONFIG_REL=configs/frontier/e97_4b_hybrid_ddp_96n_campaign.json
    case "$CAMPAIGN_PHASE" in
      1) EXPECTED_TARGET_STEPS=21504;; 2) EXPECTED_TARGET_STEPS=24832;;
      3) EXPECTED_TARGET_STEPS=28160;; 4) EXPECTED_TARGET_STEPS=31488;;
    esac
    RUN_ID=${RUN_ID:-e97-4b-hybrid-ddp-96n-campaign-r$((5 + CAMPAIGN_PHASE))-p${CAMPAIGN_PHASE}}
    : "${SEED_CHECKPOINT:?production requires the inspected prior checkpoint path}"
    : "${SEED_SHA256:?production requires the inspected prior checkpoint SHA-256}"
    ;;
  hybrid_ddp_96n_short_production|hybrid_ddp_96n_short_debug)
    [[ ${CONFIRM_96N_CAMPAIGN:-0} == 1 ]] || { echo "96-node short campaign requires CONFIRM_96N_CAMPAIGN=1" >&2; exit 64; }
    [[ ${CAMPAIGN_PHASE:-} =~ ^([1-9]|1[0-3])$ ]] || { echo "short campaign requires CAMPAIGN_PHASE=1..13" >&2; exit 64; }
    NODES=96; TIME_LIMIT=02:00:00; SEED_MODE=1
    if [[ "$MODE" == hybrid_ddp_96n_short_debug ]]; then
      [[ ${CONFIRM_DEBUG_QOS_CAMPAIGN:-0} == 1 ]] || { echo "debug-QoS campaign override requires CONFIRM_DEBUG_QOS_CAMPAIGN=1" >&2; exit 64; }
      QOS=debug
    else
      QOS=normal
    fi
    CONFIG_REL=configs/frontier/e97_4b_hybrid_ddp_96n_campaign.json
    EXPECTED_TARGET_STEPS=$((18176 + CAMPAIGN_PHASE * 1024))
    RUN_ID=${RUN_ID:-e97-4b-hybrid-ddp-96n-${QOS}-s$(printf '%02d' "$CAMPAIGN_PHASE")}
    : "${SEED_CHECKPOINT:?short campaign requires the inspected prior checkpoint path}"
    : "${SEED_SHA256:?short campaign requires the inspected prior checkpoint SHA-256}"
    ;;
  hybrid_ddp_256n_debug)
    [[ ${CONFIRM_256N_CAMPAIGN:-0} == 1 ]] || { echo "256-node campaign requires CONFIRM_256N_CAMPAIGN=1" >&2; exit 64; }
    [[ ${CAMPAIGN_PHASE:-} =~ ^[1-6]$ ]] || { echo "256-node campaign requires CAMPAIGN_PHASE=1..6" >&2; exit 64; }
    NODES=256; QOS=debug; TIME_LIMIT=02:00:00; SEED_MODE=1
    CONFIG_REL=configs/frontier/e97_4b_hybrid_ddp_256n_campaign.json
    if (( CAMPAIGN_PHASE <= 5 )); then
      EXPECTED_TARGET_STEPS=$((20224 + CAMPAIGN_PHASE * 768))
    else
      EXPECTED_TARGET_STEPS=24448
    fi
    RUN_ID=${RUN_ID:-e97-4b-hybrid-ddp-256n-debug-c${CAMPAIGN_PHASE}}
    : "${SEED_CHECKPOINT:?256-node campaign requires the inspected prior checkpoint path}"
    : "${SEED_SHA256:?256-node campaign requires the inspected prior checkpoint SHA-256}"
    ;;
  matched_clock_32n)
    [[ ${CONFIRM_MATCHED_CLOCK:-0} == 1 ]] || { echo "matched-clock test requires CONFIRM_MATCHED_CLOCK=1" >&2; exit 64; }
    NODES=32; QOS=debug; TIME_LIMIT=02:00:00
    CONFIG_REL=configs/frontier/e97_4b_matched_clock_32n.json
    RUN_ID=${RUN_ID:-e97-4b-matched-clock-w256-b1k32-$(date -u +%Y%m%dT%H%M%SZ)}
    ;;
  bootstrap)
    [[ ${CONFIRM_BOOTSTRAP:-0} == 1 ]] || { echo "bootstrap requires CONFIRM_BOOTSTRAP=1" >&2; exit 64; }
    NODES=256; QOS=debug; TIME_LIMIT=00:30:00
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048-r3}
    ;;
  debug_continuation)
    [[ ${CONFIRM_DEBUG_CONTINUATION:-0} == 1 ]] || { echo "debug continuation requires CONFIRM_DEBUG_CONTINUATION=1" >&2; exit 64; }
    NODES=256; QOS=debug; TIME_LIMIT=02:00:00; TRAIN_MINUTES=105
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048-r3}
    ;;
  production|production_6h)
    [[ ${CONFIRM_PRODUCTION:-0} == 1 ]] || { echo "production requires CONFIRM_PRODUCTION=1" >&2; exit 64; }
    NODES=256; QOS=normal; TIME_LIMIT=06:00:00; TRAIN_MINUTES=345
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048-r3}
    ;;
  production_4h)
    [[ ${CONFIRM_PRODUCTION:-0} == 1 ]] || { echo "production requires CONFIRM_PRODUCTION=1" >&2; exit 64; }
    NODES=256; QOS=normal; TIME_LIMIT=04:00:00; TRAIN_MINUTES=225
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048-r3}
    ;;
  production_8h)
    [[ ${CONFIRM_PRODUCTION:-0} == 1 ]] || { echo "production requires CONFIRM_PRODUCTION=1" >&2; exit 64; }
    NODES=256; QOS=normal; TIME_LIMIT=08:00:00; TRAIN_MINUTES=465
    RUN_ID=${RUN_ID:-e97-4b-fresh-w2048-r3}
    ;;
  *) echo "invalid MODE" >&2; exit 64;;
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
if (( SEED_MODE == 1 )); then
  [[ -f "$SEED_CHECKPOINT" ]] || { echo "missing imported seed checkpoint" >&2; exit 66; }
  observed_seed_sha=$(sha256sum "$SEED_CHECKPOINT" | awk '{print $1}')
  [[ "$observed_seed_sha" == "$SEED_SHA256" ]] || { echo "imported seed SHA-256 mismatch" >&2; exit 66; }
  mkdir -p "$RUN_DIR/train"
  if [[ -e "$RUN_DIR/train/latest.pt" || -L "$RUN_DIR/train/latest.pt" ]]; then
    [[ -L "$RUN_DIR/train/latest.pt" && $(readlink -f "$RUN_DIR/train/latest.pt") == "$SEED_CHECKPOINT" ]] || {
      echo "seed-import run already has a different latest authority" >&2; exit 65;
    }
  else
    seed_link_tmp="$RUN_DIR/train/.latest.seed-import.$$.tmp"
    ln -s "$SEED_CHECKPOINT" "$seed_link_tmp"
    mv -f "$seed_link_tmp" "$RUN_DIR/train/latest.pt"
  fi
fi
if [[ -e "$RUN_DIR/train/latest.pt" || -L "$RUN_DIR/train/latest.pt" ]]; then
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "existing resume authority is not a readable symlink" >&2; exit 65;
  }
  [[ ${CONFIRM_RESUME:-0} == 1 ]] || { echo "existing run requires CONFIRM_RESUME=1" >&2; exit 64; }
fi

mkdir -p "$BASE"/{authority,payloads,runs} "$RUN_DIR"/{identity,logs,terminal}
if (( SEED_MODE == 1 )); then
  printf '%s  %s\n' "$SEED_SHA256" "$SEED_CHECKPOINT" > "$RUN_DIR/identity/imported-seed.sha256"
fi
CORPUS_RECEIPT="$BASE/authority/commapile-mainmix-v0.1-sha256.json"
asset_hashes=$(sha256sum "$CONFIG_REL" "$PAYLOAD_REL" "$COLLECTOR_REL" train.py \
  scripts/frontier/activate_emender_frontier.sh scripts/frontier/frontier_runtime_env.sh)
payload_digest=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' \
  "$SOURCE_SHA" "$MODE" "$NODES" "$PARTITION" "$QOS" "$TIME_LIMIT" "${TRAIN_MINUTES:-0}" "$EXPECTED_TARGET_STEPS" "$asset_hashes" \
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
  --export=ALL,REPO="$repo_exact",RUN_ID="$RUN_ID",RUN_DIR="$RUN_DIR",RUN_MODE="$MODE",TRAIN_MINUTES="${TRAIN_MINUTES:-0}",CONFIG="$repo_exact/$CONFIG_REL",CORPUS_RECEIPT="$CORPUS_RECEIPT",EXPECTED_NODES="$NODES",EXPECTED_WORLD_SIZE="$WORLD_SIZE",EXPECTED_PARTITION="$PARTITION",EXPECTED_QOS="$QOS",EXPECTED_TIME_LIMIT="$TIME_LIMIT",EXPECTED_TARGET_STEPS="$EXPECTED_TARGET_STEPS" \
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
  --export=ALL,PAYLOAD_JOB_ID="$payload_id",RUN_DIR="$RUN_DIR",REPO="$repo_exact",CONFIG="$repo_exact/$CONFIG_REL",EXPECTED_WORLD_SIZE="$WORLD_SIZE",EXPECTED_PARTITION="$PARTITION",EXPECTED_QOS="$QOS" \
  "$repo_exact/$COLLECTOR_REL")

record="$RUN_DIR/identity/submission-${payload_id}.json"
"$EMENDER_PYTHON" - "$record" <<PY
import json,os,sys,time
v={'schema':'emender-e97-4b-frontier-submission-v1','payload_job_id':'$payload_id',
'collector_job_id':'$collector_id','source_sha':'$SOURCE_SHA','payload_digest':'$payload_digest',
'run_id':'$RUN_ID','run_dir':'$RUN_DIR','mode':'$MODE','nodes':$NODES,'world_size':$WORLD_SIZE,
'target_steps':$EXPECTED_TARGET_STEPS,
'partition':'$PARTITION','qos':'$QOS','time_limit':'$TIME_LIMIT','submitted_unheld':True,
'fixed_world':True,'from_scratch_if_latest_absent':True,'submitted_unix':time.time()}
p=sys.argv[1]+'.tmp'; open(p,'w').write(json.dumps(v,sort_keys=True)+'\n'); os.replace(p,sys.argv[1])
PY
printf 'payload_job_id=%s\ncollector_job_id=%s\nrun_dir=%s\nsource_sha=%s\npayload_digest=%s\n' \
  "$payload_id" "$collector_id" "$RUN_DIR" "$SOURCE_SHA" "$payload_digest"
