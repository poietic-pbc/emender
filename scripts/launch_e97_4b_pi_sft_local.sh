#!/bin/bash
# One-node/eight-GPU masked Pi SFT with pinned-CPU Schedule-Free state.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
MODE=${MODE:-qualification}
WORLD_SIZE=8
PARENT=${PARENT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/checkpoint_step_024448_loss_2.3981.pt}
PARENT_SHA256=${PARENT_SHA256:-3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568}
SOURCE_ARGS=${SOURCE_ARGS:-$(dirname "$PARENT")/args.json}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/mnt/nvme1n1/erikg/sft/e97-4b-pi-instruction-mix-v2}
PACK_ROOT=${PACK_ROOT:-$AUTHORITY_ROOT/packs-4096-v1}
AUTHORITY_SHA256=${AUTHORITY_SHA256:-$(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}')}
PACK_SHA256=${PACK_SHA256:-$(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}')}
LR=${LR:-0.00001}
WARMUP_STEPS=${WARMUP_STEPS:-8}
DILOCO_K=${DILOCO_K:-8}
SAVE_EVERY=${SAVE_EVERY:-8}
case "$MODE" in
  qualification) STEPS=${STEPS:-8} ;;
  canary)
    [[ ${CONFIRM_CANARY:-0} == 1 ]] || { echo "canary requires CONFIRM_CANARY=1" >&2; exit 64; }
    STEPS=${STEPS:-64}; SAVE_EVERY=${SAVE_EVERY:-32}
    ;;
  *) echo "MODE must be qualification or canary" >&2; exit 64;;
esac
(( STEPS > 0 && DILOCO_K > 0 && SAVE_EVERY > 0 )) || exit 64
(( STEPS % DILOCO_K == 0 && SAVE_EVERY % DILOCO_K == 0 )) || {
  echo "steps and checkpoint cadence must be K-aligned" >&2; exit 64;
}
[[ -r "$PARENT" && -r "$SOURCE_ARGS" ]] || { echo "parent or args.json missing" >&2; exit 66; }
[[ $(sha256sum "$PARENT" | awk '{print $1}') == "$PARENT_SHA256" ]] || {
  echo "parent SHA-256 mismatch" >&2; exit 66;
}
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_SHA256" ]] || exit 66
[[ $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_SHA256" ]] || exit 66
SOURCE_COMMIT=$(git rev-parse HEAD)
RUN_ID=${RUN_ID:-e97-4b-pi-sft-local-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/runs/$RUN_ID}
[[ "$RUN_ROOT" == /* && "$RUN_ID" != */* && ! -e "$RUN_ROOT" ]] || {
  echo "invalid or existing run root: $RUN_ROOT" >&2; exit 65;
}
mkdir -p "$RUN_ROOT"/{checkpoints,identity,logs,terminal}
cat > "$RUN_ROOT/identity/launch.json" <<EOF
{"schema":"emender-e97-4b-pi-sft-local-launch-v1","mode":"$MODE","source_commit":"$SOURCE_COMMIT","parent_sha256":"$PARENT_SHA256","authority_sha256":"$AUTHORITY_SHA256","pack_sha256":"$PACK_SHA256","world_size":8,"steps":$STEPS,"diloco_k":$DILOCO_K,"optimizer_state_storage":"pinned-cpu"}
EOF
COMMAND=(
  torchrun --standalone --nproc_per_node="$WORLD_SIZE"
  scripts/numa_local_rank_exec.py -- scripts/train_e97_4b_pi_sft.py
  --parent-checkpoint "$PARENT" --parent-sha256 "$PARENT_SHA256"
  --source-args-json "$SOURCE_ARGS" --source-commit "$SOURCE_COMMIT"
  --authority-root "$AUTHORITY_ROOT" --authority-sha256 "$AUTHORITY_SHA256"
  --pack-root "$PACK_ROOT" --pack-sha256 "$PACK_SHA256"
  --output-root "$RUN_ROOT/checkpoints" --log-jsonl "$RUN_ROOT/logs/training.jsonl"
  --steps "$STEPS" --save-every "$SAVE_EVERY" --diloco-k "$DILOCO_K"
  --context-size 4096 --lr "$LR" --warmup-steps "$WARMUP_STEPS"
  --sampler-key 974003 --island-size 8 --merge-bucket-numel 67108864
  --offload-schedulefree-state --schedulefree-offload-bucket-numel 67108864
)
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/identity/command.txt"; printf '\n' >> "$RUN_ROOT/identity/command.txt"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
if [[ ${ACQUIRE_GPUS:-1} == 1 ]]; then
  eval "$(scripts/gpu_lease.sh acquire 8 --no-wait)"
fi
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=/tmp/e97-4b-pi-sft-${RUN_ID}
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/run.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/return-code.txt"
if (( rc == 0 )); then
  latest=$(readlink -f "$RUN_ROOT/checkpoints/latest.pt")
  python scripts/verify_e97_4b_pi_sft_checkpoint.py "$latest" \
    --output "$RUN_ROOT/terminal/checkpoint.reload.json"
  sha256sum "$latest" > "$RUN_ROOT/terminal/checkpoint.sha256"
  echo "LOCAL_PI_SFT_QUALIFICATION_COMPLETE checkpoint=$latest"
fi
exit "$rc"
