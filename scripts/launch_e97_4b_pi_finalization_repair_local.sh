#!/bin/bash
# Fresh-optimizer finalization repair stage from a behaviorally evaluated SFT checkpoint.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
[[ ${CONFIRM_REPAIR:-0} == 1 ]] || { echo "repair requires CONFIRM_REPAIR=1" >&2; exit 64; }
: "${PARENT_CHECKPOINT:?set the selected SFT stage parent}"
: "${PARENT_SHA256:?set the selected parent SHA-256}"
SOURCE_ARGS=${SOURCE_ARGS:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/mnt/nvme1n1/erikg/sft/e97-4b-pi-finalization-repair-v1}
AUTHORITY_SHA256=${AUTHORITY_SHA256:-887120163f3531e98f9606bdbf2d02ff171d4811c60df77c9d3881c247056129}
PACK_ROOT=${PACK_ROOT:-$AUTHORITY_ROOT/packs-4096-v1}
PACK_SHA256=${PACK_SHA256:-7307fe07fcbd957c7ee27e23037f4c7d3a57de63e5f2f4b3a0ea1b0ec2b015a0}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git rev-parse HEAD)}
RUN_ID=${RUN_ID:-e97-4b-pi-finalization-repair-$(date -u +%Y%m%dT%H%M%SZ)}
BASE=${BASE:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local}
RUN_ROOT=$BASE/runs/$RUN_ID
STEPS=${STEPS:-64}; SAVE_EVERY=${SAVE_EVERY:-32}; DILOCO_K=${DILOCO_K:-8}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-3}; SAMPLER_KEY=${SAMPLER_KEY:-974103}
LR=${LR:-5e-6}; WARMUP_STEPS=${WARMUP_STEPS:-8}; GRAD_CLIP=${GRAD_CLIP:-1.0}
[[ "$RUN_ROOT" == /* && ! -e "$RUN_ROOT" ]] || { echo "run root must be new and absolute" >&2; exit 64; }
for path in "$PARENT_CHECKPOINT" "$SOURCE_ARGS" "$AUTHORITY_ROOT/manifest.json" "$PACK_ROOT/manifest.json"; do
  [[ -r "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
[[ $(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}') == "$PARENT_SHA256" ]] || { echo "parent SHA-256 mismatch" >&2; exit 65; }
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_SHA256" ]] || exit 65
[[ $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_SHA256" ]] || exit 65
[[ $((STEPS % DILOCO_K)) == 0 && $((SAVE_EVERY % DILOCO_K)) == 0 ]] || exit 64
mkdir -p "$RUN_ROOT"/{checkpoints,identity,logs,terminal}
cat > "$RUN_ROOT/identity/launch.json" <<EOF
{"schema":"emender-e97-4b-pi-finalization-repair-local-v1","run_id":"$RUN_ID","source_commit":"$SOURCE_COMMIT","parent_sha256":"$PARENT_SHA256","authority_sha256":"$AUTHORITY_SHA256","pack_sha256":"$PACK_SHA256","steps":$STEPS,"lr":$LR}
EOF
COMMAND=(
  torchrun --standalone --nproc_per_node=8
  scripts/numa_local_rank_exec.py -- scripts/train_e97_4b_pi_sft.py
  --new-stage-from "$PARENT_CHECKPOINT"
  --parent-checkpoint "$PARENT_CHECKPOINT" --parent-sha256 "$PARENT_SHA256"
  --source-args-json "$SOURCE_ARGS" --source-commit "$SOURCE_COMMIT"
  --authority-root "$AUTHORITY_ROOT" --authority-sha256 "$AUTHORITY_SHA256"
  --pack-root "$PACK_ROOT" --pack-sha256 "$PACK_SHA256"
  --output-root "$RUN_ROOT/checkpoints" --log-jsonl "$RUN_ROOT/logs/training.jsonl"
  --steps "$STEPS" --save-every "$SAVE_EVERY" --keep-checkpoints "$KEEP_CHECKPOINTS"
  --diloco-k "$DILOCO_K" --context-size 4096 --lr "$LR" --warmup-steps "$WARMUP_STEPS" --grad-clip "$GRAD_CLIP"
  --sampler-key "$SAMPLER_KEY" --island-size 8 --merge-bucket-numel 67108864
  --offload-schedulefree-state --schedulefree-offload-bucket-numel 67108864
)
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/identity/command.txt"; printf '\n' >> "$RUN_ROOT/identity/command.txt"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
eval "$(scripts/gpu_lease.sh acquire 8 --no-wait)"
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=/tmp/e97-4b-pi-final-repair-${RUN_ID}
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/run.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/return-code.txt"
(( rc == 0 )) || exit "$rc"
checkpoint=$(readlink -f "$RUN_ROOT/checkpoints/latest.pt")
digest=$(sha256sum "$checkpoint" | awk '{print $1}')
PYTHONPATH="$ROOT" python scripts/verify_e97_4b_pi_sft_checkpoint.py "$checkpoint" \
  --expected-sha256 "$digest" --output "$RUN_ROOT/terminal/checkpoint.reload.json"
printf '%s  %s\n' "$digest" "$checkpoint" > "$RUN_ROOT/terminal/checkpoint.sha256"
echo "LOCAL_PI_FINALIZATION_REPAIR_COMPLETE checkpoint=$checkpoint"
