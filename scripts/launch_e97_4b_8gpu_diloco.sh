#!/usr/bin/env bash
# Interruptible fixed-world 8-GPU DiLoCo launcher for the qualified 4.046B E97.
#
# Safe default:
#   MODE=smoke  -> 2 steps with K=1 (collective/checkpoint qualification only)
# Full pilot:
#   MODE=1b CONFIRM_1B=1 [sampler identity...] scripts/launch_e97_4b_8gpu_diloco.sh
# Effectively unbounded exact resume:
#   MODE=continuous CONFIRM_CONTINUOUS=1 RESUME=... [sampler identity...] scripts/launch_e97_4b_8gpu_diloco.sh
# Graceful early stop:
#   scripts/request_graceful_stop.sh LOGDIR
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-smoke}"
WORLD_SIZE=8
BATCH_SIZE="${BATCH_SIZE:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-0.00047431158698290157}"
SEED="${SEED:-42}"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"
TOKENIZER="${TOKENIZER:-p50k_base}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in
  smoke)
    STEPS="${STEPS:-2}"
    DILOCO_K="${DILOCO_K:-1}"
    SAVE_EVERY="${SAVE_EVERY:-2}"
    LOG_EVERY="${LOG_EVERY:-1}"
    ;;
  1b)
    [[ "${CONFIRM_1B:-0}" == 1 ]] || {
      echo "MODE=1b requires CONFIRM_1B=1" >&2; exit 64;
    }
    TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
    tokens_per_step=$((WORLD_SIZE * BATCH_SIZE * CHUNK_SIZE * GRAD_ACCUM))
    STEPS="${STEPS:-$(((TARGET_TOKENS + tokens_per_step - 1) / tokens_per_step))}"
    # Preserve the proven campaign's approximately 2.05M local tokens/merge:
    # old 4*2048*250=2,048,000; this launch 32*2048*32=2,097,152.
    DILOCO_K="${DILOCO_K:-32}"
    SAVE_EVERY="${SAVE_EVERY:-256}"
    LOG_EVERY="${LOG_EVERY:-4}"
    ;;
  continuous)
    [[ "${CONFIRM_CONTINUOUS:-0}" == 1 ]] || {
      echo "MODE=continuous requires CONFIRM_CONTINUOUS=1" >&2; exit 64;
    }
    [[ -n "$RESUME" && -e "$RESUME" ]] || {
      echo "MODE=continuous requires RESUME naming a readable checkpoint" >&2; exit 66;
    }
    # An operationally unbounded ceiling (~524T aggregate tokens). The attended
    # graceful-stop path, rather than this ceiling, owns normal termination.
    STEPS="${STEPS:-1000000000}"
    DILOCO_K="${DILOCO_K:-32}"
    SAVE_EVERY="${SAVE_EVERY:-256}"
    LOG_EVERY="${LOG_EVERY:-4}"
    ;;
  *)
    echo "MODE must be smoke, 1b, or continuous" >&2; exit 64;;
esac

for value_name in BATCH_SIZE CHUNK_SIZE GRAD_ACCUM STEPS DILOCO_K SAVE_EVERY LOG_EVERY; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer, got $value" >&2; exit 64;
  }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a non-negative integer, got $SEED" >&2; exit 64;
}
(( SAVE_EVERY % DILOCO_K == 0 )) || {
  echo "SAVE_EVERY must be a multiple of DILOCO_K" >&2; exit 64;
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="${LOGDIR:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_${MODE}_${stamp}}"
OUTPUT="${OUTPUT:-$LOGDIR/runs}"
NAME="${NAME:-e97_4b_${MODE}_8gpu_diloco}"
mkdir -p "$LOGDIR" "$OUTPUT"

SAMPLER_ARGS=()
if [[ -n "${SAMPLER_SCHEMA:-}" ]]; then
  : "${SAMPLER_CORPUS_SHA256:?counter sampler requires SAMPLER_CORPUS_SHA256}"
  : "${SAMPLER_TOKENIZER_SHA256:?counter sampler requires SAMPLER_TOKENIZER_SHA256}"
  : "${SAMPLER_KEY:?counter sampler requires SAMPLER_KEY}"
  SAMPLER_ARGS=(
    --sampler_schema "$SAMPLER_SCHEMA"
    --sampler_corpus_sha256 "$SAMPLER_CORPUS_SHA256"
    --sampler_tokenizer_sha256 "$SAMPLER_TOKENIZER_SHA256"
    --sampler_key "$SAMPLER_KEY"
    --sampler_data_world_size "$WORLD_SIZE"
  )
elif [[ "$MODE" != smoke && "${ALLOW_LEGACY_SAMPLER:-0}" != 1 ]]; then
  echo "MODE=$MODE requires a frozen counter sampler identity; set SAMPLER_SCHEMA and hashes." >&2
  echo "Set ALLOW_LEGACY_SAMPLER=1 only for a deliberately non-reproducible pilot." >&2
  exit 64
fi

RESUME_ARGS=()
[[ -z "$RESUME" ]] || RESUME_ARGS=(--resume "$RESUME")

TRAIN_ARGS=(
  torchrun --standalone --nproc_per_node="$WORLD_SIZE"
  scripts/numa_local_rank_exec.py -- train.py
  "${RESUME_ARGS[@]}"
  --level E97
  --params 4b
  --dim 3840
  --depth 18
  --n_heads 60
  --n_state 64
  --expansion 1.0
  --use_gate 1
  --gate_activation silu
  --linear_state 0
  --mlp_ratio 2.5
  --mlp_multiple 64
  --use_triton 1
  --optimizer schedulefree
  --offload_schedulefree_state
  --lr "$LR"
  --weight_decay 0.01
  --warmup_steps 0
  --bf16
  --batch_size "$BATCH_SIZE"
  --chunk_size "$CHUNK_SIZE"
  --grad_accum "$GRAD_ACCUM"
  --gradient_checkpointing
  --gradient_checkpoint_group_size 2
  # The sequential Triton wrapper currently executes its proven interval 16.
  --checkpoint_interval 16
  --projection_chunk_size 512
  --loss_chunk_size 128
  --grad_clip 1.0
  --seed "$SEED"
  --compile_warmup_steps 1
  --data "$DATA"
  --tokenizer "$TOKENIZER"
  "${SAMPLER_ARGS[@]}"
  --diloco
  --diloco_k "$DILOCO_K"
  --diloco_outer_optimizer avg
  --diloco_outer_lr 1.0
  --diloco_outer_beta 0.0
  --diloco_merge_bucket_numel 67108864
  --steps "$STEPS"
  --save_every "$SAVE_EVERY"
  --keep_checkpoints 3
  --log_every "$LOG_EVERY"
  --output "$OUTPUT"
)

ENV_ARGS=(
  env
  NCCL_P2P_DISABLE=1
  TORCH_NCCL_ENABLE_MONITORING=0
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
  PYTORCH_ALLOC_CONF=expandable_segments:True
  OMP_NUM_THREADS=4
  TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache
)

actual_tokens=$((STEPS * WORLD_SIZE * BATCH_SIZE * CHUNK_SIZE * GRAD_ACCUM))
cat >"$LOGDIR/launch_receipt.txt" <<EOF
mode=$MODE
world_size=$WORLD_SIZE
batch_size=$BATCH_SIZE
chunk_size=$CHUNK_SIZE
grad_accum=$GRAD_ACCUM
steps=$STEPS
target_or_smoke_tokens=$actual_tokens
diloco_k=$DILOCO_K
save_every=$SAVE_EVERY
lr=$LR
seed=$SEED
shape=d3840-L18-H60-n64-mlp2.5
source_commit=$(git rev-parse HEAD)
interrupt_command=scripts/request_graceful_stop.sh $LOGDIR
EOF

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'LOGDIR=%q\n' "$LOGDIR"
  printf 'OUTPUT=%q\n' "$OUTPUT"
  printf 'ACTUAL_TOKENS=%q\n' "$actual_tokens"
  printf 'COMMAND='; printf '%q ' "${ENV_ARGS[@]}" "${TRAIN_ARGS[@]}"; printf '\n'
  exit 0
elif [[ "$DRY_RUN" != 0 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2; exit 64
fi

# Do not wrap launch_detached_run.sh in command substitution: an asynchronous
# descendant can retain the substitution pipe and make this launcher wait for
# the entire training run. A regular redirected invocation returns immediately.
launcher_out="$LOGDIR/launcher.out"
scripts/launch_detached_run.sh \
  --name "$NAME" --gpus 8 --logdir "$LOGDIR" -- \
  "${ENV_ARGS[@]}" "${TRAIN_ARGS[@]}" >"$launcher_out"
pid="$(tr -d '[:space:]' <"$launcher_out")"
[[ "$pid" =~ ^[0-9]+$ ]] || {
  echo "detached launcher returned an invalid pid: $pid" >&2; exit 70;
}
printf 'pid=%s\nlogdir=%s\nlog=%s\nstop=%q\n' \
  "$pid" "$LOGDIR" "$LOGDIR/run.log" \
  "scripts/request_graceful_stop.sh $LOGDIR"
