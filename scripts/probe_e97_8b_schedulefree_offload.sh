#!/usr/bin/env bash
# One-GPU capacity probe for the square-readout E97 96x48 7.94B candidate.
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

: "${DATA:?set DATA to a text corpus path}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
STEPS="${STEPS:-1}"
LOG_EVERY="${LOG_EVERY:-1}"
PROBE_MEMORY="${PROBE_MEMORY:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-64}"
GRADIENT_CHECKPOINT_GROUP_SIZE="${GRADIENT_CHECKPOINT_GROUP_SIZE:-2}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-512}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-256}"
OUTPUT="${OUTPUT:-/tmp/emender-e97-8b-schedulefree-offload-probe-${CHUNK_SIZE}}"

probe_args=()
if [[ "$PROBE_MEMORY" == "1" ]]; then
  probe_args+=(--probe_memory)
elif [[ "$PROBE_MEMORY" != "0" ]]; then
  echo "PROBE_MEMORY must be 0 or 1" >&2
  exit 2
fi

exec env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  "$PYTHON_BIN" train.py \
    --data "$DATA" \
    --tokenizer p50k_base \
    --level E97 \
    --params 8b \
    --dim 4608 \
    --depth 25 \
    --n_heads 96 \
    --n_state 48 \
    --use_gate 1 \
    --gate_activation silu \
    --linear_state 0 \
    --mlp_ratio 2.5 \
    --mlp_multiple 64 \
    --batch_size "$BATCH_SIZE" \
    --chunk_size "$CHUNK_SIZE" \
    --steps "$STEPS" \
    --log_every "$LOG_EVERY" \
    --optimizer schedulefree \
    --offload_schedulefree_state \
    --bf16 \
    --use_triton 1 \
    --gradient_checkpointing \
    --gradient_checkpoint_group_size "$GRADIENT_CHECKPOINT_GROUP_SIZE" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --projection_chunk_size "$PROJECTION_CHUNK_SIZE" \
    --loss_chunk_size "$LOSS_CHUNK_SIZE" \
    --grad_clip 1.0 \
    --output "$OUTPUT" \
    "${probe_args[@]}" \
    "$@"
