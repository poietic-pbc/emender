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
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-64}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-512}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-256}"
OUTPUT="${OUTPUT:-/tmp/emender-e97-8b-schedulefree-offload-probe-${CHUNK_SIZE}}"

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
    --steps 1 \
    --optimizer schedulefree \
    --offload_schedulefree_state \
    --bf16 \
    --use_triton 1 \
    --gradient_checkpointing \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --projection_chunk_size "$PROJECTION_CHUNK_SIZE" \
    --loss_chunk_size "$LOSS_CHUNK_SIZE" \
    --grad_clip 1.0 \
    --probe_memory \
    --output "$OUTPUT" \
    "$@"
