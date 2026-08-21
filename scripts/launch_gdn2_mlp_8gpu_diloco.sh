#!/usr/bin/env bash
# Launch the matched from-scratch 8-GPU (I=8 islands) DiLoCo GDN2-MLP arm.
#
# This is the GDN2 analogue of scripts/launch_emender_8gpu_diloco.sh:
# keep the proven DiLoCo/data/checkpoint harness fixed, and replace only the
# model-specific E97 block plus the CMA-selected base LR.
#
# Source of truth:
#   docs/GDN2_MLP_DILOCO_HANDOFF_20260711.md
#   docs/repro/lb_gdn2_mlp_20260612/best.json
#
# Usage:
#   scripts/launch_gdn2_mlp_8gpu_diloco.sh
#   RESUME=/path/to/gdn2_ckpt.pt scripts/launch_gdn2_mlp_8gpu_diloco.sh
#   GDN2_PATH=/path/to/GatedDeltaNet-2 scripts/launch_gdn2_mlp_8gpu_diloco.sh
#
# Prints the detached run PID on stdout (see launch_detached_run.sh).
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

GDN2_PATH="${GDN2_PATH:-/home/erikg/GatedDeltaNet-2}"
LOGDIR="${LOGDIR:-/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp}"
OUTPUT="${OUTPUT:-$LOGDIR/runs}"
NAME="${NAME:-diloco_gdn2_mlp_8gpu_i8}"
RESUME="${RESUME:-}"

if [ ! -d "$GDN2_PATH" ]; then
  echo "launch_gdn2_mlp_8gpu_diloco.sh: GDN2_PATH does not exist: $GDN2_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT"

ENV_ARGS=(
  env
  GDN2_PATH="$GDN2_PATH"
  NCCL_P2P_DISABLE=1
  TORCH_NCCL_ENABLE_MONITORING=0
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

TRAIN_ARGS=(
  torchrun --standalone --nproc_per_node=8 train.py
)
if [ -n "$RESUME" ]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
TRAIN_ARGS+=(
  --level gdn2-mlp
  --dim 2176
  --depth 12
  --n_heads 30
  --expansion 1
  --gdn2_mlp_ratio 3.258732449079677
  --use_conv 1
  --d_conv 4
  --optimizer schedulefree
  --lr 0.00047431158698290157
  --bf16
  --batch_size 4
  --chunk_size 2048
  --data /home/erikg/elman/data/pile.txt
  --tokenizer p50k_base
  --diloco
  --diloco_k 250
  --diloco_outer_lr 1.0
  --diloco_outer_beta 0.0
  --steps 100000000
  --save_every 500
  --keep_checkpoints 20
  --log_every 25
  --output "$OUTPUT"
)

exec scripts/launch_detached_run.sh \
  --name "$NAME" \
  --gpus 8 \
  --logdir "$LOGDIR" \
  -- "${ENV_ARGS[@]}" "${TRAIN_ARGS[@]}"
