#!/usr/bin/env bash
# Launch the documented one-candidate-per-GPU E97 4B LR interference bracket.
set -euo pipefail
REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

ROOT="${ROOT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_lr_interference_scan}"
LOGDIR="${ROOT}.launcher"
[[ ! -e "$ROOT" ]] || { echo "scan root already exists: $ROOT" >&2; exit 64; }
[[ ! -e "$LOGDIR" ]] || { echo "launcher logdir already exists: $LOGDIR" >&2; exit 64; }
mkdir -p "$LOGDIR"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'ROOT=%q\nLOGDIR=%q\n' "$ROOT" "$LOGDIR"
  printf 'COMMAND='; printf '%q ' python scripts/run_e97_4b_lr_interference_scan.py --root "$ROOT"; printf '\n'
  exit 0
fi

scripts/launch_detached_run.sh \
  --name e97_4b_lr_interference_scan --gpus 8 --logdir "$LOGDIR" -- \
  env PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache \
  python -u scripts/run_e97_4b_lr_interference_scan.py --root "$ROOT" \
  >"$LOGDIR/launcher.out"
pid="$(tr -d '[:space:]' <"$LOGDIR/launcher.out")"
[[ "$pid" =~ ^[0-9]+$ ]] || { echo "invalid detached pid: $pid" >&2; exit 70; }
printf 'pid=%s\nroot=%s\ncontroller_log=%s\nstop=kill -TERM %s\n' \
  "$pid" "$ROOT" "$LOGDIR/run.log" "$pid"
