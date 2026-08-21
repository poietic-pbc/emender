#!/usr/bin/env bash
# Restart-on-death supervisor for the E97 8-GPU DiLoCo emender continuation.
#
# This intentionally delegates every training argument and GPU lease to
# launch_emender_8gpu_diloco.sh. The supervisor only picks the latest clean
# merge-aligned checkpoint, avoids duplicate launches, rotates logs, and
# restarts if the detached torchrun leader exits.
set -euo pipefail
trap "" HUP

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

LOGDIR="${LOGDIR:-/mnt/nvme1n1/erikg/diloco_8gpu/emender}"
OUTPUT="${OUTPUT:-$LOGDIR/runs}"
SLOG="${SLOG:-$LOGDIR/supervisor.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-30}"
NAME="${NAME:-diloco_emender_8gpu_i8_supervised}"
STOP_FILE="${STOP_FILE:-$LOGDIR/supervisor.stop}"
ADOPT_EXISTING="${ADOPT_EXISTING:-0}"

mkdir -p "$LOGDIR" "$OUTPUT"

ts() { date -u +%FT%TZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*" >> "$SLOG"; }

is_alive() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

latest_checkpoint() {
  find "$OUTPUT" -maxdepth 2 -type f -name 'checkpoint_step_*.pt' -printf '%f %p\n' |
    python3 -c '
import re
import sys

best = None
for line in sys.stdin:
    name, path = line.rstrip("\n").split(" ", 1)
    match = re.search(r"checkpoint_step_(\d+)_loss_", name)
    if not match:
        continue
    step = int(match.group(1))
    if step % 500 != 0:
        continue
    candidate = (step, path)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is None:
    raise SystemExit("no merge-aligned checkpoint found")
print(best[1])
'
}

existing_pid=""
if [ -f "$LOGDIR/run.pid" ]; then
  existing_pid="$(tr -d '[:space:]' < "$LOGDIR/run.pid" || true)"
fi
if is_alive "$existing_pid"; then
  if [ "$ADOPT_EXISTING" = "1" ]; then
    log "adopting existing live run pid=$existing_pid for restart supervision"
    adopted_pid="$existing_pid"
  else
    log "existing run is alive pid=$existing_pid; supervisor exiting without duplicate launch"
    exit 0
  fi
else
  adopted_pid=""
fi

log "supervisor up pid=$$ mode=indefinite logdir=$LOGDIR poll=${POLL_SECONDS}s"

while [ ! -e "$STOP_FILE" ]; do
  if [ -n "$adopted_pid" ]; then
    pid="$adopted_pid"
    adopted_pid=""
  else
    resume="$(latest_checkpoint)"
    if [ -f "$LOGDIR/run.log" ]; then
      mv "$LOGDIR/run.log" "$LOGDIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"
    fi

    log "launching torchrun via launch_emender_8gpu_diloco.sh resume=$resume"
    pid_tmp="$LOGDIR/supervisor.launch.pid.tmp"
    rm -f "$pid_tmp"
    LOGDIR="$LOGDIR" OUTPUT="$OUTPUT" NAME="$NAME" RESUME="$resume" \
      scripts/launch_emender_8gpu_diloco.sh > "$pid_tmp"
    pid="$(tr -d '[:space:]' < "$pid_tmp")"
    rm -f "$pid_tmp"
    log "torchrun leader pid=$pid"
  fi

  while [ ! -e "$STOP_FILE" ] && is_alive "$pid"; do
    sleep "$POLL_SECONDS"
  done

  if [ -e "$STOP_FILE" ]; then
    log "stop file present; leaving loop with last pid=$pid"
    break
  fi

  log "torchrun leader exited pid=$pid; restarting after ${RESTART_DELAY_SECONDS}s"
  sleep "$RESTART_DELAY_SECONDS"
done

log "supervisor exit pid=$$"
