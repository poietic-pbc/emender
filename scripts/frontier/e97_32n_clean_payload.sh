#!/bin/bash
# Clean instrumentation envelope for the immutable ADR-003 production launcher.
# It does not alter train.py, inject faults, select/exclude nodes, or interpose on
# srun/scontrol.  The sourced exact launcher remains the execution authority.
set -euo pipefail
: "${REPO:?exact source checkout required}" "${RUN_DIR:?}" "${RUN_ID:?}"
: "${INITIAL_CHECKPOINT:?}"
PAYLOAD_JOB_ID=${PAYLOAD_JOB_ID:-$SLURM_JOB_ID}
launcher="$REPO/scripts/frontier/e97_same_allocation_restart.sbatch"
[[ -r "$launcher" ]] || { echo "exact production launcher missing" >&2; exit 66; }
mkdir -p "$RUN_DIR"/{identity,monitor,terminal}

squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%D|%N|%P|%q|%R' | tee "$RUN_DIR/identity/squeue-live.txt"
job_record=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$job_record" > "$RUN_DIR/identity/scontrol-live.txt"
[[ "$SLURM_JOB_ID" == "$PAYLOAD_JOB_ID" && "$job_record" == *"NumNodes=32"* && \
   "$job_record" == *"Partition=batch"* && "$job_record" == *"QOS=normal"* ]] || {
  echo "fail closed: live scheduler binding is not Nodes=32 Partition=batch QOS=normal" >&2
  exit 67
}

initial_resolved=$(readlink -f "$INITIAL_CHECKPOINT")
monitor_done="$RUN_DIR/monitor/done"
monitor_log="$RUN_DIR/monitor/checkpoint-events.tsv"
: > "$monitor_log"
monitor_checkpoints() {
  local tmp target now
  local last_target="$initial_resolved"
  declare -A seen_tmp=()
  while [[ ! -e "$monitor_done" ]]; do
    if [[ -d "$RUN_DIR/train" ]]; then
      while IFS= read -r -d '' tmp; do
        if [[ -z "${seen_tmp[$tmp]:-}" ]]; then
          seen_tmp[$tmp]=1
          printf 'tmp_seen\t%s\t%s\n' "$(date +%s%N)" "$tmp" >> "$monitor_log"
        fi
      done < <(find "$RUN_DIR/train" -maxdepth 1 -type f -name '.checkpoint_step_*.tmp' -print0 2>/dev/null)
      if [[ -L "$RUN_DIR/train/latest.pt" ]]; then
        target=$(readlink -f "$RUN_DIR/train/latest.pt" 2>/dev/null || true)
        if [[ -n "$target" && "$target" != "$last_target" ]]; then
          now=$(date +%s%N)
          printf 'latest_published\t%s\t%s\n' "$now" "$target" >> "$monitor_log"
          last_target=$target
        fi
      fi
    fi
    sleep 0.10
  done
}
monitor_checkpoints &
monitor_pid=$!
cleanup_monitor() {
  touch "$monitor_done"
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

# The immutable launcher is sourced only so this clean envelope can retain live
# scheduler and checkpoint-pause observations.  samealloc_main is unchanged.
# shellcheck disable=SC1090
source "$launcher"
set +e
samealloc_main
rc=$?
set -e
cleanup_monitor
trap - EXIT
printf '%s\n' "$rc" > "$RUN_DIR/monitor/launcher-rc.txt"
exit "$rc"
