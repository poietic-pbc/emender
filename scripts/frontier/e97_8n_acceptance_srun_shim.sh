#!/bin/bash
# Acceptance-only srun shim for the immutable production launcher. It removes
# the one-shot rank-exit variables after the damaged epoch; it never retries the
# damaged bytes or changes srun arguments.
set -uo pipefail
: "${RUN_DIR:?}" "${ACCEPTANCE_STATE_DIR:?}"
REAL_SRUN=${REAL_SRUN:-/usr/bin/srun}
mkdir -p "$ACCEPTANCE_STATE_DIR"
count_file="$ACCEPTANCE_STATE_DIR/srun-invocations.txt"
count=0
[[ ! -r "$count_file" ]] || read -r count < "$count_file"
count=$((count + 1))
printf '%s\n' "$count" > "$count_file"
now_ns() { date +%s%N; }
printf '%s|%s|%q\n' "$count" "$(now_ns)" "$*" >> "$ACCEPTANCE_STATE_DIR/srun-commands.tsv"

monitor_pid=
if (( count == 1 )); then
  squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%D|%N|%P|%q|%R' \
    > "$ACCEPTANCE_STATE_DIR/squeue-live.txt" 2>&1 || true
  # Observe the synchronous step-1065200 atomic save without instrumenting
  # train.py. The first temporary checkpoint appearance and latest.pt rename
  # bound its wall-clock duration.
  (
    deadline=$(( $(date +%s) + 1200 ))
    checkpoint_started=0
    fault_seen=0
    while (( $(date +%s) < deadline )); do
      if (( checkpoint_started == 0 )) && compgen -G "$RUN_DIR/train/.checkpoint_step_1065200_loss_*.pt.*.tmp" >/dev/null; then
        checkpoint_started=1
        now_ns > "$ACCEPTANCE_STATE_DIR/checkpoint-start-epoch-ns.txt"
      fi
      target=$(readlink -f "$RUN_DIR/train/latest.pt" 2>/dev/null || true)
      if [[ $target == *checkpoint_step_1065200_loss_*.pt ]]; then
        if [[ ! -e "$ACCEPTANCE_STATE_DIR/checkpoint-published-epoch-ns.txt" ]]; then
          now_ns > "$ACCEPTANCE_STATE_DIR/checkpoint-published-epoch-ns.txt"
          printf '%s\n' "$target" > "$ACCEPTANCE_STATE_DIR/baseline-checkpoint.txt"
        fi
      fi
      epoch_log="$RUN_DIR/epochs/epoch-000001/train.out"
      if (( fault_seen == 0 )) && [[ -r $epoch_log ]] && grep -q 'DILOCO_FAULT_INJECTION' "$epoch_log"; then
        fault_seen=1
        now_ns > "$ACCEPTANCE_STATE_DIR/fault-injection-epoch-ns.txt"
      fi
      (( fault_seen == 0 )) || break
      sleep 0.10
    done
  ) &
  monitor_pid=$!
else
  # Exactly one injected exit is allowed. A fresh child receives no injection
  # variables and therefore cannot replay the unchanged damaged payload.
  unset EMENDER_DILOCO_EXIT_RANK EMENDER_DILOCO_EXIT_MERGE
  unset EMENDER_DILOCO_EXIT_BUCKET EMENDER_DILOCO_EXIT_LABEL
  unset EMENDER_DILOCO_EXIT_CODE EMENDER_DILOCO_EXIT_DELAY_SECONDS
  now_ns > "$ACCEPTANCE_STATE_DIR/relaunch-start-epoch-ns.txt"
  printf 'fault_environment_removed=true\n' > "$ACCEPTANCE_STATE_DIR/relaunch-environment.txt"
fi

start_ns=$(now_ns)
set +e
"$REAL_SRUN" "$@"
rc=$?
set -e
end_ns=$(now_ns)
printf '%s|%s|%s|%s\n' "$count" "$rc" "$start_ns" "$end_ns" >> "$ACCEPTANCE_STATE_DIR/srun-results.tsv"

if (( count == 1 )); then
  [[ -z $monitor_pid ]] || wait "$monitor_pid" 2>/dev/null || true
  now_ns > "$ACCEPTANCE_STATE_DIR/fault-step-exit-epoch-ns.txt"
  readlink -f "$RUN_DIR/train/latest.pt" > "$ACCEPTANCE_STATE_DIR/latest-after-fault.txt" 2>/dev/null || true
  find "$RUN_DIR/train" -maxdepth 1 \( -type f -o -type l \) -printf '%f|%s|%T@|%l\n' \
    | sort > "$ACCEPTANCE_STATE_DIR/train-files-after-fault.tsv"
  checkpoint=$(cat "$ACCEPTANCE_STATE_DIR/baseline-checkpoint.txt" 2>/dev/null || true)
  if [[ -n $checkpoint && -r $checkpoint ]]; then
    ( sha256sum "$checkpoint" > "$ACCEPTANCE_STATE_DIR/baseline-checkpoint.sha256.tmp" && \
      mv -f "$ACCEPTANCE_STATE_DIR/baseline-checkpoint.sha256.tmp" "$ACCEPTANCE_STATE_DIR/baseline-checkpoint.sha256" ) &
  fi
fi
exit "$rc"
