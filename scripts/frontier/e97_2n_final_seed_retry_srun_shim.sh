#!/bin/bash
# Acceptance-only wrapper: observe exactly one real failed compute child, then
# remove its one-shot injection before the production supervisor relaunches.
set -uo pipefail
: "${RUN_DIR:?}" "${FINAL_SEED_RETRY_STATE_DIR:?}"
REAL_SRUN=${REAL_SRUN:-/usr/bin/srun}

# Seed bootstrap and offline verification steps are part of the production
# launcher, not supervised model-bearing execution epochs.
if [[ " $* " == *" --gpus-per-task=0 "* ]]; then
  exec "$REAL_SRUN" "$@"
fi

mkdir -p "$FINAL_SEED_RETRY_STATE_DIR"
count_file="$FINAL_SEED_RETRY_STATE_DIR/compute-srun-invocations.txt"
count=0
[[ ! -r $count_file ]] || read -r count < "$count_file"
count=$((count + 1))
printf '%s\n' "$count" > "$count_file"
now_ns() { date +%s%N; }
printf '%s|%s|%q\n' "$count" "$(now_ns)" "$*" >> "$FINAL_SEED_RETRY_STATE_DIR/compute-srun-commands.tsv"

if (( count == 1 )); then
  squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%D|%N|%P|%q|%R' \
    > "$FINAL_SEED_RETRY_STATE_DIR/squeue-live.txt" 2>&1 || true
else
  unset EMENDER_DILOCO_EXIT_RANK EMENDER_DILOCO_EXIT_MERGE
  unset EMENDER_DILOCO_EXIT_BUCKET EMENDER_DILOCO_EXIT_LABEL
  unset EMENDER_DILOCO_EXIT_CODE EMENDER_DILOCO_EXIT_DELAY_SECONDS
  now_ns > "$FINAL_SEED_RETRY_STATE_DIR/retry-start-epoch-ns.txt"
  printf 'fault_environment_removed=true\nunchanged_failed_payload_retried=false\n' \
    > "$FINAL_SEED_RETRY_STATE_DIR/retry-environment.txt"
fi

start_ns=$(now_ns)
set +e
"$REAL_SRUN" "$@"
rc=$?
set -e
end_ns=$(now_ns)
printf '%s|%s|%s|%s\n' "$count" "$rc" "$start_ns" "$end_ns" \
  >> "$FINAL_SEED_RETRY_STATE_DIR/compute-srun-results.tsv"

if (( count == 1 )); then
  now_ns > "$FINAL_SEED_RETRY_STATE_DIR/fault-step-exit-epoch-ns.txt"
  readlink -f "$RUN_DIR/train/latest.pt" \
    > "$FINAL_SEED_RETRY_STATE_DIR/latest-after-fault.txt" 2>/dev/null || true
  find "$RUN_DIR/train" -maxdepth 1 \( -type f -o -type l \) \
    -printf '%f|%s|%T@|%l\n' | sort \
    > "$FINAL_SEED_RETRY_STATE_DIR/train-files-after-fault.tsv"
  squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%D|%N|%P|%q|%R' \
    > "$FINAL_SEED_RETRY_STATE_DIR/squeue-allocation-after-fault.txt" 2>&1 || true
  /usr/bin/scontrol show hostnames "$SLURM_JOB_NODELIST" \
    > "$FINAL_SEED_RETRY_STATE_DIR/allocation-hostnames-after-fault.txt"
  while IFS= read -r node; do
    /usr/bin/scontrol show node "$node" -o
  done < "$FINAL_SEED_RETRY_STATE_DIR/allocation-hostnames-after-fault.txt" \
    > "$FINAL_SEED_RETRY_STATE_DIR/node-health-after-fault.txt" 2>&1
fi
exit "$rc"
