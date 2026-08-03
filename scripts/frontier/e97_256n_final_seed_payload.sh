#!/bin/bash
# Clean production envelope for the immutable ADR-003 same-allocation launcher.
# This records the live scheduler binding but does not alter train.py, inject a
# fault, interpose on srun/scontrol, or provide an acceptance shim.
set -euo pipefail
: "${REPO:?exact immutable source checkout required}"
: "${RUN_ID:?stable production run identity required}" "${RUN_DIR:?stable production run directory required}"
PAYLOAD_JOB_ID=${PAYLOAD_JOB_ID:-$SLURM_JOB_ID}
EXPECTED_PARTITION=${EXPECTED_PARTITION:-batch}
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
launcher="$REPO/scripts/frontier/e97_same_allocation_restart.sbatch"
[[ -r "$launcher" ]] || { echo "exact production launcher missing" >&2; exit 66; }
mkdir -p "$RUN_DIR"/{identity,monitor,terminal}

squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%D|%N|%P|%q|%l|%R' \
  | tee "$RUN_DIR/identity/squeue-live.txt"
job_record=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$job_record" > "$RUN_DIR/identity/scontrol-live.txt"
[[ "$SLURM_JOB_ID" == "$PAYLOAD_JOB_ID" \
   && "$job_record" == *"NumNodes=256"* \
   && "$job_record" == *"NumTasks=2048"* \
   && "$job_record" == *"Partition=$EXPECTED_PARTITION"* \
   && "$job_record" == *"QOS=$EXPECTED_QOS"* \
   && "$job_record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* ]] || {
  echo "fail closed: live binding is not Nodes=256 Tasks=2048 Partition=$EXPECTED_PARTITION QOS=$EXPECTED_QOS TimeLimit=$EXPECTED_TIME_LIMIT" >&2
  exit 67
}
for name in EMENDER_DILOCO_EXIT_RANK EMENDER_DILOCO_EXIT_MERGE \
  EMENDER_DILOCO_EXIT_BUCKET EMENDER_DILOCO_EXIT_LABEL EMENDER_DILOCO_EXIT_CODE \
  EMENDER_DILOCO_EXIT_DELAY_SECONDS; do
  [[ -z ${!name+x} ]] || { echo "fault injection variable is set: $name" >&2; exit 68; }
done

# The exact launcher remains the only execution authority. Its stable
# RUN_DIR/train/latest.pt is shared by all execution epochs and requeues.
# shellcheck disable=SC1090
source "$launcher"
set +e
samealloc_main
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_DIR/monitor/launcher-rc.txt"
exit "$rc"
