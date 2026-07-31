#!/bin/bash
# Acceptance-only node-selection shim. After the single damaged execution epoch
# is durably recorded, present the final allocated node as unavailable to the
# production launcher's whole-node filter. This deliberately exercises an 8->7
# fixed-world relaunch without crashing or draining a physical Frontier node.
set -euo pipefail
REAL_SCONTROL=${REAL_SCONTROL:-/usr/bin/scontrol}
: "${RUN_DIR:?}" "${ACCEPTANCE_STATE_DIR:?}"
if [[ ${1:-} == show && ${2:-} == node && -r "$RUN_DIR/supervisor/execution-epochs.tsv" ]] \
    && awk -F'|' '$2 != 0 {found=1} END {exit !found}' "$RUN_DIR/supervisor/execution-epochs.tsv"; then
  drop_node=$($REAL_SCONTROL show hostnames "$SLURM_JOB_NODELIST" | tail -1)
  if [[ ${3:-} == "$drop_node" ]]; then
    record=$($REAL_SCONTROL "$@")
    printf '%s\n' "$record" | sed -E 's/State=[^[:space:]]+/State=DOWN/'
    if [[ ! -e "$ACCEPTANCE_STATE_DIR/deliberate-node-exclusion.txt" ]]; then
      mkdir -p "$ACCEPTANCE_STATE_DIR"
      printf 'node=%s\nmode=acceptance-only-deliberate-whole-node-exclusion\nphysical_node_crash=false\ncommunicator_shrink=false\n' \
        "$drop_node" > "$ACCEPTANCE_STATE_DIR/deliberate-node-exclusion.txt"
      squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%D|%N|%P|%q|%R' \
        > "$ACCEPTANCE_STATE_DIR/squeue-allocation-after-fault.txt" 2>&1 || true
    fi
    exit 0
  fi
fi
exec "$REAL_SCONTROL" "$@"
