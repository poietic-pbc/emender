#!/bin/bash
# Submit a dependency chain of 64-node, 2-hour E97 B4/K80 continuation jobs.
#
# Usage:
#
#   CHAIN_JOBS=6 scripts/frontier/submit_e97_1p3b_b4_k80_64n_chain.sh
#
# The jobs use afterany dependencies so a nonzero exit that still produced a
# valid latest checkpoint can be followed by the next continuation attempt.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CHAIN_JOBS=${CHAIN_JOBS:-${1:-6}}
DEPENDENCY_MODE=${DEPENDENCY_MODE:-afterany}
JOB_SCRIPT=${JOB_SCRIPT:-"${REPO}/scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch"}
WG_TASK_ID_PREFIX=${WG_TASK_ID_PREFIX:-chain-64n-b4-k80-2h}

[[ "$CHAIN_JOBS" =~ ^[0-9]+$ ]] || { echo "CHAIN_JOBS must be an integer: $CHAIN_JOBS" >&2; exit 2; }
(( CHAIN_JOBS > 0 )) || { echo "CHAIN_JOBS must be > 0" >&2; exit 2; }
[[ -r "$JOB_SCRIPT" ]] || { echo "JOB_SCRIPT is not readable: $JOB_SCRIPT" >&2; exit 4; }

prev_job=""
for i in $(seq 1 "$CHAIN_JOBS"); do
  dep_args=()
  if [[ -n "$prev_job" ]]; then
    dep_args=(--dependency="${DEPENDENCY_MODE}:${prev_job}")
  fi
  export_arg="ALL,REPO=${REPO},CHAIN_INDEX=${i},WG_TASK_ID=${WG_TASK_ID_PREFIX}-${i}"
  submission=$(sbatch --network=disable_rdzv_get "${dep_args[@]}" --export="$export_arg" "$JOB_SCRIPT")
  job_id=$(awk '{print $4}' <<<"$submission")
  echo "submitted chain_index=${i} job_id=${job_id} dependency=${prev_job:-none}"
  prev_job="$job_id"
done

echo "last_job_id=${prev_job}"
