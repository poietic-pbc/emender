#!/bin/bash
# Minimal 2-node compiled-helper diagnostic under Frontier debug QOS.

#SBATCH -A bif148
#SBATCH -J compiled-mpich-helper-2n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/compiled_mpich_helper/%x-%j.out
#SBATCH -e logs/frontier/compiled_mpich_helper/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
SUMMARY_FILE="${RUN_ROOT}/summary.md"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
HELPER_BIN="${ARTIFACT_DIR}/compiled_mpich_dense_helper"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" logs/frontier/compiled_mpich_helper
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules

export REPO ARTIFACT_DIR MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
scripts/frontier/build_compiled_mpich_dense_helper.sh | tee "${LOG_DIR}/build.log"

CMD=(
  srun
  -N 2
  -n 16
  --ntasks-per-node=8
  -c 7
  --gpus-per-task=1
  --gpu-bind=closest
  "$HELPER_BIN"
  --diagnostic
)
printf '%q ' "${CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

set +e
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/diagnostic.log"
STATUS=${PIPESTATUS[0]}
set -e

python - "$STATUS" "$RUN_ROOT" "$SUMMARY_FILE" "$COMMAND_FILE" "$LOG_DIR/diagnostic.log" <<'PY'
import sys
from pathlib import Path

status = int(sys.argv[1])
run_root = Path(sys.argv[2])
summary = Path(sys.argv[3])
command = Path(sys.argv[4])
log = Path(sys.argv[5])
text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
passed = status == 0 and '"transport":"compiled-cray-mpich-helper-p2p"' in text and '"world_size":16' in text
summary.write_text(
    "# Compiled MPICH helper 2n diagnostic\n\n"
    f"Status: {'pass' if passed else 'fail'}\n\n"
    f"Run root: `{run_root}`\n\n"
    f"Command: `{command}`\n\n"
    f"Log: `{log}`\n",
    encoding="utf-8",
)
raise SystemExit(0 if passed else 1)
PY
