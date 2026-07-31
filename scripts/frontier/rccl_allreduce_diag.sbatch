#!/bin/bash
# Frontier RCCL/Slingshot allreduce diagnostic.
#
# Examples:
#   sbatch -N 2 -J rccl-diag-current-2n \
#     --export=ALL,RCCL_DIAG_ENV_MODE=current \
#     scripts/frontier/rccl_allreduce_diag.sbatch
#
#   sbatch -N 2 -J rccl-diag-recommended-2n \
#     --export=ALL,RCCL_DIAG_ENV_MODE=recommended,AWS_OFI_RCCL_PLUGIN_DIR=/path/to/aws-ofi-rccl \
#     scripts/frontier/rccl_allreduce_diag_alt_rdzv.sbatch

#SBATCH -A bif148
#SBATCH -J rccl-allreduce-diag
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/rccl_diag/%x-%j.out
#SBATCH -e logs/frontier/rccl_diag/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-diagnose-frontier-rccl}
MODE=${RCCL_DIAG_ENV_MODE:-current}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${MODE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
RESULT_JSON="${ARTIFACT_DIR}/rank0_results.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
STDOUT_PATH="logs/frontier/rccl_diag/${SLURM_JOB_NAME:-rccl-allreduce-diag}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/rccl_diag/${SLURM_JOB_NAME:-rccl-allreduce-diag}-${SLURM_JOB_ID:-manual}.err"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SUMMARY_DIR" logs/frontier/rccl_diag
cd "$REPO"

source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules

if [[ -n "${EMENDER_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$EMENDER_CONDA_ENV"
fi

export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-3442}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export RCCL_DIAG_SIZES=${RCCL_DIAG_SIZES:-scalar=1,medium=1048576,model=1286589072}
export RCCL_DIAG_DTYPE=${RCCL_DIAG_DTYPE:-float32}
export RCCL_DIAG_JSON="$RESULT_JSON"

if [[ "$MODE" == "recommended" ]]; then
  export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-kdreg2}
  export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
  export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-2048}
  export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
  export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
  export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-net.so}
  if [[ "${RCCL_DIAG_ALT_RDZV:-0}" == "1" ]]; then
    export FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO:-alt_read}
  fi
  if [[ -n "${AWS_OFI_RCCL_PLUGIN_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${AWS_OFI_RCCL_PLUGIN_DIR%/}/lib:${LD_LIBRARY_PATH:-}"
  fi
else
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
fi
if frontier_runtime_olcf_rccl_enabled; then
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-net.so}
fi

PLUGIN_STATUS=$(frontier_resolve_librccl_net)
export RCCL_DIAG_PLUGIN_STATUS="$PLUGIN_STATUS"

{
  echo "task_id=$TASK_ID"
  echo "mode=$MODE"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "rank0_result_json=$RESULT_JSON"
  echo "plugin_status=$PLUGIN_STATUS"
  echo "git_commit=$(git rev-parse HEAD)"
  echo
  echo "=== modules ==="
  module -t list 2>&1 || true
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env "$PLUGIN_STATUS"
  echo
  echo "=== selected env ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|RCCL|FI_|MASTER|OMP|ROCR|HIP|HSA|CRAY|ROCM|LD_LIBRARY_PATH|AWS_OFI_RCCL|EMENDER_CONDA_ENV)' || true
  echo
  echo "=== command ==="
  printf '%q ' srun -N "${SLURM_JOB_NUM_NODES:-2}" -n "$((${SLURM_JOB_NUM_NODES:-2} * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc \
    'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec python -u scripts/frontier/rccl_allreduce_diag.py' 
  echo
} | tee "$ENV_FILE"

set +e
srun -N "${SLURM_JOB_NUM_NODES:-2}" -n "$((${SLURM_JOB_NUM_NODES:-2} * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest \
  bash -lc 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec python -u scripts/frontier/rccl_allreduce_diag.py' \
  2>&1 | tee "${LOG_DIR}/allreduce.log"
DIAG_STATUS=${PIPESTATUS[0]}
set -e

python - "$LOG_DIR/allreduce.log" "$SUMMARY_FILE" "$DIAG_STATUS" "$ENV_FILE" "$RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
status = int(sys.argv[3])
env_file = sys.argv[4]
result_json = Path(sys.argv[5])
lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
results = []
for line in lines:
    if line.startswith("RCCL_DIAG_RESULT "):
        try:
            results.append(json.loads(line.split(" ", 1)[1]))
        except json.JSONDecodeError:
            pass
rank0 = [row for row in results if row.get("rank") == 0]
errors = [ln for ln in lines if any(k in ln.lower() for k in ("error", "traceback", "timeout", "nccl", "rccl", "ofi", "failed"))]

with summary_path.open("w", encoding="utf-8") as out:
    out.write("# Frontier RCCL Allreduce Diagnostic Summary\n\n")
    out.write(f"Exit status: `{status}`\n\n")
    out.write(f"Env file: `{env_file}`\n\n")
    out.write(f"Rank-0 JSON: `{result_json}`\n\n")
    out.write("## Rank-0 Results\n\n")
    if rank0:
        out.write("| size | numel | dtype | status | elapsed_s | GiB/s per rank |\n")
        out.write("| --- | ---: | --- | --- | ---: | ---: |\n")
        for row in rank0:
            out.write(
                f"| {row.get('name')} | {row.get('numel')} | {row.get('dtype')} | "
                f"{row.get('status')} | {row.get('elapsed_s', '')} | "
                f"{row.get('bandwidth_gib_s_per_rank', '')} |\n"
            )
    else:
        out.write("No rank-0 result lines found.\n")
    out.write("\n## Error Lines\n\n")
    for line in errors[-80:]:
        out.write(f"```text\n{line}\n```\n")
    if not errors:
        out.write("No error lines found.\n")
PY

exit "$DIAG_STATUS"
