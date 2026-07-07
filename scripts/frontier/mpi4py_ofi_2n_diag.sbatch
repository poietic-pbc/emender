#!/bin/bash
# Minimal 2-node mpi4py/Cray MPICH OFI diagnostic for train.py dense MPI smoke.

#SBATCH -A bif148
#SBATCH -J mpi4py-ofi-2n-diag
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/mpi4py_ofi_diag/%x-%j.out
#SBATCH -e logs/frontier/mpi4py_ofi_diag/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
SUMMARY_FILE="${RUN_ROOT}/summary.md"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
SMOKE_PY="${ARTIFACT_DIR}/mpi4py_init_sendrecv.py"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" logs/frontier/mpi4py_ofi_diag
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env

PYTHON_BIN=$(command -v python)
export REPO PYTHON_BIN
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
export CRAY_MPI4PY_SITE=${CRAY_MPI4PY_SITE:-/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages}

cat > "$SMOKE_PY" <<'PY'
import json
import os
import socket
import sys
import time

site = os.environ.get("CRAY_MPI4PY_SITE")
if site and site not in sys.path:
    import site as _site
    _site.addsitedir(site)

from mpi4py import MPI

provided = MPI.Init_thread(required=MPI.THREAD_SERIALIZED)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

left = (rank - 1) % size
right = (rank + 1) % size
send_payload = {
    "rank": rank,
    "size": size,
    "host": host,
    "time": time.time(),
}
recv_payload = comm.sendrecv(send_payload, dest=right, sendtag=17, source=left, recvtag=17)
all_hosts = comm.allgather(host)
if rank == 0:
    print(json.dumps({
        "mpi_init_thread_provided": int(provided),
        "world_size": size,
        "unique_hosts": sorted(set(all_hosts)),
        "host_count": len(set(all_hosts)),
        "rank0_received_from": recv_payload["rank"],
        "mpich_gpu_support_enabled": os.environ.get("MPICH_GPU_SUPPORT_ENABLED"),
        "fi_provider": os.environ.get("FI_PROVIDER"),
        "mpich_ofi_nic_policy": os.environ.get("MPICH_OFI_NIC_POLICY"),
        "fi_cxi_rdzv_proto": os.environ.get("FI_CXI_RDZV_PROTO"),
        "fi_cxi_rx_match_mode": os.environ.get("FI_CXI_RX_MATCH_MODE"),
    }, sort_keys=True), flush=True)
comm.Barrier()
MPI.Finalize()
PY

base_srun=(
  srun
  -N 2
  -n 16
  --ntasks-per-node=8
  -c 7
  --gpus-per-task=1
  --gpu-bind=closest
)

run_case() {
  local case_name="$1"
  shift
  local case_log="${LOG_DIR}/${case_name}.log"
  local case_env="${ARTIFACT_DIR}/${case_name}.env"
  local status

  {
    echo "case=${case_name}"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'env_args='
    printf '%q ' "$@"
    printf '\n'
    env | sort | grep -E '^(FI_|MPICH_|CRAY_|LD_LIBRARY_PATH|CONDA_PREFIX|ROCM|FRONTIER_|OLCF_)=' || true
  } > "$case_env"

  printf '%q ' "${base_srun[@]}" env "$@" "$PYTHON_BIN" "$SMOKE_PY" >> "$COMMAND_FILE"
  printf '\n' >> "$COMMAND_FILE"

  set +e
  "${base_srun[@]}" env "$@" "$PYTHON_BIN" "$SMOKE_PY" 2>&1 | tee "$case_log"
  status=${PIPESTATUS[0]}
  set -e

  echo "$status" > "${ARTIFACT_DIR}/${case_name}.status"
  return 0
}

{
  echo "task_id=${WG_TASK_ID:-debug-2n-mpi}"
  echo "run_root=$RUN_ROOT"
  echo "summary_file=$SUMMARY_FILE"
  echo "env_file=$ENV_FILE"
  echo "command_file=$COMMAND_FILE"
  echo "smoke_py=$SMOKE_PY"
  echo "requested_qos=${SLURM_JOB_QOS:-debug}"
  echo "requested_partition=${SLURM_JOB_PARTITION:-batch}"
  echo "requested_nodes=${SLURM_JOB_NUM_NODES:-2}"
  echo "requested_walltime=${SLURM_TIMELIMIT:-00:10:00}"
  echo "requested_node_hours=0.333333"
  echo "mpich_gpu_support_enabled=$MPICH_GPU_SUPPORT_ENABLED"
  echo "cray_mpi4py_site=$CRAY_MPI4PY_SITE"
  echo
  echo "=== modules ==="
  module -t list 2>&1 || true
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env
} | tee "$ENV_FILE"

: > "$COMMAND_FILE"

run_case baseline \
  MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED" \
  CRAY_MPI4PY_SITE="$CRAY_MPI4PY_SITE"

run_case cxi_numa \
  MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED" \
  CRAY_MPI4PY_SITE="$CRAY_MPI4PY_SITE" \
  FI_PROVIDER="${FI_PROVIDER:-cxi}" \
  MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-NUMA}" \
  FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}" \
  FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}" \
  FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-2048}"

baseline_status=$(<"${ARTIFACT_DIR}/baseline.status")
cxi_numa_status=$(<"${ARTIFACT_DIR}/cxi_numa.status")

{
  echo "# 2-node mpi4py OFI diagnostic"
  echo
  echo "- Job id: \`${SLURM_JOB_ID:-manual}\`"
  echo "- QOS/walltime/node-hours: \`${SLURM_JOB_QOS:-debug}\` / \`${SLURM_TIMELIMIT:-00:10:00}\` / \`0.333333\`"
  echo "- Run root: \`$RUN_ROOT\`"
  echo "- Baseline status: \`$baseline_status\`"
  echo "- CXI/NUMA status: \`$cxi_numa_status\`"
  echo
  echo "## Commands"
  echo
  echo '```bash'
  cat "$COMMAND_FILE"
  echo '```'
  echo
  echo "## Environment"
  echo
  echo '```text'
  grep -E '^(mpich_gpu_support_enabled|cray_mpi4py_site|requested_|frontier_|FI_|MPICH_|CRAY_MPI4PY_SITE|CONDA_PREFIX|LD_LIBRARY_PATH)=' "$ENV_FILE" "$ARTIFACT_DIR"/*.env 2>/dev/null || true
  echo '```'
} | tee "$SUMMARY_FILE"

if [[ "$baseline_status" == "0" || "$cxi_numa_status" == "0" ]]; then
  exit 0
fi
exit "$baseline_status"
