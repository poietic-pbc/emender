#!/bin/bash
# Debug-scale Frontier benchmark for async DiLoCo x/z delta transport.
#
# Dense payloads are allocated in CPU or HIP memory and transferred via Cray
# MPICH point-to-point. Lustre is used only for source, logs, and JSON metrics.

#SBATCH -A bif148
#SBATCH -J async-diloco-transport
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/async_diloco_transport/%x-%j.out
#SBATCH -e logs/frontier/async_diloco_transport/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-async-diloco-frontier-transport-prototype}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco_transport}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
METRICS_JSON="${ARTIFACT_DIR}/transport_metrics.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
STDOUT_PATH="logs/frontier/async_diloco_transport/${SLURM_JOB_NAME:-async-diloco-transport}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/async_diloco_transport/${SLURM_JOB_NAME:-async-diloco-transport}-${SLURM_JOB_ID:-manual}.err"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SUMMARY_DIR" logs/frontier/async_diloco_transport
cd "$REPO"

source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules

export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-kdreg2}
export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-2048}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}

BENCH_DEVICE=${BENCH_DEVICE:-hip}
BENCH_MODE=${BENCH_MODE:-fanin}
BENCH_PAYLOAD_MIB=${BENCH_PAYLOAD_MIB:-256}
BENCH_CHUNK_MIB=${BENCH_CHUNK_MIB:-16}
BENCH_ITERS=${BENCH_ITERS:-20}
BENCH_WARMUP=${BENCH_WARMUP:-5}
BENCH_DTYPE_BYTES=${BENCH_DTYPE_BYTES:-2}
BENCH_BIN="${ARTIFACT_DIR}/async_diloco_transport_bench"

if [[ "$BENCH_DEVICE" == "hip" ]]; then
  CXX=${TRANSPORT_BENCH_CXX:-CC}
  CXXFLAGS=${TRANSPORT_BENCH_CXXFLAGS:--O3 -std=c++17 -x hip --offload-arch=gfx90a -DUSE_HIP}
else
  CXX=${TRANSPORT_BENCH_CXX:-CC}
  CXXFLAGS=${TRANSPORT_BENCH_CXXFLAGS:--O3 -std=c++17}
fi

{
  echo "task_id=$TASK_ID"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "slurm_job_num_nodes=${SLURM_JOB_NUM_NODES:-1}"
  echo "node_hours=$(python3 - <<'PY'
import os
nodes = float(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
minutes = 10.0
print(f"{nodes * minutes / 60.0:.6f}")
PY
)"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "metrics_json=$METRICS_JSON"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "bench_device=$BENCH_DEVICE"
  echo "bench_mode=$BENCH_MODE"
  echo "bench_payload_mib=$BENCH_PAYLOAD_MIB"
  echo "bench_chunk_mib=$BENCH_CHUNK_MIB"
  echo
  echo "=== modules ==="
  module -t list 2>&1 || true
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env
  echo
  echo "=== selected mpi env ==="
  env | sort | grep -E '^(SLURM|MPICH|FI_|CRAY|ROCM|HIP|HSA|LD_LIBRARY_PATH|FRONTIER_|OMP)=' || true
  echo
  echo "=== build command ==="
  printf '%q ' "$CXX" $CXXFLAGS scripts/frontier/async_diloco_transport_bench.cpp -o "$BENCH_BIN"
  echo
  echo
  echo "=== run command ==="
  printf '%q ' srun -N "${SLURM_JOB_NUM_NODES:-1}" -n "${SLURM_NTASKS:-8}" -c "${SLURM_CPUS_PER_TASK:-7}" --gpus-per-task=1 --gpu-bind=closest \
    "$BENCH_BIN" --mode "$BENCH_MODE" --device "$BENCH_DEVICE" --payload-mib "$BENCH_PAYLOAD_MIB" \
    --chunk-mib "$BENCH_CHUNK_MIB" --iters "$BENCH_ITERS" --warmup "$BENCH_WARMUP" \
    --dtype-bytes "$BENCH_DTYPE_BYTES" --metrics "$METRICS_JSON"
  echo
} | tee "$ENV_FILE"

# shellcheck disable=SC2086
"$CXX" $CXXFLAGS scripts/frontier/async_diloco_transport_bench.cpp -o "$BENCH_BIN" 2>&1 | tee "${LOG_DIR}/build.log"

set +e
srun -N "${SLURM_JOB_NUM_NODES:-1}" -n "${SLURM_NTASKS:-8}" -c "${SLURM_CPUS_PER_TASK:-7}" --gpus-per-task=1 --gpu-bind=closest \
  "$BENCH_BIN" \
  --mode "$BENCH_MODE" \
  --device "$BENCH_DEVICE" \
  --payload-mib "$BENCH_PAYLOAD_MIB" \
  --chunk-mib "$BENCH_CHUNK_MIB" \
  --iters "$BENCH_ITERS" \
  --warmup "$BENCH_WARMUP" \
  --dtype-bytes "$BENCH_DTYPE_BYTES" \
  --metrics "$METRICS_JSON" \
  2>&1 | tee "${LOG_DIR}/transport.log"
BENCH_STATUS=${PIPESTATUS[0]}
set -e

python - "$SUMMARY_FILE" "$BENCH_STATUS" "$ENV_FILE" "$METRICS_JSON" "$STDOUT_PATH" "$STDERR_PATH" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
status = int(sys.argv[2])
env_file = sys.argv[3]
metrics_path = Path(sys.argv[4])
stdout = sys.argv[5]
stderr = sys.argv[6]
metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())

conclusion = "pass" if status == 0 and metrics.get("status") == "pass" else "no-go"
summary.write_text(
    "# Async DiLoCo Frontier Transport Benchmark\n\n"
    f"Slurm job ID: `{metrics.get('slurm_job_id', 'see env file')}`\n\n"
    f"Exit status: `{status}`\n\n"
    f"Conclusion: `{conclusion}`\n\n"
    f"Env file: `{env_file}`\n\n"
    f"Metrics JSON: `{metrics_path}`\n\n"
    f"Stdout: `{stdout}`\n\n"
    f"Stderr: `{stderr}`\n\n"
    "Dense update data plane: MPI memory buffers only; Lustre stores logs and metrics.\n\n"
    "Metrics excerpt:\n\n"
    "```json\n"
    + json.dumps(metrics, indent=2, sort_keys=True)
    + "\n```\n",
    encoding="utf-8",
)
PY

exit "$BENCH_STATUS"
