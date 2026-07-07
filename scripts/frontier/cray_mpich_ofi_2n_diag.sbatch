#!/bin/bash
# Minimal 2-node Cray MPICH C Init_thread/sendrecv diagnostic.
#
# This intentionally avoids Python and mpi4py.  It uses the same Slurm shape as
# the dense train.py MPI smoke so Frontier OFI/PMI/launcher failures can be
# separated from Python binding failures.

#SBATCH -A bif148
#SBATCH -J cray-mpich-ofi-2n-diag
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/cray_mpich_ofi_diag/%x-%j.out
#SBATCH -e logs/frontier/cray_mpich_ofi_diag/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/cray_mpich_ofi_diag}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
SUMMARY_FILE="${RUN_ROOT}/summary.md"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
SMOKE_C="${ARTIFACT_DIR}/cray_mpich_init_sendrecv.c"
SMOKE_BIN="${ARTIFACT_DIR}/cray_mpich_init_sendrecv"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" logs/frontier/cray_mpich_ofi_diag
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules

export REPO
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}

cat > "$SMOKE_C" <<'C'
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int provided = -1;
    int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed rc=%d\n", rc);
        return 10;
    }

    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[256];
    if (gethostname(host, sizeof(host)) != 0) {
        strncpy(host, "unknown", sizeof(host));
        host[sizeof(host) - 1] = '\0';
    }

    int send_value = rank;
    int recv_value = -1;
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    MPI_Status status;

    rc = MPI_Sendrecv(&send_value, 1, MPI_INT, right, 17,
                      &recv_value, 1, MPI_INT, left, 17,
                      MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "rank=%d MPI_Sendrecv failed rc=%d\n", rank, rc);
        MPI_Abort(MPI_COMM_WORLD, 11);
    }

    int host_len = (int)strnlen(host, sizeof(host) - 1) + 1;
    int *host_lens = NULL;
    int *displs = NULL;
    char *all_hosts = NULL;
    if (rank == 0) {
        host_lens = calloc((size_t)size, sizeof(int));
        displs = calloc((size_t)size, sizeof(int));
        if (host_lens == NULL || displs == NULL) {
            fprintf(stderr, "rank=0 allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

    MPI_Gather(&host_len, 1, MPI_INT, host_lens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int total_host_bytes = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total_host_bytes;
            total_host_bytes += host_lens[i];
        }
        all_hosts = calloc((size_t)total_host_bytes, sizeof(char));
        if (all_hosts == NULL) {
            fprintf(stderr, "rank=0 host allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 13);
        }
    }
    MPI_Gatherv(host, host_len, MPI_CHAR, all_hosts, host_lens, displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int unique_hosts = 0;
        for (int i = 0; i < size; ++i) {
            const char *candidate = all_hosts + displs[i];
            int seen = 0;
            for (int j = 0; j < i; ++j) {
                const char *prior = all_hosts + displs[j];
                if (strcmp(candidate, prior) == 0) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                unique_hosts++;
            }
        }
        printf("{\"mpi_init_thread_provided\":%d,\"world_size\":%d,"
               "\"unique_hosts\":%d,\"rank0_received_from\":%d,"
               "\"mpich_gpu_support_enabled\":\"%s\",\"fi_provider\":\"%s\","
               "\"mpich_ofi_nic_policy\":\"%s\"}\n",
               provided,
               size,
               unique_hosts,
               recv_value,
               getenv("MPICH_GPU_SUPPORT_ENABLED") ? getenv("MPICH_GPU_SUPPORT_ENABLED") : "",
               getenv("FI_PROVIDER") ? getenv("FI_PROVIDER") : "",
               getenv("MPICH_OFI_NIC_POLICY") ? getenv("MPICH_OFI_NIC_POLICY") : "");
        fflush(stdout);
    }

    free(host_lens);
    free(displs);
    free(all_hosts);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
C

cc -O2 -Wall -Wextra -o "$SMOKE_BIN" "$SMOKE_C"

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
  local srun_mpi="${SRUN_MPI_TYPE:-}"
  local status
  local srun_cmd=("${base_srun[@]}")

  if [[ -n "$srun_mpi" ]]; then
    srun_cmd+=(--mpi="$srun_mpi")
  fi

  {
    echo "case=${case_name}"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "srun_mpi=${srun_mpi:-default}"
    printf 'env_args='
    printf '%q ' "$@"
    printf '\n'
    env | sort | grep -E '^(FI_|MPICH_|CRAY_|LD_LIBRARY_PATH|CONDA_PREFIX|ROCM|FRONTIER_|OLCF_)=' || true
  } > "$case_env"

  printf '%q ' "${srun_cmd[@]}" env "$@" "$SMOKE_BIN" >> "$COMMAND_FILE"
  printf '\n' >> "$COMMAND_FILE"

  set +e
  "${srun_cmd[@]}" env "$@" "$SMOKE_BIN" 2>&1 | tee "$case_log"
  status=${PIPESTATUS[0]}
  set -e

  echo "$status" > "${ARTIFACT_DIR}/${case_name}.status"
  return 0
}

{
  echo "task_id=${WG_TASK_ID:-resolve-frontier-2n}"
  echo "run_root=$RUN_ROOT"
  echo "summary_file=$SUMMARY_FILE"
  echo "env_file=$ENV_FILE"
  echo "command_file=$COMMAND_FILE"
  echo "smoke_c=$SMOKE_C"
  echo "smoke_bin=$SMOKE_BIN"
  echo "requested_qos=${SLURM_JOB_QOS:-debug}"
  echo "requested_partition=${SLURM_JOB_PARTITION:-batch}"
  echo "requested_nodes=${SLURM_JOB_NUM_NODES:-2}"
  echo "requested_walltime=${SLURM_TIMELIMIT:-00:10:00}"
  echo "requested_node_hours=0.333333"
  echo "mpich_gpu_support_enabled=$MPICH_GPU_SUPPORT_ENABLED"
  echo
  echo "=== modules ==="
  module -t list 2>&1 || true
  echo
  echo "=== cc version ==="
  cc --version 2>&1 || true
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env
} | tee "$ENV_FILE"

: > "$COMMAND_FILE"

run_case baseline \
  MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED"

SRUN_MPI_TYPE=pmi2 \
run_case pmi2 \
  MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED"

SRUN_MPI_TYPE=pmix \
run_case pmix \
  MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED"

run_case cxi_numa \
  MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED" \
  FI_PROVIDER="${FI_PROVIDER:-cxi}" \
  MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-NUMA}" \
  FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}" \
  FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}" \
  FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-2048}"

baseline_status=$(<"${ARTIFACT_DIR}/baseline.status")
pmi2_status=$(<"${ARTIFACT_DIR}/pmi2.status")
pmix_status=$(<"${ARTIFACT_DIR}/pmix.status")
cxi_numa_status=$(<"${ARTIFACT_DIR}/cxi_numa.status")

{
  echo "# 2-node Cray MPICH OFI C diagnostic"
  echo
  echo "- Job id: \`${SLURM_JOB_ID:-manual}\`"
  echo "- QOS/walltime/node-hours: \`${SLURM_JOB_QOS:-debug}\` / \`${SLURM_TIMELIMIT:-00:10:00}\` / \`0.333333\`"
  echo "- Run root: \`$RUN_ROOT\`"
  echo "- Baseline status: \`$baseline_status\`"
  echo "- PMI2 status: \`$pmi2_status\`"
  echo "- PMIx status: \`$pmix_status\`"
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
  grep -E '^(mpich_gpu_support_enabled|requested_|frontier_|FI_|MPICH_|CONDA_PREFIX|LD_LIBRARY_PATH)=' "$ENV_FILE" "$ARTIFACT_DIR"/*.env 2>/dev/null || true
  echo '```'
} | tee "$SUMMARY_FILE"

if [[ "$baseline_status" == "0" || "$pmi2_status" == "0" || "$pmix_status" == "0" || "$cxi_numa_status" == "0" ]]; then
  exit 0
fi
exit "$baseline_status"
