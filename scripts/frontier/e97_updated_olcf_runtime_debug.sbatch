#!/bin/bash
# Debug-QOS E97 resume smoke for the OLCF-aligned runtime candidate.
#
# The active production chain latest.pt is resolved at job start and passed as
# RESUME_CHECKPOINT. CHAIN_LATEST_PATH is then deliberately cleared before the
# shared E97 canary launcher runs, so this job cannot advance a production chain
# pointer.

#SBATCH -A bif148
#SBATCH -J e97-olcf-debug
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/debug/%x-%j.out
#SBATCH -e logs/frontier/debug/%x-%j.err

set -euo pipefail

REPO=${REPO:-/lustre/orion/bif148/scratch/erikgarrison/emender}
ENV_PREFIX=${ENV_PREFIX:-"${REPO}/.envs/olcf-rocm711-torch210-py312"}
GDN2_PATH=${GDN2_PATH:-"${REPO}/src/GatedDeltaNet-2"}
PRODUCTION_LATEST=${PRODUCTION_LATEST:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug}

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

mkdir -p "${REPO}/logs/frontier/debug"
cd "${REPO}"

[[ -L "${PRODUCTION_LATEST}" || -e "${PRODUCTION_LATEST}" ]] || {
  echo "production latest.pt missing: ${PRODUCTION_LATEST}" >&2
  exit 4
}
RESOLVED_PRODUCTION_LATEST=$(readlink -f "${PRODUCTION_LATEST}")
[[ -r "${RESOLVED_PRODUCTION_LATEST}" ]] || {
  echo "resolved production checkpoint is not readable: ${RESOLVED_PRODUCTION_LATEST}" >&2
  exit 4
}
PRODUCTION_LATEST_BEFORE=$(stat -Lc '%N|%s|%Y' "${PRODUCTION_LATEST}")

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load rccl-net-plugin/1.0

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export REPO
export GDN2_PATH
export PYTHONPATH="${REPO}:${GDN2_PATH}:${PYTHONPATH:-}"
export EMENDER_CONDA_ENV="${ENV_PREFIX}"
export OUTPUT_ROOT
export WG_TASK_ID=${WG_TASK_ID:-debug-updated-olcf-runtime}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_active_latest_olcf_runtime_debug}
export RESUME_CHECKPOINT="${RESOLVED_PRODUCTION_LATEST}"
export DEFAULT_RESUME_CHECKPOINT="${RESOLVED_PRODUCTION_LATEST}"
export CHAIN_SEED_CHECKPOINT="${RESOLVED_PRODUCTION_LATEST}"
export CHAIN_LATEST_PATH=
export CHAIN_MANIFEST_PATH=
export CHAIN_UPDATE_ON_FAILURE=0
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_RCCL_ALT_RDZV=${FRONTIER_RCCL_ALT_RDZV:-1}
export FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}
export FRONTIER_RUNTIME_PROFILE=${FRONTIER_RUNTIME_PROFILE:-olcf-rccl-debug}
export FRONTIER_RCCL_NET_PLUGIN_MODULE=${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}
export REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}
export NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-1800}
export FRONTIER_PER_RANK_TRITON_CACHE=${FRONTIER_PER_RANK_TRITON_CACHE:-1}
export TRAIN_MINUTES=${TRAIN_MINUTES:-6}
export BATCH_SIZE=${BATCH_SIZE:-1}
export SAVE_EVERY=${SAVE_EVERY:-999999}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-1}
export LOG_EVERY=${LOG_EVERY:-1}
export VAL_EVERY=${VAL_EVERY:-10000}
export DILOCO_K=${DILOCO_K:-1000000}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-global}
export DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-8}
export DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY:-8}
export DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-0}
export COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-86400}
export WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-1000000}
export DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-1000000}
export DISABLE_SCALAR_STATUS_COLLECTIVES=${DISABLE_SCALAR_STATUS_COLLECTIVES:-1}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-1800}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-00:30:00}
export REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-1.0}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG debug-updated-olcf-runtime: 2-node debug-QOS E97 active-latest resume smoke under OLCF runtime candidate; production chain pointer updates disabled.}

# shellcheck disable=SC1091
source activate "${ENV_PREFIX}"

PLUGIN_STATUS=not-found
IFS=':' read -r -a LD_PARTS <<< "${LD_LIBRARY_PATH:-}"
for lib_dir in "${LD_PARTS[@]}"; do
  if [[ -n "${lib_dir}" && -r "${lib_dir%/}/librccl-net.so" ]]; then
    PLUGIN_STATUS="${lib_dir%/}/librccl-net.so"
    break
  fi
done

echo "=== updated OLCF E97 debug preflight ==="
echo "production_latest=${PRODUCTION_LATEST}"
echo "resolved_production_latest=${RESOLVED_PRODUCTION_LATEST}"
echo "production_latest_before=${PRODUCTION_LATEST_BEFORE}"
echo "chain_latest_path_after_guard='${CHAIN_LATEST_PATH}'"
echo "chain_update_on_failure=${CHAIN_UPDATE_ON_FAILURE}"
echo "env_prefix=${ENV_PREFIX}"
echo "gdn2_path=${GDN2_PATH}"
echo "rccl_net_plugin_status=${PLUGIN_STATUS}"
echo "loaded_modules=${LOADEDMODULES:-}"
python - <<'PY'
import importlib.metadata as md
import json
import os
import sys

names = ["torch", "triton", "fla", "ndm"]
report = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "python": sys.version,
    "loaded_modules": os.environ.get("LOADEDMODULES"),
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "versions": {},
}
for name in names:
    try:
        report["versions"][name] = md.version(name)
    except md.PackageNotFoundError:
        report["versions"][name] = "NOT_INSTALLED"
try:
    import torch
    report["torch_version"] = torch.__version__
    report["torch_hip"] = getattr(torch.version, "hip", None)
except Exception as exc:
    report["torch_error"] = repr(exc)
print(json.dumps(report, indent=2, sort_keys=True))
PY

set +e
bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
STATUS=$?
set -e

PRODUCTION_LATEST_AFTER=$(stat -Lc '%N|%s|%Y' "${PRODUCTION_LATEST}")
echo "production_latest_after=${PRODUCTION_LATEST_AFTER}"
if [[ "${PRODUCTION_LATEST_BEFORE}" != "${PRODUCTION_LATEST_AFTER}" ]]; then
  echo "ERROR: production latest.pt metadata changed during debug smoke" >&2
  exit 44
fi

exit "${STATUS}"
