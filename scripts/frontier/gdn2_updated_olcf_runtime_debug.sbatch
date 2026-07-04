#!/bin/bash
# Debug-QOS GDN2 smoke for the OLCF-aligned runtime candidate and RCCL plugin.

#SBATCH -A bif148
#SBATCH -J gdn2-olcf-debug
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug}

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

mkdir -p "${REPO}/logs/frontier/debug"
cd "${REPO}"

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
export SMOKE_VARIANT=${SMOKE_VARIANT:-gdn2-MLP}
export DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
export VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
export TRAIN_MINUTES=${TRAIN_MINUTES:-4}
export SAVE_EVERY=${SAVE_EVERY:-999999}
export LOG_EVERY=${LOG_EVERY:-1}

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

echo "=== updated OLCF GDN2 debug preflight ==="
echo "env_prefix=${ENV_PREFIX}"
echo "gdn2_path=${GDN2_PATH}"
echo "rccl_net_plugin_status=${PLUGIN_STATUS}"
echo "loaded_modules=${LOADEDMODULES:-}"
python - <<'PY'
import importlib.metadata as md
import json
import os
import sys

report = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "python": sys.version,
    "loaded_modules": os.environ.get("LOADEDMODULES"),
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "versions": {},
}
for name in ["torch", "triton", "fla", "ndm"]:
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

exec bash "${REPO}/scripts/frontier/debug_smoke_one_node.slurm"
