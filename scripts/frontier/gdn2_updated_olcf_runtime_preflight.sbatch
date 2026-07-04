#!/bin/bash
# Direct GDN2/FLA debug-QOS preflight for the OLCF-aligned runtime candidate.
#
# This intentionally avoids the older generic debug wrapper because that wrapper
# can reset conda state before the training command. The only goal here is to
# prove candidate-runtime GDN2 imports and bf16 fwd/bwd execution inside a Slurm
# job with rccl-net-plugin/1.0 loaded.

#SBATCH -A bif148
#SBATCH -J gdn2-olcf-pre
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node=1
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
TASK_ID=${WG_TASK_ID:-debug-updated-olcf-runtime}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/gdn2_olcf_runtime_preflight/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}" "${SUMMARY_DIR}" "${REPO}/logs/frontier/debug"
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
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}

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

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
REQUESTED_SECONDS=${REQUESTED_SECONDS:-600}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-0.166667}
GIT_COMMIT=$(git rev-parse HEAD)

{
  echo "=== gdn2 olcf preflight env ==="
  echo "task_id=${TASK_ID}"
  echo "start_utc=${START_UTC}"
  echo "repo=${REPO}"
  echo "run_root=${RUN_ROOT}"
  echo "env_prefix=${ENV_PREFIX}"
  echo "gdn2_path=${GDN2_PATH}"
  echo "rccl_net_plugin_status=${PLUGIN_STATUS}"
  echo "requested_node_hours=${REQUESTED_NODE_HOURS}"
  echo "loaded_modules=${LOADEDMODULES:-}"
  echo
  echo "=== python versions ==="
  which python
  python --version
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
    report["torch_cuda_is_available"] = torch.cuda.is_available()
    report["torch_cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception as exc:
    report["torch_error"] = repr(exc)
print(json.dumps(report, indent=2, sort_keys=True))
PY
} | tee "${ENV_FILE}"

set +e
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
  python -u scripts/frontier/gdn2_rocm_preflight.py \
    --run-fwdbwd \
    --bf16 \
    --batch-size 1 \
    --seq-len 16 \
    --dim 128 \
    --n-heads 4 \
    --head-dim 16 \
    --expansion 1.0 \
    --use-conv 1 \
    --d-conv 4 \
    --mlp-ratio 2.0 \
  2>&1 | tee "${LOG_DIR}/gdn2_preflight.log"
STATUS=${PIPESTATUS[0]}
set -e

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python - <<PY
import json
manifest = {
    "task_id": "${TASK_ID}",
    "variant": "gdn2_olcf_runtime_preflight",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "job_name": "${SLURM_JOB_NAME:-gdn2-olcf-pre}",
    "nodes": int("${SLURM_JOB_NUM_NODES:-1}"),
    "requested_seconds": ${REQUESTED_SECONDS},
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "start_utc": "${START_UTC}",
    "end_utc": "${END_UTC}",
    "exit_status": int("${STATUS}"),
    "git_commit": "${GIT_COMMIT}",
    "env_prefix": "${ENV_PREFIX}",
    "gdn2_path": "${GDN2_PATH}",
    "rccl_net_plugin_status": "${PLUGIN_STATUS}",
    "run_root": "${RUN_ROOT}",
    "env_file": "${ENV_FILE}",
    "preflight_log": "${LOG_DIR}/gdn2_preflight.log",
    "summary": "${SUMMARY_FILE}",
}
with open("${MANIFEST_FILE}", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

python - "${LOG_DIR}/gdn2_preflight.log" "${SUMMARY_FILE}" "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
status = sys.argv[3]
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
lines = text.splitlines()
json_start = next((i for i, line in enumerate(lines) if line.strip().startswith("{")), None)
payload = None
if json_start is not None:
    try:
        payload = json.loads("\n".join(lines[json_start:]))
    except Exception:
        payload = None

with summary_path.open("w", encoding="utf-8") as out:
    out.write("# GDN2 OLCF Runtime Preflight Summary\n\n")
    out.write(f"Status: `{status}`\n\n")
    if payload:
        out.write("## Parsed Result\n\n")
        out.write(f"- ok: `{payload.get('ok')}`\n")
        out.write(f"- loss: `{payload.get('loss')}`\n")
        out.write(f"- finite_loss: `{payload.get('finite_loss')}`\n")
        out.write(f"- finite_output: `{payload.get('finite_output')}`\n")
        out.write(f"- finite_input_grad: `{payload.get('finite_input_grad')}`\n")
        out.write(f"- finite_param_grads: `{payload.get('finite_param_grads')}`\n")
        out.write(f"- torch_version: `{payload.get('torch_version')}`\n")
        out.write(f"- torch_version_hip: `{payload.get('torch_version_hip')}`\n")
        out.write(f"- fla_version: `{payload.get('fla_version')}`\n")
    else:
        out.write("Could not parse final JSON payload; inspect preflight log.\n")
    out.write("\n## Last Log Lines\n\n```text\n")
    out.write("\n".join(lines[-40:]))
    out.write("\n```\n")
PY

echo "manifest=${MANIFEST_FILE}"
echo "summary=${SUMMARY_FILE}"
echo "preflight_log=${LOG_DIR}/gdn2_preflight.log"

exit "${STATUS}"
