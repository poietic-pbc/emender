#!/bin/bash
# Frontier one-node debug smoke template for Emender ROCm/Triton validation.
# Submit from the repo root after creating logs/frontier/debug.
# Example:
#   mkdir -p logs/frontier/debug
#   SMOKE_VARIANT=e97-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm
#   SMOKE_VARIANT=e97-linear-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm
#   SMOKE_VARIANT=gdn2-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm

#SBATCH -A bif148
#SBATCH -J emender-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/debug/%x-%j.out
#SBATCH -e logs/frontier/debug/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-rocm-kernel-integration-smoke}
ACCOUNT=${SLURM_JOB_ACCOUNT:-bif148}
SMOKE_VARIANT=${SMOKE_VARIANT:-e97-MLP}
TRAIN_MINUTES=${TRAIN_MINUTES:-20}
DATA=${DATA:?Set DATA to a short Frontier-readable smoke training text file}
VAL_DATA=${VAL_DATA:-}
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    OUTPUT_ROOT="${MEMBERWORK}/emender/frontier_runs/debug"
  else
    OUTPUT_ROOT="${REPO}/frontier_runs/debug"
  fi
fi
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SMOKE_VARIANT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
STDOUT_PATH="logs/frontier/debug/${SLURM_JOB_NAME:-emender-smoke}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/debug/${SLURM_JOB_NAME:-emender-smoke}-${SLURM_JOB_ID:-manual}.err"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$SUMMARY_DIR"
cd "$REPO"

source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules

if [[ -n "${EMENDER_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$EMENDER_CONDA_ENV"
fi

export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
if frontier_runtime_olcf_rccl_enabled; then
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-net.so}
fi
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -i | awk '{print $1}')}
export MASTER_PORT=${MASTER_PORT:-3442}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
if [[ -z "${GDN2_PATH:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    export GDN2_PATH="${MEMBERWORK}/emender/src/GatedDeltaNet-2"
  else
    export GDN2_PATH="${REPO}/../GatedDeltaNet-2"
  fi
else
  export GDN2_PATH
fi

COMMON_ARGS=(
  --data "$DATA"
  --bf16
  --batch_size "${BATCH_SIZE:-1}"
  --chunk_size "${CHUNK_SIZE:-2048}"
  --tokenizer "${TOKENIZER:-p50k_base}"
  --optimizer schedulefree
  --seed "${SEED:-42}"
  --lr "${LR:-9e-4}"
  --train_minutes "$TRAIN_MINUTES"
  --compile_warmup_steps "${COMPILE_WARMUP_STEPS:-1}"
  --log_every "${LOG_EVERY:-10}"
  --save_every 999999
  --keep_checkpoints 1
  --walltime_final_checkpoint_margin_seconds "${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-600}"
  --walltime_check_every "${WALLTIME_CHECK_EVERY:-1}"
  --output "${RUN_ROOT}/train"
)

if [[ -n "$VAL_DATA" ]]; then
  COMMON_ARGS+=(--val_data "$VAL_DATA" --val_every "${VAL_EVERY:-50}")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -r "$RESUME_CHECKPOINT" ]] || { echo "RESUME_CHECKPOINT is not readable: $RESUME_CHECKPOINT" >&2; exit 4; }
  COMMON_ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

case "$SMOKE_VARIANT" in
  e97-MLP)
    MODEL_ARGS=(
      --level E97
      --linear_state 0
      --use_triton 1
      --n_state 32
      --n_heads 323
      --dim 1536
      --depth 10
      --expansion 1.0
      --use_gate 1
      --gate_activation silu
      --mlp_ratio 1.5
      --mlp_multiple 64
    )
    ;;
  e97-linear-MLP)
    MODEL_ARGS=(
      --level E97
      --linear_state 1
      --use_triton 1
      --use_chunked_e97 1
      --e97_chunk_size "${E97_CHUNK_SIZE:-32}"
      --n_state 32
      --n_heads 128
      --dim 1536
      --depth 21
      --expansion 1.0
      --use_gate 1
      --gate_activation silu
      --mlp_ratio 2.0
      --mlp_multiple 64
    )
    ;;
  gdn2-MLP)
    MODEL_ARGS=(
      --level gdn2-mlp
      --dim 2176
      --depth 12
      --n_heads 30
      --expansion 1.0
      --use_conv 1
      --d_conv 4
      --gdn2_mlp_ratio 3.258732449079677
    )
    ;;
  *)
    echo "Unknown SMOKE_VARIANT='$SMOKE_VARIANT' (expected e97-MLP, e97-linear-MLP, or gdn2-MLP)" >&2
    exit 2
    ;;
esac

case "$SMOKE_VARIANT" in
  e97-MLP)
    KERNEL_SMOKE_CMD=(
      python -m pytest
      tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference
      -q -s
    )
    ;;
  e97-linear-MLP)
    KERNEL_SMOKE_CMD=(
      python -m pytest
      tests/test_e97_chunked.py::test_fused_triton_forward_parity_fp32
      tests/test_e97_chunked.py::test_fused_triton_backward_parity_bf16
      tests/test_e97_chunked.py::test_strong_decay_cluster_backward_finite
      tests/test_e97_chunked.py::test_glog_floor_drift_overflow_finite
      -q -s
    )
    ;;
  gdn2-MLP)
    KERNEL_SMOKE_CMD=(
      python -u scripts/frontier/gdn2_rocm_preflight.py
      --run-fwdbwd
      --bf16
      --dim 128
      --n-heads 4
      --head-dim 16
      --seq-len 16
      --batch-size 1
      --expansion 1.0
      --use-conv 1
      --d-conv 4
      --mlp-ratio 2.0
    )
    ;;
  *)
    KERNEL_SMOKE_CMD=()
    ;;
esac

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)
LIBRCCL_NET_PATH=$(frontier_resolve_librccl_net)
REQUESTED_NODES=${SLURM_JOB_NUM_NODES:-1}
REQUESTED_SECONDS=$(python - <<'PY'
import os
t = os.environ.get("SLURM_TIMELIMIT", "30")
if "-" in t:
    d, t = t.split("-", 1)
    day_s = int(d) * 86400
else:
    day_s = 0
parts = [int(p) for p in t.split(":")]
if len(parts) == 1:
    sec = parts[0] * 60
elif len(parts) == 2:
    sec = parts[0] * 3600 + parts[1] * 60
else:
    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
print(day_s + sec)
PY
)
REQUESTED_NODE_HOURS=$(python - <<PY
nodes = float("${REQUESTED_NODES}")
seconds = float("${REQUESTED_SECONDS}")
print(f"{nodes * seconds / 3600.0:.6f}")
PY
)

{
  echo "=== frontier smoke env ==="
  echo "task_id=$TASK_ID"
  echo "variant=$SMOKE_VARIANT"
  echo "start_utc=$START_UTC"
  echo "repo=$REPO"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo
  echo "=== slurm ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|MASTER|OMP|ROCR|HIP|HSA|GDN2|NDM|DATA|VAL_DATA|TOKENIZER|TIKTOKEN|TRAIN_MINUTES|SMOKE_VARIANT)=' || true
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "librccl_net_path=$LIBRCCL_NET_PATH"
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env "$LIBRCCL_NET_PATH"
  echo
  echo "=== modules ==="
  module list 2>&1
  echo
  echo "=== rocm-smi ==="
  rocm-smi || true
  echo
  echo "=== srun gpu visibility ==="
  srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python - <<'PY' || true
import json
import os

report = {
    "hostname": os.uname().nodename,
    "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
    "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "SLURM_LOCALID": os.environ.get("SLURM_LOCALID"),
}
try:
    import torch
    report["torch.cuda.is_available"] = torch.cuda.is_available()
    report["torch.cuda.device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        report["torch.cuda.current_device"] = torch.cuda.current_device()
        report["torch.cuda.device_name_0"] = torch.cuda.get_device_name(0)
except Exception as exc:
    report["torch_error"] = repr(exc)
print(json.dumps(report, sort_keys=True))
PY
  echo
  echo "=== srun tokenizer cache visibility ==="
  srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

cache_dir = Path(os.environ["TIKTOKEN_CACHE_DIR"])
blob_url = "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
hash_path = cache_dir / hashlib.sha1(blob_url.encode()).hexdigest()
named_path = cache_dir / "p50k_base.tiktoken"
report = {
    "hostname": os.uname().nodename,
    "TIKTOKEN_CACHE_DIR": str(cache_dir),
    "hash_path": str(hash_path),
    "hash_path_readable": os.access(hash_path, os.R_OK),
    "named_path": str(named_path),
    "named_path_readable": os.access(named_path, os.R_OK),
}
if not report["hash_path_readable"]:
    raise SystemExit(f"missing readable tiktoken cache hash file: {hash_path}")
if not report["named_path_readable"]:
    raise SystemExit(f"missing readable named tiktoken cache file: {named_path}")
import tiktoken

enc = tiktoken.get_encoding("p50k_base")
report["p50k_base_n_vocab"] = enc.n_vocab
print(json.dumps(report, sort_keys=True))
PY
  echo
  echo "=== python ==="
  which python
  python --version
  python - <<'PY'
import importlib.metadata as md
for name in ["torch", "triton", "schedulefree", "numpy", "ndm"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
try:
    import torch
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception as exc:
    print(f"torch_import_error={exc!r}")
PY
  echo
  echo "=== pip freeze ==="
  python -m pip freeze
  echo
  echo "=== git ==="
  git rev-parse --show-toplevel
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git status --short
  git diff --stat
  echo
  echo "=== command ==="
  if [[ ${#KERNEL_SMOKE_CMD[@]} -gt 0 ]]; then
    printf '%q ' srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest "${KERNEL_SMOKE_CMD[@]}"
    echo
  fi
  printf '%q ' srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc 'export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK=0; exec python -u train.py "$@"' bash "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}"
  echo
} | tee "$ENV_FILE"

set +e
KERNEL_SMOKE_STATUS=0
if [[ ${#KERNEL_SMOKE_CMD[@]} -gt 0 ]]; then
  srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    "${KERNEL_SMOKE_CMD[@]}" \
    2>&1 | tee "${LOG_DIR}/kernel_smoke.log"
  KERNEL_SMOKE_STATUS=${PIPESTATUS[0]}
else
  echo "no kernel-only smoke configured for ${SMOKE_VARIANT}" | tee "${LOG_DIR}/kernel_smoke.log"
fi
if [[ "$KERNEL_SMOKE_STATUS" -eq 0 ]]; then
  srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest \
    bash -lc 'export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK=0; exec python -u train.py "$@"' bash "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/train.log"
  TRAIN_STATUS=${PIPESTATUS[0]}
  STATUS=$TRAIN_STATUS
else
  TRAIN_STATUS=77
  STATUS=$KERNEL_SMOKE_STATUS
  echo "kernel smoke failed; skipping training smoke" | tee "${LOG_DIR}/train.log"
fi
set -e

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python - <<PY
import json
manifest = {
    "task_id": "${TASK_ID}",
    "variant": "${SMOKE_VARIANT}",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "job_name": "${SLURM_JOB_NAME:-emender-smoke}",
    "account": "${ACCOUNT}",
    "partition": "${SLURM_JOB_PARTITION:-batch}",
    "qos": "${SLURM_JOB_QOS:-debug}",
    "nodes": int("${REQUESTED_NODES}"),
    "requested_walltime": "${SLURM_TIMELIMIT:-00:30:00}",
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "start_utc": "${START_UTC}",
    "end_utc": "${END_UTC}",
    "exit_status": int("${STATUS}"),
    "kernel_smoke_status": int("${KERNEL_SMOKE_STATUS}"),
    "train_status": int("${TRAIN_STATUS}"),
    "git_commit": "${GIT_COMMIT}",
    "frontier_runtime_profile": "${FRONTIER_RUNTIME_PROFILE:-default}",
    "frontier_enable_olcf_rccl_plugin": int("${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-0}"),
    "frontier_rocm_module": "${FRONTIER_ROCM_MODULE:-rocm/7.1.1}",
    "frontier_rccl_net_plugin_module": "${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}",
    "olcf_ofi_nccl_root": "${OLCF_OFI_NCCL_ROOT:-}",
    "nccl_net_plugin": "${NCCL_NET_PLUGIN:-}",
    "librccl_net_path": "${LIBRCCL_NET_PATH}",
    "tiktoken_cache_dir": "${TIKTOKEN_CACHE_DIR}",
    "resume_checkpoint": "${RESUME_CHECKPOINT:-}",
    "stdout": "${STDOUT_PATH}",
    "stderr": "${STDERR_PATH}",
    "run_root": "${RUN_ROOT}",
    "env_file": "${ENV_FILE}",
    "kernel_smoke_log": "${LOG_DIR}/kernel_smoke.log",
    "train_log": "${LOG_DIR}/train.log",
}
with open("${MANIFEST_FILE}", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

SUMMARY_METRICS=$(python - <<PY
import re
from pathlib import Path

train_log = Path("${LOG_DIR}/train.log")
kernel_log = Path("${LOG_DIR}/kernel_smoke.log")

def read_lines(path):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()

train_lines = read_lines(train_log)
kernel_lines = read_lines(kernel_log)
step_lines = [ln for ln in train_lines if "loss" in ln and ("tok/s" in ln or "global_tok/s" in ln)]
loss_lines = [ln for ln in train_lines if "loss" in ln]
runtime_lines = [ln for ln in train_lines if "[e97-runtime]" in ln or "[gdn2-runtime]" in ln or "GDN2 external deps" in ln]
memory_lines = [ln for ln in train_lines + kernel_lines if "memory" in ln.lower() or "MiB" in ln or "GiB" in ln]
error_lines = [
    ln for ln in train_lines + kernel_lines
    if re.search(r"(traceback|error|exception|failed|nan|inf)", ln, re.IGNORECASE)
]

print(f"final_metric_line: {step_lines[-1] if step_lines else 'not observed'}")
print(f"final_loss_line: {loss_lines[-1] if loss_lines else 'not observed'}")
print(f"throughput_line: {step_lines[-1] if step_lines else 'not observed'}")
print(f"runtime_path_line: {runtime_lines[-1] if runtime_lines else 'not observed'}")
print(f"peak_memory_line: {memory_lines[-1] if memory_lines else 'not observed'}")
print(f"first_actionable_error: {error_lines[0] if error_lines else 'none observed'}")
PY
)

cat > "$SUMMARY_FILE" <<EOF
# Frontier Debug Smoke Summary

- task: ${TASK_ID}
- variant: ${SMOKE_VARIANT}
- job_id: ${SLURM_JOB_ID:-manual}
- account: ${ACCOUNT}
- partition: ${SLURM_JOB_PARTITION:-batch}
- qos: ${SLURM_JOB_QOS:-debug}
- nodes: ${REQUESTED_NODES}
- requested_walltime: ${SLURM_TIMELIMIT:-00:30:00}
- elapsed: fill from sacct after completion
- requested_node_hours: ${REQUESTED_NODE_HOURS}
- actual_node_hours: fill from sacct after completion
- git_commit: ${GIT_COMMIT}
- stdout: ${STDOUT_PATH}
- stderr: ${STDERR_PATH}
- artifacts: ${RUN_ROOT}
- kernel_smoke_status: ${KERNEL_SMOKE_STATUS}
- train_status: ${TRAIN_STATUS}
- exit_status: ${STATUS}

## Observation
$(printf '%s\n' "$SUMMARY_METRICS")

## Failure Mode
Use first_actionable_error above plus ${LOG_DIR}/kernel_smoke.log and
${LOG_DIR}/train.log. Kernel failure skips training by design so the blocker is
not hidden behind a later DDP launch.

## Interpretation Boundary
This is one short debug-QOS run. Treat it as launch/import/kernel/memory/loss
sanity evidence only, not as extended-allocation readiness.
EOF

echo "manifest=$MANIFEST_FILE"
echo "summary=$SUMMARY_FILE"
echo "train_log=${LOG_DIR}/train.log"

exit "$STATUS"
