#!/bin/bash
# Forward-only fixed validation scoring for the K80 diagnostic checkpoint.
# This is not a training job. It reuses the fixed tensor from fixed-eval-e97-32.

#SBATCH -A bif148
#SBATCH -J e97-k80-fixed-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/eval/%x-%j.out
#SBATCH -e logs/frontier/eval/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-run-e97-32-node}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
TENSOR_PATH=${TENSOR_PATH:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt}
OUT_ROOT=${OUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-MLP-k80}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${RUN_ROOT:-"${OUT_ROOT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"}
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
RESULT_CSV=${RESULT_CSV:-"${ARTIFACT_DIR}/e97_fixed_eval_k80_diagnostic.csv"}
MANIFEST="${ARTIFACT_DIR}/manifest.json"

CKPT_16="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt"
CKPT_K80="${CKPT_K80:?Set CKPT_K80 to the K80 diagnostic latest.pt or final checkpoint}"

mkdir -p logs/frontier/eval "$ARTIFACT_DIR"
cd "$REPO"

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

if [[ -n "${EMENDER_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$EMENDER_CONDA_ENV"
fi

export TIKTOKEN_CACHE_DIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
export EVAL_CHECKPOINT_GPU_LEASED=1
export HELDOUT_EVAL_BS=${HELDOUT_EVAL_BS:-1}

[[ -r "$TENSOR_PATH" ]] || { echo "TENSOR_PATH is not readable: $TENSOR_PATH" >&2; exit 4; }
[[ -r "$CKPT_16" ]] || { echo "checkpoint is not readable: $CKPT_16" >&2; exit 4; }
[[ -r "$CKPT_K80" ]] || { echo "checkpoint is not readable: $CKPT_K80" >&2; exit 4; }

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)

cat > "$MANIFEST" <<EOF
{
  "task_id": "$TASK_ID",
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "start_utc": "$START_UTC",
  "repo": "$REPO",
  "git_commit": "$GIT_COMMIT",
  "nodes": ${SLURM_JOB_NUM_NODES:-1},
  "requested_walltime": "${SLURM_TIMELIMIT:-01:00:00}",
  "requested_node_hours": 1.0,
  "eval_code": "scripts/eval_checkpoint.py",
  "eval_mode": "forward-only checkpoint scoring; y-mode saved",
  "training_submitted": false,
  "research_arm": "E97-MLP only",
  "scoring_tensor": "$TENSOR_PATH",
  "result_csv": "$RESULT_CSV",
  "checkpoints": {
    "source_16node": "$CKPT_16",
    "k80_32node": "$CKPT_K80"
  }
}
EOF

run_eval () {
  local label="$1"
  local ckpt="$2"
  echo "=== scoring ${label}: ${ckpt}"
  python scripts/eval_checkpoint.py \
    --checkpoint "$ckpt" \
    --scoring-tensor "$TENSOR_PATH" \
    --y-mode saved \
    --batch-size "${HELDOUT_EVAL_BS}" \
    --out "$RESULT_CSV"
}

run_eval source_16node "$CKPT_16"
run_eval k80_32node "$CKPT_K80"

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python - <<'PY' "$MANIFEST" "$END_UTC"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data["end_utc"] = sys.argv[2]
path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "RESULT_CSV=$RESULT_CSV"
echo "MANIFEST=$MANIFEST"
