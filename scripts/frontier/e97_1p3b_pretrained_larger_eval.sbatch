#!/bin/bash
# Eval-only larger deterministic comparison for evaluate-e97-1-3b.
#
# Scores the staged E97 1.3B pretrained source checkpoint plus the completed
# 8-node K40 and K160 final checkpoints on one deterministic tensor that is
# materially larger than the 16,384-token smoke slice. This script does not run
# train.py and does not submit scaleout work.

#SBATCH -A bif148
#SBATCH -J e97-1p3b-large-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/eval/%x-%j.out
#SBATCH -e logs/frontier/eval/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-evaluate-e97-1-3b}
VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
TOKENIZER=${TOKENIZER:-p50k_base}
MAX_CHUNKS=${MAX_CHUNKS:-128}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
OUT_ROOT=${OUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-1p3b-pretrained-larger}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${RUN_ROOT:-"${OUT_ROOT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"}
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
RESULT_CSV=${RESULT_CSV:-"${ARTIFACT_DIR}/source_k40_k160_larger_eval.csv"}
TENSOR_PATH=${TENSOR_PATH:-"${ARTIFACT_DIR}/commapile_mainmix_val_p50k_2048_x${MAX_CHUNKS}.pt"}
TENSOR_META=${TENSOR_META:-"${ARTIFACT_DIR}/commapile_mainmix_val_p50k_2048_x${MAX_CHUNKS}.json"}
MANIFEST="${ARTIFACT_DIR}/manifest.json"
ONLY_LABELS=${ONLY_LABELS:-source,k40,k160}

CKPT_SOURCE="/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt"
CKPT_K40="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/train/emender_E97_1.3B_20260625_140940/latest.pt"
CKPT_K160="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/train/emender_E97_1.3B_20260625_143545/latest.pt"
CKPT_K320="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z/train/emender_E97_1.3B_20260625_152430/latest.pt"

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

[[ -r "$VAL_DATA" ]] || { echo "VAL_DATA is not readable: $VAL_DATA" >&2; exit 4; }
[[ -r "$CKPT_SOURCE" ]] || { echo "checkpoint is not readable: $CKPT_SOURCE" >&2; exit 4; }
[[ -r "$CKPT_K40" ]] || { echo "checkpoint is not readable: $CKPT_K40" >&2; exit 4; }
[[ -r "$CKPT_K160" ]] || { echo "checkpoint is not readable: $CKPT_K160" >&2; exit 4; }

case ",${ONLY_LABELS}," in
  *",k320,"*) [[ -r "$CKPT_K320" ]] || { echo "checkpoint is not readable: $CKPT_K320" >&2; exit 4; } ;;
esac

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)

python - <<'PY' "$VAL_DATA" "$TOKENIZER" "$TENSOR_PATH" "$TENSOR_META" "$MAX_CHUNKS"
import hashlib
import json
import sys
from pathlib import Path

import torch
import tiktoken

val_path = Path(sys.argv[1])
tokenizer = sys.argv[2]
tensor_path = Path(sys.argv[3])
meta_path = Path(sys.argv[4])
max_chunks = int(sys.argv[5])
chunk = 2048

raw = val_path.read_bytes()
text = raw.decode("utf-8", errors="replace")
enc = tiktoken.get_encoding(tokenizer)
tokens = enc.encode(text, disallowed_special=())
n_chunks = (len(tokens) - 1) // chunk
if n_chunks < 1:
    raise SystemExit(f"not enough tokens in {val_path}: {len(tokens)}")
if max_chunks > 0:
    n_chunks = min(n_chunks, max_chunks)
usable = tokens[: n_chunks * chunk + 1]
rows = [
    usable[i * chunk : i * chunk + chunk + 1]
    for i in range(n_chunks)
]
tensor = torch.tensor(rows, dtype=torch.long)
scored = tensor[:, 1:]
total_bytes = sum(len(enc.decode(row.tolist()).encode("utf-8")) for row in scored)
payload = {
    "chunks": tensor,
    "bytes_per_token": total_bytes / scored.numel(),
    "scored_tokens": int(scored.numel()),
    "total_utf8_bytes": int(total_bytes),
    "chunk_size": chunk,
    "n_chunks": int(n_chunks),
    "tokenizer": tokenizer,
    "source": str(val_path),
    "source_bytes": len(raw),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "construction": "contiguous non-overlapping full 2048-token chunks from start of validation file",
}
tensor_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, tensor_path)
tensor_sha256 = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
meta = {k: v for k, v in payload.items() if k != "chunks"}
meta.update(
    tensor_path=str(tensor_path),
    tensor_sha256=tensor_sha256,
    max_chunks_requested=max_chunks,
)
meta_path.write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2), flush=True)
PY

cat > "$MANIFEST" <<EOF
{
  "task_id": "$TASK_ID",
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "start_utc": "$START_UTC",
  "repo": "$REPO",
  "git_commit": "$GIT_COMMIT",
  "nodes": ${SLURM_JOB_NUM_NODES:-1},
  "requested_walltime": "${SLURM_TIMELIMIT:-02:00:00}",
  "requested_node_hours": 2.0,
  "eval_code": "scripts/eval_checkpoint.py",
  "eval_mode": "forward-only checkpoint scoring; y-mode saved",
  "training_submitted": false,
  "scaleout_submitted": false,
  "tokenizer": "$TOKENIZER",
  "batch_size": "$HELDOUT_EVAL_BS",
  "y_mode": "saved",
  "validation_source": "$VAL_DATA",
  "max_chunks": $MAX_CHUNKS,
  "scoring_tensor": "$TENSOR_PATH",
  "tensor_metadata": "$TENSOR_META",
  "result_csv": "$RESULT_CSV",
  "only_labels": "$ONLY_LABELS",
  "checkpoints": {
    "source": "$CKPT_SOURCE",
    "k40": "$CKPT_K40",
    "k160": "$CKPT_K160",
    "k320_optional": "$CKPT_K320"
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

should_run () {
  local label="$1"
  case ",${ONLY_LABELS}," in
    *",${label},"*) return 0 ;;
    *) return 1 ;;
  esac
}

should_run source && run_eval source "$CKPT_SOURCE"
should_run k40 && run_eval k40 "$CKPT_K40"
should_run k160 && run_eval k160 "$CKPT_K160"
should_run k320 && run_eval k320 "$CKPT_K320"

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
echo "TENSOR_META=$TENSOR_META"
echo "MANIFEST=$MANIFEST"
