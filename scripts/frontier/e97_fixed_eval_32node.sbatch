#!/bin/bash
# Fixed validation scoring for fixed-eval-e97-32.
#
# Scores exactly the three E97-MLP checkpoints named in the WG task on one
# Frontier node. This is forward-only evaluation; it does not launch training.

#SBATCH -A bif148
#SBATCH -J e97-fixed-eval
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
TASK_ID=${WG_TASK_ID:-fixed-eval-e97-32}
VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
TOKENIZER=${TOKENIZER:-p50k_base}
MAX_CHUNKS=${MAX_CHUNKS:-8}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
OUT_ROOT=${OUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${RUN_ROOT:-"${OUT_ROOT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"}
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
RESULT_CSV=${RESULT_CSV:-"${ARTIFACT_DIR}/e97_fixed_eval_32node.csv"}
TENSOR_PATH=${TENSOR_PATH:-"${ARTIFACT_DIR}/commapile_mainmix_val_smoke_p50k_2048.pt"}
MANIFEST="${ARTIFACT_DIR}/manifest.json"
ONLY_LABELS=${ONLY_LABELS:-source_16node,original_32node,retry_32node}

CKPT_16="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt"
CKPT_32_ORIG="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/train/emender_E97_1.3B_20260624_070413/latest.pt"
CKPT_32_RETRY="/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/train/emender_E97_1.3B_20260624_075706/latest.pt"

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
[[ -r "$CKPT_16" ]] || { echo "checkpoint is not readable: $CKPT_16" >&2; exit 4; }
[[ -r "$CKPT_32_ORIG" ]] || { echo "checkpoint is not readable: $CKPT_32_ORIG" >&2; exit 4; }
[[ -r "$CKPT_32_RETRY" ]] || { echo "checkpoint is not readable: $CKPT_32_RETRY" >&2; exit 4; }

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python - <<'PY' "$VAL_DATA" "$TOKENIZER" "$TENSOR_PATH" "$MAX_CHUNKS"
import json
import sys
from pathlib import Path

import torch
import tiktoken

val_path = Path(sys.argv[1])
tokenizer = sys.argv[2]
out = Path(sys.argv[3])
max_chunks = int(sys.argv[4])
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
    "construction": "contiguous non-overlapping full 2048-token chunks from start of validation file",
}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, out)
print(json.dumps({k: v for k, v in payload.items() if k != "chunks"}, indent=2))
PY

cat > "$MANIFEST" <<EOF
{
  "task_id": "$TASK_ID",
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "start_utc": "$START_UTC",
  "repo": "$REPO",
  "git_commit": "$(git rev-parse HEAD)",
  "nodes": ${SLURM_JOB_NUM_NODES:-1},
  "requested_walltime": "${SLURM_TIMELIMIT:-02:00:00}",
  "requested_node_hours": 2.0,
  "eval_code": "scripts/eval_checkpoint.py",
  "eval_mode": "forward-only checkpoint scoring; y-mode saved",
  "training_submitted": false,
  "research_arm": "E97-MLP only",
  "validation_source": "$VAL_DATA",
  "max_chunks": $MAX_CHUNKS,
  "scoring_tensor": "$TENSOR_PATH",
  "result_csv": "$RESULT_CSV",
  "checkpoints": {
    "source_16node": "$CKPT_16",
    "original_32node": "$CKPT_32_ORIG",
    "retry_32node": "$CKPT_32_RETRY"
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

should_run source_16node && run_eval source_16node "$CKPT_16"
should_run original_32node && run_eval original_32node "$CKPT_32_ORIG"
should_run retry_32node && run_eval retry_32node "$CKPT_32_RETRY"

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
