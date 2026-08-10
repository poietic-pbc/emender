#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-instruction-download
#SBATCH -o /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/download-%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/download-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --requeue=0

set -euo pipefail
cd /lustre/orion/bif148/scratch/erikgarrison/emender
source scripts/frontier/activate_emender_frontier.sh
export HF_TOKEN="$("$EMENDER_PYTHON" -c 'from huggingface_hub import get_token; print(get_token() or "")')"
test -n "$HF_TOKEN" || { echo 'missing Hugging Face token' >&2; exit 2; }
export HF_HOME=/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/hf-home
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME"
"$EMENDER_PYTHON" scripts/download_e97_instruction_sources.py \
  --spec configs/frontier/e97_instruction_50b_sources.json \
  --output-root /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/sources \
  --max-workers 8
