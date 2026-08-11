#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-instruction-validate
#SBATCH -o /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/validate-%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/validate-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --requeue=0
set -euo pipefail
cd /lustre/orion/bif148/scratch/erikgarrison/emender
source scripts/frontier/activate_emender_frontier.sh
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/emender/tiktoken-cache
"$EMENDER_PYTHON" scripts/validate_e97_instruction_corpus.py \
 --root /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1 --workers 48
