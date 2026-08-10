#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-instruction-build
#SBATCH -o /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/build-%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/build-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:55:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --requeue=0

set -euo pipefail
cd /lustre/orion/bif148/scratch/erikgarrison/emender
source scripts/frontier/activate_emender_frontier.sh
ROOT=/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1
"$EMENDER_PYTHON" scripts/build_e97_instruction_corpus.py \
  --spec configs/frontier/e97_instruction_50b_sources.json \
  --inventory-root "$ROOT/inventory" \
  --output-root "$ROOT" \
  --readme docs/E97_INSTRUCTION_CORPUS_50B.md \
  --buckets 256
