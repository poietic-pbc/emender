#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-tulu3-sft-build
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
INPUT_ROOT=${INPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/raw}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/authority}
WORKERS=${WORKERS:-48}
record=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$record"
[[ "$record" == *"NumNodes=1"* && "$record" == *"Partition=batch"* \
   && "$record" == *"QOS=debug"* && "$record" == *"TimeLimit=02:00:00"* \
   && "$record" == *"Requeue=0"* ]] || { echo "scheduler binding mismatch" >&2; exit 65; }
[[ -d "$INPUT_ROOT/data" && ! -e "$OUTPUT_ROOT/manifest.json" ]] || {
  echo "input missing or output authority already exists" >&2; exit 66; }
cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$OUTPUT_ROOT" logs/frontier/e97_moe
printf '%s\n' "$record" > "$OUTPUT_ROOT/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$OUTPUT_ROOT/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$OUTPUT_ROOT/source-commit.txt"
SNAPSHOT="$OUTPUT_ROOT/source-$SOURCE_COMMIT"
temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"
mkdir -p "$temporary"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"
mv -T "$temporary" "$SNAPSHOT"
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
cd "$SNAPSHOT"
export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
"$EMENDER_PYTHON" scripts/build_e97_tulu3_sft.py \
  --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" --workers "$WORKERS"
"$EMENDER_PYTHON" scripts/validate_e97_tulu3_sft.py \
  --root "$OUTPUT_ROOT" --samples 4096
printf '0\n' > "$OUTPUT_ROOT/job-${SLURM_JOB_ID}-rc.txt"
