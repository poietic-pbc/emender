#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-instruction-inventory
#SBATCH -o /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/inventory-%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/inventory-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 01:55:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --requeue=0

set -euo pipefail
cd /lustre/orion/bif148/scratch/erikgarrison/emender
source scripts/frontier/activate_emender_frontier.sh
ROOT=/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1
SRC=$ROOT/sources
INV=$ROOT/inventory
[[ -s "$ROOT/download-receipt.json" ]] || { echo 'complete download receipt missing' >&2; exit 2; }
mkdir -p "$INV"

"$EMENDER_PYTHON" scripts/restore_nemotron_chat_prompts.py \
  --input "$SRC/nemotron_instruction_chat_v3/data/chat.jsonl" \
  --output "$SRC/nemotron_instruction_chat_v3/data/chat.with_prompts.jsonl" \
  --lmsys-root "$SRC/lmsys_chat_1m" \
  --wildchat-root "$SRC/wildchat_1m" \
  --receipt "$ROOT/nemotron-prompt-restoration.json"

sources=(nemotron_instruction_chat_v3 nemotron_swe_v3 open_swe_traces nemotron_terminal swe_chat agent_trove gair_openswe swe_next open_thoughts3)
pids=()
for source in "${sources[@]}"; do
  srun --exclusive -N1 -n1 -c7 --cpu-bind=cores \
    "$EMENDER_PYTHON" scripts/inventory_e97_instruction_source.py \
      --source "$source" --input-root "$SRC/$source" --output-root "$INV" \
      >"$ROOT/inventory-$source.out" 2>"$ROOT/inventory-$source.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo 'one or more source inventories failed' >&2; exit 3; }
cp docs/E97_INSTRUCTION_CORPUS_50B.md "$ROOT/README.md.partial"
cp configs/frontier/e97_instruction_50b_sources.json "$ROOT/e97_instruction_50b_v1.sources.json.partial"
echo INVENTORY_COMPLETE
