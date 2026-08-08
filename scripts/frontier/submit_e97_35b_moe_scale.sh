#!/bin/bash
# Submit one immutable E97 MoE scale/production execution epoch. This wrapper
# performs no retry, dependency, or automatic promotion.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}
NODES=${NODES:?set NODES to the reviewed scale}
QOS=${QOS:-debug}
TIME_LIMIT=${TIME_LIMIT:-00:40:00}
TRAIN_MINUTES=${TRAIN_MINUTES:-0}
MAX_STEPS=${MAX_STEPS:-200}
RUN_ID=${RUN_ID:-e97-35b-moe-production}
RUN_ROOT=${RUN_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe-production/runs/$RUN_ID}
SAMPLER_SCHEMA=${SAMPLER_SCHEMA:-}
SAMPLER_CORPUS_SHA256=${SAMPLER_CORPUS_SHA256:-}
SAMPLER_TOKENIZER_SHA256=${SAMPLER_TOKENIZER_SHA256:-}
SAMPLER_KEY=${SAMPLER_KEY:-}
SAMPLER_DATA_WORLD_SIZE=${SAMPLER_DATA_WORLD_SIZE:-}
SAMPLER_TRANSITION_FROM_LEGACY=${SAMPLER_TRANSITION_FROM_LEGACY:-0}

[[ "$NODES" =~ ^(8|32|128|256)$ ]] || { echo "NODES must be an explicitly reviewed 8, 32, 128, or 256" >&2; exit 64; }
[[ "$QOS" == debug || "$QOS" == normal ]] || { echo "QOS must be debug or normal" >&2; exit 64; }
[[ "$RUN_ROOT" == /* && "$RUN_ID" != */* ]] || { echo "invalid stable run identity" >&2; exit 64; }
cd "$PROJECT_ROOT"
source scripts/frontier/activate_emender_frontier.sh
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked source must be clean" >&2; exit 64; }
SOURCE_COMMIT=$(git rev-parse HEAD)
[[ "$SOURCE_COMMIT" == $(git rev-parse origin/main) ]] || {
  echo "submission requires HEAD == origin/main" >&2; exit 64;
}
mkdir -p logs/frontier/e97_moe "$RUN_ROOT"

export_args="ALL,SOURCE_COMMIT=$SOURCE_COMMIT,RUN_ID=$RUN_ID,RUN_ROOT=$RUN_ROOT,TRAIN_MINUTES=$TRAIN_MINUTES,MAX_STEPS=$MAX_STEPS,EXPECTED_NODES=$NODES,EXPECTED_PARTITION=batch,EXPECTED_QOS=$QOS,EXPECTED_TIME_LIMIT=$TIME_LIMIT,SAMPLER_SCHEMA=$SAMPLER_SCHEMA,SAMPLER_CORPUS_SHA256=$SAMPLER_CORPUS_SHA256,SAMPLER_TOKENIZER_SHA256=$SAMPLER_TOKENIZER_SHA256,SAMPLER_KEY=$SAMPLER_KEY,SAMPLER_DATA_WORLD_SIZE=$SAMPLER_DATA_WORLD_SIZE,SAMPLER_TRANSITION_FROM_LEGACY=$SAMPLER_TRANSITION_FROM_LEGACY"
job_id=$(sbatch --parsable --nodes="$NODES" --ntasks=$((NODES * 8)) \
  --qos="$QOS" --time="$TIME_LIMIT" --job-name="e97-moe-${NODES}n" \
  --export="$export_args" scripts/frontier/e97_35b_moe_production.sbatch)
echo "$job_id"
for _ in $(seq 1 30); do
  binding=$(squeue -h -j "$job_id" -o '%i|%P|%q|%T|%D|%l' || true)
  if [[ -n "$binding" ]]; then
    printf '%s\n' "$binding" | tee "$RUN_ROOT/submit-binding-$job_id.txt"
    [[ "$binding" == "$job_id|batch|$QOS|"*"|$NODES|"* ]] || {
      echo "submitted scheduler binding mismatch; cancelling $job_id" >&2
      scancel "$job_id"; exit 65;
    }
    exit 0
  fi
  sleep 1
done
echo "unable to verify submitted Partition and QOS; cancelling $job_id" >&2
scancel "$job_id"
exit 65
