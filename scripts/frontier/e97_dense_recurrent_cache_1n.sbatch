#!/bin/bash
#SBATCH -J e97-dense-cache
#SBATCH -A bif148
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 00:45:00
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_dense_agent/%x-%j.out
#SBATCH -e logs/frontier/e97_dense_agent/%x-%j.err
set -euo pipefail

: "${SOURCE_COMMIT:?set SOURCE_COMMIT to an immutable commit}"
: "${RUN_ROOT:?set RUN_ROOT to the qualification artifact root}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CHECKPOINT=${CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/train-128u-lr1e5-v1/checkpoints/checkpoint_agent_sft_u000128.pt}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-1864f1c9138b17e99da698f3272017c7962da676f0a21d68a42ad98868bbbced}
ARGS_JSON=${ARGS_JSON:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/args.json}
EXPECTED_PARTITION=batch
EXPECTED_QOS=debug

cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$RUN_ROOT"/{identity,results,shards,sources,terminal} logs/frontier/e97_dense_agent
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' \
  > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
grep -q "Partition=$EXPECTED_PARTITION" "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
grep -q "QOS=$EXPECTED_QOS" "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
grep -q "Requeue=0" "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
grep -q "Partition=$EXPECTED_PARTITION|QOS=$EXPECTED_QOS" "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"

SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
mkdir -p "$SNAPSHOT"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null

LOCAL_CHECKPOINT="/tmp/e97-dense-cache-$SLURM_JOB_ID.pt"
LOCAL_ARGS="/tmp/e97-dense-cache-$SLURM_JOB_ID-args.json"
sbcast -f "$CHECKPOINT" "$LOCAL_CHECKPOINT"
sbcast -f "$ARGS_JSON" "$LOCAL_ARGS"
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$LOCAL_CHECKPOINT" | sha256sum -c - 

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'$SNAPSHOT'"
  export PYTHONPATH="'$SNAPSHOT'"
  export LOCAL_RANK=0
  export OMP_NUM_THREADS=7
  export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
  export TRITON_CACHE_DIR=/tmp/e97-dense-cache-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/qualify_e97_recurrent_cache.py \
    --checkpoint "'$LOCAL_CHECKPOINT'" \
    --args-json "'$LOCAL_ARGS'" \
    --rank "$SLURM_PROCID" \
    --world "$SLURM_NTASKS" \
    --output "'$RUN_ROOT'/shards/shard-${SLURM_PROCID}.json"
'
rc=$?
set -e

if (( rc == 0 )); then
  "$EMENDER_PYTHON" - "$RUN_ROOT" <<'PY' || rc=$?
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
shards = [json.loads(path.read_text()) for path in sorted((root / "shards").glob("shard-*.json"))]
if len(shards) != 8:
    raise SystemExit(f"expected 8 shards, found {len(shards)}")
summary = {
    "schema": "emender-e97-recurrent-cache-qualification-summary-v1",
    "status": "pass" if all(row["status"] == "pass" for row in shards) else "fail",
    "shards": len(shards),
    "hidden_max_abs_difference": max(row["measurements"]["hidden_max_abs_difference"] for row in shards),
    "next_logits_max_abs_difference": max(row["measurements"]["next_logits_max_abs_difference"] for row in shards),
    "post_generation_hidden_max_abs_difference": max(row["measurements"]["post_generation_hidden_max_abs_difference"] for row in shards),
    "post_generation_logits_max_abs_difference": max(row["measurements"]["post_generation_logits_max_abs_difference"] for row in shards),
    "generation_tokens_per_shard": sorted({len(row["generation_tokens"]) for row in shards}),
    "cache_state_bytes_per_session": sorted({row["measurements"]["cache_state_bytes_including_boundary_logits"] for row in shards}),
    "peak_allocated_bytes_per_rank": max(row["measurements"]["peak_allocated_bytes"] for row in shards),
    "all_checks": all(all(row["checks"].values()) for row in shards),
}
(root / "results" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
if summary["status"] != "pass" or not summary["all_checks"]:
    raise SystemExit(1)
PY
fi
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"
exit "$rc"
