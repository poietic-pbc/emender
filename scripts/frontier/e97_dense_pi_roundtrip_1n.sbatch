#!/bin/bash
#SBATCH -J e97-dense-pi
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
LOCAL_CHECKPOINT="/tmp/e97-dense-pi-$SLURM_JOB_ID.pt"
LOCAL_ARGS="/tmp/e97-dense-pi-$SLURM_JOB_ID-args.json"
sbcast -f "$CHECKPOINT" "$LOCAL_CHECKPOINT"
sbcast -f "$ARGS_JSON" "$LOCAL_ARGS"
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$LOCAL_CHECKPOINT" | sha256sum -c -

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'$SNAPSHOT'"
  command -v pi >/dev/null
  export PYTHONPATH="'$SNAPSHOT'" LOCAL_RANK=0 OMP_NUM_THREADS=7
  export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
  export TRITON_CACHE_DIR=/tmp/e97-dense-pi-${SLURM_JOB_ID}-${SLURM_PROCID}
  export PI_OFFLINE=1
  rm -rf "$TRITON_CACHE_DIR"
  task=/tmp/e97-pi-task-${SLURM_JOB_ID}-${SLURM_PROCID}
  config=/tmp/e97-pi-config-${SLURM_JOB_ID}-${SLURM_PROCID}
  sessions=/tmp/e97-pi-sessions-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$task" "$config" "$sessions"
  mkdir -p "$task" "$config" "$sessions"
  port=$((18000 + (SLURM_JOB_ID % 1000) * 8 + SLURM_PROCID))
  sed "s/127.0.0.1:8797/127.0.0.1:${port}/" configs/pi/e97-dense-agent.models.json > "$config/models.json"
  server_log="'$RUN_ROOT'/shards/server-${SLURM_PROCID}.log"
  "$EMENDER_PYTHON" scripts/serve_e97_agent_openai.py \
    --checkpoint "'$LOCAL_CHECKPOINT'" --args-json "'$LOCAL_ARGS'" \
    --host 127.0.0.1 --port "$port" --max-output-tokens 96 --max-sessions 2 \
    > "'$RUN_ROOT'/shards/server-${SLURM_PROCID}.out" 2> "$server_log" &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for attempt in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep 2
  done
  test "$ready" = 1
  system_prompt='"'"'You are a precise tool-using agent. Respond with either "Action:" and one JSON "Arguments:" object, or "Final:". Never invent tool results.'"'"'
  cd "$task"
  PI_CODING_AGENT_DIR="$config" PI_CODING_AGENT_SESSION_DIR="$sessions" \
    pi --mode json --provider emender-local --model e97-dense-agent \
      --no-builtin-tools -e "'$SNAPSHOT'/configs/pi/e97-v1-tools.ts" \
      --system-prompt "$system_prompt" --approve \
      "Calculate 608 * 57." \
      > "'$RUN_ROOT'/shards/pi-${SLURM_PROCID}.jsonl"
  cd "'$SNAPSHOT'"
  "$EMENDER_PYTHON" scripts/validate_e97_pi_roundtrip.py \
    --pi-jsonl "'$RUN_ROOT'/shards/pi-${SLURM_PROCID}.jsonl" \
    --server-log "$server_log" \
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
    "schema": "emender-e97-pi-roundtrip-summary-v1",
    "status": "pass" if all(row["status"] == "pass" for row in shards) else "fail",
    "shards": len(shards),
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
