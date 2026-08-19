#!/bin/bash
#SBATCH -J e97-agent-pi-v2
#SBATCH -A bif148
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_dense_agent/%x-%j.out
#SBATCH -e logs/frontier/e97_dense_agent/%x-%j.err
set -euo pipefail

: "${SOURCE_COMMIT:?}"
: "${RUN_ROOT:?}"
: "${CHECKPOINT:?}"
: "${CHECKPOINT_SHA256:?}"
LIMIT=${LIMIT:-0}
EXPECTED_TASKS=${EXPECTED_TASKS:-300}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ARGS_JSON=${ARGS_JSON:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/args.json}

cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$RUN_ROOT"/{identity,results,shards,sources,terminal} logs/frontier/e97_dense_agent
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
grep -q 'Partition=batch|QOS=debug' "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"

SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
mkdir -p "$SNAPSHOT"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
LOCAL_CHECKPOINT="/tmp/e97-agent-pi-v2-$SLURM_JOB_ID.pt"
LOCAL_ARGS="/tmp/e97-agent-pi-v2-$SLURM_JOB_ID-args.json"
sbcast -f "$CHECKPOINT" "$LOCAL_CHECKPOINT"
sbcast -f "$ARGS_JSON" "$LOCAL_ARGS"
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$LOCAL_CHECKPOINT" | sha256sum -c -

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'$SNAPSHOT'"
  command -v pi >/dev/null
  export PYTHONPATH="'$SNAPSHOT'" LOCAL_RANK=0 OMP_NUM_THREADS=7 PI_OFFLINE=1
  export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
  export TRITON_CACHE_DIR=/tmp/e97-agent-pi-v2-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$TRITON_CACHE_DIR"
  config=/tmp/e97-agent-pi-v2-config-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$config"; mkdir -p "$config"
  port=$((22000 + (SLURM_JOB_ID % 1000) * 8 + SLURM_PROCID))
  sed "s/127.0.0.1:8797/127.0.0.1:${port}/" configs/pi/e97-dense-agent.models.json > "$config/models.json"
  server_log="'$RUN_ROOT'/shards/server-${SLURM_PROCID}.log"
  "$EMENDER_PYTHON" scripts/serve_e97_agent_openai.py \
    --checkpoint "'$LOCAL_CHECKPOINT'" --args-json "'$LOCAL_ARGS'" \
    --host 127.0.0.1 --port "$port" --max-output-tokens 96 --max-sessions 2 \
    --ingest-mode tokenwise --v2-canonical-system --trace-generated-errors \
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
  "$EMENDER_PYTHON" scripts/eval_e97_dense_agent_pi_v2.py \
    --rank "$SLURM_PROCID" --world-size 8 --limit "'$LIMIT'" \
    --pi-config-dir "$config" --extension "'$SNAPSHOT'/configs/pi/e97-v2-tools.ts" \
    --output-root "'$RUN_ROOT'/shards" --timeout-seconds 120
  grep -q "completion cache=miss" "$server_log"
' 
rc=$?
set -e

if (( rc == 0 )); then
  "$EMENDER_PYTHON" "$SNAPSHOT/scripts/aggregate_e97_dense_agent_pi_v2.py" \
    --input-root "$RUN_ROOT/shards" --output "$RUN_ROOT/results/summary.json" \
    --expected-tasks "$EXPECTED_TASKS" || rc=$?
fi
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P > "$RUN_ROOT/terminal/sacct-$SLURM_JOB_ID.txt" || true
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"
exit "$rc"
