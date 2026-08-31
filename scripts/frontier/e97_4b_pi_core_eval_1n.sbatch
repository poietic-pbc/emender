#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-4b-pi-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue

set -euo pipefail
: "${SOURCE_COMMIT:?}"
: "${RUN_ROOT:?}"
: "${CHECKPOINT:?}"
: "${CHECKPOINT_SHA256:?}"
LIMIT=${LIMIT:-120}
EXPECTED_TASKS=${EXPECTED_TASKS:-$LIMIT}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ARGS_JSON=${ARGS_JSON:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-hybrid-ddp-256n-recovery-r3/train/args.json}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/pi-native-core-v1}
AUTHORITY_SHA256=${AUTHORITY_SHA256:-c0d8a8934908da87a5624fb976f7872cc6b2ca3eb667892bb3a98126e08f9bae}
CLI_IMAGE=${CLI_IMAGE:-/lustre/orion/bif148/proj-shared/emender/agent_sandbox/images/e97-cli-sandbox-296eb1e5.sif}
CLI_IMAGE_SHA256=${CLI_IMAGE_SHA256:-bb85bf69530c9b30515acc4c1946a0402309d1c629222f03fdcc2d2df5f133ee}

cd "$REPO"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?}"
record=$(scontrol show job -dd "$SLURM_JOB_ID")
[[ "$RUN_ROOT" == /* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
mkdir -p "$RUN_ROOT"/{identity,results,shards,sources,terminal}
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}-running.txt"
squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%N|%P|%q|%R' | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}-running.txt"
grep -F '|batch|debug|' "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}-running.txt" >/dev/null
git cat-file -e "${SOURCE_COMMIT}^{commit}"
[[ $(git rev-parse HEAD) == "$SOURCE_COMMIT" ]] || exit 65
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_SHA256" ]] || exit 65
[[ $(sha256sum "$CLI_IMAGE" | awk '{print $1}') == "$CLI_IMAGE_SHA256" ]] || exit 65

snapshot="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$snapshot"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$snapshot"
local_checkpoint=/tmp/e97-4b-pi-eval-${SLURM_JOB_ID}.pt
local_args=/tmp/e97-4b-pi-eval-${SLURM_JOB_ID}-args.json
sbcast -f "$CHECKPOINT" "$local_checkpoint"
sbcast -f "$ARGS_JSON" "$local_args"
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$local_checkpoint" | sha256sum -c -

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'$snapshot'"
  export PYTHONPATH="'$snapshot'" LOCAL_RANK=0 OMP_NUM_THREADS=7 PI_OFFLINE=1
  export EMENDER_CLI_IMAGE="'$CLI_IMAGE'" EMENDER_CLI_IMAGE_SHA256="'$CLI_IMAGE_SHA256'"
  export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
  export TRITON_CACHE_DIR=/tmp/e97-4b-pi-eval-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$TRITON_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR"
  config=/tmp/e97-4b-pi-eval-config-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$config"; mkdir -p "$config"
  port=$((24000 + (SLURM_JOB_ID % 1000) * 8 + SLURM_PROCID))
  sed "s/127.0.0.1:8797/127.0.0.1:${port}/" configs/pi/e97-dense-agent.models.json > "$config/models.json"
  server_log="'$RUN_ROOT'/shards/server-${SLURM_PROCID}.log"
  "$EMENDER_PYTHON" scripts/serve_e97_agent_openai.py \
    --checkpoint "'$local_checkpoint'" --args-json "'$local_args'" \
    --host 127.0.0.1 --port "$port" --max-output-tokens 256 --max-sessions 2 \
    --ingest-mode tokenwise --pi-core-canonical-system --trace-generated-errors \
    > "'$RUN_ROOT'/shards/server-${SLURM_PROCID}.out" 2> "$server_log" &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for attempt in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 2
  done
  test "$ready" = 1
  "$EMENDER_PYTHON" scripts/eval_e97_4b_pi_core.py \
    --authority-root "'$AUTHORITY_ROOT'" --rank "$SLURM_PROCID" --world-size 8 \
    --limit "'$LIMIT'" --pi-config-dir "$config" \
    --extension "'$snapshot'/configs/pi/e97-core-tools.ts" \
    --output-root "'$RUN_ROOT'/shards" --timeout-seconds 300
  grep -q "completion cache=miss" "$server_log"
'
rc=$?
set -e
if (( rc == 0 )); then
  "$EMENDER_PYTHON" "$snapshot/scripts/aggregate_e97_4b_pi_core.py" \
    --input-root "$RUN_ROOT/shards" --output "$RUN_ROOT/results/summary.json" \
    --expected-tasks "$EXPECTED_TASKS" || rc=$?
fi
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P > "$RUN_ROOT/terminal/sacct-${SLURM_JOB_ID}.txt" || true
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
exit "$rc"
