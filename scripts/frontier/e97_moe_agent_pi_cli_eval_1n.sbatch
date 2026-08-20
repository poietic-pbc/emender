#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-agent-pi
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err
set -euo pipefail

: "${SOURCE_COMMIT:?}"
: "${RUN_ROOT:?}"
GENERATION=${GENERATION:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
GENERATION_MANIFEST_SHA256=${GENERATION_MANIFEST_SHA256:-95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-cli-v1}
CLI_IMAGE=${CLI_IMAGE:-/lustre/orion/bif148/proj-shared/emender/agent_sandbox/images/e97-cli-sandbox-296eb1e5.sif}
CLI_IMAGE_SHA256=${CLI_IMAGE_SHA256:-bb85bf69530c9b30515acc4c1946a0402309d1c629222f03fdcc2d2df5f133ee}
LIMIT=${LIMIT:-4}
EXPECTED_TASKS=${EXPECTED_TASKS:-$LIMIT}
TASK_MODE=${TASK_MODE:-direct}
SYSTEM_MODE=${SYSTEM_MODE:-direct}
INGEST_MODE=${INGEST_MODE:-segment}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-96}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}

cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
[[ "$RUN_ROOT" == /* && "$LIMIT" =~ ^[1-9][0-9]*$ && "$EXPECTED_TASKS" == "$LIMIT" ]] || exit 64
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 65
[[ $(sha256sum "$GENERATION/manifest.json" | awk '{print $1}') == "$GENERATION_MANIFEST_SHA256" ]] || exit 66
[[ $(sha256sum "$CLI_IMAGE" | awk '{print $1}') == "$CLI_IMAGE_SHA256" ]] || exit 66
[[ -f "$AUTHORITY_ROOT/records.jsonl" && -f "$SEED_CHECKPOINT" && -f "$SEED_ARGS_JSON" ]] || exit 66
mkdir -p "$RUN_ROOT"/{identity,results,shards,sources,terminal} logs/frontier/e97_moe
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
grep -q 'Partition=batch|QOS=debug' "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
export MASTER_ADDR=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((37000 + SLURM_JOB_ID % 1000)) WORLD_SIZE=8 OMP_NUM_THREADS=6
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
port=$((24000 + SLURM_JOB_ID % 1000))
server_out="$RUN_ROOT/shards/server.out"; server_err="$RUN_ROOT/shards/server.err"
set +e
srun --overlap --exact -N1 -n8 -c6 --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'$SNAPSHOT'"
  export PYTHONPATH="'$SNAPSHOT'" RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/e97-moe-agent-${SLURM_JOB_ID}-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/serve_e97_moe_agent_openai.py \
    --seed-checkpoint "'$SEED_CHECKPOINT'" --seed-args-json "'$SEED_ARGS_JSON'" \
    --generation "'$GENERATION'" --host 127.0.0.1 --port "'$port'" \
    --max-output-tokens "'$MAX_OUTPUT_TOKENS'" --max-sessions 2 --ingest-mode "'$INGEST_MODE'" \
    --cli-direct-canonical-system --trace-generated-errors
' >"$server_out" 2>"$server_err" &
server_step_pid=$!
set -e
cleanup() { kill "$server_step_pid" 2>/dev/null || true; wait "$server_step_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for attempt in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_step_pid" 2>/dev/null; then break; fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'MoE agent server failed readiness' >&2; exit 67; }
config=/tmp/e97-moe-agent-config-$SLURM_JOB_ID; rm -rf "$config"; mkdir -p "$config"
sed "s/127.0.0.1:8797/127.0.0.1:${port}/" "$SNAPSHOT/configs/pi/e97-dense-agent.models.json" > "$config/models.json"
export EMENDER_CLI_IMAGE="$CLI_IMAGE" EMENDER_CLI_IMAGE_SHA256="$CLI_IMAGE_SHA256" PI_OFFLINE=1
set +e
"$EMENDER_PYTHON" "$SNAPSHOT/scripts/eval_e97_dense_agent_pi_cli.py" \
  --authority-root "$AUTHORITY_ROOT" --rank 0 --world-size 1 --limit "$LIMIT" \
  --task-mode "$TASK_MODE" --system-mode "$SYSTEM_MODE" --pi-config-dir "$config" \
  --extension "$SNAPSHOT/configs/pi/e97-cli-tools.ts" --output-root "$RUN_ROOT/shards" --timeout-seconds 300
rc=$?
set -e
cleanup; trap - EXIT
if (( rc == 0 )); then
  "$EMENDER_PYTHON" "$SNAPSHOT/scripts/aggregate_e97_dense_agent_pi_v2.py" \
    --input-root "$RUN_ROOT/shards" --output "$RUN_ROOT/results/summary.json" --expected-tasks "$EXPECTED_TASKS" || rc=$?
fi
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P > "$RUN_ROOT/terminal/sacct-$SLURM_JOB_ID.txt" || true
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"
exit "$rc"
