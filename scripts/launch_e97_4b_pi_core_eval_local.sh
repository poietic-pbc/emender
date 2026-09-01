#!/bin/bash
# Run the frozen real-Pi core-tool panel on eight local GPUs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
: "${CHECKPOINT:?set CHECKPOINT to an accepted Pi SFT checkpoint}"
: "${CHECKPOINT_SHA256:?set the checkpoint SHA-256}"
: "${CLI_IMAGE:?set CLI_IMAGE to the immutable Apptainer image}"
: "${CLI_IMAGE_SHA256:?set CLI_IMAGE_SHA256}"
SOURCE_COMMIT=${SOURCE_COMMIT:-HEAD}
SOURCE_COMMIT=$(git rev-parse "${SOURCE_COMMIT}^{commit}")
RUN_ID=${RUN_ID:-e97-4b-pi-core-local-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/evals/$RUN_ID}
ARGS_JSON=${ARGS_JSON:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/mnt/nvme1n1/erikg/sft/pi-native-core-v1}
AUTHORITY_SHA256=${AUTHORITY_SHA256:-48f6b7ecb0083f09402e2f0715b95d7ca71ba45a2711375b811472ccdeb804e1}
LIMIT=${LIMIT:-120}; WORLD_SIZE=8; EXPECTED_TASKS=${EXPECTED_TASKS:-$LIMIT}
[[ "$RUN_ROOT" == /* && ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT must be a new absolute path" >&2; exit 64; }
[[ -r "$CHECKPOINT" && -r "$ARGS_JSON" && -r "$CLI_IMAGE" ]] || exit 66
[[ $(sha256sum "$CHECKPOINT" | awk '{print $1}') == "$CHECKPOINT_SHA256" ]] || { echo "checkpoint SHA-256 mismatch" >&2; exit 65; }
[[ $(sha256sum "$CLI_IMAGE" | awk '{print $1}') == "$CLI_IMAGE_SHA256" ]] || { echo "CLI image SHA-256 mismatch" >&2; exit 65; }
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_SHA256" ]] || { echo "authority SHA-256 mismatch" >&2; exit 65; }
git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$RUN_ROOT"/{identity,results,shards,sources,terminal}
printf '%s\n' "$CHECKPOINT_SHA256  $CHECKPOINT" > "$RUN_ROOT/identity/checkpoint.sha256"
printf '%s\n' "$CLI_IMAGE_SHA256  $CLI_IMAGE" > "$RUN_ROOT/identity/cli-image.sha256"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-commit.txt"
snapshot="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$snapshot"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$snapshot"
if [[ ${DRY_RUN:-0} == 1 ]]; then echo "RUN_ROOT=$RUN_ROOT"; exit 0; fi

eval "$(scripts/gpu_lease.sh acquire 8 --no-wait)"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPUS[@]} == 8 ]] || exit 69
export PYTHONPATH="$snapshot${PYTHONPATH:+:$PYTHONPATH}" EMENDER_PYTHON=${EMENDER_PYTHON:-$(command -v python)}
export EMENDER_CLI_IMAGE="$CLI_IMAGE" EMENDER_CLI_IMAGE_SHA256="$CLI_IMAGE_SHA256"
export PI_OFFLINE=1 TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache OMP_NUM_THREADS=4
server_pids=(); eval_pids=()
cleanup() {
  trap - EXIT INT TERM
  local pid
  for pid in "${eval_pids[@]}" "${server_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${eval_pids[@]}" "${server_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  for pid in ${GPU_LEASE_HB_PIDS:-}; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
  gpu_lease_release 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for rank in $(seq 0 7); do
  config="$RUN_ROOT/shards/config-$rank"; mkdir -p "$config"
  port=$((24800 + rank))
  sed "s/127.0.0.1:8797/127.0.0.1:${port}/" "$snapshot/configs/pi/e97-dense-agent.models.json" > "$config/models.json"
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$rank]} LOCAL_RANK=0
    export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=/tmp/e97-4b-pi-core-${RUN_ID}-evalrank${rank}
    "$snapshot/scripts/numa_local_rank_exec.py" -- "$snapshot/scripts/serve_e97_agent_openai.py" \
      --checkpoint "$CHECKPOINT" --args-json "$ARGS_JSON" --host 127.0.0.1 --port "$port" \
      --max-output-tokens 256 --max-sessions 2 --ingest-mode tokenwise \
      --pi-core-canonical-system --trace-generated-errors
  ) > "$RUN_ROOT/shards/server-${rank}.out" 2> "$RUN_ROOT/shards/server-${rank}.log" &
  server_pids+=("$!")
done

for rank in $(seq 0 7); do
  ready=0; port=$((24800 + rank))
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "${server_pids[$rank]}" 2>/dev/null || break
    sleep 2
  done
  [[ $ready == 1 ]] || { echo "server rank $rank failed readiness" >&2; exit 70; }
done
echo "LOCAL_PI_CORE_EVAL_SERVERS_READY"

for rank in $(seq 0 7); do
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$rank]}
    "$EMENDER_PYTHON" "$snapshot/scripts/eval_e97_4b_pi_core.py" \
      --authority-root "$AUTHORITY_ROOT" --rank "$rank" --world-size "$WORLD_SIZE" \
      --limit "$LIMIT" --pi-config-dir "$RUN_ROOT/shards/config-$rank" \
      --extension "$snapshot/configs/pi/e97-core-tools.ts" \
      --output-root "$RUN_ROOT/shards" --timeout-seconds 300
  ) > "$RUN_ROOT/shards/eval-${rank}.out" 2> "$RUN_ROOT/shards/eval-${rank}.log" &
  eval_pids+=("$!")
done
rc=0
for pid in "${eval_pids[@]}"; do wait "$pid" || rc=$?; done
eval_pids=()
if (( rc == 0 )); then
  "$EMENDER_PYTHON" "$snapshot/scripts/aggregate_e97_4b_pi_core.py" \
    --input-root "$RUN_ROOT/shards" --output "$RUN_ROOT/results/summary.json" \
    --expected-tasks "$EXPECTED_TASKS" || rc=$?
fi
for rank in $(seq 0 7); do
  grep -q 'completion cache=miss' "$RUN_ROOT/shards/server-${rank}.log" || rc=71
done
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/return-code.txt"
if (( rc == 0 )); then echo "LOCAL_PI_CORE_EVAL_COMPLETE run_root=$RUN_ROOT"; fi
exit "$rc"
