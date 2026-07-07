#!/bin/bash
# Shared runner for train.py-backed async quorum DiLoCo debug smokes.
#
# This intentionally uses one Slurm task per GPU and calls the train.py-backed
# async entrypoint once per task. The current debug transport is the bounded
# file-quorum path in scripts/frontier/e97_async_diloco_train.py: each GPU rank
# runs real train.py local token training, writes a per-rank update manifest, and
# rank 0 publishes the authoritative latest.json/latest checkpoint metadata when
# quorum is reached.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-add-train-py}
SMOKE_NAME=${SMOKE_NAME:?SMOKE_NAME must be set by the sbatch wrapper}
SMOKE_NODE_COUNT=${SMOKE_NODE_COUNT:?SMOKE_NODE_COUNT must be set by the sbatch wrapper}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)

OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1065000_trainpy_async_quorum_${SMOKE_NAME}}
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SCALEOUT_VARIANT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
RUN_DIR="${RUN_ROOT}/async_run"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
SUMMARY_DIR="${RUN_ROOT}/summaries"
METRICS_JSON="${ARTIFACT_DIR}/metrics.json"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
RANK_START_LOG="${ARTIFACT_DIR}/rank-start.tsv"
STDOUT_PATH="logs/frontier/trainpy_async_quorum/${SLURM_JOB_NAME:-trainpy-async-quorum}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/trainpy_async_quorum/${SLURM_JOB_NAME:-trainpy-async-quorum}-${SLURM_JOB_ID:-manual}.err"

ASYNC_ENTRYPOINT=${ASYNC_ENTRYPOINT:-scripts/frontier/e97_async_diloco_train.py}
DEFAULT_E97_SEED_LATEST=${DEFAULT_E97_SEED_LATEST:-/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt}
E97_CHECKPOINT=${E97_CHECKPOINT:-$DEFAULT_E97_SEED_LATEST}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}

RANKS_PER_NODE=${RANKS_PER_NODE:-8}
CPUS_PER_RANK=${CPUS_PER_RANK:-7}
GPU_BIND=${GPU_BIND:-closest}
ASYNC_TRAINPY_RANKS=${ASYNC_TRAINPY_RANKS:-$((SMOKE_NODE_COUNT * RANKS_PER_NODE))}
ASYNC_EXPECTED_RANKS=${ASYNC_EXPECTED_RANKS:-$ASYNC_TRAINPY_RANKS}
ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$ASYNC_TRAINPY_RANKS}
ASYNC_EXPECTED_MISSING_UPDATES=${ASYNC_EXPECTED_MISSING_UPDATES:-$((ASYNC_EXPECTED_RANKS - ASYNC_TRAINPY_RANKS))}
ASYNC_TIMEOUT_S=${ASYNC_TIMEOUT_S:-120}
ASYNC_LOCAL_STEPS=${ASYNC_LOCAL_STEPS:-${DILOCO_K:-1}}
ASYNC_GENERATIONS=${ASYNC_GENERATIONS:-1}
ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=${ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM:-0}
ALLOW_SYNTHETIC_TOKEN_FALLBACK=${ALLOW_SYNTHETIC_TOKEN_FALLBACK:-0}
TRAINING_DATA_MODE=real-token

BATCH_SIZE=${BATCH_SIZE:-1}
CHUNK_SIZE=${CHUNK_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MODEL_TOKENIZER=${MODEL_TOKENIZER:-p50k_base}
MODEL_LEVEL=${MODEL_LEVEL:-E97}
MODEL_PARAMS=${MODEL_PARAMS:-100m}
MODEL_DIM=${MODEL_DIM:-1792}
MODEL_DEPTH=${MODEL_DEPTH:-11}
MODEL_N_HEADS=${MODEL_N_HEADS:-216}
MODEL_N_STATE=${MODEL_N_STATE:-32}
MODEL_N_GROUPS=${MODEL_N_GROUPS:-32}
MODEL_N_SLOTS=${MODEL_N_SLOTS:-64}
MODEL_STATE_EXPANSION=${MODEL_STATE_EXPANSION:-2}
MODEL_EXPANSION=${MODEL_EXPANSION:-1.0}
MODEL_GATE_ACTIVATION=${MODEL_GATE_ACTIVATION:-silu}
MODEL_LINEAR_STATE=${MODEL_LINEAR_STATE:-0}
MODEL_MLP_RATIO=${MODEL_MLP_RATIO:-2.2623}
MODEL_MLP_MULTIPLE=${MODEL_MLP_MULTIPLE:-64}
ASYNC_E97_BF16=${ASYNC_E97_BF16:-1}
ASYNC_E97_USE_CHUNKED=${ASYNC_E97_USE_CHUNKED:-0}
ASYNC_E97_CHUNK_SIZE=${ASYNC_E97_CHUNK_SIZE:-32}
ASYNC_E97_CHECKPOINT_INTERVAL=${ASYNC_E97_CHECKPOINT_INTERVAL:-16}
ASYNC_E97_GRADIENT_CHECKPOINTING=${ASYNC_E97_GRADIENT_CHECKPOINTING:-0}
ASYNC_E97_PROJECTION_CHUNK_SIZE=${ASYNC_E97_PROJECTION_CHUNK_SIZE:-0}
ASYNC_E97_LOSS_CHUNK_SIZE=${ASYNC_E97_LOSS_CHUNK_SIZE:-0}
RECOVERY_EVERY_GENERATIONS=${RECOVERY_EVERY_GENERATIONS:-1}
RECOVERY_EVERY_SECONDS=${RECOVERY_EVERY_SECONDS:--1}
EXPORT_EVERY_GENERATIONS=${EXPORT_EVERY_GENERATIONS:-1}
EXPORT_EVERY_SECONDS=${EXPORT_EVERY_SECONDS:--1}
FINALIZATION_BUFFER_SECONDS=${FINALIZATION_BUFFER_SECONDS:-1200}
ESTIMATED_FINALIZATION_DURATION_SECONDS=${ESTIMATED_FINALIZATION_DURATION_SECONDS:-0}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-${SLURM_TIMELIMIT:-00:20:00}}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-}
HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-WG add-train-py: bounded train.py-native async quorum debug smoke; no production approval.}

if [[ -z "$REQUESTED_NODE_HOURS" ]]; then
  REQUESTED_NODE_HOURS=$(awk -v nodes="$SMOKE_NODE_COUNT" -v wall="$REQUESTED_WALLTIME" '
    BEGIN {
      split(wall, parts, ":")
      if (length(parts) == 3) seconds = parts[1] * 3600 + parts[2] * 60 + parts[3]
      else if (length(parts) == 2) seconds = parts[1] * 60 + parts[2]
      else seconds = parts[1]
      printf "%.6f\n", nodes * seconds / 3600
    }
  ')
fi

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$SUMMARY_DIR" logs/frontier/trainpy_async_quorum
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env
PYTHON_BIN=$(command -v python)
export REPO TIKTOKEN_CACHE_DIR PYTHON_BIN RANK_START_LOG ASYNC_ENTRYPOINT

[[ -f "$ASYNC_ENTRYPOINT" ]] || { echo "ASYNC_ENTRYPOINT is missing: $ASYNC_ENTRYPOINT" >&2; exit 65; }
if [[ ! -r "$DATA" ]]; then
  if [[ "$ALLOW_SYNTHETIC_TOKEN_FALLBACK" == "1" ]]; then
    ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=1
    TRAINING_DATA_MODE=synthetic-token-fallback
  else
    echo "DATA is not readable: $DATA" >&2
    echo "Set ALLOW_SYNTHETIC_TOKEN_FALLBACK=1 to run an explicitly labeled synthetic-token fallback." >&2
    exit 4
  fi
fi
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" != "1" ]]; then
  [[ -r "$E97_CHECKPOINT" ]] || { echo "E97_CHECKPOINT is not readable: $E97_CHECKPOINT" >&2; exit 4; }
fi
if [[ "$ASYNC_EXPECTED_RANKS" -lt "$ASYNC_TRAINPY_RANKS" ]]; then
  echo "ASYNC_EXPECTED_RANKS must be >= ASYNC_TRAINPY_RANKS" >&2
  exit 64
fi
if [[ "$ASYNC_GLOBAL_QUORUM" -gt "$ASYNC_TRAINPY_RANKS" ]]; then
  echo "ASYNC_GLOBAL_QUORUM=$ASYNC_GLOBAL_QUORUM cannot exceed launched ranks=$ASYNC_TRAINPY_RANKS" >&2
  exit 64
fi

RUN_ID="${TASK_ID}-${SMOKE_NAME}-${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
CMD=(
  "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"
  --run-id "$RUN_ID"
  --run-dir "$RUN_DIR"
  --metrics-json "$METRICS_JSON"
  --data "$DATA"
  --tokenizer "$MODEL_TOKENIZER"
  --worker-count "$ASYNC_TRAINPY_RANKS"
  --node-count "$ASYNC_EXPECTED_RANKS"
  --global-quorum "$ASYNC_GLOBAL_QUORUM"
  --generations "$ASYNC_GENERATIONS"
  --local-steps "$ASYNC_LOCAL_STEPS"
  --steps "$ASYNC_LOCAL_STEPS"
  --timeout-s "$ASYNC_TIMEOUT_S"
  --level "$MODEL_LEVEL"
  --params "$MODEL_PARAMS"
  --batch-size "$BATCH_SIZE"
  --chunk-size "$CHUNK_SIZE"
  --lr "$LEARNING_RATE"
  --dim "$MODEL_DIM"
  --depth "$MODEL_DEPTH"
  --n-heads "$MODEL_N_HEADS"
  --n-state "$MODEL_N_STATE"
  --n-groups "$MODEL_N_GROUPS"
  --n-slots "$MODEL_N_SLOTS"
  --expansion "$MODEL_EXPANSION"
  --state-expansion "$MODEL_STATE_EXPANSION"
  --gate-activation "$MODEL_GATE_ACTIVATION"
  --linear-state "$MODEL_LINEAR_STATE"
  --mlp-ratio "$MODEL_MLP_RATIO"
  --mlp-multiple "$MODEL_MLP_MULTIPLE"
  --e97-chunk-size "$ASYNC_E97_CHUNK_SIZE"
  --checkpoint-interval "$ASYNC_E97_CHECKPOINT_INTERVAL"
  --projection-chunk-size "$ASYNC_E97_PROJECTION_CHUNK_SIZE"
  --loss-chunk-size "$ASYNC_E97_LOSS_CHUNK_SIZE"
  --recovery-every-generations "$RECOVERY_EVERY_GENERATIONS"
  --recovery-every-seconds "$RECOVERY_EVERY_SECONDS"
  --export-every-generations "$EXPORT_EVERY_GENERATIONS"
  --export-every-seconds "$EXPORT_EVERY_SECONDS"
  --finalization-reserve-seconds "$FINALIZATION_BUFFER_SECONDS"
  --walltime-remaining-s "$FINALIZATION_BUFFER_SECONDS"
  --estimated-finalization-duration-s "$ESTIMATED_FINALIZATION_DURATION_SECONDS"
  --actual-multinode-file-quorum
)
if [[ -r "$E97_CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$E97_CHECKPOINT")
fi
if [[ "$ASYNC_E97_BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$ASYNC_E97_USE_CHUNKED" == "1" ]]; then
  CMD+=(--use-chunked-e97)
fi
if [[ "$ASYNC_E97_GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--gradient-checkpointing)
fi
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" ]]; then
  CMD+=(--synthetic-token-stream --allow-actual-multinode-synthetic-token-stream)
fi

LAUNCH_CMD=(
  srun
  -N "$SMOKE_NODE_COUNT"
  -n "$ASYNC_TRAINPY_RANKS"
  --ntasks-per-node="$RANKS_PER_NODE"
  -c "$CPUS_PER_RANK"
  --gpus-per-task=1
  --gpu-bind="$GPU_BIND"
  bash -lc
  'source "${REPO}/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_PROCID:-}" "${SLURM_LOCALID:-}" "${SLURMD_NODENAME:-$(hostname)}" "${SLURM_NTASKS:-}" >> "$RANK_START_LOG"; exec "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT" "$@" --node-rank "${SLURM_PROCID:?missing SLURM_PROCID}" --device "cuda:${ASYNC_VISIBLE_DEVICE_ORDINAL:-0}"'
  bash
  "${CMD[@]:3}"
)

printf '%q ' "${LAUNCH_CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

{
  echo "task_id=$TASK_ID"
  echo "smoke_name=$SMOKE_NAME"
  echo "run_id=$RUN_ID"
  echo "scaleout_variant=$SCALEOUT_VARIANT"
  echo "run_root=$RUN_ROOT"
  echo "run_dir=$RUN_DIR"
  echo "metrics_json=$METRICS_JSON"
  echo "summary_file=$SUMMARY_FILE"
  echo "manifest_file=$MANIFEST_FILE"
  echo "command_file=$COMMAND_FILE"
  echo "rank_start_log=$RANK_START_LOG"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "training_data_mode=$TRAINING_DATA_MODE"
  echo "synthetic_token_stream=$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM"
  echo "data=$DATA"
  echo "e97_checkpoint=$E97_CHECKPOINT"
  echo "async_entrypoint=$ASYNC_ENTRYPOINT"
  echo "one_trainpy_rank_per_gpu=1"
  echo "slurm_nodes=$SMOKE_NODE_COUNT"
  echo "slurm_ntasks_per_node=$RANKS_PER_NODE"
  echo "slurm_gpus_per_task=1"
  echo "slurm_gpu_bind=$GPU_BIND"
  echo "trainpy_launched_ranks=$ASYNC_TRAINPY_RANKS"
  echo "async_expected_ranks=$ASYNC_EXPECTED_RANKS"
  echo "async_global_quorum=$ASYNC_GLOBAL_QUORUM"
  echo "async_expected_missing_updates=$ASYNC_EXPECTED_MISSING_UPDATES"
  echo "async_timeout_s=$ASYNC_TIMEOUT_S"
  echo "async_local_steps=$ASYNC_LOCAL_STEPS"
  echo "requested_walltime=$REQUESTED_WALLTIME"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "ddp_wrapper_expected=0"
  echo "per_step_all_reduce_expected=0"
  echo "bounded_debug_transport=actual_multinode_file_quorum"
  echo "git_commit=$(git rev-parse HEAD)"
  echo
  echo "=== modules ==="
  module -t list 2>&1 || true
  echo
  echo "=== frontier runtime ==="
  frontier_capture_runtime_env
  echo
  echo "=== command ==="
  cat "$COMMAND_FILE"
} | tee "$ENV_FILE"

set +e
"${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_DIR}/trainpy_async_quorum.log"
STATUS=${PIPESTATUS[0]}
set -e

python - "$STATUS" "$METRICS_JSON" "$ENV_FILE" "$COMMAND_FILE" "$RANK_START_LOG" "$SUMMARY_FILE" "$MANIFEST_FILE" "$SMOKE_NAME" "$STDOUT_PATH" "$STDERR_PATH" "$ASYNC_TRAINPY_RANKS" "$ASYNC_EXPECTED_MISSING_UPDATES" <<'PY'
import json
import sys
from pathlib import Path

status = int(sys.argv[1])
metrics_path = Path(sys.argv[2])
env_file = Path(sys.argv[3])
command_file = Path(sys.argv[4])
rank_start_log = Path(sys.argv[5])
summary_file = Path(sys.argv[6])
manifest_file = Path(sys.argv[7])
smoke_name = sys.argv[8]
stdout_path = sys.argv[9]
stderr_path = sys.argv[10]
expected_launched_ranks = int(sys.argv[11])
expected_missing = int(sys.argv[12])
train_log = summary_file.parents[1] / "logs" / "trainpy_async_quorum.log"

log_text = train_log.read_text(encoding="utf-8", errors="replace") if train_log.exists() else ""
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
global_metrics = ((metrics.get("global_generations") or [{}])[0].get("metrics") or {})
rank_lines = rank_start_log.read_text(encoding="utf-8", errors="replace").splitlines() if rank_start_log.exists() else []
ddp_forbidden = [
    line for line in log_text.splitlines()
    if "[DDP] wrapped model in DistributedDataParallel" in line
    or "per-step gradient all-reduce" in line
]
latest_path = Path(metrics.get("latest_path", ""))
accepted = int(global_metrics.get("accepted_updates", 0) or 0)
timed_out = int(global_metrics.get("timed_out_updates", 0) or 0)
latest_advanced = bool(global_metrics.get("latest_advanced") or metrics.get("latest_generation", -1) >= 0)
tokens = int(global_metrics.get("tokens_per_generation", 0) or 0)
checkpoint_paths = list(global_metrics.get("checkpoint_paths") or [])

validation_errors = []
if status != 0:
    validation_errors.append(f"payload_exit_status={status}")
if len(rank_lines) != expected_launched_ranks:
    validation_errors.append(f"rank_start_count={len(rank_lines)} expected={expected_launched_ranks}")
if ddp_forbidden:
    validation_errors.append("ddp_or_per_step_all_reduce_log_detected")
if not metrics:
    validation_errors.append("metrics_json_missing_or_empty")
if accepted <= 0:
    validation_errors.append("no_accepted_updates")
if tokens <= 0:
    validation_errors.append("no_training_tokens_recorded")
if expected_missing > 0 and timed_out < expected_missing:
    validation_errors.append(f"timed_out_updates={timed_out} expected_at_least={expected_missing}")
if not latest_advanced:
    validation_errors.append("latest_not_advanced")
if latest_path and not latest_path.exists():
    validation_errors.append(f"latest_path_missing={latest_path}")
if not checkpoint_paths:
    validation_errors.append("checkpoint_paths_missing")

manifest = {
    "schema_version": 1,
    "smoke_name": smoke_name,
    "exit_status": status,
    "validation_status": "pass" if not validation_errors else "fail",
    "validation_errors": validation_errors,
    "metrics_json": str(metrics_path),
    "env_file": str(env_file),
    "command_file": str(command_file),
    "rank_start_log": str(rank_start_log),
    "train_log": str(train_log),
    "stdout": stdout_path,
    "stderr": stderr_path,
    "rank_start_count": len(rank_lines),
    "expected_launched_ranks": expected_launched_ranks,
    "accepted_updates": accepted,
    "timed_out_updates": timed_out,
    "tokens_per_generation": tokens,
    "latest_path": str(latest_path),
    "checkpoint_paths": checkpoint_paths,
    "ddp_forbidden_line_count": len(ddp_forbidden),
}
manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

summary_file.write_text(
    f"# train.py async quorum {smoke_name} smoke\n\n"
    f"Validation: `{'pass' if not validation_errors else 'fail'}`\n\n"
    f"Exit status: `{status}`\n\n"
    f"Rank starts: `{len(rank_lines)}` / `{expected_launched_ranks}`\n\n"
    f"Accepted updates: `{accepted}`\n\n"
    f"Timed-out updates: `{timed_out}`\n\n"
    f"Tokens: `{tokens}`\n\n"
    f"Latest path: `{latest_path}`\n\n"
    f"Metrics JSON: `{metrics_path}`\n\n"
    f"Train log: `{train_log}`\n\n"
    f"Stdout: `{stdout_path}`\n\n"
    f"Stderr: `{stderr_path}`\n\n"
    "## Validation Errors\n\n"
    + ("\n".join(f"- `{item}`" for item in validation_errors) if validation_errors else "None\n")
    + "\n\n## Metrics Excerpt\n\n```json\n"
    + json.dumps({
        "mode": metrics.get("mode"),
        "synthetic_token_stream": metrics.get("synthetic_token_stream"),
        "node_count": metrics.get("node_count"),
        "global_quorum": metrics.get("global_quorum"),
        "latest_generation": metrics.get("latest_generation"),
        "global_metrics": global_metrics,
    }, indent=2, sort_keys=True)
    + "\n```\n",
    encoding="utf-8",
)
if validation_errors:
    raise SystemExit(90)
PY

exit "$STATUS"
