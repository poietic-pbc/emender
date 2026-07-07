#!/bin/bash
# Prepared 256-node 12-hour async quorum DiLoCo E97 launch wrapper.
#
# This production wrapper records the exact command and environment artifacts
# before any launch. It is approval-gated for production, and its debug mode
# still executes the real train.py-backed async trainer rather than the
# synthetic protocol harness.

#SBATCH -A bif148
#SBATCH -J async-diloco-e97-256n12h
#SBATCH -p batch
#SBATCH -N 256
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/async_diloco_e97/%x-%j.out
#SBATCH -e logs/frontier/async_diloco_e97/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-submit-refreshed-e97-async-256n12h}
TRAINING_TARGET=${TRAINING_TARGET:-E97_1.3B_step1065000_async_diloco_256n12h_20260706}
DEFAULT_E97_SEED_LATEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
SEED_LATEST_PATH=${SEED_LATEST_PATH:-${E97_CHECKPOINT:-$DEFAULT_E97_SEED_LATEST}}
E97_CHECKPOINT=${E97_CHECKPOINT:-$SEED_LATEST_PATH}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco}
DEFAULT_ENV_PREFIX="${REPO}/.envs/olcf-rocm711-torch210-py312"
if [[ ! -d "$DEFAULT_ENV_PREFIX" && -d /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 ]]; then
  DEFAULT_ENV_PREFIX=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
fi
ENV_PREFIX=${ENV_PREFIX:-$DEFAULT_ENV_PREFIX}
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-$ENV_PREFIX}
SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_step1065000_async_quorum_b4_k40_256n12h}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${SCALEOUT_VARIANT}/${RUN_DATE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
LOG_DIR="${RUN_ROOT}/logs"
RUN_DIR="${RUN_ROOT}/async_run"
METRICS_JSON="${ARTIFACT_DIR}/async_diloco_e97_256n_metrics.json"
COMMAND_FILE="${ARTIFACT_DIR}/command.txt"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
ASYNC_ENTRYPOINT=${ASYNC_ENTRYPOINT:-scripts/frontier/e97_async_diloco_train.py}
ASYNC_DILOCO_NON_PRODUCTION_DEBUG=${ASYNC_DILOCO_NON_PRODUCTION_DEBUG:-0}
ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=${ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM:-0}
ASYNC_ACTUAL_MULTINODE_TCP_QUORUM=${ASYNC_ACTUAL_MULTINODE_TCP_QUORUM:-0}
ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM=${ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM:-0}
ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM=${ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM:-1}
ASYNC_REQUIRE_CORRECTED_B4_LADDER=${ASYNC_REQUIRE_CORRECTED_B4_LADDER:-1}
SLURM_INTENDED_NTASKS_PER_NODE=${SLURM_INTENDED_NTASKS_PER_NODE:-8}
SLURM_INTENDED_GPUS_PER_TASK=${SLURM_INTENDED_GPUS_PER_TASK:-1}
SLURM_INTENDED_GPU_BIND=${SLURM_INTENDED_GPU_BIND:-closest}

# Async quorum recipe. Local quorum is six of eight workers per node; global
# quorum defaults to ceil(2/3 * nodes), including scaled debug node counts.
ASYNC_NODE_COUNT=${ASYNC_NODE_COUNT:-256}
ASYNC_WORKER_COUNT_PER_NODE=${ASYNC_WORKER_COUNT_PER_NODE:-8}
ASYNC_WORKER_COUNT=${ASYNC_WORKER_COUNT:-$((ASYNC_NODE_COUNT * ASYNC_WORKER_COUNT_PER_NODE))}
ASYNC_LOCAL_QUORUM=${ASYNC_LOCAL_QUORUM:-6}
ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$(((2 * ASYNC_NODE_COUNT + 2) / 3))}
ASYNC_GLOBAL_TIMEOUT_S=${ASYNC_GLOBAL_TIMEOUT_S:-240}
ASYNC_STALENESS_POLICY=${ASYNC_STALENESS_POLICY:-reject_stale}
ASYNC_WEIGHTING=${ASYNC_WEIGHTING:-tokens}
ASYNC_TRANSPORT_MODE=${ASYNC_TRANSPORT_MODE:-compiled-cray-mpich-helper-p2p}
ASYNC_MPI_DENSE_BUCKET_BYTES=${ASYNC_MPI_DENSE_BUCKET_BYTES:-67108864}
ASYNC_COMPILED_MPICH_HELPER_BIN=${ASYNC_COMPILED_MPICH_HELPER_BIN:-${ARTIFACT_DIR}/compiled_mpich_dense_helper}
ASYNC_COMPILED_MPICH_IPC_DIR=${ASYNC_COMPILED_MPICH_IPC_DIR:-${RUN_DIR}/ipc}
ASYNC_GROUP_AGGREGATOR_SIZE=${ASYNC_GROUP_AGGREGATOR_SIZE:-16}
ASYNC_DENSE_DATA_PLANE=${ASYNC_DENSE_DATA_PLANE:-mpi-p2p}
ASYNC_DENSE_UPDATE_STORAGE=${ASYNC_DENSE_UPDATE_STORAGE:-1}
ASYNC_DENSE_TRANSPORT_64N_GATE_JSON=${ASYNC_DENSE_TRANSPORT_64N_GATE_JSON:-}
ASYNC_COMPILED_MPICH_64N_GATE_JSON=${ASYNC_COMPILED_MPICH_64N_GATE_JSON:-}

# E97 B4 K40 token geometry.
BATCH_SIZE=${BATCH_SIZE:-4}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
DILOCO_K=${DILOCO_K:-40}
GENERATIONS=${GENERATIONS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
MODEL_LEVEL=${MODEL_LEVEL:-E97}
MODEL_PARAMS=${MODEL_PARAMS:-1.3b}
MODEL_TOKENIZER=${MODEL_TOKENIZER:-p50k_base}
MODEL_DIM=${MODEL_DIM:-1792}
MODEL_DEPTH=${MODEL_DEPTH:-11}
MODEL_N_HEADS=${MODEL_N_HEADS:-216}
MODEL_N_STATE=${MODEL_N_STATE:-32}
MODEL_N_GROUPS=${MODEL_N_GROUPS:-32}
MODEL_N_SLOTS=${MODEL_N_SLOTS:-64}
MODEL_EXPANSION=${MODEL_EXPANSION:-1.0}
MODEL_STATE_EXPANSION=${MODEL_STATE_EXPANSION:-2}
MODEL_GATE_ACTIVATION=${MODEL_GATE_ACTIVATION:-silu}
MODEL_LINEAR_STATE=${MODEL_LINEAR_STATE:-1}
MODEL_MLP_RATIO=${MODEL_MLP_RATIO:-2.2623}
MODEL_MLP_MULTIPLE=${MODEL_MLP_MULTIPLE:-64}
ASYNC_E97_BF16=${ASYNC_E97_BF16:-1}
ASYNC_E97_USE_CHUNKED=${ASYNC_E97_USE_CHUNKED:-1}
ASYNC_E97_CHUNK_SIZE=${ASYNC_E97_CHUNK_SIZE:-32}
ASYNC_E97_CHECKPOINT_INTERVAL=${ASYNC_E97_CHECKPOINT_INTERVAL:-64}
ASYNC_E97_GRADIENT_CHECKPOINTING=${ASYNC_E97_GRADIENT_CHECKPOINTING:-1}
ASYNC_E97_PROJECTION_CHUNK_SIZE=${ASYNC_E97_PROJECTION_CHUNK_SIZE:-256}
ASYNC_E97_LOSS_CHUNK_SIZE=${ASYNC_E97_LOSS_CHUNK_SIZE:-256}
TOKENS_PER_LOCAL_STEP=${TOKENS_PER_LOCAL_STEP:-16777216}
TOKENS_PER_DILOCO_GENERATION=${TOKENS_PER_DILOCO_GENERATION:-671088640}
TRAIN_MINUTES=${TRAIN_MINUTES:-700}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-12:00:00}
REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-3072.0}

# Durable-state policy recorded for production validation. The real trainer
# owns run-local latest.json/checkpoints; the external production latest path is
# recorded as a guard and is not mutated by this wrapper.
GENERATION_MANIFEST_EVERY=${GENERATION_MANIFEST_EVERY:-1}
RECOVERY_EVERY_GENERATIONS=${RECOVERY_EVERY_GENERATIONS:-5}
RECOVERY_EVERY_SECONDS=${RECOVERY_EVERY_SECONDS:-600}
EXPORT_EVERY_GENERATIONS=${EXPORT_EVERY_GENERATIONS:-45}
EXPORT_EVERY_SECONDS=${EXPORT_EVERY_SECONDS:-3600}
FINALIZATION_BUFFER_SECONDS=${FINALIZATION_BUFFER_SECONDS:-1200}
ASYNC_DILOCO_FORCE_FINALIZATION_CHECKPOINT=${ASYNC_DILOCO_FORCE_FINALIZATION_CHECKPOINT:-$ASYNC_DILOCO_NON_PRODUCTION_DEBUG}
CHAIN_UPDATE_ON_FAILURE=${CHAIN_UPDATE_ON_FAILURE:-0}
PRODUCTION_LATEST_GUARD=${PRODUCTION_LATEST_GUARD:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt}
PRODUCTION_LATEST_POLICY=${PRODUCTION_LATEST_POLICY:-run-local-latest-json-with-external-chain-latest-guard}

HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-User 2026-07-06 conditional approval: refreshed seed registered, 4n20m validation passed, queue 256n12h.}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" logs/frontier/async_diloco_e97
cd "$REPO"

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env
export TIKTOKEN_CACHE_DIR
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
export ASYNC_MPI_DENSE_BUCKET_BYTES ASYNC_COMPILED_MPICH_HELPER_BIN ASYNC_COMPILED_MPICH_IPC_DIR
export CRAY_MPI4PY_SITE=${CRAY_MPI4PY_SITE:-/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages}
PYTHON_BIN=$(command -v python)
[[ -n "$PYTHON_BIN" ]] || { echo "Activated runtime did not provide python" >&2; exit 66; }

[[ -r "$E97_CHECKPOINT" ]] || { echo "E97_CHECKPOINT is not readable: $E97_CHECKPOINT" >&2; exit 4; }
[[ -d "$TIKTOKEN_CACHE_DIR" ]] || { echo "TIKTOKEN_CACHE_DIR is missing: $TIKTOKEN_CACHE_DIR" >&2; exit 4; }
if [[ "$MODEL_TOKENIZER" == "p50k_base" ]]; then
  [[ -r "$TIKTOKEN_CACHE_DIR/ec7223a39ce59f226a68acc30dc1af2788490e15" || -r "$TIKTOKEN_CACHE_DIR/p50k_base.tiktoken" ]] || {
    echo "p50k_base tokenizer cache is missing under TIKTOKEN_CACHE_DIR=$TIKTOKEN_CACHE_DIR" >&2
    exit 4
  }
fi
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" != "1" ]]; then
  [[ -r "$DATA" ]] || { echo "DATA is not readable: $DATA" >&2; exit 4; }
fi
[[ -f "$ASYNC_ENTRYPOINT" ]] || {
  echo "Launch cannot start because ASYNC_ENTRYPOINT is missing: $ASYNC_ENTRYPOINT" >&2
  exit 65
}

if [[ "${ASYNC_DILOCO_RUNTIME_PROBE_ONLY:-0}" == "1" ]]; then
  echo "Production validation rejects ASYNC_DILOCO_RUNTIME_PROBE_ONLY=1; use ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1 to run the real trainer with debug-sized settings." >&2
  exit 67
fi
case "$ASYNC_ENTRYPOINT" in
  *async_diloco_e97_multinode.py|*async_diloco_e97_2n8n_debug.py)
    echo "Production validation rejects synthetic/protocol entrypoint: $ASYNC_ENTRYPOINT" >&2
    exit 68
    ;;
esac
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" && "$ASYNC_DILOCO_NON_PRODUCTION_DEBUG" != "1" ]]; then
  echo "Synthetic token streams are only allowed with ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1." >&2
  exit 69
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM" == "1" && ! -x "$ASYNC_COMPILED_MPICH_HELPER_BIN" ]]; then
  ARTIFACT_DIR="$ARTIFACT_DIR" OUT="$ASYNC_COMPILED_MPICH_HELPER_BIN" scripts/frontier/build_compiled_mpich_dense_helper.sh \
    2>&1 | tee "${LOG_DIR}/compiled_mpich_helper_build.log"
fi

CMD=(
  "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"
  --run-id "${TASK_ID}-${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
  --checkpoint "$E97_CHECKPOINT"
  --run-dir "$RUN_DIR"
  --metrics-json "$METRICS_JSON"
  --data "$DATA"
  --node-count "$ASYNC_NODE_COUNT"
  --worker-count "$ASYNC_WORKER_COUNT"
  --local-quorum "$ASYNC_LOCAL_QUORUM"
  --global-quorum "$ASYNC_GLOBAL_QUORUM"
  --generations "$GENERATIONS"
  --local-steps "$DILOCO_K"
  --timeout-s "$ASYNC_GLOBAL_TIMEOUT_S"
  --level "$MODEL_LEVEL"
  --params "$MODEL_PARAMS"
  --tokenizer "$MODEL_TOKENIZER"
  --batch-size "$BATCH_SIZE"
  --chunk-size "$CHUNK_SIZE"
  --lr "$LEARNING_RATE"
  --steps "$DILOCO_K"
  --e97-chunk-size "$ASYNC_E97_CHUNK_SIZE"
  --checkpoint-interval "$ASYNC_E97_CHECKPOINT_INTERVAL"
  --projection-chunk-size "$ASYNC_E97_PROJECTION_CHUNK_SIZE"
  --loss-chunk-size "$ASYNC_E97_LOSS_CHUNK_SIZE"
  --recovery-every-generations "$RECOVERY_EVERY_GENERATIONS"
  --recovery-every-seconds "$RECOVERY_EVERY_SECONDS"
  --export-every-generations "$EXPORT_EVERY_GENERATIONS"
  --export-every-seconds "$EXPORT_EVERY_SECONDS"
  --finalization-reserve-seconds "$FINALIZATION_BUFFER_SECONDS"
)
if [[ "$ASYNC_E97_BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$ASYNC_E97_USE_CHUNKED" == "1" ]]; then
  CMD+=(--use-chunked-e97)
fi
if [[ "$ASYNC_E97_GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--gradient-checkpointing)
fi
if [[ "$ASYNC_DILOCO_FORCE_FINALIZATION_CHECKPOINT" == "1" ]]; then
  CMD+=(--walltime-remaining-s "$FINALIZATION_BUFFER_SECONDS")
fi
if [[ -n "$MODEL_DIM" ]]; then
  CMD+=(--dim "$MODEL_DIM")
fi
if [[ -n "$MODEL_DEPTH" ]]; then
  CMD+=(--depth "$MODEL_DEPTH")
fi
if [[ -n "$MODEL_N_HEADS" ]]; then
  CMD+=(--n-heads "$MODEL_N_HEADS")
fi
if [[ -n "$MODEL_N_STATE" ]]; then
  CMD+=(--n-state "$MODEL_N_STATE")
fi
if [[ -n "$MODEL_N_GROUPS" ]]; then
  CMD+=(--n-groups "$MODEL_N_GROUPS")
fi
if [[ -n "$MODEL_N_SLOTS" ]]; then
  CMD+=(--n-slots "$MODEL_N_SLOTS")
fi
if [[ -n "$MODEL_EXPANSION" ]]; then
  CMD+=(--expansion "$MODEL_EXPANSION")
fi
if [[ -n "$MODEL_STATE_EXPANSION" ]]; then
  CMD+=(--state-expansion "$MODEL_STATE_EXPANSION")
fi
if [[ -n "$MODEL_GATE_ACTIVATION" ]]; then
  CMD+=(--gate-activation "$MODEL_GATE_ACTIVATION")
fi
if [[ -n "$MODEL_LINEAR_STATE" ]]; then
  CMD+=(--linear-state "$MODEL_LINEAR_STATE")
fi
if [[ -n "$MODEL_MLP_RATIO" ]]; then
  CMD+=(--mlp-ratio "$MODEL_MLP_RATIO")
fi
if [[ -n "$MODEL_MLP_MULTIPLE" ]]; then
  CMD+=(--mlp-multiple "$MODEL_MLP_MULTIPLE")
fi
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" ]]; then
  CMD+=(--synthetic-token-stream)
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_TCP_QUORUM" == "1" ]]; then
  CMD+=(--actual-multinode-tcp-quorum)
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM" == "1" ]]; then
  CMD+=(--actual-multinode-mpi-dense-quorum --mpi-dense-bucket-bytes "$ASYNC_MPI_DENSE_BUCKET_BYTES")
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM" == "1" ]]; then
  CMD+=(
    --actual-multinode-compiled-mpich-quorum
    --mpi-dense-bucket-bytes "$ASYNC_MPI_DENSE_BUCKET_BYTES"
    --compiled-mpich-helper-bin "$ASYNC_COMPILED_MPICH_HELPER_BIN"
    --compiled-mpich-ipc-dir "$ASYNC_COMPILED_MPICH_IPC_DIR"
  )
fi

LAUNCH_CMD=("${CMD[@]}")
if [[ "$ASYNC_NODE_COUNT" -gt 1 && ( "$ASYNC_ACTUAL_MULTINODE_TCP_QUORUM" == "1" || "$ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM" == "1" || "$ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM" == "1" ) ]]; then
  LAUNCH_CMD=(
    srun
    -N "$ASYNC_NODE_COUNT"
    -n "$ASYNC_WORKER_COUNT"
    --ntasks-per-node="$ASYNC_WORKER_COUNT_PER_NODE"
    -c 7
    --gpus-per-task=1
    --gpu-bind=closest
    "${CMD[@]}"
  )
fi

PRESUBMIT_STATUS=pass
PRESUBMIT_FAILURES=()
RUNTIME_FLAG_DRIFT=()
if [[ "$BATCH_SIZE" != "4" ]]; then
  PRESUBMIT_FAILURES+=("batch_size=$BATCH_SIZE does not match corrected B4 ladder batch_size=4")
fi
if [[ "$CHUNK_SIZE" != "2048" ]]; then
  PRESUBMIT_FAILURES+=("chunk_size=$CHUNK_SIZE does not match corrected B4 ladder chunk_size=2048")
fi
if [[ "$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM" == "1" ]]; then
  PRESUBMIT_FAILURES+=("synthetic token stream is not acceptable for corrected E97 ladder")
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_TCP_QUORUM" == "1" ]]; then
  PRESUBMIT_FAILURES+=("actual_multinode_tcp_quorum is metadata-only TCP control transport; dense async DiLoCo MPI P2P delta exchange is not implemented")
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM" == "1" ]]; then
  PRESUBMIT_FAILURES+=("actual_multinode_mpi_dense_quorum is the legacy mpi4py comparison path and is not accepted for Track B production")
fi
if [[ "$ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM" != "1" ]]; then
  PRESUBMIT_FAILURES+=("actual_multinode_compiled_mpich_quorum is not enabled")
fi
if [[ "$ASYNC_TRANSPORT_MODE" != "compiled-cray-mpich-helper-p2p" ]]; then
  PRESUBMIT_FAILURES+=("ASYNC_TRANSPORT_MODE=$ASYNC_TRANSPORT_MODE, not compiled-cray-mpich-helper-p2p")
fi
if [[ "$ASYNC_DENSE_DATA_PLANE" != "compiled-mpich-p2p" && "$ASYNC_DENSE_DATA_PLANE" != "mpi-p2p" ]]; then
  PRESUBMIT_FAILURES+=("ASYNC_DENSE_DATA_PLANE=$ASYNC_DENSE_DATA_PLANE, not compiled-mpich-p2p")
fi
if [[ "$ASYNC_NODE_COUNT" -ge 256 && "$ASYNC_DILOCO_NON_PRODUCTION_DEBUG" != "1" ]]; then
  if [[ -z "${ASYNC_COMPILED_MPICH_64N_GATE_JSON:-}" || ! -r "$ASYNC_COMPILED_MPICH_64N_GATE_JSON" ]]; then
    PRESUBMIT_FAILURES+=("256n production remains fail-closed until ASYNC_COMPILED_MPICH_64N_GATE_JSON points at a readable passing 64n compiled-helper gate artifact")
  elif ! python - "$ASYNC_COMPILED_MPICH_64N_GATE_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
text = json.dumps(payload, sort_keys=True)
ok = (
    payload.get("validation_status") == "pass"
    or payload.get("conclusion") in {"go", "pass", "dense-transport-64n-pass"}
    or '"validation_status": "pass"' in text
)
transport_ok = "compiled-cray-mpich-helper-p2p" in text
raise SystemExit(0 if ok and transport_ok else 1)
PY
  then
    PRESUBMIT_FAILURES+=("64n gate artifact did not record a passing compiled-cray-mpich-helper-p2p smoke")
  fi
fi
if [[ "$SLURM_INTENDED_NTASKS_PER_NODE" != "8" ]]; then
  PRESUBMIT_FAILURES+=("Slurm topology uses $SLURM_INTENDED_NTASKS_PER_NODE task(s) per node, not stable B4 8 GPU tasks per node")
fi
if [[ "$SLURM_INTENDED_GPUS_PER_TASK" != "1" ]]; then
  PRESUBMIT_FAILURES+=("Slurm GPU request uses --gpus-per-task=$SLURM_INTENDED_GPUS_PER_TASK, not stable B4 --gpus-per-task=1")
fi
if [[ "$SLURM_INTENDED_GPU_BIND" != "closest" ]]; then
  PRESUBMIT_FAILURES+=("Slurm GPU binding is $SLURM_INTENDED_GPU_BIND, not stable B4 --gpu-bind=closest")
fi
if [[ "$ASYNC_WORKER_COUNT_PER_NODE" != "8" ]]; then
  PRESUBMIT_FAILURES+=("configured worker_count_per_node=$ASYNC_WORKER_COUNT_PER_NODE, not B4 GPU-island 8")
fi
if [[ "$ASYNC_DENSE_UPDATE_STORAGE" != "1" ]]; then
  PRESUBMIT_FAILURES+=("ASYNC_DENSE_UPDATE_STORAGE=$ASYNC_DENSE_UPDATE_STORAGE, so dense update exchange is not recorded")
fi
if [[ "$MODEL_LINEAR_STATE" != "0" ]]; then
  RUNTIME_FLAG_DRIFT+=("linear_state=$MODEL_LINEAR_STATE vs stable train.py B4 linear_state=0")
fi
if [[ "$ASYNC_E97_USE_CHUNKED" != "0" ]]; then
  RUNTIME_FLAG_DRIFT+=("use_chunked_e97=$ASYNC_E97_USE_CHUNKED vs stable train.py B4 use_chunked_e97=0")
fi
if [[ "$ASYNC_E97_CHECKPOINT_INTERVAL" != "16" ]]; then
  RUNTIME_FLAG_DRIFT+=("checkpoint_interval=$ASYNC_E97_CHECKPOINT_INTERVAL vs stable train.py B4 checkpoint_interval=16")
fi
if ((${#RUNTIME_FLAG_DRIFT[@]} > 0)); then
  PRESUBMIT_FAILURES+=("runtime flag drift from stable train.py B4 path: ${RUNTIME_FLAG_DRIFT[*]}")
fi
if ((${#PRESUBMIT_FAILURES[@]} > 0)); then
  PRESUBMIT_STATUS=fail
fi

printf '%q ' "${LAUNCH_CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

if [[ "$ASYNC_NODE_COUNT" -gt 1 && ( "$ASYNC_ACTUAL_MULTINODE_TCP_QUORUM" == "1" || "$ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM" == "1" || "$ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM" == "1" ) ]]; then
  ASYNC_LAUNCH_USES_SRUN=1
else
  ASYNC_LAUNCH_USES_SRUN=0
fi

{
  echo "task_id=$TASK_ID"
  echo "training_target=$TRAINING_TARGET"
  echo "seed_latest_path=$SEED_LATEST_PATH"
  echo "e97_checkpoint=$E97_CHECKPOINT"
  echo "data=$DATA"
  echo "tiktoken_cache_dir=$TIKTOKEN_CACHE_DIR"
  echo "async_entrypoint=$ASYNC_ENTRYPOINT"
  echo "scaleout_variant=$SCALEOUT_VARIANT"
  echo "run_root=$RUN_ROOT"
  echo "run_dir=$RUN_DIR"
  echo "metrics_json=$METRICS_JSON"
  echo "command_file=$COMMAND_FILE"
  echo "env_file=$ENV_FILE"
  echo "env_prefix=$ENV_PREFIX"
  echo "emender_conda_env=$EMENDER_CONDA_ENV"
  echo "python_bin=$PYTHON_BIN"
  echo "requested_walltime=$REQUESTED_WALLTIME"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "presubmit_status=$PRESUBMIT_STATUS"
  echo "presubmit_requires_corrected_b4_ladder=$ASYNC_REQUIRE_CORRECTED_B4_LADDER"
  echo "presubmit_failure_count=${#PRESUBMIT_FAILURES[@]}"
  if ((${#PRESUBMIT_FAILURES[@]} == 0)); then
    echo "presubmit_failure_0=none"
  else
    for idx in "${!PRESUBMIT_FAILURES[@]}"; do
      echo "presubmit_failure_${idx}=${PRESUBMIT_FAILURES[$idx]}"
    done
  fi
  echo "runtime_flag_drift_count=${#RUNTIME_FLAG_DRIFT[@]}"
  if ((${#RUNTIME_FLAG_DRIFT[@]} == 0)); then
    echo "runtime_flag_drift_0=none"
  else
    for idx in "${!RUNTIME_FLAG_DRIFT[@]}"; do
      echo "runtime_flag_drift_${idx}=${RUNTIME_FLAG_DRIFT[$idx]}"
    done
  fi
  echo "stable_b4_reference_ntasks_per_node=8"
  echo "stable_b4_reference_gpus_per_task=1"
  echo "stable_b4_reference_gpu_bind=closest"
  echo "stable_b4_reference_linear_state=0"
  echo "stable_b4_reference_use_chunked_e97=0"
  echo "stable_b4_reference_checkpoint_interval=16"
  echo "slurm_intended_ntasks_per_node=$SLURM_INTENDED_NTASKS_PER_NODE"
  echo "slurm_intended_gpus_per_task=$SLURM_INTENDED_GPUS_PER_TASK"
  echo "slurm_intended_gpu_bind=$SLURM_INTENDED_GPU_BIND"
  echo "async_node_count=$ASYNC_NODE_COUNT"
  echo "async_worker_count_per_node=$ASYNC_WORKER_COUNT_PER_NODE"
  echo "async_worker_count=$ASYNC_WORKER_COUNT"
  echo "async_actual_multinode_tcp_quorum=$ASYNC_ACTUAL_MULTINODE_TCP_QUORUM"
  echo "async_actual_multinode_mpi_dense_quorum=$ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM"
  echo "async_actual_multinode_compiled_mpich_quorum=$ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM"
  echo "async_launch_uses_srun=$ASYNC_LAUNCH_USES_SRUN"
  echo "async_local_quorum=$ASYNC_LOCAL_QUORUM"
  echo "async_global_quorum=$ASYNC_GLOBAL_QUORUM"
  echo "async_global_timeout_s=$ASYNC_GLOBAL_TIMEOUT_S"
  echo "async_staleness_policy=$ASYNC_STALENESS_POLICY"
  echo "async_weighting=$ASYNC_WEIGHTING"
  echo "async_transport_mode=$ASYNC_TRANSPORT_MODE"
  echo "async_mpi_dense_bucket_bytes=$ASYNC_MPI_DENSE_BUCKET_BYTES"
  echo "async_compiled_mpich_helper_bin=$ASYNC_COMPILED_MPICH_HELPER_BIN"
  echo "async_compiled_mpich_ipc_dir=$ASYNC_COMPILED_MPICH_IPC_DIR"
  echo "async_group_aggregator_size=$ASYNC_GROUP_AGGREGATOR_SIZE"
  echo "async_dense_data_plane=$ASYNC_DENSE_DATA_PLANE"
  echo "async_dense_update_storage=$ASYNC_DENSE_UPDATE_STORAGE"
  echo "async_dense_transport_64n_gate_json=$ASYNC_DENSE_TRANSPORT_64N_GATE_JSON"
  echo "async_compiled_mpich_64n_gate_json=${ASYNC_COMPILED_MPICH_64N_GATE_JSON:-}"
  echo "mpich_gpu_support_enabled=$MPICH_GPU_SUPPORT_ENABLED"
  echo "cray_mpi4py_site=$CRAY_MPI4PY_SITE"
  echo "batch_size=$BATCH_SIZE"
  echo "chunk_size=$CHUNK_SIZE"
  echo "diloco_k=$DILOCO_K"
  echo "generations=$GENERATIONS"
  echo "learning_rate=$LEARNING_RATE"
  echo "model_level=$MODEL_LEVEL"
  echo "model_params=$MODEL_PARAMS"
  echo "model_tokenizer=$MODEL_TOKENIZER"
  echo "model_dim=$MODEL_DIM"
  echo "model_depth=$MODEL_DEPTH"
  echo "model_n_heads=$MODEL_N_HEADS"
  echo "model_n_state=$MODEL_N_STATE"
  echo "model_n_groups=$MODEL_N_GROUPS"
  echo "model_n_slots=$MODEL_N_SLOTS"
  echo "model_expansion=$MODEL_EXPANSION"
  echo "model_state_expansion=$MODEL_STATE_EXPANSION"
  echo "model_gate_activation=$MODEL_GATE_ACTIVATION"
  echo "model_linear_state=$MODEL_LINEAR_STATE"
  echo "model_mlp_ratio=$MODEL_MLP_RATIO"
  echo "model_mlp_multiple=$MODEL_MLP_MULTIPLE"
  echo "async_e97_bf16=$ASYNC_E97_BF16"
  echo "async_e97_use_chunked=$ASYNC_E97_USE_CHUNKED"
  echo "async_e97_chunk_size=$ASYNC_E97_CHUNK_SIZE"
  echo "async_e97_checkpoint_interval=$ASYNC_E97_CHECKPOINT_INTERVAL"
  echo "async_e97_gradient_checkpointing=$ASYNC_E97_GRADIENT_CHECKPOINTING"
  echo "async_e97_projection_chunk_size=$ASYNC_E97_PROJECTION_CHUNK_SIZE"
  echo "async_e97_loss_chunk_size=$ASYNC_E97_LOSS_CHUNK_SIZE"
  echo "tokens_per_local_step=$TOKENS_PER_LOCAL_STEP"
  echo "tokens_per_diloco_generation=$TOKENS_PER_DILOCO_GENERATION"
  echo "generation_manifest_every=$GENERATION_MANIFEST_EVERY"
  echo "recovery_every_generations=$RECOVERY_EVERY_GENERATIONS"
  echo "recovery_every_seconds=$RECOVERY_EVERY_SECONDS"
  echo "export_every_generations=$EXPORT_EVERY_GENERATIONS"
  echo "export_every_seconds=$EXPORT_EVERY_SECONDS"
  echo "finalization_buffer_seconds=$FINALIZATION_BUFFER_SECONDS"
  echo "async_diloco_force_finalization_checkpoint=$ASYNC_DILOCO_FORCE_FINALIZATION_CHECKPOINT"
  echo "chain_update_on_failure=$CHAIN_UPDATE_ON_FAILURE"
  echo "production_latest_guard=$PRODUCTION_LATEST_GUARD"
  echo "production_latest_policy=$PRODUCTION_LATEST_POLICY"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "probe_only_allowed=0"
  echo "synthetic_runtime_allowed=0"
  echo "non_production_debug=$ASYNC_DILOCO_NON_PRODUCTION_DEBUG"
  echo "synthetic_token_stream=$ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM"
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

if [[ "$ASYNC_REQUIRE_CORRECTED_B4_LADDER" == "1" && "$PRESUBMIT_STATUS" != "pass" ]]; then
  {
    echo "Corrected E97 B4 smoke ladder pre-submit gate failed; refusing to start Slurm payload."
    printf ' - %s\n' "${PRESUBMIT_FAILURES[@]}"
    echo "Set ASYNC_REQUIRE_CORRECTED_B4_LADDER=0 only for explicitly labeled diagnostic/non-comparable runs."
  } >&2
  exit 70
fi

if [[ "${ASYNC_DILOCO_HUMAN_APPROVED:-0}" != "1" && "$ASYNC_DILOCO_NON_PRODUCTION_DEBUG" != "1" ]]; then
  echo "Prepared launch package only. Refusing to start 256-node async DiLoCo without ASYNC_DILOCO_HUMAN_APPROVED=1 from WG human approval task." >&2
  exit 64
fi

exec "${LAUNCH_CMD[@]}"
