#!/bin/bash
# Production fixed-world E97 launcher: each execution epoch is one fresh srun.
#
# RUN_ID is mandatory and is deliberately independent of SLURM_JOB_ID.  On a
# child failure the parent batch shell keeps the allocation, promotes only the
# child's atomically published latest.pt (if any), and starts a new process
# group.  If Slurm reports fewer healthy whole nodes, the next epoch uses that
# smaller fixed world.  Batch-shell/allocation loss is handled by Slurm requeue,
# not by an in-process membership or communicator-shrink protocol.
#
# Example (submission is intentionally not performed by this file):
#   RUN_ID=e97-8n-acceptance-... INITIAL_CHECKPOINT=/.../latest.pt \
#     sbatch scripts/frontier/e97_same_allocation_restart.sbatch

#SBATCH -A bif148
#SBATCH -J e97-samealloc
#SBATCH -p batch
#SBATCH -N 8
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-kill
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/frontier/%x-%j.out
#SBATCH -e logs/frontier/%x-%j.err

set -euo pipefail

samealloc_bad_node_state() {
  local state=${1^^}
  [[ "$state" =~ (^|[+~,])(DOWN|DRAIN|DRAINING|DRAINED|FAIL|FAILING|FAILED|NO_RESPOND|NOT_RESPONDING|POWER_DOWN|POWERED_DOWN)([+~,]|$) ]]
}

samealloc_usable_nodes() {
  local allocation_nodelist=$1 node record state
  while IFS= read -r node; do
    [[ -n "$node" ]] || continue
    record=$(scontrol show node "$node" -o 2>/dev/null) || continue
    if [[ "$record" =~ State=([^[:space:]]+) ]]; then
      state=${BASH_REMATCH[1]}
    else
      continue
    fi
    samealloc_bad_node_state "$state" || printf '%s\n' "$node"
  done < <(scontrol show hostnames "$allocation_nodelist")
}

samealloc_atomic_latest() {
  local link=$1 target=$2 tmp
  [[ -r "$target" && -f "$target" ]] || return 1
  mkdir -p "$(dirname "$link")"
  tmp="${link}.${SLURM_JOB_ID:-manual}.$$.tmp"
  rm -f "$tmp"
  ln -s "$(readlink -f "$target")" "$tmp"
  mv -Tf "$tmp" "$link"
}

samealloc_promote_epoch_latest() {
  local epoch_output=$1 stable_latest=$2
  local -a candidates=()
  local candidate resolved
  if [[ -L "$epoch_output/latest.pt" ]]; then
    candidates+=("$epoch_output/latest.pt")
  else
    while IFS= read -r -d '' candidate; do
      candidates+=("$candidate")
    done < <(find "$epoch_output" -mindepth 2 -maxdepth 2 -type l -name latest.pt -print0 2>/dev/null)
  fi
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  if (( ${#candidates[@]} != 1 )); then
    echo "refusing ambiguous epoch latest.pt set under $epoch_output" >&2
    return 2
  fi
  resolved=$(readlink -f "${candidates[0]}") || return 1
  [[ -r "$resolved" && -f "$resolved" ]] || return 1
  samealloc_atomic_latest "$stable_latest" "$resolved"
}

samealloc_next_epoch() {
  local state_file=$1 current=0 next tmp
  if [[ -r "$state_file" ]]; then
    read -r current < "$state_file"
    [[ "$current" =~ ^[0-9]+$ ]] || {
      echo "invalid execution epoch state: $state_file" >&2
      return 65
    }
  fi
  next=$((current + 1))
  tmp="${state_file}.${SLURM_JOB_ID:-manual}.$$.tmp"
  printf '%s\n' "$next" > "$tmp"
  mv -Tf "$tmp" "$state_file"
  printf '%s\n' "$next"
}

samealloc_master_port() {
  local epoch=$1 prior_port=${2:-} job_number=0 port
  if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    job_number=$((10#${SLURM_JOB_ID}))
  fi
  port=$((20000 + ((job_number + epoch) % 40000)))
  if [[ -n "$prior_port" && "$port" == "$prior_port" ]]; then
    port=$((20000 + ((port - 19999) % 40000)))
  fi
  printf '%s\n' "$port"
}

samealloc_request_requeue() {
  local reason=$1
  echo "[samealloc] requesting Slurm requeue: $reason" >&2
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "cannot requeue without SLURM_JOB_ID" >&2
    return 75
  fi
  scontrol requeue "$SLURM_JOB_ID"
}

samealloc_main() {
  : "${SLURM_JOB_ID:?must run inside a Slurm allocation}"
  : "${SLURM_JOB_NODELIST:?Slurm allocation nodelist is missing}"
  : "${SLURM_JOB_NUM_NODES:?Slurm allocation node count is missing}"
  : "${RUN_ID:?set a stable RUN_ID independent of SLURM_JOB_ID}"

  REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
  RUN_ROOT=${RUN_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation}
  RUN_DIR=${RUN_DIR:-${RUN_ROOT%/}/$RUN_ID}
  INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-}
  TASKS_PER_NODE=${TASKS_PER_NODE:-8}
  CPUS_PER_TASK=${CPUS_PER_TASK:-7}
  TARGET_NODES=${TARGET_NODES:-$SLURM_JOB_NUM_NODES}
  MIN_NODES=${MIN_NODES:-1}
  MAX_EXECUTION_EPOCHS_PER_ATTEMPT=${MAX_EXECUTION_EPOCHS_PER_ATTEMPT:-3}
  EXECUTION_EPOCH_TIMEOUT_SECONDS=${EXECUTION_EPOCH_TIMEOUT_SECONDS:-86400}
  FAILED_STEP_WAIT_SECONDS=${FAILED_STEP_WAIT_SECONDS:-60}
  FAILED_STEP_KILL_GRACE_SECONDS=${FAILED_STEP_KILL_GRACE_SECONDS:-60}
  REQUEUE_ON_EXHAUSTION=${REQUEUE_ON_EXHAUSTION:-1}

  DILOCO_K=${DILOCO_K:-40}
  SAVE_EVERY=${SAVE_EVERY:-200}
  KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-2}
  DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-67108864}
  DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-hierarchical}
  DILOCO_MERGE_GROUP_SIZE=${DILOCO_MERGE_GROUP_SIZE:-4}
  DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=${DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY:-8}
  DILOCO_MERGE_COMPLETION_BARRIER=${DILOCO_MERGE_COMPLETION_BARRIER:-1}
  TRAIN_MINUTES=${TRAIN_MINUTES:-1440}
  BATCH_SIZE=${BATCH_SIZE:-4}
  CHUNK_SIZE=${CHUNK_SIZE:-2048}
  LOG_EVERY=${LOG_EVERY:-25}
  VAL_EVERY=${VAL_EVERY:-10000}
  COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
  WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-900}
  WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-40}
  DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-40}
  DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
  VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
  TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}

  [[ "$RUN_ID" != *"/"* && "$RUN_ID" != "." && "$RUN_ID" != ".." ]] || {
    echo "RUN_ID must be one stable path component" >&2; return 64;
  }
  [[ "$RUN_DIR" == /* ]] || { echo "RUN_DIR must be absolute" >&2; return 64; }
  [[ "$TARGET_NODES" =~ ^[1-9][0-9]*$ && "$MIN_NODES" =~ ^[1-9][0-9]*$ ]] || return 64
  (( TARGET_NODES <= SLURM_JOB_NUM_NODES && MIN_NODES <= TARGET_NODES )) || {
    echo "invalid TARGET_NODES/MIN_NODES for allocation" >&2; return 64;
  }
  (( SAVE_EVERY % DILOCO_K == 0 )) || {
    echo "SAVE_EVERY must be an exact multiple of DILOCO_K" >&2; return 64;
  }
  [[ -r "$DATA" && -r "$VAL_DATA" ]] || { echo "training data inputs are unreadable" >&2; return 64; }

  mkdir -p "$RUN_DIR"/{epochs,identity,logs,supervisor,train} logs/frontier
  cd "$REPO"

  # The canonical activation is authoritative for both submission validation and
  # the batch parent.  RCCL plugin opt-in must precede module resolution.
  export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
  export FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}
  export FRONTIER_RUNTIME_PROFILE=${FRONTIER_RUNTIME_PROFILE:-olcf-rccl-debug}
  export REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}
  # shellcheck source=activate_emender_frontier.sh
  source "$REPO/scripts/frontier/activate_emender_frontier.sh"
  PYTHON_BIN=$EMENDER_PYTHON
  TRAIN_PYTHON_BIN=$EMENDER_PYTHON
  export REPO PYTHON_BIN TRAIN_PYTHON_BIN TIKTOKEN_CACHE_DIR
  [[ $(readlink -f "$PYTHON_BIN") == $(readlink -f "$EMENDER_PYTHON") ]] || return 66
  frontier_require_requested_rccl_net_plugin "same-allocation batch parent" >/dev/null

  export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
  export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
  export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-kdreg2}
  export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
  export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-2048}
  export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
  export FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO:-alt_read}
  export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
  export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-net.so}
  export NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-240}

  local stable_latest="$RUN_DIR/train/latest.pt"
  if [[ -e "$stable_latest" && ! -L "$stable_latest" ]]; then
    echo "stable latest.pt exists but is not an atomic symlink" >&2
    return 65
  fi
  if [[ -L "$stable_latest" ]]; then
    [[ -r "$stable_latest" ]] || { echo "stable latest.pt is dangling" >&2; return 65; }
  else
    [[ -z "$INITIAL_CHECKPOINT" || $(basename "$INITIAL_CHECKPOINT") == latest.pt ]] || {
      echo "INITIAL_CHECKPOINT must name an atomically published latest.pt" >&2; return 64;
    }
    [[ -n "$INITIAL_CHECKPOINT" && -L "$INITIAL_CHECKPOINT" && -r "$INITIAL_CHECKPOINT" ]] || {
      echo "first attempt requires a readable latest.pt symlink in INITIAL_CHECKPOINT" >&2; return 64;
    }
    samealloc_atomic_latest "$stable_latest" "$INITIAL_CHECKPOINT"
  fi

  local identity_file="$RUN_DIR/identity/run-id.txt"
  if [[ -e "$identity_file" ]]; then
    [[ $(<"$identity_file") == "$RUN_ID" ]] || { echo "stable run identity conflict" >&2; return 65; }
  else
    printf '%s\n' "$RUN_ID" > "$identity_file"
  fi
  printf '%s|%s|%s|%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" \
    "${SLURM_RESTART_COUNT:-0}" "$SLURM_JOB_NODELIST" >> "$RUN_DIR/identity/slurm-attempts.tsv"
  git rev-parse HEAD > "$RUN_DIR/identity/git-commit-${SLURM_JOB_ID}.txt"

  local -a common_args=(
    --data "$DATA" --val_data "$VAL_DATA" --bf16 --batch_size "$BATCH_SIZE"
    --chunk_size "$CHUNK_SIZE" --tokenizer p50k_base --optimizer schedulefree
    --seed 42 --lr 0.001007 --train_minutes "$TRAIN_MINUTES"
    --timer_after_compile_warmup --compile_warmup_steps "$COMPILE_WARMUP_STEPS"
    --log_every "$LOG_EVERY" --val_every "$VAL_EVERY"
    --save_every "$SAVE_EVERY" --keep_checkpoints "$KEEP_CHECKPOINTS"
    --walltime_final_checkpoint_margin_seconds "$WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS"
    --walltime_check_every "$WALLTIME_CHECK_EVERY"
    --distributed_health_check_every "$DISTRIBUTED_HEALTH_CHECK_EVERY"
    --diloco --diloco_k "$DILOCO_K" --diloco_outer_optimizer avg
    --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --diloco_export_basis x
    --diloco_island_size 1 --diloco_merge_topology "$DILOCO_MERGE_TOPOLOGY"
    --diloco_merge_group_size "$DILOCO_MERGE_GROUP_SIZE"
    --diloco_merge_group_create_barrier_every "$DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY"
    --diloco_merge_completion_barrier "$DILOCO_MERGE_COMPLETION_BARRIER"
    --diloco_merge_bucket_numel "$DILOCO_MERGE_BUCKET_NUMEL"
  )
  local -a model_args=(
    --level E97 --params 100m --dim 1792 --depth 11 --n_heads 216 --n_state 32
    --n_groups 32 --n_slots 64 --state_expansion 2 --expansion 1.0 --use_gate 1
    --use_permutation 1 --gate_activation silu --linear_state 0 --use_write_gate 0
    --use_triton 1 --use_chunked_e97 0 --e97_chunk_size 32 --mlp_ratio 2.2623
    --mlp_multiple 64 --checkpoint_interval 16 --weight_decay 0.01 --grad_accum 1
    --grad_clip 1.0
  )

  local REQUEUE_REQUESTED=0 CHILD_PID=""
  samealloc_batch_signal() {
    REQUEUE_REQUESTED=1
    echo "[samealloc] batch signal received; terminating child epoch" >&2
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
      kill -TERM -- "-$CHILD_PID" 2>/dev/null || kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
  }
  trap samealloc_batch_signal TERM HUP INT

  local -a usable=()
  mapfile -t usable < <(samealloc_usable_nodes "$SLURM_JOB_NODELIST")
  (( ${#usable[@]} >= MIN_NODES )) || {
    samealloc_request_requeue "fewer than MIN_NODES healthy at startup"
    return 0
  }
  local current_nodes=$TARGET_NODES
  (( current_nodes <= ${#usable[@]} )) || current_nodes=${#usable[@]}
  usable=("${usable[@]:0:current_nodes}")

  local local_attempt epoch epoch_dir epoch_output master_host master_port prior_port=""
  local node_csv tasks rc promoted
  for ((local_attempt=1; local_attempt<=MAX_EXECUTION_EPOCHS_PER_ATTEMPT; local_attempt++)); do
    epoch=$(samealloc_next_epoch "$RUN_DIR/supervisor/execution-epoch.txt")
    epoch_dir=$(printf '%s/epochs/epoch-%06d' "$RUN_DIR" "$epoch")
    epoch_output="$RUN_DIR/train"
    mkdir -p "$epoch_output"
    master_host=${usable[0]}
    master_port=$(samealloc_master_port "$epoch" "$prior_port")
    prior_port=$master_port
    node_csv=$(IFS=,; echo "${usable[*]}")
    tasks=$((current_nodes * TASKS_PER_NODE))

    {
      echo "run_id=$RUN_ID"
      echo "slurm_job_id=$SLURM_JOB_ID"
      echo "slurm_restart_count=${SLURM_RESTART_COUNT:-0}"
      echo "execution_epoch=$epoch"
      echo "nodes=$current_nodes"
      echo "tasks=$tasks"
      echo "nodelist=$node_csv"
      echo "master_addr=$master_host"
      echo "master_port=$master_port"
      echo "resume=$stable_latest"
      echo "diloco_k=$DILOCO_K"
      echo "save_every=$SAVE_EVERY"
      echo "keep_checkpoints=$KEEP_CHECKPOINTS"
      echo "diloco_merge_bucket_numel=$DILOCO_MERGE_BUCKET_NUMEL"
    } > "$epoch_dir/launch.env"
    printf '%q ' "$PYTHON_BIN" -u train.py "${common_args[@]}" "${model_args[@]}" \
      --resume "$stable_latest" --output "$RUN_DIR" --exact_output_dir "$epoch_output" > "$epoch_dir/train-command.txt"
    printf '\n' >> "$epoch_dir/train-command.txt"

    echo "[samealloc] epoch=$epoch nodes=$current_nodes tasks=$tasks port=$master_port resume=$stable_latest"
    set +e
    setsid timeout --signal=TERM --kill-after="${FAILED_STEP_KILL_GRACE_SECONDS}s" \
      "${EXECUTION_EPOCH_TIMEOUT_SECONDS}s" \
      srun --exact --no-kill --kill-on-bad-exit=1 --wait="$FAILED_STEP_WAIT_SECONDS" \
        -N"$current_nodes" -n"$tasks" --ntasks-per-node="$TASKS_PER_NODE" \
        -c"$CPUS_PER_TASK" --gpus-per-task=1 --gpu-bind=closest --nodelist="$node_csv" \
        --export=ALL,MASTER_ADDR="$master_host",MASTER_PORT="$master_port",EMENDER_EXECUTION_EPOCH="$epoch" \
        bash -lc 'source "$REPO/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; frontier_require_requested_rccl_net_plugin "same-allocation child rank ${SLURM_PROCID}" >/dev/null; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec "$TRAIN_PYTHON_BIN" -u train.py "$@"' \
        bash "${common_args[@]}" "${model_args[@]}" --resume "$stable_latest" --output "$RUN_DIR" --exact_output_dir "$epoch_output" \
        > "$epoch_dir/train.out" 2> "$epoch_dir/train.err" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    rc=$?
    CHILD_PID=""
    set -e

    promoted=0
    if samealloc_promote_epoch_latest "$epoch_output" "$stable_latest"; then
      promoted=1
    fi
    printf '%s|%s|%s|%s|%s|%s\n' "$epoch" "$rc" "$current_nodes" "$tasks" "$master_port" "$promoted" \
      >> "$RUN_DIR/supervisor/execution-epochs.tsv"

    if (( REQUEUE_REQUESTED )); then
      samealloc_request_requeue "batch signal after epoch $epoch"
      return 0
    fi
    if (( rc == 0 )); then
      echo "[samealloc] execution epoch $epoch completed"
      return 0
    fi

    echo "[samealloc] execution epoch $epoch failed rc=$rc; latest_promoted=$promoted" >&2
    mapfile -t usable < <(samealloc_usable_nodes "$SLURM_JOB_NODELIST")
    if (( ${#usable[@]} < MIN_NODES )); then
      samealloc_request_requeue "healthy whole-node count ${#usable[@]} below MIN_NODES"
      return 0
    fi
    if (( ${#usable[@]} < current_nodes )); then
      current_nodes=${#usable[@]}
      echo "[samealloc] reducing next fixed world to $current_nodes whole nodes" >&2
    fi
    usable=("${usable[@]:0:current_nodes}")
  done

  if [[ "$REQUEUE_ON_EXHAUSTION" == "1" ]]; then
    samealloc_request_requeue "execution epoch limit exhausted"
    return 0
  fi
  echo "execution epoch limit exhausted; requeue disabled" >&2
  return 70
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  samealloc_main "$@"
fi
