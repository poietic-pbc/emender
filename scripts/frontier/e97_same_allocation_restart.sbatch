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
  [[ "$state" =~ (^|[^A-Z_])(DOWN|DRAIN|DRAINING|DRAINED|FAIL|FAILING|FAILED|NO_RESPOND|NOT_RESPONDING|INVALID_REG|INVAL|POWER_DOWN|POWERED_DOWN)([^A-Z_]|$) ]]
}

samealloc_select_usable_nodes() {
  local allocation_nodelist=$1 excluded_name=$2 usable_name=$3 hard_bad_name=$4
  local -n excluded_ref=$excluded_name usable_ref=$usable_name hard_bad_ref=$hard_bad_name
  local hostnames node record state
  usable_ref=()
  hard_bad_ref=()
  hostnames=$(scontrol show hostnames "$allocation_nodelist" 2>/dev/null) || return 69
  while IFS= read -r node; do
    [[ -n "$node" ]] || continue
    if [[ -n ${excluded_ref[$node]-} ]]; then
      continue
    fi
    if ! record=$(scontrol show node "$node" -o 2>/dev/null); then
      hard_bad_ref[$node]=SLURM_QUERY_FAILED
      excluded_ref[$node]=slurm-nonresponsive
      continue
    fi
    if [[ "$record" =~ State=([^[:space:]]+) ]]; then
      state=${BASH_REMATCH[1]}
    else
      hard_bad_ref[$node]=SLURM_STATE_MISSING
      excluded_ref[$node]=slurm-nonresponsive
      continue
    fi
    if samealloc_bad_node_state "$state"; then
      hard_bad_ref[$node]=$state
      excluded_ref[$node]="slurm-$state"
    else
      usable_ref+=("$node")
    fi
  done <<< "$hostnames"
}

samealloc_usable_nodes() {
  local allocation_nodelist=$1
  local -a selected=()
  local -A excluded=() hard_bad=()
  samealloc_select_usable_nodes "$allocation_nodelist" excluded selected hard_bad || return
  (( ${#selected[@]} == 0 )) || printf '%s\n' "${selected[@]}"
}

samealloc_direct_failure_host() {
  local log_file=$1
  shift
  local line host
  local -A allowed=() attributed=()
  for host in "$@"; do
    allowed[$host]=1
  done
  [[ -r "$log_file" ]] || return 1
  while IFS= read -r line; do
    # Only Slurm's explicit task-exit attribution is trusted. Rank-prefixed
    # tracebacks and collective error reporters are deliberately ignored.
    if [[ "$line" =~ ^srun:\ error:\ ([A-Za-z0-9._-]+):\ task\ [0-9]+:\ Exited\ with\ exit\ code\ [1-9][0-9]*[[:space:]]*$ ]]; then
      host=${BASH_REMATCH[1]}
      [[ -n ${allowed[$host]-} ]] && attributed[$host]=1
    fi
  done < "$log_file"
  (( ${#attributed[@]} == 1 )) || return 1
  printf '%s\n' "${!attributed[@]}"
}

samealloc_record_direct_failure() {
  local host=$1 strikes_name=$2 excluded_name=$3
  local -n strikes_ref=$strikes_name excluded_ref=$excluded_name
  local count=$(( ${strikes_ref[$host]:-0} + 1 ))
  strikes_ref[$host]=$count
  if (( count >= 2 )); then
    excluded_ref[$host]="direct-failure-strike-$count"
  fi
}

samealloc_update_no_progress() {
  local previous_step=$1 observed_step=$2 count_name=$3 step_name=$4
  local -n count_ref=$count_name step_ref=$step_name
  if [[ "$observed_step" =~ ^[0-9]+$ ]] && (( observed_step > previous_step )); then
    step_ref=$observed_step
    count_ref=0
  else
    count_ref=$((count_ref + 1))
  fi
}

samealloc_reset_allocation_policy() {
  local new_identity=$1 identity_name=$2 strikes_name=$3 excluded_name=$4 count_name=$5
  local -n identity_ref=$identity_name strikes_ref=$strikes_name excluded_ref=$excluded_name count_ref=$count_name
  if [[ "$identity_ref" != "$new_identity" ]]; then
    identity_ref=$new_identity
    strikes_ref=()
    excluded_ref=()
    count_ref=0
  fi
}

samealloc_checkpoint_step() {
  local latest=$1 resolved base
  resolved=$(readlink -f "$latest") || return 1
  base=$(basename "$resolved")
  [[ "$base" =~ ^checkpoint_step_([0-9]+)_ ]] || return 1
  printf '%s\n' "$((10#${BASH_REMATCH[1]}))"
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
  # The legacy total-epoch variable remains a compatibility input for accepted
  # wrappers, but now bounds only consecutive failures without checkpoint
  # progress. Epochs that promote a newer atomic checkpoint reset the bound.
  MAX_CONSECUTIVE_NO_PROGRESS_FAILURES=${MAX_CONSECUTIVE_NO_PROGRESS_FAILURES:-${MAX_EXECUTION_EPOCHS_PER_ATTEMPT:-3}}
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
  [[ "$MAX_CONSECUTIVE_NO_PROGRESS_FAILURES" =~ ^[1-9][0-9]*$ ]] || return 64
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

  local allocation_identity="${SLURM_JOB_ID}:${SLURM_RESTART_COUNT:-0}:${SLURM_JOB_NODELIST}"
  local active_allocation_identity=""
  local consecutive_no_progress=0 committed_step
  local -a usable=() launched_nodes=() filtered=()
  local -A node_strikes=() excluded_nodes=() hard_bad_nodes=()
  samealloc_reset_allocation_policy "$allocation_identity" active_allocation_identity \
    node_strikes excluded_nodes consecutive_no_progress
  printf '%s|%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$allocation_identity" \
    >> "$RUN_DIR/supervisor/allocation-identities.tsv"

  if ! samealloc_select_usable_nodes "$SLURM_JOB_NODELIST" excluded_nodes usable hard_bad_nodes; then
    samealloc_request_requeue "batch allocation node list is unavailable at startup"
    return 0
  fi
  (( ${#usable[@]} >= MIN_NODES )) || {
    samealloc_request_requeue "fewer than MIN_NODES healthy at startup"
    return 0
  }
  local current_nodes=$TARGET_NODES
  (( current_nodes <= ${#usable[@]} )) || current_nodes=${#usable[@]}
  usable=("${usable[@]:0:current_nodes}")
  committed_step=$(samealloc_checkpoint_step "$stable_latest") || {
    echo "stable latest.pt does not encode an exact checkpoint step" >&2
    return 65
  }

  local epoch epoch_dir epoch_output master_host master_port prior_port=""
  local node_csv tasks rc promoted observed_step direct_host strike_count node health_query_failed
  while true; do
    epoch=$(samealloc_next_epoch "$RUN_DIR/supervisor/execution-epoch.txt")
    epoch_dir=$(printf '%s/epochs/epoch-%06d' "$RUN_DIR" "$epoch")
    epoch_output="$RUN_DIR/train"
    # The epoch directory is owned by the production launcher and must exist
    # before launch.env, command, stdout, or stderr are opened.
    mkdir -p "$epoch_dir" "$epoch_output"
    master_host=${usable[0]}
    master_port=$(samealloc_master_port "$epoch" "$prior_port")
    prior_port=$master_port
    node_csv=$(IFS=,; echo "${usable[*]}")
    tasks=$((current_nodes * TASKS_PER_NODE))
    launched_nodes=("${usable[@]}")

    {
      echo "run_id=$RUN_ID"
      echo "slurm_job_id=$SLURM_JOB_ID"
      echo "slurm_restart_count=${SLURM_RESTART_COUNT:-0}"
      echo "allocation_identity=$allocation_identity"
      echo "execution_epoch=$epoch"
      echo "nodes=$current_nodes"
      echo "tasks=$tasks"
      echo "nodelist=$node_csv"
      echo "master_addr=$master_host"
      echo "master_port=$master_port"
      echo "resume=$stable_latest"
      echo "resume_step=$committed_step"
      echo "diloco_k=$DILOCO_K"
      echo "save_every=$SAVE_EVERY"
      echo "keep_checkpoints=$KEEP_CHECKPOINTS"
      echo "diloco_merge_bucket_numel=$DILOCO_MERGE_BUCKET_NUMEL"
    } > "$epoch_dir/launch.env"
    printf '%q ' "$PYTHON_BIN" -u train.py "${common_args[@]}" "${model_args[@]}" \
      --resume "$stable_latest" --output "$RUN_DIR" --exact_output_dir "$epoch_output" > "$epoch_dir/train-command.txt"
    printf '\n' >> "$epoch_dir/train-command.txt"

    echo "[samealloc] epoch=$epoch nodes=$current_nodes tasks=$tasks port=$master_port resume=$stable_latest"
    # Real Slurm receives these through --export below; exporting them in the
    # parent also makes the execution identity explicit to step shims/tests.
    export MASTER_ADDR=$master_host MASTER_PORT=$master_port EMENDER_EXECUTION_EPOCH=$epoch
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

    # Slurm is the first node-health authority consulted after a failed child.
    # Its hard-bad states are installed in excluded_nodes immediately.
    health_query_failed=0
    if (( rc != 0 )); then
      hard_bad_nodes=()
      if ! samealloc_select_usable_nodes "$SLURM_JOB_NODELIST" excluded_nodes usable hard_bad_nodes; then
        health_query_failed=1
      fi
    fi

    promoted=0
    if samealloc_promote_epoch_latest "$epoch_output" "$stable_latest"; then
      promoted=1
    fi
    observed_step=$(samealloc_checkpoint_step "$stable_latest") || observed_step=$committed_step
    printf '%s|%s|%s|%s|%s|%s|%s\n' "$epoch" "$rc" "$current_nodes" "$tasks" "$master_port" "$promoted" "$observed_step" \
      >> "$RUN_DIR/supervisor/execution-epochs.tsv"

    if (( REQUEUE_REQUESTED )); then
      samealloc_request_requeue "batch signal after epoch $epoch"
      return 0
    fi
    if (( rc == 0 )); then
      echo "[samealloc] execution epoch $epoch completed"
      return 0
    fi

    echo "[samealloc] execution epoch $epoch failed rc=$rc; latest_promoted=$promoted step=$observed_step" >&2
    samealloc_update_no_progress "$committed_step" "$observed_step" consecutive_no_progress committed_step

    if (( health_query_failed )); then
      samealloc_request_requeue "batch allocation node list was lost after epoch $epoch"
      return 0
    fi
    for node in "${!hard_bad_nodes[@]}"; do
      printf '%s|%s|exclude|slurm|%s\n' "$epoch" "$node" "${hard_bad_nodes[$node]}" \
        >> "$RUN_DIR/supervisor/node-policy.tsv"
      echo "[samealloc] excluding Slurm-unusable node $node state=${hard_bad_nodes[$node]}" >&2
    done

    direct_host=""
    if direct_host=$(samealloc_direct_failure_host "$epoch_dir/train.err" "${launched_nodes[@]}"); then
      samealloc_record_direct_failure "$direct_host" node_strikes excluded_nodes
      strike_count=${node_strikes[$direct_host]}
      printf '%s|%s|strike|direct-task-exit|%s\n' "$epoch" "$direct_host" "$strike_count" \
        >> "$RUN_DIR/supervisor/node-policy.tsv"
      echo "[samealloc] direct failure attribution host=$direct_host strike=$strike_count" >&2
    else
      printf '%s|-|ambiguous|no-strike|-\n' "$epoch" >> "$RUN_DIR/supervisor/node-policy.tsv"
      echo "[samealloc] no unique direct hostname attribution; no node strike" >&2
    fi

    filtered=()
    for node in "${usable[@]}"; do
      [[ -n ${excluded_nodes[$node]-} ]] || filtered+=("$node")
    done
    usable=("${filtered[@]}")
    if (( ${#usable[@]} < MIN_NODES )); then
      samealloc_request_requeue "no usable fixed-world step: ${#usable[@]} healthy nodes below MIN_NODES"
      return 0
    fi
    current_nodes=$TARGET_NODES
    (( current_nodes <= ${#usable[@]} )) || current_nodes=${#usable[@]}
    usable=("${usable[@]:0:current_nodes}")
    if (( current_nodes < ${#launched_nodes[@]} )); then
      echo "[samealloc] reducing next fixed world to $current_nodes whole nodes" >&2
    fi

    if (( consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS_FAILURES )); then
      if [[ "$REQUEUE_ON_EXHAUSTION" == "1" ]]; then
        samealloc_request_requeue "bounded consecutive failures made no checkpoint progress ($consecutive_no_progress)"
        return 0
      fi
      echo "bounded consecutive failures made no checkpoint progress; requeue disabled" >&2
      return 70
    fi
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  samealloc_main "$@"
fi
