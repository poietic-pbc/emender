#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-longctx
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

# Serialized long-context memory and correctness qualification. Submit with
# sbatch --nodes=N and never overlap this Debug-QoS study with another debug job.
set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CONTEXT_SIZE=${CONTEXT_SIZE:-8192}
LOSS_CHUNK_SIZE=${LOSS_CHUNK_SIZE:-2048}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-16}
PROJECTION_CHUNK_SIZE=${PROJECTION_CHUNK_SIZE:-0}
MAX_STEPS=${MAX_STEPS:-2}
DILOCO_K=${DILOCO_K:-1}
EXPERT_BACKEND=${EXPERT_BACKEND:-rocblas}
EMPTY_CACHE_INTERVAL=${EMPTY_CACHE_INTERVAL:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
RUN_ROOT=${RUN_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/longctx-debug-${SLURM_JOB_NUM_NODES}n-c${CONTEXT_SIZE}-j${SLURM_JOB_ID}}

[[ "$CONTEXT_SIZE" =~ ^[1-9][0-9]*$ && "$LOSS_CHUNK_SIZE" =~ ^[0-9]+$ \
   && "$CHECKPOINT_INTERVAL" =~ ^[1-9][0-9]*$ && "$MAX_STEPS" =~ ^[1-9][0-9]*$ \
   && $((CONTEXT_SIZE % CHECKPOINT_INTERVAL)) -eq 0 ]] || {
  echo "invalid long-context controls" >&2; exit 64;
}
[[ "$GRADIENT_CHECKPOINTING" =~ ^[01]$ ]] || exit 64
cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p logs/frontier/e97_moe "$RUN_ROOT"
SNAPSHOT="$RUN_ROOT/source-$SOURCE_COMMIT"
rm -rf "$SNAPSHOT"; mkdir -p "$SNAPSHOT"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|Nodes=%D|NodeList=%N|TimeLimit=%l' | tee "$RUN_ROOT/squeue.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$RUN_ROOT/scontrol.txt"
printf '%s\n' \
  "source_commit=$SOURCE_COMMIT" "context_size=$CONTEXT_SIZE" \
  "loss_chunk_size=$LOSS_CHUNK_SIZE" "checkpoint_interval=$CHECKPOINT_INTERVAL" \
  "projection_chunk_size=$PROJECTION_CHUNK_SIZE" \
  "gradient_checkpointing=$GRADIENT_CHECKPOINTING" "max_steps=$MAX_STEPS" \
  "diloco_k=$DILOCO_K" "expert_backend=$EXPERT_BACKEND" \
  "empty_cache_interval=$EMPTY_CACHE_INTERVAL" > "$RUN_ROOT/recipe.txt"

source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((34000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT

GC_FLAG=--no-gradient-checkpointing
[[ "$GRADIENT_CHECKPOINTING" == 1 ]] && GC_FLAG=--gradient-checkpointing
export GC_FLAG CONTEXT_SIZE LOSS_CHUNK_SIZE CHECKPOINT_INTERVAL PROJECTION_CHUNK_SIZE MAX_STEPS DILOCO_K EXPERT_BACKEND EMPTY_CACHE_INTERVAL
srun --label --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  source "'"$REPO"'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'"$SNAPSHOT"'"; export PYTHONPATH="'"$SNAPSHOT"'${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-moe-longctx-${SLURM_JOB_ID}-${SLURM_PROCID}"
  rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
    --max-steps "$MAX_STEPS" --minutes 0 --batch-size 1 --chunk-size "$CONTEXT_SIZE" \
    "$GC_FLAG" --loss-chunk-size "$LOSS_CHUNK_SIZE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --projection-chunk-size "$PROJECTION_CHUNK_SIZE" --diloco-k "$DILOCO_K" \
    --expert-backend "$EXPERT_BACKEND" --empty-cache-interval "$EMPTY_CACHE_INTERVAL" \
    --profile-phases \
    --log-jsonl "'"$RUN_ROOT"'/training.jsonl"
'
