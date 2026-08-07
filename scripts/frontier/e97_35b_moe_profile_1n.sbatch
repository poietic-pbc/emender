#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-35b-moe-profile-1n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SOURCE_COMMIT=${SOURCE_COMMIT:?submit with an immutable source commit}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-4}
EXPERT_BACKEND=${EXPERT_BACKEND:-rocblas}
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p logs/frontier/e97_moe
RUN_ROOT=${RUN_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/profile-1n-${EXPERT_BACKEND}-b${BATCH_SIZE}-${SLURM_JOB_ID}}
mkdir -p "$RUN_ROOT"
SNAPSHOT=$RUN_ROOT/source-$SOURCE_COMMIT
rm -rf "$SNAPSHOT"; mkdir -p "$SNAPSHOT"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N'
scontrol show job -dd "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(JobId|Partition|QOS|JobState|NodeList)=' || true
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((33000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT BATCH_SIZE MAX_STEPS EXPERT_BACKEND

srun --label --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  source "'"$REPO"'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'"$SNAPSHOT"'"; export PYTHONPATH="'"$SNAPSHOT"'${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-moe-profile-${SLURM_JOB_ID}-${SLURM_PROCID}"
  rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
    --max-steps "$MAX_STEPS" --minutes 0 --batch-size "$BATCH_SIZE" --chunk-size 2048 \
    --expert-backend "$EXPERT_BACKEND" --profile-phases --log-jsonl "'"$RUN_ROOT"'/training.jsonl"
'
