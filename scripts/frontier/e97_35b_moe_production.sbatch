#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-35b-moe-production
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 128
#SBATCH -t 00:40:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

# Fixed-world, fail-stop E97 MoE execution epoch. One canonical eight-rank
# island is checkpoint authority, matching dense DiLoCo's rank-zero authority
# without materializing the 35B graph on one GCD.
set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${RUN_ID:?stable run ID required}"
: "${RUN_ROOT:?stable absolute run root required}"
TRAIN_MINUTES=${TRAIN_MINUTES:-0}
MAX_STEPS=${MAX_STEPS:-0}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
EXPECTED_NODES=${EXPECTED_NODES:-$SLURM_JOB_NUM_NODES}
EXPECTED_PARTITION=${EXPECTED_PARTITION:-batch}
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-00:40:00}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
DATA_PATH=${DATA_PATH:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
SAVE_EVERY=${SAVE_EVERY:-200}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-2}
DILOCO_K=${DILOCO_K:-40}

[[ "$RUN_ROOT" == /* && "$RUN_ID" != */* ]] || { echo "invalid stable run identity" >&2; exit 64; }
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ && "$TRAIN_MINUTES" == 0 ]] || {
  echo "multinode production requires positive MAX_STEPS and TRAIN_MINUTES=0" >&2; exit 64;
}
[[ "$SAVE_EVERY" =~ ^[0-9]+$ && "$DILOCO_K" =~ ^[0-9]+$ \
   && $SAVE_EVERY -gt 0 && $((SAVE_EVERY % DILOCO_K)) -eq 0 ]] || {
  echo "SAVE_EVERY must be positive and K-aligned" >&2; exit 64;
}
cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$record"
[[ "$record" == *"NumNodes=$EXPECTED_NODES"* \
   && "$record" == *"NumTasks=$((EXPECTED_NODES * 8))"* \
   && "$record" == *"Partition=$EXPECTED_PARTITION"* \
   && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* \
   && "$record" == *"Requeue=0"* ]] || {
  echo "live scheduler binding mismatch" >&2; exit 65;
}

mkdir -p "$RUN_ROOT"/{identity,logs,sources,checkpoints,terminal} logs/frontier/e97_moe
if [[ -e "$RUN_ROOT/identity/run-id.txt" ]]; then
  [[ $(<"$RUN_ROOT/identity/run-id.txt") == "$RUN_ID" ]] || { echo "run identity conflict" >&2; exit 66; }
else
  printf '%s\n' "$RUN_ID" > "$RUN_ROOT/identity/run-id.txt"
fi
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"

SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/scripts/frontier/e97_35b_moe_train.py" ]]; then
  temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"
  rm -rf "$temporary"; mkdir -p "$temporary"
  git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"
  mv -T "$temporary" "$SNAPSHOT"
fi
[[ -f "$SEED_CHECKPOINT" && -f "$SEED_ARGS_JSON" && -f "$DATA_PATH" ]] || {
  echo "seed, seed args, or data unavailable" >&2; exit 66;
}

# The dense seed is needed to instantiate the graph on both cold start and
# restore. Distribute it once per node instead of making 2,048 ranks read the
# same 7.7 GB Lustre object.
JOB_SEED="/tmp/emender-e97-513b-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-513b-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"
sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
RESUME_ROOT=
if [[ -L "$CHECKPOINT_ROOT/latest" ]]; then
  generation=$(readlink -f "$CHECKPOINT_ROOT/latest")
  [[ -r "$generation/manifest.json" ]] || { echo "checkpoint latest is incomplete" >&2; exit 67; }
  RESUME_ROOT="$CHECKPOINT_ROOT"
fi

source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((31000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=$SLURM_NTASKS OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
export SNAPSHOT JOB_SEED JOB_SEED_ARGS DATA_PATH CHECKPOINT_ROOT RESUME_ROOT TRAIN_MINUTES MAX_STEPS
export SAVE_EVERY KEEP_CHECKPOINTS DILOCO_K RUN_ROOT

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "$SNAPSHOT"
  export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-35b-moe-${SLURM_JOB_ID}-${SLURM_PROCID}"
  rm -rf "$TRITON_CACHE_DIR"
  resume=()
  [[ -z "$RESUME_ROOT" ]] || resume=(--resume-root "$RESUME_ROOT")
  "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
    --seed-checkpoint "$JOB_SEED" --seed-args-json "$JOB_SEED_ARGS" --data "$DATA_PATH" \
    --max-steps "$MAX_STEPS" --minutes "$TRAIN_MINUTES" --batch-size 4 --chunk-size 2048 \
    --diloco-k "$DILOCO_K" --expert-backend rocblas \
    --checkpoint-root "$CHECKPOINT_ROOT" --save-every "$SAVE_EVERY" \
    --keep-checkpoints "$KEEP_CHECKPOINTS" "${resume[@]}" \
    --log-jsonl "$RUN_ROOT/logs/training.jsonl"
'
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
exit "$rc"
