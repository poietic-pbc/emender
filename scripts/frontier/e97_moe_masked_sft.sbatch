#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-masked-sft
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

# One fixed-world ADR-003 masked-SFT execution epoch. No restart/requeue/shrink.
set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${RUN_ROOT:?stable absolute run root required}"
EXPECTED_NODES=${EXPECTED_NODES:-$SLURM_JOB_NUM_NODES}
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
MAX_STEPS=${MAX_STEPS:-1}
DILOCO_K=${DILOCO_K:-1}
LR=${LR:-0.000002}
CONTEXT_SIZE=${CONTEXT_SIZE:-4096}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_EVERY=${SAVE_EVERY:-0}
VALIDATION_BATCHES=${VALIDATION_BATCHES:-2}
FINAL_CHECKPOINT_DELAY_SECONDS=${FINAL_CHECKPOINT_DELAY_SECONDS:-0}
SFT_SAMPLER_KEY=${SFT_SAMPLER_KEY:-42}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/authority}
AUTHORITY_MANIFEST_SHA256=${AUTHORITY_MANIFEST_SHA256:-e2461a28e7699637e282564589aeea35a150dc9e6e3cd62ffd6a2b5c0a2f47d8}
PACK_ROOT=${PACK_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/packs-4096-v1}
PACK_MANIFEST_SHA256=${PACK_MANIFEST_SHA256:-010dc5f326f351463d8af675f2d6a906c8f056e48ee4c93b1ee34dddd1b202ea}
PARENT_ROOT=${PARENT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
PARENT_MANIFEST_SHA256=${PARENT_MANIFEST_SHA256:-95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}

[[ "$RUN_ROOT" == /* && "$EXPECTED_NODES" =~ ^(1|8|16)$ \
   && "$MAX_STEPS" =~ ^[1-9][0-9]*$ && "$DILOCO_K" =~ ^[1-9][0-9]*$ \
   && "$CONTEXT_SIZE" == 4096 && "$BATCH_SIZE" == 1 \
   && "$SAVE_EVERY" =~ ^[0-9]+$ ]] || { echo "invalid masked-SFT controls" >&2; exit 64; }
(( SAVE_EVERY == 0 || SAVE_EVERY % DILOCO_K == 0 )) || exit 64
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=$EXPECTED_NODES"* \
   && "$record" == *"NumTasks=$((EXPECTED_NODES * 8))"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* && "$record" == *"Requeue=0"* ]] || {
  echo "live scheduler binding mismatch" >&2; exit 65; }
cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$RUN_ROOT"/{identity,logs,sources,checkpoints,terminal} logs/frontier/e97_moe
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-${SLURM_JOB_ID}.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/scripts/frontier/e97_35b_moe_train.py" ]]; then
  temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"; rm -rf "$temporary"; mkdir -p "$temporary"
  git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"; mv -T "$temporary" "$SNAPSHOT"
fi
[[ -f "$AUTHORITY_ROOT/manifest.json" && -f "$PACK_ROOT/manifest.json" \
   && -f "$PACK_ROOT/validation.json" && -f "$PARENT_ROOT/manifest.json" \
   && -f "$SEED_CHECKPOINT" && -f "$SEED_ARGS_JSON" ]] || exit 66
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_MANIFEST_SHA256" \
   && $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_MANIFEST_SHA256" \
   && $(sha256sum "$PARENT_ROOT/manifest.json" | awk '{print $1}') == "$PARENT_MANIFEST_SHA256" ]] || exit 66
jq -e --arg a "$AUTHORITY_MANIFEST_SHA256" --arg p "$PACK_MANIFEST_SHA256" \
  '.status == "pass" and .authority_manifest_sha256 == $a and .pack_manifest_sha256 == $p' \
  "$PACK_ROOT/validation.json" >/dev/null
printf '%s\n' \
  "schema=emender-e97-masked-sft-run-v1" "parent_manifest_sha256=$PARENT_MANIFEST_SHA256" \
  "authority_manifest_sha256=$AUTHORITY_MANIFEST_SHA256" "pack_manifest_sha256=$PACK_MANIFEST_SHA256" \
  "context_size=$CONTEXT_SIZE" "batch_size=$BATCH_SIZE" "max_steps=$MAX_STEPS" \
  "diloco_k=$DILOCO_K" "lr=$LR" "sampler_key=$SFT_SAMPLER_KEY" \
  > "$RUN_ROOT/identity/recipe-${SLURM_JOB_ID}.txt"

JOB_SEED="/tmp/emender-e97-sft-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-sft-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"
sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
RESUME_ROOT=
if [[ -L "$RUN_ROOT/checkpoints/latest" ]]; then
  generation=$(readlink -f "$RUN_ROOT/checkpoints/latest")
  [[ -f "$generation/manifest.json" ]] || exit 67
  RESUME_ROOT="$RUN_ROOT/checkpoints"
fi
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 10000)) WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7} MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
export SNAPSHOT JOB_SEED JOB_SEED_ARGS RESUME_ROOT RUN_ROOT MAX_STEPS DILOCO_K LR CONTEXT_SIZE BATCH_SIZE
export AUTHORITY_ROOT AUTHORITY_MANIFEST_SHA256 PACK_ROOT PACK_MANIFEST_SHA256 PARENT_ROOT PARENT_MANIFEST_SHA256
export SFT_SAMPLER_KEY SAVE_EVERY VALIDATION_BATCHES FINAL_CHECKPOINT_DELAY_SECONDS
set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-masked-sft-${SLURM_JOB_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
  resume=(); [[ -z "$RESUME_ROOT" ]] || resume=(--resume-root "$RESUME_ROOT")
  "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
    --seed-checkpoint "$JOB_SEED" --seed-args-json "$JOB_SEED_ARGS" \
    --max-steps "$MAX_STEPS" --batch-size "$BATCH_SIZE" --chunk-size "$CONTEXT_SIZE" \
    --gradient-checkpointing --loss-chunk-size 2048 --checkpoint-loss-chunks \
    --checkpoint-interval 16 --projection-chunk-size 2048 --moe-token-chunk-size 4096 \
    --lr "$LR" --weight-decay 0.0 --diloco-k "$DILOCO_K" --expert-backend rocblas \
    --empty-cache-interval 1 --offload-schedulefree-state \
    --checkpoint-root "$RUN_ROOT/checkpoints" --save-every "$SAVE_EVERY" --keep-checkpoints 2 \
    --sft-authority-root "$AUTHORITY_ROOT" --sft-authority-manifest-sha256 "$AUTHORITY_MANIFEST_SHA256" \
    --sft-pack-root "$PACK_ROOT" --sft-pack-manifest-sha256 "$PACK_MANIFEST_SHA256" \
    --sft-sampler-key "$SFT_SAMPLER_KEY" --sft-parent-root "$PARENT_ROOT" \
    --sft-parent-manifest-sha256 "$PARENT_MANIFEST_SHA256" \
    --sft-validation-batches "$VALIDATION_BATCHES" \
    --final-checkpoint-delay-seconds "$FINAL_CHECKPOINT_DELAY_SECONDS" \
    "${resume[@]}" --log-jsonl "$RUN_ROOT/logs/training.jsonl"
'
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,ExitCode -n -P \
  > "$RUN_ROOT/terminal/sacct-${SLURM_JOB_ID}.txt" || true
exit "$rc"
