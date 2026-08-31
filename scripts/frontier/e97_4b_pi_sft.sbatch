#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-4b-pi-sft
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300

# Fixed-world fail-stop E97 4B Pi masked-SFT qualification/canary.
set -euo pipefail
: "${REPO:?immutable submitted checkout is required}"
: "${SOURCE_COMMIT:?immutable source commit is required}"
: "${RUN_ROOT:?absolute stable run root is required}"
: "${EXPECTED_NODES:?}"
: "${EXPECTED_WORLD_SIZE:?}"
: "${AUTHORITY_ROOT:?}"
: "${AUTHORITY_SHA256:?}"
: "${PACK_ROOT:?}"
: "${PACK_SHA256:?}"

PARENT=${PARENT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-hybrid-ddp-256n-recovery-r3/train/checkpoint_step_024448_loss_2.3981.pt}
PARENT_SHA256=${PARENT_SHA256:-3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568}
SOURCE_ARGS=${SOURCE_ARGS:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-hybrid-ddp-256n-recovery-r3/train/args.json}
STEPS=${STEPS:-8}
SAVE_EVERY=${SAVE_EVERY:-$STEPS}
DILOCO_K=${DILOCO_K:-8}
LR=${LR:-0.00001}
WARMUP_STEPS=${WARMUP_STEPS:-8}
CONTEXT_SIZE=${CONTEXT_SIZE:-4096}
SAMPLER_KEY=${SAMPLER_KEY:-974003}
RESUME=${RESUME:-}
EXPECTED_PARTITION=${EXPECTED_PARTITION:-batch}
EXPECTED_QOS=${EXPECTED_QOS:-debug}

[[ "$RUN_ROOT" == /* ]] || { echo "RUN_ROOT must be absolute" >&2; exit 64; }
[[ "$EXPECTED_NODES" == 1 || "$EXPECTED_NODES" == 8 ]] || exit 64
[[ "$EXPECTED_WORLD_SIZE" == $((EXPECTED_NODES * 8)) ]] || exit 64
[[ "$SLURM_JOB_NUM_NODES" == "$EXPECTED_NODES" ]] || exit 64
[[ $((STEPS % DILOCO_K)) == 0 && $((SAVE_EVERY % DILOCO_K)) == 0 ]] || exit 64

mkdir -p "$RUN_ROOT"/{checkpoints,identity,logs,sources,terminal}
exec > >(tee -a "$RUN_ROOT/logs/batch-${SLURM_JOB_ID}.out") \
     2> >(tee -a "$RUN_ROOT/logs/batch-${SLURM_JOB_ID}.err" >&2)
cd "$REPO"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical Frontier Python is required}"

record="$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}-running.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$record"
grep -E "Partition=${EXPECTED_PARTITION}([[:space:]]|$)" "$record" >/dev/null
grep -E "QOS=${EXPECTED_QOS}([[:space:]]|$)" "$record" >/dev/null
grep -E "Requeue=0([[:space:]]|$)" "$record" >/dev/null
squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%N|%P|%q|%R' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}-running.txt"
grep -F "|${EXPECTED_PARTITION}|${EXPECTED_QOS}|" \
  "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}-running.txt" >/dev/null

[[ $(git rev-parse HEAD) == "$SOURCE_COMMIT" ]] || exit 65
git cat-file -e "${SOURCE_COMMIT}^{commit}"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-commit.txt"
printf '%s\n' "$SLURM_JOB_NODELIST" > "$RUN_ROOT/identity/nodelist-${SLURM_JOB_ID}.txt"
printf '%s\n' "$PARENT_SHA256  $PARENT" > "$RUN_ROOT/identity/parent.sha256"
printf '%s\n' "$AUTHORITY_SHA256  $AUTHORITY_ROOT/manifest.json" > "$RUN_ROOT/identity/authority.sha256"
printf '%s\n' "$PACK_SHA256  $PACK_ROOT/manifest.json" > "$RUN_ROOT/identity/packs.sha256"
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_SHA256" ]] || exit 65
[[ $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_SHA256" ]] || exit 65
[[ -r "$PARENT" && -r "$SOURCE_ARGS" ]] || exit 65

snapshot="$RUN_ROOT/sources/$SOURCE_COMMIT"
mkdir -p "$snapshot"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$snapshot"
export MASTER_PORT=$((43000 + SLURM_JOB_ID % 1000))
export WORLD_SIZE=$EXPECTED_WORLD_SIZE
export OMP_NUM_THREADS=7
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
resume_args=()
if [[ -n "$RESUME" ]]; then
  [[ -r "$RESUME" ]] || exit 65
  resume_args=(--resume "$RESUME")
fi
printf '%q ' "${resume_args[@]}" > "$RUN_ROOT/identity/resume-args-${SLURM_JOB_ID}.txt"; printf '\n' >> "$RUN_ROOT/identity/resume-args-${SLURM_JOB_ID}.txt"

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "'$snapshot'"
  export PYTHONPATH="'$snapshot'"
  export MASTER_ADDR=$(getent ahostsv4 "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" | awk "NR==1{print \$1}")
  export RANK=$SLURM_PROCID LOCAL_RANK=0
  export TRITON_CACHE_DIR=/tmp/e97-4b-pi-sft-${SLURM_JOB_ID}-${SLURM_PROCID}
  rm -rf "$TRITON_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR"
  resume=()
  if [[ -n "'$RESUME'" ]]; then resume=(--resume "'$RESUME'"); fi
  "$EMENDER_PYTHON" scripts/train_e97_4b_pi_sft.py \
    --parent-checkpoint "'$PARENT'" --parent-sha256 "'$PARENT_SHA256'" \
    --source-args-json "'$SOURCE_ARGS'" --source-commit "'$SOURCE_COMMIT'" \
    --authority-root "'$AUTHORITY_ROOT'" --authority-sha256 "'$AUTHORITY_SHA256'" \
    --pack-root "'$PACK_ROOT'" --pack-sha256 "'$PACK_SHA256'" \
    --output-root "'$RUN_ROOT'/checkpoints" --log-jsonl "'$RUN_ROOT'/logs/training.jsonl" \
    --steps "'$STEPS'" --save-every "'$SAVE_EVERY'" --diloco-k "'$DILOCO_K'" \
    --context-size "'$CONTEXT_SIZE'" --lr "'$LR'" --warmup-steps "'$WARMUP_STEPS'" \
    --sampler-key "'$SAMPLER_KEY'" "${resume[@]}"
'
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P \
  > "$RUN_ROOT/terminal/sacct-${SLURM_JOB_ID}.txt" || true
if [[ "$rc" == 0 ]]; then
  latest=$(readlink -f "$RUN_ROOT/checkpoints/latest.pt")
  "$EMENDER_PYTHON" "$snapshot/scripts/verify_e97_4b_pi_sft_checkpoint.py" "$latest" \
    --output "$RUN_ROOT/terminal/checkpoint-${SLURM_JOB_ID}.reload.json"
  sha256sum "$latest" > "$RUN_ROOT/terminal/checkpoint-${SLURM_JOB_ID}.sha256"
fi
exit "$rc"
