#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-sft-opt-state-3way
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 3
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

# Three independent one-node/eight-GCD EP worlds at 1e-4: all-fresh,
# mature-router/fresh-nonrouter, and the converse optimizer-state diagnostic.
# This is an ADR-003 fail-stop diagnostic, not a multinode DiLoCo run.
set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${CAMPAIGN_ROOT:?absolute campaign root required}"
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
MAX_STEPS=${MAX_STEPS:-64}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/authority}
AUTHORITY_MANIFEST_SHA256=${AUTHORITY_MANIFEST_SHA256:-e2461a28e7699637e282564589aeea35a150dc9e6e3cd62ffd6a2b5c0a2f47d8}
PACK_ROOT=${PACK_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/packs-4096-v1}
PACK_MANIFEST_SHA256=${PACK_MANIFEST_SHA256:-010dc5f326f351463d8af675f2d6a906c8f056e48ee4c93b1ee34dddd1b202ea}
PARENT_ROOT=${PARENT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
PARENT_MANIFEST_SHA256=${PARENT_MANIFEST_SHA256:-95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[[ "$CAMPAIGN_ROOT" == /* && "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 64
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=3"* && "$record" == *"NumTasks=24"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* && "$record" == *"Requeue=0"* ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$CAMPAIGN_ROOT"/{identity,sources,terminal} logs/frontier/e97_moe
arms=(all-fresh router-preserved nonrouter-preserved)
lrs=(0.0001 0.0001 0.0001)
states=(fresh router-preserved nonrouter-preserved)
for arm in "${arms[@]}"; do
  mkdir -p "$CAMPAIGN_ROOT/$arm"/{logs,identity}
  [[ ! -f "$CAMPAIGN_ROOT/$arm/logs/training.jsonl" ]] || {
    echo "sweep arm already has training evidence: $arm" >&2; exit 66; }
done
printf '%s\n' "$record" > "$CAMPAIGN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$CAMPAIGN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$CAMPAIGN_ROOT/identity/source-${SLURM_JOB_ID}.txt"
SNAPSHOT="$CAMPAIGN_ROOT/sources/$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/scripts/frontier/e97_35b_moe_train.py" ]]; then
  temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"; rm -rf "$temporary"; mkdir -p "$temporary"
  git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"; mv -T "$temporary" "$SNAPSHOT"
fi
[[ -f "$SEED_CHECKPOINT" && -f "$SEED_ARGS_JSON" \
   && $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_MANIFEST_SHA256" \
   && $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_MANIFEST_SHA256" \
   && $(sha256sum "$PARENT_ROOT/manifest.json" | awk '{print $1}') == "$PARENT_MANIFEST_SHA256" ]] || exit 66
jq -e --arg a "$AUTHORITY_MANIFEST_SHA256" --arg p "$PACK_MANIFEST_SHA256" \
  '.status == "pass" and .authority_manifest_sha256 == $a and .pack_manifest_sha256 == $p' \
  "$PACK_ROOT/validation.json" >/dev/null
printf '%s\n' \
  'schema=emender-e97-masked-sft-optimizer-state-factorial-v1' \
  'states=fresh,router-preserved,nonrouter-preserved' 'lr=1e-4' \
  'nodes_per_arm=1' 'world_per_arm=8' 'diloco=none-screening-only' \
  'weight_decay=0.01-matched' "max_steps=$MAX_STEPS" \
  'validation=all-1777-packs-exactly-once' \
  "authority_manifest_sha256=$AUTHORITY_MANIFEST_SHA256" \
  "pack_manifest_sha256=$PACK_MANIFEST_SHA256" \
  "parent_manifest_sha256=$PARENT_MANIFEST_SHA256" \
  > "$CAMPAIGN_ROOT/identity/recipe.txt"
JOB_SEED="/tmp/emender-e97-sft-sweep-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-sft-sweep-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"
sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#hosts[@]} == 3 ]] || exit 65
export SNAPSHOT JOB_SEED JOB_SEED_ARGS AUTHORITY_ROOT AUTHORITY_MANIFEST_SHA256 PACK_ROOT PACK_MANIFEST_SHA256
export PARENT_ROOT PARENT_MANIFEST_SHA256 MAX_STEPS SOURCE_COMMIT
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}

launch_arm() {
  local index=$1 arm=$2 lr=$3 state=$4 host=$5
  local root="$CAMPAIGN_ROOT/$arm"
  local address port delay
  address=$(getent ahostsv4 "$host" | awk 'NR==1 {print $1}')
  port=$((36000 + (SLURM_JOB_ID + index * 97) % 20000))
  delay=$((index * 10))
  srun --exclusive --exact --nodes=1 --ntasks=8 --nodelist="$host" \
    --label --kill-on-bad-exit=1 --wait=120 \
    --export=ALL,ARM_ROOT="$root",ARM_LR="$lr",ARM_STATE="$state",ARM_DELAY="$delay",MASTER_ADDR="$address",MASTER_PORT="$port",WORLD_SIZE=8 \
    bash -lc '
      set -euo pipefail
      sleep "$ARM_DELAY"
      source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
      cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
      export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
      export TRITON_CACHE_DIR="/tmp/e97-sft-sweep-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
      state_args=()
      if [[ "$ARM_STATE" != fresh ]]; then
        state_args=(--sft-parent-optimizer-split "$ARM_STATE" --resume-lr-override "$ARM_LR")
      fi
      "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
        --seed-checkpoint "$JOB_SEED" --seed-args-json "$JOB_SEED_ARGS" \
        --max-steps "$MAX_STEPS" --batch-size 1 --chunk-size 4096 \
        --gradient-checkpointing --loss-chunk-size 2048 --checkpoint-loss-chunks \
        --checkpoint-interval 16 --projection-chunk-size 2048 --moe-token-chunk-size 4096 \
        --lr "$ARM_LR" --weight-decay 0.01 --diloco-k 1 --expert-backend rocblas \
        --empty-cache-interval 1 --offload-schedulefree-state \
        --sft-authority-root "$AUTHORITY_ROOT" --sft-authority-manifest-sha256 "$AUTHORITY_MANIFEST_SHA256" \
        --sft-pack-root "$PACK_ROOT" --sft-pack-manifest-sha256 "$PACK_MANIFEST_SHA256" \
        --sft-sampler-key 42 --sft-parent-root "$PARENT_ROOT" \
        --sft-parent-manifest-sha256 "$PARENT_MANIFEST_SHA256" \
        --sft-validation-batches 0 --sft-validation-exhaustive \
        "${state_args[@]}" --log-jsonl "$ARM_ROOT/logs/training.jsonl"
    '
}
pids=()
for index in "${!arms[@]}"; do
  launch_arm "$index" "${arms[$index]}" "${lrs[$index]}" "${states[$index]}" "${hosts[$index]}" &
  pids+=("$!")
done
rc=0
for index in "${!pids[@]}"; do
  set +e; wait "${pids[$index]}"; arm_rc=$?; set -e
  printf '%s\n' "$arm_rc" > "$CAMPAIGN_ROOT/terminal/${arms[$index]}-rc.txt"
  (( arm_rc == 0 )) || rc=1
done
exit "$rc"
