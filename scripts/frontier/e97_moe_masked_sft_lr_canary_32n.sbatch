#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-sft-lr-canary
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 32
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err

# Matched 16+16-node masked-SFT LR canary. Each arm is an independent fixed
# world with node-local eight-GCD EP and corresponding-lane DiLoCo.
set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${CAMPAIGN_ROOT:?absolute campaign root required}"
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
MAX_STEPS=${MAX_STEPS:-512}
DILOCO_K=${DILOCO_K:-64}
VALIDATION_BATCHES=${VALIDATION_BATCHES:-8}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/authority}
AUTHORITY_MANIFEST_SHA256=${AUTHORITY_MANIFEST_SHA256:-e2461a28e7699637e282564589aeea35a150dc9e6e3cd62ffd6a2b5c0a2f47d8}
PACK_ROOT=${PACK_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/packs-4096-v1}
PACK_MANIFEST_SHA256=${PACK_MANIFEST_SHA256:-010dc5f326f351463d8af675f2d6a906c8f056e48ee4c93b1ee34dddd1b202ea}
PARENT_ROOT=${PARENT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
PARENT_MANIFEST_SHA256=${PARENT_MANIFEST_SHA256:-95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[[ "$CAMPAIGN_ROOT" == /* && "$MAX_STEPS" =~ ^[1-9][0-9]*$ \
   && "$DILOCO_K" == 64 && $((MAX_STEPS % DILOCO_K)) -eq 0 ]] || exit 64
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=32"* && "$record" == *"NumTasks=256"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* && "$record" == *"Requeue=0"* ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$CAMPAIGN_ROOT"/{identity,sources,terminal} logs/frontier/e97_moe
for arm in lr2e-6 lr5e-6; do
  mkdir -p "$CAMPAIGN_ROOT/$arm"/{logs,checkpoints,identity}
  [[ ! -e "$CAMPAIGN_ROOT/$arm/checkpoints/latest" ]] || {
    echo "canary arm already has a checkpoint authority" >&2; exit 66; }
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
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_MANIFEST_SHA256" \
   && $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_MANIFEST_SHA256" \
   && $(sha256sum "$PARENT_ROOT/manifest.json" | awk '{print $1}') == "$PARENT_MANIFEST_SHA256" ]] || exit 66
jq -e --arg a "$AUTHORITY_MANIFEST_SHA256" --arg p "$PACK_MANIFEST_SHA256" \
  '.status == "pass" and .authority_manifest_sha256 == $a and .pack_manifest_sha256 == $p' \
  "$PACK_ROOT/validation.json" >/dev/null
printf '%s\n' "schema=emender-e97-masked-sft-lr-canary-v1" \
  "arms=lr2e-6:2e-6,lr5e-6:5e-6" "nodes_per_arm=16" "context_size=4096" \
  "batch_size=1" "max_steps=$MAX_STEPS" "diloco_k=$DILOCO_K" \
  "authority_manifest_sha256=$AUTHORITY_MANIFEST_SHA256" \
  "pack_manifest_sha256=$PACK_MANIFEST_SHA256" \
  "parent_manifest_sha256=$PARENT_MANIFEST_SHA256" > "$CAMPAIGN_ROOT/identity/recipe.txt"
JOB_SEED="/tmp/emender-e97-sft-canary-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-sft-canary-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"; sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#hosts[@]} == 32 ]] || exit 65
nodes_a=$(IFS=,; echo "${hosts[*]:0:16}")
nodes_b=$(IFS=,; echo "${hosts[*]:16:16}")
addr_a=$(getent ahostsv4 "${hosts[0]}" | awk 'NR==1 {print $1}')
addr_b=$(getent ahostsv4 "${hosts[16]}" | awk 'NR==1 {print $1}')
port_a=$((36000 + SLURM_JOB_ID % 1000)); port_b=$((37000 + SLURM_JOB_ID % 1000))
export SNAPSHOT JOB_SEED JOB_SEED_ARGS AUTHORITY_ROOT AUTHORITY_MANIFEST_SHA256 PACK_ROOT PACK_MANIFEST_SHA256
export PARENT_ROOT PARENT_MANIFEST_SHA256 MAX_STEPS DILOCO_K VALIDATION_BATCHES SOURCE_COMMIT
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}

launch_arm() {
  local arm=$1 lr=$2 delay=$3 nodes=$4 address=$5 port=$6
  local root="$CAMPAIGN_ROOT/$arm"
  srun --exclusive --exact --nodes=16 --ntasks=128 --nodelist="$nodes" \
    --label --kill-on-bad-exit=1 --wait=120 \
    --export=ALL,ARM_ROOT="$root",ARM_LR="$lr",ARM_DELAY="$delay",MASTER_ADDR="$address",MASTER_PORT="$port",WORLD_SIZE=128 \
    bash -lc '
      set -euo pipefail
      source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
      cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
      export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
      export TRITON_CACHE_DIR="/tmp/e97-sft-canary-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
      "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
        --seed-checkpoint "$JOB_SEED" --seed-args-json "$JOB_SEED_ARGS" \
        --max-steps "$MAX_STEPS" --batch-size 1 --chunk-size 4096 \
        --gradient-checkpointing --loss-chunk-size 2048 --checkpoint-loss-chunks \
        --checkpoint-interval 16 --projection-chunk-size 2048 --moe-token-chunk-size 4096 \
        --lr "$ARM_LR" --weight-decay 0.0 --diloco-k "$DILOCO_K" --expert-backend rocblas \
        --empty-cache-interval 1 --offload-schedulefree-state \
        --checkpoint-root "$ARM_ROOT/checkpoints" --save-every 0 --keep-checkpoints 1 \
        --sft-authority-root "$AUTHORITY_ROOT" --sft-authority-manifest-sha256 "$AUTHORITY_MANIFEST_SHA256" \
        --sft-pack-root "$PACK_ROOT" --sft-pack-manifest-sha256 "$PACK_MANIFEST_SHA256" \
        --sft-sampler-key 42 --sft-parent-root "$PARENT_ROOT" \
        --sft-parent-manifest-sha256 "$PARENT_MANIFEST_SHA256" \
        --sft-validation-batches "$VALIDATION_BATCHES" \
        --final-checkpoint-delay-seconds "$ARM_DELAY" \
        --log-jsonl "$ARM_ROOT/logs/training.jsonl"
    '
}
set +e
launch_arm lr2e-6 0.000002 0 "$nodes_a" "$addr_a" "$port_a" & pid_a=$!
launch_arm lr5e-6 0.000005 120 "$nodes_b" "$addr_b" "$port_b" & pid_b=$!
wait "$pid_a"; rc_a=$?; wait "$pid_b"; rc_b=$?
set -e
printf '%s\n' "$rc_a" > "$CAMPAIGN_ROOT/terminal/lr2e-6-rc.txt"
printf '%s\n' "$rc_b" > "$CAMPAIGN_ROOT/terminal/lr5e-6-rc.txt"
(( rc_a == 0 && rc_b == 0 )) || exit 1
