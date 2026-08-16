#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-sft-router-preserved
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

# Canonical one-node promotion checkpoint for the selected masked-SFT recipe:
# mature router ScheduleFree state, fresh nonrouter state, all parameters
# trainable, 64 updates at 1e-4. Fixed-world ADR-003; no elastic behavior.
set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${RUN_ROOT:?stable absolute run root required}"
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
MAX_STEPS=64
LR=0.0001
SFT_SAMPLER_KEY=42
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/authority}
AUTHORITY_MANIFEST_SHA256=${AUTHORITY_MANIFEST_SHA256:-e2461a28e7699637e282564589aeea35a150dc9e6e3cd62ffd6a2b5c0a2f47d8}
PACK_ROOT=${PACK_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/packs-4096-v1}
PACK_MANIFEST_SHA256=${PACK_MANIFEST_SHA256:-010dc5f326f351463d8af675f2d6a906c8f056e48ee4c93b1ee34dddd1b202ea}
PARENT_ROOT=${PARENT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
PARENT_MANIFEST_SHA256=${PARENT_MANIFEST_SHA256:-95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}

[[ "$RUN_ROOT" == /* ]] || exit 64
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* && "$record" == *"Requeue=0"* ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$RUN_ROOT"/{identity,logs,sources,checkpoints,terminal} logs/frontier/e97_moe
[[ ! -e "$RUN_ROOT/checkpoints/latest" && ! -s "$RUN_ROOT/logs/training.jsonl" ]] || exit 66
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-${SLURM_JOB_ID}.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
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
  'schema=emender-e97-masked-sft-router-preserved-canonical-v1' \
  'optimizer_state=router-preserved-nonrouter-fresh' 'all_parameters_trainable=true' \
  'lr=0.0001' 'weight_decay=0.01' 'context_size=4096' 'batch_size_per_rank=1' \
  'nodes=1' 'world_size=8' 'max_steps=64' 'diloco=none-screening-only-k1' \
  'sampler=emender-record-pack-counter-v1' 'sampler_key=42' \
  'validation=all-1777-packs-4352510-targets-exactly-once' \
  "authority_manifest_sha256=$AUTHORITY_MANIFEST_SHA256" \
  "pack_manifest_sha256=$PACK_MANIFEST_SHA256" \
  "parent_manifest_sha256=$PARENT_MANIFEST_SHA256" \
  > "$RUN_ROOT/identity/recipe-${SLURM_JOB_ID}.txt"

JOB_SEED="/tmp/emender-e97-sft-canonical-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-sft-canonical-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"
sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 10000)) WORLD_SIZE=8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7} MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
export SNAPSHOT JOB_SEED JOB_SEED_ARGS RUN_ROOT AUTHORITY_ROOT AUTHORITY_MANIFEST_SHA256 PACK_ROOT PACK_MANIFEST_SHA256
export PARENT_ROOT PARENT_MANIFEST_SHA256 SFT_SAMPLER_KEY LR SOURCE_COMMIT
set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-sft-canonical-${SLURM_JOB_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py \
    --seed-checkpoint "$JOB_SEED" --seed-args-json "$JOB_SEED_ARGS" \
    --max-steps 64 --batch-size 1 --chunk-size 4096 \
    --gradient-checkpointing --loss-chunk-size 2048 --checkpoint-loss-chunks \
    --checkpoint-interval 16 --projection-chunk-size 2048 --moe-token-chunk-size 4096 \
    --lr "$LR" --weight-decay 0.01 --diloco-k 1 --expert-backend rocblas \
    --empty-cache-interval 1 --offload-schedulefree-state \
    --checkpoint-root "$RUN_ROOT/checkpoints" --save-every 64 --keep-checkpoints 2 \
    --sft-authority-root "$AUTHORITY_ROOT" --sft-authority-manifest-sha256 "$AUTHORITY_MANIFEST_SHA256" \
    --sft-pack-root "$PACK_ROOT" --sft-pack-manifest-sha256 "$PACK_MANIFEST_SHA256" \
    --sft-sampler-key "$SFT_SAMPLER_KEY" --sft-parent-root "$PARENT_ROOT" \
    --sft-parent-manifest-sha256 "$PARENT_MANIFEST_SHA256" \
    --sft-parent-optimizer-split router-preserved --resume-lr-override "$LR" \
    --sft-validation-batches 0 --sft-validation-exhaustive \
    --log-jsonl "$RUN_ROOT/logs/training.jsonl"
'
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
if (( rc == 0 )); then
  generation=$(readlink -f "$RUN_ROOT/checkpoints/latest")
  manifest="$generation/manifest.json"
  jq -e --arg source "$SOURCE_COMMIT" --arg parent "$PARENT_MANIFEST_SHA256" '
    .complete == true and .step == 2338600 and .source_commit == $source
    and (.ranks | length) == 8
    and .sampler.absolute_rank_sample_index == 64
    and .sampler.identity.schema == "emender-record-pack-counter-v1"
    and .sampler.identity.data_world_size == 8
    and .sampler.parent.manifest_sha256 == $parent
    and .sampler_transition.optimizer_state == "router-preserved"
    and (.optimizer_groups | length) == 2
  ' "$manifest" >/dev/null || rc=68
  if (( rc == 0 )); then
    while IFS=$'\t' read -r file bytes; do
      [[ -f "$generation/$file" && $(stat -c %s "$generation/$file") == "$bytes" ]] || { rc=68; break; }
    done < <(jq -r '.ranks[] | [.local, .replicated] | .[] | [.file, (.bytes|tostring)] | @tsv' "$manifest")
  fi
  if (( rc == 0 )); then
    sha256sum "$manifest" > "$RUN_ROOT/identity/canonical-manifest-${SLURM_JOB_ID}.sha256"
    printf '%s\n' "$generation" > "$RUN_ROOT/identity/canonical-generation-${SLURM_JOB_ID}.txt"
  fi
  printf '%s\n' "$rc" > "$RUN_ROOT/terminal/publication-${SLURM_JOB_ID}-rc.txt"
fi
exit "$rc"
