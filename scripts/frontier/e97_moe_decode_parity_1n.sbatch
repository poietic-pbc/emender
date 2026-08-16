#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-decode-parity
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

set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${RUN_ROOT:?absolute run root required}"
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
EXPERT_BACKEND=${EXPERT_BACKEND:-rocblas}
[[ "$EXPERT_BACKEND" == rocblas || "$EXPERT_BACKEND" == triton ]] || exit 64
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/authority}
AUTHORITY_MANIFEST_SHA256=${AUTHORITY_MANIFEST_SHA256:-e2461a28e7699637e282564589aeea35a150dc9e6e3cd62ffd6a2b5c0a2f47d8}
PACK_ROOT=${PACK_ROOT:-/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/packs-4096-v1}
PACK_MANIFEST_SHA256=${PACK_MANIFEST_SHA256:-010dc5f326f351463d8af675f2d6a906c8f056e48ee4c93b1ee34dddd1b202ea}
PANEL=${PANEL:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/panel/e97_moe_paired_v2.pt}
PANEL_SHA256=${PANEL_SHA256:-fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f}
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
mkdir -p "$RUN_ROOT"/{identity,logs,sources,terminal} logs/frontier/e97_moe
[[ ! -f "$RUN_ROOT/results/decode-parity.json" ]] || exit 66
mkdir -p "$RUN_ROOT/results"
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
   && $(sha256sum "$PANEL" | awk '{print $1}') == "$PANEL_SHA256" \
   && $(sha256sum "$PARENT_ROOT/manifest.json" | awk '{print $1}') == "$PARENT_MANIFEST_SHA256" ]] || exit 66
printf '%s\n' 'schema=emender-e97-moe-decode-parity-run-v1' \
  'teacher_forced_tokens=32' 'greedy_tokens=32' \
  "expert_backend=$EXPERT_BACKEND" \
  "panel_sha256=$PANEL_SHA256" \
  "parent_manifest_sha256=$PARENT_MANIFEST_SHA256" \
  "pack_manifest_sha256=$PACK_MANIFEST_SHA256" > "$RUN_ROOT/identity/recipe.txt"
JOB_SEED="/tmp/emender-e97-sft-exact-val-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-sft-exact-val-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"; sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 10000)) WORLD_SIZE=8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7} MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
export SNAPSHOT JOB_SEED JOB_SEED_ARGS RUN_ROOT PANEL PANEL_SHA256 PARENT_ROOT PARENT_MANIFEST_SHA256 EXPERT_BACKEND
set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-moe-decode-parity-${SLURM_JOB_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" -m pytest -q tests/test_e88_triton.py \
    -k "unaligned_inference_returns_prepadding_state or fp32_cache_is_chunk_boundary_invariant"
  "$EMENDER_PYTHON" scripts/eval_e97_moe_decode_parity.py \
    --seed-checkpoint "$JOB_SEED" --seed-args-json "$JOB_SEED_ARGS" \
    --generation "$PARENT_ROOT" --panel "$PANEL" --tokens 32 \
    --output "$RUN_ROOT/results/decode-parity.json" --expert-backend "$EXPERT_BACKEND"
'
rc=$?; set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
exit "$rc"
