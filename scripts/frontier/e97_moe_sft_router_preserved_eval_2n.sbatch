#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-sft-router-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
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
: "${RUN_ROOT:?absolute evaluation run root required}"
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
PANEL=${PANEL:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/panel/e97_moe_paired_v2.pt}
PANEL_SHA256=${PANEL_SHA256:-fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f}
PARENT=${PARENT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
PARENT_MANIFEST_SHA256=${PARENT_MANIFEST_SHA256:-95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1}
CANDIDATE=${CANDIDATE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-router-preserved-1e4-64u-canonical-v1/checkpoints/step-02338600-tokens-0000282071877429}
CANDIDATE_MANIFEST_SHA256=${CANDIDATE_MANIFEST_SHA256:-f190f0bcf735c07dc7f59ec27ac98fc8c0257718d8a331fc62ab6c529314814d}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[[ "$RUN_ROOT" == /* ]] || exit 64
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=2"* && "$record" == *"NumTasks=16"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* && "$record" == *"Requeue=0"* ]] || exit 65
[[ -f "$SEED_CHECKPOINT" && -f "$SEED_ARGS_JSON" \
   && $(sha256sum "$PANEL" | awk '{print $1}') == "$PANEL_SHA256" \
   && $(sha256sum "$PARENT/manifest.json" | awk '{print $1}') == "$PARENT_MANIFEST_SHA256" \
   && $(sha256sum "$CANDIDATE/manifest.json" | awk '{print $1}') == "$CANDIDATE_MANIFEST_SHA256" ]] || exit 66
mkdir -p "$RUN_ROOT"/{identity,results,sources,terminal} logs/frontier/e97_moe
[[ ! -e "$RUN_ROOT/results/comparison.json" ]] || exit 66
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-${SLURM_JOB_ID}.txt"
printf '%s  %s\n' "$PANEL_SHA256" "$PANEL" > "$RUN_ROOT/identity/panel.sha256"
cp "$PARENT/manifest.json" "$RUN_ROOT/identity/parent-manifest.json"
cp "$CANDIDATE/manifest.json" "$RUN_ROOT/identity/candidate-manifest.json"
printf '%s\n' 'schema=emender-e97-router-preserved-sft-eval-v1' \
  'checkpoints=282b-parent,router-preserved-1e-4-64u' \
  'generation=native-fp32-recurrent-cache' 'generation_tokens=256' \
  'stop_tokens=218,50256' 'expert_backend=rocblas' \
  "panel_sha256=$PANEL_SHA256" "parent_manifest_sha256=$PARENT_MANIFEST_SHA256" \
  "candidate_manifest_sha256=$CANDIDATE_MANIFEST_SHA256" \
  > "$RUN_ROOT/identity/recipe.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/scripts/eval_e97_moe_paired.py" ]]; then
  temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"; rm -rf "$temporary"; mkdir -p "$temporary"
  git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"; mv -T "$temporary" "$SNAPSHOT"
fi
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
JOB_SEED="/tmp/emender-e97-router-eval-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-router-eval-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"; sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); [[ ${#nodes[@]} == 2 ]] || exit 65
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
run_one() {
  local node=$1 label=$2 generation=$3 output=$4 port=$5
  srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --exclusive --nodelist="$node" \
    --kill-on-bad-exit=1 --wait=120 bash -lc '
      set -euo pipefail
      source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
      cd "'$SNAPSHOT'"; export PYTHONPATH="'$SNAPSHOT'${PYTHONPATH:+:$PYTHONPATH}"
      export MASTER_ADDR=$(getent ahostsv4 "'$node'" | awk "NR==1 {print \$1}")
      export MASTER_PORT="'$port'" WORLD_SIZE=8 RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
      export TRITON_CACHE_DIR="/tmp/e97-router-eval-${SLURM_JOB_ID}-'$label'-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
      "$EMENDER_PYTHON" scripts/eval_e97_moe_paired.py \
        --seed-checkpoint "'$JOB_SEED'" --seed-args-json "'$JOB_SEED_ARGS'" \
        --generation "'$generation'" --panel "'$PANEL'" --output "'$output'" \
        --label "'$label'" --generation-tokens 256 --expert-backend rocblas
    '
}
set +e
run_one "${nodes[0]}" parent-282b "$PARENT" "$RUN_ROOT/results/parent-282b.json" $((38000 + SLURM_JOB_ID % 1000)) & p0=$!
run_one "${nodes[1]}" router-preserved-1e-4-64u "$CANDIDATE" "$RUN_ROOT/results/candidate.json" $((39000 + SLURM_JOB_ID % 1000)) & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?
set -e
printf 'parent=%s\ncandidate=%s\n' "$r0" "$r1" > "$RUN_ROOT/terminal/runner-rc.txt"
(( r0 == 0 && r1 == 0 )) || exit 1
cd "$SNAPSHOT"
"$EMENDER_PYTHON" scripts/compare_e97_moe_sft_router_preserved.py \
  --parent "$RUN_ROOT/results/parent-282b.json" --candidate "$RUN_ROOT/results/candidate.json" \
  --output-json "$RUN_ROOT/results/comparison.json" --output-md "$RUN_ROOT/results/comparison.md"
printf '0\n' > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
