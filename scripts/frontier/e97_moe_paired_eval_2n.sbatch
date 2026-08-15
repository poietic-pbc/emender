#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-paired-eval
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
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
EXPECTED_PARTITION=${EXPECTED_PARTITION:-batch}
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
PANEL=${PANEL:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v1/panel/e97_moe_paired_v1.pt}
PANEL_SHA256=${PANEL_SHA256:-d00a54107fe6b863232c335171688f2d5aa8250907453eda69ca6c90cb304445}
BEFORE=${BEFORE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
AFTER=${AFTER:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/instruction-32k-lr1007e-3-k8-v1/milestones/step-02338872-tokens-0000304618668032}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
RUN_ROOT=${RUN_ROOT:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v1/runs/job-${SLURM_JOB_ID}}

cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$record"
[[ "$record" == *"NumNodes=2"* && "$record" == *"NumTasks=16"* \
   && "$record" == *"Partition=$EXPECTED_PARTITION"* \
   && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* \
   && "$record" == *"Requeue=0"* ]] || { echo "live scheduler binding mismatch" >&2; exit 65; }
[[ -f "$PANEL" && -f "$BEFORE/manifest.json" && -f "$AFTER/manifest.json" \
   && -f "$SEED_CHECKPOINT" && -f "$SEED_ARGS_JSON" ]] || { echo "evaluation input missing" >&2; exit 66; }
[[ $(sha256sum "$PANEL" | awk '{print $1}') == "$PANEL_SHA256" ]] || { echo "panel digest mismatch" >&2; exit 66; }

mkdir -p "$RUN_ROOT"/{identity,results,sources,terminal} logs/frontier/e97_moe
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-commit.txt"
printf '%s  %s\n' "$PANEL_SHA256" "$PANEL" > "$RUN_ROOT/identity/panel.sha256"
cp "$BEFORE/manifest.json" "$RUN_ROOT/identity/before-manifest.json"
cp "$AFTER/manifest.json" "$RUN_ROOT/identity/after-manifest.json"

SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"
mkdir -p "$temporary"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"
mv -T "$temporary" "$SNAPSHOT"

source "$REPO/scripts/frontier/activate_emender_frontier.sh"
JOB_SEED="/tmp/emender-e97-eval-seed-${SLURM_JOB_ID}.pt"
JOB_SEED_ARGS="/tmp/emender-e97-eval-seed-args-${SLURM_JOB_ID}.json"
sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"
sbcast -f "$SEED_ARGS_JSON" "$JOB_SEED_ARGS"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} == 2 ]] || { echo "expected exactly two nodes" >&2; exit 65; }
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}

run_one() {
  local node=$1 label=$2 generation=$3 output=$4 port=$5
  srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --exclusive --nodelist="$node" \
    --kill-on-bad-exit=1 --wait=120 bash -lc '
      set -euo pipefail
      source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
      cd "'$SNAPSHOT'"
      export PYTHONPATH="'$SNAPSHOT'${PYTHONPATH:+:$PYTHONPATH}"
      export MASTER_ADDR=$(getent ahostsv4 "'$node'" | awk "NR==1 {print \$1}")
      export MASTER_PORT="'$port'" WORLD_SIZE=8
      export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
      export TRITON_CACHE_DIR="/tmp/e97-moe-eval-${SLURM_JOB_ID}-'$label'-${SLURM_PROCID}"
      rm -rf "$TRITON_CACHE_DIR"
      "$EMENDER_PYTHON" scripts/eval_e97_moe_paired.py \
        --seed-checkpoint "'$JOB_SEED'" --seed-args-json "'$JOB_SEED_ARGS'" \
        --generation "'$generation'" --panel "'$PANEL'" \
        --output "'$output'" --label "'$label'" --expert-backend rocblas
    '
}

set +e
run_one "${NODES[0]}" before "$BEFORE" "$RUN_ROOT/results/before.json" "$((32000 + SLURM_JOB_ID % 1000))" &
pid_before=$!
run_one "${NODES[1]}" after "$AFTER" "$RUN_ROOT/results/after.json" "$((33000 + SLURM_JOB_ID % 1000))" &
pid_after=$!
wait "$pid_before"; rc_before=$?
wait "$pid_after"; rc_after=$?
set -e
printf 'before=%s\nafter=%s\n' "$rc_before" "$rc_after" > "$RUN_ROOT/terminal/runner-rc.txt"
[[ $rc_before == 0 && $rc_after == 0 ]] || exit 1

cd "$SNAPSHOT"
"$EMENDER_PYTHON" scripts/compare_e97_moe_paired_eval.py \
  --before "$RUN_ROOT/results/before.json" --after "$RUN_ROOT/results/after.json" \
  --output-json "$RUN_ROOT/results/comparison.json" \
  --output-md "$RUN_ROOT/results/comparison.md"
printf '0\n' > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
