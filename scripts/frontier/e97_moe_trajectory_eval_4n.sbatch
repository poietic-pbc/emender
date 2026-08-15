#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-moe-trajectory-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 4
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
PANEL=${PANEL:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/panel/e97_moe_paired_v2.pt}
PANEL_SHA256=${PANEL_SHA256:-fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f}
BASE_250B=${BASE_250B:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/scale-ladder-mainfc52/milestones/step-02338080-tokens-0000250797359104}
LONG_282B=${LONG_282B:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728}
INSTRUCTION_300B=${INSTRUCTION_300B:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/instruction-32k-lr1007e-3-k8-v1/milestones/step-02338816-tokens-0000300860571648}
INSTRUCTION_304B=${INSTRUCTION_304B:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/instruction-32k-lr1007e-3-k8-v1/milestones/step-02338872-tokens-0000304618668032}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
SEED_ARGS_JSON=${SEED_ARGS_JSON:-$(dirname "$SEED_CHECKPOINT")/args.json}
RUN_ROOT=${RUN_ROOT:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/runs/job-${SLURM_JOB_ID}}

cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$record"
[[ "$record" == *"NumNodes=4"* && "$record" == *"NumTasks=32"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* \
   && "$record" == *"TimeLimit=02:00:00"* && "$record" == *"Requeue=0"* ]] || {
  echo "live scheduler binding mismatch" >&2; exit 65; }
for path in "$PANEL" "$BASE_250B/manifest.json" "$LONG_282B/manifest.json" \
  "$INSTRUCTION_300B/manifest.json" "$INSTRUCTION_304B/manifest.json" \
  "$SEED_CHECKPOINT" "$SEED_ARGS_JSON"; do
  [[ -f "$path" ]] || { echo "missing evaluation input: $path" >&2; exit 66; }
done
[[ $(sha256sum "$PANEL" | awk '{print $1}') == "$PANEL_SHA256" ]] || {
  echo "panel digest mismatch" >&2; exit 66; }

mkdir -p "$RUN_ROOT"/{identity,results,sources,terminal} logs/frontier/e97_moe
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-commit.txt"
printf '%s  %s\n' "$PANEL_SHA256" "$PANEL" > "$RUN_ROOT/identity/panel.sha256"
for pair in "base-250b:$BASE_250B" "long-282b:$LONG_282B" \
            "instruction-300b:$INSTRUCTION_300B" "instruction-304b:$INSTRUCTION_304B"; do
  label=${pair%%:*}; path=${pair#*:}
  cp "$path/manifest.json" "$RUN_ROOT/identity/${label}-manifest.json"
done

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
[[ ${#NODES[@]} == 4 ]] || { echo "expected exactly four nodes" >&2; exit 65; }
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
      export TRITON_CACHE_DIR="/tmp/e97-moe-trajectory-${SLURM_JOB_ID}-'$label'-${SLURM_PROCID}"
      rm -rf "$TRITON_CACHE_DIR"
      "$EMENDER_PYTHON" scripts/eval_e97_moe_paired.py \
        --seed-checkpoint "'$JOB_SEED'" --seed-args-json "'$JOB_SEED_ARGS'" \
        --generation "'$generation'" --panel "'$PANEL'" \
        --output "'$output'" --label "'$label'" --expert-backend rocblas
    '
}

set +e
run_one "${NODES[0]}" base-250b "$BASE_250B" "$RUN_ROOT/results/base-250b.json" "$((34000 + SLURM_JOB_ID % 1000))" & p0=$!
run_one "${NODES[1]}" long-282b "$LONG_282B" "$RUN_ROOT/results/long-282b.json" "$((35000 + SLURM_JOB_ID % 1000))" & p1=$!
run_one "${NODES[2]}" instruction-300b "$INSTRUCTION_300B" "$RUN_ROOT/results/instruction-300b.json" "$((36000 + SLURM_JOB_ID % 1000))" & p2=$!
run_one "${NODES[3]}" instruction-304b "$INSTRUCTION_304B" "$RUN_ROOT/results/instruction-304b.json" "$((37000 + SLURM_JOB_ID % 1000))" & p3=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?
set -e
printf 'base-250b=%s\nlong-282b=%s\ninstruction-300b=%s\ninstruction-304b=%s\n' \
  "$r0" "$r1" "$r2" "$r3" > "$RUN_ROOT/terminal/runner-rc.txt"
[[ $r0 == 0 && $r1 == 0 && $r2 == 0 && $r3 == 0 ]] || exit 1

cd "$SNAPSHOT"
"$EMENDER_PYTHON" scripts/compare_e97_moe_trajectory.py \
  --base-250b "$RUN_ROOT/results/base-250b.json" \
  --long-282b "$RUN_ROOT/results/long-282b.json" \
  --instruction-300b "$RUN_ROOT/results/instruction-300b.json" \
  --instruction-304b "$RUN_ROOT/results/instruction-304b.json" \
  --output-json "$RUN_ROOT/results/trajectory.json" \
  --output-md "$RUN_ROOT/results/trajectory.md"
printf '0\n' > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
