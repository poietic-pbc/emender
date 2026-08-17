#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-dense-513b-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/eval/%x-%j.out
#SBATCH -e logs/frontier/eval/%x-%j.err

set -euo pipefail
: "${SOURCE_COMMIT:?immutable source commit required}"
: "${RUN_ROOT:?absolute run root required}"
EXPECTED_QOS=${EXPECTED_QOS:-debug}
EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}
CHECKPOINT=${CHECKPOINT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857}
ARGS_JSON=${ARGS_JSON:-$(dirname "$CHECKPOINT")/args.json}
PANEL=${PANEL:-/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/panel/e97_moe_paired_v2.pt}
PANEL_SHA256=${PANEL_SHA256:-fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f}
WEIGHT_MODE=${WEIGHT_MODE:-train}
LABEL=${LABEL:-dense-513b}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[[ "$RUN_ROOT" == /* && "$WEIGHT_MODE" =~ ^(train|saved)$ \
   && "$LABEL" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 64
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* \
   && "$record" == *"Partition=batch"* && "$record" == *"QOS=$EXPECTED_QOS"* \
   && "$record" == *"TimeLimit=$EXPECTED_TIME_LIMIT"* && "$record" == *"Requeue=0"* ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"
mkdir -p "$RUN_ROOT"/{identity,logs,results,sources,terminal} logs/frontier/eval
[[ ! -e "$RUN_ROOT/results/$LABEL.json" ]] || exit 66
[[ -f "$CHECKPOINT" && -f "$ARGS_JSON" && -f "$PANEL" ]] || exit 66
[[ $(sha256sum "$CHECKPOINT"|awk '{print $1}') == "$CHECKPOINT_SHA256" \
   && $(sha256sum "$PANEL"|awk '{print $1}') == "$PANEL_SHA256" ]] || exit 66
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-${SLURM_JOB_ID}.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N|TimeLimit=%l' \
  | tee "$RUN_ROOT/identity/squeue-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/identity/source-${SLURM_JOB_ID}.txt"
printf '%s\n' "checkpoint=$CHECKPOINT" "checkpoint_sha256=$CHECKPOINT_SHA256" \
  "panel=$PANEL" "panel_sha256=$PANEL_SHA256" \
  "weight_mode=$WEIGHT_MODE" "label=$LABEL" > "$RUN_ROOT/identity/recipe.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/scripts/eval_e97_dense_paired.py" ]]; then
  temporary="$SNAPSHOT.tmp-${SLURM_JOB_ID}"; rm -rf "$temporary"; mkdir -p "$temporary"
  git archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"; mv -T "$temporary" "$SNAPSHOT"
fi
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 10000)) WORLD_SIZE=8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7} MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export SNAPSHOT CHECKPOINT CHECKPOINT_SHA256 ARGS_JSON PANEL PANEL_SHA256 RUN_ROOT WEIGHT_MODE LABEL
set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR="/tmp/e97-dense-eval-${SLURM_JOB_ID}-${SLURM_PROCID}"
  rm -rf "$TRITON_CACHE_DIR"
  "$EMENDER_PYTHON" scripts/eval_e97_dense_paired.py \
    --checkpoint "$CHECKPOINT" --args-json "$ARGS_JSON" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" --panel "$PANEL" \
    --panel-sha256 "$PANEL_SHA256" --output "$RUN_ROOT/results/$LABEL.json" \
    --label "$LABEL" --weight-mode "$WEIGHT_MODE" --generation-tokens 256
' > "$RUN_ROOT/logs/eval.out" 2> "$RUN_ROOT/logs/eval.err"
rc=$?; set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-${SLURM_JOB_ID}-rc.txt"
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,ExitCode -n -P \
  > "$RUN_ROOT/terminal/sacct-${SLURM_JOB_ID}.txt" || true
exit "$rc"
