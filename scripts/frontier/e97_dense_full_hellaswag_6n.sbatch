#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-dense-full-hs
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 6
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/eval/%x-%j.out
#SBATCH -e logs/frontier/eval/%x-%j.err
set -euo pipefail
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
DATA=/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v1/raw/hellaswag/data/validation-00000-of-00001.parquet
ARGS=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/args.json
BASE=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones
GPT=/lustre/orion/bif148/proj-shared/emender/external_models/openai-community-gpt2-xl-15ea56d
labels=(dense-300b dense-343b dense-353b dense-401b dense-513b gpt2-xl)
backends=(e97 e97 e97 e97 e97 hf)
models=(
 "$BASE/step-2309800-tokens-299607654400/checkpoint_step_2309800_loss_2.2665.pt"
 "$BASE/step-2312400-tokens-343228416000/checkpoint_step_2312400_loss_2.2961.pt"
 "$BASE/step-2313000-tokens-353294745600/checkpoint_step_2313000_loss_2.5014.pt"
 "$BASE/step-2315840-tokens-400942039040/checkpoint_step_2315840_loss_2.2928.pt"
 "$BASE/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt"
 "$GPT")
hashes=(d8a615fe2cc0f3c7d4f5943659cc5b9382248b7748af40ac212e78c0480ab0b3 1d8a6ea45661d5defe0371c8a284148809317c8271a4ab7ffbf3ace13bb8ef78 4802ed9f7f5414aa63c99cfdce66e2e8790745104898d0b9494505ab2031823e c1f1fcb98099e7d141b998dda810baaa76208d36e3f1cc749c1b6ddddf370895 e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857 0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a)
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=6"* && "$record" == *"NumTasks=48"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
[[ -f "$DATA" && -f "$ARGS" ]] || exit 65
for i in {0..4}; do [[ $(sha256sum "${models[$i]}"|awk '{print $1}') == "${hashes[$i]}" ]] || exit 65; done
[[ $(sha256sum "$GPT/model.safetensors"|awk '{print $1}') == "${hashes[5]}" ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,results,sources,terminal} logs/frontier/eval
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; [[ -f "$SNAPSHOT/scripts/eval_full_hellaswag_shard.py" ]] || git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=7
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); [[ ${#hosts[@]} == 6 ]] || exit 66
pids=()
for i in {0..5}; do
 label=${labels[$i]}; backend=${backends[$i]}; model=${models[$i]}; mkdir -p "$RUN_ROOT/results/$label/shards"
 srun --exclusive -N1 -n8 --ntasks-per-node=8 -w "${hosts[$i]}" --label bash -lc '
   set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null
   cd "'$SNAPSHOT'"; export PYTHONPATH="'$SNAPSHOT'" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
   export TRITON_CACHE_DIR="/tmp/e97-full-hs-${SLURM_JOB_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"
   args_json=(); [[ "'$backend'" != e97 ]] || args_json=(--args-json "'$ARGS'")
   "$EMENDER_PYTHON" scripts/eval_full_hellaswag_shard.py --backend "'$backend'" --model "'$model'" "${args_json[@]}" --data "'$DATA'" --output "'$RUN_ROOT'/results/'$label'/shards/rank-${SLURM_LOCALID}.json" --rank "$SLURM_LOCALID" --world 8 --label "'$label'"
 ' > "$RUN_ROOT/logs/$label.out" 2> "$RUN_ROOT/logs/$label.err" &
 pids+=("$!")
done
rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if (( rc==0 )); then
 for label in "${labels[@]}"; do "$EMENDER_PYTHON" "$SNAPSHOT/scripts/aggregate_full_hellaswag.py" --shards "$RUN_ROOT/results/$label/shards" --output "$RUN_ROOT/results/$label.json"; done
fi
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; exit "$rc"
