#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-dense-early-hs
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
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
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}; DATA=/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v1/raw/hellaswag/data/validation-00000-of-00001.parquet
D0=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500
D1=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500
labels=(dense-step260500 dense-step383500); models=("$D0/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt" "$D1/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt"); args=("$D0/args.json" "$D1/args.json"); hashes=(46fe086e38b58026910886f3df265d8389bc2b56dc406591e4ea1177645dbfb0 b899489c77093164f92fc35f85c879204d8e1866068b320598a87cf87d99b1ca)
record=$(scontrol show job "$SLURM_JOB_ID" -o); [[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=2"* && "$record" == *"NumTasks=16"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
for i in 0 1; do [[ $(sha256sum "${models[$i]}"|awk '{print $1}') == "${hashes[$i]}" ]] || exit 65; done
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,results,sources,terminal} logs/frontier/eval
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; [[ -f "$SNAPSHOT/scripts/eval_full_hellaswag_shard.py" ]] || git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null; export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache OMP_NUM_THREADS=7
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); pids=()
for i in 0 1; do label=${labels[$i]}; model=${models[$i]}; arg=${args[$i]}; mkdir -p "$RUN_ROOT/results/$label/shards"; srun --exclusive -N1 -n8 --ntasks-per-node=8 -w "${hosts[$i]}" --label bash -lc '
 set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null; cd "'$SNAPSHOT'"; export PYTHONPATH="'$SNAPSHOT'" TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache; export TRITON_CACHE_DIR="/tmp/e97-early-hs-${SLURM_JOB_ID}-${SLURM_PROCID}"; rm -rf "$TRITON_CACHE_DIR"; "$EMENDER_PYTHON" scripts/eval_full_hellaswag_shard.py --backend e97 --model "'$model'" --args-json "'$arg'" --data "'$DATA'" --output "'$RUN_ROOT'/results/'$label'/shards/rank-${SLURM_LOCALID}.json" --rank "$SLURM_LOCALID" --world 8 --label "'$label'"
 ' > "$RUN_ROOT/logs/$label.out" 2> "$RUN_ROOT/logs/$label.err" & pids+=("$!"); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if ((rc==0)); then for label in "${labels[@]}"; do "$EMENDER_PYTHON" "$SNAPSHOT/scripts/aggregate_full_hellaswag.py" --shards "$RUN_ROOT/results/$label/shards" --output "$RUN_ROOT/results/$label.json"; done; fi
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; exit "$rc"
