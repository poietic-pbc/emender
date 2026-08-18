#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-sft-reset-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 5
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err
set -euo pipefail
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}; PANEL=/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/panel/e97_moe_paired_v2.pt; PANEL_SHA=fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f
SEED=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt; ARGS=$(dirname "$SEED")/args.json
OLD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-router-preserved-1e4-4k-k64-8n-v1/checkpoints/step-02342632-tokens-0000282978786792
BASE=/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-record-reset-1e4-4k-k8-8n-v1/checkpoints
labels=(continuous-4096u reset-16u reset-32u reset-48u reset-64u)
generations=("$OLD" "$BASE/step-02342648-tokens-0000282982329581" "$BASE/step-02342664-tokens-0000282985870875" "$BASE/step-02342680-tokens-0000282989463482" "$BASE/step-02342696-tokens-0000282993010534")
hashes=(c2eebf519580e29a2111e3925a1d1bcbac3058c32f1dad5522b92906ccc36a9c 6e19ee4f89ebbb93f4d3b88061e095621cb60030f48f50b416bcd67807b22cd9 75dc32b62eba4c0ab698ef24eac5c0e09f70753296d0b3ce8a798601a80d299f 833fa2658f5d7ad29fc63912a0ead283e54b7716a02ec451701dd0b7c778c924 0090c645389cc2d487b4fd26f473e1024d38575b6802c75c93722761ce61fc10)
record=$(scontrol show job "$SLURM_JOB_ID" -o); [[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=5"* && "$record" == *"NumTasks=40"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
[[ $(sha256sum "$PANEL"|awk '{print $1}') == "$PANEL_SHA" ]] || exit 65
for i in {0..4}; do [[ $(sha256sum "${generations[$i]}/manifest.json"|awk '{print $1}') == "${hashes[$i]}" ]] || exit 65; done
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,results,sources,terminal} logs/frontier/e97_moe
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null; JOB_SEED=/tmp/e97-reset-eval-$SLURM_JOB_ID.pt; JOB_ARGS=/tmp/e97-reset-eval-$SLURM_JOB_ID.json; sbcast -f "$SEED" "$JOB_SEED"; sbcast -f "$ARGS" "$JOB_ARGS"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); [[ ${#nodes[@]} == 5 ]] || exit 66
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache OMP_NUM_THREADS=7 MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
pids=(); for i in {0..4}; do label=${labels[$i]}; generation=${generations[$i]}; port=$((37000 + i*100 + SLURM_JOB_ID%100)); srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --exclusive --nodelist="${nodes[$i]}" --kill-on-bad-exit=1 --wait=120 bash -lc '
 set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null; cd "'$SNAPSHOT'"; export PYTHONPATH="'$SNAPSHOT'"; export MASTER_ADDR=$(getent ahostsv4 "'${nodes[$i]}'"|awk "NR==1{print \$1}") MASTER_PORT="'$port'" WORLD_SIZE=8 RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID; export TRITON_CACHE_DIR=/tmp/e97-reset-eval-${SLURM_JOB_ID}-'$label'-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"; "$EMENDER_PYTHON" scripts/eval_e97_moe_paired.py --seed-checkpoint "'$JOB_SEED'" --seed-args-json "'$JOB_ARGS'" --generation "'$generation'" --panel "'$PANEL'" --output "'$RUN_ROOT'/results/'$label'.json" --label "'$label'" --generation-tokens 256 --expert-backend rocblas
 ' > "$RUN_ROOT/logs/$label.out" 2> "$RUN_ROOT/logs/$label.err" & pids+=("$!"); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if ((rc==0)); then for label in reset-16u reset-32u reset-48u reset-64u; do "$EMENDER_PYTHON" "$SNAPSHOT/scripts/compare_e97_moe_sft_router_preserved.py" --parent "$RUN_ROOT/results/continuous-4096u.json" --candidate "$RUN_ROOT/results/$label.json" --output-json "$RUN_ROOT/results/comparison-$label.json" --output-md "$RUN_ROOT/results/comparison-$label.md"; done; fi
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; exit "$rc"
