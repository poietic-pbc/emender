#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-overfit-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 3
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_moe/%x-%j.out
#SBATCH -e logs/frontier/e97_moe/%x-%j.err
set -euo pipefail
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"; : "${TRAIN_ROOT:?}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}; AUTH=/lustre/orion/bif148/proj-shared/emender/sft/overfit-probe-v1; PANEL=$AUTH/overfit_panel.pt; PANEL_SHA=a2b5a319973815d1672f221c9ed1c7247671019d4dfbcc1cae190c8fba42e201
SEED=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt; ARGS=$(dirname "$SEED")/args.json; PARENT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-record-reset-1e4-4k-k8-8n-v1/checkpoints/step-02342760-tokens-0000283007226491
record=$(scontrol show job "$SLURM_JOB_ID" -o); [[ "$RUN_ROOT" == /* && "$TRAIN_ROOT" == /* && "$record" == *"NumNodes=3"* && "$record" == *"NumTasks=24"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
[[ $(sha256sum "$PANEL"|awk '{print $1}') == "$PANEL_SHA" ]] || exit 65
mapfile -t probes < <(find "$TRAIN_ROOT/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*'|sort); [[ ${#probes[@]} == 2 ]] || { printf 'expected two probe checkpoints, got %s\n' "${#probes[@]}" >&2; exit 66; }
labels=(parent probe-32u probe-64u); gens=("$PARENT" "${probes[0]}" "${probes[1]}"); for g in "${gens[@]}"; do [[ -f "$g/manifest.json" ]] || exit 65; done
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,results,sources,terminal} logs/frontier/e97_moe; printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; git archive "$SOURCE_COMMIT"|tar -xf - -C "$SNAPSHOT"; source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null; JS=/tmp/e97-overfit-eval-$SLURM_JOB_ID.pt; JA=/tmp/e97-overfit-eval-$SLURM_JOB_ID.json; sbcast -f "$SEED" "$JS"; sbcast -f "$ARGS" "$JA"; mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache OMP_NUM_THREADS=7 MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
pids=(); for i in {0..2}; do label=${labels[$i]}; gen=${gens[$i]}; port=$((38000+i*100+SLURM_JOB_ID%100)); srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --exclusive --nodelist="${nodes[$i]}" --kill-on-bad-exit=1 --wait=120 bash -lc '
 set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null; cd "'$SNAPSHOT'"; export PYTHONPATH="'$SNAPSHOT'" MASTER_ADDR=$(getent ahostsv4 "'${nodes[$i]}'"|awk "NR==1{print \$1}") MASTER_PORT="'$port'" WORLD_SIZE=8 RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID TRITON_CACHE_DIR=/tmp/e97-overfit-eval-${SLURM_JOB_ID}-'$label'-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"; "$EMENDER_PYTHON" scripts/eval_e97_moe_paired.py --seed-checkpoint "'$JS'" --seed-args-json "'$JA'" --generation "'$gen'" --panel "'$PANEL'" --output "'$RUN_ROOT'/results/'$label'.json" --label "'$label'" --generation-tokens 256 --expert-backend rocblas
 ' > "$RUN_ROOT/logs/$label.out" 2> "$RUN_ROOT/logs/$label.err" & pids+=("$!"); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; if ((rc==0)); then for label in "${labels[@]}"; do "$EMENDER_PYTHON" "$SNAPSHOT/scripts/score_e97_sft_overfit_probe.py" --panel "$PANEL" --evaluation "$RUN_ROOT/results/$label.json" --output "$RUN_ROOT/results/$label-score.json"; done; fi
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P > "$RUN_ROOT/terminal/sacct-$SLURM_JOB_ID.txt" || true; printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; exit "$rc"
