#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-dense-agent
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_dense_agent/%x-%j.out
#SBATCH -e logs/frontier/e97_dense_agent/%x-%j.err
set -euo pipefail
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"; STEPS=${STEPS:-8}; LR=${LR:-0.00001}; REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
AUTH=${AUTH:-/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-v1}; AUTH_SHA=${AUTH_SHA:-01382bea60c02c09b62250a17f7865cce26c2994d817522674eee252eda8a065}; PACK=${PACK:-$AUTH/packs}; PACK_SHA=${PACK_SHA:-cfb6c4ccdf3079752d3c02c6e8eb1f5f8bf87b7a9f8cce528fa45f1c255c21e6}
SEED=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt; ARGS=$(dirname "$SEED")/args.json; SEED_SHA=e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857
record=$(scontrol show job "$SLURM_JOB_ID" -o); [[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
[[ $(sha256sum "$SEED"|awk '{print $1}') == "$SEED_SHA" && $(sha256sum "$AUTH/manifest.json"|awk '{print $1}') == "$AUTH_SHA" && $(sha256sum "$PACK/manifest.json"|awk '{print $1}') == "$PACK_SHA" ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,checkpoints,sources,terminal} logs/frontier/e97_dense_agent; printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
SNAP="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAP"; git archive "$SOURCE_COMMIT"|tar -xf - -C "$SNAP"; source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null; JS=/tmp/e97-dense-agent-$SLURM_JOB_ID.pt; JA=/tmp/e97-dense-agent-$SLURM_JOB_ID.json; sbcast -f "$SEED" "$JS"; sbcast -f "$ARGS" "$JA"
export MASTER_PORT=$((41000+SLURM_JOB_ID%1000)) WORLD_SIZE=8 OMP_NUM_THREADS=7 TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
set +e; srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
 set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null; cd "'$SNAP'"; export PYTHONPATH="'$SNAP'" MASTER_ADDR=$(getent ahostsv4 "$(hostname)"|awk "NR==1{print \$1}") RANK=$SLURM_PROCID LOCAL_RANK=0 TRITON_CACHE_DIR=/tmp/e97-dense-agent-${SLURM_JOB_ID}-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"; "$EMENDER_PYTHON" scripts/train_e97_dense_agent_sft.py --source-checkpoint "'$JS'" --source-args-json "'$JA'" --authority-root "'$AUTH'" --authority-sha256 "'$AUTH_SHA'" --pack-root "'$PACK'" --pack-sha256 "'$PACK_SHA'" --output-root "'$RUN_ROOT'/checkpoints" --steps "'$STEPS'" --lr "'$LR'" --log-jsonl "'$RUN_ROOT'/logs/training.jsonl"
'; rc=$?; set -e; printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P > "$RUN_ROOT/terminal/sacct-$SLURM_JOB_ID.txt" || true; exit "$rc"
