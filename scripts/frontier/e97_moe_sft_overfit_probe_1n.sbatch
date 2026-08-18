#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-sft-overfit
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
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}; AUTH=/lustre/orion/bif148/proj-shared/emender/sft/overfit-probe-v1; PACK=$AUTH/packs
AUTH_SHA=692d3ea0ec879621509f170154896f71965aaa987a5e2351e83a62dba62676e8; PACK_SHA=7ec9f8e8d672e5d61117b1548e38d43edadde86782a8e10157888bc84ddbf411
PARENT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-record-reset-1e4-4k-k8-8n-v1/checkpoints/step-02342760-tokens-0000283007226491; PARENT_SHA=$(sha256sum "$PARENT/manifest.json"|awk '{print $1}')
SEED=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt; ARGS=$(dirname "$SEED")/args.json
record=$(scontrol show job "$SLURM_JOB_ID" -o); [[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
[[ $(sha256sum "$AUTH/manifest.json"|awk '{print $1}') == "$AUTH_SHA" && $(sha256sum "$PACK/manifest.json"|awk '{print $1}') == "$PACK_SHA" ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,sources,checkpoints,terminal} logs/frontier/e97_moe
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; git archive "$SOURCE_COMMIT"|tar -xf - -C "$SNAPSHOT"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null; JS=/tmp/e97-overfit-$SLURM_JOB_ID.pt; JA=/tmp/e97-overfit-$SLURM_JOB_ID.json; sbcast -f "$SEED" "$JS"; sbcast -f "$ARGS" "$JA"
export MASTER_PORT=$((39000+SLURM_JOB_ID%1000)) WORLD_SIZE=8 OMP_NUM_THREADS=7 TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache EMENDER_SOURCE_COMMIT=$SOURCE_COMMIT
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
 set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null; cd "'$SNAPSHOT'"; export MASTER_ADDR=$(getent ahostsv4 "$(hostname)"|awk "NR==1{print \$1}"); export PYTHONPATH="'$SNAPSHOT'" RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID TRITON_CACHE_DIR=/tmp/e97-overfit-${SLURM_JOB_ID}-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"; "$EMENDER_PYTHON" scripts/frontier/e97_35b_moe_train.py --seed-checkpoint "'$JS'" --seed-args-json "'$JA'" --max-steps 64 --batch-size 1 --chunk-size 4096 --gradient-checkpointing --loss-chunk-size 2048 --checkpoint-loss-chunks --checkpoint-interval 16 --projection-chunk-size 2048 --moe-token-chunk-size 4096 --lr 0.0001 --weight-decay 0.01 --diloco-k 1 --expert-backend rocblas --empty-cache-interval 1 --offload-schedulefree-state --checkpoint-root "'$RUN_ROOT'/checkpoints" --save-every 32 --keep-checkpoints 2 --sft-authority-root "'$AUTH'" --sft-authority-manifest-sha256 "'$AUTH_SHA'" --sft-pack-root "'$PACK'" --sft-pack-manifest-sha256 "'$PACK_SHA'" --sft-sampler-key 970035 --sft-parent-root "'$PARENT'" --sft-parent-manifest-sha256 "'$PARENT_SHA'" --sft-reset-state-between-records --log-jsonl "'$RUN_ROOT'/logs/training.jsonl"
'
