#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-agent-eval
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_dense_agent/%x-%j.out
#SBATCH -e logs/frontier/e97_dense_agent/%x-%j.err
set -euo pipefail
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"; REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}; CKPT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/train-128u-lr1e5-v1/checkpoints/checkpoint_agent_sft_u000128.pt; CKPT_SHA=1864f1c9138b17e99da698f3272017c7962da676f0a21d68a42ad98868bbbced; ARGS=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/args.json
record=$(scontrol show job "$SLURM_JOB_ID" -o); [[ "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* && $(sha256sum "$CKPT"|awk '{print $1}') == "$CKPT_SHA" ]] || exit 64
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,logs,results,shards,sources,terminal} logs/frontier/e97_dense_agent; printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"; SNAP="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAP"; git archive "$SOURCE_COMMIT"|tar -xf - -C "$SNAP"; source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null; JC=/tmp/e97-agent-eval-$SLURM_JOB_ID.pt; JA=/tmp/e97-agent-eval-$SLURM_JOB_ID.json; sbcast -f "$CKPT" "$JC"; sbcast -f "$ARGS" "$JA"
set +e; srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
 set -euo pipefail; source "'$REPO'/scripts/frontier/activate_emender_frontier.sh" >/dev/null; cd "'$SNAP'"; export PYTHONPATH="'$SNAP'" TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache TRITON_CACHE_DIR=/tmp/e97-agent-eval-${SLURM_JOB_ID}-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"; "$EMENDER_PYTHON" scripts/eval_e97_dense_agent.py --checkpoint "'$JC'" --args-json "'$JA'" --output "'$RUN_ROOT'/shards/shard-${SLURM_PROCID}.json" --rank "$SLURM_PROCID" --world 8
'; rc=$?; set -e; if ((rc==0)); then "$EMENDER_PYTHON" "$SNAP/scripts/aggregate_e97_dense_agent_eval.py" --shard-dir "$RUN_ROOT/shards" --output "$RUN_ROOT/results/summary.json" || rc=$?; fi; printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; exit "$rc"
