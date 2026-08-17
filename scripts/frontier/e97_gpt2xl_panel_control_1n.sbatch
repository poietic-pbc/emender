#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-gpt2xl-control
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/eval/%x-%j.out
#SBATCH -e logs/frontier/eval/%x-%j.err
set -euo pipefail
: "${SOURCE_COMMIT:?}"; : "${RUN_ROOT:?}"
MODEL=/lustre/orion/bif148/proj-shared/emender/external_models/openai-community-gpt2-xl-15ea56d
REVISION=15ea56dee5df4983c59b2538573817e1667135e2
PANEL=/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/panel/e97_moe_paired_v2.pt
PANEL_SHA256=fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=1"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
[[ $(sha256sum "$MODEL/model.safetensors"|awk '{print $1}') == 0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a && $(sha256sum "$PANEL"|awk '{print $1}') == "$PANEL_SHA256" ]] || exit 65
cd "$REPO"; git cat-file -e "${SOURCE_COMMIT}^{commit}"; mkdir -p "$RUN_ROOT"/{identity,results,sources,terminal} logs/frontier/eval
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"; mkdir -p "$SNAPSHOT"; [[ -f "$SNAPSHOT/scripts/eval_hf_causal_baseline.py" ]] || git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"; squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SNAPSHOT"; export PYTHONPATH="$SNAPSHOT"
set +e
"$EMENDER_PYTHON" scripts/eval_hf_causal_baseline.py --model "$MODEL" --revision "$REVISION" --panel "$PANEL" --panel-sha256 "$PANEL_SHA256" --output "$RUN_ROOT/results/gpt2-xl.json"
rc=$?; set -e; printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"; exit "$rc"
