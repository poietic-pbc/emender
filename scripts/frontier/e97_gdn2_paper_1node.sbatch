#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-paper-1node
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_paper/%x-%j.out
#SBATCH -e logs/frontier/e97_paper/%x-%j.err

set -euo pipefail
: "${ARM:?ARM must be e97-mlp, e97-linear-mlp, or gdn2-mlp}"
: "${SOURCE_COMMIT:?immutable source commit required}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
GDN2_CHECKOUT=${GDN2_CHECKOUT:-$REPO/src/GatedDeltaNet-2}
RUN_BASE=${RUN_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-gdn2-paper/qualification}
RUN_ROOT="$RUN_BASE/1node/$ARM/job-$SLURM_JOB_ID"
mkdir -p "$RUN_ROOT"/{external,logs,output,receipts} "$REPO/logs/frontier/e97_paper"
cd "$REPO"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* \
   && "$record" == *"Requeue=0"* && "$record" == *"NumNodes=1"* \
   && "$record" == *"NumTasks=8"* ]] || { echo "scheduler binding mismatch" >&2; exit 65; }
printf '%s\n' "$record" | tee "$RUN_ROOT/receipts/live-scontrol.txt"
squeue -h -j "$SLURM_JOB_ID" -o '%i|%P|%q|%T|%D|%l' \
  | tee "$RUN_ROOT/receipts/live-squeue.txt"

SNAPSHOT="$RUN_BASE/sources/code-$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/.emender-source-receipt" ]]; then
  tmp="$SNAPSHOT.tmp-$SLURM_JOB_ID"
  rm -rf "$tmp"; mkdir -p "$tmp"
  SOURCE_PATHS=(train.py ndm scripts configs tests pyproject.toml)
  git archive "$SOURCE_COMMIT" -- "${SOURCE_PATHS[@]}" | tar -xf - -C "$tmp"
  { echo "commit=$SOURCE_COMMIT"; printf 'path=%s\n' "${SOURCE_PATHS[@]}"; } \
    > "$tmp/.emender-source-receipt"
  if ! mv -T "$tmp" "$SNAPSHOT" 2>/dev/null; then
    rm -rf "$tmp"; [[ -f "$SNAPSHOT/.emender-source-receipt" ]] || exit 68
  fi
fi
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/receipts/source-commit.txt"
printf '%s\n' "$SNAPSHOT" > "$RUN_ROOT/receipts/source-snapshot.txt"
cp "$SNAPSHOT/.emender-source-receipt" "$RUN_ROOT/receipts/source-snapshot-receipt.txt"
git show "$SOURCE_COMMIT:scripts/frontier/activate_emender_frontier.sh" \
  | cmp - "$REPO/scripts/frontier/activate_emender_frontier.sh"
source "$REPO/scripts/frontier/activate_emender_frontier.sh"

GDN2_STAGE=
if [[ "$ARM" == gdn2-mlp ]]; then
  GDN2_STAGE="$RUN_ROOT/external/gdn2-95709fc"
  PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}" \
    GDN2_PATH="$GDN2_CHECKOUT" "$EMENDER_PYTHON" \
    "$SNAPSHOT/scripts/stage_gdn2_source.py" --checkout "$GDN2_CHECKOUT" \
    --expected-commit 95709fc250357c2dd109361c353192f2aa5913f9 \
    --output "$GDN2_STAGE" | tee "$RUN_ROOT/receipts/gdn2-source.json"
fi

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
export MASTER_PORT=$((32000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=8 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MPICH_GPU_SUPPORT_ENABLED=1 NCCL_DEBUG=WARN
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base
export NDM_PIN_TRITON_AUTOTUNE=1 PYTHONDONTWRITEBYTECODE=1
export EMENDER_GDN2_REQUIRE_BOUND_SOURCE=1
export EMENDER_GDN2_EXPECTED_COMMIT=95709fc250357c2dd109361c353192f2aa5913f9
export REPO SNAPSHOT RUN_ROOT ARM DATA GDN2_STAGE

run_epoch() {
  local target_steps=$1 resume_path=${2:-}
  export TARGET_STEPS=$target_steps RESUME_PATH=$resume_path
  srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
    set -euo pipefail
    source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
    cd "$SNAPSHOT"
    export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
    export RANK=$SLURM_PROCID LOCAL_RANK=0
    export TRITON_CACHE_DIR="/tmp/e97-paper-${SLURM_JOB_ID}-${ARM}-${SLURM_PROCID}"
    [[ -n "$RESUME_PATH" ]] || rm -rf "$TRITON_CACHE_DIR"
    [[ "$ARM" != gdn2-mlp ]] || export GDN2_PATH="$GDN2_STAGE"
    render=("$EMENDER_PYTHON" scripts/render_e97_gdn2_paper_args.py
      --arm "$ARM" --world-size 8 --steps "$TARGET_STEPS" --data "$DATA"
      --output "$RUN_ROOT/output" --save-every 80)
    [[ -z "$RESUME_PATH" ]] || render+=(--resume "$RESUME_PATH")
    mapfile -d "" TRAIN_ARGS < <("${render[@]}")
    "$EMENDER_PYTHON" train.py "${TRAIN_ARGS[@]}" \
      --walltime_final_checkpoint_margin_seconds 300 --log_every 10
  ' 2>&1 | tee "$RUN_ROOT/logs/train-to-${target_steps}.log"
}

run_epoch 160
LATEST="$RUN_ROOT/output/latest.pt"
[[ -L "$LATEST" && -r "$LATEST" ]] || { echo "missing first authority" >&2; exit 69; }
FIRST=$(readlink -f "$LATEST")
printf '%s\n' "$FIRST" > "$RUN_ROOT/receipts/step160-checkpoint.txt"
run_epoch 200 "$LATEST"

PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}" "$EMENDER_PYTHON" - <<PY \
  | tee "$RUN_ROOT/receipts/checkpoint-validation.json"
import json, torch
path = "$RUN_ROOT/output/latest.pt"
ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
expected = 8 * 2 * 2048 * 200
assert ckpt["step"] == 200, ckpt["step"]
assert ckpt["total_tokens"] == expected, (ckpt["total_tokens"], expected)
sampler = ckpt["checkpoint_metadata"]["sampler"]
assert sampler["total_accepted_tokens"] == expected
assert sampler["absolute_rank_sample_index"] == 400
print(json.dumps({"step": ckpt["step"], "total_tokens": ckpt["total_tokens"],
                  "sampler_cursor": sampler["absolute_rank_sample_index"],
                  "latest": str(__import__("pathlib").Path(path).resolve())}, sort_keys=True))
PY
find /tmp -maxdepth 2 -path "*/e97-paper-${SLURM_JOB_ID}-${ARM}-*/*" -type f \
  | sort > "$RUN_ROOT/receipts/triton-cache-files.txt" || true
printf 'PASS\n' > "$RUN_ROOT/receipts/verdict.txt"
echo "run_root=$RUN_ROOT"
