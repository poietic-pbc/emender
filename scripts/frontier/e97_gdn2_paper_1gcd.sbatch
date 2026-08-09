#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-paper-1gcd
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH --ntasks-per-node=1
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
RUN_ROOT="$RUN_BASE/1gcd/$ARM/job-$SLURM_JOB_ID"
mkdir -p "$RUN_ROOT"/{source,external,logs,output,receipts} "$REPO/logs/frontier/e97_paper"
cd "$REPO"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* \
   && "$record" == *"Requeue=0"* && "$record" == *"NumNodes=1"* ]] || {
  echo "scheduler binding mismatch: $record" >&2; exit 65; }
printf '%s\n' "$record" | tee "$RUN_ROOT/receipts/live-scontrol.txt"
squeue -h -j "$SLURM_JOB_ID" -o '%i|%P|%q|%T|%D|%l' \
  | tee "$RUN_ROOT/receipts/live-squeue.txt"

SNAPSHOT="$RUN_BASE/sources/code-$SOURCE_COMMIT"
if [[ ! -f "$SNAPSHOT/.emender-source-receipt" ]]; then
  tmp="$SNAPSHOT.tmp-$SLURM_JOB_ID"
  rm -rf "$tmp"; mkdir -p "$tmp"
  SOURCE_PATHS=(train.py ndm scripts configs tests pyproject.toml)
  git archive "$SOURCE_COMMIT" -- "${SOURCE_PATHS[@]}" | tar -xf - -C "$tmp"
  {
    echo "commit=$SOURCE_COMMIT"
    printf 'path=%s\n' "${SOURCE_PATHS[@]}"
  } > "$tmp/.emender-source-receipt"
  if ! mv -T "$tmp" "$SNAPSHOT" 2>/dev/null; then
    rm -rf "$tmp"
    [[ -f "$SNAPSHOT/.emender-source-receipt" ]] || exit 68
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
    "$SNAPSHOT/scripts/stage_gdn2_source.py" \
    --checkout "$GDN2_CHECKOUT" \
    --expected-commit 95709fc250357c2dd109361c353192f2aa5913f9 \
    --output "$GDN2_STAGE" | tee "$RUN_ROOT/receipts/gdn2-source.json"
fi

export SNAPSHOT RUN_ROOT ARM DATA GDN2_STAGE
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base
export NDM_PIN_TRITON_AUTOTUNE=1
export PYTHONDONTWRITEBYTECODE=1
export EMENDER_GDN2_REQUIRE_BOUND_SOURCE=1
export EMENDER_GDN2_EXPECTED_COMMIT=95709fc250357c2dd109361c353192f2aa5913f9

srun --label --kill-on-bad-exit=1 --wait=60 bash -lc '
  set -euo pipefail
  source "$REPO/scripts/frontier/activate_emender_frontier.sh" >/dev/null
  cd "$SNAPSHOT"
  export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  export TRITON_CACHE_DIR="/tmp/e97-paper-${SLURM_JOB_ID}-${ARM}"
  rm -rf "$TRITON_CACHE_DIR"
  if [[ "$ARM" == gdn2-mlp ]]; then
    export GDN2_PATH="$GDN2_STAGE"
    "$EMENDER_PYTHON" scripts/frontier/gdn2_rocm_preflight.py \
      --run-fwdbwd --bf16 --batch-size 1 --seq-len 32 --dim 128 \
      --n-heads 4 --head-dim 16 --expansion 1 --use-conv 1 \
      --d-conv 4 --mlp-ratio 2 | tee "$RUN_ROOT/logs/kernel-parity.log"
  elif [[ "$ARM" == e97-linear-mlp ]]; then
    "$EMENDER_PYTHON" -m pytest -q -s \
      tests/test_e88_triton.py::test_e97_split_edit_linear_triton_matches_reference \
      tests/test_e97_chunked.py::test_fused_triton_forward_parity_fp32 \
      tests/test_e97_chunked.py::test_fused_triton_backward_parity_fp32 \
      tests/test_e97_chunked.py::test_fused_triton_backward_parity_bf16 \
      tests/test_e97_chunked.py::test_strong_decay_cluster_backward_finite \
      tests/test_e97_chunked.py::test_glog_floor_drift_overflow_finite \
      | tee "$RUN_ROOT/logs/kernel-parity.log"
  else
    "$EMENDER_PYTHON" -m pytest -q -s \
      tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference \
      | tee "$RUN_ROOT/logs/kernel-parity.log"
  fi
  mapfile -d "" TRAIN_ARGS < <("$EMENDER_PYTHON" \
    scripts/render_e97_gdn2_paper_args.py --arm "$ARM" --world-size 1 \
    --steps 1 --data "$DATA" --output "$RUN_ROOT/output" --save-every 1)
  "$EMENDER_PYTHON" train.py "${TRAIN_ARGS[@]}" --probe_memory \
    | tee "$RUN_ROOT/logs/full-model-probe.log"
  find "$TRITON_CACHE_DIR" -type f -printf "%P\n" | sort \
    > "$RUN_ROOT/receipts/triton-cache-files.txt"
  printf "triton_cache_file_count=%s\n" "$(wc -l < "$RUN_ROOT/receipts/triton-cache-files.txt")" \
    | tee "$RUN_ROOT/receipts/triton-cache-count.txt"
'
printf 'PASS\n' > "$RUN_ROOT/receipts/verdict.txt"
echo "run_root=$RUN_ROOT"
