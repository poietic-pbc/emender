#!/bin/bash
#SBATCH --account=bif148
#SBATCH --job-name=e97-async-256-smoke
#SBATCH --partition=batch
#SBATCH --qos=debug
#SBATCH --nodes=256
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@1200
#SBATCH --output=e97-async-256-smoke-%j.out
#SBATCH --error=e97-async-256-smoke-%j.err

set -Eeuo pipefail
BUNDLE_FINGERPRINT='338b05d24fba86b1a12d40413622ba258f389a43ff470e8e9ea5fed6ea509be3'
E97_PHASE=bootstrap
e97_phase() { E97_PHASE="$1"; printf 'e97-presrun phase=%s status=begin fingerprint=%s\n' "$E97_PHASE" "$BUNDLE_FINGERPRINT" >&2; }
e97_error() { local rc=$?; trap - ERR; printf 'e97-presrun phase=%s status=error rc=%s line=%s command=%q\n' "$E97_PHASE" "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND" >&2; exit "$rc"; }
trap e97_error ERR
printf 'e97-presrun phase=bootstrap status=begin job_id=%s host=%s export=NONE\n' "${SLURM_JOB_ID:-unset}" "$(hostname)" >&2
BUNDLE_MANIFEST_SHA256='65bcd8ba05532d7f64bc42e839af57510291a98dab9c731e842165d183f2b768'
BUNDLE_DIR=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke
REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949
LAUNCHER=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
e97_phase bundle-binding
test -d "$BUNDLE_DIR" || { echo "immutable bundle is unavailable: $BUNDLE_DIR" >&2; exit 72; }
test -r "$BUNDLE_DIR/bundle-files.sha256" || { echo "bundle manifest is unavailable" >&2; exit 72; }
test "$(sha256sum "$BUNDLE_DIR/bundle-files.sha256" | cut -d' ' -f1)" = "$BUNDLE_MANIFEST_SHA256" || { echo "bundle manifest hash mismatch" >&2; exit 72; }
e97_phase module-bootstrap
if ! command -v module >/dev/null 2>&1; then test -r /etc/bash.bashrc.local || { echo "Cray PE initialization is unavailable" >&2; exit 72; }; set +u; source /etc/bash.bashrc.local; set -u; test -r /etc/profile.d/olcf-env.sh || { echo "OLCF module site initialization is unavailable" >&2; exit 72; }; source /etc/profile.d/olcf-env.sh || command -v module >/dev/null 2>&1; fi
command -v module >/dev/null 2>&1 || { echo "module command is unavailable after initialization" >&2; exit 72; }
module purge
module load PrgEnv-gnu
module load cray-mpich
module load rocm/6.3.1
export ALLOW_SYNTHETIC_TOKEN_FALLBACK=0
export ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0
export MPICH_GPU_SUPPORT_ENABLED=0
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
command -v CC >/dev/null 2>&1 || { echo "declared modules did not provide CC" >&2; exit 72; }
test -x "$LAUNCHER" || { echo "declared launcher is unavailable: $LAUNCHER" >&2; exit 72; }
e97_phase runtime-paths
RUN_ID="e97-256-${SLURM_JOB_ID:?}"
RUN_DIR="${SLURM_SUBMIT_DIR:?}/e97-256-runs/${SLURM_JOB_ID}"
METRICS="${RUN_DIR}/metrics.json"
HELPER="${RUN_DIR}/compiled_mpich_dense_helper"
IPC="${TMPDIR:-/tmp}/e97-256-${SLURM_JOB_ID}/ipc"
COORDINATOR_HOST=$(scontrol show hostnames "${SLURM_NODELIST:?}" | head -1)
COORDINATOR_PORT=29497
export RUN_ID RUN_DIR METRICS HELPER IPC COORDINATOR_HOST COORDINATOR_PORT
mkdir -p "$RUN_DIR" "$IPC"
cd "$REPO"
e97_phase preflight-before-helper
"$LAUNCHER" "$REPO/scripts/frontier/e97_async_256_preflight.py" --bundle "$BUNDLE_DIR" --repo "$REPO" --fingerprint "$BUNDLE_FINGERPRINT" --phase before-helper
e97_phase helper-build
ARTIFACT_DIR="$RUN_DIR" OUT="$HELPER" scripts/frontier/build_compiled_mpich_dense_helper.sh
e97_phase preflight-before-srun
"$LAUNCHER" "$REPO/scripts/frontier/e97_async_256_preflight.py" --bundle "$BUNDLE_DIR" --repo "$REPO" --fingerprint "$BUNDLE_FINGERPRINT" --phase before-srun --helper "$HELPER"
if [[ "${E97_STOP_BEFORE_SRUN:-0}" == 1 ]]; then printf 'e97-presrun phase=preflight-only status=complete ranks=2048 nodes=256\n' >&2; exit 0; fi
e97_phase srun
printf 'e97-presrun phase=srun status=exec ranks=2048 nodes=256\n' >&2
exec srun -N 256 -n 2048 --ntasks-per-node 8 -c 7 --gpus-per-task 1 --gpu-bind closest bash -lc 'exec /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -u scripts/frontier/e97_async_diloco_train.py --run-id ${RUN_ID} --run-dir ${RUN_DIR} --metrics-json ${METRICS} --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --checkpoint /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt --tokenizer p50k_base --worker-count 2048 --node-count 2048 --node-rank ${SLURM_PROCID:?missing SLURM_PROCID} --global-quorum 2048 --generations 1 --local-steps 40 --steps 40 --timeout-s 1200 --diloco-quorum-mode resilient_quorum --level e97 --params 100m --batch-size 4 --chunk-size 2048 --lr 0.001007 --optimizer schedulefree --weight-decay 0.01 --warmup-steps 0 --min-lr-frac 0.1 --grad-accum 1 --grad-clip 1.0 --dim 1792 --depth 11 --n-heads 216 --n-state 32 --n-groups 32 --n-slots 64 --expansion 1.0 --state-expansion 2 --gate-activation silu --linear-state 0 --mlp-ratio 2.2623 --mlp-multiple 64 --e97-chunk-size 32 --checkpoint-interval 16 --projection-chunk-size 0 --loss-chunk-size 0 --recovery-every-generations 1 --recovery-every-seconds -1 --export-every-generations 1 --export-every-seconds -1 --finalization-reserve-seconds 1200 --walltime-remaining-s 1200 --coordinator-host ${COORDINATOR_HOST} --coordinator-bind-host 0.0.0.0 --coordinator-port ${COORDINATOR_PORT} --mpi-dense-bucket-bytes 67108864 --compiled-mpich-helper-bin ${HELPER} --compiled-mpich-ipc-dir ${IPC} --actual-multinode-compiled-mpich-quorum --device cuda:0 --bf16'
