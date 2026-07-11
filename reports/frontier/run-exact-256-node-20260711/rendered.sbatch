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

set -euo pipefail
BUNDLE_FINGERPRINT='24e0fdbff99b4fcfb3b4ff729ff26a372d66db4a4b166730e92ff62526d5fb8e'
BUNDLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
test "$(sha256sum "$BUNDLE_DIR/fingerprint.sha256" | cut -d' ' -f1)" = "$(cat "$BUNDLE_DIR/fingerprint-file.sha256")" || exit 72
RUN_ID="e97-256-${SLURM_JOB_ID:?}"
RUN_DIR="${SLURM_SUBMIT_DIR}/e97-256-runs/${SLURM_JOB_ID}"
METRICS="${RUN_DIR}/metrics.json"
HELPER="${RUN_DIR}/compiled_mpich_dense_helper"
IPC="${TMPDIR:-/tmp}/e97-256-${SLURM_JOB_ID}/ipc"
COORDINATOR_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
COORDINATOR_PORT=29497
mkdir -p "$RUN_DIR" "$IPC"
cd "$REPO"
ARTIFACT_DIR="$RUN_DIR" OUT="$HELPER" scripts/frontier/build_compiled_mpich_dense_helper.sh
exec srun -N 256 -n 2048 --ntasks-per-node 8 -c 7 --gpus-per-task 1 --gpu-bind closest bash -lc 'exec .envs/olcf-rocm711-torch210-py312/bin/python -u scripts/frontier/e97_async_diloco_train.py --run-id ${RUN_ID} --run-dir ${RUN_DIR} --metrics-json ${METRICS} --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --checkpoint /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt --tokenizer p50k_base --worker-count 2048 --node-count 2048 --node-rank ${SLURM_PROCID:?missing SLURM_PROCID} --global-quorum 2048 --generations 1 --local-steps 40 --steps 40 --timeout-s 1200 --diloco-quorum-mode resilient_quorum --level e97 --params 100m --batch-size 4 --chunk-size 2048 --lr 0.001007 --optimizer schedulefree --weight-decay 0.01 --warmup-steps 0 --min-lr-frac 0.1 --grad-accum 1 --grad-clip 1.0 --dim 1792 --depth 11 --n-heads 216 --n-state 32 --n-groups 32 --n-slots 64 --expansion 1.0 --state-expansion 2 --gate-activation silu --linear-state 0 --mlp-ratio 2.2623 --mlp-multiple 64 --e97-chunk-size 32 --checkpoint-interval 16 --projection-chunk-size 0 --loss-chunk-size 0 --recovery-every-generations 1 --recovery-every-seconds -1 --export-every-generations 1 --export-every-seconds -1 --finalization-reserve-seconds 1200 --walltime-remaining-s 1200 --coordinator-host ${COORDINATOR_HOST} --coordinator-bind-host 0.0.0.0 --coordinator-port ${COORDINATOR_PORT} --mpi-dense-bucket-bytes 67108864 --compiled-mpich-helper-bin ${HELPER} --compiled-mpich-ipc-dir ${IPC} --actual-multinode-compiled-mpich-quorum --device cuda:0 --bf16'
