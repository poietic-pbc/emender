#!/bin/bash
# Bounded E97 debug smoke for the current runtime with OLCF rccl-net-plugin/1.0.
#
# This wrapper intentionally passes RESUME_CHECKPOINT as the active production
# chain latest.pt symlink and leaves CHAIN_LATEST_PATH empty so the delegated
# canary runner cannot advance the production chain pointer.

#SBATCH -A bif148
#SBATCH -J e97-current-rccl-debug
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/debug/%x-%j.out
#SBATCH -e logs/frontier/debug/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ACTIVE_CHAIN_LATEST=${ACTIVE_CHAIN_LATEST:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt}
RESOLVED_ACTIVE_CHAIN_TARGET=$(readlink -f "$ACTIVE_CHAIN_LATEST")

[[ -r "$ACTIVE_CHAIN_LATEST" ]] || { echo "active chain latest.pt is not readable: $ACTIVE_CHAIN_LATEST" >&2; exit 4; }
[[ -r "$RESOLVED_ACTIVE_CHAIN_TARGET" ]] || { echo "resolved active chain target is not readable: $RESOLVED_ACTIVE_CHAIN_TARGET" >&2; exit 4; }

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load rccl-net-plugin/1.0

RCCL_NET_PLUGIN_STATUS="not-found"
IFS=':' read -r -a LD_LIBRARY_PATH_PARTS <<< "${LD_LIBRARY_PATH:-}"
for lib_dir in "${LD_LIBRARY_PATH_PARTS[@]}"; do
  if [[ -n "$lib_dir" && -r "${lib_dir%/}/librccl-net.so" ]]; then
    RCCL_NET_PLUGIN_STATUS="${lib_dir%/}/librccl-net.so"
    break
  fi
done

echo "active_chain_latest=$ACTIVE_CHAIN_LATEST"
echo "resolved_active_chain_target=$RESOLVED_ACTIVE_CHAIN_TARGET"
echo "rccl_net_plugin_status=$RCCL_NET_PLUGIN_STATUS"
module -t list 2>&1 | sed 's/^/module=/'

[[ "$RCCL_NET_PLUGIN_STATUS" != "not-found" ]] || { echo "librccl-net.so was not found after loading rccl-net-plugin/1.0" >&2; exit 4; }

export WG_TASK_ID=${WG_TASK_ID:-debug-current-runtime-rccl-plugin}
export SCALEOUT_VARIANT=${SCALEOUT_VARIANT:-E97_1.3B_current_runtime_rccl_plugin_debug}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$ACTIVE_CHAIN_LATEST}
export CHAIN_LATEST_PATH=""
export CHAIN_MANIFEST_PATH=""
export CHAIN_UPDATE_ON_FAILURE=0
export TRAIN_MINUTES=${TRAIN_MINUTES:-3}
export DILOCO_K=${DILOCO_K:-999999}
export DILOCO_ISLAND_SIZE=${DILOCO_ISLAND_SIZE:-1}
export DILOCO_MERGE_TOPOLOGY=${DILOCO_MERGE_TOPOLOGY:-global}
export DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-0}
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_RCCL_ALT_RDZV=${FRONTIER_RCCL_ALT_RDZV:-1}
export FRONTIER_PER_RANK_TRITON_CACHE=${FRONTIER_PER_RANK_TRITON_CACHE:-1}
export BATCH_SIZE=${BATCH_SIZE:-1}
export LOG_EVERY=${LOG_EVERY:-1}
export VAL_EVERY=${VAL_EVERY:-10000}
export SAVE_EVERY=${SAVE_EVERY:-999999}
export KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-1}
export COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-1}
export WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-1200}
export WALLTIME_CHECK_EVERY=${WALLTIME_CHECK_EVERY:-60}
export DISTRIBUTED_HEALTH_CHECK_EVERY=${DISTRIBUTED_HEALTH_CHECK_EVERY:-60}
export REQUESTED_SECONDS=${REQUESTED_SECONDS:-1200}
export REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-00:20:00}
export REQUESTED_NODE_HOURS=${REQUESTED_NODE_HOURS:-0.666667}
export HUMAN_APPROVAL_RECORD=${HUMAN_APPROVAL_RECORD:-debug-current-runtime-rccl-plugin: 2-node debug smoke only; resume from active production latest.pt symlink; isolated output under frontier_runs/debug; production chain pointer update disabled.}

exec bash "${REPO}/scripts/frontier/e97_1p3b_pretrained_canary.sbatch"
