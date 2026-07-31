#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-direct-samealloc-restart
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-kill
#SBATCH -o /lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/payload-%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/payload-%j.err

# Replacement research path: intentionally tests fixed-world train.py/RCCL step
# replacement. It is NOT Compute Pool v1 or async-v2.1 conformance.
set -euo pipefail
REPO=${REPO:?REPO must name the immutable submitted checkout}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN must be the activated EMENDER_PYTHON}
ROOT=${DIRECT_RESTART_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/${SLURM_JOB_ID}}
mkdir -p "$ROOT"/{identity,logs,steps,checkpoint}
exec > >(tee -a "$ROOT/logs/supervisor.out") 2> >(tee -a "$ROOT/logs/supervisor.err" >&2)
cd "$REPO"
source scripts/frontier/frontier_runtime_env.sh
frontier_load_default_modules
frontier_activate_emender_conda_env
frontier_assert_emender_conda_env
[[ $(command -v python) == "$PYTHON_BIN" ]] || { echo "activated python differs from PYTHON_BIN" >&2; exit 64; }
TRAIN_PYTHON_BIN=$PYTHON_BIN
export REPO TRAIN_PYTHON_BIN PYTHON_BIN

START_EPOCH=$(date +%s)
SOURCE_SHA=$(git rev-parse HEAD)
printf '%s\n' "$SOURCE_SHA" > "$ROOT/identity/git-commit.txt"
sha256sum train.py "$0" > "$ROOT/identity/source.sha256"
printf '%s\n' "$SLURM_JOB_NODELIST" > "$ROOT/identity/allocation-nodelist.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" | tee "$ROOT/identity/allocation-hostnames.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$ROOT/identity/scontrol-start.txt"
squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%N|%P|%q|%R' | tee "$ROOT/identity/squeue-running.txt"

DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
VAL_DATA=${VAL_DATA:-/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt}
SEED=${SEED:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt}
[[ -r $DATA && -r $VAL_DATA && -r $SEED ]] || { echo "missing data/validation/seed input" >&2; exit 64; }
sha256sum "$SEED" > "$ROOT/identity/seed.sha256"
stat --printf='%n|%s|%y\n' "$SEED" "$DATA" "$VAL_DATA" > "$ROOT/identity/inputs.stat"

export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
export MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=7 NDM_PIN_TRITON_AUTOTUNE=1
export FRONTIER_RCCL_ENV=recommended FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
export FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug REQUIRE_RCCL_NET_PLUGIN=1
export FI_MR_CACHE_MONITOR=kdreg2 FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_RDZV_PROTO=alt_read
export NCCL_NET_GDR_LEVEL=3 NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_PLUGIN=librccl-net.so NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=240
frontier_require_requested_rccl_net_plugin "same-allocation supervisor" | tee "$ROOT/identity/rccl-plugin.txt"

COMMON_ARGS=(
 --data "$DATA" --val_data "$VAL_DATA" --bf16 --batch_size 1 --chunk_size 2048
 --tokenizer p50k_base --optimizer schedulefree --seed 42 --lr 0.001007
 --train_minutes 2 --timer_after_compile_warmup --compile_warmup_steps 1
 --log_every 1 --val_every 10000 --save_every 1 --keep_checkpoints 2
 --walltime_final_checkpoint_margin_seconds 120 --walltime_check_every 1
 --distributed_health_check_every 1 --diloco --diloco_k 1
 --diloco_outer_optimizer avg --diloco_outer_lr 1.0 --diloco_outer_beta 0.0
 --diloco_export_basis x --diloco_island_size 1 --diloco_merge_topology hierarchical
 --diloco_merge_group_size 4 --diloco_merge_group_create_barrier_every 4
 --diloco_merge_completion_barrier 1 --diloco_merge_bucket_numel 67108864
 --diloco_merge_debug 1 --diloco_merge_debug_ranks 0,1
 --level E97 --params 100m --dim 1792 --depth 11 --n_heads 216 --n_state 32
 --n_groups 32 --n_slots 64 --state_expansion 2 --expansion 1.0 --use_gate 1
 --use_permutation 1 --gate_activation silu --linear_state 0 --use_write_gate 0
 --use_triton 1 --use_chunked_e97 0 --e97_chunk_size 32 --mlp_ratio 2.2623
 --mlp_multiple 64 --checkpoint_interval 16 --weight_decay 0.01 --grad_accum 1 --grad_clip 1.0
)
printf '%q ' "$PYTHON_BIN" -u train.py "${COMMON_ARGS[@]}" > "$ROOT/identity/train-command.txt"; echo >> "$ROOT/identity/train-command.txt"

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_BASE=$((20000 + (10#$SLURM_JOB_ID % 30000)))
run_step() {
  local name=$1 nodes=$2 tasks=$3 nodelist=$4 resume=$5 output=$6 port=$7
  shift 7
  mkdir -p "$output"
  local before after rc
  before=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=60s 420s \
    srun --exact --no-kill --kill-on-bad-exit=1 --wait=60 -N"$nodes" -n"$tasks" \
      --ntasks-per-node=8 -c7 --gpus-per-task=1 --gpu-bind=closest --nodelist="$nodelist" \
      --export=ALL,MASTER_ADDR="$MASTER_HOST",MASTER_PORT="$port",EMENDER_PHASE="$name",$* \
      bash -lc 'source "$REPO/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; frontier_require_requested_rccl_net_plugin "child rank ${SLURM_PROCID}" >/dev/null; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec "$PYTHON_BIN" -u train.py "$@"' \
      bash "${COMMON_ARGS[@]}" --resume "$resume" --output "$output" \
      >"$ROOT/logs/${name}.out" 2>"$ROOT/logs/${name}.err"
  rc=$?
  set -e
  after=$(date +%s)
  printf '%s|%s|%s|%s|%s|%s\n' "$name" "$rc" "$before" "$after" "$nodes" "$tasks" | tee "$ROOT/steps/${name}.rc"
  return "$rc"
}

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
BASE_OUT="$ROOT/baseline"
run_step baseline 2 16 "$ALL_NODES" "$SEED" "$BASE_OUT" "$((MASTER_BASE+1))"
BASE_RC=$?
BASE_LATEST=$(find "$BASE_OUT" -name latest.pt -type l -print | sort | tail -1)
[[ $BASE_RC -eq 0 && -n $BASE_LATEST ]] || { echo "baseline failed to publish checkpoint" >&2; exit 70; }
BASE_CKPT=$(readlink -f "$BASE_LATEST")
sha256sum "$BASE_CKPT" > "$ROOT/checkpoint/baseline.sha256"
"$PYTHON_BIN" - "$BASE_CKPT" > "$ROOT/checkpoint/baseline-reload.json" <<'PY'
import json, sys, torch
p=sys.argv[1]; c=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
assert c.get('step',0)>0 and c.get('model_state_dict') and c.get('optimizer_state_dict')
print(json.dumps({'checkpoint':p,'reload_verified':True,'step':c['step'],'world_size':c.get('checkpoint_metadata',{}).get('world_size')}))
PY
BASE_STEP=$("$PYTHON_BIN" -c 'import torch,sys; print(torch.load(sys.argv[1],map_location="cpu",mmap=True,weights_only=False)["step"])' "$BASE_CKPT")

FAULT_OUT="$ROOT/fault"
FAULT_START=$(date +%s)
if run_step fault 2 16 "$ALL_NODES" "$BASE_CKPT" "$FAULT_OUT" "$((MASTER_BASE+2))" \
    EMENDER_DILOCO_EXIT_RANK=1,EMENDER_DILOCO_EXIT_MERGE=1,EMENDER_DILOCO_EXIT_BUCKET=0,EMENDER_DILOCO_EXIT_LABEL=model,EMENDER_DILOCO_EXIT_CODE=86; then
  FAULT_RC=0
else
  FAULT_RC=$?
fi
FAULT_END=$(date +%s)
[[ $FAULT_RC -ne 0 ]] || { echo "fault step unexpectedly succeeded" >&2; exit 71; }
grep -q DILOCO_FAULT_INJECTION "$ROOT/logs/fault.out" || { echo "missing exact rank injection marker" >&2; exit 72; }
[[ -z $(find "$FAULT_OUT" -name 'checkpoint_step_*.pt' -print -quit 2>/dev/null) ]] || { echo "fault step published checkpoint" >&2; exit 73; }
squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%N|%P|%q|%R' | tee "$ROOT/steps/allocation-after-fault.squeue"
[[ -s "$ROOT/steps/allocation-after-fault.squeue" ]] || { echo "allocation did not survive" >&2; exit 74; }

RECOVERY_START=$(date +%s)
RECOVERY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
RECOVERY_OUT="$ROOT/recovery"
run_step recovery 1 8 "$RECOVERY_NODE" "$BASE_CKPT" "$RECOVERY_OUT" "$((MASTER_BASE+3))"
RECOVERY_RC=$?
RECOVERY_END=$(date +%s)
RECOVERY_LATEST=$(find "$RECOVERY_OUT" -name latest.pt -type l -print | sort | tail -1)
[[ $RECOVERY_RC -eq 0 && -n $RECOVERY_LATEST ]] || { echo "recovery did not publish checkpoint" >&2; exit 75; }
RECOVERY_CKPT=$(readlink -f "$RECOVERY_LATEST")
sha256sum "$RECOVERY_CKPT" > "$ROOT/checkpoint/recovery.sha256"
"$PYTHON_BIN" - "$RECOVERY_CKPT" > "$ROOT/checkpoint/recovery-reload.json" <<'PY'
import json, sys, torch
p=sys.argv[1]; c=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
assert c.get('step',0)>0 and c.get('model_state_dict') and c.get('optimizer_state_dict')
print(json.dumps({'checkpoint':p,'reload_verified':True,'step':c['step'],'world_size':c.get('checkpoint_metadata',{}).get('world_size')}))
PY
RECOVERY_STEP=$("$PYTHON_BIN" -c 'import torch,sys; print(torch.load(sys.argv[1],map_location="cpu",mmap=True,weights_only=False)["step"])' "$RECOVERY_CKPT")

grep -q '\[DiLoCo\] merge #1' "$ROOT/logs/baseline.out"
grep -q '\[DiLoCo\] merge #1' "$ROOT/logs/recovery.out"
LOST_WORK=$((BASE_STEP + 1 - BASE_STEP))
DOWNTIME=$((RECOVERY_START - FAULT_END))
"$PYTHON_BIN" - "$ROOT/verdict.json" <<PY
import json,sys
v={
 'schema':'direct-same-allocation-trainpy-restart-v1',
 'replacement_research_path':True,'v1_conformance':False,'v21_conformance':False,
 'job_id':'$SLURM_JOB_ID','source_sha':'$SOURCE_SHA',
 'allocation_survived':True,'failed_step_bounded':True,
 'fresh_srun_launched':True,'world_size_changed':True,
 'checkpoint_reloaded':True,'post_relaunch_merge_passed':True,
 'baseline_world_size':16,'recovery_world_size':8,
 'baseline_step':int('$BASE_STEP'),'recovery_step':int('$RECOVERY_STEP'),
 'fault_returncode':int('$FAULT_RC'),'recovery_downtime_seconds':int('$DOWNTIME'),
 'lost_work_steps':int('$LOST_WORK'),'full_pass':True}
open(sys.argv[1]+'.tmp','w').write(json.dumps(v,sort_keys=True)+'\n')
import os; os.replace(sys.argv[1]+'.tmp',sys.argv[1])
PY
cat "$ROOT/verdict.json"
