#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-4b-fresh
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300

# Fixed-world fail-stop E97 4B from-scratch payload. Submission overrides the
# debug defaults. No elastic/native/async conformance is claimed.
set -euo pipefail
: "${REPO:?immutable submitted checkout is required}"
: "${RUN_ID:?stable run id is required}"
: "${RUN_DIR:?stable absolute run directory is required}"
: "${EXPECTED_NODES:?expected node count is required}"
: "${EXPECTED_WORLD_SIZE:?expected world size is required}"
: "${EXPECTED_PARTITION:?expected partition is required}"
: "${EXPECTED_QOS:?expected QoS is required}"
: "${EXPECTED_TIME_LIMIT:?expected time limit is required}"
RUN_MODE=${RUN_MODE:-smoke}
CONFIG=${CONFIG:-$REPO/configs/frontier/e97_4b_from_scratch.json}

[[ "$RUN_DIR" == /* && "$RUN_ID" != */* ]] || { echo "invalid run identity" >&2; exit 64; }
[[ "$SLURM_JOB_NUM_NODES" == "$EXPECTED_NODES" ]] || { echo "node-count mismatch" >&2; exit 64; }
[[ $((SLURM_JOB_NUM_NODES * 8)) == "$EXPECTED_WORLD_SIZE" ]] || { echo "world-size mismatch" >&2; exit 64; }

mkdir -p "$RUN_DIR"/{authority,epochs,identity,logs,train}
exec > >(tee -a "$RUN_DIR/logs/batch-${SLURM_JOB_ID}.out") \
     2> >(tee -a "$RUN_DIR/logs/batch-${SLURM_JOB_ID}.err" >&2)
cd "$REPO"
export FRONTIER_RCCL_ENV=${FRONTIER_RCCL_ENV:-recommended}
export FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}
export FRONTIER_RUNTIME_PROFILE=${FRONTIER_RUNTIME_PROFILE:-olcf-rccl-debug}
export REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}
# Canonical authority for module stack and Python 3.12.
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN REPO
frontier_require_requested_rccl_net_plugin "e97-4b batch parent" > "$RUN_DIR/identity/rccl-plugin-${SLURM_JOB_ID}.txt"

SOURCE_SHA=$(git rev-parse HEAD)
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
printf '%s\n' "$SOURCE_SHA" > "$RUN_DIR/identity/source-${SLURM_JOB_ID}.txt"
printf '%s\n' "$SLURM_JOB_NODELIST" > "$RUN_DIR/identity/nodelist-${SLURM_JOB_ID}.txt"
scontrol_record="$RUN_DIR/identity/scontrol-${SLURM_JOB_ID}-running.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$scontrol_record"
grep -E "Partition=${EXPECTED_PARTITION}([[:space:]]|$)" "$scontrol_record" >/dev/null
grep -E "QOS=${EXPECTED_QOS}([[:space:]]|$)" "$scontrol_record" >/dev/null
grep -E "TimeLimit=${EXPECTED_TIME_LIMIT}([[:space:]]|$)" "$scontrol_record" >/dev/null
squeue -j "$SLURM_JOB_ID" -h -o '%i|%T|%N|%P|%q|%R' \
  | tee "$RUN_DIR/identity/squeue-${SLURM_JOB_ID}-running.txt"
grep -F "|$EXPECTED_PARTITION|$EXPECTED_QOS|" \
  "$RUN_DIR/identity/squeue-${SLURM_JOB_ID}-running.txt" >/dev/null || {
    echo "live scheduler Partition/QOS mismatch" >&2; exit 64;
  }

identity_config="$RUN_DIR/identity/config.sha256"
if [[ -e "$identity_config" ]]; then
  [[ $(<"$identity_config") == "$CONFIG_SHA" ]] || { echo "run config identity drift" >&2; exit 65; }
else
  printf '%s\n' "$CONFIG_SHA" > "$identity_config"
fi
identity_world="$RUN_DIR/identity/world-size.txt"
if [[ -e "$identity_world" ]]; then
  [[ $(<"$identity_world") == "$EXPECTED_WORLD_SIZE" ]] || { echo "fixed-world drift" >&2; exit 65; }
else
  printf '%s\n' "$EXPECTED_WORLD_SIZE" > "$identity_world"
fi

readarray -t cfg < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import json,sys
v=json.load(open(sys.argv[1],encoding='utf-8'))
print(v['data']['path']); print(v['data']['expected_bytes'])
print(v['data']['tokenizer_sha256']); print(v['training']['learning_rate'])
print(v['target_steps']); print(v['training']['batch_size_per_rank'])
print(v['training']['diloco_k']); print(v['training']['save_every'])
PY
)
DATA=${cfg[0]}; EXPECTED_DATA_BYTES=${cfg[1]}; TOKENIZER_SHA256=${cfg[2]}
LR=${cfg[3]}; TARGET_STEPS=${cfg[4]}; BATCH_SIZE=${cfg[5]}
DILOCO_K=${cfg[6]}; SAVE_EVERY=${cfg[7]}
[[ -r "$DATA" && $(stat -c %s "$DATA") == "$EXPECTED_DATA_BYTES" ]] || {
  echo "reviewed corpus path/size mismatch" >&2; exit 66;
}

# The counter sampler requires a true corpus content digest. Compute it once in
# the allocation and atomically cache a receipt; later attended epochs reuse it.
CORPUS_RECEIPT=${CORPUS_RECEIPT:-$RUN_DIR/authority/corpus-sha256.json}
mkdir -p "$(dirname "$CORPUS_RECEIPT")"
if [[ ! -e "$CORPUS_RECEIPT" ]]; then
  echo "[data-authority] hashing $EXPECTED_DATA_BYTES corpus bytes"
  corpus_sha=$(sha256sum "$DATA" | awk '{print $1}')
  "$PYTHON_BIN" - "$CORPUS_RECEIPT" "$DATA" "$EXPECTED_DATA_BYTES" "$corpus_sha" <<'PY'
import json,os,sys,tempfile,time
path,data,size,digest=sys.argv[1:]
if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise SystemExit(66)
v={'schema':'emender-corpus-sha256-v1','path':data,'bytes':int(size),'sha256':digest,'created_unix':time.time()}
fd,tmp=tempfile.mkstemp(prefix='.corpus.',suffix='.tmp',dir=os.path.dirname(path),text=True)
with os.fdopen(fd,'w') as f:
 json.dump(v,f,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
fi
CORPUS_SHA=$("$PYTHON_BIN" - "$CORPUS_RECEIPT" "$DATA" "$EXPECTED_DATA_BYTES" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); assert v['schema']=='emender-corpus-sha256-v1'
assert v['path']==sys.argv[2] and v['bytes']==int(sys.argv[3]); d=v['sha256']
assert len(d)==64 and all(c in '0123456789abcdef' for c in d); print(d)
PY
)

if [[ "$RUN_MODE" == smoke ]]; then
  TARGET_STEPS=${SMOKE_STEPS:-4}; DILOCO_K=${SMOKE_DILOCO_K:-4}; SAVE_EVERY=$DILOCO_K
  LOG_EVERY=1; TRAIN_MINUTES=0
elif [[ "$RUN_MODE" == production ]]; then
  [[ "$EXPECTED_NODES" == 32 && "$EXPECTED_WORLD_SIZE" == 256 ]] || {
    echo "production authority is fixed at 32 nodes / 256 ranks" >&2; exit 64;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=${TRAIN_MINUTES:-360}
else
  echo "RUN_MODE must be smoke or production" >&2; exit 64
fi
(( SAVE_EVERY % DILOCO_K == 0 )) || { echo "checkpoint cadence is not K-aligned" >&2; exit 64; }

export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
export MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=7 NDM_PIN_TRITON_AUTOTUNE=1
export FI_MR_CACHE_MONITOR=kdreg2 FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_RDZV_PROTO=alt_read
export NCCL_NET_GDR_LEVEL=3 NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_PLUGIN=librccl-net.so NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=240

ARGS=(
 --data "$DATA" --tokenizer p50k_base --bf16 --batch_size "$BATCH_SIZE" --chunk_size 2048
 --sampler_schema emender-byte-window-counter-v1 --sampler_corpus_sha256 "$CORPUS_SHA"
 --sampler_tokenizer_sha256 "$TOKENIZER_SHA256" --sampler_key 42
 --sampler_data_world_size "$EXPECTED_WORLD_SIZE"
 --optimizer schedulefree --seed 42 --lr "$LR" --weight_decay 0.01 --warmup_steps 0
 --steps "$TARGET_STEPS" --timer_after_compile_warmup --compile_warmup_steps 1
 --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" --keep_checkpoints 3
 --walltime_final_checkpoint_margin_seconds 900 --walltime_check_every 40
 --distributed_health_check_every 40
 --diloco --diloco_k "$DILOCO_K" --diloco_outer_optimizer avg
 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --diloco_export_basis x --diloco_island_size 1
 --diloco_merge_topology hierarchical --diloco_merge_group_size 4
 --diloco_merge_group_create_barrier_every 8 --diloco_merge_completion_barrier 1
 --diloco_merge_bucket_numel 67108864
 --level E97 --params 4b --dim 3840 --depth 18 --n_heads 60 --n_state 64
 --expansion 1.0 --use_gate 1 --gate_activation silu --linear_state 0
 --use_write_gate 0 --e88_value_residual 0 --mlp_ratio 2.5 --mlp_multiple 64 --use_triton 1
 --gradient_checkpointing --gradient_checkpoint_group_size 2 --checkpoint_interval 16
 --projection_chunk_size 512 --loss_chunk_size 128 --grad_accum 1 --grad_clip 1.0
 --output "$RUN_DIR" --exact_output_dir "$RUN_DIR/train"
)
(( TRAIN_MINUTES == 0 )) || ARGS+=(--train_minutes "$TRAIN_MINUTES")
if [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]]; then
  ARGS+=(--resume "$RUN_DIR/train/latest.pt")
elif [[ -e "$RUN_DIR/train/latest.pt" || -L "$RUN_DIR/train/latest.pt" ]]; then
  echo "stable latest.pt exists but is not a readable symlink" >&2; exit 65
fi
printf '%q ' "$PYTHON_BIN" -u train.py "${ARGS[@]}" > "$RUN_DIR/identity/train-command-${SLURM_JOB_ID}.txt"
printf '\n' >> "$RUN_DIR/identity/train-command-${SLURM_JOB_ID}.txt"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=$((20000 + (10#$SLURM_JOB_ID % 40000)))
export MASTER_ADDR MASTER_PORT
CHILD_PID=""
STOP_REQUESTED=0
forward_stop() {
  STOP_REQUESTED=1
  echo "[e97-4b] batch signal received; forwarding TERM to child process group" >&2
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM -- "-$CHILD_PID" 2>/dev/null || kill -TERM "$CHILD_PID" 2>/dev/null || true
  fi
}
trap forward_stop TERM HUP INT
set +e
setsid srun --exact --kill-on-bad-exit=1 --wait=60 \
  -N"$EXPECTED_NODES" -n"$EXPECTED_WORLD_SIZE" --ntasks-per-node=8 \
  --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT" \
  bash -lc 'source "$REPO/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; frontier_require_requested_rccl_net_plugin "4B child rank ${SLURM_PROCID}" >/dev/null; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; exec "$PYTHON_BIN" -u train.py "$@"' \
  bash "${ARGS[@]}" > "$RUN_DIR/logs/train-${SLURM_JOB_ID}.out" \
  2> "$RUN_DIR/logs/train-${SLURM_JOB_ID}.err" &
CHILD_PID=$!
while true; do
  wait "$CHILD_PID"
  rc=$?
  # Bash interrupts wait after running a signal trap even while the child is
  # still finalizing. Re-enter wait so Slurm's five-minute warning can drive the
  # normal final merge/checkpoint path instead of making the batch parent exit.
  if (( STOP_REQUESTED == 1 )) && kill -0 "$CHILD_PID" 2>/dev/null; then
    continue
  fi
  break
done
CHILD_PID=""
set -e
printf '%s|%s|%s|%s\n' "$SLURM_JOB_ID" "$rc" "$EXPECTED_PARTITION" "$EXPECTED_QOS" \
  > "$RUN_DIR/identity/payload-exit-${SLURM_JOB_ID}.txt"
if (( rc == 0 )); then
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "successful child did not publish readable latest.pt" >&2; exit 70;
  }
fi
exit "$rc"
