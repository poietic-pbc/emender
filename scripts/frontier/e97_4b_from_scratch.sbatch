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
EXPECTED_TARGET_STEPS=${EXPECTED_TARGET_STEPS:-0}
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

identity_source="$RUN_DIR/identity/source.sha256"
if [[ -e "$identity_source" ]]; then
  [[ $(<"$identity_source") == "$SOURCE_SHA" ]] || { echo "run source identity drift" >&2; exit 65; }
else
  printf '%s\n' "$SOURCE_SHA" > "$identity_source"
fi
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
print(v['data']['tokenizer_sha256']); print(v['data']['corpus_sha256'])
print(v['training']['learning_rate']); print(v['target_steps'])
print(v['training']['batch_size_per_rank']); print(v['training']['diloco_k'])
print(v['training']['save_every']); print(v['bootstrap_smoke_steps'])
print(v['production_train_minutes']); print(v['debug_continuation_train_minutes'])
print(v['data']['sampler_schema']); print(v['data']['sampler_key'])
print(v['data'].get('sampler_stream_origin_accepted_tokens', 0))
print(v.get('seed', {}).get('source_step', 0))
print(v.get('seed', {}).get('source_total_tokens', 0))
print(v['training'].get('diloco_island_size', 1))
PY
)
DATA=${cfg[0]}; EXPECTED_DATA_BYTES=${cfg[1]}; TOKENIZER_SHA256=${cfg[2]}
EXPECTED_CORPUS_SHA=${cfg[3]}; LR=${cfg[4]}; TARGET_STEPS=${cfg[5]}
BATCH_SIZE=${cfg[6]}; DILOCO_K=${cfg[7]}; SAVE_EVERY=${cfg[8]}
BOOTSTRAP_STEPS=${cfg[9]}; PRODUCTION_TRAIN_MINUTES=${cfg[10]}
DEBUG_CONTINUATION_TRAIN_MINUTES=${cfg[11]}; SAMPLER_SCHEMA=${cfg[12]}
SAMPLER_KEY=${cfg[13]}; SAMPLER_STREAM_ORIGIN=${cfg[14]}
SEED_SOURCE_STEP=${cfg[15]}; SEED_SOURCE_TOKENS=${cfg[16]}
DILOCO_ISLAND_SIZE=${cfg[17]}
[[ -r "$DATA" && $(stat -c %s "$DATA") == "$EXPECTED_DATA_BYTES" ]] || {
  echo "reviewed corpus path/size mismatch" >&2; exit 66;
}

# The physical 1 TB object already has a reviewed SHA-256 authority. Materialize
# that immutable identity without spending the bootstrap allocation rehashing it.
CORPUS_RECEIPT=${CORPUS_RECEIPT:-$RUN_DIR/authority/corpus-sha256.json}
mkdir -p "$(dirname "$CORPUS_RECEIPT")"
"$PYTHON_BIN" - "$CORPUS_RECEIPT" "$DATA" "$EXPECTED_DATA_BYTES" "$EXPECTED_CORPUS_SHA" <<'PY'
import json,os,sys,tempfile,time
path,data,size,digest=sys.argv[1:]
if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise SystemExit(66)
expected={'schema':'emender-corpus-sha256-v1','path':data,'bytes':int(size),'sha256':digest}
if os.path.exists(path):
 v=json.load(open(path,encoding='utf-8'))
 if any(v.get(k)!=value for k,value in expected.items()): raise SystemExit(66)
else:
 v={**expected,'created_unix':time.time(),'authority':'committed-reviewed-physical-sha256'}
 fd,tmp=tempfile.mkstemp(prefix='.corpus.',suffix='.tmp',dir=os.path.dirname(path),text=True)
 with os.fdopen(fd,'w') as f:
  json.dump(v,f,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
print(digest)
PY
CORPUS_SHA=$EXPECTED_CORPUS_SHA

if [[ "$RUN_MODE" == smoke ]]; then
  TARGET_STEPS=${SMOKE_STEPS:-4}; DILOCO_K=${SMOKE_DILOCO_K:-4}; SAVE_EVERY=$DILOCO_K
  LOG_EVERY=1; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == rung ]]; then
  [[ "$EXPECTED_NODES" == 4 && "$EXPECTED_WORLD_SIZE" == 32 ]] || {
    echo "rung authority is fixed at 4 nodes / 32 ranks" >&2; exit 64;
  }
  TARGET_STEPS=$BOOTSTRAP_STEPS
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" =~ ^probe_b(2|4|5|6|8)$ ]]; then
  [[ "$EXPECTED_NODES" == 2 && "$EXPECTED_WORLD_SIZE" == 16 ]] || {
    echo "batch probes are fixed at 2 nodes / 16 ranks" >&2; exit 64;
  }
  BATCH_SIZE=${RUN_MODE#probe_b}
  DILOCO_K=$((128 / BATCH_SIZE))
  SAVE_EVERY=$((2 * DILOCO_K))
  TARGET_STEPS=$SAVE_EVERY
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == seed_import_32n_canary ]]; then
  [[ "$EXPECTED_NODES" == 32 && "$EXPECTED_WORLD_SIZE" == 256 ]] || {
    echo "seed-import canary is fixed at 32 nodes / 256 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 1 && "$DILOCO_K" == 32 && "$SAVE_EVERY" == 256 ]] || {
    echo "seed-import canary config must be B1/K32/save256" >&2; exit 64;
  }
  [[ "$SAMPLER_SCHEMA" == emender-byte-window-counter-v2 && \
     "$SAMPLER_STREAM_ORIGIN" == "$SEED_SOURCE_TOKENS" && \
     "$TARGET_STEPS" == $((SEED_SOURCE_STEP + 512)) ]] || {
    echo "seed-import canary requires an exact 512-step counter-v2 phase" >&2; exit 64;
  }
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "seed-import canary requires a readable imported seed latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == seed_continuation_32n_debug ]]; then
  [[ "$EXPECTED_NODES" == 32 && "$EXPECTED_WORLD_SIZE" == 256 ]] || {
    echo "seed continuation is fixed at 32 nodes / 256 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 1 && "$DILOCO_K" == 32 && "$SAVE_EVERY" == 256 && \
     "$SAMPLER_SCHEMA" == emender-byte-window-counter-v2 && \
     "$TARGET_STEPS" == 15360 ]] || {
    echo "seed continuation config must be B1/K32/save256/counter-v2/step15360" >&2; exit 64;
  }
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "seed continuation requires a readable parent latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == seed_scale_64n_canary ]]; then
  [[ "$EXPECTED_NODES" == 64 && "$EXPECTED_WORLD_SIZE" == 512 ]] || {
    echo "seed-scale canary is fixed at 64 nodes / 512 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 1 && "$DILOCO_K" == 32 && "$SAVE_EVERY" == 256 && \
     "$SAMPLER_SCHEMA" == emender-byte-window-counter-v2 && \
     "$SAMPLER_STREAM_ORIGIN" == "$SEED_SOURCE_TOKENS" && \
     "$TARGET_STEPS" == $((SEED_SOURCE_STEP + 1024)) ]] || {
    echo "seed-scale canary requires B1/K32 and an exact 1024-step counter-v2 phase" >&2; exit 64;
  }
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "seed-scale canary requires a readable parent latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == hybrid_ddp_8n_canary ]]; then
  [[ "$EXPECTED_NODES" == 8 && "$EXPECTED_WORLD_SIZE" == 64 ]] || {
    echo "hybrid-DDP canary is fixed at 8 nodes / 64 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 4 && "$DILOCO_K" == 32 && "$DILOCO_ISLAND_SIZE" == 8 && \
     "$SAVE_EVERY" == 256 && "$SAMPLER_SCHEMA" == emender-byte-window-counter-v2 && \
     "$SAMPLER_STREAM_ORIGIN" == "$SEED_SOURCE_TOKENS" && \
     "$TARGET_STEPS" == $((SEED_SOURCE_STEP + 768)) ]] || {
    echo "hybrid-DDP canary requires 8 islands / B32 effective / K32 / 768 steps" >&2; exit 64;
  }
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "hybrid-DDP canary requires a readable parent latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == hybrid_ddp_96n_debug ]]; then
  [[ "$EXPECTED_NODES" == 96 && "$EXPECTED_WORLD_SIZE" == 768 ]] || {
    echo "96-node campaign debug is fixed at 96 nodes / 768 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 4 && "$DILOCO_K" == 32 && "$DILOCO_ISLAND_SIZE" == 8 && \
     "$SAVE_EVERY" == 256 && "$SAMPLER_SCHEMA" == emender-byte-window-counter-v2 && \
     "$SAMPLER_STREAM_ORIGIN" == "$SEED_SOURCE_TOKENS" && \
     "$EXPECTED_TARGET_STEPS" == 18176 ]] || {
    echo "96-node debug requires B4/island8/K32 and target step 18176" >&2; exit 64;
  }
  TARGET_STEPS=$EXPECTED_TARGET_STEPS
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "96-node debug requires a readable parent latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == hybrid_ddp_96n_production || "$RUN_MODE" == hybrid_ddp_96n_short_production ]]; then
  [[ "$EXPECTED_NODES" == 96 && "$EXPECTED_WORLD_SIZE" == 768 ]] || {
    echo "96-node production is fixed at 96 nodes / 768 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 4 && "$DILOCO_K" == 32 && "$DILOCO_ISLAND_SIZE" == 8 && \
     "$SAVE_EVERY" == 256 && "$SAMPLER_SCHEMA" == emender-byte-window-counter-v2 ]] || {
    echo "96-node production requires B4/island8/K32/counter-v2" >&2; exit 64;
  }
  if [[ "$RUN_MODE" == hybrid_ddp_96n_production ]]; then
    case "$EXPECTED_TARGET_STEPS" in 21504|24832|28160|31488) ;; *)
      echo "invalid 96-node production campaign target" >&2; exit 64;; esac
    FINAL_MARGIN_SECONDS=600
  else
    (( EXPECTED_TARGET_STEPS >= 19200 && EXPECTED_TARGET_STEPS <= 31488 && \
       (EXPECTED_TARGET_STEPS - 18176) % 1024 == 0 )) || {
      echo "invalid 96-node short-production campaign target" >&2; exit 64;
    }
    FINAL_MARGIN_SECONDS=300
  fi
  TARGET_STEPS=$EXPECTED_TARGET_STEPS
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "96-node production requires a readable inspected parent latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0
elif [[ "$RUN_MODE" == matched_clock_32n ]]; then
  [[ "$EXPECTED_NODES" == 32 && "$EXPECTED_WORLD_SIZE" == 256 ]] || {
    echo "matched-clock qualification is fixed at 32 nodes / 256 ranks" >&2; exit 64;
  }
  [[ "$BATCH_SIZE" == 1 && "$DILOCO_K" == 32 && "$SAVE_EVERY" == 256 && "$TARGET_STEPS" == 2048 ]] || {
    echo "matched-clock config must be B1/K32/save256/2048 steps" >&2; exit 64;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == bootstrap ]]; then
  [[ "$EXPECTED_NODES" == 256 && "$EXPECTED_WORLD_SIZE" == 2048 ]] || {
    echo "bootstrap authority is fixed at 256 nodes / 2048 ranks" >&2; exit 64;
  }
  TARGET_STEPS=$BOOTSTRAP_STEPS
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=0; FINAL_MARGIN_SECONDS=300
elif [[ "$RUN_MODE" == debug_continuation ]]; then
  [[ "$EXPECTED_NODES" == 256 && "$EXPECTED_WORLD_SIZE" == 2048 ]] || {
    echo "debug continuation authority is fixed at 256 nodes / 2048 ranks" >&2; exit 64;
  }
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "debug continuation requires readable latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=${TRAIN_MINUTES:-$DEBUG_CONTINUATION_TRAIN_MINUTES}
  FINAL_MARGIN_SECONDS=600
elif [[ "$RUN_MODE" =~ ^production(_4h|_6h|_8h)?$ ]]; then
  [[ "$EXPECTED_NODES" == 256 && "$EXPECTED_WORLD_SIZE" == 2048 ]] || {
    echo "production authority is fixed at 256 nodes / 2048 ranks" >&2; exit 64;
  }
  [[ -L "$RUN_DIR/train/latest.pt" && -r "$RUN_DIR/train/latest.pt" ]] || {
    echo "production requires readable latest.pt" >&2; exit 65;
  }
  LOG_EVERY=${LOG_EVERY:-4}; TRAIN_MINUTES=${TRAIN_MINUTES:-$PRODUCTION_TRAIN_MINUTES}
  FINAL_MARGIN_SECONDS=600
else
  echo "invalid RUN_MODE" >&2; exit 64
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
 --sampler_schema "$SAMPLER_SCHEMA" --sampler_corpus_sha256 "$CORPUS_SHA"
 --sampler_tokenizer_sha256 "$TOKENIZER_SHA256" --sampler_key "$SAMPLER_KEY"
 --sampler_data_world_size "$EXPECTED_WORLD_SIZE"
 --optimizer schedulefree --seed 42 --lr "$LR" --weight_decay 0.01 --warmup_steps 0
 --steps "$TARGET_STEPS" --timer_after_compile_warmup --compile_warmup_steps 1
 --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" --keep_checkpoints 3
 --walltime_final_checkpoint_margin_seconds "$FINAL_MARGIN_SECONDS" --walltime_check_every 8
 --distributed_health_check_every 40
 --diloco --diloco_k "$DILOCO_K" --diloco_outer_optimizer avg
 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --diloco_export_basis x
 --diloco_island_size "$DILOCO_ISLAND_SIZE"
 --diloco_merge_topology hierarchical --diloco_merge_group_size 4
 --diloco_merge_group_create_barrier_every 8 --diloco_merge_completion_barrier 1
 --diloco_merge_bucket_numel 67108864
 --level E97 --params 4b --dim 3840 --depth 18 --n_heads 60 --n_state 64
 --expansion 1.0 --use_gate 1 --gate_activation silu --linear_state 0
 --use_write_gate 0 --e88_value_residual 0 --mlp_ratio 2.5 --mlp_multiple 64 --use_triton 1
 --checkpoint_interval 16
 --projection_chunk_size 512 --loss_chunk_size 128 --grad_accum 1 --grad_clip 1.0
 --output "$RUN_DIR" --exact_output_dir "$RUN_DIR/train"
)
(( SAMPLER_STREAM_ORIGIN == 0 )) || \
  ARGS+=(--sampler_stream_origin_accepted_tokens "$SAMPLER_STREAM_ORIGIN")
if [[ "$RUN_MODE" == seed_import_32n_canary || "$RUN_MODE" == seed_scale_64n_canary || \
      "$RUN_MODE" == hybrid_ddp_8n_canary || "$RUN_MODE" == hybrid_ddp_96n_debug ]]; then
  ARGS+=(--sampler_transition_from_counter)
fi
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
  # Signalling srun cancels every task before train.py can merge and publish.
  # Instead publish the same atomic shared stop request that train.py polls at
  # collective-safe boundaries. Slurm retains the hard allocation deadline.
  local request="$RUN_DIR/train/.final_checkpoint_request"
  local tmp="${request}.batch-${SLURM_JOB_ID}.tmp"
  echo "[e97-4b] batch signal received; requesting graceful final checkpoint" >&2
  printf '{"reason":"batch_signal","step":-1,"time":%s}\n' "$(date +%s)" > "$tmp"
  mv -f "$tmp" "$request"
}
trap forward_stop TERM HUP INT
set +e
setsid srun --exact --kill-on-bad-exit=1 --wait=60 \
  -N"$EXPECTED_NODES" -n"$EXPECTED_WORLD_SIZE" --ntasks-per-node=8 \
  --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT" \
  bash -lc 'source "$REPO/scripts/frontier/frontier_runtime_env.sh"; frontier_activate_emender_conda_env; frontier_assert_emender_conda_env; frontier_require_requested_rccl_net_plugin "4B child rank ${SLURM_PROCID}" >/dev/null; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; export TRITON_CACHE_DIR=/tmp/e97-4b-${SLURM_JOB_ID}-${SLURM_PROCID}; rm -rf "$TRITON_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR"; exec "$PYTHON_BIN" -u train.py "$@"' \
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
