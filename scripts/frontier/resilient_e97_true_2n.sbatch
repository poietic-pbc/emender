#!/bin/bash
# True resilient E97 bounded-lag v2 gate: one manager and eight real trainers
# per node.  ADR-002 authorizes exactly the 2-node batch/debug rung only.
# Managers and trainers are separate srun steps; --overlap and --no-kill keep a
# failed step from cancelling healthy siblings. There are no sentinel ranks;
# every GPU step is a real trainer. The debug QoS bounds this G3 admission
# rung; it does not permit a Python dense transport or an unattested binary.
#SBATCH -A bif148
#SBATCH -J resilient-e97-true-2n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH -t 00:20:00
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/frontier/trainpy_async_quorum/%x-%j.out
#SBATCH -e logs/frontier/trainpy_async_quorum/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
APPROVED_EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
: "${EMENDER_CONDA_ENV:?immutable submit payload must export EMENDER_CONDA_ENV=$APPROVED_EMENDER_CONDA_ENV}"
[[ $(realpath "$EMENDER_CONDA_ENV") == $(realpath "$APPROVED_EMENDER_CONDA_ENV") ]] || {
  echo "EMENDER_CONDA_ENV must resolve to $APPROVED_EMENDER_CONDA_ENV" >&2; exit 64;
}
RESILIENT_E97_NODE_COUNT=${RESILIENT_E97_NODE_COUNT:-2}
[[ $RESILIENT_E97_NODE_COUNT == 2 ]] || {
  echo "async-decoupled-v2 qualification permits exactly two nodes" >&2; exit 64;
}
export EMENDER_CONDA_ENV
source "$REPO/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env
frontier_assert_emender_conda_env
TRAIN_PYTHON_BIN=$(command -v python || command -v python3)
: "${TRAIN_PYTHON_BIN:?Frontier runtime activation did not select TRAIN_PYTHON_BIN}"
[[ $(realpath "$TRAIN_PYTHON_BIN") == $(realpath "$APPROVED_EMENDER_CONDA_ENV/bin/python") ]] || {
  echo "activated Python is not the approved immutable runtime" >&2; exit 64;
}
RUNTIME_IDENTITY=$($TRAIN_PYTHON_BIN - <<'PY'
import json, platform, sys
import torch, triton
identity = {
    "executable": sys.executable,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "triton": triton.__version__,
}
expected = {
    "python": "3.12.13", "torch": "2.10.0+rocm7.1",
    "torch_hip": "7.1.25424", "triton": "3.6.0",
}
if any(identity[key] != value for key, value in expected.items()):
    raise SystemExit("runtime identity mismatch: " + json.dumps(identity, sort_keys=True))
print(json.dumps(identity, sort_keys=True))
PY
)
printf 'resilient_e97_runtime_identity=%s\n' "$RUNTIME_IDENTITY"
mkdir -p "${RUN_DIR:?set immutable run directory}"
printf '%s\n' "$RUNTIME_IDENTITY" > "$RUN_DIR/runtime-identity.json"

# The full-layout real-model launcher has no Python-TCP escape hatch.  Bind the
# exact local library, transport library, and service binary to the retained
# exact-code G2 artifact before the supervisor may acquire a lease or admit a
# model-bearing trainer.
DILOCO_DATAPLANE=${DILOCO_DATAPLANE:-native-cxi}
[[ $DILOCO_DATAPLANE == native-cxi ]] || {
  echo "full-layout Frontier launcher requires DILOCO_DATAPLANE=native-cxi; Python TCP is debug-fixture only" >&2
  exit 64
}
FI_PROVIDER=${FI_PROVIDER:-cxi}
[[ $FI_PROVIDER == cxi ]] || {
  echo "native-cxi production requires exact FI_PROVIDER=cxi" >&2
  exit 64
}
: "${NDP_BUILD_MANIFEST:?set exact installed native build manifest}"
: "${NDP_FULL_LAYOUT_GATE_JSON:?set retained exact-code G2 full-layout gate}"
NDP_LAUNCH_ATTESTATION="$RUN_DIR/native-dataplane-launch-attestation.json"
"$TRAIN_PYTHON_BIN" "$REPO/scripts/frontier/attest_native_dataplane.py" verify \
  --backend "$DILOCO_DATAPLANE" --production --full-layout \
  --build-manifest "$NDP_BUILD_MANIFEST" \
  --gate-json "$NDP_FULL_LAYOUT_GATE_JSON" --source-root "$REPO" \
  --output "$NDP_LAUNCH_ATTESTATION"
export DILOCO_DATAPLANE FI_PROVIDER NDP_BUILD_MANIFEST \
  NDP_FULL_LAYOUT_GATE_JSON NDP_LAUNCH_ATTESTATION
readarray -t NDP_ARTIFACT_PATHS < <("$TRAIN_PYTHON_BIN" - "$NDP_BUILD_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys
manifest = Path(sys.argv[1]).resolve()
value = json.loads(manifest.read_text())
for name in ("service_binary", "local_library", "transport_library"):
    path = (manifest.parent / value["artifacts"][name]["path"]).resolve()
    path.relative_to(manifest.parent)
    print(path)
PY
)
NDP_SERVICE_BINARY=${NDP_ARTIFACT_PATHS[0]}
EMENDER_NDP_LIBRARY=${NDP_ARTIFACT_PATHS[1]}
EMENDER_NDP_TRANSPORT_LIBRARY=${NDP_ARTIFACT_PATHS[2]}
[[ -x $NDP_SERVICE_BINARY && -f $EMENDER_NDP_LIBRARY && -f $EMENDER_NDP_TRANSPORT_LIBRARY ]] || {
  echo "attested native service/client artifacts are missing" >&2; exit 64;
}
export EMENDER_NDP_LIBRARY EMENDER_NDP_TRANSPORT_LIBRARY
export EMENDER_NDP_MAX_SHARED_BYTES=${RESILIENT_E97_MAX_SHARED_BYTES:-68719476736}
export EMENDER_NDP_INTERMEDIATE_F64=1
export NDP_SERVICE_COMMAND="$NDP_SERVICE_BINARY --provider cxi --require-provider cxi --production --serve --domain cxi0 --payload-max ${RESILIENT_E97_BULK_CHUNK_BYTES:-67108864} --resident-limit ${EMENDER_NDP_MAX_SHARED_BYTES} --tx-slots 4 --rx-slots 4"

[[ ${SLURM_JOB_NUM_NODES:?} == "$RESILIENT_E97_NODE_COUNT" ]] || {
  echo "Slurm node count differs from explicit resilient-pool capacity" >&2; exit 64;
}
[[ ${SLURM_JOB_PARTITION:?} == batch ]] || { echo "requires batch partition" >&2; exit 64; }
[[ ${SLURM_JOB_QOS:?} == debug ]] || { echo "requires debug QoS" >&2; exit 64; }
if [[ ${RESILIENT_E97_STARTUP_SMOKE:-0} == 1 ]]; then
  # Frontier does not consistently export SLURM_TIMELIMIT into batch jobs.
  # The submitter must therefore bind the requested value into the immutable
  # payload; when Slurm does expose its value, require the two to agree.
  [[ ${RESILIENT_E97_REQUESTED_WALLTIME:-} == 00:20:00 ]] || {
    echo "startup smoke requires exactly 00:20:00" >&2; exit 64;
  }
  requested_minutes=20
  [[ -z ${SLURM_TIMELIMIT:-} || ${SLURM_TIMELIMIT} == ${RESILIENT_E97_REQUESTED_WALLTIME} || ${SLURM_TIMELIMIT} == ${requested_minutes} ]] || {
    echo "startup smoke Slurm walltime disagrees with requested ${RESILIENT_E97_REQUESTED_WALLTIME}" >&2; exit 64;
  }
  [[ ${RESILIENT_E97_GENERATIONS:-1} == 1 ]] || {
    echo "startup smoke requires exactly one finalized generation" >&2; exit 64;
  }
  [[ -z ${RESILIENT_E97_INJECT_TRAINER:-}${RESILIENT_E97_INJECT_MANAGER:-}${RESILIENT_E97_INJECT_NODE_STEP:-}${RESILIENT_E97_INJECT_NATIVE_SERVICE:-}${RESILIENT_E97_DELAY_READY:-} ]] || {
    echo "startup smoke forbids failure injection" >&2; exit 64;
  }
else
  if [[ -n ${RESILIENT_E97_ACCEPTANCE_MANIFEST:-} ]]; then
    [[ ${RESILIENT_E97_NODE_COUNT} == 2 ]] || {
      echo "exact acceptance manifest requires exactly two nodes" >&2; exit 64;
    }
    [[ ${SLURM_TIMELIMIT:-02:00:00} == 02:00:00 || ${SLURM_TIMELIMIT:-} == 120 ]] || {
      echo "exact acceptance phase requires the honest 02:00:00 debug walltime" >&2; exit 64;
    }
    : "${RESILIENT_E97_ACCEPTANCE_PHASE:?acceptance manifest requires a named phase}"
  else
    [[ ${SLURM_TIMELIMIT:-00:20:00} == 00:20:00 || ${SLURM_TIMELIMIT:-} == 20 || ${SLURM_TIMELIMIT:-} == 00:30:00 || ${SLURM_TIMELIMIT:-} == 30 ]] || {
      echo "requires a bounded 00:20:00 restart/startup or 00:30:00 resilience gate" >&2; exit 64;
    }
  fi
fi
[[ ${ASYNC_LOCAL_STEPS:-40} == 40 ]] || { echo "E97 gate requires local_steps=40" >&2; exit 64; }
RUN_DIR=${RUN_DIR:?set immutable run directory}
: "${RESILIENT_E97_RUN_ID:?set fenced run identity}"
: "${RESILIENT_E97_SOURCE_ID:?set verified seed/source identity}"
: "${RESILIENT_E97_PAYLOAD_ID:?set tensor layout identity}"
: "${RESILIENT_E97_SEED_CONFIG:?set canonical immutable E97 seed config}"
: "${RESILIENT_E97_SEED_STEP:?set canonical immutable E97 seed step}"
: "${RESILIENT_E97_SEED_TOKENS:?set canonical immutable E97 seed token count}"
: "${RESILIENT_E97_SEED_SIZE:?set canonical immutable E97 seed byte size}"
: "${RESILIENT_E97_SEED_SHA256:?set canonical immutable E97 seed SHA256}"
: "${RESILIENT_E97_TRAIN_ARGS_JSON:?set approved E97 training arguments JSON}"
: "${RESILIENT_E97_DATA:?set approved token dataset}"
: "${RESILIENT_E97_TIKTOKEN_CACHE_FILE:?set verified p50k tokenizer cache file}"
APPROVED_SEED_CONFIG="$REPO/configs/frontier/e97_async_256.yaml"
[[ $(realpath "$RESILIENT_E97_SEED_CONFIG") == $(realpath "$APPROVED_SEED_CONFIG") ]] || {
  echo "seed configuration must be canonical: $APPROVED_SEED_CONFIG" >&2
  exit 64
}
readarray -t CANONICAL_SEED_IDENTITY < <("$TRAIN_PYTHON_BIN" - "$APPROVED_SEED_CONFIG" <<'PY'
import json
import sys
seed = json.load(open(sys.argv[1], encoding="utf-8"))["seed"]
for key in ("step", "tokens", "size", "sha256"):
    print(seed[key])
PY
)
[[ ${CANONICAL_SEED_IDENTITY[0]} == "$RESILIENT_E97_SEED_STEP"
  && ${CANONICAL_SEED_IDENTITY[1]} == "$RESILIENT_E97_SEED_TOKENS"
  && ${CANONICAL_SEED_IDENTITY[2]} == "$RESILIENT_E97_SEED_SIZE"
  && ${CANONICAL_SEED_IDENTITY[3]} == "$RESILIENT_E97_SEED_SHA256" ]] || {
  echo "rendered seed identity drifted from canonical config" >&2
  exit 64
}

# The submitter has already resolved both S3 authorities, verified all 7.7 GB,
# and atomically published this content-addressed cold artifact.  Compute
# nodes perform no S3/HTTP request: Slurm broadcasts the pinned files to the
# exact live-job-scoped node-local destination, then every node independently
# verifies both artifacts before any model-bearing process can start.
: "${SLURM_JOB_ID:?job-local seed materialization requires SLURM_JOB_ID}"
: "${RESILIENT_E97_SEED_CACHE:?submitter must export verified cold seed cache}"
: "${RESILIENT_E97_SEED_ATTESTATION:?submitter must export pinned authority attestation}"
: "${RESILIENT_E97_SEED_ATTESTATION_SHA256:?submitter must export authority attestation digest}"
[[ $(basename "$RESILIENT_E97_SEED_CACHE") == "sha256-$RESILIENT_E97_SEED_SHA256.pt" ]] || {
  echo "cold seed cache is not content addressed by the immutable digest" >&2
  exit 64
}
[[ $(sha256sum "$RESILIENT_E97_SEED_ATTESTATION" | awk '{print $1}') == "$RESILIENT_E97_SEED_ATTESTATION_SHA256" ]] || {
  echo "submit-side seed authority attestation digest mismatch" >&2
  exit 64
}
RESILIENT_E97_JOB_SEED_TEMPLATE=${RESILIENT_E97_JOB_SEED_TEMPLATE:-'/tmp/emender-e97-seed-${SLURM_JOB_ID}'}
[[ $RESILIENT_E97_JOB_SEED_TEMPLATE == '/tmp/emender-e97-seed-${SLURM_JOB_ID}' ]] || {
  echo "seed destination template must retain the deferred batch SLURM_JOB_ID" >&2
  exit 64
}
RESILIENT_E97_JOB_SEED_DIR=${RESILIENT_E97_JOB_SEED_TEMPLATE/'${SLURM_JOB_ID}'/"$SLURM_JOB_ID"}
[[ $RESILIENT_E97_JOB_SEED_DIR == "/tmp/emender-e97-seed-$SLURM_JOB_ID" ]] || {
  echo "seed destination must be the immutable job-local /tmp directory" >&2
  exit 64
}
RESILIENT_E97_SEED="$RESILIENT_E97_JOB_SEED_DIR/checkpoint-step-$RESILIENT_E97_SEED_STEP.pt"
RESILIENT_E97_NODE_ATTESTATION="$RESILIENT_E97_JOB_SEED_DIR/seed-bootstrap-attestation.json"
SEED_EVIDENCE_DIR="$RUN_DIR/seed-materialization"
mkdir -p "$SEED_EVIDENCE_DIR"
SBCAST=$(command -v sbcast) || {
  echo "compiled Slurm sbcast is required for seed bootstrap" >&2
  exit 64
}
[[ -x $SBCAST ]] || { echo "sbcast is not executable: $SBCAST" >&2; exit 64; }
cleanup_seed_bootstrap() {
  srun --nodes="$RESILIENT_E97_NODE_COUNT" --ntasks="$RESILIENT_E97_NODE_COUNT" \
    --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=0 \
    rm -rf -- "$RESILIENT_E97_JOB_SEED_DIR" >/dev/null 2>&1 || true
}
trap cleanup_seed_bootstrap ERR INT TERM
srun --nodes=$RESILIENT_E97_NODE_COUNT --ntasks=$RESILIENT_E97_NODE_COUNT --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=0 \
  bash -c 'set -euo pipefail; test ! -e "$1"; mkdir -m 700 "$1"' \
  bash "$RESILIENT_E97_JOB_SEED_DIR"
"$SBCAST" -f "$RESILIENT_E97_SEED_CACHE" "$RESILIENT_E97_SEED"
"$SBCAST" -f "$RESILIENT_E97_SEED_ATTESTATION" "$RESILIENT_E97_NODE_ATTESTATION"
srun --nodes=$RESILIENT_E97_NODE_COUNT --ntasks=$RESILIENT_E97_NODE_COUNT --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=0 \
  bash -c 'set -euo pipefail
node=${SLURMD_NODENAME:-$(hostname)}
"$1" "$2" --verify-local --seed-config "$3" --destination "$4" \
  --attestation "$5" --expected-job-id "$6" --runtime-manifest "$7/$node.json" \
  --expected-attestation-sha256 "$8"' \
  bash "$TRAIN_PYTHON_BIN" "$REPO/scripts/frontier/materialize_e97_s3_seed.py" \
  "$APPROVED_SEED_CONFIG" "$RESILIENT_E97_SEED" \
  "$RESILIENT_E97_NODE_ATTESTATION" "$SLURM_JOB_ID" "$SEED_EVIDENCE_DIR" \
  "$RESILIENT_E97_SEED_ATTESTATION_SHA256"
trap - ERR INT TERM
# The verified node-local seed must remain resident for every trainer in this
# allocation, but it is not authoritative after the supervisor exits.  Remove
# it on every subsequent success/failure path so a retry with the same
# externally supplied job-id cannot observe or overwrite stale bootstrap state.
trap cleanup_seed_bootstrap EXIT
export RESILIENT_E97_SEED

RESILIENT_E97_TIKTOKEN_SHA256=${RESILIENT_E97_TIKTOKEN_SHA256:-94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069}
[[ $(sha256sum "$RESILIENT_E97_TIKTOKEN_CACHE_FILE" | awk '{print $1}') == "$RESILIENT_E97_TIKTOKEN_SHA256" ]] || {
  echo "p50k tokenizer cache checksum mismatch" >&2
  exit 64
}
APPROVED_TRAIN_ARGS="$REPO/configs/frontier/e97_resilient_split_role_flat.json"
[[ $(realpath "$RESILIENT_E97_TRAIN_ARGS_JSON") == $(realpath "$APPROVED_TRAIN_ARGS") ]] || {
  echo "training arguments must be the approved flat E97 split-role configuration: $APPROVED_TRAIN_ARGS" >&2
  exit 64
}
mapfile -t ALLOCATION_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#ALLOCATION_NODES[@]} == "$RESILIENT_E97_NODE_COUNT" ]] || {
  echo "allocation discovery differs from explicit resilient-pool capacity" >&2; exit 64;
}
RESILIENT_E97_GLOBAL_QUORUM=${RESILIENT_E97_GLOBAL_QUORUM:-2}
RESILIENT_E97_GLOBAL_TOKEN_MIN=${RESILIENT_E97_GLOBAL_TOKEN_MIN:-3934080}
((RESILIENT_E97_GLOBAL_QUORUM >= 2
  && RESILIENT_E97_GLOBAL_QUORUM <= RESILIENT_E97_NODE_COUNT)) || {
  echo "global quorum must be between two and the bounded node capacity" >&2; exit 64;
}
((RESILIENT_E97_GLOBAL_TOKEN_MIN > 0)) || {
  echo "global token floor must be positive" >&2; exit 64;
}
RESILIENT_E97_COORDINATOR_HOST=${RESILIENT_E97_COORDINATOR_HOST:-${ALLOCATION_NODES[0]}}
RESILIENT_E97_INITIAL_GENERATION=${RESILIENT_E97_INITIAL_GENERATION:-0}
RESILIENT_E97_RESUME_HANDOFF=${RESILIENT_E97_RESUME_HANDOFF:-}
# Compute nodes cannot fetch tiktoken vocabulary blobs from the public network.
# Stage the immutable, checksum-verified p50k input once per node before roles
# start; all tokenizer reads after this point are node-local.
RESILIENT_E97_NODE_TIKTOKEN_CACHE=${RESILIENT_E97_NODE_TIKTOKEN_CACHE:-/tmp/resilient-e97-tiktoken-$RESILIENT_E97_PAYLOAD_ID}
srun --nodes=$RESILIENT_E97_NODE_COUNT --ntasks=$RESILIENT_E97_NODE_COUNT --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=0 \
  bash -c 'set -euo pipefail; mkdir -p "$1"; cp "$2" "$1/ec7223a39ce59f226a68acc30dc1af2788490e15"; echo "$3  $1/ec7223a39ce59f226a68acc30dc1af2788490e15" | sha256sum --check --status' \
  bash "$RESILIENT_E97_NODE_TIKTOKEN_CACHE" "$RESILIENT_E97_TIKTOKEN_CACHE_FILE" "$RESILIENT_E97_TIKTOKEN_SHA256"
export TIKTOKEN_CACHE_DIR=$RESILIENT_E97_NODE_TIKTOKEN_CACHE
ROLE="$REPO/scripts/frontier/resilient_e97_role.py"
RESILIENT_E97_BULK_ROOT=${RESILIENT_E97_BULK_ROOT:-/tmp/resilient-e97}
RESILIENT_E97_DILOCO_POLICY=${RESILIENT_E97_DILOCO_POLICY:-async-decoupled-v2.0-exp}
RESILIENT_E97_TAU_HARD=${RESILIENT_E97_TAU_HARD:-6}
RESILIENT_E97_TAU_TARGET=${RESILIENT_E97_TAU_TARGET:-2}
RESILIENT_E97_SIGMA_HARD=${RESILIENT_E97_SIGMA_HARD:-8}
RESILIENT_E97_SIGMA_TARGET=${RESILIENT_E97_SIGMA_TARGET:-2}
[[ $RESILIENT_E97_DILOCO_POLICY == async-decoupled-v2.0-exp \
   && $RESILIENT_E97_TAU_HARD == 6 && $RESILIENT_E97_TAU_TARGET == 2 \
   && $RESILIENT_E97_SIGMA_HARD == 8 && $RESILIENT_E97_SIGMA_TARGET == 2 ]] || {
  echo "rendered async-v2 policy differs from reviewed ADR-002" >&2; exit 64;
}
COMMON="--run-dir $RUN_DIR --run-id $RESILIENT_E97_RUN_ID --generations ${RESILIENT_E97_GENERATIONS:-1} --initial-generation $RESILIENT_E97_INITIAL_GENERATION --local-steps 40 --node-count $RESILIENT_E97_NODE_COUNT --global-quorum $RESILIENT_E97_GLOBAL_QUORUM --global-token-min $RESILIENT_E97_GLOBAL_TOKEN_MIN --coordinator-host $RESILIENT_E97_COORDINATOR_HOST --coordinator-port ${RESILIENT_E97_COORDINATOR_PORT:-29571} --deadline-s ${RESILIENT_E97_GENERATION_DEADLINE_S:-600} --source-id $RESILIENT_E97_SOURCE_ID --payload-id $RESILIENT_E97_PAYLOAD_ID --code-id ${RESILIENT_E97_CODE_ID:-unknown} --coordinator-epoch ${RESILIENT_E97_COORDINATOR_EPOCH:-1} --bulk-root $RESILIENT_E97_BULK_ROOT --max-spool-bytes ${RESILIENT_E97_MAX_SPOOL_BYTES:-68719476736} --bulk-chunk-bytes ${RESILIENT_E97_BULK_CHUNK_BYTES:-67108864} --local-spool-chunk-bytes ${RESILIENT_E97_LOCAL_SPOOL_CHUNK_BYTES:-67108864} --dataplane-backend $DILOCO_DATAPLANE --native-build-manifest $NDP_BUILD_MANIFEST --native-gate-json $NDP_FULL_LAYOUT_GATE_JSON --diloco-policy $RESILIENT_E97_DILOCO_POLICY --tau-hard $RESILIENT_E97_TAU_HARD --tau-target $RESILIENT_E97_TAU_TARGET --sigma-hard $RESILIENT_E97_SIGMA_HARD --sigma-target $RESILIENT_E97_SIGMA_TARGET --eta-outer 0.5"
export RESILIENT_E97_MANAGER_COMMAND="$TRAIN_PYTHON_BIN $ROLE manager $COMMON --local-quorum 8"
RESUME_ARGS=()
[[ -z $RESILIENT_E97_RESUME_HANDOFF ]] || RESUME_ARGS=(--resume-handoff "$RESILIENT_E97_RESUME_HANDOFF")
RESUME_COMMAND=
if ((${#RESUME_ARGS[@]})); then
  printf -v RESUME_COMMAND ' %q' "${RESUME_ARGS[@]}"
fi
export RESILIENT_E97_TRAINER_COMMAND="$TRAIN_PYTHON_BIN $ROLE trainer $COMMON --seed $RESILIENT_E97_SEED --train-args-json $RESILIENT_E97_TRAIN_ARGS_JSON --data $RESILIENT_E97_DATA --migration-policy initialize-from-approved-config$RESUME_COMMAND"
printf 'topology managers=%s real_trainers=%s trainers_per_node=8 local_steps=40 collective=none\n' \
  "$RESILIENT_E97_NODE_COUNT" "$((RESILIENT_E97_NODE_COUNT * 8))"
"$TRAIN_PYTHON_BIN" "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"
if [[ ${RESILIENT_E97_ACCEPTANCE_PHASE:-} == clean-overlap ]]; then
  "$TRAIN_PYTHON_BIN" "$REPO/scripts/frontier/validate_pipelined_e97_performance.py" \
    --telemetry-root "$RUN_DIR/retained-evidence" \
    --output "$RUN_DIR/pipelined-performance.json"
fi
