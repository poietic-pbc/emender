#!/bin/bash
# Native G2/G2-fault gate: exactly two persistent CXI endpoints, no model/GPU.
#SBATCH -A bif148
#SBATCH -J native-ndp-2n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --network=job_vni
#SBATCH --signal=B:TERM@120
#SBATCH -o logs/frontier/native-dataplane/%x-%j.out
#SBATCH -e logs/frontier/native-dataplane/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
# A batch allocation is a clean runtime boundary. Load the reviewed module
# stack here; never depend on the submit/login shell's loader state.
# shellcheck source=frontier_runtime_env.sh
source "$REPO/scripts/frontier/frontier_runtime_env.sh"
# Discard submit-shell loader state before Lmod constructs the canonical batch
# runtime. Everything captured after this point came from reviewed modules.
unset LD_LIBRARY_PATH
frontier_load_default_modules
ROCM_RUNTIME_DIR=$(frontier_resolve_rocm_runtime_dir)
FABRIC_RUNTIME_DIR=$(frontier_resolve_libfabric_runtime_dir)
MODULE_RUNTIME_PATH=${LD_LIBRARY_PATH:-}
: "${NDP_BUILD_MANIFEST:?set the one retained clean native build manifest}"
: "${NDP_ARTIFACT_ROOT:?set retained Frontier artifact root}"
: "${NDP_RUN_ID:?set fenced synthetic run identity}"
: "${NDP_PAYLOAD_ID:?set immutable clean or changed fault payload identity}"
: "${NDP_MODE:?set clean or fault}"
[[ $NDP_MODE == clean || $NDP_MODE == fault ]] || { echo "invalid NDP_MODE" >&2; exit 64; }
[[ ${SLURM_JOB_NUM_NODES:?} == 2 ]] || { echo "native gate requires exactly two nodes" >&2; exit 64; }
[[ ${SLURM_JOB_QOS:?} == debug ]] || { echo "native gate requires debug QoS" >&2; exit 64; }
[[ ${SLURM_JOB_PARTITION:?} == batch ]] || { echo "native gate requires batch partition" >&2; exit 64; }
[[ ${NDP_REQUESTED_WALLTIME:-} == 00:20:00 ]] || { echo "native gate requires exact 20m payload" >&2; exit 64; }
[[ ${NDP_GATE:-} == full-layout ]] || { echo "native gate requires full-layout identity" >&2; exit 64; }
[[ ,${SLURM_NETWORK:-}, == *,job_vni,* ]] || {
  echo "native gate requires an allocation-time cross-node job VNI" >&2; exit 64;
}
[[ ${FI_PROVIDER:-} == cxi ]] || { echo "native Frontier gate requires exact FI_PROVIDER=cxi" >&2; exit 64; }
[[ ${NDP_LAYOUT:-} == e97-f64-5506770496 ]] || { echo "native gate requires exact E97 f64 layout" >&2; exit 64; }
[[ ${NDP_TRAINERS_PER_NODE:-} == 8 ]] || { echo "native gate requires eight local lanes" >&2; exit 64; }
NDP_WEIGHTS=${NDP_WEIGHT_NODE0:-},${NDP_WEIGHT_NODE1:-}
[[ $NDP_WEIGHTS == 1966080,1968000 ]] || { echo "native gate weights changed" >&2; exit 64; }
if [[ $NDP_MODE == clean ]]; then
  [[ ${NDP_GENERATIONS:-} == 3 && ${NDP_WARMUP_GENERATIONS:-} == 1 ]] || {
    echo "clean gate requires one warm-up plus three timed generations" >&2; exit 64;
  }
  [[ -z ${NDP_CLEAN_GATE_JSON:-} ]] || { echo "clean gate cannot consume prior promotion evidence" >&2; exit 64; }
else
  [[ ${NDP_GENERATIONS:-} == 1 && ${NDP_WARMUP_GENERATIONS:-} == 1 ]] || {
    echo "fault gate requires one warm-up plus one changed injected generation" >&2; exit 64;
  }
  : "${NDP_CLEAN_GATE_JSON:?fault payload requires the passing clean gate}"
  [[ -s $NDP_CLEAN_GATE_JSON ]] || { echo "clean gate is unreadable" >&2; exit 64; }
fi

mkdir -p "$REPO/logs/frontier/native-dataplane"
ARTIFACT_DIR="$NDP_ARTIFACT_ROOT/$SLURM_JOB_ID"
[[ ! -e $ARTIFACT_DIR ]] || { echo "refusing to overwrite retained job evidence" >&2; exit 73; }
mkdir -p "$ARTIFACT_DIR"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${NDP_PYTHON_BIN:-$REPO/.envs/olcf-rocm711-torch210-py312/bin/python}
[[ -x $PYTHON_BIN ]] || { echo "bounded native gate Python is unavailable: $PYTHON_BIN" >&2; exit 66; }
"$PYTHON_BIN" - "$NDP_BUILD_MANIFEST" "$REPO" <<'PY'
import sys
from ndm.native_artifacts import validate_build_manifest
validate_build_manifest(sys.argv[1], source_root=sys.argv[2], require_clean=True)
PY

INSTALL_DIR=$(dirname "$NDP_BUILD_MANIFEST")
GATE_BINARY="$INSTALL_DIR/bin/ndp_frontier_2n_gate"
[[ -x $GATE_BINARY ]] || { echo "installed hashed gate binary is missing" >&2; exit 66; }
# Use only bundle and module-reviewed runtime paths. In particular, do not
# admit arbitrary directories inherited through --export=ALL.
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$ROCM_RUNTIME_DIR:$FABRIC_RUNTIME_DIR${MODULE_RUNTIME_PATH:+:$MODULE_RUNTIME_PATH}"

LOADER_PREFLIGHT="$ARTIFACT_DIR/loader-preflight.txt"
{
  printf 'gate_binary=%s\nrocm_runtime_dir=%s\nfabric_runtime_dir=%s\n' \
    "$(readlink -f "$GATE_BINARY")" "$ROCM_RUNTIME_DIR" "$FABRIC_RUNTIME_DIR"
  readelf -d "$GATE_BINARY" | grep -E 'RPATH|RUNPATH'
  ldd "$GATE_BINARY"
} >"$LOADER_PREFLIGHT"
! grep -q 'not found' "$LOADER_PREFLIGHT" || {
  cat "$LOADER_PREFLIGHT" >&2
  echo "native gate has unresolved runtime libraries" >&2
  exit 66
}
HIP_RESOLVED=$(sed -n 's/^[[:space:]]*libamdhip64\.so\.7 => \([^ ]*\).*/\1/p' "$LOADER_PREFLIGHT")
[[ -n "$HIP_RESOLVED" && $(readlink -f "$HIP_RESOLVED") == "$ROCM_RUNTIME_DIR"/* ]] || {
  cat "$LOADER_PREFLIGHT" >&2
  echo "libamdhip64.so.7 did not resolve from the reviewed ROCm module" >&2
  exit 66
}
while read -r library; do
  [[ -r "$library" ]] || continue
  printf 'resolved_library=%s sha256=%s\n' "$(readlink -f "$library")" \
    "$(sha256sum "$library" | awk '{print $1}')"
done < <(sed -n 's/^[[:space:]]*[^ ]* => \([^ ]*\).*/\1/p' "$LOADER_PREFLIGHT") \
  >>"$LOADER_PREFLIGHT"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} == 2 ]] || { echo "Slurm allocation did not resolve two nodes" >&2; exit 64; }
CONTROLLER_HOST=${NODES[0]}
CONTROLLER_PORT=$((31000 + SLURM_JOB_ID % 20000))

"$PYTHON_BIN" - "$ARTIFACT_DIR/submission.json" <<PY
import json, os, pathlib, socket, subprocess, time
path = pathlib.Path(os.sys.argv[1])
value = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "job_name": os.environ["SLURM_JOB_NAME"],
    "account": os.environ.get("SLURM_JOB_ACCOUNT", "bif148"),
    "partition": os.environ["SLURM_JOB_PARTITION"],
    "qos": os.environ["SLURM_JOB_QOS"],
    "nodes": int(os.environ["SLURM_JOB_NUM_NODES"]),
    "nodelist": os.environ["SLURM_JOB_NODELIST"],
    "requested_walltime": os.environ["NDP_REQUESTED_WALLTIME"],
    "slurm_network": os.environ["SLURM_NETWORK"],
    "slingshot_step_env_at_batch_scope": {
        key: value for key, value in os.environ.items() if key.startswith("SLINGSHOT_")
    },
    "submit_host": socket.gethostname(),
    "batch_started_unix_ns": time.time_ns(),
    "mode": os.environ["NDP_MODE"],
    "run_id": os.environ["NDP_RUN_ID"],
    "payload_id": os.environ["NDP_PAYLOAD_ID"],
    "sbatch_script": "scripts/frontier/native_dataplane_2n_gate.sbatch",
    "submit_command": "scripts/frontier/submit_native_dataplane_2n_gate.sh " + os.environ["NDP_MODE"],
    "scontrol_job_sha256_input": subprocess.check_output(
        ["scontrol", "show", "job", "-dd", os.environ["SLURM_JOB_ID"]], text=True
    ),
}
path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY

# The batch shell already runs on CONTROLLER_HOST.  Keeping bounded metadata
# here avoids a needless one-node Slurm step and its single_node_vni path; the
# only job step is the exact two-node native CXI data plane below.
"$PYTHON_BIN" "$REPO/scripts/frontier/native_dataplane_2n_controller.py" \
  --host 0.0.0.0 --port "$CONTROLLER_PORT" --provider cxi --mode "$NDP_MODE" \
  --output "$ARTIFACT_DIR/membership.json" &
CONTROLLER_STEP_PID=$!

set +e
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=32 \
  --cpu-bind=cores --overlap --exact --kill-on-bad-exit=0 \
  bash -c 'if [[ $SLURM_PROCID == 0 ]]; then
      printf "native_step_network=%s slingshot_env_names=" "${SLURM_NETWORK:-unset}" >&2
      env | sed -n "s/^\(SLINGSHOT_[^=]*\)=.*/\1/p" | sort | paste -sd, - >&2
    fi
    exec "$1" --controller-host "$2" --controller-port "$3" \
    --rank "$SLURM_PROCID" --provider cxi --production --mode "$4" \
    --layout-bytes 5506770496 --payload-max 67108864 \
    --generations "$5" --warmup "$6" --reduction-threads 16 \
    --deadline-seconds 900 --run-id "$7" --payload-id "$8" \
    --output "$9/node-$SLURM_PROCID.json"' \
  bash "$GATE_BINARY" "$CONTROLLER_HOST" "$CONTROLLER_PORT" "$NDP_MODE" \
  "$NDP_GENERATIONS" "$NDP_WARMUP_GENERATIONS" "$NDP_RUN_ID" \
  "$NDP_PAYLOAD_ID" "$ARTIFACT_DIR"
GATE_STEP_RC=$?
wait "$CONTROLLER_STEP_PID"
CONTROLLER_RC=$?
set -e
if ((GATE_STEP_RC != 0 || CONTROLLER_RC != 0)); then
  printf 'native gate failed: data_step=%d controller_step=%d\n' \
    "$GATE_STEP_RC" "$CONTROLLER_RC" >&2
  exit 1
fi

VALIDATOR=(
  "$PYTHON_BIN" "$REPO/scripts/frontier/validate_native_dataplane_2n_gate.py"
  --mode "$NDP_MODE"
  --build-manifest "$NDP_BUILD_MANIFEST"
  --membership "$ARTIFACT_DIR/membership.json"
  --node-0 "$ARTIFACT_DIR/node-0.json"
  --node-1 "$ARTIFACT_DIR/node-1.json"
  --submission "$ARTIFACT_DIR/submission.json"
  --source-root "$REPO"
)
if [[ $NDP_MODE == clean ]]; then
  VALIDATOR+=(--output "$ARTIFACT_DIR/full-layout-gate.json")
else
  VALIDATOR+=(--clean-gate "$NDP_CLEAN_GATE_JSON" \
              --output "$ARTIFACT_DIR/failure-injection-gate.json")
fi
"${VALIDATOR[@]}"

(
  cd "$ARTIFACT_DIR"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' |
    sort -z |
    while IFS= read -r -d '' artifact; do
      sha256sum -- "$artifact"
    done > SHA256SUMS
)
printf 'native_gate_status=passed mode=%s job_id=%s artifact_dir=%s\n' \
  "$NDP_MODE" "$SLURM_JOB_ID" "$ARTIFACT_DIR"
