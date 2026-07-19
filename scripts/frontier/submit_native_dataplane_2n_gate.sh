#!/bin/bash
set -euo pipefail

MODE=${1:?usage: submit_native_dataplane_2n_gate.sh clean|fault}
[[ $MODE == clean || $MODE == fault ]] || { echo "mode must be clean or fault" >&2; exit 64; }
REPO=${REPO:-$(git rev-parse --show-toplevel)}
: "${NDP_BUILD_MANIFEST:?set retained exact-code build manifest}"
NDP_ARTIFACT_ROOT=${NDP_ARTIFACT_ROOT:-$REPO/reports/frontier/native-dataplane}
SOURCE_COMMIT=$(git -C "$REPO" rev-parse HEAD)
SHORT_COMMIT=${SOURCE_COMMIT:0:12}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
NDP_RUN_ID=${NDP_RUN_ID:-native-g2-$MODE-$SHORT_COMMIT-$STAMP}
NDP_PAYLOAD_ID=${NDP_PAYLOAD_ID:-$SHORT_COMMIT-e97-g2-$MODE-$STAMP}
NDP_CLEAN_GATE_JSON=${NDP_CLEAN_GATE_JSON:-}
NDP_PYTHON_BIN=${NDP_PYTHON_BIN:-$REPO/.envs/olcf-rocm711-torch210-py312/bin/python}

[[ $(git -C "$REPO" branch --show-current) == main ]] || {
  echo "authoritative Frontier gate must be submitted from main" >&2; exit 64;
}
git -C "$REPO" diff --quiet --ignore-submodules -- || {
  echo "authoritative Frontier gate requires a clean tracked source tree" >&2; exit 64;
}
[[ -z $(squeue -u "$USER" -h -o '%i') ]] || {
  echo "refusing to overlap another user allocation" >&2; exit 69;
}

if [[ $MODE == clean ]]; then
  [[ -z $NDP_CLEAN_GATE_JSON ]] || { echo "clean submission cannot consume a gate" >&2; exit 64; }
  GENERATIONS=3
else
  [[ -s $NDP_CLEAN_GATE_JSON ]] || { echo "fault submission requires readable NDP_CLEAN_GATE_JSON" >&2; exit 64; }
  "$NDP_PYTHON_BIN" "$REPO/scripts/frontier/attest_native_dataplane.py" verify \
    --backend native-cxi --production --full-layout \
    --build-manifest "$NDP_BUILD_MANIFEST" --gate-json "$NDP_CLEAN_GATE_JSON" \
    --source-root "$REPO" >/dev/null
  CLEAN_PAYLOAD=$($NDP_PYTHON_BIN - "$NDP_CLEAN_GATE_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["payload_id"])
PY
)
  [[ $NDP_PAYLOAD_ID != "$CLEAN_PAYLOAD" ]] || { echo "fault payload ID is unchanged" >&2; exit 64; }
  GENERATIONS=1
fi

mkdir -p "$REPO/logs/frontier/native-dataplane" "$NDP_ARTIFACT_ROOT"
export NDP_GATE=full-layout NDP_WEIGHTS=1966080,1968000
export REPO NDP_BUILD_MANIFEST NDP_ARTIFACT_ROOT NDP_RUN_ID NDP_PAYLOAD_ID \
  NDP_CLEAN_GATE_JSON NDP_PYTHON_BIN
exec sbatch --parsable \
  -A bif148 -p batch --qos=debug -N 2 -t 00:20:00 \
  --network=job_vni \
  -J "native-ndp-g2-$MODE" --chdir="$REPO" \
  --export=ALL,REPO="$REPO",NDP_BUILD_MANIFEST="$NDP_BUILD_MANIFEST",NDP_ARTIFACT_ROOT="$NDP_ARTIFACT_ROOT",NDP_RUN_ID="$NDP_RUN_ID",NDP_PAYLOAD_ID="$NDP_PAYLOAD_ID",NDP_MODE="$MODE",NDP_CLEAN_GATE_JSON="$NDP_CLEAN_GATE_JSON",NDP_PYTHON_BIN="$NDP_PYTHON_BIN",NDP_REQUESTED_WALLTIME=00:20:00,NDP_LAYOUT=e97-f64-5506770496,NDP_TRAINERS_PER_NODE=8,NDP_WEIGHT_NODE0=1966080,NDP_WEIGHT_NODE1=1968000,NDP_GENERATIONS="$GENERATIONS",NDP_WARMUP_GENERATIONS=1,FI_PROVIDER=cxi,FI_MR_CACHE_MONITOR=kdreg2,FI_CXI_ATS=0 \
  scripts/frontier/native_dataplane_2n_gate.sbatch
