#!/bin/bash
# One unchanged-payload attempt: held debug payload -> durable afterany collector -> release.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$REPO"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?activation did not set EMENDER_PYTHON}"
PAYLOAD=$REPO/scripts/frontier/direct_same_allocation_trainpy_restart.sbatch
COLLECTOR=$REPO/scripts/frontier/direct_same_allocation_collector.sh
BASE=${DIRECT_RESTART_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation}
mkdir -p "$BASE/attempts" "$BASE/collectors"
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked source must be clean" >&2; exit 64; }
SOURCE_SHA=$(git rev-parse HEAD)
PAYLOAD_DIGEST=$(sha256sum train.py "$PAYLOAD" "$COLLECTOR" | sha256sum | awk '{print $1}')
SENTINEL="$BASE/attempts/$PAYLOAD_DIGEST"
( set -o noclobber; printf '%s|%s|%s\n' "$SOURCE_SHA" "$(date -u +%FT%TZ)" "$PAYLOAD_DIGEST" > "$SENTINEL" ) 2>/dev/null || {
  echo "unchanged payload already attempted: $PAYLOAD_DIGEST" >&2; exit 65;
}
PAYLOAD_ID=""
trap 'if [[ -z $PAYLOAD_ID ]]; then rm -f "$SENTINEL"; fi' EXIT
PAYLOAD_ID=$(sbatch --parsable --hold --no-kill -A bif148 -p batch -q debug -N2 -t 00:30:00 \
  --export=ALL,REPO="$REPO",PYTHON_BIN="$EMENDER_PYTHON" "$PAYLOAD")
ROOT="$BASE/$PAYLOAD_ID"
mkdir -p "$ROOT/identity"
RECORD="$ROOT/identity/submission.json"
"$EMENDER_PYTHON" - "$RECORD" <<PY
import json,os,sys,tempfile
v={'schema':'direct-same-allocation-submission-v1','payload_job_id':'$PAYLOAD_ID',
'source_sha':'$SOURCE_SHA','payload_digest':'$PAYLOAD_DIGEST','nodes':2,
'partition':'batch','qos':'debug','sbatch_no_kill':True,'released':False,
'python_bin':'$EMENDER_PYTHON'}
open(sys.argv[1],'w').write(json.dumps(v,sort_keys=True)+'\n')
PY
COLLECTOR_ID=$(sbatch --parsable -A bif148 -p batch -q normal -N1 -t 00:10:00 \
  --dependency="afterany:$PAYLOAD_ID" \
  --export=ALL,PAYLOAD_JOB_ID="$PAYLOAD_ID",DIRECT_RESTART_ROOT="$ROOT",SUBMISSION_RECORD="$RECORD" "$COLLECTOR")
# Retain explicit queued scheduler evidence naming partition and QoS before release.
squeue -j "$PAYLOAD_ID,$COLLECTOR_ID" -h -o '%i|%T|%N|%P|%q|%R' | tee "$ROOT/identity/squeue-held.txt"
"$EMENDER_PYTHON" - "$RECORD" "$COLLECTOR_ID" <<'PY'
import json,os,sys
p=sys.argv[1]; v=json.load(open(p)); v['collector_job_id']=sys.argv[2]
v['dependency']='afterany:'+v['payload_job_id']; v['collector_registered_before_release']=True
q=p+'.tmp'; open(q,'w').write(json.dumps(v,sort_keys=True)+'\n'); os.replace(q,p)
PY
scontrol release "$PAYLOAD_ID"
"$EMENDER_PYTHON" - "$RECORD" <<'PY'
import json,os,sys
p=sys.argv[1]; v=json.load(open(p)); v['released']=True
q=p+'.tmp'; open(q,'w').write(json.dumps(v,sort_keys=True)+'\n'); os.replace(q,p)
PY
printf 'payload_job_id=%s\ncollector_job_id=%s\nroot=%s\npayload_digest=%s\n' "$PAYLOAD_ID" "$COLLECTOR_ID" "$ROOT" "$PAYLOAD_DIGEST"
