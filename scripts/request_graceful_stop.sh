#!/usr/bin/env bash
# Request coordinated final merge + atomic checkpoint at the next safe step.
set -euo pipefail

if [[ $# -ne 1 || "$1" == -* ]]; then
  echo "usage: scripts/request_graceful_stop.sh LOGDIR" >&2
  exit 64
fi
LOGDIR="$(cd -P "$1" 2>/dev/null && pwd)" || {
  echo "log directory does not exist: $1" >&2; exit 66;
}
LOGFILE="$LOGDIR/run.log"
[[ -f "$LOGFILE" ]] || { echo "missing run log: $LOGFILE" >&2; exit 66; }

output_dir="$(python - "$LOGFILE" <<'PY'
import sys
prefix = "Output directory: "
found = []
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        line = line.strip()
        if line.startswith(prefix):
            found.append(line[len(prefix):].strip())
if not found:
    raise SystemExit(1)
print(found[-1])
PY
)" || {
  echo "training output directory is not visible in $LOGFILE yet; retry after initialization" >&2
  exit 75
}
[[ -d "$output_dir" ]] || {
  echo "reported training output directory does not exist: $output_dir" >&2; exit 66;
}

request="$output_dir/.final_checkpoint_request"
python - "$request" <<'PY'
import json
import os
import sys
import tempfile
import time

path = sys.argv[1]
payload = {
    "reason": "operator_stop",
    "requester_pid": os.getpid(),
    "time": time.time(),
}
fd, temporary = tempfile.mkstemp(
    prefix=".final_checkpoint_request.", suffix=".tmp",
    dir=os.path.dirname(path), text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(path)
PY

echo "Graceful stop requested. Do not kill the process group."
echo "Ranks will finish the current optimizer step, perform a final DiLoCo merge,"
echo "and rank 0 will atomically publish latest.pt. Monitor:"
echo "  tail -f $LOGFILE"
echo "Completion marker: [final-checkpoint] END"
