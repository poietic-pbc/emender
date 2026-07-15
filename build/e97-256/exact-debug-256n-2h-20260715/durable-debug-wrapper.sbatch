#!/bin/bash
# Scheduler-exit continuation shim for the immutable job-4980157 payload.
# Training is still executed from ATTESTED_TREE without source substitution.
#SBATCH -A bif148
#SBATCH -J trainpy-async-quorum-2n
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/trainpy_async_quorum/%x-%j.out
#SBATCH -e logs/frontier/trainpy_async_quorum/%x-%j.err
#SBATCH --signal=B:TERM@300

set -euo pipefail

readonly ATTESTED_TREE=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1023/build/e97-256/repeated-smoke-4979704/render/production/attested-tree-9fff689c9f9252b6a264773c207f8f8ca8509666-jj6etgng
readonly ATTESTED_LAUNCHER="$ATTESTED_TREE/scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch"
child=

run_root() {
  find "${OUTPUT_ROOT:?}/${RUN_DATE}/${SCALEOUT_VARIANT:?}" -mindepth 1 -maxdepth 1 \
    -type d -name "${SLURM_JOB_ID:?}-*" -print -quit 2>/dev/null || true
}

publish_handoff() {
  local root latest tmp immutable
  root=$(run_root)
  [[ -n "$root" && -r "$root/async_run/latest.json" ]] || return 0
  mkdir -p "$root/continuation"
  immutable="$root/continuation/last-valid-$(date -u +%Y%m%dT%H%M%SZ).json"
  tmp="${immutable}.tmp.$$"
  python3 - "$root" "$tmp" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

root, output = map(Path, sys.argv[1:])
latest = root / "async_run/latest.json"
record = json.loads(latest.read_text())
candidates = []
for key in ("model_checkpoint_path", "checkpoint_path"):
    if record.get(key): candidates.append(record[key])
candidates.extend(record.get("checkpoint_paths") or [])
checkpoint = next((Path(p) for p in candidates if str(p).endswith(".pt") and Path(p).is_file()), None)
if checkpoint is None:
    raise SystemExit("latest.json has no readable immutable .pt checkpoint")
h = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        h.update(block)
payload = {
    "schema_version": 1,
    "job_id": os.environ["SLURM_JOB_ID"],
    "run_root": str(root),
    "source_latest_json": str(latest),
    "checkpoint_path": str(checkpoint.resolve()),
    "checkpoint_sha256": h.hexdigest(),
    "checkpoint_size_bytes": checkpoint.stat().st_size,
    "generation": record.get("generation"),
    "step": record.get("step"),
    "published_at": datetime.now(timezone.utc).isoformat(),
    "next_launch_selector": "select-continuation-seed.py",
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  mv "$tmp" "$immutable"
  ln -sfn "$(basename "$immutable")" "$root/continuation/last-valid.json.tmp.$$"
  mv -Tf "$root/continuation/last-valid.json.tmp.$$" "$root/continuation/last-valid.json"
}

on_term() {
  [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
  [[ -z "$child" ]] || wait "$child" || true
  publish_handoff
  exit 143
}
trap on_term TERM
trap publish_handoff EXIT

export RUN_DATE=${RUN_DATE:-$(date -u +%Y%m%d)}
export REPO="$ATTESTED_TREE"
bash "$ATTESTED_LAUNCHER" &
child=$!
set +e
wait "$child"
status=$?
set -e
child=
exit "$status"
