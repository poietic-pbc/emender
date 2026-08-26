#!/bin/bash
# Download and validate a commit-pinned private E97 4B training checkpoint.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
source scripts/frontier/activate_emender_frontier.sh
: "${EMENDER_PYTHON:?canonical Frontier activation did not set EMENDER_PYTHON}"
: "${REVISION:?set REVISION to the immutable 40-character Hub commit}"
: "${REMOTE_DIR:?set REMOTE_DIR, e.g. checkpoints/step_011776_tokens_6174015488}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "REVISION must be a full Hub commit" >&2; exit 64; }
REPO_ID=${REPO_ID:-spinozans/e97-4b-training-checkpoints}
DEST_ROOT=${DEST_ROOT:-/lustre/orion/bif148/proj-shared/emender/checkpoints/e97-4b}
TRANSFER_MODE=${TRANSFER_MODE:-exact}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-8}
case "$TRANSFER_MODE" in
  exact) ;;
  model-only)
    [[ ${CONFIRM_MODEL_ONLY:-0} == 1 ]] || {
      echo "model-only transfer requires CONFIRM_MODEL_ONLY=1" >&2; exit 64;
    }
    ;;
  *) echo "TRANSFER_MODE must be exact or model-only" >&2; exit 64;;
esac
mkdir -p "$DEST_ROOT"
"$EMENDER_PYTHON" - "$REPO_ID" "$REVISION" "$REMOTE_DIR" "$DEST_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo,revision,remote,dest=sys.argv[1:]
path=snapshot_download(repo_id=repo,repo_type='model',revision=revision,
                       allow_patterns=[f'{remote}/*'],local_dir=dest)
print(path)
PY
LOCAL_DIR="$DEST_ROOT/$REMOTE_DIR"
[[ -r "$LOCAL_DIR/metadata.json" && -r "$LOCAL_DIR/SHA256SUMS" ]] || {
  echo "download lacks metadata/checksum authority" >&2; exit 66;
}
(
  cd "$LOCAL_DIR"
  sha256sum -c SHA256SUMS
)
CHECKPOINT=$("$EMENDER_PYTHON" - "$LOCAL_DIR" "$TRANSFER_MODE" "$EXPECTED_WORLD_SIZE" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]; expected_world=int(sys.argv[3])
m=json.loads((root/'metadata.json').read_text())
if m.get('schema')!='emender-e97-4b-durable-checkpoint-v1': raise SystemExit('bad receipt schema')
p=root/m['checkpoint']
if p.stat().st_size!=int(m['size_bytes']): raise SystemExit('checkpoint size mismatch')
c=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
cm=c.get('checkpoint_metadata') or {}; sampler=cm.get('sampler') or {}; ident=sampler.get('identity') or {}
checks={
 'step':int(c.get('step',-1))==int(m['step']),
 'tokens':int(c.get('total_tokens',-1))==int(m['total_accepted_tokens']),
 'params':int((cm.get('model') or {}).get('total_params',-1))==int(m['parameters']),
 'sampler_schema':ident.get('schema')=='emender-byte-window-counter-v1',
 'world':int(cm.get('world_size',-1))==int(m['world_size']),
}
if not all(checks.values()): raise SystemExit(f'checkpoint metadata mismatch: {checks}')
if mode=='exact' and int(m['world_size'])!=expected_world:
 raise SystemExit(f'exact resume world mismatch: checkpoint={m["world_size"]} expected={expected_world}')
print(p.resolve())
PY
)
printf 'validated_checkpoint=%s\ntransfer_mode=%s\ncheckpoint_world_size=%s\n' \
  "$CHECKPOINT" "$TRANSFER_MODE" \
  "$("$EMENDER_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["world_size"])' "$LOCAL_DIR/metadata.json")"
if [[ "$TRANSFER_MODE" == model-only ]]; then
  echo "WARNING: this is not an exact optimizer/sampler resume; no training launch was performed" >&2
fi
