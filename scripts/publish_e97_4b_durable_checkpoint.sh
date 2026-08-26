#!/bin/bash
# Archive, checksum, stage, and explicitly publish one trusted training checkpoint.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
: "${CHECKPOINT:?set CHECKPOINT to a readable E97 4B checkpoint or latest.pt}"
: "${PUBLISH_CONFIRM:?set PUBLISH_CONFIRM=1 to permit the Hub upload}"
[[ "$PUBLISH_CONFIRM" == 1 ]] || { echo "PUBLISH_CONFIRM must equal 1" >&2; exit 64; }
REPO_ID=${REPO_ID:-spinozans/e97-4b-training-checkpoints}
DURABLE_ROOT=${DURABLE_ROOT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_durable_checkpoints}
STAGING_ROOT=${STAGING_ROOT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_hf_staging}
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the commit that launched this checkpoint lineage}"
ARGS_JSON=${ARGS_JSON:-$(dirname "$(readlink -f "$CHECKPOINT")")/args.json}
[[ -r "$CHECKPOINT" && -r "$ARGS_JSON" ]] || { echo "checkpoint or args.json is unreadable" >&2; exit 66; }
command -v hf >/dev/null || { echo "hf CLI is required" >&2; exit 69; }
hf auth whoami >/dev/null
mkdir -p "$STAGING_ROOT"
cp -f hf_templates/e97_training_checkpoints/README.md "$STAGING_ROOT/README.md"
receipt=$(
  "$PYTHON_BIN" scripts/archive_e97_4b_durable_checkpoint.py \
    --checkpoint "$CHECKPOINT" --args-json "$ARGS_JSON" \
    --durable-root "$DURABLE_ROOT" --staging-root "$STAGING_ROOT" \
    --repo-id "$REPO_ID" --source-commit "$SOURCE_COMMIT"
)
printf '%s\n' "$receipt"
HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1} \
  hf upload-large-folder "$REPO_ID" "$STAGING_ROOT" \
    --repo-type model --private --num-workers "${HF_UPLOAD_WORKERS:-4}"
"$PYTHON_BIN" - "$REPO_ID" "$receipt" <<'PY'
import json,sys
from huggingface_hub import HfApi
repo,receipt=sys.argv[1:]
info=HfApi().repo_info(repo,repo_type='model')
print(json.dumps({'repository':repo,'commit':info.sha,'checkpoint':json.loads(receipt)},sort_keys=True))
PY
