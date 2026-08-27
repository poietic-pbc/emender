#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-4b-collector
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --no-requeue
set -euo pipefail
: "${PAYLOAD_JOB_ID:?payload job id is required}"
: "${RUN_DIR:?run directory is required}"
: "${EXPECTED_PARTITION:?expected partition is required}"
: "${EXPECTED_QOS:?expected QoS is required}"
: "${REPO:?immutable submitted checkout is required}"
: "${EXPECTED_WORLD_SIZE:?expected world size is required}"
CONFIG=${CONFIG:-$REPO/configs/frontier/e97_4b_from_scratch.json}
[[ "$CONFIG" == "$REPO"/* && -r "$CONFIG" ]] || { echo "collector config must be readable immutable source" >&2; exit 66; }
mkdir -p "$RUN_DIR/terminal"
out="$RUN_DIR/terminal/payload-${PAYLOAD_JOB_ID}.sacct"
for _ in $(seq 1 30); do
  sacct -j "$PAYLOAD_JOB_ID" -n -P \
    --format=JobIDRaw,Partition,QOS,State,Elapsed,NNodes,ExitCode > "$out" || true
  grep -E "^${PAYLOAD_JOB_ID}\|${EXPECTED_PARTITION}\|${EXPECTED_QOS}\|" "$out" >/dev/null && break
  sleep 2
done
grep -E "^${PAYLOAD_JOB_ID}\|${EXPECTED_PARTITION}\|${EXPECTED_QOS}\|" "$out" >/dev/null || {
  echo "terminal accounting lacks exact Partition/QOS evidence" >&2; exit 66;
}
sha256sum "$out" > "$out.sha256"
latest="$RUN_DIR/train/latest.pt"
if [[ -L "$latest" && -r "$latest" ]]; then
  checkpoint=$(readlink -f "$latest")
  checkpoint_record="$RUN_DIR/terminal/checkpoint-${PAYLOAD_JOB_ID}.sha256"
  sha256sum "$checkpoint" > "$checkpoint_record.tmp"
  mv "$checkpoint_record.tmp" "$checkpoint_record"
  stat -c 'path=%n bytes=%s mtime=%Y' "$checkpoint" \
    > "$RUN_DIR/terminal/checkpoint-${PAYLOAD_JOB_ID}.stat"
  source "$REPO/scripts/frontier/activate_emender_frontier.sh"
  : "${EMENDER_PYTHON:?canonical activation did not set EMENDER_PYTHON}"
  "$EMENDER_PYTHON" - "$checkpoint" "$CONFIG" \
    "$EXPECTED_WORLD_SIZE" "$RUN_DIR/terminal/checkpoint-${PAYLOAD_JOB_ID}.reload.json" <<'PY'
import json, os, sys, tempfile
import torch
checkpoint_path, config_path, expected_world, output = sys.argv[1:]
config = json.load(open(config_path, encoding='utf-8'))
value = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
step = value.get('step'); total_tokens = value.get('total_tokens')
metadata = value.get('checkpoint_metadata'); model = value.get('model_state_dict')
optimizer = value.get('optimizer_state_dict')
if type(step) is not int or step <= 0: raise SystemExit('invalid checkpoint step')
if type(total_tokens) is not int or total_tokens <= 0: raise SystemExit('invalid total_tokens')
training = config['training']; data = config['data']; expected_world = int(expected_world)
if step % training['diloco_k'] != 0: raise SystemExit('checkpoint is not K-aligned')
if not isinstance(metadata, dict) or metadata.get('world_size') != expected_world:
    raise SystemExit('checkpoint world-size mismatch')
sampler = metadata.get('sampler', {}); identity = sampler.get('identity', {})
expected_identity = {
    'schema': data['sampler_schema'], 'corpus_sha256': data['corpus_sha256'],
    'tokenizer_sha256': data['tokenizer_sha256'], 'sampler_key': data['sampler_key'],
    'data_world_size': expected_world, 'context_size': training['context_size'],
}
origin = int(data.get('sampler_stream_origin_accepted_tokens', 0))
if data['sampler_schema'] == 'emender-byte-window-counter-v2':
    expected_identity['stream_origin_accepted_tokens'] = origin
if identity != expected_identity: raise SystemExit('sampler identity mismatch')
if sampler.get('total_accepted_tokens') != total_tokens: raise SystemExit('sampler token mismatch')
relative_tokens = total_tokens - origin
per_rank_sample_tokens = expected_world * training['context_size']
if relative_tokens < 0 or relative_tokens % per_rank_sample_tokens:
    raise SystemExit('checkpoint token clock mismatch')
cursor = relative_tokens // per_rank_sample_tokens
if sampler.get('absolute_rank_sample_index') != cursor: raise SystemExit('sampler index mismatch')
if cursor % training['batch_size_per_rank']:
    raise SystemExit('sampler cursor is not optimizer-step aligned')
source_step = int(config.get('seed', {}).get('source_step', 0))
if step != source_step + cursor // training['batch_size_per_rank']:
    raise SystemExit('checkpoint step disagrees with phase-relative sampler clock')
if origin:
    transition = metadata.get('sampler_transition', {})
    if (transition.get('boundary_total_tokens') != origin
            or transition.get('source_step') != source_step
            or transition.get('new_identity') != expected_identity):
        raise SystemExit('missing or invalid sampler transition provenance')
if not isinstance(model, dict) or not model: raise SystemExit('missing model state')
if not isinstance(optimizer, dict) or not optimizer.get('state') or not optimizer.get('param_groups'):
    raise SystemExit('missing optimizer state')
report = {'schema':'emender-e97-4b-checkpoint-reload-v1','checkpoint':os.path.realpath(checkpoint_path),
          'step':step,'total_tokens':total_tokens,'world_size':expected_world,
          'batch_size_per_rank':training['batch_size_per_rank'],'diloco_k':training['diloco_k'],
          'model_tensors':len(model),'optimizer_state_entries':len(optimizer['state']),
          'mmap_deserialization':True,'reloadable':True}
os.makedirs(os.path.dirname(output), exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix='.reload.', suffix='.tmp', dir=os.path.dirname(output), text=True)
with os.fdopen(fd, 'w') as handle:
    json.dump(report, handle, sort_keys=True); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp, output)
PY
fi
cat "$out"
