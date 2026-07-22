# Trace E97 smoothed-loss plot and S3 checkpoint upload

Date: 2026-07-22 UTC
Task: `trace-e97-smoothed`

This report records a read-only investigation of the E97 loss plot, the E97
checkpoint upload path, and whether either mechanism is following the live GDN2
run. Runtime files, processes, training state, and S3 objects were not changed.

## Sources inspected

- `docs/GDN2_MLP_DILOCO_HANDOFF_20260711.md`
- `docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md`
- `docs/E97_LOSS_INFO_FIGURE_20260629_0735.md`
- `docs/E97_LOSS_CURVE_BPB_REFRESH_20260630_0755.md`
- `scripts/plot_e97_diloco_loss.py`
- `scripts/launch_emender_8gpu_diloco.sh`
- `scripts/supervise_emender_8gpu_diloco.sh`
- `scripts/launch_gdn2_mlp_8gpu_diloco.sh`
- `train.py`
- Local read-only process, log, checkpoint, cron, systemd, tmux, and S3 listings.

## E97 Plot

Verified plot script:

```bash
scripts/plot_e97_diloco_loss.py
```

Verified documented refresh command from the current ops handoff:

```bash
cd /home/erikg/ndm
RUN_ROOT=/mnt/nvme1n1/erikg/diloco_8gpu/emender
PLOT=/tmp/e97_diloco_loss_curve_20260623.png
OUT=/tmp/e97_diloco_loss_curve_20260623.refresh.txt

python scripts/plot_e97_diloco_loss.py \
  --run-root "$RUN_ROOT" \
  --output "$PLOT" | tee "$OUT"
```

Older documented/local default output command:

```bash
python scripts/plot_e97_diloco_loss.py \
  --run-root /mnt/nvme1n1/erikg/diloco_8gpu/emender \
  --output docs/experiments/figures/e97_diloco_loss_curve_20260623.png
```

Verified local output artifact:

```text
docs/experiments/figures/e97_diloco_loss_curve_20260623.png
mtime: 2026-07-07 14:43:56.838 +0000
size: 286402 bytes
```

Verified documented hosted artifact:

```text
http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png
SSH target: erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png
```

No live plot updater was visible by read-only inspection. Searches of process
state, user cron, user systemd units/timers, and tmux sessions found no
`plot_e97_diloco_loss.py`, matplotlib, E97 plot, GDN2 plot, or related running
service. The local PNG mtime is stale relative to the final E97 checkpoint, so
the current visible mechanism is operator refresh by running the script, not an
active service that keeps the artifact current. This matches
`docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md`: it documents a status/refresh
protocol to run when the human asks for status or plot update, not a required
daemon.

The documented hosted publish step is also operator-run:

```bash
scp "$PLOT" erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png
```

The documented verification is SHA comparison across local PNG, SSH target, and
HTTP download plus an HTTP header check for `200` and `image/png`.

### Plot Inputs

The plot script reads `run*.log` files directly under the run root:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260702T111434Z.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260706T083721Z.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260709T084543Z.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260722T055705Z.log
/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log
```

It parses step/loss lines matching this shape:

```text
step <n> | loss <x> | lr ... | grad ... | tok/s ... | global_tok/s ... | elapsed_h ... | time <iso>
```

It also parses resume lines and saved checkpoint lines. Checkpoint markers come
from both saved-checkpoint log lines and checkpoint filenames under:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/*/checkpoint_step_*_loss_*.pt
```

Rank provenance: `train.py` prints the parsed training loss lines only inside
`if is_main:` at the logging site, so the plotted loss records are main-process
aggregate log entries. The files are combined detached-run logs, not per-rank
log files.

### Lineage And Smoothing

The plot script deduplicates/resolves resumes before plotting. It sorts parsed
records by timestamp and file/order, then keeps the latest observed record for
each optimizer step; older duplicated records are marked superseded.

Verified final E97 parse on 2026-07-22:

```text
raw_points: 94976
effective_points: 92037
superseded_points: 2939
resume_steps: 500, 72000, 72500, 741500, 1067500, 1260000, 2300500
```

Smoothing algorithm:

```python
avg_window = min(80, max(5, len(points) // 40))
smoothed[i] = mean(loss[max(0, i - avg_window + 1):i + 1])
```

For final E97, `len(points) == 92037`, so:

```text
avg_window: 80 plotted log points
```

This is a trailing moving average over plotted loss records, not over raw
optimizer steps. Because E97 logged every 25 steps, the final 80-point smoothing
window spans about 2,000 optimizer steps.

### Token Calculation

E97 was launched with:

```text
batch_size: 4
chunk_size: 2048
world_size: 8
```

So one optimizer step corresponds to:

```text
4 * 2048 * 8 = 65536 tokens
```

Token count is:

```text
tokens = optimizer_step * 65536
```

### Final E97 Values

Verified final plotted point from log parsing:

```text
latest_plotted_step: 2300925
latest_plotted_raw_loss: 2.4770
latest_plotted_time: 2026-07-22T05:56:09+00:00
latest_plotted_tokens: 150793420800
plot_smoothed_loss: 2.437045
last100_loss_from_log_records: 2.436512
```

Verified final checkpoint from the handoff and local/S3 inspection:

```text
final_checkpoint_step: 2300930
final_checkpoint_loss_in_filename: 2.4365
final_checkpoint_tokens: 150793748480
local_checkpoint:
  /mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/checkpoint_step_2300930_loss_2.4365.pt
S3_checkpoint:
  s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt
S3_manifest:
  s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json
sha256:
  0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
```

The final checkpoint is five optimizer steps after the latest regular plotted
log point because logging was every 25 steps and final checkpointing stopped at
step 2,300,930.

## Live GDN2 Plot State

Verified live GDN2 process:

```text
torchrun PID: 3754241
wrapper bash PID: 3754406
worker rank PIDs: 3754608, 3754609, 3754611, 3754612, 3754613, 3754614, 3754615, 3754616
log:
  /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log
run directory:
  /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444
local checkpoints visible by read-only listing at 2026-07-22T09:08Z:
  checkpoint_step_000500_loss_5.8263.pt
  checkpoint_step_001000_loss_5.3572.pt
  checkpoint_step_001500_loss_4.9854.pt
```

Verified launch source:

```bash
scripts/launch_gdn2_mlp_8gpu_diloco.sh
```

No analogous plot process was visible for GDN2. The GDN2 launch script contains
the training launch only; it does not invoke `scripts/plot_e97_diloco_loss.py`,
a GDN2-specific plotter, S3 upload, cron registration, or supervisor. No GDN2
plot output path was found.

The E97 parser can read the GDN2 log shape if run manually, but that is only
parser compatibility, not evidence of an active plot updater. A manual read-only
parse of the live GDN2 log at inspection time showed:

```text
raw_points: 77
effective_points: 77
superseded_points: 0
avg_window: 5
latest_step: 1925
latest_loss: 4.7270
latest_time: 2026-07-22T09:09:14+00:00
smoothed_loss: 4.722640
tokens_per_step: 65536
latest_tokens: 126156800
checkpoint_steps: 500, 1000, 1500
```

Those values are a point-in-time observation and will be stale as the live run
continues.

## E97 Upload Path

No persistent repository upload script, launcher hook, cron job, systemd unit,
tmux session, or running upload daemon was found for E97 checkpoint uploads.
Searches of launch/supervisor/train scripts showed no `aws s3`, `boto3`, or
S3 upload integration in the training path. The current visible source of truth
is `docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md`, which documents an
operator-run upload protocol rather than a daemon or supervisor hook.

Verified documented local checkpoint source for final E97:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/checkpoint_step_2300930_loss_2.4365.pt
```

Verified bucket and run prefix:

```text
bucket: spinozans
current E97 prefix:
  emender/e97-diloco/emender_E97_1.3B_20260709_084606/
latest pointer:
  emender/e97-diloco/latest_emender_E97_1.3B.json
```

Documented upload selection and trigger:

```text
select exactly one intended checkpoint by numeric step;
ignore temporary/incomplete files;
verify plausible nonzero size and completed local sha256sum;
upload only when requested/confirmed for sharing;
do not use aws s3 sync from the whole run directory.
```

Documented upload command shape:

```bash
aws s3 cp "$CKPT" "${S3_PREFIX}/$(basename "$CKPT")" --sse AES256
aws s3 cp "$META_DIR/" "$S3_PREFIX/" --recursive --sse AES256
aws s3 cp "$META_DIR/latest_emender_E97_1.3B.json" \
  s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json \
  --sse AES256
```

The metadata bundle is generated under `/tmp/${RUN_NAME}_step_${STEP}_s3_meta`
and includes sidecar SHA files, `args.json`, `run_manifest.json`,
`supervisor.log` if present, `latest_symlink_target.txt`, a process snapshot,
`run_log_tail_1000.txt`, `manifest.json`, `metadata_files.sha256`, and the
top-level latest pointer JSON.

Cadence/trigger is manual/operator milestone upload. Retry/synchronization
behavior visible in the protocol is AWS CLI `cp`, including its multipart
behavior for large files, plus explicit post-upload verification. There is no
documented loop, daemon, retry supervisor, or periodic sync job, and read-only
process/scheduler checks found none.

Verified current final remote checkpoint metadata:

```json
{
  "ContentLength": 7719680116,
  "LastModified": "2026-07-22T08:03:56+00:00",
  "ServerSideEncryption": "AES256",
  "ETag": "\"f3f88f4a11fad751ee203baa5c10822f-116\""
}
```

Current remote E97 checkpoint steps visible under
`s3://spinozans/emender/e97-diloco/`:

```text
step_1282500/checkpoint_step_1282500_loss_2.5175.pt
step_1525000/checkpoint_step_1525000_loss_2.4378.pt
step_2300930/checkpoint_step_2300930_loss_2.4365.pt
```

Each checkpoint prefix includes the large checkpoint object plus metadata such
as `args.json`, `launch_manifest.json`, `run_manifest.json`, `supervisor.log`,
`run_log_tail_1000.txt`, `manifest.json`, SHA sidecar files, and latest-pointer
metadata. The `step_1525000` prefix also contains
`e97_diloco_loss_curve_smoothed_20260623.png` and `loss_metrics.json`.

Historical WG archive records show earlier manual upload tasks selected a
single stable checkpoint, checked stable size/mtime and no open writer, uploaded
the checkpoint plus metadata/manifest/SHA files with AWS CLI, and then verified
the remote listing/metadata. Those records used an older prefix:

```text
s3://spinozans/emender/e97-diloco/levelE97_100m_20260623_103742/
```

Those older-prefix objects were not present in the current recursive listing of
`s3://spinozans/emender/e97-diloco/`, so they should be treated as historical
WG evidence rather than current remote state.

Inferred upload trigger/cadence: manual/operator or WG-task invocation at
selected milestones, not periodic upload. This is an inference from the absence
of a repository uploader, active uploader process, supervisor/cron/systemd
caller, and from the archived upload task logs. No retry loop or continuous sync
behavior was visible; the archived tasks verified completion after upload rather
than configuring a daemon.

## Live GDN2 Upload State

Verified GDN2 local checkpoint source directory:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444
```

Verified first live GDN2 checkpoint at inspection time:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444/checkpoint_step_500_loss_5.8263.pt
```

No upload configuration for GDN2 was found in
`scripts/launch_gdn2_mlp_8gpu_diloco.sh`, the live command line, process state,
tmux panes, cron, or user systemd. The ops handoff maps to GDN2 only as a
procedure an operator could intentionally adapt after selecting a stable GDN2
checkpoint; it does not configure the currently running GDN2 control to upload.
Bounded read-only S3 prefix checks for `emender/gdn2` returned `null`, and a
bounded query for GDN2-named keys under `emender/` returned an empty list.
A bounded top-level delimiter listing under `emender/` returned only the E97
lineage:

```text
emender/e97-diloco/
```

Conclusion: the live GDN2 run is not visibly configured to upload checkpoints
and no GDN2 remote checkpoint objects were visible at inspection time.

## Safe Read-Only Commands

Inspect E97 plot artifact freshness:

```bash
stat -c '%y %s %n' docs/experiments/figures/e97_diloco_loss_curve_20260623.png
```

Inspect E97 log freshness and recent loss values:

```bash
ls -lt /mnt/nvme1n1/erikg/diloco_8gpu/emender/run*.log | head
rg 'step\\s+[0-9]+ \\| loss' /mnt/nvme1n1/erikg/diloco_8gpu/emender/run*.log | tail -20
tail -80 /mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260722T055705Z.log
```

Check whether any plot or upload process is currently running:

```bash
ps -eo pid,ppid,lstart,etime,stat,cmd | rg -i 'plot_e97|plot.*gdn2|matplotlib|aws s3|s3api|spinozans' || true
```

Recompute E97 parsed summary without writing a plot:

```bash
python - <<'PY'
from pathlib import Path
import scripts.plot_e97_diloco_loss as p
root = Path('/mnt/nvme1n1/erikg/diloco_8gpu/emender')
logs = sorted(root.glob('run*.log'), key=lambda x: (x.stat().st_mtime, x.name))
records, resumes, saved = p.parse_logs(logs)
kept, superseded = p.effective_lineage(records)
window = p.smoothing_window(len(kept)) if kept else 0
smoothed = p.moving_average([r.loss for r in kept], window) if kept else []
latest = kept[-1]
print('raw_points', len(records))
print('effective_points', len(kept))
print('superseded_points', len(superseded))
print('resume_steps', sorted({r.step for r in resumes}))
print('avg_window', window)
print('latest_step', latest.step)
print('latest_raw_loss', latest.loss)
print('latest_time', latest.timestamp.isoformat())
print('smoothed_loss', smoothed[-1])
print('tokens_per_step', 4 * 2048 * 8)
print('latest_tokens', latest.step * 4 * 2048 * 8)
PY
```

Recompute the same current summary for the live GDN2 log without writing a
plot:

```bash
python - <<'PY'
from pathlib import Path
import scripts.plot_e97_diloco_loss as p
root = Path('/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp')
logs = sorted(root.glob('run*.log'), key=lambda x: (x.stat().st_mtime, x.name))
records, resumes, saved = p.parse_logs(logs)
kept, superseded = p.effective_lineage(records)
window = p.smoothing_window(len(kept)) if kept else 0
smoothed = p.moving_average([r.loss for r in kept], window) if kept else []
latest = kept[-1]
print('raw_points', len(records))
print('effective_points', len(kept))
print('superseded_points', len(superseded))
print('checkpoint_steps', sorted({c.step for c in saved + p.checkpoint_files(root)}))
print('avg_window', window)
print('latest_step', latest.step)
print('latest_raw_loss', latest.loss)
print('latest_time', latest.timestamp.isoformat())
print('smoothed_loss', smoothed[-1])
print('tokens_per_step', 4 * 2048 * 8)
print('latest_tokens', latest.step * 4 * 2048 * 8)
PY
```

Inspect live GDN2 process and log freshness:

```bash
cat /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.pid
ps -fp "$(cat /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.pid)"
stat -c '%y %s %n' /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log
rg 'step\\s+[0-9]+ \\| loss|saved checkpoint' /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log | tail -30
find /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs -maxdepth 2 -name 'checkpoint_step_*_loss_*.pt' -printf '%T+ %s %p\n' | sort | tail -20
```

List a bounded sample of E97 remote checkpoint objects:

```bash
timeout 30s aws s3api list-objects-v2 \
  --bucket spinozans \
  --prefix emender/e97-diloco/ \
  --max-items 100 \
  --query 'Contents[].{LastModified:LastModified,Size:Size,Key:Key}' \
  --output table
```

List only the known final E97 run prefix:

```bash
timeout 30s aws s3api list-objects-v2 \
  --bucket spinozans \
  --prefix emender/e97-diloco/emender_E97_1.3B_20260709_084606/ \
  --max-items 100 \
  --query 'Contents[].{LastModified:LastModified,Size:Size,Key:Key}' \
  --output table
```

Inspect exact metadata for the final E97 checkpoint:

```bash
aws s3api head-object \
  --bucket spinozans \
  --key emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt \
  --query '{ContentLength:ContentLength,LastModified:LastModified,ServerSideEncryption:ServerSideEncryption,ETag:ETag}' \
  --output json
```

Read the final E97 SHA sidecar without downloading the checkpoint:

```bash
aws s3 cp s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt.sha256 -
```

Check whether likely GDN2 prefixes are visible remotely:

```bash
for prefix in emender/gdn2 emender/gdn2_mlp emender/gdn2-mlp; do
  echo "== $prefix =="
  timeout 30s aws s3api list-objects-v2 \
    --bucket spinozans \
    --prefix "$prefix" \
    --max-items 50 \
    --query 'Contents[].Key' \
    --output text
done
```

List top-level `emender/` prefixes without recursing:

```bash
timeout 30s aws s3api list-objects-v2 \
  --bucket spinozans \
  --prefix emender/ \
  --delimiter / \
  --max-items 100 \
  --query 'CommonPrefixes[].Prefix' \
  --output text
```

Check local scheduler hooks for plot/upload jobs:

```bash
crontab -l 2>/dev/null | rg -i 'e97|gdn2|plot|s3|spinozans|aws' || true
systemctl --user list-units --all --no-pager | rg -i 'e97|gdn2|plot|s3|spinozans|aws' || true
systemctl --user list-timers --all --no-pager | rg -i 'e97|gdn2|plot|s3|spinozans|aws' || true
tmux list-sessions 2>/dev/null || true
```

## Validation Checklist

- Relevant scripts, processes, outputs, local source paths, S3 bucket, and S3
  prefixes are named above.
- Plot smoothing and token conversion are described with final E97 values.
- E97 remote checkpoint presence and live GDN2 plot/upload state were checked
  read-only.
- Runtime files, processes, training state, and S3 objects were not changed.
- Read-only monitoring commands are included above.
