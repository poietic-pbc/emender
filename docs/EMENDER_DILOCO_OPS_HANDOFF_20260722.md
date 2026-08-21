# Emender DiLoCo Ops Handoff

Date: 2026-07-22

Purpose: give a fresh chat agent the operational protocol for answering
"status" / "how's it going" requests on the Emender DiLoCo runs, refreshing the
hosted smoothed loss plot, and uploading shareable checkpoints to the
`spinozans` S3 bucket.

This complements `docs/GDN2_MLP_DILOCO_HANDOFF_20260711.md`, which covers the
scientific GDN2 model switch. This document is about live-run observability and
artifact publishing.

## Current E97 Final Reference

The completed E97 run is:

```text
run_root: /mnt/nvme1n1/erikg/diloco_8gpu/emender
run_dir:  /mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606
final_checkpoint: checkpoint_step_2300930_loss_2.4365.pt
final_step: 2,300,930
tokens_per_step: 65,536
final_tokens: 150,793,748,480
final_loss_last100: 2.4365
sha256: 0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
s3_checkpoint: s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt
s3_manifest:   s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json
latest_pointer: s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json
```

The local E97 supervisor stop file is:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.stop
```

Do not remove it unless intentionally resuming E97.

## Status Reply Contract

When the human asks "status", "how's it going", "update plot", or similar, the
agent should produce a compact but complete status packet. Do the refresh first
if asked, then report:

- run state: whether `supervise_emender`, `torchrun`, or `train.py` is active;
- latest plotted step, timestamp, raw loss, and smoothed loss if available;
- tokens seen and percent of target, using the run's true tokens-per-step;
- estimated BPB and the multiplier used;
- latest checkpoint path, step, loss, file size, and whether it is healthy enough
  to upload/share;
- hosted plot URL and verification result;
- S3 checkpoint URI if an upload was requested or already confirmed;
- any risk that matters now, such as disk pressure, missing logs, stalled
  process, old supervisor restart, or S3 upload still in progress.

Keep raw loss and checkpoint/save loss separate. The plot script reports the
latest raw log loss. A checkpoint filename usually records the run's save-time
loss summary, which can differ.

Example response shape:

```text
Plot refreshed: http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png
Run state: stopped; no torchrun/train.py/supervisor process active.
Latest point: step 2,300,930, 150.794B tokens, raw loss <from plot stdout>.
Smoothed/status loss: use the moving-average value if computed separately, or
state that the red curve is the script's moving average.
Estimated BPB: loss * 0.381362685934 using bytes_per_token=3.783.
Latest checkpoint: <path>, 7.19 GiB, sha256 <hash>.
S3: <checkpoint URI or "not uploaded in this refresh">.
```

## Smoothed Plot Refresh

Canonical plot script:

```text
scripts/plot_e97_diloco_loss.py
```

Canonical E97 run root:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender
```

Canonical hosted target:

```text
ssh/scp target: erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png
public URL:     http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png
```

The script deduplicates resumed-over steps, plots raw observed loss in blue, and
plots a red moving average. The moving-average window is:

```text
min(80, max(5, effective_point_count // 40))
```

For the mature E97 run this is normally `80` points.

Generate into `/tmp` to avoid dirtying the repo. The historical default output
is `docs/experiments/figures/e97_diloco_loss_curve_20260623.png`, but the human
explicitly does not want generated PNGs committed.

```bash
cd /home/erikg/ndm
RUN_ROOT=/mnt/nvme1n1/erikg/diloco_8gpu/emender
PLOT=/tmp/e97_diloco_loss_curve_20260623.png
OUT=/tmp/e97_diloco_loss_curve_20260623.refresh.txt

python scripts/plot_e97_diloco_loss.py \
  --run-root "$RUN_ROOT" \
  --output "$PLOT" | tee "$OUT"
```

The script prints:

```text
output=...
sources=...
raw_points=...
effective_points=...
superseded_points=...
resume_steps=...
checkpoint_steps=...
moving_average_window=...
latest_step=...
latest_loss=...
latest_smoothed_loss=...
latest_time=...
```

Upload and verify:

```bash
scp "$PLOT" erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png

sha256sum "$PLOT"
ssh erik@hypervolu.me \
  'sha256sum www/emender/e97_diloco_loss_curve_20260623.png'
curl -fsSI http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png \
  | sed -n '1p;/content-type/Ip'
curl -fsSL http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png \
  | sha256sum
```

The local, SSH remote, and HTTP SHA-256 values should match. The HTTP headers
should show status `200` and content type `image/png`.

Do not stage the PNG:

```bash
git status --short
git diff --cached --name-only
```

If a PNG under `docs/experiments/figures/` was generated for convenience, it is
an operational artifact. Leave it uncommitted or intentionally restore only that
file after confirming no other agent's changes are involved.

## Token And BPB Math

For the completed E97 local 8-GPU run:

```text
tokens_per_step = world_size * batch_size * chunk_size * grad_accum
                = 8 * 4 * 2048 * 1
                = 65,536

tokens_seen = latest_step * 65,536
```

For the Pile/p50k E97 run, `heldout_bytes_per_token=3.783` was recorded in the
run `args.json`. The approximate BPB conversion is:

```text
BPB = loss_nats_per_token * log2(e) / bytes_per_token
BPB = loss_nats_per_token * 0.381362685934
```

Use this only as an estimate for perspective. It is not a replacement for a
proper held-out evaluation.

Quick calculation:

```bash
LOSS=2.4365 python - <<'PY'
import math, os
loss = float(os.environ["LOSS"])
bytes_per_token = 3.783
print(f"bpb_est={loss * math.log2(math.e) / bytes_per_token:.6f}")
PY
```

## Finding The Latest Good Checkpoint

Use numeric step ordering, not lexicographic ordering. Ignore temp and incomplete
files.

```bash
RUN_DIR=/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606

find "$RUN_DIR" -maxdepth 1 -type f \
  -name 'checkpoint_step_*_loss_*.pt' \
  ! -name '*.tmp' \
  ! -name '*.incomplete*' \
  -printf '%f %s %p\n' | sort -V | tail -20
```

Before upload, confirm:

- filename step is the intended step;
- size is plausible and nonzero; final E97 is about `7.19 GiB`;
- the save completed and no `.tmp` or `.incomplete*` file is being selected;
- if training is active, the selected checkpoint is not still being written;
- `sha256sum` completes locally.

For final E97:

```bash
CKPT=/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/checkpoint_step_2300930_loss_2.4365.pt
stat -c '%n %s bytes' "$CKPT"
sha256sum "$CKPT"
```

Do not compress PyTorch checkpoint files for normal handoff. They are mostly
high-entropy tensor data, compression usually saves little, slows the upload,
and makes direct resume on Frontier less convenient. Upload the raw `.pt` plus
metadata and checksums.

## S3 Checkpoint Upload Protocol

Bucket:

```text
s3://spinozans
```

Canonical E97 prefix shape:

```text
s3://spinozans/emender/e97-diloco/<run_name>/step_<step>/
```

For final E97:

```text
s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/
```

Use server-side encryption. Do not use `aws s3 sync` from a whole run directory;
it is too easy to upload old checkpoints or temp files. Upload a selected
checkpoint and a small metadata bundle.

Recommended command template:

```bash
cd /home/erikg/ndm

RUN_NAME=emender_E97_1.3B_20260709_084606
RUN_ROOT=/mnt/nvme1n1/erikg/diloco_8gpu/emender
RUN_DIR="$RUN_ROOT/runs/$RUN_NAME"
STEP=2300930
LOSS=2.4365
CKPT="$RUN_DIR/checkpoint_step_${STEP}_loss_${LOSS}.pt"
S3_PREFIX="s3://spinozans/emender/e97-diloco/${RUN_NAME}/step_${STEP}"
META_DIR="/tmp/${RUN_NAME}_step_${STEP}_s3_meta"

rm -rf "$META_DIR"
mkdir -p "$META_DIR"

stat -c '%n %s bytes' "$CKPT"
sha256sum "$CKPT" | tee "$META_DIR/$(basename "$CKPT").sha256"

cp "$RUN_DIR/args.json" "$META_DIR/args.json"
cp "$RUN_DIR/run_manifest.json" "$META_DIR/run_manifest.json"
[ -f "$RUN_ROOT/supervisor.log" ] && cp "$RUN_ROOT/supervisor.log" "$META_DIR/supervisor.log"
readlink "$RUN_DIR/latest.pt" > "$META_DIR/latest_symlink_target.txt" 2>/dev/null || true

ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd \
  | rg 'supervise_emender|torchrun|train.py' \
  > "$META_DIR/run_command_process_snapshot.txt" || true

tail -n 1000 "$RUN_ROOT"/run*.log > "$META_DIR/run_log_tail_1000.txt"

cat > "$META_DIR/manifest.json" <<EOF
{
  "model": "emender_E97_1.3B",
  "run": "$RUN_NAME",
  "step": $STEP,
  "loss": $LOSS,
  "tokens_per_step": 65536,
  "tokens": $((STEP * 65536)),
  "world_size": 8,
  "per_gpu_batch_size": 4,
  "chunk_size": 2048,
  "diloco_island_size": 0,
  "diloco_k": 250,
  "diloco_outer_optimizer": "avg",
  "diloco_outer_lr": 1.0,
  "diloco_outer_beta": 0.0,
  "semantics": "pure local-DiLoCo: one independent replica per GPU, periodic all-rank model averaging",
  "checkpoint_local_path": "$CKPT",
  "checkpoint_name": "$(basename "$CKPT")",
  "checkpoint_s3_uri": "${S3_PREFIX}/$(basename "$CKPT")",
  "checkpoint_sha256": "$(cut -d' ' -f1 "$META_DIR/$(basename "$CKPT").sha256")",
  "checkpoint_size_bytes": $(stat -c '%s' "$CKPT"),
  "s3_prefix": "${S3_PREFIX}/",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

sha256sum "$META_DIR"/* > "$META_DIR/metadata_files.sha256"
cp "$META_DIR/manifest.json" "$META_DIR/latest_emender_E97_1.3B.json"

aws s3 cp "$CKPT" "${S3_PREFIX}/$(basename "$CKPT")" --sse AES256
aws s3 cp "$META_DIR/" "$S3_PREFIX/" --recursive --sse AES256
aws s3 cp "$META_DIR/latest_emender_E97_1.3B.json" \
  s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json \
  --sse AES256
```

If a checkpoint is much larger than normal, keep using `aws s3 cp`; the AWS CLI
will do multipart upload. For very large files, add `--expected-size` with the
exact byte size if the CLI warns about multipart part limits.

## S3 Verification

First verify the object exists, has the expected byte length, and is encrypted:

```bash
aws s3api head-object \
  --bucket spinozans \
  --key emender/e97-diloco/${RUN_NAME}/step_${STEP}/$(basename "$CKPT")

aws s3 ls "${S3_PREFIX}/" --human-readable --summarize
aws s3 cp "${S3_PREFIX}/manifest.json" -
aws s3 cp s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json -
```

The ETag of a multipart upload is not a SHA-256 checksum. The authoritative
content checksum is the uploaded `.sha256` sidecar. If a byte-for-byte remote
checksum is required and the bandwidth/time cost is acceptable:

```bash
aws s3 cp "${S3_PREFIX}/$(basename "$CKPT")" - | sha256sum
```

Then compare with the sidecar:

```bash
aws s3 cp "${S3_PREFIX}/$(basename "$CKPT").sha256" -
```

## S3 Retention And Cleanup

Use S3 cleanup only as a separate explicit action. The safe sequence is:

1. Upload and verify the new latest checkpoint.
2. Confirm the top-level latest pointer names the new checkpoint.
3. List old prefixes:

```bash
aws s3 ls s3://spinozans/emender/e97-diloco/${RUN_NAME}/
```

4. Remove only exact obsolete `step_<step>/` prefixes after the human has
   agreed they are no longer needed:

```bash
aws s3 rm --recursive \
  s3://spinozans/emender/e97-diloco/${RUN_NAME}/step_<old_step>/
```

Never run recursive deletion on `s3://spinozans/emender/` or the run-name prefix
without first listing the target and confirming that it contains only the
intended checkpoint step.

## Commit Hygiene

Commit handoff docs and scripts only when requested. Do not commit generated
PNGs, run logs, checkpoints, temporary metadata directories, or large artifacts.

Before any commit:

```bash
git status --short
git diff --cached --name-only
```

Stage explicit files only, for example:

```bash
git add docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md
```
