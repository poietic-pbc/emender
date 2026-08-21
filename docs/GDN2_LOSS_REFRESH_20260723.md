# GDN2 Loss Refresh - 2026-07-23

Task: `refresh-gdn2-loss`

## Source And Health

- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Run root: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp`
- Run dir: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444`
- Log source: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log`
- Rank/source policy: rank-0/main training stdout in shared `run.log`; one complete logged loss record every `log_every=25` optimizer steps; duplicate optimizer steps would keep the latest timestamp/order record.
- Health before refresh: torchrun PID `3754241` active, eight worker PIDs `3754608`, `3754609`, `3754611`, `3754612`, `3754613`, `3754614`, `3754615`, `3754616` active.
- Health after publication: same torchrun and eight worker PIDs active; `nvidia-smi pmon -c 1` showed one `python3` client on each GPU 0-7 at 98-99% SM utilization.

The launch manifest and `args.json` validate:

```text
world_size=8
batch_size=4
chunk_size=2048
grad_accum=1
log_every=25
tokens_per_step = 8 * 4 * 2048 * 1 = 65,536
```

## Local Artifacts

- Ops directory: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-loss_20260723T075822Z`
- Snapshot UTC: `2026-07-23T07:58:22Z`
- Snapshot: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-loss_20260723T075822Z/run.log.snapshot.tail_2097152_20260723T075822Z.log`
- Snapshot size: `482699` bytes, bounded tail limit `2097152` bytes
- Parser summary: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-loss_20260723T075822Z/gdn2_mlp_diloco_loss_curve_summary_20260723T075822Z.json`
- PNG: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-loss_20260723T075822Z/gdn2_mlp_diloco_loss_curve_20260723T075822Z.png`
- PNG SHA-256: `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec`
- PNG size/readability: `191088` bytes, readable PNG via matplotlib, image shape `[1200, 2240, 4]`

## Parser And Plot Summary

- Parsed raw/effective points: `3124` / `3124`
- Duplicate steps removed: `0`
- Superseded points: `0`
- Step range: `25..78100`
- Token range: `1638400..5118361600`
- Sanity checks: finite effective records, strictly increasing steps, strictly increasing tokens, no malformed step-like lines, no dropped final partial line, no rejected final record.
- Smoothing method: trailing moving average over effective plotted loss records.
- Smoothing window: `min(80, max(5, effective_point_count // 40)) = 78` points.
- Latest raw loss: `2.8518` at step `78100`, logged at `2026-07-23T07:58:06+00:00`.
- Latest plotted smoothed loss: `2.858776923076924`.
- Current aggregate tokens: `5118361600` (`5.1183616` billion).

Intervals are optimizer-step intervals, but loss is logged every 25 optimizer steps. The means below are arithmetic means over logged observations inside each interval, with no interpolation.

- Last 100 optimizer-step interval: bounds `78001..78100`; observations `4` at cadence `25` optimizer steps; observed steps `78025, 78050, 78075, 78100`; every optimizer step represented: `false`; mean logged-observation loss `2.7674`.
- Last 1000 optimizer-step interval: bounds `77101..78100`; observations `40` at cadence `25` optimizer steps; first/last observed steps `77125..78100`; every optimizer step represented: `false`; mean logged-observation loss `2.852405`.

## Publication Verification

- Public URL: `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`
- Prior GDN2 remote SHA-256: `b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333`
- Prior GDN2 HTTP SHA-256: `b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333`
- Same-directory temporary remote path: `www/emender/.gdn2_mlp_diloco_loss_curve_20260722.png.tmp.refresh-gdn2-loss.20260723T075822Z.301427`
- Temporary remote SHA-256 before rename: `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec`
- New GDN2 remote SHA-256: `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec`
- New GDN2 HTTP SHA-256: `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec`
- GDN2 HTTP verification: `HTTP/1.1 200 OK`, `Content-Type: image/png`, `Content-Length: 191088`

Protected E97 target `www/emender/e97_diloco_loss_curve_20260623.png` remained unchanged:

- E97 remote before/after SHA-256: `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`
- E97 HTTP before/after SHA-256: `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`
- E97 HTTP after verification: `HTTP/1.1 200 OK`, `Content-Type: image/png`, `Content-Length: 290596`

No checkpoint, S3, or training control/process-state command was run. Checkpoint files were only listed/stat-read for context; no checkpoint path was written.
