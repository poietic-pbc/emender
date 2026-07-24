# GDN2 Loss Refresh - 2026-07-24

Task: `gdn2-refresh-20260724`

## Source And Health

- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Run root: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp`
- Run dir: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444`
- Log source: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log`
- Rank/source policy: rank-0/main training stdout in shared `run.log`; one complete logged loss record every `log_every=25` optimizer steps; duplicate optimizer steps keep the latest timestamp/order record.
- Health before refresh: torchrun PID `3754241` active with eight worker PIDs `3754608`, `3754609`, `3754611`, `3754612`, `3754613`, `3754614`, `3754615`, `3754616`.
- Health after publication: the same torchrun PID and eight worker PIDs were active; `nvidia-smi pmon -c 1` showed one `python3` client on each GPU 0-7 at 98-99% SM utilization.

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

- Ops directory: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/gdn2-refresh-20260724_20260724T054317Z`
- Snapshot UTC: `2026-07-24T05:43:17Z`
- Snapshot: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/gdn2-refresh-20260724_20260724T054317Z/run.log.snapshot.tail_2097152_20260724T054317Z.log`
- Snapshot size: `926022` bytes, bounded tail limit `2097152` bytes
- Parser summary: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/gdn2-refresh-20260724_20260724T054317Z/gdn2_mlp_diloco_loss_curve_summary_20260724T054317Z.json`
- Parser summary SHA-256: `ca912aff8a6a929f0907f8b1418265894a7c8c5eba96ea105bf1348fa9ea482a`
- PNG: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/gdn2-refresh-20260724_20260724T054317Z/gdn2_mlp_diloco_loss_curve_20260724T054317Z.png`
- PNG SHA-256: `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13`
- PNG size/readability: `185158` bytes, readable PNG via matplotlib, image shape `[1200, 2240, 4]`

## Parser And Plot Summary

- Parsed raw/effective points: `6024` / `6024`
- Duplicate steps removed: `0`
- Superseded points: `0`
- Step range: `25..150600`
- Token range: `1638400..9869721600`
- Sanity checks: finite effective records, strictly increasing steps, strictly increasing tokens, no malformed step-like lines, no dropped final partial line, no rejected final record.
- Smoothing method: trailing moving average over effective plotted loss records.
- Smoothing window: `min(80, max(5, effective_point_count // 40)) = 80` points.
- Latest raw loss: `2.6162` at step `150600`, logged at `2026-07-24T05:43:00+00:00`.
- Latest plotted smoothed loss: `2.73039`.
- Current aggregate tokens: `9869721600` (`9.8697216` billion).

Intervals are optimizer-step intervals, but loss is logged every 25 optimizer steps. The means below are arithmetic means over logged observations inside each interval, with no interpolation.

- Last 100 optimizer-step interval: bounds `150501..150600`; observations `4` at cadence `25` optimizer steps; observed steps `150525, 150550, 150575, 150600`; every optimizer step represented: `false`; mean logged-observation loss `2.62155`.
- Last 1000 optimizer-step interval: bounds `149601..150600`; observations `40` at cadence `25` optimizer steps; first/last observed steps `149625..150600`; every optimizer step represented: `false`; mean logged-observation loss `2.72214`.

## Publication Verification

- Public URL: `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`
- Prior GDN2 remote SHA-256: `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec`
- Prior GDN2 HTTP SHA-256: `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec`
- Same-directory temporary remote path: `www/emender/.gdn2_mlp_diloco_loss_curve_20260722.png.tmp.gdn2-refresh-20260724.20260724T054317Z.1334600`
- Temporary remote SHA-256 before rename: `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13`
- New GDN2 remote SHA-256: `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13`
- New GDN2 HTTP SHA-256: `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13`
- GDN2 HTTP verification: `HTTP/1.1 200 OK`, `Content-Type: image/png`, `Content-Length: 185158`

Protected E97 target `www/emender/e97_diloco_loss_curve_20260623.png` remained unchanged:

- E97 remote before/after SHA-256: `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`
- E97 HTTP before/after SHA-256: `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`
- E97 HTTP after verification: `HTTP/1.1 200 OK`, `Content-Type: image/png`, `Content-Length: 290596`

No checkpoint, S3, or training control/process-state command was run. Checkpoint files were only listed/stat-read for context; no checkpoint path was written.
