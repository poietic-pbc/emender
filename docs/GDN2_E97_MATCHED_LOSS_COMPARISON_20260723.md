# GDN2 vs E97 Matched Loss Comparison - 2026-07-23

Task: `compare-gdn2-and`

## Cutoff And Sources

- GDN2 snapshot UTC: `2026-07-23T07:58:22Z`
- Matched cutoff: optimizer step `78100`, aggregate tokens `5118361600`
  (`5.1183616B`)
- GDN2 source snapshot:
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-loss_20260723T075822Z/run.log.snapshot.tail_2097152_20260723T075822Z.log`
- E97 source policy: canonical
  `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run*.log` files, matching
  `scripts/plot_e97_diloco_loss.py`.
- Snapshot/output directory:
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/compare-gdn2-and_20260723T081011Z`
- Summary JSON:
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/compare-gdn2-and_20260723T081011Z/gdn2_vs_e97_matched_5p118b_tokens_20260723_summary.json`
- Overlay PNG:
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/compare-gdn2-and_20260723T081011Z/gdn2_vs_e97_matched_5p118b_tokens_20260723.png`

Both runs validate the same aggregate token semantics:

```text
tokens_per_step = world_size * batch_size * chunk_size * grad_accum
                = 8 * 4 * 2048 * 1
                = 65536
```

No token-count fallback was needed because matched optimizer steps are matched
aggregate token counts.

## Reconstruction

- Parsed only complete finite `step | loss | tok/s | global_tok/s | time`
  records.
- Applied rank-0/main stdout policy documented for the hosted curves.
- Deduplicated by optimizer step, keeping the latest timestamp/order record.
- Verified finite records and strictly increasing effective steps/tokens.
- Aligned only on exact common logged optimizer steps through the cutoff.
- Used no interpolation.
- Applied the GDN2 cutoff smoothing window `78` to both aligned series:
  `min(80, max(5, 3124 // 40)) = 78`.

Record counts:

| Series | Raw records through cutoff | Effective records | Duplicates removed | Step range | Token range |
| --- | ---: | ---: | ---: | --- | --- |
| E97 | 6039 | 3124 | 2915 | `25..78100` | `1638400..5118361600` |
| GDN2-MLP | 3124 | 3124 | 0 | `25..78100` | `1638400..5118361600` |

Alignment:

| Metric | Value |
| --- | ---: |
| Aligned record count | 3124 |
| Common step range | `25..78100` |
| Common token range | `1638400..5118361600` |
| E97-only logged steps through cutoff | 0 |
| GDN2-only logged steps through cutoff | 0 |

## Numerical Result

Signed deltas are `GDN2 - E97`, so negative means lower GDN2 logged loss.

| Comparison at cutoff | E97 | GDN2-MLP | Delta |
| --- | ---: | ---: | ---: |
| Raw loss at step `78100` | 2.899800 | 2.851800 | -0.048000 |
| 78-record trailing smoothed loss | 2.930032 | 2.858777 | -0.071255 |

Interval means are arithmetic means over aligned logged observations inside the
optimizer-step interval, with no interpolation. The logging cadence is every 25
optimizer steps, so not every optimizer step is represented.

| Interval | Bounds | Observed bounds | Samples | Cadence | E97 mean | GDN2-MLP mean | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Last 100 optimizer steps | `78001..78100` | `78025..78100` | 4 | 25 | 2.915400 | 2.767400 | -0.148000 |
| Last 1000 optimizer steps | `77101..78100` | `77125..78100` | 40 | 25 | 2.921803 | 2.852405 | -0.069398 |

Descriptive aggregate deltas:

| Metric | Mean signed delta |
| --- | ---: |
| All aligned logged points | 0.082815 |
| Recent 1000-step interval aligned logged points | -0.069398 |

These are descriptive arithmetic means over aligned logged records and do not
imply statistical significance.

## Publication

The overlay was published to a new distinct Hypervolume target:

```text
http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_5p118b_tokens_20260723.png
```

Publication verification:

| Check | Value |
| --- | --- |
| Local PNG SHA-256 | `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` |
| SSH remote SHA-256 | `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` |
| HTTP SHA-256 | `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` |
| HTTP status | `HTTP/1.1 200 OK` |
| HTTP content type | `image/png` |
| HTTP content length | `259569` |
| Collision handling | Target absent before upload; uploaded to same-directory temp and atomically renamed. |

Protected hosted curves remained unchanged:

| Protected curve | Before SSH/HTTP SHA-256 | After SSH/HTTP SHA-256 |
| --- | --- | --- |
| E97 hosted curve | `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` | `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` |
| GDN2 hosted curve | `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec` / `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec` | `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec` / `5c4d47c4d891d6b122517fec6cbf8e3b806cefd9acfda948edbebb088e0cf7ec` |

No training control, checkpoint modification, or S3 command was performed.
