# GDN2-MLP vs E97 Matched-Token Loss Comparison - 2026-07-24

This comparison reproduces the prior `compare-gdn2-and` protocol at the
authoritative `gdn2-refresh-20260724` cutoff/state.

## Cutoff and Sources

- Predecessor snapshot UTC: `2026-07-24T05:43:17Z`.
- GDN2 authoritative cutoff: optimizer step `150600`, aggregate tokens
  `9869721600` (`9.8697216B`).
- Token validation: both runs use exactly `65536` aggregate tokens per optimizer
  step (`world_size=8 * batch_size=4 * chunk_size=2048 * grad_accum=1`).
- GDN2 source: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log`, snapshotted
  read-only into the ops artifact directory.
- E97 sources: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run*.log`, snapshotted
  read-only into the ops artifact directory.
- Record policy: finite complete rank-0/main stdout records only; deduplicate by
  optimizer step keeping the latest timestamp/order record; no interpolation.

## Source Validation

| Series | Raw through cutoff | Effective | Duplicates removed | Step range | Token range | Finite | Monotonic |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| E97 | `8939` | `6024` | `2915` | `25..150600` | `1638400..9869721600` | yes | strict steps/tokens |
| GDN2-MLP | `6024` | `6024` | `0` | `25..150600` | `1638400..9869721600` | yes | strict steps/tokens |

Malformed step-like lines were `0` for both. No final partial line was dropped
for either source.

## Alignment

- Exact aligned logged steps: `6024`.
- Common optimizer-step range: `25..150600`.
- Common token range: `1638400..9869721600`.
- Missing/nonoverlapping records through cutoff: E97-only `0`, GDN2-only `0`.
- Cadence in the recent intervals: every `25` optimizer steps; every optimizer
  step is not represented; no interpolation was used.

## Metrics

Signed deltas are `GDN2 - E97`.

| Metric | E97 | GDN2-MLP | Signed delta |
| --- | ---: | ---: | ---: |
| Raw loss at step `150600` | `2.9559` | `2.6162` | `-0.3397000000` |
| Trailing MA loss at step `150600` | `2.8038425000` | `2.7303900000` | `-0.0734525000` |
| Last 100 optimizer-step interval mean | `2.8190750000` | `2.6215500000` | `-0.1975250000` |
| Last 1000 optimizer-step interval mean | `2.8256225000` | `2.7221400000` | `-0.1034825000` |

Smoothing used the predecessor GDN2 rule identically for both aligned series:
trailing moving average over effective plotted loss records,
`window=min(80,max(5,n//40))=80` for `6024` GDN2 effective records.

Interval coverage:

- Last 100 optimizer steps: bounds `150501..150600`, `4` logged observations for
  both, first/last observed steps `150525..150600`, token bounds
  `9863233536..9869721600`, observed token span `9864806400..9869721600`.
- Last 1000 optimizer steps: bounds `149601..150600`, `40` logged observations
  for both, first/last observed steps `149625..150600`, token bounds
  `9804251136..9869721600`, observed token span `9805824000..9869721600`.

Descriptive aligned deltas:

- Mean signed delta over all aligned points: `0.0148174303`.
- Mean signed delta over recent 1000-step interval: `-0.1034825000`.

## Artifacts

- Local overlay:
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/e97-gdn2-comparison-20260724_20260724T055900Z/gdn2_vs_e97_matched_tokens_20260724.png`
- Local overlay SHA-256:
  `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`
- Local summary:
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/e97-gdn2-comparison-20260724_20260724T055900Z/gdn2_vs_e97_matched_tokens_20260724_summary.json`
- Local summary SHA-256:
  `fc471be311b422c316d1f965cbd953f4b4cb9e2fd82b58f1b77ce48f23a1c06f`
- Public URL:
  `http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260724.png`
- Public verification: HTTP `200 OK`, `Content-Type=image/png`,
  `Content-Length=239480`, SHA-256
  `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`.

The target was absent before publication. The file was uploaded to a hidden
same-directory temporary path on `erik@hypervolu.me`, verified by SHA-256, then
atomically renamed to
`www/emender/gdn2_vs_e97_matched_tokens_20260724.png`.

## Protected Artifact Hashes

| Artifact | Before | After |
| --- | --- | --- |
| E97 hosted standalone plot, SSH/HTTP | `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` | `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` |
| GDN2 hosted standalone plot, SSH/HTTP | `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13` | `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13` |
| Prior 5.118B comparison, SSH/HTTP | `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` | `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` |

No training control, checkpoint write/modification, or S3 command was run.
