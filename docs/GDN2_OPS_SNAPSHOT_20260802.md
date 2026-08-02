# GDN2 Ops Snapshot - 20260802

- Snapshot UTC: `2026-08-02T16:08:25Z`
- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Health: `torchrun active with 8 worker ranks`; torchrun PIDs `[3754241]`, worker rank PIDs `[3754608, 3754609, 3754611, 3754612, 3754613, 3754614, 3754615, 3754616]`.
- GPU occupancy: `GPU 0: pid 3754608 sm=98 mem=43; GPU 1: pid 3754609 sm=98 mem=45; GPU 2: pid 3754611 sm=99 mem=44; GPU 3: pid 3754612 sm=99 mem=44; GPU 4: pid 3754613 sm=99 mem=44; GPU 5: pid 3754614 sm=99 mem=44; GPU 6: pid 3754615 sm=99 mem=43; GPU 7: pid 3754616 sm=99 mem=43`.
- GPU temperatures: `GPU0=86C GPU1=72C GPU2=80C GPU3=73C GPU4=81C GPU5=72C GPU6=83C GPU7=81C`.
- Latest DiLoCo merge: `merge_3601_step_900250_4229ms`.
- Latest checkpoint: `checkpoint_step_900000_loss_2.5186.pt`.
- Fatal/error/OOM/NaN scan: `none_found_no_fatal_error_oom_nan_evidence` with `0` hits.
- Source policy: `rank-0/main stdout, finite complete records only, dedupe by optimizer step keeping latest timestamp/order, no interpolation`
- Token validation: `8 * 4 * 2048 * 1 = 65536` aggregate tokens/optimizer step.

## Current GDN2

- Effective records: `36012` from step `25` to `900300`; raw records `36012`, duplicates removed `0`.
- Token range: `1638400` to `59002060800`; current `59002060800` (`59.002060800B`), `39.334707200%` of 150B.
- Latest loss at step `900300` (`2026-08-02T16:08:15Z`): raw `2.4952000000`, smoothed `2.4842462500` with trailing MA window `80`.
- Last 100 optimizer-step mean: `2.5339000000` over `4` observations, bounds `[900201, 900300]`, cadence `25`, every-step coverage `False`.
- Last 1000 optimizer-step mean: `2.4901350000` over `40` observations, bounds `[899301, 900300]`, cadence `25`, every-step coverage `False`.

## Matched E97 Comparison

- Cutoff: step `900300`, tokens `59002060800` (`59.002060800B`).
- Alignment: `36012` exact common records, step range `[25, 900300]`, token range `[1638400, 59002060800]`, E97-only `0`, GDN2-only `0`.
- Raw cutoff losses: E97 `2.5503000000`, GDN2 `2.4952000000`, delta GDN2-E97 `-0.0551000000`.
- Smoothed cutoff losses: E97 `2.5330975000`, GDN2 `2.4842462500`, delta `-0.0488512500`.
- Last 100 mean delta: `-0.0559750000`; E97 `2.5898750000`, GDN2 `2.5339000000`.
- Last 1000 mean delta: `-0.0456600000`; E97 `2.5357950000`, GDN2 `2.4901350000`.
- Mean signed delta over all aligned observations: `-0.0414437410`; recent 1000-step interval: `-0.0456600000`.

## Publication

- GDN2 plot: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260802_20260802T160825Z/gdn2_mlp_diloco_loss_curve_20260722.png`, SHA-256 `879bce2fb3272f852b63345cdf1ceb7e86a4b6edeba231052462a4700398e7c6`, URL `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`.
- Comparison overlay: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260802_20260802T160825Z/gdn2_vs_e97_matched_tokens_20260802.png`, SHA-256 `a0ca3f09fecb16d4761928a3ec40d4ca7c2770dca4d483b98fd873effc6cf4d7`, URL `http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260802.png`.
- GDN2 stable local/SSH/HTTP hashes: `879bce2fb3272f852b63345cdf1ceb7e86a4b6edeba231052462a4700398e7c6` / `879bce2fb3272f852b63345cdf1ceb7e86a4b6edeba231052462a4700398e7c6` / `879bce2fb3272f852b63345cdf1ceb7e86a4b6edeba231052462a4700398e7c6`; HTTP `200` `image/png`.
- Overlay local/SSH/HTTP hashes: `a0ca3f09fecb16d4761928a3ec40d4ca7c2770dca4d483b98fd873effc6cf4d7` / `a0ca3f09fecb16d4761928a3ec40d4ca7c2770dca4d483b98fd873effc6cf4d7` / `a0ca3f09fecb16d4761928a3ec40d4ca7c2770dca4d483b98fd873effc6cf4d7`; HTTP `200` `image/png`.

## Protected Artifact Hashes

- E97_standalone: before refresh SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after GDN2 publish SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after overlay SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; unchanged after overlay vs after GDN2 `True`.
- prior_20260724_comparison: before refresh SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after GDN2 publish SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after overlay SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; unchanged after overlay vs after GDN2 `True`.
- prior_20260728_comparison: before refresh SSH/HTTP `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; after GDN2 publish SSH/HTTP `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; after overlay SSH/HTTP `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; unchanged after overlay vs after GDN2 `True`.
- prior_20260729_comparison: before refresh SSH/HTTP `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04` / `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04`; after GDN2 publish SSH/HTTP `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04` / `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04`; after overlay SSH/HTTP `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04` / `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04`; unchanged after overlay vs after GDN2 `True`.

## Throughput and ETA

- recent_1h: bounds `2026-08-02T15:08:39Z` step `897000` to `2026-08-02T16:08:15Z` step `900300`; samples `133`, steps/sec `0.922818792`, tokens/sec `60477.852`.
- recent_6h: bounds `2026-08-02T10:08:38Z` step `880450` to `2026-08-02T16:08:15Z` step `900300`; samples `795`, steps/sec `0.919961070`, tokens/sec `60290.569`.
- since_launch: bounds `2026-07-22T08:35:16Z` step `25` to `2026-08-02T16:08:15Z` step `900300`; samples `36012`, steps/sec `0.920923015`, tokens/sec `60353.611`.
- target_150b: target `150000000000`, percent `39.334707200%`, remaining `90997939200` tokens / `1388518.359375` steps, step at/above `2288819`, overshoot `41984` tokens, primary ETA `17d 11h 15m 23s` ending `2026-08-20T03:23:37Z`, range `2026-08-20T02:05:43Z` to `2026-08-20T02:57:21Z`.
- target_e97_parity: target `150793748480`, percent `39.127657078%`, remaining `91791687680` tokens / `1400630.000000` steps, step at/above `2300930`, overshoot `0` tokens, primary ETA `17d 14h 54m 48s` ending `2026-08-20T07:03:03Z`, range `2026-08-20T05:44:28Z` to `2026-08-20T06:36:33Z`.

Assumptions: no downtime, unchanged 8-GPU rate, and 65,536 aggregate tokens per optimizer step.

No training control, checkpoint write/modification, or S3 command was run.
