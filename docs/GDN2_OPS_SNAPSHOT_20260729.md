# GDN2 Ops Snapshot - 2026-07-29

- Snapshot UTC: `2026-07-29T14:49:36Z`
- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Health: `torchrun active with 8 worker ranks`; torchrun PIDs `[3754241]`, worker rank PIDs `[3754608, 3754609, 3754611, 3754612, 3754613, 3754614, 3754615, 3754616]`.
- Source policy: `rank-0/main stdout, finite complete records only, dedupe by optimizer step keeping latest timestamp/order, no interpolation`
- Token validation: `8 * 4 * 2048 * 1 = 65536` aggregate tokens/optimizer step.

## Current GDN2

- Effective records: `23185` from step `25` to `579625`; raw records `23185`, duplicates removed `0`.
- Token range: `1638400` to `37986304000`; current `37986304000` (`37.986304000B`), `25.324202667%` of 150B.
- Latest loss at step `579625` (`2026-07-29T14:49:38Z`): raw `2.4394000000`, smoothed `2.5346462500` with trailing MA window `80`.
- Last 100 optimizer-step mean: `2.5397000000` over `4` observations, bounds `[579526, 579625]`, cadence `25`, every-step coverage `False`.
- Last 1000 optimizer-step mean: `2.5338525000` over `40` observations, bounds `[578626, 579625]`, cadence `25`, every-step coverage `False`.

## Matched E97 Comparison

- Cutoff: step `579625`, tokens `37986304000` (`37.986304000B`).
- Alignment: `23185` exact common records, step range `[25, 579625]`, token range `[1638400, 37986304000]`, E97-only `0`, GDN2-only `0`.
- Raw cutoff losses: E97 `2.5946000000`, GDN2 `2.4394000000`, delta GDN2-E97 `-0.1552000000`.
- Smoothed cutoff losses: E97 `2.5382487500`, GDN2 `2.5346462500`, delta `-0.0036025000`.
- Last 100 mean delta: `-0.0597000000`; E97 `2.5994000000`, GDN2 `2.5397000000`.
- Last 1000 mean delta: `0.0039375000`; E97 `2.5299150000`, GDN2 `2.5338525000`.
- Mean signed delta over all aligned observations: `-0.0341800776`; recent 1000-step interval: `0.0039375000`.

## Publication

- GDN2 plot: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260729_20260729T144936Z/gdn2_mlp_diloco_loss_curve_20260729T144936Z.png`, SHA-256 `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4`, URL `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`.
- Comparison overlay: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260729_20260729T144936Z/gdn2_vs_e97_matched_tokens_20260729.png`, SHA-256 `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04`, URL `http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260729.png`.
- GDN2 stable local/SSH/HTTP hashes: `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4` / `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4` / `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4`; HTTP `200` `image/png`.
- Overlay local/SSH/HTTP hashes: `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04` / `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04` / `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04`; HTTP `200` `image/png`.

## Protected Artifact Hashes

- E97_standalone: before refresh SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after GDN2 publish SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after overlay SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; unchanged after overlay vs after GDN2 `True`.
- GDN2_standalone: before refresh SSH/HTTP `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1` / `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1`; after GDN2 publish SSH/HTTP `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4` / `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4`; after overlay SSH/HTTP `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4` / `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4`; unchanged after overlay vs after GDN2 `True`.
- prior_5p118B_comparison: before refresh SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; after GDN2 publish SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; after overlay SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; unchanged after overlay vs after GDN2 `True`.
- prior_20260724_comparison: before refresh SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after GDN2 publish SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after overlay SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; unchanged after overlay vs after GDN2 `True`.
- prior_20260728_comparison: before refresh SSH/HTTP `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; after GDN2 publish SSH/HTTP `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; after overlay SSH/HTTP `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; unchanged after overlay vs after GDN2 `True`.

## Throughput and ETA

- recent_1h: bounds `2026-07-29T13:49:48Z` step `576300` to `2026-07-29T14:49:38Z` step `579625`; samples `134`, steps/sec `0.926183844`, tokens/sec `60698.384`.
- recent_6h: bounds `2026-07-29T08:49:51Z` step `559650` to `2026-07-29T14:49:38Z` step `579625`; samples `800`, steps/sec `0.925325427`, tokens/sec `60642.127`.
- since_launch: bounds `2026-07-22T08:35:16Z` step `25` to `2026-07-29T14:49:38Z` step `579625`; samples `23185`, steps/sec `0.924015802`, tokens/sec `60556.300`.
- target_150b: target `150000000000`, percent `25.324202667%`, remaining `112013696000` tokens / `1709193.359375` steps, step at/above `2288819`, overshoot `41984` tokens, primary ETA `21d 9h 5m 27s` ending `2026-08-19T23:55:04Z`, range `2026-08-19T23:26:32Z` to `2026-08-20T00:38:42Z`.
- target_e97_parity: target `150793748480`, percent `25.190901070%`, remaining `112807444480` tokens / `1721305.000000` steps, step at/above `2300930`, overshoot `0` tokens, primary ETA `21d 12h 43m 36s` ending `2026-08-20T03:33:13Z`, range `2026-08-20T03:04:29Z` to `2026-08-20T04:17:10Z`.

Assumptions: no downtime, unchanged 8-GPU rate, and 65,536 aggregate tokens per optimizer step.

No training control, checkpoint write/modification, or S3 command was run.
