# GDN2 Ops Snapshot - 2026-07-28

- Snapshot UTC: `2026-07-28T13:45:51Z`
- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Health: `torchrun active with 8 worker ranks`; torchrun PID `[3754241]`, launch wrapper PID `[3754406]`, worker rank PIDs `[3754608, 3754609, 3754611, 3754612, 3754613, 3754614, 3754615, 3754616]`.
- Source policy: `rank-0/main stdout, finite complete records only, dedupe by optimizer step keeping latest timestamp/order, no interpolation`
- Token validation: `8 * 4 * 2048 * 1 = 65536` aggregate tokens/optimizer step.

## Current GDN2

- Effective records: `19868` from step `25` to `496700`; raw records `19868`, duplicates removed `0`.
- Token range: `1638400` to `32551731200`; current `32551731200` (`32.551731200B`), `21.701154133%` of 150B.
- Latest loss at step `496700` (`2026-07-28T13:45:55Z`): raw `2.5990000000`, smoothed `2.5529237500` with trailing MA window `80`.
- Last 100 optimizer-step mean: `2.5411250000` over `4` observations, bounds `[496601, 496700]`, cadence `25`, every-step coverage `False`.
- Last 1000 optimizer-step mean: `2.5553575000` over `40` observations, bounds `[495701, 496700]`, cadence `25`, every-step coverage `False`.

## Matched E97 Comparison

- Cutoff: step `496700`, tokens `32551731200` (`32.551731200B`).
- Alignment: `19868` exact common records, step range `[25, 496700]`, token range `[1638400, 32551731200]`, E97-only `0`, GDN2-only `0`.
- Raw cutoff losses: E97 `2.6758000000`, GDN2 `2.5990000000`, delta GDN2-E97 `-0.0768000000`.
- Smoothed cutoff losses: E97 `2.5871875000`, GDN2 `2.5529237500`, delta `-0.0342637500`.
- Last 100 mean delta: `-0.0476000000`; E97 `2.5887250000`, GDN2 `2.5411250000`.
- Last 1000 mean delta: `-0.0285300000`; E97 `2.5838875000`, GDN2 `2.5553575000`.
- Mean signed delta over all aligned observations: `-0.0328986914`; recent 1000-step interval: `-0.0285300000`.

## Publication

- GDN2 plot: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260728_20260728T134551Z/gdn2_mlp_diloco_loss_curve_20260728T134551Z.png`, SHA-256 `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1`, URL `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`.
- Comparison overlay: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260728_20260728T134551Z/gdn2_vs_e97_matched_tokens_20260728.png`, SHA-256 `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`, URL `http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260728.png`.
- GDN2 stable local/SSH/HTTP hashes: `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1` / `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1` / `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1`; HTTP `200` `image/png`.
- Overlay local/SSH/HTTP hashes: `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41` / `e687e1ef0055b362f481cc201c83a6aa6ebbff1536229a9ee6ec493a4db3df41`; HTTP `200` `image/png`.

## Protected Artifact Hashes

- E97_standalone: before refresh SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after GDN2 publish SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after overlay SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; unchanged from initial `true`.
- GDN2_standalone: before refresh SSH/HTTP `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13` / `fb82c1d967872639e372b2b248d21884e36de27f0c79db035644a66dbae23f13`; after GDN2 publish SSH/HTTP `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1` / `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1`; after overlay SSH/HTTP `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1` / `720457670ec20e281658656b2fb8f6ddfe35a6bc20c59630b054bfab69dcddb1`; unchanged after overlay vs after GDN2 `true`.
- prior_5p118B_comparison: before refresh SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; after GDN2 publish SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; after overlay SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; unchanged from initial `true`.
- prior_20260724_comparison: before refresh SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after GDN2 publish SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after overlay SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; unchanged from initial `true`.

## Throughput and ETA

- recent_1h: bounds `2026-07-28T12:45:57Z` step `493375` to `2026-07-28T13:45:55Z` step `496700`; samples `134`, steps/sec `0.924124514`, tokens/sec `60563.424`.
- recent_6h: bounds `2026-07-28T07:45:55Z` step `476800` to `2026-07-28T13:45:55Z` step `496700`; samples `797`, steps/sec `0.921296296`, tokens/sec `60378.074`.
- since_launch: bounds `2026-07-22T08:35:16Z` step `25` to `2026-07-28T13:45:55Z` step `496700`; samples `19868`, steps/sec `0.924839723`, tokens/sec `60610.296`.
- target_150b: target `150000000000`, percent `21.701154133%`, remaining `117448268800` tokens / `1792118.359375` steps, step at/above `2288819`, overshoot `41984` tokens, primary ETA `22d 12h 20m 14s` ending `2026-08-20T02:06:08Z`, range `2026-08-20T00:01:56Z` to `2026-08-20T00:26:55Z`.
- target_e97_parity: target `150793748480`, percent `21.586923548%`, remaining `118242017280` tokens / `1804230.000000` steps, step at/above `2300930`, overshoot `0` tokens, primary ETA `22d 15h 59m 20s` ending `2026-08-20T05:45:15Z`, range `2026-08-20T03:40:11Z` to `2026-08-20T04:05:21Z`.

Assumptions: no downtime, unchanged 8-GPU rate, and 65,536 aggregate tokens per optimizer step.

No training control, checkpoint write/modification, or S3 command was run.
