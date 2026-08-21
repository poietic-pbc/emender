# GDN2 Ops Snapshot - 2026-07-28

- Snapshot UTC: `2026-08-21T12:03:30Z`
- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Health: `stopped; no training process active`; torchrun PIDs `[]`, worker rank PIDs `[]`.
- Source policy: `rank-0/main stdout, finite complete records only, dedupe by optimizer step keeping latest timestamp/order, no interpolation`
- Token validation: `8 * 4 * 2048 * 1 = 65536` aggregate tokens/optimizer step.

## Current GDN2

- Effective records: `92944` from step `25` to `2323600`; raw records `92944`, duplicates removed `0`.
- Token range: `1638400` to `152279449600`; current `152279449600` (`152.279449600B`), `101.519633067%` of 150B.
- Latest loss at step `2323600` (`2026-08-20T20:00:32Z`): raw `2.4345000000`, smoothed `2.4007612500` with trailing MA window `80`.
- Last 100 optimizer-step mean: `2.3736750000` over `4` observations, bounds `[2323501, 2323600]`, cadence `25`, every-step coverage `False`.
- Last 1000 optimizer-step mean: `2.4100675000` over `40` observations, bounds `[2322601, 2323600]`, cadence `25`, every-step coverage `False`.

## Matched E97 Comparison

- Cutoff: step `2300925`, tokens `150793420800` (`150.793420800B`).
- Alignment: `92037` exact common records, step range `[25, 2300925]`, token range `[1638400, 150793420800]`, E97-only `0`, GDN2-only `0`.
- Raw cutoff losses: E97 `2.4770000000`, GDN2 `2.4324000000`, delta GDN2-E97 `-0.0446000000`.
- Smoothed cutoff losses: E97 `2.4370450000`, GDN2 `2.4267050000`, delta `-0.0103400000`.
- Last 100 mean delta: `0.0139500000`; E97 `2.4283500000`, GDN2 `2.4423000000`.
- Last 1000 mean delta: `-0.0085025000`; E97 `2.4358200000`, GDN2 `2.4273175000`.
- Mean signed delta over all aligned observations: `-0.0316291133`; recent 1000-step interval: `-0.0085025000`.

## Publication

- GDN2 plot: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260728_20260821T120330Z/gdn2_mlp_diloco_loss_curve_20260821T120330Z.png`, SHA-256 `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84`, URL `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`.
- Comparison overlay: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260728_20260821T120330Z/gdn2_vs_e97_matched_tokens_20260728.png`, SHA-256 `ca0582aad0f5176197080bd823889d13987be8b2e16df862c45458fb0036db1b`, URL `http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260728_20260821T120347Z.png`.
- GDN2 stable local/SSH/HTTP hashes: `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84` / `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84` / `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84`; HTTP `200` `image/png`.
- Overlay local/SSH/HTTP hashes: `ca0582aad0f5176197080bd823889d13987be8b2e16df862c45458fb0036db1b` / `ca0582aad0f5176197080bd823889d13987be8b2e16df862c45458fb0036db1b` / `ca0582aad0f5176197080bd823889d13987be8b2e16df862c45458fb0036db1b`; HTTP `200` `image/png`.

## Protected Artifact Hashes

- E97_standalone: before refresh SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after GDN2 publish SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; after overlay SSH/HTTP `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc` / `4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc`; unchanged after overlay vs after GDN2 `True`.
- GDN2_standalone: before refresh SSH/HTTP `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84` / `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84`; after GDN2 publish SSH/HTTP `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84` / `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84`; after overlay SSH/HTTP `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84` / `6124c4b32bba468f4f7ea9898a582c692cf8227861dce6abe98def44e09cbe84`; unchanged after overlay vs after GDN2 `True`.
- prior_5p118B_comparison: before refresh SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; after GDN2 publish SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; after overlay SSH/HTTP `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1` / `54bfcf831ad43c0d20423617777de377e1cb81c716035816cba2850893610bd1`; unchanged after overlay vs after GDN2 `True`.
- prior_20260724_comparison: before refresh SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after GDN2 publish SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; after overlay SSH/HTTP `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7` / `61afa4ee38a05d9a4b4fc537b4f5998e017cf366f110cb7247eb1e99f4bb93b7`; unchanged after overlay vs after GDN2 `True`.

## Throughput and ETA

- recent_1h: bounds `2026-08-20T19:00:56Z` step `2320300` to `2026-08-20T20:00:32Z` step `2323600`; samples `133`, steps/sec `0.922818792`, tokens/sec `60477.852`.
- recent_6h: bounds `2026-08-20T14:00:56Z` step `2303675` to `2026-08-20T20:00:32Z` step `2323600`; samples `798`, steps/sec `0.923479792`, tokens/sec `60521.172`.
- since_launch: bounds `2026-07-22T08:35:16Z` step `25` to `2026-08-20T20:00:32Z` step `2323600`; samples `92944`, steps/sec `0.912380886`, tokens/sec `59793.794`.
- target_150b: target `150000000000`, percent `101.519633067%`, remaining `0` tokens / `0.000000` steps, step at/above `2288819`, overshoot `41984` tokens, primary ETA `0d 0h 0m 0s` ending `2026-08-20T20:00:32Z`, range `2026-08-20T20:00:32Z` to `2026-08-20T20:00:32Z`.
- target_e97_parity: target `150793748480`, percent `100.985253789%`, remaining `0` tokens / `0.000000` steps, step at/above `2300930`, overshoot `0` tokens, primary ETA `0d 0h 0m 0s` ending `2026-08-20T20:00:32Z`, range `2026-08-20T20:00:32Z` to `2026-08-20T20:00:32Z`.

Assumptions: no downtime, unchanged 8-GPU rate, and 65,536 aggregate tokens per optimizer step.

No training control, checkpoint write/modification, or S3 command was run.
