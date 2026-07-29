# GDN2 Ops Refresh 20260729 Retry Closeout

This retry resumed after the previous agent completed the GDN2 operations
refresh and committed it as `7f3045b2`.

No new live `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log` snapshot was
taken during this retry. The completed bounded snapshot remains:

- Summary JSON: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/refresh-gdn2-ops_20260729_20260729T144936Z/refresh_gdn2_ops_20260729_summary.json`
- Report: `docs/GDN2_OPS_SNAPSHOT_20260729.md`
- WG `RESULT:` log timestamp: `2026-07-29T14:50:20.154652093+00:00`

Retry validation performed at `2026-07-29T16:55Z`:

- Parsed the completed summary JSON with Python and asserted:
  - `worker_rank_count == 8`
  - `tokens_per_step == 65536`
  - GDN2 records are finite, deduplicated, and strictly monotonic
  - stable GDN2 local/SSH/HTTP hashes in the summary match
  - 20260729 overlay local/SSH/HTTP hashes in the summary match
  - both published images recorded HTTP `200` and `image/png`
  - no training control, checkpoint write/modification, or S3 command was run
- Re-fetched the public stable GDN2 plot:
  - URL: `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`
  - HTTP: `200`, `image/png`
  - SHA-256: `08f9e5d464d174f512ec3289a488a1cbe3cc68eee58224ec18cdaf05641acaf4`
- Re-fetched the public 20260729 E97/GDN2 overlay:
  - URL: `http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260729.png`
  - HTTP: `200`, `image/png`
  - SHA-256: `92c35c362347d9b9ea20b2c2932672798004b628332b837f35ca42f52da16a04`

Direct SSH hash re-check from this retry shell was unavailable because SSH auth
to `hypervolu.me` was rejected. The original completed refresh summary and WG
`RESULT:` entry contain matching SSH hashes for both published images and for
the protected historical artifacts.
