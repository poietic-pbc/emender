# Validate native data plane on two Frontier nodes

**Status: PASSED.** This file is the WG-declared Frontier delivery index for
`validate-native-dataplane-2n-synthetic-v1`. The complete methodology,
failure/correction ledger, commands, conformance analysis, and metrics are in
the [canonical full report](../validate-native-dataplane-2n-synthetic-v1.md)
and [machine-readable metrics](../validate-native-dataplane-2n-synthetic-v1-metrics.json).

## Final exact pair

| Item | Clean G2 | Changed fault G2 |
|---|---|---|
| Job | `5031461` | `5031553` |
| Source | `1c179e4ba014b2e54989a552fa5c99df010d7bbe` | same |
| Bundle | `411d7d92a5e23ea6838b370c0086265cd46d160878b4c2b1b8efc64e88293df1` | same |
| Nodes | `frontier[00122-00123]` | `frontier[00122-00123]` |
| Queue / runtime | 25 / 171 s | 9 / 113 s |
| Slurm | 2 nodes, `batch`, `debug`, `00:20:00`, `job_vni` | same |
| Payload | `1c179e4ba014-e97-g2-clean-20260719T053908Z` | `1c179e4ba014-e97-g2-fault-20260719T054304Z` |
| Gate | [full-layout-gate.json](native-dataplane/5031461/full-layout-gate.json) | [failure-injection-gate.json](native-dataplane/5031553/failure-injection-gate.json) |
| Result | passed | passed |

The pair used the retained manifest record
[native-artifacts-1c179e4.json](native-dataplane/native-artifacts-1c179e4.json),
SHA-256 `16daf9781253b1eff05afda7cfd72230adffbb3eea3b5713b3e00a2d907dcb6c`.
The synthetic gate binary SHA-256 is
`77b34e9899d3fdbbfcd44d79cc44eaf100cff12bc55c36868b92928910e95c42`.
There was no code change, build, or commit between the clean and changed fault
submissions.

## Acceptance results

- Both jobs attested two persistent native libfabric `FI_EP_RDM` endpoints,
  exact provider/fabric/domain `cxi`/`cxi`/`cxi0`, and no fallback provider.
- Exact layout was 5,506,770,496 float64 bytes / 688,346,312 elements / 83
  shards; eight lanes per node used weights 1,966,080 and 1,968,000 for global
  weight 3,934,080.
- Logical contribution and redistribution were each exactly 11,013,540,992
  bytes per generation. Each node's local-reduction input shape was
  22,027,081,984 bytes.
- The independent analytical/per-shard reference matched on both nodes for all
  generations. The full result digest was
  `e8d95487da2e901938bc5cdeb08fd5cceba9b537ec07eb63a167d27c0037c304`.
- Clean preflight rejected exactly two stale and two corrupt inputs total;
  measured clean generations had no rejection.
- Clean measured samples were 23.259562736, 23.219041387, and 23.150270638
  seconds. Median was **23.219041387 seconds**, logical goodput was
  **948,664,573.05 B/s**, and retained-Python speedup was **4.2620814925x**,
  passing 24.740361642 seconds and 4x.
- Clean CQ/route errors were 0/0. Wire TX/RX were each 44,323,138,304 bytes;
  retries were 1,895. In-flight/retained terminal bytes were zero.
- Clean owner admission was 14,372,851,648 bytes against 14,440,737,184;
  process RSS high-water was 14,695,383,040 against 20,215,943,136;
  in-flight and retained high-water were 134,219,008 and 268,436,736 bytes.
  Post-release RSS passed its floor.
- Fault rank 1 closed and reopened at endpoint epoch 2 with a new incarnation.
  Each rank performed exactly one reassignment and reported exactly 134,217,728
  logical replay bytes. One 67,108,864-byte replay shard crossed the wire and
  one stayed on its local native owner. Physical contribution was exactly
  5,573,879,360 bytes; redistribution was exactly 5,506,770,496 bytes.
- The changed fault result matched the independent reference, rejected exactly
  one old-owner-epoch input, reported `partial_commit=false` on both ranks,
  had zero CQ/route errors, released all retained transport state, and returned
  RSS to the required floor.
- Native CTest passed 8/8. The final installed native/resilient Python suite
  passed 121/121 in 156.52 seconds.

## Forbidden paths and conformance

No Python dense socket bytes, MPI collective, all-rank barrier, Lustre dense
hot path, trainer spool, disk replay, GPU training, central full-model broker,
or allocation larger than two nodes was used. Python carried bounded endpoint
metadata only; all dense bytes used bounded native memory and CXI.

The final gates embed the conformance checklist for **R01–R16** and
**NDP01–NDP17**, with `Q_min=2`, `T_min=3,934,080`, READY membership rather
than launched-rank admission, bounded non-Lustre hot paths, and atomic result
or no commit. This synthetic pass does not substitute for the downstream
real-model optimizer/checkpoint gate.

## Evidence integrity

- [Clean raw evidence and accounting](native-dataplane/5031461)
- [Fault raw evidence and accounting](native-dataplane/5031553)
- [Clean SHA-256 manifest](native-dataplane/5031461/SHA256SUMS)
- [Fault SHA-256 manifest](native-dataplane/5031553/SHA256SUMS)
- [Top-level report SHA-256 manifest](../validate-native-dataplane-2n-synthetic-v1-SHA256SUMS)

All manifests were verified after packaging. The final evidence commit is
`4263416`; a subsequent delivery-index commit contains documentation only.
