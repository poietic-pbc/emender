# Native resilient pool v1 two-node startup acceptance

Date: 2026-07-19

WG task: `validate-native-pool-v1-2n-startup`

Live source commit: `85cf5a09eae00f19612e30b4b7001cab0e1a541f`

Live Slurm job: `5033125`

Synthetic full-layout prerequisite: G2 job `5033120`

## Result

**Passed.** One real E97 generation ran on exactly two Frontier nodes in the
`debug` QoS, with a 20-minute allocation, 8 GPU trainers per node, no failure
injection, no restart, and `DILOCO_DATAPLANE=native-cxi`. Slurm job `5033125`
completed `0:0` in 6 minutes 27 seconds on
`frontier[00526,00528]`.

Both model-free managers reached native-service READY in less than 15 seconds
from their respective role starts. All 16 trainers completed the real K40
interval in 135.13–137.01 seconds from trainer start. Native local reduction,
route readiness, owner exchange, redistribution, and every independent trainer
apply remained inside the unmodified stage bounds. The fenced generation-1
manifest became durable 352 seconds after the Slurm allocation started, and
both managers completed native publication by 381.94 seconds. All 16 trainers
and both managers then exited zero; the two persistent native services were
stopped only after allocation completion.

The machine-readable summary is
[`validate-native-pool-v1-2n-startup-metrics.json`](validate-native-pool-v1-2n-startup-metrics.json).
The compact repository evidence is under
[`frontier/evidence/job-5033125`](frontier/evidence/job-5033125); the immutable
7.90-GB checkpoint is deliberately not duplicated into git and remains at the
absolute run path recorded below.

## Authorized launch and provenance

The immutable submission recipe is
[`validate-native-pool-v1-2n-startup-submit-20260719T192710Z.sh`](frontier/validate-native-pool-v1-2n-startup-submit-20260719T192710Z.sh).
It hard-gated all of the following before `sbatch`:

- authoritative repository branch `main`, with local `HEAD` and `origin/main`
  both equal to `85cf5a09eae00f19612e30b4b7001cab0e1a541f`;
- no tracked source or index changes;
- the approved Python 3.12 Frontier environment;
- pinned step-1525000 seed SHA-256
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`;
- pinned tokenizer SHA-256
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`;
- clean attested native bundle
  `f2ac884eb52623896ca40777f2debc828751f50695ce812fa6f2bf239409282c`;
- passing exact-source, full-E97-layout two-node G2 gate `5033120`;
- exactly two nodes, eight GPUs and eight trainers per node, one generation,
  K40, quorum 2, zero permitted restarts, empty injection selectors,
  `native-cxi`, exact `cxi`, and `job_vni`.

The real run identity was
`validate-native-pool-v1-2n-startup-20260719T192710Z-85cf5a0`, with payload
`85cf5a0-20260719T192710Z-native-pool-v1-startup-2n20m-k40` and allocation
fence 1. The source tree was clean when the canonical RelWithDebInfo build was
recorded; tests and XPMEM support were enabled. Its exact artifact digests are:

| Artifact | SHA-256 |
|---|---|
| `ndp_cxi_service` | `370aae986599df5a51f8ed8e1ff5b5a2eb6528b227895344226afc3e4fde9686` |
| `libemender_ndp.so.1` | `9bc1e1077fcb30a3c0d46ebfe8d04fc3a4aa6e83f81c1657266328883b2d7c2c` |
| `libemender_ndp_transport.so.1` | `2fc10d9bb46c3527949d2f7d78778ec63046eac5f5f3ef64636a1b459287cba5` |
| `ndp_frontier_2n_gate` | `ba96d446679145fa13564678086886d209753da98332cdfba1c13aab4f4ba174` |

## Acceptance measurements

### Startup and K40

The supervisor's append-only event stream contains 20 role starts: two native
services, two managers, and sixteen trainers. Every start has `restart=0`.

| Measurement | Observed | Hard bound | Result |
|---|---:|---:|---|
| node-0 manager READY from manager start | 14.784 s | 180 s | pass |
| node-1 manager READY from manager start | 14.758 s | 180 s | pass |
| fastest trainer start → K40 `streaming_delta` | 135.133 s | 420 s | pass |
| slowest trainer start → K40 `streaming_delta` | 137.010 s | 420 s | pass |
| trainers reaching K40 | 16/16 | 16/16 | pass |

The K40 measurement uses each trainer's supervisor `started` time and its
`streaming_delta` progress heartbeat. It therefore includes real model forward,
backward, loss synchronization, optimizer update, and bounded delta streaming,
not a launcher-only proxy.

### Native stages

The live stage files carry their original hard bound and `within_slo=true`.
No deadline was raised in response to an observed run.

| Native stage | node 0 | node 1 | Original hard bound | Dense data |
|---|---:|---:|---:|---|
| local f32→weighted-f64 reduction, 8 contributions | 17.082 s | 17.164 s | 180 s | 11,013,540,992-byte node numerator |
| pairwise route readiness | 0.002 s | 0.012 s | 15 s | metadata only |
| CXI owner exchange | 48.059 s | 47.069 s | 90 s | 11,013,540,992 bytes sent and received per node |
| two-owner f64→final-f32 redistribution | 18.625 s | 18.598 s | 60 s | 5,506,770,496-byte shared result |
| slowest ordered trainer apply | 36.233 s | 45.866 s | 60 s | independent read-only result view |

All 16 `native_direct_memfd` records report `producer_direct=true`,
`trainer_spool_bytes=0`, and `python_dense_socket_bytes=0`. All 16
`native_trainer_apply` records exist, have the expected lane ranks 0–7 per
node, apply the same 5,506,770,496-byte result, and remain below 60 seconds.

During payload exchange, both manager endpoints reported exact production
`cxi/cxi0/FI_EP_RDM`, 11,013,540,992 useful bytes in each direction, zero
replay bytes, zero route errors, zero CQ errors, zero retained bytes after
release, and the same final result root. The large native `retries` counters
are bounded nonblocking-provider backpressure polls; they are not payload
replay and not a generation retry. The generation identity remained attempt 0,
the supervisor observed no restart, and no unchanged generation was retried.

For transparency, closing each native manager session canceled two outstanding
receive completions after result release. Those records have provider errno
`ECANCELED`, owner state `DRAINING`, zero useful bytes, and occur only after the
payload-phase counters had reached zero CQ/route errors. Both managers and the
job exited zero. They are normal collective-free endpoint teardown, not stage
failure or timeout.

### Atomic fenced commit

Generation 1 was atomically committed under fence 1. The immutable manifest was
written 352 seconds after the Slurm allocation began, below the 720-second
first-commit bound and the task's 12-minute limit. The authoritative SQLite
transaction contains exactly the linked `commit`, `checkpoint`, and `latest`
rows under fence 1; their checkpoint and manifest digests agree with the
filesystem handoff. Both managers subsequently committed their native sessions
after observing all eight local applied markers and published generation 1 by
381.94 seconds from allocation start.

The committed state includes:

- model and nonempty inner optimizer at step 1,525,040;
- Python-owned outer update state `{algorithm: weighted-mean, eta_outer: 1}`
  and digest
  `79661b97a27fce6f9057a16642b0cabdc6f6f7ab7782315e12da17a3a136c712`;
- accepted-token clock 5,245,440;
- global membership `node-0,node-1`, frozen node incarnations, local rank
  membership, and the node-0 trainer-0 checkpoint-writer identity;
- run, source, payload, coordinator epoch, generation, attempt, and fence;
- native provider, build, config, source, and artifact digests;
- native result root
  `ec3c1c31e618ce4b8322e6ff9a98d45c7782758d87c555a17ceffee905497fc4`,
  layout digest
  `e3fb15da10a151dbd33d6f66a3a2f8723be69bbaf7b34a6b3652bee0f5a352e2`,
  base digest
  `1ac9d84d60b8618b4a0318da49d79260469f7052e637c10198a954ded91f43b9`,
  exact global weight, and result size.

Native result `attempt=2` denotes the specified second arithmetic stage that
combines the two preweighted node numerators. It is not a repeated generation;
the fenced generation identity itself remains attempt 0.

### Independent reload

After Slurm completion, a new approved-Python process independently:

1. streamed and recomputed the 7,899,873,331-byte checkpoint SHA-256;
2. recomputed the immutable manifest SHA-256;
3. loaded the checkpoint with `torch.load(map_location="cpu", mmap=True,
   weights_only=True)`;
4. audited the complete model, optimizer, outer state, clock, membership,
   identities, fence, and native runtime digests;
5. independently read the SQLite publication transaction and compared its
   three rows to the handoff.

The recomputed checkpoint digest is
`d9f23dcd62e9e6464feca07af6a8c638f82e69b52926d7061a156133f303d07e`;
the recomputed manifest digest is
`114eb3229ae1b43d91a0226b9df45438a4348f582522f2227247d6e60a83d575`.
Both match the manifest, compatibility latest pointer, and authoritative
SQLite rows. Reload found 146 bfloat16 model tensors with 1,376,692,624
elements, 145 nonempty optimizer state entries, one optimizer parameter group,
and 268 bfloat16 optimizer tensors with 2,573,173,392 elements. The global
checkpoint membership is represented by `accepted_peers=[node-0,node-1]`; its
separate `membership=[0..7]` field records the local trainer layout. The
manifest's global membership matches `accepted_peers`, and the native runtime
and outer-state digests match exactly.

The detailed proof is
[`independent-reload-proof.json`](frontier/evidence/job-5033125/independent-reload-proof.json).
The checkpoint remains at:

`/lustre/orion/bif148/proj-shared/emender/runs/validate-native-pool-v1-2n-startup-20260719T192710Z-85cf5a0/checkpoints/generation-00000001-fence-00000001.pt`

## Prohibited-path audit

- **No Python TCP dense payload:** every live manager/trainer stage reports
  `python_dense_socket_bytes=0`; the immutable launcher selects only
  `native-cxi`. Python carries metadata and publication policy.
- **No global MPI collective:** the native full-layout G2 reports
  `mpi_collectives=0` and `all_rank_barriers=0`; the live launcher uses
  independent node-local role steps and point-to-point CXI, with no fixed-world
  MPI path.
- **No Lustre dense hot path:** all trainer deltas, node numerators, owner
  transfers, redistributed results, and trainer views use service-owned memfds
  and CXI. Node-local retention explicitly excludes `*.data` and `*.pt`.
  Shared storage receives only small control/evidence plus the one immutable
  post-update checkpoint/handoff. Reading the pinned seed is initialization,
  not the generation's dense aggregation hot path.
- **No central dense broker:** each node owns its node numerator and exchanges
  directly with its frozen peer. The SQLite/control endpoint carries leases,
  READY membership, receipts, and fenced publication only. G2 records
  `central_full_model_broker=false`.
- **No unchanged retry:** `RESILIENT_E97_MAX_RESTARTS=0`, all supervisor restart
  fields are zero, generation identity attempt is zero, payload replay bytes
  are zero, and exactly one generation was committed. Provider backpressure
  polling is separately measured and does not change generation identity.

The exact-source G2 prerequisite independently moved the full E97 layout on two
production CXI endpoints, matched the analytical f64 reference for all three
timed generations, recorded zero MPI collectives/Python dense socket/disk
replay/trainer spool/CQ/route errors, and completed at a 19.492-second median—
5.077× the pinned Python baseline, above the required 4× threshold.

## Focused fixes discovered by live acceptance

Every listed fix was committed and pushed to authoritative `main` before final
job `5033125` was built and launched:

| Main commit | Focused correction |
|---|---|
| `0e81e869` | Digest real E97 bfloat16 state bytes correctly. |
| `ded89b9b` | Parallelize sealed submission validation within its finite bound. |
| `ddc9ec70` | Parallelize native local reduction while preserving rank order. |
| `538de66b` | Exclude lease-only metadata from the frozen owner snapshot. |
| `be35e143` | Overlap bounded native owner result imports. |
| `98ff1f9d` | Remove redundant sealed result scans. |
| `c3e6f631` | Fuse bounded native final-apply passes. |
| `6b0224eb` | Parallelize finite native admission work. |
| `d64214e4` | Fence pairwise native route readiness before payload exchange. |
| `1fc8c618` | Use the native OpenSSL SHA path for dense digests. |
| `2ab182cc` | Refresh the authoritative native result view after attempt-2 redistribution. |
| `2a306d42` | Let terminal followers reuse the one fenced checkpoint instead of serializing 15 redundant 7.9-GB files. |
| `c7b30c83` | Order per-node trainer result-view apply lanes and record their measured durations. |
| `29a1fd32` | Do not re-enter READY after publishing the terminal generation. |
| `85cf5a09` | Do not contact the already-closing pool server to drain after terminal publication. |

The fixes retain the normative arithmetic and deadlines. They remove redundant
work and terminal control-plane races; they do not widen any stage timeout.

## Validation

- [x] Managers READY within 3 minutes: 14.784 s maximum.
- [x] All 16 trainers complete K40 within 7 minutes of trainer start: 137.010 s
  maximum.
- [x] Native local reduction, route readiness, owner exchange, redistribution,
  and all 16 trainer applies meet their original measured stage bounds.
- [x] Generation 1 is atomically committed under active fence 1 within 12
  minutes with model, optimizer, outer state, token clock, membership,
  identities, and native digests.
- [x] A separate approved-Python process reloads the checkpoint and matches the
  checkpoint, manifest, latest pointer, outer-state digest, native-runtime
  digests, and SQLite publication rows.
- [x] No Python TCP dense payload, global MPI collective, Lustre dense hot path,
  central dense broker, or unchanged generation retry occurs.
- [x] Canonical native build passes 10/10 CTests.
- [x] Exact-source clean G2 job `5033120` passes on two production CXI nodes.
- [x] The selected native/resilient Python pre-submit passes 162/162 tests in
  279.07 seconds.
- [x] Complete compact evidence, the immutable submission recipe, focused
  fixes, metrics, report, and checksums are committed and pushed to
  authoritative `main`.

## Design-authority conformance checklist

This validation was checked against the normative
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../docs/RESILIENT_DILOCO_COMPUTE_POOL.md)
and its traceability matrix
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../docs/RESILIENT_DILOCO_GAP_MATRIX.md).
Failure injection is intentionally outside this startup job and remains the
scope of downstream `validate-native-pool-v1-2n-failures`; this report does not
claim that downstream run.

### Compute-pool requirements R01–R16

- **R01–R04 (lease, lifecycle, live membership, identity):** allocation fence 1
  was acquired before role/model launch; READY membership contains the two live
  node incarnations; the frozen contribution record binds run, fence,
  generation, attempt, worker, incarnation, and sequence. Terminal completion
  drains locally without rejoining or contacting a closed server.
- **R05–R06 (exact weighted aggregation and bounded quorum):** eight ranked f32
  trainer contributions per node reduce into preweighted f64 numerators; the
  two-node frozen set meets `q_min=2`, advances the exact 5,245,440-token clock,
  and closes under bounded stage deadlines.
- **R07–R08 (atomic commit and bounded direct owners):** the current lease gates
  one atomic commit/checkpoint/latest bundle; fixed-size CXI frames, explicit
  route readiness, byte admission, checksums, backpressure, zero payload replay,
  prompt release, and direct owners avoid a dense broker.
- **R09–R10 (model ownership and non-Lustre hot path):** trainers alone own the
  model/optimizer; managers own metadata/native descriptors. Producer-direct
  memfds and CXI carry the dense hot path, with zero trainer spool.
- **R11–R12 (incarnation/recovery and outer ownership):** the no-failure startup
  freezes current incarnations and emits an authoritative fresh-allocation
  handoff. The outer weighted-mean state, migration provenance, accepted-token
  clock, and digests reload independently. Loss/rejoin timing is reserved for
  the downstream injected-failure gate.
- **R13–R14 (adapter isolation and measured deadlines):** the backend-neutral
  pool protocol is exercised through the Frontier adapter, and retained live
  telemetry proves READY, K40, exchange, apply, release, and commit bounds.
- **R15–R16 (numerics/accounting and two-node order):** exact-source G2 matches
  the independent analytical reference, this real two-node job preserves the
  frozen global weight/result root, and no 4+ launch preceded acceptance.

### Native data-plane requirements NDP01–NDP17

- **NDP01–NDP04 (hard boundary and persistent service):** production selects no
  Python dense TCP or MPI collective; each node runs one persistent C++17
  `FI_EP_RDM/cxi` service; every trainer uses producer-direct service memfd and
  records zero extra dense handoff writes.
- **NDP05–NDP07 (exact arithmetic and fenced routes):** rank-sequenced f32
  deltas become weighted f64 node numerators, are exchanged under fixed fenced
  identities over current leased endpoints, then are divided/projected once
  into the shared f32 result.
- **NDP08–NDP11 (admission, credits, integrity, bounded replay):** the run uses
  the fixed 64-GiB service ledger and 64-MiB chunks; authenticated credits are
  distinct from CQ ownership; header/payload/result identities validate; live
  replay bytes are zero and the generation is never retried.
- **NDP12–NDP13 (owner-direct redistribution and deadlines):** both native
  owners import the peer f64 numerator directly into bounded memfd state,
  redistribute one result, expose it to eight local trainers, and finish every
  recorded stage inside the absolute original bound.
- **NDP14–NDP15 (stable ABI and fenced handoff):** attested
  `libemender_ndp.so.1` and metadata-only authenticated control are used; Python
  policy atomically publishes the current-fence handoff; all independent apply
  markers arrive before collective-free terminal close.
- **NDP16 (telemetry):** retained evidence binds provider, provider/build/config/
  source/artifact digests, bytes, throughput, backpressure, release, result
  identity, membership, token weight, and explicit zero prohibited-path fields.
- **NDP17 (ordered gate):** exact-source clean full-layout G2 `5033120` passed
  before real job `5033125`; both used exactly two debug-QoS nodes. The next
  authorized rung is the two-node injected-failure task, not 4+ scale.

## Evidence inventory

Repository evidence includes:

- immutable Slurm accounting and top-level stdout/stderr;
- exact build manifest and full-layout G2 JSON/stdout/stderr;
- allocation lease, runtime identity, and launch attestation;
- complete supervisor event stream;
- SQLite control store and fenced handoff JSON (but not the 7.90-GB checkpoint);
- all per-role logs and native service transport summaries;
- retained node-local control, supervision, manager/native/stage telemetry, and
  all sixteen trainer apply records;
- independent reload proof and recursive evidence checksums.

`reports/validate-native-pool-v1-2n-startup-SHA256SUMS` binds this report and
its machine-readable metrics. The evidence directory's own `SHA256SUMS` binds
every compact captured artifact beneath job `5033125`.
