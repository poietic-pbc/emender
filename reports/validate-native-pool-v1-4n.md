# Native resilient pool v1 four-node validation

Date: 2026-07-19/20 UTC

WG task: `validate-native-pool-v1-4n`

Final live source commit: `d57c7dbea7038e44afdf93ba9fab875474a7ac45`

Peer-loss/rejoin job: `5034117`

Final same-code continuation job: `5034180`

Exact-source full-layout prerequisite: G2 job `5034114`

## Result

**Passed.** The first post-two-node native rung ran on exactly four Frontier
nodes with the authoritative E97 seed, workload, K40 cadence, fenced protocol,
and production `native-cxi`/`cxi`/`FI_EP_RDM` data plane. The accepted scale
policy was explicit: `Q_min=4`, `T_min=3,934,080`, four active owners, eight
real trainers per node, and no launched-rank shortcut. The normalized workload
SHA-256 is
`0493ba1ebc6f28c3244b6845bffb2de722639ee8e9df7d6ffa1918742e0691f4`.

Job `5034117` committed generations 9 and 10 under fence 7. Immediately after
all four managers published generation 9, the supervisor injected one bounded
loss of the node-3 manager, drained its node-local native service, started a
fresh service and replacement manager, delayed node 3 for the prescribed 30
seconds, and admitted the replacement only with restart/incarnation 1. That
replacement joined generation 10's native owner exchange and reached terminal
`published` at generation 10. All 32 trainers applied generation 10 exactly
once. The allocation finished `COMPLETED 0:0` in 13:59 of its 30-minute debug
limit.

The final continuation loaded generation 10 under strictly newer fence 9 and
committed generation 11 with the identical `d57c7dbe` source and native bundle.
Together generations 9, 10, and 11 are three consecutive atomic commits from
the same validated code. Job 5034180 finished `COMPLETED 0:0` in 7:28 of its
30-minute debug limit on exactly four nodes.

The durable store ends with one commit and one checkpoint publication for each
generation, one authoritative latest pointer, `PRAGMA integrity_check=ok`, and
no active lease. There was no partial publication, duplicate generation,
duplicate trainer application, Python dense transport, trainer spool, disk
replay, MPI collective, all-rank barrier, or central model broker.

This is a **validation-only rung**. The production policy remains
`defined-no-submission-authority`, its `production_qualified_rungs` array
remains empty, and this report grants no normal-QoS or production authority.
No 8-, 32-, 64-, or larger-node job was submitted.

The machine-readable result is
[`validate-native-pool-v1-4n-metrics.json`](validate-native-pool-v1-4n-metrics.json).
Compact live evidence is retained under
[`frontier/evidence/job-5034117`](frontier/evidence/job-5034117) and
[`frontier/evidence/job-5034180`](frontier/evidence/job-5034180). The
exact-source G2 prerequisite is under
[`frontier/native-dataplane/5034114`](frontier/native-dataplane/5034114).
The 7.9 GB checkpoint payloads stay at their immutable recorded run paths;
Git contains their manifests and independent full-file digest/reload results.

## Authorities and admission decision

The authoritative predecessors were accepted before submission:

- `validate-native-pool-v1-2n-restart`, final commit `39de56fb`, proves a
  fresh two-node allocation can acquire a newer fence, reload immutable state,
  reconstruct native endpoints, and commit twice;
- `validate-native-pool-v1-2n-failures`, job `5033384`, is the source of this
  exact run and payload identity and proves three two-node K40 commits with
  trainer and manager loss/rejoin;
- `define-native-pool-v1-production-policy`, final commit `f9b44727`, defines
  ordered validation admission and explicitly records no production-qualified
  rung;
- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1 and
  `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md` version 1 remain the design
  authorities; the gap matrix is traceability, not a competing design.

The four-node rung was allowed only after a clean exact-source G2 rerun. G2 job
5034114 completed on two nodes with three exact timed generations, median
23.648376788 seconds, 931,441,603 logical bytes/s, and 4.1847x speedup over the
independent Python baseline. It reported exact reference equality, zero route
or CQ errors during operation, zero post-release transport bytes, and bounded
resident/transport high-water.

The immutable launch recipes check the clean pushed snapshot, exact seed and
tokenizer digests, current handoff digest, current fenced latest row, absent
lease, exact G2 attestation, no overlapping allocation, native backend, and
exact `cxi` provider before calling `sbatch`. The launcher itself admits only
two or four nodes and rejects 8+.

## Payload preservation and explicit scale controls

[`validate-native-pool-v1-4n-normalized-payload.json`](frontier/validate-native-pool-v1-4n-normalized-payload.json)
retains both the normalized workload and the scale/queue controls. Canonical
`jq -cS .normalized_workload` bytes hash to
`0493ba1ebc6f28c3244b6845bffb2de722639ee8e9df7d6ffa1918742e0691f4`.

The normalized payload preserves the accepted two-node values:

- exact run/payload lineage
  `npv1f-progress-20260719T223459Z-f56e27a` /
  `f56e27a-20260719T223459Z-native-pool-v1-failures-progress-2n30m-k40`;
- seed/source SHA-256
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`,
  config SHA-256
  `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c`,
  tokenizer SHA-256
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`,
  and the same CommaPile path;
- eight trainers per node, ScheduleFree inner optimizer, weighted-mean outer
  update with `eta_outer=1`, 40 local steps, and staleness `tau=0`;
- exact 5,506,770,496-byte E97 layout, 83 shards, 64 MiB chunks, 64 GiB shared
  admission, four TX/four RX slots, and checkpoint-every-generation cadence;
- bundle
  `7b4c77696011cfc2b68fce541a34f9a1072ef767e13cce94ed2a4e2befc57d04`,
  `native-cxi`, exact `cxi`, `FI_EP_RDM`, `job_vni`, `kdreg2`, and ATS disabled;
- the validated READY/K40/route/exchange/redistribution/apply/commit deadlines,
  bounded restart/reassignment rules, and zero disk fallback.

The recorded scale controls changed capacity from two to four nodes, explicit
quorum/owners from 2/2 to 4/4, and retained `T_min=3,934,080`. Allocation ID,
incarnation, fence, resume generation, nodelist, and node-local roots are
dynamic identity, while partition/QoS/walltime are queue records. The required
failure event maps the manager target to node 3 so the newly added cohort is
the peer that leaves and rejoins; it does not alter model, optimizer, layout,
transport, numerical, or K40 payload bytes. The source commit changed only to
the exact reviewed four-node implementation, while the attested native bundle
and normalized training payload stayed identical.

## Frontier allocation accounting

All compute attempts requested exactly four nodes, `batch`, debug QoS, eight
GPUs per node, `job_vni`, and a 30-minute cap. No larger request occurred.

| Job | Result | Durable result | Purpose |
|---|---|---|---|
| `5033611` | cancelled pending; no compute | none | First queue attempt, cancelled without a node/fence |
| `5033672` | failed closed | no commit | Exposed that three sequential peer waves exceeded the 90-second owner-exchange bound |
| `5033788` | failed after commit | generation 6/fence 4 | Proved concurrent four-owner exchange; exposed stale static trainer handoff |
| `5033953` | failed after commit | generation 7/fence 5 | Proved newer authoritative trainer reload; exposed manager lifecycle gap |
| `5034067` | failed after commit | generation 8/fence 6 | Fresh service restart occurred; replacement inherited admission-token EOF and failed closed |
| `5034117` | `COMPLETED 0:0`, 13:59 | generations 9 and 10/fence 7 | Final bounded peer loss/rejoin and two commits |
| `5034173` | failed closed before model load, 1:01 | no commit; fence only advanced to 8 | Overlong node-local socket path; zero managers/trainers started |
| `5034180` | `COMPLETED 0:0`, 7:28 | generation 11/fence 9 | Same-code third commit and newer-fence reload |

Every failed compute attempt either published one complete immutable generation
or none. The fenced latest pointer advanced only after the checkpoint/handoff
was durable. Thus diagnostic runs contributed useful immutable generations
without ever exposing a partial result. The accepted evidence does not hide
the attempts: recipes, accounting, shared supervisor events, and fail-closed
audits record them.

## Three consecutive same-code atomic generations

The accepted final-code chain is:

| Generation | Fence | Step | Accepted tokens | Manifest SHA-256 | Checkpoint SHA-256 | Source |
|---:|---:|---:|---:|---|---|---|
| 9 | 7 | 1,525,360 | 68,190,720 | `3c9f4735c703f24fe378225b3b158a5708a68407f919d98296d29346bf377ba1` | `c619f9044bebf99384789209b30e833934359fbec74d1d648f8ff64a607d6983` | `d57c7dbe` |
| 10 | 7 | 1,525,400 | 78,681,600 | `dc53a24b2b723463e8a5fab0640285d10543db93852989461ae5de339ecb88f7` | `89058e658b5a21ce1aa87fdbc93d83d38928cf4976f4a4c76f75d888ea81cb37` | `d57c7dbe` |
| 11 | 9 | 1,525,440 | 89,172,480 | `c1896f28f805ca9cf519f470d9e5f1b5132eae33d6decec232abac43b0f16a1e` | `5084db3032037ab511aed09296c77f6d120eed5b7e15b758f9f4dcbf09bb158c` | `d57c7dbe` |

Each increment is exactly 10,490,880 accepted tokens from four frozen manager
contributions. Each generation has one SQLite `commit` row, one `checkpoint`
row, one immutable handoff, a monotonic latest update, four manager published
receipts, and 32 unique `(node,rank,generation)` apply receipts. The SQLite
primary key `(run_id,kind,name)` prevents duplicate publications, while native
receipt identities prevent duplicate lane application.

Earlier four-node generations 6, 7, and 8 also committed atomically under
fences 4, 5, and 6 with token clocks 36,718,080, 47,208,960, and 57,699,840.
They are retained as fail-closed diagnostic evidence; the table above is the
strict three-generation final-code acceptance chain.

## Bounded peer loss and fresh rejoin

Job 5034117's shared supervisor evidence gives this exact sequence:

1. Four restart-0 managers and 32 trainers completed input generation 8.
   SQLite atomically published generation 9/fence 7 at 68,190,720 tokens; all
   four managers reached `published` and all 32 trainers reached `applied`.
2. At Unix time `1784518656.1723623`, the generation-gated injector observed
   node-3 manager generation 9, stage `published`. It therefore could not cut
   through the generation-9 atomic commit.
3. The old manager exited zero for `injected_generation_gate`. Its same-node
   service drained zero for `manager_rejoin:injected_generation_gate`.
4. At `1784518660.1793382` the supervisor recorded
   `native_service_rejoin`, manager restart 1 and service restart 1. It started
   fresh service PID 891750 and replacement manager PID 891758. The new
   admission-token FD was rewound before spawn, and the service created a new
   CXI endpoint.
5. The replacement manager progressed through runtime import,
   `native_service_ready` at generation 9, the exact 30-second delayed READY,
   training wait, freeze, owner transport, redistribution, and finally
   `published` generation 10 with restart 1.
6. The other three managers published generation 10 with restart 0. Exactly
   32 trainers applied generation 10 once, all roles completed zero, and all
   four services drained zero for allocation completion.

No manager or trainer had an unplanned eviction. The only two non-terminal
evictions were the prescribed old node-3 manager and its service. The bounded
event introduced no duplicate, partial, skipped, or mixed-fence publication.

## Native throughput, high-water, and release evidence

All observed stage records were inside their hard limits:

| Stage | Samples | Observed range (s) | Hard limit (s) |
|---|---:|---:|---:|
| local f32-to-f64 reduction | 8 | 17.0076–17.9108 | 180 |
| pairwise route readiness | 24 | 0.00061–0.20941 | 15 |
| three-peer owner exchange | 8 | 81.5204–84.0288 | 90 |
| four-owner redistribution | 8 | 19.9239–20.7032 | 60 |
| independent trainer apply | 64 | 3.1742–13.0649 | 60 |

Each owner-exchange record sent and received exactly 33,040,622,976 useful
bytes to/from three peers with concurrency limit 3. Its bounded RX queue used
one to four 64 MiB frames depending on arrival order. `python_dense_socket_bytes`
was zero in every manager stage; every trainer reported direct service memfd,
zero spool bytes, and a unique result lane.

The three uninterrupted managers each shut down after two generations with
66,081,245,952 useful bytes in each direction, 268,436,736-byte in-flight and
retained high-water, 132,163,759,104 released bytes, and zero bytes retained or
in flight. Their terminal measured transport rates were 173.715, 178.317, and
178.808 MB/s. The replacement node-3 manager measured one post-rejoin exchange:
33,040,622,976 useful bytes each way, 268,436,736-byte highs,
66,081,879,552 released bytes, zero residual bytes, and 218.770 MB/s.

All four terminal records have `replay_bytes=0` and `route_errors=0`. Their two
CQ status increments are the expected collective-free drain cancellations
(`status=-12`, provider errno 125, owner state `DRAINING`), not operational
transfer errors. The exact-source G2 separately reports zero CQ errors and zero
post-release bytes at its measurement boundary.

## Checkpoint reload and fence progression

Job 5034117 acquired fence 7 only after fence-6 generation 8 was authoritative.
All trainers independently verified the 7,899,873,779-byte generation-8
checkpoint SHA-256, loaded it with `mmap=True`, and ran K40. A post-job audit
streamed the full file again and reproduced
`664a5d3b4e592656edd5ce1f26ef69471fbb4634c50f08808e1245fcbbcdaa20`.
The loaded checkpoint has generation 8, step 1,525,320, 57,699,840 tokens, 146
model tensors, 145 optimizer state entries, outer weighted mean state, and
model digest
`85314e43ac3e0a1ae7652048e50efa5a1eabdae186de9c777efe4848863e3e3b`.

The same independent process reloaded generation 10/fence 7, reproduced the
7,899,873,907-byte checkpoint SHA-256
`89058e658b5a21ce1aa87fdbc93d83d38928cf4976f4a4c76f75d888ea81cb37`,
and computed model digest
`2e3eef16a6fb2394b4538fbcde4fa8618d0543871882c7e1a3cba237ce656614`.
The final continuation then loaded that exact manifest/checkpoint under fence
9 before producing generation 11. Fence 8's socket-path attempt never loaded a
model and could not publish. An independent post-job process streamed and
reloaded the 7,899,874,035-byte generation-11 checkpoint, reproduced SHA-256
`5084db3032037ab511aed09296c77f6d120eed5b7e15b758f9f4dcbf09bb158c`,
and computed model digest
`217258beececb4ad6c31e49acc63470807be90962accb9d46eb4a4eef635074c`.

The terminal store has `last_fence=9`, no lease, 11 commit rows, the same
number of checkpoint rows,
and exactly one authoritative latest row at generation 11. This proves restart
uses a strictly newer fence without allowing a fence downgrade or stale latest
overwrite.

## Local-state authority and prohibited paths

Each allocation used a unique `/tmp` bulk and kernel-cache root. Managers were
model-free. Trainers alone loaded model and optimizer, produced deltas directly
in service-owned sealed memfds, mapped the shared read-only result, and wrote
bounded recovery metadata. Retained snapshots intentionally include only
supervision, telemetry, and small JSON/JSONL controls; they exclude mailbox
data, `*.data`, `*.pt`, and kernel caches.

The authoritative state is the fenced SQLite publication plus immutable
checkpoint/handoff. Node-local unfinished buffers were never accepted after a
new fence. Across the accepted jobs, trainer spool, disk replay, Python dense
socket, Lustre dense hot-path, MPI collective, all-rank barrier, and central
full-model broker counts are all zero.

## Production-policy decision

The rung meets validation evidence requirements, but the production policy is
unchanged:

- status remains `defined-no-submission-authority`;
- `measured_policy_baseline_rungs` remains `[2]` in the committed authority;
- `production_qualified_rungs` remains `[]`;
- this run used debug QoS, not production `normal` QoS;
- no promotion record was created and no byte-identical production payload was
  admitted;
- the next graph task may evaluate the four-node evidence and then authorize
  the already ordered eight-node *validation* rung, but this task submitted no
  8+ allocation.

## Architecture conformance checklist

This is the required conformance check against
`RESILIENT_DILOCO_COMPUTE_POOL.md` and
`RESILIENT_DILOCO_GAP_MATRIX.md`.

| Requirement | Four-node live conformance evidence |
|---|---|
| R01 | SQLite granted exclusive leases at strictly increasing fences 4–9 before role/model start; terminal leases are zero. |
| R02 | Stable `node-N` workers and per-start incarnations traversed import/READY/ACTIVE/drain; node 3 rejoined with restart/incarnation 1. |
| R03 | Each generation froze four leased READY managers, not launched ranks; delayed node 3 was admitted only after READY. |
| R04 | Run/fence/generation/attempt/worker/incarnation/sequence identities were exact; SQLite/native receipts show no duplicate or stale acceptance. |
| R05 | Eight local deltas per node reduced f32-to-f64, four exact node numerators reduced deterministically, then projected once to f32. |
| R06 | `Q_min=4`, `T_min=3,934,080`, K40/freeze/exchange deadlines, and fail-closed bounds were explicit. |
| R07 | Every accepted generation has one immutable checkpoint/handoff and atomic fenced commit/checkpoint/latest publications. |
| R08 | Deterministic owners, 64 MiB chunks, authenticated checksums, credit backpressure, bounded queues, prompt release, and no broker are observed. |
| R09 | Four managers remained model-free; 32 trainers owned model/optimizer and independently applied the shared result. |
| R10 | Dense update/aggregate/redistribution stayed in node-local memfd/CXI paths; shared storage held only lease/publication/checkpoint evidence. |
| R11 | Node-3 disappearance drained its service; a fresh service/manager caught up from generation 9 and joined generation 10. |
| R12 | Weighted-mean outer state and accepted-token clock restored exactly at newer fences 7 and 9; inner state remained trainer-owned. |
| R13 | The backend-neutral protocol was bound through the Frontier adapter; no scheduler or MPI identity entered numerical reduction. |
| R14 | READY, K40, exchange, redistribution, apply, checkpoint, and drain stages emitted bounded live telemetry and committed-generation evidence. |
| R15 | Exact frozen weights advanced by 10,490,880 tokens per four-node generation; f64 arithmetic/layout and dtype restoration matched the retained reference contract. |
| R16 | Accepted two-node startup/failure/restart plus exact-source G2 job 5034114 preceded four-node submission; no 8+ rung was submitted. |

| Requirement | Native data-plane v1 conformance evidence |
|---|---|
| NDP01 | Python retained control only; all 5.5 GB dense movement used compiled native memfd/CXI paths, with zero Python dense bytes. |
| NDP02 | Point-to-point routes survived a peer replacement; MPI collectives and all-rank barriers were zero. |
| NDP03 | One persistent C++17 service per node used exact production `cxi` and `FI_EP_RDM`; node 3 received a fresh service on rejoin. |
| NDP04 | All 64 accepted trainer submissions in job 5034117 were producer-direct service memfds with zero spool/full-copy bytes. |
| NDP05 | Native f32/f64/f64/f32 weighted arithmetic, rank order, exact token totals, and retained exact-source G2 analytical equality bind numerical behavior. |
| NDP06 | Fixed framed identities include run/fence/generation/attempt/owner/worker/incarnation/chunk plus checksums and absolute deadlines. |
| NDP07 | Four current-fence endpoint records formed pairwise CXI routes; the replacement installed fresh generation-9 routes. |
| NDP08 | 64 GiB service admission, immutable extents, fixed TX/RX slots, and 268,436,736-byte observed highs stayed within policy bounds. |
| NDP09 | Authenticated receiver credits and separate four-slot CQ bounds kept every peer exchange bounded; RX queue high-water was at most four frames. |
| NDP10 | Header/payload/root validation plus unique result-lane receipts yielded 32 once-only applies per generation and no conflicts. |
| NDP11 | Replay/reassignment mechanisms remained bounded; observed replay and disk fallback were both zero. |
| NDP12 | Four owner numerators redistributed one service-owned 5,506,770,496-byte f32 result to eight independent trainer views per node. |
| NDP13 | Absolute stage deadlines contained every diagnostic failure and the injected node-3 loss locally; no partial commit escaped. |
| NDP14 | Stable v1 metadata-only seqpacket RPC used a sealed admission-token FD; the replacement service received a rewound protected FD. |
| NDP15 | Python checkpoint policy published only after native result, leader proposal, 32 applies, and current-fence CAS; drain was collective-free. |
| NDP16 | Manifests bind source/config/provider/artifact/result/token identities; retained telemetry contains throughput, byte, high-water, release, retry, and zero-path counters. |
| NDP17 | Exact-source full-layout production-CXI G2 job 5034114 passed before the real four-node rung; ordered scale stopped at four nodes. |

## Validation

- [x] Predecessor two-node startup/failure/restart evidence and production
  policy were treated as authoritative.
- [x] Normalized model/K40/native payload is retained and unchanged; node
  capacity, explicit Q/T/owners, dynamic run/fence identity, failure target,
  and recorded queue/walltime are isolated as controls.
- [x] Three exact same-code atomic generations 9–11 committed with native
  throughput, high-water, and release telemetry.
- [x] One bounded node-3 peer loss/rejoin completed without partial or duplicate
  commit or apply.
- [x] Generation-8/fence-6 and generation-10/fence-7 checkpoints reloaded under
  strictly newer fences.
- [x] Exact immutable recipes, accounting, SQLite snapshots, manifests,
  telemetry, audits, and checksums are committed; no 8+ job was submitted.
- [x] Compute-pool requirements R01–R16 and native requirements NDP01–NDP17
  are addressed above.
- [x] Exact-source tests passed: 102 selected Python tests, 10/10 native CTests,
  JSON/checksum audits, and `git diff --check`.

## Evidence index

- peer loss/rejoin allocation: `reports/frontier/evidence/job-5034117`;
- same-code final continuation: `reports/frontier/evidence/job-5034180`;
- fail-closed noncompute attempt: `reports/frontier/evidence/job-5034173`;
- exact-source G2: `reports/frontier/native-dataplane/5034114`;
- immutable recipes: `reports/frontier/validate-native-pool-v1-4n-*.sh`;
- normalized payload: `reports/frontier/validate-native-pool-v1-4n-normalized-payload.json`;
- machine-readable acceptance: `reports/validate-native-pool-v1-4n-metrics.json`;
- top-level integrity manifest: `reports/validate-native-pool-v1-4n-SHA256SUMS`.
