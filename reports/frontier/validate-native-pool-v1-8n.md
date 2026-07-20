# Native resilient pool v1 eight-node validation

Date: 2026-07-20 UTC

WG task: `validate-native-pool-v1-8n`

Final live source commit: `5121539bcd7c9679e32b42bb2ebf1722672d9015`

Two-generation atomic job: `5034745`

Unavailable-owner job: `5034865`

Fresh-fence continuation job: `5034875`

Exact-source clean/fault G2 jobs: `5034807` / `5034843`

## Result

**Passed.** The ordered native resilient-pool ladder ran on exactly eight
Frontier nodes only after the four-node dependency was complete and
authoritative. It preserved the accepted E97 seed, train configuration,
tokenizer, CommaPile source, K40 workload, optimizer, numerical layout,
5,506,770,496-byte result, native artifact bundle, `native-cxi` backend,
production `cxi` provider, and `FI_EP_RDM` endpoint. The canonical normalized
workload digest remains
`0493ba1ebc6f28c3244b6845bffb2de722639ee8e9df7d6ffa1918742e0691f4`.

Eight-node generations 12, 13, and 14 committed atomically at token clocks
110,154,240, 131,136,000, and 152,117,760. Every increment is exactly
20,981,760 accepted tokens from eight frozen READY managers and 64 real
trainers. Each generation has one fenced commit, one checkpoint publication,
one immutable finalized handoff, and 64 unique `(node,rank)` native apply
receipts. The READY worker names stayed stable while every recorded
incarnation set changed between the accepted generation-11 input,
generation-12 input, fault allocation, and final fresh-fence allocation.

Job 5034865 made the node-6 shard owner unavailable during generation-13
`owner_transport`. The old manager was killed, its node-local service was
drained, and the supervisor attempted exactly the configured two fresh
service/manager assignments. They could not reconstruct a same-fence
eight-owner result after peers had failed their bounded wait, so restart
exhaustion stopped the allocation. No generation-14 bytes, checkpoint, commit,
or latest pointer escaped fence 15. The exact-source fault gate 5034843
separately completed the deterministic replay branch with one reassignment,
a fresh owner epoch, rejection of the old epoch, exactly 134,217,728 replay
bytes, no partial commit, and throughput above policy.

Job 5034875 then acquired strictly newer fence 16, synchronized all eight
managers from the generation-13/fence-14 handoff, reconstructed eight fresh
native endpoints, completed K40, and committed generation 14. All eight
managers and all 64 trainers exited zero, and all eight native services drained
zero for `allocation_complete`. The terminal SQLite database reports
`integrity_check=ok`, `last_fence=16`, no active lease, 14 commit rows, 14
checkpoint rows, and exactly one latest row at generation 14/fence 16.

The final layout placed 165 compact contribution shards round-robin over eight
owners. Per-owner contribution extent differed by at most one 64 MiB frame
(1,342,177,280–1,409,286,144 bytes), and result redistribution extent differed
by 32 MiB (671,088,640–704,643,072 bytes). All 56 routes became ready in
0.0032–1.0386 seconds; owner contribution completed in 14.7757–16.8382
seconds; owner redistribution completed in 12.1820–13.2339 seconds; all
records were inside policy.

The final eight services each published one endpoint/fabric-ready record and
one shutdown record. No service restarted, so the normative one endpoint, one
TX CQ, one RX CQ, and one progress thread per service were created once and
could not grow across the generation. Operational CQ and route errors were
zero. The only CQ increments were the fixed TX/RX cancellation pair during
collective-free drain. All services ended with zero in-flight, retained,
replay, and route-error bytes/counters.

This is a **validation-only rung**. The production policy remains
`defined-no-submission-authority`; it does not authorize normal QoS or a
production payload. No 32-, 64-, or larger-node job was submitted.

The machine-readable result is
[`validate-native-pool-v1-8n-metrics.json`](validate-native-pool-v1-8n-metrics.json).
Live evidence is retained under
[`frontier/evidence/job-5034865`](frontier/evidence/job-5034865) and
[`frontier/evidence/job-5034875`](frontier/evidence/job-5034875). The final
exact-source clean and replay gates are under
[`frontier/native-dataplane/5034807`](frontier/native-dataplane/5034807) and
[`frontier/native-dataplane/5034843`](frontier/native-dataplane/5034843).
The 7.9 GB checkpoint payloads remain at their immutable recorded run paths;
Git retains their handoffs, publications, byte counts, independent full-file
digests, and reload evidence.

## Four-node authority and ordered admission

The dependency was a hard launch gate, not an inference from the presence of
old files. `validate-native-pool-v1-4n` completed evaluator acceptance at
`2026-07-20T04:16:40Z` and pushed immutable evidence in commit `41951669`.
Its authoritative latest pointer was generation 11/fence 9 at 89,172,480
tokens, with handoff SHA-256
`c1896f28f805ca9cf519f470d9e5f1b5132eae33d6decec232abac43b0f16a1e`.

The first eight-node submission occurred at `2026-07-20T05:08:48Z`, 3,128
seconds after the accepted dependency. Its immutable recipe asserted all of
the following before `sbatch`:

- a clean pushed exact source snapshot on the task branch;
- the accepted seed, tokenizer, config, bundle, and generation-11 handoff
  digests;
- SQLite `last_fence=9`, no lease, and latest
  `9:11:89172480`;
- a clean exact-source full-layout production-CXI G2 attestation;
- no overlapping user allocation;
- exactly eight requested nodes, eight GPUs per node, debug QoS, a 30-minute
  cap, and `job_vni`.

The admission implementation introduced for this rung accepts eight only in
the ordered validation context and explicitly rejects 32 or more. Each later
recipe repeated the exact source/digest/latest/lease/G2 preflight for its
snapshot and intended fence. No task recipe contains a 32+ request.

## Byte-identical payload and scale controls

[`validate-native-pool-v1-8n-normalized-payload.json`](frontier/validate-native-pool-v1-8n-normalized-payload.json)
retains the normalized training and data-plane payload. Canonical
`jq -cS .normalized_workload` bytes reproduce the accepted four-node digest
`0493ba1e...`.

The following payload facts did not change:

- seed/source SHA-256
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`,
  config SHA-256
  `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c`,
  and tokenizer SHA-256
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`;
- eight trainers per node, ScheduleFree inner optimizer, weighted-mean outer
  update with `eta_outer=1`, 40 local steps, and staleness `tau=0`;
- the exact 5,506,770,496-byte f32 result layout, 83 base shards, 64 MiB
  frames, 64 GiB node-local admission, four TX/four RX slots, and
  checkpoint-every-generation cadence;
- native bundle
  `7b4c77696011cfc2b68fce541a34f9a1072ef767e13cce94ed2a4e2befc57d04`,
  exact provider `cxi`, `FI_EP_RDM`, `job_vni`, `kdreg2`, and ATS disabled;
- model/optimizer ownership, f32-to-f64 local reduction, deterministic f64
  aggregate, one f64-to-f32 projection, and exact accepted-token weighting.

The isolated scale controls changed four nodes/owners and 32 trainers to eight
nodes/owners and 64 trainers. To keep peak memory and route concurrency
bounded, the eight-node mapper represents the same aggregate layout as 165
compact round-robin owner shards and uses an explicit 180-second scale-stage
bound. This changes ownership/control scheduling, not model, optimizer,
training, provider, native binary bundle, numerical layout, or K40 bytes.
Fences, incarnations, nodelists, allocation IDs, source snapshots, and
node-local `/tmp` roots are dynamic identity. The single fault target is an
explicit validation control.

The final generation uses source `5121539b` rather than `95049241` because the
two-generation run exposed a focused control-plane lifetime defect: the short
INSTALL RPC deadline was also expiring the service-owned result before the
longer apply/checkpoint/commit sequence finished. Commit `5121539b` separates
the RPC wait from a bounded 180-second recovery/commit lifetime. It does not
change any dense payload or native artifact byte; the bundle digest is exactly
the same in generations 12–14.

## Frontier allocation accounting

All eight-node attempts used `batch`, debug QoS, eight GPUs per node,
`job_vni`, and a 30-minute cap. The full immutable accounting is
[`validate-native-pool-v1-8n-attempts-sacct.txt`](frontier/evidence/validate-native-pool-v1-8n-attempts-sacct.txt).

| Job | Result | Durable result | Purpose |
|---|---|---|---|
| `5034361` | failed closed, 1:07 | fence 10, no commit | Exposed an overlong node-local service socket path. |
| `5034380` | failed closed, 7:29 | fence 11, no commit | Full all-to-all f64 owner exchange exceeded its bounded scale deadline. |
| `5034531` | cancelled after 0:21 | no fence/model/commit | Sparse-memory audit was stopped before work began. |
| `5034633` | failed closed, 7:07 | fence 12, no commit | Compact owner layout exposed CREDIT/DATA RDM phase reordering. |
| `5034692` | failed closed, 6:43 | fence 13, no commit | Per-peer phase serialization exposed the contribution/redistribution direction boundary. |
| `5034745` | failed after 14:51 | generations 12 and 13/fence 14 | Two complete atomic publications; then result-lifetime expiry prevented generation 14. |
| `5034865` | failed closed, 6:11 | fence 15, no new commit | Injected node-6 owner loss; two bounded fresh assignments exhausted. |
| `5034875` | `COMPLETED 0:0`, 7:04 | generation 14/fence 16 | Fresh-fence reload, native reconstruction, and third commit. |

Failures were contained at protocol boundaries. Jobs 5034361, 5034380,
5034633, 5034692, and 5034865 advanced at most the lease fence and published
no model. Job 5034745 published two fully durable generations before a later
stage failed; its two results are legitimate immutable commits, not partial
attempts. Job 5034531 was cancelled before fence/model admission. The latest
pointer therefore advanced only for generations 12, 13, and 14.

## Three exact weighted generations and changing READY membership

| Generation | Fence | Step | Accepted tokens | Exact increment | Manifest SHA-256 | Checkpoint SHA-256 | Source |
|---:|---:|---:|---:|---:|---|---|---|
| 12 | 14 | 1,525,480 | 110,154,240 | 20,981,760 | `20f82519d568cde46d77f3c16715a3cbabf2e9a236b809660848d0c977682dc8` | `d4255d0f3432027b953483fa4df863f9eee98c8ea1610778b604ae697e062dd5` | `95049241` |
| 13 | 14 | 1,525,520 | 131,136,000 | 20,981,760 | `040ea79b0592ffae3f13e8d60cbd50c7fcc1e19341201cdd2e95fc959cbe4e5b` | `6ca549b545b262c84d4265b6c3caa094997ed5cb38dd3f8494138d2b0eb89457` | `95049241` |
| 14 | 16 | 1,525,560 | 152,117,760 | 20,981,760 | `31693c27af8b9eb18616dfb9c41cc71e5b6d5410288ca7c37a2a94f00550405a` | `24cb63a8814dfc410b96c72ffcecf5b6aa688e54b7f0f612b8c6a0bdb8f0e0ad` | `5121539b` |

The weight is exact: 64 trainer contributions form eight node-local weights,
and the frozen global weight is 20,981,760 for every accepted generation.
All three handoffs bind the same layout digest
`e3fb15da10a151dbd33d6f66a3a2f8723be69bbaf7b34a6b3652bee0f5a352e2`
and the same native bundle. Generation-12, -13, and -14 result roots are,
respectively, `a2199532...`, `fe1494cc...`, and `915b19f8...`.

Retained control receipts contain exactly 64 unique node/rank applies for each
result. For generation 12 those receipts bind fence 14 and root `a2199532...`;
for generation 13 they bind fence 14 and root `fe1494cc...`; for generation 14
they bind fence 16 and root `915b19f8...`. No `(node,rank,generation)` is
missing or duplicated.

Membership was READY-derived rather than inferred from launched ranks. The
eight stable worker IDs were constant, but their current incarnations changed:

- generation-11/fence-14 began with node-0/node-7 incarnations
  `9ad5be54...` / `a2cf0198...`;
- generation-12/fence-14 froze `7d51008b...` / `8d493360...`;
- the generation-13/fence-15 fault attempt froze `9a049d5f...` /
  `3a6a1f0e...`;
- generation-13/fence-16 continuation froze `8b199afd...` /
  `9c4a2f36...`.

Every set contains eight distinct current-fence identities, records
`required_contributions=8`, `accepted_tokens=20,981,760`, and reason
`accepted_floor_met`. No old incarnation was admitted into a newer freeze.

## Unavailable owner, bounded replay/reassignment, and fresh fence

The accepted failure sequence is deliberately split between the live
eight-node containment test and the exact-source deterministic replay gate.
This makes both properties observable without weakening the atomic commit
rule.

In job 5034865, all eight managers reached generation 13 and froze eight
current fence-15 identities. When node 6 entered `owner_transport`, the gated
injector killed `node-6-manager`. The old service drained for
`injected_generation_gate`; the supervisor created a new service and manager
with restart 1, then a second fresh pair with restart 2. Neither replacement
could accept a stale partially assembled owner result after peer deadlines.
`restart_exhausted` terminated the allocation exactly at the configured
bound. The fence-14 latest pointer remained generation 13 at 131,136,000
tokens, fence 15 produced zero generation-14 publications, and the lease was
released.

Exact-source G2-fault job 5034843 exercises the successful bounded transport
replay branch in isolation. It reports:

- peer loss observed and one owner reassignment;
- owner epoch 1 replaced by epoch 2 with a new incarnation;
- stale epoch rejection;
- exactly 134,217,728 replay bytes;
- no partial commit, no disk replay, and exact independent-reference equality;
- 920,277,319 logical bytes/s, above the 890,329,830-byte/s production floor.

The clean companion G2 job 5034807 completed three exact timed generations at
24.5563, 23.0060, and 22.9423 seconds. Its median rate was 957,448,319
logical bytes/s, 4.3015x the independent Python baseline, also above the
policy floor. Both jobs used source `5121539b`, the exact `cxi` provider, the
same native bundle as the model run, zero operational CQ/route errors, and
zero post-release bytes.

The continuation job then demonstrated durable recovery. It acquired fence 16
after the failed fence-15 lease was gone. Every manager wrote a synchronized
receipt for generation-13/fence-14 manifest `040ea79b...` under fence 16,
with a fresh incarnation. All eight manager processes reached `published`
generation 14, and all 64 trainer processes reached `applied` generation 14.
This is the required fresh-fence checkpoint continuation after an unavailable
owner, without reusing any unfinished node-local state.

## Balanced placement, resource lifecycle, and throughput

The final owner placement records are explicit rather than reconstructed from
rank order:

| Stage | Samples | Owner range | Maximum skew | Time range | Hard limit |
|---|---:|---:|---:|---:|---:|
| compact contribution ownership | 8 | 1,342,177,280–1,409,286,144 B | 67,108,864 B | 14.7757–16.8382 s | 180 s |
| compact result ownership | 8 | 671,088,640–704,643,072 B | 33,554,432 B | 12.1820–13.2339 s | 180 s |
| full local reduction | 8 | 8 contributions/node | n/a | 17.3006–17.7930 s | 180 s |
| pairwise route readiness | 56 | 7 routes/node | n/a | 0.0032–1.0386 s | 15 s |
| full native redistribution | 8 | 5,506,770,496 B result | n/a | 55.0905–56.3066 s | 180 s |
| independent trainer apply | 64 | 1 lane/trainer | n/a | 3.1986–10.7569 s | 60 s |

Every owner record says `balanced_owner_placement=true`. Contribution skew is
exactly one 64 MiB frame, result skew is half a frame, and the RX queue
high-water is at most two frames. `python_dense_socket_bytes` and
`trainer_spool_bytes` are zero in every relevant record.

Final job 5034875 has a fixed lifecycle:

- exactly eight service, eight manager, and 64 trainer starts;
- exactly one `fabric_ready` and one endpoint record per service;
- no service, manager, or trainer restart;
- 64 zero-exit trainers, eight zero-exit managers, and eight zero-drain
  services;
- one TX CQ, one RX CQ, and one native progress thread per service by the
  normative v1 construction, with no second construction path entered.

The eight terminal telemetry records total 115,642,180,416 useful TX bytes,
the same useful RX bytes, and 231,287,353,472 released bytes. High-water is
268,436,736 bytes for both in-flight and retained storage. Every terminal has
zero in-flight bytes, retained bytes, replay bytes, and route errors. The
lowest whole-allocation service rate is 85,300,500 B/s; that clock includes
model training, checkpointing, and idle commit lifetime and is not the
full-layout production gate. Production throughput is decided by the exact
source G2 measurements above, both of which exceed the recorded floor. Every
live model stage also remained inside its hard SLO.

Operational CQ errors stayed zero through all transfer completions. On drain,
each service reports exactly two `status=-12`, provider errno 125
cancellations—one for each fixed TX/RX CQ side—followed by shutdown. This is
the expected collective-free cancellation pair, not CQ accumulation or a
data-path error.

## Checkpoint continuation and terminal authority

An independent post-job `sha256sum` streamed all three full checkpoints from
the immutable run directory and reproduced their handoff/SQLite digests:

| Generation/fence | Bytes | Independent SHA-256 |
|---|---:|---|
| 12/14 | 7,899,874,291 | `d4255d0f3432027b953483fa4df863f9eee98c8ea1610778b604ae697e062dd5` |
| 13/14 | 7,899,874,291 | `6ca549b545b262c84d4265b6c3caa094997ed5cb38dd3f8494138d2b0eb89457` |
| 14/16 | 7,899,874,419 | `24cb63a8814dfc410b96c72ffcecf5b6aa688e54b7f0f612b8c6a0bdb8f0e0ad` |

The fence-16 manager synchronization receipts independently bind source
generation 13, source fence 14, and handoff `040ea79b...`. Native services and
manager processes were new; unfinished fence-15 `/tmp` state was neither
authoritative nor eligible. Sixty-four trainers built device state and
optimizer state from the authoritative continuation, ran K40, submitted direct
memfds, independently applied the new result, and recorded generation-14
receipts.

The terminal database contains 14 unique `commit` rows and 14 matching
`checkpoint` rows for the logical run. Generations 12–14 each appear exactly
once. The sole latest row is generation 14/fence 16 at 152,117,760 tokens.
`PRAGMA integrity_check` is `ok`; `last_fence` is 16; active leases are zero.
There is no fence downgrade, stale latest overwrite, duplicate publication,
or partial generation.

## Focused fixes and exact-source verification

The eight-node rung exposed bounded scale defects that could not occur at four
owners. Each was fixed narrowly and validated before the next live attempt:

- `4ec9e770`: ordered eight-node admission and explicit 32+ rejection;
- `9332ccca`: bounded sharded owner exchange;
- `d31fbbcb`: deterministic compact round-robin owner layout;
- `a8f8a93a`: serialization of each peer's transfer phases;
- `95049241`: fenced direction change between contribution and redistribution;
- `5121539b`: distinct short INSTALL RPC and bounded commit-lifetime result
  deadlines.

The final fix has regression coverage proving that an INSTALL call can time
out locally while the installed generation remains recoverable/abortable for
its explicit generation lifetime. It propagates that lifetime through the
native client/runtime boundary and gives the final reduction attempt the
remaining owner deadline plus a bounded 180-second recovery window.

With the canonical Frontier environment activated, the final source passed:

- 110/110 focused Python tests across native failure handling, native pool
  integration, E97 runtime/source contracts, launcher admission, and resilient
  pool runtime in 135.90 seconds;
- 10/10 compiled native CTests in the clean exact-source snapshot;
- exact-source clean and fault G2 jobs 5034807/5034843, both `COMPLETED 0:0`;
- independent checkpoint streaming, SQLite integrity, JSON/JSONL parsing,
  nested and top-level SHA-256 verification, and `git diff --check`.

The command/result summary is retained in
[`validate-native-pool-v1-8n-validation.txt`](frontier/evidence/validate-native-pool-v1-8n-validation.txt).

## Local-state authority and prohibited paths

Every allocation used a unique `/tmp` bulk/kernel-cache root. Managers stayed
model-free. Trainers alone owned model/optimizer state, wrote deltas into
service-owned memfds, and mapped service-owned result lanes. Dense update,
owner contribution, owner redistribution, and independent apply remained on
node-local memory/CXI paths. Shared storage held only fenced control,
immutable handoffs/checkpoints, and retained small evidence.

Across accepted evidence, Python dense socket bytes, trainer spool bytes,
Lustre dense hot-path bytes, disk replay bytes, MPI collectives, all-rank
barriers, and central full-model brokers are all zero. The only nonzero replay
is the explicitly bounded 128 MiB in the exact-source fault gate, held inside
the native transport recovery path.

## Production-policy decision

The rung meets its validation acceptance criteria but does not promote a
production payload:

- policy status remains `defined-no-submission-authority`;
- `production_qualified_rungs` remains empty;
- all model jobs used debug rather than normal QoS;
- no production promotion record was created;
- no 32+ job was submitted;
- downstream `validate-native-pool-v1-32n` remains dependent on independent
  evaluation of this evidence and receives no authority merely from this
  report.

## Architecture conformance checklist

This is the required conformance check against
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1 and requirement IDs from
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. The native checks also bind
`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md` version 1.

| Requirement | Eight-node live conformance evidence |
|---|---|
| R01 | Exclusive SQLite leases advanced from authoritative fence 9 through fences 10–16; terminal fence is 16 and active leases are zero. |
| R02 | Stable `node-0`–`node-7` workers used per-start incarnations; all four frozen incarnation sets differ, and replacements re-entered only through READY. |
| R03 | Each accepted generation froze exactly eight leased READY managers and 20,981,760 tokens, never launched-rank membership. |
| R04 | Run/fence/generation/attempt/worker/incarnation/sequence identities are bound in pool/native receipts; stale owner epoch was rejected and no duplicate was accepted. |
| R05 | Eight local f32 contributions per node reduced to f64, eight weighted node numerators reduced deterministically, then the 5,506,770,496-byte result projected once to f32. |
| R06 | `Q_min=8`, `T_min=3,934,080`, K40, route, owner, apply, generation, and restart limits were explicit; diagnostic attempts failed closed at them. |
| R07 | Generations 12–14 each have one finalized checkpoint/handoff and atomic fenced commit/checkpoint/latest publication. |
| R08 | Deterministic compact owners, 64 MiB frames, checksums, credits, phase fencing, bounded queues, prompt release, and no broker are observed. |
| R09 | Eight managers remained model-free; 64 trainers owned model/optimizer state and independently applied one native result lane each. |
| R10 | Dense bytes stayed in memfd/CXI paths; shared storage contains lease/publication/checkpoint/evidence only. |
| R11 | Node-6 disappearance drained its service; two bounded fresh same-fence assignments failed closed, and fresh fence 16 reconstructed from durable generation 13. |
| R12 | Weighted-mean state and accepted-token clock restored from generation-13/fence-14 under fence 16; inner optimizer state remained trainer-owned. |
| R13 | Backend-neutral protocol bound through the Frontier native adapter; neither scheduler rank nor MPI identity entered numerical reduction. |
| R14 | READY, K40, route, owner contribution, redistribution, apply, checkpoint, commit, and drain stages emitted bounded telemetry/evidence. |
| R15 | Exact frozen weights advanced 20,981,760 tokens per generation; f64 arithmetic/layout/dtype contract and full-layout exact-reference G2 passed. |
| R16 | Evaluated four-node acceptance preceded the first eight-node submission; exact-source G2 preceded final live jobs; ordered scale stopped at eight and no 32+ job was submitted. |

| Requirement | Native data-plane v1 conformance evidence |
|---|---|
| NDP01 | Python retained control only; dense movement used compiled native memfd/CXI paths with zero Python dense bytes. |
| NDP02 | Eight point-to-point endpoint routes were constructed without MPI collectives or all-rank barriers; failure stayed peer/owner scoped. |
| NDP03 | Exactly one persistent C++17 service per node used `cxi`/`FI_EP_RDM`; the final allocation created each once and drained it once. |
| NDP04 | Sixty-four final trainer submissions were producer-direct service memfds with zero spool bytes; each had a unique result lane/apply receipt. |
| NDP05 | Native f32/f64/f64/f32 weighted arithmetic, stable rank order, exact 20,981,760 weights, and independent G2 equality bind numerical behavior. |
| NDP06 | Framed identities bind run/fence/generation/attempt/owner/worker/incarnation/chunk, checksums, phases, and absolute deadlines. |
| NDP07 | Eight current-fence endpoint records formed 56 pairwise routes; fence 16 constructed a wholly fresh endpoint/incarnation set. |
| NDP08 | 64 GiB service admission, immutable extents, fixed four TX/four RX slots, and 268,436,736-byte observed highs remained bounded. |
| NDP09 | Authenticated credits and serialized contribution/redistribution phases bounded RX queues to at most two 64 MiB frames. |
| NDP10 | Header/payload/root validation plus unique lane receipts yielded 64 once-only applies per generation and no conflicts. |
| NDP11 | The live owner failure exhausted exactly two assignments and failed closed; G2 replay succeeded with one reassignment and exactly 128 MiB native replay; disk replay stayed zero. |
| NDP12 | Eight owner results redistributed one service-owned 5,506,770,496-byte f32 result to eight independent trainer views per node. |
| NDP13 | Absolute stage deadlines contained every diagnostic failure and the injected node-6 loss; no partial result escaped. |
| NDP14 | Stable v1 metadata-only seqpacket RPC used sealed admission tokens; replacement services received fresh protected state. |
| NDP15 | Python checkpoint policy published only after native result, leader proposal, 64 applies, and current-fence CAS; drain was collective-free. |
| NDP16 | Manifests bind source/config/provider/artifact/result/token identities; telemetry retains placement, throughput, bytes, highs, release, retry, replay, and zero-path counters. |
| NDP17 | Final-source production-CXI clean and failure/replay G2 gates passed before the owner-fault/fresh-fence conclusion; scale stopped at eight. |

## Validation

- [x] No launch occurred before the four-node gate was evaluator-complete and
  authoritative: 4n completion `04:16:40Z`, first 8n submit `05:08:48Z`.
- [x] Generations 12–14 committed with exact 20,981,760-token weights,
  64 unique applies each, and distinct READY incarnation sets.
- [x] Eight-owner compact placement is balanced; the final allocation created
  its fixed endpoint/CQ/progress-thread topology once with no restart/growth;
  all live stages met SLO and final-source clean/fault G2 throughput exceeded
  the 890,329,830-byte/s policy floor.
- [x] The unavailable node-6 owner exercised two bounded assignments and
  failed closed; exact-source G2 completed one bounded 128 MiB replay and
  reassignment; fence 16 reloaded fence-14 generation 13 and committed 14.
- [x] Immutable evidence, recipes, metrics, focused fixes, and checksums are
  committed and pushed; no 32+ job was submitted.
- [x] Compute-pool requirements R01–R16 and native requirements NDP01–NDP17
  are addressed above.
- [x] Final verification passed: 110 Python tests, 10 native CTests, clean and
  fault G2, three independent checkpoint streams, SQLite/JSON/checksum audits,
  and `git diff --check`.

## Evidence index

- final fresh-fence allocation: `reports/frontier/evidence/job-5034875`;
- unavailable-owner allocation: `reports/frontier/evidence/job-5034865`;
- retained early full-exchange diagnostic: `reports/frontier/evidence/job-5034380`;
- all-attempt Slurm accounting: `reports/frontier/evidence/validate-native-pool-v1-8n-attempts-sacct.txt`;
- exact-source clean/fault gates: `reports/frontier/native-dataplane/5034807` and `5034843`;
- immutable submission recipes: `reports/frontier/validate-native-pool-v1-8n-*.sh`;
- normalized payload: `reports/frontier/validate-native-pool-v1-8n-normalized-payload.json`;
- validation transcript summary: `reports/frontier/evidence/validate-native-pool-v1-8n-validation.txt`;
- machine-readable acceptance: `reports/validate-native-pool-v1-8n-metrics.json`;
- top-level integrity manifest: `reports/validate-native-pool-v1-8n-SHA256SUMS`.
