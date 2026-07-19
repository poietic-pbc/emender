# Native resilient pool v1 production policy

**Status:** reviewed policy definition; **no production submission authority**

**Version:** 1.0.0, 2026-07-19

**Profile:** E97 1.3B, K40, eight trainers per node, native CXI

**Machine-readable authority:**
[`native_resilient_pool_v1_production_policy.json`](../configs/frontier/native_resilient_pool_v1_production_policy.json)

This document resolves the production-policy decisions left open by
[*Resilient DiLoCo Compute Pool*](RESILIENT_DILOCO_COMPUTE_POOL.md), version 1,
for the measured native E97 profile. It specializes, but does not weaken, that
authority or [*Native resilient DiLoCo data plane v1*](NATIVE_RESILIENT_DILOCO_DATAPLANE.md).
The applicable traceability requirements are **R01–R16** and **NDP01–NDP17**
from [the gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md).

The policy is deliberately fail-closed. The retained two-node results define
the numeric baseline, but do not authorize a production or Slurm job. The
machine-readable policy currently has no `production_qualified_rungs`; a
reviewed packaging/promotion record must add an exact rung only after all
ordered prerequisites, including fresh-allocation recovery, have passed. The
next validation capacity is four nodes, not a four-node production launch.

## Decision summary

For the measured two-node profile:

| Decision | Production value | Basis |
|---|---:|---|
| Complete node contributions, `Q_min` | 2 | Both accepted real runs froze two leased READY node incarnations. One remaining node is below the native owner and global contribution floor. |
| Accepted-token floor, `T_min` | 3,934,080 | Exact full-layout G2 node weights `1,966,080 + 1,968,000`; real generations exceeded it at 5,245,440. |
| READY fraction | none | The explicit Q/T floor is authoritative and is evaluated against the live leased snapshot, never launched ranks. |
| Local contribution floor | 8/8 trainers on each contributing node | The accepted native handoff used all lanes 0–7. A partial node numerator is not admissible. |
| Native owners | exactly 2 for the two-node profile | G2 and all accepted real generations used two distributed owners; v1 requires at least two. |
| Staleness | `tau = 0` | Only current fence/generation/attempt input is admissible. |
| Generation retries | 0 | All three accepted generations stayed at generation attempt 0; the owner-loss run aborted rather than repeating an unchanged generation. |
| Manager/trainer restarts | at most 2 per role, inside the original absolute deadline | The accepted progress run used one trainer restart and one manager restart with `MAX_RESTARTS=2`; it had no exhaustion or unplanned restart. |
| Native-service restarts after a generation opens | 0 | The measured persistent-service/owner loss failed closed. A future service-restart policy requires a separately qualified payload. |
| Owner reassignment | at most 2; deadline never resets | NDP11 hard bound. The two-node profile cannot use it after losing one of its two required owners. |
| Sender replay | initial send plus at most `2*L = 11,013,540,992` bytes | NDP11 with E97 `L=5,506,770,496`; accepted runs used zero replay. |
| Checkpoint cadence | every committed generation | The progress run produced three independently verified 7,899,873,331-byte checkpoints for three commits. |
| Retained open generations | 1 plus compact prior receipts | NDP08 bound; all dense bytes must release before next admission. |
| Production scale rungs presently registered | none | Two nodes are a measured policy baseline. This policy task grants no submission authority. |

`T_min` is an independent safety condition even though `Q_min=2` caused the
accepted real generations to exceed it. A production renderer must pass
`--global-token-min 3934080` explicitly; a default or implicit value is not
acceptable. Later node-count profiles must carry separately measured Q/T and
owner values in their exact qualification record. This policy does not invent
a formula that extrapolates the two-node values to larger rungs.

## Retained measurements and fixed deadlines

The policy binds these immutable inputs by path and SHA-256 in the JSON:

- `reports/validate-native-pool-v1-2n-startup-metrics.json`, SHA-256
  `d4d12cbd4a743ede611bef98c8f48a9b4f687eb961e1968d624ec29a2c62b7ee`;
- `reports/validate-native-pool-v1-2n-failures-metrics.json`, SHA-256
  `335cb0e5fc6f268f5700aec680c43c0dd9f99ea063de499c998faa1710c99d10`;
- exact-source G2 `5033380` `full-layout-gate.json`, SHA-256
  `0f813eccd2fb54cf8d543f173e0027d5849a8f611458bf3ef6acaa2509270ab6`;
  and
- the immutable two-node failure recipe, SHA-256
  `f0fcf2ff0f999ff3fc0980e6068a69fdde0eb454bff9679f6f6614bdc69ea07e`.

No timeout was enlarged from these runs. The production values preserve the
exact qualification bounds:

| Stage | Retained observation | Production hard bound | Required response at bound |
|---|---:|---:|---|
| Allocation lease | renewed every 10 s | TTL 60 s | Renewal failure immediately stops admission/publication. |
| Manager/native READY | 14.784 s maximum | 180 s | Exclude/restart the role; do not wait for launched ranks. |
| Heartbeat / loss detection | native-service loss observed at 19.015 s | 60 s | Expire the incarnation and evaluate floors. |
| Real K40 training | 137.010 s maximum | 420 s | Discard unfinished local work; node contribution is absent. |
| Freeze | immediate after the floor in accepted runs | 15 s | Abort the attempt. |
| Native local reduction | 17.164 s maximum | 180 s | Abort/exclude the node contribution. |
| Pairwise route READY | 0.012 s maximum | 15 s | Expire the route; no payload send. |
| CXI owner exchange | 48.059 s maximum | 90 s | Bounded replay/reassignment or abort. |
| Redistribution | 18.625 s maximum | 60 s | Abort; never expose a partial aggregate. |
| Slowest ordered trainer apply | 45.866 s | 60 s | Attempt fails closed; speculative trainers reload `S_g`. |
| Exchange + checkpoint commit | inside accepted 352/381.931 s first commit | 180 s substage | No `latest` advance; abort/reload the last commit. |
| Steady generation, open through durable publication | three generations in 1,049 s | 600 s absolute | No same-generation retry; drain from the prior commit. |
| First atomic commit from allocation start | manifest 352 s; full publication 381.931 s | 720 s absolute | Exit without production progress. |
| Fenced TERM handoff | 37.685 s after injection | 45 s total; 30 s drain | Kill remaining local processes; keep prior durable state. |

Retries, reassignment, backpressure polling, and manager reincarnation never
reset a parent deadline. The 600-second generation limit is end-to-end through
current-fence publication. Therefore a supervisor must not open a steady
generation with less than `600 + 45 = 645` seconds of allocation walltime
remaining. It must not begin first-commit work unless the allocation provides
at least `720 + 45 = 765` seconds. Scheduler TERM stops new admission and may
checkpoint only state already durably valid under the current fence.

### Throughput floors

G2 moved `11,013,540,992` logical contribution bytes and the same number of
redistribution bytes per timed generation. Its retained median was
`22.690315566 s` or `970,770,191.3589156 logical B/s`. Production pre-admission
retains the accepted 4x-native gate:

- G2 logical throughput at least **890,329,830.36 B/s**; and
- G2 median no greater than **24.740361642 s**.

The real owner exchange took 48.059 seconds, or
`229,166,734.92595 B/s`. The live production health floor is the retained
Python baseline **222,582,457.59 B/s**, which the real run exceeded without
widening its 90-second stage bound. A completed, correctly fenced generation
does not become invalid merely because it falls below this operational floor;
it remains authoritative, but the allocation must not open another generation
and must drain after publishing it. Crossing the absolute stage deadline
aborts the current attempt instead.

At the Q/T/deadline floor, committed-token throughput is at least
`3,934,080 / 600 = 6,556.8 tokens/s`. The accepted three-generation failure
run achieved `15,736,320 / 1,049 = 15,001.258341 tokens/s`. Both byte and token
rates are mandatory telemetry; missing values fail qualification.

## Memory, credit, and buffer policy

The E97 layout is 5,506,770,496 bytes in 83 shards. Production admits no
larger value and no unbounded allocation:

| Bound | Value |
|---|---:|
| Node-local shared-byte ledger | 68,719,476,736 bytes (64 GiB) |
| Node-local spool ledger | 68,719,476,736 bytes; admitted dense trainer/disk spool usage must remain zero |
| Frame payload | at most 67,108,864 bytes (64 MiB) plus the fixed 320-byte header |
| Registered fabric slots | 4 TX and 4 RX |
| Two-owner accumulator | at most `ceil(L/2)+64 MiB = 2,820,494,112` bytes |
| Owner resident admission | 14,372,851,648 bytes |
| Owner resident hard bound | 14,440,737,184 bytes |
| Process RSS hard bound | 20,215,943,136 bytes; G2 observed 14,730,473,472 |
| Observed transport in-flight high-water | 67,110,144 bytes |
| Observed transport retained high-water | 268,436,736 bytes |

Every plan is checked before attaching trainer buffers. Credit exhaustion may
poll the provider only inside the current absolute deadline; `FI_EAGAIN`
polling is not a payload replay or a reason to allocate more memory. A commit
is not complete until admitted dense bytes and handles release. The accepted
runs ended with zero in-flight, retained, replay, and post-release transport
bytes. Any nonzero post-generation dense retention, credit/memory bound
violation, trainer spool, disk replay, or Python dense payload blocks the next
generation and production promotion.

## Admission: all checks occur before model load

The production packager must materialize a candidate record and pass it to
`ndm.native_pool_production_policy.validate_production_candidate` (or its
module CLI). Admission is conjunctive; there is no warning-only mode.

1. **Exclusive current lease (R01, R07, R12; NDP06, NDP15).** Acquire the
   logical run lease before loading a model. A live competing owner produces a
   successful zero-work exit. The admitted allocation must hold the current
   60-second lease, renew every 10 seconds, and have a strictly newer fence
   than a resumed checkpoint.
2. **Exact native backend/provider (R08, R10, R13; NDP01–NDP03, NDP07).** The
   backend is exactly `native-cxi`, effective provider exactly `cxi`, endpoint
   exactly `FI_EP_RDM`, and network `job_vni`. `python-tcp-debug`,
   `native-test`, utility/test providers, fixed-world MPI, or provider fallback
   is a hard refusal.
3. **Clean build and complete digests (R04, R08, R16; NDP05–NDP07, NDP14,
   NDP16–NDP17).** Source HEAD, clean build manifest, G2, scale qualification,
   and candidate must name one 40-hex source commit and one native bundle.
   SHA-256 is required for `ndp_cxi_service`, `libemender_ndp.so.1`,
   `libemender_ndp_transport.so.1`, and the synthetic gate binary. Missing,
   changed, dirty, or mismatched bytes refuse production.
4. **Exact ordered rung (R16; NDP17).** Node count must appear in
   `production_qualified_rungs`, and the record must bind node count, all
   smaller rungs, source, bundle, provider, config, normalized payload, layout,
   owners, and bounds. Qualification at N nodes never admits N+1 or a changed
   payload. The checked-in list is intentionally empty.
5. **Exact runtime policy (R02–R06, R08–R15; NDP04–NDP13, NDP15–NDP16).** All
   Q/T, K40, trainer, staleness, owner, deadline, retry, byte, checkpoint, and
   memory fields match the JSON. A defaulted or omitted value is a mismatch.
6. **Fresh checkpoint/fence (R01, R04, R07, R11–R12; NDP06, NDP10, NDP15).**
   Cold start accepts only the pinned seed SHA-256. Resume requires zero
   committed-generation lag: authoritative SQLite `latest`, immutable
   manifest, and independently recomputed checkpoint must have identical
   generation, accepted-token clock, and digest. The new allocation fence is
   strictly greater than the checkpoint/latest fence. Dynamic compatibility
   `latest.json`, node-local recovery, a same-fence allocation, or any stale
   generation cannot authorize resume.
7. **Promotion parity.** The retained validation qualification, normalized
   validation payload, and production payload have the same SHA-256. Only the
   scheduler envelope differences described below are permitted.

This explicitly refuses Python TCP, a non-CXI provider, a missing native
digest, an unqualified scale rung, and a stale checkpoint/fence. Refusal is
terminal for that candidate; no code path silently substitutes a backend,
widens a timeout, lowers a floor, or selects an older checkpoint.

## Node, rank, owner, and allocation-loss behavior

The active world is the current leased READY snapshot, not the Slurm node or
rank count. The following thresholds are production behavior, not suggestions:

| Event | Bounded local action | Continue condition | Whole-allocation action |
|---|---|---|---|
| Trainer/rank loss | Discard unfinished trainer work; restart that role with a new process/incarnation, at most twice, inside the 420/600-second parent deadline. | All eight lanes still form one complete node contribution and global Q/T/owner floors remain possible. | If the node cannot produce 8/8 lanes by the deadline, its contribution is absent. On two nodes that makes `Q=1<2`; abort, drain, and exit from the prior checkpoint. |
| Model-free manager loss | Expire the old incarnation; restart at most twice; validate the latest handoff; use a new incarnation; sync before READY. | New manager becomes synchronized/READY within 180 seconds and all floors remain possible. The accepted run proved one such loss and a 30-second delayed READY. | Restart exhaustion, stale sync evidence, or floor collapse aborts the open attempt and exits. |
| Persistent native-service loss before any generation opens | Recreate and re-attest the local service only before model work/admission. | Exact provider/build/routes/bounds re-pass before model load. | Otherwise exit zero-work/nonzero according to whether the allocation lease was acquired. |
| Persistent native-service or node loss after generation open | No service restart in this profile; expire its READY incarnation and native routes. | At a future qualified larger rung only: Q/T stay met, at least two current owners remain, replay source exists, and reassignment/replay completes under the original deadline. | At two nodes, owner count falls from 2 to 1 and is unsafe. Abort with no partial publication, finish fenced handoff in 45 seconds, exit nonzero. |
| Owner/route loss before commit | At most two deterministic owner reassignments and `2*L` replay per sender; never reset transfer deadline. | A qualified plan still has at least two owners and all frozen sources. | Bound exhaustion, missing replay source, or `<2` owners aborts and exits. |
| Corrupt/nonfinite/stale/conflicting contribution | Reject without accumulator mutation; quarantine/expire its incarnation; identical replay gets the original receipt. | Frozen complete set still meets Q/T by the 600-second deadline. | Otherwise abort; no same-generation retry. |
| Q/T deadline miss | Do not freeze or publish. | Never for the expired attempt. | Preserve the prior commit, drain, and exit nonzero for later fresh-allocation recovery. |
| Checkpoint/publication failure | Applied trainer state is speculative; do not advance authoritative `latest`; reload `S_g`. | Only if publication succeeds inside the same current-fence deadline. | At 180/600-second expiry, abort and exit from `S_g`. |
| Lease renewal/current-fence loss | Immediately stop new admission, native finalize, checkpoint publication, and `latest` CAS. | Never under the stale allocation. | Abort/drain within 45 seconds; a later allocation must acquire a newer fence. |
| Throughput below the live health floor after a valid commit | Retain the valid commit and evidence. | Do not continue in the same allocation. | Drain cleanly and require a new qualification/diagnosis before another production generation. |
| Scheduler/allocation loss | No attempt to resurrect node-local or unfinished trainer state. | Never in the lost allocation. | Later job acquires a newer fence and reloads only the last immutable authoritative checkpoint. |

The supervisor exits the whole allocation when any of these becomes true:

- the current lease/fence is lost or cannot renew;
- the 600-second generation deadline expires without Q/T and complete native
  publication;
- the current or future-possible READY set is below `Q_min=2`, accepted tokens
  cannot reach `T_min=3,934,080`, or current owners fall below two;
- a required manager/trainer exhausts two restarts, or a post-open native
  service is lost;
- either of the two owner reassignments or `2*L` replay budget is exhausted;
- a memory, credit, digest, identity, provider, corruption, or prohibited-path
  invariant fails;
- checkpoint cadence/freshness cannot be maintained; or
- less than 645 seconds remains before walltime for a new steady generation.

The only non-error allocation exit is losing the initial exclusive lease: it
must exit successfully before model load and before any run-state write.

## Checkpoint cadence, freshness, and rollback

Every committed generation writes and independently reload/verifies one
immutable checkpoint containing model, required outer optimizer, inner state
selected by the E97 payload, accepted-token clock, membership, policy/config,
native digests, and fence. The checkpoint, manifest, and authoritative SQLite
`latest` advance atomically under the current fence. Maximum staleness is zero
committed generations; wall-clock freshness is not a substitute.

Rollback never edits or decrements `latest` and never reuses a fence:

1. Stop new generation admission.
2. Abort the open uncommitted attempt, cancel routes, release dense buffers,
   and drain within the 30/45-second bounds.
3. Leave the latest independently verified immutable checkpoint untouched.
4. Start a fresh allocation only through normal admission, with a strictly
   newer fence and the last production-qualified byte-identical payload.
5. Reload and recompute all checkpoint/manifest/control-store digests before
   READY or model mutation.

If there is no earlier production-qualified native payload, rollback means
**halt**. Python TCP, a test provider, fixed-world MPICH, an older checkpoint,
or a lowered Q/T/owner floor is not an automatic fallback. The compiled MPICH
reference may run only as a separately approved fixed-world workload with a
different run identity and policy; it cannot continue the elastic native run
by implication.

## Byte-identical validation-to-production promotion

Validation and production are rendered from the same canonical payload. The
normalized payload SHA-256 covers source commit, launcher/body, node/GPU/trainer
topology, model/data/seed/tokenizer digests, train config, K40, Q/T, owners,
deadlines, retry/replay limits, native bundle/artifacts/G2, provider/network,
buffer limits, checkpoint cadence, signals, environment, and all non-scheduler
arguments.

The only policy-bearing differences allowed are:

1. Slurm QoS/queue: `debug` -> `normal`; and
2. Slurm walltime: the reviewed validation limit -> the reviewed production
   limit.

The partition remains `batch`. Node count, GPUs, payload exports, code,
provider, config, and every policy value remain identical. Generated Slurm job
ID, allocation incarnation, unique run ID/directory, and the newly acquired
fence are runtime identities, not payload differences; their schemas are
fixed and they are recorded in the launch attestation. Job name/log paths may
encode that identity but may not change payload or policy.

Any additional diff—including node count, partition, environment, timeout,
checkpoint cadence, native artifact, seed, data, or failure setting—invalidates
promotion and requires a new validation rung. Production cannot dispatch until
the exact validation artifact is terminally accepted, committed/pushed, and
its normalized payload digest is registered in `production_qualified_rungs`.

## Validation and conformance

No production or Slurm job was submitted while defining this policy.

- [x] **R01–R16 and NDP01–NDP17:** READY leased membership, bounded waits,
  current fences, strict identity/staleness, exact weighted math, atomic
  checkpoints, native CXI, bounded owners/buffers/replay, failure handling,
  sequential scale, and rollback remain conformant with both authorities.
- [x] The Q/T, owner, deadline, throughput, buffer, retry, checkpoint, and
  failure thresholds cite retained two-node observations or existing v1 hard
  bounds; no observed failure widened a timeout.
- [x] Admission refuses Python TCP, non-`cxi`, missing/mismatched native
  digests, unqualified node counts, stale checkpoint/latest generations, and
  non-increasing resume fences.
- [x] Trainer/rank, manager, native-service, owner, node, lease, checkpoint,
  throughput, and whole-allocation exit behavior is explicit.
- [x] Promotion permits only QoS and walltime changes around a byte-identical
  payload; partition and node count do not change.
- [x] Rollback preserves the last immutable checkpoint, requires a newer
  fence, and has no implicit debug/fixed-world fallback.

The focused executable validation is:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_native_pool_production_policy.py
```

The checker can be invoked without submission side effects:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m ndm.native_pool_production_policy \
  --policy configs/frontier/native_resilient_pool_v1_production_policy.json \
  candidate.json
```

It prints an admission record only when every conjunctive gate passes. On the
checked-in policy it refuses every production candidate because the reviewed
production-qualified rung list is intentionally empty.
