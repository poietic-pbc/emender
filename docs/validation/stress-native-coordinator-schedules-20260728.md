# Native coordinator schedule stress gate

Date: 2026-07-28  
Task: `stress-native-coordinator-schedules`  
Status: **passed — zero native safety invariant failures**

## Authorities and scope

This gate applies, without weakening or substitution:

- [`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
  version 1, including its required compute-pool conformance checklist and
  bounded asynchronous overlap contract.
- [`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md),
  including every R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 row.
- [`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
  version 1.
- [`ASYNC_DECOUPLED_DILOCO_V2.md`](../ASYNC_DECOUPLED_DILOCO_V2.md),
  ADR-002, for the simple asynchronous v2.1 policy boundary.
- [`harden-native-coordination-kernel-20260728.md`](harden-native-coordination-kernel-20260728.md),
  the exact upstream hardened-kernel validation manifest.

This is a **native transition-safety stress gate only**. It does not load E97,
construct a trainer, start an external peer, touch a GPU, initialize
libfabric, request a Frontier allocation, or invoke Slurm. It does not replace
native binary64 numerical/reference evidence, dense memfd/fabric byte-path
evidence, immutable snapshot ownership, causal live timing, exact two-node
CXI qualification, scheduler `Partition`/`QOS` evidence, convergence, or any
scale rung. Those boundaries remain mandatory downstream.

## Result

The production-linked local explorer passed:

| Measure | Exact result |
|---|---:|
| Systematic schedules | 348 |
| Seeded randomized schedules | 50,000 |
| Permanent corpus schedules | 3 |
| Authoritative schedules per full campaign | 50,351 |
| Production native transitions per full campaign | 1,979,407 |
| Meaningful pair classes | 29 |
| Ordered pair cases | 58, exactly `AB` and `BA` for every class |
| Targeted three-race classes | 4 |
| Three-race permutations | 24, all six permutations per class |
| Restart role/phase cases | 32, all `4 roles × 8 phases` |
| Event kinds covered | 13/13 |
| Typed dispositions covered | 13/13 |
| Known-bad variants detected/minimized | 2/2 |
| Commits admitted under named progress predicates | 107 |
| Native safety invariant failures | **0** |

The final campaign transcript digest is
`a2b752a16328962eee1d2743cd67bbd867bda3cf7196ee63ab1f34a17b3a9fb3`.
The immutable passed manifest is
[`reports/frontier/native-coordination-stress-v1.json`](../../reports/frontier/native-coordination-stress-v1.json)
with SHA-256
`65e50db231a8ccdba484cb64828f7c83387a07a377ab32addb724c5614cfc646`.

## Actual production transition path

The explorer is
[`coordination_schedule_stress.cpp`](../../src/native_resilient_dataplane/tests/coordination_schedule_stress.cpp).
Its CMake target links `emender_ndp_service_core`, the same compiled service
core that owns the live manager state. Every native event, deterministic
repeat, replay, randomized schedule, corpus schedule, and shrink predicate is
evaluated by:

```text
emender_ndp::coordination::step(AuthorityState, Event) -> Transition
```

from
[`coordination_kernel.cpp`](../../src/native_resilient_dataplane/src/coordination_kernel.cpp).
The explorer contains no alternative coordination state machine or transition
policy. Its separate code is limited to generating causally meaningful
production events, invoking the production function, checking its returned
state/effects/trace, and retaining replay evidence.

The relevant production linkage is:

```text
ndp_coordination_schedule_stress
  -> emender_ndp_service_core
  -> coordination::step

live ndp_cxi_service
  -> LocalServiceCore::coordination_step
  -> coordination::step
  -> sole Service::coordination_state_ assignment
```

The stress source directly calls `coordination::step` for the schedule
transition, for an identical `(pre-state,event)` determinism comparison, and
for immediate accepted-event replay. `state_digest`, `invariant`, canonical
effects, and the kernel-produced trace are checked after every transition.

## Immutable identity chain

| Identity | Value |
|---|---|
| Hardened-kernel source commit | `e3243158b27ce6f54a4b6543199e40b87be691f8` |
| Hardened-kernel validation manifest SHA-256 | `ba5fb5052b7aa10840613d69b91f280e64ffa2a1a2f6c33beb0f4aa4d53a1f57` |
| Stress implementation source commit | `aafd509ba566ad2c4f3ea43adaf69ee543959f42` |
| Immutable stress-manifest commit | `f0e1828305ba84a6f4eb0e3070275c1cc918e299` |
| Stress source-bundle SHA-256 | `0f962e9523259c7834c293d394e5639305e2f0042dd3e8a14d2a91190ff3b457` |
| Exact native stress binary SHA-256 | `661f4ae778287341b63044345d3e0d74e3cf1f85061357fef77a902428e66535` |
| Coordination ABI header SHA-256 | `e02caec4a725a60d737cfebf464da9a86f8e245b81f9515fb11d56e8d6088c62` |
| ABI descriptor SHA-256 | `1b22884af9551fa0c54f463a46017bed9bcd02c73e45e40a4774dc0bd8165e0e` |
| Generator/schema SHA-256 | `3e16ce33f4d4546046531e6384351cfcd816d7ac9f05d5cecdbed4444ba2b562` |
| Permanent corpus SHA-256 | `a7f5f9d42a8d872952b07ac9bb9249465c661bdb8a903398c1a8fd7b5fe29791` |

The stress executable refuses to run if the exact hardened-kernel validation
manifest digest changes. The passed stress manifest binds the exact source
bundle, binary, public coordination ABI, derived ABI descriptor,
generator/schema, corpus, and upstream hardened manifest. It is therefore an
immutable native-first input for `scale-v21-direct-8n`, not a floating
`latest` reference.

The source branch and remote were equal at the immutable-manifest commit:

```text
f0e1828305ba84a6f4eb0e3070275c1cc918e299
f0e1828305ba84a6f4eb0e3070275c1cc918e299  refs/heads/wg/agent-1657/stress-native-coordinator-schedules
```

The final report commit and final remote-equality check are recorded in the WG
task log after this self-referential report is committed.

## Versioned generator

### PRNG

The PRNG identifier is `pcg-xsh-rr-64-32-v1`. It is specified in source and
bound into the generator schema digest:

```text
state := state * 6364136223846793005 + (stream | 1) mod 2^64
xorshifted := uint32(((old_state >> 18) xor old_state) >> 27)
rotation := old_state >> 59
output := rotr32(xorshifted, rotation)
```

Seeding is:

```text
state = 0
advance once
state = state + seed mod 2^64
advance once
```

Each random schedule is independently derived with versioned SplitMix64 from
`(base_seed, schedule_index)`, so replay does not depend on generating or
discarding earlier schedules. The four final seed partitions are:

| Seed | Schedules |
|---:|---:|
| `5105811` | 12,500 |
| `2611923443488327891` (`0x243f6a8885a308d3`) | 12,500 |
| `11400714819323198485` (`0x9e3779b97f4a7c15`) | 12,500 |
| `15111065706836454659` (`0xd1b54a32d192ed03`) | 12,500 |

### Grammar

Generator schema: `emender-native-coordination-generator-v1`.

```text
schedule ::=
  authority
  (recover-peer (ready | delayed-ready | expire)){2,4}
  open
  generation-event{0,32}

generation-event ::=
    recover-peer | ready | expire
  | open
  | contribution | identical-contribution | conflicting-contribution | drop
  | finite-close | deadline-close | close-probe
  | result-receipt | identical-result | conflicting-result
  | owner-loss | identical-owner-loss
  | commit | corrupt-commit | query-checkpoint
  | seven-trainer-apply | all-eight-apply | duplicate-apply
  | recover-node-apply
  | trainer-restart | manager-restart
  | service-restart | peer-control-restart
  | stale-fence | stale-incarnation | stale-generation
```

Opaque keys and SHA-256 digests stand in for node, incarnation, policy,
payload, result, receipt, checkpoint-manifest, and prior-receipt identities.
The only quantitative weight is an exact positive `uint64_t` token integer.
There are no model parameters, dense buffers, floats, endpoints, ranks, or
wall clocks in a generated event.

### Causal preconditions

The causal checker is deliberately distinct from semantic admissibility:
causal construction permits an early, late, stale, duplicate, or conflicting
delivery, while the production kernel must decide whether that delivery is
accepted, deferred, stale, closed, corrupt, conflicting, or fatal.

The versioned rules are:

1. Non-authority events require a prior nonzero run/fence/policy authority.
2. READY, expiry, apply, and contribution identities require a previously
   recovered stable-node/incarnation cause.
3. Open requires an attempt identity and snapshots the prior READY history.
4. A contribution requires a prior open plus a recovered peer cause. The
   production kernel, not the generator, decides whether that peer belongs to
   the immutable open cohort; this permits meaningful delayed-READY and stale
   incarnation deliveries.
5. Close and owner loss require the corresponding prior open. Result receipt
   requires a planned contribution identity.
6. Commit requires a planned prior contribution/result identity. It may be
   delivered before the final result receipt and must then defer.
7. Node apply requires a recovered peer cause. Seven-trainer and
   generation-zero reports are allowed as typed corrupt/invalid boundary
   tests but cannot grant authority.
8. A service or peer-control restart discards volatile READY/open/contribution
   causes and reconstructs only the exact durable commit identity through a
   production `RecoverAuthority` transition.

### Bounds

- Logical nodes: 2–4.
- Random generation events after bootstrap: at most 32.
- State: one active generation, at most 256 members/cohort entries/
  contributions/results/recovered applies.
- Effects: at most 8 per transition.
- Trace: less than 4,096 bytes per transition.
- Owner reassignments: at most 2.
- Trainer receipts required for node apply: exactly 8.
- Token/sample weights and clocks: exact unsigned 64-bit integers with
  checked addition.
- Random schedules: exactly 50,000 in the passed manifest.

## Systematic schedule coverage

Every pair class was executed in both `AB` and `BA` order. The 29 classes are:

1. `ready-expire`
2. `ready-open`
3. `open-expire`
4. `contribution-contribution`
5. `contribution-expire`
6. `contribution-finite-close`
7. `contribution-deadline-close`
8. `contribution-stale-fence`
9. `contribution-stale-incarnation`
10. `contribution-duplicate-conflict`
11. `duplicate-contribution-close`
12. `conflicting-contribution-close`
13. `close-expire`
14. `close-owner-loss`
15. `finite-close-deadline-close`
16. `result-result`
17. `result-owner-loss`
18. `result-duplicate-conflict`
19. `result-commit`
20. `commit-expire`
21. `commit-owner-loss`
22. `commit-recover`
23. `commit-query-checkpoint`
24. `apply-ready`
25. `apply-expire`
26. `apply-duplicate-receipt`
27. `recover-stale-incarnation`
28. `owner-loss-replay`
29. `recover-apply-recover-peer`

The four required targeted three-event races each ran all
`abc`, `acb`, `bac`, `bca`, `cab`, and `cba` permutations:

- close / participant failure / contribution;
- commit / participant failure / rejoin;
- owner loss / contribution replay / result receipt;
- all-eight apply / service restart / duplicate apply receipt.

The restart matrix covers each of trainer, manager, persistent native service,
and peer control at:

- recovered authority;
- READY membership;
- open generation;
- accepted contribution;
- finite closed generation;
- result publication/receipt;
- committed/checkpoint authority;
- node apply.

Targeted schedules additionally cover READY delay then expiry, a dropped
contribution at deadline, insufficient finite close, owner reassignment and
bounded replay exhaustion, current-commit conflict, invalid generation-zero
partial apply, stale closed work, checkpoint query, durable recovered node
apply, and next-generation rejoin.

## Transition and disposition coverage

Every production event kind and typed disposition was observed:

| Event | Count | Disposition | Count |
|---|---:|---|---:|
| recover-authority | 228,668 | accepted | 1,057,151 |
| recover-node-apply | 6 | identical-duplicate | 58,725 |
| recover-peer | 240,328 | conflicting-duplicate | 4,853 |
| ready | 214,682 | stale-fence | 88,608 |
| open-generation | 139,035 | stale-incarnation | 56,229 |
| contribution | 37,711 | stale-generation | 292,831 |
| close-generation | 24,978 | generation-closed | 2,445 |
| result-receipt | 6,594 | deferred | 335,324 |
| commit | 6,304 | retry-next-generation | 12,366 |
| node-apply | 82 | insufficient-cohort | 70,863 |
| expire-peer | 88,872 | corrupt | 6 |
| owner-lost | 24,894 | invalid-event | 3 |
| query-commit | 967,253 | fatal-invariant | 3 |

Exactly one kernel-produced `emit-trace` effect was observed for each of the
1,979,407 transitions. Accepted progress effects included 642 frozen cohorts,
185 owner reassignments, 132 commit-eligible reports, 107 published commits,
and 52 node-apply records.

`fatal-invariant` observations are expected fail-closed responses to
deliberately conflicting exact-once authority. They preserve the supplied
state digest and are not safety invariant failures.

## Safety oracle after every transition

Every transition checks:

1. **Unique commit.** A committed generation maps to one immutable
   `(token_clock, receipt, manifest, result)` identity across ordinary
   transitions and restart recovery.
2. **Stale noninterference.** Stale fence/incarnation/generation, closed,
   conflicting, corrupt, invalid, deferred, insufficient, identical replay,
   and fatal-conflict outcomes preserve the exact state digest as applicable.
3. **Idempotence.** Every accepted mutating event is immediately replayed; the
   replay must preserve the post-state and return identical duplicate
   (`QueryCommit` is the explicit accepted/read-only exception).
4. **Immutable cohort closure.** Once closed, cohort and accepted contribution
   maps cannot change while the generation/attempt identity remains active.
5. **No partial authority.** Seven-trainer apply and partial recovered apply
   are non-mutating; READY above generation zero requires live,
   synchronized, current-generation all-eight apply authority.
6. **Monotonic recovery.** Fence, committed generation, exact token clock,
   committed identity, and nonzero node applied generation never roll back
   across service/peer-control recovery.
7. **Bounded protocol state.** Member/cohort/contribution/result/recovery maps,
   effects, trace bytes, owner replay, and active-generation state remain
   within compiled constants.
8. **Deterministic state digests.** The returned pre/post digests equal fresh
   `state_digest` computation. A second production call on the identical
   pre-state/event must produce the same disposition, effects, trace bytes,
   and digests.
9. **Kernel invariant.** `coordination::invariant` succeeds for every returned
   authoritative state.

Progress is asserted only when all named predicates hold:

- `Q_min` complete contributions and `T_min` positive exact tokens;
- a named finite close or deadline;
- delivery of one agreeing result receipt for every frozen contributor;
- an explicitly scheduled commit fairness step.

The campaign does not claim progress under arbitrary message loss, absent
quorum, absent deadline/finite close, absent delivery, or absent scheduling
fairness.

## Shrinking, replay, and permanent corpus

Shrinker identifier:
`causal-ddmin-suffix-chunk-single-scalar-v1`.

The fixed shrink order is:

1. remove the largest prefix-preserving failing suffix;
2. delta-debug the largest contiguous chunks, left to right;
3. remove individual events newest to oldest to a fixed point;
4. simplify flags, sequence, exact-token, and other scalar fields in stable
   field order.

A candidate is accepted only if the versioned causal checker succeeds and a
fresh production replay reproduces the **same original predicate string**.
On a production failure the runner automatically writes:

- `regression-<predicate>-<digest>.schedule`, a canonical directly replayable
  schedule, into the permanent corpus directory;
- `regression-<predicate>-<digest>.native-trace.jsonl`, containing the exact
  minimized kernel traces;
- the exact `--replay-file` command.

The two built-in mutation operators prove the detector and shrinker:

- a stale-fence writer is detected as `stale-noninterference`;
- a seven-trainer authority grant is detected as `no-partial-authority`.

Both minimized witnesses remain permanent inputs:

- [`known-bad-stale-fence.schedule`](../../tests/corpus/native_coordination/known-bad-stale-fence.schedule);
- [`known-bad-partial-apply.schedule`](../../tests/corpus/native_coordination/known-bad-partial-apply.schedule).

They contain five total steps before and after shrinking, confirming they are
already causal fixed points. The required job 5105811 regression is permanent
at
[`job5105811.schedule`](../../tests/corpus/native_coordination/job5105811.schedule).
It replays the live three-peer ordering, closed-generation contribution,
nonfatal catch-up, seven-trainer rejection, all-eight apply, READY, and
generation-1 rejoin through the production kernel.

Example exact replays:

```bash
build/native-coordination-stress/ndp_coordination_schedule_stress \
  --source-root "$PWD" \
  --corpus-dir "$PWD/tests/corpus/native_coordination" \
  --replay-file \
    "$PWD/tests/corpus/native_coordination/job5105811.schedule"

build/native-coordination-stress/ndp_coordination_schedule_stress \
  --source-root "$PWD" \
  --corpus-dir "$PWD/tests/corpus/native_coordination" \
  --replay-seed 5105811 --replay-index 0
```

## Determinism and runtime

The final runner performed two full campaign repeats internally. The entire
runner was then invoked independently a second time with the exact same
arguments and output path. Thus four executions of the 50,351-schedule
campaign were completed for repeat evidence.

Both independent manifests were byte-identical:

```text
65e50db231a8ccdba484cb64828f7c83387a07a377ab32addb724c5614cfc646  first.json
65e50db231a8ccdba484cb64828f7c83387a07a377ab32addb724c5614cfc646  reports/frontier/native-coordination-stress-v1.json
```

There are no timestamps, process IDs, pointers, addresses, or wall-duration
fields in the authoritative manifest. `non_authoritative_timestamp_fields` is
the empty list. The manifest records deterministic logical runtime as
3,958,814 native transition evaluations (`1,979,407 × 2` internal repeats).

Measured wall runtimes, retained only in this validation report so the
authoritative manifest stays byte-identical, were:

| Run | Real | User | System |
|---|---:|---:|---:|
| Final full runner 1 | 236.17 s | 227.15 s | 10.27 s |
| Final full runner 2 | 225.47 s | 211.41 s | 9.20 s |

The focused Debug CTest stress scenario completed in 44.76 seconds. A separate
10,351-schedule ASan/UBSan campaign completed with the same
`18e4b26a1508828b5dc58bbe0ac262cb6a0ed60ff913615cb92f383ff28cd83c`
campaign digest and no sanitizer diagnostic.

## Compute-pool conformance checklist

| Required checklist obligation | Applied evidence and retained boundary |
|---|---|
| Cite the design authority and complete requirement sets | This report cites the compute-pool authority, native specialization, ADR-002, gap matrix, and exact hardened-kernel manifest. Every R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 row is mapped below. |
| Peer-owned READY membership, bounded waits, and no launched-rank/all-rank invariant | Recover/READY/delay/expiry/rejoin events and open snapshots are stressed for 2–4 logical nodes. The event/state ABI has no launched-rank or collective input. Deadline/finite-close decisions are total and nonblocking. |
| Rendered compute-role closure has no SQLite/database/lock/metadata heartbeat | The stress target constructs only local in-memory native state and file-backed input/output evidence. Its source contains no SQLite/database/lock/membership-heartbeat implementation. The exact upstream hardened manifest retains the full production compute-role audit; stress does not replace it. |
| Fenced identity, deterministic weighted math, idempotence, stale/corrupt rejection, atomic committed evidence | Every event is run/fence/policy/generation/attempt/node/incarnation/sequence fenced. Exact tokens are the sole kernel quantity. All typed dispositions, duplicate/conflict classes, unique commit, prior-receipt lineage, deterministic digests, and all-eight apply are checked. Numerical binary64 reduction remains external. |
| Bounded non-Lustre hot-path transport, backpressure/release, no central broker | The transition kernel has no transport or dense bytes and exposes fixed state/effect/trace bounds. Native memfd/CXI replay/release evidence remains NDP02–NDP12/NDP17 boundary evidence and is not inferred from this gate. |
| Applicable failure/deadline/recovery path and minimum progress floor | READY expiry, participant/owner loss, two owner reassignments, contribution drop, finite/deadline close, insufficient Q/T, trainer/manager/service/peer-control restart, stale fence/incarnation, durable recovery, apply, and rejoin are exercised. The stress profile uses explicit positive Q/T; commit progress requires the named predicates above. |
| Exact commands and committed artifacts; prior rung for scale | Commands and immutable manifest are below. This is a prerequisite stress input only. It supplies no two-node or scale authorization and cannot satisfy a prior rung. |
| Bounded async V21S/ISP timing/ownership evidence | All V21S01–V21S17 and ISP01–ISP07 are cited below. Transition traces prove only coordination state/effect determinism. Snapshot coherence, background live-state exclusion, per-phase max/p99, zero foreground result wait, apply pause, and approximately-200-second tail rejection remain explicit external gates. |

## Requirement crosswalk

“Direct” means the production transition was stressed here. “Boundary
retained” means this gate binds or preserves the requirement but deliberately
does not claim the required numerical, byte-path, ownership, timing,
scheduler, two-node, Frontier, or scale evidence.

### R01–R16

| ID | Stress application |
|---|---|
| R01 | **Direct/boundary:** every event is run/fence bound; stale fence is non-mutating. The scheduler claim and no-database compute-role audit remain bound through the exact hardened manifest. |
| R02 | **Direct:** recover, delayed READY, READY replay, expiry, higher incarnation/sequence, and rejoin are scheduled across 2–4 logical nodes. |
| R03 | **Direct:** open snapshots only native READY state; ready/open and open/expiry run in both orders. No launched-rank field exists. |
| R04 | **Direct:** complete generation/contribution identity, one stable-node contribution, identical duplicate, conflict, stale incarnation/fence/generation, and closed work are covered. |
| R05 | **Direct/boundary:** exact positive token integers alone drive Q/T and token clocks. Binary64 weighted reduction/reference parity is not a transition property and remains required. |
| R06 | **Direct:** positive Q/T, finite close, deadline close, close probe, insufficient cohort, dropped contribution, abort/retry, and bounded attempt behavior are covered. |
| R07 | **Direct/boundary:** unique exactly-once commit, previous receipt, manifest, result, and token lineage are asserted. Durable checkpoint publication/CAS remains external. |
| R08 | **Direct/boundary:** frozen cohort identities, owner loss/replay, no more than two reassignments, and result reset are stressed. Dense chunks, credits, checksums, and release remain native transport evidence. |
| R09 | **Boundary retained:** the target is model-free and has only opaque identities/tokens. Trainer snapshot coherence and overlap are not claimed. |
| R10 | **Boundary retained:** the production transition has no filesystem/network/database API. The harness writes only corpus/manifest/failure evidence. Full rendered compute closure remains upstream evidence. |
| R11 | **Direct:** participant loss, new-incarnation catch-up, all role restarts at every phase, owner retry, old closed work, job 5105811, apply, and next-generation rejoin are covered. |
| R12 | **Direct/boundary:** fence/generation/token/result/receipt/applied clocks cannot roll back across recovery. Model/outer checkpoint restoration remains immutable-manifest evidence. |
| R13 | **Boundary retained:** events are backend-neutral opaque metadata. No Frontier backend or scheduler is invoked. |
| R14 | **Direct/boundary:** deadline/defer/close outcomes and canonical traces are deterministic. Human-time phase max/p99 and tail evidence remain external. |
| R15 | **Direct/boundary:** exact accepted tokens alone advance the commit clock while explicit membership changes. Numerical parity remains external. |
| R16 | **Boundary retained:** the kernel has no node-count promotion decision. This stress pass authorizes no two-node or scale rung. |

### NDP01–NDP17

| ID | Stress application |
|---|---|
| NDP01 | **Direct:** the executable links the production persistent-service core and invokes its sole pure `coordination::step`; no substitute policy exists. |
| NDP02 | **Boundary retained:** the event path is metadata-only and has no collective. Point-to-point fabric/no-MPI proof remains native service evidence. |
| NDP03 | **Boundary retained:** the same compiled service core is linked. Exact `cxi` provider evidence is not exercised locally. |
| NDP04 | **Boundary retained:** only opaque keys/digests/tokens cross this test boundary. Producer snapshot coherence remains ISP01/ISP03 evidence. |
| NDP05 | **Boundary retained:** exact tokens are stressed; deterministic binary64 owner-local reduction remains numerical evidence. |
| NDP06 | **Direct:** ABI version/constants/record sizes are bound into the ABI descriptor digest; every event is fenced and receipt identities are replayed/conflicted. |
| NDP07 | **Boundary retained:** READY identity/cache effects are exercised as data. Endpoint/AV installation remains transport evidence. |
| NDP08 | **Direct/boundary:** fixed members, effects, trace, generation, receipt, and replay bounds are asserted. Dense registered-slot exhaustion remains external. |
| NDP09 | **Direct/boundary:** effects never mutate authority without a later event. Fabric credit and foreground nonblocking behavior remain ISP02/ISP04 evidence. |
| NDP10 | **Direct/boundary:** receipts are idempotent and conflicts fail closed. Dense CRC/SHA/nonfinite validation remains native transport/reducer evidence. |
| NDP11 | **Direct/boundary:** owner loss is replayed and capped at two reassignments before retry/abort. Dense byte replay/NVMe remains external. |
| NDP12 | **Direct/boundary:** every frozen contributor must report the identical result before commit. Owner-direct dense redistribution remains external. |
| NDP13 | **Direct/boundary:** delayed/late/deadline/closed deliveries are total, typed, and local. Zero foreground wait remains live timing evidence. |
| NDP14 | **Direct:** stable public coordination ABI, bounded metadata records, compiled service linkage, and canonical result/trace bounds are checked. |
| NDP15 | **Direct/boundary:** commit requires complete result agreement; seven trainers never grant apply/READY; all eight do. Atomic trainer swap/checkpoint I/O remains external. |
| NDP16 | **Direct/boundary:** every transition carries complete event/effect/state digests and a canonical trace. Live phase/max/p99 evidence remains ISP06/ISP07. |
| NDP17 | **Boundary retained:** all stress work is local and allocation-free. No exact two-node CXI or scale promotion is claimed. |

### V21S01–V21S17

| ID | Stress application |
|---|---|
| V21S01 | **Direct/boundary:** kernel, trace, manifest, generator, schedule, PRNG, ABI, policy, source, binary, and corpus identities are versioned/digested. Historical policy rejection remains ingress evidence. |
| V21S02 | **Direct/boundary:** commit/applied generation clocks are monotonic; stale/defer/drop/restart outcomes are stressed. Full lag/speculative foreground behavior remains async runtime evidence. |
| V21S03 | **Direct:** positive exact tokens are the sole Q/T and cumulative token quantity; no aggregation-weight field exists. |
| V21S04 | **Boundary retained:** commit advances only generation/token/result authority. K40, eta-one, and outer numerical behavior remain trainer/reference evidence. |
| V21S05 | **Direct/boundary:** stable node/incarnation/sequence/generation/attempt/token/payload identities and one contribution per stable node are stressed. Dense layout/code/window ingress remains external. |
| V21S06 | **Boundary retained:** the model-free harness cannot read trainer state. Coherent immutable capture and bounded admission pause are not proven here. |
| V21S07 | **Direct/boundary:** result/commit/apply are once-only; READY requires current all-eight apply. Atomic `x/z/interval` translation and 60-second bound remain external. |
| V21S08 | **Direct/boundary:** stale/conflicting/corrupt result and receipt deliveries are non-mutating/fail-closed. Capacity-one mailbox behavior remains external. |
| V21S09 | **Direct/boundary:** one active generation, fixed state/effect/receipt/replay bounds, and explicit defer/drop/retry outcomes are asserted. Dense cohort/mailbox capacity remains ISP04. |
| V21S10 | **Direct:** leased READY snapshot, expiry, higher incarnation recovery, old-work rejection, Q floor, and nonblocking typed missing-peer outcomes are covered. |
| V21S11 | **Direct/boundary:** seven-trainer reports never authorize READY; current all-eight apply does. Actual all-eight trainer atomicity and timing remain ISP05. |
| V21S12 | **Boundary retained:** the production model-free compiled service core is used. Persistent memfd/libfabric/no-MPI data movement remains native evidence. |
| V21S13 | **Direct/boundary:** complete canonical causal transition traces and byte-deterministic repeats are retained. Live freeze/admission/network/aggregation/checkpoint/result-wait/apply/idle max/p99 evidence remains external. |
| V21S14 | **Direct/boundary:** higher-fence recovery preserves exact commit/result/token/apply authority. Durable E97 checkpoint/seed/sbcast evidence remains external. |
| V21S15 | **Boundary retained:** no two-node job, scheduler queue, or qualification gate was attempted. |
| V21S16 | **Boundary retained:** no review authorization or `4 -> 8` rung pass is claimed. |
| V21S17 | **Direct/boundary:** finite close races cover every causally delivered pre-close contribution admitted by the immutable READY cohort and never inspect launched ranks. Evidence-derived live close arithmetic remains controller evidence. |

### ISP01–ISP07

| ID | Stress application |
|---|---|
| ISP01 | **Boundary retained:** the kernel and stress input contain no model/optimizer/live trainer object. Coherent capture and negative background-access evidence are not claimed. |
| ISP02 | **Direct/boundary:** coordination transitions are total and contain no wait; delayed READY/result/failure/deadline outcomes are typed. The 1-second admission bound and advancing K window require live evidence. |
| ISP03 | **Boundary retained:** only immutable opaque digests enter transition events. Background immutable snapshot/checkpoint ownership is not established by metadata traces. |
| ISP04 | **Direct/boundary:** fixed protocol bounds and defer/drop/retry behavior are asserted. Snapshot slots, fabric credits, dense replay, and mailbox exhaustion with advancing foreground work remain external. |
| ISP05 | **Direct/boundary:** partial/unready/late/invalid apply cannot grant READY; apply/restart/duplicate receipt is permuted. Atomic trainer-visible swap and 60-second pause remain external. |
| ISP06 | **Direct/boundary:** every transition has deterministic event/effect/state evidence. Required live phase identities, every-event max/p99, and zero foreground result-wait timing remain unclaimed. |
| ISP07 | **Boundary retained:** raw transition schedules cannot hide a coordination event and repeated bytes are exact. Checkpoint/restart alone does not satisfy the approximately-200-second tail-stall rejection gate. |

## Exact validation commands

Every native build/test command was run only after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

No bare system Python was used. The JSON assertions used
`"$EMENDER_PYTHON"`. No Slurm command was called.

Focused Debug build and CTest:

```bash
cmake -S src/native_resilient_dataplane \
  -B build/coord-stress-agent1657 \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/coord-stress-agent1657 --parallel 4
ctest --test-dir build/coord-stress-agent1657 --output-on-failure \
  -R 'ndp_coordination_(kernel|schedule_stress)_test'
```

Result:

```text
ndp_coordination_kernel_test ............ Passed
ndp_coordination_schedule_stress_test ... Passed
100% tests passed, 0 tests failed out of 2
```

ASan/UBSan campaign:

```bash
cmake -S src/native_resilient_dataplane \
  -B build/coord-stress-agent1657-sanitize \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug \
  -DNDP_ENABLE_SANITIZERS=ON
cmake --build build/coord-stress-agent1657-sanitize \
  --target ndp_coordination_schedule_stress --parallel 4
build/coord-stress-agent1657-sanitize/ndp_coordination_schedule_stress \
  --source-root . \
  --corpus-dir tests/corpus/native_coordination \
  --random-schedules 10000 \
  --maximum-events 32 \
  --determinism-repeats 1 \
  --no-output
```

Result:

```text
systematic=348 random=10000 corpus=3 total=10351
transitions=399056 pairs=29 ordered_pairs=58 three_races=24
digest=18e4b26a1508828b5dc58bbe0ac262cb6a0ed60ff913615cb92f383ff28cd83c
```

Final full runner command, executed twice unchanged:

```bash
/usr/bin/time -p env \
  REPO="$PWD" \
  BUILD_DIR="$PWD/build/native-coordination-stress" \
  OUTPUT="$PWD/reports/frontier/native-coordination-stress-v1.json" \
  RANDOM_SCHEDULES=50000 \
  MAXIMUM_EVENTS=32 \
  DETERMINISM_REPEATS=2 \
  BUILD_JOBS=4 \
  scripts/frontier/run_native_coordination_stress.sh
```

Byte comparison and manifest validation:

```bash
cmp /tmp/agent1657-native-coordination-stress-first.json \
  reports/frontier/native-coordination-stress-v1.json

"$EMENDER_PYTHON" - <<'PY'
import json
from pathlib import Path
d = json.loads(
    Path("reports/frontier/native-coordination-stress-v1.json").read_text())
assert d["status"] == "passed"
assert d["counts"]["total_schedules"] == 50351
assert d["counts"]["native_transitions"] == 1979407
assert d["counts"]["native_safety_failures"] == 0
assert d["counts"]["ordered_pair_cases"] == 58
assert d["counts"]["three_race_permutations"] == 24
assert d["counts"]["known_bad_detected"] == 2
assert d["counts"]["known_bad_minimized"] == 2
assert all(d["coverage"]["event_kinds"].values())
assert all(d["coverage"]["dispositions"].values())
PY

sha256sum \
  /tmp/agent1657-native-coordination-stress-first.json \
  reports/frontier/native-coordination-stress-v1.json
```

Zero-Slurm source evidence:

```bash
if rg -n '\b(sbatch|srun|salloc|squeue|sacct|scancel)\b' \
  src/native_resilient_dataplane/tests/coordination_schedule_stress.cpp \
  scripts/frontier/run_native_coordination_stress.sh \
  src/native_resilient_dataplane/CMakeLists.txt \
  tests/corpus/native_coordination
then
  exit 1
else
  echo "zero Slurm command tokens in stress implementation, runner, CMake registration, and corpus"
fi
```

Result:

```text
zero Slurm command tokens in stress implementation, runner, CMake registration, and corpus
```

## Validation

- [x] Cited and applied the compute-pool checklist and every R01–R16,
  NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 row, while retaining explicit
  numerical, byte-path, ownership, timing, two-node, Frontier, and scale
  nonclaims.
- [x] Exercised the hardened actual production native transition function
  with deterministic seeds; no test-only coordination implementation exists.
- [x] Covered all 29 declared meaningful pair classes in both orders and all
  six permutations of the four required three-event races.
- [x] Asserted unique commit, stale noninterference, idempotence, immutable
  cohort closure, no partial authority, monotonic recovery, bounded protocol
  state, native invariants, and deterministic state/trace bytes after every
  transition. Progress is asserted only under named Q/T, close/deadline,
  delivery, and fairness predicates.
- [x] Executed 50,351 bounded schedules and 1,979,407 production transitions
  per full campaign with exact systematic/random/corpus counts, four seeds,
  2–4-node/32-event bounds, full event/disposition coverage, measured runtime,
  and zero native safety failures.
- [x] Detected and minimized both known-bad variants with causal and
  predicate-preserving shrinking; retained job 5105811 and both minimized
  mutation witnesses permanently.
- [x] Repeated the complete campaign twice internally and twice independently;
  complete passed manifests were byte-identical, with directly replayable
  canonical native failure/corpus traces.
- [x] Produced an immutable passed stress manifest bound to the exact
  hardened-kernel manifest, source commit, native binary, ABI, schema, corpus,
  and transcript digest for `scale-v21-direct-8n`.
- [x] Recorded exact commands, zero-Slurm evidence, source/manifest commit
  SHAs, pushed remote equality, and zero native safety failures.
