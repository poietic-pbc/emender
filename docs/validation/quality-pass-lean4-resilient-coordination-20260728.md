# Lean 4 resilient coordination workstream quality pass

**Task:** `.quality-pass-lean4-resilient-coordination`  
**Date:** 2026-07-28  
**Scope:** WG task metadata and dependency edges only. No implementation file
was changed and no Slurm job was submitted.

## Verdict

The reviewed architecture is executable after the metadata corrections recorded
below:

- Lean 4 is the authoritative executable model of the pure resilient
  coordination transition kernel and the machine-checked safety oracle.
- The production persistent compiled native service remains responsible for
  networking, timers, buffer ownership, process supervision, and runtime
  effects.
- A versioned trace adapter must instrument the actual production transition
  path. Reimplementing native decisions in a test-only mock is not conformance.
- Local CI and deterministic chaos testing compare every native event
  disposition and authoritative post-state digest with the Lean runner.
- All five tasks are local/no-Slurm work. The formal evidence is an additional
  fail-closed prerequisite of the already-defined 8-node systems probe; it is
  not Frontier qualification and it authorizes no job by itself.

The original graph had two blocking review defects:

1. `conform-native-coordinator-to-lean4` did not depend on the Lean model it
   was meant to conform to.
2. `stress-native-coordinator-schedules` and
   `integrate-formal-coordination-scale-gate` could complete without the proof
   task.

Both are corrected by the final edge list below.

## Authority anchors and conformance checklist

The design-authority chain is explicit in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:3-23`. The live authority belongs to
leased native peer control rather than launched ranks or a shared database
(`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:25-45`), while the lifecycle,
generation identity, exact accepted set, idempotence, commit, all-eight apply,
and stale-input rules are at
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:100-154`.

Every task must apply the conformance checklist at
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:276-297`. In particular, the work must
retain:

- leased READY membership and finite deadlines, with no launched-rank or
  all-rank completion invariant;
- fenced allocation/generation/attempt/worker/incarnation/sequence identity;
- deterministic accepted-set and exact-token accounting, idempotent identical
  replay, conflicting-duplicate/stale/corrupt rejection, and one atomic
  authoritative commit or none;
- bounded point-to-point runtime behavior, prompt release, no shared SQLite
  control path, no Python/Lustre dense hot path, and no central full-model
  broker;
- explicit minimum-progress floors, failure/deadline/recovery outcomes, and
  exact commands and immutable evidence;
- for v2.1, all four lag clocks and the per-phase/tail evidence required by
  ISP01-ISP07. Formal coordination evidence does not replace the existing
  snapshot, transport, numerical, performance, or two-node evidence.

The four namespaces are independently normative. Their definitions begin at
`docs/RESILIENT_DILOCO_GAP_MATRIX.md:18` (R01-R16), line 39 (NDP01-NDP17),
line 67 (V21S01-V21S17), and line 95 (ISP01-ISP07). ADR-002 requires every
implementation/runner/scale task to cite all four namespaces
(`docs/ASYNC_DECOUPLED_DILOCO_V2.md:425-435`). The native authority makes the
same requirement at
`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md:1166-1192`.

## Requirement allocation

Abbreviations:

- **B** — `bootstrap-lean4-resilient-protocol`
- **P** — `prove-lean4-resilient-protocol-safety`
- **N** — `conform-native-coordinator-to-lean4`
- **S** — `stress-native-coordinator-schedules`
- **G** — `integrate-formal-coordination-scale-gate`

“Boundary” means Lean records or reasons about the coordination-visible
contract, but the native/local/controller evidence remains the acceptance
authority for physical transport, memory, snapshot, numerical, or scheduling
behavior.

### R01-R16

| ID | Primary task allocation | Required batch evidence |
|---|---|---|
| R01 | B, P, N, S, G | Monotonic allocation fence/claim identity; old fences are total non-mutating dispositions; exact formal evidence is gate-bound. |
| R02 | B, P, N, S, G | DISCOVER/BOOT/SYNC/READY/DRAIN/EXPIRE and stable-worker/new-incarnation lifecycle. |
| R03 | B, P, N, S, G | Generation eligibility and closure use leased READY membership, never launched ranks. |
| R04 | B, P, N, S, G | Full contribution identity, identical replay idempotence, and stale/conflicting/corrupt rejection. |
| R05 | B, P, N, S, G | Lean carries exact positive tokens, deterministic admitted order/result identity, and token-clock rules; native/reference evidence remains authoritative for floating-point bytes. |
| R06 | B, P, N, S, G | Explicit quorum/token floors and finite close/deadline; progress theorems name quorum and fairness assumptions. |
| R07 | B, P, N, S, G | Exact-once commit, digest-linked receipt lineage, and one authoritative result or none. |
| R08 | B, P, N, S, G | Coordination-visible owner epochs, bounded replay/reassignment and no-partial-publication; physical byte/buffer proof is native evidence. |
| R09 | N, S, G (B/P boundary) | Lean assumes immutable coordination payloads and never claims model ownership; native evidence proves model-free service and exclusive trainer live-state ownership. |
| R10 | N, S, G | Conformance/gate manifests preserve no-SQLite, non-Lustre hot path and no Python dense transport. |
| R11 | B, P, N, S, G | Loss, expiry, catch-up, new-incarnation rejoin and next-admissible-generation rules. |
| R12 | B, P, N, S, G | Monotonic restart authority, exact base receipt/token clock, and no rollback on fresh allocation. |
| R13 | B, N, S, G | Pure model is backend-neutral; native adapter tests point-to-point runtime without collective assumptions. |
| R14 | B, P, N, S, G | Every transition/deadline has canonical trace identity and invariant verdict; gate binds causal telemetry separately. |
| R15 | B, P, N, S, G | Exact accepted-token accounting and deterministic result identity; native/reference tests retain numerical authority. |
| R16 | G (S prerequisite) | Formal/local evidence is fail-closed input to the policy-controlled 8-node probe and never a scale authorization by itself. |

### NDP01-NDP17

| ID | Primary task allocation | Required batch evidence |
|---|---|---|
| NDP01 | B, P, N, S, G | Lean models peer-control authority; native trace comes from the live production transition path; no shared control database. |
| NDP02 | N, S, G (B boundary) | Local conformance preserves bounded point-to-point semantics and records source/symbol evidence rejecting collective control. |
| NDP03 | N, S, G | Native service/provider facts are evidence inputs; Lean does not model libfabric. |
| NDP04 | N, S, G | Trace boundary admits only coherent immutable descriptors; ISP01/native tests prove physical ownership and no live-state reads. |
| NDP05 | N, S, G (B/P identity) | Lean fixes admitted order/token/result-root semantics; native/reference tests prove exact arithmetic bytes. |
| NDP06 | B, P, N, S, G | Every event/receipt/replay/result is fenced and identity-complete. |
| NDP07 | B, N, S, G | Endpoint eligibility is derived only from current leased READY records. |
| NDP08 | B, N, S, G | Bounded capacity is coordination-visible; native stress proves fixed pools and fail-closed admission. |
| NDP09 | B, N, S, G | Credit/receipt distinction and nonblocking exhaustion dispositions are traced and stressed. |
| NDP10 | B, P, N, S, G | Checksummed identity, exactly-once application, idempotent receipt and corrupt/nonfinite rejection. |
| NDP11 | B, P, N, S, G | Replay/reassignment is bounded by owner epoch/deadline and cannot create a partial commit. |
| NDP12 | B, P, N, S, G | Only a complete authoritative result reaches apply authority; physical redistribution remains native evidence. |
| NDP13 | B, P, N, S, G | Absolute deadlines produce typed abort/defer/retry outcomes and never hidden unconditional liveness. |
| NDP14 | N, S, G | Versioned native ABI/trace adapter identity is digest-bound; dense bytes never cross the trace control schema. |
| NDP15 | B, P, N, S, G | Atomic commit/checkpoint/apply state, all-eight receipt reduction, and restart after partial apply. |
| NDP16 | B, N, S, G | Canonical traces and manifests bind identities, transitions, bounds and terminal reasons; runtime phase evidence remains separate. |
| NDP17 | G | Existing exact-source native/two-node evidence remains a prerequisite; formal local work neither submits nor substitutes for it. |

### V21S01-V21S17

| ID | Primary task allocation | Required batch evidence |
|---|---|---|
| V21S01 | B, P, N, S, G | Pinned policy/schema/trace/toolchain identities; v1/v2.0/unknown identities fail before mutation. |
| V21S02 | B, P, N, S, G | Commit, applied-anchor, result-version and speculative-window clocks remain distinct; lag-three outcomes are typed nonblocking drop/defer paths. |
| V21S03 | B, P, N, S, G | Positive exact tokens are the sole quorum/clock/weight identity; native reducer proves bytes. |
| V21S04 | B, P, N, S, G | K40/eta-one/outer-step and accepted-token transition identity is modeled and differentially checked. |
| V21S05 | B, P, N, S, G | Full contribution identity, exact two-node floors/deadlines, and at most one stable-worker contribution per transition. |
| V21S06 | N, S, G (B boundary) | Lean begins at immutable admission/OWNED; native/ISP evidence proves coherent capture, exclusive live ownership and immediate resume. |
| V21S07 | B, P, N, S, G | Once-only complete result apply at a safe boundary; absent/late/invalid/unready is non-mutating defer; physical x/z translation remains native evidence. |
| V21S08 | B, P, N, S, G | Capacity-one verified-latest mailbox identity, replacement/idempotence/conflict rules and no foreground wait. |
| V21S09 | B, N, S, G | Capacity edges are explicit skip/replace/defer transitions; native tests prove byte formulas and no third dense cohort. |
| V21S10 | B, P, N, S, G | Leased READY expiry/rejoin, no old-incarnation mutation and no one-node authority. |
| V21S11 | B, P, N, S, G | All-eight apply/recovery transaction: no READY after partial/timed-out apply; fenced all-lane restart. |
| V21S12 | N, S, G | Native service/point-to-point/no-MPI/no-broker evidence is digest-bound, not claimed as a Lean theorem. |
| V21S13 | B, N, S, G | Trace schema carries causal phase identities and lag/bound facts; native/controller validators own real timing evidence. |
| V21S14 | B, P, N, S, G | Fenced immutable result/receipt/restart chain and exact restored clocks; external seed/staging evidence remains gate-bound. |
| V21S15 | G (S local input) | Formal build, corpus conformance and deterministic stress add local evidence but do not replace the exact two-node gates. |
| V21S16 | G | Policy task plus all exact formal/native/chaos manifests must pass before the already-defined immediate 8-node rung. |
| V21S17 | B, P, N, S, G | Finite leased-READY closure is modeled/traced and checked against policy evidence; it never closes from launched ranks or merely Q_min arrival. |

### ISP01-ISP07

| ID | Primary task allocation | Required batch evidence |
|---|---|---|
| ISP01 | N, S, G | Adapter accepts only sealed immutable work; native/source/race evidence proves coherent capture and no background live-state read. |
| ISP02 | B, N, S, G | Coordination events have nonblocking OWNED/defer outcomes; native causal timing proves immediate next-K progress. |
| ISP03 | N, S, G | Trace/manifests bind immutable snapshot/result identities; native tests prove hashing/checkpoint work never reads mutable foreground state. |
| ISP04 | B, N, S, G | Capacity exhaustion is a total skip/replace/drop/defer disposition; local stress covers every bounded queue/credit/mailbox edge. |
| ISP05 | B, P, N, S, G | Atomic all-eight apply is a kernel safety property; native faults prove bounded visibility and all-lane recovery. |
| ISP06 | B, N, S, G | Trace schema carries causal phase IDs; controller/native validators prove complete intervals, zero foreground result wait, max/p99 and bounds. |
| ISP07 | S, G | Deterministic local corpus includes tail/adversarial schedules; gate still consumes the independent approximately-200-second-stall validator and live raw timing evidence. |

All 57 IDs are therefore allocated at least once, and no physical-runtime
obligation is incorrectly discharged by a theorem about the pure kernel.

## Transition and proof coverage required by the batch

The executable kernel and its tests/proofs must cover:

1. allocation fence and generation/attempt authority;
2. DISCOVER/BOOT/SYNC/leased-READY/DRAIN/EXPIRE and new incarnations;
3. deterministic finite closure over the eligible leased READY snapshot;
4. contribution admission, exact-token floors, accepted-set immutability and
   owner/replay epochs;
5. accepted, identical-duplicate, conflicting-duplicate, stale-fence,
   stale-incarnation, late, generation-closed, corrupt, deferred, abort and
   retry-next-generation dispositions;
6. exact-once commit receipts, one authoritative result or none, and atomic
   all-eight apply/READY authority;
7. trainer/manager/native owner/peer-control/whole-allocation loss, restart,
   catch-up and rejoin without rollback;
8. the permanent job-5105811 ordering: generation closes/commits, node 0
   fails/reincarnates, node 1 submits to the closed generation, receives a
   typed nonfatal non-mutating recovery disposition, preserves restart budget,
   and later joins the next admissible generation; and
9. bounded progress only under explicitly named finite-close/deadline,
   surviving-quorum and fair-delivery/scheduling assumptions.

Safety must be stated for every transition from a well-formed state without
fairness assumptions wherever possible. Liveness/progress statements must
never hide participant-survival, quorum, deadline, delivery, or scheduler
fairness hypotheses.

## Differential and chaos acceptance

The canonical trace schema must include the schema/policy/toolchain/source
identity, pre-state digest and authoritative identities, event, disposition,
post-state digest and authoritative identities, invariant verdict, and
causality/replay metadata. Malformed, identity-incomplete, ambiguous, or
forbidden-reordered traces fail closed.

Native differential execution must:

- instrument the actual state mutation path used by production;
- translate only at the trace boundary and never copy native admission,
  closure, receipt, commit, apply, or recovery decisions into a mock;
- compare every event disposition and authoritative post-state digest with the
  Lean runner; and
- retain job 5105811 plus every later minimized divergence as permanent corpus
  entries.

Schedule stress must use a specified deterministic PRNG/version and record
seed, generator configuration, execution-source/toolchain/schema digests and
the exact replay command. Systematic pair orders and targeted three-event
races are supplemented by seeded randomized schedules. A failing schedule is
automatically shrunk while preserving causal preconditions and the failure
predicate; the minimal Lean/native trace is directly replayable. All of this
runs locally with opaque small payload digests—no GPU, Frontier allocation,
external peer service, or Slurm command.

## Final reviewed edges

Primary workstream:

```text
.quality-pass-lean4-resilient-coordination
  -> bootstrap-lean4-resilient-protocol
  -> [prove-lean4-resilient-protocol-safety,
      conform-native-coordinator-to-lean4]
  -> stress-native-coordinator-schedules
  -> integrate-formal-coordination-scale-gate
  -> scale-v21-direct-8n
```

Explicit join edges:

```text
stress-native-coordinator-schedules
  after prove-lean4-resilient-protocol-safety
  after conform-native-coordinator-to-lean4

integrate-formal-coordination-scale-gate
  after prove-lean4-resilient-protocol-safety
  after conform-native-coordinator-to-lean4
  after stress-native-coordinator-schedules
  after codify-v21-direct-scale-policy

scale-v21-direct-8n
  after integrate-formal-coordination-scale-gate
```

The pre-existing direct-scale dependencies remain intact:
`codify-v21-direct-scale-policy` follows the exact-source two-node
fault/restart chain, and `scale-v21-direct-8n` also directly requires that
fault/restart task and the policy task. Assignment/evaluation bookkeeping edges
are omitted from this human-readable list but are not removed.

## Gate manifest contract

`integrate-formal-coordination-scale-gate` must fail closed unless one immutable
manifest binds:

- exact execution source and the completed scale-policy/controller manifest;
- pinned Lean toolchain/Lake dependency lock and executable model digest;
- theorem/proof manifest with no `sorry`, `admit`, unsafe axiom or boolean
  substitute;
- native production-transition source/binary digest and versioned trace adapter
  digest;
- canonical trace schema and permanent regression corpus digests, including
  job 5105811;
- zero-divergence differential manifest;
- deterministic systematic/random schedule manifest with seeds, coverage,
  replay and shrinker evidence; and
- the already-required exact-source clean/fault/native/controller/scale
  evidence.

Lean or theorem proving is not required on compute nodes. CI builds the pinned
proofs and replays the permanent corpus locally; the scale controller consumes
immutable prebuilt evidence. Missing, stale, partial, evaluator-only,
digest-mismatched, or failed evidence independently blocks the non-submitting
8-node preflight and therefore `scale-v21-direct-8n`.
