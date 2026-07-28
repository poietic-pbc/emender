# Resilient DiLoCo pure coordination kernel

This is the project-local Lean 4 workspace for the authoritative **pure**
resilient DiLoCo coordination model. It contains one total deterministic
transition, `ResilientProtocol.transition`, and the `resilient-trace`
executable that replays the canonical JSON trace produced by that same
function. It is not another networking runtime.

The compiled native service remains authoritative for transport, physical
timers and scheduling, model/tensor bytes, IEEE-754 reduction, registered
buffers, process supervision, libfabric, XPMEM/memfd, checkpoint I/O, and
production execution. Lean consumes opaque bounded evidence identities for
those facts and decides only admission, lifecycle, closure, fencing,
commit/mailbox/apply authority, and recovery disposition.

## Reproducible toolchain and dependency lock

The workspace pins:

- `lean-toolchain`: `leanprover/lean4:v4.26.0`;
- upstream Lean commit:
  `d8204c9fd894f91bbb2cdfec5912ec8196fd8562`;
- Lake: `5.0.0-src+d8204c9`;
- Linux x86-64 release archive SHA-256:
  `873c252b1c6b1392e5720ad8d5a137aabbe72c9f96a930fdb5a1dd1ddc5da454`;
- no external Lake packages (`lake-manifest.json` has an empty package set);
- trace schema: `emender-resilient-coordination-trace-v1`, whose exact
  file SHA-256 is bound in `ResilientProtocol.Types`.

The bootstrap never invokes elan and never writes `$HOME`, `~/.elan`, or a
machine-global toolchain. It downloads, verifies, and extracts under
`formal/resilient/.cache/resilient-lean` by default. Override
`EMENDER_RESILIENT_LEAN_CACHE` only with another project-owned cache path.
After the first verified download, set
`EMENDER_RESILIENT_LEAN_OFFLINE=1` to prohibit network access.

On a Frontier login node:

```bash
cd formal/resilient
scripts/lake.sh build
scripts/lake.sh test
scripts/smoke.sh
```

The first command may populate the project cache from the pinned GitHub
release. Subsequent commands are cache-only; no Slurm allocation, GPU, model
checkpoint, libfabric peer, Python process, or external runtime service is
used. A shared checkout may set `EMENDER_RESILIENT_LEAN_CACHE` to a durable
project-owned cache so WG worktrees reuse the verified archive/toolchain.
Concurrent completed caches are read-only in practice; bootstrap refuses to
overwrite an incomplete target.

## Authority and scope mapping

The model applies the conformance checklist in
[`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`](../../docs/RESILIENT_DILOCO_COMPUTE_POOL.md),
Version 1, and the independent definitions in
[`docs/RESILIENT_DILOCO_GAP_MATRIX.md`](../../docs/RESILIENT_DILOCO_GAP_MATRIX.md).
It also follows ADR-002, the native boundary, and execution-source identity:

- [`docs/ASYNC_DECOUPLED_DILOCO_V2.md`](../../docs/ASYNC_DECOUPLED_DILOCO_V2.md);
- [`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md);
- [`docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`](../../docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md).

Pure-kernel obligations represented by executable state and dispositions are:

- R01–R08 and R11–R16;
- NDP01, NDP06, NDP10–NDP13, NDP15, and NDP16;
- V21S01–V21S05, V21S07–V21S11, V21S13, V21S14, and V21S17;
- ISP04–ISP07 only where represented by a pure coordination transition,
  finite checked bound, typed disposition, or evidence identity.

Runtime-boundary-only obligations are represented only by typed evidence
identities and fail-closed boundaries, and are discharged by native
conformance/stress/gate work rather than claimed as Lean proofs:

- R09, R10, and R13;
- NDP02–NDP05, NDP07–NDP09, NDP14, and NDP17;
- V21S06, V21S12, V21S15, and V21S16;
- ISP01–ISP03.

The ISP04–ISP07 intersection is intentionally narrow. Lean decides represented
skip/defer/recovery outcomes and checks finite structural bounds and evidence
identities. Native/controller evidence must still prove physical byte
capacity, foreground nonblocking time, atomic model/optimizer visibility,
causal telemetry completeness, tail latency, and live qualification behavior.
The transition-by-transition crosswalk is
[`PROOF_COVERAGE.md`](PROOF_COVERAGE.md).

## Model surface

`Types.lean` keeps nominal types for run/allocation fence, policy/schema,
generation, attempt, worker, node, trainer, incarnation, contribution
sequence, owner epoch, receipt, result, digest, tick, and native evidence.
Lifecycle covers DISCOVER, BOOT, SYNC, leased READY, DRAIN, and EXPIRE.
Generation state covers open, closed, aborted, committed, and applied.

`Kernel.lean` implements the sole transition function. Highlights:

- all applicable run/fence/policy/schema/source/generation/attempt/owner and
  worker/incarnation/sequence/result/receipt identities are checked before a
  mutation;
- membership is an immutable, sorted snapshot of live leased READY peers,
  never launched ranks;
- fixed V21S17 closure remains open through its finite evidence-derived close
  even after `Q_min=2`, and includes every admissible pre-close arrival;
- contribution tokens are positive exact integers; the accepted list is
  deterministic and contains at most one stable worker per transition;
- identical replay is idempotent, while conflicting/stale/corrupt/late/closed
  input is a normal non-mutating disposition;
- closure below `Q_min`/`T_min` is `insufficient_cohort`, not an unbounded
  wait;
- a closed cohort never changes; owner reassignment is capped at two and
  exhaustion aborts without publication;
- one commit receipt/result is authoritative or none; a verified mailbox has
  capacity one;
- eight distinct current-incarnation trainer receipts must reduce to one node
  receipt before that node can advertise READY for the next generation;
- participant, service, manager, trainer, owner, and whole-allocation loss,
  restart, catch-up, and new-incarnation rejoin are explicit events/outcomes.

`strictV1` enforces fresh-only lag zero. `asyncV21` pins K40, exact two-node
`Q_min=2`, `T_min=3,934,080`, four independent lag maxima of two, lag-three
catch-up/defer, zero attempt retry, and exact-token-only coordination
accounting. Historical v2.0 or unknown policy/schema/source identities return
`unknown_identity` without mutation.

## Machine-checked safety

[`Safety.lean`](ResilientProtocol/Safety.lean) proves propositions about the
same `transition` called by traces and executables. It does not introduce a
simplified proof transition and does not promote `invariantHolds` or another
runtime Boolean to a theorem. `WellFormedState` contains only structural
authority, valid policy/schema, finite owner/floor bounds, frozen-cohort
consistency, commit ancestry, and all-eight node-apply consistency. It has no
delivery, scheduling, failure, or fairness field.

The checked results include:

- independent monotonic fence, generation, accepted-token, and receipt
  authority for all 15 event constructors;
- exact noninterference for stale fence/incarnation, identical/conflicting
  duplicate, corrupt, deferred, closed, late, catch-up, retry, and unknown
  identity outcomes;
- typed recovery actions for closed/late/catch-up/retry responses;
- one immutable accepted cohort and one authoritative commit/result over any
  in-frame trace;
- contribution admission only from the leased READY snapshot and exact
  event-bound identity;
- commit only for the declared frozen cohort, quorum, exact-token floor, and
  prior receipt;
- owner loss, peer restart, and abort cannot manufacture a commit or mailbox
  publication;
- node authority only after eight distinct matching trainer receipts, with a
  deferred/partial reduction preserving the exact state;
- restart/fresh-fence recovery cannot roll generation/token/receipt authority
  backward or resurrect historical work.

[`Regression.lean`](ResilientProtocol/Regression.lean) fixes the job-5105811
ordering at generation 3: close and commit; node-0 service/trainer loss and
new incarnation; concurrent node-1 submission to the committed generation;
typed catch-up with the literal pre-state preserved; no node-1 termination,
unrelated owner budget charge, second commit, partial apply, or rollback.
The canonical executable scenario contains that exact event after its full
peer bootstrap and then continues to generation 4.

[`Mutations.lean`](ResilientProtocol/Mutations.lean) contains deliberately
invalid double-commit, stale-write, conflicting-duplicate-write,
mutable-closed-cohort, partial-publication, and partial-node-apply variants.
Each adjacent theorem proves the mutant violates its corresponding safety
obligation.

## Conditional bounded progress

[`Progress.lean`](ResilientProtocol/Progress.lean) deliberately exposes no
unconditional liveness result. `BoundedProgressAssumptions` requires all of:

- finite close and stage deadlines;
- a surviving eligible stable-worker quorum;
- the surviving exact-token floor;
- bounded permitted failures and owner reassignments;
- eventual delivery and processing at a finite schedule index; and
- fair scheduling of the enabled executable transition.

Only under that complete package does
`bounded_next_generation_participation_under_exact_assumptions` establish a
bounded accepted next-generation contribution. Separate exclusion theorems
show that the package cannot be used under total participant loss, permanent
quorum loss, an expired deadline without the floor, over-budget/unbounded
faults, or unfair scheduling. The job-5105811 continuation theorem takes the
same package explicitly; its name and statement make no unconditional
progress claim.

## Proof manifest and trust boundary

[`proof-manifest-v1.json`](proof-manifest-v1.json) binds the pinned Lean
toolchain, authoritative transition/model identity, trace schema, proof
modules, coverage matrix, and their SHA-256 digests.
`scripts/verify-proof-manifest.sh` recomputes every digest and rejects a
missing, duplicate, absolute, or escaping artifact path. The smoke suite also
rejects `sorry`, `admit`, `native_decide`, axioms, opaque declarations, and
unsafe declarations throughout the proof and executable roots.

## Canonical trace and replay

[`trace-schema-v1.json`](trace-schema-v1.json) is the single interchange
schema for Lean and production differential replay. Every step binds:

- schema/policy/toolchain/execution-source identities;
- claimed pre-state digest and authoritative identity view;
- one fully fenced typed event and typed disposition;
- claimed post-state digest and authoritative identity view;
- invariant verdict and violations;
- trace/event indices, predecessor digest, causal parents, and replay target.

`Trace.strictFromJson` re-encodes the complete derived record and compares the
JSON trees. That rejects missing fields, unknown fields at any nesting depth,
and ambiguous tagged events. `Trace.parseTrace` additionally requires the raw
input (apart from leading/trailing whitespace) to equal the deterministic
compressed encoder, which rejects duplicate object keys, alternate field
orders, and other forbidden reorderings that a JSON object parser could
otherwise normalize. Replay additionally rejects reordered indices,
missing/impossible causal parents, future replay targets, stale pre-state,
mismatched disposition/post-state, and any false invariant verdict.

Coordination state and step lineage use the pinned
`emender-lean-coordination-state-v1` structural digest implemented in
`Trace.lean`. Opaque payload/result/receipt/checkpoint fields remain SHA-256
evidence supplied by native/reference code; the Lean structural digest makes
no cryptographic or floating-point claim.

Useful commands:

```bash
scripts/lake.sh exe resilient-examples --list
scripts/lake.sh exe resilient-examples
scripts/lake.sh exe resilient-examples \
  --trace job-5105811-generation-closed-restart-rejoin > /tmp/job-5105811.json
scripts/lake.sh exe resilient-trace replay /tmp/job-5105811.json
scripts/lake.sh exe resilient-conformance-corpus
scripts/lake.sh exe resilient-conformance \
  corpus/native-v1/native-job-5105811-generation-3-close-restart-rejoin.json
```

The executable corpus covers normal commit/apply, duplicate and conflicting
receipts, stale fence/incarnation, corrupt and lag-three input, peer and owner
loss, bounded replay/abort, all four runtime role-loss classes, fresh-fence
restart, late contribution after close, the permanent job-5105811 generation-3
close/commit → node-0 service loss/reincarnation → node-1 typed non-mutating
catch-up ordering, and new-incarnation admission into generation 4 under the
concrete finite executable schedule.

## Production native differential boundary

`Conformance.lean` derives
`emender-native-lean-authority-view-v1` only after calling the authoritative
`transition`. `ConformanceMain.lean` emits one oracle disposition, normalized
post-state, structural digest, and full Lean state digest per canonical event.
It is a view over the proof state, not a second transition system.

`ConformanceExamples.lean` and `corpus/native-v1/manifest.json` retain fifteen
production-bound traces with stable IDs, SHA-256 identities, event counts, and
direct replay commands. The job-5105811 trace is permanent. Regenerate the
checked corpus only from the pinned executable:

```bash
scripts/lake.sh build resilient-conformance resilient-conformance-corpus
"$EMENDER_PYTHON" \
  ../../scripts/conformance/generate_native_lean_corpus.py
```

The production differential runner is documented in
[`docs/NATIVE_LEAN_COORDINATION_CONFORMANCE.md`](../../docs/NATIVE_LEAN_COORDINATION_CONFORMANCE.md).
It starts the actual compiled persistent service and reaches
`coordination::step` through the public C ABI and service RPC. Agreement is
limited to pure coordination. It is not ISP timing, dense byte-path,
floating-point, Frontier provider, or scale evidence.
