# Bootstrap Lean 4 resilient protocol — validation record

Date: 2026-07-28

Task: `bootstrap-lean4-resilient-protocol`

Validated implementation commit: `1ef10e52c09b8a289b5bd1ec086428a11f9ffb0e`

Published branch: `wg/agent-1646/bootstrap-lean4-resilient-protocol`

## Result

The project now has a reproducible, project-local, Std/Lean-core-only Lean 4
workspace for the pure resilient DiLoCo coordination kernel. The workspace
contains one total deterministic transition function,
`ResilientProtocol.transition`, and the executable trace runner and examples
call that function directly. The native implementation remains authoritative
for transport, physical time, byte buffers, process effects, floating-point
behavior, and production execution; no second networking runtime was added.

The validated implementation commit was pushed before this record was written.
The local and remote branch object IDs were compared directly and were both:

```text
1ef10e52c09b8a289b5bd1ec086428a11f9ffb0e
```

## Design-authority and conformance check

The implementation was checked against the Version 1 conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, all four independently defined
namespaces in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, ADR-002 in
`docs/ASYNC_DECOUPLED_DILOCO_V2.md`, the native authority boundary in
`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, and
`docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`.

Pure coordination behavior is represented for:

- R01–R08 and R11–R16;
- NDP01, NDP02, NDP06–NDP13, and NDP15–NDP17;
- V21S01–V21S05, V21S07–V21S11, and V21S13–V21S17;
- ISP02 and ISP04–ISP07.

The following obligations are deliberately runtime-boundary-only. Lean binds
typed identities/evidence and fails closed, but does not claim to prove their
physical implementation:

- R09, R10, and the physical backend portion of R13;
- NDP03–NDP05, the physical buffer/credit portions of NDP08–NDP09, and NDP14;
- V21S06 and V21S12;
- ISP01 and ISP03.

The R13, NDP08, and NDP09 intersections are intentional. The pure kernel can
decide a backend-neutral or capacity-exhaustion disposition while native
conformance/stress/gate tasks must prove actual point-to-point transport,
registered-byte capacity, and credit behavior.

The compute-pool checklist was applied as follows:

1. Versioned authority and the independent R/NDP/V21S/ISP definitions are
   cited above and in `formal/resilient/README.md`; the implementation does not
   substitute older scaffolding for the Version 1 authority.
2. Generation closure is over an immutable sorted snapshot of live,
   unexpired, leased-READY peer identities. It never consults launched-rank
   unanimity. Fixed scale closure stays open through the evidence-derived
   finite close even if the two-node floor was already reached.
3. The pure kernel has no SQLite, TCP collective, libfabric, XPMEM, memfd,
   Slurm, GPU, model-checkpoint, or tensor implementation. These remain native
   boundaries.
4. Every mutating event carries and checks its applicable run/allocation
   fence, policy/schema/source, generation, attempt, worker/node/trainer,
   incarnation, sequence, owner epoch, result, receipt, and evidence
   identities. Identity-incomplete JSON does not decode.
5. Exact positive token accounting, lag eligibility, deterministic admitted
   order, one stable worker admission, immutable cohorts, bounded owner replay,
   exact-once commit, capacity-one verified mailbox, and all-eight trainer
   reduction are executable transition guards.
6. Loss and recovery paths cover peer/participant, service, manager, trainer,
   and owner loss, allocation restart, catch-up, and new-incarnation rejoin.
   Insufficient closure is a finite typed outcome rather than deadlock.
7. The canonical trace binds pre/post authority and digests, event and
   disposition, invariant verdict, schema/policy/toolchain/source identities,
   and causality/replay metadata. Missing/unknown fields, duplicate keys,
   alternate field order, forbidden event reordering, stale predecessors, and
   impossible replay targets fail closed.
8. Exact build/test/replay commands and durable artifacts are recorded below.

## Reproducibility pins

- `lean-toolchain`: `leanprover/lean4:v4.26.0`
- Lean upstream commit:
  `d8204c9fd894f91bbb2cdfec5912ec8196fd8562`
- Lake: `5.0.0-src+d8204c9`
- Linux x86-64 release archive SHA-256:
  `873c252b1c6b1392e5720ad8d5a137aabbe72c9f96a930fdb5a1dd1ddc5da454`
- Lake dependencies: none; `lake-manifest.json` has an empty `packages` array
- Trace schema identity: `emender-resilient-coordination-trace-v1`
- Trace schema SHA-256:
  `cf654525395e63b31b2d76e8109ee2bcc6a652f6273d1c6e4ca5bec9ecb776b4`

`formal/resilient/scripts/bootstrap.sh` verifies every toolchain identity and
installs only under the ignored project cache
`formal/resilient/.cache/resilient-lean` (or an explicitly selected
project-owned cache). It never invokes elan or writes a global user
environment. Frontier login-node and offline-cache behavior is documented in
`formal/resilient/README.md`.

## Exact validation commands and outcomes

The validation was run from repository root. The project-local wrapper selected
the pinned toolchain:

```text
$ formal/resilient/scripts/lake.sh --version
Lake version 5.0.0-src+d8204c9 (Lean version 4.26.0)

$ formal/resilient/scripts/lake.sh env lean --version
Lean (version 4.26.0, x86_64-unknown-linux-gnu, commit d8204c9fd894f91bbb2cdfec5912ec8196fd8562, Release)
```

Dependency-lock refresh was clean:

```text
$ formal/resilient/scripts/lake.sh update
info: toolchain not updated; already up-to-date
```

The complete local/CI smoke gate passed:

```text
$ formal/resilient/scripts/smoke.sh
formal/resilient/trace-schema-v1.json: OK
Build completed successfully (17 jobs).
PASS resilient protocol: 10 scenarios
...
"verdict":"accepted"
```

The smoke script performs these exact checks:

1. verifies the pinned trace-schema SHA-256;
2. rejects `sorry`, `admit`, `native_decide`, explicit `axiom`, and `opaque`;
3. runs `scripts/lake.sh build`;
4. runs `scripts/lake.sh test`;
5. runs every executable example;
6. emits the permanent job-5105811 trace through the example runner; and
7. parses and replays that canonical trace through `resilient-trace`.

The ten passing executable scenarios are:

1. normal commit/apply plus duplicate, conflict, stale, corrupt, catch-up,
   late, mailbox, and trainer/node receipt handling;
2. peer/manager loss with insufficient cohort;
3. bounded owner replay and abort;
4. participant loss;
5. service loss;
6. trainer loss;
7. job-5105811 generation-closed restart, catch-up, and next-generation rejoin;
8. fresh allocation-fence restart;
9. leased-READY finite snapshot closure; and
10. strict-v1 admission plus fail-closed historical/unknown identity.

Focused assertions also prove by execution that all non-mutating dispositions
preserve the exact state, canonical trace round-trip/replay matches direct
execution, and unknown fields, duplicate JSON keys, reordered fields,
reordered trace steps, and incomplete identities are rejected.

`git diff --check` passed before the implementation commit. The smoke used no
Slurm allocation, GPU, model tensor/checkpoint, libfabric peer, Python helper,
or external service. The only possible network operation is the initial
download of the exact SHA-pinned official Lean archive; offline mode fails
closed if that archive is absent and performs no network request once it is
cached.

## Downstream handoff

Safety-proof work can import `ResilientProtocol.Kernel` and reason about the
single `transition` function and `invariantViolations`. Native-conformance work
can emit `emender-resilient-coordination-trace-v1` and use
`resilient-trace replay` as the pure coordination differential oracle. Neither
consumer should infer transport, timer, floating-point, or physical-buffer
claims from the Lean model.
