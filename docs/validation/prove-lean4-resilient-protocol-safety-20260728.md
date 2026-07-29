# Prove Lean 4 resilient protocol safety — validation record

Date: 2026-07-28
Task: `prove-lean4-resilient-protocol-safety`
Implementation commit:
`9a548394082c40bfb7f9ac4598b48bd56ae711e1`

## Scope and authorities

This work applied the Version 1 checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and kept the independent R, NDP,
V21S, and ISP namespaces from `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.
`formal/resilient/PROOF_COVERAGE.md` is the transition-constructor/theorem/
assumption/requirement crosswalk.

The proved coordination-kernel scope is:

- R01–R08, R11, R12, R14–R16;
- NDP01, NDP06, NDP10–NDP13, NDP15, NDP16;
- V21S01–V21S05, V21S07–V21S11, V21S13, V21S14, V21S17;
- ISP04–ISP07 only for facts represented by pure coordination transitions,
  typed dispositions, finite checked bounds, or evidence identities.

The following remain explicit native/controller evidence obligations and are
not Lean theorem claims:

- R09, R10, R13;
- NDP02–NDP05, NDP07–NDP09, NDP14, NDP17;
- V21S06, V21S12, V21S15, V21S16;
- ISP01–ISP03.

## Environment and isolation

All proof and trace validation ran locally in the WG worktree. The verified
Lean archive/toolchain cache was reused in mandatory offline mode:

```bash
export EMENDER_RESILIENT_LEAN_CACHE=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1646/formal/resilient/.cache/resilient-lean
export EMENDER_RESILIENT_LEAN_OFFLINE=1
```

No Python helper, native helper, Slurm command/allocation, GPU, model tensor or
checkpoint, native runtime service, libfabric peer, package download, or
external proof/runtime network was used. Therefore the Frontier Python
activation rule was not triggered. The only remote operation was the
task-mandated Git publication after local validation.

## Exact validation commands and results

From `formal/resilient`:

```bash
./scripts/lake.sh build
```

Result: pass; 21/21 targets. This compiled `Types`, the single executable
`Kernel.transition`, `Safety`, `Progress`, `Regression`, `Mutations`, trace
replay, examples, and the authoritative import root. There were no warnings
and no theorem failures.

```bash
./scripts/lake.sh test
```

Result:

```text
PASS resilient protocol: 10 scenarios
```

The test driver also checked the literal job-5105811 generation-3 state before
the raced contribution: committed/frozen authority, node-0 service loss and
new incarnation, surviving node 1, one expected restart, zero owner
reassignments, no node apply, typed catch-up, and exact nonmutation. The
continuation admitted the new node-0 incarnation in generation 4.

```bash
./scripts/smoke.sh
```

Result: pass. The smoke gate checked:

- trace schema SHA-256;
- 18 proof-manifest artifact digests;
- absence of `sorry`, `admit`, `native_decide`, `axiom`, `opaque`, and
  `unsafe` trust escapes;
- `lake build`;
- `lake test`;
- all 10 executable scenarios;
- the 63-transition canonical
  `job-5105811-generation-closed-restart-rejoin` trace; and
- parse/replay through `resilient-trace`.

The canonical job trace replay ended with:

```text
verdict=accepted
generation=4
generationStatus=open
acceptedCount=1
finalStateDigest=47e34332f49a82e16f7adafe5531d371a3c3808cfe5fb0fe6be8158eb9197f81
```

Additional local checks:

```bash
formal/resilient/scripts/verify-proof-manifest.sh
git diff --check
bash -n formal/resilient/scripts/smoke.sh \
  formal/resilient/scripts/verify-proof-manifest.sh
jq -e . formal/resilient/proof-manifest-v1.json
rg -n '\b(sorry|admit|native_decide)\b|^\s*axiom\b|^\s*opaque\b|^\s*unsafe\b' \
  formal/resilient/ResilientProtocol.lean \
  formal/resilient/ResilientProtocol \
  formal/resilient/TraceMain.lean \
  formal/resilient/ExamplesMain.lean \
  formal/resilient/TestsMain.lean
```

Results: manifest pass (`18 bound artifacts`), clean diff/shell/JSON checks,
and no forbidden Lean matches.

## Safety and progress disposition

Safety theorems are propositions over the same executable transition and
decision stages used by trace replay. `WellFormedState` contains structural
policy/schema, finite-bound, frozen-cohort, commit ancestry, and all-eight
apply facts; it contains no delivery, scheduling, failure, or fairness
hypothesis.

The proof suite covers all 15 event constructors and proves independent
fence/generation/token/receipt authority monotonicity, exact typed-rejection
noninterference, duplicate semantics, immutable cohort/commit authority,
leased-READY admission, declared-quorum/exact-token commit, no partial
publication from loss/restart/abort, all-eight node authority, and recovery
non-rollback/non-resurrection.

`BoundedProgressAssumptions` explicitly requires finite close/stage deadlines,
surviving eligible stable-worker quorum, the exact-token floor, bounded
failures/reassignments, eventual delivery/processing, and fair scheduling of
the enabled executable transition. Exclusion theorems cover total participant
loss, permanent quorum loss, expired deadline without the floor,
over-budget/unbounded faults, and unfair scheduling. No unconditional progress
theorem is present.

The mutation module proves rejection of deliberately invalid double-commit,
stale-write, conflicting-duplicate-write, mutable-closed-cohort,
partial-publication, and partial-node-apply variants.

## Commit and pushed equality

Surgical staging listed all 14 task files explicitly; neither `git add -A` nor
`git add .` was used.

```bash
git commit -m \
  "feat: prove resilient protocol safety (prove-lean4-resilient-protocol-safety)"
git push --set-upstream origin \
  wg/agent-1654/prove-lean4-resilient-protocol-safety
git rev-parse HEAD
git rev-parse '@{u}'
```

Both local `HEAD` and the pushed upstream ref resolved to:

```text
9a548394082c40bfb7f9ac4598b48bd56ae711e1
```
