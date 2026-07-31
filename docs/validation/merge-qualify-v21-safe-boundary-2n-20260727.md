# Deferred merge reconciliation: async v2.1 safe-boundary qualification

**Task:** `.merge-qualify-v21-safe-boundary-2n`  
**Date:** 2026-07-27  
**Source:** `wg/agent-1594/qualify-v21-safe-boundary-2n`  
**Target:** `main`

## Outcome

The deferred merge required no content-level conflict resolution. Before this
task began, the source tip and published target already named the same commit:

```text
source       ef879d0c38bcc4e94bc3be6eae301e20afac5750
origin/main  ef879d0c38bcc4e94bc3be6eae301e20afac5750
```

The dependency task had non-force published its complete validated history
directly to `origin/main`; only the local `main` worktree remained stale at
`243b94c7e791fad873964ddf622c95dacc6a936f`. This task fast-forwarded that
local target to the already-published source tip. Recreating a squash commit
would have required rewriting public `main` and invalidated the retained
commit- and digest-bound qualification evidence, so no force push or history
rewrite was performed.

The seven files recorded as conflicts by the stale done-time merge are now
exactly the source versions on `main`:

- `ndm/async_diloco_real.py`
- `scripts/frontier/resilient_e97_role.py`
- `scripts/frontier/validate_pipelined_e97_performance.py`
- `tests/test_async_snapshot_pipeline.py`
- `tests/test_resilient_e97_runtime.py`
- `tests/test_resilient_e97_true_2n_launcher.py`
- `tests/test_validate_pipelined_e97_performance.py`

## Architecture conformance

This reconciliation was checked against
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1, ADR-002, and
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. It preserves the qualified source tree
without semantic edits. Applicable requirements are R01–R16, NDP01–NDP17, and
V2A01–V2A18.

The retained qualification evidence continues to demonstrate leased READY
membership, bounded waits, fenced identities, deterministic token/lag
weighting, idempotence and rejection behavior, atomic commit evidence, bounded
non-Lustre native transport, backpressure/release, no central full-model
broker, exact two-node minimum progress floors, and the applicable failure,
deadline, recovery, snapshot, safe-boundary, and reload paths. No 4+ node
promotion is claimed by this merge reconciliation.

## Validation

- `git merge-base --is-ancestor main source` passed before reconciliation.
- `git diff --quiet origin/main source` passed before reconciliation.
- `git pull --ff-only origin main` advanced local `main` without conflict or
  rewrite.
- Source and target resolved to
  `ef879d0c38bcc4e94bc3be6eae301e20afac5750` immediately after the
  fast-forward.
- The source qualification retained an independent external audit of 544
  checks, the named OWNED regression, correction gates of 178 + 53 + 2 tests,
  native CTest 10/10, and clean `git diff --check`.
- The four conflict-scope Python test files passed 160/161 tests in one combined
  run. The sole failure was the timing-sensitive production overlap fixture
  observing 240 rather than 120 synthetic calls; an immediate isolated rerun
  of that exact test passed 1/1 in 30.83 seconds. This is recorded as a
  transient combined-suite timing result, not hidden as an all-green run.
- External retained evidence:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/qualify-v21-safe-boundary-2n/46c8043791e9b14c4cb3376c1fb03ebe7fe6932f/FINAL-SHA256SUMS.txt`
  and
  `/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/qualify-v21-safe-boundary-2n/46c8043791e9b14c4cb3376c1fb03ebe7fe6932f/scheduler-evidence/clean-5099195/final-validation.txt`.
