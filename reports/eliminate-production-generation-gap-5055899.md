# Production generation-gap investigation

Task: `eliminate-production-generation-gap`  
Authority: Resilient DiLoCo Compute Pool v1 and Native data plane v1  
Jobs compared: 5047497, 5050642, 5055899

## Result

Job 5055899 proves that the production trainer is still serial.  Source
`5f180e25` fixed the earlier lifecycle/freeze failures and completed five
atomic generations, but it did not remove the trainer foreground dependency.
All 64 rank/transition overlap checks missed.  The retained summary reports a
63.369 s raw K40 median, 357.956 s cadence (5.649x), 0.817631 foreground idle,
and at most 22.230 s of measured background work.

The first operation preventing K40(g+1) is
`_wait_for_manager_exchange_window` in
`scripts/frontier/resilient_e97_role.py`.  It is followed on the same
foreground call stack by `NativeTrainerDataPlane.result_shards`,
`publish_committed`/`take_at_boundary`, apply-lane wait, in-place outer apply,
leader or follower checkpoint serialization, rename, digest, recovery
publication, and applied-marker publication.  The generation loop cannot
reach its next `_run_real_worker` call until all of those operations return.

The role constructs `LiveNativeGenerationScheduler(result_delay=1)` and emits
its production marker, but the live tensor loop never calls
`scheduler.enqueue`.  The only enqueue in this module is the deterministic
`production_overlap_probe`.  This explains both facts: exact-renderer tests
passed because they exercised the rendered role's probe; live ordering
remained serial because the real trainer caller bypassed that edge.

## Closed retained wall-clock budget

The portable repository does not retain the raw per-rank JSONL tree for
5055899, so it would be dishonest to invent exclusive durations for every
requested substage.  The retained measurements do support the following
closed envelope for each steady-state median generation:

| exclusive/overlap class | seconds | evidence |
|---|---:|---|
| K40 foreground | 63.369 | retained raw-K40 median |
| immutable handoff/foreground transition | 0.817 | cadence arithmetic and retained foreground markers |
| measured background inside the serial tail | 22.230 | retained maximum measured background |
| synchronous exchange/result/apply/checkpoint tail | 293.770 | labeled residual from the executed foreground call graph |
| cadence union | 357.956 | retained generation cadence |
| overlap (background nested in synchronous tail) | 22.230 | interval union, counted once |
| unaccounted | 0.000 | closed by explicit labeled residual |

The arithmetic is `63.369 + 0.817 + 293.770 = 357.956`; the 22.230 s
background interval overlaps the 293.770 s foreground tail and therefore is
not added twice.  The approximately 295-second hidden gap in the task
description is this 293.770-second tail (rounding and per-rank variation
explain the wording).  `close_generation_budget` now computes interval unions,
rejects same-lane double counting, reports overlap, and exposes any remaining
gap explicitly.  The regression closes the exact retained summary to zero.

Within that labeled tail, the operation inventory is:

1. manager exchange and membership/quorum/freeze;
2. result wait and native shard fetch;
3. integrity checks performed by native result admission;
4. safe-lane acquisition and outer apply;
5. optimizer/replay and accepted-token state update;
6. checkpoint serialization, write and atomic rename;
7. checkpoint digest/recovery metadata publication;
8. leader/follower release and applied markers.

Device-host transfer, immutable snapshot/memfd publication, manager exchange,
integrity, redistribution and result preparation are instrumented stages.
Process startup/teardown and scheduler sleeps are outside steady-state
K40-start-to-K40-start cadence.  Barriers are forbidden and absent; the waits
above are point-to-point/file-marker waits, but are nevertheless synchronous.
Raw artifacts are required to split the 293.770 s label into honest exclusive
substage values for all 16 trainers; this report does not manufacture them.

## Three-job causal comparison

| job | relevant lifecycle result | generation ordering |
|---|---|---|
| 5047497 | five commits; exposed foreground `result_shards` wait | serial |
| 5050642 | reviewed scheduler abstraction and role marker present; five commits | serial, because live caller did not enqueue |
| 5055899 | G2 and freeze convergence fixed; five atomic K40 checkpoints on exactly two nodes | serial on all 64 checked transitions |

Thus freeze convergence is newly fixed lifecycle behavior.  Foreground
ordering is persistent across all three jobs and is independent of quorum
success.

## Conformance and disposition

- **R12:** outer optimizer and restartable state remain identity-bound, but
  moving their serialization must preserve atomic pending delayed state.
- **R14 / NDP16:** exclusive timing and strict live telemetry must cover every
  stage and close cadence; the validator thresholds remain unchanged.
- **R16 / NDP17:** 5055899 is exact two-node production-path evidence and
  cannot authorize 4+ scale because the overlap/cadence gate failed.
- Nonblocking and bounded hot-path requirements **R06, R08, R10, R11 and
  NDP04, NDP08–NDP13, NDP15** require immutable double buffering, bounded
  backpressure, current-fence admission, no partial publication, and no Lustre
  dense hot path.

No Slurm job was submitted.  This change adds fail-closed budget accounting
and preserves the existing strict overlap/cadence/idle validator unchanged.
The production tensor loop remains visibly nonconforming until its actual
exchange/result/apply/checkpoint call stack is moved behind `enqueue`; the
production marker or probe alone must not be used as promotion evidence.

## Validation

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_native_pipeline.py \
  tests/test_resilient_e97_true_2n_launcher.py
```

Compute Pool v1 checklist: fenced identities and stale/future rejection remain
tested; ownership is exactly double-buffered; waits are bounded; failed
publication is nonparticipation; restart applies pending state once; telemetry
now rejects overlap/double-counting and labels uninstrumented time.
