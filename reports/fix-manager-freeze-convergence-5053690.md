# Job 5053690 manager FREEZE convergence report

## Scope and authority

This implementation follows *Resilient DiLoCo Compute Pool*, version 1, and
addresses R02–R04, R06–R08, R11, R14, R16 and NDP06, NDP07, NDP10, NDP13,
NDP15, NDP17.  No Slurm job was submitted.  The source under test was
`53441395`; its exact-source G2 prerequisite was job 5053588.  The failing K40
allocation was job 5053690 on frontier05345 and frontier05350.

## Monotonic generation-0 timeline

The retained job summary and the production call graph establish the following
single happens-before order (timestamps in the job telemetry are monotonic
stage/progress timestamps; the job bundle itself is not present in this
worktree, so this report does not invent wall-clock values):

1. The allocation holder admitted one run/fence and both managers advertised a
   unique incarnation and current-fence native endpoint.  Generation 0,
   attempt 1 was opened from that fenced READY snapshot.
2. All 16 rank contributions completed K40 and were admitted.  This rules out
   seed, source, G2, trainer, token-floor, and corrupt-contribution failures.
3. Each manager installed the same native `(run, fence, generation=0,
   attempt=1, owner_epoch=1, base, plan, layout)` identity and locally reduced
   its accepted ranks.  Manager supervision entered `freeze`.
4. FREEZE established the immutable local accepted set.  One manager advanced
   through `FINALIZE_OWNERS` to native `RESULT_READY` while reordered/retried
   FREEZE control traffic was still possible on the other route.
5. The native service accepted repeated FREEZE only in exactly `FROZEN`.
   Consequently the already-advanced manager returned `NDP_ESTATE`, while its
   peer remained on the progress-deadline path.  Python made this worse by
   rejecting every second `NativeManagerSession.freeze()` solely from its
   local `_frozen` boolean rather than asking the fenced service for the
   established operation.
6. The manager waiting for rank metadata refreshed progress only at the phase
   boundary, not for each accepted contribution.  Its deadline could therefore
   be based on the stale `training_wait` origin even while ranks were arriving.
   This explains the observed asymmetric `progress_deadline` versus invalid
   native lifecycle state from identical generation/fence inputs.
7. Peer-route recovery could not repair a lifecycle split: the immutable
   accepted set existed, but the advanced side rejected the replay before
   route replacement could converge.  Teardown followed with zero atomic
   generations published.  There is no evidence of a manager restart or fence
   change before this split; route recovery was subsequent, not causal.

## Root cause and correction

The primary defect was a non-idempotent *state-dependent reply* to an
identity-idempotent FREEZE.  `FROZEN -> RESULT_READY` is valid monotonic
progress, but it accidentally removed the ability to acknowledge the earlier
FREEZE.  The correction retains and returns the original freeze operation in
`FROZEN`, `RESULT_READY`, and `COMMITTED`; stale generation/attempt/fence input
continues to fail before that branch.  The Python manager now delegates replay
to that authoritative fenced lifecycle.  It does not reduce twice, create a
second result, or weaken corruption checks.

The production manager also refreshes monotonic progress after each accepted
rank contribution.  This changes no queue or overlap bounds and prevents an
actively advancing generation from expiring against its phase-entry time.
Rank acceptance remains policy-driven by `local_quorum`; the loop no longer
treats a silent multi-rank list comprehension as one indivisible progress
event.  The existing rank membership certificate and fenced quorum policy
remain authoritative, so an absent rank is recorded as nonparticipation rather
than requiring an inferred eight-contribution manager invariant.

Finally, terminal harvesting records the first observation of the transient
state in which a job has left `squeue` but is not yet visible to `sacct`.  It
retries for a bounded 120-second propagation window, clears the marker when a
terminal record appears, and fails explicitly after the bound instead of
misclassifying the job as missing or waiting forever.

## Validation mapping

- Native service RPC integration replays FREEZE after `RESULT_READY` and
  requires the exact original operation handle.
- Exact-2n renderer tests cover delayed `sacct` appearance and the bounded
  propagation failure.
- Existing native failure/reference/integration tests retain stale fence,
  corruption, duplicate contribution, bounded replay/queues, one-result, and
  strict overlap coverage.
- No scheduler submission command was run during this task.
