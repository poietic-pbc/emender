# Pipelined generation-start lifecycle fix

Date: 2026-07-20  
Task: `fix-pipelined-generation`

## Root cause and correction

Exact two-node K40 job 5037971 reached generation-0 `commit_ready` with Q=2,
5,245,440 accepted tokens, and result root
`63b3fef285173902e0ee4b54f4e7cab61fac8860c7427d9ef2c3750b9c641477`.
The trainer then evaluated native pipeline telemetry using
`generation_started`, but the variable was never assigned in `trainer`.  The
similarly named manager loop timestamp hid the omission from source-contract
coverage.  Because the reference occurs only after result admission, the bug
could not appear until the expensive production path had completed training,
freeze, native reduction, and result publication.

The trainer now initializes the monotonic generation origin at the sole entry
to its generation loop, before the TERM check and every training, handoff,
delayed-result, rejected-result, safe-boundary, and failure branch.  The same
entry is used after fresh start, authoritative handoff resume, local recovery,
and supervisor restart.  No elapsed time is synthesized and no exception is
suppressed.

## Failure-chain conclusion

The retained ordering supports one primary application failure followed by
shutdown effects: both trainer leaders raised the `NameError`; a native route
then completed with `-12` while the role group was draining; manager `FREEZE`
subsequently saw lifecycle `-3`; and the supervisor exhausted restarts.  The
focused deterministic test now takes the exact accepted-token count through a
verified committed-result safe boundary and reserves the generation-1 slot
without foreground blocking.  Existing native pipeline tests continue to
prove that malformed, non-finite, stale, wrong-fence, wrong-base, and
pre-checkpoint-CAS results fail closed.  With the initiating exception removed,
the focused runtime/launcher suite has no route, freeze, or restart lifecycle
failure.  Therefore the retained `-12`, `-3`, and restart exhaustion are
cascading consequences; no independent lifecycle defect was reproduced.

## Validation and conformance

- `tests/test_resilient_e97_runtime.py::test_native_trainer_generation_timer_covers_every_result_lifecycle_path`
  is the focused regression for the undefined timestamp and verifies dominance
  over result lifecycle branches.
- `tests/test_resilient_e97_runtime.py::test_native_pipeline_commit_ready_advances_without_foreground_blocking`
  deterministically advances generation 0 through commit/apply and begins
  generation 1 using the retained production token count.
- The focused runtime, pipeline, and exact-two-node launcher suite passed: 90
  tests.
- The canonical native release build completed and CTest passed 10/10 tests.

The change conforms to the *Resilient DiLoCo Compute Pool* checklist, notably
R04 (generation identity/stale rejection), R07 (atomic commit), R11-R12
(restart/resume), R14 (monotonic bounded stage evidence), and R16 (two-node
gate ordering), and to native requirements NDP10, NDP13, NDP15-NDP17.  It does
not change admission, integrity verification, checkpoint CAS, failure
propagation, or the ordered scale ladder.  No Slurm job was submitted.
