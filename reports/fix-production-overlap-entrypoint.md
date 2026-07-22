# Production overlap entrypoint investigation

Task: `fix-production-overlap-entrypoint`  
Date: 2026-07-22

## Source and rendered path

The exact acceptance renderer records
`scripts/frontier/resilient_e97_true_2n.sbatch`.  That batch file exports the
fully rendered manager and trainer commands with
`ROLE=$REPO/scripts/frontier/resilient_e97_role.py`; the allocation supervisor
executes those commands.  The trainer now writes
`emender-production-delayed-pipeline-v1` after native attestation and recovery,
including the role source, run/code/fence identity, and the concrete
`ndm.native_pipeline.LiveNativeGenerationScheduler` implementation.  This is
an explicit production-path marker, rather than an inference from an imported
module.

## Monotonic ordering and first dependency

The retained terminal records for pre-fix job 5047497 and exact-source
post-abstraction-fix job 5050642 agree on the following generation-0/1 order.
Both jobs completed five atomic generations; 5047497 ran 30:58 and 5050642 ran
31:08.  The strict, unchanged validator rejected both at the same edge:
generation-0 background ended before generation-1 K40 began.

| order | production operation | foreground dependency |
|---:|---|---|
| 1 | generation 0 K40 start/end | model-owning trainer |
| 2 | sealed memfd publish and handoff | `publish_model_delta`; bounded producer ownership |
| 3 | manager exchange-window wait | synchronous `_wait_for_manager_exchange_window` |
| 4 | result shards | synchronous `native_plane.result_shards().__enter__` |
| 5 | result admission | synchronous `publish_committed` / `take_at_boundary` |
| 6 | apply lane and outer apply | synchronous lane wait and in-place apply |
| 7 | checkpoint/proposal | synchronous save, atomic rename, fenced proposal |
| 8 | follower/recovery checkpoint | synchronous save/promotion |
| 9 | generation 1 K40 start | loop can only advance after every row above |

There was no scheduler enqueue in either live timeline.  Thus the first
synchronous dependency after the snapshot/handoff is the manager exchange
window at `resilient_e97_role.py`; `result_shards`, apply-lane, checkpoint, and
recovery joins extend the same foreground critical path.  The memfd ownership
token itself is released promptly and is not the cause.

## Why the previous tests were green

Commit 84a81e40 tested `LiveNativeGenerationScheduler` directly.  Its overlap
test manually enqueued fake work and manually emitted generation-1 K40 events.
The production role neither constructed nor enqueued that scheduler, so those
tests proved the policy object but not its caller.  This is a missing
production-entrypoint test, not a telemetry threshold issue.  The new focused
test follows the renderer to the batch role and requires the concrete
production marker and scheduler construction.

## Conformance

The investigation used the Compute Pool v1 checklist and gap-matrix IDs
R01-R16 and NDP01-NDP17.  The marker preserves source/fence identity (R01,
R10, R13; NDP02, NDP03, NDP13).  Queue/identity/result behavior remains owned
by the reviewed bounded scheduler (R04, R06, R07, R11; NDP10, NDP15).  No
validator thresholds were changed and no Slurm job was submitted.

## Remaining implementation boundary

The production marker and K40 scheduler telemetry are now present, but moving
exchange, result-shard collection, apply, and checkpoint publication off the
foreground loop requires a larger state-machine conversion than the marker
patch alone.  Until that conversion is complete, the live overlap requirement
R12/R14/R16 and NDP16/NDP17 must remain fail-closed.
