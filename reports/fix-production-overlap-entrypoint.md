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

## Exact retained topology and authority

Both retained runs used the same logical topology: one two-node Slurm
allocation, two model-free managers/native services, and sixteen real trainers
(eight local ranks per node).  The retained stdout explicitly reports
`managers=2 real_trainers=16 trainers_per_node=8 collective=none`.  Slurm's
expanded NodeList and opaque CXI endpoint bytes were not copied into the
portable retained report, so they cannot be reconstructed honestly from the
validator summary; the stable protocol identities are `node-0` and `node-1`.

| role | cardinality / identity | responsibility in 5047497 and 5050642 |
|---|---|---|
| allocation supervisor / lease holder | one, allocation fence owner | launches node-local roles, supervises deadlines, finalizes the accepted trainer proposal |
| manager + persistent native service | `node-0-manager`, `node-1-manager` | READY discovery, epoch-scoped membership snapshot, local 8-rank reduction, point-to-point owner exchange, quorum/result publication |
| trainers | `node-{0,1}-trainer-{0..7}` | K40, immutable delta/memfd handoff, result mapping and apply |
| checkpoint leader | `node-0-trainer-0` | streamed apply, atomic checkpoint rename, fenced proposal |
| followers | remaining 15 trainers; node-0 peers wait for leader release | apply the same result root and publish local recovery acknowledgement |

Node 0 is therefore the publication head, but it is not a central dense
aggregator: deterministic native shard owners exchange data directly.  The
head observes the accepted native result and publishes the checkpoint; the
allocation supervisor finalizes it.  The production argv comes from the exact
renderer to `resilient_e97_true_2n.sbatch`, whose `ROLE` is
`scripts/frontier/resilient_e97_role.py`; that role connects
`NativeTrainerDataPlane` to `EMENDER_NDP_SOCKET`.  Job 5050642's source was
32fd9ab1 and included reviewed scheduler 84a81e40.  This proves the executed
implementation, while the new runtime marker additionally binds code ID,
role-source path, run, fence, trainer identity, scheduler class, and the
one-generation result delay.

## Generation certificate and causal evidence

For generations 0 and 1, both jobs formed the same two-peer eligible manager
set and accepted the same class of complete 16-trainer contribution set before
atomic publication.  Retained portable evidence does not contain each CXI
endpoint, per-contribution digest, membership epoch, or decision timestamp;
claiming exact values would manufacture evidence.  Those fields remain in the
per-node native/pool JSONL and generation manifests retained with job 5050642.
The relevant certificate is the already-fenced tuple: parent/base digest,
allocation fence (membership epoch), generation/attempt, frozen accepted peer
set and threshold, contribution identities/digests/weights, membership root,
result root, checkpoint digest, and leader proposal.  Monotonic telemetry must
associate every discovery, freeze, contribution, decision, head observation,
result and apply event with that tuple.

The decisive causal fact needs no inference from completion ordering.  In the
executed trainer source, control cannot reach the next loop's `training_start`
until `_wait_for_manager_exchange_window`, `result_shards().__enter__`,
`take_at_boundary`, apply-lane acquisition, in-place outer apply, checkpoint
save/proposal and recovery publication have returned.  Certification of g is
therefore foreground permission for K40(g+1).  That coupling is the first
missing edge; discovery/quorum may be slow, but even an immediate quorum still
runs on the foreground dependency chain.  Scheduler-only tests passed because
they manually enqueued work and emitted K40(g+1), bypassing this caller.

The production scheduler is now configured with `result_delay=1`: a committed
result for g is admissible only at boundary g+1, never at g and never after it
has become two generations stale.  This encodes the intended safe-boundary
identity and prevents an apparently green same-generation scheduler test from
silently changing the algorithm.

## Split-brain safety

Two-node quorum policy cannot simultaneously provide one-failure availability
and partition safety.  With an unchanged two-member epoch, quorum 1 permits
both sides of a partition to certify conflicting children; quorum 2 cannot
progress after either member disappears.  The only safe one-node continuation
is a lease-holder-authorized membership reconfiguration under a strictly newer
fenced epoch, rooted at the last accepted parent.  A late or recovered peer
syncs that parent and joins a subsequent frozen epoch.  The head follows the
single certificate accepted under the current allocation fence; it never
lowers quorum locally and is not itself a synchronous global gate.  No
Byzantine consensus or proof-of-work is required for this crash-fault model.
