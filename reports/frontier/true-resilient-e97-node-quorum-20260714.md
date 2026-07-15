# Initial true-resilient E97 node-quorum transport

Task: `build-true-resilient-e97`  
Date: 2026-07-14

## Audit and feasibility conclusion

The reusable pieces are the pure quorum/catch-up/checkpoint math in
`ndm/async_diloco.py`, the checksummed bucket envelope in
`ndm/async_diloco_mpi.py`, and the failure-accounting tests.  The TCP path in
`ndm/async_diloco_real.py` is a bounded debug metadata coordinator, not a
model-sized scalable data plane.  `run_mpi_dense_quorum` is also not resilient:
it creates `MPI_COMM_WORLD`, sends every rank's dense update to one root, and
waits for a root reply.  The compiled helper is a healthy-world strict
`MPI_Reduce` control and remains unchanged.  Existing Frontier results prove
that control's reachability/performance, not process-failure tolerance.

The selected new explicit mode is `resilient-node-quorum-sharded-p2p`.  Eight
training ranks must first aggregate locally behind one node manager; an
incomplete local group excludes that node for the generation.  Node managers
then stream checksummed buckets to independently replaceable shard owners.  A
small fenced metadata coordinator freezes and durably publishes the exact
accepted-node set.  Senders retain generation buckets until commit, allowing a
replacement shard owner to request replay.  Model payloads do not pass through
the metadata directory.  The initial implementation in
`ndm/resilient_node_quorum.py` supplies the protocol/state primitives; wiring a
Frontier network backend and node-step supervisor remains before this is a
production trainer mode.

## Guarantees demonstrated locally

- generation, attempt, run, and coordinator-epoch fencing;
- stale/late attempts cannot enter a current accepted set;
- exact accepted set is immutable once its atomic manifest and `latest.json`
  record are committed;
- quorum advances with a completely absent/stuck node and computes an exact
  token-weighted mean;
- quorum loss fails closed without advancing latest;
- sender-retained buckets replay to a replacement owner after owner failure;
- coordinator failover fences the old writer;
- a restarted/lagging manager verifies the latest-manifest checksum and catches
  up from the committed generation;
- checksums and hard retained-byte limits reject corrupt or unbounded payloads.

These are deterministic protocol tests, not Frontier network evidence.  They
do not claim that a killed physical node can be replaced inside an allocation.

## Frontier and Slurm boundary

No job was submitted by this implementation pass.  In particular, production
job 4980157, attested tree `9fff689c9f9252b6a264773c207f8f8ca8509666`, its
attempt marker/run root, `production/latest`, and the pinned step-1525000 seed
pointer were not mutated.

Before the required two-node test, the launcher must use independent node
manager steps with explicit deadlines rather than a single long-lived
world-size collective.  `--no-kill` can keep an allocation/step alive under
some task failures, but it is not evidence that Frontier permits arbitrary
replacement of a failed physical node or safe creation of a replacement step;
that exact behavior must be tested in a unique debug allocation and recorded
with `scontrol`, `sacct`, `sstat`, and the fault timeline.  Slurm cannot add a
node outside the fixed allocation unless the site exposes a supported
mechanism.  Across-job checkpoint/restart is therefore still required.

The follow-up implementation adds a real, structured TCP bucket data-plane
primitive (`ShardOwnerServer`/`send_bucket`) rather than an in-memory-only
simulation. It length-prefixes JSON envelopes, caps every frame before
allocation, validates schema/checksum and the full generation/attempt/epoch
fence, returns a checksummed receipt, and stores the accepted payload under a
hard byte bound. `supervise_until_quorum` uses a monotonic progress deadline,
freezes the deterministic accepted set, terminates only incomplete node-manager
processes, and fails closed when too few complete nodes remain. A process fault
test sends two healthy nodes through real loopback sockets, leaves a third OS
process stuck, proves that the supervisor terminates only that process, commits
the exact mean, and releases retained sender payloads.

`ndm/resilient_node_transport.py` extends that primitive into a complete local
node-manager exchange: a crash-surviving bounded disk spool retains float64
buckets, independent bucket connections permit replay, the server freezes a
quorum and computes the exact weighted mean, and committed buckets stream back
to every accepted client. The metadata manifest contains only membership,
sizes, and hashes. Its integration suite proves a completely missing node does
not prevent redistribution, verifies restart replay, exercises deadline
failure and stale-epoch fencing, and kills a stuck child step while leaving the
supervisor reusable.

This is now a production-shaped network/supervision primitive, but it is not
yet wired into `e97_async_diloco_train.py`'s model/optimizer state transition or
an independent-per-node `srun` launcher. Aggregate redistribution is represented
by verified committed bucket payloads and `catch_up`, not yet applied to E97
tensors. Consequently the required unique 2-node/16-GPU injected-failure smoke
is still blocked by implementation, not by Frontier: submitting the existing
launchers would exercise TCP-debug or strict MPI and would be false evidence.
No resilient-mode Slurm job was submitted in this pass. The remaining gate is
the tensor codec/apply path, per-node step launcher and on-allocation restart;
only after those exist is one <=20-minute attempt justified. The <=8-node rung
remains gated on it; 64/128/256-node and production-QoS jobs are out of scope.

## Validation

Run:

```text
pytest -q tests/test_resilient_node_quorum.py
pytest -q tests/test_async_diloco_mpi_transport.py tests/test_async_diloco_compiled_mpich.py
git diff --check
```

The first suite includes real loopback TCP and real OS-process termination as
well as independent protocol stores and missing/stale/failover events. It proves
local forward progress without an MPI communicator. It is not a Frontier smoke;
that remains an explicit completion blocker rather than being overstated here.
