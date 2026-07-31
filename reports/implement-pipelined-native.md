# Pipelined native DiLoCo implementation evidence

**Authority.** This implementation conforms to *Resilient DiLoCo Compute
Pool*, version 1, and the companion gap matrix requirements R01–R16 and
NDP01–NDP17.  The implemented policy layer is
`ndm/native_pipeline.py`; dense bytes remain exclusively in the existing
compiled service and owner-sharded transport.

## Implemented contract

`NativeGenerationPipeline` provides exactly two immutable contribution slots
and one latest-only committed-result mailbox. A writer must reserve, seal, and
hand off a slot with the full run/fence/generation/attempt/incarnation/layout/
base identity. The service releases it using the exact returned ownership
token, so stale receipts cannot free a reused slot. Exhaustion waits only to an
explicit deadline and records foreground wait and queue high-water telemetry.

Only complete results with positive weight, full digests, the current fence and
incarnation, and a successful caller-supplied durability/integrity verifier can
enter the mailbox. Newer committed results replace older volatile results;
stale results cannot replace newer ones. A trainer reads the mailbox without
waiting and applies only at an explicit K boundary when generation, fence,
incarnation, and base digest all match. No result means local training
continues. Rebind atomically invalidates all volatile slots and results after a
new incarnation or fence.

The class deliberately owns no tensor storage and performs no transport. Thus
it cannot introduce Python dense traffic, a central broker, inter-node copies,
or a collective. The native service continues to fuse finite validation into
its reduction/projection passes and to authenticate sealed memfds and framed
owner traffic at their established trust boundaries (NDP06, NDP09–NDP12,
NDP16).

## Validation

Canonical Frontier environment commands and results:

```text
source scripts/frontier/activate_emender_frontier.sh
$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py
8 passed in 21.01s

bash scripts/frontier/build_native_resilient_dataplane.sh
ctest: 10/10 passed

$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py \
  tests/test_native_dataplane_failure.py tests/test_native_pool_integration.py
33 passed in 28.70s

$EMENDER_PYTHON -m pytest -q tests/test_resilient_e97_runtime.py
28 passed (as part of the initial 61-test run)
```

The new tests cover double-buffer ownership and bounded exhaustion, latest-only
replacement, stale/partial/corrupt/non-finite rejection, delayed quorum without
foreground waiting, missing-owner abort/release, new-incarnation rejoin, wrong
fence/base rejection, and death before checkpoint CAS. Existing compiled tests
continue to cover native replay, checksum and finite rejection, fixed memory
bounds, owner transport, local fanout, fenced checkpoint commit, and all eight
trainers.

## Two-node gate status

No new Slurm job was submitted from this worktree. At the required submission
checkpoint, `squeue` showed job 5035221 (`validate-native-pool-32n-clean-stable`)
already running for this user. The authoritative two-node launcher explicitly
refuses overlapping allocations, and also requires submission from a clean
`main` checkout. Consequently, this commit does **not** claim the requested
five-generation two-node performance/failure gate, foreground idle below 10%,
or cadence at most 1.25x K40. Those remain required before scale-out; the
existing 32-node allocation is not evidence for this change and was neither
submitted nor modified here.

## Conformance checklist

- R01–R05, R08: fenced identities, replay-safe ownership, stale/corrupt
  rejection, and incarnation reset are explicit and tested.
- R06, R11–R13: waits and storage are bounded; a quorum miss publishes no
  partial result and does not block a trainer with a free slot.
- R07, NDP15: mailbox admission occurs only after the caller's durable commit
  verifier succeeds; pre-CAS death leaves no admissible result.
- R09–R10, NDP01–NDP14: this metadata-only layer leaves model ownership with
  trainers and dense transport/reduction/fanout with the compiled node service.
- R14–R16, NDP16–NDP17: queue high-water, replacement/rejection, apply, and
  foreground-wait counters are exposed. Local gates pass; the exact two-node
  retained timing/failure artifact is explicitly outstanding.
