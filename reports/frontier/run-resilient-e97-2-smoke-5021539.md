# Resilient E97 changed-payload startup smoke — job 5021539

## Submission

This is a real Slurm submission, not `--test-only`. It was submitted at
`2026-07-17T16:55:17-04:00` from fetched authoritative commit
`2d47a46e249d95cc52129b9994a72af5a4d29b26`. The worktree's tracked HEAD
equaled `origin/main` at submission.

- Job: `5021539`
- Run: `run-resilient-e97-2-smoke-20260717T205515Z-2d47a46`
- Payload: `2d47a46-20260717T205515Z-startup-smoke-allocation-gpu-request`
- Run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T205515Z-2d47a46`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

The exact command was rendered and retained as `exact-command.sh` in the run
directory before submission. The payload requests the Frontier allocation's
eight GPUs per node and is unique relative to failed smoke 5021395. It runs
one generation with no failure injection. No full resilience-gate pass is
claimed.

## Required smoke result

Before a full `02:00:00` failure gate may be submitted, this allocation must
demonstrate both model-free managers, all sixteen real GPU trainers, first
heartbeat and network connectivity, and at least one finalized generation.
Queue time and runtime will be recorded separately. Pending queue state is not
a failure and will not trigger cancellation or resubmission.

## Validation

Conformance is checked against *Resilient DiLoCo Compute Pool*, version 1.
Applicable gap-matrix requirements are R02, R03, R04, R06, R08, R09, R10,
R14, and R16. The preceding changed payload passed the focused launcher suite,
rendered parity, Python compilation, shell syntax, and diff checks. Final
scheduler accounting and generation evidence will be appended after the job
terminates.

## Terminal result

The job started at `2026-07-17T16:56:24-04:00` and ended at
`2026-07-17T17:11:00-04:00`; queue time was `00:01:07` and allocation runtime
was `00:14:36`.  Slurm reports `COMPLETED` for the batch wrapper, but the smoke
gate **failed before any manager/trainer heartbeat or finalized generation**.
Both node-supervisor steps repeatedly reported `Requested nodes are busy` and
were evicted at the bounded 300-second startup deadline.  Their retry steps
were blocked identically until TERM handoff.  The retained `events.jsonl`
contains only node-supervisor starts/evictions; there are no manager/trainer,
network-connectivity, or generation-finalization events.

The concrete diagnosis is that payload `2d47a46` requested the allocation's
eight GPUs again on each overlapping node-supervisor step.  Frontier does not
share that GRES allocation between the two nested steps.  The next payload is
therefore changed by commit `34d73e5b21a732a7d52fc95596e17a87153a235e`,
which reuses the batch allocation's device cgroup while each direct trainer
still binds `ROCR_VISIBLE_DEVICES=0..7`.  Per the queue-efficiency directive,
another full two-hour gate remains forbidden until that changed payload passes
the short startup smoke.
