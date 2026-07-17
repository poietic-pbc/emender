# Resilient E97 changed-payload startup smoke — job 5021768

## Submission

This is a real Slurm submission, not `--test-only`, submitted at
`2026-07-17T17:54:49-04:00` from fetched authoritative commit
`c2afd316e2d240bfcc4c84a9091caf103545842e`. Tracked `HEAD` equaled fetched
`origin/main` before submission.

- Job: `5021768`
- Run: `run-resilient-e97-2-smoke-20260717T215430Z-c2afd31`
- Payload: `c2afd31-20260717T215430Z-startup-smoke-inherit-allocation-gpus`
- Run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T215430Z-c2afd31`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`, eight GPUs per node
- Seed: pinned step 1525000, SHA256 `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

The exact executable command was retained as `exact-command.sh` before the
submission. This unique changed payload follows the pre-generation failure of
job 5021737. It keeps the allocation-level GPU request but removes the repeated
step-level GRES request, letting the single two-node supervisor step inherit
the allocation device cgroup. The smoke requests one finalized generation and
has no fault injection.

## Validation and gate state

The focused ROCm runtime, launcher, and node-transport matrix passed 34/34 in
86.25 seconds. Rendered production parity returned `ok=true` with no forbidden
or missing fields. Python compilation, shell syntax, and `git diff --check`
passed. Conformance was checked against *Resilient DiLoCo Compute Pool*,
version 1; applicable gap-matrix requirements are R02, R03, R04, R06, R08,
R09, R10, R14, and R16.

No full resilience-gate pass is claimed. Job 5021768 must prove both model-free
managers, all sixteen real GPU trainers, first heartbeat, network connectivity,
and at least one finalized generation before any full `02:00:00` gate can be
submitted. Queue time and runtime are recorded separately, and pending queue
state is not a failure or a reason to retry.

## Result and diagnosis

Job 5021768 queued for 36 minutes 16 seconds, from `2026-07-17T17:54:49-04:00`
until `2026-07-17T18:31:05-04:00`, then ran for 14 minutes 38 seconds. It failed
before any role heartbeat or finalized generation. The only application output
was the expected topology declaration; Slurm repeatedly reported
`Requested nodes are busy` for the single two-node supervisor step and
terminated the batch step at `2026-07-17T18:45:43-04:00` with state `FAILED`
and exit code `0:15`.

The changed payload correctly stopped re-requesting GPU GRES, but the step
still requested `-c64` on each Frontier node. Accounting shows this allocation
contained 112 CPUs total, exactly 56 per node, so a 64-CPU-per-task step could
never be admitted. The next changed payload requests the allocation's actual
56 Slurm-visible CPUs per node. Per the queue-efficiency directive, no full
gate or unchanged retry follows this pre-generation failure; another short
two-node startup smoke is required first.
