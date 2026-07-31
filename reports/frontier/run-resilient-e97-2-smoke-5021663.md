# Resilient E97 changed-payload startup smoke — job 5021663

## Submission

This is a real Slurm submission, not `--test-only`. It was submitted at
`2026-07-17T17:17:17-04:00` from fetched authoritative commit
`89c0211dfae722d8b580d4dd631f599b492ea9ba`. The tracked worktree HEAD
equaled fetched `origin/main` at submission.

- Job: `5021663`
- Run: `run-resilient-e97-2-smoke-20260717T211649Z-89c0211`
- Payload: `89c0211-20260717T211649Z-startup-smoke-reuse-allocation-gpus`
- Run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T211649Z-89c0211`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

The exact executable command was retained as `exact-command.sh` before
submission. This unique payload includes the focused fix that lets nested
node-supervisor steps reuse the batch allocation's GPU device cgroup. It runs
one generation, has no failure injection, and is the mandatory short startup
smoke after the preceding pre-generation failure. Queue time and allocation
runtime are tracked separately. Pending queue state is not failure and will
not cause cancellation or resubmission.

## Pre-submit validation

The focused runtime, launcher, and node-transport matrix passed 33/33 in
92.91 seconds. Rendered production parity returned `ok=true`, with no
forbidden or missing fields. Compilation, shell syntax, and diff checks also
passed. Conformance is checked against *Resilient DiLoCo Compute Pool*,
version 1; applicable gap-matrix requirements are R02, R03, R04, R06, R08,
R09, R10, R14, and R16.

No full resilience-gate pass is claimed. Before a full `02:00:00` gate may be
submitted, job 5021663 must demonstrate both model-free managers, all sixteen
real GPU trainers, first heartbeat, network connectivity, and at least one
finalized generation.
