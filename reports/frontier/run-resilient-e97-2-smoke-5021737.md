# Resilient E97 changed-payload startup smoke — job 5021737

## Submission

This is a real Slurm submission, not `--test-only`. It was submitted at
`2026-07-17T17:38:17-04:00` from fetched authoritative commit
`2f0b6c6d9fa5e52df46245546eb452e17beeff19`. Tracked `HEAD` equaled fetched
`origin/main` before submission.

- Job: `5021737`
- Run: `run-resilient-e97-2-smoke-20260717T213815Z-2f0b6c6`
- Payload: `2f0b6c6-20260717T213815Z-startup-smoke-single-gpu-step`
- Run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T213815Z-2f0b6c6`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

The exact executable command was retained as `exact-command.sh` before
submission. This payload is unique and contains commit `2f0b6c6`, which launches
both node-local supervisors in one GPU-bearing two-node Slurm step. It follows
the pre-generation failure of smoke 5021663 and is the required short startup
smoke before any full two-hour resilience gate. It requests one finalized
generation and has no failure injection.

## Pre-submit validation

The focused runtime, launcher, and node-transport matrix passed 34/34. The
rendered production parity check returned `ok=true`, with no forbidden or
missing fields. Python compilation, shell syntax, diff checks, and the pinned
seed SHA256 check passed. Conformance is checked against *Resilient DiLoCo
Compute Pool*, version 1; applicable gap-matrix requirements are R02, R03, R04,
R06, R08, R09, R10, R14, and R16.

No full resilience-gate pass is claimed. Before a full `02:00:00` gate may be
submitted, job 5021737 must demonstrate both model-free managers, all sixteen
real GPU trainers, first heartbeat, network connectivity, and at least one
finalized generation. Queue time and allocation runtime are recorded
separately; pending queue state is not failure and will not trigger a retry.
