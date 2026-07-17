# Resilient E97 changed-payload startup smoke — job 5021992

## Submission

This is a real Slurm submission, not `--test-only`, submitted at
`2026-07-17T19:01:52-04:00` from fetched authoritative commit
`2b7e4a41eff08558c84c3a262e56f0955709eeb5`. Tracked `HEAD` equaled fetched
`origin/main` before submission.

- Job: `5021992`
- Run: `run-resilient-e97-2-smoke-20260717T230012Z-2b7e4a4`
- Payload: `2b7e4a4-20260717T230012Z-startup-smoke-valid-56cpu-step`
- Run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T230012Z-2b7e4a4`
- Immediate state at `2026-07-17T19:02:04-04:00`: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`, eight GPUs per node
- Failure injection: none
- Seed: pinned step 1525000, independently verified SHA256 `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

The exact executable command was retained as `exact-command.sh` before the
submission. This unique payload follows the pre-generation failure of job
5021768 and changes its unsatisfiable 64-CPU-per-node supervisor step to the
56 CPUs per node shown by Slurm accounting. It is the required short startup
smoke before any full `02:00:00` resilience gate; no duplicate or full job was
submitted.

## Validation and architecture conformance

The focused runtime/launcher rerun passed 24/24. The complete focused matrix
passed 71/72 on its first pass with one timing-sensitive aggregate-apply test;
that runtime suite passed on immediate focused rerun. Rendered production
parity returned `ok=true` with no forbidden or missing fields. Python
compilation, shell syntax, and `git diff --check` passed.

Conformance was checked against *Resilient DiLoCo Compute Pool*, version 1.
Applicable gap-matrix requirements are R02, R03, R04, R06, R08, R09, R10,
R14, and R16. This smoke must show two model-free managers, sixteen real GPU
trainers, bounded startup/heartbeat/generation deadlines, node-local/network
bulk transport rather than a Lustre hot path, and at least one immutable
finalized generation. It does not by itself satisfy the later failure,
TERM-checkpoint, or fresh-allocation gates.
