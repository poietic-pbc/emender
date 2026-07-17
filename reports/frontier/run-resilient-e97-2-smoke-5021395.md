# Resilient E97 changed-payload startup smoke — job 5021395

Real Slurm submission (not `--test-only`) at `2026-07-17T20:37:27Z`:

- job: `5021395`
- run: `run-resilient-e97-2-smoke-20260717T203727Z-a9b8da6`
- payload: `a9b8da6-20260717T203727Z-startup-smoke-node-gpu-resource`
- fetched authoritative code: `a9b8da691c205c2e5b19c89c4b9f44117a988b44`
- run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T203727Z-a9b8da6`
- immediate state: `PENDING/Priority`; queue clock begins at submission and
  runtime remains `00:00:00`
- resources: exactly 2 nodes, debug QoS, `00:20:00`; no injection

The exact executable command is retained as `exact-command.sh` in the run
directory. It binds the approved step-1525000 seed and SHA256 identity,
production train-argument JSON and CommaPile data, 40 local steps, node-local
bulk root, bounded startup/heartbeat/progress/generation deadlines, and the
unique run/payload/code identities. This required smoke must prove all 18
roles, network connectivity, and one finalized generation before a full
`02:00:00` gate can be submitted.

## Validation

Conformance was checked against *Resilient DiLoCo Compute Pool*, version 1.
Applicable gap-matrix requirements: R02, R03, R04, R06, R08, R09, R10, R14,
and R16. The changed resource-shape launcher suite passed 17/17; rendered
parity returned `ok=true` with no forbidden or missing fields; compileall,
shell syntax, and diff checks passed. No acceptance-gate pass is claimed.
