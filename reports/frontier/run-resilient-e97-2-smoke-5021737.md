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

## Terminal outcome and changed payload

Job 5021737 started at `2026-07-17T17:39:22-04:00` after `00:01:05` queued.
It was cancelled at `2026-07-17T17:50:59-04:00` after `00:11:37` runtime and
zero role heartbeats or finalized generations. The single node-supervisor
`srun` repeatedly reported `Requested nodes are busy`.

While the allocation was still assigned, this exact diagnostic launched on
both nodes immediately and returned their hostnames:

```bash
timeout 20s srun --jobid 5021737 --overlap --no-kill --exact \
  -N2 -n2 --ntasks-per-node=1 -c1 hostname
```

That isolated the redundant step-level `--gpus-per-node=8` request: the batch
allocation already owned eight GPUs per node, and asking for the same GRES on
the child step blocked it. The changed payload retains allocation-level
`#SBATCH --gpus-per-node=8`, but the one two-node supervisor step now inherits
that device cgroup without requesting GRES again. This is a pre-generation
failure, so no checkpoint is eligible for restart and no unchanged retry is
allowed.
