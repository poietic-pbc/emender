# Resilient E97 changed-payload startup smoke — job 5021314

## Terminal result

- Queue time: `00:00:53` (`Submit=2026-07-17T16:21:27`,
  `Start=2026-07-17T16:22:20`, America/New_York).
- Runtime before the documented startup fail-fast: `00:11:07`; cancelled at
  `2026-07-17T16:33:27` only after two complete 300-second startup-deadline
  intervals produced zero manager/trainer heartbeats and zero finalized
  generations.
- Both node-supervisor steps remained pending with `Requested nodes are busy`.
  The preserved supervisor event stream records startup-deadline eviction at
  300 seconds and again at 600 seconds. No full-gate pass is claimed.
- Root cause: the Frontier node-supervisor step requested all eight GPUs with
  task-level `--gpus-per-task=8`; Frontier's node-level GCD allocation requires
  `--gpus-per-node=8`. The changed payload is covered by
  `test_node_supervisor_requests_frontier_gpus_as_node_resources` before the
  next unique startup smoke.

Submitted at `2026-07-17T20:21:27Z` from fetched authoritative commit
`cec0abceb0a1d1d5b50a615e2375f07c3421c6c1`. This is a real Slurm
submission, not `--test-only`. Immediate state was `PENDING (Priority)`.
Queue time and runtime are accounted separately.

The job is restricted to exactly two Frontier nodes, debug QoS, and a
`00:20:00` startup-smoke walltime. It must produce all 18 role starts,
network connectivity, and at least one finalized generation before another
full `02:00:00` gate may be submitted. It has no fault injection.

Identities:

- job: `5021314`
- run: `run-resilient-e97-2-smoke-20260717T202127Z-cec0abc`
- payload: `cec0abc-20260717T202127Z-startup-smoke-node-local-supervision`
- run directory: `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T202127Z-cec0abc`
- seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

Exact command (line-wrapped only for readability):

```bash
sbatch -N 2 -q debug -t 00:20:00 \
  --export=ALL,RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T202127Z-cec0abc,RESILIENT_E97_RUN_ID=run-resilient-e97-2-smoke-20260717T202127Z-cec0abc,RESILIENT_E97_SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9,RESILIENT_E97_PAYLOAD_ID=cec0abc-20260717T202127Z-startup-smoke-node-local-supervision,RESILIENT_E97_CODE_ID=cec0abceb0a1d1d5b50a615e2375f07c3421c6c1,RESILIENT_E97_SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt,RESILIENT_E97_TRAIN_ARGS_JSON=/lustre/orion/bif148/scratch/erikgarrison/emender/configs/frontier/e97_async_256_job4962400_golden.json,RESILIENT_E97_DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,RESILIENT_E97_GENERATIONS=1,RESILIENT_E97_STARTUP_SMOKE=1,RESILIENT_E97_REQUESTED_WALLTIME=00:20:00,RESILIENT_E97_LAUNCH_MODE=node-local,RESILIENT_E97_STARTUP_DEADLINE_S=300,RESILIENT_E97_HEARTBEAT_DEADLINE_S=60,RESILIENT_E97_PROGRESS_DEADLINE_S=900,RESILIENT_E97_GENERATION_DEADLINE_S=900,RESILIENT_E97_BULK_ROOT=/tmp/resilient-e97,RESILIENT_E97_MAX_RESTARTS=2 \
  scripts/frontier/resilient_e97_true_2n.sbatch
```

## Validation

Conformance is checked against *Resilient DiLoCo Compute Pool*, version 1.
Applicable gap-matrix requirements are R02, R03, R04, R06, R08, R09, R10,
R14, and R16. The pre-submit focused runtime/topology/transport/launcher matrix
passed 32 tests in 94.58 seconds. Rendered production parity returned
`ok=true`, with no forbidden or missing fields. Compile, shell syntax, and
diff checks passed.

The startup smoke is not an acceptance-gate pass. Final scheduler accounting,
role events, finalized-generation evidence, and the next state-machine action
will be appended only after the job leaves the queue and completes.
