# Monitor Refreshed E97 Async DiLoCo 256n12h

Task: `monitor-refreshed-e97-async-256n12h`

Report type: **interim pre-start monitor**

Observation time: `2026-07-06T10:19:36Z`

## Summary

The refreshed E97 async DiLoCo 256-node 12-hour production job is still queued
and has not started. Slurm job `4946500` is `PENDING` with reason `Priority`;
`scontrol` reports projected start `2026-07-06T11:40:00` and projected end
`2026-07-06T23:40:00`.

No cancel condition has fired. Because the job is not allocated yet, it has
produced no stdout/stderr, no run root, no rendezvous/startup evidence, no
metrics JSON, no checkpoints, and no run-local latest file. Node-hours burned
so far are `0.0`, and observed production tokens are `0`.

## Slurm State

Commands checked:

```bash
squeue -j 4946500 -o '%i|%T|%M|%l|%D|%R|%S|%V'
sacct -j 4946500 --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End,NNodes,AllocCPUS,ReqTRES%80,AllocTRES%80 -P
scontrol show job 4946500
```

Observed `squeue` row:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4946500|PENDING|0:00|12:00:00|256|(Priority)|2026-07-06T11:40:00|2026-07-06T06:11:07
```

Observed `sacct` row:

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|NNodes|AllocCPUS|ReqTRES|AllocTRES
4946500|async-diloco-e97-256n12h|PENDING|0:0|00:00:00|Unknown|Unknown|256|0|billing=14336,cpu=14336,mem=125T,node=256|
```

Relevant `scontrol` fields:

- Job name: `async-diloco-e97-256n12h`
- State: `PENDING`
- Reason: `Priority`
- Account/QoS/partition: `bif148` / `normal` / `batch`
- Nodes: `256`
- Tasks: `256`
- CPUs: `14336`
- Time limit: `12:00:00`
- Submit time: `2026-07-06T06:11:07`
- Eligible time: `2026-07-06T06:11:07`
- Start time estimate: `2026-07-06T11:40:00`
- End time estimate: `2026-07-06T23:40:00`
- Stdout: `/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.out`
- Stderr: `/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.err`

## Startup And Rendezvous

Expected startup artifacts from the submit report were checked:

- Job-specific stdout: not present yet.
- Job-specific stderr: not present yet.
- Run root under `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/`: not present yet.
- `<run-root>/artifacts/env.txt`: not present yet.
- `<run-root>/artifacts/command.txt`: not present yet.
- `<run-root>/artifacts/async_diloco_e97_256n_metrics.json`: not present yet.

This is consistent with the job still being pending and not yet allocated. No
rendezvous verdict can be made until Slurm enters `RUNNING` and the wrapper
creates its run root.

## Metrics Monitor

No 256-node metrics JSON exists yet, so all runtime metrics are unavailable
rather than failed:

| Field | Expected source | Current observation |
| --- | --- | --- |
| Configured local quorum | Submit wrapper/env | `8` |
| Configured global quorum | Submit wrapper/env | `240` |
| Effective local quorum | Metrics JSON | not emitted yet |
| Effective global quorum | Metrics JSON | not emitted yet |
| Accepted updates | Metrics JSON | not emitted yet |
| Dropped/stale updates | Metrics JSON | not emitted yet |
| Timed-out updates | Metrics JSON | not emitted yet |
| Failed/invalid updates | Metrics JSON | not emitted yet |
| Loss windows | Metrics JSON | not emitted yet |
| Tokens/sec | Metrics JSON | not emitted yet |
| Checkpoint overhead | Metrics JSON | not emitted yet |

The monitor confirmed the expected metrics schema from prior async E97
32-node and 64-node metrics artifacts, including `effective_quorum`,
`update_counts`, `global_generation_metrics`, `checkpoint_finalization`,
`loss_moving_average`, and checkpoint overhead fields. Those reference artifacts
are historical context only; they are not evidence for job `4946500`.

## Latest And Finalization Behavior

Production latest policy from the submit package:

- Run-local `async_run/latest.json` may advance after the 256-node run starts.
- External production chain `latest.pt` is guard-only and must not change.
- Guard path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

Current guard observation:

```text
path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
size=231
mtime=2026-07-03 11:32:54.000000000 -0400
sha256=6cdca7edcf208d96c99f7be5f58b996216eba81de553975c58f04a5bdb5563a3
```

No run-local `async_run/latest.json` exists yet for job `4946500`, which is
expected before startup. No production latest/finalization failure is observed.

## Accounting

- Requested node-hours: `3072.0` (`256` nodes x `12` hours).
- Slurm elapsed time: `00:00:00`.
- Allocated nodes: `0` so far.
- Consumed node-hours from this observation: `0.0`.
- Checkpoints produced by job `4946500`: `0`.
- Production tokens observed: `0`.
- Planned token geometry: `16777216` tokens/local step; `40` local steps per
  DiLoCo generation; `671088640` tokens/node/generation; `171798691840`
  aggregate tokens per 256-node generation.

Because the job has not started, no measured generation count or tokens/sec is
available. Token accounting should switch from `0` to metrics-derived counts
after the first metrics JSON is emitted.

## Cancel Decision

Decision: **do not cancel at this observation**.

Evidence:

- The job is queued normally as `PENDING(Priority)`.
- Slurm still has an estimated start time and scheduled node list.
- No node-hours have been consumed.
- No stdout/stderr, run root, metrics, or latest artifacts are expected before
  allocation.
- The external production latest guard exists and has not been observed
  changing during this pre-start monitor pass.

Cancel criteria from the submit report remain active for the next pass:

- Cancel if the started run records a wrong seed path or resolved checkpoint.
- Cancel if the async entrypoint is missing or not
  `scripts/frontier/async_diloco_e97_multinode.py`.
- Cancel if local quorum is below `8`, global quorum is below `240`, or stale,
  timeout, failed, or invalid counts indicate unsafe progress.
- Cancel if the production latest guard changes.
- Cancel if checkpoint finalization or run-local latest advancement fails.
- Cancel if material node-hours are burned without stdout/stderr, run-root
  artifacts, or metrics.

## Next Recommendation

Continue monitoring after the projected Slurm start time
`2026-07-06T11:40:00`. The next monitor pass should:

1. Re-check `squeue`, `sacct`, and `scontrol` for state transition and elapsed
   time.
2. Tail stdout/stderr for wrapper startup and entrypoint/rendezvous messages.
3. Locate the `4946500-<UTC stamp>Z` run root and read `artifacts/env.txt` and
   `artifacts/command.txt`.
4. Parse `artifacts/async_diloco_e97_256n_metrics.json` once emitted and record
   effective quorum, update counts, loss windows, tokens/sec, checkpoint
   overhead, checkpoint paths, and latest/finalization status.
5. Re-stat and re-hash the production latest guard.
6. Compute node-hours as `256 * elapsed_hours` once allocated and derive tokens
   from emitted generation metrics.
