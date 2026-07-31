# Continue Monitor Refreshed E97 Async DiLoCo 256n12h

Task: `continue-monitor-refreshed`

Report type: **continued pre-start monitor**

Observation time: `2026-07-06T10:30:07Z` (`2026-07-06T06:30:07-0400` on the login node)

## Summary

Slurm job `4946500` has not started yet. At this observation it is still
`PENDING` with reason `Priority`, elapsed time `00:00:00`, allocated nodes `0`,
and projected Slurm start `2026-07-06T11:40:00`. The job has therefore burned
`0.0` node-hours so far.

No cancel condition has fired. Because the job is not allocated, there is still
no job stdout/stderr, no run root, no env or command artifact, no metrics JSON,
no checkpoints, and no run-local latest/finalization evidence. The external
production latest guard is present and its stat/hash match the prior monitor
observation.

Decision: **continue waiting; do not cancel**.

## Slurm State

Commands checked:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ
date +%Y-%m-%dT%H:%M:%S%z
squeue -j 4946500 -o '%i|%T|%M|%l|%D|%R|%S|%V|%u|%j'
sacct -j 4946500 --format=JobID,JobName%40,State,Elapsed,Start,End,NNodes,AllocNodes,ReqNodes,NodeList%80,ExitCode -P
scontrol show job 4946500
```

Observed `squeue` row:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME|USER|NAME
4946500|PENDING|0:00|12:00:00|256|(Priority)|2026-07-06T11:40:00|2026-07-06T06:11:07|erikgarrison|async-diloco-e97-256n12h
```

Observed `sacct` row:

```text
JobID|JobName|State|Elapsed|Start|End|NNodes|AllocNodes|ReqNodes|NodeList|ExitCode
4946500|async-diloco-e97-256n12h|PENDING|00:00:00|Unknown|Unknown|256|0|256|None assigned|0:0
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
- Last scheduler evaluation: `2026-07-06T06:30:02`
- Scheduler: `Backfill:*`
- Requested TRES: `cpu=14336,mem=125T,node=256,billing=14336`
- Allocated TRES: `(null)`
- Stdout: `/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.out`
- Stderr: `/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.err`
- Command: `/lustre/orion/bif148/scratch/erikgarrison/emender/scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`

The scheduled node list remains populated in `scontrol`, but `NodeList` is
empty and `AllocTRES` is null, confirming that the allocation has not begun.

## Startup And Rendezvous Evidence

Expected startup artifacts from
`reports/frontier/submit-refreshed-e97-async-256n12h-20260706.md` were checked.

Log files:

```text
ls: cannot access 'logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.out': No such file or directory
ls: cannot access 'logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.err': No such file or directory
```

Run-root search:

```bash
find /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706 -maxdepth 2 -type d -name '4946500-*' -print
```

Result: no directories found.

Artifact search:

```bash
find /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706 -maxdepth 4 \( -name 'async_diloco_e97_256n_metrics.json' -o -name 'latest.json' -o -name 'env.txt' -o -name 'command.txt' \) -print
```

Result: no files found.

Because the job remains pending, there is no startup or rendezvous evidence to
judge yet. This is expected for a job that has not received an allocation.

## Metrics JSON

Metrics path expected after startup:

```text
<run-root>/artifacts/async_diloco_e97_256n_metrics.json
```

No run root or metrics JSON exists for job `4946500`, so metrics are unavailable
rather than failing.

| Metric | Current observation |
| --- | --- |
| Effective local quorum | not emitted |
| Effective global quorum | not emitted |
| Accepted update count | not emitted |
| Stale/drop count | not emitted |
| Timeout count | not emitted |
| Failure/invalid count | not emitted |
| Loss windows | not emitted |
| Tokens/sec | not emitted |
| Checkpoint overhead | not emitted |

Configured expectations from the submit report remain:

- Local quorum: `8`
- Global quorum: `240`
- Worker count per node: `8`
- DiLoCo K: `40`
- Tokens per local step: `16777216`
- Tokens per node per generation: `671088640`
- Aggregate tokens per 256-node generation: `171798691840`

## Checkpoints And Finalization

Checkpoint and manifest search:

```bash
find /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706 -maxdepth 5 \( -name '*checkpoint*' -o -name '*manifest*' \) -print
```

Result: no files found.

Expected cadence after startup:

- Generation manifests: every generation
- Recovery checkpoint: every `5` generations or `600` seconds, whichever comes first
- Export checkpoint: every `45` generations or `3600` seconds, whichever comes first
- Finalization buffer: `1200` seconds

No checkpoint cadence or finalization behavior can be evaluated until the run
root exists and the job emits artifacts.

## Latest Guard And Run-Local Latest

Production latest policy from the submit report:

- Run-local `async_run/latest.json` may advance after the 256-node run starts.
- External production chain `latest.pt` is guard-only and must not change.
- Guard path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

Current guard observation:

```text
path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt size=231 mtime=2026-07-03 11:32:54.000000000 -0400
6cdca7edcf208d96c99f7be5f58b996216eba81de553975c58f04a5bdb5563a3  /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

This matches the prior monitor observation:

- Size: `231`
- Mtime: `2026-07-03 11:32:54.000000000 -0400`
- SHA256: `6cdca7edcf208d96c99f7be5f58b996216eba81de553975c58f04a5bdb5563a3`

No run-local `async_run/latest.json` exists yet for job `4946500`, which is
expected before startup.

## Accounting

- Requested node-hours: `3072.0` (`256` nodes x `12` hours)
- Slurm elapsed time: `00:00:00`
- Allocated nodes from `sacct`: `0`
- Consumed node-hours at this observation: `0.0`
- Checkpoints produced by job `4946500`: `0`
- Observed production tokens: `0`
- Measured tokens/sec: unavailable until metrics JSON is emitted

Token accounting should switch from `0` to metrics-derived generation counts
after the first metrics JSON is written.

## Cancel Or Continue Decision

Decision: **continue waiting; do not cancel**.

Direct evidence:

- Job `4946500` is still queued as `PENDING(Priority)`.
- Elapsed time is `00:00:00`.
- Allocated nodes are `0`.
- Consumed node-hours are `0.0`.
- Stdout/stderr have not been created, which is expected before allocation.
- No run root, env, command, metrics, checkpoint, or run-local latest artifacts
  exist, which is expected before allocation.
- The external production latest guard is present and unchanged from the prior
  monitor hash/stat.

Cancel criteria remain armed for the next pass:

- Cancel if the started run records a wrong seed path or resolved checkpoint.
- Cancel if the async entrypoint is missing or is not
  `scripts/frontier/async_diloco_e97_multinode.py`.
- Cancel if local quorum is below `8`, global quorum is below `240`, or stale,
  timeout, failed, invalid, or dropped update counts indicate unsafe progress.
- Cancel if the production latest guard changes.
- Cancel if checkpoint finalization or run-local latest advancement fails.
- Cancel if material node-hours are burned without stdout/stderr, run-root
  artifacts, or metrics.

## Next Polling Recommendation

Poll again after the projected Slurm start time visible in the latest `scontrol`
snapshot: `2026-07-06T11:40:00` local Frontier time. A practical next check is
`2026-07-06T15:45:00Z` (`2026-07-06T11:45:00-0400`) to allow several minutes
for allocation, wrapper startup, and run-root creation.

The next monitor pass should:

1. Re-check `squeue`, `sacct`, and `scontrol` for transition to `RUNNING` or a
   revised pending reason/start estimate.
2. Tail the job stdout/stderr paths if they exist.
3. Locate the `4946500-<UTC stamp>Z` run root.
4. Read `<run-root>/artifacts/env.txt` and `<run-root>/artifacts/command.txt`.
5. Parse `<run-root>/artifacts/async_diloco_e97_256n_metrics.json` if emitted.
6. Inspect checkpoint cadence, run-local `async_run/latest.json`, and
   finalization state.
7. Re-stat and re-hash the production latest guard.
8. Recompute node-hours and token accounting from Slurm elapsed time and
   metrics-derived generation counts.
