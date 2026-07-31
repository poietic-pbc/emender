# Monitor production 64-node E97 B4/K80 chain: final pass

Task: `monitor-production-64-node`
Timestamp: 2026-07-03T22:46:00-04:00 / 2026-07-04T02:46:00+00:00

## Summary

This pass found no completed training run for the original production B4/K80
chain. The active monitored jobs from the updated scope both reached terminal
Slurm accounting states before starting:

- `4931004`, the original 64-node x 2h production job, is `CANCELLED by 19032`.
- `4932059`, the independent 96-node x 6h continuation job, is also
  `CANCELLED by 19032`.

Both jobs show `Start=None`, `Elapsed=00:00:00`, `AllocNodes=0`, and
`NodeList=None assigned`, so neither job allocated nodes, started training,
wrote logs, saved checkpoints, or advanced the production chain pointer.

The production `latest.pt` pointer now exists, but it is a manual chain
initialization to the verified seed checkpoint, not a checkpoint produced by
`4931004` or `4932059`.

No persistent hang is left unattended. `squeue` has no entries for `4931004`,
`4932059`, or current `e97-b4-k80` jobs.

## Scope Notes

The original task listed jobs `4931004` and `4931005`. Later operator/user
updates changed the active scope:

- `4931005` was cancelled after an operator mistake and should not be counted
  as a training failure.
- `4931947` and `4932047` were intentional replacement attempts and were later
  intentionally cancelled.
- The final active scope before this pass was `4931004` and independent
  `4932059`.
- A later operator update cancelled `4932059` at user request and initialized
  `latest.pt` from the seed checkpoint so latest-required jobs would not fail
  solely because the chain pointer was absent.

## Scheduler State

Observed command:

```bash
sacct -j 4931004,4932059 --format=JobID,JobName%40,Partition,Account,State,ExitCode,Elapsed,Submit,Eligible,Start,End,AllocNodes,NNodes,NodeList%50,Reason%40 -P
```

Results:

| Job | Role | State | ExitCode | Elapsed | Submit | Start | End | AllocNodes | NNodes | NodeList |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `4931004` | original 64n production chain job | `CANCELLED by 19032` | `0:0` | `00:00:00` | `2026-07-02T08:30:50` | `None` | `2026-07-03T11:37:24` | `0` | `64` | `None assigned` |
| `4932059` | independent 96n continuation | `CANCELLED by 19032` | `0:0` | `00:00:00` | `2026-07-02T12:56:18` | `None` | `2026-07-03T11:32:41` | `0` | `96` | `None assigned` |

`scontrol show job 4931004` and `scontrol show job 4932059` now return
`slurm_load_jobs error: Invalid job id specified`, which is expected once the
jobs are no longer present in live Slurm control state. Accounting data remains
available through `sacct`.

Current queue check:

```bash
squeue -j 4931004,4932059,4936017 -o '%i|%T|%M|%L|%D|%R|%S|%j'
```

Only unrelated follow-on work was present:

```text
4936017|PENDING|0:00|12:00:00|256|(Priority)|N/A|e97-b4-k40-256n12h
```

There were no `4931004`, `4932059`, or `e97-b4-k80` rows in the user's current
queue.

## Log Inspection

Expected logs:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.out
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.err
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-96n6h-4932059.out
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-96n6h-4932059.err
```

All four paths are absent. A job-id scan under the run log tree also found no
files containing `4931004` or `4932059`. This is consistent with both jobs being
cancelled before start with no allocated nodes.

Because no training logs exist, these metrics are not available for this
production chain attempt:

- token throughput
- final or last logged loss trend
- DiLoCo merge timing
- NCCL/RCCL collective/op/step diagnostics
- final checkpoint save line

No NCCL/RCCL timeout or recurrent instability was observed in these jobs; they
did not start.

## Checkpoint Pointer Evidence

Production chain latest path:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

Current chain directory:

```text
lrwxrwxrwx latest.pt -> /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
-rw-r--r-- latest.pt.manifest.json
```

`readlink -f latest.pt` resolves to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

Seed checkpoint stat:

```text
-rw------- erikgarrison bif148 7719679569 2026-06-30 16:04:37 -0400 checkpoint_step_489920_loss_2.4894.pt
```

Manifest summary:

```json
{
  "kind": "manual_chain_initialization",
  "checkpoint_step": 489920,
  "checkpoint_loss": 2.4894,
  "input_source_job": "4911454",
  "updated_at_utc": "2026-07-03T15:33:15+00:00"
}
```

The manifest states that the pointer was initialized manually after cancelling
`4932059`, with the reason:

```text
Initialize active production chain latest.pt to verified step-489920 seed so latest-required jobs do not fail if scheduled before first chained run completes.
```

Therefore, `latest.pt` is valid as a seed pointer, but it did not advance to a
new checkpoint from `4931004`.

## Validation Status

- Job `4931004` terminal state: satisfied as a scheduler terminal state
  (`CANCELLED by 19032`), but not as a completed training run.
- Job `4931004` logs inspected: expected log paths and job-id scan were
  inspected; no logs exist because the job never started.
- `4931004` final/latest checkpoint advancement: not satisfied; no
  `4931004` checkpoint exists and `latest.pt` points to the manually initialized
  seed checkpoint.
- Job `4932059` startup from production `latest.pt`: not applicable; accounting
  shows `Start=None` and no logs exist.
- Job `4932059` terminal state: satisfied as a scheduler terminal state
  (`CANCELLED by 19032`), but not as a completed training run.
- Job `4932059` logs inspected: expected log paths and job-id scan were
  inspected; no logs exist because the job never started.
- Token throughput, loss trend, DiLoCo merge timing, and checkpoint paths:
  unavailable for these jobs because neither started. The only checkpoint path
  to report is the seed checkpoint currently targeted by `latest.pt`.
- Hang handling: satisfied; no monitored job remains running or wedged.

## Final Assessment

The production B4/K80 chain did not produce training evidence in this attempt.
The correct preserved evidence is:

- `4931004` and `4932059` were cancelled before start.
- No logs or job-specific checkpoints were created.
- `latest.pt` exists and resolves to the verified step-489920 seed checkpoint.
- The chain pointer did not advance from `4931004`.
- There is no unattended running hang to cancel.
