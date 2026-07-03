# Follow-up Recheck: 256-node K40 E97 probe

Task: `follow-up-recheck-256`
Job: `4936017` (`e97-b4-k40-256n12h`)
Checked: `2026-07-03T14:45:54+00:00` (`2026-07-03T10:45:54-04:00`)

## Summary

The job has not started. It remains `PENDING` for Slurm priority, with
`Start=Unknown` in `sacct` and an estimated `squeue`/`scontrol` start time of
`2026-07-04T03:54:00`.

The required input chain pointer has not been restored:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

Both the input chain directory and output chain directory are absent. Because
the job has not allocated yet, no startup failure has occurred in this recheck,
but the same pre-start `RESUME_CHECKPOINT` blocker remains. If the path is still
missing at allocation time, the submitted wrapper is expected to exit before
training with the unreadable checkpoint check.

No job submit or cancel action was performed during this recheck.

## Slurm State

Timestamp: `2026-07-03T10:45:27-04:00`

`squeue -j 4936017 -o '%i|%T|%M|%l|%D|%R|%S|%V|%u|%j' --noheader`

```text
4936017|PENDING|0:00|12:00:00|256|(Priority)|2026-07-04T03:54:00|2026-07-03T10:32:31|erikgarrison|e97-b4-k40-256n12h
```

`sacct -j 4936017 --format=JobID,JobName,Partition,Account,AllocNodes,State,ExitCode,Submit,Eligible,Start,End,Elapsed,Timelimit -P`

```text
JobID|JobName|Partition|Account|AllocNodes|State|ExitCode|Submit|Eligible|Start|End|Elapsed|Timelimit
4936017|e97-b4-k40-256n12h|batch|bif148|0|PENDING|0:0|2026-07-03T10:32:31|2026-07-03T10:32:31|Unknown|Unknown|00:00:00|12:00:00
```

Relevant `scontrol show job 4936017` fields:

```text
JobState=PENDING Reason=Priority
RunTime=00:00:00 TimeLimit=12:00:00
SubmitTime=2026-07-03T10:32:31 EligibleTime=2026-07-03T10:32:31
StartTime=2026-07-04T03:54:00 EndTime=2026-07-04T15:54:00
NumNodes=256-256 NumTasks=2048
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.err
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.out
```

## Pointer Checks

Required input pointer:

```text
ls -l /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
ls: cannot access '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt': No such file or directory
```

Input chain directory:

```text
ls -ld /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg
ls: cannot access '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg': No such file or directory
```

Output pointer:

```text
ls -l /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h/latest.pt
ls: cannot access '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h/latest.pt': No such file or directory
```

Output chain directory:

```text
ls -ld /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h
ls: cannot access '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h': No such file or directory
```

No matching `*step489920*` chain directory was found under
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains` in this pass.

## Log Checks

The Slurm-recorded stdout/stderr paths do not exist yet:

```text
ls -l /lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.out
ls: cannot access '/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.out': No such file or directory

ls -l /lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.err
ls: cannot access '/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.err': No such file or directory
```

`find ... -name '*4936017*' -ls` under the scaleout log directory returned no
entries.

Because no logs exist and the job is still pending:

- Missing input startup message: not yet present in logs.
- Loss blowup: not assessable.
- Non-finite loss: not assessable.
- RCCL/NCCL failures: not assessable.
- Checkpoint/finalization: not assessable.
- Token accounting: actual processed tokens remain `0`.

## Validation Coverage

- `squeue` and `sacct` state recorded with timestamp.
- Required input pointer checked and recorded as missing.
- Output pointer checked and recorded as missing.
- Slurm-recorded log paths checked; logs absent because job has not started.
- No submit/cancel action performed.
- Findings recorded in this artifact and logged to WG.
