# Cancellation record: production job 4980157

Task: `cancel-unsafe-pending`

Date: 2026-07-15

## Decision and scope

The explicitly authorized cancellation of the pending 256-node, 12-hour
production job `4980157` is complete. The safety reason is that debug job
`5000436`, which exercised the same core 256-node path, stalled after
generation 0. Production remains blocked pending successful progressive scale
validation. No replacement is authorized or submitted.

This task changed scheduler state only for job `4980157`. The first worker for
this WG task started at `2026-07-15T07:27:36Z`; Slurm records the cancellation
at `2026-07-15T03:28:54-04:00` (`2026-07-15T07:28:54Z`). When the recovery
worker began at `2026-07-15T07:29:24Z`, the job was already terminal, so it did
not issue a duplicate `scancel`. There was one authorized cancellation, with no
submission, requeue, hold, release, update, or cancellation of any other job.

## Scheduler history

The retained initial queue snapshot in launch evidence records:

```text
JOBID   NAME                         QOS     STATE    NODES  TIME  TIME_LIMIT  START_TIME  NODELIST(REASON)
4980157 async-b4k40-ladder-256n     normal  PENDING  256    0:00  12:00:00   N/A         (Priority)
```

The last pre-cancel monitor at `2026-07-15T06:17Z` still recorded the job as
pending. The post-cancel read-only queries returned no row from `squeue` and
the following job record from `sacct -X`:

```text
JobIDRaw|JobName|User|Account|Partition|State|Reason|Submit|Eligible|Start|End|Elapsed|Timelimit|NNodes|AllocNodes|AllocTRES|ReqTRES|ExitCode
4980157|async-b4k40-ladder-256n|erikgarrison|bif148|batch|CANCELLED by 19032|None|2026-07-13T08:06:54|2026-07-13T08:06:54|None|2026-07-15T03:28:54|00:00:00|12:00:00|256|0||billing=14336,cpu=14336,mem=125T,node=256|0:0
```

This is terminal `CANCELLED` state. Accounting is also affirmative evidence
that the job never entered `RUNNING` and consumed no new allocation:
`Start=None`, `Elapsed=00:00:00`, `AllocNodes=0`, and empty `AllocTRES`.
`scontrol` agrees on `JobState=CANCELLED`, `RunTime=00:00:00`, and
`AllocTRES=(null)`. Its synthesized `StartTime` equals `EndTime`; the accounting
record above is authoritative for the never-started determination.

## Immutable launch identity and evidence

The production launch remains preserved in Git commit `fcdc37d` (descended
from approved evidence commit `9ac7fe2`). The attested execution-tree commit
is `9fff689c9f9252b6a264773c207f8f8ca8509666`; the reviewed parity fingerprint
is `a4a493eb60c6425f3df2dea71436ded0b43fee282de6f2bac87a0a49e4f0ad5b`.
The pinned seed hash remains
`1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.

The immutable production tree includes the render, manifest, approval,
submission-attempt marker, initial scheduler snapshots, all queue-monitor
snapshots through `monitor-20260715T0617Z.txt`, and submission identity. Key
Git blob identities verified after cancellation are:

```text
bb32764ba60655655ea9127f660c66a0009bf4a7 rendered.sbatch
2f68cc6fc17f5879f205e6922ab573e43608b225 golden-manifest.json
c7f886e1972b4b87a44d027fc9601eca93f2e3d9 launch-inputs.json
586bcbcfee2c510dd932d695592ba42cdaa38420 reviewed-approval.json
2d03f9737b7a388021bfcef5a7549f477c476123 evidence/submission-attempt.json
f5133a9fad4ebf385931acf48aa24f57515228e3 evidence/squeue-initial.txt
907ee029b0cf589aca3d40f01086a631ea5f2ca7 evidence/sacct-initial.txt
836a034c4cba201f45ae0286e785498b8861e954 evidence/monitor-20260715T0605Z.txt
e830e1b33d64cc25c311ed9eb0cca4bdb769ab37 evidence/monitor-20260715T0617Z.txt
```

No immutable launch, render, attestation, approval, submission, or queue-history
artifact was edited during cancellation. This report adds the durable
cancellation and terminal-accounting record alongside those artifacts.

## Operational gate

Production is blocked pending successful progressive scale validation. In
particular, the stalled 256-node debug path must be repaired and revalidated
through the approved scale ladder before any new production submission is
considered. This cancellation does not authorize a replacement.
