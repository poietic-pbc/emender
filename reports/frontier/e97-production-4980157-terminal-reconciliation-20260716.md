# E97 production job 4980157 terminal reconciliation

Task: `deploy-repaired-e97-256`

Observed at `2026-07-16T10:33:37-04:00`. This was a read-only reconciliation
after the explicitly authorized cancellation recorded in
`cancel-production-job4980157-20260715.md`. No Slurm mutation or submission was
performed during this pass.

## Production terminal state

Job `4980157` is absent from `squeue`. The allocation-only accounting record is:

```text
JobIDRaw|JobName|State|ExitCode|Submit|Eligible|Start|End|Elapsed|Timelimit|QOS|Partition|ReqNodes|AllocNodes|Reason
4980157|async-b4k40-ladder-256n|CANCELLED by 19032|0:0|2026-07-13T08:06:54|2026-07-13T08:06:54|None|2026-07-15T03:28:54|00:00:00|12:00:00|normal|batch|256|0|None
```

This independently confirms terminal `CANCELLED` state and that the job never
started: `Start=None`, `Elapsed=00:00:00`, and `AllocNodes=0`. It therefore
cannot satisfy the original production-success checks for 12-hour
scheduler-controlled runtime, training progress, repeated merges, or a final
checkpoint. Those checks are superseded operationally by the user's later
cancellation authorization; they are not represented as successful.

The one submitted production job remains `4980157`. It was never resubmitted,
and no replacement was submitted. Its immutable render, approval, structured
allowlist diff, submission marker, seed identity, and monitoring history remain
preserved in the retained launch bundle and in the cancellation record. A new
production launch remains blocked pending successful progressive-scale
validation.

## Requested debug-job follow-up

Debug job `5000354` is also absent from `squeue`. Its current accounting record
is:

```text
JobIDRaw|JobName|State|ExitCode|Submit|Eligible|Start|End|Elapsed|Timelimit|QOS|Partition|ReqNodes|AllocNodes|Reason
5000354|async-b4k40-ladder-256n|CANCELLED by 19032|0:0|2026-07-15T02:06:33|2026-07-15T02:06:33|None|2026-07-15T02:22:11|00:00:00|02:00:00|debug|batch|256|0|None
```

Because the job is terminal, Slurm no longer publishes a projected `StartTime`
or pending reason for it. The last live read-only observation, at
`2026-07-15T02:20:41-04:00`, recorded `StartTime=2026-07-15T02:24:00-04:00`,
unchanged from the `02:17:05` snapshot (delta `0` seconds), with pending reason
`Priority`. Accounting now shows that it was cancelled at
`2026-07-15T02:22:11-04:00`, before that projection, with no start or
allocation. This task did not modify or cancel job `5000354`.

## Final disposition

- Production job `4980157`: terminal `CANCELLED`, never allocated.
- Debug job `5000354`: terminal `CANCELLED`, never allocated.
- Production submissions made by this deployment: exactly one (`4980157`).
- Replacement or retry submissions: none.
- Current action: retain evidence and keep production blocked.
