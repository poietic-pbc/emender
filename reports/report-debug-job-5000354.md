# Slurm start estimate for debug job 5000354

## Snapshot

- Query timestamp: `2026-07-15T02:21:41.237680645-04:00` (`2026-07-15T06:21:41.237680645Z`).
- Follow-up timestamp: `2026-07-15T02:21:49.175548535-04:00`.
- Read-only commands used: `scontrol show job -o`, `squeue`, and `sprio`. No job-changing command was run.

At the primary timestamp, Slurm reported job `5000354` as exactly:

- `JobState=PENDING`
- `Reason=Priority`
- partition `batch`, QOS `debug`
- priority `43373465`
- requested nodes `256`
- `StartTime=2026-07-15T02:24:00-04:00`
- `LastSchedEval=2026-07-15T02:21:39-04:00`

The projected start was **2 minutes 18.762 seconds after the primary query** (about 2 minutes 19 seconds). `squeue --start` independently showed the same `2026-07-15T02:24:00` estimate and pending reason. The follow-up query approximately 7.94 seconds later still showed the same state, reason, priority, and StartTime; at that point the estimate was about 2 minutes 11 seconds away.

## Queue-position evidence

Slurm does not expose a single authoritative numeric queue position because scheduling also depends on eligibility, resource shape, reservations, backfill, and dependencies. A priority-sorted `squeue --qos=debug --states=PENDING,RUNNING` snapshot provided the following relevant pending entries:

| Job | Priority | Nodes | StartTime | Reason | Eligibility relative to 5000354 |
|---|---:|---:|---|---|---|
| 4991551 | 43545600 | 1 | unavailable | `DependencyNeverSatisfied` | Higher priority, but not eligible/runnable |
| **5000354** | **43373465** | **256** | **2026-07-15 02:24 EDT** | **`Priority`** | **Target** |
| 4995459 | 39080632 | 80 | unavailable | `Priority` | Lower priority |
| 4996598 | 39075506 | 275 | unavailable | `Priority` | Lower priority |
| 4998530 | 39073563 | 5 | unavailable | `Priority` | Lower priority |
| 4991935 | 13164702 | 1 | unavailable | `Priority` | Lower priority |
| 4994411 | 13161780 | 8 | unavailable | `Priority` | Lower priority |
| 4998730 | 13152704 | 1 | unavailable | `Priority` | Lower priority |

Therefore, within the visible pending debug-QOS snapshot, job `5000354` was **first by priority among eligible pending jobs**. There were **zero higher-priority eligible pending debug jobs**. Four higher-priority debug-QOS jobs were already running and are not pending jobs ahead of it.

## Estimate stability

Confidence is **moderate for the immediate estimate, but not guaranteed**. Positive evidence is that Slurm assigned a concrete 256-node `SchedNodeList`, had evaluated the job two seconds before the primary query, and preserved the exact StartTime over two queries roughly eight seconds apart. The estimate was also only about two minutes away. The caveat is that `squeue --start` is a projection, not a reservation guarantee; a 256-node allocation is sensitive to scheduler changes, node availability, higher-priority arrivals, and running-job completion timing. Re-query near 02:24 EDT for confirmation.

## Non-modification confirmation

Jobs `5000354` and `4980157` were **not modified**. This inspection used only read-only Slurm commands. As a reference snapshot, job `4980157` remained `JobState=PENDING`, `Reason=Priority`, QOS `normal`, priority `43351842`, and projected `StartTime=2026-07-16T09:10:00-04:00`.
