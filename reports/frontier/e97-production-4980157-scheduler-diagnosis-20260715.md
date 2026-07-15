# E97 production job 4980157 scheduler diagnosis

Observed at `2026-07-15T06:17:05Z` (`02:17:05` Frontier local). This was a
read-only monitoring pass. Job `4980157` was not modified, cancelled, or
resubmitted. The separate 256-node debug job visible as `5000354` was also not
touched.

## Current production state

Slurm reports job `4980157` as `PENDING (Priority)`, normal QoS, requesting
exactly 256 nodes, 2,048 tasks at eight tasks per node, and `12:00:00`. It has
no dependency and no allocation. Its current estimated interval is
`2026-07-16T09:10:00-04:00` through `2026-07-16T21:10:00-04:00`.

The start estimate has demonstrably moved later:

| observation | estimated start |
| --- | --- |
| 2026-07-13 08:57 EDT | 2026-07-13 14:24 EDT |
| 2026-07-14 04:52 EDT | 2026-07-14 16:00 EDT |
| 2026-07-14 06:30 EDT | 2026-07-14 16:30 EDT |
| 2026-07-15 02:05 EDT | 2026-07-16 09:10 EDT |
| 2026-07-15 02:17 EDT | 2026-07-16 09:10 EDT |

The estimate is therefore a rolling backfill projection, not a reservation or
guarantee. The latest projection is stable across the two July 15 observations
but is about 42 hours 46 minutes later than the first recorded projection.

## Priority and competing demand

`sprio -l` decomposes the production priority `43,351,542` as association
`43,200,000` plus age `151,542`; site, fair-share, job-size, partition, QoS,
nice, and TRES components are all zero. The cluster configuration confirms
multifactor priority, `PriorityWeightFairShare=0`, age weight `31,536,000`, and
association/QoS weights of `86,400`. Thus the scheduler does not currently
attribute this delay to a worsening fair-share score; the displayed fair-share
component is exactly zero for this job.

At the snapshot, 258 pending batch jobs had a numerically higher priority,
requesting 49,229 nodes in aggregate. Of those, 47 were immediately relevant
`Priority` or `Resources` contenders; 256 of the 258 were normal QoS. Large
observed jobs ahead included normal-QoS requests for 9,248 and 8,192 nodes, as
well as numerous 256--863-node requests. This is direct evidence of substantial
higher-priority normal-QoS demand competing for the contiguous scheduling
window needed by a 256-node, 12-hour job.

The batch partition is exclusive, reports 9,536 total nodes, and at this
snapshot `sinfo` showed 9,302 allocated nodes plus additional planned,
maintenance, drained, down, and booting nodes. Slurm's backfill scheduler uses a
48-hour window (`bf_window=2880`) and reserves running jobs while evaluating at
most 1,000 jobs per pass. The job had a concrete `SchedNodeList`, so Slurm found
a prospective placement, but current higher-priority work prevents it from
starting now. No user reservation covering job `4980157` was visible; the
listed reservations were small maintenance/help reservations rather than a
256-node reservation for this workload.

## Does repeated debug usage explain the delay?

Observed facts:

- Ten debug-QoS jobs were visible: two running and eight pending, totaling 657
  requested or allocated nodes. Four had priority above `4980157`.
- The separate same-user debug job `5000354` requested 256 nodes and had an
  estimated start of `2026-07-15T02:24:00-04:00` at the snapshot.
- Debug QoS has priority factor 2 and a two-hour maximum walltime; normal QoS
  has priority factor 0. Cluster preemption is disabled (`PreemptMode=OFF`, no
  `PreemptType`).
- The overwhelming majority of pending jobs above production were normal QoS:
  256 normal versus two debug in the raw numeric-priority count.

Inference: debug jobs can materially delay the normal job at the margin because
they consume exclusive nodes and their QoS boost may let them enter short
backfill holes first. Repeated 256-node debug launches can temporarily remove
exactly the node count production needs. They cannot preempt a running normal
job, however, and the snapshot does not support debug as the primary cause of
the multi-day backward movement. The stronger explanation is heavy
higher-priority normal-QoS demand combined with the difficulty of finding a
256-node window that remains free for the full 12-hour walltime. Only Slurm's
internal future-plan evolution could prove which individual arrivals displaced
each earlier estimate; the public fields establish correlation and scheduling
constraints, not a per-job causal trace.

### Debug job 5000354 follow-up

At `2026-07-15T02:20:41-04:00`, job `5000354` was still `PENDING` with
reason `Priority`, requesting 256 nodes and 2,048 tasks under debug QoS for
`02:00:00`. Its StartTime remained `2026-07-15T02:24:00-04:00`, a delta of
exactly zero seconds from the `02:17:05` observation. Its priority had aged
from `43,373,165` to `43,373,465`; the latter decomposed into association
`43,200,000`, debug QoS `172,800`, and age `665`. This is reported for context
only; neither the debug job nor production was modified.

## Monitoring cadence

The projected start was more than two hours away, so the required pending
policy parks monitoring until approximately 90 minutes before the estimate:
`2026-07-16T07:40:00-04:00`. A materially revised estimate resets that target.
Five-minute progress polling does not begin until the production job is
actually `RUNNING`.
