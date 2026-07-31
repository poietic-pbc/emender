# Diagnosis: E97 job 4975950 premature completion

Task: `diagnose-e97-job-4975950`

Date: 2026-07-13

## Verdict

Job `4975950` was an objective production failure even though Slurm recorded
`COMPLETED 0:0`. It did not run a 12-hour training workload. It ran the retained
20-minute smoke payload once: one generation containing exactly 40 local
training steps, then published a checkpoint and returned normally.

The exact stopping mechanism was the launcher environment and argv combination
`ASYNC_GENERATIONS=1` (the wrapper default) plus `ASYNC_LOCAL_STEPS=40`. The
actual multinode code path is even more restrictive than the displayed
`--generations 1`: it constructs one `RealAsyncFileRankConfig`, hard-codes
`generation = 0`, calls the local supervisor once with `local_steps=40`,
publishes that one generation, and returns. Therefore the seed step advanced
from `1525000` to `1525040` and the process exited successfully.

The retained `ASYNC_TIMEOUT_S=1200` was a per-generation quorum/transport
timeout, not a job-duration timer, and it did not fire. The retained
`--walltime-remaining-s 1200` was passed as a static checkpoint-finalization
hint; it caused the first and only publication to include a
`walltime_finalization.json` record, but did not keep training alive or stop it.
There was no Slurm signal, requeue, cancellation, exception, or nonzero exit.

## Scheduler and terminal evidence

The retained scheduler capture is
`build/e97-256/production/evidence/sacct-terminal.txt`. A fresh read-only
`sacct -X` query on 2026-07-13 returned the same allocation row:

```text
JobIDRaw|JobName|State|ExitCode|NNodes|NTasks|Elapsed|Timelimit|Start|End|Submit|Partition|QOS|Reason
4975950|async-b4k40-ladder-256n|COMPLETED|0:0|256||00:09:11|12:00:00|2026-07-12T23:09:16|2026-07-12T23:18:27|2026-07-12T15:41:20|batch|normal|None
```

The batch step also elapsed `00:09:11`; the 2,048-task step `4975950.0`
elapsed `00:08:41`. Thus only 551 seconds of the 43,200-second allocation limit
were used (1.28%). The terminal reason was normal exhaustion of the program's
single-generation work list, followed by exit 0—not scheduler walltime.

The retained stdout/stderr locations are recorded in
`build/e97-256/production/evidence/log-paths.txt`. Stdout prints the actual
`srun` command, including:

```text
--generations 1 --local-steps 40 --steps 40 --timeout-s 1200
...
--finalization-reserve-seconds 1200 --walltime-remaining-s 1200
```

It then prints one coordinator result for `generation: 0`, with 2,048 accepted
updates, finite loss `2.4854`, and a successful merge. The final checkpoint in
`build/e97-256/production/evidence/terminal-validation.json` is step `1525040`,
exactly `1525000 + 40`. Stderr contains only environment-module replacement
notices; it contains no termination or signal error.

No job submission or resubmission was performed during this diagnosis. The
only scheduler command used was the read-only accounting query above.

## Retained launch-input reconciliation

Both `build/e97-256/smoke/launch-inputs.json` and
`build/e97-256/production/launch-inputs.json` retain this identical effective
stop budget:

```json
{
  "generations": 1,
  "local_steps": 40,
  "steps": 40,
  "timeout_s": 1200,
  "walltime_remaining_s": 1200
}
```

The production `sbatch_argv` changed only the requested walltime from
`00:20:00` to `12:00:00` and QoS from `debug` to `normal`; its exported smoke
values remained `ASYNC_TIMEOUT_S=1200,DILOCO_K=40,ASYNC_LOCAL_STEPS=40`.
`ASYNC_GENERATIONS` was absent, so
`scripts/frontier/trainpy_async_quorum_smoke_common.sh` selected its default of
one. The same common wrapper rendered:

```text
--generations "$ASYNC_GENERATIONS"
--local-steps "$ASYNC_LOCAL_STEPS"
--steps "$ASYNC_LOCAL_STEPS"
--timeout-s "$ASYNC_TIMEOUT_S"
```

The CLI parser in `scripts/frontier/e97_async_diloco_train.py` accepts
`--generations`, but the selected actual-multinode branch does not pass it into
`RealAsyncFileRankConfig`. `ndm/async_diloco_real.py`'s
`run_real_async_diloco_file_rank` sets `generation = 0`, builds one worker spec
with `local_steps=config.local_steps`, invokes `_run_real_node_supervisor` once,
coordinates/publishes once, and returns. In contrast, only the local simulated
path uses `for offset in range(config.generations)`. Consequently even raising
`--generations` alone would not make this selected production path sustained.

The retained launch input explicitly records `signal: null` and
`requeue: null`; neither rendered batch file contains an `#SBATCH --signal` or
requeue directive. The allocation reason is `None`, all exit codes are `0:0`,
and no signal path explains the terminal state.

## Why promotion validation produced a false positive

The pre-submit parity checker encoded the wrong invariant. Policy
`configs/frontier/e97_async_256_parity_policy.json` says
`training_stop_budget_must_match: true`, and
`scripts/frontier/check_e97_async_promotion.py` rejects a production stop
budget that differs from smoke. The test
`test_job_4962400_launcher_is_byte_exact_and_only_queue_time_differ` goes
further and asserts the known-bad matched value `{generations: 1,
local_steps: 40, steps: 40, timeout_s: 1200, walltime_remaining_s: 1200}`.
Matching the smoke payload was therefore treated as proof of production parity,
although it guaranteed smoke-length work.

The post-run validation then conflated process correctness with production
objective success. It required Slurm `COMPLETED 0:0`, all ranks, a finite loss,
one advanced quorum/merge, and a reloadable checkpoint. Those checks correctly
show that one 40-step generation was healthy, but no check required:

- elapsed time consistent with the 12-hour allocation;
- more than the smoke's single generation/40 steps;
- a scheduler-derived deadline as the effective runtime source; or
- a structured terminal reason identifying deadline-reserve finalization.

The prior report consequently described the retained smoke stop budget as
"intentionally" identical and accepted early completion. That is the precise
validation gap: `COMPLETED 0:0` was used as sufficient success evidence while
the declared 12-hour duration was never tested.

## Minimal invariant and gate change

Smoke and production should retain identical training configuration and differ
only in queue selection and scheduler walltime, but *effective duration must not
be a fixed launch input*. The minimal invariant is:

> The common workload runs repeated K-step generations until the deadline
> supplied by its current Slurm allocation, then finalizes inside a fixed safety
> reserve. No finite generation count, total-step count, static remaining-time
> value, or independent runtime limit may terminate a production-capable run.

Concretely:

1. Keep `DILOCO_K=40`/`local_steps=40` as the algorithmic steps per generation;
   do not interpret it as total job steps.
2. Refactor the actual multinode path to loop over generations, chaining the
   newly merged checkpoint/state, until a monotonic deadline derived at runtime
   from Slurm (prefer `SLURM_JOB_END_TIME`; fail closed if a production-capable
   allocation has no trustworthy scheduler deadline).
3. Keep one identical finalization reserve for smoke and production. Derive
   `walltime_remaining_s` dynamically on every generation from that deadline;
   remove the static `--walltime-remaining-s 1200` value.
4. Remove finite `ASYNC_GENERATIONS`/total `--steps` as termination controls
   from the production-capable launcher. If retained for unit/debug-only modes,
   the promotion gate must reject them for both promotable smoke and production.
5. Replace `training_stop_budget_must_match` with a fail-closed
   `scheduler_deadline_controls_duration` invariant. Normalization may still
   allow only scheduler `-t`, partition, and QoS differences because the same
   workload automatically receives the allocation-specific deadline.

The terminal gate must separately distinguish workload health from duration
success. For an allocation that exits 0, require a structured terminal record
such as `terminal_reason=scheduler_deadline_finalization`, the observed Slurm
deadline and reserve, finalization before the deadline, and elapsed time within
a documented reserve/tolerance of the requested limit. An early `COMPLETED 0:0`
without that deadline reason must fail promotion/production validation.

## Regression-test plan

Add tests before any future submission:

1. A renderer/gate test that rejects promotable launch inputs containing
   `generations=1`, total `steps=40`, or a static `walltime_remaining_s=1200`,
   while still proving smoke and production differ only in `-t`, partition, and
   QoS.
2. A deterministic clock test of the actual multinode loop: with K=40 and a
   synthetic scheduler deadline long enough for three generations, assert it
   executes generations 0, 1, and 2, chains their state/checkpoints, and stops
   only when the remaining time reaches the reserve.
3. A paired-duration test using the identical workload and fake scheduler
   deadlines of 20 minutes and 12 hours. Assert the latter runs more
   generations and that neither duration is encoded as a step/generation cap.
4. A missing-deadline production test that fails closed rather than falling
   back to one generation or 1,200 seconds.
5. Terminal-validator fixtures where `COMPLETED 0:0` at 9:11 of a 12-hour limit
   fails, while a run ending in the reserve window with the structured
   scheduler-deadline reason passes. Also assert a signal, timeout, exception,
   cancellation, or arbitrary max-step exit cannot masquerade as success.

No future 12-hour production submission should be authorized until these tests
and the sustained actual-multinode loop pass locally and on a bounded smoke.
