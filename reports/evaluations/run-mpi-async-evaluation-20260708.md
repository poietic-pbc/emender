# Evaluation: run-mpi-async

Date: 2026-07-08

Task: `run-mpi-async`

Evaluator: `agent-911`

## Verdict

Score: **0.02 / 1.00**

Confidence: **0.94**

Rubric underspecified: **no**. The task supplied an explicit validation
checklist, a sequential ladder policy, a required transport policy, and clear
negative constraints on larger submissions.

Disposition: **incomplete / retry required**. I found no evidence that the
assigned runner submitted or monitored the MPI compiled-helper 1n -> 8n -> 64n
ladder, and no first-failed-rung report was produced. Downstream 256n gating
must not proceed from this task state.

## Evidence Reviewed

- `wg show run-mpi-async`: task was still `in-progress`, assigned to
  `agent-911`, with no artifacts, no runner logs, and only coordinator/evaluator
  start logs.
- `wg log run-mpi-async --list`: no actor progress entries before this
  evaluation.
- `wg show .assign-run-mpi-async`: assignment task completed, but only assigned
  the runner.
- `.wg/agency/assignments/run-mpi-async.yaml`: records native assignment metadata
  only.
- `.wg/output/.assign-run-mpi-async/artifacts.json`: empty artifact list.
- `.wg/output/.assign-run-mpi-async/log.json`: contains only "Spawned assignment
  inline" and "Task marked as done".

I did not treat unrelated existing Frontier logs under `logs/frontier/` as
completion evidence for this task because the task itself records no job IDs,
no artifact paths, no report, and no linkage from the assigned actor to those
runs.

## Validation Dimensions

| Dimension | Score | Rationale |
| --- | ---: | --- |
| 1n/8n/64n job IDs, configs, QOS, walltime, node-hours, output roots or first failed rung | 0.00 | No 1n job ID was reported, and no first failed rung was identified. There is no evidence of 8n/64n eligibility or non-submission due to failure. |
| MPI/compiled-helper transport evidence and quorum metrics for passed rungs | 0.00 | No rung was reported as passed. No transport metadata, rank-starts, accepted/missing/stale/late/timed-out/rejected counts, catchup events, merge latency/bytes, loss window, or checkpoint/latest behavior were recorded. |
| TCP hot-path rejection/confirmation | 0.00 | The actor did not report `ASYNC_QUORUM_TRANSPORT`, transport metadata, or TCP byte counters, so TCP exclusion is unproven. |
| Production latest/last unchanged | 0.00 | No before/after production-latest check, artifact inspection, or explicit confirmation was produced. |
| Bounded 256n MPI debug gate recommendation | 0.00 | No recommendation was provided. Based on the empty ladder evidence, the evaluator recommendation is no-go/retry the ladder first. |
| Negative constraint: no forbidden larger submissions | 0.10 | I found no task-linked evidence of forbidden 128n/256n/1h/12h/production submission. This receives minimal credit because absence of evidence is weak and the task produced no audit trail. |
| WG task hygiene and artifacts | 0.00 | The actor left no report, artifacts, commit, or validation logs for the runner task. |

Overall grade: **0.02 / 1.00**.

## Rationale

The central deliverable was operational: submit the corrected MPI/compiled-helper
ladder sequentially, stop on first hard failure, and report enough evidence to
support the next gate. None of that evidence exists on the task. The only
completed predecessor here is `.assign-run-mpi-async`, which assigned a worker
and marked the assignment complete; it did not perform the runner task.

The score is not exactly zero only because I did not find task-linked evidence
that the actor violated the explicit "do not submit larger runs" constraint.
That weak negative evidence is not enough to satisfy any positive validation
criterion.

## Gate Recommendation

Do **not** proceed to `gate-mpi-async` or any 256n debug submission based on
this task. Retry `run-mpi-async` with a runner/intake/implementation-capable
agent that must either:

1. submit and monitor the 1n, 8n, and 64n MPI/compiled-helper ladder in order,
   reporting every required field; or
2. report the first failed rung with job/config/output-root evidence and prove
   no later rung was submitted.

Production latest/last must be checked explicitly before and after the retry,
and TCP must be rejected or ruled out from the hot aggregation path using
recorded metadata and/or byte counters.
