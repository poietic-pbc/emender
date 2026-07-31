# Evaluation: run-post-patch

Date: 2026-07-07

Task: `run-post-patch`

Evaluator: `agent-792`

## Verdict

Score: 0.20 / 1.00

Confidence: 0.86

Rubric underspecified: no. The task has an explicit `## Validation` checklist
and a clear ladder policy.

Disposition: incomplete. The smoke ladder was not completed and should be
retried by a runner/intake/implementation-capable agent after job `4953629`
finishes or is cancelled/classified.

## Evidence Reviewed

- `wg show run-post-patch`: task was still `in-progress`, had no artifacts, and
  only contained the evaluator-start log entry at review time.
- `.wg/output/.assign-run-post-patch/artifacts.json`: empty artifact list.
- `.wg/output/.assign-run-post-patch/changes.patch`: no changes detected.
- Adopted job stdout/stderr in `.wg-worktrees/agent-788/logs/frontier/trainpy_async_quorum/`.
- Adopted run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z`.
- `squeue -j 4953629`: job was still `RUNNING` at review, on 1 node in
  `batch` partition with `debug` QOS, time `4:49/20:00`.
- `sacct -j 4953629`: job and steps were still `RUNNING`.

## Validation Dimensions

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Adopted 1n job report completeness | 0.20 | Some raw evidence exists externally: job `4953629` stdout names the run root, command, debug QOS intent, expected 8 ranks, and run-local artifact paths. Rank-start evidence shows 8/8 ranks started. However no `run-post-patch` report was produced, the job was still running, `metrics.json`, `manifest.json`, and `summary.md` were absent, and there was no pass/fail decision. |
| Conditional ladder submissions | 1.00 | No 2n/8n/64n submissions were found, which is appropriate because the adopted 1n rung had not passed. |
| Submission details for later rungs | N/A | No later rung was eligible for submission. This dimension is not penalized directly, but it also provides no positive completion evidence. |
| No speculative larger rung submissions | 1.00 | I found no evidence of speculative 2n/8n/64n/256n submissions from this task. |
| No production latest mutation confirmation | 0.00 | The task required an affirmative report confirming no production latest path was mutated. No such report or artifact exists for `run-post-patch`. The adopted job intent says run-local latest only, but the required confirmation was not delivered. |
| Failure/pass classification | 0.00 | There was no terminal rung result to classify. At review time the operational state was pending/running, not a transport, launcher, data/checkpoint, runtime/MPI, or scheduler/accounting failure report. |

## Operational State For Retry

Adopted 1n job `4953629` had not reached a gate decision at review time. The
available artifacts showed:

- `rank-start.tsv`: 8 lines, ranks 0 through 7 all started on `frontier08198`.
- Heartbeats: all eight ranks reached `stage=mpi_dense_send_starting` with
  `transport=mpi-dense`, `global_quorum=8`, and
  `mpi_bucket_bytes=67108864`.
- Missing expected pass artifacts:
  - `artifacts/metrics.json`
  - `artifacts/manifest.json`
  - `summaries/summary.md`
- The named Slurm output was still live:
  `.wg-worktrees/agent-788/logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4953629.out`
  and
  `.wg-worktrees/agent-788/logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4953629.err`.

Because the 1n job was still running and lacked metrics/manifest/summary
artifacts, the correct ladder action was to continue monitoring `4953629`, not
submit 2n.

## Grade Rationale

The actor did not satisfy the central task outcome: there is no concise
post-patch smoke ladder report and no terminal pass/fail decision for the
adopted 1n rung. The only substantial positive evidence is that the adopted job
exists, is in debug QOS, launched all eight ranks, and reached MPI dense send
heartbeats. That evidence is insufficient for a passing ladder gate because the
job was still running and the required metrics/latest/checkpoint artifacts were
absent.

I assign 0.20 rather than 0.00 because the available external evidence confirms
that the first rung was launched/adopted in the intended shape and that the
agent did not violate the ladder by submitting larger rungs prematurely. The
grade remains low because completion requires a monitored terminal outcome and
an explicit report, neither of which exists.
