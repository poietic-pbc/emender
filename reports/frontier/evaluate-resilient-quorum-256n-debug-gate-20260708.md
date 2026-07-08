# Resilient quorum 256n debug gate evaluation

Task: `evaluate-resilient-quorum-256n-debug-gate`
Date: 2026-07-08

## Decision

**No-go. Do not submit the 256-node debug smoke.**

The prerequisite resilient-quorum 1n/8n/64n ladder is not clean. The only
attempted rung, Slurm job `4956022`, failed at Python argument parsing before
the 1n run emitted usable quorum, checkpoint, or terminal-state metrics. The
8n and 64n rungs were not submitted, so the required scale ladder evidence is
absent.

No 256n debug, 1h, 12h, production, latest/last-mutating, or
training-continuation job was submitted from this gate task.

## Cited Evidence

- Implementation task `implement-resilient-quorum-diloco`: status `done`;
  focused validation logged `py_compile` plus 43 focused async quorum, MPI,
  compiled-MPICH, real-trainer, and E97 entrypoint tests passing; commit
  `03cd39f` pushed to `origin/main`.
- Failure-injection task `validate-resilient-quorum-failure-injection`: status
  `done`; focused suite logged `36 passed` via the project Python environment;
  commit `f15557a` pushed to `origin/wg/agent-879/validate-resilient-quorum-failure-injection`.
- Ladder task `run-resilient-quorum-1n8n64n-ladder`: status `done`, but its
  own conclusion is a stop-on-first-failure no-go. The dependency log records
  `1n rung failed hard before metrics: job 4956022 FAILED exit 2:0 due
  wrapper/entrypoint CLI mismatch; no 8n/64n/128n/256n/1h jobs submitted`.
- Ladder summary artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/summaries/summary.md`
  reports conclusion `no-go-missing-metrics`, exit status `2`, and a metrics
  excerpt where required fields such as `configured_quorum`,
  `effective_quorum`, `checkpoint_finalization`, `production_latest_guard`,
  `resume_check`, and `update_counts` are all `null`.
- Ladder log artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/logs/async_diloco_e97_2n8n.log`
  shows `async_diloco_e97_multinode.py: error: unrecognized arguments` for
  wrapper-only flags including `--worker-count-per-node`,
  `--tokens-per-step`, `--delta-scale`, `--task-id`, `--slurm-job-id`,
  `--requested-walltime`, `--requested-node-hours`, `--resume-check`, and
  `--production-latest-path`.
- Ladder environment artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/artifacts/env.txt`
  records the intended 1n debug configuration: checkpoint seed
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`,
  QOS/debug walltime `00:20:00`, requested node-hours `0.333333`, local quorum
  `8`, global quorum `1`, and production latest guard
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`.

## Go Criteria Status

| Criterion | Status | Rationale |
| --- | --- | --- |
| Implementation task passed | Met | `implement-resilient-quorum-diloco` is `done`; validation logs cite focused tests and commit `03cd39f`. |
| Failure-injection task passed | Met | `validate-resilient-quorum-failure-injection` is `done`; validation logs cite 36 passing focused tests and commit `f15557a`. |
| 1n resilient-quorum rung passed | Not met | Job `4956022` exited `2:0` at CLI parsing before metrics. |
| 8n resilient-quorum rung passed | Not met | Not submitted after the 1n stop-on-first-failure. |
| 64n resilient-quorum rung passed | Not met | Not submitted after the 1n stop-on-first-failure. |
| Strict `MPI_Reduce` fast path available | Partially evidenced | Implementation/failure-injection logs and existing `tests/test_async_diloco_compiled_mpich.py` preserve/check `MPI_Reduce`, but the required ladder did not reach a run that could exercise fallback/control behavior. |
| Scale-safety metrics complete | Not met | Summary metrics are `null` for quorum, update counts, checkpoint finalization, guard, and terminal fields; no 8n/64n metrics exist. |

## Required Resilience Evidence Status

| Required evidence item | Status | Evidence and gap |
| --- | --- | --- |
| Nonjoining rank tolerance | Unit/failure-injection evidence exists; ladder evidence missing | Failure-injection task passed with automated coverage for quorum advance without all ranks. The failed 1n rung produced no run artifact showing missing/nonjoining rank accounting at ladder scale. |
| Stuck-rank timeout | Unit/failure-injection evidence exists; ladder evidence missing | Failure-injection task passed stuck/missing timeout coverage. Ladder summary has no `timed_out`, `missing`, or terminal quorum metrics because execution failed before the run. |
| Stale-generation handling | Unit/failure-injection evidence exists; ladder evidence missing | Focused tests cover stale update rejection/metrics; the ladder did not emit stale-generation metrics or artifacts. |
| Stale/restarted rank catchup | Unit/failure-injection evidence exists; ladder evidence missing | Focused tests cover catchup behavior, but the ladder did not emit catchup events, checkpoint ids, or resume/catchup artifacts. |
| Checkpoint finalization remains run-local | Not evidenced by this ladder | The env file shows an intended run-local `run_dir` and production guard, but the rung failed before checkpoint finalization. Summary `checkpoint_finalization` is `null`. |
| Production latest/last guard intact | Intent recorded; no mutation authorized by this task | The env file records `production_latest_guard=.../latest.pt` and the ladder task log says the guard was unchanged. This gate task submitted no jobs and did not mutate production latest/last. A successful rerun should include before/after stat/hash evidence. |

## Missing 256n Debug Metrics

Because this is a no-go and no 256n job was submitted, there is no 256n report
for ranks started/joined, accepted quorum, missing/stale/late/timed-out/rejected
counts, catchup events, merge latency, bytes, loss window, checkpoint/latest
behavior, or terminal state. These remain required before any later 256n or 1h
approval package consumes this workstream.

## Next Fix And Retest

1. Fix the wrapper/entrypoint CLI contract for
   `scripts/frontier/async_diloco_e97_multinode.py` or its ladder wrapper so
   wrapper-only metadata flags are either accepted by the entrypoint or stripped
   before invocation.
2. Add a focused test that runs the generated 1n ladder command through
   argument parsing, including `--worker-count-per-node`, `--tokens-per-step`,
   `--delta-scale`, `--task-id`, `--slurm-job-id`, `--requested-walltime`,
   `--requested-node-hours`, `--resume-check`, and
   `--production-latest-path`.
3. Rerun the resilient-quorum ladder from 1n, stopping on first failure. Only if
   1n, 8n, and 64n all pass with the required metrics and production guard
   evidence should a later gate submit exactly one bounded 10-20 minute 256n
   debug smoke.

## Evaluation Grade

Readiness score: **0.22 / 1.00** with high confidence.

Dimension scores:

- Upstream implementation/failure-injection readiness: `0.90`
- 1n/8n/64n ladder readiness: `0.00`
- Required resilience evidence at ladder/run-artifact level: `0.25`
- Metrics sufficiency for 256n scale safety: `0.00`
- Checkpoint/latest and production guard evidence: `0.35`
- Constraint fidelity for this gate task: `1.00`

The score is low because the explicit go criteria are conjunctive: a clean
implementation and unit-level failure injection are not enough without a passing
1n/8n/64n ladder and complete scale-safety metrics. The gate-task constraint
score is high because this evaluation did not submit prohibited jobs or mutate
production latest/last.
