# Evaluation: validate-merged-real-2

Task: `validate-merged-real-2` (`Validate merged real async E97 8n20m`)

Evaluator: `agent-732`

Date: 2026-07-06

## Verdict

Score: 0.00 / 1.00

Confidence: 0.95

Rubric underspecified: no. The task supplied concrete validation checklist items, and each item is directly checkable from WG artifacts, logs, Slurm output, and run-directory evidence.

Outcome: fail / incomplete. The actor did not provide evidence that an 8-node <=20 minute debug-QOS validation was submitted or completed using the production wrapper and real trainer path. No task-owned artifacts, Slurm job id, sbatch command, logs, metrics, elapsed time, node-hours, checkpoint/finalization record, or pass/no-go decision were recorded for this task.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Slurm submission evidence | 0.00 | `wg show validate-merged-real-2` and `wg log validate-merged-real-2 --list` show no actor evidence beyond task spawn. No 8-node job id, exact `sbatch` command, queue/QOS, elapsed time, node-hours, or task-owned Slurm logs were recorded. |
| Production-shaped run artifacts | 0.00 | No artifact demonstrates the production wrapper/real trainer path on `main`, activated `PYTHON_BIN`, refreshed `latest.pt`, `batch_size=4`, `chunk_size=2048`, or `DILOCO_K=40` for an 8-node run. |
| Training metrics and quorum distributions | 0.00 | No 8-node metrics file was recorded. A broad check of `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/.../20260706` found only prior 1-node and 2-node fix-run metrics, not an 8-node validation from this task. |
| Checkpoint/finalization behavior | 0.00 | No checkpoint manifest, recovery/export checkpoint record, or finalization behavior was recorded for the requested 8-node run. |
| Clear 256n12h pass/no-go | 0.00 | The task contains no clear pass/no-go decision for preparing the 256-node 12-hour submission. In the absence of an 8-node validation, the correct operational decision is no-go, but that was not supplied by the actor. |

## Evidence Reviewed

- `wg show validate-merged-real-2`: task is assigned/in-progress, has zero commits ahead in its worktree, no uncommitted files, and no artifacts listed.
- `wg log validate-merged-real-2 --list`: only the coordinator spawn entry and this evaluator's review log are present.
- `wg show .assign-validate-merged-real-2`: assignment task completed, but it only spawned the validation task; it contains no validation artifacts.
- `find docs logs/frontier/async_diloco_e97 ... | rg '8n|validate.*2|merged_real|494'`: found older unrelated log names, with no task-owned 8-node validation report for `validate-merged-real-2`.
- `find /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco ... | rg '20260706|8n|4949|4950'`: found only the prior 1-node and 2-node fix-run metrics from jobs `4949348` and `4949402`.

## Calibration Rationale

A score above zero would require at least partial evidence that the actor attempted the requested 8-node Slurm validation or produced some subset of the required artifacts. The available task record shows no such attempt. Prior dependency work established 1-node and 2-node readiness, but the task under evaluation specifically required a new 8-node <=20 minute debug-QOS run and a recorded readiness decision for 256n12h. Dependency evidence cannot satisfy that scale gate.

Recommended WG action: mark `validate-merged-real-2` incomplete and retry with instructions to actually submit or attempt the 8-node production-wrapper run and record all checklist artifacts.
