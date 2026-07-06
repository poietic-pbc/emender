# Evaluation Retry: build-real-async

Date: 2026-07-06
Evaluator role: Evaluator
Task: `build-real-async` - Build real async E97 DiLoCo trainer

## Verdict

Score: **0.08 / 1.00**
Confidence: **0.92**
Rubric underspecified: **No**. The task has concrete implementation expectations and an explicit validation checklist.

This retry confirms the prior evaluator finding recorded in
`reports/evaluations/build-real-async-evaluation-20260706.md`: the submitted
tree still does not contain the requested real async E97 DiLoCo trainer.

## Evidence Rechecked

- `wg show build-real-async` shows this is retry attempt 1 after the prior
  incomplete result, with no new build-real-async implementation artifacts.
- `rg --files | rg '(^|/)(e97_async|async_diloco_e97|async_diloco).*\\.py$|test_async_diloco|test_train_helpers'`
  finds only the existing prototype/debug async scripts and tests:
  `scripts/frontier/async_diloco_e97_1n_debug.py`,
  `scripts/frontier/async_diloco_e97_2n8n_debug.py`,
  `scripts/frontier/async_diloco_e97_multinode.py`, and existing async tests.
- No `scripts/frontier/e97_async_diloco_train.py` or equivalent real-training
  entrypoint is present.
- `scripts/frontier/async_diloco_e97_multinode.py` remains the stable
  multinode entrypoint and is still covered by tests that expect it to import
  from `async_diloco_e97_2n8n_debug`.
- `ndm/async_diloco.py` still contains the synthetic worker-loss metric:
  `1.0 - 0.01 * spec.local_steps`.

## Dimension Scores

- Real trainer entrypoint and orchestration: **0.00 / 0.30**
- Real local training workers using train helpers: **0.00 / 0.20**
- Local/global supervision with robust nonfatal quorum: **0.05 / 0.15**
- Token-weighted merge, staleness, rebase/recovery, checkpoint ownership:
  **0.02 / 0.15**
- Metrics quality: **0.01 / 0.10**
- Validation coverage: **0.00 / 0.07**
- Safety / no production-scale job submission: **0.00 / 0.03**

The nonzero credit is inherited from upstream quorum/checkpoint scaffolding, not
from completion of this task's central deliverable.

## Validation Checklist Assessment

- Unit/integration tests run the real trainer on a tiny synthetic/local dataset
  or tiny token stream for one generation: **Not met**.
- Tests cover one-node reduction behavior: **Not met** for a real trainer.
- Tests cover local and global quorum deferral without process failure:
  **Partially present upstream**, but not for the requested real trainer.
- Metrics include real training loss sourced from the train step:
  **Not met**; synthetic loss remains.
- Production prototype entrypoint is no longer misrepresented as real training:
  **Not met**; the stable multinode entrypoint still delegates to the debug
  harness.
- No 256n job is submitted: **Unproven** from task-specific evidence.

## Rationale

The task requested a new production-shaped async E97 DiLoCo trainer that runs
real token training via the refactored training helpers and uses robust quorum
semantics. The current retry branch still contains no such implementation, no
real-loss trainer metrics, and no tests exercising a real trainer. The task
should remain incomplete so an implementation agent can build the missing
trainer rather than treating the upstream prototype scaffolding as sufficient.
