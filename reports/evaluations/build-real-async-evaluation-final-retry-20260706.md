# Final Retry Evaluation: build-real-async

Date: 2026-07-06
Evaluator role: Evaluator
Task: `build-real-async` - Build real async E97 DiLoCo trainer

## Verdict

Score: **0.08 / 1.00**
Confidence: **0.93**
Rubric underspecified: **No**. The task description gives a concrete expected
implementation shape and a validation checklist.

This final retry rechecks the same tree state as the prior two evaluations and
finds no implementation progress toward the requested real async E97 DiLoCo
trainer. The task should remain incomplete and be retried by an implementation
agent, not passed through evaluation.

## Evidence Checked

- `wg show build-real-async` reports retry count 2 and this worktree contains
  only evaluation-report commits ahead of `main`.
- `rg --files | rg '(^|/)(e97_async|async_diloco_e97|async_diloco).*\.py$|test_async_diloco|test_train_helpers'`
  finds only existing debug/prototype scripts and existing tests. There is no
  `scripts/frontier/e97_async_diloco_train.py` or equivalent real-training
  orchestrator.
- `scripts/frontier/async_diloco_e97_multinode.py` still consists of a wrapper
  around `async_diloco_e97_2n8n_debug`.
- `tests/test_async_diloco_e97_2n8n_debug_runner.py` still asserts that the
  multinode entrypoint imports from the debug harness.
- `ndm/async_diloco.py` still contains the synthetic worker loss expression
  `1.0 - 0.01 * spec.local_steps`.
- The expected real-trainer tests for tiny local token training, one-node
  reduction, and nonfatal local/global quorum deferral are absent.

## Dimension Scores

- Real trainer entrypoint and orchestration: **0.00 / 0.30**
- Real local training workers using train helpers: **0.00 / 0.20**
- Local/global supervision with robust nonfatal quorum: **0.05 / 0.15**
- Token-weighted merge, staleness, rebase/recovery, checkpoint ownership:
  **0.02 / 0.15**
- Metrics quality: **0.01 / 0.10**
- Validation coverage: **0.00 / 0.07**
- Safety / no production-scale job submission: **0.00 / 0.03**

The small amount of credit reflects inherited async quorum scaffolding from
upstream dependencies, not fulfillment of this task's required deliverable.

## Validation Checklist Assessment

- Unit/integration tests run the real trainer on a tiny synthetic/local dataset
  or tiny token stream for one generation: **Not met**.
- Tests cover one-node reduction behavior: **Not met** for a real trainer.
- Tests cover local and global quorum deferral without process failure:
  **Partially present upstream**, but not for a real trainer.
- Metrics include real training loss sourced from the train step:
  **Not met**; the synthetic loss remains.
- Production prototype entrypoint is no longer misrepresented as real training:
  **Not met**.
- No 256n job is submitted: **No evidence of a 256n submission found**, but this
  safety item cannot compensate for the missing implementation.

## Validation Commands

- Evidence scan:
  `rg --files | rg '(^|/)(e97_async|async_diloco_e97|async_diloco).*\.py$|test_async_diloco|test_train_helpers'`
- Entrypoint/synthetic-loss scan:
  `rg -n "1\.0 - 0\.01 \* spec\.local_steps|async_diloco_e97_2n8n_debug|e97_async_diloco_train|real async" ndm scripts tests reports/evaluations/build-real-async-evaluation-retry-20260706.md`

I did not rerun the pytest suite on this final retry because no implementation
files changed since the prior retry and the prior retry already recorded that
`python3 -m pytest tests/test_async_diloco*.py tests/test_train_helpers.py`
was blocked in this environment by `No module named pytest`.
