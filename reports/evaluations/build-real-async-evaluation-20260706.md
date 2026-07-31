# Evaluation: build-real-async

Date: 2026-07-06
Evaluator role: Evaluator
Task: `build-real-async` - Build real async E97 DiLoCo trainer

## Verdict

Score: **0.08 / 1.00**
Confidence: **0.92**
Rubric underspecified: **No**. The task included concrete implementation expectations and validation checklist items.

The submitted tree does not implement the requested real async E97 DiLoCo trainer. The only relevant branch tip visible during evaluation was `f475a50` (`feat: implement-nonfatal-async (agent-694)`), which belongs to the upstream nonfatal quorum task, not this task. No new `scripts/frontier/e97_async_diloco_train.py` or equivalent real-training entrypoint is present, and `scripts/frontier/async_diloco_e97_multinode.py` still delegates directly to the synthetic debug harness.

## Evidence Checked

- `wg show build-real-async`: task was still `in-progress`, with no build-real-async-specific artifact list or progress logs beyond assignment and evaluator log.
- `git log --oneline -n 20`: `HEAD` was `f475a50 feat: implement-nonfatal-async (agent-694)`, not a build-real-async implementation commit.
- `git show --stat --name-only f475a50`: touched `ndm/async_diloco.py`, `scripts/frontier/async_diloco_e97_2n8n_debug.py`, and async prototype tests from the upstream dependency.
- `rg --files | rg 'e97_async|async_diloco|test_async|train_helpers|frontier'`: no new real async trainer module found.
- `scripts/frontier/async_diloco_e97_multinode.py`: imports `main` from `async_diloco_e97_2n8n_debug`, preserving production reliance on the debug harness.
- `ndm/async_diloco.py`: `_run_prototype_worker` still fabricates worker state deltas and reports synthetic loss via `1.0 - 0.01 * spec.local_steps`.
- Attempted validation command: `python3 -m pytest tests/test_async_diloco*.py tests/test_train_helpers.py`, but this base environment lacks `pytest` (`No module named pytest`). This does not materially affect the verdict because the required implementation artifacts are absent.

## Dimension Scores

- Real trainer entrypoint and orchestration: **0.00 / 0.30**
  No new real trainer script/module was added. The stable multinode entrypoint still calls the synthetic debug harness.

- Real local training workers using train helpers: **0.00 / 0.20**
  There is no evidence that workers run real token training for K optimizer steps from global state. The visible worker path remains `_run_prototype_worker`, which fabricates deltas by adding a scalar shift to tensors.

- Local/global supervision with robust nonfatal quorum: **0.05 / 0.15**
  The upstream nonfatal-quorum utilities exist and provide some relevant semantics, but this task did not integrate them into a real training orchestrator.

- Token-weighted merge, staleness, rebase/recovery, checkpoint ownership: **0.02 / 0.15**
  Some upstream prototype utilities support token weighting and checkpoint finalization. However, these are still prototype/debug-path mechanics, not a real trainer owning run-local latest and recovery from accepted global state.

- Metrics quality: **0.01 / 0.10**
  Prototype metrics exist, but the required real training loss is not present. Synthetic loss remains in `ndm/async_diloco.py`.

- Validation coverage: **0.00 / 0.07**
  No tests were added or identified that run a real trainer on a tiny token stream for one generation, cover real one-node reduction, or prove nonfatal local/global deferral in the real trainer.

- Safety / no production-scale job submission: **0.00 / 0.03**
  I did not find a build-real-async implementation log or artifact proving no production-scale job was submitted. The repository contains prior untracked Frontier log files including 256n outputs, but I did not attribute those to this task. This item is therefore unproven rather than counted as satisfied.

## Validation Checklist Assessment

- Unit/integration tests run the real trainer on a tiny synthetic/local dataset or token stream for one generation: **Not met**.
- Tests cover one-node reduction behavior: **Not met** for a real trainer.
- Tests cover local and global quorum deferral without process failure: **Partially present upstream**, but not for the requested real trainer.
- Metrics include real training loss sourced from the train step: **Not met**. Synthetic loss formula remains.
- Production prototype entrypoint is no longer misrepresented as real training: **Not met**. `async_diloco_e97_multinode.py` still delegates to `async_diloco_e97_2n8n_debug.py`.
- No 256n job is submitted: **Unproven** from task-specific logs.

## Rationale

This task asked for a production-shaped real async E97 DiLoCo training orchestrator replacing reliance on the synthetic debug harness. The accepted tree contains useful upstream infrastructure for nonfatal quorum semantics, but it does not satisfy the central deliverable: no real token-training worker path, no real trainer entrypoint, no real-loss metrics, no real-trainer tests, and no removal of debug-harness reliance from the stable production prototype entrypoint.

The small nonzero score credits the inherited quorum/checkpoint/metrics scaffolding that could support a future implementation, but the actor did not complete the requested build-real-async task.

