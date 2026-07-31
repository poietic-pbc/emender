# Frontier E97 32-Node Loss Regression Investigation

Date: 2026-06-24
Task: `investigate-32-node-e97`

## Verdict

Do not launch `run-64-node-e97` yet.

The 32-node E97-MLP avg smoke was operationally clean, but its loss regression
is too large and too coherent across the run to accept as a green 64-node gate.
The most defensible next action is one bounded 32-node retry from the same
16-node source checkpoint before any 64-node E97 avg smoke. If the retry is
flat or improves against the source, treat the current run as noisy bounded
smoke behavior. If the retry repeats the upward drift, treat it as a systematic
32-node scale-out/configuration issue and fix that before larger jobs.

I did not submit any diagnostic job from this task. Requested and actual
node-hours for this investigation are therefore `0`.

## Evidence Inspected

Primary predecessor report:

```text
docs/FRONTIER_E97_32NODE_AVG_SMOKE_20260624.md
```

Referenced run artifacts:

```text
logs/frontier/scaleout/e97-32n-avg-smoke-4894111.out
logs/frontier/scaleout/e97-32n-avg-smoke-4894111.err
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/summaries/summary.md
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/train/emender_E97_1.3B_20260624_070413/latest.pt
```

Comparison source report:

```text
docs/FRONTIER_E97_16NODE_AVG_SMOKE_20260624.md
```

## Operational Findings

The 32-node run did not show a launch, communication, checkpoint, or
finalization failure:

- Slurm job `4894111` completed with exit `0:0`.
- Accounting reported `COMPLETED`, 32 nodes, elapsed `00:20:22`, and actual
  node-hours `10.862222`.
- The manifest recorded the intended E97-MLP avg outer configuration:
  `nodes=32`, `diloco_outer_optimizer=avg`, `diloco_export_basis=x`, source
  checkpoint equal to the 16-node `latest.pt`, and runtime commit
  `17440fddc52ea58c4328fb78171e830046b85d3e`.
- Rank 0 logged `world_size=256`, `32 islands x 8 GPUs`, `DILOCO_K=10`, and
  `outer optimizer: avg`.
- The run completed 46 DiLoCo merges, wrote a final checkpoint, and advanced
  `latest.pt` to `checkpoint_step_001780_loss_5.8701.pt`.
- The final merge was correctly skipped at step `1780` because the step had
  already landed on the regular K=10 merge cadence.
- The train log and Slurm stderr did not show a fatal traceback, runtime
  error, watchdog timeout, collective mismatch, OOM, non-finite loss, NaN, or
  cancellation signature.

The non-blocking warnings match the prior operational smoke class: PyTorch/NCCL
rank-to-GPU startup warnings, c10d address-family warnings, a Triton version
warning, module replacement notices, and an AMD low-power-state notice. These
do not explain the loss regression by themselves because the job formed the
process group, trained, merged, checkpointed, and exited cleanly.

## Loss Findings

The source checkpoint for 32-node was the clean 16-node final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

That predecessor was effectively flat against its 8-node source:

| Run | Source final loss | Final loss | Delta |
| --- | ---: | ---: | ---: |
| 16-node avg smoke | 5.2538 | 5.2531 | -0.0007 |
| 32-node avg smoke | 5.2531 | 5.8701 | +0.6170 |

The raw 32-node metric sequence also drifted upward during the run, rather
than only producing one bad final metric:

| Metric window | 16-node mean | 32-node mean |
| --- | ---: | ---: |
| all rank-0 metric samples | 5.2531 | 5.8701 |
| first 20 samples | 5.1588 | 5.4394 |
| last 20 samples | 5.4237 | 6.1958 |

Five equal windows of the 32-node rank-0 metric samples show a coherent rise:

| 32-node window | Steps | Mean loss |
| --- | --- | ---: |
| quintile 1 | 1330-1415 | 5.4295 |
| quintile 2 | 1420-1505 | 5.7480 |
| quintile 3 | 1510-1595 | 5.9166 |
| quintile 4 | 1600-1685 | 6.0427 |
| quintile 5 | 1690-1780 | 6.1957 |

The comparable 16-node sequence was much flatter:

| 16-node window | Steps | Mean loss |
| --- | --- | ---: |
| quintile 1 | 975-1040 | 5.1584 |
| quintile 2 | 1045-1110 | 5.1743 |
| quintile 3 | 1115-1180 | 5.2440 |
| quintile 4 | 1185-1250 | 5.2204 |
| quintile 5 | 1255-1325 | 5.4540 |

This is not enough evidence to call a specific code bug: the loss metric is a
short training-loss window over changing training tokens, not a fixed
validation metric, and 32 nodes advanced more steps in the same bounded wall
time than 16 nodes did. However, the magnitude of the 32-node final delta and
the monotonic window-level rise are too large to dismiss as acceptable noise
for the 64-node gate.

## Decision

Classification: requires a 32-node retry before 64-node, not an immediate
code/config fix.

Rationale:

- Acceptable-noise classification is too weak because the final loss regressed
  by `+0.6170` versus source and the within-run windows rose from `5.4394` to
  `6.1958`.
- Immediate code/config fix is premature because the operational path was
  clean, the avg outer configuration was as requested, and there is only one
  32-node quality sample.
- A retry is the narrowest diagnostic that separates stochastic/data-window
  behavior from reproducible 32-node degradation while staying inside the
  E97-MLP avg-outer scope.

## Recommendation For `run-64-node-e97`

Keep `run-64-node-e97` blocked until one bounded 32-node retry has completed
and has been evaluated.

Suggested retry gate:

- Same E97-MLP avg outer scope.
- Resume from the same 16-node final source checkpoint if still retained.
- Use at most one bounded `<=32` node job unless a human explicitly broadens
  scope.
- Record requested and actual node-hours.
- Pass criteria: clean Slurm/process/checkpoint/finalization mechanics and no
  material final-loss regression versus the 16-node source. A repeated
  `FINAL_LOSS_LAST100` near the current `5.8701` class, or a similarly rising
  last-window mean, should trigger a code/config investigation instead of a
  64-node submission.

No 64-node or larger job was submitted from this task.
