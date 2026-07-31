# Frontier E97 32-Node K40 Diagnostic

Date: 2026-06-24
Task: `diagnose-e97-32-node`

## Pre-Launch Decision

The fixed validation gate completed before this diagnostic decision. Task
`fixed-eval-e97-32` reached `done` at `2026-06-24T13:22:26Z`, after recording
all three fixed-eval rows in
`docs/FRONTIER_E97_32NODE_FIXED_EVAL_20260624.md`.

Fixed validation says the 32-node regression is real on the fixed slice, not
only a rank-0 train-loss comparability artifact:

| Checkpoint | Fixed CE | Fixed BPB | BPB delta vs 16-node |
| --- | ---: | ---: | ---: |
| 16-node source, step 1328 | 10.49609756 | 4.36699062 | 0.00000000 |
| 32-node original, step 1780 | 10.68286133 | 4.44469527 | +0.07770465 |
| 32-node retry, step 1740 | 10.67199516 | 4.44017431 | +0.07318369 |

## DiLoCo Audit Summary

The failed 32-node jobs were operationally clean and used the intended plain
averaging path:

- 32 nodes, 256 ranks, `DILOCO_ISLAND_SIZE=8`, therefore 32 islands.
- `DILOCO_OUTER_OPTIMIZER=avg`, `DILOCO_OUTER_LR=1.0`,
  `DILOCO_OUTER_BETA=0.0`, and `DILOCO_EXPORT_BASIS=x`.
- `DILOCO_K=10`, `SAVE_EVERY=10`, and final steps landed exactly on merge
  boundaries.
- `train.py` broadcasts rank-0 weights before DiLoCo training, forms per-island
  DDP groups for within-island gradient all-reduce, and uses a global
  `diloco_merge()` across all ranks at each K boundary.
- For schedule-free inner optimization with avg outer, `diloco_merge()` switches
  to eval mode to average `x`, averages `z`, preserves schedule-free scalar
  clock state, rebuilds train weights, and skips a redundant final merge when
  `step % K == 0`.

That audit does not identify a launch, checkpoint, finalization, or basic avg
merge semantic failure. The most conservative single-variable diagnostic is
therefore to change the merge cadence only.

## Chosen Diagnostic

Submit one bounded 32-node E97-MLP diagnostic from the same 16-node source
checkpoint with `DILOCO_K=40`.

Rationale:

- It changes one scale-control variable only: DiLoCo cadence.
- It keeps the same E97-MLP model, same 16-node source checkpoint, same 32-node
  launch shape, same island size, same stateless avg outer, same optimizer, and
  same no-GDN2/no-CMAES/no-schedule-free scope.
- The repeated K=10 failures suggest that with 32 islands, very frequent global
  averaging may be injecting consensus shocks or over-constraining the
  schedule-free trajectory. K=40 reduces global merge frequency while preserving
  enough merges in a 20-minute bounded smoke to observe directionality.

Committed wrapper:

```text
scripts/frontier/e97_32node_k40_diagnostic.sbatch
```

Planned requested node-hours:

```text
32 nodes * 0.5 hours = 16.000000 node-hours
```

## Submission And Accounting

Exactly one bounded diagnostic training job was submitted from this task:

| Job | Name | Nodes | Walltime | Requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4894517` | `e97-32n-k40-diag` | 32 | `00:30:00` | 16.000000 | `COMPLETED` | `0:0` | `00:20:17` | 10.817778 |

Scheduler accounting:

```text
4894517|e97-32n-k40-diag|batch|COMPLETED|0:0|00:20:17|00:30:00|32|billing=3584,cpu=3584,energy=53232854,mem=16000G,node=32|2026-06-24T09:46:20|2026-06-24T10:06:37
4894517.batch|batch||COMPLETED|0:0|00:20:17||1|cpu=56,mem=500G,node=1|2026-06-24T09:46:20|2026-06-24T10:06:37
4894517.extern|extern||COMPLETED|0:0|00:20:17||32|billing=3584,cpu=3584,mem=16000G,node=32|2026-06-24T09:46:20|2026-06-24T10:06:37
4894517.0|bash||COMPLETED|0:0|00:19:54||32|cpu=1792,mem=16000G,node=32|2026-06-24T09:46:43|2026-06-24T10:06:37
```

One forward-only fixed-eval job was submitted after the training diagnostic
completed. It did not launch training:

| Job | Name | Nodes | Walltime | Requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4894795` | `e97-k40-fixed-eval` | 1 | `01:00:00` | 1.000000 | `COMPLETED` | `0:0` | `00:01:27` | 0.024167 |

Total requested node-hours from this task: `17.000000`.

Total actual node-hours from top-level allocation elapsed times:
`10.841945`.

No second diagnostic training job was submitted.

## Training Evidence

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894517-20260624T134624Z
```

Primary artifacts:

```text
logs/frontier/scaleout/e97-32n-k40-diag-4894517.out
logs/frontier/scaleout/e97-32n-k40-diag-4894517.err
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894517-20260624T134624Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894517-20260624T134624Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894517-20260624T134624Z/summaries/summary.md
```

Manifest-confirmed settings:

```text
task_id=diagnose-e97-32-node
variant=e97-MLP
job_id=4894517
nodes=32
requested_walltime=00:30:00
requested_node_hours=16.0
diloco_outer_optimizer=avg
diloco_export_basis=x
resume_checkpoint=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
exit_status=0
```

Rank 0 logged the intended 32-island K40 configuration:

```text
[DiLoCo] world_size=256 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] periodic model-weight averaging: K=40 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)
[DiLoCo-hybrid] 32 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=40 (subgroup comms warmed sequentially)
```

Training progressed from source step `1328` to final step `2109`.

Loss-window summary from 156 rank-0 metric samples:

| Window | Steps | Mean loss |
| --- | --- | ---: |
| all samples | 1330-2105 | 5.2780 |
| first 20 samples | 1330-1425 | 5.2305 |
| last 20 samples | 2010-2105 | 5.3872 |
| first 10 samples | 1330-1375 | 5.1211 |
| last 10 samples | 2060-2105 | 5.4331 |

Five equal windows:

| Quintile | Steps | Mean loss |
| --- | --- | ---: |
| 1 | 1330-1480 | 5.2544 |
| 2 | 1485-1635 | 5.2674 |
| 3 | 1640-1790 | 5.1965 |
| 4 | 1795-1945 | 5.2611 |
| 5 | 1950-2105 | 5.4066 |

This is materially better than the failed K10 32-node class:

| Run | DILOCO_K | Final trailing loss | First-20 mean | Last-20 mean |
| --- | ---: | ---: | ---: | ---: |
| 16-node source run | 10 | 5.2531 | 5.1588 | 5.4237 |
| 32-node original `4894111` | 10 | 5.8701 | 5.4394 | 6.1958 |
| 32-node retry `4894206` | 10 | 5.8164 | 5.4486 | 6.1179 |
| 32-node K40 diagnostic `4894517` | 40 | 5.2822 | 5.2305 | 5.3872 |

Throughput summary from rank-0 metric samples:

```text
global_tok_median=576538
global_tok_mean=496593
```

DiLoCo and finalization summary:

```text
>>> [DiLoCo] FINAL merge #20 at step 2109: consensus model averaged across 256 ranks (4216 ms)
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=2109 loss=5.2822 remaining_s=595.2 ...
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894517-20260624T134624Z/train/emender_E97_1.3B_20260624_094942/checkpoint_step_002109_loss_5.2822.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894517-20260624T134624Z/train/emender_E97_1.3B_20260624_094942/latest.pt ...
Training complete! Final step: 2109
FINAL_LOSS_LAST100: 5.2822
DILOCO_MERGES: 20
DILOCO_K: 40
DILOCO_SYNC_TOTAL_S: 87.531
DILOCO_SYNC_AVG_MS: 4376.6
```

Final retained checkpoints:

```text
checkpoint_step_002000_loss_5.5250.pt
checkpoint_step_002040_loss_5.3793.pt
checkpoint_step_002080_loss_5.5170.pt
checkpoint_step_002109_loss_5.2822.pt
latest.pt -> checkpoint_step_002109_loss_5.2822.pt
```

Fatal-signature scans over Slurm stdout, Slurm stderr, and `train.log` found
no traceback, runtime error, exception, non-finite loss, NaN, OOM/out-of-memory,
watchdog, collective mismatch, timeout, cancellation, fatal signal, or failed
job signature. Observed warnings were the same non-blocking class as prior
Frontier runs: module reload notices, AMD low-power-state warning, Triton/Python
recommendation warnings, and PyTorch/NCCL rank-to-GPU startup warnings.

## Fixed Validation Evidence

Forward-only fixed eval job `4894795` scored the 16-node source and K40 final
checkpoint on the same saved fixed tensor used by `fixed-eval-e97-32`:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Result CSV:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP-k40/4894795-20260624T140942Z/artifacts/e97_fixed_eval_k40_diagnostic.csv
```

Fixed validation rows:

| Checkpoint | Step | Fixed CE | Fixed BPB | CE delta vs 16-node | BPB delta vs 16-node |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-node source | 1328 | 10.49609756 | 4.36699062 | 0.00000000 | 0.00000000 |
| 32-node K40 diagnostic | 2109 | 10.62739754 | 4.42161909 | +0.13129998 | +0.05462847 |

Interpretation: K40 substantially improves the 32-node scale-out behavior
relative to the K10 failures, but it has not fully cleared the fixed-validation
gate against the 16-node source on this small fixed slice. For comparison, the
K10 fixed BPB deltas were `+0.07770465` and `+0.07318369`; K40 reduces that
delta to `+0.05462847`.

## Recommendation

Do not resume or submit `run-64-node-e97` yet.

The K40 diagnostic is a useful scale-control signal: changing only DiLoCo
cadence from K10 to K40 repaired the 32-node train-loss dynamics and produced a
clean checkpoint/finalization path. However, the same fixed validation slice
still shows K40 worse than the 16-node source, only less worse than K10.

Recommended next gate:

- Treat `DILOCO_K` as the leading 32-node scale-control variable.
- Before any 64-node run, require a 32-node E97-MLP checkpoint to pass both:
  clean Slurm/process/DiLoCo/checkpoint/finalization mechanics and row-matched
  fixed-validation CE/BPB that is not worse than the 16-node source beyond a
  small predeclared tolerance.
- Rank-0 train loss alone is not a sufficient gate. It is useful as a fast
  smoke signal, but the acceptance gate should be the fixed validation CE/BPB
  comparison used here, ideally on a slightly larger fixed slice once cost is
  acceptable.
- A follow-up should tune only the cadence family next, for example one
  predeclared `DILOCO_K` value larger than 40 or a short cadence bracket, rather
  than switching to GDN2, CMAES, or broad schedule-free exploration.

## Scope Confirmations

- `run-64-node-e97` remains open and paused.
- No 64+ node job was submitted or authorized from this task.
- Exactly one bounded `<=32` node E97-MLP diagnostic training job was submitted.
- The only post-training extra job was one one-node forward-only fixed eval.
- GDN2, CMAES, and broad schedule-free exploration remained out of scope.
