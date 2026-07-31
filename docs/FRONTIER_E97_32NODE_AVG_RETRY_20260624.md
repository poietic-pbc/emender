# Frontier E97 32-Node Avg Diagnostic Retry

Date: 2026-06-24
Task: `retry-32-node-e97`

## Verdict

FAIL for the 64-node gate. The bounded 32-node E97-MLP avg-outer retry was
operationally clean, but it repeated the loss regression class seen in job
`4894111`.

Do not launch `run-64-node-e97` yet. The next action should be a focused
32-node scale-out/configuration investigation, not another larger scale-out
submission. This retry used the same retained 16-node final source checkpoint,
stayed in the E97-MLP avg-outer scope, and submitted no 64-node or larger job.

## Submitted Job

Exactly one bounded diagnostic retry was submitted:

| Job | Partition | QOS | Nodes | Walltime | Max requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4894206` | `batch` | `normal` | 32 | `00:30:00` | `16.000000` | `COMPLETED` | `0:0` | `00:20:22` | `10.862222` |

Actual node-hours are computed from the top-level allocation elapsed time:
`32 * (20m22s / 3600) = 10.862222`.

Scheduler accounting:

```text
4894206|e97-32n-avg-retry|batch|normal|COMPLETED|0:0|00:20:22|00:30:00|32|billing=3584,cpu=3584,energy=41814508,mem=16000G,node=32|2026-06-24T07:53:13|2026-06-24T07:53:40|2026-06-24T08:14:02
4894206.batch|batch|||COMPLETED|0:0|00:20:22||1|cpu=56,mem=500G,node=1|2026-06-24T07:53:40|2026-06-24T07:53:40|2026-06-24T08:14:02
4894206.extern|extern|||COMPLETED|0:0|00:20:22||32|billing=3584,cpu=3584,mem=16000G,node=32|2026-06-24T07:53:40|2026-06-24T07:53:40|2026-06-24T08:14:02
4894206.0|bash|||COMPLETED|0:0|00:20:01||32|cpu=1792,mem=16000G,node=32|2026-06-24T07:54:01|2026-06-24T07:54:01|2026-06-24T08:14:02
```

No additional diagnostic retry and no 64-node-or-larger job were submitted from
this task.

## Source Checkpoint

The retry resumed from the same 16-node final source checkpoint used by the
original 32-node smoke:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

Before submission, that symlink was still retained and resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/checkpoint_step_001328_loss_5.2531.pt
```

The retry train log repeatedly recorded the expected resume path and rank 0
formed the intended 256-rank/32-island job:

```text
Resuming from /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
[DiLoCo] world_size=256 backend=nccl; this is rank 0 on cuda:0
[DiLoCo-hybrid] 32 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=10 (subgroup comms warmed sequentially)
```

## Run Configuration

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z
```

Primary artifacts inspected:

```text
logs/frontier/scaleout/e97-32n-avg-retry-4894206.out
logs/frontier/scaleout/e97-32n-avg-retry-4894206.err
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/summaries/summary.md
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/train/emender_E97_1.3B_20260624_075706/latest.pt
```

Manifest-confirmed settings:

```text
task_id=retry-32-node-e97
variant=e97-MLP
job_id=4894206
job_name=e97-32n-avg-retry
nodes=32
requested_walltime=00:30:00
requested_node_hours=16
diloco_outer_optimizer=avg
diloco_export_basis=x
resume_checkpoint=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
exit_status=0
```

The Slurm stdout command line also recorded the intended bounded retry
settings: `TRAIN_MINUTES=20`, `DILOCO_K=10`, `DILOCO_ISLAND_SIZE=8`,
`DILOCO_OUTER_OPTIMIZER=avg`, `DILOCO_OUTER_LR=1.0`,
`DILOCO_OUTER_BETA=0.0`, `DILOCO_EXPORT_BASIS=x`, `SAVE_EVERY=10`, and
`KEEP_CHECKPOINTS=4`.

## Loss Findings

The retry progressed from source step `1328` to final step `1740`. It wrote
`42` DiLoCo merges and ended with:

```text
Training complete! Final step: 1740
FINAL_LOSS_LAST100: 5.8164
DILOCO_MERGES: 42
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 191.752
DILOCO_SYNC_AVG_MS: 4565.5
```

Final trailing metric comparison:

| Run | Source final loss | Final loss | Delta |
| --- | ---: | ---: | ---: |
| 16-node avg source | n/a | 5.2531 | n/a |
| 32-node avg smoke `4894111` | 5.2531 | 5.8701 | +0.6170 |
| 32-node avg retry `4894206` | 5.2531 | 5.8164 | +0.5633 |

The retry is slightly less bad than the first 32-node smoke by final trailing
loss, but still materially regresses versus the identical 16-node source
checkpoint.

Comparable rank-0 metric-window summaries:

| Run | Samples | Steps | Mean loss | First-20 mean | Last-20 mean | First-10 mean | Last-10 mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16-node source run | 71 | 975-1325 | 5.2531 | 5.1588 | 5.4237 | 5.2110 | 5.5150 |
| 32-node smoke `4894111` | 91 | 1330-1780 | 5.8701 | 5.4394 | 6.1958 | 5.1964 | 6.2543 |
| 32-node retry `4894206` | 83 | 1330-1740 | 5.8164 | 5.4486 | 6.1179 | 5.1954 | 6.1421 |

Retry quintiles show the same coherent upward pattern:

| Retry window | Steps | Mean loss |
| --- | --- | ---: |
| quintile 1 | 1330-1405 | 5.4096 |
| quintile 2 | 1410-1490 | 5.7280 |
| quintile 3 | 1495-1570 | 5.8462 |
| quintile 4 | 1575-1655 | 5.9570 |
| quintile 5 | 1660-1740 | 6.1189 |

This repeats the original 32-node failure mode rather than clearing it.

## Checkpoint And Retention State

Finalization triggered from the Slurm walltime margin. Because final step
`1740` already landed on the regular `DILOCO_K=10` merge cadence, the final
merge was skipped correctly:

```text
>>> [DiLoCo] final merge SKIPPED at step 1740: last step already merged (step % K == 0); checkpoint is already consensus (avoids spurious outer-momentum double-step)
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=1740 loss=5.8164 remaining_s=590.3 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/256 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/train/emender_E97_1.3B_20260624_075706/checkpoint_step_001740_loss_5.8164.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/train/emender_E97_1.3B_20260624_075706/latest.pt model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/256 is_head=True
```

Final retained checkpoint state:

```text
checkpoint_step_001720_loss_6.3747.pt
checkpoint_step_001730_loss_5.9560.pt
checkpoint_step_001740_loss_5.8164.pt
checkpoint_step_001740_loss_6.5654.pt
latest.pt -> checkpoint_step_001740_loss_5.8164.pt
```

`readlink -f latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894206-20260624T115342Z/train/emender_E97_1.3B_20260624_075706/checkpoint_step_001740_loss_5.8164.pt
```

No checkpoint temporary, partial, or incomplete files were found in the final
run directory.

## Warning And Failure Scan

The Slurm job completed with exit `0:0`. Final scans over Slurm stdout,
Slurm stderr, and the train log found no traceback, runtime error, exception,
non-finite loss, NaN, OOM/out-of-memory, watchdog, collective mismatch,
timeout, cancellation, fatal signal, or failed-job signature.

Observed warnings match the previous clean operational smoke class:

- module replacement/reload notices during launch;
- Frontier low-power GPU state warning before training;
- PyTorch/NCCL rank-to-GPU startup warnings;
- Triton 3.2.0 recommendation warning.

These warnings did not prevent process-group formation, 42 DiLoCo merges,
final checkpointing, or clean Slurm completion.

## Recommendation For `run-64-node-e97`

Keep `run-64-node-e97` blocked.

The diagnostic question was whether the first 32-node loss regression was
stochastic/noisy or reproducible. The retry reproduced the problem:
`FINAL_LOSS_LAST100` remained in the same bad class (`5.8164` retry,
`5.8701` original) versus the same source checkpoint (`5.2531`), and the
within-run window again rose from roughly `5.45` early to above `6.11` late.

The next WG task should inspect 32-node E97-MLP avg configuration and scale-out
behavior before any larger run. Useful targets include learning-rate/optimizer
state continuity after resume, data-window comparability around the resumed
steps, 32-island averaging cadence semantics, and whether the 32-node loss
shock is specific to E97-MLP avg or to the current 32-island launch shape. Do
not submit a 64-node E97 avg job until that investigation produces a concrete
fix or a stronger acceptance criterion than this retry provides.
