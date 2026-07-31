# Frontier E97 32-Node K80 Diagnostic

Date: 2026-06-25
Task: `run-e97-32-node`

## Pre-Launch Decision

Task `plan-e97-32-node` was complete before this training decision. Its plan
selected a cadence-only 32-node E97-MLP diagnostic as the next rung:
same 16-node source checkpoint, same island size 8, same stateless avg outer,
same fixed validation tensor, and `DILOCO_K=80`.

Fixed validation tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

16-node source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

## Planned Diagnostic

Committed training wrapper:

```text
scripts/frontier/e97_32node_k80_diagnostic.sbatch
```

Planned shape:

```text
32 Frontier nodes, E97-MLP, resume from the 16-node source checkpoint,
DILOCO_ISLAND_SIZE=8, DILOCO_OUTER_OPTIMIZER=avg,
DILOCO_OUTER_LR=1.0, DILOCO_OUTER_BETA=0.0,
DILOCO_EXPORT_BASIS=x, DILOCO_K=80, SAVE_EVERY=80,
TRAIN_MINUTES=20, requested walltime=00:30:00.
```

Requested training node-hours:

```text
32 nodes * 0.5 hours = 16.000000 node-hours
```

## Submission And Accounting

Exactly one bounded diagnostic training job was submitted from this task:

| Job | Name | Nodes | Walltime | Requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4899126` | `e97-32n-k80-diag` | 32 | `00:30:00` | 16.000000 | `COMPLETED` | `0:0` | `00:20:18` | 10.826667 |

Scheduler accounting:

```text
4899126|e97-32n-k80-diag|COMPLETED|0:0|00:20:18|00:30:00|32|billing=3584,cpu=3584,energy=59226408,mem=16000G,node=32|2026-06-25T04:49:05|2026-06-25T05:09:23
4899126.batch|batch|COMPLETED|0:0|00:20:18||1|cpu=56,mem=500G,node=1|2026-06-25T04:49:05|2026-06-25T05:09:23
4899126.extern|extern|COMPLETED|0:0|00:20:18||32|billing=3584,cpu=3584,mem=16000G,node=32|2026-06-25T04:49:05|2026-06-25T05:09:23
4899126.0|bash|COMPLETED|0:0|00:19:56||32|cpu=1792,mem=16000G,node=32|2026-06-25T04:49:27|2026-06-25T05:09:23
```

One forward-only fixed-eval job was submitted after the training diagnostic
completed. It did not launch training:

| Job | Name | Nodes | Walltime | Requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4899139` | `e97-k80-fixed-eval` | 1 | `01:00:00` | 1.000000 | `COMPLETED` | `0:0` | `00:01:25` | 0.023611 |

Total requested node-hours from this task: `17.000000`.

Total actual node-hours from top-level allocation elapsed times: `10.850278`.

## Training Evidence

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899126-20260625T084907Z
```

Primary artifacts:

```text
logs/frontier/scaleout/e97-32n-k80-diag-4899126.out
logs/frontier/scaleout/e97-32n-k80-diag-4899126.err
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899126-20260625T084907Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899126-20260625T084907Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899126-20260625T084907Z/summaries/summary.md
```

Manifest-confirmed settings:

```text
task_id=run-e97-32-node
variant=e97-MLP
job_id=4899126
nodes=32
requested_walltime=00:30:00
requested_node_hours=16.0
diloco_outer_optimizer=avg
diloco_export_basis=x
resume_checkpoint=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
exit_status=0
```

Rank 0 logged the intended 32-island K80 configuration:

```text
[DiLoCo] world_size=256 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] periodic model-weight averaging: K=80 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)
[DiLoCo-hybrid] 32 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=80 (subgroup comms warmed sequentially)
```

Training progressed from source step `1328` to final step `2279`.

Loss-window summary from 190 rank-0 metric samples:

| Window | Steps | Mean loss |
| --- | --- | ---: |
| all samples | 1330-2275 | 5.1488 |
| first 20 samples | 1330-1425 | 5.2264 |
| last 20 samples | 2180-2275 | 5.1834 |
| first 10 samples | 1330-1375 | 5.1208 |
| last 10 samples | 2230-2275 | 5.2306 |

Five equal windows:

| Quintile | Steps | Mean loss |
| --- | --- | ---: |
| 1 | 1330-1515 | 5.2747 |
| 2 | 1520-1705 | 5.1993 |
| 3 | 1710-1895 | 5.0095 |
| 4 | 1900-2085 | 5.1441 |
| 5 | 2090-2275 | 5.1164 |

This is materially better than the failed K10 32-node class and better than the
K40 trailing-loss checkpoint result:

| Run | DILOCO_K | Final trailing loss | First-20 mean | Last-20 mean |
| --- | ---: | ---: | ---: | ---: |
| 16-node source run | 10 | 5.2531 | 5.1588 | 5.4237 |
| 32-node original `4894111` | 10 | 5.8701 | 5.4394 | 6.1958 |
| 32-node retry `4894206` | 10 | 5.8164 | 5.4486 | 6.1179 |
| 32-node K40 diagnostic `4894517` | 40 | 5.2822 | 5.2305 | 5.3872 |
| 32-node K80 diagnostic `4899126` | 80 | 5.1084 | 5.2264 | 5.1834 |

Throughput summary from rank-0 metric samples:

```text
global_tok_median=572196
global_tok_mean=529179
```

DiLoCo and finalization summary:

```text
>>> [DiLoCo] FINAL merge #13 at step 2279: consensus model averaged across 256 ranks (4601 ms)
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=2279 loss=5.1084 remaining_s=594.7 ...
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899126-20260625T084907Z/train/emender_E97_1.3B_20260625_045105/checkpoint_step_002279_loss_5.1084.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899126-20260625T084907Z/train/emender_E97_1.3B_20260625_045105/latest.pt ...
Training complete! Final step: 2279
FINAL_LOSS_LAST100: 5.1084
DILOCO_MERGES: 13
DILOCO_K: 80
DILOCO_SYNC_TOTAL_S: 62.182
DILOCO_SYNC_AVG_MS: 4783.2
```

Final retained checkpoints:

```text
checkpoint_step_002080_loss_5.1636.pt
checkpoint_step_002160_loss_5.0673.pt
checkpoint_step_002240_loss_5.1349.pt
checkpoint_step_002279_loss_5.1084.pt
latest.pt -> checkpoint_step_002279_loss_5.1084.pt
```

Fatal-signature scans over Slurm stdout, Slurm stderr, and `train.log` found
no traceback, runtime error, exception, non-finite loss, NaN, OOM/out-of-memory,
watchdog, collective mismatch, timeout, cancellation, fatal signal, or failed
job signature. Observed warnings were the same non-blocking class as prior
Frontier runs: module reload notices, AMD low-power-state warning, and
Triton/Python recommendation warnings.

## Fixed Validation Evidence

Committed forward-only fixed-eval wrapper:

```text
scripts/frontier/e97_fixed_eval_k80_diagnostic.sbatch
```

The wrapper requires `CKPT_K80` to point at the completed K80 `latest.pt` or
final checkpoint and scores the 16-node source versus K80 on the same fixed
tensor.

Forward-only fixed eval job `4899139` scored the 16-node source and K80 final
checkpoint on the same saved fixed tensor used by `fixed-eval-e97-32` and the
K40 diagnostic.

Result CSV:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-MLP-k80/4899139-20260625T091116Z/artifacts/e97_fixed_eval_k80_diagnostic.csv
```

Fixed validation rows:

| Checkpoint | Step | Fixed CE | Fixed BPB | CE delta vs 16-node | BPB delta vs 16-node |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-node source | 1328 | 10.49609756 | 4.36699062 | 0.00000000 | 0.00000000 |
| 32-node K80 diagnostic | 2279 | 10.57016373 | 4.39780647 | +0.07406617 | +0.03081585 |

Green gate from `plan-e97-32-node`: fixed BPB delta versus the 16-node source
`<= +0.010` and fixed CE delta `<= +0.025` on this smoke tensor. A result with
`0.010 < BPB delta <= 0.05462847` improves over K40 but still fails the 32-node
gate; worse than K40 worsens the cadence direction.

Verdict: K80 **improves but fails** the 32-node gate. It is operationally clean,
has a better train-loss trajectory than the 16-node/K40 class, and improves the
fixed BPB delta from K40's `+0.05462847` to `+0.03081585`. It does not clear
the green threshold because CE delta is `+0.07406617` and BPB delta is
`+0.03081585`, both above the K80 gate (`+0.025` CE and `+0.010` BPB).

## Scope Confirmations

- Exactly one bounded 32-node E97-MLP training diagnostic was submitted from
  this task: job `4899126`.
- No 64+ node job was submitted from this task.
- No GDN2, CMAES, or schedule-free run was submitted from this task.
- `run-64-node-e97` remains `open (PAUSED)`.
