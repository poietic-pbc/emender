# Frontier E97 8-Node Avg Continuation

Date: 2026-06-24
Task: `continue-8-node-e97`

## Verdict

PASS. The bounded 8-node E97-MLP DiLoCo regular-averaging continuation
resumed from the prior 8-node smoke checkpoint, progressed from step `451` to
step `970`, completed cleanly with Slurm exit `0:0`, wrote regular checkpoints
approximately every 10 steps, wrote a final checkpoint, advanced `latest.pt` to
the final checkpoint, and retained a bounded checkpoint set.

The continuation loss evidence is positive: prior baseline final
`FINAL_LOSS_LAST100` was `5.7391`; this continuation ended at
`FINAL_LOSS_LAST100: 5.2538`.

No 16-node, 32-node, or 64-node jobs were submitted. No schedule-free outer run
was submitted. No GDN2 control was submitted.

## Upstream State

- Confirmed before launch that `origin/main` and `HEAD` were both
  `3242926b51e3ea57d08cac505382daf32aabcb42`.
- Confirmed `origin/main` includes
  `docs/FRONTIER_E97_8NODE_AVG_SMOKE_20260624.md`.
- Runtime commit recorded by the job:
  `3242926b51e3ea57d08cac505382daf32aabcb42`.

## Source Checkpoint

The task text named:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092536Z
```

That path did not exist. The integrated 8-node smoke report records the actual
successful job `4893934` run root as:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z
```

The corrected source checkpoint was:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/train/emender_E97_1.3B_20260624_052918/latest.pt
```

`latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/train/emender_E97_1.3B_20260624_052918/checkpoint_step_000451_loss_5.7391.pt
```

The checkpoint step `451` is equal to the prior final checkpoint step recorded
in the 8-node smoke report, satisfying the resume-step gate.

## Submitted Job

Exactly one bounded 8-node continuation was submitted:

| Job | QOS | Nodes | Walltime | Max requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4893959` | debug | 8 | `00:30:00` | `4.000000` | `COMPLETED` | `0:0` | `00:20:17` | `2.704444` |

Actual node-hours are computed from the top-level allocation elapsed time:
`8 * (20m17s / 3600) = 2.704444`.

Scheduler accounting:

```text
4893959|e97-8n-avg-cont|batch|debug|COMPLETED|0:0|00:20:17|00:30:00|8|billing=896,cpu=896,energy=11594840,mem=4000G,node=8|2026-06-24T05:56:22|2026-06-24T06:16:39
4893959.batch|batch|||COMPLETED|0:0|00:20:17||1|cpu=56,mem=500G,node=1|2026-06-24T05:56:22|2026-06-24T06:16:39
4893959.extern|extern|||COMPLETED|0:0|00:20:17||8|billing=896,cpu=896,mem=4000G,node=8|2026-06-24T05:56:22|2026-06-24T06:16:39
4893959.0|bash|||COMPLETED|0:0|00:19:55||8|cpu=448,mem=4000G,node=8|2026-06-24T05:56:44|2026-06-24T06:16:39
```

## Run Configuration

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z
```

Primary artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/summaries/summary.md
```

Key launch settings:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=8
SCALEOUT_WALLTIME=00:30:00
TRAIN_MINUTES=20
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
DILOCO_K=10
DILOCO_ISLAND_SIZE=8
DILOCO_OUTER_OPTIMIZER=avg
DILOCO_OUTER_LR=1.0
DILOCO_OUTER_BETA=0.0
DILOCO_EXPORT_BASIS=x
BATCH_SIZE=1
CHUNK_SIZE=2048
LOG_EVERY=5
VAL_EVERY=10000
SAVE_EVERY=10
KEEP_CHECKPOINTS=4
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
```

Run label:

```text
emender_E97_1.3B_20260624_055747
```

Rank 0 logged the expected 64-rank, 8-island, avg-outer configuration:

```text
[DiLoCo] world_size=64 backend=nccl; this is rank 0 on cuda:0
[DiLoCo-hybrid] 8 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=10 (subgroup comms warmed sequentially)
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
```

## Resume And Progress

All ranks loaded the source checkpoint and rank 0 logged:

```text
Resumed at step 451
Starting training from step 451...
```

Representative loss and throughput samples:

```text
step    455 | loss 4.0856 | lr 9.00e-04 | grad 1.03 | tok/s 2036 | global_tok/s 130289 | elapsed_h 0.001 | time 2026-06-24T09:58:08+00:00
step    500 | loss 5.0507 | lr 9.00e-04 | grad 1.22 | tok/s 1206 | global_tok/s 77160 | elapsed_h 0.026 | time 2026-06-24T09:59:38+00:00
step    690 | loss 5.0149 | lr 9.00e-04 | grad 1.71 | tok/s 1198 | global_tok/s 76661 | elapsed_h 0.138 | time 2026-06-24T10:06:22+00:00
step    870 | loss 4.5319 | lr 9.00e-04 | grad 1.08 | tok/s 1207 | global_tok/s 77229 | elapsed_h 0.244 | time 2026-06-24T10:12:43+00:00
step    970 | loss 5.5773 | lr 9.00e-04 | grad 0.96 | tok/s 1202 | global_tok/s 76900 | elapsed_h 0.303 | time 2026-06-24T10:16:16+00:00
```

Loss-window summary from the 104 rank-0 metric samples:

```text
samples=104 first_step=455 last_step=970 mean_all=5.2380 first20_mean=5.0906 last20_mean=5.3702
```

The last-window average is higher than the first-window average, so the short
within-run sample is noisy and does not show monotonic local improvement.
However, the final trailing metric is the primary comparison requested here and
is clearly below the prior 8-node smoke baseline:

```text
Prior 8-node final FINAL_LOSS_LAST100: 5.7391
Continuation FINAL_LOSS_LAST100:       5.2538
Delta:                                -0.4853
```

Summary metrics:

```text
Training complete! Final step: 970
FINAL_LOSS_LAST100: 5.2538
DILOCO_MERGES: 52
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 215.922
DILOCO_SYNC_AVG_MS: 4152.3
```

## Checkpoint And Retention Verification

Regular checkpoints were written every 10 steps after resume. The finalization
path triggered at step `970` due to the Slurm walltime margin. Because step
`970` had already completed a normal DiLoCo merge, the final merge was skipped
as already-consensus:

```text
>>> [DiLoCo] final merge SKIPPED at step 970: last step already merged (step % K == 0); checkpoint is already consensus (avoids spurious outer-momentum double-step)
```

Final checkpoint status:

```text
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=970 loss=5.2538 remaining_s=596.4 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/64 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/train/emender_E97_1.3B_20260624_055747/checkpoint_step_000970_loss_5.2538.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/train/emender_E97_1.3B_20260624_055747/latest.pt model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/64 is_head=True
```

Final retained run-directory contents:

```text
checkpoint_step_000950_loss_5.5007.pt
checkpoint_step_000960_loss_5.2058.pt
checkpoint_step_000970_loss_5.5773.pt
checkpoint_step_000970_loss_5.2538.pt
latest.pt -> checkpoint_step_000970_loss_5.2538.pt
```

The duplicate step-970 files are expected from the regular save at the save
boundary followed by the final checkpoint with the last-100 loss. Retention
remained bounded at four checkpoint files, and no checkpoint temporary files
remained after completion.

`readlink -f latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/train/emender_E97_1.3B_20260624_055747/checkpoint_step_000970_loss_5.2538.pt
```

## Communication And Warning Scan

No NCCL/RCCL watchdog timeout, collective mismatch, traceback, runtime error,
OOM, timeout, or nonzero-exit signature was found in the final scan. Slurm
accounting reports `COMPLETED` with exit `0:0`.

Non-blocking warnings matched prior runs:

- PyTorch/NCCL emitted the rank-to-GPU startup warning, but the job completed
  52 DiLoCo merges, final checkpointing, and clean process exit.
- Triton 3.2.0 emitted the existing recommendation warning for 3.3.0.
- Frontier reported low-power GPU state in stderr before training; the run
  proceeded normally.

## Recommendation

Next task should be a bounded 16-node E97-MLP avg smoke. Rationale: 8-node avg
now has clean launch, checkpoint retention, resume, final checkpoint,
`latest.pt`, clean finalization, and productive continuation evidence with
`FINAL_LOSS_LAST100` improving from `5.7391` to `5.2538`. Keep the same
guardrails: current `origin/main`, explicit shared `OUTPUT_ROOT`, `DILOCO_K=10`,
checkpoint interval around 10 steps, bounded retention, and short debug
walltime. Schedule-free outer should remain a separate comparison task after
the next avg scale step if systems scale-out remains the immediate priority.
