# Frontier E97 16-Node Avg Scaleout Smoke

Date: 2026-06-24
Task: `run-16-node-e97`

## Verdict

PASS. The bounded 16-node E97-MLP DiLoCo regular-averaging smoke resumed from
the clean 8-node continuation checkpoint, launched exactly one 16-node job,
ran to the walltime finalization margin, completed cleanly with Slurm exit
`0:0`, wrote regular checkpoints every 10 steps, wrote a final consensus
checkpoint, advanced `latest.pt` to the final checkpoint, and retained a
bounded checkpoint set.

No 32-node or 64-node job was submitted. No schedule-free outer run was
submitted. No GDN2 run was submitted.

## Upstream 8-Node Gate

The prerequisite `continue-8-node-e97` was checked before submission through
`docs/FRONTIER_E97_8NODE_AVG_CONTINUATION_20260624.md` and WG dependency logs.

The 8-node continuation passed:

- Job `4893959` completed with Slurm exit `0:0`.
- It resumed from step `451` and reached final step `970`.
- It recorded `DILOCO_MERGES: 52`.
- It advanced `latest.pt` to the final step-970 checkpoint.
- Its final loss comparison was productive:
  - prior 8-node smoke `FINAL_LOSS_LAST100: 5.7391`
  - 8-node continuation `FINAL_LOSS_LAST100: 5.2538`
  - delta `-0.4853`
- No NCCL/RCCL watchdog timeout or collective mismatch signature was reported.

This satisfied the gate to proceed with the 16-node smoke.

## Submitted Job

Exactly one bounded 16-node E97-MLP avg smoke was submitted:

| Job | Partition | QOS | Nodes | Walltime | Max requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4893977` | `batch` | `normal` | 16 | `00:30:00` | `8.000000` | `COMPLETED` | `0:0` | `00:20:18` | `5.413333` |

Actual node-hours are computed from the top-level allocation elapsed time:
`16 * (20m18s / 3600) = 5.413333`.

Scheduler accounting:

```text
4893977|e97-16n-avg-smoke|batch|normal|COMPLETED|0:0|00:20:18|00:30:00|16|billing=1792,cpu=1792,energy=18969404,mem=8000G,node=16|2026-06-24T06:26:44|2026-06-24T06:30:40|2026-06-24T06:50:58
4893977.batch|batch|||COMPLETED|0:0|00:20:18||1|cpu=56,mem=500G,node=1|2026-06-24T06:30:40|2026-06-24T06:30:40|2026-06-24T06:50:58
4893977.extern|extern|||COMPLETED|0:0|00:20:19||16|billing=1792,cpu=1792,mem=8000G,node=16|2026-06-24T06:30:40|2026-06-24T06:30:40|2026-06-24T06:50:59
4893977.0|bash|||COMPLETED|0:0|00:19:53||16|cpu=896,mem=8000G,node=16|2026-06-24T06:31:05|2026-06-24T06:31:05|2026-06-24T06:50:58
```

The job initially queued in normal QOS for `Priority` and consumed zero
node-hours while pending. It was not cancelled or resubmitted.

## Source State

Before submission, `origin/main` and `HEAD` were both:

```text
b6d486c178369365a80d15a21d737bd23593d702
```

The runtime manifest recorded the same git commit:

```text
b6d486c178369365a80d15a21d737bd23593d702
```

## Source Checkpoint

The 16-node smoke resumed from the 8-node continuation `latest.pt`:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/train/emender_E97_1.3B_20260624_055747/latest.pt
```

Before submission, that symlink resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893959-20260624T095625Z/train/emender_E97_1.3B_20260624_055747/checkpoint_step_000970_loss_5.2538.pt
```

All ranks loaded the checkpoint, and rank 0 logged:

```text
Resumed at step 970
Starting training from step 970...
```

## Run Configuration

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z
```

Primary artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/summaries/summary.md
logs/frontier/scaleout/e97-16n-avg-smoke-4893977.out
logs/frontier/scaleout/e97-16n-avg-smoke-4893977.err
```

Run label:

```text
emender_E97_1.3B_20260624_063731
```

Key launch settings:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=16
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

Rank 0 logged the expected 128-rank, 16-island, avg-outer configuration:

```text
[DiLoCo] world_size=128 backend=nccl; this is rank 0 on cuda:0
[DiLoCo-hybrid] 16 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=10 (subgroup comms warmed sequentially)
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
```

## Loss And Throughput

The run progressed from source step `970` to final step `1328`.

Representative rank-0 metric lines:

```text
step    975 | loss 5.4350 | lr 9.00e-04 | grad 1.08 | tok/s 1405 | global_tok/s 179779 | elapsed_h 0.002 | time 2026-06-24T10:38:02+00:00
step   1000 | loss 4.9058 | lr 9.00e-04 | grad 1.21 | tok/s 1178 | global_tok/s 150772 | elapsed_h 0.016 | time 2026-06-24T10:38:53+00:00
step   1100 | loss 5.4560 | lr 9.00e-04 | grad 1.03 | tok/s 1182 | global_tok/s 151291 | elapsed_h 0.075 | time 2026-06-24T10:42:27+00:00
step   1200 | loss 5.1246 | lr 9.00e-04 | grad 0.96 | tok/s 1180 | global_tok/s 151020 | elapsed_h 0.136 | time 2026-06-24T10:46:04+00:00
step   1300 | loss 5.9739 | lr 9.00e-04 | grad 2.25 | tok/s 1152 | global_tok/s 147476 | elapsed_h 0.196 | time 2026-06-24T10:49:41+00:00
step   1325 | loss 5.3282 | lr 9.00e-04 | grad 1.36 | tok/s 821 | global_tok/s 105046 | elapsed_h 0.212 | time 2026-06-24T10:50:37+00:00
```

Loss-window summary from the 71 rank-0 metric samples:

```text
samples=71 first_step=975 last_step=1325 mean_all=5.2531 first20_mean=5.1588 last20_mean=5.4237
global_tok_median=124898 global_tok_mean=127386
```

The short within-run window is noisy and the last-20 mean is higher than the
first-20 mean. The final trailing metric is effectively flat against the
source 8-node continuation final loss:

```text
8-node continuation FINAL_LOSS_LAST100: 5.2538
16-node smoke FINAL_LOSS_LAST100:       5.2531
Delta:                                -0.0007
```

For a bounded launch/checkpoint/finalization smoke, this is acceptable:
training stayed finite, resumed correctly, advanced 358 steps, completed 36
DiLoCo merges, and did not regress the trailing final-loss comparison.

Summary metrics:

```text
Training complete! Final step: 1328
FINAL_LOSS_LAST100: 5.2531
DILOCO_MERGES: 36
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 151.728
DILOCO_SYNC_AVG_MS: 4214.7
```

## Checkpoint And Retention Verification

Regular checkpoints were written every 10 steps after resume through step
`1320`. Finalization triggered at step `1328` due to the Slurm walltime margin.
All ranks entered finalization, and rank 0 performed the final consensus merge
and checkpoint write:

```text
>>> [DiLoCo] FINAL merge #36 at step 1328: consensus model averaged across 128 ranks (4145 ms)
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=1328 loss=5.2531 remaining_s=595.4 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/128 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/checkpoint_step_001328_loss_5.2531.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/128 is_head=True
```

Final retained run-directory contents:

```text
checkpoint_step_001300_loss_5.9739.pt
checkpoint_step_001310_loss_4.9961.pt
checkpoint_step_001320_loss_5.8916.pt
checkpoint_step_001328_loss_5.2531.pt
latest.pt -> checkpoint_step_001328_loss_5.2531.pt
```

`readlink -f latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/checkpoint_step_001328_loss_5.2531.pt
```

No checkpoint temporary or partial files remained after completion.

## Communication And Warning Scan

Final scans over the train log, Slurm stdout, and Slurm stderr found no
watchdog, collective mismatch, timeout, traceback, runtime error, exception,
non-finite, NaN, OOM, out-of-memory, failed, or cancelled signature. Slurm
accounting reports `COMPLETED` with exit `0:0`.

Non-blocking warnings matched the prior clean 8-node runs:

- PyTorch/NCCL emitted the rank-to-GPU startup warning. The job still completed
  36 DiLoCo merges, final consensus averaging, final checkpointing, and clean
  process exit.
- c10d emitted transient IPv6 socket initialization warnings during rendezvous.
  The process group initialized and trained normally afterward.
- Triton 3.2.0 emitted the existing recommendation warning for 3.3.0.
- Frontier reported low-power GPU state in stderr before training; the run
  proceeded normally.

## Recommendation

Proceed to the bounded 32-node E97-MLP avg smoke. Rationale: the 16-node job
satisfies the launch, resume, checkpoint cadence, final consensus checkpoint,
`latest.pt`, retention, communication-cleanliness, and clean-exit gates. The
loss signal is flat-to-slightly-better versus the 8-node continuation
(`5.2531` vs `5.2538`) rather than a strong continuation decrease, but this
task is explicitly a bounded scale-out smoke, and there is no negative systems
evidence that would justify stopping or retrying 16 nodes before the next
bounded scale-out check.
