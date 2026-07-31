# Frontier E97 32-Node Avg Scaleout Smoke

Date: 2026-06-24
Task: `run-32-node-e97`

## Verdict

MIXED. The bounded 32-node E97-MLP DiLoCo regular-averaging smoke completed
the launch, checkpoint, retention, finalization, and Slurm-clean-exit checks.
It resumed from the clean 16-node final checkpoint, launched exactly one
32-node job, completed with Slurm exit `0:0`, wrote regular checkpoints every
10 steps, wrote a final checkpoint, advanced `latest.pt` to the final
checkpoint, and retained a bounded checkpoint set.

The loss behavior was not productive. The run's final trailing loss was
`5.8701`, compared with the 16-node source final trailing loss of `5.2531`.
The within-run metric window also drifted up: first-20 mean `5.4394`,
last-20 mean `6.1958`. This is acceptable as an operational smoke but not as a
quality gate to launch the 64-node task.

Recommendation: do not proceed directly to 64 nodes. Add a focused 32-node
loss-regression follow-up before any 64-node E97 avg smoke.

No 64-node job was submitted. No schedule-free outer run was submitted. No GDN2
run was submitted.

## Upstream 16-Node Gate

The prerequisite `run-16-node-e97` was checked before submission using
`docs/FRONTIER_E97_16NODE_AVG_SMOKE_20260624.md` and WG dependency logs.

The 16-node smoke passed:

- Job `4893977` completed with Slurm exit `0:0`.
- It resumed from the 8-node continuation and reached final step `1328`.
- It recorded `DILOCO_MERGES: 36`.
- It advanced `latest.pt` to the final step-1328 checkpoint.
- It retained a bounded set of four checkpoints.
- Its final trailing loss was effectively flat to the 8-node source:
  - 8-node continuation `FINAL_LOSS_LAST100: 5.2538`
  - 16-node smoke `FINAL_LOSS_LAST100: 5.2531`
  - delta `-0.0007`
- No NCCL/RCCL watchdog timeout or collective mismatch signature was reported.

This satisfied the gate to run one bounded 32-node smoke.

## Submitted Job

Exactly one bounded 32-node E97-MLP avg smoke was submitted:

| Job | Partition | QOS | Nodes | Walltime | Max requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4894111` | `batch` | `normal` | 32 | `00:30:00` | `16.000000` | `COMPLETED` | `0:0` | `00:20:22` | `10.862222` |

Actual node-hours are computed from the top-level allocation elapsed time:
`32 * (20m22s / 3600) = 10.862222`.

Scheduler accounting:

```text
4894111|e97-32n-avg-smoke|batch|normal|COMPLETED|0:0|00:20:22|00:30:00|32|billing=3584,cpu=3584,energy=16955234148,mem=16000G,node=32|2026-06-24T07:01:33|2026-06-24T07:01:42|2026-06-24T07:22:04
4894111.batch|batch|||COMPLETED|0:0|00:20:22||1|cpu=56,mem=500G,node=1|2026-06-24T07:01:42|2026-06-24T07:01:42|2026-06-24T07:22:04
4894111.extern|extern|||COMPLETED|0:0|00:20:22||32|billing=3584,cpu=3584,mem=16000G,node=32|2026-06-24T07:01:42|2026-06-24T07:01:42|2026-06-24T07:22:04
4894111.0|bash|||COMPLETED|0:0|00:19:59||32|cpu=1792,mem=16000G,node=32|2026-06-24T07:02:05|2026-06-24T07:02:05|2026-06-24T07:22:04
```

## Source State

Before submission, `HEAD` and `origin/main` were both:

```text
17440fddc52ea58c4328fb78171e830046b85d3e
```

The runtime manifest recorded the same git commit:

```text
17440fddc52ea58c4328fb78171e830046b85d3e
```

## Source Checkpoint

The 32-node smoke resumed from the clean 16-node final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

Before submission, that symlink resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/checkpoint_step_001328_loss_5.2531.pt
```

Rank 0 logged:

```text
Resumed at step 1328
Starting training from step 1328...
```

## Run Configuration

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z
```

Primary artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/summaries/summary.md
logs/frontier/scaleout/e97-32n-avg-smoke-4894111.out
logs/frontier/scaleout/e97-32n-avg-smoke-4894111.err
```

Run label:

```text
emender_E97_1.3B_20260624_070413
```

Key launch settings:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=32
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

Rank 0 logged the expected 256-rank, 32-island, avg-outer configuration:

```text
[DiLoCo] world_size=256 backend=nccl; this is rank 0 on cuda:0
[DiLoCo-hybrid] 32 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=10 (subgroup comms warmed sequentially)
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
```

## Loss And Throughput

The run progressed from source step `1328` to final step `1780`.

Representative rank-0 metric lines:

```text
step   1330 | loss 2.2782 | lr 9.00e-04 | grad 1.16 | tok/s 591 | global_tok/s 151169 | elapsed_h 0.002 | time 2026-06-24T11:04:56+00:00
step   1400 | loss 6.1078 | lr 9.00e-04 | grad 2.12 | tok/s 1120 | global_tok/s 286723 | elapsed_h 0.044 | time 2026-06-24T11:07:28+00:00
step   1500 | loss 5.6864 | lr 9.00e-04 | grad 0.95 | tok/s 1117 | global_tok/s 285949 | elapsed_h 0.106 | time 2026-06-24T11:11:10+00:00
step   1600 | loss 5.7320 | lr 9.00e-04 | grad 1.12 | tok/s 1110 | global_tok/s 284150 | elapsed_h 0.167 | time 2026-06-24T11:14:51+00:00
step   1700 | loss 6.0197 | lr 9.00e-04 | grad 1.53 | tok/s 1113 | global_tok/s 284871 | elapsed_h 0.232 | time 2026-06-24T11:18:46+00:00
step   1780 | loss 6.2220 | lr 9.00e-04 | grad 1.08 | tok/s 1128 | global_tok/s 288832 | elapsed_h 0.282 | time 2026-06-24T11:21:43+00:00
```

Loss-window summary from the 91 rank-0 metric samples:

```text
samples=91 first_step=1330 last_step=1780 mean_all=5.8701 first20_mean=5.4394 last20_mean=6.1958 first10_mean=5.1964 last10_mean=6.2543
global_tok_median=220157 global_tok_mean=241997
```

Final trailing metric comparison:

```text
16-node smoke FINAL_LOSS_LAST100: 5.2531
32-node smoke FINAL_LOSS_LAST100: 5.8701
Delta:                           +0.6170
```

The short within-run window is noisy, but the final trailing loss and
last-window mean are materially worse than the source. This is a quality
regression signal, not a launch failure.

Summary metrics:

```text
Training complete! Final step: 1780
FINAL_LOSS_LAST100: 5.8701
DILOCO_MERGES: 46
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 210.828
DILOCO_SYNC_AVG_MS: 4583.2
PEAK_MEMORY_MB: 15595
RESERVED_MEMORY_MB: 19002
```

## Checkpoint And Retention Verification

Regular checkpoints were written every 10 steps after resume through step
`1780`. Finalization triggered due to the Slurm walltime margin. Since step
`1780` was already a DiLoCo merge step, the final merge was skipped correctly:

```text
>>> [DiLoCo] final merge SKIPPED at step 1780: last step already merged (step % K == 0); checkpoint is already consensus (avoids spurious outer-momentum double-step)
```

Rank 0 then wrote the final checkpoint:

```text
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=1780 loss=5.8701 remaining_s=589.3 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/256 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/train/emender_E97_1.3B_20260624_070413/checkpoint_step_001780_loss_5.8701.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/train/emender_E97_1.3B_20260624_070413/latest.pt model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/256 is_head=True
```

Final retained run-directory contents:

```text
checkpoint_step_001760_loss_5.9794.pt
checkpoint_step_001770_loss_6.1837.pt
checkpoint_step_001780_loss_5.8701.pt
checkpoint_step_001780_loss_6.2220.pt
latest.pt -> checkpoint_step_001780_loss_5.8701.pt
```

The extra step-1780 regular checkpoint is expected because step `1780` landed
on the regular `SAVE_EVERY=10` cadence before final checkpoint finalization.
`latest.pt` points to the final checkpoint with the `FINAL_LOSS_LAST100` value.

`readlink -f latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4894111-20260624T110146Z/train/emender_E97_1.3B_20260624_070413/checkpoint_step_001780_loss_5.8701.pt
```

No checkpoint temporary, partial, or incomplete files remained after
completion.

## Communication And Warning Scan

Final scans over the train log, Slurm stdout, and Slurm stderr found no
watchdog, collective mismatch, timeout, traceback, runtime error, exception,
non-finite, NaN, OOM, out-of-memory, failed, or cancelled signature. Slurm
accounting reports `COMPLETED` with exit `0:0`.

Non-blocking warnings matched prior clean runs or existing runtime warnings:

- PyTorch/NCCL emitted the rank-to-GPU startup warning. The job still completed
  and performed 46 successful DiLoCo merges across 256 ranks.
- c10d emitted address-family socket warnings during startup. The process group
  formed and completed training.
- Triton 3.2.0 printed a recommended-version warning. This was not fatal.
- Slurm stderr contained module replacement notices and the AMD low-power-state
  warning.

## Downstream Recommendation

Hold the 64-node task. The 32-node operational path is clean enough to trust
the launch/checkpoint/finalization mechanics, but the loss behavior is not
productive enough to treat this as a green scale-out gate.

Recommended next action: run a focused follow-up before 64 nodes that explains
or mitigates the 32-node loss regression. Reasonable options include a bounded
32-node retry from the same source checkpoint to separate stochastic/noisy loss
from systematic scale-out degradation, or a smaller diagnostic comparing
32-node DiLoCo settings while staying within the avg-outer-only scope.
