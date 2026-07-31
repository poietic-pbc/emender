# Frontier E97 16-node schedule-free outer smoke

Date: 2026-06-25
Task: `run-e97-schedule-3`

## Verdict

The bounded 16-node E97-MLP `sfsgd_y` schedule-free outer smoke completed
cleanly and is strong enough to justify creating a separate bounded 32-node
schedule-free diagnostic task.

This result does not authorize a 32-node or 64-node launch from
`run-e97-schedule-3`. It only says the 16-node schedule-free path is
operationally ready for a separately-scoped next-scale task after synthesis.

## Gating decision

I read `docs/FRONTIER_E97_SCHEDULEFREE_4N_8N_SYNTHESIS_20260625.md` before
launching. That synthesis recommended exactly one bounded 16-node E97-MLP
schedule-free `sfsgd_y` smoke and explicitly did not authorize 32-node or
64-node schedule-free work.

The 16-node launch therefore proceeded. No blocker from the 4-node or 8-node
evidence was present.

## Scope and controls

- Submitted training jobs from this task: exactly one.
- Training job id: `4899221`.
- Fixed eval job id: `4899229`.
- No 32-node or 64-node schedule-free job was submitted.
- No GDN2, CMAES, beta sweep, LR sweep, K sweep, fixed momentum, or export-basis
  `x` job was submitted.
- `run-64-node-e97` was checked before and after this task and remains
  `open (PAUSED)`.
- Source checkpoint: none. The run started from scratch, matching the audited
  schedule-free plan because the avg checkpoint cannot initialize coherent
  `sfsgd` outer state.

## Training job

| Field | Value |
| --- | --- |
| Job | `e97-16n-sfsgd-y-smoke` |
| Job id | `4899221` |
| Account / partition / QOS | `bif148` / `batch` / `normal` |
| Nodes | `16` |
| Walltime requested | `00:30:00` |
| Requested node-hours | `8.000000` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:20:29` |
| Actual node-hours | `5.464444` |
| Runtime git commit | `2ab8330adfeb72710cd1bf37cc9aad2e277d5f65` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899221-20260625T100521Z` |
| Run label | `emender_E97_1.3B_20260625_060650` |
| Training log | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899221-20260625T100521Z/logs/train.log` |
| Slurm stdout / stderr | `logs/frontier/scaleout/e97-16n-sfsgd-y-smoke-4899221.out` / `logs/frontier/scaleout/e97-16n-sfsgd-y-smoke-4899221.err` |

Relevant accounting:

```text
4899221|e97-16n-sfsgd-y-smoke|batch|bif148|COMPLETED|0:0|00:20:29|00:30:00|16|2026-06-25T06:04:43|2026-06-25T06:05:18|2026-06-25T06:25:47
```

## Launch configuration

The recorded args matched the audited schedule-free family:

```text
optimizer=schedulefree
diloco=true
diloco_k=250
diloco_outer_optimizer=sfsgd
diloco_outer_lr=1.0
diloco_outer_beta=0.1
diloco_export_basis=y
diloco_island_size=8
batch_size=1
chunk_size=2048
save_every=250
keep_checkpoints=4
compile_warmup_steps=1
train_minutes=20
resume=null
```

The training log confirmed 128 ranks as `16 islands x 8 GPUs` and the intended
outer optimizer:

```text
[DiLoCo-hybrid] 16 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=250
[DiLoCo] outer optimizer: sfsgd (outer_lr=1.0, outer_beta=0.1, export_basis=y)
```

## Loss and throughput

Loss stayed finite and productive over the short bounded window:

| Step | Loss | Global tok/s | Note |
| ---: | ---: | ---: | --- |
| 5 | `10.4739` | `258230` | first logged training step |
| 250 | checkpoint loss retained only through later pruning | n/a | merge #1 |
| 500 | `5.6342` | n/a | retained checkpoint |
| 750 | `4.9755` | `118249` | merge/checkpoint pause |
| 1000 | `4.9031` | `118717` | merge/checkpoint pause |
| 1075 | `4.0296` | `301178` | late-window sample |
| 1115 | `4.4199` | `297452` | final pre-finalization sample |
| final | `4.8905` | n/a | `FINAL_LOSS_LAST100` |

The final summary was:

```text
Training complete! Final step: 1119
FINAL_LOSS_LAST100: 4.8905
```

Late-window non-merge throughput was roughly `294k-302k` global tokens/s.

## DiLoCo and checkpoint behavior

The run completed four periodic merges and one final consensus merge:

```text
merge #1 at step 250
merge #2 at step 500
merge #3 at step 750
merge #4 at step 1000
FINAL merge #5 at step 1119
DILOCO_MERGES: 5
DILOCO_K: 250
DILOCO_SYNC_TOTAL_S: 33.193
DILOCO_SYNC_AVG_MS: 6638.7
```

Finalization performed the expected consensus merge before writing the final
checkpoint:

```text
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=1119 loss=4.8905
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899221-20260625T100521Z/train/emender_E97_1.3B_20260625_060650/checkpoint_step_001119_loss_4.8905.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899221-20260625T100521Z/train/emender_E97_1.3B_20260625_060650/latest.pt
```

Retention was bounded at four checkpoint files:

```text
checkpoint_step_000500_loss_5.6342.pt
checkpoint_step_000750_loss_4.9755.pt
checkpoint_step_001000_loss_4.9031.pt
checkpoint_step_001119_loss_4.8905.pt
latest.pt -> checkpoint_step_001119_loss_4.8905.pt
```

## Schedule-free checkpoint state

The final `latest.pt` checkpoint was inspected with the Frontier module stack:

```text
top_keys ['checkpoint_metadata', 'diloco_outer_state', 'loss', 'model_state_dict', 'optimizer_state_dict', 'step']
step 1119
loss 4.890534532070159
has_diloco_outer_state True
outer_state_keys ['k', 'lr_max', 'mode', 'weight_sum', 'x', 'y', 'z']
mode=sfsgd
k=5
weight_sum=5.0
lr_max=1.0
x_type=list
y_type=list
z_type=list
```

## Fixed eval

The saved-basis fixed eval was practical and completed:

| Field | Value |
| --- | --- |
| Job | `e97-16n-sfsgd-y-fixed-eval` |
| Job id | `4899229` |
| Nodes | `1` |
| Walltime requested | `01:00:00` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:00:49` |
| Actual node-hours | `0.013611` |
| Eval mode | `--y-mode saved` |
| Scoring tensor | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt` |
| Output CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899221-20260625T100521Z/artifacts/e97_16n_sfsgd_y_fixed_eval.csv` |

Eval row:

```csv
step,tokens,ce,bpb,split,checkpoint
1119,2291712,4.84385067,2.01532525,primary,/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899221-20260625T100521Z/train/emender_E97_1.3B_20260625_060650/checkpoint_step_001119_loss_4.8905.pt
```

Relative to the 4-node and 8-node schedule-free fixed eval rows from
`synthesize-e97-schedule`, the 16-node CE/BPB is slightly lower:

| Run | Step | Fixed CE | Fixed BPB |
| --- | ---: | ---: | ---: |
| 4-node `sfsgd_y` | 1204 | `4.85631931` | `2.02051293` |
| 8-node `sfsgd_y` | 1190 | `4.85920626` | `2.02171407` |
| 16-node `sfsgd_y` | 1119 | `4.84385067` | `2.01532525` |

This remains an operational/saved-basis health comparison, not a same-trajectory
win over the older avg-outer checkpoint.

## Warnings and non-issues

The run emitted the same expected PyTorch NCCL device-id inference warnings seen
in earlier probes. The training and eval jobs still exited `COMPLETED` with
exit code `0:0`.

I found no training traceback, non-finite loss, OOM/out-of-memory, NCCL/RCCL
watchdog timeout, collective mismatch, checkpoint write failure, finalization
failure, or eval load failure in the inspected logs.

Unlike the 8-node run, I did not find post-final TCPStore heartbeat warnings in
the final inspected 16-node output.

## Decision

The 16-node schedule-free smoke is ready to justify a separate bounded 32-node
schedule-free task:

- 128-rank launch completed cleanly;
- loss remained finite and productive;
- merge cadence and final consensus worked;
- checkpoint retention and `latest.pt` behavior were correct;
- `diloco_outer_state` was retained with the expected `sfsgd` fields;
- saved-basis fixed eval completed and improved slightly versus the 4-node and
  8-node schedule-free rows.

Do not infer that 32-node or 64-node schedule-free work was authorized or
submitted here. This task stopped after the one permitted 16-node training job
and its one-node fixed eval.
