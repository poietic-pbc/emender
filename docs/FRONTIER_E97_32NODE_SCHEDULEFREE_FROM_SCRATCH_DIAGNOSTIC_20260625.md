# Frontier E97 32-node schedule-free from-scratch diagnostic

Date: 2026-06-25
Task: `run-e97-32-node-4`

## Verdict

The bounded 32-node E97-MLP schedule-free `sfsgd_y` from-scratch diagnostic
completed cleanly and produced a usable saved-basis fixed-eval row.

This is schedule-free scale-probe evidence only. It is not a same-source
avg-ladder rung, does not supersede the avg K80/partial-average ladder, and
does not unblock `run-64-node-e97`.

## Gating and scope

I read `docs/FRONTIER_E97_SCHEDULEFREE_SCALE_SYNTHESIS_20260625.md` before
launch. That report authorized exactly one bounded 32-node E97-MLP
schedule-free from-scratch diagnostic with `sfsgd`, export basis `y`, and no
resume checkpoint.

Scope controls:

- Submitted training jobs from this task: exactly one.
- Training job id: `4899248`.
- Fixed eval jobs from this task: exactly one.
- Fixed eval job id: `4899282`.
- No 64-node, GDN2, CMAES, schedule-free sweep, export-basis `x`, fixed
  momentum, changed island size, or resumed non-avg job was submitted.
- `run-64-node-e97` was checked after the run and remains `open (PAUSED)`.
- Source checkpoint: none. The run started from scratch.

## Training job

| Field | Value |
| --- | --- |
| Job | `e97-32n-sfsgd-y-diag` |
| Job id | `4899248` |
| Account / partition / QOS | `bif148` / `batch` / `normal` |
| Nodes | `32` |
| Walltime requested | `00:30:00` |
| Requested node-hours | `16.000000` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:20:30` |
| Actual node-hours | `10.933333` |
| Runtime git commit | `7d9e954e94a6b78899b75bfed8f492adba825ddc` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899248-20260625T110924Z` |
| Run label | `emender_E97_1.3B_20260625_071201` |
| Training log | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899248-20260625T110924Z/logs/train.log` |
| Slurm stdout / stderr | `logs/frontier/scaleout/e97-32n-sfsgd-y-diag-4899248.out` / `logs/frontier/scaleout/e97-32n-sfsgd-y-diag-4899248.err` |

Relevant accounting:

```text
4899248|e97-32n-sfsgd-y-diag|batch|bif148|COMPLETED|0:0|00:20:30|00:30:00|32|2026-06-25T07:09:10|2026-06-25T07:09:22|2026-06-25T07:29:52
```

## Launch configuration

The recorded launch matched the synthesis config:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=32
TRAIN_MINUTES=20
walltime=00:30:00
RESUME_CHECKPOINT unset / empty
optimizer=schedulefree
diloco=true
diloco_k=250
diloco_island_size=8
diloco_outer_optimizer=sfsgd
diloco_outer_lr=1.0
diloco_outer_beta=0.1
diloco_export_basis=y
save_every=250
keep_checkpoints=4
batch_size=1
chunk_size=2048
compile_warmup_steps=1
```

The Slurm stdout recorded `resume_checkpoint=` and the `srun` command did not
include `--resume`. It launched `32` nodes, `256` ranks, and the training log
confirmed the intended topology:

```text
[DiLoCo-hybrid] 32 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=250
[DiLoCo] outer optimizer: sfsgd (outer_lr=1.0, outer_beta=0.1, export_basis=y)
```

## Loss and throughput

Loss stayed finite and productive over the bounded window:

| Step | Loss | Global tok/s | Note |
| ---: | ---: | ---: | --- |
| 5 | `10.5608` | `400345` | first logged training step |
| 250 | `5.3045` | merge/checkpoint | merge #1 |
| 500 | `5.2374` | merge/checkpoint | merge #2 |
| 750 | `4.9101` | merge/checkpoint | merge #3 |
| 905 | `4.7470` | `572787` | late-window sample |
| 950 | `4.6898` | `580443` | late-window sample |
| 995 | `4.9633` | `581836` | late-window sample |
| 1000 periodic | `4.8150` | `232183` | merge #4 and periodic checkpoint |
| final | `5.0053` | n/a | final consensus checkpoint and `FINAL_LOSS_LAST100` |

Late-window non-merge throughput was roughly `570k-590k` global tokens/s.

Final summary:

```text
Training complete! Final step: 1000
FINAL_LOSS_LAST100: 5.0053
```

## DiLoCo and checkpoint behavior

The run completed four DiLoCo merges:

```text
merge #1 at step 250: 6807 ms
merge #2 at step 500: 6761 ms
merge #3 at step 750: 6698 ms
merge #4 at step 1000: 6783 ms
final merge SKIPPED at step 1000: last step already merged (step % K == 0); checkpoint is already consensus
DILOCO_MERGES: 4
DILOCO_K: 250
DILOCO_SYNC_TOTAL_S: 27.050
DILOCO_SYNC_AVG_MS: 6762.5
```

Checkpoint retention was bounded at four checkpoint files:

```text
checkpoint_step_000500_loss_5.2374.pt
checkpoint_step_000750_loss_4.9101.pt
checkpoint_step_001000_loss_4.8150.pt
checkpoint_step_001000_loss_5.0053.pt
latest.pt -> checkpoint_step_001000_loss_5.0053.pt
```

There are two step-1000 checkpoint files because the periodic step-1000
checkpoint was followed by finalization at the same step. `latest.pt` points to
the later final consensus checkpoint.

Final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899248-20260625T110924Z/train/emender_E97_1.3B_20260625_071201/checkpoint_step_001000_loss_5.0053.pt
```

## Schedule-free checkpoint state

The final `latest.pt` checkpoint was inspected with the Frontier module stack:

```text
top_keys ['checkpoint_metadata', 'diloco_outer_state', 'loss', 'model_state_dict', 'optimizer_state_dict', 'step']
step 1000
loss 5.005345113754273
has_diloco_outer_state True
outer_state_keys ['k', 'lr_max', 'mode', 'weight_sum', 'x', 'y', 'z']
mode=sfsgd
k=4
weight_sum=4.0
lr_max=1.0
x_present=True type=list
y_present=True type=list
z_present=True type=list
```

The `k=4` / `weight_sum=4.0` values match the four actual DiLoCo outer merges.
The final merge was skipped because step `1000` had already been merged.

## Fixed eval

The saved-basis fixed eval completed as the single permitted one-node eval:

| Field | Value |
| --- | --- |
| Job | `e97-32n-sfsgd-y-fixed-eval` |
| Job id | `4899282` |
| Nodes | `1` |
| Walltime requested | `01:00:00` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:01:00` |
| Actual node-hours | `0.016667` |
| Eval mode | `--y-mode saved` |
| Scoring tensor | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt` |
| Output CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899248-20260625T110924Z/artifacts/e97_32n_sfsgd_y_fixed_eval.csv` |

Eval row:

```csv
step,tokens,ce,bpb,split,checkpoint
1000,2048000,4.85214365,2.01877561,primary,/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899248-20260625T110924Z/train/emender_E97_1.3B_20260625_071201/checkpoint_step_001000_loss_5.0053.pt
```

## Fatal-signature scan

The train and eval jobs exited `COMPLETED` with `0:0`.

I found no training traceback from user code, non-finite loss, OOM/out-of-memory,
RCCL/NCCL watchdog timeout, collective mismatch, checkpoint write failure,
finalization failure, or eval load failure.

The training stdout emitted post-completion TCPStore / `ProcessGroupNCCL`
heartbeat warnings while ranks were shutting down:

```text
Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: failed to recv, got 0 bytes
```

This happened after final checkpoint completion and with Slurm state
`COMPLETED` / exit `0:0`. I am recording it as a non-fatal shutdown warning,
not as a failed collective.

The eval stderr contained module reload notices and Triton/Python version
warnings, but the eval strict load and scoring completed successfully.

## Schedule-free comparison

All rows below are from-scratch `sfsgd_y` probes with saved-basis fixed eval on
the same tensor.

| Run | Train job | Eval job | Nodes / islands | Final step | Final train loss | Fixed CE | Fixed BPB | Merge state |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4-node `sfsgd_y` | `4899141` | `4899197` | 4 / 4 | 1204 | `4.7773` | `4.85631931` | `2.02051293` | 5 merges; final consensus |
| 8-node `sfsgd_y` | `4899142` | `4899198` | 8 / 8 | 1190 | `4.8115` | `4.85920626` | `2.02171407` | 5 merges; final consensus |
| 16-node `sfsgd_y` | `4899221` | `4899229` | 16 / 16 | 1119 | `4.8905` | `4.84385067` | `2.01532525` | 5 merges; final consensus |
| 32-node `sfsgd_y` | `4899248` | `4899282` | 32 / 32 | 1000 | `5.0053` | `4.85214365` | `2.01877561` | 4 merges; final merge skipped because step 1000 already merged |

The 32-node row is operationally clean, but fixed CE/BPB is between the 16-node
row and the 4/8-node rows rather than a monotonic improvement. The shorter
effective step count and the skipped duplicate final merge make it a valid
systems/scale probe, not a quality promotion signal.

## Relation to avg K80 / partial-average ladder

This run remains outside the same-source avg ladder. The avg K80 and
partial-average ladder asks whether a 32-node recipe can continue from the
common 16-node avg checkpoint without regressing fixed validation. This
schedule-free run asks whether the audited `sfsgd_y` schedule-free path scales
cleanly from scratch to 32 nodes.

Therefore:

- It supports operational health of the schedule-free `sfsgd_y` path through
  32 nodes.
- It does not replace the avg K80 result as the best same-source 32-node avg
  evidence.
- It does not repair the partial-average/non-avg resume-state blocker.
- It does not authorize or unblock `run-64-node-e97`, which remains
  `open (PAUSED)`.
