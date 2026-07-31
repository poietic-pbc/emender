# Frontier E97 4-node schedule-free smoke

Date: 2026-06-25
Task: `run-e97-schedule`

## Verdict

The bounded 4-node E97-MLP schedule-free outer DiLoCo smoke is operationally
clean and promising as a 4-node canary.

The run used the audited `sfsgd_y` configuration from
`docs/FRONTIER_E97_SCHEDULEFREE_OUTER_AUDIT_20260625.md`, did not resume from
the 16-node avg checkpoint, completed with Slurm exit `0:0`, performed periodic
and final DiLoCo consensus merges, wrote retained checkpoints, advanced
`latest.pt` to the final checkpoint, retained `diloco_outer_state`, and passed
the saved-basis fixed eval on the audited tensor.

This is not same-trajectory evidence against the 16-node avg source because the
audit required the first schedule-free probes to run from scratch. It is valid
as an operational and stability probe for the schedule-free outer path.

## Scope and controls

- Audit dependency: `audit-e97-schedule` completed before launch.
- Config source: exact 4-node launch recipe from
  `docs/FRONTIER_E97_SCHEDULEFREE_OUTER_AUDIT_20260625.md`.
- Submitted training jobs from this task: exactly one.
- No `RESUME_CHECKPOINT` was set.
- No 16-node, 32-node, or 64-node job was submitted from this task.
- No GDN2, CMAES, export-basis `x`, fixed momentum, beta sweep, LR sweep, or K
  sweep was submitted from this task.
- `run-64-node-e97` was checked and left paused.

## Training job

| Field | Value |
| --- | --- |
| Job id | `4899141` |
| Job name | `e97-4n-sfsgd-y-smoke` |
| Account / partition / QOS | `bif148` / `batch` / `normal` |
| Nodes | `4` |
| Walltime requested | `00:30:00` |
| Requested node-hours | `2.000000` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:20:26` |
| Actual node-hours | `1.362222` |
| Runtime git commit | `098c135f67e2463df5da20822d2a93a9ebb5529c` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899141-20260625T091236Z` |
| Run label | `emender_E97_1.3B_20260625_051345` |
| Source checkpoint | none; from-scratch per audit |
| Fixed-eval comparator source | 16-node avg source checkpoint from audit: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt` |

Relevant Slurm accounting:

```text
4899141|e97-4n-sfsgd-y-smoke|batch|bif148|COMPLETED|0:0|00:20:26|00:30:00|4|2026-06-25T05:12:21|2026-06-25T05:12:34|2026-06-25T05:33:00
```

## Launch configuration

The effective training command included:

```text
--optimizer schedulefree
--diloco
--diloco_k 250
--diloco_outer_optimizer sfsgd
--diloco_outer_lr 1.0
--diloco_outer_beta 0.1
--diloco_export_basis y
--diloco_island_size 8
--save_every 250
--keep_checkpoints 4
--train_minutes 20
```

The log confirmed:

```text
[DiLoCo-hybrid] 4 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=250
[DiLoCo] outer optimizer: sfsgd (outer_lr=1.0, outer_beta=0.1, export_basis=y)
```

## Loss and throughput signal

Loss was finite throughout and improved materially over the short run:

| Step | Loss | Global tok/s | Note |
| --- | ---: | ---: | --- |
| warmup | about `11.0` to `11.2` | n/a | one compile warmup step per rank |
| `5` | `10.4834` | `73081` | first logged train step |
| `250` | `5.5634` | `30281` | merge/checkpoint window |
| `500` | `5.2217` | `29928` | merge/checkpoint window |
| `750` | `4.5560` | `30385` | merge/checkpoint window |
| `1000` | `5.4247` | `30465` | merge/checkpoint window |
| `1200` | `4.2488` | `77395` | final pre-stop logged sample |
| final | `4.7773` | n/a | `FINAL_LOSS_LAST100` |

The final logged summary was:

```text
Training complete! Final step: 1204
FINAL_LOSS_LAST100: 4.7773
```

## DiLoCo and checkpoint behavior

The run completed four periodic merges and one final consensus merge:

```text
>>> [DiLoCo] merge #1 at step 250
>>> [DiLoCo] merge #2 at step 500
>>> [DiLoCo] merge #3 at step 750
>>> [DiLoCo] merge #4 at step 1000
>>> [DiLoCo] FINAL merge #5 at step 1204
DILOCO_MERGES: 5
DILOCO_K: 250
DILOCO_SYNC_TOTAL_S: 32.998
DILOCO_SYNC_AVG_MS: 6599.5
```

Final checkpoint behavior was clean:

```text
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=1204 loss=4.7773
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899141-20260625T091236Z/train/emender_E97_1.3B_20260625_051345/checkpoint_step_001204_loss_4.7773.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899141-20260625T091236Z/train/emender_E97_1.3B_20260625_051345/latest.pt
```

Retained checkpoint files at completion:

```text
checkpoint_step_000500_loss_5.2217.pt
checkpoint_step_000750_loss_4.5560.pt
checkpoint_step_001000_loss_5.4247.pt
checkpoint_step_001204_loss_4.7773.pt
latest.pt -> checkpoint_step_001204_loss_4.7773.pt
```

No final `.tmp`, `.partial`, or incomplete checkpoint files were found in the
run directory.

## Schedule-free checkpoint state

`latest.pt` was inspected with the Frontier module stack and contained:

```text
top_keys ['checkpoint_metadata', 'diloco_outer_state', 'loss', 'model_state_dict', 'optimizer_state_dict', 'step']
step 1204
loss 4.777303514003753
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

The task requested `mode=sfsgd`, `k`, `weight_sum`, and `lr_max`; all were
present.

## Fixed eval

The audited saved-basis eval was practical and completed:

| Field | Value |
| --- | --- |
| Job id | `4899197` |
| Job name | `e97-4n-sfsgd-y-fixed-eval` |
| Nodes | `1` |
| Slurm state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:00:50` |
| Eval mode | `--y-mode saved` |
| Scoring tensor | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt` |
| Output CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899141-20260625T091236Z/artifacts/e97_4n_sfsgd_y_fixed_eval.csv` |

Eval output:

```text
step,tokens,ce,bpb,split,checkpoint
1204,2465792,4.85631931,2.02051293,primary,/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899141-20260625T091236Z/train/emender_E97_1.3B_20260625_051345/checkpoint_step_001204_loss_4.7773.pt
```

Documented 16-node source fixed-eval row from
`docs/FRONTIER_E97_32NODE_FIXED_EVAL_20260624.md`:

| Checkpoint | Step | Fixed CE | Fixed BPB |
| --- | ---: | ---: | ---: |
| 16-node avg source | `1328` | `10.49609756` | `4.36699062` |
| 4-node `sfsgd_y` from scratch | `1204` | `4.85631931` | `2.02051293` |

The 4-node row is much lower on this fixed slice, but because it is a
from-scratch schedule-free probe, not a continuation from the 16-node avg
source, the safe interpretation is "fixed eval is comparable and clean" rather
than a direct trajectory win.

## Warnings and non-issues

The run emitted expected environment warnings about module replacement, Triton
version, Python version, and PyTorch NCCL device-id inference. It did not show a
traceback, runtime error, non-finite loss, NaN, OOM/out-of-memory, RCCL/NCCL
watchdog timeout, collective mismatch, checkpoint write failure, or eval load
failure.

One fixed-eval `sbatch` attempt omitted `-A bif148` and was rejected before job
creation. The successful eval job was `4899197`.

## Decision

The 4-node schedule-free outer path is operationally clean and promising:

- the audited `sfsgd_y` path launches at 4 nodes;
- loss is finite and productive over a short bounded window;
- periodic and final DiLoCo merges work;
- finalization writes a valid `latest.pt`;
- retention remains bounded at four checkpoints;
- schedule-free outer state is retained in the checkpoint;
- saved-basis fixed eval can load and score the checkpoint.

Do not launch a 16-node schedule-free job from this result alone. Wait for
`run-e97-schedule-2` and `synthesize-e97-schedule` to compare 4-node and 8-node
evidence and explicitly authorize any next scale.
