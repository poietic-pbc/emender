# Frontier E97 8-node schedule-free outer smoke

Date: 2026-06-25
Task: `run-e97-schedule-2`

## Verdict

The 8-node E97-MLP `sfsgd_y` schedule-free outer smoke was operationally clean
enough for the planned 4/8 synthesis step and promising as a cheap scale signal.
It completed within the bounded window, produced finite loss through five DiLoCo
merges, finalized a consensus checkpoint, retained `diloco_outer_state`, advanced
`latest.pt`, and produced the required saved-basis fixed eval row.

Do not treat the fixed-eval delta against the 16-node source as a same-trajectory
quality comparison: the audit required this first schedule-free probe to start
from scratch because the 16-node avg checkpoint cannot initialize coherent
`sfsgd` outer state. The useful signal here is launch health, finite training,
checkpoint integrity, and fixed-eval viability.

## Audit and scope confirmation

- Audit dependency: `audit-e97-schedule` completed before this run.
- Audit artifact used: `docs/FRONTIER_E97_SCHEDULEFREE_OUTER_AUDIT_20260625.md`.
- Training submission: exactly one 8-node training job was submitted.
- Training job id: `4899142`.
- Fixed eval job id: `4899198`.
- No 16-node, 32-node, 64-node, GDN2, CMAES, beta sweep, LR sweep, K sweep, fixed
  momentum, or `sfsgd` export-`x` job was submitted from this task.
- `run-64-node-e97` was checked and left paused.

## Training job

```text
Job: e97-8n-sfsgd-y-smoke
Job id: 4899142
Partition/QOS/account: batch / normal / bif148
Nodes: 8
Walltime requested: 00:30:00
Requested node-hours: 4.000000
Actual elapsed: 00:20:27
Actual node-hours: 2.726667
Slurm state: COMPLETED
Exit code: 0:0
Start/end: 2026-06-25T05:13:10 to 2026-06-25T05:33:37 America/New_York
```

The submitted config matched the audit recipe:

```text
optimizer=schedulefree
diloco=true
diloco_k=250
diloco_outer_optimizer=sfsgd
diloco_export_basis=y
diloco_outer_lr=1.0
diloco_outer_beta=0.1
batch_size=1
chunk_size=2048
save_every=250
keep_checkpoints=4
compile_warmup_steps=1
resume_checkpoint=<unset>
```

Run paths:

```text
Run root:
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899142-20260625T091312Z

Run label:
emender_E97_1.3B_20260625_051423

Training log:
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899142-20260625T091312Z/logs/train.log

Summary:
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899142-20260625T091312Z/summaries/summary.md

Slurm stdout/stderr:
logs/frontier/scaleout/e97-8n-sfsgd-y-smoke-4899142.out
logs/frontier/scaleout/e97-8n-sfsgd-y-smoke-4899142.err
```

The run manifest records repository commit
`098c135f67e2463df5da20822d2a93a9ebb5529c`.

## Loss and throughput

The run reached final step `1190`, with `FINAL_LOSS_LAST100: 4.8115`. The loss
stream stayed finite. Representative samples:

| Step | Loss | Global tok/s | Note |
| ---: | ---: | ---: | --- |
| 500 | 5.3428 | n/a | DiLoCo merge #2 checkpoint |
| 750 | 4.8561 | n/a | DiLoCo merge #3 checkpoint |
| 1000 | 5.0750 | n/a | DiLoCo merge #4 checkpoint |
| 1095 | 4.8625 | 152536 | late-window sample |
| 1125 | 4.5373 | 154956 | late-window sample |
| 1155 | 4.3016 | 152346 | late-window sample |
| 1190 | 4.5732 | 152181 | final logged step before final checkpoint |
| final | 4.8115 | n/a | final-checkpoint loss / `FINAL_LOSS_LAST100` |

Steady late-window throughput was roughly `150k-157k` global tokens/s outside
merge/checkpoint pauses.

## DiLoCo and checkpoint behavior

The run used 64 ranks as `8 islands x 8 GPUs`, with per-step DDP within each
island and DiLoCo averaging across islands every `K=250`.

DiLoCo merge lines:

```text
merge #1 at step 250
merge #2 at step 500
merge #3 at step 750
merge #4 at step 1000
FINAL merge #5 at step 1190
DILOCO_MERGES: 5
DILOCO_K: 250
DILOCO_SYNC_TOTAL_S: 32.523
DILOCO_SYNC_AVG_MS: 6504.5
```

Retention and finalization were correct for `KEEP_CHECKPOINTS=4`:

```text
checkpoint_step_000500_loss_5.3428.pt
checkpoint_step_000750_loss_4.8561.pt
checkpoint_step_001000_loss_5.0750.pt
checkpoint_step_001190_loss_4.8115.pt
latest.pt -> checkpoint_step_001190_loss_4.8115.pt
```

The final checkpoint contains:

```text
step: 1190
loss: 4.811515504837035
diloco_outer_state.present: true
diloco_outer_state.mode: sfsgd
diloco_outer_state.k: 5
diloco_outer_state.weight_sum: 5.0
diloco_outer_state.lr_max: 1.0
diloco_outer_state.x/y/z: present
```

Finalization performed the expected consensus merge before saving:

```text
[DiLoCo] FINAL merge #5 at step 1190: consensus model averaged across 64 ranks
[final-checkpoint] END path=.../checkpoint_step_001190_loss_4.8115.pt latest=.../latest.pt
Training complete! Final step: 1190
FINAL_LOSS_LAST100: 4.8115
```

The Slurm job exited cleanly. The stdout includes post-exit PyTorch TCPStore
heartbeat warnings such as `Failed to check the "should dump" flag on TCPStore`
after final checkpoint completion. These occurred during teardown after the
successful final save and did not produce a nonzero Slurm exit. I found no NCCL
watchdog timeout, no collective mismatch, and no training traceback before the
final checkpoint.

## Fixed eval

The source/eval basis came from the audit:

```text
Source checkpoint comparator:
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt

Scoring tensor:
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt

Eval mode:
--y-mode saved
```

The source checkpoint already had a saved-basis row on this same tensor in
`docs/FRONTIER_E97_32NODE_FIXED_EVAL_20260624.md`, so I did not spend another
one-node eval allocation re-scoring it. The 8-node schedule-free checkpoint was
evaluated by job `4899198`:

```text
Job: e97-8n-sfsgd-y-fixed-eval
Job id: 4899198
Nodes: 1
Walltime requested: 01:00:00
Actual elapsed: 00:00:51
Actual node-hours: 0.014167
Slurm state: COMPLETED
Exit code: 0:0
```

Eval output:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899142-20260625T091312Z/artifacts/e97_8n_sfsgd_y_fixed_eval.csv
```

Fixed-eval comparison:

| Checkpoint | Step | Train checkpoint loss | Fixed CE | Fixed BPB | Basis |
| --- | ---: | ---: | ---: | ---: | --- |
| 16-node avg source | 1328 | 5.2531 | 10.49609756 | 4.36699062 | saved |
| 8-node `sfsgd_y` from-scratch smoke | 1190 | 4.8115 | 4.85920626 | 2.02171407 | saved |

The 8-node CSV row is:

```csv
step,tokens,ce,bpb,split,checkpoint
1190,2437120,4.85920626,2.02171407,primary,/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899142-20260625T091312Z/train/emender_E97_1.3B_20260625_051423/checkpoint_step_001190_loss_4.8115.pt
```

## Decision

The 8-node schedule-free run is operationally clean enough to synthesize with
the 4-node probe:

- launch path and audited config worked without resume or hidden variable drift;
- 64-rank / 8-island DiLoCo topology initialized and completed;
- schedule-free outer state survived into the final checkpoint;
- merge cadence and checkpoint retention behaved as expected;
- saved-basis fixed eval ran successfully.

The only caveat is the benign-looking TCPStore heartbeat warning after final
checkpoint completion. It should be mentioned in synthesis, but it did not block
finalization or evaluation. Based on this 8-node result alone, schedule-free is
promising as an operational canary, but acceleration to 16 nodes should wait for
the paired 4-node comparison and synthesis rather than relying on the 16-node
avg source delta.
