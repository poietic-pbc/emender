# E97 1.3B Pretrained K160 32-Node Retry - 2026-06-26

WG task: `run-e97-1p3b-k160-32n`

## Result

Status: **failed on systems grounds**.

The single allowed retry job, Slurm `4904173`, started correctly on 32 Frontier nodes / 256 ranks from the requested job-4903889 K-aligned checkpoint, but failed before the first post-resume K160 DiLoCo merge and before any final checkpoint. The failure was a NCCL process-group watchdog timeout on a 1-element scalar `ALLREDUCE`:

```text
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=313, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
```

This is an actual systems failure under the relaxed gate. No 64-node successor checkpoint is valid from this retry.

## Submission

Submitted command:

```bash
sbatch -N 32 -J e97-1p3b-k160-32n-r2 --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-32n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_32n_r2,RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt,TRAIN_MINUTES=35,WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200,WALLTIME_CHECK_EVERY=40,REQUESTED_NODE_HOURS=32.0 scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

Only one allocated retry job was submitted for this task.

## Code And Source Checkpoint

- Worktree HEAD: `0e77129f022d6f574f4a9e02e36877b3ef22027d`.
- `0e77129` is an ancestor of HEAD.
- `train.py` contains the `--walltime_check_every` gated walltime/final-checkpoint consensus check.
- `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch` defaults:
  - `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200`
  - `WALLTIME_CHECK_EVERY=40`

Source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt
```

It resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/checkpoint_step_269920_loss_2.9823.pt
```

This is the requested job-4903889 K-aligned periodic checkpoint at step `269920`.

## Runtime Topology

Confirmed from the Slurm submit line, launcher, `args.json`, and train log:

- Nodes: `32`
- Ranks/tasks: `256`
- `DILOCO_ISLAND_SIZE=1`
- No DDP islands; singleton GPU islands with no per-step gradient all-reduce.
- `--diloco_k 160`
- `--diloco_outer_optimizer avg`
- `--diloco_outer_lr 1.0`
- `--diloco_outer_beta 0.0`
- `--diloco_export_basis x`
- `TRAIN_MINUTES=35`
- `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200`
- `WALLTIME_CHECK_EVERY=40`
- No GDN2, CMAES, schedule-free outer optimizer, K sweep, LR sweep, or DDP/island-size change was submitted.

The model optimizer was schedule-free AdamW inside the validated launcher path; the DiLoCo outer optimizer was `avg`, as required.

## Artifacts

- Slurm job id: `4904173`
- Slurm stdout: `logs/frontier/scaleout/e97-1p3b-k160-32n-r2-4904173.out`
- Slurm stderr: `logs/frontier/scaleout/e97-1p3b-k160-32n-r2-4904173.err`
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r2/4904173-20260626T084525Z`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r2/4904173-20260626T084525Z/logs/train.log`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r2/4904173-20260626T084525Z/train/emender_E97_1.3B_20260626_044654/run_manifest.json`
- Args: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r2/4904173-20260626T084525Z/train/emender_E97_1.3B_20260626_044654/args.json`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r2/4904173-20260626T084525Z/summaries/summary.md`
- Final `latest.pt`: **none**. The run directory contains only `args.json`, `run_manifest.json`, and `.final_checkpoint_ready/`; no `.pt` checkpoint was written.

## Slurm Accounting

`sacct` terminal state:

```text
JobID|JobName|State|ExitCode|Elapsed|AllocNodes|NTasks
4904173|e97-1p3b-k160-32n-r2|FAILED|137:0|00:14:18|32|
4904173.batch|batch|FAILED|137:0|00:14:18|1|1
4904173.extern|extern|COMPLETED|0:0|00:14:18|32|32
4904173.0|bash|CANCELLED|0:9|00:14:11|32|256
```

- Requested node-hours: `32.0`
- Actual elapsed: `00:14:18`
- Actual node-hours: `32 * 858 / 3600 = 7.6267`

## Training Metrics Before Failure

Rank-0 training metric lines were available from step `269925` through `270000` only, 16 logged points covering 80 local steps after resume. The run failed before there were 500, 1000, or 2000 logged local steps available.

Available trailing rank-0 loss averages:

- Last 500 local steps where available: 16 logged points, average loss `2.572225`
- Last 1000 local steps where available: 16 logged points, average loss `2.572225`
- Last 2000 local steps where available: 16 logged points, average loss `2.572225`

Last rank-0 metric:

```text
step 270000 | loss 2.0811 | lr 1.01e-03 | grad 1.56 | tok/s 2720 | global_tok/s 696278 | elapsed_h 0.017 | time 2026-06-26T08:48:08+00:00
```

Average global throughput over the 16 available rank-0 metric lines: `685683.5 tok/s`.

Approximate ensemble tokens processed after resume through last logged rank-0 step:

```text
(270000 - 269920) * 256 ranks * 2048 tokens = 41,943,040 tokens
```

The failure occurred around `2026-06-26T04:59:08-04:00`, roughly 10 minutes after the last rank-0 metric line. Because no later successful training step was logged, the conservative reported processed-token count is `41,943,040`.

## DiLoCo And Sync

- `DILOCO_MERGES`: `0` after resume.
- The retry started at step `269920`; the next K160 merge would have been at step `270080`.
- The last logged rank-0 training step was `270000`, so the job failed before reaching the first post-resume DiLoCo merge.
- No successful DiLoCo merge/sync duration was logged in this retry.
- Sync total/avg/fraction: not computable from successful sync events; the only observed synchronization outcome was the failed 1-element `ALLREDUCE` watchdog timeout after about 600 seconds.

## Loss Classification

Loss behavior: **indeterminate**.

Reason: all logged rank-0 losses were finite and did not show a large sustained blowup, but the run ended after only 16 rank-0 metric points and before the first post-resume K160 merge. The stop condition is systems failure, not quality.

## Finalization

Clean finalization did not occur:

- No final consensus merge.
- No explicit already-consensus finalization skip.
- No final checkpoint.
- No `latest.pt` pointing at a final checkpoint.
- No valid 64-node successor checkpoint.

The run did create `.final_checkpoint_ready/` during startup, but it failed long before final checkpoint entry and wrote no checkpoint file.

## Validation Checklist

- [x] Main/HEAD is at commit `0e77129` or later and contains the walltime scalar-collective fix.
- [x] Submitted no more than one allocated 32-node retry job for this task.
- [x] Confirmed source checkpoint resolves to job-4903889 K-aligned periodic `latest.pt` at step `269920`.
- [x] Confirmed topology: 32 nodes, 256 ranks, `DILOCO_ISLAND_SIZE=1`, no DDP, avg outer, K160, export basis `x`.
- [x] Confirmed runtime settings: `TRAIN_MINUTES=35`, `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200`, `WALLTIME_CHECK_EVERY=40`.
- [x] Checked finalization; it failed before final checkpoint, so this criterion is not satisfied by the run.
- [x] Reported Slurm job id, elapsed time, requested and actual node-hours, run root, train log, manifest, summary, and final `latest.pt` status.
- [x] Reported trailing rank-0 training loss averages where available.
- [x] Classified loss behavior as indeterminate, not blowup.
- [x] Reported throughput, approximate ensemble tokens processed, `DILOCO_MERGES`, sync outcome, and sync fraction availability.
- [x] Confirmed no GDN2, CMAES, schedule-free outer, K sweep, LR sweep, or DDP/island-size change was submitted.
