# E97 1.3B Pretrained K160 32-Node Retry R3 - 2026-06-26

WG task: `run-e97-1p3b-k160-32n`

## Result

Status: **failed on systems grounds**.

The single allowed r3 retry job, Slurm `4904355`, started correctly on 32 Frontier
nodes / 256 ranks from the requested job-4903889 K-aligned checkpoint. It
resumed at step `269920`, logged finite rank-0 training through step `270000`,
then hung until NCCL watchdog termination on a 1-element scalar `ALLREDUCE`:

```text
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=153, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
```

The run failed before the first post-resume K160 DiLoCo merge, before any final
consensus/finalization step, and before any checkpoint was written. This is an
actual systems failure under the relaxed gate. No 64-node successor checkpoint is
valid from this retry.

## Submission

Submitted command:

```bash
sbatch -N 32 -J e97-1p3b-k160-32n-r3 --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-32n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_32n_r3,RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt,TRAIN_MINUTES=35,WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200,WALLTIME_CHECK_EVERY=160,DISTRIBUTED_HEALTH_CHECK_EVERY=160,REQUESTED_NODE_HOURS=32.0 scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

Only one allocated 32-node r3 retry job was submitted for this task.

## Code And Source Checkpoint

- Worktree HEAD / run manifest commit:
  `3e10964b7aa6d97f0c71a980d275b7ba58366a33`.
- `3e10964` is the current `main` commit in this worktree.
- `train.py` contains gated distributed non-finite loss/gradient health checks
  through `--distributed_health_check_every`.
- `train.py` contains gated walltime/finalization consensus checks through
  `--walltime_check_every`.
- `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch` defaults:
  - `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200`
  - `WALLTIME_CHECK_EVERY=160`
  - `DISTRIBUTED_HEALTH_CHECK_EVERY=160`

Source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt
```

It resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/checkpoint_step_269920_loss_2.9823.pt
```

This is the requested job-4903889 K-aligned periodic checkpoint at step
`269920`.

## Runtime Topology

Confirmed from the Slurm submit line, launcher, `args.json`, run manifest, and
train log:

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
- `WALLTIME_CHECK_EVERY=160`
- `DISTRIBUTED_HEALTH_CHECK_EVERY=160`
- No GDN2, CMAES, schedule-free outer optimizer, K sweep, LR sweep, or
  DDP/island-size change was submitted.

The model optimizer inside `train.py` remained the validated `schedulefree`
inner optimizer path; the DiLoCo outer optimizer was `avg`, as required.

## Artifacts

- Slurm job id: `4904355`
- Slurm stdout: `logs/frontier/scaleout/e97-1p3b-k160-32n-r3-4904355.out`
- Slurm stderr: `logs/frontier/scaleout/e97-1p3b-k160-32n-r3-4904355.err`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r3/4904355-20260626T105907Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r3/4904355-20260626T105907Z/logs/train.log`
- Artifact manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r3/4904355-20260626T105907Z/artifacts/manifest.json`
- Training run manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r3/4904355-20260626T105907Z/train/emender_E97_1.3B_20260626_070123/run_manifest.json`
- Args:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r3/4904355-20260626T105907Z/train/emender_E97_1.3B_20260626_070123/args.json`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r3/4904355-20260626T105907Z/summaries/summary.md`
- Final `latest.pt`: **none**. The run directory contains only `args.json`
  and `run_manifest.json` under the train run; no `.pt` checkpoint was written.

## Slurm Accounting

Terminal `sacct` state:

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|AllocTRES|Submit|Start|End
4904355|e97-1p3b-k160-32n-r3|FAILED|137:0|00:16:03|32|billing=3584,cpu=3584,energy=25416768,mem=16000G,node=32|2026-06-26T06:46:24|2026-06-26T06:59:05|2026-06-26T07:15:08
4904355.batch|batch|FAILED|137:0|00:16:03|1|cpu=56,mem=500G,node=1|2026-06-26T06:59:05|2026-06-26T06:59:05|2026-06-26T07:15:08
4904355.extern|extern|COMPLETED|0:0|00:16:03|32|billing=3584,cpu=3584,mem=16000G,node=32|2026-06-26T06:59:05|2026-06-26T06:59:05|2026-06-26T07:15:08
4904355.0|bash|CANCELLED|0:9|00:15:55|32|cpu=1792,mem=16000G,node=32|2026-06-26T06:59:12|2026-06-26T06:59:12|2026-06-26T07:15:07
```

- Requested node-hours: `32.0`
- Actual elapsed: `00:16:03` (`963` seconds)
- Actual node-hours: `32 * 963 / 3600 = 8.56`

## Training Metrics Before Failure

Rank-0 training metric lines were available from step `269925` through
`270000`, 16 logged points covering 80 local steps after resume. The job failed
before 500, 1000, or 2000 local steps were available.

Available trailing rank-0 loss averages:

- Last 500 local steps where available: 16 logged points, average loss
  `2.571875`
- Last 1000 local steps where available: 16 logged points, average loss
  `2.571875`
- Last 2000 local steps where available: 16 logged points, average loss
  `2.571875`

First and last rank-0 metrics:

```text
step 269925 | loss 2.8726 | lr 1.01e-03 | grad 1.66 | tok/s 880 | global_tok/s 225365 | elapsed_h 0.003 | time 2026-06-26T11:01:47+00:00
step 270000 | loss 2.1674 | lr 1.01e-03 | grad 1.69 | tok/s 2926 | global_tok/s 749128 | elapsed_h 0.018 | time 2026-06-26T11:02:41+00:00
```

Throughput:

- Average global throughput excluding the first warmup-adjacent metric:
  `743,376 tok/s`
- Observed global throughput range: `225,365` to `750,980 tok/s`

Approximate ensemble tokens processed after resume through the last logged
rank-0 step:

```text
(270000 - 269920) * 256 ranks * 2048 tokens = 41,943,040 tokens
```

Because no later successful training step was logged, the conservative reported
processed-token count is `41,943,040`.

## DiLoCo And Sync

- `DILOCO_MERGES`: `0` after resume.
- The retry started at step `269920`; the next K160 merge would have been at
  step `270080`.
- The last logged rank-0 training step was `270000`, so the job failed before
  reaching the first post-resume DiLoCo merge.
- No successful DiLoCo merge/sync duration was logged in this retry.
- Sync total/avg/fraction: not computable from successful sync events. The only
  observed synchronization outcome was the failed 1-element `ALLREDUCE`
  watchdog timeout after about 600 seconds.

## Loss Classification

Loss behavior: **indeterminate**.

Reason: all logged rank-0 losses were finite and did not show a large sustained
blowup, but the run ended after only 16 rank-0 metric points and before the
first post-resume K160 merge. The stop condition is systems failure, not model
quality.

## Finalization

Clean finalization did not occur:

- No final consensus merge.
- No explicit already-consensus finalization skip.
- No final checkpoint.
- No `latest.pt` pointing at a final checkpoint.
- No final-ready mismatch was observed; the run failed before the finalization
  path.
- No valid 64-node successor checkpoint.

## Validation Checklist

- [x] Confirmed main/HEAD is at commit `3e10964` or later and `train.py` /
  launcher contain both scalar-collective cadence fixes.
- [x] Submitted no more than one allocated 32-node retry job for this task.
- [x] Confirmed source checkpoint resolves to job-4903889 K-aligned periodic
  `latest.pt` at step `269920`.
- [x] Confirmed topology: 32 nodes, 256 ranks, `DILOCO_ISLAND_SIZE=1`, no DDP,
  avg outer, K160, export basis `x`.
- [x] Confirmed runtime settings: `TRAIN_MINUTES=35`,
  `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200`,
  `WALLTIME_CHECK_EVERY=160`, `DISTRIBUTED_HEALTH_CHECK_EVERY=160`.
- [x] Checked finalization; it failed before final checkpoint, so this criterion
  is not satisfied by the run.
- [x] Reported Slurm job id, elapsed time, requested and actual node-hours, run
  root, train log, manifest, summary, and final `latest.pt` status.
- [x] Reported trailing rank-0 training loss averages where available.
- [x] Classified loss behavior as indeterminate, not blowup.
- [x] Reported throughput, approximate ensemble tokens processed,
  `DILOCO_MERGES`, sync outcome, and sync fraction availability.
- [x] Confirmed no GDN2, CMAES, schedule-free outer, K sweep, LR sweep, or
  DDP/island-size change was submitted.
