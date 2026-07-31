# E97 1.3B Pretrained K160 32-Node Retry R4 - 2026-06-26

WG task: `run-e97-1p3b-k160-32n`

## Result

Status: **failed on systems grounds**.

The single allowed r4 retry job, Slurm `4904689`, started correctly on 32
Frontier nodes / 256 ranks from the requested job-4903889 K-aligned periodic
checkpoint. It resumed at step `269920`, logged finite rank-0 training through
step `270000`, then failed during the first post-resume K160 synchronization
window. The NCCL watchdog evidence is no longer the 1-element scalar-clock
collective seen in r3. It is a model-sized tensor `ALLREDUCE`:

```text
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=153, OpType=ALLREDUCE, NumelIn=1286589072, NumelOut=1286589072, Timeout(ms)=600000)
```

`1286589072` matches the run manifest's trainable parameter count for the
E97 1.3B model, so the r4 probe reached the intended post-scalar-fix failure
surface: the first K160 DiLoCo model-state merge did not complete. No successful
merge duration, final consensus, final checkpoint, or valid 64-node successor
checkpoint was produced.

## Submission

Submitted command:

```bash
sbatch -N 32 -J e97-1p3b-k160-32n-r4 --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-32n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_32n_r4,RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt,TRAIN_MINUTES=35,WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200,WALLTIME_CHECK_EVERY=160,DISTRIBUTED_HEALTH_CHECK_EVERY=160,REQUESTED_NODE_HOURS=32.0 scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

Only one allocated 32-node r4 retry job was submitted for this task.

## Code And Source Checkpoint

- Worktree HEAD / `origin/main`: `133b069`.
- Run manifest commit:
  `133b06932ec73f16fb079b3794dcb4f5ac708ef9`.
- `train.py` contains the ScheduleFree DiLoCo scalar-collective removal from
  `27c21b7`: the merge averages tensor state (`x`, `z`, and optional exported
  `y`) while keeping scalar optimizer clocks (`weight_sum`, `k`, `lr_max`)
  local.
- `133b069` contains the r3 failure report tying the prior `SeqNum=153`
  1-element timeout to the first K160 merge timing.

Source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt
```

It resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/checkpoint_step_269920_loss_2.9823.pt
```

This is the requested job-4903889 K-aligned periodic `latest.pt` at step
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

- Slurm job id: `4904689`
- Slurm stdout: `logs/frontier/scaleout/e97-1p3b-k160-32n-r4-4904689.out`
- Slurm stderr: `logs/frontier/scaleout/e97-1p3b-k160-32n-r4-4904689.err`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z/logs/train.log`
- Artifact manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z/artifacts/manifest.json`
- Training run manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z/train/emender_E97_1.3B_20260626_105757/run_manifest.json`
- Args:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z/train/emender_E97_1.3B_20260626_105757/args.json`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z/summaries/summary.md`
- Final `latest.pt`: **none**. The training run directory contains only
  `args.json` and `run_manifest.json`; no `.pt` checkpoint was written.

## Slurm Accounting

Terminal `sacct` state:

```text
JobID|JobName|State|ExitCode|Elapsed|AllocNodes|NTasks|AllocCPUS
4904689|e97-1p3b-k160-32n-r4|FAILED|137:0|00:16:00|32||3584
4904689.batch|batch|FAILED|137:0|00:16:00|1|1|56
4904689.extern|extern|COMPLETED|0:0|00:16:00|32|32|3584
4904689.0|bash|CANCELLED|0:9|00:15:52|32|256|1792
```

- Submitted: `2026-06-26T10:04:53-04:00`
- Started: `2026-06-26T10:55:38-04:00`
- Ended: `2026-06-26T11:11:38-04:00`
- Requested node-hours: `32.0`
- Actual elapsed: `00:16:00` (`960` seconds)
- Actual node-hours: `32 * 960 / 3600 = 8.53`

## Training Metrics Before Failure

Rank-0 training metric lines were available from step `269925` through
`270000`, 16 logged points covering 80 local steps after resume. The job failed
before 500, 1000, or 2000 local steps were available.

Available trailing rank-0 loss averages:

- Last 500 local steps where available: 16 logged points, average loss
  `2.587975`
- Last 1000 local steps where available: 16 logged points, average loss
  `2.587975`
- Last 2000 local steps where available: 16 logged points, average loss
  `2.587975`

First and last rank-0 metrics:

```text
step 269925 | loss 2.9070 | lr 1.01e-03 | grad 1.49 | tok/s 1268 | global_tok/s 324719 | elapsed_h 0.002 | time 2026-06-26T14:58:18+00:00
step 270000 | loss 2.1434 | lr 1.01e-03 | grad 1.63 | tok/s 2910 | global_tok/s 745054 | elapsed_h 0.017 | time 2026-06-26T14:59:12+00:00
```

Throughput:

- Average global throughput across the 16 logged rank-0 metric lines:
  `719,551 tok/s`
- Last logged global throughput: `745,054 tok/s`
- Observed logged loss range: `2.1434` to `2.9070`

Approximate ensemble tokens processed after resume through the last logged
rank-0 step:

```text
(270000 - 269920) * 256 ranks * 2048 tokens = 41,943,040 tokens
```

Because no later successful training step was logged, the conservative reported
processed-token count is `41,943,040`.

## DiLoCo And Sync

- `DILOCO_MERGES`: `0` successful logged merges after resume.
- The retry started at step `269920`; the next K160 merge was due at step
  `270080`.
- No `step 270080` metric line and no successful DiLoCo merge duration were
  logged.
- The failure evidence is a 600-second NCCL watchdog timeout on
  `SeqNum=153`, `OpType=ALLREDUCE`, `NumelIn=1286589072`,
  `NumelOut=1286589072`.
- Sync total/avg/fraction: not computable from successful sync events. The only
  observed sync outcome was the failed model-sized ALLREDUCE, so the sync
  fraction is pathological/failed rather than a meaningful steady-state metric.

## Loss Classification

Loss behavior: **indeterminate**.

Reason: all logged rank-0 losses were finite and did not show a large sustained
blowup, but the run ended after only 16 rank-0 metric points and failed during
the first post-resume K160 synchronization window. The stop condition is systems
failure, not model quality.

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

- [x] Confirmed `origin/main`/HEAD is at commit `133b069` and `train.py`
  contains the ScheduleFree DiLoCo merge scalar-collective removal.
- [x] Submitted no more than one allocated 32-node r4 retry job for this task.
- [x] Confirmed source checkpoint resolves to job-4903889 K-aligned periodic
  `latest.pt` at step `269920`.
- [x] Confirmed topology: 32 nodes, 256 ranks, `DILOCO_ISLAND_SIZE=1`, no DDP,
  avg outer, K160, export basis `x`.
- [x] Confirmed runtime settings: `TRAIN_MINUTES=35`,
  `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200`,
  `WALLTIME_CHECK_EVERY=160`, `DISTRIBUTED_HEALTH_CHECK_EVERY=160`.
- [x] Confirmed the first post-resume K160 DiLoCo merge at step `270080` did
  not succeed; the evidence is a model-sized NCCL `ALLREDUCE` timeout at
  `SeqNum=153`.
- [x] Checked finalization; it failed before final checkpoint, so this
  criterion is not satisfied by the run.
- [x] Reported Slurm job id, elapsed time, requested and actual node-hours, run
  root, train log, manifest, summary, and final `latest.pt` status.
- [x] Reported trailing rank-0 training loss averages where available.
- [x] Classified loss behavior as indeterminate, not blowup.
- [x] Reported throughput, approximate ensemble tokens processed,
  `DILOCO_MERGES`, sync outcome, and sync fraction availability.
- [x] Confirmed no GDN2, CMAES, schedule-free outer, K sweep, LR sweep, or
  DDP/island-size change was submitted.
