# E97 Async DiLoCo 256n x 12h Production Run

Date: 2026-07-09
Task: `submit-and-monitor`
Slurm job: `4963853`
Submission time: 2026-07-09T18:13:18Z / 2026-07-09T14:13:18-04:00

## Submission Verdict

Submitted exactly one real E97 async DiLoCo 256-node x 12-hour job to the
regular production `batch` partition. No debug QOS/queue was requested.

Current scheduler state at first check:

```text
JOBID|NAME|PARTITION|QOS|STATE|NODES|TIME_LIMIT|START_TIME|NODELIST(REASON)
4963853|async-diloco-e97-256n12h|batch|normal|PENDING|256|12:00:00|N/A|(Priority)
```

`squeue --start` estimate:

```text
             JOBID PARTITION     NAME     USER ST          START_TIME  NODES SCHEDNODES           NODELIST(REASON)
           4963853     batch async-di erikgarr PD                 N/A    256 (null)               (Priority)
```

Follow-up scheduler check after commit/push produced a concrete estimate:

```text
JOBID|NAME|PARTITION|QOS|STATE|NODES|TIME_LIMIT|START_TIME|NODELIST(REASON)
4963853|async-diloco-e97-256n12h|batch|normal|PENDING|256|12:00:00|2026-07-10T01:32:00|(Priority)
```

```text
             JOBID PARTITION     NAME     USER ST          START_TIME  NODES SCHEDNODES           NODELIST(REASON)
           4963853     batch async-di erikgarr PD 2026-07-10T01:32:00    256 frontier[00062-00065 (Priority)
```

## Git And Seed Gate

- `HEAD`, `main`, and `origin/main`: `005d869389a6380c0185cd17b326491a99ec2d00`.
- Working tree status was noisy with pre-existing untracked WG/log/data files;
  no source edits were staged for this task.
- Seed latest pointer:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt`
- Pointer target:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/checkpoint_step_1282500_loss_2.5175.pt`
- Recorded seed SHA256 from validated state:
  `0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4`
- Recorded token count from validated state: 84.050B tokens.

The wrapper resolves `latest.pt` inside the Slurm job at job start through
`SEED_LATEST_PATH` / `E97_CHECKPOINT`; the submitted command passes the
`latest.pt` path, not a resolved checkpoint baked into a local command.

## Wrapper And Gate

Production wrapper:

```text
scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Wrapper SBATCH settings:

```text
partition=batch
qos=normal (no debug QOS requested; Slurm assigned normal)
nodes=256
walltime=12:00:00
ntasks-per-node=8
cpus-per-task=7
gpus-per-task=1
gpu-bind=closest
stdout=logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-%j.out
stderr=logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-%j.err
```

The production wrapper required a readable passing 64n compiled-helper gate
JSON with `compiled-cray-mpich-helper-p2p`. The original validated artifacts
split this evidence between manifest and metrics files, so this task created a
report-owned machine-readable gate summary:

```text
reports/frontier/e97-async-256n12h-production-64n-gate-20260709.json
```

That gate summary points to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/metrics.json
```

Gate fields used by the wrapper:

```text
validation_status=pass
conclusion=pass
transport=compiled-cray-mpich-helper-p2p
actual_transport=compiled-cray-mpich-helper-collective-reduce
nodes=64
world_size=512
accepted_updates=512
stale/failed/invalid/timed_out updates=0/0/0/0
tcp_dense_data_plane=false
```

## Exact Submission

```bash
sbatch --parsable --export=ALL,WG_TASK_ID=submit-and-monitor,TASK_ID=submit-and-monitor,ASYNC_DILOCO_HUMAN_APPROVED=1,HUMAN_APPROVAL_RECORD=Erik_approval_2026-07-09_queue_one_256n12h_E97_run_now,SEED_LATEST_PATH=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt,E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt,ASYNC_COMPILED_MPICH_64N_GATE_JSON=/lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/e97-async-256n12h-production-64n-gate-20260709.json,BATCH_SIZE=4,CHUNK_SIZE=2048,DILOCO_K=40,ASYNC_LOCAL_STEPS=40,LEARNING_RATE=0.001007,OPTIMIZER=schedulefree,WEIGHT_DECAY=0.01,WARMUP_STEPS=0,GRAD_ACCUM=1,GRAD_CLIP=1.0,MODEL_TOKENIZER=p50k_base,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_ACTUAL_MULTINODE_TCP_QUORUM=0,ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM=0,ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM=1,ASYNC_TRANSPORT_MODE=compiled-cray-mpich-helper-p2p,ASYNC_DENSE_DATA_PLANE=compiled-mpich-p2p,MODEL_LINEAR_STATE=0,ASYNC_E97_USE_CHUNKED=0,ASYNC_E97_CHECKPOINT_INTERVAL=16 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

`sbatch --parsable` returned:

```text
4963853
```

## Key Runtime Recipe

```text
BATCH_SIZE=4
CHUNK_SIZE=2048
DILOCO_K=40
ASYNC_LOCAL_STEPS=40
LEARNING_RATE=0.001007
OPTIMIZER=schedulefree
WEIGHT_DECAY=0.01
WARMUP_STEPS=0
GRAD_ACCUM=1
GRAD_CLIP=1.0
MODEL_TOKENIZER=p50k_base
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0
ASYNC_ACTUAL_MULTINODE_TCP_QUORUM=0
ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM=0
ASYNC_ACTUAL_MULTINODE_COMPILED_MPICH_QUORUM=1
ASYNC_TRANSPORT_MODE=compiled-cray-mpich-helper-p2p
ASYNC_DENSE_DATA_PLANE=compiled-mpich-p2p
MODEL_LINEAR_STATE=0
ASYNC_E97_USE_CHUNKED=0
ASYNC_E97_CHECKPOINT_INTERVAL=16
```

The submitted job command from `scontrol show job 4963853` confirms:

```text
JobState=PENDING Reason=Priority
Account=bif148 QOS=normal
Partition=batch
NumNodes=256-256 NumTasks=2048 CPUs/Task=7
TimeLimit=12:00:00
Command=/lustre/orion/bif148/scratch/erikgarrison/emender/scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4963853.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4963853.err
TresBind=gres/gpu:closest
TresPerTask=cpu=7,gres/gpu=1
```

## Monitoring Plan

Pending monitoring:

- Check `squeue --start -j 4963853` about every 2 hours while the start time is
  `N/A` or while the estimate remains more than 2 hours away.
- If Slurm provides an estimated start time, check at about half the remaining
  estimated time or every 2 hours, whichever is sooner.
- Record each scheduler check in WG logs and append material state changes here.

Next pending check target after the 2026-07-09T18:13Z follow-up is about
2026-07-09T20:13Z, unless the scheduler state changes sooner.

Running monitoring:

- At start, verify wrapper startup artifacts in the Slurm stdout/stderr and run
  artifact directory, including `seed_latest_path`, `e97_checkpoint`,
  `presubmit_status=pass`, and the expanded command.
- Monitor rank joins until all expected 2048 ranks start, then monitor the first
  merge closely.
- Confirm compiled MPICH helper path is active and TCP dense quorum remains
  disabled.
- Confirm first checkpoint/latest publication succeeds and the checkpoint
  includes optimizer state.
- During early running, check every 30-60 minutes; if stable, check about every
  2 hours.

Cancellation criteria:

- Sustained catastrophic instability.
- Non-finite loss.
- Rank startup/join failure.
- Stale, failed, invalid, or timed-out update failure that breaks the run.
- Checkpoint verification, checkpoint save, or latest-pointer update failure.

Small noisy finite loss movement is not a cancellation reason.
