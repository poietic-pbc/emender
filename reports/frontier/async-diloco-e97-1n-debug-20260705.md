# Async DiLoCo E97 1-Node Debug

Task: `async-diloco-e97-1n-debug`
Date: 2026-07-05
Conclusion: `pass`

## Slurm Job

- Job ID: `4943217`
- Job name: `async-diloco-e97-1n`
- Node: `frontier09044`
- State: `COMPLETED`
- Exit code: `0:0`
- Slurm elapsed: `00:00:40`
- Requested allocation: 1 node for `00:20:00`
- Requested node-hours: `0.333333`
- Debug run elapsed inside runner: `7.914347027195618` seconds

## Command

```bash
WG_TASK_ID=async-diloco-e97-1n-debug \
TRAINING_TARGET=E97_1.3B_step483000_async_diloco_debug \
E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt \
PRODUCTION_LATEST_GUARD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt \
ASYNC_WORKER_COUNT=8 \
ASYNC_LOCAL_QUORUM=8 \
ASYNC_LOCAL_STEPS=1 \
REQUESTED_WALLTIME=00:20:00 \
REQUESTED_NODE_HOURS=0.333333 \
sbatch scripts/frontier/async_diloco_e97_1n_debug.sbatch
```

The exact command executed inside the job is recorded at:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/artifacts/command.txt`

## Artifacts

- Metrics JSON:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/artifacts/async_diloco_e97_1n_metrics.json`
- Job environment:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/artifacts/env.txt`
- Run summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/summaries/summary.md`
- Prototype metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/async_run/prototype_metrics.json`
- Debug latest pointer:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/async_run/latest.json`
- Generation manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/async_run/generations/gen_000000/manifest.json`
- Slurm stdout:
  `logs/frontier/async_diloco_e97/async-diloco-e97-1n-4943217.out`
- Slurm stderr:
  `logs/frontier/async_diloco_e97/async-diloco-e97-1n-4943217.err`

## E97 Checkpoint And Target

The job loaded the intended non-production debug target:

- Training target: `E97_1.3B_step483000_async_diloco_debug`
- Source checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt`
- Source checkpoint size: `7719673482` bytes
- Source checkpoint mtime before and after: `1782635295000000000`
- Source checkpoint modified by run: `false`
- Run directory:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z`

## Metrics

Machine-readable metrics reported:

- Configured quorum: local `8`, global `1`, worker count `8`
- Effective quorum: local `8`, global `1`
- Local updates: accepted `8`, stale `0`, timed out `0`, failed `0`, invalid `0`
- Global updates: accepted `1`, stale `0`, timed out `0`, failed `0`, invalid `0`
- Local generation duration: `1.2379735128488392` seconds
- Global generation duration: `1.2379735128488392` seconds
- Local tokens/sec: `6617.265971344147`
- Global tokens/sec: `6617.265971344147`
- Local update bytes: worker `22027081984`, node `2753385248`
- Global update bytes: node `2753385248`, global state `2753385248`
- Checkpoint finalization latest advanced: `true`
- Checkpoint finalization duration: `0.0` seconds
- Checkpoint path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/async_run/generations/gen_000000/manifest.json`
- Checkpoint manifest size: `8383` bytes

The metrics JSON includes the required configured quorum, effective quorum,
update counts, generation duration, tokens/sec, checkpoint paths/durations, and
latest advancement fields.

## Latest Pointer Guard

The only latest pointer advanced by this run was the debug run directory latest:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/async_run/latest.json`

Production latest guard was checked before and after:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

The guard resolved to the same checkpoint before and after:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt`

Production latest changed: `false`.

## Validation

- Slurm job ID, command, logs, elapsed time, and node-hours are recorded above.
- The job loaded the intended E97 checkpoint/training target from a
  non-production debug run directory.
- One async local quorum generation completed with 8 accepted worker updates.
- The metrics artifact is machine-readable JSON and reports the required quorum,
  update, timing, throughput, checkpoint, and latest fields.
- The debug run directory latest pointer advanced; the guarded production latest
  pointer did not change.
- Pass/no-go conclusion: `pass`.
