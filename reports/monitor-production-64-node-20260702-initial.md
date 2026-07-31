# Monitor production 64-node E97 B4/K80 chain: initial pass

Task: `monitor-production-64-node`
Timestamp: 2026-07-02T08:33:11-04:00 / 2026-07-02T12:33:11+00:00

## Summary

The two-job production chain was still queued during this monitoring pass. No
training logs or checkpoints from jobs `4931004` or `4931005` existed yet, so
the terminal-state, loss-trend, throughput, merge-timing, final-checkpoint, and
latest-pointer advancement checks could not be completed in this pass.

The task should resume when `4931004` starts or after the current scheduler
estimate. At this point there is no unattended hang: both jobs are pending, not
running.

## Scheduler state

Observed with:

```bash
sacct -j 4931004,4931005 --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End,NodeList%40 -P
squeue -j 4931004,4931005 -o '%i|%t|%M|%l|%D|%R|%j'
scontrol show job 4931004
scontrol show job 4931005
```

Results:

| Job | State | Reason / dependency | Start estimate | Time limit | Nodes | Log paths |
| --- | --- | --- | --- | --- | --- | --- |
| `4931004` | `PENDING` | `Priority` | `2026-07-02T23:16:00` | `02:00:00` | 64 | `logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.{out,err}` |
| `4931005` | `PENDING` | `afterany:4931004(unfulfilled)` | `Unknown` | `02:00:00` | 64 | `logs/frontier/scaleout/e97-b4-k80-64n2h-4931005.{out,err}` |

`sacct` reported both jobs with `Elapsed=00:00:00`, `ExitCode=0:0`, `Start=Unknown`,
and `End=Unknown`. `squeue` reported `4931004|PD|0:00|2:00:00|64|(Priority)`
and `4931005|PD|0:00|2:00:00|64|(Dependency)`.

`scontrol show job 4931004` reported:

- Submit time: `2026-07-02T08:30:50`
- Workdir: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free`
- Command: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch`
- Stdout: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.out`
- Stderr: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.err`
- Submit line included `CHAIN_INDEX=1,WG_TASK_ID=chain-64n-b4-k80-2h-1`

`scontrol show job 4931005` reported the same command/workdir pattern, with:

- Dependency: `afterany:4931004(unfulfilled)`
- Stdout: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931005.out`
- Stderr: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931005.err`
- Submit line included `CHAIN_INDEX=2,WG_TASK_ID=chain-64n-b4-k80-2h-2`

## Log files

The expected log files did not exist yet:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.out
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.err
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931005.out
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931005.err
```

## Checkpoint pointer baseline

Production chain latest path:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

The chain directory and `latest.pt` path did not exist at the time of this pass:

```text
ls: cannot access '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg': No such file or directory
```

Seed checkpoint existed:

```text
-rw------- 1 erikgarrison bif148 7719679569 Jun 30 16:04 /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

## Submission recipe confirmation

The submitted script was:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch
```

Key environment defaults in the script:

- `CHAIN_NAME=E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg`
- `CHAIN_LATEST_PATH=${CHAIN_DIR}/latest.pt`
- `CHAIN_MANIFEST_PATH=${CHAIN_DIR}/latest.pt.manifest.json`
- `CHAIN_SEED_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt`
- `CHAIN_UPDATE_ON_FAILURE=1`
- `TRAIN_MINUTES=105`
- `DILOCO_K=80`
- `DILOCO_MERGE_TOPOLOGY=hierarchical`
- `DILOCO_MERGE_GROUP_SIZE=4`
- `DILOCO_MERGE_BUCKET_NUMEL=67108864`
- `BATCH_SIZE=4`
- `SAVE_EVERY=160`
- `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=900`
- `DISTRIBUTED_HEALTH_CHECK_EVERY=80`
- `DISABLE_SCALAR_STATUS_COLLECTIVES=1`
- `REQUESTED_SECONDS=7200`
- `REQUESTED_WALLTIME=02:00:00`
- `REQUESTED_NODE_HOURS=128.0`

## Resume checklist

When the task is resumed:

1. Check queue state for `4931004` and `4931005`.
2. Tail the expected stdout/stderr files once they appear.
3. For `4931004`, record terminal state, final/last loss trend, token throughput,
   DiLoCo merge timing, final checkpoint path, and whether the chain `latest.pt`
   and manifest advanced.
4. Confirm `4931005` started from the advanced `latest.pt` if `4931004`
   advanced it; otherwise confirm it used the seed or last good latest.
5. Repeat terminal/log/loss/throughput/merge/checkpoint checks for `4931005`.
6. If NCCL/RCCL instability occurs, preserve the exact collective/op/step and
   chain pointer evidence before any cancellation.
