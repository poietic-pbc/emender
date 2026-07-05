# Async DiLoCo E97 2/8-Node Debug Ladder

Task: `async-diloco-e97-2n8n-debug`
Date: 2026-07-05
Conclusion: `pass`

## Scope

This ladder ran the E97 async quorum DiLoCo debug path on Frontier at 2 nodes
and then 8 nodes. It used the E97 checkpoint:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt`

All latest pointer writes were confined to non-production debug run directories.
The production latest guard was checked before and after both passing jobs:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

Production latest changed: `false` for both passing jobs.

## 2-Node Job

- Slurm job ID: `4943251`
- Job name: `async-diloco-e97-2n`
- Nodes: `frontier[04543-04544]`
- State: `COMPLETED`
- Exit code: `0:0`
- Slurm elapsed: `00:00:44`
- Runner elapsed: `11.983046225039288` seconds
- Requested allocation: 2 nodes for `00:20:00`
- Requested node-hours: `0.666667`
- Pass/no-go conclusion: `pass`

Command submitted from this WG task:

```bash
WG_TASK_ID=async-diloco-e97-2n8n-debug \
TRAINING_TARGET=E97_1.3B_step483000_async_diloco_debug \
E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt \
PRODUCTION_LATEST_GUARD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt \
ASYNC_WORKER_COUNT_PER_NODE=8 \
ASYNC_LOCAL_QUORUM=8 \
ASYNC_GLOBAL_QUORUM=2 \
ASYNC_LOCAL_STEPS=1 \
ASYNC_RESUME_CHECK=1 \
REQUESTED_WALLTIME=00:20:00 \
REQUESTED_NODE_HOURS=0.666667 \
sbatch -N 2 -J async-diloco-e97-2n scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

Artifacts:

- Metrics JSON:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/artifacts/async_diloco_e97_2n_metrics.json`
- Command file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/artifacts/command.txt`
- Environment file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/artifacts/env.txt`
- Run summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/summaries/summary.md`
- Debug latest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/async_run/latest.json`
- Generation manifests:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/async_run/generations/gen_000000/manifest.json`
  and
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/async_run/generations/gen_000001/manifest.json`
- Slurm stdout:
  `logs/frontier/async_diloco_e97/async-diloco-e97-2n-4943251.out`
- Slurm stderr:
  `logs/frontier/async_diloco_e97/async-diloco-e97-2n-4943251.err`

Machine-readable metric excerpt:

```json
{
  "configured_quorum": {
    "global_quorum": 2,
    "local_quorum": 8,
    "nodes": 2,
    "worker_count_per_node": 8
  },
  "effective_quorum": {
    "global": 2,
    "local_distribution": {
      "average": 8.0,
      "max": 8,
      "min": 8
    },
    "local_by_node": {
      "node-000": 8,
      "node-001": 8
    }
  },
  "update_counts": {
    "global": {
      "accepted": 2,
      "failed": 0,
      "invalid": 0,
      "stale": 0,
      "timed_out": 0
    },
    "local_total": {
      "accepted": 16,
      "failed": 0,
      "invalid": 0,
      "stale": 0,
      "timed_out": 0
    }
  },
  "duration_s": {
    "checkpoint": 0.0,
    "global_merge": 1.1271432670764625,
    "global_rebase": 0.0,
    "local_merge_by_node": {
      "node-000": 1.7437401190400124,
      "node-001": 1.6965356038417667
    }
  },
  "tokens_per_sec": {
    "aggregate": 1367.2650252958765,
    "global": 13082.890776699498,
    "local_by_node": {
      "node-000": 6541.445388349749,
      "node-001": 6584.688558240504
    }
  }
}
```

Checkpoint resume was tested in the same non-production debug run directory:
latest finalized generation `0` was selected and generation `1` was published.

## 8-Node Induced Drop Job

- Slurm job ID: `4943254`
- Job name: `async-diloco-e97-8n`
- Nodes:
  `frontier[01041,02859,03801,03948,04543,04657,04939,05057]`
- State: `COMPLETED`
- Exit code: `0:0`
- Slurm elapsed: `00:01:04`
- Runner elapsed: `30.394721147837117` seconds
- Requested allocation: 8 nodes for `00:20:00`
- Requested node-hours: `2.666667`
- Pass/no-go conclusion: `pass`

Command submitted from this WG task:

```bash
WG_TASK_ID=async-diloco-e97-2n8n-debug \
TRAINING_TARGET=E97_1.3B_step483000_async_diloco_debug \
E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt \
PRODUCTION_LATEST_GUARD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt \
ASYNC_WORKER_COUNT_PER_NODE=8 \
ASYNC_LOCAL_QUORUM=8 \
ASYNC_GLOBAL_QUORUM=6 \
ASYNC_GLOBAL_DROP_NODE_IDS=6,7 \
ASYNC_LOCAL_STEPS=1 \
ASYNC_RESUME_CHECK=1 \
REQUESTED_WALLTIME=00:20:00 \
REQUESTED_NODE_HOURS=2.666667 \
sbatch -N 8 -J async-diloco-e97-8n scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

Artifacts:

- Metrics JSON:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/artifacts/async_diloco_e97_8n_metrics.json`
- Command file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/artifacts/command.txt`
- Environment file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/artifacts/env.txt`
- Run summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/summaries/summary.md`
- Debug latest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/async_run/latest.json`
- Generation manifests:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/async_run/generations/gen_000000/manifest.json`
  and
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/async_run/generations/gen_000001/manifest.json`
- Slurm stdout:
  `logs/frontier/async_diloco_e97/async-diloco-e97-8n-4943254.out`
- Slurm stderr:
  `logs/frontier/async_diloco_e97/async-diloco-e97-8n-4943254.err`

Machine-readable metric excerpt:

```json
{
  "configured_quorum": {
    "global_quorum": 6,
    "local_quorum": 8,
    "nodes": 8,
    "worker_count_per_node": 8
  },
  "effective_quorum": {
    "global": 6,
    "local_distribution": {
      "average": 8.0,
      "max": 8,
      "min": 8
    }
  },
  "induced_lag_drop": {
    "global_dropped_node_ids": [6, 7],
    "measured": true
  },
  "update_counts": {
    "global": {
      "accepted": 6,
      "failed": 0,
      "invalid": 0,
      "stale": 0,
      "timed_out": 2
    },
    "local_total": {
      "accepted": 64,
      "failed": 0,
      "invalid": 0,
      "stale": 0,
      "timed_out": 0
    }
  },
  "duration_s": {
    "checkpoint": 0.0,
    "global_merge": 1.9192170419264585,
    "global_rebase": 0.0
  },
  "tokens_per_sec": {
    "aggregate": 2156.1638838941453,
    "global": 36778.23752569986
  }
}
```

The induced case is measured at the global node-update layer: nodes 6 and 7
were intentionally represented as timed-out/missing updates, while global
quorum was set to 6. The generation finalized with 6 accepted node updates and
2 timed-out node updates, proving this debug path does not require all node
workers to join the generation.

Checkpoint resume was also tested in this non-production run directory:
latest finalized generation `0` was selected and generation `1` was published.

## Failed Attempt

The first 2-node submission, Slurm job `4943244`, failed before Python in
`00:00:24` because the sbatch wrapper defaulted `ENV_PREFIX` to the WG worktree
path, where the conda environment does not exist. The wrapper was patched to
fall back to:

`/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`

The passing jobs above used that fallback.

## Gate Recommendation

Recommendation for the paused 32/64-node approval gate: **go for the next
bounded approval step, but do not treat this as full production readiness**.

Evidence supporting go:

- E97 checkpoint loading is read-only; source checkpoint metadata was unchanged.
- 2-node and 8-node Frontier allocations completed with pass conclusions.
- 8-node induced drop case finalized with global quorum 6/8 and recorded two
  timed-out/missing node updates.
- Checkpoint resume from latest finalized generation was tested in both passing
  non-production run directories.
- Production latest guard remained unchanged.
- No 32+ node jobs were submitted from this task.

Residual limits:

- This is still a short debug-scale prototype path. It validates quorum,
  metrics, checkpoint/latest semantics, and Frontier launch plumbing, not
  sustained E97 training throughput or full async transport performance.
- Stale rejected updates remained `0` in these Frontier jobs. The induced case
  measured missing/timed-out node updates rather than late stale arrivals.
