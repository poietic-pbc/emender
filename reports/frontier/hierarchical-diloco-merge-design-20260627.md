# Hierarchical DiLoCo Merge Reduction Design

Task: `design-hierarchical-diloco`
Date: 2026-06-27

## Feasibility

An exact application-level hierarchical DiLoCo merge is feasible with the current
one-GPU-per-process `torch.distributed` setup. The implementation in this branch
is opt-in via `--diloco_merge_topology hierarchical` or
`DILOCO_MERGE_TOPOLOGY=hierarchical`; the default remains the existing global
all-reduce path.

This is distinct from NCCL/RCCL's internal tree or ring choices. NCCL can choose
a tree algorithm inside one collective, but the application still presents a
single all-rank `all_reduce` over a full bucket. The hierarchical path changes
the application communication schedule:

1. Reduce each bucket to one root per first-level group.
2. All-reduce only the roots.
3. Divide by the global world size.
4. Broadcast the exact average from each root back to its group.

The reduction is exact for current DiLoCo averaging semantics because first-level
groups produce sums, not averages. Unequal group sizes are therefore handled by
construction: the final value is `sum(all ranks) / world_size`. A naive
average-of-averages would be correct only when all first-level groups have the
same size.

## Current Code Inspected

- `train.py:diloco_merge`: merges ScheduleFree `sf_x`, `sf_z`, optional `sf_y`
  for `sfsgd`/`y` export, or non-ScheduleFree `params`.
- `train.py:_diloco_allreduce_average_flat`: existing bucketed global
  `SUM / world_size` helper.
- `train.py` distributed setup: one process per GPU, `srun` exports
  `RANK=$SLURM_PROCID`, `WORLD_SIZE=$SLURM_NTASKS`, and `LOCAL_RANK=0` because
  each Slurm task receives one visible GPU.
- `train.py` hybrid island setup: already uses `dist.new_group` safely by having
  all ranks create every subgroup in the same order, then warms subgroup NCCL
  communicators sequentially.
- `scripts/frontier/diloco_scaleout_readiness.sbatch`: Frontier launch template
  uses 8 tasks per node and consecutive global ranks, so groups of 8 map to GPU
  islands/nodes for the scaleout runs.

## Prototype In This Branch

`train.py` adds:

- `--diloco_merge_topology {global,hierarchical}`
- `--diloco_merge_group_size`, default `8`
- `_build_diloco_hierarchical_merge_groups(...)`
- topology-aware `_diloco_allreduce_average_flat(...)`

For every existing merge tensor, the same bucket loop is retained. Each bucket
uses one of two paths:

- `global`: exactly the previous `dist.all_reduce(SUM)` then `div_(world_size)`.
- `hierarchical`: `dist.reduce(SUM)` within the local group to its root, root
  `dist.all_reduce(SUM)` across roots, root `div_(world_size)`, then
  `dist.broadcast` back to local group members.

The implementation deliberately does not change tensor selection or DiLoCo
optimizer semantics. It only changes how the existing flat tensor buckets are
communicated.

## ScheduleFree And Non-ScheduleFree Semantics

The hierarchical helper is called by the same sites as the global helper:

- `sf_x`: averaged exactly across all ranks.
- `sf_z`: averaged exactly across all ranks when present.
- `sf_y`: averaged exactly across all ranks for the `sfsgd` outer optimizer with
  `--diloco_export_basis y`.
- `params`: averaged exactly across all ranks for non-ScheduleFree optimizers.

ScheduleFree scalar clocks remain local, as before. The new topology does not
introduce scalar collectives.

## Frontier Topology Recommendation

For Frontier scaleout, use:

- `DILOCO_MERGE_TOPOLOGY=hierarchical`
- `DILOCO_MERGE_GROUP_SIZE=8`
- `DILOCO_MERGE_BUCKET_NUMEL=67108864` initially, matching the successful
  32-node bucketed diagnostic.

This forms one first-level group per 8-rank node/GPU island for the current
`srun --ntasks-per-node=8` launch layout. With 32 nodes, roots are ranks
`0,8,16,...,248`; with 64 nodes, roots are `0,8,16,...,504`.

If a future launcher no longer assigns consecutive ranks by node, the grouping
rule should be revisited. In the current Slurm templates, consecutive global
ranks are the practical topology signal; `LOCAL_RANK` is intentionally set to
`0` inside each one-GPU Slurm task and is not suitable for grouping.

## Risk Assessment

Primary risks:

- Extra process groups and collectives increase ordering complexity. The branch
  follows the existing hybrid-island pattern: all ranks call `new_group` for all
  subgroups in deterministic order, and subgroup communicators are warmed before
  training reaches the first model bucket.
- Hierarchical root all-reduce changes root-rank traffic shape. It should reduce
  full-world participation in the inter-node step, but it adds a local reduce and
  broadcast around each root collective.
- Performance is topology-dependent. This is a correctness-preserving prototype,
  not proof of faster wall-clock synchronization.

The change is low risk operationally because it is opt-in and `global` remains
the default.

## Validation Performed

Runnable in this bare worktree:

```bash
python3 -m py_compile train.py tests/test_diloco_merge.py tests/test_diloco_hierarchical_math.py
python3 tests/test_diloco_hierarchical_math.py
```

The dependency-free simulation verifies that weighted hierarchical sums match a
global average for unequal groups and that naive average-of-averages is wrong in
that case.

Added but not runnable in this environment because `/usr/bin/python3` lacks
`torch`, `schedulefree`, and `pytest`:

```bash
python3 -m pytest tests/test_diloco_merge.py::test_hierarchical_merge_matches_global_average_for_unequal_groups -q -s
```

That test uses a real 3-rank Gloo process group split as `[0,1]` and `[2]`, and
checks hierarchical results for non-ScheduleFree params plus ScheduleFree `x`
and `z` against the global mean.

## Recommended Next Slurm Experiment

Do not jump directly to a long 64-node run. The next experiment should be a short
32-node replay of the known-good bucketed diagnostic, changing only the merge
topology:

```bash
sbatch -A bif148 -p batch --qos=debug -N 32 -t 00:30:00 \
  --job-name emender-e97-hier-merge-32n \
  --export=ALL,WG_TASK_ID=verify-hierarchical-diloco-32n,\
SCALEOUT_VARIANT=e97-MLP,SCALEOUT_NODES=32,SCALEOUT_WALLTIME=00:30:00,\
TRAIN_MINUTES=8,DILOCO_K=20,DILOCO_OUTER_OPTIMIZER=avg,\
DILOCO_OUTER_LR=1.0,DILOCO_OUTER_BETA=0.0,DILOCO_EXPORT_BASIS=x,\
DILOCO_ISLAND_SIZE=8,DILOCO_MERGE_TOPOLOGY=hierarchical,\
DILOCO_MERGE_GROUP_SIZE=8,DILOCO_MERGE_BUCKET_NUMEL=67108864,\
DILOCO_MERGE_DEBUG=1,DILOCO_MERGE_DEBUG_RANKS=0,8,255,\
<DATA/VAL/TIKTOKEN/HUMAN_APPROVAL/RESUME_CHECKPOINT vars> \
  scripts/frontier/diloco_scaleout_readiness.sbatch
```

Success criteria:

- All selected debug ranks enter and exit every `sf_x` and `sf_z` bucket.
- Log prints `topology=hierarchical group_size=8`.
- At least several DiLoCo merges complete with no NCCL/RCCL watchdog timeout.
- Merge duration is compared against the existing 32-node bucketed global
  diagnostic, not against the old monolithic all-reduce failure.

If that passes, repeat at 64 nodes with the same bucket size and debug roots
`0,8,504`, still as a short diagnostic before any production-length run.
