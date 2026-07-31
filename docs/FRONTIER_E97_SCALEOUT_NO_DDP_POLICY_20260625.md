# Frontier E97 Scaleout No-DDP Policy

Task: `update-e97-scaleout`
Date: 2026-06-25

## Decision

For future Frontier E97-MLP scaleout tasks, the primary hypothesis is
GPU-island DiLoCo with no per-step DDP inside an island:

```text
DILOCO_ISLAND_SIZE=1
one GPU per DiLoCo island
no per-step DDP gradient averaging
periodic DiLoCo/model averaging only
```

This preserves the single-GPU learner on each GPU between DiLoCo merge
boundaries. It is the preferred E97-MLP scaleout path when the model fits on one
GPU.

## Baseline Evidence, Not Preferred Recipe

The prior 16-node and 32-node E97-MLP avg scaleout runs were node-island DDP
controls, not evidence that within-node DDP should remain the default recipe.
Those jobs used:

```text
DILOCO_ISLAND_SIZE=8
one Frontier node per island
8 GPUs per island with per-step DDP gradient averaging inside the island
periodic DiLoCo/model averaging across node islands
```

This applies to the 16-node source and the 32-node K10, K40, and K80 avg
diagnostics summarized in the 32-node ladder notes. Interpret those runs as
baseline/control evidence for the old node-island topology. They are useful for
operational comparison, cadence sensitivity, checkpoint behavior, and fixed-eval
gating, but they should not steer future launch tasks back to a
within-node-DDP default.

## Tensor Parallelism Boundary

Tensor/model parallelism is a separate mechanism from DiLoCo island topology.
Do not introduce tensor parallelism for current E97-MLP scaleout solely because
the run spans many nodes. Consider tensor/model parallelism only if a selected
model no longer fits on one GPU. The current E97-MLP scaleout target fits on one
GPU, so tensor parallelism is out of scope for the primary no-DDP path.

## Launch Guidance

Future E97-MLP scaleout planning and execution tasks should:

- Treat `DILOCO_ISLAND_SIZE=1` as the primary recipe when model memory permits.
- Explicitly state that no per-step DDP gradient all-reduce is expected in the
  primary path.
- Use prior `DILOCO_ISLAND_SIZE=8` node-island runs only as controls or
  baselines.
- Avoid changing training code or Slurm templates unless a separate
  implementation task authorizes that work.
- Keep 32-node and larger no-DDP work gated by the existing audit/probe track;
  do not infer authorization from old 16/32-node node-island DDP runs.

## 64-Node Gate

`run-64-node-e97` remains paused and must not be resumed using the old
DDP-shaped default. A future 64-node task would need explicit approval and a
fresh recipe that states whether it is:

- a no-DDP GPU-island primary-path run with `DILOCO_ISLAND_SIZE=1`, or
- a deliberate node-island DDP control/baseline run with `DILOCO_ISLAND_SIZE=8`.

The paused `run-64-node-e97` task is not that approval.

## Validation

- No Slurm jobs were submitted for this documentation update.
- No training code was changed.
- `wg show run-64-node-e97` reports status `open (PAUSED)` as of this update.
