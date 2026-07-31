# Frontier E97 32-Node Regression Main Merge Record

Date: 2026-06-24
Task: `merge-32-node-e97`

## Merge State

After fetching `origin`, local `main`, `origin/main`, and this task worktree
all resolved to commit `7da366d`.

The requested worker evidence was already present in that main history:

```text
dd44996 feat: investigate-32-node-e97 (agent-171)
b28b318 feat: retry-32-node-e97 (agent-174)
```

The named remote worker refs `origin/wg/agent-171/investigate-32-node-e97`
and `origin/wg/agent-174/retry-32-node-e97` were not resolvable after fetch in
this worktree, but matching local WG branches were present. Those branch tips
had no additional report evidence beyond what main already contained, and they
were behind current main because current main also includes the 32-node quality
pass commit.

## Evidence Present On Main

Current main contains the 32-node E97-MLP regression evidence files:

```text
docs/FRONTIER_E97_32NODE_LOSS_REGRESSION_INVESTIGATION_20260624.md
docs/FRONTIER_E97_32NODE_AVG_RETRY_20260624.md
logs/frontier/scaleout/e97-32n-avg-retry-4894206.err
logs/frontier/scaleout/e97-32n-avg-retry-4894206.out
```

The merged evidence preserves the intended conclusion:

- the original 32-node E97-MLP avg smoke completed operationally cleanly;
- the bounded 32-node retry also completed operationally cleanly;
- the retry repeated the material loss-regression class versus the same
  16-node source checkpoint;
- `run-64-node-e97` should remain blocked/paused until a focused 32-node
  scale-out/configuration investigation produces a fix or stronger acceptance
  criterion.

## Scope Confirmation

This merge record is documentation-only. It does not modify launcher code,
training code, model code, Slurm scripts, or runtime configuration.

No Slurm job was submitted or authorized by this merge task. No 64-node or
larger job was submitted or authorized by this merge task.

The resolution remains E97-MLP-first. It does not add GDN2, schedule-free,
CMAES, or other training-arm work to this batch.
