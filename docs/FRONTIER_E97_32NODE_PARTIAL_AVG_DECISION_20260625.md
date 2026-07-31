# Frontier E97 32-Node Partial-Average Decision

Date: 2026-06-25
Task: `run-e97-32-node-2`

## Decision

No 32-node partial-average training job was submitted.

The K80 rung did not clear the fixed-validation gate, so this task was eligible
to consider the merge-strength diagnostic. The implementation audit found that
the intended partial-average semantics exist for the `momentum` DiLoCo outer
optimizer, but the required resume shape is not safe for this diagnostic:
`train.py` refuses to resume from a checkpoint without matching
`diloco_outer_state` when `--diloco_outer_optimizer` is anything other than
`avg`. The required source checkpoint is the same 16-node source checkpoint used
by the K80 rung, not a checkpoint with momentum outer state.

Launching the preferred candidate would therefore change more than merge
strength. It would require either bypassing the resume guard, altering
checkpoint semantics, or starting from a checkpoint with different optimizer
provenance. Per the task instruction, the safe action is to submit no job and
record the decision.

## K80 Gate Result

The preceding K80 diagnostic completed cleanly but failed the green gate:

| Checkpoint | Step | Fixed CE | Fixed BPB | CE delta vs source | BPB delta vs source |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-node source | 1328 | 10.49609756 | 4.36699062 | 0.00000000 | 0.00000000 |
| 32-node K80 diagnostic | 2279 | 10.57016373 | 4.39780647 | +0.07406617 | +0.03081585 |

Gate from `plan-e97-32-node`: fixed BPB delta `<= +0.010` and fixed CE delta
`<= +0.025`. K80 improved over K40 but still failed both thresholds.

K80 evidence source:

```text
docs/FRONTIER_E97_32NODE_K80_DIAGNOSTIC_20260625.md
logs/frontier/eval/e97-k80-fixed-eval-4899139.out
```

## Partial-Average Semantics Audit

The relevant command surface is present in `train.py`:

```text
--diloco_outer_optimizer {avg,momentum,sfsgd}
--diloco_outer_lr
--diloco_outer_beta
--diloco_export_basis
```

The intended partial-average formula is implemented in `diloco_merge()` for
the `momentum` outer optimizer:

```text
delta     = mean_i(W_{r,i}) - W_r
outer_mom = outer_beta * outer_mom + delta
W_{r+1}   = W_r + outer_lr * outer_mom
```

With `outer_beta=0.0`, this becomes:

```text
W_{r+1} = W_r + outer_lr * (mean_i(W_{r,i}) - W_r)
```

So the preferred candidate, if resume-safe, would have been:

```text
DILOCO_K=80
DILOCO_ISLAND_SIZE=8
DILOCO_OUTER_OPTIMIZER=momentum
DILOCO_OUTER_LR=0.5
DILOCO_OUTER_BETA=0.0
DILOCO_EXPORT_BASIS=x
```

This would be a half step from the outer anchor toward the cross-island/global
mean at each DiLoCo boundary.

However, the same audit found the resume guard:

```text
if loaded_outer_state is None and args.resume and args.diloco_outer_optimizer != 'avg':
    raise ValueError(...)
```

That guard is load-bearing for this task because the required run must resume
from the same 16-node source checkpoint. A momentum outer diagnostic from that
source would fail before training unless the code or source checkpoint
semantics were changed.

`avg` with `DILOCO_OUTER_LR < 1` is not a substitute. In the `avg` route,
`diloco_merge()` performs the all-reduce average directly and skips the
outer-optimizer step, so `outer_lr` does not express partial averaging.

## Job And Scope Confirmation

No training job was submitted for `run-e97-32-node-2`.

Scheduler check from this task window:

```text
sacct -S 2026-06-25T09:24:00 -o JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,NNodes,Submit,Start,End -P
```

No matching E97 partial-average, merge-strength, GDN2, CMAES, schedule-free, or
64+ node job appeared in that accounting query.

Because no training job was submitted, there is no candidate checkpoint and no
source-vs-candidate fixed eval was run for this task.

`run-64-node-e97` remains `open (PAUSED)`.

## Recommendation

Do not treat partial averaging as tested. The current codebase supports the
mathematical operation only through the stateful `momentum` outer path, while
this rung requires resuming from a source checkpoint without that state. The
next decision point should either:

1. keep the K80 result as the best clean cadence-only 32-node evidence so far,
   or
2. create a separate implementation/design task to define a stateless
   partial-average outer mode, or an explicit fresh-initialization policy for
   momentum outer state on resume, before spending 32 nodes on this diagnostic.

That follow-up would be a code/semantics change, not the one-concept
merge-strength diagnostic requested here.
