# Frontier E97 32-Node Schedule-Free Decision

Date: 2026-06-25
Task: `run-e97-32-node-3`

## Decision

No 32-node schedule-free training job was submitted.

The preceding K80 rung did not clear the fixed-validation gate, and the
partial-average/merge-strength rung submitted no job because the only existing
partial-average semantics require a non-`avg` outer optimizer state that the
required source checkpoint does not contain. That made this task eligible to
consider schedule-free as the next 32-node ladder rung.

However, the task requires the same 16-node source checkpoint used by K80 and
the same fixed validation tensor/gate. The current implementation deliberately
rejects resuming any non-`avg` DiLoCo outer optimizer from a checkpoint that
lacks `diloco_outer_state`. The required source checkpoint was produced by the
stateless `avg` outer path, so it cannot be used for a conforming
`sfsgd` schedule-free outer continuation without changing checkpoint semantics,
bypassing the resume guard, or starting from a different trajectory.

Those alternatives would add a new uncontrolled variable to this rung. The safe
action is therefore to submit no job and keep schedule-free as a deferred
comparison arm, pending either lower-rung schedule-free synthesis or an explicit
implementation/design task for initializing non-`avg` outer state from an
`avg` source checkpoint.

## Required Source And Gate

16-node source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

Fixed validation tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Gate from the 32-node ladder plan:

```text
fixed CE delta <= +0.025 versus the 16-node source
fixed BPB delta <= +0.010 versus the 16-node source
```

Because no conforming schedule-free candidate checkpoint was produced, no
source-versus-candidate fixed eval was run for this task.

## Preceding Rungs Read Before Decision

K80 evidence source:

```text
docs/FRONTIER_E97_32NODE_K80_DIAGNOSTIC_20260625.md
```

K80 completed cleanly but improved-and-failed the 32-node gate:

| Checkpoint | Step | Fixed CE | Fixed BPB | CE delta vs source | BPB delta vs source |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-node source | 1328 | 10.49609756 | 4.36699062 | 0.00000000 | 0.00000000 |
| 32-node K80 diagnostic | 2279 | 10.57016373 | 4.39780647 | +0.07406617 | +0.03081585 |

Partial-average decision source:

```text
docs/FRONTIER_E97_32NODE_PARTIAL_AVG_DECISION_20260625.md
```

Partial-average submitted no job. The intended half-average operation exists
only through the `momentum` non-`avg` outer route, but the same source checkpoint
lacks `diloco_outer_state`, and `train.py` rejects that resume shape before
training.

## Schedule-Free Configuration Considered

The only locally supported schedule-free outer arm with prior positive evidence
is the `sfsgd_y` route described in the schedule-free audit:

```text
--optimizer schedulefree
--diloco
--diloco_outer_optimizer sfsgd
--diloco_export_basis y
--diloco_outer_lr 1.0
--diloco_outer_beta 0.1
```

The audit's first-run recipe also uses `DILOCO_K=250`, `SAVE_EVERY=250`, and
leaves `RESUME_CHECKPOINT` unset for 4-node and 8-node from-scratch probes.
That is useful schedule-free implementation evidence, but it is not a
conforming configuration for this 32-node task because this task explicitly
requires the same 16-node source checkpoint.

A conforming 32-node continuation would need to combine `sfsgd_y` with:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=32
DILOCO_ISLAND_SIZE=8
RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

Current `train.py` rejects that shape before work begins:

```text
if loaded_outer_state is None and args.resume and args.diloco_outer_optimizer != 'avg':
    raise ValueError(
        f"checkpoint {args.resume} does not contain diloco_outer_state; "
        f"cannot coherently resume --diloco_outer_optimizer "
        f"{args.diloco_outer_optimizer}")
```

This is the same class of blocker that prevented the partial-average rung from
launching a valid non-`avg` continuation from the required avg source.

## Scheduler And Scope Evidence

No Slurm training job was submitted from this task. Requested training
node-hours from this task:

```text
0.000000
```

Actual training node-hours from this task:

```text
0.000000
```

Scheduler accounting check from the task window, using scheduler-local time:

```text
sacct -S 2026-06-25T05:36:00 -o JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,NNodes,Submit,Start,End -P
```

Result at decision time:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|Submit|Start|End
4899197|e97-4n-sfsgd-y-fixed-eval|PENDING|0:0|00:00:00|01:00:00|1|2026-06-25T05:37:17|Unknown|Unknown
4899198|e97-8n-sfsgd-y-fixed-eval|PENDING|0:0|00:00:00|01:00:00|1|2026-06-25T05:37:21|Unknown|Unknown
```

Those are one-node fixed-eval jobs from separate 4-node and 8-node
schedule-free tasks. They are not 32-node training jobs and are not source-
checkpoint continuation evidence for this gate.

Current queue check at decision time:

```text
JOBID|NAME|STATE|NODES|TIME|TIME_LIMIT|SUBMIT_TIME
4899198|e97-8n-sfsgd-y-fixed-eval|PENDING|1|0:00|1:00:00|2026-06-25T05:37:21
4899197|e97-4n-sfsgd-y-fixed-eval|PENDING|1|0:00|1:00:00|2026-06-25T05:37:17
```

## Gate Verdict

Schedule-free does not clear the 32-node same-source fixed-validation gate in
this task because no conforming candidate could be launched under the current
implementation constraints.

Classification for the 32-node ladder:

```text
deferred / no valid candidate
```

Do not count schedule-free as worse than K80 on model quality; no candidate was
evaluated. Do not count it as improving or clearing; no candidate checkpoint
exists. For synthesis, K80 remains the best clean 32-node same-source evidence
in this ladder, while schedule-free should remain a comparison arm until the
resume-state question is deliberately resolved.

## Scope Confirmations

- Read the K80 result before deciding: K80 improved over K40 but failed the
  gate with CE delta `+0.07406617` and BPB delta `+0.03081585`.
- Read the partial-average result before deciding: no job was submitted because
  the required non-`avg` resume shape is rejected from the avg source.
- Submitted no 32-node schedule-free training job.
- Requested and actual training node-hours from this task are both `0.000000`.
- Ran no fixed eval from this task because there is no candidate checkpoint.
- Submitted no 64+ job, no GDN2 job, no CMAES job, and no run with changed
  island size, learning rate, or source trajectory from this task.
- `run-64-node-e97` remains `open (PAUSED)`.
