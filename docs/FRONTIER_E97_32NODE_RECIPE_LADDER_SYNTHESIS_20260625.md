# Frontier E97 32-Node Recipe Ladder Synthesis

Date: 2026-06-25
Task: `synthesize-e97-32-node`

## Decision

2026-06-25 policy update: interpret this synthesis as node-island DDP
baseline/control evidence. The primary future E97-MLP scaleout hypothesis is
`DILOCO_ISLAND_SIZE=1`, one GPU per island, no per-step DDP gradient averaging,
and periodic DiLoCo/model averaging only. The `DILOCO_ISLAND_SIZE=8` recipes
below should not be used as the preferred default for new primary launch tasks.

No available 32-node E97-MLP recipe is green against the fixed gate. The current
best recipe is the clean K80 cadence-only avg-outer run: it repaired the train
loss behavior most strongly and reduced the fixed-validation regression, but it
still failed the green threshold versus the same 16-node source checkpoint.

Keep `run-64-node-e97` paused. Do not resume it automatically. Do not replace it
with a revised 64-node task yet, because there is not yet a 32-node recipe that
clears the fixed source-vs-candidate gate.

Recommended next actions, capped at two:

1. Run a narrow local LR-schedule diagnostic at 32 nodes from the same 16-node
   source, keeping K80, island size 8, avg outer, and the fixed eval tensor
   unchanged; change only a predeclared resume LR schedule feature such as a
   short post-resume warmup or LR multiplier.
2. In parallel or before spending more 32-node allocation, create a small
   implementation/design task for a stateless partial-average outer mode, or an
   explicit coherent non-`avg` outer-state initialization policy, so the
   partial-average and schedule-free arms can later be tested without changing
   source provenance.

## Common Source And Gate

All candidate rows below are compared against the same E97-MLP 16-node source:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

That source resolves to step `1328`, train checkpoint loss `5.2531`, fixed CE
`10.49609756`, and fixed BPB `4.36699062`.

The fixed validation tensor is the same row-matched smoke tensor used across
the ladder:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Green gate from the ladder plan:

- fixed CE delta versus source `<= +0.025`
- fixed BPB delta versus source `<= +0.010`
- operationally clean training and final checkpoint mechanics
- train-loss trajectory not in the repeated K10 failure class

## Recipe Ladder

Node-hours are listed as requested / actual top-level elapsed allocation where
available. Training and fixed-eval jobs are separated in the job column because
the fixed evals are one-node forward-only jobs and do not launch training.
All 16/32-node avg recipes below used `DILOCO_ISLAND_SIZE=8`: one Frontier node
per DiLoCo island, with 8-GPU per-step DDP inside each island. They are controls
for the old topology, not the primary no-DDP GPU-island path.

| Recipe | Config | Job ids | Node-hours | Train loss behavior | Fixed CE / BPB delta vs 16-node source | Verdict |
| --- | --- | --- | ---: | --- | --- | --- |
| 16-node source | 16 nodes, avg outer, K10, source checkpoint for all comparisons | train `4893977`; fixed eval row in `4894484` | train `8.000000 / 5.413333`; fixed eval shared in `4894484` | final trailing `5.2531` | `+0.00000000` CE / `+0.00000000` BPB | Baseline only |
| K10 original | 32 nodes, 32 islands x 8 GPUs, avg outer, `outer_lr=1.0`, `outer_beta=0.0`, export `x`, `DILOCO_K=10`, `SAVE_EVERY=10` | train `4894111`; fixed eval `4894484` | train `16.000000 / 10.862222`; fixed eval shared in `4894484` | Operationally clean, but final trailing `5.8701`; first-20 `5.4394`, last-20 `6.1958`; coherent late rise | `+0.18676377` CE / `+0.07770465` BPB | Red: operational smoke passed, quality failed |
| K10 retry | Same as K10 original; repeat from same 16-node source | train `4894206`; fixed eval `4894488` | train `16.000000 / 10.862222`; fixed eval `1.000000 / 0.016100` | Operationally clean, but repeated failure class; final trailing `5.8164`; first-20 `5.4486`, last-20 `6.1179` | `+0.17589760` CE / `+0.07318369` BPB | Red: reproducible K10 failure |
| K40 cadence | Same source and island topology, avg outer, export `x`, `DILOCO_K=40`, `SAVE_EVERY=40` | train `4894517`; fixed eval `4894795` | train+eval `17.000000 / 10.841945` | Operationally clean; train loss mostly repaired versus K10; final trailing `5.2822`; first-20 `5.2305`, last-20 `5.3872` | `+0.13129998` CE / `+0.05462847` BPB | Yellow/red: improves K10, fails fixed gate |
| K80 cadence | Same source and island topology, avg outer, export `x`, `DILOCO_K=80`, `SAVE_EVERY=80` | train `4899126`; fixed eval `4899139` | train+eval `17.000000 / 10.850278` | Operationally clean; best train-loss behavior in ladder; final trailing `5.1084`; first-20 `5.2264`, last-20 `5.1834` | `+0.07406617` CE / `+0.03081585` BPB | Best available, but not green |
| Partial-average | Intended K80 plus partial merge strength through `momentum`, `outer_lr=0.5`, `outer_beta=0.0`, export `x` | none | `0.000000 / 0.000000` | No candidate: conforming same-source run blocked because non-`avg` resume requires `diloco_outer_state` absent from avg source checkpoint | not evaluated | Deferred/no-op; not worse or better than K80 |
| Schedule-free outer | Considered `sfsgd_y`: schedule-free optimizer, `diloco_outer_optimizer=sfsgd`, export `y`, `outer_lr=1.0`, `outer_beta=0.1`; same 32-node source continuation required | none | `0.000000 / 0.000000` | No candidate: same non-`avg` outer-state resume guard blocks coherent continuation from the required avg source | not evaluated | Deferred/no-op; comparison arm remains untested for this gate |

## Operational Health

The launched 32-node candidates were operationally clean. K10 original, K10
retry, K40, and K80 completed with Slurm exit `0:0`; formed the intended
256-rank, 32-island job; wrote final checkpoints; advanced `latest.pt`; and had
no fatal signatures such as traceback, non-finite loss, OOM, NCCL/RCCL watchdog,
collective mismatch, timeout, cancellation, or fatal signal.

Partial-average and schedule-free did not run. Those no-ops were correct for
this ladder because both would have required changing checkpoint semantics,
bypassing the resume guard, or using a different source trajectory.

## Train-Loss Behavior

K10 is the clear failure class: both K10 runs drifted from roughly `5.44` early
to above `6.11` late and ended with final trailing losses `5.8701` and
`5.8164`. K40 repaired most of that train-loss symptom, ending at `5.2822`.
K80 is best on train loss, ending at `5.1084` with a late-window mean
`5.1834`.

Train loss alone is not sufficient to call a recipe green. K80 looks best by
training dynamics, but the fixed-validation row still regresses versus the same
16-node source.

## Fixed-Validation Behavior

The fixed smoke tensor shows a monotone improvement as cadence is relaxed from
K10 to K40 to K80, but not enough to clear the gate:

| Recipe | Fixed CE delta | Fixed BPB delta | Gate status |
| --- | ---: | ---: | --- |
| K10 original | `+0.18676377` | `+0.07770465` | Fail |
| K10 retry | `+0.17589760` | `+0.07318369` | Fail |
| K40 | `+0.13129998` | `+0.05462847` | Fail |
| K80 | `+0.07406617` | `+0.03081585` | Fail |

K80 is the current best 32-node recipe by fixed BPB delta and train-loss
behavior, but it is not green because both CE and BPB deltas exceed the
predeclared thresholds.

## Allocation Cost

The ladder spent bounded allocation on four 32-node training jobs and
forward-only fixed evals. K10 original and retry each requested `16.000000`
training node-hours and used `10.862222` actual training node-hours. K40 and
K80 each requested `17.000000` train+eval node-hours and used about `10.84` to
`10.85` actual train+eval node-hours. The partial-average and schedule-free
rungs spent zero 32-node training node-hours because they explicitly no-oped.

The next action should therefore stay narrow. The data already says cadence is
the strongest observed control knob, but cadence-only K80 still leaves a fixed
validation gap; broad 64-node exploration would multiply cost before a
32-node recipe is green.

## Scope Confirmations

- No Slurm jobs were submitted from this synthesis task.
- `run-64-node-e97` should remain paused. It should not be replaced by a
  revised 64-node task until a 32-node recipe is green on the fixed gate and,
  preferably, confirmed on a larger fixed slice.
- `run-64-node-e97` must not be resumed using the old `DILOCO_ISLAND_SIZE=8`
  DDP-shaped default. Any future 64-node work needs a fresh task that labels
  `DILOCO_ISLAND_SIZE=1` as the primary no-DDP path or explicitly declares a
  node-island DDP control.
- Tensor/model parallelism was not introduced and remains out of scope unless a
  future E97 variant does not fit on one GPU.
- This synthesis is strictly E97-MLP scope. GDN2 remains a separate control
  track and is not part of this 32-node scale-fix ladder.
- CMAES, GDN2, 64+ node training, and uncontrolled source/checkpoint changes
  were not introduced.
