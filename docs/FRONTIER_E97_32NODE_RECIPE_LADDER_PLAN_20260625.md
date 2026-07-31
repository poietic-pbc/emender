# Frontier E97 32-node recipe ladder plan (2026-06-25)

Task: `plan-e97-32-node`

## Decision

2026-06-25 policy update: this ladder is now baseline/control evidence for the
old node-island DDP topology. Future primary E97-MLP scaleout should use
`DILOCO_ISLAND_SIZE=1` GPU islands with no per-step DDP gradient averaging when
the model fits on one GPU. Do not copy this ladder's `DILOCO_ISLAND_SIZE=8`
default into new primary launch tasks; use it only for deliberate controls.

Run one cadence-only diagnostic next: 32 Frontier nodes, E97-MLP, same 16-node
source checkpoint, same island size 8, same avg outer, but `DILOCO_K=80`.

Do not resume `run-64-node-e97` yet. The 32-node K10 path failed twice on the
fixed gate, and K40 improved the train trajectory and fixed BPB delta but did
not clear the 16-node source comparison.

## Evidence basis

| Evidence | Result | Planning implication |
| --- | --- | --- |
| 16-node avg source | Operationally clean; final trailing loss `5.2531`; no 32/64-node, schedule-free, or GDN2 job submitted from that task | Valid source checkpoint and baseline gate |
| 32-node K10 original | Operationally clean but final trailing loss `5.8701`; fixed BPB delta `+0.07770465` vs 16-node source | K10 at 32 nodes is not green |
| 32-node K10 retry | Operationally clean but repeated failure; final trailing loss `5.8164`; fixed BPB delta `+0.07318369` | Regression is reproducible enough to block 64 nodes |
| 32-node K40 | Operationally clean; train loss largely repaired; final trailing loss `5.2822`; fixed BPB delta improved to `+0.05462847` but still worse than source | Cadence is the leading scale-control knob, but K40 is not sufficient |

The fixed tensor remains the row-matched comparator:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

## Ordered ladder by expected information per node-hour

1. **Cadence: K80 or nearby larger K.**
   This is the highest-information next node-hour because K40 changed one
   variable from the failed K10 runs and moved both train loss and fixed BPB in
   the right direction. K80 tests the obvious monotone continuation without
   introducing optimizer, LR, island, or schedule-free confounds. Use K80 as the
   next rung unless an audit immediately before launch finds an implementation
   reason to choose a nearby larger K instead.

2. **Merge strength: partial average via momentum outer with `beta=0` and
   `outer_lr<1`, only if K80 fails to clear.**
   `train.py` documents and implements the momentum outer update as
   `W_{r+1} = W_r + outer_lr * (outer_beta * outer_mom + mean(W_i) - W_r)`;
   with `outer_beta=0`, `outer_lr=0.5` is a half-step from the anchor toward the
   global mean on the first round. That directly tests whether full consensus is
   too strong. It should follow K80 because it changes outer-optimizer mode and
   state handling, so it is slightly less isolated than a cadence-only rung.

3. **Local optimizer schedule: resume LR reduction or short warmup.**
   The K10 failures could include a post-resume shock from using the same local
   learning rate after doubling from 16 to 32 nodes. This is a plausible
   low-cost diagnostic, but it is ranked below cadence and merge strength
   because the K40 result already points at global merge frequency/strength as
   the more direct lever. If used, change exactly one schedule feature, for
   example a short warmup after resume or a predeclared LR multiplier, and keep
   the fixed eval unchanged.

4. **Island topology: defer until cadence and merge-strength diagnostics finish.**
   Island size 8 remains the right default unit for the next run because it was
   used by the clean 16-node source, the failed 32-node K10 runs, and the
   improved 32-node K40 run. A smaller or larger island test changes the number
   of DDP groups and the local/global communication geometry, so it is best kept
   as a later topology arm if K80 and partial averaging do not clear the gate.

5. **Validation slice size: keep the current fixed tensor for gating each rung;
   expand only before declaring a recipe generally green.**
   The small 8-chunk fixed tensor is the only row-matched comparator across
   K10, retry, and K40. Use it for the immediate ladder. If K80 or a later rung
   is within tolerance or better on this tensor, run a larger fixed validation
   slice before authorizing 64 nodes or calling the recipe robust.

6. **Schedule-free outer: controlled later arm only.**
   Schedule-free is not the default first fix. Reserve it for the existing later
   downstream arm only if K80 and partial averaging do not clear the fixed gate,
   or if the partial-average audit shows schedule-free is the smallest direct
   test of the observed failure. Do not combine it with a simultaneous K, LR, or
   island-topology change.

## Exact first diagnostic and acceptance gate

First diagnostic to run next:

```text
32 nodes, E97-MLP, resume from the 16-node source checkpoint,
DILOCO_ISLAND_SIZE=8, DILOCO_OUTER_OPTIMIZER=avg,
DILOCO_OUTER_LR=1.0, DILOCO_OUTER_BETA=0.0,
DILOCO_EXPORT_BASIS=x, DILOCO_K=80,
same fixed validation tensor source-vs-K80 after training.
```

Acceptance gate for green:

- Training job is operationally clean: Slurm exit `0:0`, no NCCL/RCCL watchdog
  or collective mismatch, no non-finite/OOM/traceback/fatal signature, valid
  final checkpoint, valid `latest.pt`, bounded retention, and recorded DiLoCo
  merge count.
- Train-loss trajectory is not in the K10 failure class: final trailing loss is
  close to the 16-node/K40 class, not the `5.8-5.9` class, and the late window
  does not show the coherent K10 rise.
- Same fixed tensor source-vs-K80 eval is run. Green requires fixed BPB delta
  versus the 16-node source `<= +0.010` and fixed CE delta `<= +0.025` on this
  smoke tensor. A negative delta is green; `0.010 < BPB delta <= 0.05462847`
  is "improves but fails"; worse than K40 is a failed cadence direction.
- Even if K80 clears the small tensor, require a larger fixed validation slice
  before declaring the recipe robust enough to unblock `run-64-node-e97`.

## Scope confirmations

- No Slurm jobs were submitted from this planning task.
- `run-64-node-e97` remains `open (PAUSED)` and must stay paused until a
  32-node recipe clears the fixed gate and a larger fixed validation check is
  satisfactory.
- The 16/32-node K10/K40/K80 avg evidence in this ladder used
  `DILOCO_ISLAND_SIZE=8`, meaning one node island with 8-GPU per-step DDP inside
  each island. It is baseline/control evidence, not the primary no-DDP recipe.
- The primary E97-MLP scaleout hypothesis is now `DILOCO_ISLAND_SIZE=1`: one GPU
  per island, no within-island DDP gradient averaging, and only periodic
  DiLoCo/model averaging. Tensor/model parallelism remains separate and should
  only be considered if the model does not fit on one GPU.
- GDN2 and CMAES are out of scope for this E97-MLP 32-node scale-fix ladder.
- Schedule-free is not treated as the default first fix; it remains a controlled
  later arm after cadence and merge-strength diagnostics.
