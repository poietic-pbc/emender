# Frontier E97 DiLoCo convergence expectations

Date: 2026-06-25

Task: `research-e97-diloco`

## Executive summary

E97-MLP remains the primary target. The current Frontier evidence says the
32-node same-source avg-outer ladder is not green: K10 reproducibly fails, K40
improves but remains worse than the 16-node source on the fixed gate, and K80 is
the best available same-source 32-node recipe but still fails the fixed CE/BPB
thresholds. That pattern points to scale-control dynamics rather than a basic
Slurm, checkpoint, or finalization failure.

The literature supports DiLoCo/local-SGD as a valid low-communication training
family, but it does not imply that every early 32-island E97 continuation should
look monotone on a tiny fixed validation slice. The key knobs remain local
period K, outer merge strength and momentum/schedule-free state, post-scale
learning-rate warmup, island count/topology, data ordering, and a stronger
validation gate.

The two highest-value next actions are:

1. A narrow same-source 32-node K80 avg-outer diagnostic that changes only the
   post-resume LR schedule, such as a short warmup or LR multiplier, and keeps
   the fixed tensor unchanged. This is an experiment recommendation only; it
   requires a separate human-authorized task before any Slurm launch.
2. A code/design task for coherent non-`avg` outer-state initialization or a
   stateless partial-average outer mode, so merge strength and schedule-free
   continuation can be tested from the same 16-node source without bypassing the
   current resume guard.

No Slurm jobs were submitted for this research task. I did not run `sbatch`.
`run-64-node-e97` remains `open (PAUSED)` as confirmed by `wg show
run-64-node-e97` during this task.

## Literature-backed expectations

| Question | Literature-backed expectation | Source |
| --- | --- | --- |
| What is DiLoCo? | DiLoCo is a federated/local-SGD-style method for language-model training over islands, with many inner steps, AdamW inner updates, and a momentum-style outer optimizer. The DiLoCo paper reports 8-worker C4 training comparable to fully synchronous optimization with much lower communication. | Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models", arXiv:2311.08105, abstract lines 38-40: https://arxiv.org/abs/2311.08105 |
| Is local SGD expected to converge? | Local SGD can match mini-batch SGD convergence rates under stated assumptions while reducing communication, but the claims are asymptotic/theoretical and depend on problem assumptions, stepsize, worker count, and synchronization frequency. | Stich, "Local SGD Converges Fast and Communicates Little", arXiv:1805.09767, abstract lines 38-42: https://arxiv.org/abs/1805.09767 |
| Does heterogeneity matter? | Local-SGD theory distinguishes identical and heterogeneous data regimes; empirical/theoretical work reports severe impact from data heterogeneity and uses a variance notion specific to local SGD with different data. | Khaled, Mishchenko, Richtarik, "Tighter Theory for Local SGD on Identical and Heterogeneous Data", PMLR AISTATS 2020 abstract: https://proceedings.mlr.press/v108/bayoumi20a.html |
| Should larger from-scratch models show slower early convergence? | Not as a universal loss-vs-token statement. Scaling-law work reports larger language models are more sample-efficient and that loss follows predictable trends with model size, data, and compute. But larger models can be slower in wall-clock per optimizer step, more sensitive to LR/warmup, and may have noisier early fixed-slice measurements. The safe claim is metric-dependent: wall-clock convergence can be slower, while loss/token can be equal or better once optimization is tuned. | Kaplan et al., "Scaling Laws for Neural Language Models", arXiv:2001.08361, abstract lines 42-43: https://arxiv.org/abs/2001.08361 |

Practical reading for E97-MLP:

- Larger from-scratch E97 should not be presumed "bad" because early loss is
  noisier or because a small fixed slice is high variance. On a loss-vs-token
  horizon, a larger tuned model may be more sample-efficient. On a wall-clock,
  short-smoke, or just-after-scale-change horizon, it can look slower because
  each step is heavier and the optimizer may be in a transient regime.
- Averaging many partially independent random trajectories can hurt early
  structure formation when workers have not yet entered the same basin, local
  data order differs enough to create high between-island variance, the merge is
  too frequent or too strong, or the outer optimizer state is incoherent. In
  that regime, the global average can erase useful island-specific early
  features or inject repeated "consensus shocks." This is an inference from
  local-SGD/DiLoCo mechanics plus the heterogeneity literature, not a proven
  E97-specific theorem.
- K is not simply "larger is always better." Small K tracks synchronous training
  more closely but can over-constrain emerging structure; large K reduces
  communication and consensus shocks but increases island drift. The optimal K
  depends on LR, batch/data order, outer merge strength, worker count, and the
  training horizon.

## Frontier observations from this repo and WG graph

These are repo/WG empirical observations, not literature claims.

| Evidence | What happened | Implication |
| --- | --- | --- |
| 16-node avg source | The clean 16-node avg source checkpoint used for the 32-node ladder is step `1328`, train checkpoint loss `5.2531`, fixed CE `10.49609756`, fixed BPB `4.36699062`. See `docs/FRONTIER_E97_32NODE_RECIPE_LADDER_SYNTHESIS_20260625.md`. | This is the baseline for same-source 32-node continuation, not a universal E97 quality number. |
| 32-node K10 original/retry | Both K10 32-node avg jobs were operationally clean but failed quality: final trailing losses `5.8701` and `5.8164`, fixed BPB deltas `+0.07770465` and `+0.07318369`. See `docs/FRONTIER_E97_32NODE_FIXED_EVAL_20260624.md` and the ladder synthesis. | The K10 failure class is reproducible and not only rank-0 train-loss logging noise. |
| 32-node K40 | K40 repaired much of the train-loss symptom: final trailing `5.2822`, fixed BPB delta improved to `+0.05462847`, but fixed eval still failed the source comparison. See `docs/FRONTIER_E97_32NODE_K40_DIAGNOSTIC_20260624.md`. | Merge cadence is a real scale-control variable. K40 is better than K10 but not green. |
| 32-node K80 | K80 is the current best same-source avg-outer row: final trailing `5.1084`, fixed BPB delta `+0.03081585`, fixed CE delta `+0.07406617`; still above the green thresholds of `+0.010` BPB and `+0.025` CE. See `docs/FRONTIER_E97_32NODE_K80_DIAGNOSTIC_20260625.md`. | Relaxing cadence monotonically improved the fixed-smoke deltas from K10 to K40 to K80, but cadence alone has not cleared the gate. |
| Partial-average blocker | No 32-node partial-average job was submitted. The intended half-average formula exists through the `momentum` outer path, but resuming non-`avg` from the avg source lacks `diloco_outer_state` and is rejected. See `docs/FRONTIER_E97_32NODE_PARTIAL_AVG_DECISION_20260625.md`. | Merge strength is relevant but currently blocked by state/provenance semantics, not disproven empirically. |
| Same-source schedule-free blocker | No 32-node same-source schedule-free continuation was submitted. `sfsgd_y` is the plausible schedule-free arm, but current code rejects resuming non-`avg` outer optimizer from the required avg checkpoint without `diloco_outer_state`. See `docs/FRONTIER_E97_32NODE_SCHEDULEFREE_DECISION_20260625.md`. | Schedule-free has not been tested as a same-source 32-node continuation. It should not be counted better or worse than K80 for that gate. |
| From-scratch schedule-free 4/8 | The from-scratch 4-node and 8-node `sfsgd_y` probes completed cleanly, had finite/productive short-window loss, retained `diloco_outer_state`, and produced close fixed rows: 4-node CE/BPB `4.85631931`/`2.02051293`, 8-node `4.85920626`/`2.02171407`. See `docs/FRONTIER_E97_SCHEDULEFREE_4N_8N_SYNTHESIS_20260625.md`. | This supports operational health of from-scratch `sfsgd_y` at 4/8 nodes, not a same-trajectory win over avg continuation. |
| From-scratch schedule-free 16 | WG task `run-e97-schedule-3` was pending evaluation while this report was written. Its task log and unmerged artifact report one clean 16-node `sfsgd_y` from-scratch job `4899221`, fixed eval job `4899229`, final step `1119`, final loss `4.8905`, fixed CE/BPB `4.84385067`/`2.01532525`, and no 32/64-node submission. | If accepted by evaluation/merge, this strengthens the from-scratch schedule-free scale-health track through 16 nodes. It still does not answer the same-source 32-node avg-continuation regression. |

## Hypotheses and inferences not yet validated

| Hypothesis | Why it is plausible | What would validate or falsify it |
| --- | --- | --- |
| Frequent 32-island K10 averaging disrupts early E97 structure formation. | K10 failed twice; K40 and K80 improved monotonically on fixed BPB deltas and train-loss windows while keeping source/islands/avg outer fixed. | A controlled cadence bracket or merge-geometry metrics showing reduced cross-island shock with larger K; a K80+LR-warmup run clearing fixed eval would support it. |
| K80 still fails because the LR schedule is too abrupt after scale change, not because K is intrinsically insufficient. | K80 train loss is better than source/K40 but fixed eval remains worse, suggesting optimization transient or validation mismatch rather than systems failure. | Same-source K80 with only a short post-resume warmup or LR multiplier. Keep fixed tensor/source unchanged. |
| Partial averaging would reduce consensus shocks better than full averaging. | The implemented momentum formula with `outer_beta=0` and `outer_lr<1` would move partway from anchor to island mean, which directly targets merge strength. | Requires stateless partial-average mode or safe initialization of non-`avg` state, then a same-source 32-node comparison. |
| Schedule-free `sfsgd_y` may scale better from scratch than avg continuation. | 4/8, and pending 16-node, from-scratch probes are operationally clean and have close fixed rows. | A separately authorized 32-node from-scratch schedule-free diagnostic, plus a matched from-scratch avg/control if the question is model-quality rather than operational health. |
| The tiny fixed slice overstates or understates quality gaps. | The fixed eval doc explicitly notes absolute CE/BPB values are near-random on the very small smoke slice; only row-matched deltas are currently useful. | Larger fixed tensor, multiple shards/domains, confidence intervals or bootstrap CIs, and repeated source/candidate scoring under identical invocation. |

## Knobs that matter

| Knob | Expected effect | Current status |
| --- | --- | --- |
| Local steps K | Controls communication frequency and island drift. Too small can behave like repeated full-consensus shocks; too large can let islands diverge. | Strongest observed same-source knob: K10 red, K40 better, K80 best but not green. |
| Outer LR / merge strength | Controls how far the outer anchor moves toward the island mean. | Not tested same-source. `avg` ignores `outer_lr<1`; partial averaging requires `momentum` or a new stateless mode. |
| Outer momentum / schedule-free outer | Can smooth or accelerate outer dynamics but requires coherent state. | Same-source non-`avg` continuation blocked by missing `diloco_outer_state`; from-scratch `sfsgd_y` is operationally promising. |
| Warmup after scale change | Can reduce optimizer shock when node/island count changes at resume. | High-value next experiment because it changes one local schedule variable while keeping K80/source/fixed tensor constant. |
| Island count/topology | More islands increase the number of partially independent trajectories being merged; topology affects communication and variance. | 32 islands x 8 GPUs is where avg K10/K40/K80 remains not green. Do not jump to 64 nodes before 32-node green. |
| Global batch and data order | Affects gradient noise, data heterogeneity across islands, and validation comparability. | Current evidence uses row-matched fixed eval for 32-node ladder, but data-order/island-specific variance is not fully characterized. |
| Validation split quality | Determines whether fixed-eval gate detects real quality or smoke-slice noise. | Current 8-chunk/16,384-token tensor is useful for row-matched smoke, fragile for absolute quality. Needs larger/multi-shard gate before major scale decisions. |

## From-scratch versus pretrained or continuation

Same-source continuation and from-scratch scale health answer different
questions.

The 32-node avg ladder is a continuation question: can the same E97-MLP
checkpoint that was clean at 16 nodes be scaled to 32 nodes without quality
regression under a fixed row-matched eval gate? The answer is currently no:
K80 is best but still not green.

The schedule-free 4/8/16 probes are from-scratch operational-health questions:
does `sfsgd_y` launch, merge, checkpoint, retain outer state, and score
consistently as island count grows? The answer is yes through merged 4/8
evidence and likely yes through the pending 16-node WG result, but those runs do
not compare directly to the 16-node avg source because they start from a
different trajectory and optimizer state.

A pretrained continuation would be a third question. It should not inherit
claims from either track until checkpoint path, E97-MLP geometry, tokenizer,
context length, dtype, tensor names/shapes, and optimizer/outer-state semantics
are verified. If a pretrained checkpoint lacks compatible non-`avg` outer state,
schedule-free or momentum continuation needs an explicit bootstrap policy.

## Fixed eval/source comparability is useful but fragile

The fixed eval gate is better than rank-0 train loss because it scores source
and candidate checkpoints through the same forward-only evaluator on the same
saved tensor. It correctly caught the K10 regression and showed monotone
improvement through K40/K80.

It is still fragile:

- The current fixed tensor is only eight 2048-token chunks, 16,384 scored
  tokens. The fixed-eval report warns that absolute CE/BPB values are near
  random on this smoke slice.
- It is a source-vs-candidate continuation gate. It is not valid for declaring
  from-scratch schedule-free better than an older avg checkpoint.
- Schedule-free basis handling matters. Saved-basis eval is appropriate for the
  current smoke comparisons; changing to y-swap/train-basis rows would change
  the measured object.
- A single fixed slice can be sensitive to data ordering, tokenizer artifacts,
  and local stream overlap or mismatch.

Improve the gate by:

- using a larger fixed tensor or several predeclared fixed shards;
- reporting CE/BPB deltas with bootstrap confidence intervals across chunks;
- keeping row-matched source/candidate invocation, checkpoint basis, tokenizer,
  batch size, and code commit fixed;
- adding a same-source validation criterion for continuation tasks and a
  separate from-scratch matched-control criterion for schedule-free scale-health
  tasks;
- retaining the tiny tensor as a cheap smoke gate but not as the final quality
  arbiter for 64-node or extended runs.

## Recommended next actions

1. Same-source experiment design: prepare, but do not launch from this task, one
   32-node E97-MLP K80 avg diagnostic that changes only post-resume LR behavior
   such as a short warmup or multiplier. Keep the same 16-node source checkpoint,
   island size 8, avg outer, export basis `x`, fixed eval tensor, and green
   thresholds. This directly tests whether K80's remaining fixed-eval gap is an
   optimizer transient after scale change.
2. Code/design change: implement or specify a fail-closed way to test merge
   strength and non-`avg` outer continuation from the same loaded model weights.
   Acceptable shapes are a stateless partial-average outer mode, or an explicit
   bootstrap policy for momentum/schedule-free outer state that preserves loaded
   model tensors and records metadata. This unlocks the partial-average and
   same-source schedule-free questions without silently changing provenance.

GDN2 remains only a control/comparator for later interpretation. It is not the
main target of this E97-MLP scale-out diagnosis.
