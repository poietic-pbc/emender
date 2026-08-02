# HANDOFF — Emender / E97 architecture research → Frontier scaleout

**Last updated: 2026-06-20** (supersedes the Jun-15 revision; the DiLoCo, recipe,
scaleout-protocol, and Frontier-operations sections were rewritten against the
Jun-16…Jun-20 committed results — see the dated deltas in §3 and the CLAIM→EVIDENCE
fact-check in §7).

**Audience:** a fresh research + engineering team with **zero prior context** —
new models, new systems, no knowledge of this project's tooling or history. This
document is **self-contained**: it defines the architecture from scratch, states
every validated finding with committed provenance (file paths + git commits), and
lays out the scaleout plan, the artifact map, the methodology guardrails, and the
open questions. Every quantitative claim is tied to a committed artifact; the
fact-check table in §7 maps each number to its source and flags anything external.

> **One-paragraph orientation.** This project studies a recurrent ("token-mixer")
> cell called the **Emender** (lead instance tag **E97**), a *gated-delta
> linear/nonlinear recurrence* that generalizes the linear "Gated DeltaNet-2"
> (GDN-2) baseline, paired with a standard SwiGLU MLP ("channel-mixer") — the
> ordinary modern block, not a frankenstein. Three measured results, fairly
> compared at 1.3B parameters with the MLP held constant: (1) **language-model
> loss is a convergent-loss null** — `emender-mlp` and `gdn2-mlp` reach the **same
> held-out BPB band** at matched compute, so LM loss does **not** distinguish the
> architectures (this is a *finding*, not "inconclusive": the distinguisher must be
> capability, not loss); (2) **capability shows a PROVED, causally-isolated temporal
> class separation** — per-step nonlinearity-in-time lets the nonlinear cell
> extrapolate iterated maps where the byte-identical linear cell memorizes-then-
> collapses; the separation appears **exactly where the theory requires it and
> nowhere it does not** (two-sided confirmed); (3) **throughput is a tie** at 1.3B.
> The forward experiment the scaleout enables: at ≤2B tokens **both arms are near
> chance (pre-emergence)**, so whether the proved expressivity separation surfaces
> as *measurable real-LM* capability is the at-scale test — an **extension** of
> conclusive work. The plan: a local **equal-FLOPs E97-vs-GDN2 convergence race**
> on license-clean **commapile**, then a Frontier-class HPC scaleout via
> hierarchical plain-average DiLoCo. The DiLoCo recipe is **locked** (plain
> averaging, `outer_beta=0`) and the merge bug that corrupted Schedule-Free is
> **fixed** (§3).

---

## Table of contents

1. [Architecture: the Emender / E97 cell + MLP, from scratch](#1-architecture)
2. [Validated findings (with provenance)](#2-validated-findings)
3. [Scaleout + frontier plan (recipe, primitives, gaps, phases)](#3-scaleout--frontier)
4. [Artifact map: where everything lives](#4-artifact-map)
5. [Methodology guardrails (avoiding the documented failure modes)](#5-methodology-guardrails)
6. [Open questions / immediate next steps](#6-open-questions--immediate-next)
7. [CLAIM → EVIDENCE fact-check](#7-claim--evidence-fact-check)

---

## 1. Architecture

### 1.1 The family: a dynamics taxonomy of head-types

The Emender is a **mixture of recurrent "head-types"** inside a single layer. Each
head-type is one per-token state-update map of the general form

```
S_t = f( A_t · S_{t-1} + B_t )
```

where `S_t` is the head's matrix state, `A_t` is an input-dependent state transition,
`B_t` is the input-driven write, and `f` is an optional pointwise nonlinearity. The
taxonomy is read from the **dynamics**, on **two independent axes** (canonical
reference: `paper/review/EMENDER_TAXONOMY.md`; paper anchor `paper/main.typ`,
subsection *"A dynamics taxonomy of head-types: eigenvalue placement × saturation"*):

**Axis 1 — eigenvalue placement in the complex unit disk** (where the along-key
eigenvalue of `A_t` sits):

| Placement | Name | Dynamics |
|---|---|---|
| real, positive | `decay` | stored value fades toward zero (vanilla GDN) |
| real, negative | `reflect` | sign flips each step — the tracking lever (lets a recurrence track non-solvable group structure such as S₅) |
| complex pair | `rot` | rotation / oscillation of the stored value |

**Axis 2 — state map: linear vs saturated:**

| State map | Suffix | Canonical fn | Dynamics |
|---|---|---|---|
| linear | *(none)* | identity | unbounded along eigendirections |
| + saturation | `-nonlin` | `hardtanh` (or smooth `tanh`) | latches a driven slot → finite-state regime |

The named cells form a grid: `decay`, `reflect`, `nonlin` (= **E97**, the saturated
delta-correcting head), `rot` (= complex-eigenvalue head), `rot-nonlin` (reserved).

**Key structural claim — GDN-2 is a special case of the Emender.** Gated DeltaNet-2,
the linear-recurrent baseline, is *exactly* the restriction of the Emender to the
**real diameter of the unit disk with no saturation**: the `{decay, reflect} × linear`
sub-grid. The Emender generalizes GDN-2 along two axes: real diameter → full unit disk
(adds `rot`), and linear state → saturated state (adds `nonlin`). The Emender layer is
the within-layer pool over this grid; **GDN-2 is its linear, real-axis corner.** This
matters for fair comparison: any Emender-vs-GDN-2 result is a comparison of a model to
its own restriction.

> **Naming note.** The experiment tags **E97 / E88 / E98 / E99** are retired as
> architecture names and kept *only* as historical run identifiers. E97 = the `nonlin`
> head (real × saturation). E88 is the original 1.3B instance tag. Use the dynamics
> names (`decay`/`reflect`/`nonlin`/`rot`) in new work.

### 1.2 The E97 cell — split-edit gated-delta recurrence with per-step nonlinearity

The lead head-type is the **E97 split-edit delta** cell. It is a gated-delta-rule
recurrence with *separate* gates on the key (erase/read) axis and the value (write)
axis, plus a per-step bounded nonlinearity on the state. Per (batch, head), with state
`S` of shape `[N, V]` (canonical source: kernel docstring in
`ndm/triton/e97_chunked.py`; cell in `ndm/models/e88_fla_hybrid.py`,
`use_split_edit=True`):

```
read_key_t  = e_t ⊙ k_t                      # erase/read gate on the key axis,  [N]
write_val_t = w_t ⊙ v_t                      # value-write gate,                 [V]
delta_t     = write_val_t − S_{t-1}^T read_key_t          # delta correction,    [V]
S_t         = f( decay_t · S_{t-1} + k_t · delta_t^T )    # state update,     [N,V]
out_t       = S_t^T q_t                                   # readout,             [V]
```

- **`decay_t`** is a Mamba2-style input-dependent exponential decay
  (`g = −exp(A_log) · softplus(a_proj(x) + dt_bias)`), giving the along-key eigenvalue.
- **Short depthwise convolutions + SiLU** are applied to `k, v, q` after projection;
  `q, k` are L2-normalized (FLA-GatedDeltaNet design elements).
- **The per-step nonlinearity `f`** is the saturation axis. The paper writes the cell
  as `S ← tanh(d·S + k·(silu(v) − S^T k)^T)`. Three ablation variants isolate the
  pieces, all byte-identical code except for the marked change:
  - **`emender` / `nonlin` / E97** (lead): `f = tanh`, delta-correcting. This is the
    capability-retaining cell.
  - **`e97-linear`**: `f = identity` (drop the per-step nonlinearity). Isolates the
    causal effect of the nonlinearity — same kernel, only `tanh` vs `identity` differs.
  - **`e97-raw`**: drop the delta correction, write the gated value directly.

When the per-step nonlinearity is **on**, the recurrence is genuinely nonlinear in time
and must run a sequential scan; when **off** (`e97-linear`), the recurrence is an
*affine asymmetric gated-delta* form `S_t = (decay_t·I − k_t·read_key_t^T)·S_{t-1} +
k_t·write_val_t^T`, which admits a **chunked-parallel** kernel (intra-chunk matmuls,
recurrent state threaded across chunks — the same trick FLA uses for GDN-2). The fused
Triton kernel `ndm/triton/e97_chunked.py` implements the chunked form; the nonlinear
sequential fused path routes through `ndm/triton/e88_triton_optimized.py` into the
shared E88-derived Triton forward/backward engine with `SPLIT_EDIT=True`. See
[`E97_E88_KERNEL_NAMING_CLARIFICATION_20260802.md`](E97_E88_KERNEL_NAMING_CLARIFICATION_20260802.md)
for the historical naming issue and compatibility-preserving cleanup plan.

### 1.3 The lead configuration: `emender-mlp`

The seed config handed to scaleout:

```
emender-mlp = E97 split-edit DELTA (e88_raw_write=0, delta-correcting; NOT raw-write)
            + bias-free LLaMA-style SwiGLU MLP  (w3(silu(w1 x) · w2 x))
geometry:   dim 1792 · n_heads 216 · n_state 32 · depth 11 · mlp_ratio 2.2623
params:     1,286,589,072  (measured, byte-identical to its CMA search)
```

The SwiGLU MLP is added by `MixerMLPWrapper` / `SwiGLUMLP` in
`ndm/models/ladder_lm.py` (post-mixer RMSNorm + MLP), making it a fair counterpart of
the `gdn2-mlp` control (GDN-2 mixer + the *same* SwiGLU MLP). **The MLP matters and
the comparison is MLP-controlled by design:** the MLP (channel-mixer) carries LM
quality; the recurrence (token-mixer) carries the expressivity claim. The MLP was
worth ~0.42 search-loss to GDN-2, so the canonical fair fight is **MLP-vs-MLP**
(`emender-mlp` vs `gdn2-mlp`), never naked-cell-vs-MLP (see §5).

The primary control:

```
gdn2-mlp = GDN-2 mixer (the linear {decay,reflect} corner) + SwiGLU MLP
geometry: dim 2176 · n_heads 30 · depth 12 · mlp_ratio 3.2587 · use_conv
params:   1,286,713,448  (measured)
```

Both configs are built **byte-identically** to their CMA searches via
`scripts/cmaes_search_v2.build_train_command` → `train.py` (`--level E97` / `--level
gdn2-mlp`). All param counts verified against each source's recorded `actual_params`.

---

## 2. Validated findings

All findings below are from committed, measured artifacts. The calibration is
deliberate: the project's own post-mortem (`docs/RESEARCH_ASTRAY_POSTMORTEM.md`)
documents ~10 days of comparisons that were quietly rigged *against* the Emender and
had to be corrected by the PI. The discipline is to read the **primary measured metric
head-to-head**, to put the **burden of proof on NULLs** (no stop verdict without a cleared
confound audit), and to **not hedge a proved result**.

### 2.1 LM loss → the CONVERGENT-LOSS NULL is a real FINDING

**Provenance:** `experiments/lb_compare_20260613/{LEADERBOARD,REPRODUCTION}.md`
(commit `8acd929`, corrections `ece1b16`); paper anchor `paper/main.typ`
("same held-out bpb band", lines ~1281/1286/1309). Apples-to-apples: all 5 CMA-best
1.3B models at their *own* found geometry, same protocol — pile.txt seed42, 15-min
train budget (matching the CMA search), bf16 uniform + fused kernels, p50k_base, ctx
2048, schedule-free AdamW. Held-out = ONE fixed disjoint pile.txt-tail slice (131,072
scored tokens, byte-identical for every model). BPB = (CE_nats/ln2)/3.878 bytes/token.

| `emender-mlp` vs `gdn2-mlp` | `emender-mlp` | `gdn2-mlp` | winner |
|---|---:|---:|---|
| CMA search avg-loss | **5.8606** | 5.8949 | emender-mlp (−0.034) |
| held-out BPB, **non-avg** (primary basis) | **2.0911** | 2.1013 | emender-mlp (−0.010) |
| held-out BPB, averaged (inferior basis) | 2.1783 | 2.1550 | gdn2-mlp (+0.023) |

**This is a FINDING, not "inconclusive."** At matched compute, many distinct cells
converge to the **same held-out BPB band** — the architectures are indistinguishable
*on LM loss*. That is the **convergent-loss null**, and it is the load-bearing reason
the project pivoted to capability: **LM loss does not distinguish the architectures, so
the distinguisher must be capability/expressivity** (§2.2). It is robust across
architecture, optimization, and scale (consistent with `emender-real-1p3b`,
`emender-cap-sweep`, `opt-1p3b`). Within that band, `emender-mlp` is ahead on **both**
primary metrics (search avg-loss −0.034; non-averaged held-out BPB −0.010), losing only
on the schedule-free *averaged-weights* basis the run itself flags as the inferior /
artifact basis (at a 15-min budget the polyak average is uniformly worse than the final
weights, by an architecture-dependent margin up to ~0.70 BPB for mixer-only cells). The
±0.088 single-seed / 15-min noise band means the honest one-line statement is:
**`emender-mlp` ties-or-leads on LM loss and is never clearly worse — and LM loss is a
convergent null either way.**

> ⚠️ **Provenance caveat — read the corrections, not the auto-verdict.** The
> auto-generated VERDICT block inside `LEADERBOARD.md` says the emender is a clean loss and "gdn2-mlp best
> all-around." That verdict is **superseded** by the CORRECTIONS at the top of both
> files (and by `RESEARCH_ASTRAY_POSTMORTEM.md` Addendum): it (a) mislabeled
> `emender-mlp` as raw-write when it is the delta cell, (b) leaned on the averaged
> (artifact) ordering, and (c) used a grok-suppressed separator battery (see §2.2). The
> corrected reading is the table above. **Follow-up owed:** multi-seed BPB to tighten
> the margin (the null is the headline; the within-band lead is secondary).

### 2.2 Capability → a PROVED, causally-isolated, two-sided temporal class separation

**Provenance:** `experiments/grok_symmetric_width/{RESULTS,CONFIRM}.md` (commits
`d489955`, `ce7ff39`). This is the load-bearing capability result. It is **proved and
two-sided** — the separation appears exactly where the theory requires it and is absent
exactly where the theory says it should be. **This is not "scoped" or "narrow": the
task-family domain IS the theorem's domain, and the negatives CONFIRM the theory.**

**The positive (proved, seed-robust, mechanistically isolated).** On
`modular_quadratic` (`x_t = (x_{t-1}² + c_t) mod p`, per-position supervision), trained
at sequence length T=128 and evaluated by **length-extrapolation** out to T=4096:

- the **nonlinear** cell `e97` is **length-invariant** (e.g. p=256/dim1024: 0.982 @
  T=128 → 0.981 @ T=4096);
- both **linear-state** arms (`e97-linear` and `gdn2`) **memorize the train length and
  collapse** toward baseline at far T, **even at maximum width (dim 1024)** (e.g. same
  cell: e97-lin 0.619, gdn2 0.560 @ T=4096);
- the gap is **positive in every decisive cell** and **grows with the modulus p**; the
  **8-seed confirmation run** (`grok_symmetric_width/CONFIRM.md`, commit `ce7ff39`) puts
  e97−gdn2 at **+0.17 … +0.32 @ T=4096** (the 4-seed symmetric-width sweep in `RESULTS.md`
  shows even larger decisive-cell gaps, +0.26 … +0.42);
- **`e97` vs `e97-linear` is byte-identical code on the same fused kernel — the only
  difference is the per-step state map (`tanh` vs `identity`).** This **causally
  isolates** the **per-step nonlinearity-in-time** as the lever — not capacity, depth,
  or the specific linear architecture.

This corrects a predecessor false-negative (`grok_highp_temporal` concluded "width
closes the gap → it's capacity") which (a) widened only the linear arms and (b) scored
at the *train* length — capacity buying memorization, which length-extrapolation defeats.
The symmetric-width control (all arms widened, scored at far T) flips it.

**The negative side CONFIRMS the theory (it is not a limitation).** Tested on two
*other* task families (`grok_symmetric_width/CONFIRM.md`), the signature is **correctly
absent**:

- **`iterated_nonlinear_map`** (logistic `h_t = a_t·h_{t-1}(1−h_{t-1})`, binned): 0/24
  grok; all arms plateau flat across T. The map is **contractive (fading memory)** — no
  long-memory to memorize-then-fail, so the per-step nonlinearity is correctly not
  load-bearing. No separation **as predicted**.
- **`anbncn_viability`** (a^n b^n c^n, count comparisons): 24/24 fit; all arms decay
  *together* with length; the e97−gdn2 gap is tiny and **sign-flipping**. Counting is
  additive accumulation where linear cumulative state extrapolates *as well as* bounded
  nonlinear state — **as predicted**.

**Mechanism — the separation requires BOTH conditions the theory names, which only
`modular_quadratic` satisfies:** (1) a per-step state-nonlinearity the linear arm
**cannot represent** (x² mod p is non-invertible; counting and a fading map are not),
**and** (2) **non-contractive, full-precision long memory** so the failure-to-represent
*compounds* with length. Remove either and the separation vanishes — which is exactly
what the two non-replications measure. **Two-sided confirmation of a stated mechanism is
the strongest form a small-scale capability claim can take.**

> **Bottom line:** a **proved, mechanism-specific, causally-isolated** class
> separation, two-sided confirmed (8-seed positive on `modular_quadratic`, predicted-
> negative on the two families that lack a required condition). Do **not** deflate this
> to "scoped/narrow" — the domain is the theorem's domain. The one thing it is *not* is
> a universal "nonlinear-in-time always beats linear" law (the two negatives forbid
> that overclaim). Note: the **separator battery inside `lb_compare`** is
> **grok-suppressed** (LR pinned, no weight-decay sweep, short training) and includes
> *bounded* finite-state counting where linear-state is *expected* to win — so that
> battery neither tests nor refutes the claim.

### 2.3 Throughput → TIE at 1.3B (the grok-scale speedup did NOT survive to scale)

**Provenance:** `experiments/preflight_100b/RESULTS.md` (commit `0e914a3`); grok-scale
microbenchmark in `experiments/grok_symmetric_width/`.

At the literal `emender-mlp` geometry (dim1792/nh216/ns32/dep11/mlp2.26), measured on a
leased RTX 6000 Ada, identical conditions:

| arm | batch | per-GPU tok/s | global tok/s (7-GPU) | peak mem/GPU |
|---|---|---:|---:|---:|
| emender-mlp | bs4 (matched) | 3,211 | 22,474 | 28,942 MB |
| gdn2-mlp | bs4 (matched) | 3,290 | 23,034 | 35,715 MB |
| emender-mlp | bs6 (its DDP max) | 4,470 | 31,291 | 38,497 MB |

- **Matched-conditions per-token throughput ratio emender/gdn2 = 0.976× → a TIE.**
- The earlier **1.26–1.56× speedup** measured at grok scale (small dims, isolated
  1-GPU) **is NOT reproduced at 1.3B — REFUTED**, consistent with the post-mortem warning
  that prior E97 throughput claims were unreliable.
- The Emender's *real* edge at scale is **memory**: 28.9 GB vs 35.7 GB at bs4, so it
  fits bs6 where gdn2 OOMs → a ~1.36× *aggregate* throughput advantage, but that is a
  batch-size/memory effect, **not** a per-token kernel speedup. State the honest version.

### 2.4 Architecture side-probes → both NULL, both confound-cleared. The pair is fixed.

Two attempts to *extend* the E97 cell were run to a 1.3B-comparable verdict and both
came back NULL after a full confound audit. **The experiment is the E97-vs-GDN2 pair;
do NOT add arms to the scaleout.**

- **M2 — multi-query rank-R readout — rank R statistically null at matched capacity.** Provenance:
  `paper/review/M2_CMAES_1P3B_RESULTS.md` (homologous CMA-ES at 1.3B). M2's CMA-best
  (R\*=3) places **LAST** on the SCALE_PLAN §1 leaderboard (search avg-loss **6.1843**
  vs emender-mlp 5.8606, +0.234 worse than its closest sibling pure-E97). The rank knob
  R is **statistically null** at matched capacity: an iso-geometry 3-seed control gives
  R1→R4 mean difference **0.035 < within-R seed std (0.025–0.040)**, and the two-sample
  t-test R1-vs-R4 is **not significant** (**t≈1.0, p≈0.36**, recomputed from the committed
  `iso_geometry_R_control/summary.json`; the source MD's stated p≈0.27 is slightly
  optimistic, but the not-significant conclusion is robust). Confound audit (capacity, fused-guard
  96/96, aggregator, precision, iso-param, unequal-sampling) all cleared.
- **M1 — state-aware MLP (pre-`o_proj` nonlinear head-mixing) — NULL at grok scale.**
  Provenance: `paper/review/STATE_AWARE_MLP_M1_RESULTS.md` (18-run grok probe, fused E97,
  iso-param certified). M1b **ties-or-loses** the plain-MLP baseline at **every metric,
  every T, both p**; per-seed grok rate **2/6 ≤ plain-MLP 8/12**; never raises the grok
  ceiling. The decision rule required beating BOTH baseline and the plain-wider control;
  it did the opposite → **NULL, do not escalate to 1.3B.** Mechanistically expected: M1
  exposes no new state information and does not touch temporal dynamics (the documented
  separator is nonlinearity-IN-TIME inside the recurrence, §2.2).

Both NULLs survive the same audit a positive would, consistent with the standing
convergent-loss-null pattern. They are *defended* negatives, not abandoned ideas.

---

## 3. Scaleout + frontier

### 3.1 Local seed run — the equal-FLOPs E97-vs-GDN2 convergence race

**Goal (provenance: `docs/SCALE_PLAN.md`, commit `c0cdf28`):** stay at **1.3B
parameters** and push the **token count** into the emergence regime (~77 tokens/param,
~4× Chinchilla), tracking the capability-eval suite (held-out BPB + length-extrapolation
+ the algorithmic battery) on checkpoints **as the token count grows**. At ≤2B tokens
both arms are near chance (pre-emergence); the central experiment is whether the
**proved** modquad expressivity separation (§2.2) surfaces as a *measurable real-LM*
divergence between `emender-mlp` and `gdn2-mlp` at scale — an **extension** of
conclusive work, not a hole in it. The 3B/7B parameter scale-up is a separate,
compute-gated, later phase.

**The race protocol (the local gate).** Run E97 and GDN2 as a head-to-head
**convergence race**, decided on the metric, not pre-baked:

- **1 GPU per arm**, **EQUAL banked compute** (equal wall-clock), **resume-from-
  checkpoint**, **indefinite** — there is **no `--steps` cap**; the race runs until the
  metric decides.
- **Re-equalization rule:** if one arm stops (crash, lease loss, hang), bring it back
  level by running the **behind** arm alone for the wall-clock difference, then resume
  both together. Banked compute must stay equal for the verdict to be fair.
- **Hang watchdog (mandatory on a loaded box).** Silent CUDA wedges happen — a step
  stops advancing while the process is still alive, and an 8-hour silent hang can eat a
  race. `scripts/racer_watchdog.sh` (commit `8d5f507`) polls each arm's log every
  `INTERVAL=600 s`; after `STALL_CHECKS=2` consecutive no-advance checks (~20 min) it
  **`kill -KILL`s the wedged pid and writes a LOUD `ALERT … HUNG … Re-equalize needed`**
  to `/mnt/nvme1n1/erikg/race/watchdog.log`. Run it alongside the race.

**The hardware reality (provenance: `experiments/preflight_100b/RESULTS.md`).** The
local box is **8× RTX 6000 Ada (49 GB, PCIe, NO NVLink)**. Measured scaling:

| configuration | aggregate tok/s | efficiency vs 1-GPU |
|---|---:|---:|
| 1× GPU (emender bs6) | 8,600 | 100% (baseline) |
| **7× GPU vanilla DDP** (bs6) | **31,291** | **52%** |
| 7× independent processes (no DDP) | ~62,000 | ~103% (near-linear) |

Vanilla per-step DDP **wastes ~48%** of the GPUs: the bottleneck is the per-step
all-reduce of the 1.29B bf16 gradient (~2.6 GB) over PCIe with no NVLink — *not* CPU,
NVMe, or power (7 independent procs scale near-linearly). `grad_accum` does not fix it.
This is **why DiLoCo (periodic averaging) is the parallelism path**, not synchronous DDP.

#### THE DiLoCo RECIPE — plain averaging, `outer_beta=0`. State this loudly.

```
--diloco  --diloco_k 250  --diloco_outer_lr 1.0  --diloco_outer_beta 0.0
```

**`outer_beta=0` (plain local-SGD weight average) is the ONLY safe outer optimizer
measured. Outer momentum DIVERGES — do not use it.** `outer_beta=0.9` blows up
regardless of seed maturity or island count: from a 528 M seed at I=4 the consensus
loss goes 3.54 → 16.95 (held-out **+1.8 → +35 BPB**), and even at the gentler
`outer_lr=0.5` from the *most-mature* 1.966 B seed at I=6 it degrades **2.76 → 6.47**
(`experiments/seed_maturity_threshold/final_threshold_table.csv`; commits `1b272c1`,
`d9d4433`). This is an outer-optimizer overshoot, not an island-count or maturity
effect; **ramp with the plain β=0 average.**

#### The Schedule-Free × DiLoCo merge bug — FIXED (the merge that corrupted SF).

The naive merge (averaging only the model weights, or resetting `weight_sum=0` /
`z=p.data`) **corrupts Schedule-Free**: it lobotomizes SF's lr-weighted iterate average
into a mere K-step trailing average, so the eval (x) weights regress after every merge.
The **principled merge** averages **both** the SF base iterate `z` **and** the eval
weights `x` across replicas and **PRESERVES** the SF clock scalars
(`weight_sum`, `k`, `lr_max`). Provenance: commit **`a06480a`**, `train.py:diloco_merge`,
`tests/test_diloco_merge.py`, smoke `evaluations/sf_diloco_merge_smoke/README.md` — the
principled merge shows **no post-merge spike** and a better eval-weight loss than the
reset band-aid (final train CE on x **0.0568 vs 0.0614**), with loss decreasing through
all three K=250 merge windows. Every DiLoCo result below uses this principled merge.

#### DiLoCo throughput → GO (~1.85× DDP).

`experiments/diloco_100b/RESULTS.md` (commit `7997419`): K∈{250,500} reaches
**57.7–57.9 k global tok/s = ~1.85× the 31,291 DDP baseline**, recovering 93–93.4% of the
~62 k independent ceiling (98.3–99.6% of the no-merge ceiling); the periodic merge costs
only ~2.1 s amortized. Merge correctness verified (ScheduleFree y-mode swap; gloo unit
test; bit-faithful consensus checkpoint). DiLoCo recovers essentially all of the
independent ceiling that per-step DDP throws away.

#### DiLoCo scaling-law → NO matched-token degradation through I≤4; PARITY from a mature seed.

Provenance: `experiments/diloco_scaling_law/degradation_summary.json`,
`evaluations/diloco-scaling-law-evaluation.md` (agent-1936, Evaluator). With the
principled merge + plain β=0, DiLoCo consensus matched-token held-out BPB **vs the
single-GPU sequential reference** (the relevant sample-efficiency baseline):

| cell | islands | degradation vs single-GPU ref @ matched tokens |
|---|---:|---:|
| from-scratch | I=4 | **+0.009 BPB** (≈ parity, within noise) @ 708 M |
| seed (528 M) | I=2 | **−0.115 … −0.135 BPB** |
| seed (528 M) | I=4 | **−0.075 … −0.129 BPB** |

**There is no matched-token island penalty through I=4 on either seed.** From a mature
seed it is **PARITY** — and the apparent SWA-style "beat" (the negative numbers) is
**WITHIN the reference's ±0.12–0.17 BPB constant-LR noise floor; do NOT call it a beat.**
The {2,4} island-count curve is **FLAT**. **Honest ceiling: this is measured only to
I≤4 on the leasable 4-GPU pool; the 100s–1000s-island Frontier regime is UNTESTED — a
100s/1000s viability number is REFUSED as unsupported extrapolation (a flat ≈0 curve has
no slope to fit).**

#### Seed-maturity → at I≤6 the OUTER OPTIMIZER decides, not seed maturity (counterintuitive).

Provenance: `experiments/seed_maturity_threshold/final_threshold_table.csv`, `DESIGN.md`,
`merge_continuity_summary.json` (commits `1b272c1`, `d9d4433`). Harvested with **no GPU**
from the per-merge loss trajectories the production DiLoCo runs already wrote to disk;
4-lens adversarial audit passed. The maturity × island × β matrix:

| seed maturity | islands | β | merges | loss first → last | verdict |
|---|---:|---:|---:|---|---|
| scratch / 0 tok | 4 | 0.0 | 91 | 9.137 → **2.912** | **IN-BASIN** |
| 528 M | 2 | 0.0 | 22 | 3.543 → 3.089 | IN-BASIN |
| 528 M | 4 | 0.0 | 22 | 3.522 → 3.095 | IN-BASIN |
| 1.233 B | 4 | 0.0 | 358 | 3.380 → 3.109 | IN-BASIN |
| 528 M | 4 | 0.9 | 22 | 3.543 → 16.953 | **BLOW-UP** |
| 1.966 B | 6 | 0.9 (lr 0.5) | 35 | 2.763 → 6.472 | **BLOW-UP** |

**Counterintuitive result: at I≤6, seed maturity is NOT the factor that decides safe
scaleout — the outer optimizer is.** Plain averaging (β=0) is **in-basin at EVERY
maturity, including from-scratch** (9.14 → 2.91 over 91 merges). β=0.9 blows up
regardless of maturity (it diverges even from the most-mature 1.966 B seed). So the
**safe-seed maturity floor at I≤4 is 0 tokens (random-init), directly measured.**
**CAVEAT: this is an I≤6 ceiling; the 512-island Frontier regime is UNTESTED** — a floor
could still matter at island counts large enough to induce drift between merges.

#### The matched-token-vs-synchronous-DDP comparison (distinct baseline, for completeness).

When DiLoCo (β=0, K=250) is compared **against synchronous large-batch DDP** (not the
single-GPU reference) under the **healthy** recipe, it carries a persistent **+0.35 BPB**
matched-token penalty that does **not** close from 64.5 M through 215 M
(`experiments/diloco_100b/longhorizon_fix/RESULTS.md`, e.g. +0.358 → +0.357). This is
**not** in tension with the parity result above — it is a *different baseline*: DiLoCo
matches the single-GPU sequential learner (its natural baseline) at I≤4, and only lags
the expensive 7×-batch synchronous DDP that the no-NVLink box cannot afford anyway (52%
efficiency). For Frontier, where global synchronous DDP over ~1.05 M-token batches is
infeasible, **parity-with-sequential is the relevant and positive result.**

#### The training recipe — the long-horizon blocker is RESOLVED.

Provenance: `experiments/diloco_100b/longhorizon_fix/RESULTS.md` (follow-up
`fix-long-horizon`, commits `5579f11`/`e343d65`). The predecessor found held-out BPB
**collapses for BOTH DDP and DiLoCo** under the constant CMA-tuned LR (1.007e-3,
`warmup_steps=0`, no decay) — a *recipe* bug, not a parallelism bug (the SF eval/x
average rolls over while train loss keeps falling). Two `train.py` plumbing bugs were
fixed (`warmup_steps` never reached the SF optimizer; `get_lr` cosine was malformed →
`lr_scale_at`). **The corrected recipe — AdamW + linear warmup + cosine decay to a small
floor — is strictly monotone**, reaching **1.205 BPB @ 215 M** at the `emender-mlp`
geometry (vs the broken recipe's 3.234 endpoint), beating the broken recipe's *global
minimum* (1.571) by step 750. **The 100B seed run MUST use AdamW + warmup + cosine
decay** (scale `warmup_steps` to ~1–2% of the step budget), not constant-LR
schedule-free.

**→ Current local recommendation:** run the **equal-FLOPs race** (above) to decide
E97-vs-GDN2 on the metric; use the **AdamW+cosine** recipe; if multi-GPU throughput is
needed, **DiLoCo with the locked recipe (β=0, K=250, principled merge)** is parity with
sequential at I≤4 and ~1.85× DDP throughput.

### 3.2 Frontier — hierarchical plain-average DiLoCo-OUTER × DP/3D-INNER over RCCL/Slingshot

The frontier target is an OLCF-Frontier-class allocation (AMD MI250X). The plan is
**NOT plain DDP across all ranks** — that forces an enormous global batch
(64 nodes × 8 GCDs × 2048 = **~1.05 M tokens/update** even at per-GCD bs=1, risky for
learning). Instead use **hierarchical plain-average (β=0) DiLoCo**: synchronous
training (DDP, ideally 1 node = 8 GCDs) *inside* each island, periodic weight averaging
*between* islands. (Design: `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`; systems notes:
`docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`.)

#### Frontier scheduler reality (OLCF) — plan the job shape around it.

> **Provenance flag.** The scheduler-policy numbers in this subsection are **external
> OLCF Frontier scheduling policy / operator directive**, not internal measured
> artifacts of this repo (the only repo-committed pieces are the 64-node × 24 h target
> and the 64 nodes × 8 GCDs = 512-GCD batch arithmetic in
> `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`). **Re-verify against current OLCF docs
> before relying on them for an allocation.** They are recorded here because they shape
> the job design.

- **`extended` partition:** 64-node max, 24 h walltime, **1 running + 1 eligible** job
  per user → throughput is **SERIAL chained-resume** via
  `sbatch --dependency=afterany:<jobid>`. **Our checkpoint/resume design fits this
  natively** (every run already saves and resumes from a consensus checkpoint).
- **`batch` bins:** the **aging boost only applies at ≥1882 nodes** (leadership-class);
  below that you do not get the priority bump.
- **Realistic throughput:** at 64 nodes (~512 GCDs, MI250X at roughly half the per-GCD
  rate) ≈ **~150–180 B tokens per 24 h job** — **NOT 100B/hour** (that is the
  leadership-class regime, not the 64-node `extended` regime). *(Planning estimate;
  derive the exact figure from the P1 single-GCD throughput measurement.)*
- **Pack BOTH arms (E97 and GDN2) into ONE 64-node job** (32 nodes each) so they stay
  concurrent under the 1-running-job limit — otherwise the second arm waits in the
  eligible slot.
- **R&D loop:** use the **`batch` partition at ≤91 nodes / ≤2 h** for the fast iteration
  loop (those backfill quickly).

#### Data — commapile is MANDATORY on Frontier (legal hard requirement).

> **`pile.txt` is NOT license-clean → it is illegal to train the releasable model on it
> on Frontier.** The releasable model MUST be **commapile END-TO-END** (commapile *seed*,
> not a pile-seeded model fine-tuned on commapile).

- **Corpus:** `commapile_mainmix_v0.1_1tb.txt` — comma v0.1 **main-stage weighted mix**,
  **31 sources** (manifest `commapile_mainmix_v0.1_1tb.txt.manifest.json`:
  `sources=31`, `mixture` weights), **license-clean** (`docs/SCALE_PLAN.md`),
  **interleaved on disk** (single shuffled stream, not concatenated). 1.000 TB raw
  (≈250 B p50k tokens — one epoch); already the corpus used by every preflight/DiLoCo
  measurement above.
- **Artifact for transfer:** `commapile_mainmix_v0.1_1tb.txt.zst` — **251.7 GB**
  (compressed, valid; `251,655,400,225` bytes on disk at
  `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/`).
- **Upload target:** `s3://garrisonlab/commapile/` — *(intended staging location; the
  S3 path itself is an operator directive, not yet a committed artifact — confirm the
  bucket/prefix and integrity hash on upload).*
- `pile.txt` remains usable only for the *local* `<1 bpb` convergence-race gate (it is
  the corpus that history's BPB anchors were computed on); the **Frontier / releasable**
  path is commapile-only.

#### Primitives that EXIST (cite, build on — do not reinvent):

| Primitive | Reference | Role |
|---|---|---|
| DiLoCo (low-communication distributed training) | arXiv:2311.08105 | the outer periodic-averaging method |
| OpenDiLoCo (open reproduction, global scale) | arXiv:2407.07852 | scaled open implementation |
| Decoupled momentum / outer-optimizer variants | arXiv:2604.21428 | outer-optimizer design space (note: β=0 is our locked recipe — §3.1) |
| Eager updates (reduce DiLoCo comm stalls) | arXiv:2502.12996 | overlapping comm with compute |
| Scaling laws for DiLoCo | arXiv:2503.09799 | how the method scales with model/island count |
| Pier (DiLoCo systems / partitioning) | arXiv:2511.17849 | systems-level DiLoCo |
| Frontier 3D-parallelism + ROCm precedent (Dash et al.) | arXiv:2312.12705 | optimizing distributed LLM training on Frontier/MI250X |
| Low-bandwidth model partitioning | arXiv:2501.04266 | partitioning for slow interconnects |

Supporting systems facts (from `FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`): use
**ROCm/Megatron-LM** (`github.com/ROCm/Megatron-LM`, NOT Microsoft Megatron-DeepSpeed);
the **`aws-ofi-rccl` plugin MUST be built + LD_PRELOAD'd** (default RCCL uses TCP/IP on
Slingshot — catastrophic); pre-build DeepSpeed JIT ops at image time (they fail on
ROCm); checkpoint with PyTorch DCP `SHARDED_STATE_DICT`, tiered NVMe→Orion-Lustre; watch
the known `torch.compile + bf16 + ROCm` NaN gotcha. For non-ParaRNN parallelism, the
recommended inner shape is TP=8 intra-node, PP=4 (interleaved 1F1B), DP cross-node
(ZeRO-1). Optional high-risk axis: **ParaRNN** (arXiv:2510.21450) parallelizes nonlinear
RNNs across the sequence via Newton's method — *if* it converges on the E97 matrix-state
recurrence (untested; matrix state makes the Newton block expensive), sequence
parallelism becomes available; prototype before betting budget on it.

#### The 3 GAPS to build/validate (unpublished — this is the research+engineering work):

- **(a) The composition is unpublished.** Plain-average DiLoCo-outer over RCCL with a
  **DP/3D-parallel inner** (DP within an island, optionally TP×PP) at multi-node scale
  has not been published. The primitives exist separately; composing them — and
  validating throughput, fault-tolerance, and learning parity at multi-node scale — is
  novel work.
- **(b) Schedule-Free AdamW as the DiLoCo INNER optimizer is unstudied.** Published
  DiLoCo uses AdamW-inner + SGD/Nesterov-outer. This project's inner optimizer is
  **Schedule-Free AdamW**, whose internal train/eval (x/y/z) weight semantics interact
  with the merge — **resolved locally** by the principled merge (§3.1, commit `a06480a`):
  average both `x` and `z`, preserve the SF clock. The merge target is the eval (x)
  weights with the base sequence `z` reset to consensus (see
  `SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` §ScheduleFree Interaction). The remaining work
  is confirming this holds at multi-node scale. *(If the final run uses AdamW+cosine —
  the recipe §3.1 mandates for the long horizon — the SF-merge interaction may be moot;
  decide the inner optimizer before P2.)*
- **(c) The E97 fused split-edit kernel → ROCm/MI250X port (biggest eng risk, hard
  prereq).** The fused E97 kernel is CUDA/Triton and **bf16-only**, and has had chunked
  fp32-overflow / NaN edge cases on CUDA already (`complex-eig-chunked-overflow`,
  `fuse-2kernel` notes). It **must** be ported to ROCm/MI250X and **parity-verified**
  (≤ few e-3 in bf16, T∈{128,512,1024,2048}) on a single MI250X GCD before any multi-node
  run. There is **no fp32 safety net** for the fused path. This is the gating risk for
  choosing `emender-mlp` (vs the FLA-mature `gdn2-mlp`) as the frontier arm.

### 3.3 Phased, risk-ordered plan

- **P0 — Local race (now).** Run the equal-FLOPs E97-vs-GDN2 convergence race (§3.1) on
  the AdamW+cosine recipe with the watchdog, deciding the arm on the metric. Produce the
  seed checkpoint(s) + capability evals at a fixed token cadence. If multi-GPU is needed,
  use DiLoCo with the locked recipe (β=0, K=250, principled merge).
- **P1 — ROCm kernel port (de-risk EARLY).** Port the E97 fused split-edit kernel to
  ROCm/MI250X and parity-verify on a single GCD; **measure single-GCD throughput** to
  pin the §3.2 token-per-job estimate. Gap (c), hard prereq — do it *before* committing
  HPC allocation. Keep `gdn2-mlp` (mature ROCm via FLA) as the fallback arm.
- **P2 — HPC DiLoCo over RCCL + small-multinode validation.** Stand up hierarchical
  plain-average (β=0) DiLoCo-outer × DP/3D-inner (gap a) on a small node count (e.g. 16
  nodes); validate throughput, learning parity at matched tokens, and fault-tolerance
  (straggler/failed-island handling) before scaling. **Train on commapile** (legal).
- **P3 — Scale.** Scale node count (64 → larger) and, separately and compute-gated,
  parameters (1.3B → 3B → 7B → 13B) with token scaling, tracking the scaling law and
  capability divergence. Remember the I≤6 ceiling on the seed-maturity/scaling result —
  the large-island regime is the first thing to re-measure at P2/P3.

---

## 4. Artifact map

| What | Where | Notes |
|---|---|---|
| **CMA-best 1.3B configs** | `experiments/lb_compare_20260613/REPRODUCTION.md` | exact geometries for all 5 arms; param counts verified |
| **LM leaderboard / convergent-loss null** | `experiments/lb_compare_20260613/{LEADERBOARD.md, bpb_results.json, sep_results.json}`; `paper/main.typ` "same held-out bpb band" | read the CORRECTIONS, not the auto-verdict |
| **Capability separation (modquad), two-sided** | `experiments/grok_symmetric_width/{RESULTS.md, CONFIRM.md, *.json}` | 216 + 96/24/24 runs; length-extrapolation to T=4096 |
| **Architecture side-probes (both NULL)** | `paper/review/M2_CMAES_1P3B_RESULTS.md`, `paper/review/STATE_AWARE_MLP_M1_RESULTS.md` | M2 last + R-null; M1 null at grok; confound-cleared |
| **Preflight throughput / DDP scaling** | `experiments/preflight_100b/RESULTS.md`, `run_ddp.sh`, `ckpt_roundtrip.py` | the 0.976× tie + 52% DDP scaling |
| **SF×DiLoCo merge fix** | commit `a06480a`; `train.py:diloco_merge`; `tests/test_diloco_merge.py`; `evaluations/sf_diloco_merge_smoke/README.md` | average x AND z, preserve weight_sum/k/lr_max |
| **DiLoCo throughput + merge correctness** | `experiments/diloco_100b/RESULTS.md`, `run_diloco.sh` | 1.85× DDP, y-mode merge verified |
| **DiLoCo scaling-law (I≤4 parity)** | `experiments/diloco_scaling_law/degradation_summary.json`, `evaluations/diloco-scaling-law-evaluation.md` | no matched-token degradation; SWA "beat" within noise |
| **Seed-maturity threshold (β decides)** | `experiments/seed_maturity_threshold/{final_threshold_table.csv, DESIGN.md, merge_continuity_summary.json}` | β=0 in-basin at every maturity; β=0.9 blows up |
| **DiLoCo long-horizon recipe fix + vs-DDP gap** | `experiments/diloco_100b/longhorizon_fix/RESULTS.md`; `experiments/diloco_100b/longhorizon/RESULTS.md` | AdamW+cosine monotone to 1.205; +0.35 vs synchronous DDP |
| **Equal-FLOPs race + watchdog** | `docs/SCALE_PLAN.md` (race = the gate); `scripts/racer_watchdog.sh` (commit `8d5f507`); `scripts/run_diloco_seed_race_i4.sh`, `scripts/racer_eval_suite.py` | 1 GPU/arm, equal banked compute, re-equalize on stall |
| **Training data — releasable (legal)** | `commapile_mainmix_v0.1_1tb.txt` (+`.zst` 251.7 GB, `.manifest.json` sources=31) at `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/`; intended `s3://garrisonlab/commapile/` | **MANDATORY on Frontier**; license-clean, 31-source main-stage mix, interleaved |
| **Training data — local gate only** | `/mnt/nvme2n1/erikg/pile.txt` (symlink `/home/erikg/elman/data/pile.txt`) | the `<1 bpb` race gate corpus; **NOT license-clean → not for the releasable Frontier model** |
| **The cell (source)** | `ndm/models/e88_fla_hybrid.py` (`use_split_edit=True`); ablations via `--e88_raw_write` / `--linear_state` | E97 split-edit delta |
| **Fused E97 kernel** | `ndm/triton/e97_chunked.py` (linear-state chunked); `ndm/triton/e88_triton_optimized.py` + `e88_triton_{forward,backward}.py` with `SPLIT_EDIT=True` (nonlinear sequential); tests `tests/test_e97_chunked.py`, `tests/test_e88_triton.py` | Shared E88-derived sequential core; see the [naming clarification](E97_E88_KERNEL_NAMING_CLARIFICATION_20260802.md) |
| **Model wiring (SwiGLU MLP, GDN-2 control)** | `ndm/models/ladder_lm.py` (`SwiGLUMLP`, `MixerMLPWrapper`, `gdn2-mlp`) | the fair MLP counterpart |
| **Trainer (DDP + DiLoCo opt-in)** | `train.py` (`--level`, `--use_triton`, `--diloco`, `--diloco_k/outer_lr/outer_beta`, `--warmup_steps`, `--min_lr_frac`, `--heldout_tensor`, `--data_rank/--data_world_size`) | single-GPU path byte-identical |
| **CMA driver (standard HPO)** | `scripts/cmaes_search_v2.py` (`build_train_command`, search spaces) | use this, full geometry, ≥96 evals — not bespoke searches |
| **Eval harnesses** | separators: `experiments/expressivity_tasks/train_hybrid.py`, `experiments/lb_compare_20260613/run_separators.py`; grok: `experiments/grok_symmetric_width/train_grok.py`; held-out BPB: `experiments/*/run_bpb.py`, `build_heldout_tensor.py` | |
| **Paper** | `paper/main.typ` (build: `bash paper/build.sh`); taxonomy fig anchor `@fig_taxonomy` | |
| **Key docs** | `docs/SCALE_PLAN.md`, `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`, `docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`, `docs/RESEARCH_ASTRAY_POSTMORTEM.md`, `docs/MODEL_CARD_TEMPLATE.md`, `paper/review/EMENDER_TAXONOMY.md` | |
| **GPU lease broker (this box)** | `scripts/gpu_lease.sh` (`eval "$(scripts/gpu_lease.sh N)"`) | ALWAYS lease before touching a GPU; no central allocator, 8 shared GPUs |

Provenance commits (most recent first): `8d5f507` racer watchdog · `1b272c1` /
`d9d4433` seed-maturity-threshold · `a06480a` sf-diloco-merge · `92a153f` /
`5579f11` / `e343d65` diloco-loss-parity + fix-long-horizon · `7997419` diloco-periodic
· `0e914a3` preflight-100b · `c0cdf28` scale-plan lock + postmortem addendum · `ce7ff39`
grok-confirm · `d489955` grok-symmetric-width · `8acd929` / `ece1b16` lb-compare +
corrections · `baf47f2` research-astray post-mortem.

---

## 5. Methodology guardrails

`docs/RESEARCH_ASTRAY_POSTMORTEM.md` is required reading: it documents ~10 days in which
**every "the Emender loses / it's a null" verdict came from a comparison quietly rigged
against the Emender**, each corrected (always by the human PI) toward a positive, with
the final fair test flipping to a win. The errors were *one-directional* — a neutral
error process would scatter. So the guardrails are not optional:

1. **Burden of proof is on NULLs; do not hedge a proved result.** No stop verdict without a
   cleared confound audit (M1/M2 in §2.4 model this). Symmetrically, do not deflate a
   proved, two-sided result (§2.2) into "scoped/narrow," and do not soften a real finding
   (the convergent-loss null, §2.1) into "inconclusive."
2. **Trust measured artifacts over synthesis.** No verdict until the committed measured
   data is on screen. Read the **primary metric head-to-head**; do not inherit a prior
   agent's verdict without checking the basis (e.g. averaged vs non-averaged weights —
   that flips the lb-compare ranking).
3. **Test class-separation at CONSTRAINED capacity + LENGTH-EXTRAPOLATION.** Capacity
   buys *memorization*, which masquerades as capability if you score at the train length.
   The real test is whether a model holds as T grows past the trained length (this
   flipped the `grok_highp_temporal` false-negative — §2.2).
4. **Fair = best-vs-best, MLP-controlled, with documented symmetric search space, budget,
   precision, and geometry — fixed *before* any conclusion.** Past failures: searching
   the Emender only over a 2-D mixture axis while giving GDN-2 the full geometry; running
   the Emender in fp32 while controls ran bf16; comparing GDN-2-with-MLP to
   Emender-without-MLP; under-searching (64 vs 104+ evals).
5. **NON-NEGOTIABLE #1 — fused Triton only.** Every recurrence runs through the fused
   `@triton.jit` kernel with the `[fused-guard] … NO eager fallback` assert; any eager
   path is INVALID and scores 0. Eager signal does not transfer to fused (documented
   inert/untrainable fused cases).
6. **Test capability AT GROK** (AdamW + wd sweep + 10–100× the steps memorization needs);
   the short-schedule, no-wd-sweep `lb_compare` separators are grok-suppressed and do not
   test the claim. Use real separators (iterated maps / unbounded counting at
   length-extrapolation), not finite-state proxies where linear is *expected* to win.
7. **No premature "done"; liveness = step/merge count advancing, not GPU-util snapshots.**
   A task is done only when its run process exits and the result is committed.

---

## 6. Open questions / immediate next

**The central open question.** Does the **proved** `modular_quadratic` expressivity
separation (§2.2) surface as **measurable, real-LM capability** at the emergence regime
(~77 tok/param), as a **divergence between `emender-mlp` and `gdn2-mlp`**? LM loss is a
convergent null (§2.1) and at ≤2B tokens both arms are pre-emergence — so this is the
**forward experiment the scaleout enables**, an extension of conclusive work. The
equal-FLOPs race (§3.1) is the local instrument; track held-out BPB + length-extrapolation
+ the algorithmic battery on checkpoints at a fixed token cadence.

**Resolved since the prior handoff** (do not re-litigate): the SF×DiLoCo merge bug
(`a06480a`); the long-horizon recipe collapse (AdamW+cosine, monotone to 1.205 BPB); the
DiLoCo recipe (plain β=0, K=250); DiLoCo I≤4 matched-token parity; the M1 and M2 NULLs.

**The 3 scaleout gaps (§3.2), risk-ordered:**
- **(c) ROCm/MI250X port of the bf16-only fused E97 kernel** — the long pole and hard
  prereq; parity-verify + throughput-measure on a single GCD before any HPC allocation.
  `gdn2-mlp` is the fallback arm.
- **(a) The hierarchical composition** (plain-average DiLoCo-outer over RCCL × DP/3D
  inner) — unpublished; validate throughput/parity/fault-tolerance at small node count
  (P2) before scaling. **Re-measure the seed-maturity/scaling result at the large-island
  regime (the I≤6 ceiling is the known blind spot).**
- **(b) Inner optimizer at multi-node scale** — the SF merge is fixed locally; confirm
  the choice (SF vs AdamW+cosine) and its merge interaction before P2.

**Smaller follow-ups owed:** multi-seed held-out BPB to tighten the within-band
`emender-mlp` vs `gdn2-mlp` margin (§2.1); confirm the `s3://garrisonlab/commapile/`
upload (path + integrity hash); a proper grok-protocol separator battery if the
`lb_compare` separators are ever to be cited (currently grok-suppressed, not
load-bearing).

---

## 7. CLAIM → EVIDENCE fact-check

Every quantitative claim in this document, mapped to the committed artifact that
supports it. **VERIFIED** = the number was read from the cited committed artifact during
this update. **EXTERNAL** = an external policy/operator directive, not an internal
measured artifact (flagged in-text; re-verify before relying on it). No
mock/aspirational numbers appear in this document.

| # | Claim | Source (committed) | Status |
|---|---|---|---|
| 1 | `emender-mlp` params = 1,286,589,072; `gdn2-mlp` = 1,286,713,448 | `lb_compare_20260613/REPRODUCTION.md`; `longhorizon_fix/RESULTS.md` | VERIFIED |
| 2 | Geometry emender dim1792/nh216/ns32/dep11/mlp2.2623; gdn2 dim2176/nh30/dep12/mlp3.2587 | `lb_compare_20260613/REPRODUCTION.md`; `diloco-scaling-law-evaluation.md` §1 | VERIFIED |
| 3 | Search avg-loss 5.8606 vs 5.8949; held-out BPB non-avg 2.0911 vs 2.1013; avg 2.1783 vs 2.1550 | `lb_compare_20260613/LEADERBOARD.md` (commit `8acd929`/`ece1b16`) | VERIFIED |
| 4 | "same held-out bpb band" (convergent-loss null) | `paper/main.typ` lines ~1281/1286/1309 | VERIFIED |
| 5 | modquad: e97 0.982→0.981 length-invariant; e97-lin 0.619 / gdn2 0.560 @ T4096 (4-seed sweep, decisive-cell gaps +0.26…+0.42) | `grok_symmetric_width/RESULTS.md` (commit `d489955`) | VERIFIED |
| 5b | 8-seed gap e97−gdn2 = +0.17…+0.32 @ T4096 (corrected: this range is the 8-seed confirmation, not the 4-seed RESULTS.md) | `grok_symmetric_width/CONFIRM.md` (commit `ce7ff39`) | VERIFIED (fixed citation) |
| 6 | Two-sided: iterated_nonlinear_map 0/24 grok (contractive); anbncn 24/24, gap sign-flipping | `grok_symmetric_width/CONFIRM.md` (commit `ce7ff39`) | VERIFIED |
| 7 | e97 vs e97-linear byte-identical code (only tanh vs identity) | `ndm/models/e88_fla_hybrid.py`; `grok_symmetric_width/RESULTS.md` | VERIFIED |
| 8 | Throughput tie 0.976×; emender 28.9 GB vs gdn2 35.7 GB @ bs4; 1.26–1.56× refuted at scale | `preflight_100b/RESULTS.md` (commit `0e914a3`) | VERIFIED |
| 9 | DDP scaling: 1-GPU 8,600; 7-GPU DDP 31,291 (52%); 7 indep ~62,000 (~103%) | `preflight_100b/RESULTS.md` | VERIFIED |
| 10 | M2 last on leaderboard (6.1843 vs emender 5.8606); R null (R1→R4 mean diff 0.035 < seed std 0.025–0.040; t-test p≈0.36 recomputed from summary.json, not significant — corrected from the source MD's optimistic p≈0.27) | `paper/review/M2_CMAES_1P3B_RESULTS.md`; `experiments/cmaes_m2_1p3b_20260616/iso_geometry_R_control/summary.json` | VERIFIED (fixed statistic) |
| 11 | M1 ties-or-loses plain-MLP every T/both p; grok 2/6 ≤ 8/12; NULL | `paper/review/STATE_AWARE_MLP_M1_RESULTS.md` | VERIFIED |
| 12 | Principled merge: averages x AND z, preserves weight_sum/k/lr_max; smoke x-CE 0.0568 vs 0.0614, no spike | commit `a06480a`; `evaluations/sf_diloco_merge_smoke/README.md`; `tests/test_diloco_merge.py` | VERIFIED |
| 13 | DiLoCo throughput K250/500 = 57.7–57.9k tok/s = 1.85× DDP; 93–93.4% of 62k | `diloco_100b/RESULTS.md` (commit `7997419`) | VERIFIED |
| 14 | Scaling-law: scratch/I4 +0.009; seed/I2 −0.115…−0.135; seed/I4 −0.075…−0.129; SWA "beat" within ±0.12–0.17 noise | `diloco_scaling_law/degradation_summary.json`; `diloco-scaling-law-evaluation.md` | VERIFIED |
| 15 | Seed-maturity: β=0 in-basin scratch 9.137→2.912 (91 merges) … 1.233B (358 merges); β=0.9 blow-up 3.543→16.953 and 2.763→6.472 | `seed_maturity_threshold/final_threshold_table.csv`, `merge_continuity_summary.json` | VERIFIED |
| 16 | Recipe β=0 K=250 outer_lr=1.0; β=0.9 diverges incl outer_lr=0.5 (2.7634→6.4722) | `final_threshold_table.csv`; `diloco-scaling-law-evaluation.md` §4 | VERIFIED |
| 17 | Long-horizon: broken SF rolls 1.571→3.234; AdamW+cosine monotone to 1.205 @215M; DiLoCo +0.35 vs DDP (plateau) | `diloco_100b/longhorizon_fix/RESULTS.md` | VERIFIED |
| 18 | Racer watchdog: INTERVAL 600s, STALL_CHECKS 2 (~20 min), kill+ALERT on stall | `scripts/racer_watchdog.sh` (commit `8d5f507`) | VERIFIED |
| 19 | commapile: 31 sources, license-clean main-stage mix; `.zst` 251.7 GB (251,655,400,225 B) | `…/commapile_mainmix_v0.1_1tb.txt.manifest.json` (sources=31); on-disk `.zst`; `docs/SCALE_PLAN.md` | VERIFIED |
| 20 | Frontier batch arithmetic 64 nodes × 8 GCDs × 2048 = ~1.05M tokens/update; 512 GCDs | `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` | VERIFIED |
| 21 | OLCF `extended` 64-node/24h/1-running+1-eligible/afterany; batch aging-boost ≥1882 nodes; ~150–180B tok/24h @64 nodes; pack both arms 32+32; batch ≤91 nodes/≤2h R&D | external OLCF scheduler policy / operator directive | **EXTERNAL — re-verify** |
| 22 | `s3://garrisonlab/commapile/` upload target | operator directive | **EXTERNAL — confirm on upload** |
| 23 | Frontier primitive arXiv ids (2311.08105, 2407.07852, 2604.21428, 2502.12996, 2503.09799, 2511.17849, 2312.12705, 2501.04266, 2510.21450) | external literature; carried from prior handoff via `docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md` | EXTERNAL (citations) |

**Hedge audit:** §2.1 states the convergent-loss null as a finding (not "inconclusive");
§2.2 states the separation as proved + causally-isolated + two-sided (not "scoped" /
"narrow"); §3.1 states the recipe (β=0) loudly and the I≤4 parity as parity (the SWA
"beat" explicitly de-rated to within-noise, not called a beat); the only soft language is
the **explicitly-flagged honest ceilings** (I≤6 untested at 512 islands; the EXTERNAL
scheduler/S3 numbers) — these are calibration, not deflation of a proved result.

---

*Self-contained handoff authored from committed measured artifacts. Every finding cites
its provenance (path + commit) and is fact-checked in §7. The framing is calibrated and
non-deflating: the expressivity separation is PROVED and two-sided; the convergent-loss
null is a real finding; the DiLoCo recipe is locked at plain β=0; commapile is mandatory
on Frontier; and the binding near-term work is the local equal-FLOPs race plus the ROCm
kernel port.*
