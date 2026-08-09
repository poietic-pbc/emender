# E97 paper primary graph selection

**Decision date:** 2026-08-08

## Decision

The paper's E97-MLP and matched E97-linear-MLP arms use the exact dense graph
that produced the 513,013,841,920-token production seed and was subsequently
upcycled into the 35B MoE. The older canonical CMA `e97` entry is retained as
historical mixer-only provenance, not mislabeled as E97-MLP.

The paper standardizes all three from-scratch arms to B2. The production dense
seed used B4; microbatch does not alter the graph, and common B2 preserves
literal sample grouping and accepted-token cadence across the paper arms.

## Verified dense graph

Source evidence:

- `configs/frontier/e97_resilient_split_role_flat.json`
- `docs/validation/e97-35b-moe-graph-inspection-step2322520-513b.json`
- checkpoint metadata from the protected 513.014B milestone

```text
level=E97
dim=1792
depth=11
n_heads=216
n_state=32
linear_state=0
e88_raw_write=0
use_gate=1
gate_activation=silu
mlp_ratio=2.2623
mlp_multiple=64
exact_mlp_hidden=4032
vocab_size=50281
```

Exact instantiated counts:

```text
total                              1,286,589,072
non-MLP recurrent/shared           1,048,152,720
11 post-mixer SwiGLU MLPs            238,436,352
one MLP: 3 * 1792 * 4032              21,676,032
MLP fraction                              18.5324%
```

The graph inspection proves all 11 layers are uniform and records the exact
FFN tensor shapes (`w1/w2 [4032,1792]`, `w3 [1792,4032]`), residual and
normalization order, tied embedding/head, and checkpoint name/shape match.

The primary linear-state ablation clones this complete graph and changes only
`linear_state=0 -> 1`.

## Size match to GDN2-MLP

```text
verified E97-MLP                  1,286,589,072
retained GDN2-MLP                 1,285,245,320
difference                           1,343,752
relative difference                      0.1046%
```

## Upcycle cross-check

The 35B MoE conversion replaced only `layers.*.mlp`; embeddings, recurrent
mixers, state transitions, projections, gates, norms, and output head remained
protected.

```text
shared non-MoE backbone           1,048,152,720
routed experts/layer                         64
shared experts/layer                          1
top-k                                          3
expert hidden width                         8832
router parameters                       1,261,568
total unique parameters             34,998,209,168
active parameters/token               3,138,570,896
```

The running immutable job 5208321 independently passed its executable packed
count guards:

```text
local parameters/rank                5,750,016,656
local routed-expert parameters       4,178,313,216
```

This running continuation remains independent of the from-scratch paper study
and is not modified or retroactively assigned the new sampler.

## Superseded ambiguity

The historical CMA entry
`dim=2176, n_heads=170, n_state=32, depth=14` has exactly 1,274,697,304
parameters only with `mlp_ratio=0`. Adding a post-mixer MLP would make it a
different, substantially larger graph. It therefore must not be used as the
paper's E97-MLP recipe or cited as including MLP parameters.
