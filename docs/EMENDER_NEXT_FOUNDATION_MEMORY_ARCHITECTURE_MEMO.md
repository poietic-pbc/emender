# Memo: Increasing Useful Recurrent State in the Next EMENDER Foundation

Status: research direction following the dense-agent and post-mixer MoE studies  
Purpose: record the architectural conclusions and candidate experiments for a new 3–8B seed

## Executive conclusion

The next EMENDER should remain a simple, homogeneous recurrent language model. Its central design objective should be to make persistent hidden state abundant, easy to preserve, easy to address, and easy to read relative to the number of learned parameters.

The present 1.3B E97 is an unusually good likelihood model for its size, and targeted SFT can teach it reliable narrow tool behavior. It nevertheless has weak semantic control and unfamiliar-task behavior. The 35B post-mixer MoE added conditional FFN storage without enlarging the shared recurrent state or residual representation. That experiment therefore did not test the more promising form of scaling: more useful recurrent computation and memory.

The recommended next step is not a more elaborate external-memory system and not another large MLP-only MoE. It is a dense 3–4B recurrent seed that reallocates capacity toward:

1. a wider residual stream;
2. fewer but richer matrix-state heads;
3. more persistent state per projection parameter;
4. optionally several homogeneous state banks driven by shared projections;
5. straightforward, nonlinear access to recurrent readouts;
6. a recurrence with an easy learned no-op/retain path;
7. training tasks that make retention and retrieval necessary.

A 7–8B version is a credible target after this design demonstrates behavioral scaling. Tens of billions of parameters should be deferred until the recurrent bottleneck itself scales.

## Evidence motivating the redesign

### Dense E97

The dense 1.3B model has:

- residual width `1,792`;
- 11 recurrent layers;
- 216 heads/layer;
- matrix state size `32 x 32` per head;
- approximately 2.4 million FP32 recurrent values across the model;
- approximately 9.38 MiB of recurrent cache per session.

It achieved strong language-model loss and learned narrow action grammars after sufficient targeted training. The promoted direct CLI checkpoint reached 40/40 exact real-Pi tasks. It did not reliably infer unfamiliar CLI control flow from help.

This is not evidence that recurrence cannot support agents. It is evidence that good next-token loss and a numerically large cache do not guarantee that the model has learned useful state allocation, addressing, retention, and control.

### Post-mixer MoE

The 35B MoE replaced only post-mixer MLPs. It retained the dense model's:

- residual width;
- recurrent projections;
- matrix-state geometry;
- shared recurrent cache;
- layer count;
- state-update policy.

It therefore scaled conditional storage, not recurrent reasoning capacity.

The first RS-free MoE agent study is still informative. The untrained parent and update 8 produced no valid actions, while the cumulative update-32 checkpoint began producing real behavior: its first four-task segment-prefill canary passed 2/4 strictly, used the correct operational argv on 3/4, and grounded 3/4 submissions. This confirms that likelihood improvement eventually transfers to action behavior, but it does not change the architectural point: MLP experts do not provide independent memory tracks.

## Where the current parameter budget goes

Let:

- `d` be residual width;
- `H` be the number of recurrent heads;
- `n` be matrix-state side length;
- `m = H*n` be the concatenated projected memory width.

For current E97:

```text
d = 1,792
H = 216
n = 32
m = 6,912
```

Approximate learned parameters per layer are:

| Component | Parameters |
|---|---:|
| fused QKV, `d -> 3m` | 37.2M |
| output gate, `d -> m` | 12.4M |
| erase gate, `d -> m` | 12.4M |
| value-write gate, `d -> m` | 12.4M |
| output projection, `m -> d` | 12.4M |
| decay projection and small terms | 0.4M |
| post-mixer SwiGLU | 21.7M |

Approximately 87M of roughly 109M parameters/layer are recurrent projections and gates. The model projects a 1,792-wide residual into a 6,912-wide head representation, updates many small states, and linearly compresses the result back to 1,792.

This explains why simply adding more `n=32` heads is unattractive. It increases hidden state, projection parameters, projection FLOPs, and cache almost proportionally.

## The essential state-efficiency relation

Matrix-state values per layer scale as:

```text
state values = H*n^2
```

The dominant dense projection parameters scale approximately as:

```text
projection parameters = C*d*H*n
```

where `C` counts the QKV, gate, and output families.

Their ratio is therefore approximately:

```text
state values / projection parameters ~= n / (C*d)
```

The number of heads cancels. At fixed state side length, adding heads does not improve hidden-state capacity per learned projection parameter. The most direct ways to improve the ratio are:

1. increase `n`;
2. reduce redundant projection families without weakening the cell;
3. let several independent states share projections;
4. reduce residual-to-state dense connectivity through grouping or factorization.

This state-efficiency ratio should be a first-class design and reporting metric for the next model.

## Primary recommendation: fewer, richer heads

At a similar projected width:

```text
current: 216 * 32 = 6,912 projected channels
option:  128 * 48 = 6,144 projected channels
option:   96 * 64 = 6,144 projected channels
```

Their matrix-state capacities are:

```text
216 * 32^2 = 221,184 values/layer
128 * 48^2 = 294,912 values/layer
 96 * 64^2 = 393,216 values/layer
```

The `96 x 64` arrangement uses slightly fewer projected channels but has about 1.8 times as much persistent state per layer. Its recurrent state update is more expensive, but that is the desired trade: spend more computation on actual state rather than repeated dense projections.

`n=48` is a useful intermediate candidate if `n=64` is too expensive. These alternatives require kernel qualification; the existing `n=32` implementation should remain the control.

## Homogeneous multi-bank recurrence

A simple way to increase state without proportionally increasing projections is to give each head several state banks driven by the same projected query/key/value and split-edit signals.

For `B` banks:

```text
projection width remains approximately H*n
state values become B*H*n^2
```

Each bank runs the same recurrent cell. Banks can differ only through small learned retention/decay parameters or initialization. Their readouts are combined through a small learned gate before the ordinary output projection.

This preserves the desired homogeneity:

- one cell type;
- one projection interface;
- no symbolic memory controller;
- no hand-written semantic roles;
- no external read/write language;
- the model decides what the banks represent.

A two-bank version doubles state with little projection growth. A four-bank version is an important upper probe. Recurrent FLOPs and cache grow with the number of banks, so this is a compute-for-memory exchange rather than free capacity.

The principal risk is symmetry: identically driven banks may learn redundant states. The least invasive symmetry breakers are independent decay biases, bank embeddings applied only to gates, or small per-bank/per-head transforms. These should remain much smaller than another complete `d -> H*n` projection.

## Make remembering easy

The model should not need a fragile sequence of precise floating-point cancellations to retain a value. The recurrent update should provide an easy path for:

```text
retain previous state nearly unchanged
write a bounded new association
selectively erase or replace an association
read without modifying
```

E97's split-edit cell is a reasonable starting point because it separates read/erase and value write. The redesign should preserve that useful structure unless controlled experiments reject it.

Important initialization and training properties include:

- retention initialized near a useful long-memory regime;
- stable identity/no-op behavior;
- sparse or conservative writes early in training;
- explicit reset only at true record/task boundaries;
- FP32 recurrent state during inference;
- state-continuous training segments;
- examples whose loss cannot be minimized without using distant information.

The goal is not to assign memories manually. It is to make useful memory a simple solution for gradient descent.

## Make state easy to read

A state that is retained but difficult to expose is not useful memory.

Current E97 concatenates many head readouts and largely collapses them through one linear output projection before the main nonlinear computation can integrate them. The next model should test a modest nonlinear readout while remaining homogeneous:

```text
per-head readout
  -> shared small per-head nonlinearity
  -> grouped summaries
  -> shared cross-group mixer
  -> residual projection
```

The transformation should be shared across heads/groups wherever possible. It should not introduce manually specialized memory types. The model should be able to form nonlinear relations among remembered values before compressing them into the residual stream.

A simpler first arm is to expose a compact recurrent-readout summary directly to the block's SwiGLU. This follows the existing state-aware-MLP direction. It does not add literal state, but it tests whether access, rather than storage, is the bottleneck.

## Reduce projection waste conservatively

Projection reduction should be used to fund more residual width and state, not pursued as an end in itself.

Low-risk candidates include:

1. derive erase/write gates from already projected key/value channels using shared per-head transforms;
2. use a shared low-rank residual basis for multiple gate projections;
3. factor QKV only after an iso-parameter control confirms that rank is not being removed too aggressively;
4. group projections into recurrent lanes, with periodic shared mixing.

The proven split-edit behavior should not be removed merely to save parameters. Each reduction needs an equal-parameter or equal-FLOP behavioral comparison.

## Recurrent lanes

A compatible extension of homogeneous heads is to partition the residual channels and heads into four or eight identical lanes. Each lane uses the same recurrent cell and local projection pattern. Channel permutations or occasional shared mixers allow information to cross lanes.

This approximates several RNNs working together while retaining one uniform architecture. It can reduce dense projection cost because most maps become block structured. It also gives gradient descent separable state subspaces without prescribing their semantic roles.

This is preferable to routing among 64 recurrent experts at every token. Token-level recurrent routing creates difficult state-identity, cache, and training problems. If recurrent expertization is revisited later, routing should be stable over segments or conversations.

## Fixed local attention

A small sliding-window attention layer keeps exact keys and values for the most recent `W` tokens in a ring buffer. Its cache remains constant with transcript length. It is good at exact recent copying, paths, syntax, and tool-output use.

However, it introduces a second memory mechanism and an abrupt eviction boundary. The current preference is therefore:

- do not make local attention part of the first foundation hypothesis;
- retain it as a controlled hybrid arm;
- use grouped-query attention and a small 512–1,024-token window if tested;
- require that the recurrent-only model remain the architectural control.

Local attention can compensate for poor recurrent learning. It should not obscure whether the redesigned recurrence learned to remember.

## Literal or external memory

Literal episodic memory is attractive for exact hashes, paths, command outputs, and evidence. A pointer-based store could retain immutable token spans and retrieve them into a bounded working window.

It is also difficult:

- the write and eviction policy must be trained or supervised;
- retrieval creates another failure surface;
- untrusted stored text can become prompt injection;
- branch and provenance semantics must be exact;
- training/inference parity is hard;
- an external store can hide a weak foundation model.

It should remain an agent-system layer, not the foundation's primary memory mechanism. Pi can later provide verified append-only evidence memory. The base model should first demonstrate that it can learn internal retention and retrieval under a simple homogeneous recurrence.

## Candidate 3–4B seed

A plausible first design region is:

```text
residual width:       2,560–3,072
layers:               18–24
heads/layer:          96–128
state side length:    48 or 64
state banks:          1 and 2 as primary arms
cell:                 homogeneous E97 split-edit derivative
post-mixer MLP:       shared SwiGLU in every layer
readout:              linear control plus one small nonlinear-summary arm
local attention:      absent in baseline
```

One indicative configuration is `d=3,072`, 20 layers, `H=96`, `n=64`. With the current full projection/gate family and a normal SwiGLU, it lies around the 4B range. One state bank would have about 7.9 million recurrent values; two banks about 15.7 million, roughly 63 MiB of FP32 state/session. Exact counts must come from instantiated graphs.

This configuration is not an immutable recipe. It is a center point for parameter, FLOP, HBM, and kernel studies.

## Candidate 7–8B scale

If the 3–4B family demonstrates better state use and behavioral scaling, expand toward:

```text
residual width:       3,584–4,096
layers:               24–32
heads/layer:          96–128
state side length:    64
state banks:          1–2
```

An 8B model is a safer scale for broad usefulness than 3–4B, but only if it can receive sufficient high-quality tokens. A rough 20-token/parameter lower-bound implies approximately 160B tokens for 8B parameters. A several-week single-node run should be selected only after measured sustained throughput establishes the reachable token budget.

An undertrained 8B model with the old projection bottleneck is not preferable to a well-trained 4B redesign.

## HETU experimental ladder

Before a multi-week seed:

1. implement an instantiated graph/parameter/cache/FLOP calculator;
2. benchmark `n=32`, `48`, and `64` recurrent kernels;
3. run iso-parameter controls for fewer/larger heads;
4. test one versus two shared-projection state banks;
5. test linear versus compact nonlinear readout;
6. test conservative gate factorization only after the above;
7. retain the current E97 cell as a control.

Short proxy runs should compare more than validation loss:

- distant retrieval accuracy and margin;
- exact copying at varying delays;
- objective retention after distractors;
- state sensitivity and state-ablation loss;
- state saturation/health;
- multi-turn action grammar;
- error-to-correction transitions;
- held-out semantic benchmarks.

The final long seed should be chosen from behavioral scaling, not from loss alone.

## Decision

The next foundation research program should optimize persistent hidden-state capacity per parameter and per unit of projection compute. The preferred path is a simple homogeneous E97-derived recurrence with fewer larger heads and shared-projection state banks. Complexity should be added only when a controlled experiment demonstrates that the simple recurrence cannot learn the required retention or access policy.

The guiding criterion is:

> Make remembering an easy behavior for the model to learn, while leaving the content, organization, and use of memory to the model itself.
