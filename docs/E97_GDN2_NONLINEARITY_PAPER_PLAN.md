# E97–GDN2 Nonlinearity Study and Long-Context Follow-up

**Status:** operator-authorized autonomous execution plan for the scoped three-arm study
**Date:** 2026-08-08  
**Primary question:** Does E97's nonlinear matrix-state update improve language
modeling over (a) the same E97 transition with the state `tanh` removed and
(b) the strongest closely related established recurrent baseline, when all
systems see literally identical training tokens?

## 1. Scientific scope

The core study has three approximately 1.3B-parameter arms:

1. **E97:** canonical split key-axis erase/read plus value-axis delta-write,
   with the nonlinear matrix-state update.
2. **E97-linear:** the exact same instantiated E97 architecture and training
   recipe, changing only `linear_state=False` to `linear_state=True`.
3. **GDN2-MLP:** official-style GDN2 mixer plus SwiGLU MLP, using the best
   retained CMA-ES configuration as the established neighboring baseline.

M²RNN is not a core arm. Its public XMA implementation currently identifies
M²RNN as unsupported on ROCm. A local HIP implementation would create an
avoidable implementation-fidelity confound and reviewer surface. It remains an
optional engineering stress goal, not evidence required by this paper.

The base study trains at context 2,048 to an immutable 50.332B-token milestone.
Long-context adaptation is a separate second phase forked from those immutable
base checkpoints. It must not alter or overwrite the base comparison.

## 2. Architecture decision

### 2.1 Primary E97 geometry

Use the retained canonical-E97 CMA-ES winner:

```json
{
  "dim": 2176,
  "n_heads": 170,
  "n_state": 32,
  "depth": 14,
  "lr": 0.0010403731352768883,
  "batch_size": 2
}
```

The retained leaderboard reports approximately 1,274.7M parameters. Before a
training submission, instantiate from the exact production graph and record the
exact parameter count, tensor schema, initialized-state digest, and rendered
arguments.

This is deliberately canonical E97 rather than `e97-raw`: the raw-write winner
removes the delta correction that is central to the proposed E97 mechanism.

### 2.2 Matched E97-linear ablation

The primary ablation clones the canonical E97 geometry, initialization policy,
optimizer, learning rate, B2 microbatch, DiLoCo policy, and all non-state
components. It changes only:

```text
linear_state: false -> true
```

At the state activation this means:

```text
S_t = tanh(pre_t) -> S_t = pre_t
```

and the backward pass drops only the `tanh` derivative. The whole network is
not a linear model: its input-dependent gates, normalization, projections,
MLPs, and other activations remain nonlinear. “Linear-state E97” is the precise
term. IEEE floating-point rounding means the computed transition does not obey
exact real-arithmetic linear identities, but rounding is not treated as a
learned source of nonlinear expressivity.

A separate CMA-ES `e97-linear` winner does exist:

```json
{
  "dim": 2176,
  "n_heads": 224,
  "n_state": 32,
  "depth": 11,
  "lr": 0.0010531703750126676,
  "batch_size": 4
}
```

It is substantially different: head count, depth, microbatch, parameter count,
and update cadence differ from canonical E97. It estimates an independently
optimized linear-state architecture's ceiling, but it does **not** isolate the
`tanh`. Therefore it is not the primary ablation. If allocation remains after
the three-arm study, train it as a clearly labeled secondary sensitivity arm.

Historical Frontier evidence also records that an older E97-linear chunked
smoke failed parity/finiteness. Current-source linear forward/backward parity
and sustained ROCm execution are mandatory; nonlinear E97 readiness cannot be
silently inherited by the ablation.

### 2.3 GDN2-MLP geometry

Use the retained GDN2-MLP CMA-ES winner:

```json
{
  "dim": 2304,
  "expansion": 2,
  "depth": 17,
  "n_heads": 8,
  "mlp_ratio": 2.854220752778522,
  "lr": 0.0003907570359771844,
  "batch_size": 2
}
```

Retained parameter count: 1,285,245,320. Its CMA-ES rank was based on 64
evaluations and was more provisional than the 100-evaluation E97 families, so
the paper must describe that history honestly. It remains the most defensible
closely related baseline.

The GDN2 source must not be an ambient checkout. Production binds commit
`95709fc250357c2dd109361c353192f2aa5913f9` or a subsequently reviewed immutable
replacement, stages it with the job source, and records its digest. Python/eager
fallback is forbidden in GPU production.

## 3. Canonical CommaPile data

Canonical source:

```text
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
```

Recorded facts:

- 1,000,000,725,401 decoded bytes (approximately 932 GiB);
- 12,308,526,802 lines;
- approximately 214.5M payload records from 31 sources;
- records separated by byte `0x1e`;
- corpus-construction/interleaving seed 42;
- manifest SHA-256
  `44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9`;
- compressed source size 251,655,400,225 bytes;
- weighted/interleaved main-stage mixture, led by peS2o, Stack Exchange,
  Stack-v2 educational, Wikimedia, CCCC, and GitHub Archive.

The project describes this as the license-clean releasable corpus. The paper
must cite the corpus/manifest rather than treating the local filename as data
provenance. A one-time batch verification must recompute the canonical source
hash before deriving a new scientific stream receipt.

The tokenizer is pinned `p50k_base`, including the exact cache artifact and its
SHA-256 (`94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`).

## 4. Deterministic identical-token stream

### 4.1 Current defect and required replacement

The current tokenized loader selects random byte positions from mutable RNG
state. The production resume workaround seeds approximately as
`42 + restored_step + rank`. That avoids replaying the oldest prefix but does
not reproduce the exact uninterrupted next sample, and models with different
step clocks need not consume identical samples.

The replacement is a versioned counter-based sampler. The fixed sampler key may
be 42, but **accepted tokens are a cursor, not a changing RNG seed**. Each
sample is a pure deterministic function of:

```text
sampler_schema
canonical_corpus_digest
tokenizer_digest
fixed_sampler_key
data_world_size
global_rank
absolute_rank_sample_index
bounded_retry_index
```

It preserves the current byte-window distribution initially: deterministic
byte position, fixed read size, UTF-8 replacement policy, drop-first-token
rule, and fixed p50k sample length. Retry positions are derived from the same
sample identity, so an unusually short/invalid read cannot perturb later
samples. A future record-aware or pretokenized corpus is a new sampler schema,
never a silent behavior change.

For fixed context 2,048 and world size `W`:

```text
absolute_rank_sample_index = total_accepted_tokens / (W * 2048)
```

This must divide exactly at every authoritative resume. Each B2 step consumes
two consecutive sample indices per rank. A checkpoint stores the sampler
schema/key, corpus/tokenizer digests, data world, predicted tokens per sample,
absolute cursor, accepted-token clock, and transition origin if an older run is
migrated.

Only successfully merged, canonically checkpointed tokens are accepted.
Provisional work after the last authority is not counted. A retry from the same
checkpoint deliberately reproduces precisely that unaccepted provisional
work.

### 4.2 Scope of application

The same sampler implementation and metadata contract must be used by:

- generic dense `train.py` (E97, E97-linear, and GDN2);
- E97 MoE production in `scripts/frontier/e97_35b_moe_train.py`;
- held-out/fixed-stream generators where applicable.

Existing checkpoints are never relabeled as having used the new schema. The
currently running MoE job remains tied to its immutable source and old stream.
A future MoE continuation may transition only at a complete K40 checkpoint,
records the old/new sampler boundary explicitly, and thereafter obtains exact
retry/resume behavior.

The scientific three-arm run starts from scratch at sampler cursor zero with a
fixed 2,048-rank data world. Systems-ladder checkpoints at smaller worlds are
throwaway qualification artifacts and are not promoted into the scientific
training chain.

### 4.3 Literal equivalence claim

With identical B2, context 2,048, world 2,048, sampler identity, and accepted
cursor, the three arms receive the same input and target token tensors, in the
same order, on the same global ranks. This is stronger than “same distribution.”
Training may still differ because architecture, kernel arithmetic, and
model-specific learning rate differ.

Add tests proving:

1. uninterrupted and checkpoint-resumed streams are byte-for-byte equal;
2. retries reproduce unaccepted work;
3. different ranks are deterministic and distinct;
4. E97/GDN2/MoE call sites produce the same sample IDs and tensors;
5. inconsistent token cursor, world, corpus, tokenizer, or schema fails closed;
6. counter generation does not depend on batch grouping;
7. sampler metadata survives checkpoint publication and fresh-process restore.

## 5. Matched token and DiLoCo arithmetic

All primary arms use B2, context 2,048, 2,048 GCDs, and K40:

```text
accepted tokens/step
  = 2048 ranks * 2 samples * 2048 tokens
  = 8,388,608

accepted tokens/K40 merge
  = 335,544,320
```

Exact milestones:

| Milestone | Steps | K40 merges | Accepted tokens |
|---|---:|---:|---:|
| systems only | exact configured multiple of 40 | configured | recorded |
| base paper checkpoint | 6,000 | 150 | 50,331,648,000 |
| optional continuation | 12,000 | 300 | 100,663,296,000 |

The initial commitment is 50.332B per arm. Preserve all three authorities and
run fixed-stream evaluation before approving continuation. If evidence and
allocation support it, continue the same chains to 100.663B; do not restart.
Runtime is measured, not assumed from MoE throughput. The qualification ladder
must establish per-arm step, merge, and checkpoint time before reserving the
normal-QoS envelope.

## 6. Implementation and qualification sequence

### Phase A — freeze identities and implement the sampler

1. Recompute/verify the CommaPile digest in a scheduler job and publish a
   durable corpus receipt.
2. Implement the counter sampler once in `ndm/data/tokenized_dataset.py`.
3. Thread accepted-token cursor and metadata through dense and MoE checkpoint
   paths.
4. Add deterministic cross-model/restart tests.
5. Freeze exact argument JSON, parameter schemas/counts, initialization seeds,
   source commit, external-source digest, tokenizer, corpus, sampler, and
   optimizer/DiLoCo policy for each arm.

### Phase B — one-GCD and one-node kernel gates

For every arm:

1. CPU/reference or trusted-kernel output and gradient parity where available;
2. bf16 finite forward/backward and optimizer step on gfx90a;
3. fail-closed confirmation against Python/eager GPU fallback;
4. changing-shape/long-loop compiled-module-cardinality test for HIP-209;
5. exact model instantiation and parameter count;
6. one-node eight-GCD sustained run through several K40 merges;
7. checkpoint, fresh-process restore, and identical next-batch proof;
8. peak HBM, ordinary-step throughput, loss, and merge timing.

GDN2 additionally requires iterative source compatibility work. Its dynamic
T/N/B kernel arguments already use `do_not_specialize` in important upstream
kernels, but this is not a Frontier qualification. Audit all current call paths,
pin ROCm-safe autotuning, test forward/backward, and retain compile-cache
cardinality. Do not scale until it passes sustained one-node training.

E97-linear receives the same scrutiny because of its historical failed
chunked-ROCm smoke.

### Phase C — systems ladder

The operator has explicitly authorized autonomous progression for this scoped
study without a 128-node rung or a separate human gate at each transition. The
execution ladder is:

1. **8 nodes:** approximately 20-minute safety envelope, exact positive
   K-aligned steps; validate data cursor, loss, kernels, merge, and checkpoint.
2. **32 nodes:** same exact source and identities, 20–30-minute envelope.
3. **256 nodes / 2,048 GCDs:** advance automatically only after the 32-node
   machine criteria pass; run long enough to demonstrate multiple merges, at
   least one periodic checkpoint, stable loss, sustained HBM, and bounded
   compiled-module count.

A failed rung is diagnosed and retried autonomously from immutable evidence;
unchanged unexplained failures do not advance. No 128-node allocation is
required for these arms.

These are throwaway from-scratch systems runs. They do not become paper model
initialization. Every run uses eight ranks/node, one GCD/rank, fixed world,
`Partition=batch`, separately verified QoS, `Requeue=0`, fail-stop child
execution, exact positive `MAX_STEPS`, `TRAIN_MINUTES=0`, and K-aligned saves.
Debug QoS is limited to its scheduler envelope; sustained production uses
normal QoS. Retain both live and terminal `Partition` and `QOS` evidence.

### Phase D — three production arms

After all exact-source gates pass, submit the three immutable 6,000-step
production specifications. Queue concurrency is an operational opportunity,
not an assumption: retain scheduler eligibility/start evidence and do not allow
one arm's queue timing to alter its data/model identity.

For each arm retain:

- scheduler binding and terminal state;
- exact source/config/external dependency identities;
- initialization receipt;
- sampler/corpus/tokenizer receipt;
- step/token/merge/checkpoint clocks;
- ordinary-step, merge, and checkpoint timings;
- training loss and numerical health;
- HBM and compiled-kernel-cardinality evidence;
- complete canonical checkpoint authority and immutable 50.332B milestone.

No inline held-out evaluation runs inside production training.

## 7. Evaluation plan

Evaluate the initial models, all three 50.332B checkpoints, and any later
100.663B checkpoints on identical immutable inputs.

Required categories:

- held-out next-token cross-entropy, perplexity, and tokenizer-independent BPB;
- fixed language-model benchmarks selected before inspecting arm results;
- recurrent/state-tracking and controlled synthetic tasks;
- loss-versus-accepted-token and loss-versus-compute curves;
- throughput, HBM, merge/checkpoint overhead, and node-hour cost;
- multiple evaluation bootstrap slices or confidence intervals;
- failure/kernel evidence reported separately from model quality.

Training loss is never substituted for held-out evidence. The held-out corpus
must be provenance-separated from the training artifact, immutable, and hashed.

Primary comparisons:

1. E97 versus matched E97-linear: controlled effect of state `tanh`.
2. E97 versus GDN2-MLP: proposed architecture versus established close baseline.
3. E97-linear versus GDN2-MLP: behavior of two linear-state neighboring systems.

The independently CMA-optimized E97-linear configuration, if run, is a
secondary sensitivity analysis and must not replace the matched ablation.

## 8. Long-context phase

Long-context work begins only after base-context checkpoints and evaluation are
immutable. It is a fork, not a continuation that overwrites base evidence.

First evaluate zero-shot length behavior at 2K, 4K, 8K, 16K, and the maximum
safe length. Then define a matched adaptation grid using the same long-document
corpus/sampler, accepted-token budget, and token-based DiLoCo merge interval for
all arms. Context changes alter samples/step, so K is chosen per context to
match **tokens per merge**, with exact boundary arithmetic recorded before
submission.

Tune on a development suite, not the final held-out suite. Candidate variables
include context curriculum, adaptation learning rate, and checkpoint/token
budget. Architecture-specific maximum safe microbatching may use gradient
accumulation, but effective tokens/update and tokens/merge must be matched and
reported. Preserve both pre-adaptation and post-adaptation checkpoints.

The long-context report separates:

- zero-shot extrapolation from 2K training;
- gains from matched long-context adaptation;
- quality versus context length;
- state stability/numerical range;
- throughput and memory scaling;
- retrieval/state-tracking versus ordinary language modeling.

No long-context result is required to decide whether the base 50B nonlinear
ablation succeeded.

## 9. Stop/go criteria

Stop an arm before scale or production on eager fallback, nonfinite values,
unbounded kernel specialization, inconsistent sampler metadata, mismatched next
batch after restore, checkpoint corruption, rank/world drift, or an unexplained
loss discontinuity. A failed fixed world publishes no emergency checkpoint.
Resume only from the newest complete accepted K-boundary authority in a fresh,
non-requeueing job after automated evidence review.

Proceed from 50B to 100B only after all three 50B milestones exist and a fixed
held-out evaluation shows that continuation is scientifically useful. A systems
pass does not imply a quality result; a quality result does not waive systems
or data-integrity gates.

## 10. Deliverables

- versioned deterministic sampler implementation and tests;
- canonical CommaPile/tokenizer/sampler receipt;
- exact three-arm model/config/initialization manifests;
- GDN2 immutable source bundle and ROCm qualification;
- E97-linear current-source parity qualification;
- 8/32/256 systems reports per newly qualified kernel path;
- three immutable 50.332B checkpoint authorities;
- held-out base-context comparison report;
- optional three-arm 100.663B continuation report;
- separate long-context adaptation plan, checkpoints, and report.

## Validation and architecture conformance

This plan conforms to `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003
(2026-07-31 plus the 2026-08-04 one-epoch/no-requeue safety decision), and the
ADR-003 crosswalk in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

Applicable production safety requirements are **R07** (atomic checkpoint
intent), **R12** (stable authoritative restart), **R14/NDP13** (bounded
fail-stop execution), and **NDP15 checkpoint atomicity only**. The
elastic/native clauses of those rows are unclaimed. For this explicitly scoped
study, the operator replaces **R16**'s generic `8 -> 32 -> 128` promotion rule
with an autonomous exact-source `8 -> 32 -> 256` ladder. No general elastic or
other production-path promotion claim follows from this exception. **R02–R06, R08–R11; NDP01, NDP03–NDP12, NDP14,
NDP16–NDP17; V21S01–V21S17; and ISP01–ISP07** are explicitly retired/not
claimed for this fixed-world production study, as are NDP02's no-all-rank
property and background/apply clauses of NDP15.

The production path uses one bounded fixed-world child, never preserves,
shrinks, or automatically relaunches a broken communicator, contains no SQLite
or filesystem membership/heartbeat authority, publishes only complete atomic
K-aligned checkpoints, and starts a fresh non-requeueing job after automated
failure-evidence review.
There is no elastic minimum-progress floor: the required fixed-world floor is
the complete launched world; failure of any rank aborts provisional work.

Minimum local validation before any scheduler mutation:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_train_helpers.py \
  tests/test_e97_moe_production_launcher.py
```

The sampler work adds focused tests to that command. Model-specific ROCm tests,
launcher tests, rendered source/config checks, and exact scheduler commands are
recorded in each rung's validation report. Every Frontier receipt separately
names `Partition` and `QOS` in live and terminal evidence.
