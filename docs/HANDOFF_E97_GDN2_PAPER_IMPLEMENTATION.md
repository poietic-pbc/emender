# Handoff: implement the E97-MLP / E97-linear-MLP / GDN2-MLP paper study

**Audience:** implementation agents taking over the paper experiment  
**Repository:** <https://github.com/spinozans/emender>  
**Authoritative branch:** `main`  
**Detailed design:** [`E97_GDN2_NONLINEARITY_PAPER_PLAN.md`](E97_GDN2_NONLINEARITY_PAPER_PLAN.md)

## Mission

Build and execute a defensible, identical-data comparison among:

1. **E97-MLP** — canonical nonlinear E97 recurrent mixer followed by the
   standard SwiGLU MLP in each layer;
2. **E97-linear-MLP** — the exact same graph and recipe, changing only the
   matrix-state activation from `tanh` to identity;
3. **GDN2-MLP** — the strongest retained closely related recurrent baseline.

Train all three from scratch on literally identical CommaPile token samples to
an immutable **50.332B accepted-token** milestone at context 2,048. Evaluate
all three on the same held-out artifacts. A continuation to 100.663B and
long-context adaptation are later decisions, not prerequisites for the base
study.

M²RNN is deliberately excluded. Its public XMA implementation is not supported
on ROCm, and a local reimplementation would add an unnecessary fidelity
confound.

## Live execution checklist

**Last updated:** 2026-08-08. This is the authoritative progress ledger for the
study. Update it in the same commit as each completed implementation or evidence
milestone. `[~]` means actively in progress; `[x]` requires committed evidence.

### Operating mode

- [x] Work is running directly in one attended session; no WG dispatcher or
  worker service is active.
- [x] Frontier debugging is serialized to at most one active debug-QoS job.
- [x] Independent MoE continuation job 5208321 is left untouched by this study.

### Phase A — identities and deterministic sampling

- [x] Recompute the canonical CommaPile and tokenizer hashes in bounded Frontier
  job 5207741. It completed `0:0` in 00:57:46 on `Partition=batch`, `QOS=debug`;
  both pinned digests matched. Receipt:
  `docs/validation/e97-paper-corpus-sampler-receipt.md`.
- [~] Implement the versioned counter-based sampler in
  `ndm/data/tokenized_dataset.py` with focused deterministic/resume tests.
- [ ] Thread sampler identity and accepted-token cursor through dense training
  and atomic checkpoint restore.
- [ ] Thread the same contract through MoE manifests and training without
  relabeling legacy checkpoints.
- [ ] Add cross-model/call-site tests and launcher/config receipts covering all
  eight sampler requirements.
- [ ] Freeze exact graph, parameter, tensor-schema, initialization, optimizer,
  corpus, tokenizer, sampler, and external-source manifests for all three arms.

### Phase B — kernel qualification

- [ ] E97-MLP current-source one-GCD and one-node qualification.
- [ ] Matched E97-linear-MLP parity, finiteness, specialization-cardinality,
  sustained K40, and restore qualification.
- [ ] Bind GDN2 commit `95709fc250357c2dd109361c353192f2aa5913f9` and
  complete ROCm kernel/fallback audit.
- [ ] GDN2-MLP parity, one-GCD, sustained eight-GCD, and restore qualification.

### Phase C — fixed-world systems ladders

- [ ] E97-MLP: 8 nodes → 32 nodes → 256 nodes.
- [ ] E97-linear-MLP: 8 nodes → 32 nodes → 256 nodes.
- [ ] GDN2-MLP: 8 nodes → 32 nodes → 256 nodes.

Each box is checked only after exact-source predecessor evidence passes. Ladder
checkpoints are throwaway and never enter the scientific chain.

### Phase D — production and evaluation

- [ ] E97-MLP immutable 6,000-step / 50,331,648,000-token authority.
- [ ] E97-linear-MLP immutable 6,000-step / 50,331,648,000-token authority.
- [ ] GDN2-MLP immutable 6,000-step / 50,331,648,000-token authority.
- [ ] Fixed-stream held-out comparison report for all three authorities.
- [ ] Post-base long-context adaptation plan.

## Read first

1. `docs/E97_GDN2_NONLINEARITY_PAPER_PLAN.md` — normative scientific and
   execution plan.
2. `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` — fixed-world production safety.
3. `docs/RESILIENT_DILOCO_GAP_MATRIX.md` — requirement crosswalk.
4. `docs/REPRODUCE_E97_RAW_1P3B.md` and
   `docs/E97_RAW_1P3B_LEADERBOARD.md` — CMA-ES provenance.
5. `docs/validation/e97-35b-moe-hip209-ragged-specialization-fix.md` — why
   runtime-varying Triton values must not become unbounded specializations.
6. `docs/EMENDER_E97_35B_MOE_IMPLEMENTATION.md` — current MoE hierarchy and
   checkpoint background.

Work directly from a clean, freshly fetched `main`. Keep `main` synchronized
with `origin/main`; do not accumulate essential work only on a feature branch.
Every accepted implementation/evidence change must be committed and pushed.

## Current operational state at handoff

The 35B MoE continuation is independent of this from-scratch study and is still
running as Frontier job **5201882** from an immutable source snapshot:

- 256 nodes / 2,048 ranks;
- `Partition=batch`, `QOS=normal`, `Requeue=0`;
- target step `2329120`, accepted tokens `100,473,503,744`;
- most recently observed around step `2328960`, approximately 97.789B accepted
  tokens, finite loss, and a successful K40 merge;
- the mutable repository checkout cannot change its loaded code.

Do not cancel, relabel, or modify that job. When it completes, separately verify
terminal scheduler evidence, final eight-shard authority, and preserve the
100.474B checkpoint as an immutable milestone. The deterministic sampler below
applies only to future executions and must not retroactively relabel old data
streams.

## Frozen primary model recipes

### E97-MLP and matched E97-linear-MLP

Use the canonical-E97 CMA winner for both arms:

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

Instantiate the exact current graph and record its exact parameter count and
tensor schema. E97-linear-MLP must reuse the same geometry, initialization
policy, optimizer, learning rate, microbatch, MLPs, and DiLoCo configuration.
The sole architectural change is `linear_state=True`.

There is a separately CMA-optimized linear configuration (depth 11, 224 heads,
B4), but it confounds the nonlinearity ablation. It is optional secondary work,
not a replacement for the matched primary arm.

Canonical E97 is used rather than `e97-raw`; raw-write removes the delta
correction central to the mechanism under study.

### GDN2-MLP

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

Retained parameter count: `1,285,245,320`. Bind the external GDN2 source to
commit `95709fc250357c2dd109361c353192f2aa5913f9`, or document and review an
explicit replacement. Do not rely on an ambient checkout. Stage the immutable
source with each job. GPU eager/Python fallback is forbidden.

## First implementation: deterministic accepted-token sampling

This is the immediate priority and must be shared by dense and MoE training.

The old stream uses mutable RNG state and approximately
`42 + restored_step + rank`. It avoids the oldest replay but does not reproduce
the exact uninterrupted next sample. Replace it with a versioned counter-based
sampler in `ndm/data/tokenized_dataset.py`.

Accepted tokens are a **cursor**, not a changing seed. A sample must be a pure
function of:

```text
sampler schema
canonical corpus digest
tokenizer digest
fixed sampler key (42 is acceptable)
data world size
global rank
absolute per-rank sample index
bounded retry index
```

For context 2,048 and fixed data world `W`:

```text
absolute_rank_sample_index = total_accepted_tokens / (W * 2048)
```

Fail closed unless division is exact. B2 consumes two consecutive indices per
rank. A retry from the same checkpoint must reproduce exactly the unaccepted
provisional batches. A successful continuation resumes at the exact next
sample.

Initially preserve the current sampling distribution: deterministic byte
position, fixed read size, UTF-8 replacement, drop-first-token behavior, and
pinned p50k tokenization. Record a new sampler schema for any future
record-aware or pretokenized representation.

Thread the sampler identity and cursor through:

- `train.py`;
- `scripts/frontier/e97_35b_moe_train.py`;
- dense checkpoint metadata;
- `ndm/e97_moe_checkpoint.py` manifests;
- launchers and source/config receipts.

The canonical data identity is:

```text
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
```

Manifest SHA-256:

```text
44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9
```

Pinned p50k cache SHA-256:

```text
94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069
```

Recompute the one-terabyte corpus hash in a scheduler job and retain a durable
receipt before scientific production.

Required sampler tests:

1. uninterrupted versus checkpoint-resumed tensors are byte-identical;
2. retry reproduces unaccepted work;
3. successful continuation advances to the exact next sample;
4. batch grouping does not alter sample identity;
5. different ranks are deterministic and distinct;
6. dense E97, GDN2, and MoE call sites agree on sample IDs/tensors;
7. schema, cursor, world, context, corpus, or tokenizer mismatch fails closed;
8. metadata survives atomic publication and fresh-process restore.

Existing checkpoints without the schema are legacy. Never silently claim they
used the new stream. A future MoE transition occurs only at a complete K40
checkpoint and records the old/new boundary explicitly.

## Kernel qualification before scaling

### E97-MLP

Nonlinear E97 already has substantial Frontier evidence. Re-run current-source
one-GCD and one-node gates after sampler integration; do not assume data-path
changes are operationally neutral.

### E97-linear-MLP

Treat as a separate kernel qualification. Historical evidence records an older
chunked linear-state parity/finiteness failure. Prove current-source:

- forward and backward parity;
- finite bf16 optimizer steps on gfx90a;
- sustained one-node execution through several K40 boundaries;
- bounded compiled-module cardinality;
- checkpoint restore and identical next batch;
- no eager GPU fallback.

### GDN2-MLP

Expect iterative ROCm work. The external kernels already use
`do_not_specialize` for important dynamic T/N/B arguments, but that is not a
production qualification. Required work:

1. bind and stage the immutable external source;
2. validate imports and remove CUDA-only assumptions;
3. audit every Triton signature and autotune key for unbounded HIP module
   growth;
4. pin ROCm-safe autotune candidates;
5. run output/gradient parity against a trusted implementation;
6. run one-GCD and sustained eight-GCD training;
7. measure HBM, throughput, loss, compile-cache cardinality, and checkpoint
   restore;
8. fail closed against Python/eager fallback.

Use the canonical Frontier environment before Python or tests:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest ...
```

## Autonomous scale ladder

For this scoped study the operator has authorized autonomous progression:

```text
8 nodes -> 32 nodes -> 256 nodes
```

There is no 128-node rung and no separate human promotion gate. Advance only
when machine-readable criteria pass; diagnose and retry failures autonomously,
but do not advance an unchanged unexplained failure.

Each systems run is throwaway and starts from scratch. Do not promote a
small-world checkpoint into the scientific chain because the data-world mapping
would differ.

Every Frontier run must use:

- eight ranks/node, one GCD/rank;
- fixed-world fail-stop execution;
- `Partition=batch` and separately verified `QOS`;
- `Requeue=0`;
- exact positive `MAX_STEPS` and `TRAIN_MINUTES=0` for multinode work;
- K-aligned stopping and checkpoint publication;
- immutable source/config/external dependency receipts;
- live and terminal scheduler evidence naming both Partition and QoS.

Suggested qualification envelopes:

1. 8 nodes: approximately 20 minutes and multiple K40 merges;
2. 32 nodes: 20–30 minutes, multiple merges, and a checkpoint;
3. 256 nodes: enough exact steps for sustained loss/HBM/throughput evidence and
   at least one periodic checkpoint.

## Production arithmetic

All three primary arms use B2, context 2,048, K40, and 2,048 ranks:

```text
8,388,608 accepted tokens/step
335,544,320 accepted tokens/K40 merge
```

Primary milestone:

```text
6,000 steps = 150 merges = 50,331,648,000 accepted tokens
```

Optional continuation:

```text
12,000 steps = 300 merges = 100,663,296,000 accepted tokens
```

Submit three immutable 6,000-step specifications only after the corresponding
kernel and scale gates pass. Parallel normal-QoS execution is allowed when the
scheduler permits it. Preserve each 50.332B authority outside rotating
retention before evaluation.

## Evaluation and long context

No inline held-out evaluation in production training. Evaluate initial models
and all 50.332B milestones on the same immutable held-out inputs:

- cross-entropy, perplexity, and tokenizer-independent BPB;
- predetermined language-model benchmarks;
- state-tracking and controlled recurrent tasks;
- loss versus accepted tokens and compute;
- throughput, HBM, checkpoint/merge overhead, and node-hours;
- confidence intervals or bootstrap slices.

The primary claims are:

1. E97-MLP versus matched E97-linear-MLP isolates the state `tanh`;
2. E97-MLP versus GDN2-MLP compares against the established close baseline;
3. E97-linear-MLP versus GDN2-MLP compares neighboring linear-state systems.

Long-context work is a later fork from immutable base checkpoints. First measure
zero-shot 2K/4K/8K/16K behavior. Then tune context curriculum and adaptation
learning rate on a development suite, matching accepted tokens and tokens per
DiLoCo merge across arms. Never overwrite the 2K base checkpoints or tune on
the final held-out suite.

## Definition of done

The implementation handoff is complete when the repository contains and
`origin/main` publishes:

- deterministic sampler code and cross-model resume tests;
- corpus/tokenizer/sampler receipt;
- exact three-arm graph/config/initialization manifests;
- GDN2 immutable source binding and ROCm qualification;
- E97-linear current-source qualification;
- 8/32/256 reports for each newly qualified path;
- three complete immutable 50.332B checkpoints;
- one fixed-stream held-out comparison report;
- a separately scoped long-context plan after base evaluation.

Do not report streaming training loss as held-out quality. Do not publish an
emergency checkpoint from a broken world. Do not infer data equivalence from
equal token counts alone: the sampler/corpus/tokenizer identities and absolute
sample cursor must also match.

## Validation and conformance

The fixed-world safety intent is from
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` ADR-003 and the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable requirements are **R07**
(atomic checkpoint intent), **R12** (authoritative restart), **R14/NDP13**
(bounded fail-stop execution), and **NDP15 checkpoint atomicity only**. Elastic,
native, async-v2.1, and immutable-snapshot overlap claims are not made.

For this scoped study, the recorded operator decision replaces **R16**'s generic
ladder with autonomous exact-source `8 -> 32 -> 256`. This does not authorize a
general elastic or unrelated production-path promotion policy.

At minimum, sampler changes must pass focused dataset/checkpoint tests plus:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_train_helpers.py \
  tests/test_e97_moe_checkpoint.py \
  tests/test_e97_moe_production_launcher.py
```

Add exact validation commands and artifacts to every implementation and runner
report. A clean test pass is preflight, not evidence that an unrun Frontier rung
or model-quality result exists.
