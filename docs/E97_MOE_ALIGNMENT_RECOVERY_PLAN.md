# E97-MoE alignment and long-context recovery plan

Status: **active execution plan**  
Created: 2026-08-15  
Current implementation commit at creation: `00d9f65d`  

This document is the execution authority for evaluating and recovering assistant
behavior and usable long-context memory in the trained E97 35B MoE. Update it
as evidence arrives. Do not replace failed or ambiguous results with informal
memory, and do not begin a large continuation merely because a small diagnostic
is disappointing.

## Current conclusion

The preserved 282.070B checkpoint is the provisional masked-SFT parent: a viable
pretrained language model, not an aligned assistant and not a demonstrated
long-context retriever. Corrected trajectory evaluation `5275758` found that it
has the best paired WikiText and external assistant-response likelihood of the
four authorities, while remaining near chance on natural-filler retrieval. The
304.619B checkpoint received 22.549B tokens of
instruction-shaped **all-token causal continued training**, not conventional
assistant-masked SFT. The first matched evaluation did not show useful assistant
behavior and found modest general-language and retrieval regression after that
continued-training phase.

The long-context phase used full BPTT, but almost all of its 31.273B tokens used
32K context at an accidental learning rate of `1e-4` rather than the original
`0.001007`. On the first synthetic passkey probe, the 282B checkpoint succeeded
at 2K and was approximately chance from 8K through 30K. This is evidence against
claiming usable long-context retrieval, but it is not yet a measured 250B-to-282B
adaptation delta and the repetitive filler may be unusually hostile to recurrent
state.

No further generic long-context continuation or all-token instruction
continuation is authorized until the gates below are evaluated.

## Preserved authorities

| Name | Step | Accepted tokens | Path |
|---|---:|---:|---|
| Base | 2,338,080 | 250,797,359,104 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/scale-ladder-mainfc52/milestones/step-02338080-tokens-0000250797359104` |
| Long-context | 2,338,536 | 282,070,089,728 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728` |
| Intermediate instruction | 2,338,816 | 300,860,571,648 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/instruction-32k-lr1007e-3-k8-v1/milestones/step-02338816-tokens-0000300860571648` |
| Final instruction | 2,338,872 | 304,618,668,032 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/instruction-32k-lr1007e-3-k8-v1/milestones/step-02338872-tokens-0000304618668032` |

Treat every authority as immutable. New branches and evaluation outputs must use
new directories.

## Existing paired-evaluation evidence

Job `5274663` evaluated 282B and 304B on two nodes with one complete eight-GCD
EP island per checkpoint. Both GPU runners completed successfully. Slurm marked
the wrapper `FAILED 2:0` only because a trailing shell continuation passed
`printf` as an argument to the CPU comparison script. The comparison was
recovered on the login node, and commit `00d9f65d` fixed the launcher. No GPU
rerun is required for this evidence.

Frozen panel SHA-256:
`d00a54107fe6b863232c335171688f2d5aa8250907453eda69ca6c90cb304445`

Results:
`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v1/runs/job-5274663/results/`

Final-minus-pre-instruction observations:

- WikiText NLL: approximately `+0.022` nats/token at 2K, 8K, and 32K;
- MMLU: `20.31% -> 18.36%` on the 256-example diagnostic;
- normalized HellaSwag: `44.53% -> 44.14%`, effectively flat;
- 2K synthetic retrieval: `100% -> 71.88%`;
- 8K through 30K synthetic retrieval: approximately chance for both;
- recurrent states: fully finite;
- peak evaluation HBM: approximately 13GB;
- generic greedy generations: incoherent and dominated by short fragments and
  repeated newlines for both checkpoints.

Caveats:

- generation stopped on EOT but not RS token 218;
- generic prompts did not cover every native serialization, including
  `Assistant reasoning:` trajectories;
- benchmark contamination is not ruled out;
- synthetic retrieval used repetitive filler;
- 250B and 300.861B were not included.

These caveats require corrected evaluation; they do not justify calling either
checkpoint an assistant.

## Execution phases

### Phase 1 — corrected four-checkpoint evaluation

Run one four-node Debug job, one eight-GCD EP island per authority. Freeze one
panel and use byte-identical examples for all checkpoints.

Required panels:

1. WikiText NLL at 2K, 8K, and 32K with loss by position.
2. MMLU and HellaSwag diagnostics.
3. Exact corpus-native prompt formats:
   - `System:` / `User:` / `Assistant:`;
   - `Assistant reasoning:` followed by `Assistant:`;
   - native tool-call and JSON formats.
4. Greedy generation and fixed-seed top-p sampling.
5. Generation termination on either RS token 218 or EOT.
6. Assistant-response likelihood on externally held-out prompts/responses.
7. Strengthened retrieval:
   - varied natural-text filler;
   - single and multiple key/value tasks;
   - distances 2K, 4K, 8K, 16K, 24K, and 30K;
   - multiple query positions and distractors;
   - RS-separated records;
   - real long-document QA where feasible.
8. Router load and recurrent-state health by layer and position.

Deliverables:

- four result JSON files;
- paired deltas for 250B->282B, 282B->300.861B, and
  300.861B->304.619B;
- a human-readable generation comparison;
- scheduler evidence naming both `Partition=batch` and `QOS=debug`.

Decision:

- 282B is the default SFT parent unless corrected evidence or a matched SFT
  canary proves that 304B is materially more alignable.

### Phase 2 — immutable assistant-masked SFT corpus

The first assistant-foundation authority uses the stock
`allenai/tulu-3-sft-mixture` at immutable revision
`b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`. This is the published mixture
used for Tülu 3, including its 70B SFT model, and avoids inventing another
mixture before proving that masked SFT works. The six pinned parquet shards
contain 939,343 observed rows, 1,110,933 assistant messages, 1,110,918 user
messages, and 795 system messages; every record ends in an assistant message.
The card advertises 939,344 examples, so receipts must preserve the observed
file count rather than silently adopting the card total.

The raw pinned authority is downloaded at:

`/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/raw`

Tülu 3 is ODC-BY overall, but component sources have mixed licenses and some
are noncommercial. Preserve the upstream README, source field, component
counts, revisions, and digests. This is a research artifact, not a blanket
license conclusion.

The first canary uses this stock mixture without custom reweighting. A later
long-instruction phase may add genuine retrieval/document-QA data only after
ordinary assistant behavior is demonstrated. Smol-Magpie-Ultra is the reserved
Apache-2.0 cleanliness fallback if Tülu results are ambiguous.

Build a record-aware tokenized authority. Do not use random byte windows.

Target-mask contract:

| Span | Target loss |
|---|---:|
| system text | 0 |
| user text | 0 |
| assistant header | 0 |
| assistant content | 1 |
| assistant tool call | 1 |
| tool result | 0 |
| terminal RS | 1 |

The immutable artifact must record token IDs, one mask bit per target token,
record offsets, conversation boundaries, source identity, train/held-out split,
total tokens, assistant target tokens, tokenizer identity, and SHA-256 digests.
Held-out membership must be selected by hash before training selection.

Pack complete records into 32K windows. RS remains visible and does not reset
recurrent state. Any deterministic slicing of long records must retain role and
mask metadata.

### Phase 3 — masked-SFT implementation

Extend the proven E97-MoE runner with:

- deterministic record-aware packing and resume;
- token-aligned assistant masks;
- chunked masked cross-entropy;
- explicit total-token and assistant-target-token accounting;
- correct normalization with variable target counts;
- RS/EOT-aware generation;
- canonical sampler identity and checkpoint metadata;
- fresh optimizer state for a new SFT phase.

Preserve the graph, fused MoE path, 32K full BPTT, 16K bounded MoE execution,
eight-way node-local EP, K8 DiLoCo, router objective, and canonical eight-shard
checkpoint publication.

The official Tülu 3 full-parameter anchors are `5e-6` for Llama 3.1 8B and
`2e-6` for Llama 3.1 70B, both at 4K for two epochs with 3% warmup and no weight
decay. E97 uses a different optimizer and sparse graph, so these are canary
anchors rather than copied truth. Initial E97 candidates are `2e-6` and `5e-6`.
Reusing `0.001007`, or beginning at `1e-5`/`3e-5`, is forbidden without separate
evidence.

### Phase 4 — qualification

CPU tests must cover mask alignment, header exclusion, assistant/RS inclusion,
tool-result exclusion, packed boundaries, deterministic sampling, exact
resume/retry, accounting, and chunked-loss parity with an oracle.

Frontier qualification sequence:

1. one-node forward/backward;
2. fresh-process checkpoint restore;
3. eight-node 4K/K64 run (token-distance equivalent to 32K/K8);
4. variable assistant-token counts;
5. finite state and zero dropped-token gates;
6. complete canonical checkpoint publication;
7. explicit scheduler evidence for partition and QoS.

### Phase 5 — matched masked-SFT LR canary

Corrected evaluation rejected 300B/304B as primary parents: all-token instruction
continuation worsened external assistant-response and WikiText likelihood and did
not reveal native-template assistant behavior. Qualified source uses one 32-node
Debug allocation split into independent fixed worlds:

- 16 nodes: masked SFT from 282B at `2e-6`;
- 16 nodes: identical masked SFT from 282B at `5e-6`.

Each arm uses complete-record 4K packs, 512 updates, K64, eight target-weighted
DiLoCo merges, and fresh optimizer state. Expected accepted exposure is about
225M packed tokens and 160M assistant targets per arm, measured exactly at
runtime. Use identical data coordinates and do not infer quality from training
loss alone. Keep 250B as rollback/control; run a small 250B arm only if masked
SFT from 282B fails.

A selectable checkpoint must show coherent greedy responses, reliable RS/EOT
termination, improved held-out assistant NLL and constraint following, no
newline collapse, healthy routing, and bounded general-language regression.
Use `+0.03` WikiText nats/token as an initial regression warning boundary, not
an automatic scientific law.

### Phase 6 — LR and long-supervision canaries

Select between the matched `2e-6` and `5e-6` arms, then compare ordinary masked
SFT against the same mixture containing 10–15% long examples whose target truly
depends on distant information.

If targeted supervision rapidly improves 4K/8K retrieval, continue. If it
remains near chance, inspect recurrent-state decay and reproduce on the 1.3B E97
proxy before spending more tokens. Treat persistent failure as a possible
architectural capacity limit, not an invitation for blind continuation.

### Phase 7 — bounded production alignment

Only after canary acceptance:

1. assistant foundation: 0.5–1B total tokens;
2. long-instruction specialization: 0.5–1B tokens;
3. agent/tool specialization: 0.2–0.5B tokens.

Expected useful scale is 1–3B tokens, with intermediate evaluation and early
stopping. Preference optimization follows only after coherent supervised
behavior exists.

## Non-negotiable operational controls

- Work from clean pushed `main`; production jobs use immutable source snapshots.
- Do not overwrite or relabel preserved authorities.
- GPU MoE remains fused and fail-closed.
- Use eight ranks per node and one GCD per rank.
- Debug acceptance uses `Partition=batch`, `QOS=debug`, and `Requeue=0`.
- Record both partition and QoS while live and in terminal accounting.
- Do not infer held-out quality from training-stream loss.
- Do not change production LR or stop/modify a scheduler job without explicit
  operator agreement.
- Preserve compute for evaluation and final alignment.

## Active checklist

- [x] Preserve 250B, 282B, 300.861B, and 304.619B authorities.
- [x] Run initial paired 282B/304B evaluation (`5274663`).
- [x] Recover comparison and fix postprocessing launcher (`00d9f65d`).
- [x] Build corrected four-checkpoint panel (`emender-e97-moe-paired-eval-panel-v2`, SHA-256 `fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f`).
- [x] Run and analyze corrected four-node evaluation (`5275758`, `COMPLETED 0:0`, 17m05s, `Partition=batch`, `QOS=debug`); select 282B provisionally and reject further all-token instruction continuation (`docs/validation/e97-moe-trajectory-eval-job5275758.md`).
- [x] Select and freeze stock Tülu 3 masked-SFT source at revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`.
- [x] Download and inspect all six raw parquet shards (939,343 observed records; roles restricted to system/user/assistant; every record ends assistant).
- [x] Build and independently validate immutable token-plus-mask SFT artifact (job `5276678`; manifest `e2461a28...`).
- [x] Implement masked objective and record-aware sampler (`18c8eae0`; focused CPU parity passes).
- [x] Complete one-node, fresh-process restore, and eight-node 4K/K64 qualification (jobs `5276974`, `5277148`, `5277224`; `docs/validation/e97-moe-masked-sft-qualification.md`).
- [x] Run matched 282B `2e-6`/`5e-6` 32-node masked-SFT canary (`5277510`) and corrected three-node evaluation (`5280428`); 5e-6 wins likelihood but neither arm is coherent and both collapse final-layer routing (`docs/validation/e97-moe-masked-sft-canary-job5280428.md`).
- [x] Test exact continuation of the mature 282B ScheduleFree state at preserved `1e-4`, overridden `1.007e-3`, and overridden `5e-6`; every arm regressed held-out assistant NLL within 8–64 updates, so none may scale (`docs/validation/e97-moe-sft-preserved-optimizer-canaries.md`).
- [ ] Qualify cached/full-prefix decoding parity, then select a full-capacity fresh-state SFT correction; low-rank adaptation is excluded. Retain 250B as fallback control.
- [ ] Run long-supervision canary.
- [ ] Authorize or reject bounded production SFT.

## Current next action

Stop 35B continuation. Exact preservation of mature ScheduleFree history has
now been physically rejected at `5e-6`, stored `1e-4`, and `1.007e-3`; do not
scale those branches. First prove cached-versus-full-prefix decoding parity and
exact-format held-out Tülu generation. Any subsequent adaptation must retain
full model capacity and use fresh objective-appropriate state; low-rank
adaptation is excluded. Qualify update precision on the 1.3B proxy and one-node
35B path before another multinode SFT canary.
