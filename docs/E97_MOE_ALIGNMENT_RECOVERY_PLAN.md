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

The preserved 282.070B checkpoint is a viable pretrained language-model parent,
not an aligned assistant. The 304.619B checkpoint received 22.549B tokens of
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

Build a record-aware tokenized authority. Do not use random byte windows.

Initial target mixture:

- 50–60% clean single- and multi-turn instruction/chat;
- 15–20% reasoning and math;
- 15–20% direct code generation and debugging;
- 10–15% genuine long-context retrieval and document QA.

Terminal transcripts and long agent/tool trajectories must not dominate this
foundation stage. Add agent specialization only after ordinary instruction
behavior is demonstrated.

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

Initial learning-rate candidates are `1e-5` and `3e-5`. Reusing `0.001007` for
masked SFT is forbidden without separate evidence.

### Phase 4 — qualification

CPU tests must cover mask alignment, header exclusion, assistant/RS inclusion,
tool-result exclusion, packed boundaries, deterministic sampling, exact
resume/retry, accounting, and chunked-loss parity with an oracle.

Frontier qualification sequence:

1. one-node forward/backward;
2. fresh-process checkpoint restore;
3. eight-node K8 run;
4. variable assistant-token counts;
5. finite state and zero dropped-token gates;
6. complete canonical checkpoint publication;
7. explicit scheduler evidence for partition and QoS.

### Phase 5 — matched parent canary

Use one 16-node Debug allocation:

- eight nodes: masked SFT from 282B;
- eight nodes: identical masked SFT from 304B.

Initial budget: approximately 100M total tokens and 30–50M assistant target
tokens per arm at a common midpoint LR near `2e-5`. Use identical data and
ordering. Do not infer parent quality from training loss alone.

A selectable checkpoint must show coherent greedy responses, reliable RS/EOT
termination, improved held-out assistant NLL and constraint following, no
newline collapse, healthy routing, and bounded general-language regression.
Use `+0.03` WikiText nats/token as an initial regression warning boundary, not
an automatic scientific law.

### Phase 6 — LR and long-supervision canaries

On the selected parent, compare `1e-5` and `3e-5`, then compare ordinary masked
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
- [ ] Run corrected four-node evaluation.
- [ ] Select and freeze initial masked-SFT data sources and held-out split.
- [ ] Build immutable token-plus-mask SFT artifact.
- [ ] Implement masked objective and record-aware sampler.
- [ ] Complete CPU and Frontier qualification.
- [ ] Run matched 282B/304B parent canary.
- [ ] Select parent and LR.
- [ ] Run long-supervision canary.
- [ ] Authorize or reject bounded production SFT.

## Current next action

Implement the corrected four-checkpoint evaluation first. It is the cheapest
way to measure the 250B->282B long-context delta, the instruction trajectory,
and prompt-template sensitivity before changing training code.
