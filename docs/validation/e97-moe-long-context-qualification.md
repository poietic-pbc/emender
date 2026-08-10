# E97-MoE 32K/128K Frontier qualification

Date: 2026-08-09

## Scope

This receipt qualifies unchanged E97-MoE model parameters at 32,768- and
131,072-token windows. It covers the training execution path, memory controls,
record-separator preservation, explicit sampler phase transitions, one-node
correctness, two-node DiLoCo, and 32-node scaling. It does not authorize a
long-context production token budget.

Frozen trained parent authority:

- checkpoint root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/scale-ladder-mainfc52/checkpoints`
- generation: `step-02338080-tokens-0000250797359104`
- step: `2,338,080`
- accepted tokens: `250,797,359,104`
- manifest complete: `true`
- parent sampler: counter-v2, context 2,048, world 2,048
- parent source commit: `b0706594610d690f2ae389f3792eba02a0a4746c`
- parent scheduler job: `5216579`, `COMPLETED 0:0`, 256 nodes, 05:02:58

## Existing mechanisms retained

The qualification uses the existing sparse E97 Triton recurrence:

- state is saved every `checkpoint_interval` and replayed in backward;
- LM-head cross entropy is streamed with the existing `loss_chunk_size`;
- routed/shared rocBLAS experts retain checkpointed inputs and recompute SwiGLU;
- node-local eight-GCD expert parallelism and cross-node DiLoCo are unchanged.

No parameter, state-dict, router, expert, recurrence, or model-shape change was
made.

## Required execution fixes

1. `scripts/frontier/e97_35b_moe_train.py` now exposes and logs:
   - layer activation checkpointing;
   - loss chunk size;
   - recurrence checkpoint interval;
   - projection chunk size;
   - state-continuous TBPTT chunk size;
   - explicit post-restore LR override.
2. Nonlinear split-edit projections can be recomputed in bounded time segments.
   Each segment calls the existing sequential split-edit Triton recurrence and
   carries `S_final` into the next segment without changing the recurrence.
3. The fallback 128K recipe uses state-continuous TBPTT with a 32K gradient
   horizon. This remains available as `128k_tbptt_fallback.json`.
4. Literal 128K full BPTT builds one graph from four bounded 32K forward
   segments. Recurrent states remain attached across every segment boundary,
   so one final backward traverses the complete 128K window.
5. Complete-block grouped checkpointing retains one `(x, residual)` boundary
   for all 11 blocks. Each MoE records compact top-3 routing plans during
   forward and consumes those plans in reverse during replay, eliminating the
   ragged-shape failure of naive rerouting. Checkpointed 2K loss chunks and
   complete ScheduleFree-state CPU offload provide the remaining HBM margin.
6. Counter-to-counter phase transitions are explicit and fail closed. A new
   context/world identity begins at the exact parent accepted-token boundary;
   historical 2K samples are not relabelled.

Implementation commits, in order:

- `1b1c11ad` — expose long-context controls and phase memory
- `5958567a` — split-edit projection recomputation
- `8883841a` — state-continuous bounded TBPTT
- `a62022d2` — explicit counter-to-counter sampler transition
- `a46e0187` — trained-checkpoint transition launcher
- `2dbf11a0` — explicit LR override after optimizer restore
- `5e4bae1e` — grouped block replay and bounded MoE execution
- `53600b54` — deterministic top-3 route replay
- `176db9ba` — checkpointed LM-loss chunks
- `f73b252d` — optional pre-backward HIP cache trim
- `0bff17d9`, `466bba66` — ScheduleFree optimizer-state CPU offload
- `43dbd4aa` — full BPTT across bounded context segments

Focused integrated tests: 48 passed.

## Diagnostic sequence

All qualification jobs used `Partition=batch`, `QOS=debug`, `Requeue=0` and
were serialized so only one Debug-QoS job was active.

### Baselines and failure localization

| Job | Nodes | Context | Recipe | Result |
|---|---:|---:|---|---|
| 5218644 | 1 | 8K | layer checkpoint, loss C=2K | PASS, 2 steps |
| 5218756 | 1 | 32K | full projections | PASS, peak 54,320,634,880 B |
| 5218863 | 1 | 64K | full projections, ckpt interval 64 | expected OOM on second backward |
| 5219049 | 1 | 64K | same plus cache trim | expected OOM on second backward |
| 5219154 | 1 | 8K | projection C=2K parity probe | PASS |
| 5219315 | 1 | 64K | projection C=2K | PASS, peak 60,803,138,560 B |

At 8K, projection recomputation changed the first-step BF16 loss only from
`2.1704211` to `2.1703861` (3.5e-5), with router auxiliary difference below
4e-9. This is the expected segmented GEMM/rounding effect, not a semantic
change.

A whole-model activation-checkpoint experiment, job `5219406`, was rejected.
Dropless routing row counts changed during recomputation and PyTorch correctly
raised `CheckpointError` for different tensor metadata. This path is not used.

### Accepted 128K/32K-TBPTT fallback

| Job | Asset | Nodes | Context | Gradient horizon | K | Result |
|---|---|---:|---:|---:|---:|---|
| 5219435 | dense seed converted to MoE | 1 | 8K | 2K TBPTT gate | 1 | PASS |
| 5219480 | dense seed converted to MoE | 1 | 128K | 32K | 1 | PASS |
| 5219561 | dense seed converted to MoE | 2 | 128K | 32K | 1 | PASS, 2 merges |
| 5219603 | dense seed converted to MoE | 32 | 128K | 32K | 1 | PASS, 2 merges |
| 5219935 | trained 250.797B MoE | 1 | 128K | 32K | 1 | PASS |

Decisive trained-asset 128K receipt, job `5219935`:

- scheduler: `COMPLETED 0:0`, 1 node, 00:15:16;
- restored step/tokens: `2,338,080` / `250,797,359,104`;
- sampler restore: `counter-transition`;
- new identity: counter-v2, context 131,072, world 8, origin
  `250,797,359,104`;
- initial phase cursor: 0;
- two optimizer steps, final accepted tokens `250,799,456,256`;
- losses: `1.79976`, `1.81551`;
- peak allocated HBM: `60,302,868,992` B (56.16 GiB);
- steady measured throughput: approximately 11.1K tokens/s/node.

Decisive 32-node 128K receipt, job `5219603`:

- scheduler: `COMPLETED 0:0`, 32 nodes, 00:16:10;
- two optimizer steps and two K1 DiLoCo merges;
- accepted tokens: `67,108,864`;
- peak allocated HBM: `59,377,508,352` B (55.30 GiB);
- steady step time: 143.95 seconds;
- steady global throughput: 233,093 tokens/s;
- DiLoCo merge: 23.52 seconds.

### Accepted literal 128K full-BPTT path

The accepted exact recipe uses four 32K materialization segments, retains
attached recurrent states across all four, checkpoints all 11 complete blocks
as one replay group, checkpoints 2K loss chunks, and offloads the complete
ScheduleFree state. Setting the MoE execution bound to 32K preserves the
original full-segment router auxiliary objective exactly while bounding all
routing/exchange buffers to one segment.

| Job | Nodes | Steps | K | Result |
|---|---:|---:|---:|---|
| 5222503 | 1 | 2 | 1 | PASS, repeated full-BPTT optimizer updates |
| 5222295 | 2 | 1 | 1 | PASS, DiLoCo merge |
| 5222617 | 32 | 1 | 1 | PASS, scale qualification |

Decisive one-node receipt, job `5222503`:

- scheduler: `COMPLETED 0:0`, 1 node, 00:14:57;
- two literal full-128K optimizer steps;
- losses: `1.7997439`, `1.8153541`;
- first auxiliary loss: `0.01143238`, matching the 32K-segment objective;
- steady step time: 97.53 seconds, 10,751 tokens/s;
- peak allocated HBM: `60,535,007,232` B (56.38 GiB).

Two-node job `5222295` completed `0:0` with a K1 DiLoCo merge in 22.68 seconds.
The 32-node gate, job `5222617`, completed `0:0`:

- one optimizer step and one K1 DiLoCo merge;
- accepted tokens: `33,554,432`;
- peak allocated HBM: `59,829,827,584` B (55.72 GiB);
- step time: 156.18 seconds;
- global throughput: 214,842 tokens/s;
- DiLoCo merge: 50.50 seconds.

An 8K MoE execution bound also completed through 32 nodes (`5222024`,
`5222102`, `5222295`, and `5222368`), but diagnosis showed that it averages
router balance losses per MoE chunk rather than preserving the full-32K
auxiliary objective. It is rejected. The runner now fails closed when a
training MoE bound differs from the effective sequence segment.

### Accepted 32K full-BPTT path

| Job | Asset | Nodes | Context | Gradient horizon | K | Result |
|---|---|---:|---:|---:|---:|---|
| 5219673 | dense seed converted to MoE | 32 | 32K | full 32K | 2 | PASS |
| 5220015 | trained 250.797B MoE | 1 | 32K | full 32K | 2 | PASS |

Decisive trained-asset 32K receipt, job `5220015`:

- scheduler: `COMPLETED 0:0`, 1 node, 00:12:30;
- sampler restore: explicit counter transition, cursor 0;
- two optimizer steps, final accepted tokens `250,797,883,392`;
- losses: `1.83522`, `1.58148`;
- peak allocated HBM: `58,090,042,880` B (54.10 GiB).

Decisive 32-node 32K receipt, job `5219673`:

- scheduler: `COMPLETED 0:0`, 32 nodes, 00:12:58;
- two steps, one K2 merge;
- accepted tokens: `16,777,216`;
- peak allocated HBM: `50,675,260,416` B (47.19 GiB);
- non-merge step: 32.04 seconds, 261,795 tokens/s;
- K2 merge: 21.83 seconds.

## Record separator receipt

The unchanged corpus/tokenizer path was sampled directly:

- p50k encodes ASCII RS `0x1e` as token `218`;
- deterministic rank-0/seed-42 32K window contained 16 RS tokens;
- the corresponding 128K window contained 94 RS tokens;
- RS remains an ordinary visible token; recurrent state is not reset at RS.

### Effect of the new controls at 32K

Job `5222225` isolated checkpointed 2K loss chunks on the unchanged trained
32K full-BPTT path. Forward live allocation fell from approximately 47.48 GB
to 37.59 GB (9.89 GB / 20.8%) while steady step time changed from 23.54 to
23.83 seconds (1.2%). Backward parameter gradients still set the overall peak,
so this control is optional at 32K. Grouped whole-model replay and optimizer
state offload are not needed for 32K and remain disabled in `32k.json`.

## Qualification conclusion

- **32K:** qualified for full-BPTT training through 32 nodes; the new loss
  checkpoint control provides substantial forward headroom at negligible cost.
- **128K:** literal full 128K BPTT is qualified on the trained asset through 32
  nodes, including repeated one-node updates and two-/32-node K1 DiLoCo.
- **Fallback:** 128K forward context / 32K gradient horizon remains separately
  frozen and qualified; it must not be described as full 128K BPTT.
- Production adaptation still requires a separately frozen token budget, LR,
  node count, sampler world identity, checkpoint root, and evaluation plan.
