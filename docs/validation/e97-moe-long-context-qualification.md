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
3. 128K uses state-continuous TBPTT with 32K gradient segments. Forward state
   spans the complete 128K window; recurrent gradients are deliberately
   truncated at 32K boundaries. Gradients from all four segments are averaged
   before one optimizer update.
4. Counter-to-counter phase transitions are explicit and fail closed. A new
   context/world identity begins at the exact parent accepted-token boundary;
   historical 2K samples are not relabelled.

Implementation commits, in order:

- `1b1c11ad` — expose long-context controls and phase memory
- `5958567a` — split-edit projection recomputation
- `8883841a` — state-continuous bounded TBPTT
- `a62022d2` — explicit counter-to-counter sampler transition
- `a46e0187` — trained-checkpoint transition launcher
- `2dbf11a0` — explicit LR override after optimizer restore

Focused integrated tests: 46 passed.

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

### Accepted 128K execution path

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

## Qualification conclusion

- **32K:** qualified for full-BPTT training through 32 nodes.
- **128K:** qualified through 32 nodes with continuous forward state and 32K
  TBPTT gradient horizon.
- **Not qualified:** literal full 128K BPTT. The present HBM envelope does not
  support it, and the nondeterministic dropless-routing shape makes naive
  whole-model recomputation invalid.
- Production adaptation still requires a separately frozen token budget, LR,
  node count, sampler world identity, checkpoint root, and evaluation plan.
