# E97 4B mature-seed 64-node equal-token canary preflight

**Status:** submitted as payload `5357795` with collector `5357796` from
immutable source `f70dbbd2f04e511071f1b4dfb684385cd055e916`. No continuation
is authorized by this preflight.

## Question

Does doubling the mature seed's world from 256 to 512 ranks improve or preserve
loss progress per node-hour, or does the resulting twofold effective-batch
increase waste updates?

The preceding 32-node block processed 1,073,741,824 new tokens in 2,048 B1
optimizer steps and 64 K32 merges, ending at step 15,360 / 8,053,063,680 tokens.
This canary processes the same number of new tokens and approximately the same
node-hours using 64 nodes, 1,024 B1 optimizer steps and 32 K32 merges. Context,
LR, local batch and local K remain unchanged. This isolates the world-size and
effective-batch change; it deliberately does not also change context length.

## Parent

- path: `e97-4b-seed-cont-w256-b1k32-r2/train/checkpoint_step_015360_loss_2.8084.pt`;
- step/token: 15,360 / 8,053,063,680;
- final last-100 loss: 2.8331;
- SHA-256: `9da49a274934135de4b5ac4f9265c0e6eff5ec7672fb7d9c3fb6388b0da18f16`;
- payload/collector `5354989`/`5354990` both completed `0:0`;
- mmap reload, sampler transition chain, model and optimizer state passed.

## Bound canary

- 64 nodes / 512 ranks, B1, context 2,048, K32;
- `Partition=batch`, `QOS=debug`, `Requeue=0`;
- one-hour limit;
- step 15,360 to 16,384: 1,024 new updates;
- 1,073,741,824 new tokens, ending at 9,126,805,504 cumulative tokens;
- 32 merges; saves every 256 steps; every boundary is K-aligned;
- unchanged LR `0.00047431158698290157`;
- explicit counter-v2 CommaPile world-256 to world-512 phase transition at
  8,053,063,680 accepted tokens;
- durable transition metadata chains the earlier Pile/world-eight import.

The fair decision metric is fixed-panel loss and loss reduction per node-hour,
not raw token throughput. Training-window loss is supporting evidence because
the counter identities select different samples.

## Context-length follow-up

The model should plausibly fit an 8K B1 pilot using the already implemented
gradient-checkpointing path, but that must be a separate experiment. Changing
world size and context together would make this scale canary uninterpretable.
A one-node memory/kernel qualification followed by a bounded 32-node 8K pilot
is the next context test if this checkpoint remains healthy.

## Architecture scope

This is ADR-003 fixed-world qualification. Applicable safety intent is R07,
R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic
R02--R06/R08--R11/NDP02, async-v2.1 V21S01--V21S17 and ISP01--ISP07, and native
NDP17 are explicitly unclaimed.
