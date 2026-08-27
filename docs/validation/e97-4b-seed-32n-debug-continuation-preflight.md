# E97 4B imported-seed 32-node debug continuation preflight

**Status:** attended two-hour debug-QoS continuation authorized. No later job is
authorized by this preflight.

## Parent authority

The 32-node seed-import canary payload `5354034` and collector `5354035`
completed `0:0` on `Partition=batch`, `QOS=debug`:

- step 13,312 / 6,979,321,856 cumulative tokens;
- final last-100 loss 2.8908; last logged window 2.7593;
- 16 K32 merges averaging 4.853 seconds;
- 31,404 MiB allocated / 35,850 MiB reserved HBM;
- checkpoint SHA-256
  `fa7b53f8ea31ca177aac0bba6b1fd174a970d8d8db68314c96333efd80a50ade`;
- atomic latest publication and independent mmap reload passed;
- the counter-v1 Pile/world-eight to counter-v2 CommaPile/world-256 transition
  is retained in checkpoint metadata.

## Bound continuation

- 32 nodes / 256 ranks, B1, context 2,048, K32;
- `Partition=batch`, `QOS=debug`, `Requeue=0`;
- two-hour allocation limit;
- exact same-world model, ScheduleFree state, sampler identity and cursor;
- step 13,312 to 15,360: 2,048 new optimizer updates;
- 1,073,741,824 new tokens, ending at 8,053,063,680 cumulative tokens;
- 64 K32 merges;
- saves every 256 steps; target and every periodic checkpoint are K-aligned;
- unchanged LR `0.00047431158698290157`;
- no context, batch, world, corpus, tokenizer, optimizer, or merge-policy change.

Expected training time from the canary is approximately 81 minutes plus load
and checkpoint overhead, safely within two hours. Expected use is roughly
45--55 node-hours. No 64-node or longer-context transition is implied.

## Validation and scope

Canonical Frontier activation, Bash syntax checks, targeted pytest, immutable
source/config binding, parent SHA verification, queued/running scheduler field
capture, and terminal collector validation are required.

This is ADR-003 fixed-world continuation. Applicable safety intent is R07,
R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic
R02--R06/R08--R11/NDP02, async-v2.1 V21S01--V21S17 and ISP01--ISP07, and native
NDP17 are explicitly unclaimed.
