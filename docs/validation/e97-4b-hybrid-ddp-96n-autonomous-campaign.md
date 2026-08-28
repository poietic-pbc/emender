# E97 4B 96-node hybrid-DDP autonomous campaign

**Status:** operator-authorized autonomous execution for the final four-day
allocation window. The 96-node debug gate completed successfully as payload
`5361411` / collector `5361412`. The first six-hour normal-QoS payload
`5362809` remained pending on priority for twelve hours and was cancelled,
without starting or consuming compute, together with collector `5362810` after
explicit operator authorization. Production now uses thirteen two-hour
normal-QoS epochs to improve backfill access without mutating the immutable
queued payload. Short normal phase 1 completed successfully as payload
`5364644` / collector `5364645`, reaching step 19,200 / 22,414,360,576 tokens
with final last-100 loss 2.5905 and a reloadable checkpoint. Short normal phase
2 (`5366556` / `5366557`) then remained pending on priority and consumed no
compute. The operator explicitly accepted the debug-QoS policy tradeoff and
authorized cancellation plus repeated two-hour debug-QoS continuation to get
a scientific answer before allocation expiry. The campaign began with the
bounded debug gate below.
If that gate has a hard systems, nonfinite, checkpoint, reload, or catastrophic
learning failure, stop and report alternatives. If healthy, submit and inspect
each of the four production epochs sequentially without waiting for another
operator message. No blind scheduler dependency chain is authorized.

## Motivation

Frontier permits at most two hours below 92 nodes. A 96-node job is in the
92--183-node bin and may request six hours. The successful eight-node topology
is preserved locally on every node:

```text
96 independent node islands
8 DDP ranks/node * B4/rank = effective B32/island
context 2048, K32, stateless outer average
```

Increasing island count changes statistical batch and must pass the debug gate,
but does not return to singleton B1 optimizer trajectories. Every node owns one
B32 ScheduleFree trajectory matching the proven Lambda-local geometry.

## Immutable parent

- step/token: 17,152 / 9,529,458,688;
- checkpoint: `e97-4b-hybrid-ddp-8n-b4k32-r4/train/checkpoint_step_017152_loss_2.6381.pt`;
- SHA-256: `cfb25848725912da577452e1a23fc91c541c6bb891145732d3b9c08f7ba9cfc9`;
- final last-100 loss: 2.6970;
- payload/collector `5358779`/`5358780` completed `0:0`;
- transition chain, atomic publication and mmap reload passed.

## Phase table

All jobs use 96 nodes, 768 ranks, B4, island size eight, K32, context 2,048,
LR `0.00047431158698290157`, saves every 256 steps, `Requeue=0`, and exact
same-world counter-v2 continuation after the debug transition.

| Phase | QoS / limit | Start -> target step | New tokens | End tokens |
|---|---|---:|---:|---:|
| debug | debug / 02:00 | 17,152 -> 18,176 | 6,442,450,944 | 15,971,909,632 |
| p1 | normal / 06:00 | 18,176 -> 21,504 | 20,937,965,568 | 36,909,875,200 |
| p2 | normal / 06:00 | 21,504 -> 24,832 | 20,937,965,568 | 57,847,840,768 |
| p3 | normal / 06:00 | 24,832 -> 28,160 | 20,937,965,568 | 78,785,806,336 |
| p4 | normal / 06:00 | 28,160 -> 31,488 | 20,937,965,568 | 99,723,771,904 |

The replacement short-production schedule divides the exact same remaining
13,312 updates into thirteen epochs of 1,024 updates each (phase 1 used normal QoS; phases 2--13 use the
operator-authorized debug-QoS override). Targets
are 19,200, 20,224, ..., 31,488; every epoch adds 6,442,450,944 tokens and is
both K32- and save256-aligned. The two-hour limit preserves a roughly
15-minute margin relative to the qualified 1h44m debug runtime. Each epoch is
submitted only after its predecessor passes collector inspection.

The operational target is approximately 100B, so 99.724B satisfies the
campaign objective. Every target is K32- and save256-aligned.

## Autonomous gate and stop rules

For every epoch, retain live and terminal records naming both
`Partition=batch` and the exact QoS. Require payload `COMPLETED/0:0`, no OOM,
traceback, nonfinite loss/gradient, distributed timeout, or hard data error;
all expected merges and 768 finalization receipts; K-aligned atomic latest;
independent SHA-256 and mmap reload receipt; finite loss; and HBM evidence.

The debug gate is healthy when those systems requirements pass and its recent
loss is not catastrophically worse than its initial quarter (more than 0.15
nats sustained regression) or above 3.0. Flat but finite behavior is diagnostic,
not a hard failure, under the operator's explicit instruction to use the
remaining allocation. Production epochs are submitted one at a time only
after the prior collector and inspection pass. Any later hard failure also
stops fresh submission; do not automatically resubmit a damaged epoch.

## Architecture scope

This is ADR-003 fixed-world execution. Applicable safety intent is R07, R12,
R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic
R02--R06/R08--R11/NDP02, async-v2.1 V21S01--V21S17 and ISP01--ISP07, and native
NDP17 are explicitly unclaimed.
