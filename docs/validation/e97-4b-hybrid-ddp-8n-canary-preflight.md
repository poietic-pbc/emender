# E97 4B Lambda-topology hybrid-DDP canary preflight

**Status:** submitted as payload `5358779` with collector `5358780` from
immutable source `25b0be049029d3a85f7c1130178c1011c3599765`. No longer run is
authorized until this exact topology passes and is inspected.

## Scientific question

The singleton-island Frontier continuations are mechanically healthy, but their
training loss is a noisy rank-zero stream and has not established strong fixed
loss improvement. This canary tests whether reproducing the successful Lambda
optimizer topology improves mature-seed learning dynamics.

The mathematical topology match is:

```text
Lambda:   8 independent optimizer islands * local B32 = 256 samples/update
Frontier: 8 node islands * (8-rank DDP * B4)       = 256 samples/update
```

Both use context 2,048, K32, 524,288 aggregate tokens/update, stateless outer
averaging, and LR `0.00047431158698290157`. Within each Frontier node, DDP
all-reduces gradients every step so all eight ranks maintain one identical
ScheduleFree trajectory with effective batch 32. DiLoCo averages the eight node
trajectories every 32 steps. This is not global DDP.

## Parent

- step/token: 16,384 / 9,126,805,504;
- path: `e97-4b-seed-scale-w512-b1k32-r3/train/checkpoint_step_016384_loss_2.6330.pt`;
- final last-100 training loss: 2.8425;
- SHA-256: `bb1a1350c37126793ad2de2b4477014c09f671a0c731184a4c5496f21cfcd8bd`;
- payload/collector `5357795`/`5357796` completed `0:0` with mmap reload.

## Bounded canary

- 8 nodes / 64 ranks / eight DDP ranks per node / eight optimizer islands;
- B4/rank, effective B32/island, context 2,048, K32;
- `Partition=batch`, `QOS=debug`, `Requeue=0`, two-hour limit;
- explicit counter-v2 world-512 to world-64 phase transition at the parent
  boundary, retaining the complete prior transition chain;
- 768 new updates / 402,653,184 new tokens;
- step 16,384 to 17,152; total tokens 9,529,458,688;
- 24 merges, saves every 256 steps, all boundaries K-aligned;
- GPU-resident ScheduleFree state and the already qualified B4 memory geometry.

This qualifies the topology; it does not yet provide the requested multi-billion
token verdict. On success, an inspected 6--8 hour normal-QoS continuation of
approximately 1.6--2.1B new tokens is the next separate decision.

## Validation and scope

Canonical Frontier activation, Bash syntax checks, targeted tests, immutable
source/config binding, parent hashing, explicit `Partition`/`QOS` evidence,
atomic checkpoint publication and mmap reload are mandatory.

This is ADR-003 fixed-world qualification. Applicable safety intent is R07,
R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic
R02--R06/R08--R11/NDP02, async-v2.1 V21S01--V21S17 and ISP01--ISP07, and native
NDP17 are explicitly unclaimed.
