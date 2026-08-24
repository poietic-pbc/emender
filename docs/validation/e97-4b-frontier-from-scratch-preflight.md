# E97 4B Frontier from-scratch preflight

Status: source/config/launcher prepared; live Frontier canary and production
submission not yet claimed.

## Immutable training identity

* Shape: `d=3840`, `L=18`, `H=60`, `n=64`, MLP ratio 2.5;
  4,045,972,080 parameters.
* BF16 fused E97, Schedule-Free AdamW, LR
  `0.00047431158698290157`, weight decay 0.01, clip 1.0.
* Production fixed world: 32 nodes x 8 GCDs = 256 ranks.
* Batch four per rank, context 2,048: 2,097,152 aggregate accepted tokens per
  optimizer step.
* Target: 38,586 optimizer steps / 80,920,707,072 accepted tokens, just above
  20 tokens per parameter.
* DiLoCo `avg`, K=256. Batch x K gives 1,024 local samples per merge, matching
  the qualified local batch-32/K-32 sample cadence.
* Checkpoint every 1,024 steps (four merges), retain three, plus final/pre-signal
  publication.

The production global batch is four times the local eight-GPU run, but retains
38,586 sequential optimizer updates to the Chinchilla target. A 256-node
from-scratch launch was rejected: even batch one would expose only about 19,293
updates and would spend most of the node-hour budget replicating optimizer
work.

## Budget model

One seven-hour 32-node epoch costs 224 node-hours. Twelve attended epochs cost
2,688 node-hours and provide 84 wall-clock hours; thirteen cost 2,912
node-hours and provide 91 hours. At an initial planning estimate of eight
seconds per optimizer step, the target requires about 85.7 hours. This is a
budget envelope, not throughput evidence. The debug canary must measure real
MI250X memory and step time before the production series is authorized.

## Data and restart authority

The reviewed CommaPile object is
`/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
with expected size 1,000,000,725,401 bytes. The first allocation computes its
true SHA-256 once and atomically writes a durable receipt. Counter sampler v1
binds that digest, p50k digest
`94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`,
key 42, context 2,048, and the fixed world of 256.

The first production epoch starts from random initialization only when stable
`train/latest.pt` is absent. Later attended submissions require
`CONFIRM_RESUME=1`; `train.py` validates embedded counter identity and accepted
token cursor before model/optimizer mutation. World-size drift fails closed.

## Scheduler and safety contract

The submitter requires clean `HEAD == main == origin/main`, creates an immutable
checkout, sources `activate_emender_frontier.sh`, and submits explicit
`Partition=batch` and QoS (`debug` smoke, `normal` production). It retains
`squeue` output naming both fields while queued/running. A durable afterany
collector retains terminal `sacct` fields `Partition` and `QOS` separately.

The payload is one fixed-world fail-stop `srun`, `--kill-on-bad-exit`, no
requeue, with batch-signal forwarding into the normal final consensus and
atomic checkpoint path. This retains applicable gap-matrix safety intent:

* R07: atomic checkpoint and `latest.pt` publication;
* R12: exact model/inner optimizer/token-cursor restore;
* R14: bounded job/child lifetime and retained scheduler evidence;
* R15: fixed-world weighted accounting.

No elastic, native-dataplane, changing-world, or asynchronous conformance is
claimed (ADR-003 fixed-world authority; NDP02 is retired/incompatible here).

## Required live ladder

1. Pull the immutable origin commit on Frontier.
2. Run `MODE=smoke scripts/frontier/submit_e97_4b_from_scratch.sh` under
   `Partition=batch`, `QOS=debug`; retain queued/running and terminal evidence.
3. Require eight fused guards, finite loss/gradients, measured peak HBM,
   successful K4 consensus, atomic ~24 GB checkpoint, and reloadable sampler
   metadata.
4. Review measured step time and revise only through a new committed config.
5. Submit `MODE=production CONFIRM_PRODUCTION=1 ...` under
   `Partition=batch`, `QOS=normal` for the stable 32-node run id.
6. Before every later epoch, inspect terminal accounting and checkpoint, then
   submit manually with `CONFIRM_RESUME=1`. No automatic requeue/chain.

## Current limitation

The attended host could not resolve the Frontier login alias, so no scheduler
mutation or live canary was attempted while preparing this source. Repository
readiness is not Frontier execution evidence.
