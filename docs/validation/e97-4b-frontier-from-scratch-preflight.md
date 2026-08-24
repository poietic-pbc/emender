# E97 4B Frontier from-scratch preflight

Status: source/config/launcher prepared; live Frontier canary and production
submission not yet claimed.

## Immutable training identity

* Shape: `d=3840`, `L=18`, `H=60`, `n=64`, MLP ratio 2.5;
  4,045,972,080 parameters.
* BF16 fused E97, Schedule-Free AdamW, LR
  `0.00047431158698290157`, weight decay 0.01, clip 1.0.
* Production fixed world: 256 nodes x 8 GCDs = 2,048 ranks.
* Batch one per rank, context 2,048: 4,194,304 aggregate accepted tokens per
  optimizer step.
* Target: 19,293 optimizer steps / 80,920,707,072 accepted tokens, just above
  20 tokens per parameter.
* DiLoCo `avg`, K=128 at the unchanged validated LR. This gives 128 local
  samples and 536,870,912 global accepted tokens per merge. It is slightly more
  conservative than the proven 256-node E97 1.3B K40/B4 authority (160 local
  samples and 671,088,640 global tokens per merge).
* Checkpoint every 256 steps (two merges), retain three, plus final/pre-signal
  publication. The exact bootstrap measures checkpoint cost; cadence may be
  relaxed to 512 only through reviewed evidence if a 24 GB publication takes
  more than 60 seconds.

The minimum 256-node batch exposes 19,293 sequential optimizer updates to the
Chinchilla target. DiLoCo worker count does not trigger conventional global-
batch LR scaling: the inner Schedule-Free LR remains
`0.00047431158698290157`.

## Budget model

The scheduler-derived balance on 2026-08-24 was 3,220.56 node-hours. One exact
256-node 30-minute bootstrap costs at most 128 node-hours. Two 256-node 6:05
normal allocations cost at most 3,114.67 node-hours, so the complete maximum
envelope is 3,242.67 and is not authorized as a blind fixed pair. Instead the
first production epoch is attended and the continuation wall time is sized
from actual bootstrap/epoch accounting. The nominal two-epoch envelope uses
about 350 productive minutes per job and requires an effective cadence no worse
than about 2.18 seconds per update; the bootstrap gate is <=2.15 seconds per
update.

The 30-minute bootstrap is useful training, not a throwaway canary: if and only
if its exact-source fixed-world gates pass, its 512-step / 2,147,483,648-token
checkpoint becomes the genesis training authority and counts toward the 20
accepted tokens/parameter target.

## Data and restart authority

The reviewed CommaPile object is
`/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
with expected size 1,000,000,725,401 bytes. Its reviewed physical SHA-256 is
`44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9`;
the payload materializes that committed authority instead of re-reading 1 TB
inside the allocation. Counter sampler v1 binds that digest, p50k digest
`94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`,
key 42, context 2,048, and the fixed world of 256.

The exact 256-node bootstrap starts from random initialization only when stable
`train/latest.pt` is absent. No earlier trained seed is used. The immutable
source/config/seed plus the first K-aligned checkpoint define the genesis
lineage. Later attended submissions require
`CONFIRM_RESUME=1`; `train.py` validates embedded counter identity and accepted
token cursor before model/optimizer mutation. World-size drift fails closed.

## Scheduler and safety contract

The submitter requires clean `HEAD == main == origin/main`, creates an immutable
checkout, sources `activate_emender_frontier.sh`, and submits explicit
`Partition=batch` and QoS (`debug` smoke, `normal` production). It retains
`squeue` output naming both fields while queued/running. A durable afterany
collector retains terminal `sacct` fields `Partition` and `QOS` separately.

The bootstrap payload is one fixed-world fail-stop `srun`,
`--kill-on-bad-exit`, no requeue, with batch-signal forwarding into the normal
final consensus and atomic checkpoint path. Production additionally requires a
bounded same-allocation fresh-communicator restart path before authorization;
the bootstrap does not claim that gate. This retains applicable gap-matrix
safety intent:

* R07: atomic checkpoint and `latest.pt` publication;
* R12: exact model/inner optimizer/token-cursor restore;
* R14: bounded job/child lifetime and retained scheduler evidence;
* R15: fixed-world weighted accounting.

No elastic, native-dataplane, changing-world, or asynchronous conformance is
claimed (ADR-003 fixed-world authority; NDP02 is retired/incompatible here).

## Required live ladder

1. Pull the immutable origin commit on Frontier.
2. Submit `MODE=bootstrap CONFIRM_BOOTSTRAP=1` for exactly 256 nodes / 2,048
   ranks / 30 minutes under `Partition=batch`, `QOS=debug`.
3. Require eight fused guards, finite loss/gradients, <=2.15 effective seconds
   per update, measured peak HBM, four successful K128 consensuses, two atomic
   ~24 GB checkpoints, and reloadable sampler/optimizer metadata.
4. Promote the step-512 checkpoint only after terminal `Partition` and `QOS`
   evidence, exact accepted-token accounting, and checkpoint integrity review.
5. Add and locally qualify bounded same-allocation restart and node-local
   checkpoint staging before production authorization.
6. Submit one attended `MODE=production CONFIRM_PRODUCTION=1 CONFIRM_RESUME=1`
   epoch under `Partition=batch`, `QOS=normal` for the stable 256-node run id.
7. Inspect actual accounting, throughput, loss, and checkpoint authority before
   sizing any continuation. No automatic scheduler resubmission or chain.

## Current limitation

No live 4B Frontier bootstrap has passed yet. Repository readiness is not
Frontier execution evidence.
