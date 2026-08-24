# E97 4B Frontier from-scratch preflight

Status: source/config/launcher prepared; live Frontier canary and production
submission not yet claimed.

## Immutable training identity

* Shape: `d=3840`, `L=18`, `H=60`, `n=64`, MLP ratio 2.5;
  4,045,972,080 parameters.
* BF16 fused E97, Schedule-Free AdamW, LR
  `0.00047431158698290157`, weight decay 0.01, clip 1.0.
* Production fixed world: 256 nodes x 8 GCDs = 2,048 ranks. Every rank uses a
  private node-local Triton cache keyed by job and global rank; job 5337283
  proved that a shared home-directory Triton cache races and fails at this scale.
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

After failed job 5337283, the scheduler-derived balance on 2026-08-24 is about
3,170.14 node-hours. The revised maximum envelope is 1.33 node-hours for a
four-node 20-minute cache/model rung, 85.33 for an exact 256-node 20-minute
bootstrap, and 3,072 for two 256-node six-hour normal allocations: 3,158.67
node-hours, leaving about 11.47 before small collectors. Production uses 345
productive minutes per job with a ten-minute final-checkpoint margin. The
bootstrap gate remains <=2.15 effective seconds per update.

The 20-minute 256-node bootstrap is useful training, not a throwaway canary: if
and only if its exact-source fixed-world gates pass, its step-256 /
1,073,741,824-token checkpoint becomes the genesis training authority and
counts toward the 20 accepted tokens/parameter target. The four-node rung has a
different sampler world and is qualification-only; its checkpoint is never
promoted into the production lineage.

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
2. Submit `MODE=rung CONFIRM_RUNG=1` for exactly four nodes / 32 ranks / 20
   minutes. Require private rank-local Triton caches, finite training, one K128
   consensus, safe HBM, and a reloadable step-256 checkpoint. Do not promote it.
3. Use two-node probes to select batch size while preserving DiLoCo work:
   B2/K64/save128/target128, then B4/K32/save64/target64, bounded
   B5/K25/save50/target50 and B6/K21/save42/target42 interpolations, and
   B8/K16/save32/target32 only as an HBM boundary probe. The power-of-two arms
   process 256 local samples; B5 processes 250 and B6 processes 252 (125/126
   per merge, within 2.4% of the 128-sample authority).
   Every arm performs two merges, keeps LR unchanged, and uses a separate
   non-promotable run identity. Select from finite loss, peak reserved HBM,
   merge/checkpoint time, and sustained tokens/s/GCD.
4. Commit the selected production B/K/step/checkpoint geometry.
5. Submit `MODE=bootstrap CONFIRM_BOOTSTRAP=1` for exactly 256 nodes / 2,048
   ranks / 20 minutes under `batch/debug`.
6. Require eight fused guards, finite loss/gradients, measured peak HBM, two
   successful selected-K consensuses, one atomic ~24 GB checkpoint, and
   reloadable sampler/optimizer metadata. Throughput must project inside the
   reviewed production allocation envelope.
7. Promote the bootstrap checkpoint only after terminal `Partition` and `QOS`
   evidence, exact accepted-token accounting, and checkpoint integrity review.
8. Add and locally qualify bounded same-allocation restart and node-local
   checkpoint staging before production authorization.
9. Submit one attended `MODE=production CONFIRM_PRODUCTION=1 CONFIRM_RESUME=1`
   epoch under `Partition=batch`, `QOS=normal` for the stable 256-node run id.
10. Inspect actual accounting, throughput, loss, and checkpoint authority before
    sizing any continuation. No automatic scheduler resubmission or chain.

## Current limitation

Job 5337283 reached initialization and fused warmup but failed before the first
optimizer update because 2,048 ranks concurrently mutated the shared
`~/.triton/cache` temporary directory. It published no checkpoint and commits
no tokens. Four-node repair rung 5337432 then proved rank-private caches, 256
finite updates, two K128 merges (5.3 and 6.2 seconds), about 2.84 seconds per ordinary update,
38,738 MiB peak allocated / 41,386 MiB reserved HBM, and one atomic
24,276,098,175-byte step-256
checkpoint with SHA-256
`a24f91b1ef2fc553decc517382b849d2362dee780f0a76e74289d183f6bcb9e0`.
It nevertheless exited 137: after the periodic checkpoint completed, generic
finalization redundantly began serializing the same state a second time and the
five-minute scheduler warning killed that duplicate temporary write. The
checkpoint is qualification-only and the rung is not a clean pass. Source now
reuses a completed same-step periodic checkpoint during finalization. Clean
replacement rung 5337548 completed 256 updates, two approximately six-second
K128 merges, periodic-checkpoint reuse, and terminal exit zero. Removing outer
layer-group checkpointing improved ordinary throughput from about 720 to
840--860 tokens/s/GCD while peak allocated remained 38,738 MiB and peak reserved
rose modestly to about 42,682 MiB. This is memory-safe but still below the
throughput target. B2 job 5337664 completed at roughly 1,200--1,250
tokens/s/GCD with 41,526 MiB reserved. B4 job 5337831 completed at roughly
1,420--1,505 tokens/s/GCD with 49,002 MiB reserved, although its two merges took
20.7--23.3 seconds. B8 job 5337929 failed before its first optimizer update:
58.96 GiB was allocated and 3.92 GiB reserved-but-unallocated, leaving no room
for a 100--198 MiB request. B8 is rejected even if allocator tuning could make
it barely fit. B6 job 5337975 completed cleanly at roughly 1,690--1,775
tokens/s/GCD, but 52,380 MiB allocated / 58,208 MiB reserved leaves only about
5.8 GiB physical headroom before the larger 256-node communicator topology;
B6 is therefore not production-safe as measured. One final B5 interpolation
probe is justified to target at least 8--10 GiB headroom while improving on B4
throughput. Repository readiness is not Frontier execution evidence.
