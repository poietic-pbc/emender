# E97 4B Frontier from-scratch preflight

Status: historical preflight. The B5/K25 exact-world bootstrap passed, but the
from-scratch strategy was stopped for extreme large-batch token inefficiency.
See [`e97-4b-frontier-from-scratch-verdict.md`](e97-4b-frontier-from-scratch-verdict.md).

## Immutable training identity

* Shape: `d=3840`, `L=18`, `H=60`, `n=64`, MLP ratio 2.5;
  4,045,972,080 parameters.
* BF16 fused E97, Schedule-Free AdamW, LR
  `0.00047431158698290157`, weight decay 0.01, clip 1.0.
* Production fixed world: 256 nodes x 8 GCDs = 2,048 ranks. Every rank uses a
  private node-local Triton cache keyed by job and global rank; job 5337283
  proved that a shared home-directory Triton cache races and fails at this scale.
* Batch five per rank, context 2,048: 20,971,520 aggregate accepted tokens per
  optimizer step.
* Operational target: 4,769 optimizer steps / 100,013,178,880 accepted tokens,
  just above the requested approximately 100B budget (24.7 tokens/parameter).
* DiLoCo `avg`, K=25 at the unchanged validated LR. This intentionally changes
  cadence from 128 to 125 local samples per merge (2.34%) and yields 524,288,000
  global accepted tokens per merge. The operator explicitly accepted this
  throughput-driven deviation after the controlled B2/B4/B5/B6/B8 sweep.
* Checkpoint every 50 steps (two merges), retain three, plus final/pre-signal
  publication. The exact bootstrap measures checkpoint cost before continuation.

DiLoCo worker count does not trigger conventional global-batch LR scaling: the
inner Schedule-Free LR remains `0.00047431158698290157`.

## Budget model

After failed job 5337283, the scheduler-derived balance on 2026-08-24 was
about 3,170.14 node-hours; the subsequent small qualification jobs consumed
only a few additional node-hours. The diagnosed 256-node 30-minute replacement
bootstrap costs 128 node-hours, a two-hour debug continuation costs 512, and a six-hour
normal continuation costs 1,536. The attended plan is bootstrap -> inspect ->
two-hour debug resume -> inspect -> a measured four-, six-, or eight-hour
normal resume, stopping at the configured 100B token ceiling. No job is chained
or automatically resubmitted after failure.

The 30-minute 256-node bootstrap is useful training, not a throwaway canary: if
and only if its exact-source fixed-world gates pass, its step-50 /
1,048,576,000-token checkpoint becomes the genesis training authority and
counts toward the 100B target. The four-node rung has a
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
key 42, context 2,048, and the fixed world of 2,048 ranks.

The exact 256-node bootstrap starts from random initialization only when stable
`train/latest.pt` is absent. No earlier trained seed is used. The immutable
source/config/seed plus the first K-aligned checkpoint define the genesis
lineage. Later attended submissions require
`CONFIRM_RESUME=1`; `train.py` validates embedded counter identity and accepted
token cursor before model/optimizer mutation. World-size drift fails closed.

## Scheduler and safety contract

The submitter requires clean `HEAD == main == origin/main`, creates an immutable
checkout, pins one stable source/config identity across the run, sources
`activate_emender_frontier.sh`, and submits explicit
`Partition=batch` and QoS (`debug` smoke, `normal` production). It retains
`squeue` output naming both fields while queued/running. A durable afterany
collector retains terminal `sacct` fields `Partition` and `QOS` separately.

The bootstrap payload is one fixed-world fail-stop `srun`,
`--kill-on-bad-exit`, no requeue, with batch-signal forwarding into the normal
final consensus and atomic checkpoint path. Each submitted job has one execution
epoch. A later human-reviewed job constructs a fresh process group and resumes
only atomic `latest.pt`; it never preserves or shrinks a failed communicator.
This retains ADR-003 gap-matrix safety intent for R07, R12, R14/NDP13, R16, and
NDP15 checkpoint atomicity. R02--R06/R08--R11, NDP02 and other native/elastic
clauses, and all V21S01--V21S17/ISP01--ISP07 requirements are explicitly retired
or unclaimed for this fixed-world production path.

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
   ranks / 30 minutes under `batch/debug`.
6. Require eight fused guards, finite loss/gradients, measured peak HBM, two
   successful selected-K consensuses, one atomic ~24 GB checkpoint, and
   reloadable sampler/optimizer metadata. Throughput must project inside the
   reviewed production allocation envelope.
7. Promote the bootstrap checkpoint only after terminal `Partition` and `QOS`
   evidence, exact accepted-token accounting, and checkpoint integrity review.
8. Submit one attended `MODE=debug_continuation
   CONFIRM_DEBUG_CONTINUATION=1 CONFIRM_RESUME=1` two-hour epoch under
   `Partition=batch`, `QOS=debug`; this is the fresh-process-group restart gate.
9. Inspect it, then choose an attended `production_4h`, `production_6h`, or
   `production_8h` normal-QoS epoch from measured accepted-token throughput and
   remaining allocation.
10. Inspect actual accounting, throughput, loss, and checkpoint authority before
    every continuation. No automatic scheduler resubmission or chain.

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
tokens/s/GCD with 52,380 MiB allocated / 58,208 MiB reserved. B5 job 5338084
completed cleanly at 1,711 tokens/s/GCD mean ordinary throughput, two merges in
12.3 and 6.7 seconds, 47,721 MiB allocated / 57,562 MiB reserved, and a valid
step-50 checkpoint. B5 was selected because it is only 1.9% slower than B6 in
ordinary updates, used about 4.5 GiB less live allocation, and had higher
observed merge-inclusive throughput. Its 7.77 GiB reserve-based headroom still
requires the exact 2,048-rank bootstrap.

First exact-world attempt 5338307 used `Partition=batch`, `QOS=debug` and proved
28 finite B5 updates plus one 2,048-rank K25 merge in 63.7 seconds, with ordinary
throughput reaching 1,767 tokens/s/GCD. It failed at the scheduler five-minute
warning after 14m50s: the batch trap forwarded TERM to `srun`, which cancelled
all ranks before finalization. No checkpoint was published, so its implied
587,202,560 processed tokens are not committed. The failed `r2` directory is
retained as evidence. The repair uses a new immutable `r3` lineage, requests 30
minutes so step 50 can complete before the warning, polls the internal walltime
controller every eight steps, and translates a batch signal into an atomic
`.final_checkpoint_request` rather than signalling `srun`. Repository readiness
is not replacement execution evidence.
