# E97/GDN2 paper Frontier qualification

Date: 2026-08-09

## Verdict

- **E97-MLP: PASS through the serialized 8 → 32 → 256-node ladder.**
- **GDN2-MLP: PASS through the serialized 8 → 32 → 256-node ladder.**
- **E97-linear-MLP: FAIL CLOSED.** One-GCD, eight-GCD restore, and the latest
  8-node rung pass, but independent ranks produce rare non-finite losses before
  the first K40 merge at larger rank populations. The failure occurs with both
  chunked and sequential fused E97 routes. No production job is authorized.

Every qualification allocation used `Partition=batch`, `QOS=debug`, and
`Requeue=0`. Live `scontrol` and `squeue` records, source receipts, checkpoint
validation, logs, and PASS verdicts are retained below
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-gdn2-paper/qualification`.
Jobs were submitted serially, with at most one active Debug-QoS job.

## Kernel and one-node gates

The priority arms passed one-GCD fused-kernel/full-model gates in jobs 5213741
(E97-linear, initial chunked route) and 5214006 (GDN2). Nonlinear E97's short
regression gate passed in 5214118. After matching both E97 arms on the sequential
fused route, E97-linear passed current-source one-GCD job 5214798. Its runtime
receipt states `recurrence=sequential`, `state=linear`, `eager_fallback=False`,
and peak probe allocation 12,451.8 MiB.

Eight-GCD, one-node pure-DiLoCo qualification trained to step 160 (four K40
merges), checkpointed, restored in a fresh process, and trained to step 200:

| Arm | Job | Result | Step-160 tokens | Peak allocated |
|---|---:|---|---:|---:|
| E97-MLP | 5214372 | `COMPLETED 0:0` | 5,242,880 | 16,982 MiB |
| E97-linear-MLP, sequential | 5214859 | `COMPLETED 0:0` | 5,242,880 | 16,982 MiB |
| GDN2-MLP | 5214351 | `COMPLETED 0:0` | 5,242,880 | 17,932 MiB |

Each final checkpoint records step 200, 6,553,600 total accepted tokens, and
sampler cursor 400. GDN2 used the immutable staged commit
`95709fc250357c2dd109361c353192f2aa5913f9`; fused guards reported no eager
fallback.

## Passing systems ladders

All rungs used source commit `0a49bb6fabb4084a2a465950ad2801fb60b69c88`,
B2, context 2,048, pure DiLoCo K40, four merges, hierarchical groups of eight,
160 exact steps, and validated the checkpoint's accepted-token authority.

| Arm | Nodes / ranks | Job | State | Accepted tokens | Peak allocated | Mean merge |
|---|---:|---:|---|---:|---:|---:|
| E97-MLP | 8 / 64 | 5215231 | `COMPLETED 0:0` | 41,943,040 | 16,939 MiB | 8,781.5 ms |
| E97-MLP | 32 / 256 | 5215343 | `COMPLETED 0:0` | 167,772,160 | 16,939 MiB | 9,976.7 ms |
| E97-MLP | 256 / 2,048 | 5215375 | `COMPLETED 0:0` | 1,342,177,280 | 16,939 MiB | 10,063.5 ms |
| GDN2-MLP | 8 / 64 | 5215056 | `COMPLETED 0:0` | 41,943,040 | 17,941 MiB | 8,627.5 ms |
| GDN2-MLP | 32 / 256 | 5215081 | `COMPLETED 0:0` | 167,772,160 | 17,941 MiB | 9,197.2 ms |
| GDN2-MLP | 256 / 2,048 | 5215097 | `COMPLETED 0:0` | 1,342,177,280 | 17,942 MiB | 10,817.8 ms |

The 256-node GDN2 gate completed in 00:14:51 and the E97 gate in 00:15:22,
including immutable-source setup, process-group construction, compilation,
training, checkpointing, and validation.

## E97-linear failure investigation

The matched arm initially used the chunked fused route. The following evidence
showed a population-dependent intermittent failure:

- 8-node job 5214470: rank 41, non-finite loss at step 29.
- Exact-source retry 5214583: passed 160 steps on 64 ranks.
- 32-node job 5214603: passed 160 steps on 256 ranks.
- 256-node job 5214686: rank 886, non-finite loss at step 28.

One-GCD reproductions of the exact rank streams (jobs 5214566 and 5214773)
remained finite through 40 steps. To remove the chunked route as a confound, the
frozen E97 configs were made identical in kernel routing: both now use the
sequential split-edit Triton kernel and differ only in `arm` and `linear_state`.
The manifest parameter/tensor schema remained unchanged. The sequential arm
then passed one-GCD job 5214798, one-node restore job 5214859, and 8-node job
5214917. Nevertheless, 32-node job 5214939 failed on rank 179 with a non-finite
loss at step 36, still before the first K40 merge.

Thus the chunked route is not the sole cause. At the frozen matched learning
rate (`0.001007`), the unbounded linear-state arm has not demonstrated the
required fail-stop stability over the larger independent-rank population.
Changing its learning rate, clipping, state transition, or optimizer would
change the frozen matched comparison and requires an explicit scientific-plan
decision. Qualification therefore stops here rather than masking, retrying, or
launching production from a non-reproducibly passing allocation.

## Production boundary

No 6,000-step production allocation was submitted. E97-MLP and GDN2-MLP have
complete pre-production systems evidence, but the three-arm study as specified
is **not production-ready** until the E97-linear optimization/stability policy
is explicitly revised and requalified from the affected gates.
