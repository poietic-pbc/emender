# E97 35B MoE stage-3 continuation to 250.797B accepted tokens

Date: 2026-08-09

## Decision

The operator authorizes one additional exact 3,000-step continuation from the
immutable 200.466B canonical authority to 250,797,359,104 accepted tokens. The
run retains the established counter-v2 stream without another sampler
transition.

```text
nodes=256
ranks=2048
partition=batch
qos=normal
time_limit=06:00:00
max_steps=3000
train_minutes=0
diloco_k=40
save_every=200
keep_checkpoints=2
requeue=0
```

The restored complete eight-shard checkpoint is:

```text
step=2335080
accepted_tokens=200465711104
checkpoint=step-02335080-tokens-0000200465711104
manifest_sha256=14aaf718eea5fd18509a50f99176da5c8a7d922968bbd2bb0bddd23a5b002189
```

It is protected by a complete 25-file hard-linked milestone. Exact arithmetic:

```text
3000 * 16777216 = 50331648000 new accepted tokens
200465711104 + 50331648000 = 250797359104 final accepted tokens
2335080 + 3000 = 2338080 final step
3000 / 40 = 75 K40 merges
```

With the K-aligned 200-step cadence, periodic publications occur at steps
2,335,200 through 2,338,000, followed by the exact final publication at
2,338,080: 16 complete canonical checkpoints. The six-hour request is a safety
envelope; stopping is the exact positive step boundary. Stage-2 performance
predicts approximately five hours and about 1,288 node-hours; maximum requested
exposure is 1,536 node-hours.

## Sampler continuity

The exact retained sampler identity is:

```text
schema=emender-byte-window-counter-v2
corpus_sha256=44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9
tokenizer_sha256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069
sampler_key=42
data_world_size=2048
context_size=2048
stream_origin_accepted_tokens=150134063104
transition_from_legacy=0
```

The restored per-rank cursor is 12,000. Every B4 step advances it by exactly
four, so the final cursor must be 24,000. Resume and retry retain the existing
identity and continue at the checkpoint-bound cursor; there is no reseeding,
zero-origin reset, legacy transition, or sampler fallback.

## Concurrent paper diagnostic safety

At review time the only same-user scheduler activity was job `5216040`, a
32-node debug-QoS E97-linear diagnostic. It uses the separate
`e97_gdn2_paper_scale.sbatch` path, separate qualification output root, dense
1.3B model, B2 workload, and counter-v1 paper stream. It does not read or write
the MoE production run root or canonical checkpoints. The MoE submission uses
256 nodes at normal QoS and an immutable source snapshot. Changes on `main`
since the 200B launch touch only paper configuration, diagnostics, and dense
model files; the MoE production launcher, trainer, checkpoint, sampler, model,
and fused kernels are byte-unchanged. Scheduler overlap is therefore safe if
Slurm elects to run both jobs concurrently.

## Stop and evidence policy

Any rank, node, collective, timeout, nonfinite value, HIP error, sampler
identity/cursor mismatch, or checkpoint publication failure terminates the
fixed world nonzero. There is no automatic restart, shrink, requeue, emergency
checkpoint, or sampler fallback. Only complete K-aligned canonical checkpoints
are restart authority.

After completion, preserve the complete 250.797B authority as an immutable
hard-linked milestone before retention. Report exact scheduler state, steps,
merges, checkpoints, training and auxiliary loss, throughput, HBM, phase tails,
manifest identity, sampler cursor, and terminal accounting.

## Validation and architecture conformance

Authority is `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 production
same-allocation execution epochs, and the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable safety intent is **R07**,
**R12**, **R14/NDP13**, **R16**, and **NDP15** checkpoint atomicity. The
operator explicitly reviewed this 256-node stage. The rendered role is one
bounded fixed-world child with no SQLite/database/lock/metadata-heartbeat path
and no attempt to preserve, shrink, or relaunch a broken communicator.

Explicitly retired and unclaimed are **R02-R06, R08-R11; NDP01,
NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and ISP01-ISP07**. No elastic,
native-data-plane, asynchronous-overlap, background-checkpoint, or
communicator-shrink claim is made.

Pre-submission validation:

```text
checkpoints/latest resolves to step-02335080-tokens-0000200465711104
checkpoint manifest complete=true
checkpoint step=2335080
checkpoint accepted_tokens=200465711104
checkpoint sampler cursor=12000
checkpoint and milestone manifest SHA-256 both equal
  14aaf718eea5fd18509a50f99176da5c8a7d922968bbd2bb0bddd23a5b002189
HEAD equals origin/main before the authorization commit
tracked source is clean
no MoE production-relevant file differs from source 7bafaae4
```
