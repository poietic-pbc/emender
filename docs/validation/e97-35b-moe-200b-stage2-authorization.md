# E97 35B MoE stage-2 continuation to 200.466B accepted tokens

Date: 2026-08-08

## Decision

The operator authorizes stage 2 from the immutable 150.134B canonical authority
to exactly 200,465,711,104 accepted tokens. The operator explicitly declines a
held-out-loss gate at this sub-epoch exposure regime. The run introduces the
reviewed boundary-relative counter-v2 sampler at the complete 150.134B K40
authority; historical samples remain labelled legacy.

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
step=2332080
accepted_tokens=150134063104
checkpoint=step-02332080-tokens-0000150134063104
manifest_sha256=89383b916292c0d804a44a6835c21066d38c8108b953c8a3e9c58d88f79db2d4
```

It is protected by a complete 25-file hard-linked milestone. Exact arithmetic:

```text
3000 * 16777216 = 50331648000 new accepted tokens
150134063104 + 50331648000 = 200465711104 final accepted tokens
2332080 + 3000 = 2335080 final step
3000 / 40 = 75 K40 merges
```

With the K-aligned 200-step cadence, periodic publications occur at steps
2,332,200 through 2,335,000, followed by the exact final publication at
2,335,080: 16 canonical checkpoints. The six-hour request is a safety envelope;
stopping is the exact positive step boundary. Stage-1 performance predicts
about 5h05m and roughly 1,300 node-hours; maximum requested exposure is 1,536
node-hours.

## Sampler transition

The exact stage-2 sampler identity is:

```text
schema=emender-byte-window-counter-v2
corpus_sha256=44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9
tokenizer_sha256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069
sampler_key=42
data_world_size=2048
context_size=2048
stream_origin_accepted_tokens=150134063104
transition_from_legacy=1
```

The initial per-rank cursor is zero. Every B4 step advances it by exactly four.
The origin is part of every SHA-256 sample identity, creating a new deterministic
stream. Sampling retains the same corpus/tokenizer/uniform-byte-window
distribution and remains with replacement. Every canonical checkpoint binds
the origin, total accepted-token clock, exact relative cursor, and transition
boundary. Retry replays unaccepted work; a successful resume produces the exact
uninterrupted next tensors.

## Stop and evidence policy

Any rank, node, collective, timeout, nonfinite value, HIP error, sampler
identity/cursor mismatch, or checkpoint publication failure terminates the
fixed world nonzero. There is no automatic restart, shrink, requeue, emergency
checkpoint, or sampler fallback. Only complete K-aligned canonical checkpoints
are restart authority.

After completion, preserve the complete 200.466B authority as an immutable
hard-linked milestone before retention. Report exact scheduler state, steps,
merges, checkpoints, loss/auxiliary loss, throughput, HBM, phase tails, manifest
identity, sampler transition/cursor, and terminal accounting.

## Validation and architecture conformance

Authority is `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 production
same-allocation execution epochs (2026-07-31), and the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable safety intent is **R07**,
**R12**, **R14/NDP13**, **R16**, and **NDP15** checkpoint atomicity. The
operator explicitly reviewed this 256-node stage. The rendered role is one
bounded fixed-world child with no SQLite/database/lock/metadata-heartbeat path
and no attempt to preserve, shrink, or relaunch a broken communicator.

Explicitly retired and unclaimed are **R02-R06, R08-R11; NDP01,
NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and ISP01-ISP07**. No elastic,
native-data-plane, asynchronous-overlap, background-checkpoint, or
communicator-shrink claim is made.

Pre-submission evidence is
`docs/validation/e97-moe-counter-v2-150b-transition.md`: 56 tests passed and a
real-corpus preflight validated the actual 150B manifest transition, cursor
zero, one-step cursor four, and uninterrupted/resumed tensor equality.
