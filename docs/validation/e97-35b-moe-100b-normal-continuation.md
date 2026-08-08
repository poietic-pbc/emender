# E97 35B MoE continuation from 40B to 100B accepted tokens

Date: 2026-08-08

## Decision

Continue the verified canonical 40,075,526,144-token MoE checkpoint in one
fixed-world 256-node production epoch.  Use normal QoS to avoid five repeated
model-load/staging cycles and request a seven-hour safety envelope for an
estimated 6h05m--6h15m execution.  Scientific progress remains exact-step
bounded; wall time is not a stopping condition.

```text
nodes=256
ranks=2048
partition=batch
qos=normal
time_limit=07:00:00
max_steps=3600
train_minutes=0
diloco_k=40
save_every=160
keep_checkpoints=2
requeue=0
```

The restored authority is the complete checksummed eight-shard checkpoint:

```text
step=2325520
accepted_tokens=40075526144
checkpoint=step-02325520-tokens-0000040075526144
```

That checkpoint is also retained as an immutable hard-linked milestone with
manifest SHA-256
`08ccf835253f97b88aadecda42f23eddc10470cb1c53c291e08987b34cff175a`.
It descends from the verified 513B processed-exposure dense E97 seed through
function-preserving MoE upcycling; 513B is not claimed as unique data.

At 16,777,216 tokens per exact 256-node step, the epoch arithmetic is:

```text
3600 * 16777216 = 60397977600 new accepted tokens
40075526144 + 60397977600 = 100473503744 final accepted tokens
```

The run therefore contains exactly 90 K40 merges.  Because the restored step
is 80 modulo 160, periodic checkpoints occur first after 80 steps and then
every 160 steps, including the exact final step: 23 publications total.  The
maximum durable rollback after the first save is 160 steps.  A rank, node,
collective, timeout, or publication failure terminates the complete job
nonzero; there is no automatic restart, shrink, scheduler requeue, or damaged
communicator reuse.

## Basis

The operational source includes the bounded ragged-row Triton fix documented
in `e97-35b-moe-hip209-ragged-specialization-fix.md`.  On that exact code,
256-node jobs 5195332 and 5195870 completed 320 and 880 steps.  Evidence-only
source then ran job 5200112 for another exact 880 steps, 22 merges, and 11
checkpoints, terminating `COMPLETED 0:0` in 1:49:37 at the authority above.
The last run sustained approximately 5.14M token/s median, finite whole-run
mean language-model loss 2.11196, and peak allocated HBM 54,460,125,696
bytes/GCD without HIP-209, collective, memory, or numerical failure.

The longer epoch is expected to save roughly 400 node-hours relative to four
880-step debug epochs plus a short tail by paying source staging and model
restore once.  The seven-hour request is a safety envelope; allocation usage
is actual elapsed node-hours.

## Validation and architecture conformance

Before model load, the immutable production launcher must verify and retain
scheduler output naming both `Partition=batch` and `QOS=normal`, plus
`Requeue=0`, `NumNodes=256`, `NumTasks=2048`, and `TimeLimit=07:00:00`.
Terminal evidence must again name Partition and QoS separately, exit `0:0`,
show exactly 3,600 steps, 90 merges, 23 checkpoint publications, finite loss
and routing auxiliary loss, bounded HBM, and final complete eight-shard
authority `(2329120, 100473503744)`.  The final checkpoint will be preserved as
a milestone before retention can remove it.

Authority is `RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 production
same-allocation execution epochs (2026-07-31), and the production crosswalk in
`RESILIENT_DILOCO_GAP_MATRIX.md`.  Applicable safety intent is **R07**,
**R12**, **R14/NDP13**, **R16**, and **NDP15** checkpoint atomicity.  The
operator explicitly reviewed 256 nodes in this session.  The path remains one
fixed-world fail-stop child with canonical atomic restart authority.

Explicitly retired and unclaimed here are **R02-R06, R08-R11; NDP01,
NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and ISP01-ISP07**.  This run
makes no elastic membership, native data-plane, asynchronous overlap,
background checkpoint, or communicator-shrink claim, and adds no SQLite,
database, filesystem lock, or metadata-heartbeat dependency.
