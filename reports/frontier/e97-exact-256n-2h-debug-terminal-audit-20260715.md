# Terminal audit: exact E97 256-node debug validation

Audit time: 2026-07-15 (post-terminal accounting)

## Disposition

The validation is **incomplete**, with no further submission authorized or
performed. Debug job `5000436` was cancelled at the user's explicit direction
after 42 minutes 43 seconds, so it did not reach scheduler-controlled
finalization near the requested two-hour limit. An independent deserialization
of model, optimizer, step, and async-chain state was also not completed before
cancellation.

The useful scale gates did pass before cancellation: 256 nodes and 2,048 ranks
started, generations 0 through 9 completed, ten all-rank merges finalized, and
training advanced from approved seed step 1,525,000 to checkpoint step
1,525,400. Preserved monitoring evidence reports no OOM, node/rank loss,
non-finite loss, or failed/missing merge participants.

## Terminal Slurm accounting

Read-only `sacct` returned:

```text
4980157|CANCELLED by 19032|0:0|2026-07-13T08:06:54|None|2026-07-15T03:28:54|00:00:00|12:00:00|256|0|None
5000436|CANCELLED by 19032|0:0|2026-07-15T02:31:50|2026-07-15T02:45:14|2026-07-15T03:27:57|00:42:43|02:00:00|256|28672|None
```

Production job `4980157` never started or received an allocation. Its later
cancellation was external to this task; this task issued no mutation command
against it.

## Immutable continuation handoff

Persistent run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z
```

The atomic pointer `continuation/last-valid.json` resolves to immutable manifest
`continuation/last-valid-20260715T072807Z.json`. It selects generation 9 and:

```text
async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt
```

Checkpoint size is `15439252298` bytes and SHA256 is
`ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`.
The recorded next-launch selector was executed post-terminal and passed its
existence, size, and full-file SHA256 checks, returning that exact checkpoint.
The manifest's `step` JSON field is null, but the immutable checkpoint filename
and the finalized generation evidence identify step 1,525,400; downstream
registration must preserve this caveat rather than claim a populated manifest
field.

## Acceptance result

- PASS: exact 256-node/2,048-rank allocation and approved attested seed.
- PASS: at least five generations, five all-rank merges, and progress beyond
  step 40 (ten generations; 400 steps).
- PASS: multiple durable finalized checkpoints and a checksum-verified,
  immutable continuation selection.
- FAIL/NOT OBSERVED: scheduler-controlled finalization near two hours, because
  the user authorized early cancellation.
- FAIL/NOT PERFORMED: independent model/optimizer/step/async-chain reload.
- PASS: no duplicate replacement submission after job `5000436`.
- PASS: no production-job mutation by this task.

The correct task disposition is therefore incomplete, not successful and not a
deterministic training failure. No unchanged 256-node retry should be launched.
