# Final disposition: job 5000436

Recorded: 2026-07-15 after terminal accounting and continuation recovery.

## Outcome

This validation is **incomplete**. Do not submit or retry another 256-node job
from this task.

- Debug job `5000436` was cancelled exactly once at the user's explicit
  direction and is terminal `CANCELLED` after `00:42:43` elapsed.
- The task did not reach scheduler-controlled finalization near two hours.
- An independent reload of the final checkpoint's model, optimizer, step, and
  async-chain metadata was not performed.
- This is classified as **user-directed early termination / incomplete
  scheduler validation**, not as a demonstrated deterministic code/config,
  resource, node/rank-loss, OOM, or non-finite-loss failure.
- Production job `4980157` was never mutated by this task. Its later terminal
  cancellation was performed outside this task and it never started or
  received an allocation.

## Scale evidence retained

- 256 nodes and 2,048 ranks started from the approved step-1,525,000 seed.
- Generations 0 through 9 and ten all-rank merges finalized.
- Training advanced 400 steps to step 1,525,400.
- Preserved evidence showed no OOM, node/rank loss, missing merge participant,
  or non-finite-loss signal before cancellation.

## Immutable continuation

Persistent run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z
```

The `step1065000` directory component is a cosmetic stale variant label; the
submitted and loaded seed was the approved step-1,525,000 checkpoint.

Atomic pointer:

```text
continuation/last-valid.json
```

Immutable manifest:

```text
continuation/last-valid-20260715T072807Z.json
```

Last valid checkpoint:

```text
async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt
```

- Size: `15439252298` bytes
- SHA256: `ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`
- The post-terminal selector verified existence, size, and full-file SHA256.
- The immutable manifest's `step` JSON field is null. Consumers must derive
  step 1,525,400 from the checkpoint filename and finalized generation record,
  and must not represent the null field as populated.

## Canonical reports

- `reports/frontier/e97-exact-256n-2h-debug-20260715.md`
- `reports/frontier/e97-exact-256n-2h-debug-terminal-audit-20260715.md`
- `build/e97-256/exact-debug-256n-2h-20260715/replacement-terminal-20260715T0728Z.txt`
