# Async DiLoCo B4/K40 Parity Debug

Date: 2026-07-09

## Summary

The earlier async-quorum smoke loss around `13.x` was caused by recipe drift and incomplete
resume semantics, not by the E97 checkpoint being from scratch.

Two fixes were landed:

- `9b70b39` `Align async DiLoCo smoke recipe with B4 K40`
  - async wrapper defaults changed from B1/chunk128/lr1e-4/K1-ish/tiny smoke behavior to
    B4/chunk2048/K40/lr0.001007/schedulefree/weight_decay=0.01.
  - async CLI now accepts and forwards optimizer, weight decay, warmup, min LR frac,
    grad accumulation, and grad clipping.
- `8b07aaa` `Load optimizer state for async DiLoCo resume`
  - async resume now loads checkpoint `optimizer_state_dict` into worker optimizers,
    matching `train.py --resume`, then resets param-group LR to the requested CLI LR.

Both commits were pushed to `origin/main`.

## Why Loss Was High

The previous async path was not the exact train.py smoke recipe:

| Field | Proven train.py smoke | Old async default | Fixed async parity |
| --- | --- | --- | --- |
| batch size | `4` | `1` | `4` |
| chunk size | `2048` | `128` | `2048` |
| local steps / K | `40` | could fall to `1` | `40` |
| LR | `0.001007` | `0.0001` | `0.001007` |
| optimizer | `schedulefree` | `adamw` default in async args | `schedulefree` |
| weight decay | `0.01` | `0.0` default in async args | `0.01` |
| checkpoint optimizer state | loaded by `train.py --resume` | not loaded | loaded |
| data | real CommaPile | real CommaPile | real CommaPile |
| synthetic stream | off | off | off |
| transport | train.py sync DiLoCo | compiled helper async quorum | compiled helper async quorum |

## Jobs Run

No jobs larger than 1 node were submitted during this debug. No 2n job was submitted because
the first parity attempt still had high loss.

### Job 4961602: recipe parity without optimizer-state resume

- Job: `4961602`
- Name: `async-b4k40-1n`
- State: `COMPLETED 0:0`
- Elapsed: `00:04:35`
- Nodes: `1`
- Queue: `batch/debug`
- Commit: `9b70b39`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_exact_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4_k40_exact_1n/4961602-20260709T113216Z`
- Result:
  - rank starts `8/8`
  - accepted updates `8`
  - timed out `0`
  - tokens `2,622,720`
  - loss `7.94267`

Conclusion: fixing batch/chunk/K/LR/optimizer CLI was not enough. The remaining mismatch was
that async loaded model weights only, while `train.py --resume` loaded optimizer state.

### Job 4961611: recipe parity with optimizer-state resume

- Job: `4961611`
- Name: `async-b4k40-opt-1n`
- State: `COMPLETED 0:0`
- Elapsed: `00:04:26`
- Nodes: `1`
- Queue: `batch/debug`
- Commit: `8b07aaa`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_exact_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4_k40_optstate_1n/4961611-20260709T114016Z`
- Result:
  - rank starts `8/8`
  - accepted updates `8`
  - timed out `0`
  - tokens `2,622,720`
  - tokens/sec `38,868.41`
  - transport `compiled-cray-mpich-helper-collective-reduce`
  - TCP dense data plane `false`
  - loss `2.52576`

This matches the successful train.py B4/K40 smoke ladder range.

## Current Status

The async DiLoCo exact-parity 1n smoke now passes the quality sanity check.

The WG task `debug-async-quorum` is paused because WG service dispatch crashed on that task's
assignment path. Do not resume it until the WG dispatcher issue is resolved or the work is
reframed as explicit shell/report tasks.

## Recommended Next Step

Create a new WG task, after the WG dispatcher issue is addressed, to run the exact same
fixed async DiLoCo recipe at 2 nodes in `batch/debug`. The only intended variable should be
node count.

