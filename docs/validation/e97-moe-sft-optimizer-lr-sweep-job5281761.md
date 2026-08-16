# E97-MoE masked-SFT optimizer/LR sweep — jobs 5281761 and 5281927

Date: 2026-08-16  
Verdict: **full-model masked SFT learns strongly with fresh state near `1e-4`;
preserved full-model ScheduleFree state is uniformly harmful. Fresh router
state destabilizes routing, motivating a crossed router/backbone-state canary
before multinode promotion.**

## Execution and validation authority

Job `5281761` completed `0:0` in 22m29s with ten independent one-node,
eight-GCD node-local EP worlds. Every child step completed `0:0`. Scheduler
evidence explicitly recorded `Partition=batch`, `QOS=debug`, `Requeue=0`, ten
nodes, and 80 tasks. Source was
`c3628f1ff026f145fafcf6125de8c2784cec22a1`.

Each arm trained for 64 updates on exactly 1,787,701 packed tokens and 1,206,497
assistant-target tokens. No cross-node DiLoCo was used; this was a local
optimizer screen. Weight decay was matched at `0.01`. No ~200GB diagnostic
checkpoints were published.

The prior two-batch metric covered only 16 replacement-sampled packs and about
42K targets. This sweep introduced exact validation enumeration: all 1,777
held-out packs and all 4,352,510 assistant targets exactly once, with 2,000
fixed-seed pack bootstraps and layer-wise routing counts.

A source-followup validation-only job, `5281927`, completed `0:0` in 13m20s on
`Partition=batch`, `QOS=debug`. It established the exact parent baseline:

- target-weighted NLL: `1.6707942057`;
- bootstrap interval: `[1.6376605230, 1.7041276405]`;
- layer-10 routing max/mean `5.882`, normalized entropy `0.9175`, zero unused
  experts;
- zero unused experts in every layer.

## Results

| State | LR | Exact NLL | Delta from parent | Worst max/mean | Min entropy | Max unused experts |
|---|---:|---:|---:|---:|---:|---:|
| fresh | `5e-6` | 1.63863 | -0.03216 | 5.454 | 0.8016 | 5 |
| fresh | `3e-5` | 1.41086 | -0.25994 | 6.563 | 0.8142 | 5 |
| **fresh** | **`1e-4`** | **1.28110** | **-0.38970** | **5.870** | **0.8666** | **3** |
| fresh | `3e-4` | 1.30598 | -0.36482 | 11.317 | 0.7883 | 8 |
| fresh | `1.007e-3` | 1.67141 | +0.00062 | 15.910 | 0.5759 | 4 |
| preserved | `5e-6` | 1.68204 | +0.01125 | 5.860 | 0.9169 | 0 |
| preserved | `3e-5` | 1.94501 | +0.27421 | 6.114 | 0.9143 | 0 |
| preserved | `1e-4` | 2.08017 | +0.40938 | 5.862 | 0.9137 | 0 |
| preserved | `3e-4` | 1.96070 | +0.28991 | 6.238 | 0.9091 | 0 |
| preserved | `1.007e-3` | 1.80262 | +0.13183 | 5.316 | 0.8935 | 0 |

Negative NLL deltas are improvements. The fresh-state LR curve is well resolved
and peaks around `1e-4`; `3e-4` is close in likelihood but substantially worse
in routing. `1.007e-3` is too large: it produces no held-out gain and severe
routing concentration. The tiny `5e-6` rate learns only weakly over this short
exposure, validating the concern that it is too small for direct BF16 updates.

Preserved full-model state keeps routing close to parent, but every LR worsens
held-out NLL. Exact enumeration confirms, rather than weakens, the earlier
small-sample indication that long-context ScheduleFree history conflicts with
the masked objective.

## New mechanistic conclusion

The crossed results isolate two effects:

1. backbone/experts need fresh objective-appropriate state to learn;
2. routers benefit from mature state and become unstable when their FP32
   ScheduleFree state is reset.

The next lowest-risk full-capacity experiment is therefore not LoRA and not
another plain LR sweep. It is a one-node `1e-4` optimizer-state factorial:

- preserve router `z`/second moment/clocks while resetting non-router state;
- the converse diagnostic, fresh router state with preserved non-router state.

This keeps every model parameter trainable. If the first hybrid retains the
fresh-`1e-4` likelihood gain while preserving parent-like router coverage, it
becomes the promotion candidate. Plain fresh `1e-4` remains the control.

No 8-node or larger SFT run is authorized until cached/full-prefix decoding
parity and this state-factorial pass.

## Architecture conformance

These are ADR-003 fixed-world diagnostic worlds. Applicable safety intent:
R07/R12 atomic authority binding, R14/NDP13 bounded fail-stop execution, R16
evidence discipline, and NDP15 checkpoint atomicity (no model checkpoint was
requested in the sweep). Elastic R02-R06/R08-R11, NDP02 dynamic semantics,
V21S01-V21S17, and ISP01-ISP07 are explicitly retired and unclaimed.
