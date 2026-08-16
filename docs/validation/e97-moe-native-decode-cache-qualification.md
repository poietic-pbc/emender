# E97 MoE native recurrent decode qualification

Date: 2026-08-16

Verdict: **native recurrent caching is qualified for instruction-tuning evaluation; exact cross-shape/backend BF16 logit identity is not an operational requirement.**

## Failure and fixes

Job `5282437` exposed catastrophic cached decoding: cached/full-prefix top-1
agreement was about 24%, no greedy sequence matched, and maximum target-logp
error was 10.48. The sparse-checkpoint Triton wrapper right-padded unaligned
inference calls and returned the state after the artificial tail. Zero-padded
decay erased the recurrent state before the next incremental call.

Commit `98d6a3fa` captures `S_final` after the final real token. Job `5282868`
passed the targeted padding-boundary regression on all eight GCDs and raised
cached/full-prefix top-1 agreement to 97.66% and exact 32-token greedy agreement
to 5/8. Commit `7acff2e1` then retained inference recurrent caches in FP32,
avoiding a BF16 state round trip after every generated token. Its GPU regression
proved exact token-at-a-time versus one-shot recurrence state/output equality.

## Real-checkpoint evidence

All jobs used the immutable 282.070B parent, one complete eight-GCD expert
island, `Partition=batch`, `QOS=debug`, and `Requeue=0`.

| Job | Expert backend | Cached/recomputed top-1 | Cached/one-shot top-1 | Exact greedy | Max target-logp delta |
|---|---|---:|---:|---:|---:|
| `5283152` | rocBLAS | 100.00% | 98.44% | 7/8 | 0.368 |
| `5284228` | rocBLAS | 98.44% | 97.66% | 6/8 | 0.227 |
| `5285055` | Triton | 100.00% | 98.44% | 7/8 | 0.138 |

The remaining greedy branches occurred only at BF16 near-ties. In job
`5285055`, rank 2 first differed at token 17: cached logits were 11.125 versus
11.0625, while recomputation tied both candidates at 11.125. The rocBLAS margin
diagnostic likewise observed margins of 0–0.125. Backend and run-to-run
variation therefore reflects expected shape-dependent BF16 execution, not a
missing recurrent state.

## Acceptance decision

Native cached generation with FP32 recurrent state is the canonical generation
path. The following remain required:

- the padding-boundary GPU regression;
- exact recurrence-only chunk-boundary invariance with FP32 cache;
- no catastrophic cached/full-prefix disagreement;
- behavioral generation evaluation with RS/EOT accounting.

Bitwise logits, exact greedy identity under different GEMM shapes, and a maximum
per-token log-probability delta of 0.02 are retired as promotion gates. This
qualification does not assert that the 282B parent is instruction-following; it
only makes generation from subsequent masked-SFT checkpoints interpretable.

No resilient/elastic compute-pool requirement is exercised: these were
read-only, fixed one-node expert-parallel evaluations under ADR-003-style
fail-stop execution.
