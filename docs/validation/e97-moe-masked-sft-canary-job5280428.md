# E97-MoE masked-SFT LR canary evaluation — job 5280428

Date: 2026-08-16  
Verdict: **masked SFT is optimizing the intended likelihood, but neither arm is
a usable assistant; 5e-6 is the stronger LR while the final router becomes
unacceptably concentrated. Do not extend either checkpoint unchanged.**

## Execution

Evaluation job `5280428` completed `0:0` in 15m49s on three independent
node-local eight-GCD EP islands with `Partition=batch`, `QOS=debug`, and
`Requeue=0`. Source was `f6b9cedf`. The frozen corrected panel SHA-256 was
`fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f`.

Inputs were the immutable 282B parent and matched 512-update SFT checkpoints.
Each SFT arm consumed exactly 227,417,738 packed tokens and 161,788,210
assistant-target tokens at 4K, with eight K64 target-weighted DiLoCo merges.

Results:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-sft-canary-v1/runs/job-pending/results/`

## Likelihood and general diagnostics

| Checkpoint | Wiki 2K | Wiki 8K | Wiki 32K | External assistant NLL | MMLU | HellaSwag norm |
|---|---:|---:|---:|---:|---:|---:|
| 282B parent | 3.35645 | 3.27295 | 3.25537 | 2.07251 | 20.31% | 44.92% |
| SFT 2e-6 | 3.35107 | 3.26770 | 3.24988 | 2.04625 | 18.36% | 44.92% |
| SFT 5e-6 | **3.33130** | **3.24829** | **3.22729** | **1.93784** | 19.53% | 44.92% |

Paired parent-to-5e-6 changes were favorable and statistically resolved:

- WikiText: -0.02515 at 2K, -0.02466 at 8K, and -0.02808 at 32K;
- per-example assistant-response NLL: -0.22550, bootstrap interval
  `[-0.29529, -0.16006]`;
- HellaSwag normalized accuracy: unchanged;
- MMLU: -0.78 point with interval crossing zero.

The 2e-6 arm made smaller likelihood gains and had a larger observed MMLU
reduction. The 5e-6 arm is therefore the better LR candidate on likelihood and
general-LM evidence. This proves that target masks, target normalization, and
fresh optimization produce a real external response-likelihood update; it does
not prove assistant behavior.

## Generation remains a hard failure

All 16 greedy generations from every checkpoint exhausted the 256-token cap
without RS/EOT. Parent and SFT outputs remained dominated by short generic
fragments such as `The results of the same time.` followed by blank lines.
There was no reliable arithmetic, constrained list, JSON, code, SQL, editing,
translation, refusal, or tool-call behavior.

Sampled generations remained incoherent mixtures of prose, code, markup,
citations, and numbers. Stop counts over 16 sampled prompts were:

- parent: 2;
- 2e-6: 0;
- 5e-6: 5.

The extra 5e-6 stops followed incoherent text and are not instruction
following. Neither arm passes the assistant-foundation gate.

## Retrieval remains absent

All retrieval variants remained near eight-choice chance with negative mean
margins at every distance. The small accuracy movements are not monotonic and
provide no evidence that 4K SFT created or damaged robust long retrieval.
Long-context specialization remains deferred.

## Routing regression

The final MoE layer changed from already-skewed to effectively collapsed on the
evaluation panel:

| Checkpoint | Layer-10 max/mean | Layer-10 CV | Layer-10 entropy | Minimum expert count |
|---|---:|---:|---:|---:|
| parent | 4.709 | 1.058 | 0.897 | 2,015 |
| 2e-6 | 8.611 | 1.716 | 0.723 | 0 |
| 5e-6 | 8.876 | 1.840 | 0.718 | 0 |

Other layers remained close to parent routing. Direct checkpoint tensor audit
confirmed that layer 10 is unusually sensitive: router-weight relative delta
was 0.00513 at 2e-6 and 0.01178 at 5e-6, versus roughly 0.0016–0.0053 in the
other layers. Thus the routing result is model-state drift, not merely noisy
evaluation accounting.

All recurrent states remained finite. Peak training/evaluation memory was
bounded. The failure is behavioral and routing-specific, not numerical
instability.

## Decision and next experiment

Do not continue either trained checkpoint unchanged and do not begin bounded
production SFT. Preserve both as scientific evidence.

The next canary should restart from the clean 282B parent at 5e-6 and isolate
router stabilization while keeping identical data and token budget:

1. frozen-router arm: zero all router-weight gradients, retaining the parent
   routing function while training backbone and experts;
2. strengthened-router-objective arm: retain trainable routers but multiply the
   existing load-balance/z auxiliary by a reviewed fixed factor.

Require parent-like layer-10 load/entropy, continued assistant-NLL improvement,
and the first coherent native-template generations before any extension. If
routing is healthy but generation remains incoherent after a larger cumulative
supervised exposure, investigate an architectural/generation limitation on the
1.3B E97 proxy rather than spending billions of 35B tokens blindly.
