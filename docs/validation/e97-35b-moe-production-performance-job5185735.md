# E97 35B MoE production performance correction — job 5185735

## Verdict

PASS for the corrected one-node production path. The earlier correctness-first
custom expert GEMMs were not performance-qualified. Production now retains the
proven fused E97 recurrence plus Triton routing/packing/combine and uses the
same tuned ROCm linear backend as dense E97 for shared and routed expert GEMMs.
Selective non-reentrant checkpointing retains packed expert inputs instead of
the 8,832-wide SwiGLU activation.

## Root causes corrected

1. Each of eight custom expert weight-gradient programs scanned the complete
   rank-level packed row range. Expert-local dynamic spans removed this roughly
   eightfold masked traversal.
2. The old autograd path retained the large activated tensor while also
   recomputing gate/up. The corrected checkpoint policy retains compact packed
   inputs and recomputes the expert activation once.
3. The fixed, correctness-first Triton GEMM tiles were substantially slower
   than Frontier's tuned ROCm GEMMs. An explicit `rocblas` backend now handles
   shared and routed expert linear operations; it is not an implicit fallback.
4. Batch one underfilled the expert GEMMs. Batch four was selected after B1/B2/B4
   sweeps; B4 remained bounded during a full 20-minute run.

## One-node acceptance

- Job: `5185735`
- Source: `65e1a678`
- Node: `frontier08716`
- Scheduler: `Partition=batch`, `QOS=debug`
- State: `COMPLETED`, exit `0:0`
- Measured training: `1,200.959 s`
- Steps: `181`
- Accepted tokens: `11,862,016`
- Mean/median loss: `2.61438 / 2.62639`
- Mean/median throughput: `11,690.96 / 11,681.83 tok/s`
- Median step time: `5.610 s`
- Maximum allocated HBM/GCD: `57,040,789,504 bytes`

The prior qualified job `5182403` achieved median `1,110.88 tok/s` at batch
one. The corrected production run is **10.52x faster per node** while retaining
finite training for at least 20 minutes. It processed 11,862,016 tokens versus
1,343,488 in the earlier qualification.

Short phase profiles established the progression:

| Job | Backend / batch | Warm median throughput | Representative forward / backward |
|---:|---|---:|---:|
| 5185088 | corrected custom Triton / B1 | ~1.5k tok/s | ~4.0 / 6.4 s |
| 5185219 | corrected custom Triton / B2 | ~2.7k tok/s | ~4.3 / 7.7 s |
| 5185308 | ROCm routed, Triton shared / B2 | ~7.3k tok/s | ~1.8 / 2.4 s |
| 5185400 | ROCm routed+shared / B2 | ~9.3k tok/s | ~1.7 / 1.6 s |
| 5185426 | ROCm routed+shared / B4 | ~13k tok/s short-run | ~2.2 / 2.6 s |

The long run settled lower than the short B4 profile as routing evolved, but
remained over ten times the old accepted throughput. Router balance and removal
of host-visible ragged split synchronization are the next optimization targets.

## Parity and authority

`tests/test_e97_moe_ep_triton.py` compares production ROCm expert outputs and
all input/weight gradients against the Triton implementation and the FP32
oracle. The complete relevant local suite passed 21 tests before submission.
Only post-mixer FFNs changed; E97 recurrent mixers, state, projections, gates,
embeddings, and norms remain on the established E97 implementation.

This remains one eight-GCD node-local expert island. Expert tokens do not cross
nodes. The change conforms to R07, R12, R14/NDP13, R16 and the applicable
fixed-world ADR-003 checklist in `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.
