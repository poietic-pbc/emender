# Deterministic E97 tensor reducer evidence

This implementation conforms to `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`,
architecture decision version 1 (2026-07-17), for R05, R08, and R15.

- R05: lexicographically bound full tensor layouts, float64 wire values,
  identity-sorted `math.fsum` token-weighted reduction, and unequal-weight
  high-precision reference comparison.
- R08: deterministic per-attempt owner mapping, checksummed bounded chunks,
  owner-local byte backpressure, idempotent receipts, conflicting replay
  rejection, and prompt payload release. Reducers own one shard; there is no
  central full-model broker and no shared-filesystem tensor hot path.
- R15: the eight-member fresh equal-token cohort is compared with synchronous
  DiLoCo at `rtol=1e-6, atol=1e-7`; changing token weights are compared with a
  float64 reference at `rtol=6e-8, atol=2e-7`.

## Exact validation commands and numerical artifacts

```bash
PY=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
"$PY" -m pytest -q tests/test_resilient_e97_reducer.py
"$PY" -m pytest -q tests/test_resilient_node_transport.py tests/test_resilient_e97_split_roles.py tests/test_resilient_e97_runtime.py
python3.11 -m compileall -q ndm/resilient_e97_reducer.py tests/test_resilient_e97_reducer.py
git diff --check
```

The deterministic fixture contains seven representative E97/full-model tensor
classes, 511 float32 parameters, 40 bounded 104-byte shards, unequal token
weights `(3, 1000003, 29)`, and two opposing arrival orders. The equal-weight
artifact uses eight fresh contributions of 4096 tokens each and 80-byte shards.
The materialized hashes and tolerances are in the adjacent JSON artifact.
