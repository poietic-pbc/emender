# E97/GDN2 paper deterministic sampler validation

**Status:** implementation preflight PASS

**Date:** 2026-08-08
**Scope:** CPU/login-node implementation evidence only; this is not a Frontier
GPU qualification or scale-rung result.

## Bound identities

- schema: `emender-byte-window-counter-v1`
- corpus SHA-256:
  `44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9`
- p50k cache SHA-256:
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`
- fixed key: `42`
- context: 2,048 accepted/predicted tokens per sample
- world and accepted cursor: exact launch/checkpoint fields, never inferred from
  a mutable seed

The scheduler verification for the corpus and tokenizer is retained separately
in `e97-paper-corpus-sampler-receipt.md`.

## Implemented contract

`ndm/data/tokenized_dataset.py` derives each candidate byte position from a
canonical SHA-256 encoding of schema, corpus digest, tokenizer digest, key,
world, global rank, absolute per-rank sample index, and bounded retry index.
Accepted tokens convert to the sample cursor only when exactly divisible by
`world * context`. B2 consumes consecutive indices.

Dense `train.py` now:

- accepts the complete sampler identity as rendered CLI arguments;
- rejects partial identities and launched-world drift;
- writes identity/cursor/accepted tokens into run and checkpoint metadata;
- validates metadata through a checkpoint preflight callback before loading
  model or optimizer state;
- publishes the metadata in the same atomic checkpoint payload addressed by
  `latest.pt`;
- explicitly labels newly written mutable-RNG checkpoints as legacy.

MoE training and `ndm/e97_moe_checkpoint.py` now:

- use the same sampler implementation and identity;
- persist sampler metadata in shard payloads, rank sidecars, and the canonical
  manifest;
- validate manifest/sidecar/shard clocks and identities before tensor restore;
- refuse to relabel missing/legacy sampler metadata;
- permit an explicit legacy-to-counter transition only at a complete
  K-aligned checkpoint and retain the old/new boundary;
- render the sampler identity and transition intent in launcher receipts.

Historical jobs and checkpoints are unchanged. In particular, independently
running job 5208321 uses its immutable earlier source snapshot.

## Requirement coverage

1. uninterrupted versus resumed tensors: executable equality test;
2. retry reproduces unaccepted work: executable tensors/sample-ID test;
3. successful continuation advances exactly: executable cursor/next-sample test;
4. batch grouping independence: B4 versus B2+B2 executable test;
5. ranks deterministic and distinct: executable ID/position/tensor test;
6. E97, E97-linear, GDN2, and MoE call-site agreement: shared-identity and
   byte-identical batch test (the dense arms share `build_training_dataset`);
7. schema/cursor/world/context/corpus/tokenizer drift: parameterized fail-closed
   tests for dense and MoE;
8. atomic publication/fresh restore: `latest.pt` metadata test plus a spawned
   fresh-Python-process next-batch test.

Bounded retry exhaustion and the property that retrying one sample cannot
perturb later samples are tested separately.

## Exact validation

```bash
source scripts/frontier/activate_emender_frontier.sh
bash -n \
  scripts/frontier/e97_35b_moe_production.sbatch \
  scripts/frontier/submit_e97_35b_moe_scale.sh
"$EMENDER_PYTHON" -m py_compile \
  train.py \
  ndm/data/tokenized_dataset.py \
  ndm/e97_moe_checkpoint.py \
  scripts/frontier/e97_35b_moe_train.py
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_tokenized_counter_sampler.py \
  tests/test_train_helpers.py \
  tests/test_e97_moe_checkpoint.py \
  tests/test_e97_moe_production_launcher.py
```

Result: `51 passed in 33.78s`.

The next gate is exact three-arm graph/config initialization manifests, followed
by current-source one-GCD and one-node kernel qualification. A clean preflight
does not claim those unrun machine gates.
