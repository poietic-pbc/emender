# Exact two-node batch seed destination integration

## Integrated history

The reviewed implementation from
`fix-exact-2n-batch-seed-destination` is commit
`04bc3441c3f0cd3c2e7aa27fb33ae43f547bf860` in this integration history
(the worker branch published the equivalent reviewed commit as `a6b9ac2d`).
Authoritative `origin/main` had advanced independently to
`7ab92adabcd63ae4c5d0cf2d2c408b0fe182a944`, so the histories were merged
instead of replacing either side. This retains the reviewed fix and concurrent
upstream work.

The implementation review confirmed that the renderer copies the production
batch script into `rendered.sbatch` and requires the single-quoted literal
`'/tmp/emender-e97-seed-${SLURM_JOB_ID}'`. Consequently, submit-time rendering
does not consume `SLURM_JOB_ID`. The batch script validates that exact template
and performs its sole substitution only after Slurm supplies the live
`SLURM_JOB_ID`. The materializer independently requires the exact parent
directory `emender-e97-seed-<live-job-id>`, rejecting absent, empty,
mismatched, or merely containing identities before download or model work.

## Immutable bindings

The integrated exact two-node launch remains:

- seed step: `2300930`
- accepted tokens: `150793748480`
- checkpoint size: `7719680116` bytes
- checkpoint SHA256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`
- nodes: `2`
- Slurm partition: `batch`
- Slurm QoS: `debug`

No Slurm submission was made during this integration.

## Design conformance

This integration was reviewed against Resilient DiLoCo Compute Pool version 1
and its companion gap matrix:

- **R01/R09:** admission rejects an invalid batch-scoped seed destination
  before a model-owning trainer starts, preserving zero-model-work failure.
- **R10:** the materialized seed is live-job-scoped node-local `/tmp` state;
  shared-filesystem destinations remain fail-closed.
- **R14/R16:** the bounded two-node acceptance controller, terminal evidence,
  and exact `Partition=batch`/`QOS=debug` rung remain unchanged.
- **NDP13:** malformed or stale destination identity is contained locally and
  rejected before later training stages.
- **NDP15:** the immutable, digest-bound checkpoint handoff is unchanged.
- **NDP16:** runtime materialization evidence retains job identity, path,
  bytes, and digest.
- **NDP17:** the change repairs only the exact two-node rung and neither
  weakens the retained G2 gate nor authorizes a 4+ node launch.

## Validation

The following validation ran after reconciling current `origin/main`, in the
canonical Frontier environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
"$EMENDER_PYTHON" -m compileall -q \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  scripts/frontier/materialize_e97_s3_seed.py
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_s3_seed.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_resilient_e97_true_2n_launcher.py
git diff --check origin/main...HEAD
```

Result: `88 passed in 41.16s`; shell syntax, compilation, and whitespace checks
also passed. The regression renders with submit-time `SLURM_JOB_ID` absent,
asserts the literal remains in the actual submission artifact, expands it
under synthetic batch job `5059548`, and passes the result through the
production destination validator. Negative cases cover unset, empty,
mismatched, legacy, shared-filesystem, stale, wrong-size, and wrong-digest
inputs.
