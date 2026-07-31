# Exact two-node batch-scoped seed destination fix

## Failure and fix

Slurm job `5059548` reached the exact two-node batch allocation with
`Partition=batch` and `QOS=debug`, but both node-local seed materializers
rejected their destinations before model load because submit-side processing
had erased the job-id component. No K40 work or model load occurred.

The production acceptance controller now writes the exact script it passes to
`sbatch` as each phase's `rendered.sbatch`. Rendering requires and preserves
the single-quoted literal template
`/tmp/emender-e97-seed-${SLURM_JOB_ID}`. The batch script admits only that
template and substitutes its token once, after Slurm has supplied the live job
ID. Ambient, empty, unset, or mismatched destination identities cannot select a
different path. The materializer additionally requires the exact
`emender-e97-seed-<live-job-id>` directory name before any authority fetch or
model load.

The immutable seed remains step `2300930`, `150793748480` tokens,
`7719680116` bytes, SHA256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The renderer and batch directives remain exactly `Partition=batch`,
`QOS=debug`, and two nodes. This implementation submitted no Slurm job.

## Conformance

This change conforms to Resilient DiLoCo Compute Pool version 1:

- **R01/R09:** the destination and immutable seed are validated before the
  model-owning trainer starts; rejection remains zero-model-work.
- **R10:** every node receives a live-job-scoped `/tmp` destination, while
  shared filesystem destinations remain rejected.
- **R14/R16:** the bounded exact two-node admission path and its retained
  scheduler binding are unchanged.
- **NDP13:** destination admission fails locally and before later stages.
- **NDP15:** immutable checkpoint handoff identity is unchanged.
- **NDP16:** per-node materialization evidence retains the live Slurm job ID,
  staged path, bytes, and digest.
- **NDP17:** this is the exact two-node rung; it neither weakens the G2 gate nor
  authorizes a larger or new job.

## Validation

Run from the canonical Frontier environment:

```text
source scripts/frontier/activate_emender_frontier.sh
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_s3_seed.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_resilient_e97_true_2n_launcher.py
```

The regression invokes the production renderer with `SLURM_JOB_ID` absent,
inspects the actual submission artifact for the deferred literal, evaluates
the same batch-time expansion with synthetic job `5059548`, and passes the
result to the production materializer validator. Separate cases prove unset,
empty, mismatched, and merely-containing-the-ID destinations fail closed.
