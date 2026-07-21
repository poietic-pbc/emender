# Exact two-node pipelined telemetry harvest fix

The clean-overlap acceptance phase now validates the supervisor's durable
post-run evidence at `RUN_DIR/retained-evidence`.  The performance validator
recursively reads JSONL files below that root, so telemetry harvested from both
`node-0/telemetry` and `node-1/telemetry` participates in admission.  Its exact
K40 counts, background-overlap requirement, strict foreground-idle maximum,
and conditional 1.25x cadence maximum are unchanged.

## Validation

- `test_post_supervisor_retained_evidence_recursively_harvests_both_nodes`
  reproduces the job-5043045 layout and proves both trainer identities and both
  node managers are consumed before the unchanged performance gates pass.
- `test_clean_overlap_validates_post_supervisor_retained_node_telemetry` proves
  the batch invokes the validator after the supervisor and selects retained
  evidence rather than the already-harvested node-local telemetry directory.
- `pytest -q tests/test_resilient_e97_true_2n_launcher.py
  tests/test_validate_pipelined_e97_performance.py tests/test_native_pipeline.py`:
  69 passed.  Adding the acceptance-renderer file produced 73 passed and its
  expected clean-tree refusal while these source edits were uncommitted.
- The canonical Frontier Release native build completed.  CTest passed 10/10.
- The runtime suite advanced through 29 tests after the native library was
  built and configured, then a native subprocess terminated the invoking login
  shell without a pytest failure report.  The telemetry-specific focused suite
  is fully green.
- `git diff --check` passed.
- No `sbatch`, `srun` job launch, or other Slurm submission was performed.

## Resilient DiLoCo conformance

Checked against *Resilient DiLoCo Compute Pool*, version 1, and the companion
gap matrix.  This evidence-path-only change preserves:

- **R07, R12, NDP15:** no commit, checkpoint, fence, outer-state, handoff, or
  publication behavior changes; the validator reads retained evidence only.
- **R11:** telemetry from every retained node is included recursively, without
  introducing a launched-rank admission invariant or changing rejoin/catch-up.
- **R14, NDP13:** the exact K40, overlap, idle, cadence, and bounded-stage SLO
  evidence continues to fail closed; only its post-supervisor location changes.
- **R16, NDP17:** this remains the exact two-node gate and makes no 4+ scale
  claim.  No Slurm rung was submitted.
- **NDP10:** checksummed/idempotent native data-plane behavior is untouched.
- **NDP16:** both nodes' required stage/trainer telemetry is now visible to the
  acceptance result after bounded node-local evidence release.

The minimum progress floor and production topology remain exactly two READY
node managers, sixteen real trainers, global quorum two, positive configured
token minimum, and two steady-state exact-K40 generations per observed trainer.
