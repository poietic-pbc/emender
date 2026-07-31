# Deferred merge: production overlap entrypoint

Date: 2026-07-22
Task: `.merge-fix-production-overlap-entrypoint`
Source: `wg/agent-1414/fix-production-overlap-entrypoint` at `76ed9ac6`

## Resolution

The source branch was applied with squash semantics. The only content conflict
was in `reports/fix-production-overlap-entrypoint.md`. The resolution preserves
the target's retained-run topology, certificate, split-brain, and production
boundary analysis and adds the source branch's deterministic production-role
regression section. No evidence or limitation was discarded: the report still
states that this local probe does not constitute a new Frontier run.

The resulting code adds `production_overlap_probe` to the exact rendered role
and its regression to `tests/test_resilient_e97_true_2n_launcher.py`. The test
proves the production role records generation-1 K40 start after generation-0
discovery/quorum work starts but before generation-0 checkpoint publication.

## Validation

Validation used the canonical Frontier environment from
`scripts/frontier/activate_emender_frontier.sh` and exercised the exact new
production-role test together with `tests/test_native_pipeline.py`. Git's
whitespace/conflict checks passed, and the staged paths were limited to the
conflicting report, production role, and exact-renderer test.

Conformance was checked against *Resilient DiLoCo Compute Pool*, version 1
(2026-07-17). The change is a deterministic scheduling-edge regression for
R04, R06-R07, R11-R12, R14, and R16 and NDP10, NDP13, NDP15-NDP17. It does not
change READY membership, quorum floors, fencing, weighted math, transport,
backpressure, publication authority, or the minimum-progress policy. No Slurm
submission was made.
