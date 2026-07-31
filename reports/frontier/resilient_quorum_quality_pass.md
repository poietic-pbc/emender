# Resilient quorum workstream quality pass

Task: `quality-pass-resilient-quorum-workstream`
Date: 2026-07-08

## Scope

This pass reviewed WG task metadata only. It did not inspect or edit source
code.

## Dependency Check

The downstream workstream is a sequential ladder:

1. `design-resilient-quorum-diloco-catchup`
2. `implement-resilient-quorum-diloco`
3. `validate-resilient-quorum-failure-injection`
4. `run-resilient-quorum-1n8n64n-ladder`
5. `evaluate-resilient-quorum-256n-debug-gate`
6. `synthesize-resilient-quorum-256n1h-package`

This order matches the intended progression: design, implementation,
failure-injection validation, 1n/8n/64n scale ladder, optional bounded 256n
debug smoke, then a report-only 256n x 1h approval package.

## Metadata Tightening

The initial chain already separated implementation from validation and blocked
large Slurm submissions in the early tasks. Two downstream descriptions were
tightened:

- `evaluate-resilient-quorum-256n-debug-gate`: made the 256n debug smoke a
  single bounded submission only, added node-hour and guard recording, and
  required separate evidence for nonjoining rank tolerance, stuck-rank timeout,
  stale-generation handling, stale/restarted-rank catchup, checkpoint
  finalization, and production latest/last protection.
- `synthesize-resilient-quorum-256n1h-package`: made the task explicitly
  report-only, prohibited auto-submitting follow-ups, required the final ladder
  boundary in the report, and required separate evidence grading for each
  resilience claim before any 256n x 1h recommendation.

## Final Ladder Boundary

The approved workstream boundary is:

`1n -> 8n -> 64n -> optional single bounded 256n debug smoke`

The 256n x 1h run is not authorized by this workstream. The last task may only
prepare an approval package. Any 1h or 12h submission requires a later explicit
human approval message or task.

## Production Guard

Development and debug rungs must use run-local output roots and must not mutate
production `latest` or `last` pointers. Evidence for checkpoint/latest behavior
must distinguish run-local finalization from production state.

## Validation Summary

- Dependencies were checked for the intended sequential ladder.
- No downstream task authorizes production latest/last mutation.
- No downstream task authorizes 1h or 12h submission.
- Metrics requirements now include concrete quorum, catchup, checkpoint, and
  terminal-state fields needed by downstream evaluators.
- Ambiguous approval-boundary language was edited in WG task metadata.
