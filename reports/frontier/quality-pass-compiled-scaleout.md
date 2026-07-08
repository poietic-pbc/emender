# Compiled Helper Scaleout Quality Pass

Task: `quality-pass-compiled-scaleout`
Date: 2026-07-08

## Scope

This was a graph-quality pass over WG task metadata only. No source code was
inspected or edited.

## Final Intended Ladder

1. `run-compiled-helper` owns the 8n and 64n debug smoke ladder only.
   It must not submit 128n/256n rungs, production jobs, production latest/last
   mutations, shared production chain pointer mutations, or live Lustre update
   collections.
2. `run-compiled-helper-128n` is the 128n debug bridge. It is gated on this
   quality pass and a clean `run-compiled-helper` result with 8n and 64n
   evidence.
3. `evaluate-compiled-helper` is the 256n debug gate. It is gated on
   `run-compiled-helper` and `run-compiled-helper-128n`; it may submit only a
   bounded 10-20 minute 256n debug smoke after clean prior rungs.
4. `synthesize-compiled-helper-scaleout` is a report-only synthesis task after
   the 256n debug gate resolves. It must not submit Slurm jobs or mutate
   production latest/last state.

## Metadata Edits Applied

- Tightened `run-compiled-helper` to state explicit 8n/64n ownership and no
  128n/256n submission.
- Tightened `run-compiled-helper-128n` to forbid production QOS/walltime and
  require the exact metric set.
- Tightened `evaluate-compiled-helper` to allow only bounded 256n debug and to
  require human approval for 1h+ or production-scale runs.
- Tightened `synthesize-compiled-helper-scaleout` to require exact per-rung job,
  QOS, walltime, node-hour, artifact, latest/checkpoint, and helper metrics, and
  to remain report-only.

## Required Metrics Now Explicit

The relevant task descriptions now require concrete reporting of job id,
command/export vars, partition/QOS, requested walltime, estimated and actual
node-hours, seed path, output root, artifact paths, ranks started,
accepted/quorum counts, stale/failed/timed-out counts, reduce latency, aggregate
bytes, loss window, latest/checkpoint behavior, production/latest guard outcome,
and terminal state.

## Validation Result

Pass. The downstream graph is precise enough to run after this quality gate:
production latest mutation is prohibited during debug rungs, 8n/64n remain owned
by `run-compiled-helper`, 128n is a bridge after clean 64n evidence, 256n remains
bounded debug only unless a human explicitly approves longer, and synthesis is
report-only.
