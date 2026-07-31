# E97 GPU-Island No-DDP Quality Pass

Task: `quality-pass-e97-3`
Date: 2026-06-25

## Verdict

Pass. The downstream track is sufficiently specified to test the intended
E97-MLP regime where each GPU is a singleton DiLoCo island and no per-step
within-island DDP gradient averaging should occur. The task chain keeps the
first training probe bounded to 4 nodes, makes the 8-node probe conditional on
the 4-node evidence, and does not authorize any 32-node or 64-node GPU-island
job.

Calibrated grade: 0.94
Confidence: 0.86
Rubric underspecified: no

## Checklist

- Downstream tasks have concrete validation criteria: yes.
  `audit-e97-gpu`, `run-e97-gpu`, `run-e97-gpu-2`, and
  `synthesize-e97-gpu` each include a `## Validation` checklist with concrete
  required outputs or gating decisions.
- Audit verifies `DILOCO_ISLAND_SIZE=1` semantics: yes.
  `audit-e97-gpu` explicitly requires stating whether `DILOCO_ISLAND_SIZE=1`
  truly avoids DDP gradient all-reduce, and forbids Slurm submission.
- First training probe is bounded to <=4 nodes and records fixed eval: yes.
  `run-e97-gpu` allows at most one bounded <=4-node E97-MLP GPU-island job and
  requires a fixed source-vs-candidate eval with CE/BPB deltas.
- 8-node probe is conditional on 4-node evidence: yes.
  `run-e97-gpu-2` depends on `run-e97-gpu`, requires reading the 4-node result,
  and requires submitting no job if the 4-node result is not clean/promising.
- No 32/64-node no-DDP job is authorized: yes.
  The 4-node task explicitly forbids 8/16/32/64-node submissions from that
  task; the 8-node task forbids 16/32/64-node submissions from that task; the
  synthesis may only recommend a later 16/32-node task.
- `run-64-node-e97` remains paused: yes.
  `wg show run-64-node-e97` reports status `open (PAUSED)`.
- E97-MLP remains primary and GDN2/CMAES are out of scope: yes.
  The run tasks forbid GDN2/CMAES submissions, and the synthesis requires E97-MLP
  scope preservation.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Validation specificity | 0.95 | All downstream tasks have concrete checklists with job bounds, required recorded fields, eval outputs, and no-job blocker paths. |
| No-DDP semantic guard | 0.96 | The audit is explicitly responsible for confirming whether singleton islands avoid DDP gradient all-reduce before any run proceeds. |
| Operational bounding | 0.95 | The first executable training probe is capped at <=4 nodes and one submission; the follow-up is capped at <=8 nodes and one submission. |
| Conditional sequencing | 0.94 | Dependencies enforce audit -> 4-node -> 8-node -> synthesis, and task text requires evidence-based gating. |
| Scope control | 0.93 | Large GPU-island jobs are not authorized, `run-64-node-e97` is paused, and GDN2/CMAES are excluded. Minor naming differences from the intended task names are harmless but worth noting. |
| Evaluability | 0.92 | The artifacts expected from each task are clear enough for later FLIP/evaluator review. The exact fixed eval tensor is intentionally delegated to the audit, which is acceptable. |

## Notes

The actual task identifiers differ from the names in the quality-pass prompt:
`audit-e97-gpu`, `run-e97-gpu`, `run-e97-gpu-2`, and `synthesize-e97-gpu` are
the implemented downstream tasks. This does not weaken the pass because their
descriptions match the intended track and preserve the same dependency order.

No extra decomposition was needed for this quality pass. The task was a bounded
review of existing downstream WG task specifications, not an implementation or
multi-file code change.
