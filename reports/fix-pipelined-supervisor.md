# Pipelined supervisor deadline recovery fix

Date: 2026-07-21

## Failure and fix

The allocation supervisor previously derived progress exclusively from each
role's coarse `progress_time`. During pipelined native redistribution, the
manager can remain in one stage while trainers independently apply the result
and publish durable `native-applied-<generation>-<rank>.json` receipts. The
supervisor therefore evicted a manager that was making observable forward
progress. It now treats validated, same-run and same-generation apply-receipt
publication times as manager/node-supervisor progress. Heartbeat expiry remains
independent and fail closed, and a stalled apply phase still expires after the
configured stage budget.

After a manager restart, the transient role state can temporarily report the
launcher's initial generation. The first-atomic-generation deadline previously
interpreted this as a run with no commit even when `handoff/latest.json` named a
finalized generation-2 checkpoint. The supervisor now restores this accounting
from the durable handoff only after verifying the manifest checksum, finalized
flag, generation, and logical run identity. Missing, malformed, mismatched, or
corrupt handoff data does not suppress the deadline.

## Architecture conformance

Checked against the conformance checklist in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md` and the normative Resilient DiLoCo
Compute Pool design:

- **R12:** restart accounting uses only the finalized, checksum-verified global
  checkpoint handoff; unfinished local work remains irrelevant.
- **R14:** bounded stage deadlines remain enforced, while immutable per-trainer
  apply receipts count as genuine exchange/apply progress. Corrupt or stale
  evidence remains fail closed.
- **R16:** this local correction does not advance any scale rung or authorize a
  live allocation. No Slurm job was submitted.
- **NDP16:** supervisor accounting consumes the native per-trainer apply
  telemetry already bound to run and generation identity; heartbeat and
  deadline telemetry contracts are unchanged.
- **NDP17:** the ordered two-node native gate remains the next downstream live
  validation step; this task makes no G3 or larger-scale claim.

## Validation

- Regression tests cover active generation-2 trainer apply redistribution and
  restart after a finalized generation-2 checkpoint, including corrupt durable
  pointer rejection.
- Canonical runtime/launcher suites and native CTest results are recorded in
  the WG task log.
- No Slurm job was submitted.
