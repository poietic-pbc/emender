# Resilient E97 live gate job 5017344 startup failure

## Allocation identity

- Payload/code commit: `ae89c9900f5b3d29a0c10e4db3470e0032c3fa37`.
- Submission was a real `sbatch`, not `--test-only`.
- Job: `5017344`; debug QoS, batch partition, exactly two nodes, exactly
  `02:00:00`; run ID
  `run-resilient-e97-2-live-20260717T101540Z-ae89c99`.
- Immediate state was `PENDING (Priority)`. The allocation later ran on
  `frontier02381` and `frontier02510` from 2026-07-17 10:00:50 EDT.
- Rendered parity was `ok=true`, with only `failure_injection`, `nodes`, `qos`,
  and `walltime` different from the production rendering.

## Concrete failure and disposition

The batch step launched two allocation-level `node-supervisor` sruns. Each
occupied its requested node, then attempted to create nested manager/trainer
sruns. Frontier repeatedly rejected both child steps with `Requested nodes are
busy`. Only the batch and extern steps existed. The retained supervisor event
stream contains exactly two node-supervisor starts; no manager or trainer
heartbeat, finalized generation, or checkpoint was produced.

At 2026-07-17 10:42 EDT, after more than 41 minutes and therefore well beyond
the configured 900-second progress deadline, the unusable allocation was
cancelled as a documented startup fail-fast. Final accounting was
`CANCELLED by 19032`, elapsed `00:41:44`, nodes
`frontier[02381,02510]`. This job does **not** satisfy any live-generation
acceptance criterion and is not called a passed gate.

## Changed-payload repair

The supervisor now defaults to independent allocation-level manager/trainer
steps, avoiding nested `srun` on Frontier while preserving the same 2-manager,
16-trainer identities and resource assignments. A bounded startup deadline now
evicts a child that never publishes its first heartbeat, closing the prior hole
where absence of a state file disabled all supervision deadlines. Focused tests
cover both the independent-step default and the first-heartbeat deadline.

No unchanged retry was submitted. A retry is permitted only after this repair
passes the focused regression matrix and the changed commit is pushed, fetched,
and verified as authoritative `origin/main`.
