# Resilient E97 direct-model-stream startup smoke — job 5025627

Real changed-payload submission (not `--test-only`) at
`2026-07-17T23:22:16-04:00`, from fetched authoritative commit
`10d7ad9d5ef87090afbcd9abaf22f0bb4a91848b` with clean tracked checkout
`HEAD == origin/main`.

- Run: `run-resilient-e97-2-smoke-20260718T032139Z-10d7ad9`.
- Payload: `10d7ad9-20260718T032139Z-startup-smoke-direct-model-stream`.
- Job: `5025627`; initial inspection was `RUNNING` on exactly two nodes.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:50:00`; no injection.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- Exact executable command: `exact-command.sh` in the immutable run directory.

This is not an unchanged retry after terminal smoke 5025325.  The real worker
now consumes the trained model directly through a bounded delta callback; the
role copies and subtracts one configured parameter shard at a time into the
node-local spool.  The same monotonic deadline covers local training,
publication, aggregation, and apply.  The model, CommaPile data, ScheduleFree
optimizer, pinned seed, 40 local steps, and 2-manager/16-trainer topology are
unchanged.

Pre-submit validation: the complete launcher/topology suite passed 24 tests.
The combined heavy runtime test invocation was terminated by login-host memory
pressure after its first test without a failure traceback; compile and diff
checks passed.  Production parity remains restricted to the allowlisted Slurm
scale/QoS/walltime and injection differences.

Conformance: *Resilient DiLoCo Compute Pool* version 1, applicable R03, R05,
R06, R08, R09, R10, R14, and R16.  This smoke must finalize one immutable
generation before any full resilience gate is authorized.
