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

## Terminal result — 2026-07-17T23:40:16-04:00

The allocation was deliberately failed fast after crossing its configured
900-second whole-generation deadline with zero finalized generations.  Slurm
recorded `CANCELLED by 19032`, start `23:22:27-04:00`, end
`23:40:16-04:00`, and runtime `00:17:49`; queue time was 11 seconds.  This was
not a pending-job cancellation and no duplicate or unchanged retry was
submitted.

At the deadline, the retained supervisor stream accounted for exactly two
model-free managers and sixteen real GPU trainers on `frontier09044` and
`frontier09064`.  All eighteen node-local liveness files were fresh.  Every
trainer log proved the real HIP/Triton E97 path and showed the trainer inside
the local-training call, but no trainer had emitted a first optimizer-step
progress record, streamed an update, or submitted to its manager.  The Lustre
run directory contained only retained logs, supervision events, and the small
coordinator discovery record; no live bulk payload traversed Lustre.  No
generation, aggregate, checkpoint, handoff, or loader result exists, so this
smoke failed and cannot authorize the full resilience gate.

The 900-second setting is bounded but is below observed cold-node startup and
compilation variance: earlier job 5024996 completed all 40 real optimizer
steps in about five minutes after initialization, whereas this allocation had
not completed its first callback by the same generation bound.  The next
changed payload will retain a finite whole-generation deadline but restore the
previously exercised 2700-second bound, which leaves five minutes for Slurm's
`TERM@300` handoff in the 50-minute startup allocation.  This changes the
explicit deadline and unique payload identity; it does not change the model,
data, optimizer, seed, local-step count, role topology, transport, or launcher.

Conformance result: R03/R09 role topology and R10 node-local live-path
ownership were observed; R05/R08 aggregation was not reached; R06/R14 failed
closed at the declared bound; R16 remains closed pending a finalized immutable
generation.
