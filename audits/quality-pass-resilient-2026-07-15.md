# Resilient E97 autonomous ladder quality pass

Task: `quality-pass-resilient`
Date: 2026-07-15

## Graph reviewed

The executable workstream is the existing sequential chain:

1. `complete-resilient-e97` — foundation implementation plus the 2-node live gate.
2. `run-autonomous-resilient` — sequential 4, 8, 16, 32, 64, 128, and 256-node live gates.
3. `package-resilient-e97` — immutable authorization package only, with no submission.

The task IDs differ from the prose aliases in the quality-pass objective, but the titles and objectives are the intended work. Dependencies are strictly foundation → ladder → package. All are full-execution tasks with graph/full context, remain paused until the quality pass is accepted, and have bounded worker timeouts of 72 hours, 14 days, and 24 hours respectively. Every Slurm scale attempt is independently capped at debug QoS and two hours.

## Hardening applied

- Required independent per-node managers and local trainer supervision, with no all-rank blocking MPI, RCCL, TCPStore, or equivalent collective in the failure-sensitive coordination boundary.
- Required bounded detection/progress deadlines, dynamic quorum membership, generation/coordinator fencing, exact weighted aggregate redistribution/apply, durable bounded replay/catch-up, integrity rejection, and explicit atomic checkpoint/restart semantics.
- Separated survivable in-allocation rank/node-step failures (healthy quorum must continue) from whole-allocation Slurm termination (durable restart from the last verified immutable handoff must progress in a new debug allocation).
- Required each 2, 4, 8, 16, 32, 64, 128, and 256-node gate to use debug QoS for two hours, begin from the preceding verified immutable checkpoint, establish at least two baseline generations, inject controlled failure only afterward, and finalize at least three post-failure generations.
- Made every rung a hard gate. A failure requires evidence preservation, root-cause diagnosis, a code/config change, focused regression tests, commit and push, and a changed payload identity before retry. Unchanged resubmission and advancement after failure are prohibited.
- Required fail-fast monitoring based on finalized generation/checkpoint progress rather than allocation state or CPU use, plus bounded memory, scratch, spool, replay, and checkpoint growth.
- Prohibited every normal-QoS or production submission or modification. The final task may only render and audit a candidate command; execution requires a new task created from separate explicit user authorization.

## Validation

- [x] All three downstream tasks exist in strict dependency order.
- [x] Foundation and ladder require live Frontier execution and cannot pass on local tests, dry runs, or evaluator reports alone.
- [x] The 2, 4, 8, 16, 32, 64, 128, and 256-node gates and controlled failure criteria are explicit.
- [x] Immutable checkpoint chaining, resource bounds, fail-fast monitoring, and commit/push/clean-worktree hygiene are explicit.
- [x] In-allocation continued progress and whole-allocation durable restart are separately proven.
- [x] Normal-QoS and production submission are prohibited pending a separate user-authorized task.
- [x] Every downstream description retains a literal `## Validation` checklist with concrete Frontier evidence requirements.
