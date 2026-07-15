## Objective
Implement the production-shaped resilient E97 foundation, diagnose strict-MPI job 5000436's generation-0 stall, and pass a live 2-node/16-rank Frontier gate. Local tests, dry runs, designs, or evaluator reports are necessary supporting evidence but cannot complete this task.

## Required implementation
- Audit commits 5786eb1, 6367b01, and e8bd90a; do not treat an earlier Done verdict as Frontier readiness.
- Preserve the approved E97 model/data/optimizer, `local_steps=40`, outer-update semantics, checkpoint identity, and bounded per-node memory behavior.
- Remove every all-rank blocking MPI, RCCL, TCPStore, or equivalent collective from the failure-sensitive coordination boundary. Use a real non-MPI node-manager network backend, one independently supervised manager per node, and independently supervised local trainers.
- Implement bounded connect/heartbeat/progress/aggregation/apply deadlines; dynamic quorum membership and eviction; generation, coordinator, and payload fencing; exact weighted aggregation and aggregate redistribution/apply; bounded crash-surviving replay; catch-up/rejoin rules; and stale, duplicate, corrupt, nonfinite, or incompatible-state rejection.
- Make checkpoint semantics explicit: atomically finalize each generation, preserve immutable last-valid handoffs, and record path, step/generation, byte size, SHA256, source job, code commit, and payload identity. Restart must never select a partial or unfenced generation.
- Treat failure classes separately. For a survivable in-allocation rank, trainer, manager, or node-step loss, healthy quorum members must continue advancing without an all-rank abort. For Slurm whole-allocation termination, fail promptly, preserve the last verified handoff, and demonstrate durable restart and renewed progress in a fresh debug allocation.

## Mandatory live Frontier gate
- Diagnose job 5000436 from retained scheduler, rank, heartbeat, bridge, generation, and checkpoint evidence. Log the exact last progress boundary and strongest supported root cause before changing code.
- Run focused local/control tests, commit and push fixes, then submit a unique 2-node/16-rank job using debug QoS and exactly a 2-hour walltime. Start from the newest verified immutable handoff; use approved step1525000 only if the newer candidate fails checksum and loader verification.
- Account for every node, manager, trainer, and rank. Establish at least two finalized baseline generations before any injection. Then inject a documented non-coordinator trainer/rank failure and a manager/node-step failure without corrupting the checkpoint chain.
- Prove bounded detection, eviction/quorum shrink, fencing, aggregate apply, and at least three finalized post-injection generations with advancing step/loss and healthy-participant accounting. Continue to TERM@300 handoff unless a documented fail-fast deadline fires.
- After the survivable-failure gate, cause or use a controlled whole-allocation termination only after a verified checkpoint exists; in a new 2-node debug allocation, reload that exact immutable handoff and finalize at least two new generations. This restart proof may use a separate attempt but must remain debug QoS with a 2-hour walltime.
- Monitor continuously enough to cancel on missed bounded deadlines, corruption, unbounded resource growth, or lack of finalized-generation progress. RUNNING state or CPU utilization alone is not progress.
- On failure, preserve evidence, diagnose root cause, change code/config, run focused regression tests, commit and push, and verify a new payload identity before retry. Never resubmit an unchanged payload or advance on a failed gate.
- Never submit, modify, or cancel a normal-QoS or production job.

## Validation
- [ ] Job 5000436 is diagnosed from retained live evidence with the last progress boundary recorded.
- [ ] The resilient merge/apply path contains no all-rank blocking collective in its failure-sensitive boundary.
- [ ] Focused protocol, network, supervisor, apply, deadline, fencing, replay, catch-up, stale/duplicate/corrupt/nonfinite, checkpoint, and restart tests pass.
- [ ] Live 2-node/16-rank Frontier evidence shows a debug-QoS 2-hour allocation, two pre-injection and three post-injection finalized generations, both injected failure classes, quorum change, continued progress, and bounded resources.
- [ ] A distinct whole-allocation termination is followed by reload of the last verified immutable handoff in a fresh debug allocation and at least two newly finalized generations.
- [ ] Exact job IDs, submission and injection commands/timestamps, code and payload hashes, logs, manifests, checkpoint paths/checksums, Slurm accounting, and failure evidence are logged and registered as artifacts.
- [ ] Every failed attempt is followed by a diagnosed and regression-tested change committed and pushed before retry; there is no unchanged resubmission.
- [ ] HEAD matches its remote, the worktree is clean, and all code/evidence changes are committed and pushed.
- [ ] No normal-QoS or production job was submitted, modified, or cancelled.
