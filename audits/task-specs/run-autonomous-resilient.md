## Objective
Autonomously pass sequential live Frontier resilient-E97 gates at 4, 8, 16, 32, 64, 128, and 256 nodes after the 2-node foundation gate. Each successful debug rung creates useful durable state and the immutable seed for the next; reports, local tests, queued jobs, or partial rungs cannot complete this task.

## Fixed semantics and bounds
- Use the exact passing resilient E97 foundation: model/data/optimizer, `local_steps=40`, outer update, checkpoint cadence, independent per-node supervision, bounded deadlines, quorum/membership, fencing, exact weighted aggregate redistribution/apply, durable bounded replay/catch-up, TERM@300, fail-fast policy, and eight ranks per node.
- Every attempt uses debug QoS and exactly a 2-hour walltime. Never submit, modify, or cancel a normal-QoS or production job.
- Between adjacent rungs, only node/derived-rank count, mechanically unique job/evidence/run identity, and the predecessor's immutable verified checkpoint may differ. Render a normalized parity diff before submission.
- Never use `/tmp` for cross-node or durable state. Enforce and record bounds for manager/trainer memory, scratch, spool, replay, and checkpoint growth.

## Sequential per-rung hard gate
For each node count in the strict order 4, 8, 16, 32, 64, 128, 256:
- Verify the immediately preceding rung's immutable checkpoint for completeness, size, SHA256, step/generation, source job/code/payload identity, and loader compatibility. Render an immutable manifest before submission.
- Submit one unique debug-QoS, 2-hour Frontier job per attempt and monitor startup through scheduler exit. Account for every expected node, manager, trainer, and rank.
- Finalize at least two baseline generations before controlled failure injection. Then inject a documented non-coordinator trainer/rank loss and manager/node-step loss appropriate to that rung.
- Prove bounded detection, membership eviction/quorum update, generation/coordinator fencing, aggregate redistribution/apply, checkpoint integrity, and at least three finalized post-injection generations with advancing step/loss and healthy-participant accounting. Continue to TERM@300 and publish a verified immutable handoff unless fail-fast criteria fire.
- Verify checkpoint reload and dry-render the next rung from that handoff. At 256 nodes, also demonstrate durable whole-allocation restart: after a verified checkpoint and controlled allocation termination, start a fresh 256-node debug-QoS 2-hour allocation from that handoff and finalize at least two new generations. Do not claim that Slurm whole-allocation termination is survivable in-place.
- Treat a failed rung as a hard block on every larger rung. Cancel promptly on missed deadlines, corruption, unexpected quorum loss, unbounded growth, or absent finalized-generation progress; allocation RUNNING state and high CPU are not progress.
- Before retry, preserve evidence, diagnose root cause, change code/config, run focused regressions, commit and push the fix, and verify a changed payload identity. Never resubmit unchanged.
- If shared resilient code changes, rerun the 2-node injected-failure regression and the most recent passed rung before advancing.

## Validation
- [ ] Passing live debug-QoS, 2-hour Frontier evidence exists sequentially for 4, 8, 16, 32, 64, 128, and 256 nodes, downstream of the passed 2-node gate.
- [ ] Every rung begins from the immediately preceding immutable verified checkpoint and records at least two pre-injection plus three post-injection finalized generations.
- [ ] Every injected failure records injection time/command, bounded detection, eviction/quorum, fencing, continued progress, aggregate apply, and checkpoint integrity.
- [ ] The 256-node rung includes a separate whole-allocation termination followed by durable restart in a fresh debug allocation and at least two new finalized generations.
- [ ] Normalized diffs contain only allowlisted scale, predecessor checkpoint, and unique identity differences.
- [ ] Memory, scratch, spool, replay, and checkpoint growth are bounded at every rung; fail-fast monitoring prevents silent allocation consumption.
- [ ] A failed rung blocks all larger rungs; every repair is regression-tested, committed, pushed, and payload-changed before retry, with no unchanged resubmission.
- [ ] Exact job IDs, commands, manifests, hashes, logs, checkpoints/checksums, Slurm accounting, failure evidence, and immutable handoffs are registered as WG artifacts.
- [ ] Final 256-node handoff is complete, checksum-valid, reloadable, and tied to clean pushed HEAD.
- [ ] No normal-QoS or production job was submitted, modified, or cancelled.
