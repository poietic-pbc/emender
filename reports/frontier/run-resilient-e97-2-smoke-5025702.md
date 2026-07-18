# Resilient E97 cold-compilation-bound startup smoke — job 5025702

Real changed-payload submission (not `--test-only`) at
`2026-07-17T23:44:04-04:00`, from fetched authoritative commit
`0f70099d792aefcf4a269a98395e6085b88d997c`.  The tracked checkout was clean
and `HEAD == origin/main`; retained scheduler log files are untracked evidence.

- Run: `run-resilient-e97-2-smoke-20260718T034300Z-0f70099`.
- Payload: `0f70099-20260718T034300Z-startup-smoke-cold-compile-bound`.
- Job: `5025702`; immediate state `PENDING (Priority)`.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:50:00`; no injection.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- Whole-generation and progress deadlines: 2700 seconds; Slurm `TERM@300`.
- Exact executable command: `exact-command.sh` in the immutable run directory.

This is not an unchanged retry after terminal job 5025627.  Its unique payload
restores the previously exercised finite 2700-second cold-node bound after the
900-second run failed closed before its first optimizer-step callback.  The
model, pinned checkpoint, CommaPile data, ScheduleFree optimizer, 40 local
steps, 2 model-free managers, 16 real HIP trainers, dynamic 6/8 local quorum,
bounded node-local spool, network manager exchange, and direct bounded model
delta stream are unchanged.  No synthetic/control model or data path is
enabled.

The scheduler retained the exact `SubmitLine`; it proves two nodes, debug QoS,
50-minute startup-smoke walltime, no injection variables, node-local bulk root,
unique run/payload/code identities, verified seed and tokenizer identities,
and the explicit finite deadlines.  Queue time and allocation runtime will be
recorded separately.  Pending state alone will not trigger cancellation or a
duplicate submission.

Production parity remains restricted to the allowlisted injection, node-count,
QoS, and walltime differences.  Conformance: *Resilient DiLoCo Compute Pool*
version 1, applicable R03, R05, R06, R08, R09, R10, R14, and R16.  This job
must finalize one immutable generation before any full `02:00:00` resilience
gate is authorized.

## Live runtime checkpoint

At `2026-07-18T00:18:12-04:00`, Slurm reported the allocation `RUNNING` on
`frontier[08022-08023]` with elapsed time `00:33:55`. Queue time was 13 seconds
(`Submit=2026-07-17T23:44:04`, `Start=2026-07-17T23:44:17`) and is tracked
separately from runtime. The supervisor evidence records both model-free
managers and all sixteen real GPU trainers starting. The immutable run
directory contained no generation manifest or checkpoint at this checkpoint;
process presence was therefore not counted as progress. Trainer logs had last
advanced during model startup between 23:46 and 23:50, still inside the
explicit 2,700-second whole-generation deadline. The job was not cancelled and
no duplicate or follow-on allocation was submitted.

At `2026-07-18T00:25:50-04:00` (approximately 40 minutes after the node-local
roles started), a live `srun --jobid=5025702 --overlap` inspection confirmed
that both managers and all sixteen trainer processes remained alive. Each
trainer held approximately 6.3 GiB RSS. Node-local liveness files continued
to advance on both nodes, but there was still no bulk update spool object,
finalized-generation manifest, or immutable checkpoint. The retained trainer
output had reached the real HIP E97 forward/backward path and its first loss
conversion, but had not reached the first optimizer-step callback. This is
process-liveness evidence only and is not counted as a successful smoke
generation. The allocation remains subject to its finite 2,700-second
whole-generation deadline and was neither cancelled nor duplicated.

## Terminal outcome

Slurm delivered the configured pre-walltime termination signal at
`2026-07-18T00:29:12-04:00`.  Accounting records `FAILED`, `ExitCode=0:15`,
`Submit=2026-07-17T23:44:04`, `Start=2026-07-17T23:44:17`,
`End=2026-07-18T00:29:13`, and `Elapsed=00:44:56`; queue time was 13 seconds and
allocation runtime was 44 minutes 56 seconds.  The supervisor recorded both
managers exiting on `allocation_term_handoff` and terminated trainer children
without publishing a partial generation.

The terminal immutable run directory contains no update spool object,
finalized-generation manifest, or checkpoint.  All sixteen real HIP trainers
reached the first real forward/backward computation and remained live, but none
reached the first optimizer-step progress callback before TERM.  Thus this is a
preserved pre-generation smoke failure, not a successful gate and not a restart
point.  Peak RSS for the two-node trainer step was 92,370,224 KiB; no bulk
update, aggregate, heartbeat, membership, quorum, or redistribution payload was
written through Lustre.

The finite 2,700-second generation deadline was longer than the usable runtime:
Slurm's five-minute TERM lead reduced a 50-minute request to roughly 45 minutes.
The next payload must therefore be unique, committed, pushed, and fetched, and
must remain a no-injection startup smoke.  Because 50 minutes was empirically
insufficient for the cold compiled first optimizer step, its requested runtime
must provide the full bounded generation window plus TERM margin.  No full
failure-injection allocation is authorized until that changed smoke finalizes
one immutable generation.

Conformance check against *Resilient DiLoCo Compute Pool*, version 1: this run
exercised bounded waits and shutdown (R06, R14), the model-free-manager and
trainer split (R09), node-local/network-only hot-path configuration (R08, R10),
and the mandatory two-node rung (R16).  It did not satisfy committed-generation
evidence (R07), failure/rejoin (R11), fresh-allocation outer-state restoration
(R12), or the complete R16 gate, so none of those requirements is claimed.
