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
