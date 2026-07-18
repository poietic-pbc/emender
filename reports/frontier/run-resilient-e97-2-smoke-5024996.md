# Resilient E97 step-progress startup smoke — job 5024996

Real submission from fetched authoritative commit
`4ca7e9664a0f1e8409a503cfcefb6ce2157a8c86`; not `--test-only`.

- Submitted: `2026-07-17T21:15:36-04:00`; immediate state `PENDING (Priority)`.
- Run: `run-resilient-e97-2-smoke-20260718T011356Z-4ca7e96`.
- Payload: `4ca7e96-20260718T011356Z-startup-smoke-step-progress`.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:50:00`, no injection.
- One finalized generation is mandatory; progress/generation deadlines are
  2700 seconds and Slurm sends TERM@300.
- Seed SHA256 was independently verified as
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.

This is a changed payload after 5024821 reached TERM at 34m53s before its
first 40-step generation. It emits a durable node-local heartbeat after every
real optimizer step, allowing runtime progress and the exact generation
duration to be distinguished from process presence. The 50-minute smoke gives
45 minutes before TERM and remains shorter than the required full
`02:00:00` gate. Focused validation passed 24 tests after one isolated
fixed-port collision was rerun successfully. Rendered production parity is
`ok=true`, with no forbidden or missing fields; allowlisted differences are
failure injection, nodes, QoS, and walltime (the rendered partition values are
identical).

The exact scheduler command is retained by Slurm in `SubmitLine` and will be
written into the immutable run directory by the launcher. It binds the run,
payload, code, seed, approved flat E97 arguments, CommaPile data, verified
node-local tokenizer staging, one generation, node-local launch, bounded
deadlines, node-local bulk root, and two role restarts.

Conformance: *Resilient DiLoCo Compute Pool* version 1; applicable R02, R03,
R04, R06, R08, R09, R10, R14, R16. No full gate will be submitted unless this
job finalizes one immutable generation.
