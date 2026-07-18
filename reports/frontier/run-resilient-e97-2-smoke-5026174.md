# Resilient E97 full-bound startup smoke — job 5026174

Real changed-payload submission (not `--test-only`) at
`2026-07-18T00:37:36-04:00`, from fetched authoritative commit
`d00fe600afc97b16d69b2477735ed408706e8934`.  The tracked checkout was clean
and `HEAD == origin/main` immediately before submission.

- Run: `run-resilient-e97-2-smoke-20260718T-current-d00fe60`.
- Payload: `d00fe60-20260718-startup-smoke-full-bound`.
- Job: `5026174`; immediate state `RUNNING` on `frontier[00016,00018]`.
- Submit: `2026-07-18T00:37:36-04:00`; start:
  `2026-07-18T00:37:50-04:00`; queue time: 14 seconds.
- Exactly 2 nodes and 16 GPUs, debug QoS, `02:00:00`; no injection variables.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- Whole-generation and progress deadlines: 2,700 seconds; Slurm `TERM@300`.

This is not a full failure-injection gate.  It is the required changed-payload
startup smoke after terminal pre-generation job 5025702.  That job proved a
50-minute request supplied only 44:56 runtime and terminated before the first
optimizer-step callback.  The two-hour debug ceiling is used here to contain
the measured cold-compilation bound, the complete 40-local-step generation,
and the five-minute TERM margin.  Exactly one immutable finalized generation
is required before a full resilience allocation is authorized.

The focused pinned ROCm runtime, split-role, transport, quorum, launcher,
topology, Frontier-plumbing, checkpoint, and walltime matrix passed immediately
before submission: `79 passed in 97.72s`.  The rendered production parity check
reported `ok=true`, no forbidden or missing fields, and only the allowlisted
failure-injection, node-count, QoS, and walltime differences.

The command retained by Slurm's exact `SubmitLine` binds the original seed,
CommaPile data, p50k tokenizer digest, flat E97 model configuration,
ScheduleFree optimizer, `local_steps=40`, two model-free managers, sixteen real
HIP trainers, dynamic 6/8 local quorum, bounded deadlines, node-local bulk root,
network manager exchange, and the new-harness checkpoint contract.  There is no
synthetic/control model or data, MPI collective launcher, normal QoS, 4+ node
request, failure injection, or production mutation.

Conformance check against *Resilient DiLoCo Compute Pool*, version 1: applicable
requirements are R03, R05, R06, R08, R09, R10, R14, and R16.  Success requires
committed-generation evidence, not process presence.  Failure/rejoin (R11),
complete fenced checkpoint publication (R07/R12), and the full R16 lifecycle
remain unclaimed until their later gates actually pass.

## Terminal outcome

The batch payload failed closed at `2026-07-18T00:37:56-04:00`, six seconds
after allocation start and before any role launch.  Slurm accounting records
`FAILED`, `ExitCode=64:0`; queue time was 14 seconds and allocation runtime was
six seconds.  The exact diagnostic was `startup smoke requires exactly
00:50:00`: the launch guard still hard-coded the earlier smoke duration and
rejected the empirically required two-hour deadline-plus-TERM window.

No trainer or manager heartbeat, update spool, generation, or checkpoint was
created, so this is a pre-generation payload failure and not a restart point.
The changed payload adds a focused launcher regression for a two-hour,
one-generation, no-injection smoke while retaining the 50-minute option for
shorter cold starts; all other two-node, debug-only, role-count, local-step,
transport, and generation guards remain fail closed.
