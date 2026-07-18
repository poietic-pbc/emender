# Resilient E97 per-rank kernel-cache startup smoke — job 5026498

## Submission

Real `sbatch` submission (not `--test-only`) at
`2026-07-18T05:35:55Z`:

- Job: `5026498`; initial state `PD (Priority)`.
- Exactly two nodes, debug QoS, `02:00:00`, with the script's `TERM@300`.
- Run: `run-resilient-e97-2-smoke-20260718T053555Z-572e3e5`.
- Payload: `572e3e5-20260718T053555Z-startup-smoke-rank-kernel-cache`.
- Code: fetched authoritative `HEAD == origin/main == 572e3e5` with a clean
  tracked worktree at submission.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- One finalized generation required; failure injection disabled.
- The exact shell-escaped command is retained at
  `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260718T053555Z-572e3e5/exact-submit-command.txt`.

This is a changed payload after job 5026188's bounded pre-generation failure.
It preserves the approved flat E97 model, ScheduleFree optimizer, CommaPile
data, 40 local steps, two model-free managers, sixteen real HIP trainers,
dynamic 6/8 local quorum, bounded 2,700-second generation/progress deadlines,
node-local/network bulk transport, and node-local launch mode. The only runtime
change isolates Triton and Inductor caches by local trainer under `/tmp`,
preventing shared home-cache lock contention and keeping kernel-build traffic
off Lustre.

The rendered debug/production parity check is `ok=true`, with no forbidden or
missing fields. The allowlisted differences remain failure injection, nodes,
QoS, and walltime; production injection is disabled. This smoke is not the full
failure/restart gate, and no pass is claimed until an immutable finalized
generation exists.

## Validation and design conformance

Pre-submit validation: launcher/topology `29 passed`; transport `10 passed`;
parity `ok=true`; compileall, `bash -n`, and `git diff --check` passed. The
monolithic runtime test process exceeded login-node memory after its initial
tests and is not represented as a pass.

This runner conforms to version 1 of
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`; applicable requirement IDs are R03,
R05, R06, R08, R09, R10, R14, and R16. Success requires committed-generation
evidence, not process presence. R07/R11/R12 and complete R16 acceptance remain
unclaimed until failure/rejoin, immutable checkpoint, and fresh-allocation
continuation are exercised live.
