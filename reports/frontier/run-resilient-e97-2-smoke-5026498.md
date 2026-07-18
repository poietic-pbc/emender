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

## Live checkpoint at 45 minutes

Slurm started the allocation at `2026-07-18T05:36:28Z`, after 33 seconds in
the queue. At elapsed `00:45:40`, job `5026498` and its two-node role step were
still `RUNNING`; no cancellation or retry was requested. The supervision log
accounts for exactly two managers and sixteen trainers. Both managers formed
the coordinator network, and every trainer entered the real HIP E97 loop using
the `e88-sequential-split-edit-triton` path. The role step reported
`MaxRSS=88,904,292 KiB`, `MaxDiskRead=83,631,620,024`, and
`MaxDiskWrite=37,281,603,171` bytes.

There was still no optimizer-step heartbeat, trainer contribution, manager
quorum freeze, or finalized-generation/checkpoint marker. Trainer logs stopped
advancing between `05:41:30Z` and `05:41:42Z`, after entering the first real
training step. The manager generation deadline begins after initialization;
the first manager reached its local-quorum wait at approximately `05:40:39Z`,
putting the configured 2,700-second fail-fast near `06:25:39Z`. The job is being
left to enforce that bounded deadline. This checkpoint is live evidence only,
not a successful smoke or R16 claim.

## Terminal result and root-cause correction

Slurm records job `5026498` as `CANCELLED by 19032` after `00:50:54` of
runtime (`2026-07-18T01:36:28` through `02:27:22` in Slurm accounting). The
role step was cancelled after both managers reached `progress_deadline`, local
trainer quorum disappeared, and zero updates or finalized generations had been
published. This is a fundamental pre-generation failure, not acceptable
cold-start behavior. The 2,700-second experimental deadline is rejected; the
checked-in launcher default remains 900 seconds and no retry is permitted until
a changed payload restores minutes-scale progress.

An exact comparison with the retained working production command in
`logs/frontier/trainpy_async_quorum/async-b4k40-ladder-256n-4979526.out`
found two unintended training-flag differences in the resilient flat config:
the failed payload used `batch_size=1` and `gradient_checkpointing=true`, while
the working pinned E97 path used `--batch-size 4` and did not enable gradient
checkpointing. The corrected config restores batch size 4 and disables
gradient checkpointing. Model dimensions, ScheduleFree parameters, pinned seed,
CommaPile data, bf16, Triton split-edit path, chunk size 2048, and K=40 remain
identical.

The changed payload also records node-local per-trainer JSONL timestamps around
data load, forward, backward, scalar loss/item synchronization, optimizer
update, optimizer-step completion, and delta streaming. Each phase refreshes a
stage-specific heartbeat without treating liveness as completed-generation
progress. This makes a subsequent startup smoke diagnose the exact stalled
phase while the unchanged 900-second generation/progress fail-fast remains
bounded. Applicable authority requirements are R03, R06, R09, R10, R14, and
R16; no R16 success is claimed.
