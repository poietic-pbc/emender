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

## Live runtime checkpoint

At `2026-07-17T21:50:09-04:00` the job remained `RUNNING` on exactly
`frontier04928` and `frontier04929`.  Queue time was `00:07:16`; runtime was
`00:27:17`.  A read-only overlapping diagnostic step inspected the node-local
bulk roots on both nodes.  All sixteen trainer progress records reported real
optimizer step `1525040` for generation 0, with finite losses, and all sixteen
trainer liveness records plus both model-free manager liveness records were
fresh.  Both managers remained in `collecting`; no update file, finalized
generation, or checkpoint existed yet.  This proves 40 real local steps and
healthy roles, but it does **not** satisfy the smoke gate until the 1.3B-parameter
updates finish materializing and one generation is finalized.  The allocation
was left running; no cancellation or duplicate submission was made.

The active files observed by the diagnostic were exclusively below
`/tmp/resilient-e97/<run-id>/node-{0,1}/supervision`.  The retained Lustre run
directory contained only logs, the supervisor event record, and coordinator
discovery metadata at this checkpoint; it contained no bulk update payload.

## Terminal result and focused diagnosis

At `2026-07-17T22:07:39-04:00` Slurm ended job 5024996 as `FAILED`
(`ExitCode=0:15`) after `00:44:47` runtime.  The separately recorded queue time
remains `00:07:16`.  No update, aggregate, finalized generation, immutable
checkpoint, or loader result was produced, so this smoke **failed** and no full
`02:00:00` resilience allocation is authorized from this payload.  There was no
manual cancellation and no retry was submitted.

The retained supervisor record accounts for all two managers and sixteen real
trainers.  On allocation handoff both managers exited on SIGTERM and the two
trainer leaders required SIGKILL; Slurm accounting reports a peak RSS of
`294350988K` for the combined node-local Python step, about 287.5 GiB.  The
trainers had completed optimizer step 1525040 by approximately 5m16s runtime,
but remained in post-training update construction until TERM.  Node-local
inventory immediately before termination contained only supervision/liveness
records and no mailbox update.

The focused code-path diagnosis is that the current trainer creates several
full 1.3B-parameter CPU representations after the optimizer steps:

1. `_floating_delta_from_model()` materializes a dense CPU delta;
2. `trainer()` materializes `after = before + delta`; and
3. `flatten_delta()` converts both full states to float64, subtracts them again,
   concatenates the entire update, and then clones it into chunks.

This is consistent with the high-water RSS, the absence of a first mailbox
file, and the long post-step interval.  It is not evidence of a network or
manager-quorum failure because no update reached either transport.  A next
payload must remove the redundant `after` state and whole-update float64
concatenation in favor of bounded streaming delta publication, add stage-level
progress evidence around delta construction/publication, pass focused runtime
and transport regressions, and then pass a new unique 2-node startup smoke
before another full gate.  The failed payload must not be resubmitted unchanged.
