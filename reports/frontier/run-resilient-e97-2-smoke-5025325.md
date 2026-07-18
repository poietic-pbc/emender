# Resilient E97 bounded-streaming startup smoke — job 5025325

Real changed-payload submission (not `--test-only`) at
`2026-07-17T22:25:34-04:00`, from fetched authoritative commit
`41ce274340e0edb7143e8f60138233e27bd20a1e` with clean tracked checkout
`HEAD == origin/main`.

- Run: `run-resilient-e97-2-smoke-20260718T022418Z-41ce274`.
- Payload: `41ce274-20260718T022418Z-startup-smoke-streaming-delta`.
- Immediate state: `PENDING (Priority)`; queue runtime `00:00:00`.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:50:00`; no injection.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- One finalized generation is mandatory before any full resilience gate.
- Exact executable command: `exact-command.sh` in the immutable run directory.

This payload follows terminal smoke 5024996 and is not an unchanged retry. It
removes the redundant full base clone and post-training `after` state, streams
model-precision delta shards into the bounded node-local spool without a
whole-model flatten/cat allocation, promotes each shard to float64 only in the
model-free manager accumulator, and emits a `streaming_delta` progress stage.

Pre-submit validation was run sequentially to avoid fixed-port interference:
7 runtime/checkpoint, 22 launcher/topology, 4 split-role, and 10 transport tests
passed (43 total). Production parity returned `ok=true`,
`forbidden_diff=[]`, and `missing_required=[]`; compileall, shell syntax, and
diff checks passed. The seed and tokenizer cache SHA256 values were independently
verified immediately before submission.

Conformance: *Resilient DiLoCo Compute Pool* version 1, applicable R03, R05,
R06, R08, R09, R10, R14, and R16. This smoke must prove both model-free
managers, all sixteen real trainers, node-local/network-only live transport,
bounded progress, and one immutable finalized generation. Queue time will be
recorded separately from runtime; pending state alone is not a retry trigger.

## Live checkpoint — 2026-07-17T22:34:38-04:00

The allocation started at `2026-07-17T22:26:05-04:00` on
`frontier[09130,09180]`, after 31 seconds of queue time. At runtime `00:08:33`,
Slurm still reported the job `RUNNING`. Supervisor evidence records two
model-free managers and all sixteen trainers starting between
`22:26:12-04:00` and `22:26:13-04:00`. The representative trainer log proves
the real HIP/Triton E97 path entered training with the pinned autotune registry;
no synthetic/model-free trainer mode was used.

No immutable generation had finalized at this checkpoint. That is not a pass
or a retry trigger: the configured generation deadline is 900 seconds. The
next inspection must check finalized generation/checkpoint artifacts and the
manager/network events, not process presence alone. Job 5025325 must not be
cancelled or duplicated merely because it is still within that deadline.
