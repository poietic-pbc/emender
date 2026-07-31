# Resilient E97 compiler-cache startup smoke — job 5026188

## Exact allocation identity and result

- Submitted `2026-07-18T00:40:57-04:00`; started `00:41:24`; ended
  `01:27:11`. Queue time was 27 seconds and allocation runtime was 45:47.
- Slurm job `5026188`, debug QoS, exactly two nodes, `02:00:00`, with
  `TERM@300`; terminal state `FAILED`, exit `1:0`.
- Run: `run-resilient-e97-2-smoke-20260718T0040Z-0984def`.
- Payload: `0984def-20260718T0040Z-startup-smoke-full-bound`; code `0984def`.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- The immutable exact submission command is retained by `scontrol show job -dd
  5026188` and binds the approved flat E97 configuration, CommaPile input,
  verified p50k cache, 40 local steps, two managers, sixteen trainers, dynamic
  6/8 local quorum, node-local bulk root, and 2,700-second bounded generation
  and progress deadlines. Injection was disabled.

## Preserved evidence and diagnosis

The supervision stream accounts for two model-free managers and all sixteen
real HIP trainers. The role processes launched at `00:41:29`. No trainer
published a contribution, no manager froze an accepted set, and no generation,
checkpoint, or handoff was finalized. Both managers were evicted at the
2,700-second local-quorum deadline and exhausted their two bounded restarts;
the batch failed closed. This is a pre-generation failure, not a restart point.

Every trainer output contains exactly eleven E97 layer-runtime entries and no
optimizer-step heartbeat. The two-node process step peaked at `92,939,088 KiB`
RSS. The evidence is consistent across all sixteen trainers: concurrent cold
Triton/Inductor compilation stalled inside the first optimizer step. The live
launcher did not isolate compiler caches, so eight trainers per node used the
default shared home-directory caches and contended on compiler cache locks.
Established Frontier production launchers use per-rank Triton caches for this
path.

The next changed payload assigns each trainer a payload-specific, node-local
`TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR`. It does not alter model, data,
optimizer, local-step count, role topology, quorum, tensor transport, failure
policy, or launcher mode. Job 5026188 will not be retried unchanged, and no full
failure-injection allocation is authorized until the changed startup smoke
finalizes an immutable generation.

## Validation and design conformance

This runner conforms to version 1 of
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`; applicable matrix requirements are
R03, R05, R06, R08, R09, R10, R14, and R16. READY-subset progress remains
represented by dynamic 6/8 local quorum rather than launched-rank collectives;
generation waits are finite; identities and transport remain fenced and
bounded; bulk traffic is node-local/network-only; managers remain model-free.
The smoke did not satisfy R16 because it produced no committed generation.
Failure/rejoin (R11), fenced checkpoint publication (R07/R12), and fresh
allocation continuation remain unclaimed until their live gates pass.
