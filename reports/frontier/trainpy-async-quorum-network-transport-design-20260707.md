# train.py async quorum network transport design

Task: `implement-train-py-4`

## Decision

The train.py-backed async quorum smoke/debug path now uses a rank-0 TCP
coordinator for live quorum collection. Each one-rank-per-GPU worker trains
independently, builds its local metadata/update summary in memory, and submits a
length-prefixed JSON payload to rank 0 over a socket. Rank 0 accepts updates
until the configured absolute quorum is met or the timeout expires, then writes
run-local metrics, generation manifests, latest metadata, and recovery/export
checkpoint records.

Shared storage is retired from the live quorum path. It is now used only for
final/recovery checkpoint artifacts, metrics JSON, manifests, logs, and post-run
per-rank artifacts. The coordinator no longer scans `node_updates/*.json` to
decide quorum progress.

## Transport comparison

MPI point-to-point is the preferred Frontier-scale dense data-plane target. It
matches the existing Cray MPICH/GPU-aware transport benchmark, avoids
torch.distributed collective ordering hazards, and can carry sharded X/Z tensor
payloads without routing through Lustre. It should be used for the production
dense update stream once train.py sends real tensor buckets rather than bounded
debug metadata.

torch.distributed P2P/RCCL was not selected for this control/update path. It
would keep the implementation inside PyTorch, but it still depends on process
group setup and backend behavior that is easy to confuse with DDP/collective
semantics. The requirement is one rank per GPU with no DDP and no unanimity
barrier, so the control path should not depend on an all-rank process group.

TCP sockets were selected for the Python train.py smoke/debug control plane
because they are available in the current launcher without compiling MPI
bindings, work across Slurm-launched ranks, support retry/timeout behavior, and
make quorum progress independent of Lustre metadata visibility. The payloads in
this path are bounded metadata summaries, not dense E97 tensors, so TCP is
adequate for the debug quorum decision while keeping the code production-shaped:
coordinator, workers, absolute quorum, timeout, byte counts, submit latency, and
run-local latest/checkpoint ownership.

## Metrics and guards

Per generation, the metrics payload records:

- configured quorum and accepted quorum size;
- timed-out rank IDs and timed-out/failed/invalid counts;
- TCP bytes sent and submit/receive latency distributions;
- token/loss summaries from accepted ranks;
- merge duration and checkpoint duration from the existing async metrics schema;
- run-local latest/checkpoint paths and whether latest advanced.

Production latest paths remain guarded by launcher policy. Debug jobs publish
only under their run-local `async_run` directory; no production `latest.pt` path
is mutated by this transport.
