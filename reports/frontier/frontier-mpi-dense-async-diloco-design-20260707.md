# Frontier MPI dense async DiLoCo transport design

Task: `implement-frontier-mpi`

Date: 2026-07-07

## Transport selection

The dense update data plane is implemented as an explicit `mpi-dense` transport
for `scripts/frontier/e97_async_diloco_train.py`.

The selected production target remains Cray MPICH point-to-point on Frontier.
The Python implementation uses `mpi4py` over Cray MPICH and moves serialized
dense tensor-delta buckets with explicit `MPI.BYTE` `Isend`/`Irecv` calls.
Headers are also transferred as length-prefixed byte buffers so large E97 tensor
metadata does not go through mpi4py's Python-pickle message path.

The current Python path is host-staged: tensors are packed into CPU byte buckets
before MPI sends. This is dense MPI point-to-point and removes TCP/Lustre from
the live tensor data plane, but it is not yet true GPU-buffer MPI. Frontier
runtime attempts showed Cray's bundled `mpi4py` is not linked for GTL/GPU-aware
MPI (`MPICH_GPU_SUPPORT_ENABLED=1` aborts with `GTL library is not linked`), so
the launch wrappers default `MPICH_GPU_SUPPORT_ENABLED=0` for this Python
transport. A future lower-level extension can replace the pack/send layer with
real device-buffer sends while keeping the same wire metadata and quorum logic.

TCP remains selectable only as `ASYNC_QUORUM_TRANSPORT=tcp` for bounded
metadata/control-plane debugging. TCP is not the dense data plane. Lustre is
used only for metrics, manifests, logs, checkpoints, and post-run reports.

## Wire format

`ndm.async_diloco_mpi.DenseUpdateEnvelope` is the wire object. It contains:

- `schema_version`
- transport name: `cray-mpich-gpu-aware-p2p`
- `run_id`
- sender `rank` and `worker_id`
- `generation`
- `base_generation`
- optional `base_checkpoint`
- accepted-token count
- local step count
- loss moving-average window
- explicit staleness integer
- failed/timed-out/invalid status flags
- target bucket size
- total payload byte count
- full-payload SHA256
- per-bucket metadata: index, stream offset, byte count, SHA256, and tensor entries
- per-tensor metadata: name, shape, dtype, stream offset, byte count, SHA256

The packer sorts tensor names for deterministic serialization. Unpack validates
full-payload, bucket, and tensor checksums before reconstructing an
`AsyncDiLoCoUpdate`.

## Bucket and tree shape

The default bucket target is `67108864` bytes, matching the existing 64 MiB
scaleout bucket convention. The current root shape is:

1. Each GPU rank trains one local E97 worker from the shared base checkpoint.
2. Each rank computes a delta from base state to local state.
3. Each rank packs the delta into checksummed buckets.
4. Non-root ranks send a header length, header bytes, and bucket bytes to rank 0
   via nonblocking MPI point-to-point.
5. Rank 0 receives until quorum advances or timeout expires, then merges only
   fresh accepted updates.
6. Rank 0 publishes run-local latest/checkpoint metadata only after quorum
   advances.

The code keeps the quorum layer separate from the rank-0 collection shape:
`collect_dense_quorum_from_envelopes` can be reused by a future tree or
hierarchical aggregator. The intended production extension is:

- GPU rank to node/group aggregator
- group aggregators to a coordinator tree
- coordinator result broadcast back to workers

That keeps rank 0 from becoming the only dense data hot spot at 256 nodes.

## Delta representation and merge

Updates are delta-form, not full endpoint checkpoints. Each update carries
`worker_state - base_state` and the merge applies weighted mean deltas to the
base state using the existing async DiLoCo `quorum_merge` math.

Accepted updates must match both attempted `generation` and `base_generation`.
Stale updates are rejected by default and accounted in metrics. The quorum
threshold can be absolute (`quorum`) or fractional (`quorum_fraction`).

## Metrics

The dense transport records:

- quorum size
- timed-out ranks
- stale ranks
- failed ranks
- bytes sent and received
- per-rank bucket receive timing
- merge latency
- rebase/checkpoint latency fields
- MPI dense payload bytes in `update_bytes`

The train.py wrapper still emits the existing run-local metrics JSON, summary,
manifest, rank-start log, progress heartbeats, and checkpoint/latest metadata.

## Production guard

`scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` now defaults to:

- one Slurm task per GPU: `--ntasks-per-node=8`, `--gpus-per-task=1`
- `ASYNC_ACTUAL_MULTINODE_MPI_DENSE_QUORUM=1`
- `ASYNC_DENSE_DATA_PLANE=mpi-p2p`
- `ASYNC_DENSE_UPDATE_STORAGE=1`
- TCP disabled by default

The 256-node wrapper remains fail-closed. For non-debug 256-node use,
`ASYNC_DENSE_TRANSPORT_64N_GATE_JSON` must point at a readable passing 64-node
MPI dense transport metrics/manifest artifact that mentions MPI and dense
transport. Without that artifact the presubmit gate fails before launching.

## Why TCP and Lustre are not the dense data plane

TCP quorum is useful for small metadata/control-plane debugging because it is
easy to inspect and does not require MPI Python runtime details. It is not
appropriate for E97 dense deltas because each generation moves model-scale
tensor payloads, and a rank-0 TCP socket would serialize traffic through a
single host path.

Lustre is intentionally excluded from live update collection. It remains a
durable artifact plane only. Live dense updates through shared files would turn
metadata services and storage bandwidth into the synchronization bottleneck and
would make slow/missing ranks look like filesystem consistency problems instead
of transport/quorum events.
