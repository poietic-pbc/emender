# Compiled Cray MPICH dense helper design

Task: `design-compiled-mpich`

Date: 2026-07-07

## Decision

Build the first compiled dense transport as a standalone C++ Cray MPICH helper
process launched under the same Slurm MPI allocation as `train.py`, with local
handoff through run-local shared-memory or staged files and a small JSON control
contract. Do not start with a Python extension.

The selected route is:

- one Slurm MPI rank per GPU, still no DDP and no `torch.distributed` collectives
- Python `scripts/frontier/e97_async_diloco_train.py` owns model training,
  delta computation, merge math, metrics, and run-local latest/checkpoint
  publication
- C++ helper binary owns `MPI_Init_thread`, Cray MPICH point-to-point progress,
  dense bucket send/receive, and rank-level timeout/error reporting
- local Python-to-helper handoff uses node-local shared memory when available
  and falls back to run-local staged files under `$RUN_DIR/ipc/`; Lustre remains
  artifact/checkpoint storage only and is not a live quorum transport
- 256-node production launch remains fail-closed until a 64-node helper gate
  artifact passes

This is intentionally minimal. The objective is to prove the MPI substrate and
dense bucket movement at 1n, 2n, 8n, and 64n before optimizing the Python/local
handoff or adding a pybind API.

## Why this route

### Subprocess helper

Advantages:

- Matches the passing evidence: the 2-node C/Cray MPICH
  `Init_thread`/sendrecv diagnostic worked while the 2-node mpi4py paths failed
  in `PMPI_Init_thread` / OFI `fi_getinfo`.
- Keeps MPI lifetime outside the Python interpreter. Python never imports
  `mpi4py`, never calls `MPI_Init_thread`, and never links against Python MPI
  wheels or Cray Python site packages.
- Keeps failure containment simple. If helper initialization fails, rank-local
  Python sees a nonzero helper status, emits a metrics record, and the job fails
  closed before any production latest pointer can advance.
- Keeps the first implementation buildable with Cray wrappers (`CC`/`cc`) and
  ordinary C++/C MPI APIs. No Python ABI, pybind, Cython, or wheel packaging is
  required on Frontier.
- Allows staged CPU byte payloads first. GPU-aware MPICH can be tested later
  behind the same helper control API after the host-staged 1n/2n/8n/64n ladder
  is green.

Costs:

- Local handoff adds one extra copy for host-staged payloads.
- Python and helper processes need a small control protocol and cleanup rules.
- A root-rank helper still needs careful timeout and memory accounting for E97
  payloads.

### Python extension / pybind helper

Advantages:

- Lowest-latency integration once stable: Python could pass buffer views
  directly into compiled MPICH code.
- Easier in-process return of result metrics and future GPU buffer handles.
- Natural place to reuse Python-side envelope types if the ABI is stable.

Costs:

- Reintroduces MPI initialization into the Python process, which is the failure
  surface that triggered this track.
- Adds Frontier packaging risk: Python version, PyTorch ABI, pybind ABI, Cray
  MPI library paths, and GTL/GPU-aware linking all have to line up before the
  2-node smoke can even isolate transport behavior.
- Harder failure containment. A bad `MPI_Init_thread` or libfabric setup aborts
  the training interpreter instead of a narrow helper binary.

Use this only after the subprocess route passes 64n and the remaining overhead
is measured to be material.

### Other IPC choices

Unix domain sockets or pipes are reasonable for small control messages, but
they are not the first choice for E97 dense payloads because they add stream
framing and backpressure failure modes without helping with MPI initialization.
They remain acceptable for the helper control channel if shared-memory creation
is unavailable.

Live Lustre files are explicitly rejected for dense update movement. Staged
files may be used only as a local handoff fallback between the Python rank and
its same-rank helper, under a run-local scratch directory with atomic manifest
renames. They must not be polled by remote ranks and must not implement quorum.

## Process model

The production Slurm shape stays one task per GPU:

```text
srun -N <nodes> --ntasks-per-node=8 --gpus-per-task=1 \
  scripts/frontier/run_trainpy_with_mpich_helper.sh
```

Each Slurm task starts one Python trainer process and one helper process on the
same node/GPU binding. The recommended first wrapper is:

```text
rank N shell
  export LOCAL_RANK / ROCR_VISIBLE_DEVICES from Slurm
  start compiled helper for rank N
  start python train.py async DiLoCo rank N
  wait for both; if either exits nonzero, fail rank N
```

The helper binary is compiled by Cray C++:

```text
CC -O2 -std=c++17 -Wall -Wextra \
  scripts/frontier/compiled_mpich_dense_helper.cpp \
  -o "$ARTIFACT_DIR/compiled_mpich_dense_helper"
```

The helper calls:

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
```

and exits before accepting work if `provided < MPI_THREAD_SERIALIZED`, world
size differs from `ASYNC_EXPECTED_RANKS`, rank/GPU binding is inconsistent, or
required run-local directories cannot be created.

## train.py integration boundary

Add a transport mode such as:

```text
--actual-multinode-compiled-mpich-quorum
--compiled-mpich-helper-bin <path>
--compiled-mpich-ipc-dir <run_dir>/ipc
--compiled-mpich-bucket-bytes 67108864
--compiled-mpich-timeout-s <seconds>
```

Python keeps the current async DiLoCo responsibilities:

1. load the base checkpoint read-only
2. train one local E97 worker on the rank's GPU
3. compute `worker_state - base_state`
4. pack deterministic dense update headers and buckets using the existing
   `ndm.async_diloco_mpi` schema or a byte-compatible v2 schema
5. hand bucket descriptors to the helper
6. receive helper result metrics and, on rank 0 only, merge accepted updates
   through the existing quorum math
7. publish run-local manifests, recovery checkpoints, finalization checkpoints,
   and `latest.json` only after quorum advances

The helper does not write checkpoints, does not advance latest pointers, does
not merge tensors, and does not know about training data. It moves bytes and
reports which rank payloads arrived, failed checksum validation, timed out, or
were marked stale by header metadata.

## Local IPC API

Use a run-local request/response directory:

```text
$RUN_DIR/ipc/
  rank_00000/
    helper.ready.json
    request.gen000123.json
    request.gen000123.done
    result.gen000123.json
    error.gen000123.json
    shm/
```

Control files are written with temp-file plus atomic rename. Payloads are not
remote quorum files; they are same-rank handoff artifacts only.

Request JSON:

```json
{
  "schema_version": 1,
  "command": "dense_quorum_generation",
  "run_id": "E97_1.3B_step1065000_trainpy_async_quorum_2n",
  "rank": 6,
  "world_size": 16,
  "generation": 12,
  "base_generation": 11,
  "base_checkpoint": "/path/to/run-local/base/latest.pt",
  "quorum": 11,
  "timeout_s": 900.0,
  "bucket_bytes_target": 67108864,
  "header_bytes": 74170,
  "payload_bytes": 1234567890,
  "header_path": "rank_00006/gen000012/header.json",
  "bucket_descriptors": [
    {
      "index": 0,
      "nbytes": 67108864,
      "checksum_sha256": "...",
      "ipc": {
        "kind": "shm",
        "name": "/emender_dense_rank6_gen12_bucket0",
        "offset": 0
      }
    }
  ]
}
```

Allowed `ipc.kind` values:

- `shm`: POSIX shared memory or Linux `memfd` exported by descriptor/name in
  the rank-local namespace; preferred for 1n/2n/8n/64n.
- `file`: run-local staged file under `$RUN_DIR/ipc/rank_<rank>/gen<gen>/`;
  fallback only, using atomic rename after the Python writer fsyncs and closes.

Result JSON:

```json
{
  "schema_version": 1,
  "status": "advanced",
  "rank": 0,
  "generation": 12,
  "base_generation": 11,
  "accepted_ranks": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
  "timed_out_ranks": [8, 12, 13, 14, 15],
  "failed_ranks": [],
  "stale_ranks": [],
  "bytes_sent": 1234567890,
  "bytes_received": 13580246790,
  "helper_exit_code": 0,
  "mpi": {
    "provided_thread_level": "MPI_THREAD_SERIALIZED",
    "world_size": 16,
    "root_rank": 0
  },
  "received_payloads": [
    {
      "rank": 1,
      "header_path": "rank_00000/gen000012/from_rank_00001.header.json",
      "bucket_paths": ["rank_00000/gen000012/from_rank_00001.bucket00000.bin"]
    }
  ]
}
```

For the first implementation, rank 0 Python can reconstruct
`DenseUpdateEnvelope` objects from `received_payloads` and call
`collect_dense_quorum_from_envelopes`. Later, root helper can optionally return
a packed accepted-rank stream over shared memory to avoid root-side staged
files.

## MPI data movement

Dense delta movement is helper-to-helper over Cray MPICH only:

1. Every Python rank packs its local delta into the existing deterministic
   header plus bucket stream.
2. Every local helper waits for its request manifest, maps the local bucket
   payloads, and validates declared byte counts before entering the generation.
3. Non-root helpers send a fixed-size control preamble to root with
   `MPI_Isend`, then send the JSON header as `MPI_BYTE`, then send each dense
   bucket as `MPI_BYTE` using tags derived from generation, source rank, and
   bucket index.
4. Root helper posts nonblocking receives for header lengths, headers, and
   buckets. It stops waiting once quorum-sized valid payloads have arrived or
   the generation deadline expires.
5. Root helper writes received rank payloads into root-local IPC buffers/files
   and emits `result.genNNN.json`.
6. Root Python validates SHA256 checksums using the existing envelope unpacker,
   rejects stale generation/base-generation payloads, runs quorum merge, and
   publishes run-local latest/checkpoint metadata if advanced.
7. Root helper broadcasts a compact decision payload to non-root helpers. Each
   non-root helper writes its local result JSON. Python ranks use this only for
   metrics/rebase control and never publish global latest.

No mpi4py is imported. No dense update is sent through TCP. No remote rank polls
Lustre for live quorum progress. Lustre-visible files remain limited to normal
run artifacts: logs, metrics, manifests, recovery/export/finalization
checkpoints, and post-run reports.

## Data format

Reuse the current dense envelope fields so the Python validation and merge code
does not churn:

- `schema_version`
- `transport`: change to `compiled-cray-mpich-helper-p2p`
- `run_id`
- `rank` and `worker_id`
- `generation`
- `base_generation`
- optional `base_checkpoint`
- `tokens`, `local_steps`, and loss moving-average fields
- `staleness`, `failed`, `timed_out`, `invalid`
- target bucket bytes, total payload bytes, payload SHA256
- bucket index, offset, byte count, SHA256
- per-tensor name, shape, dtype, byte offset, byte count, SHA256

The first helper treats headers and buckets as opaque bytes except for the
fields needed to size receives, route ranks, enforce generation/world metadata,
and write metrics. Tensor reconstruction and merge remain Python-side.

Tag allocation should avoid unbounded tag growth:

```text
TAG_CONTROL_BASE + generation_mod_window
TAG_HEADER_BASE  + generation_mod_window
TAG_BUCKET_BASE  + (bucket_index % bucket_tag_window)
TAG_RESULT_BASE  + generation_mod_window
```

The helper must check that `bucket_count * bucket_bytes_target` fits configured
memory limits before posting all receives. For 8n/64n, root can process buckets
rank-by-rank or in bounded windows to avoid allocating every E97 payload at
once.

## Lifecycle

Startup:

1. Slurm launches one rank per GPU.
2. Wrapper compiles or locates helper binary and records its path in the run
   manifest.
3. Helper starts, initializes Cray MPICH, writes `helper.ready.json`, and waits
   for generation requests.
4. Python waits for helper readiness before starting dense quorum mode.

Per generation:

1. Python writes payload buffers and request manifest.
2. Helper consumes the request and enters the MPI generation.
3. Helper writes result or error manifest.
4. Python consumes result. Rank 0 merges and publishes only if quorum advanced.
5. Python writes heartbeat/metrics including helper timings and rank status.
6. Python deletes or rotates per-generation IPC payloads after result
   consumption, keeping enough artifacts for debug when configured.

Shutdown:

1. Python writes a `shutdown` request after finalization or fail-closed abort.
2. Helper broadcasts shutdown, calls `MPI_Finalize`, and exits zero only if all
   completed generations had valid result/error manifests.
3. Wrapper fails the Slurm task if helper or Python exits nonzero.

## Failure handling

Fail closed before publication:

- helper MPI init failure
- helper world-size mismatch
- local rank/GPU binding mismatch
- missing or corrupt request manifest
- bucket byte count exceeds configured cap
- checksum mismatch for a rank needed to satisfy quorum
- root cannot write result manifest
- Python cannot validate helper result

Quorum-deferred but job can continue when policy allows:

- some ranks time out but accepted fresh ranks meet quorum
- stale ranks arrive for a previous base generation
- a rank reports local training failure but quorum still advances without it

In both cases, production latest/checkpoint semantics stay Python-owned:

- no helper path updates production `latest.pt`, chain pointers, or run-local
  `latest.json`
- rank 0 Python advances run-local latest only after the existing quorum merge
  returns `latest_advanced=True`
- debug jobs always point at isolated run directories and record before/after
  production latest identity
- non-debug 256n launch refuses to start without the 64n compiled-helper gate
  artifact

## Minimal implementation plan

1. Add `scripts/frontier/compiled_mpich_dense_helper.cpp`.
   - Implement `MPI_Init_thread`, rank/world validation, ready/error manifests,
     one-generation request parsing, header/bucket `MPI_BYTE` send/receive, and
     result JSON output.
   - Start host-staged only with `MPICH_GPU_SUPPORT_ENABLED=0`.

2. Add a narrow Python bridge module, for example
   `ndm/async_diloco_compiled_mpich.py`.
   - Reuse `pack_dense_update`, `unpack_dense_update`, and
     `collect_dense_quorum_from_envelopes`.
   - Write request manifests and local IPC payloads.
   - Parse helper results and return the same payload shape as
     `run_mpi_dense_quorum`.

3. Wire `scripts/frontier/e97_async_diloco_train.py`.
   - Add `--actual-multinode-compiled-mpich-quorum`.
   - Keep `--actual-multinode-mpi-dense-quorum` as the old mpi4py path for
     comparison only, not as production Track B.
   - Reject simultaneous TCP, mpi4py MPI dense, and compiled helper modes.

4. Add launch wrappers.
   - `scripts/frontier/trainpy_compiled_mpich_quorum_smoke_common.sh`
   - 1n, 2n, 8n, and 64n sbatch wrappers derived from existing train.py async
     quorum scripts.
   - Update the 256n wrapper to require a compiled-helper 64n gate artifact,
     not the mpi4py dense gate.

5. Add tests.
   - Unit tests for request/result JSON schema and local handoff cleanup.
   - Local helper compile smoke with a tiny two-rank synthetic payload when MPI
     is available; otherwise skip loudly.
   - Existing async DiLoCo merge and checkpoint tests continue to validate
     Python-owned semantics.

## Validation ladder

1n gate:

- compile helper with Cray `CC`
- run 8 ranks on one Frontier node, one rank per GPU
- prove Python never imports `mpi4py`
- prove helper initializes Cray MPICH and completes one E97 dense generation
- verify run-local `latest.json` and checkpoint paths exist
- verify production latest/checkpoint identity is unchanged

2n gate:

- run 16 ranks across two nodes
- reproduce the prior failing topology with the compiled helper
- require helper `MPI_Init_thread` success and dense bucket movement between
  nodes
- validate missing/slow rank accounting with quorum below unanimity
- record `FI_*`, `MPICH_*`, Slurm MPI type, helper build command, and result
  metrics

8n gate:

- run 64 ranks
- require bounded root memory use, no live Lustre update files, no TCP dense
  path, and no DDP
- verify accepted/stale/timed-out rank metrics and per-rank byte counts
- compare helper payload checksums against Python-side unpack validation

64n gate:

- run 512 ranks with the selected E97 debug cadence
- require at least one completed dense generation, run-local latest advance,
  and checkpoint finalization in the isolated run directory
- require no helper crash, no MPI abort, and no production pointer mutation
- emit a machine-readable gate artifact containing:
  `transport=compiled-cray-mpich-helper-p2p`,
  `world_size=512`, `nodes=64`, quorum status, accepted ranks, bytes moved,
  helper build hash, and log paths

256n debug/production gate:

- 256n debug may launch only with explicit debug approval and isolated output;
  it must not update production chain latest pointers
- non-debug 256n production launch must fail closed unless
  `ASYNC_COMPILED_MPICH_64N_GATE_JSON` points to a readable passing 64n gate
  artifact for the compiled helper
- the 256n wrapper must reject mpi4py dense gate artifacts for this Track B
  path

## Initial non-goals

- no pybind/Python extension in the first implementation
- no GPU device-buffer MPI requirement until host-staged helper passes 64n
- no DDP, no NCCL/RCCL collectives, no torch distributed process group
- no live Lustre quorum or remote file polling for dense updates
- no 256n production approval based only on 1n/2n/8n evidence
