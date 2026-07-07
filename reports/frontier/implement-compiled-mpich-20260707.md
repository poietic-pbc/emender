# Implement compiled MPICH dense helper

Task: `implement-compiled-mpich`

Date: 2026-07-07

## Summary

Implemented the first compiled dense transport path for train.py-native async
DiLoCo as a standalone C++ Cray MPICH helper plus a Python request/result bridge.

The explicit production transport name is:

```text
compiled-cray-mpich-helper-p2p
```

TCP remains selectable as a metadata/debug fallback.  The old `mpi-dense`
mpi4py path remains present only as an explicit legacy comparison path and is
not the smoke-wrapper or 256-node production default.

No 8-node, 64-node, or 256-node training job was submitted by this task.  The
only Slurm submission was the bounded 2-node helper diagnostic required by this
task's validation section.

## Code Artifacts

- `scripts/frontier/compiled_mpich_dense_helper.cpp`
  - C++17 helper using `MPI_Init_thread(..., MPI_THREAD_SERIALIZED, ...)`.
  - Supports `--diagnostic` sendrecv mode.
  - Supports one request/result generation with file IPC payload handoff and
    MPI byte movement from non-root ranks to rank 0.
- `scripts/frontier/build_compiled_mpich_dense_helper.sh`
  - Frontier build wrapper using `CC` by default.
- `scripts/frontier/compiled_mpich_dense_helper_2n_diag.sbatch`
  - 2-node debug-QOS helper-only diagnostic.
- `ndm/async_diloco_compiled_mpich.py`
  - Python bridge for request manifests, bucket files, helper invocation,
    result loading, checksums via the existing dense envelope unpacker, and
    root-side merge through existing quorum math.
- `ndm/async_diloco_real.py`
  - Adds `compiled-cray-mpich-helper-p2p` as a separate actual multinode
    transport.
  - Rank 0 still owns checkpoint/latest publication after helper result
    validation.
- `scripts/frontier/e97_async_diloco_train.py`
  - Adds `--actual-multinode-compiled-mpich-quorum`,
    `--compiled-mpich-helper-bin`, and `--compiled-mpich-ipc-dir`.
  - Rejects simultaneous TCP, mpi4py dense, and compiled-helper modes.
- `scripts/frontier/trainpy_async_quorum_smoke_common.sh`
  - Defaults `ASYNC_QUORUM_TRANSPORT` to
    `compiled-cray-mpich-helper-p2p`.
  - Builds the helper into the run artifact directory when needed.
- `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
  - Defaults Track B production selection to compiled helper.
  - Fails closed for non-debug 256n unless
    `ASYNC_COMPILED_MPICH_64N_GATE_JSON` points to a readable passing 64n
    compiled-helper gate artifact containing
    `compiled-cray-mpich-helper-p2p`.

## Build Commands

Local Frontier login build:

```bash
CXX=CC ARTIFACT_DIR=/tmp/compiled-mpich-helper-build-${USER:-agent} \
  scripts/frontier/build_compiled_mpich_dense_helper.sh
```

Observed output artifact:

```text
/tmp/compiled-mpich-helper-build-erikgarrison/compiled_mpich_dense_helper
```

The 2-node diagnostic wrapper build command recorded by Slurm job `4953892`:

```text
CC -O2 -std=c++17 -Wall -Wextra \
  /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-816/scripts/frontier/compiled_mpich_dense_helper.cpp \
  -o /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z/artifacts/compiled_mpich_dense_helper
```

## Validation

Local syntax:

```bash
python3.11 -m py_compile \
  ndm/async_diloco_compiled_mpich.py \
  ndm/async_diloco_real.py \
  scripts/frontier/e97_async_diloco_train.py
```

Local C++ syntax:

```bash
CC -O2 -std=c++17 -Wall -Wextra -fsyntax-only \
  scripts/frontier/compiled_mpich_dense_helper.cpp
```

Local helper build:

```bash
CXX=CC ARTIFACT_DIR=/tmp/compiled-mpich-helper-build-${USER:-agent} \
  scripts/frontier/build_compiled_mpich_dense_helper.sh
```

Local/unit tests:

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest \
  tests/test_async_diloco_compiled_mpich.py \
  tests/test_trainpy_async_quorum_smoke_launchers.py \
  tests/test_async_diloco_e97_2n8n_debug_runner.py \
  -q
```

Result:

```text
15 passed in 22.29s
```

Helper diagnostic local one-rank mode:

```bash
/tmp/compiled-mpich-helper-build-erikgarrison/compiled_mpich_dense_helper --diagnostic
```

Result:

```json
{"diagnostic":"compiled_mpich_dense_helper","transport":"compiled-cray-mpich-helper-p2p","world_size":1,"provided_thread_level":"MPI_THREAD_SERIALIZED","rank0_received_from":0}
```

2-node debug-QOS helper diagnostic:

```bash
sbatch scripts/frontier/compiled_mpich_dense_helper_2n_diag.sbatch
```

Result:

```text
Submitted batch job 4953892
4953892|compiled-mpich-helper-2n|COMPLETED|0:0|00:00:14|2
4953892.0|compiled_mpich_dense_helper|COMPLETED|0:0|00:00:02|2
```

Diagnostic payload:

```json
{"diagnostic":"compiled_mpich_dense_helper","transport":"compiled-cray-mpich-helper-p2p","world_size":16,"provided_thread_level":"MPI_THREAD_SERIALIZED","rank0_received_from":15}
```

Diagnostic artifact root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z
```

Key artifacts:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z/summary.md`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z/artifacts/command.txt`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z/artifacts/compiled_mpich_dense_helper`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z/logs/build.log`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/compiled_mpich_helper/20260707/4953892-20260707T221643Z/logs/diagnostic.log`

## Next Smoke Command

The next bounded smoke should be a 1-node train.py async quorum run using the
compiled helper default:

```bash
sbatch --export=ALL,WG_TASK_ID=integrate-compiled-mpich-main,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p \
  scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch
```

If the 1-node smoke passes, continue sequentially to the existing 2-node smoke
wrapper.  Do not submit 8n/64n/256n training from this implementation task.
