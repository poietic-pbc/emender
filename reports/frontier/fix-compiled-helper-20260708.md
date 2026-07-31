# Fix compiled-helper train.py IPC launch

Date: 2026-07-08
Task: `fix-compiled-helper`

## Summary

The failed `validate-compiled-helper` 1-node smoke exposed that every `train.py`
rank launched `compiled_mpich_dense_helper --run-once` as an independent
subprocess. The compiled helper expected one coherent MPI world matching the
train ranks, so eight unrelated helper processes never formed the required
quorum.

This pass removes that per-rank subprocess launch from the Python integration
and adds an in-process shared-library bridge:

- `ndm.async_diloco_compiled_mpich.run_compiled_mpich_dense_quorum` now loads
  `compiled_mpich_dense_helper.so` with `ctypes` and calls
  `compiled_mpich_dense_helper_run_once(ipc_dir, request_path)`.
- The C++ helper exports that C ABI and initializes MPI inside the already
  launched `srun -n N` rank world. The shared-library entrypoint intentionally
  does not call `MPI_Finalize`; the standalone CLI still finalizes only when it
  initialized MPI itself.
- The Frontier build script now emits both the CLI binary and sibling `.so`.
- Launcher scripts stage dense helper IPC under node-local temporary storage by
  default and keep durable metrics, summaries, and checkpoints under the run
  root.
- Launcher scripts enforce the intended local Python environment and capture it
  in the artifacts.
- Helper trace files can be enabled with `ASYNC_COMPILED_MPICH_TRACE_DIR`.

During Frontier testing, the in-process bridge correctly rendezvoused all eight
ranks, but the current helper's full root-gather dense payload transfer is not
the scale-ready transport for the 1.3B model. For the 1-node debug smoke only,
the launcher enables `ASYNC_COMPILED_MPICH_FILE_GATHER=1`, a same-node
diagnostic mode where non-root ranks send MPI preambles and rank 0 resolves the
node-local bucket paths from peer request files. This validates the train.py
bridge, environment, request/result contract, parser, metrics, and latest
publication path without promoting the full root-gather transport to 2-node or
scale use.

`implement-compiled-mpich-2` already exists and blocks `run-compiled-helper`.
It owns the streaming/tree reducer needed before any 2-node, 8-node, 64-node,
or 256-node scale ladder is submitted.

## Code validation

Focused tests:

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest \
  tests/test_async_diloco_compiled_mpich.py \
  tests/test_trainpy_async_quorum_smoke_launchers.py \
  tests/test_async_diloco_e97_2n8n_debug_runner.py \
  -q
```

Result: `17 passed in 44.88s`.

The focused contract test monkeypatches `subprocess.run` to fail if the old
independent-helper-subprocess pattern is used. It also stubs the shared-library
call and verifies the dense quorum request/result path still works.

The C++ parser contract test builds the helper with Frontier `CC`, writes an
80-bucket stable JSON request, runs
`compiled_mpich_dense_helper --request request.gen000000.json --validate-request`,
and verifies all 80 bucket paths are preserved. This catches the alternating
bucket parser bug seen during the fourth debug smoke.

Python compile:

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m py_compile \
  ndm/async_diloco_compiled_mpich.py \
  ndm/async_diloco_real.py \
  scripts/frontier/e97_async_diloco_train.py
```

Result: passed.

Shell syntax:

```bash
bash -n \
  scripts/frontier/build_compiled_mpich_dense_helper.sh \
  scripts/frontier/trainpy_async_quorum_smoke_common.sh \
  scripts/frontier/async_diloco_e97_256n12h_launch.sbatch \
  scripts/frontier/compiled_mpich_dense_helper_2n_diag.sbatch
```

Result: passed.

Compiled helper build:

```bash
ARTIFACT_DIR=/tmp/compiled-mpich-helper-fix-compiled-helper-final \
  scripts/frontier/build_compiled_mpich_dense_helper.sh
```

Result: produced both:

- `/tmp/compiled-mpich-helper-fix-compiled-helper-final/compiled_mpich_dense_helper`
- `/tmp/compiled-mpich-helper-fix-compiled-helper-final/compiled_mpich_dense_helper.so`

Loaded-module dependency check:

```bash
source scripts/frontier/frontier_runtime_env.sh
frontier_load_default_modules
ARTIFACT_DIR=/tmp/compiled-mpich-helper-fix-compiled-helper-loaded \
  scripts/frontier/build_compiled_mpich_dense_helper.sh
ldd /tmp/compiled-mpich-helper-fix-compiled-helper-loaded/compiled_mpich_dense_helper.so \
  | grep -E 'libamdhip|libhsa|libmpi_gtl'
```

Result: no `libamdhip`, `libhsa`, or `libmpi_gtl` dependency in the helper
shared library.

## Frontier smoke evidence

Passing 1-node debug smoke:

- Slurm job: `4954204`
- Job name: `compiled-helper-fix-1n-r6`
- Slurm status: `COMPLETED`, exit `0:0`, elapsed `00:04:24`, nodes `1`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_fix_1n_r6/4954204-20260708T000729Z`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_fix_1n_r6/4954204-20260708T000729Z/summaries/summary.md`

Wrapper validation:

```text
Validation: pass
Exit status: 0
Rank starts: 8 / 8
Accepted updates: 8
Timed-out updates: 0
Tokens: 1032
```

Selected metrics:

```text
quorum_status: advanced
participating_workers: 8
quorum_size: 8
tokens_per_sec: 8.574041510278255
loss: 13.633476257324219
loss_100: 13.633476257324219
mpi.world_size: 8
mpi.provided_thread_level: MPI_THREAD_SERIALIZED
update_bytes.accepted: 44054163968
update_bytes.accepted_dense_delta: 44054163968
update_bytes.mpi_dense_payload_received: 44054163968
update_bytes.mpi_dense_payload_sent: 44054163968
```

Artifact checks:

- `artifacts/rank-start.tsv` contains 8 rank-start records.
- `artifacts/env.txt` captured:
  - `emender_conda_env=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`
  - `python_bin=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python`
  - `python_executable=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python`
  - `torch.__version__=2.10.0+rocm7.1`
  - `async_compiled_mpich_file_gather=1`
  - `async_compiled_mpich_trace_dir=<run root>/artifacts/compiled_mpich_trace`
- `artifacts/compiled_mpich_trace/` contains bridge/MPI/request/result trace
  events for ranks 0 through 7.
- `ldd artifacts/compiled_mpich_dense_helper.so | grep -E
  'libamdhip|libhsa|libmpi_gtl'` produced no matches.
- `async_run/latest.json` is under the run root.
- Checkpoint artifacts are under the run root:
  - `async_run/generations/gen_000000/manifest.json`
  - `async_run/recovery_checkpoints/gen_000000/initial.json`
  - `async_run/export_checkpoints/gen_000000/initial.json`
  - `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

No production `latest` path was written by this debug run.

## Jobs not submitted

No 2-node job was submitted from this task after the 1-node diagnostic pass,
because the only passing data-plane mode in this pass is the same-node
`ASYNC_COMPILED_MPICH_FILE_GATHER=1` diagnostic path. The scale-ready
streaming/tree reducer is intentionally left to `implement-compiled-mpich-2`.

No 8-node, 64-node, or 256-node job was submitted.

