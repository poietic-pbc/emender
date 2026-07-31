# Frontier MPI dense async DiLoCo validation, 2026-07-07

Task: `implement-frontier-mpi`

## Summary

Implemented the importable MPI dense transport and wired train.py async DiLoCo
launchers to select it explicitly. Local unit/integration tests pass. Frontier
debug smoke attempts reached real E97 checkpoint/data loading and exposed
several concrete runtime transport blockers. The latest code includes fixes for
the blockers observed up to job `4952476`, but no passing Frontier MPI dense
smoke was obtained in this pass. The 256-node production wrapper remains
fail-closed behind a required 64-node dense-transport gate artifact.

## Local validation

Commands:

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m py_compile ndm/async_diloco_mpi.py ndm/async_diloco_real.py scripts/frontier/e97_async_diloco_train.py
bash -n scripts/frontier/trainpy_async_quorum_smoke_common.sh scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_async_diloco_mpi_transport.py tests/test_async_diloco_real_trainer.py tests/test_trainpy_async_quorum_smoke_launchers.py tests/test_async_diloco_e97_2n8n_debug_runner.py
```

Result:

```text
28 passed in 29.97s
```

Covered locally:

- Delta pack/unpack preserves tensor payload and wire metadata.
- SHA256 checksum validation rejects corrupted bucket payloads.
- Stale-generation/staleness metadata rejects stale updates while accepting a
  fresh quorum.
- Timeout/missing-rank accounting advances without unanimity when quorum is met.
- Fractional quorum and weighted bucket merge math are correct.
- Existing real train.py async trainer tests still pass.
- Launchers select MPI dense by default, preserve TCP as explicit fallback, and
  keep one rank per GPU.
- Production wrapper remains fail-closed without a passing 64-node dense gate.

## Frontier debug attempts

### Job `4952434`: 2n MPI dense, pre-fix

Command:

```bash
WG_TASK_ID=implement-frontier-mpi ASYNC_QUORUM_TRANSPORT=mpi-dense sbatch --parsable scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Result: `FAILED`, exit `2:0`, elapsed `00:00:21`.

Artifacts:

- stdout: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4952434.out`
- stderr: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4952434.err`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_2n/4952434-20260707T154245Z`

Failure evidence:

- The old TCP smoke expected one phantom missing rank (`node-count=17` with
  `srun -n 16`), which is invalid for MPI because `requested_ranks` cannot
  exceed `MPI_COMM_WORLD`.
- The shell manifest block also had nested command-substitution quoting that
  failed before launch validation completed.

Fix applied:

- MPI dense smoke expected-rank accounting now clamps to launched ranks.
- Manifest `bounded_debug_transport` is computed in a variable, not inline.

### Job `4952443`: 1n MPI dense, missing `mpi4py`

Command:

```bash
WG_TASK_ID=implement-frontier-mpi ASYNC_QUORUM_TRANSPORT=mpi-dense sbatch --parsable scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch
```

Result: `FAILED`, exit `90:0`, elapsed `00:01:36`.

Artifacts:

- stdout: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4952443.out`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4952443-20260707T154530Z`

Failure evidence:

```text
ModuleNotFoundError: No module named 'mpi4py'
RuntimeError: mpi4py is required for MPI dense transport
```

Positive evidence:

- Real E97 data/checkpoint path was readable.
- All 8 ranks started before transport import failure.

Fix applied:

- The transport module now appends Cray's bundled `mpi4py` site-packages using
  `site.addsitedir` when normal `mpi4py` import fails.
- Launchers record `CRAY_MPI4PY_SITE` for reproducibility.

### Job `4952450`: 1n MPI dense, `PYTHONPATH` ordering conflict

Result: `FAILED`, exit `90:0`, elapsed `00:00:11`.

Artifacts:

- stdout: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4952450.out`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4952450-20260707T155014Z`

Failure evidence:

```text
ImportError: cannot import name 'TypeIs' from 'typing_extensions'
(/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages/typing_extensions.py)
```

Fix applied:

- Removed launcher `PYTHONPATH` mutation.
- Kept Cray `mpi4py` discovery inside the transport module so it does not shadow
  the training environment's newer dependencies.

### Job `4952455`: 1n MPI dense, GTL/GPU-aware abort

Result: `FAILED`, exit `90:0`, elapsed `00:02:29`.

Artifacts:

- stdout: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4952455.out`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4952455-20260707T155255Z`

Failure evidence:

```text
MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked
```

Positive evidence:

- The job loaded the real E97 checkpoint/data path.
- All ranks reached `mpi_dense_send_starting` heartbeats.

Fix applied:

- Python MPI dense launch defaults `MPICH_GPU_SUPPORT_ENABLED=0` because this
  implementation sends host-staged byte buckets. True GPU-aware MPI requires a
  GTL-linked lower-level extension or a compatible `mpi4py` build.

### Job `4952464`: 1n MPI dense, large payload pickle truncation

Result: `FAILED`, exit `90:0`, elapsed `00:04:07`.

Artifacts:

- stdout: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4952464.out`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4952464-20260707T155703Z`

Failure evidence:

```text
mpi4py.MPI.Exception: Message truncated
Message from rank 0 and tag 0 truncated; 32768 bytes received but buffer size is 74172
TimeoutError: rank <n> timed out sending dense MPI update
```

Fix applied:

- Bucket payloads now use explicit `MPI.BYTE` `Isend`/`Irecv` buffers instead
  of mpi4py Python-object pickle sends.

### Job `4952476`: 1n MPI dense, large header pickle truncation

Result: `FAILED`, exit `90:0`, elapsed `00:03:55`.

Artifacts:

- stdout: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4952476.out`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4952476-20260707T160310Z`

Failure evidence:

```text
mpi4py.MPI.Exception: Message truncated
Message from rank 6 and tag 0 truncated; 32768 bytes received but buffer size is 74170
TimeoutError: rank <n> timed out sending dense MPI update
```

Positive evidence:

- Real E97 training ran long enough to produce local deltas.
- The failure is at MPI dense header movement, not data/checkpoint loading.

Fix applied after this job:

- Headers now use `length -> MPI.BYTE header -> MPI.BYTE buckets`.
- No additional Slurm retry was submitted after this patch in this pass.

## 8n / 64n / 256n status

Initial 8n and 64n submission attempts were blocked by
`QOSMaxSubmitJobPerUserLimit` while smaller jobs were in flight. After the final
1n failure there was no passing 1n dense smoke, so 8n and 64n were not retried.

No 256-node production job was launched. The production wrapper still refuses
non-debug 256-node launch unless `ASYNC_DENSE_TRANSPORT_64N_GATE_JSON` points at
a readable passing 64-node MPI dense transport artifact.

## Current go/no-go

No-go for production.

The code now has local coverage for the update wire format, checksums, stale
handling, timeout quorum behavior, and merge math. Frontier debug evidence
shows the path reaches real E97 training and the MPI dense send stage, but a
post-`4952476` Slurm retry is still required to validate the latest
length-prefixed header/bucket patch. A 2n/8n/64n ladder must pass before any
256-node production run.
