# Debug 2n MPI OFI init failure, 2026-07-07

Task: `debug-2n-mpi`

## Verdict

The first failing post-patch 2-node MPI dense rung is reproduced by a minimal
2-node `mpi4py` `MPI.Init_thread` plus ring send/recv smoke. This is not a
train.py model, checkpoint, tensor bucket, or quorum bug: Cray MPICH aborts
inside OFI initialization before user-level MPI communication.

No fix was identified in this diagnostic pass. The attempted Frontier MPI/OFI
launcher changes did not make minimal `mpi4py` succeed:

- Baseline train.py-equivalent environment: failed with `OFI fi_getinfo()`.
- Explicit `FI_PROVIDER=cxi` plus `MPICH_OFI_NIC_POLICY=NUMA`: failed with the
  same `OFI fi_getinfo()`.
- Adding Slurm `--network=disable_rdzv_get` and `FI_CXI_RDZV_PROTO=alt_read`
  to the explicit CXI/NUMA environment: failed with the same `OFI fi_getinfo()`.

Do not retry the 2-node train.py MPI dense smoke until this minimal diagnostic
passes on Frontier. No 8n, 64n, or 256n follow-on job was submitted from this
task.

## Diagnostic Wrapper

Added reusable wrapper:

```text
scripts/frontier/mpi4py_ofi_2n_diag.sbatch
```

It uses the same Frontier module helper as the train.py smoke:

```bash
source "${REPO}/scripts/frontier/frontier_runtime_env.sh"
frontier_load_default_modules
frontier_activate_emender_conda_env
```

It uses the same launcher shape as the failed train.py rung:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest ...
```

The minimal Python payload appends Cray's bundled `mpi4py` site path, calls
`MPI.Init_thread(required=MPI.THREAD_SERIALIZED)`, then performs a ring
`sendrecv` and an `allgather` before `MPI.Finalize()`.

I also broadened `frontier_capture_runtime_env` in
`scripts/frontier/frontier_runtime_env.sh` to capture generic `FI_*` and
`MPICH_*` settings, not only `FI_CXI*`, so future artifacts preserve the exact
MPI/OFI knobs.

## Job 4953690: Baseline and Explicit CXI/NUMA

Exact submit command:

```bash
sbatch --parsable --export=ALL,WG_TASK_ID=debug-2n-mpi scripts/frontier/mpi4py_ofi_2n_diag.sbatch
```

Slurm request:

- Job id: `4953690`
- Partition/QOS: `batch` / `debug`
- Walltime: `00:10:00`
- Nodes: `2`
- Requested node-hours: `0.333333`
- Node list: `frontier[01515,01517]`

Slurm accounting:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NodeList
4953690|mpi4py-ofi-2n-diag|batch|debug|FAILED|255:0|00:00:26|2|frontier[01515,01517]
4953690.0|env|||FAILED|255:0|00:00:02|2|frontier[01515,01517]
4953690.1|env|||FAILED|255:0|00:00:02|2|frontier[01515,01517]
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953690-20260707T210438Z
```

Per-step commands recorded by the wrapper:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest env MPICH_GPU_SUPPORT_ENABLED=0 CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953690-20260707T210438Z/artifacts/mpi4py_init_sendrecv.py
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest env MPICH_GPU_SUPPORT_ENABLED=0 CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages FI_PROVIDER=cxi MPICH_OFI_NIC_POLICY=NUMA FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953690-20260707T210438Z/artifacts/mpi4py_init_sendrecv.py
```

Environment exports:

```text
MPICH_GPU_SUPPORT_ENABLED=0
CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages

# second step only
FI_PROVIDER=cxi
MPICH_OFI_NIC_POLICY=NUMA
FI_CXI_RX_MATCH_MODE=hybrid
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
```

Result:

- Baseline step status: `255`
- Explicit CXI/NUMA step status: `255`
- Minimal `mpi4py` did not reach send/recv.
- Failure reproduced the original train.py OFI error:

```text
Fatal error in PMPI_Init_thread
MPIDI_OFI_mpi_init_hook
open_fabric(1559): OFI fi_getinfo() failed (ofi_init.c:1559:open_fabric:No data available)
```

## Job 4953693: Slurm Network/RDZV Candidate

Exact submit command:

```bash
sbatch --parsable --network=disable_rdzv_get --export=ALL,WG_TASK_ID=debug-2n-mpi,FI_PROVIDER=cxi,MPICH_OFI_NIC_POLICY=NUMA,FI_CXI_RDZV_PROTO=alt_read,FI_CXI_RX_MATCH_MODE=hybrid,FI_CXI_DEFAULT_CQ_SIZE=131072,FI_CXI_DEFAULT_TX_SIZE=2048 scripts/frontier/mpi4py_ofi_2n_diag.sbatch
```

Slurm request:

- Job id: `4953693`
- Partition/QOS: `batch` / `debug`
- Walltime: `00:10:00`
- Nodes: `2`
- Requested node-hours: `0.333333`
- Slurm network: `disable_rdzv_get`
- Node list: `frontier[01517,01520]`

Slurm accounting:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NodeList
4953693|mpi4py-ofi-2n-diag|batch|debug|FAILED|255:0|00:00:25|2|frontier[01517,01520]
4953693.0|env|||FAILED|255:0|00:00:02|2|frontier[01517,01520]
4953693.1|env|||FAILED|255:0|00:00:01|2|frontier[01517,01520]
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953693-20260707T210606Z
```

Environment exports:

```text
MPICH_GPU_SUPPORT_ENABLED=0
CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages
FI_PROVIDER=cxi
MPICH_OFI_NIC_POLICY=NUMA
FI_CXI_RDZV_PROTO=alt_read
FI_CXI_RX_MATCH_MODE=hybrid
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
```

Result:

- Baseline step status: `255`
- Explicit CXI/NUMA step status: `255`
- Minimal `mpi4py` again reproduced `PMPI_Init_thread` /
  `MPIDI_OFI_mpi_init_hook` / `OFI fi_getinfo() failed`.
- `--network=disable_rdzv_get` plus `FI_CXI_RDZV_PROTO=alt_read` is therefore
  not sufficient to fix this 2-node mpi4py/MPICH OFI initialization failure.

## Comparison With Failed Train.py Rung

The failed train.py rung, Slurm job `4953646`, used:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest ...
```

and captured:

```text
mpich_gpu_support_enabled=0
cray_mpi4py_site=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages
modules include libfabric/2.3.1, craype-network-ofi, cpe/26.03, cray-mpich/9.1.0, cray-pmi/6.1.17, miniforge3/23.11.0-0, rocm/7.1.1, craype-accel-amd-gfx90a
```

The diagnostic jobs used the same module stack and `MPICH_GPU_SUPPORT_ENABLED=0`
host-staged Python MPI posture. Because the minimal `MPI.Init_thread` fails
before any model payload runs, the dense train.py rung is blocked below the
application layer.

## Retry Recipe

There is no fixed 2n train.py retry recipe from this task. The safe gate is:

1. First make this command pass on two nodes:

   ```bash
   sbatch --parsable --export=ALL,WG_TASK_ID=debug-2n-mpi scripts/frontier/mpi4py_ofi_2n_diag.sbatch
   ```

   Passing means at least one 16-rank step exits `0` and prints the rank-0 JSON
   from `mpi4py_init_sendrecv.py`, including `world_size: 16` and two unique
   Frontier hosts.

2. Only after that, retry the 2n train.py MPI dense smoke with the original
   ladder command plus whatever MPI/OFI launcher or environment change made the
   minimal diagnostic pass:

   ```bash
   sbatch --parsable --export=ALL,WG_TASK_ID=run-post-patch,TASK_ID=run-post-patch,SMOKE_NAME=2n-mpi-dense-postpatch,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch,ASYNC_QUORUM_TRANSPORT=mpi-dense,HUMAN_APPROVAL_RECORD='WG run-post-patch: 2-node postpatch MPI dense debug smoke after minimal mpi4py 2-node Init_thread/sendrecv passed; debug QOS; run-local latest only; no production latest mutation authorized.' scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
   ```

3. Do not submit 8n, 64n, or 256n until the 2n train.py retry passes wrapper
   validation.

The specific candidate change tested here:

```bash
--network=disable_rdzv_get
FI_PROVIDER=cxi
MPICH_OFI_NIC_POLICY=NUMA
FI_CXI_RDZV_PROTO=alt_read
FI_CXI_RX_MATCH_MODE=hybrid
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
```

is **not** sufficient and should not be treated as the fix.

## Validation Checklist

- Diagnostic sbatch/srun command, job id, QOS/walltime/node-hours, and
  environment exports are recorded above.
- Minimal 2-node `mpi4py` `MPI.Init_thread` reproduces the OFI `fi_getinfo`
  failure across two nodes.
- No fixed train.py retry environment was found; the report gives the exact
  gated retry recipe and explicitly marks the tested candidate as insufficient.
- No 8n, 64n, or 256n follow-on job was submitted.
