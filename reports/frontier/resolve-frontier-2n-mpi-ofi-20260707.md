# Resolve Frontier 2n MPI/OFI transport, 2026-07-07

Task: `resolve-frontier-2n`

## Verdict

The 2-node Frontier transport substrate itself is usable with Cray MPICH C:
job `4953725` passed a compiled `cc` `MPI_Init_thread(MPI_THREAD_SERIALIZED)`
plus ring `MPI_Sendrecv` diagnostic across 16 ranks on 2 nodes.  The same
Slurm shape still fails through Cray's bundled `mpi4py` before user-level
communication, including after forcing mpi4py's requested thread level to
`serialized`.

Conclusion: the current Python `mpi4py` binding path is unsuitable for the
dense async DiLoCo 2-node data plane on Frontier.  The next transport layer
should be a compiled Cray MPICH helper/extension linked by `cc`/`CC`, with
Python using the helper as a subprocess or native extension boundary.  I did
not submit the gated 2-node train.py retry because the required minimal
2-node mpi4py gate did not pass.  No 8n, 64n, or 256n follow-on job was
submitted from this task.

## Code/artifacts added

- `scripts/frontier/cray_mpich_ofi_2n_diag.sbatch`
  - Builds a C diagnostic with Frontier `cc`.
  - Runs the same 2-node/16-rank/1-GPU-per-rank Slurm shape as train.py dense.
  - Captures modules, environment, exact `srun` commands, per-case logs, and
    status files under the debug run root.
- `scripts/frontier/mpi4py_ofi_2n_diag.sbatch`
  - Updated to configure `mpi4py.rc.thread_level=serialized` before importing
    `mpi4py.MPI`, then query the provided thread level instead of calling
    `MPI.Init_thread` after import.
- `ndm/async_diloco_mpi.py`
  - Updated the train.py dense import path to request mpi4py serialized thread
    support before importing `MPI`.  This was a reasonable candidate because
    the compiled C diagnostic passes with `MPI_THREAD_SERIALIZED`, but the
    patched mpi4py Slurm retry still fails.

## Shared Slurm shape

All diagnostics used the same resource shape as the failed 2-node train.py MPI
dense rung:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest ...
```

The debug QOS requests were:

- Account/partition/QOS: `bif148` / `batch` / `debug`
- Nodes: `2`
- Walltime: `00:10:00`
- Requested node-hours: `0.333333`
- Tasks per node: `8`
- CPUs per task: `7`
- GPUs per task: `1`

## Module/runtime stack

The C and mpi4py diagnostics both used:

```bash
source scripts/frontier/frontier_runtime_env.sh
frontier_load_default_modules
```

Captured module stack from job `4953725` and `4953735`:

```text
craype-x86-trento
libfabric/2.3.1
craype-network-ofi
xpmem/1.0.1-1.5_1_gfb6998056825
Core/25.03
DefApps
cray-dsmml/0.3.0
PrgEnv-gnu/8.7.0
gcc-native/14.2
cray-libsci/26.03.0
cray-mpich/9.1.0
cray-pmi/6.1.17
craype/2.7.36
perftools-base/26.03.0
cpe/26.03
darshan-runtime/3.4.7-mpi
miniforge3/23.11.0-0
rocm/7.1.1
craype-accel-amd-gfx90a
```

Common captured communication environment:

```text
MPICH_GPU_SUPPORT_ENABLED=0
CONDA_PREFIX=/autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0
LD_LIBRARY_PATH includes:
  /opt/cray/pe/pmi/6.1.17/lib
  /opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3/lib
  /opt/cray/pe/mpich/9.1.0/gtl/lib
  /opt/cray/libfabric/2.3.1/lib64
```

The mpi4py path used Cray's bundled mpi4py:

```text
CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages
mpi4py_file=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages/mpi4py/__init__.py
mpi4py_version=3.1.4
mpi4py_extensions=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages/mpi4py/MPI.cpython-310-x86_64-linux-gnu.so
```

## Job 4953725: C Cray MPICH diagnostic

Exact submit command:

```bash
sbatch --parsable --export=ALL,WG_TASK_ID=resolve-frontier-2n scripts/frontier/cray_mpich_ofi_2n_diag.sbatch
```

Slurm accounting:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NodeList
4953725|cray-mpich-ofi-2n-diag|batch|debug|COMPLETED|0:0|00:00:28|2|frontier[08247-08248]
4953725.batch|batch|||COMPLETED|0:0|00:00:28|1|frontier08247
4953725.extern|extern|||COMPLETED|0:0|00:00:28|2|frontier[08247-08248]
4953725.0|env|||COMPLETED|0:0|00:00:02|2|frontier[08247-08248]
4953725.1|env|||COMPLETED|0:0|00:00:01|2|frontier[08247-08248]
4953725.2|env|||COMPLETED|0:0|00:00:01|2|frontier[08247-08248]
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/cray_mpich_ofi_diag/20260707/4953725-20260707T212132Z
```

Exact diagnostic commands recorded by the wrapper:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest env MPICH_GPU_SUPPORT_ENABLED=0 /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/cray_mpich_ofi_diag/20260707/4953725-20260707T212132Z/artifacts/cray_mpich_init_sendrecv
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest --mpi=pmi2 env MPICH_GPU_SUPPORT_ENABLED=0 /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/cray_mpich_ofi_diag/20260707/4953725-20260707T212132Z/artifacts/cray_mpich_init_sendrecv
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest --mpi=pmix env MPICH_GPU_SUPPORT_ENABLED=0 /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/cray_mpich_ofi_diag/20260707/4953725-20260707T212132Z/artifacts/cray_mpich_init_sendrecv
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest env MPICH_GPU_SUPPORT_ENABLED=0 FI_PROVIDER=cxi MPICH_OFI_NIC_POLICY=NUMA FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/cray_mpich_ofi_diag/20260707/4953725-20260707T212132Z/artifacts/cray_mpich_init_sendrecv
```

Results:

- Baseline/default `srun`: `0`, valid MPI world.
  - Rank-0 JSON:
    `{"mpi_init_thread_provided":2,"world_size":16,"unique_hosts":2,"rank0_received_from":15,"mpich_gpu_support_enabled":"0","fi_provider":"","mpich_ofi_nic_policy":"NUMA"}`
- `--mpi=pmi2`: process exit `0`, but invalid for this workload because each
  task initialized as `world_size=1`; do not use it for dense DiLoCo.
- `--mpi=pmix`: failed immediately:
  `Invalid MPI type 'pmix', --mpi=list for acceptable types`.
- Explicit `FI_PROVIDER=cxi` / `MPICH_OFI_NIC_POLICY=NUMA`: `0`, valid MPI
  world.
  - Rank-0 JSON:
    `{"mpi_init_thread_provided":2,"world_size":16,"unique_hosts":2,"rank0_received_from":15,"mpich_gpu_support_enabled":"0","fi_provider":"cxi","mpich_ofi_nic_policy":"NUMA"}`

This proves the 2-node Cray MPICH/Slurm/OFI substrate can initialize and
exchange user messages under the train.py Slurm shape when the client is a
compiled `cc` binary.

## Job 4953735: patched mpi4py diagnostic

Patch tested:

```python
import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.threads = True
mpi4py.rc.thread_level = os.environ.get("MPI4PY_RC_THREAD_LEVEL", "serialized")
from mpi4py import MPI
provided = MPI.Query_thread()
```

Exact submit command:

```bash
sbatch --parsable --export=ALL,WG_TASK_ID=resolve-frontier-2n scripts/frontier/mpi4py_ofi_2n_diag.sbatch
```

Slurm accounting:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NodeList
4953735|mpi4py-ofi-2n-diag|batch|debug|FAILED|255:0|00:00:26|2|frontier[07258,07272]
4953735.batch|batch|||FAILED|255:0|00:00:26|1|frontier07258
4953735.extern|extern|||COMPLETED|0:0|00:00:26|2|frontier[07258,07272]
4953735.0|env|||FAILED|255:0|00:00:03|2|frontier[07258,07272]
4953735.1|env|||FAILED|255:0|00:00:00|2|frontier[07258,07272]
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953735-20260707T212420Z
```

Exact diagnostic commands recorded by the wrapper:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest env MPICH_GPU_SUPPORT_ENABLED=0 CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953735-20260707T212420Z/artifacts/mpi4py_init_sendrecv.py
srun -N 2 -n 16 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest env MPICH_GPU_SUPPORT_ENABLED=0 CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages FI_PROVIDER=cxi MPICH_OFI_NIC_POLICY=NUMA FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/mpi4py_ofi_diag/20260707/4953735-20260707T212420Z/artifacts/mpi4py_init_sendrecv.py
```

Environment exports:

```text
MPICH_GPU_SUPPORT_ENABLED=0
CRAY_MPI4PY_SITE=/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages
MPI4PY_RC_THREAD_LEVEL defaulted to serialized

# second step only
FI_PROVIDER=cxi
MPICH_OFI_NIC_POLICY=NUMA
FI_CXI_RX_MATCH_MODE=hybrid
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
```

Result:

- Baseline status: `255`
- Explicit CXI/NUMA status: `255`
- Minimal mpi4py never reached ring `sendrecv`.
- Failure signature is unchanged:

```text
Fatal error in PMPI_Init_thread
MPIDI_OFI_mpi_init_hook
open_fabric(1559): OFI fi_getinfo() failed (ofi_init.c:1559:open_fabric:No data available)
```

Prior mpi4py diagnostics `4953690` and `4953693` tested baseline, explicit
`FI_PROVIDER=cxi`, `MPICH_OFI_NIC_POLICY=NUMA`, `--network=disable_rdzv_get`,
and `FI_CXI_RDZV_PROTO=alt_read`.  Job `4953735` adds the serialized
mpi4py-thread-level candidate.  None produced a passing 2-node mpi4py command.

## Launcher/module finding

- Default `srun` is the correct launcher substrate for compiled Cray MPICH.
- `--mpi=pmix` is not available in this Slurm configuration.
- `--mpi=pmi2` is not acceptable for this workload: the C program exits `0`
  but each process reports singleton `MPI_COMM_WORLD` (`world_size=1`), so it
  is not a valid multi-rank transport.
- The module stack is sufficient for a compiled `cc` Cray MPICH binary.
- The remaining failure is specific to Python/mpi4py initialization with this
  Cray mpi4py/Python runtime, not a general 2-node OFI failure.

## Selected next transport layer

Use a compiled Cray MPICH helper/extension as the dense data plane, not Python
`mpi4py`.

Recommended design:

1. Keep Python responsible for train.py local steps, state-dict delta packing,
   checksum metadata, and metrics formatting.
2. Move the MPI process initialization and point-to-point transport into a
   `cc`/`CC` built helper linked by Cray MPICH.
3. Use one of two integration modes:
   - Native extension: expose `init`, `rank`, `size`, `send_bucket`,
     `recv_bucket`, `barrier`, and `finalize` through a small CPython or
     pybind11 wrapper.  This is the cleanest long-term path but must ensure MPI
     is initialized exactly once before Python imports conflicting MPI symbols.
   - Subprocess helper: each Slurm rank launches the compiled helper as the MPI
     process and communicates with rank-local Python over stdin/stdout or Unix
     domain socket using length-prefixed bucket frames.  This has more plumbing
     but isolates MPI initialization from Python and matches the evidence from
     job `4953725`.
4. Preserve the existing dense wire format:
   - JSON header with schema, rank, generation, tensor metadata, bucket checksums.
   - Raw bucket bytes as `MPI_BYTE`.
   - Root-rank quorum and merge semantics unchanged.
5. Add a permanent two-step gate before train.py scale-out:
   - `scripts/frontier/cray_mpich_ofi_2n_diag.sbatch` must pass with
     `world_size=16` and `unique_hosts=2`.
   - The compiled helper transport must pass a 2-node synthetic dense bucket
     exchange before train.py is retried.

## Train.py retry status

No 2-node train.py MPI dense retry was submitted from this task because the
minimal mpi4py transport gate failed in job `4953735`.  Retrying train.py on
the existing Python mpi4py data plane would reproduce the same
`PMPI_Init_thread` / `OFI fi_getinfo` failure before producing metrics.

The next train.py retry should wait for a passing compiled-helper 2-node bucket
exchange, then submit only the 2-node debug QOS rung.  Do not submit 8n, 64n,
or 256n until that 2-node train.py rung passes.

## Validation checklist

- Exact diagnostic commands, job IDs, QOS/walltime/node-hours, modules, and
  environment exports are recorded above.
- 2-node C Cray MPICH `Init_thread`/`sendrecv` passes under the same Slurm
  shape: job `4953725`, default `srun`, `world_size=16`, `unique_hosts=2`.
- 2-node mpi4py cannot be fixed by the tested launcher/module/env changes:
  baseline, explicit CXI/NUMA, RDZV candidate from prior jobs, and serialized
  mpi4py thread-level all fail with `OFI fi_getinfo`.
- Because C passes and mpi4py fails, the selected design is a compiled
  Cray MPICH helper/extension for dense buckets.
- No train.py retry was submitted because no mpi4py fix was found.
- No 8n, 64n, or 256n follow-on job was submitted from this task.
