# Async DiLoCo Frontier Transport Prototype

Task: `async-diloco-frontier-transport-prototype`
Date: 2026-07-05

## Scope

This is standalone transport plumbing for async DiLoCo x/z deltas. It does not
wire into training, does not submit production jobs, and does not change
checkpoint or `latest` pointers.

Dense update bytes are explicitly kept off Lustre. Lustre is used only for
source checkout, Slurm stdout/stderr, environment captures, and small JSON or
Markdown summaries. The benchmark allocates x-like and z-like payload buffers
and moves the dense bytes through Cray MPICH point-to-point calls.

## Added Artifacts

- `scripts/frontier/async_diloco_transport_bench.cpp`
  - C++17 MPI point-to-point benchmark.
  - Modes:
    - `pair`: rank pairs exchange x/z chunks.
    - `fanin`: ranks 1..N-1 send x/z chunks to rank 0, matching a prototype
      worker-to-merger data path.
  - Devices:
    - `cpu`: host buffers.
    - `hip`: HIP device buffers when compiled with `-DUSE_HIP`; requires
      GPU-aware MPI for real device-buffer transport.
  - Emits machine-readable JSON containing payload size, chunk size, latency,
    effective bandwidth, staging behavior, MPI GPU env state, and failure
    modes.

- `scripts/frontier/async_diloco_transport_bench.sbatch`
  - Debug-QOS Frontier wrapper.
  - Default size is one node for ten minutes: `#SBATCH -N 1`, `#SBATCH -q debug`,
    `#SBATCH -t 00:10:00`, eight tasks per node, one GPU per task.
  - Builds the benchmark in the run artifact directory and runs it with `srun`.
  - Captures modules, environment, command, Slurm paths, approximate node-hours,
    and JSON/Markdown summaries.

- `reports/frontier/async-diloco-transport-local-metrics-20260705.json`
  - Local single-rank validation metrics from this task.

## Frontier Modules And Environment

The debug wrapper sources `scripts/frontier/frontier_runtime_env.sh` and uses
the same module stack documented by the runtime plumbing task:

```bash
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
```

The transport-specific GPU-aware MPI environment is:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export OMP_NUM_THREADS=7
```

The default HIP build command on Frontier is:

```bash
CC -O3 -std=c++17 -x hip --offload-arch=gfx90a -DUSE_HIP \
  scripts/frontier/async_diloco_transport_bench.cpp \
  -o <artifact_dir>/async_diloco_transport_bench
```

For CPU-only local validation, use:

```bash
mpicxx -O3 -std=c++17 \
  scripts/frontier/async_diloco_transport_bench.cpp \
  -o /tmp/async_diloco_transport_bench
```

## Launch Recipe

Small single-node debug run only:

```bash
mkdir -p logs/frontier/async_diloco_transport
WG_TASK_ID=async-diloco-frontier-transport-prototype \
BENCH_DEVICE=hip \
BENCH_MODE=fanin \
BENCH_PAYLOAD_MIB=256 \
BENCH_CHUNK_MIB=16 \
BENCH_ITERS=20 \
BENCH_WARMUP=5 \
sbatch scripts/frontier/async_diloco_transport_bench.sbatch
```

Two-node debug scaling is intentionally not the default. If run later, keep it
debug-scale and WG-tracked, for example:

```bash
WG_TASK_ID=async-diloco-frontier-transport-prototype \
BENCH_DEVICE=hip \
BENCH_MODE=fanin \
BENCH_PAYLOAD_MIB=256 \
BENCH_CHUNK_MIB=16 \
BENCH_ITERS=20 \
BENCH_WARMUP=5 \
sbatch -N 2 -t 00:10:00 scripts/frontier/async_diloco_transport_bench.sbatch
```

## Local Validation

Local validation was run before any Frontier multi-node benchmark. No Frontier
Slurm job was submitted by this task.

Build:

```bash
mpicxx -O3 -std=c++17 scripts/frontier/async_diloco_transport_bench.cpp \
  -o /tmp/async_diloco_transport_bench
```

Run:

```bash
/tmp/async_diloco_transport_bench --mode pair --device cpu \
  --payload-mib 1 --chunk-mib 1 --iters 3 --warmup 1 \
  --metrics reports/frontier/async-diloco-transport-local-metrics-20260705.json
```

The single-rank run validates argument parsing, allocation, chunk walking,
metrics emission, and the no-Lustre payload contract. It cannot validate
network injection or GPU-aware MPI; the metrics explicitly record that as a
failure mode.

## Frontier Job Ledger

No Frontier benchmark job was submitted in this task. Debug wrapper fields are
ready to record:

- Slurm job ID.
- Exact command.
- stdout/stderr paths.
- run artifact directory.
- elapsed benchmark latency and effective bandwidth.
- approximate node-hours for the debug allocation.
- pass/no-go conclusion.

## Current No-Go / Next Gate

The prototype is ready for a one-node debug run on Frontier, but the current
pass/no-go conclusion for multi-node transport is `no-go-not-run`: local
single-rank validation passed, while GPU-aware device-buffer bandwidth and
Slingshot latency still require a small Frontier debug submission.

This work does not use `torch.distributed` all-rank collectives as the merge
mechanism. The only global synchronization in the benchmark is MPI timing
barriers and small metadata reduction for failure status; dense x/z payloads use
MPI point-to-point sends and receives.
