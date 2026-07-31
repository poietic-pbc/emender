# Native local resilient data plane v1 validation

Date: 2026-07-18

Task: `implement-native-local-dataplane-v1`

Authority: [Resilient DiLoCo Compute Pool v1](../../docs/RESILIENT_DILOCO_COMPUTE_POOL.md)
and [Native resilient DiLoCo data plane v1](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md)

This record covers the compiled node-local core and Python bridge only. The
parallel transport task owns libfabric/CXI endpoint, wire, credit, owner, and
service files under `native/dataplane`; this branch adds no network class and
no collective. Python continues to own leases, READY membership, `Q_min`,
`T_min`, bounded generation closure, checkpoint policy, and atomic durable
publication.

## Conformance checklist

- **R04 / NDP06:** every client, control command, event, operation, and result
  retains the 128-bit run/worker/incarnation keys and exact fence, generation,
  attempt, owner epoch, and submission sequence. A newer controller fence
  cancels volatile old state; old commands reject with `NDP_EFENCE`.
- **R05, R15 / NDP05:** sources are reduced in raw trainer-key order. Native
  code converts f32, f64, or bfloat16 to f64, uses exact positive integer
  weights, `FE_TONEAREST`, `-fno-fast-math`, `-ffp-contract=off`, finite checks,
  and one f32 projection. All three dtype policies match the Python v1
  reference bit-for-bit across unequal weights and arrival permutations.
- **R08 / NDP08-NDP10:** at most 64 local buffers and a configurable shared-byte
  limit are enforced before admission. Sealed memfds, SHA-256, identity-bound
  receipts, idempotent duplicate replay, conflicting-reuse rejection, and
  explicit operation/buffer release are tested. Mapping high-water is one
  trainer buffer, not the full local cohort.
- **R09 / NDP01:** the library is model-free and owns no lease, membership,
  quorum, optimizer, checkpoint, scheduler, or training policy.
- **R10 / NDP04, NDP12, NDP14:** trainers produce directly into service-allocated
  memfds; Python marshals metadata only. The result is one sealed shared f32
  memfd with read-only fd duplicates. Metrics prove zero Python dense-socket,
  handoff-copy, and trainer-spool bytes. `nm -D` finds no MPI/PMPI symbol.
- **R14 / NDP15-NDP16:** every call uses an absolute deadline; events and metrics
  expose state, fenced identity, logical bytes, rejects, high-water, projection,
  replay, and release. Commit is accepted only after a result view exists;
  native code never publishes `latest`.
- **Failure/recovery:** tests cover exception cleanup, process-restart-unique
  handles, stale fence/incarnation, checksum corruption, nonfinite data,
  duplicate/conflicting replay, cancellation, slot/byte exhaustion, and the
  explicitly enabled one-journal fallback. The minimum progress floor remains
  the existing Python control plane and is unchanged.
- **No rank invariant or central broker:** the local core contains no launched
  rank/world-size input, collective, model, network owner map, or full-model
  broker. It consumes only the control plane's explicit local/frozen identity.

## Builds and exact tests

Portable GNU build:

```bash
cmake -S src/native_resilient_dataplane \
  -B build/native-dataplane-portable \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DNDP_ENABLE_XPMEM=OFF
cmake --build build/native-dataplane-portable --parallel
ctest --test-dir build/native-dataplane-portable --output-on-failure
```

Result: GNU C++ 7.5.0, CTest 1/1 passed. Final binary SHA-256:
`b514c0c28c3afed3205d571e6bca37c436fa7a6769e5056482f3ed7dadb90ddc`.

Supported Frontier programming environment build:

```bash
cmake -S src/native_resilient_dataplane \
  -B build/native-dataplane-frontier \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DNDP_ENABLE_XPMEM=ON
cmake --build build/native-dataplane-frontier --parallel
ctest --test-dir build/native-dataplane-frontier --output-on-failure
```

Result: Cray clang/C++ 18.0.1 under PrgEnv-cray 8.6.0, CTest 1/1 passed.
Final binary SHA-256:
`8a591eabc1f86a5bd8b44d8a244970971dd581f7ba6ede1b41cb9d77a2aea8d3`.

The same bridge/failure suite was run against each binary:

```bash
EMENDER_NDP_LIBRARY=$PWD/build/<compiler>/libemender_ndp.so \
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -q \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py
```

Result for each compiler: **14 passed**. Cases include unequal weights
`(3, 1000003, 29)`, two changing arrival orders, all three source dtypes,
duplicate/conflict, stale fence, corruption, nonfinite, cancellation, 64-slot
exhaustion, byte exhaustion, exception cleanup, and two-process restart.

The authoritative existing resilient-pool set remains green:

```bash
python -m pytest -q \
  tests/test_resilient_e97_reducer.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_pool_runtime.py
```

Result: **21 passed in 24.69 seconds**. A wider local run collected 121
resilient/fencing tests and reached 120 passes; the sole failure was the
pre-existing `test_real_stuck_node_process_is_killed_while_quorum_continues`
five-second multiprocessing termination assertion. That exact test and the
other timing failure observed during an earlier loaded run both pass in an
isolated rerun (**2 passed in 28.32 seconds**). The failing test imports none of
the new implementation.

## Measured bounds

The retained machine-readable record is
[`implement-native-local-dataplane-v1-metrics.json`](implement-native-local-dataplane-v1-metrics.json).
For eight f32 trainer mappings of 128 elements each:

| Metric | Default | Explicit fallback |
|---|---:|---:|
| Shared-memory high-water | 4,096 B | 4,096 B |
| Per-lane mapped high-water | 512 B | 512 B |
| Prompt source release | 4,096 B | 4,096 B |
| One shared f32 result | 512 B | 512 B |
| Admitted / released after completion | 4,608 / 4,608 B | 4,608 / 4,608 B |
| Projection count | 1 | 1 |
| Trainer spool files / bytes | 0 / 0 | 0 / 0 |
| Disk replay files / bytes | 0 / 0 | 1 / 1,120 B |
| Python dense socket / handoff-copy bytes | 0 / 0 | 0 / 0 |
| Post-generation shared bytes / files | 0 / 0 | 0 / 0 |

The explicit fallback is the 1,024-byte f64 node numerator plus a 64-byte
header and 32-byte checksum. It is one file, not eight trainer streams; it is
below `layout_bytes + 1 MiB` and is removed at commit. Default steady state
writes no disk payload.

## Submission statement

No `sbatch`, `srun`, `scancel`, or other Slurm submission/control command was
run by this task. All validation was allocation-free and local. No Frontier
job result or G1/G2 transport qualification is claimed.
