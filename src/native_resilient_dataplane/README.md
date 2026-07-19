# Native resilient data plane: node-local v1 core

This directory implements the node-local half of
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md):
the stable client C ABI, bounded seqpacket RPC, service-owned memfd lifetime,
deterministic native reduction, one shared f32 apply view, prompt release, and
the optional single reduced-node replay journal. It is model-free and contains
no lease, membership, quorum, checkpoint, Slurm, MPI, or all-rank policy.

`service_core.cpp` is a linkable static authority instantiated only by
`native/dataplane/src/service_main.cpp`. `libemender_ndp.so.1` contains the
process-local client handle table and the `AF_UNIX/SOCK_SEQPACKET` v1 client;
it never constructs an authoritative `ServiceCore`. The transport sibling in
`native/dataplane` links the same core into `ndp_cxi_service`, so that one
process owns local state and `FabricEndpoint` for the same lifetime.

## Build

Portable compiler:

```bash
cmake -S src/native_resilient_dataplane -B build/native-dataplane \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=g++ \
  -DNDP_ENABLE_XPMEM=OFF
cmake --build build/native-dataplane --parallel
ctest --test-dir build/native-dataplane --output-on-failure
```

Frontier programming environment:

```bash
CXX_COMPILER=CC NDP_ENABLE_XPMEM=ON \
  scripts/frontier/build_native_local_dataplane.sh
```

The reduction translation unit is compiled as C++17 with `-fno-fast-math`,
`-ffp-contract=off`, and `-frounding-math`. The installed SONAME is
`libemender_ndp.so.1`; the installed header is `include/emender/ndp.h`.

## Ownership and lifecycle

1. The controller authenticates to the mode-0600 service socket using
   `SO_PEERCRED`, run, admission-token, and fence identity, then installs a
   sealed canonical layout descriptor and a fenced generation identity.
2. A trainer requests a sealed-size memfd, maps it writable, produces directly
   into the final mapping, closes the mapping, and seals it.
3. `ndp_submit_local_v1` validates the exact identity, byte bounds, SHA-256,
   dtype, finiteness, weight, and deadline. Identical replay receives an
   idempotent operation; conflicting reuse rejects.
4. `FREEZE` reduces retained sources in raw `trainer_key` order. It maps one
   source at a time, converts each element once to binary64, performs the
   specified checked weighted additions, emits `BUFFER_RELEASED`, and retains
   one node numerator.
5. `FINALIZE_OWNERS` divides once in binary64 and projects exactly once into a
   single sealed f32 memfd. Any number of local trainers receive read-only fd
   duplicates of those same pages; there is no per-trainer aggregate file or
   service copy.
6. Every buffer, view, and operation is released explicitly. Python context
   managers close them safely on exceptions and stale-fence supersession.
   Public handles contain a client-process boot cookie and translate to
   service-private handles only inside bounded RPC records. Service restart or
   a newer fence invalidates both domains deterministically.

The default replay mode is memory-only and writes zero disk bytes. Setting
`EMENDER_NDP_FALLBACK_SPOOL_DIR` explicitly materializes exactly one
checksummed float64 node numerator. The native path rejects Lustre, NFS, home,
and autofs paths; the journal is bounded by `layout_bytes + 1 MiB` and removed
on commit, abort, or fence change.

## Python bridge

`ndm.native_dataplane.Client` loads the ABI through typed `ctypes` calls.
`Buffer.mapped(...)` exposes NumPy/`torch.frombuffer`-compatible pages directly;
only fixed-size metadata is marshalled by Python. `ResultView` is read-only and
reports the full `(run_key, fence, generation, attempt, layout, base, root,
global_weight)` result identity. `Client.metrics` reports shared/mapped
high-water, admitted/released bytes, projection count, rejects, buffer
exhaustion, and the required zero-copy/spool counters.

## Local validation

```bash
EMENDER_NDP_LIBRARY=$PWD/build/native-dataplane/libemender_ndp.so \
python -m pytest -q \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py
```

The suite launches the compiled service and covers unequal weights, arrival
permutations, f32/f64/bfloat16
source policy, duplicate and conflicting replay, stale fences/incarnations,
checksum corruption, nonfinite input, cancellation, slot/byte exhaustion,
exception cleanup, controller/trainer disconnect, restart-unique handles,
exact ancillary-fd cardinality, one read-only shared result, default zero disk
writes, and the bounded fallback journal.

Conformance scope: Compute Pool v1 R04, R05, R08, R09, R10, R14, R15 and
native v1 NDP01, NDP04-NDP06, NDP08-NDP10, NDP12, NDP14-NDP16. Python remains
the sole owner of READY membership, minimum progress, bounded generation
closure, and atomic durable publication.
