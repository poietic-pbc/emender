# Native resilient data plane: node-local v1 core

This directory implements the node-local half of
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md):
the stable C ABI, bounded memfd ownership, deterministic native reduction, one
shared f32 apply view, prompt release, and the optional single reduced-node
replay journal. `libemender_ndp.so.1` is now a metadata-only RPC client. The
authoritative `LocalServiceCore` is linked only into `ndp_cxi_service`, beside
the persistent `FabricEndpoint`; client processes never construct a local
dense-state singleton. The core is model-free and contains no lease,
membership, quorum, checkpoint, Slurm, MPI, or all-rank policy.

The transport sibling intentionally lives in `native/dataplane`. The single
service executable links both ownership domains for one process lifetime while
preserving their versioned C ABIs.

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

1. The service binds a mode-`0600` `AF_UNIX/SOCK_SEQPACKET` socket, validates
   `SO_PEERCRED`, the 256-bit admission token, and run/fence identity, and caps
   every RPC packet at 64 KiB. Every packet carries run, fence, generation,
   attempt, incarnation, sequence, extent, layout, and metadata digest fields.
2. The controller installs a sealed canonical layout descriptor and a fenced
   generation identity.
3. A trainer requests a sealed-size memfd, maps it writable, produces directly
   into the final mapping, closes the mapping, and seals it.
4. `ndp_submit_local_v1` validates the exact identity, byte bounds, SHA-256,
   dtype, finiteness, weight, and deadline. Identical replay receives an
   idempotent operation; conflicting reuse rejects.
5. `FREEZE` reduces retained sources in raw `trainer_key` order. It maps one
   source at a time, converts each element once to binary64, performs the
   specified checked weighted additions, emits `BUFFER_RELEASED`, and retains
   one node numerator.
6. `FINALIZE_OWNERS` divides once in binary64 and projects exactly once into a
   single sealed f32 memfd. Clients receive an `O_RDONLY|O_CLOEXEC` descriptor
   for those same pages using `SCM_RIGHTS`; there is no per-trainer aggregate
   file or service copy.
7. Every buffer, view, and operation is released explicitly. Python context
   managers close them safely on exceptions and stale-fence supersession.
   Handles contain a service-boot cookie and cannot be reused after restart.
   A controller or trainer disconnect does not destroy an admitted attempt;
   explicit abort/drain or a newer fence does.

The default replay mode is memory-only and writes zero disk bytes. Setting
`EMENDER_NDP_FALLBACK_SPOOL_DIR` explicitly materializes exactly one
checksummed float64 node numerator. The native path rejects Lustre, NFS, home,
and autofs paths; the journal is bounded by `layout_bytes + 1 MiB` and removed
on commit, abort, or fence change.

## Python bridge

`ndm.native_dataplane.Client` loads the RPC client ABI through typed `ctypes`
calls. `EMENDER_NDP_SOCKET` and `EMENDER_NDP_ADMISSION_TOKEN_HEX` identify the
already-running service selected by the post-lease supervisor. Production
launch passes the 32-byte token to `ndp_cxi_service --admission-token-fd FD`;
the hex command-line form is test-only so the token is not exposed in the
process list.
`Buffer.mapped(...)` exposes NumPy/`torch.frombuffer`-compatible pages directly;
only fixed-size metadata is marshalled by Python. `ResultView` is read-only and
reports the full `(run_key, fence, generation, attempt, layout, base, root,
global_weight)` result identity. `Client.metrics` reports shared/mapped
high-water, admitted/released bytes, projection count, rejects, buffer
exhaustion, and the required zero-copy/spool counters.

## Local validation

```bash
source scripts/frontier/activate_emender_frontier.sh
EMENDER_NDP_LIBRARY=$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1 \
EMENDER_NDP_SERVICE=$PWD/build/native-resilient-dataplane/bin/ndp_cxi_service \
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py
```

The suite covers unequal weights, arrival permutations, f32/f64/bfloat16
source policy, duplicate and conflicting replay, stale fences/incarnations,
checksum corruption, nonfinite input, cancellation, slot/byte exhaustion,
exception cleanup, restart-unique handles, one shared result, default zero
disk writes, and the bounded fallback journal.

Conformance scope: Compute Pool v1 R04, R05, R08, R09, R10, R14, R15 and
native v1 NDP01, NDP04-NDP06, NDP08-NDP10, NDP12, NDP14-NDP16. Python remains
the sole owner of READY membership, minimum progress, bounded generation
closure, and atomic durable publication.
