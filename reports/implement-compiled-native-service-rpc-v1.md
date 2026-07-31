# Compiled persistent native service RPC v1

**Task:** `implement-compiled-native-service-rpc-v1`
**Authority:** Resilient DiLoCo Compute Pool v1 (2026-07-17) and Native
resilient DiLoCo data plane v1 (2026-07-18)
**Result:** implemented and locally qualified; no Slurm job was submitted

## Delivered boundary

`libemender_ndp.so.1` retains the published v1 C ABI, but it is now a thin
`AF_UNIX/SOCK_SEQPACKET` RPC client. It owns only connection bookkeeping needed
to serialize calls; it does not construct or own authoritative generation,
buffer, operation, receipt, or result state. `LocalServiceCore` contains that
state and is linked statically only into `ndp_cxi_service`.

The service executable constructs one `LocalServiceCore`, one
`LocalRpcServer`, and one `FabricEndpoint` and retains them for the daemon
lifetime. The persistent endpoint still enforces the existing exact-CXI
production policy. This task did not remove the live production fail-closed
guard, wire the E97 roles, or submit a Slurm job; those remain the scope of
`wire-native-dataplane-e97-v1`.

The local RPC protocol is version 1.0 and has a fixed 188-byte field-serialized
header. A complete seqpacket is bounded to 65,536 bytes. Each header binds the
opcode, request ID, client handle, run key, fence, generation, attempt, client
incarnation, sequence/owner epoch, byte extent, layout digest, metadata digest,
status, declared payload length, and ancillary-fd count. Payloads are ABI
metadata structs or bounded event arrays. Dense tensor bytes are never present
in a packet.

The socket is a pathname socket with mode `0600`. The server checks
`SO_PEERCRED` against its effective UID, the exact socket path, a constant-time
256-bit admission-token comparison, run identity, fence, and per-connection
incarnation. Production accepts the token through a protected inherited file
descriptor; the command-line hex form is test-only. `recvmsg` rejects packet
or control truncation, unexpected ancillary types, digest mismatch, and any
declared/actual descriptor-cardinality mismatch.

Trainer-created dense input uses one size/write-sealed memfd sent with one
`SCM_RIGHTS` descriptor. Service-allocated direct-producer mappings remain
supported without a return copy because the service already owns the original
memfd. Finalization creates one size/write/seal-sealed result memfd. A result
view sends one new `O_RDONLY|O_CLOEXEC` descriptor for those same pages; a
writable shared mapping is impossible.

Controller and trainer socket disconnects close only their public connection
resources. Retained admitted submissions, receipts, generation state, and a
finalized result stay in the service. A later authenticated controller can
reclaim the result operation. Explicit abort/drain, service shutdown, or a
strictly newer fence invalidates the retained handles. Service-boot cookies
make pre-restart handles invalid in a new daemon.

## Separate-process integration evidence

`ndp_service_rpc_integration_test` executes the installed topology rather than
sharing a core in test threads:

1. The test execs `ndp_cxi_service` using local `tcp;ofi_rxm`; the daemon owns
   both its real `FabricEndpoint` and `LocalServiceCore`.
2. A controller child installs a sealed layout and generation, closes its ABI
   client, and exits.
3. A trainer child first proves unsealed input rejection, then passes exactly
   one sealed dense memfd, submits `(run,fence,generation,attempt,trainer,
   incarnation,sequence,extent,layout,digest,weight)`, closes, and exits.
4. A reconnected trainer receives an idempotent result for an identical replay,
   gets `NDP_ECONFLICT` for a changed-weight replay, and gets
   `NDP_ENONFINITE` for a distinct NaN contribution.
5. A second controller child freezes and finalizes, writes only the service
   operation handle to its parent, closes, and exits.
6. A third controller child obtains the exact result from that service-owned
   operation. It verifies values `[1,2,3,4]`, weight `3`, all fenced result
   identity fields, four required seals, `O_RDONLY`, `CLOEXEC`, and failed
   writable mapping before commit.
7. The test proves wrong-token rejection, exact `SCM_RIGHTS` cardinality,
   oversized-frame rejection, newer-fence rejection of an old buffer handle,
   service-restart rejection of the stale handle, bounded SIGTERM exit, no
   remaining child, and removal of the socket path.

The retained Python ABI/failure/reference tests also cover short sealed extents,
checksum rejection, all supported dtypes, unequal weights and arrival
permutations, byte/slot exhaustion, stale incarnation/fence behavior, result
sharing, fallback-journal bounds, and zero dense Python/spool counters. Their
fixture starts a fresh compiled daemon for each stateful test; it does not
provide a library singleton or production autostart path.

## Compute Pool v1 conformance

| ID | Check |
|---|---|
| R03 | The service admits only explicitly authenticated clients and never observes a launched rank count. The local test uses independently exiting/reconnecting processes; no all-rank invariant or collective exists. |
| R04 | Run/fence/generation/attempt/trainer/incarnation/sequence are carried and validated. Identical replay is idempotent; changed metadata, stale fence, newer fence, and restart handles reject deterministically. |
| R05 | The extracted core preserves the specified sorted-trainer float64 weighted reduction. Exact native/reference tests cover unequal weights/dtypes/order; the process integration verifies the exact weighted result and weight. |
| R08 | RPC packets, ancillary descriptors, client count, local buffers, layouts, events, extents, resident bytes, and memfd seals are bounded before use. Retained dense bytes remain in service-owned memfds, not socket allocations. |
| R09 | The daemon and manager-side client are model-free. Trainers remain the model/optimizer owners; the service owns only data-plane state. |
| R10 | The tested hot path is node-local seqpacket metadata plus memfd pages and native libfabric. Dense bytes are neither Python-serialized nor written to Lustre/TCP; default spool and dense-socket counters remain zero. |
| R14 | Open, submit, control, poll, and drain use absolute/bounded deadlines. SIGTERM stops admission, closes clients and endpoint, joins threads, unlinks the socket, and exits without a rendezvous. |
| R15 | Final math remains bitwise reference-tested, and checkpoint/apply clients receive a fenced, sealed, read-only result view with exact global weight and result identity. |

This boundary does not choose `Q_min`, `T_min`, READY membership, accepted
owners, generation closure, checkpoint cadence, or durable publication. Those
remain Python control-plane policy. The minimum progress floor in this local
single-contribution integration is explicitly one complete contribution with
positive weight; it is test policy, not a production training policy.

## Native v1 conformance

| ID | Check |
|---|---|
| NDP01 | The compiled daemon is the sole owner of local dense handoff/reduction/buffer lifetime while the controller invokes only metadata commands. No dense bytes cross Python or the RPC payload. |
| NDP03 | One service process owns both the local core and persistent `FI_EP_RDM` endpoint. Exact production `cxi` validation and explicit test-provider labeling remain unchanged. |
| NDP04 | The service validates one sealed dense input memfd transferred by `SCM_RIGHTS`; direct allocation adds no handoff copy. Result pages are shared, not rewritten per client. |
| NDP06 | Every RPC request binds run/fence/generation/attempt/incarnation/sequence/extent/layout/metadata digest. Core contributions/results retain their existing full fenced identities. |
| NDP08 | The 64-KiB control bound, exact fd cardinality, 64 local-buffer bound, 16-GiB hard layout bound, extent/seal checks, and resident-byte limit fail closed before mapping/reduction. |
| NDP10 | Packet metadata SHA-256, dense SHA-256, once-only receipts, identical replay, conflicting replay, unsealed/short input, checksum, and nonfinite rejection are exercised without accumulator mutation. |
| NDP11 | Retained submission memfds survive disconnect for replay/finalization; explicit abort/newer fence/shutdown releases them. The existing one-node-numerator optional local replay journal stays bounded and default-disabled. |
| NDP12 | Finalization creates one service-owned aggregate. Another process maps the exact same sealed result memfd read-only; no per-client aggregate copy/file is created. |
| NDP13 | RPC poll is capped at 30 seconds, all data-plane calls carry absolute deadlines, disconnect is route-local, and service shutdown is bounded and collective-free. |
| NDP14 | The SONAME-1 v1 C ABI now connects to a versioned, metadata-only seqpacket daemon. The authoritative singleton has been removed from the client library. |
| NDP15 | A later authenticated controller receives a fenced read-only result descriptor. Native code still does not choose checkpoint policy or advance durable `latest`; commit remains an explicit controller command after publication. |

## Validation commands and results

All Python, CMake, native builds, and CTest commands ran after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Canonical release bundle and attestation:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: configure/build/install succeeded; canonical CTest passed **9/9**;
`build/native-resilient-dataplane/native-artifacts.json` was regenerated by
the attestation command.

Independent normal build:

```bash
cmake -S native -B build/native-rpc-v1 -DCMAKE_BUILD_TYPE=Debug \
  -DNDP_ENABLE_XPMEM=OFF -DNDP_BUILD_TESTS=ON
cmake --build build/native-rpc-v1 --parallel 8
ctest --test-dir build/native-rpc-v1 --output-on-failure
```

Result: **9/9 passed**, including the new separate-process RPC integration.

Independent ASan/UBSan build:

```bash
cmake -S native -B build/native-rpc-v1-asan -DCMAKE_BUILD_TYPE=Debug \
  -DNDP_ENABLE_XPMEM=OFF -DNDP_BUILD_TESTS=ON \
  -DNDP_ENABLE_SANITIZERS=ON
cmake --build build/native-rpc-v1-asan --parallel 8
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/native-rpc-v1-asan --output-on-failure
```

Result: **9/9 passed** with ASan and UBSan halt-on-error. LeakSanitizer's
optional process-exit scan was disabled because the activated Frontier ROCm
runtime retains 61,512 bytes in `libhsa-runtime64.so` in every libfabric
process, including unchanged transport tests; no project allocation appears in
that report. The integration's explicit child/fd/socket lifetime checks pass.

Required Python sweep against the canonical installed bundle:

```bash
export EMENDER_NDP_LIBRARY=$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1
export EMENDER_NDP_TRANSPORT_LIBRARY=$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1
export EMENDER_NDP_SERVICE=$PWD/build/native-resilient-dataplane/bin/ndp_cxi_service
"$EMENDER_PYTHON" -m pytest -q tests/test_native*.py tests/test_resilient*.py
```

Result after refreshing the checksum-linked gap-matrix reference: **144/144
passed**.

Post-test process/socket checks found no surviving task service process or
service socket. No `sbatch`, `srun`, Slurm submission wrapper, real-model
launcher, or 4+ node command was invoked.
