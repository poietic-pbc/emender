# Compiled persistent native service RPC v1

**Task:** `implement-compiled-native-service-rpc-v1`<br>
**Status:** passed<br>
**Implementation source:** `f6446f6c9e9e635a30ec3f9087cd80e3fee05a62`<br>
**Implementation commits:** `9d4c93d6`, `f6446f6c`<br>
**Architecture authority:** `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, conformance checklist and Native data-plane binding; `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md` v1; requirement status is tracked in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.<br>
**Promotion rung:** G0 component boundary only. No Slurm job was submitted.

## Outcome

The local native ABI is now a client of one compiled, persistent service process instead of the owner of an authoritative process-local `Service` singleton. `ndp_cxi_service` constructs a linkable `ServiceCore`, an `AF_UNIX` `SOCK_SEQPACKET` metadata server, and the existing libfabric `FabricEndpoint` in one process and keeps all three alive for the same service lifetime. The service owns admitted clients, generations, buffers, operations, results, retained trainer memfds, and the fabric endpoint. Client libraries retain only process-local public-to-service handle translations needed to preserve the stable v1 C ABI.

The RPC wire contract is version 1 with a fixed 32-byte header, a maximum packet size of 4,096 bytes, monotonically increasing per-connection sequence numbers, and a maximum of 16 returned events per poll. Dense bytes are never copied into a packet. A dense input crosses the process boundary as exactly one `SCM_RIGHTS` descriptor for a producer memfd, after which the service validates the exact extent, required write/grow/shrink seals, layout identity, and SHA-256 source digest. A result crosses back as exactly one descriptor reopened `O_RDONLY`; the integration test confirms that `pread` returns the exact result and `pwrite` fails with `EBADF`.

The listener is mode `0600`, enables `SO_PASSCRED`, obtains `SO_PEERCRED` from every accepted connection, and admits only the service UID. The first frame must additionally carry the exact configured socket path, run key, constant-time-compared admission token, and a current-or-newer fence. Frames with the wrong magic/version/length/descriptor count, truncation, ancillary truncation, an oversized body, unexpected descriptors, replayed/out-of-order sequence, or an opcode-specific body/cardinality mismatch fail closed. The service caps live connections at 64 and reaps completed connection threads.

Disconnect now releases only the connection/client view; it does not abort admitted generation state. A same-fence controller can reconnect and resume freeze/finalize ownership. An admitted trainer may exit without releasing its local handles, while the service retains its sealed input reference through exact reduction. A newer authenticated fence aborts the prior generation and returns `NDP_EFENCE` to stale clients. Service shutdown closes the listener and connections, joins all connection threads, unlinks the socket, and causes an old client channel to return stable `NDP_ESHUTDOWN`. A restarted service has a new private handle domain and rejects a stale public buffer handle.

The production provider policy is unchanged and continues to fail closed: production still requires exact `cxi`; `tcp;ofi_rxm` is accepted only under explicit `--test-only`. This task does not route real E97 roles into the component, remove the production wiring guard, infer READY membership or close policy inside C++, publish checkpoints, or promote beyond G0. Those are deliberately left to `wire-native-dataplane-e97-v1` and its later Frontier gates.

## Multiprocess regression

`ndp_service_rpc_integration_test` is a black-box process test, not a thread test linked to a shared registry. It creates a private socket directory, forks and `exec`s the built `ndp_cxi_service`, and drives it through the installed public client library. Its first service/socket assertion is failing-first against the pre-RPC service, which had neither the serve arguments nor a seqpacket listener.

The test performs the following sequence:

1. It rejects a wrong admission token, a 4,097-byte packet, and a forged metadata request carrying two descriptors.
2. A controller process installs the sealed layout descriptor, installs generation `(run, fence=1, generation=7, attempt=2, owner_epoch=3)`, then disconnects.
3. A trainer child passes exactly one sealed f32 memfd with exact extent/layout/source digest and identity `(worker, incarnation, sequence=9)`, submits weight 3, and exits without releasing its buffer or operation.
4. A nonfinite child is rejected without mutating the accumulator. An identical reconnect/replay is idempotent; reuse of the contribution identity with conflicting weight is rejected.
5. A new controller client freezes and finalizes the retained generation. It receives one `O_RDONLY` result memfd, exact values `{1,3,5,7}`, global weight 3, and exact fence/generation/attempt/layout identity, then commits.
6. A fence-2 controller invalidates the old controller. The test then stops and reaps the service, verifies that the old channel reports shutdown, restarts at fence 3, and verifies that a stale handle is invalid in the new service domain.
7. Both service children are reaped. Each orderly shutdown removes its socket, and the successful test removes its private directory. A post-suite audit found no live `ndp_cxi_service` process and no remaining RPC/Python service socket.

Existing native Python ABI, reference, failure, manager-session, and pool-integration fixtures now launch the compiled service explicitly and pass its socket path and admission token. They do not instantiate a Python dense listener. The production Python handoff/runtime code remains fail-closed until the downstream wiring task replaces that live path.

## Validation

All build and test commands were run after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The clean canonical build and attestation command was:

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
BUILD_DIR="$PWD/build/native-resilient-dataplane-rpc-v1-build" \
INSTALL_DIR="$PWD/build/native-resilient-dataplane" \
BUILD_JOBS=8 \
scripts/frontier/build_native_resilient_dataplane.sh
```

The first attempt to reuse the older canonical build directory encountered a concrete stale-toolchain link failure at `__cray_dset_detect` after the Frontier module stack changed. Reconfiguring in the clean build directory above succeeded. The retained build manifest reports:

- schema `emender-native-dataplane-build-v1`;
- ABI `65536` and protocol `1.0`;
- source `f6446f6c9e9e635a30ec3f9087cd80e3fee05a62`, clean;
- bundle SHA-256 `edd5b1ed5b74a4fde3122f3e22ddce81afb9df1bea31939dcf3fae798c5a7ca4`;
- client SHA-256 `6e94a888be855116ecd8326dc5b46c08d2fa84697bc0320b64605c560ae3a360`;
- service SHA-256 `3d47d930d08f61657f10ec59539d6c20fed5eaa00302030334028a52dcb3c6b8`.

Normal tests:

```bash
ctest --test-dir build/native-resilient-dataplane-rpc-v1-build \
  --output-on-failure
```

Result: **9/9 passed**, including the new separate-process service RPC integration test.

Sanitizer build and tests:

```bash
cmake -S native -B build/compiled-native-service-rpc-v1-asan \
  -DCMAKE_BUILD_TYPE=Debug -DNDP_ENABLE_XPMEM=OFF \
  -DNDP_BUILD_TESTS=ON -DNDP_ENABLE_SANITIZERS=ON
cmake --build build/compiled-native-service-rpc-v1-asan --parallel 8
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build/compiled-native-service-rpc-v1-asan \
  --output-on-failure
```

Result: **9/9 passed** with AddressSanitizer and UndefinedBehaviorSanitizer active. LeakSanitizer alone was disabled because the initial run reported the same process-exit allocation in unchanged transport tests and the new service test: 61,512 bytes originating in Frontier ROCm `libhsa-runtime64.so.1` while libfabric initializes. There was no application stack frame in that report. Address and undefined-behavior failures remained fatal in the passing run.

Python regressions against the canonical installed client and service:

```bash
export EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1"
export EMENDER_NDP_SERVICE="$PWD/build/native-resilient-dataplane/bin/ndp_cxi_service"
"$EMENDER_PYTHON" -m pytest -q tests/test_native*.py
"$EMENDER_PYTHON" -m pytest -q tests/test_resilient*.py
```

Results: **36/36 native tests passed** in 54.83 seconds; **107/107 resilient tests passed** in 130.87 seconds.

Binary/source/lifecycle audits:

```bash
nm -D --defined-only build/native-resilient-dataplane/lib64/libemender_ndp.so.1
nm -D --defined-only build/native-resilient-dataplane/bin/ndp_cxi_service
rg 'static Service|Service[[:space:]]*&[[:space:]]*service' \
  src/native_resilient_dataplane/src/ndp.cpp
pgrep -a -u "$(id -u)" ndp_cxi_service
find /tmp -maxdepth 2 -user "$(id -u)" -type s \
  \( -path '/tmp/emender-ndp-pytest-*/*' -o \
     -path '/tmp/emender-ndp-rpc-*/*' \)
```

Result: the expected stable v1 C entry points are exported; neither the client library nor service exports MPI/PMPI symbols; the client source contains no authoritative `Service` singleton; and no service child or socket remains. One socket created by the intentionally failing initial LeakSanitizer run was identified by its unique test directory and removed before the definitive passing/audit run; passing shutdowns removed their own sockets.

## Compute Pool conformance

This section applies the required conformance checklist from `RESILIENT_DILOCO_COMPUTE_POOL.md` and names every task-required matrix ID.

| Requirement | Conformance check at this boundary |
|---|---|
| **R03** | No launched-rank or all-rank invariant was added. Controller/trainer identities arrive over independent point-to-point local connections; the service does not initialize MPI. READY lease selection remains Python policy and is not inferred from connections. |
| **R04** | The service binds run, fence, generation, attempt, worker, incarnation, and submission sequence; identical replay is idempotent, conflicting identity reuse is rejected, and newer fences deterministically invalidate stale clients/state. |
| **R05** | Sealed source data is reduced inside the persistent service using the existing deterministic float64 weighted accumulator. The process integration verifies exact source values and global weight 3; the native reference suite covers unequal weights, arrival order, f32/f64/bf16, and digest parity. |
| **R08** | Metadata frames are capped at 4,096 bytes, event batches at 16, connections at 64, and ancillary descriptors by both wire and opcode cardinality. Dense state stays in service-owned bounded memfds/core storage; no central Python full-model broker is introduced. Existing owner credit/replay bounds remain unchanged. |
| **R09** | The service is model-free. Trainers remain model/optimizer owners and pass only a sealed producer-direct memfd; a trainer disconnect is disposable while its admitted contribution remains service-owned. Controller policy stays outside the service. |
| **R10** | Dense input/result bytes use node-local memfd mappings and the compiled boundary; metadata uses a protected node-local UNIX socket. No dense Python serialization, Python TCP listener, or Lustre hot-path payload is used by this boundary. Live-role replacement is still downstream and fail-closed. |
| **R14** | Requests retain absolute deadlines; polls are capped at 30 seconds; the service has a configured lifetime deadline, bounded accept/progress polling, bounded connection shutdown, terminal metrics, deterministic socket unlink, and reaped children. Live stage timing remains a later runner artifact. |
| **R15** | SHA-256 source/layout identity, exact extent, seal checks, nonfinite rejection, deterministic weighted float64 reduction, total weight, and exact result bytes are exercised. Accepted-token/generation-close policy remains Python-owned and is not inferred here. |

The applicable recovery path is disconnect/reconnect, newer-fence invalidation, and full service restart. The minimum progress floor is unchanged from the authoritative design: Python policy decides whether a generation has enough admitted READY work to freeze; this native boundary makes no independent forward-progress claim. Atomic committed evidence is represented at this component rung by the fenced finalized result/root and commit transition; durable checkpoint publication remains Python policy and is not claimed by this task.

## Native data-plane conformance

| Requirement | Conformance check at this boundary |
|---|---|
| **NDP01** | The hard compiled boundary now exists: stable client ABI to seqpacket metadata RPC, descriptor-only dense transfer, persistent compiled owner. No production Python dense TCP is used by the component. Real E97 role wiring remains fail-closed and downstream. |
| **NDP03** | One `ndp_cxi_service` lifetime owns `ServiceCore`, `ServiceRpcServer`, and `FabricEndpoint`. The process test uses explicit `tcp;ofi_rxm`; exact production `cxi` policy is preserved, not weakened. G1/G2 CXI evidence is not claimed. |
| **NDP04** | Producer-direct input transfers exactly one sealed memfd, with exact extent and seal verification and no socket serialization/full-layout copy. The result is one separately shared read-only memfd. Full-E97 live lane accounting remains downstream. |
| **NDP06** | The RPC and core preserve fixed run/fence/generation/attempt/owner/worker/incarnation/sequence/layout/source identities. Per-connection RPC sequences must start at one and increase exactly once. |
| **NDP08** | The packet, event, connection, local layout, buffer, result, and resident-byte bounds fail closed. Existing transport TX/RX admission remains in the co-owned endpoint. The required live full-E97 configured accounting is not inferred. |
| **NDP10** | Core SHA-256 validation, nonfinite rejection, once-only application, identical receipt replay, and conflicting identity reuse rejection are preserved behind the service. Existing transport CRC32C/receipt checks remain unchanged. |
| **NDP11** | Admitted trainer inputs survive disconnect in service-owned references and existing bounded replay/journal semantics remain intact. No new unbounded socket replay or thread ledger was added. Live owner reassignment and zero-disk G2 evidence remain downstream. |
| **NDP12** | Finalization creates one service-owned result buffer and returns one `O_RDONLY` descriptor to a reconnecting controller. The test proves exact mapping and write rejection. Live trainer redistribution is not claimed. |
| **NDP13** | Absolute request deadlines, bounded poll intervals, route-local connection failure, newer-fence abort, orderly drain, service-death shutdown status, thread joins, child reap, and socket cleanup are exercised. Live CXI failure evidence remains downstream. |
| **NDP14** | `libemender_ndp.so.1` keeps ABI v1 and its public symbols while production calls now use versioned, bounded `AF_UNIX/SOCK_SEQPACKET` RPC. Only descriptors, commands, identities, receipts, metrics, and results metadata traverse the socket. |
| **NDP15** | Fenced finalization yields a read-only result tied to run/fence/generation/attempt/layout/base/result-root/weight/bytes, survives client reconnect, and is invalidated by a newer fence/service restart. Python checkpoint policy/publication and live collective-free drain remain downstream. |

## Scope and handoff

This closes the compiled per-node service/client boundary that was the specific NDP14 singleton gap. It does not change the authoritative limitations for live native E97 traffic: the production selection guard remains in place, no live trainer/manager role has been rewired, no provider/mount/traffic attestation has been claimed, and no Slurm job has been submitted. The next task may wire the stable socket/admission/fence ABI into real E97 roles without reintroducing a client-owned singleton or a Python dense listener.
