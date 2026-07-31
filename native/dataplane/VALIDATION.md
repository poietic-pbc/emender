# Libfabric owner transport v1 validation

Validation date: 2026-07-18. This is local G0 evidence for
`implement-libfabric-cxi-dataplane-v1`; it is not a Frontier CXI qualification
or permission to launch a model. No Slurm command was submitted.

## Acceptance evidence

| Acceptance criterion | Evidence |
| --- | --- |
| Native service and stable ABI | A C++17 release build used the active libfabric 2.3.1 headers and `libfabric.so.1`. Installation produced `include/emender/ndp_transport.h`, `libemender_ndp_transport.so.1.0.0` with SONAME `libemender_ndp_transport.so.1`, and `ndp_cxi_service`. The dynamic export scan contained only the twelve versioned `ndp_transport_*@@EMENDER_NDP_TRANSPORT_1.0` functions. The ABI test checks forward-prefix handling and the published structure sizes. |
| Local multiprocess faults | `ndp_fabric_multiprocess_test` exchanges real messages between two OS processes through `tcp;ofi_rxm` and `FI_EP_RDM`. It covers fragmented endpoint exchange, out-of-order arrival, identical duplicate replay, receiver corruption, stale fencing, expired deadlines, abrupt peer exit, cancellation, removal, and post-removal route failure. `ndp_protocol_owner_test` additionally covers truncated frames, header and payload corruption, conflicting identity reuse, nonfinite input, receipt replay, two owner reassignments, replay exhaustion, and result redistribution. |
| Hard memory bounds and release | Admission checks the normative overflow-safe resident-byte formula. Slot counts are limited to 16 and payloads to 64 MiB. Completed frames remain in their registered RX slots until consumption rather than allocating a second service payload; the multiprocess test proves a too-small consumer buffer preserves that slot for retry. It also asserts TX/RX slot high-water marks do not exceed two, final fabric in-flight and retained bytes are zero, and sent/received wire bytes have been released. The owner tests prove the replay buffer rejects its first over-limit retain, stays below `2*layout_bytes`, releases exactly 800 bytes, and releases the single redistribution aggregate. |
| Elastic membership | The service owns one RDM endpoint and AV. `peer_upsert`, expiry, cancellation, and `peer_remove` operate on fenced endpoint records supplied by the caller. The C ABI test proves idempotent upsert, monotonic endpoint epochs, reconnect with a new incarnation, and stale-record rejection. The binaries have no `MPI_` or `PMPI_` symbol and contain no communicator or barrier discovery path. |
| Provider isolation | Production requires both selected and required provider names to equal `cxi`, rejects a conflicting `FI_PROVIDER`, rejects any non-CXI result, and requires successful MR registration. Test mode accepts only explicit `tcp;ofi_rxm` or `shm`, rejects `cxi`, and always reports `production_provider=false`. Local-provider registration exceptions therefore cannot become production evidence. |
| Telemetry | JSON Lines schema `emender-native-dataplane-telemetry-v1` and `ndp_transport_metrics_v1` report useful and wire TX/RX bytes, retries, replay bytes, CQ and route errors, throughput, mean send latency, in-flight/retained/released bytes and high-water marks, live peers, and owner state, together with selected provider facts. |

## Commands and results

Release configure/build/test:

```text
cmake -S native/dataplane -B build/native-dataplane -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/native-dataplane --parallel 8
ctest --test-dir build/native-dataplane --output-on-failure
Result: 7/7 tests passed
```

The seven tests are the protocol/owner test, the real multiprocess fabric
test, the stable C ABI test, production fail-closed selection, explicit local
test selection, rejection of CXI in test mode, and rejection of an
environment override that would weaken production policy.

Repeated fault-path test:

```text
ctest --test-dir build/native-dataplane \
  -R 'ndp_(protocol_owner|fabric_multiprocess|transport_c_abi)_test' \
  --repeat until-fail:30
Result: 90/90 executions passed
```

Sanitizer configure/build/test:

```text
cmake -S native/dataplane -B build/native-dataplane-asan \
  -DCMAKE_BUILD_TYPE=Debug -DNDP_ENABLE_SANITIZERS=ON
cmake --build build/native-dataplane-asan --parallel 8
ctest --test-dir build/native-dataplane-asan --output-on-failure
Result: 7/7 tests passed under AddressSanitizer and UndefinedBehaviorSanitizer
```

Artifact checks:

```text
cmake --install build/native-dataplane --prefix <temporary-prefix>
readelf -d build/native-dataplane/libemender_ndp_transport.so.1
nm -D --defined-only build/native-dataplane/libemender_ndp_transport.so.1
nm -D build/native-dataplane/libemender_ndp_transport.so.1 \
  build/native-dataplane/ndp_cxi_service | rg 'MPI_|PMPI_'
Result: install succeeded; SONAME is .so.1; exports are versioned; MPI scan empty
```

## Architecture conformance

The implementation checklist is against the normative
`RESILIENT_DILOCO_COMPUTE_POOL.md`, with gap-matrix requirement identifiers:

- R03 / NDP02 and NDP07: endpoints are installed from fenced controller
  records; no launched-rank membership, fixed communicator, or all-rank
  barrier exists.
- R04 and R08 / NDP06 and NDP09–NDP11: generation, fence, attempt, owner epoch,
  contribution, shard, and message identities are field-serialized and
  validated; owner order and mapping are deterministic; application receipts
  make duplicate replay idempotent.
- R06, R11, and R14 / NDP11–NDP13: deadlines, cancellation, peer-local CQ
  failure, endpoint expiry/removal, reconnect, owner reassignment hooks, and
  clean bounded shutdown avoid an allocation-wide failure domain.
- R08 and R10 / NDP08, NDP09, and NDP12: registered fixed pools in production,
  absolute receiver credit, capped event queues, replay limits, admission
  preflight, and one aggregate redistribution buffer hard-bound residency.
- R13 / NDP02: the compiled service is persistent and elastic, with one
  connectionless RDM route per active endpoint rather than one connection per
  shard.
- R14 and R15 / NDP03 and NDP16: provider facts and useful/wire/retry/error,
  throughput, latency, memory, route, and owner-state metrics are emitted.
- R16 / NDP17: this report is explicitly G0-only. CXI G1/G2 evidence remains
  the responsibility of the downstream qualification task.

Generation freeze/closure, minimum progress policy, durable commit/checkpoint,
and Slurm supervision remain in the fenced Python control plane as required by
the architecture; this transport neither duplicates nor weakens them.
