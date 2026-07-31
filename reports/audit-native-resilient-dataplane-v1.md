# Native resilient data plane v1 adversarial audit

Date: 2026-07-18

Task: `audit-native-resilient-dataplane-v1`

Design authority: [Resilient DiLoCo Compute Pool v1](../docs/RESILIENT_DILOCO_COMPUTE_POOL.md) and [Native resilient DiLoCo data plane v1](../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md)

## Findings and release decision

**Decision: HOLD native live testing.** The integrated tree contains useful,
locally tested native local-reduction and libfabric components, but it did not
contain an integrated native split-role dense path. The production role could
previously attest those components and then continue through the Python
file/TCP implementation. This audit fixes that unsafe admission decision by
failing closed before model load. The native functionality remains a release
blocker tracked by WG task `implement-live-split`; component G0 evidence is not
G1/G2 authorization.

No admitted critical or high correctness defect remains silently enabled.
The unresolved live-integration and full-E97 lane-accounting work is isolated
behind the fail-closed gate. No Slurm command or job was submitted.

| ID | Severity | Finding | Evidence | Disposition |
|---|---|---|---|---|
| F01 | **Critical** | Native selection relabeled the live Python spool/TCP dense path; `NativeManagerSession` was component-test-only and its Python transport wrapper did not expose send/receive. | `scripts/frontier/resilient_e97_role.py:316-340,435-458,610-624`; `ndm/native_transport.py:318-438`; repository call-site search | **Safety fixed:** `scripts/frontier/resilient_e97_role.py:132-155` rejects `native-test` and `native-cxi` after artifact attestation and before model load. Regression: `tests/test_resilient_e97_runtime.py:31-38`. Functional wiring remains blocked under `implement-live-split`. |
| F02 | **Critical** | Production did not request `FI_SOURCE`; unknown provider addresses were queued with peer 0, and peer 0 bypassed the C bridge's route check. A rogue endpoint could claim an installed worker identity. | Prior `native/dataplane/src/fabric.cpp:125-145,443-523`; prior `native/dataplane/src/transport_c.cpp` receive binding | **Fixed:** source reporting is mandatory, unknown/expired sources are event-only and their RX slot is reposted, and decoded identities require an exact nonzero source peer (`fabric.cpp:125-145,443-523`; `transport_c.cpp:348-386`). A real third RDM endpoint now attempts impersonation (`fabric_multiprocess_test.cpp:140-172`). |
| F03 | **High** | `ResultAssembler` accepted result bytes without binding run, fence, generation, attempt, owner epoch, layout, base, global weight, chunk count or shard size. Stale-fence results could enter the final aggregate. | Prior `native/dataplane/src/owner.cpp` `ResultAssembler::accept` | **Fixed:** the assembler owns a validated immutable `GenerationPlan` and rejects every mismatched identity/framing field before mutation (`owner.cpp:535-586`). Regression injects stale fence, stale owner epoch and conflicting weight and proves incomplete state (`protocol_owner_test.cpp:243-285`). |
| F04 | **High** | Native checkpoint proposal/commit checked only a subset of result/publication identity. A current-fence manifest could acknowledge a foreign or stale native result root. | Prior `ndm/native_pool_runtime.py:217-270` | **Fixed:** proposal binds the exact client, run, fence, generation, attempt, layout, base, nonzero result root and weight; commit additionally requires the publication's root, weight and byte count (`native_pool_runtime.py:215-297`). Regression rejects stale result fence, missing CAS and mismatched result root on the same session before a valid commit (`test_native_pool_integration.py:43-98`). |
| F05 | **High** | Shutdown stopped progress before canceling operations and ignored endpoint-close failure before releasing registered buffers, risking provider access to freed storage. | Prior `native/dataplane/src/fabric.cpp` `FabricEndpoint::shutdown` | **Fixed:** active work is canceled while the endpoint/storage are live, completions are drained, and registered memory is released only after successful endpoint close (`fabric.cpp:753-808`). If the provider remains busy, provider objects and malloc-backed slot storage are deliberately retained until supervised process exit rather than freed DMA-visible. ASan/UBSan and normal multiprocess close paths pass. |
| F06 | **High** | C++ allocation/container/provider exceptions could escape the exported transport C ABI or terminate the progress thread. Output handles could retain caller garbage after failure. | Prior `native/dataplane/src/transport_c.cpp`; prior `fabric.cpp:512-532` | **Fixed:** every exported transport operation is guarded and maps `bad_alloc` to `ENOMEM`, other exceptions to `EIO`, failure outputs are initialized, and the progress thread contains exceptions (`transport_c.cpp:65-75,439-516`; `fabric.cpp:535-565`). |
| F07 | **High** | A sealed memfd shorter than its claimed registration length could be mapped past EOF (SIGBUS); XPMEM address spans were not overflow checked. The Python loader also omitted the 64-bit `ndp_buffer_register_v1` signature, truncating direct client handles. | Prior `src/native_resilient_dataplane/src/ndp.cpp:730-777`; prior `ndm/native_dataplane.py:395-420` | **Fixed:** exact `fstat` extent and checked `uintptr_t` span are required (`ndp.cpp:730-777`); ctypes now binds the full signature (`native_dataplane.py:403-410`). Regression claims eight bytes from a sealed four-byte allocation and receives `EBOUNDS` with a zero output handle (`test_native_dataplane_failure.py:108-124`). |
| F08 | **High** | The Python reference transport built a commit header from a shared aggregate map, released its lock, and streamed from that same map. Another bucket connection could acknowledge delivery and clear the map between those actions, producing a commit that promised two frames but sent fewer—a partial result publication. | Focused-suite failure `RuntimeError('expected aggregate frame')`; prior `ndm/resilient_node_transport.py:360-388` | **Fixed:** accepted identity, token weight and immutable aggregate-byte references are captured atomically under the generation lock; header count and frames come from that one snapshot (`resilient_node_transport.py:365-388`). The exact missing-node/weighted-mean concurrent test passed five consecutive runs (`test_resilient_node_transport.py:103-129`). |
| F09 | **Medium** | An RX CQ error consumed a posted receive permanently, and partial event polling drained eventfd readiness even while queued events remained. Repeated faults could exhaust progress/notifications. | Prior `fabric.cpp:410-440`; prior `ndp.cpp:1310-1334` | **Fixed:** failed RX slots repost outside the state lock, rejected source slots repost, and eventfd is re-armed when the event deque remains nonempty. Multiprocess rogue/corrupt/replay traffic and local ABI polling pass. |
| F10 | **Medium** | The shared-login two-manager fixture selected its three-port block from Linux's ephemeral client range; the closed probe was repeatedly stolen by an outbound connection before manager bind. | Two consecutive focused-suite failures at `DistributedOwnerServer.bind`, `EADDRINUSE`; `/proc/sys/net/ipv4/ip_local_port_range=32768 60999` | **Fixed:** the fixture performs a PID-distributed, three-port scan in 20000-29999 (`test_resilient_e97_runtime.py:115-142`). The exact failing test then passed. This changes test isolation only, not the production protocol. |

### F01 detail: why the integration claim was unsafe

The live manager constructs `LocalTrainerSpool` and
`DistributedOwnerServer`, then calls `submit_owned_shards` and
`fetch_owned_shards`. The live trainer also constructs `LocalTrainerSpool`.
No live call site constructs `NativeManagerSession`; the sole caller is the
component integration test. `NativeTransport` exposes bind, route upsert,
poll, metrics and close, but not the C ABI's send/receive functions. The local
ABI is a process-local singleton and validates the supplied socket path and
admission token without using them as a cross-process trainer/manager control
channel.

Consequently, a clean bundle and even a matching G2 JSON could not prove that
live dense bytes used the bundle. The new gate occurs after cryptographic
artifact attestation but before manager/trainer dispatch and before trainer
model load. This preserves evidence checking while preventing component
evidence from authorizing the wrong implementation.

The live follow-up must, at minimum, provide a real cross-process local ABI,
bind transport send/receive in Python or move orchestration into the service,
replace all four Python dense/spool call sites, install only frozen leased
routes, drive owner/replay/result state machines, bind checkpoint publication,
and account/release all full-layout lanes. Until then, `native-test` and
`native-cxi` are intentionally unusable in the split-role process.

## Audit coverage

| Area | Adversarial check and result |
|---|---|
| Memory ownership | Local buffers retain public and operation references; result views close their duplicated fd; fixed fabric slots are freed only after endpoint close. Short memfd extent, XPMEM span, buffer-count and shared-byte exhaustion are tested. Provider-busy shutdown retains storage until process exit rather than risking UAF. |
| Integer/size overflow | Local layout and shared-byte additions use explicit limits; XPMEM pointer addition is now checked. Transport frame sizes convert to `size_t` only after bounds checks. `GenerationPlan` validates 16 GiB layout, shard/payload/slot caps, checked `2L+A+slots+ledger+64MiB` admission and total weight below `2^63`. Result assembler now consumes that validated plan. |
| Native/Python lifetime | ctypes signatures are size/versioned and the missing registration binding is fixed. `Buffer`/`ResultView` close duplicated fds before releasing native handles. Manager close drains/aborts local state, cancels each current route, closes transport/local handles and closes telemetry fd. Concurrent close/use remains intentionally outside the manager-scoped Python API contract. |
| CQ progress | One progress thread services distinct TX/RX CQs; `FI_THREAD_SAFE` is requested. CQ errors release TX accounting and repost RX slots. Unknown sources cannot consume slots. Thread exceptions become route-error events rather than crossing a thread boundary. |
| Fencing | Local contributions, owner frames, result redistribution and checkpoint acknowledgement are bound to run/fence/generation/attempt plus their subordinate identity. Stale input is rejected before aggregate/publication mutation. |
| Idempotence/replay | Contribution order is deterministic, duplicate identity/digest returns the stable receipt, conflicting reuse is quarantined, credit is independent of CQ completion, replay is capped at `2L`, and reassignment is capped at two. Result shard duplicates require identical digest. |
| Peer loss | Routes expire/cancel independently, loss does not require an all-peer rendezvous, replay buffers are bounded, and an unknown provider source is observable but never consumable. Local TCP multiprocess owner-loss tests remain reference evidence, not proof of the unwired native live path. |
| Shutdown | Admission is stopped by caller policy; native cancel occurs before progress join and endpoint close; buffers/MRs are released after provider ownership ends. Close is idempotent. An endpoint that refuses close emits `shutdown_endpoint_busy` and retains resources for supervised process exit. |
| Provider selection | Production requires exact `cxi`, test mode cannot select `cxi`, `FI_PROVIDER` cannot weaken the required provider, `FI_EP_RDM` is required and `FI_SOURCE` is now mandatory for route authentication. Local tests use explicit `tcp;ofi_rxm`. No CXI result is claimed here. |
| File descriptors | Local client/eventfd, memfd duplicates, telemetry fd and installed binary handles have explicit owners and close paths; fixed buffer count is 64. Libfabric fids close in endpoint→MR/CQ/AV→domain→fabric order after drain. The deliberate provider-busy path retains fids/storage until process termination and is telemetry-visible. |
| Thread safety | C++ provider calls are serialized by `api_mutex_`; state/counters/events use `mutex_`; telemetry writes use a separate mutex and handle EINTR/partial writes. Libfabric is opened with `FI_THREAD_SAFE`. Local ABI global state is serialized by its service mutex. |
| Static/sanitizer checks | CrayClang 18.0.1 builds with the project's warning sets. ASan+UBSan now covers both local and transport children and passes all 8 CTests with leak detection. `clang-tidy`, `cppcheck` and `scan-build` are not installed; this limitation is recorded rather than silently omitted. |

## 256-node and full-E97 complexity/buffer accounting

The authoritative E97 layout is:

```text
elements E = 688,346,312
layout L   = E * 8 = 5,506,770,496 bytes (native f64 wire/accumulator)
payload C  = 67,108,864 bytes
shards S   = ceil(L/C) = 83
header H   = 320 bytes
nodes N    = 256
```

### Traffic and state complexity

Each accepted node contributes one logical `L` and each node receives one
logical `L` result. At 256 nodes:

| Quantity | Formula | Exact value |
|---|---:|---:|
| Contribution logical bytes | `N*L` | 1,409,733,246,976 |
| Redistribution logical bytes | `N*L` | 1,409,733,246,976 |
| Total logical dense bytes | `2*N*L` | 2,819,466,493,952 |
| Data frames per direction | `N*S` | 21,248 |
| Total data frames | `2*N*S` | 42,496 |
| Header bytes per direction | `N*S*H` | 6,799,360 |
| Total data-frame header bytes | `2*N*S*H` | 13,598,720 |
| AV routes per service | `N-1` | 255 |
| Directed AV entries cluster-wide | `N*(N-1)` | 65,280 |

Per-node route and transfer work is bounded `O(N)` and cluster transfer work is
`O(N*S)`. Directed route state is `O(N^2)` cluster-wide but only 65,280 small
AV entries at N=256. Deterministic contribution order is computed per shard,
`O(S*N log N)` with the current sort; ledger memory is `O(S*N)`. There is no
all-rank collective or central `O(N*L)` broker in the component design.

### Native resident admission formula

For four TX and four RX slots, the component plan's conservative per-service
admission is:

```text
2*L                                      = 11,013,540,992
2*K*(C+H), K=4                          =    536,873,472
R = N*S*128 receipt/ledger bytes        =      2,719,744
slack                                     =     67,108,864
base before assigned-owner bytes          = 11,620,243,072
```

With two owners, the enforced owner-skew allowance
`A=ceil(L/2)+C=2,820,494,112` produces 14,440,737,184 bytes
(13.449 GiB). With 256 owners,
`A=ceil(L/256)+C=88,619,687` produces 11,708,862,759 bytes
(10.905 GiB). Both fit the component's 16 GiB default transport resident
limit, and every addition/multiplication is checked before admission.

That formula does **not** solve the missing live local-lane budget. One full
E97 f32 trainer output is 2,753,385,248 bytes (2.564 GiB). Six concurrent
lanes consume 16,520,311,488 bytes (15.386 GiB); eight consume
22,027,081,984 bytes (20.514 GiB), before the 5,506,770,496-byte f64 numerator,
the 2,753,385,248-byte result view, duplicated fds/mappings or allocator
overhead. Therefore the current 16 GiB local default cannot admit eight full
trainer buffers. This is part of F01/`implement-live-split`, and the production
fail-closed gate prevents it becoming an OOM or partial-generation surprise
during live testing.

## Fault simulations and atomicity result

The focused tests prove the following failure boundaries:

1. A third real `FI_EP_RDM` endpoint sends a checksum-valid contribution frame
   claiming an installed worker. The receiver emits a route error with no peer
   identity, makes no frame consumable, reposts the slot, and then processes
   legitimate cross-process traffic.
2. Stale result fence, stale owner epoch and wrong global weight are rejected
   before a result shard is marked received. `complete()` remains false after
   the stale-fence attempt; only the exact plan completes.
3. A stale `ResultView` cannot create a checkpoint proposal. Missing CAS,
   stale authoritative fence and mismatched result root cannot issue native
   `COMMIT`; the same session remains usable and commits only after the exact
   publication identity is supplied. This proves no partial native commit.
4. Concurrent Python reference bucket connections cannot clear result storage
   between a commit header and its promised aggregate frames: each response
   uses one immutable under-lock snapshot. The missing-node exact weighted
   exchange passed five consecutive stress reruns after the fix.
5. The retained atomic-publication fault test expires the old allocation,
   rejects its generation-2 finalize, proves no generation-2 handoff exists,
   leaves the prior immutable latest authoritative, and advances only under a
   newer lease/fence. This proves no stale-fence publication.
6. Corrupt payload, nonfinite input, conflicting duplicate, lost receipt,
   replay exhaustion, peer expiry/removal and cancellation tests prove no
   accumulator mutation before validation and bounded release afterward.

## Validation

This audit applied the Compute Pool v1 conformance checklist to **R01–R16** and
the native specialization checklist to **NDP01–NDP17**. The corrected status
and bounded gap for every ID is recorded in
[the gap matrix](../docs/RESILIENT_DILOCO_GAP_MATRIX.md). In particular, the
audit does not convert component G0 evidence into a claim of NDP01, NDP04,
NDP07, NDP12, NDP14, NDP15 or NDP16 live-path conformance.

Commands executed locally include:

```bash
cmake -S native -B build/audit-native-resilient-dataplane \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DNDP_BUILD_TESTS=ON
cmake --build build/audit-native-resilient-dataplane -j 8
ctest --test-dir build/audit-native-resilient-dataplane --output-on-failure
# result: 8/8 passed

cmake -S native -B build/audit-native-asan \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DNDP_BUILD_TESTS=ON \
  -DNDP_ENABLE_SANITIZERS=ON
cmake --build build/audit-native-asan -j 8
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build/audit-native-asan --output-on-failure
# result: 8/8 passed

EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1" \
EMENDER_NDP_TRANSPORT_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1" \
  .envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_transport_bridge.py \
  tests/test_native_artifact_attestation.py \
  tests/test_native_pool_integration.py \
  tests/test_resilient_e97_reducer.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_peer_membership.py \
  tests/test_resilient_shard_owner.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py
# final result: 105/105 passed

.envs/olcf-rocm711-torch210-py312/bin/python -m compileall -q ndm scripts/frontier
bash -n scripts/frontier/build_native_resilient_dataplane.sh \
  scripts/frontier/resilient_e97_true_2n.sbatch
nm -D --defined-only build/native-resilient-dataplane/lib64/libemender_ndp.so.1
nm -D --defined-only build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1
git diff --check
cmp -s AGENTS.md CLAUDE.md
```

The normal and sanitized native suites pass, the complete focused Python suite
passes, generated Python compiles, shell syntax and JSON parse, both SONAMEs
remain ABI v1, dynamic symbol scans contain no direct `MPI_`/`PMPI_`, and the
project guides remain identical. Standalone `clang-tidy`, `cppcheck` and
`scan-build` were unavailable. The authoritative post-commit build manifest is
regenerated from a clean exact source tip before publication.

No `sbatch`, `srun`, `scancel` or other Slurm operation was executed.
