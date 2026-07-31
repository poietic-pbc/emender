# Native resilient owner transport v1

This directory is the compiled elastic-network half of the native resilient
DiLoCo data plane. It specializes the authoritative
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
without taking over the Python control plane or the sibling local handoff ABI.
The local task owns `include/emender/ndp.h`, trainer buffers, native local
reduction, result-view handoff, and the Python bridge. This directory exports
the additive transport ABI in `include/emender/ndp_transport.h`; the downstream
integration task binds those calls to the authoritative local service handles.

## Delivered service

`ndp_cxi_service` owns one persistent local reduction/handle core, one bounded
seqpacket RPC server, one libfabric endpoint, one address vector,
distinct transmit and receive completion queues, fixed 2 MiB-aligned TX/RX
slot pools, and one native progress thread. It uses `FI_EP_RDM`, `FI_MSG`, and
`FI_CONTEXT`. Peer endpoint names are opaque inputs to `peer_upsert`; the
service neither initializes MPI nor discovers launched ranks. Routes can be
added, cancelled, expired, and removed independently.

Production policy accepts only the explicit pair `--provider=cxi
--require-provider=cxi --production`. It rejects a conflicting `FI_PROVIDER`,
non-CXI returned provider, non-RDM endpoint, non-equivalent provider matches,
insufficient maximum message size, and failure to register the fixed pools.
Test mode accepts only explicit `tcp;ofi_rxm` or `shm`; a test-mode endpoint is
always reported with `production_provider=false`. Some local providers
advertise `mr_mode=0` and reject `fi_mr_reg` because host registration is not
part of their contract. Their slots remain fixed/reusable for failure tests,
but that result is never production evidence. CXI production always requires
successful explicit MR registration.

`protocol.cpp` field-serializes the 320-byte v1 header, CRC32C-checks bytes
0–311, SHA-256-checks complete payloads, and enforces exact body and shard
bounds before allocation or field use. `owner.cpp` enforces current
run/fence/generation/attempt/owner epoch, deterministic per-shard contribution
order, receiver-issued absolute credit, finite float64 application, a compact
once-only receipt ledger, identical duplicate replay, conflicting-reuse
quarantine, at most two reassignments, at most `2*layout_bytes` replay, and
one preallocated redistribution aggregate.

The installed `libemender_ndp.so.1` is a client library for the service's
version-1 `AF_UNIX/SOCK_SEQPACKET` protocol. The socket is mode `0600`; peers
must match `SO_PEERCRED`, admission token, run, and fence. Packets are at most
64 KiB and metadata-only. Dense trainer input and read-only sealed results move
as exactly one memfd descriptor per operation through `SCM_RIGHTS`.

There is one route per current endpoint, not one connection per shard.
Generation, fence, contribution, owner epoch, shard, and sequence identities
are carried in every fixed header. No Python socket payload path or fabric
collective exists in these binaries.

## Hard bounds and release

Before owner admission, overflow-checked preflight implements the v1 bound:

```text
2*layout_bytes + assigned_owner_bytes
+ (tx_slots + rx_slots) * (payload_max + 320)
+ frozen_contributions * shard_count * 128
+ 64 MiB
```

Payloads are capped at 64 MiB, shards at 256, contributions at 4,096, and each
slot pool at 16. TX bytes stay live only through their fabric completion. The
receive queue never exceeds the configured RX window: receive reposting stops
when queued frames plus posted slots reach the limit and resumes only after a
consumer release. A completed receive remains in its registered slot until the
consumer copies it; there is no second service-owned payload allocation, and a
too-small consumer buffer leaves the slot available for a bounded retry. Wire
validation uses a non-owning view over fixed slots or the caller buffer, and
owner result-root hashing reads finalized shards in place.
Metadata event queues are separately capped. Sender replay
sources release only on an application receipt or cancellation. Owner results
release on commit/abort/cancel. Tests assert zero final in-flight and retained
bytes plus exact released-byte accounting.

## Telemetry

The service emits JSON Lines schema
`emender-native-dataplane-telemetry-v1`. Each snapshot carries provider facts,
useful and wire TX/RX bytes, retry and replay bytes, CQ/route errors, derived
throughput, mean send latency, in-flight/retained/released byte counters and
high-water marks, and owner/route state. The stable metrics ABI exposes the
same counters without parsing logs.

## Local validation

```bash
cmake -S native/dataplane -B build/native-dataplane \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/native-dataplane --parallel
ctest --test-dir build/native-dataplane --output-on-failure
```

The compiled tests cover:

- fragmented endpoint exchange and truncated frames (partial writes);
- out-of-order application and redistribution;
- duplicate network delivery and lost-receipt replay;
- CRC/payload corruption and nonfinite input;
- absolute timeout and credit stalls;
- abrupt peer exit, route cancellation/removal, and bounded shutdown;
- stale fences/generations/owner epochs and conflicting identities;
- owner reassignment/replay caps and deterministic owner mapping;
- resident/in-flight/retained bounds and prompt release;
- preservation on a too-small consumer buffer without an internal payload copy;
- C ABI lifecycle/forward-prefix rules and provider fail-closed policy; and
- a separate-process service/controller/trainer RPC lifecycle including
  disconnect, replay, nonfinite input, newer-fence and service-restart rejects,
  read-only result descriptors, and socket/child cleanup; and
- an actual two-process `tcp;ofi_rxm` `FI_EP_RDM` exchange using the active
  libfabric headers and shared library.

The elastic binary symbol scan must contain no `MPI_`/`PMPI_` entry points.
These local tests are G0 evidence only. They are not G1/G2 CXI evidence and do
not authorize a real model or a 4+ node launch. This implementation task
submits no Slurm job.

## Conformance checklist

This implementation conforms to Resilient DiLoCo Compute Pool v1 requirements
R03, R04, R06, R08, R10, R11, and R13–R16 and native requirements NDP02,
NDP03, and NDP06–NDP13, NDP16, and NDP17:

- membership and routes come only from explicitly installed, expiring endpoint
  records; there is no launched-rank or all-rank invariant (R03, R11, R13;
  NDP02, NDP07);
- fenced identities, deterministic owner/order math, idempotent replay, and
  stale/corrupt rejection are compiled and tested (R04, R08; NDP06,
  NDP09–NDP11);
- fixed pools, admission formulas, receive credits, release invariants, and no
  central broker bound the non-Lustre point-to-point path (R08, R10; NDP08,
  NDP09, NDP12);
- absolute deadlines, route-local failure, cancellation, peer removal,
  reassignment hooks, and clean drain avoid allocation-wide aborts (R06, R11,
  R14; NDP11–NDP13);
- exact provider facts, useful/wire/retry/error/throughput/latency/owner-state
  telemetry are mandatory outputs (R14, R15; NDP03, NDP16); and
- local tests are explicitly non-promoting; G1/G2 CXI and every later ordered
  gate remain downstream prerequisites (R16; NDP17).

Minimum progress floors, generation freeze/closure, durable commit/checkpoint,
and Slurm supervision remain Python policy and are intentionally absent here.
