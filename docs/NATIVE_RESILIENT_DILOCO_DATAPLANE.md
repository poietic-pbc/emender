# Native resilient DiLoCo data plane v1

**Status:** Normative, implementation-ready data-plane authority, version 1
(2026-07-18).

**Semantic authority:** This document specializes the transport, local handoff,
native ABI, and tensor-reduction requirements of
[Resilient DiLoCo Compute Pool](RESILIENT_DILOCO_COMPUTE_POOL.md), version 1.
That document remains authoritative for allocation claims, membership, generation closure,
commit semantics, and training correctness.
[ADR-002](ASYNC_DECOUPLED_DILOCO_V2.md) is the normative semantic extension
when `async-decoupled-v2.1-simple` is selected; it preserves NDP01–NDP17 but
requires a v2.1 protocol/ABI extension rather than relabeling this v1 wire
format. The
[companion gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md) traces both authorities.
An implementation conforms only if it satisfies both. A contradiction fails
closed and requires a reviewed design change; it is not resolved by choosing
the easier behavior.

Normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their RFC
2119 meanings. Exact integer constants in this document are part of the v1
contract.

## Executive contract

This table is the design in operational form. Read it before the protocol
references; the later “Normative reference” sections are implementation
appendices and do not change these decisions.

| Concern | V1 contract |
|---|---|
| Control plane | One pure deterministic transition kernel in the persistent model-free C++ service owns live fence/incarnation validation, READY membership, generation open/freeze/abort/commit, node-apply state, and recovery handshakes. Python submits typed events, executes explicit effects, adapts Slurm, chooses checkpoint/outer policy, and writes immutable restart evidence; no shared database participates. |
| Dense data plane | One persistent model-free C++17 service per node owns local dense handoff, exact weighted reduction, network payloads, replay, and redistribution. Production is libfabric `FI_EP_RDM` with exact provider `cxi`; Python TCP carries zero production dense bytes. |
| Failure boundary | The elastic backend uses bounded point-to-point work over explicitly named contributions/owners. It never requires launched ranks, MPI, or an all-rank collective. Loss triggers bounded owner reassignment/replay or a clean no-commit abort. |
| Node-local path | Trainers produce directly into XPMEM or service-allocated memfd buffers. Handoff adds no trainer-sized write; redistribution creates one shared node aggregate, not eight files/copies. Disk may retain one reduced node contribution only as an explicit bounded NVMe replay fallback. |
| Correctness | Python reports clocks and locally complete, checksummed, replayable node contributions as typed events. The native kernel freezes the immutable accepted set and authorizes commit/apply; native owners apply each fenced identity exactly once, in the specified deterministic float64 order, with exact integer weights, checksums, credits, and idempotent receipts. |
| Checkpoint/commit | Native code returns a fenced read-only aggregate view. Native peers agree the exact next result/token/receipt identity; background policy materializes and reload-verifies the immutable checkpoint from immutable inputs, and a digest-linked receipt makes it durable. In bounded asynchronous mode, live trainers apply only the complete verified result later at a safe boundary. `latest.json` and fabric receipts are not authority. |
| Admission gates | G0 local -> G1 two-node CXI probe -> **G2 full-layout two-node synthetic** -> current-source G3 clean -> G4 failure/rejoin -> G5 fresh-allocation restart -> G6 direct 8/32/128 systems scale -> explicit 256 review. No real model or scale native job is allowed before exact-code G2 and current-source G3–G5 pass. |

## Decisions and requirements

The production dense path is a persistent, model-free C++17 service on every
node. It uses libfabric `FI_EP_RDM`; Frontier production selects provider `cxi`
explicitly. Python remains the scheduler/policy/effect adapter around the
compiled coordination authority. Python TCP, Python object
serialization, MPI, and any failure-sensitive all-rank collective are forbidden
from the elastic dense hot path.

| ID | Normative decision |
|---|---|
| NDP01 | Native peer control owns live allocation-fence/incarnation checks, READY membership, generation open/freeze/abort/commit, owner agreement, node-apply state, and recovery handshakes. Python owns only the scheduler adapter, outer/checkpoint policy, immutable publication, and supervision. C++ owns local dense handoff, native reduction, fabric movement, redistribution, dense-buffer lifetime, and independent stale-fence rejection. No compute role may open a shared control database. |
| NDP02 | The elastic backend uses only bounded point-to-point operations. It MUST NOT call an all-rank barrier, broadcast, reduce, all-reduce, all-gather, MPI initialization/finalization, or any operation whose completion requires every launched or READY peer. |
| NDP03 | One persistent C++17 node service uses libfabric `FI_EP_RDM` and `FI_MSG`. Frontier production MUST resolve exactly `cxi`; test providers are never silently promotable. |
| NDP04 | Trainer/service handoff uses XPMEM or a service-allocated `memfd`. The trainer exclusively owns live mutable model/optimizer state and captures a coherent fenced immutable snapshot into a preallocated/COW/equivalent bounded exported buffer at a safe boundary. The service never reads live mutable state. No extra trainer-sized file or socket copy is permitted in steady state. |
| NDP05 | Layout, conversion, weight bounds, accumulation order, rounding mode, and result encoding are fixed below. Native output is bitwise comparable with the v1 reference, independent of arrival order. |
| NDP06 | Every command, frame, contribution, receipt, replay, result, and checkpoint handoff carries the fenced wire identity defined below. A stale or conflicting identity is rejected, never guessed or overwritten. |
| NDP07 | Python exchanges opaque libfabric endpoint names as leased membership metadata. Native services install only current-fence routes; PMI, DNS fan-out, filesystem polling, and native all-gather are not endpoint exchange mechanisms. |
| NDP08 | Snapshot, result, and fabric buffers are fixed, capacity-bounded, pre-registered pools. Frame, layout, contribution, replay, mailbox, and owner-memory bounds are checked before admission. No unbounded allocation is allowed after a generation opens, and exhaustion skips/replaces/defers background work rather than blocking foreground training. |
| NDP09 | Receiver-issued byte and slot credits bound traffic. A fabric send completion is not an application receipt and does not restore logical credit. Credit or mailbox backpressure remains in background work and MUST NOT become a trainer wait after immutable snapshot admission. |
| NDP10 | SHA-256 payload checksums, CRC32C headers, once-only owner application, and idempotent receipts are mandatory. Nonfinite or corrupt data cannot enter an accumulator. |
| NDP11 | Sender memory is the primary replay source. Replay and owner reassignment are bounded by bytes, owner epochs, and deadlines. Disk is an optional one-node-contribution, node-local fallback only. |
| NDP12 | Committed aggregate shards are redistributed directly from owners into one service-owned node aggregate. Trainers map that shared result; the service does not write eight aggregate files. |
| NDP13 | Every stage has an absolute deadline and a defined abort/retry result. Fabric or owner failure is contained to routes/generation attempts and never invokes an allocation-wide abort. In bounded asynchronous mode, background deadline expiry skips/defers the result and cannot create a foreground result wait. |
| NDP14 | Python talks to a versioned C ABI and a local `AF_UNIX/SOCK_SEQPACKET` service. Only metadata crosses that control channel; dense bytes are referenced by native buffer handles. |
| NDP15 | Python selects checkpoint cadence/publisher and writes complete immutable state only from admitted immutable snapshots/results. Native peers agree the exact commit and return a fenced read-only aggregate view in the background. A complete verified result is applied/swapped atomically at a later safe boundary under a separate finite foreground bound; all eight trainer receipts reduce to one node-applied receipt. Checkpoint I/O is never on that foreground apply path. |
| NDP16 | Required counters, timings, identities, bounds, provider facts, and terminal reasons are emitted as structured telemetry. Bounded asynchronous artifacts separately time freeze/snapshot, admission, publish/network, aggregation, checkpoint, result wait, apply/swap, and total foreground idle for causally identified work. Missing or median-only telemetry fails an acceptance gate. |
| NDP17 | A full-E97-layout, two-node synthetic CXI gate is a hard prerequisite for every real model or larger-node job using this backend. Scale proceeds in the fixed order specified below. |

### Explicitly non-goals

The C++ dense service does not choose the scheduler fence, quorum policy,
outer-optimizer policy, checkpoint naming/retention, Slurm allocation changes,
simultaneous-allocation federation, stale-update policy, GPU-direct fabric, or
confidentiality in v1. The model-free peer-control protocol does own live
membership, generation closure/commit, and recovery handshakes. The v1 payload
is host-resident float64. GPU-direct
is a future ABI capability and cannot be enabled by configuration alone.
ADR-002 resolves the v2.1 asynchronous policy above this layer but does not
make the v1 ABI capable of carrying it.

The existing `scripts/frontier/compiled_mpich_dense_helper.cpp` is the
fixed-world performance/numerical reference and an explicitly selected
fixed-world fallback. Its `MPI_Allreduce`, `MPI_Allgather`, node barriers,
leader communicator, and launched `world_size` are deliberately nonconforming
to NDP02. It MUST NOT be used as the membership or failure domain of the
elastic backend.

### Simple asynchronous v2.1 extension boundary

All NDP01–NDP17 decisions remain mandatory for
`async-decoupled-v2.1-simple`. The native service remains persistent,
model-free, point-to-point, credit bounded, checksummed, deterministic, and
fenced. It has no authority to invent lag, exact-token floors, outer math,
checkpoint cadence, or scale-close timing. Those policy values and external
observations arrive as typed events. The service kernel is nevertheless the
sole authority that applies READY membership, immutable accepted sets,
generation phase, commit, node apply, and recovery transitions.

The current 76-byte v1 contribution identity and v1 metadata kinds are
insufficient. A conforming implementation MUST use
`NDP_ABI_V21 = 0x00020001`, wire protocol major 2/minor 1, and the exact v2.1
policy/metadata/manifest identities in ADR-002. It adds v2.1 symbols and
records; it does not grow a v1 struct or accept historical v2.0 records under
the new version. The extension carries:

- stable worker plus current incarnation, contribution sequence, cumulative
  local K-window range, interval endpoint, and eight-trainer cohort identity;
- base global version/digest, commit lag, applied-anchor lag,
  result-version-at-apply lag, and speculative-window lag as distinct clocks;
- policy schema/digest and code digest in addition to layout/payload digests;
- one positive `exact_tokens` field used for the token floor, accepted-token
  clock, deterministic NDP05 binary64 multiplication, denominator, manifest,
  and telemetry;
- bounded one-owned/one-mutable cohort admission, correction-ledger identity,
  capacity-one mailbox publication/replacement/release, and all-eight-trainer
  node-apply/recovery facts; and
- scale-authorization and leased-READY-snapshot closure digests for any
  direct scale
  group plan.

No v2.1 production record contains a distinct aggregation/effective/staleness
weight. Owners apply NDP05 in deterministic contribution order using
`exact_tokens` directly and divide by the checked exact-token sum. The
extension MUST NOT encode lagged work as a v1 `generation` with `tau = 0`,
accept a v2.0 policy/schema/digest, retain unbounded generations, make a
trainer or background worker share the live mutable model, copy weights while
the optimizer mutates them, make a trainer wait for discovery, quorum,
send/receipt, aggregation, hashing, checkpoint, or result readiness after
bounded local `OWNED`, add a dense Python/Lustre path, introduce a collective,
or close a scale group from launched-rank state or merely because two
contributions arrived. Until the versioned extension and V21S01–V21S17 gates
pass, native v1 remains fresh-only compatibility behavior and v2.0 remains
historical evidence only.

## Plane boundary and process topology

There is one Python allocation holder, one Python manager and one native data
service per live node, and normally eight Python trainer processes per Frontier
node. The manager is model-free. Trainers own model/inner optimizer state. The
native service owns no training policy, but its pure kernel is the sole
coordination commit authority after external durable-publication evidence
arrives as a typed event. In bounded asynchronous mode, trainers exclusively own all live mutable state;
managers and native services receive only fenced immutable snapshots and
immutable results. No background process may map or inspect a trainer's live
mutable model or optimizer.

| Operation | Python control plane | C++ native data plane |
|---|---|---|
| Publish immutable scheduler-fenced allocation claim | scheduler adapter only | peer control validates claim/fence; C++ rejects stale binds |
| DISCOVER/BOOT/SYNC/READY/ACTIVE/DRAIN/EXPIRE | invokes native peer protocol | peer control is live authority; C++ reports local readiness/faults |
| Open, close, freeze, commit, defer, or abort generation | supplies typed policy/timer/storage events and executes effects | pure kernel is the only authority that changes generation/commit state |
| Choose accepted identities, `Q_min`, `T_min`, deadlines, and owners | supplies configured floors, timer events, and deterministic owner inputs | snapshots READY membership, validates Q/T, freezes identities, and authorizes bounded owner epochs; no policy substitution |
| Export trainer delta and local weight | invokes stable C ABI | maps, validates, and reduces native buffers |
| Endpoint publication and route installation | exchanges opaque endpoint records | creates endpoint, validates and installs routes |
| Dense contribution/aggregate/replay/redistribution bytes | MUST NOT carry them over TCP or serialize Python objects | sole owner |
| Checkpoint cadence, publisher, outer state, immutable manifest/receipt | Python policy/writer | peer agrees commit and returns fenced aggregate/result digests |
| Slurm signals/restarts | sole owner | bounded drain/exit on local command or signal |

```text
immutable claim/commit/checkpoint chain <-> Python policy/publisher
                           |
              native in-memory peer control
                           |
  +------------------------+------------------------+
  | node A                                          | node B
  | Python manager -- local C ABI -- ndp service    | ...
  |      |                         || FI_EP_RDM/cxi  |
  |  trainers 0..7 -- XPMEM/memfd ||==============||
  +-------------------------------------------------+
```

No data-plane step waits for a launched rank count. A manager may wait for a
finite set of owners or frozen contributions named in one attempt, each under
its own deadline; loss produces a control-plane replan or abort, not an
all-rank rendezvous.

## Normative reference A — service lifecycle and generation protocol

The process state machine is normative. Commands not listed for the current
state return `NDP_ESTATE`; they do not cause an implicit transition.

| State | Entry | Allowed transition and result |
|---|---|---|
| `STARTING` | process start | initialize local control socket -> `CONTROL_BOUND`; fatal local setup -> `FAULT` |
| `CONTROL_BOUND` | socket mode `0600`, peer credentials enabled | successful provider selection, endpoint, AV, CQs, and registered pools -> `FABRIC_READY`; otherwise `FAULT` |
| `FABRIC_READY` | endpoint record available | current fenced controller binds -> `IDLE`; newer fence aborts any older volatile state first |
| `IDLE` | no open attempt | `INSTALL_GENERATION` with valid bounds -> `LOCAL_COLLECT`; `DRAIN` -> `DRAINING` |
| `LOCAL_COLLECT` | layout and eligible local trainers installed | all selected local buffers validated/reduced -> `PREPARED`; local deadline/fault -> `ABORTING` |
| `PREPARED` | node numerator, weight, contribution root retained | Python `FREEZE` including this identity -> `FROZEN`; exclusion/abort/newer fence -> `ABORTING` |
| `FROZEN` | immutable accepted set, endpoint set, owner map, and owner epoch installed | credit-driven sends/receives -> `TRANSFERRING`; invalid plan -> `ABORTING` |
| `TRANSFERRING` | frozen payload movement/replay | all locally owned shards applied exactly once -> `OWNED_READY`; owner/sender loss -> bounded `REASSIGN` or `ABORTING` |
| `OWNED_READY` | owned numerators and roots complete | `FINALIZE_OWNERS` divides by global weight -> `REDISTRIBUTING`; disagreement/missing shard -> `ABORTING` |
| `REDISTRIBUTING` | owner results immutable in memory | every required local aggregate shard received/verified -> `RESULT_READY`; deadline/owner loss -> reassign or `ABORTING` |
| `RESULT_READY` | read-only aggregate view and root digest available | Python reports durable commit -> `COMMITTED`; generation rejection -> `ABORTING`; newer fence -> `ABORTING` |
| `COMMITTED` | current fence/generation recorded | release after all local view handles close -> `IDLE` |
| `ABORTING` | no new sends/admission | cancel routes, drain CQs, retain only permitted replay evidence -> `ABORTED` -> `IDLE` |
| `DRAINING` | no new generation accepted | finish/cancel bounded local operations, unregister/detach -> `STOPPED` |
| `FAULT` | unrecoverable service-local invariant/provider failure | report and exit nonzero; never publish or call an allocation abort |

Generation closure is a two-phase control/data operation:

1. A node is `PREPARED` only after all bytes of its node contribution exist in
   replayable native memory, every local input was finite and checksummed, and
   its metadata/root/weight are reported to Python. “Complete contribution” in
   the compute-pool authority means this locally complete, retained state; a
   metadata promise without the retained bytes is not complete.
2. Python reports READY/expiry and finite/deadline observations with configured
   `Q_min`/`T_min` as typed events. The pure native kernel snapshots leased
   READY identities and freezes a deterministic complete set. Owners accept
   dense bytes only for that exact set. If any frozen payload cannot finish,
   the kernel authorizes bounded reassignment or abort with no publication.
3. Each owner reports `OWNED_READY` independently. Python verifies complete
   shard coverage and sends `FINALIZE_OWNERS`; this is point-to-point control,
   not a collective.
4. Each node fetches result shards independently and exposes `RESULT_READY`.
   Background Python policy materializes and reload-verifies immutable
   publication from the immutable base/result. Native peer exact-once commit
   agreement plus the current-fence digest-linked receipt permits `COMMITTED`.
   A trainer consumes the complete verified result only through an atomic,
   separately bounded apply/swap at a later safe boundary. Neither publication
   nor result readiness gates its next foreground K window.

## Normative reference B — provider and endpoint contract

### Production provider selection

Frontier starts the service with `--provider=cxi --require-provider=cxi` and
records the effective libfabric environment. Native setup MUST:

1. call `fi_allocinfo`; set `ep_attr->type = FI_EP_RDM`, `caps = FI_MSG`,
   `mode = FI_CONTEXT`, and `fabric_attr->prov_name = "cxi"`;
2. call `fi_getinfo(FI_VERSION(1, 18), ...)` and reject zero or multiple
   non-equivalent matches rather than choosing by list order;
3. verify the returned endpoint type is `FI_EP_RDM`, provider is exactly
   `cxi` (not a utility-provider stack whose terminal provider is not CXI),
   `max_msg_size > 320`, and the returned MR mode/caps cover the requested host
   `FI_SEND|FI_RECV` registrations;
4. create one endpoint, one address vector, distinct TX/RX completion queues,
   and a single native progress thread before publishing readiness; and
5. call `fi_getname` and publish the opaque address plus provider/fabric/domain,
   `addr_format`, libfabric API/provider versions, service incarnation, and
   endpoint epoch.

The production launcher MUST fail before model load if these checks fail.
`FI_PROVIDER` from the environment is attested but cannot weaken
`--require-provider=cxi`. `FI_MR_CACHE_MONITOR=kdreg2` MAY be used only if the
returned provider facts and the registered-pool tests pass; the service does
not depend on implicit registration caching for correctness.

Local CI MAY use explicit `tcp;ofi_rxm` or `shm` where they provide
`FI_EP_RDM`. Telemetry then says `production_provider=false`. A gate artifact
from any provider other than exact `cxi` cannot authorize a Frontier model job.

### Endpoint exchange

The endpoint record is bounded to 4,096 bytes and has this no-padding encoding:

```text
run_key[16], fence_epoch:u64, worker_key[16], incarnation[16],
endpoint_epoch:u64, expires_unix_ns:u64,
provider_name_len:u16, provider_name[UTF-8],
fabric_name_len:u16, fabric_name[UTF-8],
domain_name_len:u16, domain_name[UTF-8],
addr_format:u32, endpoint_name_len:u16, endpoint_name[opaque],
record_sha256[32]
```

All integers in data-plane records are little-endian. Strings are length-
prefixed UTF-8 without NUL. `record_sha256` covers the domain separator
`"emender-ndp-endpoint-v1\0"` followed by every preceding encoded byte.

The service gives its record to its local Python manager. The allocation holder
publishes it with the READY lease. Python distributes only current-fence,
unexpired records to participating services. A service validates the digest,
exact identities, provider policy, expiry, and monotonic endpoint epoch before
`fi_av_insert`. `fi_av_insert` failure is a route failure. Route deletion and
expiry are local and never wait for another endpoint. G1 admission requires
node clocks to agree within 250 ms; a receiver uses the earlier of the endpoint
expiry, the frame deadline, and its locally installed monotonic stage deadline.

## Normative reference C — local trainer handoff

### Primary XPMEM path

Each trainer allocates the final contiguous CPU delta buffer before producing
the delta. The v1 buffer is little-endian IEEE binary32, bfloat16, or binary64
as declared by the immutable layout; conversion occurs in native code. The
trainer exports the already-produced address range with `xpmem_make`, and the
thin native client sends the segment ID, offset, length, handle generation,
layout digest, trainer identity, and weight over the local seqpacket socket.
The service uses `xpmem_get`/`xpmem_attach`, reads it without modification, and
detaches/releases it immediately after local reduction and checksum. The
trainer MUST keep the allocation stable until `BUFFER_RELEASED`.

For bounded asynchronous training, producing the delta is the local immutable
snapshot operation: the trainer fences optimizer mutation at one safe K
boundary and fills or exposes a preallocated double buffer, copy-on-write view,
or equivalent bounded snapshot before sealing it. Exporting live parameter
storage, copying it concurrently with an optimizer step, or letting the
service reread it after foreground mutation resumes is nonconforming.

### Required memfd fallback

If XPMEM is unavailable, the service allocates a sealed-size `memfd`, sizes it
once with `ftruncate`, and passes a duplicate descriptor using `SCM_RIGHTS`.
The trainer maps the descriptor and produces directly into that mapping. No
intermediate tensor-sized Python `bytes`, pipe, Unix-socket payload, or file
copy is allowed. After production, the trainer marks the buffer read-only to
the protocol; the service maps the same pages and reduces them. The descriptor
is unlinked by construction and closed after `BUFFER_RELEASED`.

Snapshot admission ends when the complete fenced descriptor becomes `OWNED`.
From that point, discovery, quorum, credit waits, publication, hashing,
network, reduction, validation, checkpoint I/O, and result readiness execute
only in background workers. Capacity exhaustion is an explicit
skip/replace/defer outcome and never extends the snapshot pause into a trainer
wait.

In both paths, “directly” means one producer fill of the trainer's delta is
unavoidable, but handoff adds **zero** full-layout writes. In particular the
steady hot path MUST report:

- `trainer_spool_files = 0`, `trainer_spool_bytes = 0`;
- `python_dense_socket_bytes = 0`;
- `handoff_full_copy_bytes = 0`; and
- at most one mapped trainer contribution per local reduction lane.

The service reduces trainers in `trainer_key` byte order and releases each
input before attaching the next when possible. It maintains one node numerator,
not eight trainer-sized service copies. A trainer exit before release invalidates
that local contribution; Python may exclude/restart it under the generation
deadline.

### Disk replay fallback

Disk is not a handoff. An explicitly configured recovery mode MAY write the
single already-reduced node numerator to node-local NVMe after `PREPARED`.
Before opening it, the service must prove by `realpath`, mount identity, and
`statfs` policy that the target is node-local and not Lustre/NFS/Orion/home.
The journal is exactly one open generation, no larger than `layout_bytes + 1
MiB`, has a checksummed completion marker, and is deleted on commit, abort,
fence change, or retention expiry. Default production is memory replay with
`disk_replay_bytes = 0`. Eight trainer files are forbidden even in fallback.

## Normative reference D — layout and exact weighted reduction

### Canonical layout

The layout descriptor is at most 16 MiB and is passed by read-only `memfd`, not
re-parsed from a mutable filesystem path. Its no-padding byte encoding is:

```text
magic[8] = "NDPLAY1\0"
record_count:u32, source_dtype_mask:u32, total_elements:u64,
payload_max:u64, shard_count:u32, reserved:u32=0
for records sorted by raw UTF-8 name bytes:
  name_len:u16, name[name_len], source_dtype:u16, ndim:u16,
  dims[ndim]:u64, element_offset:u64, element_count:u64
```

Names are nonempty valid UTF-8 without NUL and unique; `ndim <= 16`; dimensions
and ranges are checked for overflow, non-overlap, contiguity, and exact
`total_elements`. Dtype codes are `1=binary32`, `2=bfloat16`, and
`3=binary64`. `layout_digest = SHA256("emender-ndp-layout-v1\0" || descriptor)`.
Payloads are always converted to float64 numerators; `layout_bytes =
total_elements * 8` must not overflow and must satisfy the configured bounds.
`payload_max` is positive, 8-byte aligned, and no greater than 64 MiB;
`shard_count` must equal `ceil(layout_bytes/payload_max)`.

The retained E97 full layout is normative gate input:

```text
total_elements = 688,346,312
layout_bytes   = 5,506,770,496
payload_max    = 67,108,864 (64 MiB)
shard_count    = 83 (82 full shards plus 3,843,648 bytes)
```

### Identities and weights

Python assigns `run_key` and stable `worker_key` as opaque 128-bit values and
records their full string mapping in the committed manifest. Each service boot
uses a cryptographically random 128-bit incarnation. A contribution identity
is the following 76-byte no-padding value:

```text
run_key[16] | fence_epoch:u64 | generation:u64 | attempt:u32 |
worker_key[16] | incarnation[16] | contribution_seq:u64
```

`contribution_digest = SHA256("emender-ndp-contribution-v1\0" || identity)`.
The positive integer weight is not part of the identity: it is receipt-bound
metadata so reuse of an identity with a different weight is a detectable
conflict. Each local weight MUST be in `[1, 2^53-1]`; the frozen total weight
MUST fit unsigned 63 bits. Sums use checked unsigned 64-bit integer arithmetic.

Python also assigns every local trainer an opaque 128-bit `trainer_key`; every
trainer boot has a random 128-bit incarnation and a monotonically increasing
submission sequence. `local_set_digest` is SHA-256 over the sorted no-padding
records `(trainer_key[16], trainer_incarnation[16], seq:u64, weight:u64,
source_buffer_sha256[32])`. It binds the trainers and weights that produced the
node contribution and is retained in the Python generation manifest.

### Reference arithmetic

“Exact weighted” in v1 means exact integer token/sample accounting and bitwise
execution of this specified IEEE-754 algorithm; it does not claim exact real-
number summation. Implementations set `FE_TONEAREST`, compile the reduction
translation unit with `-fno-fast-math -ffp-contract=off`, and reject a platform
that does not provide IEC 60559 binary64.

For trainer `t`, convert each source element once to binary64 using round-to-
nearest, ties-to-even. In ascending `trainer_key` order:

```text
term_t = round_binary64(delta_t * round_binary64(uint64_weight_t))
node_numerator = round_binary64(node_numerator + term_t)
node_weight = checked_integer_sum(weight_t)
```

The node sends numerator shards, not a rounded local mean. For shard `j`, the
global contribution order is ascending by the tuple
`(SHA256("emender-ndp-order-v1\0" || layout_digest || u32le(j) ||
contribution_digest), contribution_digest)`. Owners grant application credit
in exactly that order, allowing different shards to start with different
contributors while preserving deterministic per-element order:

```text
global_numerator_j = repeated round_binary64(global_numerator_j + node_numerator_ij)
global_weight = checked_integer_sum(node_weight_i)
delta_j = round_binary64(global_numerator_j / round_binary64(global_weight))
```

Every conversion, term, partial sum, and quotient is checked finite. An owner
never applies a contribution outside the frozen set or out of its shard order.
SIMD is allowed across elements, not across contribution order. The reference
test emits the exact little-endian float64 bytes and SHA-256 for every shard;
zero absolute tolerance is required for the synthetic v1 reference. Restoring
the recorded model dtype is a later local apply operation and uses PyTorch's
documented cast; it is not part of the wire reduction.

## Normative reference E — wire protocol

### Fixed header

Each RDM message is one 320-byte header and, for `CONTRIBUTION_DATA` or
`RESULT_DATA`, exactly `payload_bytes` contiguous bytes. Header-only messages
may use `payload_bytes` and `payload_digest` to identify the data being granted,
requested, or acknowledged; their actual body length is zero. Message-type
validation occurs before the receiver trusts either interpretation.
The header has no compiler padding and is serialized field-by-field; a C/C++
`reinterpret_cast` of a host struct is nonconforming.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | magic bytes `45 4d 4e 44 50 31 00 00` (`EMNDP1\0\0`) |
| 8 | 2 | protocol major, `1` |
| 10 | 2 | protocol minor, `0` |
| 12 | 2 | message type |
| 14 | 2 | flags; unknown set bits reject |
| 16 | 4 | `header_bytes`, exactly `320` |
| 20 | 4 | reserved, zero |
| 24 | 16 | `run_key` |
| 40 | 8 | `fence_epoch` |
| 48 | 8 | `generation` |
| 56 | 4 | `attempt` |
| 60 | 4 | `shard_id` (`0xffffffff` when not shard-specific) |
| 64 | 8 | `owner_epoch` |
| 72 | 8 | `contribution_seq` |
| 80 | 16 | `worker_key` |
| 96 | 16 | `incarnation` |
| 112 | 32 | `layout_digest` |
| 144 | 32 | `base_digest` |
| 176 | 32 | `payload_digest` (SHA-256 of payload; SHA-256 of empty for no payload) |
| 208 | 32 | `contribution_digest` (zero only for route-wide control) |
| 240 | 8 | `payload_offset` in canonical float64 stream |
| 248 | 8 | `payload_bytes` |
| 256 | 8 | `shard_bytes` |
| 264 | 8 | positive contribution/global weight as applicable |
| 272 | 8 | sender-monotonic `message_seq` |
| 280 | 8 | absolute `deadline_unix_ns` |
| 288 | 8 | absolute advertised `credit_bytes` |
| 296 | 4 | `chunk_index` |
| 300 | 4 | `chunk_count` |
| 304 | 4 | status code |
| 308 | 4 | reason code |
| 312 | 4 | CRC32C of bytes 0–311 with this field absent |
| 316 | 4 | reserved, zero |

V1 defines no flag bits, so `flags` is zero. The CRC definition is Castagnoli
polynomial `0x1EDC6F41`, initial/final XOR
`0xffffffff`, over the serialized bytes at offsets 0–311. Receivers validate
magic/version/length/reserved/CRC and all configured byte bounds before using
any field or allocating memory. Major mismatch rejects. A higher minor is
accepted only when all unknown flag/message/extension fields are absent.

Message types are:

| Value | Name | Payload / semantics |
|---:|---|---|
| 1 | `ROUTE_PROBE` | empty; validates current route identity |
| 2 | `ROUTE_PROBE_ACK` | empty; echoes route and receiver endpoint epoch |
| 3 | `CREDIT` | empty; absolute credit epoch/bytes and permitted shard ordinal |
| 4 | `CONTRIBUTION_DATA` | one complete float64 numerator shard |
| 5 | `RECEIPT` | empty; exact once-only application result |
| 6 | `RESULT_ANNOUNCE` | empty; finalized owned-shard root |
| 7 | `FETCH` | empty; requests one finalized result shard |
| 8 | `RESULT_DATA` | one complete float64 result shard |
| 9 | `CANCEL` | empty; stops the named attempt/owner epoch |
| 10 | `GOODBYE` | empty; advisory route drain, never a barrier |

For `CREDIT`, `message_seq` is the monotonic credit epoch, `status/reason` are
zero, and `(shard_id, contribution_digest)` names the only payload currently
permitted on that shard; both sides derive its ordinal from the frozen plan.
For `CONTRIBUTION_DATA`,
`worker_key/incarnation` name the contributor. For `RECEIPT`, those fields name
the receiving owner, `contribution_seq` echoes the source sequence, and the
payload offset/length/digest fields acknowledge the original data despite the
empty body. For `RESULT_ANNOUNCE`, `payload_digest` is the owner's canonical
owned-shard root. `FETCH` names the expected result shard/digest; `RESULT_DATA`
contains it. The fabric source address must map to the current endpoint record
for the message's sender role.

Wire status values are `0=NONE/REQUEST`, `1=APPLIED`, `2=DUPLICATE`,
`3=FINALIZED`, `4=REJECTED`, and `5=RETRYABLE`. Reason values are `0=NONE`,
`1=STALE_FENCE`, `2=STALE_GENERATION_OR_ATTEMPT`, `3=STALE_OWNER_EPOCH`,
`4=NOT_ACCEPTED`, `5=LAYOUT_OR_BASE`, `6=BYTE_BOUNDS`, `7=CHECKSUM`,
`8=NONFINITE`, `9=CONFLICT`, `10=NO_CREDIT`, `11=DEADLINE`, `12=ROUTE`,
`13=PROVIDER`, and `14=SHUTDOWN`. Unknown values reject as a protocol error.

V1 has one frame per shard, so `chunk_index == shard_id`, `chunk_count ==
shard_count`, `payload_offset == shard_id * payload_max`, and `payload_bytes ==
shard_bytes <= payload_max`. A provider whose `max_msg_size` cannot hold
`320 + payload_max` lowers `payload_max` to the greatest 8-byte-aligned value
not above 64 MiB and recomputes the layout digest/shard count before model load.
It cannot change framing after generation installation.

### Checksums, receipt identity, and conflict handling

The contribution root is:

```text
SHA256("emender-ndp-root-v1\0" || contribution_digest ||
       u64le(weight) || local_set_digest || layout_digest || base_digest ||
       concat(shard_id:u32le, shard_bytes:u64le, payload_sha256))
```

with shards in ascending ID order. The owner receipt key is
`(run_key,fence,generation,attempt,owner_epoch,contribution_digest,shard_id,
payload_offset,payload_bytes,payload_digest,weight)`. A receipt includes that
key, receiver identity/incarnation/endpoint epoch, the deterministic
application ordinal, status/reason, and `receipt_digest =
SHA256("emender-ndp-receipt-v1\0" || encoded_receipt)`. The no-padding
`encoded_receipt` is, in order:

```text
run_key[16], fence:u64, generation:u64, attempt:u32, owner_epoch:u64,
contribution_digest[32], contribution_seq:u64, shard_id:u32,
payload_offset:u64, payload_bytes:u64, payload_digest[32], weight:u64,
receiver_worker_key[16], receiver_incarnation[16], receiver_endpoint_epoch:u64,
application_ordinal:u32, status:u32, reason:u32
```

The receipt digest is recomputed from header fields, the frozen plan, and the
installed endpoint record; it does not require a receipt body.

After division, an owner announces:

```text
owner_result_root = SHA256("emender-ndp-owner-result-v1\0" ||
  run_key || u64le(fence) || u64le(generation) || u32le(attempt) ||
  u64le(owner_epoch) || owner_worker_key || owner_incarnation ||
  concat(assigned_shard_id:u32le, shard_bytes:u64le, result_payload_sha256))
```

The canonical full result used by every node and checkpoint handoff is:

```text
result_root = SHA256("emender-ndp-result-v1\0" || run_key || u64le(fence) ||
  u64le(generation) || u32le(attempt) || u64le(owner_epoch) ||
  layout_digest || base_digest || u64le(global_weight) ||
  concat(all shard_id:u32le, shard_bytes:u64le, result_payload_sha256))
```

Assigned/all shard tuples are in ascending shard ID order. Python accepts owner
announcements only when their disjoint assigned sets exactly cover the plan;
every node recomputes the same full `result_root` after redistribution.

An owner maintains a compact key-to-receipt ledger through commit/abort and for
at most two generation slots. First valid receipt application adds the
numerator exactly once. An identical replay returns the identical receipt and
does not add. Reuse of the logical identity with any different bound field is
`REJECT_CONFLICT`, quarantines the route for the attempt, and is reported to
Python. Checksum, nonfinite, layout, offset, weight, fence, generation, owner,
or accepted-set failure is a rejecting receipt and cannot return credit as a
successful application.

### Credits and registered pools

All data frames use memory registered once during `CONTROL_BOUND`. The service
allocates 2 MiB-aligned host slabs and registers fixed TX and RX slots. The
default is four TX and four RX slots, each `320 + payload_max`; the configured
range is 1–16 slots. Accumulators and aggregate buffers are allocated and
bounded before `INSTALL_GENERATION`; providers requiring their registration
register them once for the attempt, never once per message.

`CREDIT` advertises an absolute monotonically versioned byte/slot allowance for
one receiver/owner epoch. The receiver grants only the next deterministic
contribution ordinal for an active shard and never grants total unreceipted
payload above `rx_slots * payload_max`. The sender additionally caps all routes
by its global TX pool. `FI_EAGAIN` drives progress until the operation deadline;
it does not allocate a new slot. Fabric completion releases only the fabric TX
slot after the provider permits reuse. Logical credit and replay-source release
occur only on a valid application `RECEIPT`.

The sender retains its prepared node numerator until Python reports that all
owner shards are complete or the generation terminates. Result owners retain
final shards through durable commit/abort. This separates bounded network
staging from bounded semantic replay.

## Normative reference F — bounds and admission

V1 hard bounds are checked with overflow-safe arithmetic before generation
admission. A deployment may configure lower values, never higher ones without
a protocol-major review.

| Item | V1 hard bound | Frontier v1 default |
|---|---:|---:|
| layout descriptor | 16 MiB | exact descriptor |
| float64 layout payload | 16 GiB | 5,506,770,496 bytes for E97 |
| frame header | 320 bytes | 320 bytes |
| frame payload | 64 MiB | 64 MiB if `cxi max_msg_size` permits |
| shards per layout | 256 | 83 for E97 |
| frozen node contributions | 4,096 | active READY snapshot/policy |
| local trainer buffers | 64 | 8 |
| endpoint record / local control packet | 4 KiB / 64 KiB | same |
| TX/RX registered slots | 16 each | 4 each |
| retained open generations | 1 plus compact prior receipt slot | 1 |
| owner epochs | initial plus 2 reassignments | same |
| sender replay payload | initial send plus at most `2 * layout_bytes` replay | same |
| disk replay | one node numerator, `layout_bytes + 1 MiB` | disabled |

For `L=layout_bytes`, `C=payload_max`, `K=registered slots`, `A=bytes of shards
assigned to this owner`, and `R=receipt_ledger_bound`, the service preflight
requires room for at most:

```text
2*L + A + 2*K*(C + 320) + R + 64 MiB
R = frozen_contributions * shard_count * 128 bytes
A <= ceil(L / owner_count) + C
```

The `2*L` terms are the prepared numerator and node aggregate; an implementation
may safely reuse them after the corresponding replay boundary but cannot admit
based on hoped-for reuse. Frontier E97 requires at least two owners and
`max_owner_accumulator_bytes >= ceil(L/owner_count)+C`. A plan exceeding the
configured resident-memory budget, credit pools, receipt bound, deadline, or
owner bound returns `NDP_EBOUNDS` before attaching trainer buffers.

Logical payload counters include same-service delivery so gates remain
comparable; physical `fabric_tx_payload_bytes` is separate and may be lower.
For two full E97 node contributions the required logical contribution bytes are
exactly `2*L = 11,013,540,992`, and redistribution to two nodes is exactly the
same value.

## Normative reference G — replay, reassignment, redistribution, and failure

### Bounded replay and owner reassignment

A missing receipt may retry the identical key on the same owner with the same
`message_seq` only until the route deadline. The sender never invents a new
contribution sequence for a retry. Duplicate delivery is resolved by the
receipt ledger.

On owner failure before commit, Python removes the expired incarnation,
increments `owner_epoch`, and deterministically remaps only incomplete shards
over the remaining current-fence READY native endpoints. The mapping is
round-robin from `SHA256(run_key || u64le(fence) || u64le(generation) ||
u32le(attempt) || u64le(owner_epoch))` over sorted worker keys: if `h` is the
unsigned little-endian value of the first eight digest bytes, shard `j` is
owned by `owners[(h % owner_count + j) % owner_count]`. Every surviving service
receives the same immutable plan digest. Old-owner receipts do not satisfy a
new owner epoch. Senders replay affected shards from their retained numerator;
completed unaffected shards remain valid only if named by the new plan.

There are at most two reassignments and at most `2*L` replay payload bytes per
sender beyond the initial send. The transfer/replay deadline is not reset by
reassignment. If no plan satisfies the owner count/memory bound, a frozen
sender disappears without a valid replay source, or any bound/deadline is
exhausted, Python aborts the attempt. No partial aggregate is published.

### Redistribution

After all owners finalize, each node independently fetches every result shard
from its current owner with the same framing, checksums, credits, and deadlines.
The service writes into exactly one preallocated node aggregate buffer at the
canonical offset. A per-shard received bitmap and root verification prevent
holes/duplicates. Only a complete matching root becomes `RESULT_READY`.

The service exports read-only XPMEM/memfd views of that one buffer to local
trainers. Each trainer necessarily reads/applies the aggregate to its own model,
but no trainer-sized aggregate file, Python TCP copy, or per-trainer service
copy is created. The aggregate remains pinned until every exported handle is
released or the bounded apply deadline aborts the generation.

### Failure table

| Failure | Native response | Python/control response |
|---|---|---|
| trainer dies before local release | detach safely, reject local input, keep other lanes | exclude/restart under deadline; new trainer incarnation |
| service dies before `PREPARED` | contribution absent | restart service/new incarnation or exclude |
| service dies after `PREPARED` | memory replay lost; optional valid NVMe journal may restore only exact fence/attempt | rebind/replay within deadline or abort frozen attempt |
| route expiry/CQ error/endpoint loss | stop credit, report route error, preserve sender source | expire membership; reassign owner or abort |
| `FI_EAGAIN` | progress/retry without allocating | no membership change until route deadline |
| CQ overflow, truncation, MR/access error | route/service fault; never trust affected frame | reassign/abort; capture provider error |
| stale fence/generation/attempt/owner epoch | rejecting receipt; no accumulator mutation | fence/expire source, direct catch-up |
| duplicate identical frame | return identical receipt, no second add | none |
| conflicting duplicate or bad checksum/nonfinite | reject/quarantine route | record evidence; exclude/abort by policy |
| owner loss after receipt | old receipt is valid only if new plan retains that owner result; otherwise replay new owner epoch | bounded reassignment; no partial commit |
| redistribution owner loss | preserve prepared/result buffers and report missing result | reassign/recompute within same absolute deadline or abort |
| claim/fence superseded | atomically reject old commands/frames, cancel old routes, release volatile state | newer peer authority controls next attempt; immutable base receipt selects restart state |
| checkpoint/publication failure | keep `RESULT_READY` only to deadline, then abort/release | do not acknowledge a commit receipt or next-generation READY |
| SIGTERM | stop admission and enter bounded `DRAINING` | checkpoint only previously/current durably valid state per policy |

Libfabric errors MUST NOT call `abort()`, `MPI_Abort`, signal peer processes, or
kill the allocation. A service may exit nonzero after reporting a terminal
local fault; Python supervision decides restart/exclusion.

## Normative reference H — stable Python/native ABI

The installed public header is `include/emender/ndp.h` and the shared library
SONAME is `libemender_ndp.so.1`. Only `extern "C"` symbols, fixed-width integer
types, opaque 64-bit handles, and explicit byte spans cross the boundary. No
C++ STL, CPython object, PyTorch ABI, exceptions, callbacks, or borrowed output
pointers cross it.

All input/output structs begin with `struct_size:u32` and `abi_version:u32`.
`NDP_ABI_V1 = 0x00010000`. Callers zero unknown tail bytes. A v1 library accepts
a larger struct only when the v1 prefix is valid; it never reads the tail. A
major mismatch returns `NDP_EVERSION`. Input spans are copied before the call
returns. Returned file descriptors are `CLOEXEC` duplicates owned by the
caller. Handles are process-local, never serialized, and become invalid after
release/close.

Required v1 symbols are:

```c
uint32_t ndp_abi_version(void);
const char *ndp_error_string(int code);                 /* static UTF-8 */

int ndp_client_open_v1(const struct ndp_open_v1 *, ndp_client_t *);
int ndp_client_poll_fd_v1(ndp_client_t, int *dup_fd);
int ndp_client_close_v1(ndp_client_t);

int ndp_layout_install_v1(ndp_client_t, const struct ndp_layout_v1 *);
int ndp_buffer_register_v1(ndp_client_t, const struct ndp_buffer_v1 *,
                           ndp_buffer_t *);
int ndp_buffer_allocate_v1(ndp_client_t, const struct ndp_alloc_v1 *,
                           ndp_buffer_t *, int *dup_fd);
int ndp_buffer_seal_v1(ndp_client_t, ndp_buffer_t);
int ndp_buffer_release_v1(ndp_client_t, ndp_buffer_t);

int ndp_submit_local_v1(ndp_client_t, const struct ndp_submit_v1 *,
                        ndp_op_t *);
int ndp_control_v1(ndp_client_t, const struct ndp_control_v1 *, ndp_op_t *);
int ndp_poll_v1(ndp_client_t, struct ndp_event_v1 *events,
                uint32_t capacity, uint32_t *count, int timeout_ms);
int ndp_result_view_v1(ndp_client_t, ndp_op_t,
                       struct ndp_result_v1 *, ndp_buffer_t *, int *dup_fd);
int ndp_op_release_v1(ndp_client_t, ndp_op_t);
```

V1 is a Linux LP64, little-endian ABI with natural C alignment capped at 8
bytes; the installed header contains `static_assert`/`_Static_assert` checks for
the declared sizes. The normative struct prefixes are:

```c
typedef uint64_t ndp_client_t;
typedef uint64_t ndp_buffer_t;
typedef uint64_t ndp_op_t;

struct ndp_open_v1 {                 /* sizeof = 224 */
  uint32_t struct_size, abi_version, role, flags;
  uint32_t socket_path_len; uint8_t socket_path[108];
  uint8_t run_key[16]; uint64_t fence_epoch;
  uint8_t worker_key[16], incarnation[16], admission_token[32];
  uint64_t deadline_unix_ns;
};
struct ndp_layout_v1 {               /* sizeof = 56 */
  uint32_t struct_size, abi_version; int32_t descriptor_fd;
  uint32_t reserved0; uint64_t descriptor_bytes; uint8_t layout_digest[32];
};
struct ndp_buffer_v1 {               /* sizeof = 88 */
  uint32_t struct_size, abi_version, kind, flags;
  uint64_t address_or_segid, offset, length, handle_generation;
  int32_t fd; uint32_t reserved0; uint8_t layout_digest[32];
};
struct ndp_alloc_v1 {                /* sizeof = 32 */
  uint32_t struct_size, abi_version, flags, reserved0;
  uint64_t bytes, deadline_unix_ns;
};
struct ndp_submit_v1 {               /* sizeof = 128 */
  uint32_t struct_size, abi_version; uint64_t buffer;
  uint8_t trainer_key[16], trainer_incarnation[16];
  uint64_t submission_seq, weight, element_offset, element_count;
  uint32_t source_dtype, flags; uint64_t deadline_unix_ns;
  uint8_t source_buffer_sha256[32];
};
struct ndp_control_v1 {              /* sizeof = 216 */
  uint32_t struct_size, abi_version, command, flags;
  uint8_t run_key[16]; uint64_t fence_epoch, generation;
  uint32_t attempt, metadata_kind; uint64_t owner_epoch, deadline_unix_ns;
  int32_t metadata_fd; uint32_t reserved0; uint64_t metadata_bytes;
  uint8_t layout_digest[32], base_digest[32], plan_digest[32], metadata_sha256[32];
};
struct ndp_event_v1 {                /* sizeof = 96 */
  uint32_t struct_size, abi_version, event, status, reason, state;
  uint64_t op, generation; uint32_t attempt, shard_id;
  uint64_t owner_epoch, logical_bytes; uint8_t detail_digest[32];
};
struct ndp_result_v1 {               /* sizeof = 168 */
  uint32_t struct_size, abi_version, flags, dtype;
  uint8_t run_key[16]; uint64_t fence_epoch, generation;
  uint32_t attempt, reserved0;
  uint8_t layout_digest[32], base_digest[32], result_root[32];
  uint64_t global_weight, result_bytes;
};
```

The compiler-checked definition is the authority if a displayed size and the
listed fields ever disagree; such disagreement is a documentation bug and
blocks release. `role` is `1=TRAINER`, `2=CONTROLLER`. Buffer flags are
`READ=1`, `WRITE=2`; result views are `READ` only. For XPMEM registration,
`address_or_segid` is the caller address and the client library creates and
passes the segment; for memfd it is zero and `fd` is required. Unused integer,
array tail, and reserved fields are zero.

Every metadata fd begins with this no-padding stream and ends exactly after its
SHA-256; extra bytes reject:

```text
magic[8]="NDPMD1\0\0", kind:u16, version:u16=1, record_count:u32,
payload_bytes:u64, payload[payload_bytes],
sha256[32] = SHA256("emender-ndp-metadata-v1\0" || all preceding bytes)
```

Kind `1=ENDPOINTS` has `record_count` repetitions of
`record_bytes:u32 | endpoint_record[record_bytes]` using the endpoint encoding
above. Kind `2=ACCEPTED_SET` has 180-byte records in ascending
`contribution_digest` order:

```text
contribution_identity[76], contribution_digest[32], weight:u64,
local_set_digest[32], contribution_root[32]
```

Kind `3=FROZEN_PLAN` payload is `endpoint_count:u32`, the kind-1 endpoint
records, `accepted_count:u32`, the kind-2 accepted records,
`owner_count:u32`, and sorted `owner_worker_keys[owner_count][16]`. Counts must
match the envelope and hard bounds (`record_count = endpoint_count +
accepted_count + owner_count` for kind 3); weights must sum to the separately
checked global total. `INSTALL_GENERATION` needs no metadata after the layout is
installed; `FREEZE` and `REASSIGN` require kind 3. Other commands require
`metadata_kind=0`, `metadata_fd=-1`, zero bytes/digest. Plan digest is SHA-256
over the complete kind-3 stream plus the control struct's fenced identity,
layout/base digests, owner epoch, and deadline.

`ndp_open_v1` contains role (`TRAINER` or `CONTROLLER`), socket path, run/fence,
worker/incarnation, and a 256-bit allocation admission token delivered by the
post-lease supervisor through a protected local pipe. The service socket is
mode `0600`, validates `SO_PEERCRED` UID, and permits exactly one current-fence
controller. A higher fence supersedes; a lower fence returns `NDP_EFENCE`.

`ndp_buffer_v1` kinds are `NDP_BUFFER_XPMEM_ADDRESS=1` and
`NDP_BUFFER_MEMFD=2`. It contains address/offset/length, access flags, handle
generation, and layout digest. `ndp_submit_v1` contains the complete trainer
identity, buffer handle, source dtype, element range, positive weight, and
absolute deadline. `ndp_control_v1` commands correspond exactly to the state
machine (`BIND_FENCE`, `INSTALL_GENERATION`, `FREEZE`, `REASSIGN`,
`FINALIZE_OWNERS`, `COMMIT`, `ABORT`, `DRAIN`). Variable endpoint/accepted-set
arrays are passed as read-only metadata memfds with the exact v1 encodings
above, never as JSON.

`ndp_poll_v1` is the only potentially blocking call and is capped by
`timeout_ms`; `-1` is forbidden. Events include state, operation handle,
fenced identity, status/reason, byte counters, and optional native buffer
handle. Calls are thread-safe at the client library boundary; event ordering is
per client. The service performs all fabric progress on native threads.

Return codes are `0` success, `1` accepted/in progress, and fixed negative
values: `NDP_EINVAL=-1`, `NDP_EVERSION=-2`, `NDP_ESTATE=-3`, `NDP_EFENCE=-4`,
`NDP_ESTALE=-5`, `NDP_ECONFLICT=-6`, `NDP_ECHECKSUM=-7`,
`NDP_ENONFINITE=-8`, `NDP_EBOUNDS=-9`, `NDP_ECREDIT=-10`,
`NDP_EDEADLINE=-11`, `NDP_EROUTE=-12`, `NDP_EPROVIDER=-13`,
`NDP_ENOMEM=-14`, `NDP_EIO=-15`, and `NDP_ESHUTDOWN=-16`. `errno` is diagnostic
only and never the ABI result.

The Python package `ndm.native_dataplane` is a thin owner of these handles. It
may use a CPython limited-API extension or `ctypes`, but its public Python
objects must expose context managers and must not copy dense data. ABI tests
run a v1 Python wheel against the current v1 service and a current Python
binding against a retained v1 service binary.

## Normative reference I — checkpoint handoff and shutdown

At `RESULT_READY`, native peers agree one current-fence result and Python
chooses one checkpoint publisher. The
publisher obtains a read-only aggregate fd/view plus `(run_key,fence,
generation,attempt,layout_digest,base_digest,result_root,global_weight)`. It
materializes the global delta and required outer state from immutable
base/result inputs, writes and reload-verifies the immutable checkpoint
selected by policy, and appends a digest-linked commit receipt under the
immutable allocation claim. It does not read or mutate a live trainer model or
optimizer. Native peer
control acknowledges exactly that receipt/result/token identity; it never
declares a commit from a fabric receipt. A compatibility `latest.json` may be
written after the receipt but is never read as authority. No database, lock
file, or filesystem heartbeat participates.

For strict-fresh v1, nonpublisher trainers may map the same node aggregate for
local apply under the generation lifecycle. For bounded asynchronous v2.1,
checkpoint/publication is background work on immutable inputs and does not
gate the next K window. Only after the complete result is verified may all
eight trainers atomically apply/swap it at a later safe boundary under the
separate foreground pause bound. A late, absent, invalid, failed, or unready
result is skipped/deferred without a foreground wait. A handle is released
explicitly. Leaked handles expire at the background/apply deadline and make
the result fail closed; the service does not reuse their buffer for a later
generation. No partial result becomes READY or mutates a subset of trainers.

Shutdown sequence is: Python stops new local work; sends `ABORT` for an open
uncommitted attempt or `COMMIT` for already durable state; sends `DRAIN`; the
service stops credit, sends advisory `GOODBYE`, waits at most 30 seconds for
posted completions/handles, calls `fi_cancel` where supported, drains CQ errors,
removes AV entries, closes endpoint/CQs/AV/domain/fabric, deregisters MRs,
detaches XPMEM, closes memfds/socket, and exits. The supervisor sends SIGTERM at
the allocation shutdown deadline and SIGKILL after 45 seconds total. There is
no peer rendezvous or native final barrier.

## Normative reference J — deadlines and telemetry

All deadlines are absolute Unix nanoseconds in commands/frames and monotonic
durations internally. A retry/reassignment does not extend the parent stage.
The two-node qualification defaults are:

| Stage | Default bound |
|---|---:|
| local socket/control bind | 10 s |
| provider/endpoint/registered pools ready | 30 s |
| endpoint record install/probe | 30 s |
| local trainer handoff after trainers report delta ready | 180 s |
| frozen contribution transfer, including replay/reassignment | 180 s |
| owner finalization | 30 s |
| redistribution and aggregate root validation | 180 s |
| background checkpoint handoff/publication | policy bound, at most 180 s for 2-node gate |
| foreground snapshot capture/admission | v2.1 at most 1 s through `OWNED` |
| foreground atomic result apply/swap | v2.1 at most 60 s for all eight trainers |
| drain | 30 s; process kill bound 45 s |

Every service emits JSON Lines schema `emender-native-dataplane-telemetry-v1` to
node-local storage, and Python retains the bounded summary with the generation
manifest. Required fields/counters are:

- build/git/ABI/protocol versions; provider, fabric/domain, endpoint type,
  libfabric API/provider versions, `max_msg_size`, MR mode, effective `FI_*`;
- full fenced identity, endpoint/owner epoch, plan/layout/base/result digests,
  accepted contribution digests and exact weights;
- state transition timestamps, configured/observed deadlines, terminal
  status/reason, CQ/provider error data;
- `logical_contribution_bytes`, `fabric_tx_payload_bytes`,
  `logical_redistribution_bytes`, header/control bytes, throughput and per-stage
  durations;
- TX/RX credits and slot high-water, accumulator/result/resident high-water,
  receipt count, duplicates/conflicts/checksum/nonfinite/stale rejects;
- initial/replay bytes, reassignment count, disk replay bytes/path class,
  prompt released bytes and post-generation resident bytes;
- local handoff kind, mapped bytes, `handoff_full_copy_bytes`, trainer spool
  file/bytes, Python dense socket bytes, open/released view counts; and
- for bounded asynchronous work, causally identified `freeze_snapshot`,
  `snapshot_admission`, `publish_network`, `aggregation`, `checkpoint`,
  `result_wait`, and `apply_swap` intervals plus total foreground idle,
  zero foreground result-wait time, and every-event maximum/p99
  snapshot/admission and apply/swap pauses.

Counter invariants are checked, not merely printed. Successful steady-state
gates require `trainer_spool_bytes=0`, `python_dense_socket_bytes=0`,
`handoff_full_copy_bytes=0`, `disk_replay_bytes=0`, no credit/memory bound
violation, and released logical bytes equal admitted logical bytes.
Checkpoint/restart success, checkpoint count, median cadence, and aggregate
idle are not overlap evidence. A bursty trace with alternating K windows and
approximately 200-second foreground gaps fails even if those summaries pass.

## Normative gate reference — acceptance commands and metrics

The following is the required command/artifact surface for implementation and
qualification tasks. The scripts/executables named here are deliverables of
those downstream tasks. These commands are specifications, not evidence that
this design task submitted a job. **No real model, 4+ node, or production job
is authorized until gate G2's retained artifact matches the exact code,
provider, layout, and configuration being launched.**

### G0: local ABI, math, protocol, and bounded-provider tests

```bash
cmake -S native/dataplane -B build/native-dataplane \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNDP_ENABLE_XPMEM=ON
cmake --build build/native-dataplane --parallel
ctest --test-dir build/native-dataplane --output-on-failure
python3 -m pytest -q \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py
build/native-dataplane/ndp_synthetic_gate \
  --provider 'tcp;ofi_rxm' --nodes 2 --processes-on-one-host \
  --layout-bytes 536870912 --trainers-per-node 8 --weights 3,1000003 \
  --inject duplicate,stale,checksum,credit-stall \
  --json-out artifacts/native-dataplane/g0.json
```

G0 passes only with bitwise Python/C++ reference roots for unequal weights and
arrival permutations; ABI forward/backward tests; exact duplicate/stale/
corrupt results; bounded credit/replay/memory; no collective symbols in the
elastic binaries (`nm -D` scan rejects `MPI_`, `PMPI_`, `torch.distributed`,
and Python socket transport imports); and all steady-state zero-copy/spool
counters above.

### G1: Frontier CXI probe

```bash
sbatch --parsable --nodes=2 --time=00:10:00 \
  --export=ALL,NDP_GATE=cxi-probe,FI_PROVIDER=cxi \
  scripts/frontier/native_dataplane_2n_gate.sbatch
```

G1 uses two service processes and no model. It passes only if both select exact
`cxi`/`FI_EP_RDM`, exchange endpoints through Python membership, move at least
64 GiB logical data in 64 MiB-or-smaller registered frames, show bounded
credits with zero corruption/deadline/CQ errors, survive one route restart, and
retain `gate.json`, provider facts, checksums, and per-node telemetry.

### G2: hard full-layout two-node synthetic gate

```bash
sbatch --parsable --nodes=2 --time=00:20:00 \
  --export=ALL,NDP_GATE=full-layout,NDP_LAYOUT=e97-f64-5506770496,NDP_TRAINERS_PER_NODE=8,NDP_WEIGHTS=1966080,1968000,FI_PROVIDER=cxi \
  scripts/frontier/native_dataplane_2n_gate.sbatch
```

The job generates deterministic buffers; it does not load a model/checkpoint.
All eight lanes are exercised: each node-0 trainer has weight 245,760 and each
node-1 trainer has weight 246,000, giving node weights 1,966,080 and 1,968,000.
After one warm-up it runs three timed generations. Every generation requires:

- two complete node contributions, `global_weight=3,934,080`, 83 shards,
  `logical_contribution_bytes=11,013,540,992`, and
  `logical_redistribution_bytes=11,013,540,992`;
- bitwise result equality with the offline v1 reference and identical roots on
  both nodes; exact `cxi`, registered pools, two distributed owners, and no
  Python dense bytes, trainer files, disk replay, rejects, replay, or leaks;
- observed resident/credit/owner high-water at or below the admitted formulas,
  and released bytes equal logical admitted bytes; and
- median end-to-end transfer-plus-redistribution time no greater than the
  retained Python baseline `98.961446568 s` (at least
  `222,582,457.59 logical B/s` over both directions for the exact byte/time
  constants; the retained report rounds this to 222.6 MB/s), with no timed iteration
  above `118.753735882 s` (20% noise ceiling).

The retained artifact is
`reports/frontier/native-dataplane/<job-id>/full-layout-gate.json` and binds the
source commit, binary SHA-256, provider facts, layout/plan digests, commands,
metrics, and telemetry hashes. Model launchers reject a missing/mismatched
artifact via `NDP_FULL_LAYOUT_GATE_JSON`.

#### G2 retained-artifact ownership

`NDP_ARTIFACT_ROOT` is a shared container, not an invitation for every actor
to create `<job-id>`. Its immutable `ARTIFACT-OWNERSHIP.json` uses schema
`emender-native-g2-artifact-ownership-v1` and assigns these exclusive
namespaces:

| Actor | Sole writable authoritative namespace | Permitted reads and publication rule |
|---|---|---|
| submit/controller monitor | `controller/<payload-job-id>/scheduler-evidence/` | May capture `squeue -o '%i|%T|%P|%q'`, `scontrol`, and reconciliation records. Complete records are published by content-addressed hard link without replacement. It MUST NOT create `<payload-job-id>/`. |
| native G2 batch | `<payload-job-id>/` | Sole writer of job artifacts. On Frontier Lustre it atomically creates a no-replace relative symlink to a batch-owned directory under `.batch-storage/` that already contains `.artifact-owner.json`. Thus neither `mkdir` nor a directory rename can observe and replace an empty authoritative directory. Any existing path fails closed with exit 73 and is never overwritten. |
| scheduler-owned `afterany` collector | `collectors/<collector-job-id>/payload-<payload-job-id>/` | May read the batch directory after dependency release and write only immutable content-addressed hard-link publications in its own directory. It MUST NOT create or mutate the batch or controller directory. |

No actor may pre-create another actor's authoritative root. Legitimate
controller evidence, including prior failed or repeated observations, is
therefore never placed below the batch root and cannot trip its create-once
guard. The canonical submit helper initializes the schema, durably records the
numeric job identity returned by `sbatch`, then immediately captures both
`Partition` and `QOS` outside the batch namespace. Repeated observation and
collector reconciliation are idempotent because equal canonical records have
one digest-derived filename; a new scheduler state creates a new immutable
record rather than overwriting history.

The only supported ownership operations are:

```bash
"$NDP_PYTHON_BIN" scripts/frontier/native_g2_artifact_namespace.py \
  observe-scheduler --artifact-root "$NDP_ARTIFACT_ROOT" \
  --job-id "$JOB_ID" --kind monitor

"$NDP_PYTHON_BIN" scripts/frontier/native_g2_artifact_namespace.py \
  record-collector --artifact-root "$NDP_ARTIFACT_ROOT" \
  --collector-job-id "$SLURM_JOB_ID" --payload-job-id "$PAYLOAD_JOB_ID" \
  --kind terminal --evidence-json '{"dependency":"afterany"}'
```

Callers provide identities and evidence, never destination paths. Direct
operator writes under `NDP_ARTIFACT_ROOT/<job-id>` are conflicting
authoritative artifacts and deliberately remain an exit-73 condition.

### G3: real two-node generation

Only after G2:

```bash
sbatch --parsable --nodes=2 --time=00:20:00 \
  --export=ALL,DILOCO_DATAPLANE=native-cxi,NDP_FULL_LAYOUT_GATE_JSON=/absolute/retained/full-layout-gate.json,FI_PROVIDER=cxi \
  scripts/frontier/resilient_e97_true_2n.sbatch
```

G3 passes with two leased READY node incarnations, all real local trainers using
XPMEM/memfd, one current-fence generation committed at exactly 3,934,080
accepted tokens, finite K40 training, bitwise native/reference aggregate roots,
complete outer state, a reload-verified immutable checkpoint and atomic latest,
the exact full-layout byte counters, all bounds/release invariants, and zero
trainer spool/Python dense/disk replay bytes. A valid data-plane result followed
by checkpoint or clean-shutdown failure is not a pass.

### G4: failure, late join, disappearance, and rejoin

```bash
sbatch --parsable --nodes=2 --time=00:30:00 \
  --export=ALL,NDP_GATE=resilience,NDP_SCENARIOS=delayed-boot,late-join,trainer-loss,owner-loss-after-receipt,duplicate,stale,checksum,rejoin,FI_PROVIDER=cxi \
  scripts/frontier/native_dataplane_2n_gate.sbatch
```

G4 runs at least three generations and passes only when delayed/late peers do
not block the first valid commit; injected owner-role state loss leaves sender
sources alive, increments owner epoch, replays no more than `2*L`, and produces
the same result root; duplicate data is
acknowledged once; stale/corrupt data is rejected without accumulator change;
the old incarnation never contributes after expiry; the rejoined worker has a
new incarnation, synchronizes the last commit, and contributes only to a later
generation; every wait is within its original deadline; and at least one
generation commits while an injected contributor is absent if `Q_min/T_min`
remain satisfied (its still-live native service may remain an owner without
joining the frozen contribution set). Otherwise the attempt must abort with no
partial publication.

### G5: fresh-service and fresh-allocation restart

```bash
seed_job=$(sbatch --parsable --nodes=2 --time=00:20:00 \
  --export=ALL,NDP_GATE=restart-seed,FI_PROVIDER=cxi \
  scripts/frontier/native_dataplane_restart_2n.sbatch)
sbatch --parsable --dependency=afterok:${seed_job} --nodes=2 --time=00:20:00 \
  --export=ALL,NDP_GATE=restart-resume,NDP_SEED_JOB=${seed_job},FI_PROVIDER=cxi \
  scripts/frontier/native_dataplane_restart_2n.sbatch
```

The seed allocation commits and reload-verifies generation 1, then exits. The
resume allocation acquires a strictly higher fence, starts new service/manager
incarnations, loads only the immutable latest state, rejects injected old-fence
frames/receipts/journal data, and commits generation 2 with a continuous token
clock. Artifacts must prove the two Slurm allocations, fence increase,
independent reload, old-frame rejection, no mutation by the stale process, and
bounded shutdown. A same-process “restart” is insufficient.

### G6: direct systems-scale ladder and compiled reference

Each rung is submitted separately only after the preceding rung's artifact is
accepted. The exact command shape is:

```bash
sbatch --parsable --nodes=<8|32|128> --time=00:20:00 \
  --export=ALL,NDP_GATE=scale,NDP_SCALE_NODES=<same>,NDP_LAYOUT=e97-f64-5506770496,FI_PROVIDER=cxi \
  scripts/frontier/native_dataplane_scale_gate.sbatch
```

At every rung, one warm-up plus three timed synthetic generations require all
READY contributions, exact weights/roots, `nodes * 5,506,770,496` logical
contribution and redistribution bytes, no missing/reject/replay in the clean
case, owner count at least two, maximum owner assignment within the formula,
credit/resident bounds, full release, and p50 no slower than
`98.961446568 s`; no sample may exceed 1.20 times its rung median. The gate also
injects one owner loss after the clean measurements and requires bounded
reassignment or a clean no-commit result according to the configured floor.

The 8, 32, and 128 systems rungs are strictly ordered after the current-source
two-node clean, fault/rejoin, and fresh-allocation recovery pass. A real model
scale rung is separately authorized only after its same-size synthetic
artifact and exact immediate real-policy predecessor. Four, 16, and 64 are
not rungs. After 128, 256 is an explicit evidence review only and no G6
256-node runner is authorized. V2.1 additionally requires the
authorization-pinned finite close over the leased READY snapshot defined by
ADR-002: the v1 clean all-READY measurement is a reference, not permission to
wait for launched ranks or close at `Q_min=2`. Convergence/model quality is a
separate study and does not authorize or block G6. For a possible future
256-node proposal (2,048 trainer lanes), the retained fixed-world compiled reference is
`5.304643992334604 s`. Elastic v1 acceptance requires the native clean
reduction-plus-redistribution median no greater than twice that value,
`10.609287984669208 s`; matching or beating `5.304643992334604 s` is the
performance target. Correctness without the 10.609-second cap leaves native
CXI unpromoted and the compiled MPICH path as a fixed-world fallback only; it
does not justify weakening resilience semantics.

The retained compiled reference itself is requalified, without changing its
role, by the downstream compiled-transport task using the existing build and
2048-rank smoke launcher. Its artifact must still show 2,048/2,048 accepted,
zero missing/timed-out/stale updates, 5,506,770,496 aggregate bytes, finite
loss, and the observed merge duration.

## Pure coordination differential

The cross-authority canonical schema, compiled-service adapter, permanent
job-5105811/corner-case corpus, source/runtime call-path audit, and
first-divergence replay contract are specified in
[Native/Lean coordination conformance](NATIVE_LEAN_COORDINATION_CONFORMANCE.md).
That gate calls the production `coordination::step` through the installed
library, metadata RPC, and persistent service. It does not replace the native
trace or extend pure agreement into a dense-path, timing, numerical, Frontier,
or scale claim.

## Conformance and traceability checklist

Every implementation, runner, qualification, or scale task using this design
MUST include the compute-pool conformance checklist in its `## Validation` and
cite applicable matrix IDs R01–R16 plus native IDs NDP01–NDP17. At minimum:

- local implementation: R04, R05, R08–R10, R14, R15 and NDP01, NDP04–NDP06,
  NDP08–NDP10, NDP12, NDP14–NDP16;
- CXI transport: R03, R04, R06, R08, R10, R11, R13–R16 and NDP02, NDP03,
  NDP06–NDP13, NDP16, NDP17;
- two-node/scale runners: R01–R16 and NDP01–NDP17; and
- compiled reference qualification: R05, R10, R15, R16 and NDP02, NDP03,
  NDP05, NDP16, NDP17, explicitly recording its non-elastic fallback status.

A v2.1 asynchronous task MUST additionally cite ADR-002, V21S01–V21S17, and
ISP01–ISP07.
Its native artifact must prove the versioned identity/exact-token/descriptor/
coalescing/correction/mailbox/node-apply extensions above, immutable snapshot
coherence and foreground/background phase timing, while retaining every
applicable NDP01–NDP17 invariant. A scale artifact must also prove the reviewed
leased-READY scale closure. Passing the v1 G2/G3 gate is a prerequisite and
reference, not evidence that v2.1 transport has passed.

The acceptance record must contain exact commands, immutable artifacts, code
and binary digests, provider facts, committed generation/checkpoint evidence,
and an explicit statement that no failure-sensitive all-rank operation or
Python dense TCP path was used by the elastic backend.
