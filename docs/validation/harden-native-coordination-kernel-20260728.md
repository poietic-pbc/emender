# Hardened production native coordination kernel

Date: 2026-07-28
Task: `harden-native-coordination-kernel`
Design authorities:

- [`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
  especially its required conformance checklist and Native data-plane binding.
- [`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md),
  requirements R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07.
- [`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md).
- [`quality-pass-native-first-coordination-20260728.md`](quality-pass-native-first-coordination-20260728.md),
  which routes direct kernel obligations separately from native transport,
  numerical, snapshot-ownership, timing, scheduler, and scale evidence.

## Result

The live native manager now uses one production C++ transition function:

```text
coordination::step(
    const AuthorityState &authoritative_state,
    const Event &event
) noexcept -> Transition {
    state, disposition, effects, trace, pre_state_digest, post_state_digest
}
```

The function is pure and deterministic. It has no sockets, clocks, files,
process control, Slurm calls, dense buffers, or effect execution. Every handler
is total. Expected race and replay inputs return a typed nonfatal disposition
and preserve the state digest. A corrupt authoritative input or a conflicting
exact-once commit/apply authority is the only fatal class.

The actual production path is:

```text
scripts/frontier/resilient_e97_role.py::_native_manager
  -> NativeManagerSession.coordination_authority
  -> NativePoolControlServer (serial event adapter; effects only)
  -> NativeCoordinationAuthority
  -> Client.coordination_step / ndp_coord_step_v1
  -> bounded AF_UNIX RPC CoordinationStep
  -> LocalServiceCore::coordination_step (service mutex)
  -> coordination::step (pure decision)
  -> sole assignment to Service::coordination_state_
  -> typed result/effects + canonical trace
```

`PoolControlServer`, `PeerMembership`, and `GenerationAdmission` remain only
behind the explicitly selected `python-tcp-debug` reference backend. The native
branch does not construct them. A production source assertion permanently
checks this separation.

## Kernel boundary and stable API

The additive `ndp_coord_step_v1` ABI uses fixed-size, versioned records:

- 312-byte event;
- 52,016-byte bounded result;
- no more than 256 members;
- no more than eight explicit effects per transition;
- no more than 4,096 bytes for one canonical trace record.

Every event carries:

- run/job key and allocation fence;
- source/result generation and attempt;
- stable node key, process/cohort incarnation, and monotonic sequence;
- exact tokens and all-eight trainer count where applicable;
- policy, payload, result, contribution/commit/apply receipt, previous receipt,
  and manifest digests where applicable.

Typed events are `RecoverAuthority`, `RecoverNodeApply`, `RecoverPeer`, `Ready`,
`OpenGeneration`, `Contribution`, `CloseGeneration`, `ResultReceipt`, `Commit`,
`NodeApply`, `ExpirePeer`, `OwnerLost`, and `QueryCommit`.

Typed dispositions are `accepted`, `identical-duplicate`,
`conflicting-duplicate`, `stale-fence`, `stale-incarnation`,
`stale-generation`, `generation-closed`, `deferred`,
`retry-next-generation`, `insufficient-cohort`, `corrupt`, `invalid-event`,
and `fatal-invariant`.

Effects are data, not calls from the kernel: bind/sync authority, advertise
READY, start/freeze a generation, acknowledge a receipt, reassign an owner,
announce commit eligibility, publish a commit, record node apply, expire a
peer, retry, and emit the trace. Endpoint validation, lease/deadline clocks,
libfabric, durable receipt loading/publication, cache pruning, and process
supervision remain outside and re-enter only through typed events.

## Authoritative invariants

The production kernel and tests enforce:

1. **Fence and identity.** A configured authority has one nonzero run/fence and
   policy. Wrong run/fence is `stale-fence`; old incarnation/sequence is
   `stale-incarnation`.
2. **Immutable leased READY close.** Open snapshots current READY
   node/incarnation pairs into an ordered cohort. Contributions may only bind a
   snapshot identity. Close never changes that cohort or an accepted receipt.
3. **Monotonic phases.** One bounded active generation moves
   `open -> closed -> committed -> applied`, or `open/closed -> aborted`.
   A higher retry attempt may replace only an aborted attempt. No event rolls
   committed generation, token clock, result, receipt, or applied generation
   backward.
4. **Idempotent receipts.** Byte-identical contribution, result, close, commit,
   READY, recovered apply, node-apply, and owner-loss replays preserve the
   state digest. Lease expiry cannot revoke an already acknowledged
   contribution. Conflicting contribution/result sequences are typed conflicts
   rather than exceptions.
5. **Exactly-once commit authority.** Commit is continuous, extends the exact
   previous receipt, uses the exact cumulative token clock, requires every
   frozen contributor to report one identical result, and rejects conflicting
   current authority fatally.
6. **All-eight node apply.** Seven-trainer apply is `corrupt` and
   non-mutating. READY above generation zero is deferred until the node has an
   eight-trainer apply receipt for the current committed generation, and a
   mismatched receipt is rejected.
7. **Recovery without rollback.** Recovery installs only equal/newer durable
   generation/token/receipt authority. Durable node-apply receipts are loaded
   before peer recovery; a stale higher-fence recovery proposal is
   non-mutating. Prior applied-generation evidence remains monotonic across a
   later commit without granting READY for that later generation.
8. **Bounded authority.** Membership and recovered-apply maps cap at 256;
   cohort/contribution/result maps are subsets of that bound; there is one
   active generation and a two-reassignment owner limit with bounded
   owner-loss replay receipts. Effect-side Python caches prune on commit and
   retain at most current/frozen incarnations.
9. **Fatal means invariant failure.** Malformed events are `invalid-event` or
   `corrupt`. Stale, late, duplicate, insufficient, closed, and deferred inputs
   do not throw and do not mutate authority.

## Permanent job5105811 regression

Both the direct production-kernel CTest and the actual
Python -> C ABI -> AF_UNIX service RPC -> C++ kernel integration encode the
minimized ordering:

1. node-0, node-1, and node-2 are live/READY for generation 0;
2. node-0 and node-2 contribute, the generation closes, both report the same
   result, and generation 1 commits once;
3. node-0 recovers under a new incarnation/sequence;
4. uninjected node-1 submits to closed generation 0;
5. the transition returns `generation-closed` plus immutable catch-up
   authority, with identical pre/post state digests;
6. node-1 remains live; no expire, kill, or restart-budget effect exists;
7. a seven-trainer apply cannot authorize READY;
8. node-0 and node-1 each report all eight, advertise READY, and open
   generation 1 together.

The live role treats that catch-up as a successful reload handoff and returns
zero. It does not throw through the manager, kill node-1, consume the
supervisor restart budget, or create `restart_exhausted`.

## Canonical replay trace

Every call emits exactly one deterministic JSON object with schema
`emender-native-coordination-trace-v1` and kernel identifier
`emender-native-coordination-kernel-v1`. It contains:

- pre-state digest;
- complete event identity and flags;
- typed disposition;
- ordered explicit effects;
- authoritative fence, committed generation, exact token clock, phase, bounded
  counts, commit receipt, and committed result;
- post-state digest and invariant status.

No timestamp, address, pointer, process ID, map iteration nondeterminism, or
effect-execution result enters the record. The production adapter appends the
kernel-produced bytes unchanged to
`retained-evidence/pool-control/native-coordination-trace-v1.jsonl`. This is
the canonical schedule-replay and later Lean-conformance hook; there is no
test-only model.

## Conformance checklist

The following mapping distinguishes **direct** transition-kernel evidence from
**boundary** obligations retained in existing native/controller gates. A
“retained” row is deliberately not claimed as proven by a pure state machine.

### R01–R16

| ID | Scope and implementation/test mapping |
|---|---|
| R01 | **Direct/boundary:** run/fence on every event; stale-fence is non-mutating. Immutable scheduler claim and no-database audit remain external. |
| R02 | **Direct:** recover/READY/expire with stable node, incarnation, sequence, live/ready/recovering flags; direct and live RPC tests. |
| R03 | **Direct:** open snapshots only current READY identities; no launched-rank input exists. Endpoint lease clocks feed expiry events externally. |
| R04 | **Direct:** complete event identity, one stable-node contribution, idempotent identical receipts, typed stale/conflict/closed outcomes. |
| R05 | **Boundary retained:** exact tokens are the only kernel weight and token clock. Native binary64 numerical correctness remains reducer/reference evidence. |
| R06 | **Direct:** positive Q/T, finite/deadline flags, insufficient/deferred/abort outcomes, bounded attempts. |
| R07 | **Direct/boundary:** exactly-once native commit and immutable previous-receipt lineage; durable manifest/CAS publication remains `ManifestPeerAuthority`. |
| R08 | **Direct/boundary:** immutable frozen identities and bounded two-owner reassignment authority. Dense chunks, credits, replay bytes, and release remain compiled transport evidence. |
| R09 | **Boundary retained:** kernel contains no model/optimizer/dense state. Trainer snapshot ownership and overlap remain ISP01–ISP03 live evidence. |
| R10 | **Boundary retained:** no kernel filesystem/database/network API; production adapter writes only canonical evidence and immutable publication effects. Existing compute-closure audit remains authoritative. |
| R11 | **Direct:** recovery, higher sequence/incarnation, closed-generation catch-up, owner-loss retry, next-generation rejoin; job5105811 regression. |
| R12 | **Direct/boundary:** monotonic committed generation/token/result/receipt recovery. Outer optimizer/checkpoint restoration remains immutable-manifest evidence. |
| R13 | **Boundary retained:** event ABI is scheduler/backend neutral; Frontier and future backend adapters remain external. |
| R14 | **Direct/boundary:** typed deadline/defer decisions and deterministic traces. Human-time phase/tail measurements remain live telemetry. |
| R15 | **Direct/boundary:** token clock advances by exact frozen tokens and changing membership is explicit; numerical parity remains native reducer evidence. |
| R16 | **Boundary retained:** kernel has no node-count scale gate. Two-node/scale authorization and scheduler evidence remain ordered controller tasks. |

### NDP01–NDP17

| ID | Scope and implementation/test mapping |
|---|---|
| NDP01 | **Direct:** persistent C++ service owns the sole state and pure transition; native production call-site audit proves Python has no duplicate authority. |
| NDP02 | **Boundary retained:** serialized metadata RPC and point-to-point effects contain no all-rank operation; existing MPI/symbol/failure gates remain required. |
| NDP03 | **Boundary retained:** kernel is compiled into the persistent service. Provider=`cxi` selection remains service/Frontier evidence. |
| NDP04 | **Boundary retained:** no dense/live mutable state crosses the kernel. Coherent producer snapshot ownership remains ISP01 evidence. |
| NDP05 | **Boundary retained:** kernel supplies exact tokens only; deterministic binary64 arithmetic remains native numerical tests. |
| NDP06 | **Direct:** fixed 312-byte fenced event and domain-separated receipt identity; ABI size gates plus stale/conflict tests. |
| NDP07 | **Boundary retained:** READY effect cache validates opaque same-run/fence/backend/bundle endpoints; AV installation remains transport evidence. |
| NDP08 | **Direct/boundary:** fixed authority/result/member/effect bounds and cache pruning. Dense buffer/slot exhaustion remains native capacity evidence. |
| NDP09 | **Boundary retained:** effect execution cannot alter authority except through a later event; fabric credit/foreground behavior remains ISP02/ISP04 evidence. |
| NDP10 | **Direct/boundary:** idempotent coordination receipts and fail-closed authority; CRC/SHA/nonfinite dense validation remains native transport/reducer evidence. |
| NDP11 | **Direct/boundary:** no more than two owner reassignments and typed abort/retry; byte replay and optional NVMe remain transport evidence. |
| NDP12 | **Direct/boundary:** result agreement covers every frozen contributor before commit; owner-direct dense redistribution remains native data-path evidence. |
| NDP13 | **Direct/boundary:** deadline/late/closed inputs are total and route-local; zero foreground waiting remains ISP02/ISP05 live evidence. |
| NDP14 | **Direct:** additive stable C ABI, bounded metadata RPC, fixed trace/result buffers; actual service integration test. |
| NDP15 | **Direct/boundary:** result agreement, commit, all-eight node apply, and no partial READY authority. Checkpoint I/O/atomic trainer swap remains external. |
| NDP16 | **Direct/boundary:** canonical identity/effect/state trace. Required causal timing phases/max/p99 remain ISP06/ISP07 live evidence. |
| NDP17 | **Boundary retained:** build/CTest is allocation-free. Exact two-node CXI and scale ladder remain later gates; no promotion is claimed. |

### V21S01–V21S17

| ID | Scope and implementation/test mapping |
|---|---|
| V21S01 | **Direct/boundary:** versioned kernel/event/trace schemas and policy digest. Historical policy rejection remains v2.1 ingress evidence. |
| V21S02 | **Direct/boundary:** monotonic commit/applied generation clocks and stale/defer/drop outcomes. Full lag-clock/foreground behavior remains async-policy evidence. |
| V21S03 | **Direct:** positive exact tokens alone determine Q/T, result totals, and cumulative commit clock; aggregation-weight is rejected by live adapter. |
| V21S04 | **Boundary retained:** commit advances only generation/token authority; K40/eta-one/outer-math behavior remains trainer/policy tests. |
| V21S05 | **Direct/boundary:** fenced stable node/incarnation/sequence/token/payload identity and one contribution per stable node. Window/base/layout/code details remain v2.1 dense ingress validation. |
| V21S06 | **Boundary retained:** kernel cannot access live trainer state; coherent snapshot capture and negative ownership proof remain ISP01/ISP02. |
| V21S07 | **Direct/boundary:** commit/result is once-only; node READY requires current all-eight apply. Atomic `x/z/interval` swap and 60-second measurement remain trainer evidence. |
| V21S08 | **Direct/boundary:** stale/conflicting/corrupt results are non-mutating or fail closed; mailbox capacity/replacement behavior remains external. |
| V21S09 | **Direct/boundary:** one active generation, 256 members, eight effects, bounded owner replay authority, pruned effect caches. Dense capacity edges remain ISP04. |
| V21S10 | **Direct:** leased READY snapshot, expiry, new sequence/incarnation recovery, closed old-work rejection, no one-node authority below Q. |
| V21S11 | **Direct/boundary:** seven-trainer report never grants authority; eight grants per-node apply and then READY. Actual trainer atomicity/restart timing remains ISP05. |
| V21S12 | **Boundary retained:** production kernel runs in the persistent model-free C++ service and metadata-only RPC; libfabric/memfd/no-MPI facts remain native gates. |
| V21S13 | **Direct/boundary:** canonical causal coordination trace with every disposition/effect; phase timing/max/p99/foreground-idle evidence remains ISP06/ISP07. |
| V21S14 | **Direct/boundary:** higher-fence recovery installs exact monotonic commit/result/token/apply authority; durable checkpoint/seed/sbcast evidence remains external. |
| V21S15 | **Boundary retained:** no Slurm run or two-node gate was attempted or claimed. Kernel becomes an input to later exact-source qualification. |
| V21S16 | **Boundary retained:** no scale authorization or rung pass is claimed; existing fail-closed controller ordering is unchanged. |
| V21S17 | **Direct/boundary:** external reviewed finite-close timing feeds `FINITE_CLOSE`; kernel snapshots leased READY identities and includes every admitted pre-close contribution, never launched ranks. Evidence-derived scale formula remains policy/controller evidence. |

### ISP01–ISP07

| ID | Scope and implementation/test mapping |
|---|---|
| ISP01 | **Boundary retained:** kernel has no trainer/live-state references. Coherent immutable capture and negative background-access proof are not claimed here. |
| ISP02 | **Direct/boundary:** deferred/closed/deadline coordination never waits in the kernel. Foreground K-window overlap remains live timing evidence. |
| ISP03 | **Boundary retained:** kernel consumes only digests/identities. Immutable snapshot/checkpoint ownership is not proven by coordination traces. |
| ISP04 | **Direct/boundary:** fixed state/effect/result bounds and explicit deferred/retry outcomes; every dense/mailbox/credit full-capacity path remains external evidence. |
| ISP05 | **Direct/boundary:** incomplete apply cannot grant READY; late/invalid results are typed and non-mutating. Trainer-visible atomic swap/timing remains external. |
| ISP06 | **Direct/boundary:** every coordination event has deterministic phase/effect/state evidence. Required eight live timing classes, zero result wait, max, and p99 remain unclaimed. |
| ISP07 | **Boundary retained:** raw transition traces cannot hide a coordination event, but checkpoint/restart and approximately-200-second tail rejection remain later live-validator obligations. |

## Validation

All commands were run after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Python commands used `"$EMENDER_PYTHON"` and build wrappers received
`PYTHON_BIN="$EMENDER_PYTHON"`. No `srun`, `sbatch`, `salloc`, scheduler query,
GPU allocation, or model load was used.

Validated commands/results:

```text
cmake -S src/native_resilient_dataplane \
  -B build/coord-kernel-agent1651 \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/coord-kernel-agent1651 --parallel 4
ctest --test-dir build/coord-kernel-agent1651 --output-on-failure
  -> 2/2 passed

PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
  -> unified native CTest 11/11 passed; artifact manifest recorded

"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent1651-pytest-full-final \
  tests/test_native_pool_integration.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_dataplane_reference.py
  -> 118 passed

"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent1651-kernel-final \
  tests/test_native_pool_integration.py::test_production_native_coordination_job5105811_and_rejoin \
  tests/test_resilient_e97_true_2n_launcher.py::test_native_manager_treats_generation_catch_up_as_successful_reload_handoff \
  tests/test_resilient_e97_true_2n_launcher.py::test_native_manager_uses_the_production_compiled_coordination_authority
  -> 3 passed after the final kernel/service rebuild
```

The final canonical rebuild/test, diff check, commit SHA, push result, and
remote equality are recorded in the WG task log.
