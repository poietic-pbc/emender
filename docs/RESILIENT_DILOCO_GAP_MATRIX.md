# Resilient DiLoCo traceability and bounded backlog

Companion to the authoritative
[Resilient DiLoCo Compute Pool](RESILIENT_DILOCO_COMPUTE_POOL.md), its
normative
[Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
specialization, and
[ADR-002: simple asynchronous DiLoCo v2.1](ASYNC_DECOUPLED_DILOCO_V2.md).
This file records v1 implementation state and is the single normative
definition/crosswalk matrix for V21S01–V21S17 and the immutable-snapshot
pipeline namespace ISP01–ISP07. Status is **present**,
**partial**, or **gap**. Line numbers are avoided because implementation paths
move; paths and test names are stable review anchors. “Present” means wired and
locally exercised, never that a Frontier or promotion gate passed. After the
2026-07-31 ADR-003 decision, these namespaces remain independently normative
for elastic/native/async research that claims them; they are not implicitly
production E97 launch requirements.

## ADR-003 production fixed-world crosswalk (2026-07-31)

Frontier job **5125415** is the decision evidence for the selected
same-allocation execution-epoch boundary; see
[`validation/direct-same-allocation-trainpy-restart-5125415.md`](validation/direct-same-allocation-trainpy-restart-5125415.md).
It proved 64M-bucket hierarchical `train.py` merge, atomic checkpoint reload,
bounded failed-child teardown, allocation survival, and fresh smaller-world
relaunch. It remains honestly `v1_conformance=false` and
`v21_conformance=false`; this crosswalk does not relabel it.

| ID / set | ADR-003 production disposition | Implementation / remaining evidence |
|---|---|---|
| R07 | **Applicable safety intent, elastic receipt-chain clause unclaimed.** A child checkpoint and `latest.pt` are synchronously published with temp-file/rename and temp-symlink/rename. The parent advances only a readable atomic epoch `latest.pt`; partial files are never selected. | `train.save_checkpoint`; `samealloc_promote_epoch_latest`; checkpoint and launcher tests. No native peer receipt chain is claimed. |
| R12 | **Applicable safety intent.** The single child receives stable run-level `latest.pt` through `--resume`, restoring model, inner optimizer, and recorded DiLoCo outer state. Stable `RUN_ID` and pointer survive a failed job and a human-approved replacement job. | `scripts/frontier/e97_same_allocation_restart.sbatch`; restart/outer-state tests. Automatic child restart and scheduler requeue are forbidden; no allocation claim or native recovery handshake is claimed. |
| R14 / NDP13 | **Applicable after translation to an execution-epoch boundary.** `timeout` TERM/KILL bounds the epoch, `srun --kill-on-bad-exit --wait` bounds rank-failure cleanup, and existing final/pre-walltime checkpoint controls remain enabled. | Job 5125415 terminated the damaged step in 99 seconds; launcher syntax/contract tests cover configured bounds. Async phase/foreground-wait telemetry is retired, not satisfied. |
| R16 | **Replaced production ladder.** The attended approval plus job 5125415 selects immutable exact-source `8 -> 32 -> 128`, each rung requiring its immediate predecessor. 256 remains review-only. | The dependent immutable 8-node acceptance consumes the production launcher commit. The historical native G2–G6 authorization is research-only. |
| NDP02 | **Retired/incompatible for production.** Each child deliberately uses fixed-world RCCL collectives. Failure safety comes from terminating the entire child and constructing a fresh process group, never from continuing or shrinking its communicator. | Job 5125415 is the physical containment evidence. No no-all-rank claim is made. |
| NDP15 | **Checkpoint atomicity retained; background/apply clauses retired.** Production checkpoints synchronously at K-aligned step 200, retains two, and keeps final/pre-walltime publication. | Atomic `train.py` tests plus launcher promotion tests. No hashing, background checkpoint I/O, mailbox, or later apply is added or claimed. |
| NDP17 | **Retired/replaced for production.** The full-layout native CXI G2–G6 chain remains research evidence. Job 5125415 and the approved exact-source fixed-world ladder are the production gate. | No native-bundle or communicator-shrink claim; each production rung must retain its immutable predecessor pass. |
| R02–R06, R08–R11; NDP01, NDP03–NDP12, NDP14, NDP16 | **Retired from production, retained as elastic/native research.** | Their matrices and artifacts below remain unchanged evidence; ADR-003 adds no cell, owner-tree, service, database, or new coordination protocol. |
| V21S01–V21S17; ISP01–ISP07 | **Entirely retired from production, retained as non-production async-v2.1 research.** | No overlap, distinct lag clocks, background immutable-snapshot pipeline, cell/owner-tree, or communicator-shrink conformance is claimed. Existing gaps remain honest research gaps. |

Production defaults are `DILOCO_K=40`, `SAVE_EVERY=200` (five outer merges),
and `KEEP_CHECKPOINTS=2`, all explicit overrides with K-alignment enforced.
The launcher preserves hierarchical `DILOCO_MERGE_BUCKET_NUMEL=67108864`, uses
a stable run directory independent of `SLURM_JOB_ID`, launches exactly one
execution epoch, performs no inline validation, and fails the job after the
first child failure. Submission uses `--no-requeue` and verifies `Requeue=0`.

## Elastic research requirements-to-current-harness matrix

| ID | Requirement | Current code / evidence | Tests | Status and bounded gap |
|---|---|---|---|---|
| R01 | Exclusive scheduler-fenced allocation claim and strictly newer fence before model load; no shared database | `_allocation_admission` publishes an immutable claim anchored to the exact base receipt; Slurm job ID supplies the fence; native peer commands carry claim/fence/incarnation and stale writers fail closed | `test_allocation_fence_is_acquired_before_roles_and_loser_is_zero_work`; `test_sqlite_connect_is_fatal_across_admission_diagnostic_commit_and_restart`; stale-allocation tests | **Present:** no renewal row, lock, or compute-node database exists. |
| R02 | DISCOVER→BOOTING→SYNCING→READY/ACTIVE→DRAINING/EXPIRED; stable worker plus incarnation | Production `ndp_cxi_service` applies serialized `RecoverPeer`, `Ready`, and `ExpirePeer` events in `coordination::step`; every event carries the stable node, incarnation, and monotonic sequence. `PoolControlServer`/`PeerMembership` remains only the explicit Python-TCP debug reference. | native coordination kernel test; live RPC job5105811/rejoin regression; leased-peer debug tests | **Present** in the live native manager control path. |
| R03 | Active world is live leased READY membership; never launched ranks | `OpenGeneration` immutably snapshots only current leased READY identities in the native kernel; Python supplies typed expiry events from external lease clocks. `node_count` remains address/capacity policy and is not a kernel admission input. | native coordination kernel test; live RPC 3-READY/2-contributor close and rejoin regression | **Present** in the live path; no tensor collective or launched-rank admission remains there. |
| R04 | Fresh generation identity, strict stale rejection, idempotent duplicate handling | The compiled event ABI binds run/fence/generation/attempt/node/incarnation/sequence. `coordination::step` returns typed accepted, identical/conflicting duplicate, stale, closed, deferred, retry, corrupt, or fatal dispositions with deterministic receipts. | `ndp_coordination_kernel_test`; `test_production_native_coordination_job5105811_and_rejoin`; debug receipt tests | **Present**. |
| R05 | Exact token/sample-weighted incremental sharded aggregation | `CpuNodeManager.collect` performs streaming local float64 weighted reduction; `TensorLayout` and `ExactWeightedShardReducer` perform deterministic owner-local weighted reduction with vectorized codecs | reducer unequal-weight/changing-order reference; live representative-layout distributed gate | **Present**; the E97 flat layout is bound without a whole-model concat or Python scalar serialization. |
| R06 | Explicit quorum/token floor plus generation deadline; no unbounded wait | `RecoverAuthority` installs positive fixed `q_min`/`t_min`; external finite/deadline clocks feed flags into total `CloseGeneration` transitions, which either freeze, defer, or abort/retry without blocking in the kernel. | native insufficient/deadline/immutable-close tests; existing policy/deadline tests | **Present** as a mechanism; production Q/T values remain a training-policy choice. |
| R07 | Exact-once peer commit and immutable checkpoint/receipt chain under current fence | The native kernel alone admits the continuous generation/result/token/previous-receipt identity after every frozen contributor reports the same result; identical replay is idempotent and conflicting current authority is fatal. `finalize_checkpoint` and `ManifestPeerAuthority` remain the durable effect executor/receipt chain. | native two-commit/duplicate/conflict tests; live RPC commit; manifest exact-once/publication-failure tests | **Present:** current-fence native agreement plus append-only receipt lineage replaces the database CAS. |
| R08 | Deterministic sharded owners; bounded chunks, checksums, backpressure, replay, prompt release; no broker | The native manager now installs leased endpoint records, moves frozen 320-byte framed memfd chunks through the compiled provider, validates CRC32C/SHA-256 identities, caps each `(peer,root,chunk)` at one send plus two replays, and redistributes one shared result. The Python owner server remains only in the explicit debug branch. | `ndp_protocol_owner_test`; `ndp_fabric_multiprocess_test`; `test_frozen_owner_frame_moves_memfd_to_memfd_over_native_provider`; bounded-replay and result-root tests | **Present in the wired local path;** real-model CXI failure timing remains a downstream G3/G4 measurement. |
| R09 | Manager is model-free; trainers exclusively own live mutable model/optimizer state; background roles consume only immutable snapshots; unfinished work is disposable | The live `_native_manager` owns metadata/native descriptors/membership/checkpoint policy, but the existing overlap fixture does not yet prove that snapshot bytes are captured coherently before optimizer mutation resumes or that every background path is unable to read live state. | manager source audit; required `test_snapshot_capture_is_coherent_and_background_never_reads_live_state`; race/invariant test mutating weights immediately after `OWNED` while verifying stable snapshot bytes | **Partial:** role separation is present; immutable snapshot coherence and negative live-state access evidence are still required by ISP01/ISP03. |
| R10 | Non-Lustre update, aggregate, heartbeat, membership, redistribution hot path | `assert_node_local_path` guards role-local control/heartbeat/telemetry; service-owned memfds and compiled point-to-point transport carry dense state. Shared-run writes are append-only allocation/commit/apply receipts and immutable checkpoints/manifests only. | rendered compute-closure SQLite tripwire; node-local rejection; direct-memfd telemetry and launcher topology tests | **Present in code:** no live filesystem database, lock, membership, or heartbeat traffic remains. |
| R11 | Catch-up, disappear/rejoin with new incarnation, late join next generation | Native recovery requires authoritative commit plus a higher incarnation sequence; old work returns a non-mutating typed disposition, at most two owner reassignments are authoritative, and an all-eight-applied peer can join the next immutable READY snapshot. | permanent job5105811 live RPC/direct-kernel regression; owner-replay and recovery tests | **Present** in the live control/role path. |
| R12 | Outer optimizer globally owned/restored; inner work optional; fresh-allocation resume | Immutable receipts bind manifest/checkpoint/result roots, outer step, accepted-token clock, membership and fence. A newer claim anchors the exact prior receipt; native recovery handshake installs it before READY; stale local state is discarded | outer-state migration; fresh-allocation manifest recovery; stale-fence/incarnation and exact receipt-chain tests | **Present** without database bootstrap. |
| R13 | Backend-neutral protocol | `ndm/resilient_pool_runtime.py` is scheduler/MPI independent; Frontier hostname and allocation admission are isolated in supervisor/role adapters | local TCP multi-peer gate | **Partial:** Frontier adapter is live; a hyperscale-local adapter fixture is still absent. |
| R14 | Stage deadlines, immutable committed evidence, and disjoint foreground/background phase timing | Existing telemetry records READY/K40/exchange/commit deadlines and some overlap fields, but does not yet require causally matched freeze/snapshot, admission, publish/network, aggregation, checkpoint, result-wait, apply/swap, every-event tail pauses, and total foreground idle. | stage-deadline tests; required semantic-validator fixtures for every missing phase, a 200-second alternating-stall rejection, and maximum/p99 snapshot/apply budgets; two-node live JSONL | **Partial:** ordinary stage deadlines are present; ISP06/ISP07 telemetry and tail-stall gates remain a documented implementation/qualification gap. |
| R15 | Numerical/reference correctness and changing-participation accounting | Accepted-token clock advances only by the frozen global weight; vectorized float64 reducers are arrival-order stable and restore recorded E97 dtype only at redistribution | unequal weights `(3,1000003,29)`, equal-cohort parity, changing READY membership and 3/2 pool gate | **Present** for numerical/runtime accounting; convergence research remains separate policy work. |
| R16 | Current-source two-node clean/fault/fresh-recovery gate before direct systems scale; exact predecessor at `8 -> 32 -> 128`; explicit 256 review | The controller/launcher reject every removed rung and 256 submission, require exact source/policy/schema/native/seed/launcher identities, and bind complete collector-backed systems evidence before each immediate successor. | retained G2 reports/checksums; current-source G3–G5 verdicts; `test_v21_direct_ladder_rejects_removed_rungs_and_review_only_256`; exact predecessor/identity/evidence tests | **Present as policy/code; no live rung pass is claimed.** |

## Native data-plane v1 requirements-to-implementation matrix

These IDs are requirements, not a claim that local integration promotes a
real-model Frontier run. The compiled local and libfabric implementations share
one digest-attested bundle; the split-role native branch now calls both ABIs.
Rows distinguish wired code/local evidence from the retained synthetic G2 rung
and from later real-model G3–G5 execution.

| ID | Native requirement | Current code / evidence | Required validation anchor | Status and bounded gap |
|---|---|---|---|---|
| NDP01 | Native peer control owns live fence/incarnation, membership, generation/commit and recovery; hard metadata-control/C++-dense boundary; no production Python dense TCP or shared database | One `AuthorityState` in persistent `ndp_cxi_service` is changed only by pure `coordination::step(state,event)`. `_native_manager` constructs `NativePoolControlServer` around the service ABI; Python retains endpoint/timer/storage effect caches only. `PoolControlServer` is confined to `python-tcp-debug`. | production call-site source audit; direct kernel CTest; actual service/RPC job5105811 integration | **Present.** |
| NDP02 | No failure-sensitive all-rank operation in elastic backend | Native local/transport binaries and the live role are process-local or point-to-point; dynamic exports are scanned for `MPI_`/`PMPI_`, and the launcher rejects fixed-world MPI. | manifest symbol scan; peer exit/rejoin/removal tests; retained G2 fault evidence | **Present.** |
| NDP03 | Persistent C++17 `FI_EP_RDM` service; exact Frontier `cxi` | The node-local supervisor starts one `ndp_cxi_service` before its model-free manager and eight trainers, passes a sealed admission-token fd, and independently monitors/drains it. The same persistent service now owns coordination authority. Production commands require exact `cxi`; test paths require an explicit test provider. | eleven unified CTests; persistent cross-process service/RPC tests; launcher topology/provider tests; retained G2 provider evidence | **Present in code and synthetic G2;** real startup is the immediate downstream gate. |
| NDP04 | XPMEM or producer-direct memfd immutable snapshot; zero extra full-layout handoff writes; no service read of live mutable state | Each trainer writes a service allocation and seals a descriptor, but current tests prove zero-copy ownership transfer rather than coherent capture against concurrent optimizer mutation. | eight-trainer K40 parity; required snapshot race test and native/source audit proving the service maps only sealed immutable extents; shared-result views | **Partial:** direct handoff exists; ISP01 coherence and the prohibition on background live-state reads need executable evidence. |
| NDP05 | Bitwise v1 exact weighted native arithmetic/layout | Rank-sequenced f32 K40 deltas accumulate into token-weighted binary64 node numerators; owner attempt 2 adds preweighted node numerators, divides once by exact global tokens, then projects once to f32. | native reference suite; `test_persistent_service_preserves_exact_global_numerator`; eight-trainer mixed-precision K40 parity | **Present locally.** |
| NDP06 | Fixed fenced contribution/frame/receipt identities | In addition to the 320-byte dense frames, the fixed 312-byte coordination event carries run/fence/generation/attempt/node/incarnation/sequence/exact-token and digest identity; the service derives an immutable domain-separated contribution receipt. | coordination ABI size gates and CTest; golden frame tests; live stale/duplicate/closed RPC regression | **Present.** |
| NDP07 | Leased Python endpoint exchange and current-fence AV routes | Managers publish opaque endpoint records in READY metadata and install only frozen same-backend/same-bundle current-fence records in the native AV. | endpoint checksum/fence/provider tests; live route wiring; retained G2 | **Present.** |
| NDP08 | Pre-registered, capacity-bounded snapshot/result/fabric buffers with explicit byte/memory admission formulas and nonblocking exhaustion policy | Service-wide byte admission, immutable exact extents, TX/RX slots, and the 64-GiB ledger are bounded; the current tests do not prove every full snapshot/mailbox path returns skip/replace/defer without stalling a trainer. | shared-byte exhaustion; required `test_snapshot_and_mailbox_capacity_never_blocks_foreground`; eight-trainer metrics; slot/payload CTests | **Partial:** memory bounds are present; ISP04 requires foreground nonblocking evidence at every capacity edge. |
| NDP09 | Byte/slot credit backpressure distinct from fabric completion and foreground progress | Authenticated credits and fixed fabric slots are implemented, but the evidence must also show credit exhaustion after `OWNED` cannot delay the next K-window start. | owner credit exhaustion/recovery CTests; required integration trace holding all credits while the trainer completes the next K window; provider slot tests | **Partial:** credit correctness is present; the ISP02/ISP04 foreground boundary is not yet an acceptance gate. |
| NDP10 | CRC32C headers, SHA-256 payloads, once-only apply, idempotent receipts | Provider ingress retains its fixed header/route/CRC/payload checks. The coordination kernel additionally makes contribution, result, commit, and node-apply receipts idempotent; conflicting authority fails closed while stale/late races remain nonfatal and non-mutating. | native corruption/conflict tests; direct provider frame test; coordination duplicate/stale/job5105811 tests | **Present.** |
| NDP11 | Bounded replay, at most two owner reassignments, optional one-contribution local-NVMe fallback | `transfer_frozen_fd` keys replay by peer/result-root/chunk and permits at most the initial send plus two replays; default live fallback is disabled and reports zero disk replay/spool bytes. | fd replay-exhaustion test; retained G2 fault/reassignment evidence; fallback cleanup test | **Present as a bounded mechanism;** real-role owner-loss timing remains G4 evidence. |
| NDP12 | Owner-direct redistribution into one shared node aggregate | Frozen f64 numerator frames are received into bounded memfds, validated, registered directly for deterministic attempt-2 reduction, and exposed as one service-owned read-only f32 result to all eight trainers. | provider memfd transfer; exact two-stage numerator test; eight independent result views | **Present.** |
| NDP13 | Absolute stage deadlines, route-local failure containment, and no foreground wait on background expiry | Live stages have absolute deadlines and local containment; accepted wording now requires result/publication expiry to skip/defer rather than become a trainer catch-up pause. | native failure/multiprocess suites; required late/missing/failed-result integration with continuing K starts and zero foreground result wait; launcher TERM/service-loss tests | **Partial:** deadline containment exists; ISP02/ISP05 nonblocking expiry evidence remains required. |
| NDP14 | Stable `libemender_ndp.so.1` C ABI and metadata-only seqpacket control | The live trainer/controller clients use the bounded v1 RPC, sealed token fd and descriptor-only AF_UNIX protocol. Additive `ndp_coord_step_v1` uses fixed 312-byte events, fixed bounded results/effects, and a 4096-byte canonical trace; the authoritative registry and state remain solely in the persistent service. | ABI size/prefix/auth tests; direct and cross-process service/RPC coordination tests | **Present.** |
| NDP15 | Fenced immutable checkpoint handoff; background publication; later atomic bounded apply with no checkpoint I/O in the foreground pause | Peer result/receipt agreement and eight-lane recovery markers exist. Existing lifecycle/source-order tests do not yet prove checkpoint hashing/I/O is driven only from immutable snapshots or that late/unready results are deferred before an atomic apply begins. | manifest identity and restart tests; required atomic-apply fault injection at every lane with no partial visibility; checkpoint-I/O block while K windows continue | **Partial:** atomic recovery mechanisms exist; ISP03/ISP05 separation needs direct integration evidence. |
| NDP16 | Required provider/identity/byte/bound/release and per-phase foreground/background telemetry | Every coordination transition now emits versioned canonical JSON with event identity, typed disposition/effects, pre/post state digests, authoritative phase/counts, and invariant status. Existing phase telemetry still lacks all required ISP06/ISP07 live timing/tail evidence. | canonical trace assertions and replay-ready JSONL; manifest/metric assertions; required semantic-validator live timing gates | **Partial:** deterministic coordination telemetry is present; it does not substitute for missing foreground/background timing evidence. |
| NDP17 | Full-layout two-node synthetic CXI hard gate before a real/scale job; current-source G3 clean, G4 fault/rejoin, G5 fresh recovery, then direct G6 `8 -> 32 -> 128 -> review 256` | Production validates the exact bundle/gate and rejects Python TCP, test providers, dirty artifacts, fixed-world MPI, removed rungs, and a 256 runner before roles. | artifact tamper/backend tests; retained G2 reports/checksums; direct-ladder controller/launcher contract | **Present as policy/code through the submission boundary; later physical rungs are intentionally unclaimed.** |

## Normative simple asynchronous v2.1 matrix

This table is the only normative definition of the V21S namespace. ADR-002
provides the full semantics. The base-map column is additive: every named R and
NDP requirement still applies in full. Existing code and artifacts implement
historical v2.0 identities and therefore cannot be marked present for v2.1
until the versioned production path and its tests change.

| ID | Normative v2.1 requirement | Preserved base map | Required test / evidence | Bounded gap |
|---|---|---|---|---|
| V21S01 | Pin `async-decoupled-v2.1-simple`, canonical policy/contribution/manifest/native schema identities and digests, reject unknown or historical v2.0 identities before load/mutation, and never relabel v1/v2.0 work. | R01, R04, R07, R12, R14; NDP06, NDP10, NDP14, NDP16 | `ndm/async_diloco_v2.py`; `src/native_resilient_dataplane/include/emender/ndp.h`; `native/dataplane/src/protocol.cpp`; `test_v21_rejects_v20_policy_schema_and_digest`; native ABI/protocol golden tests. | **Present:** every v2.1 ingress checks the versioned policy, contribution, wire, ABI, and checkpoint identities; v2.0 remains only as rejectable historical/reference data. |
| V21S02 | Keep commit, applied-anchor, result-version-at-apply, and speculative-snapshot lag as distinct integer clocks; accept each only through 2, drop a lag-3 contribution/result, defer a third snapshot, and never pause foreground training for catch-up. | R04, R06, R11, R14; NDP06, NDP13, NDP16 | `AsyncV21ContributionIdentity.validate`; required replacement of pause/catch-up fixtures with `test_v21_lag_three_defers_snapshot_without_blocking_next_k_window`; live zero-foreground-result-wait trace. | **Partial:** clock/drop logic exists, but older pause/catch-up behavior and tests must be reconciled to ISP02/ISP05. |
| V21S03 | Use positive exact tokens as the sole quantitative quorum, accepted-token clock, deterministic binary64 numerator weight, and denominator; permit no distinct aggregation/effective/staleness weight in v2.1. | R05, R06, R12, R15; NDP05, NDP06, NDP09, NDP10, NDP16 | `reference_aggregate`; `ndp_submit_local_v21`; native coordination kernel; `test_v21_exact_tokens_are_only_weight_and_eta_one`; native unequal-token permutation roots. | **Present:** v2.1 schemas and native coordination expose only `exact_tokens`; commit advances the accepted-token clock by exactly the frozen generation total. |
| V21S04 | Run exact K40 intervals and apply the stateless exact-token mean with `eta_outer=1.0`, advancing only outer step and exact accepted tokens; no outer momentum tensor. | R05, R12, R15; NDP05, NDP15 | `AsyncV21Policy`; `AsyncV21CommitAuthority.commit`; `PersistentRealWorkerSession`; `test_v21_exact_tokens_are_only_weight_and_eta_one`; persistent-session tests. | **Present:** K is pinned to 40 and outer state contains only step, exact accepted tokens, and the immutable eta-one policy value. |
| V21S05 | Bind the full fenced stable-worker/incarnation/window/base/policy/layout/code/token/payload identity; at exact two nodes require `Q_min=2`, `T_min=3,934,080`, disabled active fraction, zero attempt retry, fixed ADR deadlines, and at most one contribution per stable worker per transition. | R01, R02, R04, R06, R14; NDP01, NDP06, NDP10, NDP13, NDP16 | `AsyncV21ContributionIdentity`; `_pool_config`; `coordination::step`; native identity conflict/replay/incarnation tests; exact render/controller tests. | **Present:** two-node constants and one-stable-worker native admission are pinned; scale replaces the two-node close only through reviewed V21S17 evidence. |
| V21S06 | Trainer exclusively owns resident mutable model/optimizer/iterator/hidden state; at K boundaries it captures a coherent fenced immutable snapshot into a bounded mechanism; `OWNED` ends trainer responsibility and immediately resumes training. Background live-state reads and concurrent weight copying are forbidden. | R08, R09, R14; NDP04, NDP08, NDP09, NDP11, NDP14 | persistent-session tests; required coherent-snapshot race test, negative background-access audit, and blocked-result integration proving next-K start after `OWNED` | **Partial:** resident overlap exists; ISP01/ISP02 snapshot coherence and exclusive-ownership evidence remain required. |
| V21S07 | At a safe K boundary apply a complete verified result exactly once and atomically to ScheduleFree `x`, `z`, and the mutable interval start within the `60 s` all-eight bound; skipped versions retain ledger correctness, while late/absent/invalid/failed results defer without waiting. | R09, R11, R12, R15; NDP04, NDP12, NDP14, NDP15 | correction-ledger fixtures; required late/missing/invalid-result test with uninterrupted K starts and apply fault injection proving no partial `x/z/interval` visibility | **Partial:** correction math exists; ISP05 atomic visibility, bounded pause, and nonblocking failure evidence remain required. |
| V21S08 | Admit only reload-verified current-fence peer-committed results into a capacity-one latest mailbox, allow one bounded replacement staging view, reject old/conflicting/corrupt results, and skip/defer capacity or readiness failures without blocking foreground training. | R01, R07, R08, R12; NDP06, NDP10, NDP12, NDP15 | `VerifiedLatestMailbox`; peer authority; skipped-version/corrupt/fence fixtures; required held-view/full-staging integration proving continuing K starts | **Partial:** validation/replacement exists; ISP04 nonblocking held-view behavior lacks foreground evidence. |
| V21S09 | Preflight finite resident, registered-slot, credit, replay, reassignment, receipt, mailbox, and deadline bounds; retain one immutable snapshot plus one mutable interval, forbid a third dense cohort/spill, and make every capacity edge skip/replace/defer rather than wait in foreground. | R06, R08, R10, R14; NDP08, NDP09, NDP11, NDP13, NDP16 | `_v21_resident_bytes`; native bounds; required parameterized exhaustion test for snapshot slot, mailbox view/staging, credit, replay, and receipt bounds with an advancing training counter | **Partial:** the `64,001,671,648`-byte formula is present; ISP04 behavior at all full-capacity paths is not yet proven. |
| V21S10 | Use only leased READY membership; expire failures, require a returning stable worker to use a new incarnation and authoritative latest, reject old work except an already frozen exact identity, and never turn the two-node floor into one-node authority or a foreground missing-peer wait. | R01, R02, R03, R06, R11; NDP01, NDP02, NDP06, NDP07, NDP13 | native recover/READY/open/expire transitions; job5105811/rejoin regression; required missing-peer foreground timing test | **Partial:** native leased membership and nonfatal defer are present; the ISP02 foreground timing evidence remains required. |
| V21S11 | Treat apply/recovery as one node transaction: background prepares one verified result for all eight trainers; only a complete safe-boundary transaction begins, finishes within `60 s`, and emits one node marker. Unready results defer; any partial/timed-out apply permits no READY and restarts all eight from verified latest. | R07, R09, R11, R12, R14; NDP12, NDP13, NDP15, NDP16 | `AtomicEightTrainerApply`; node-applied/READY ordering; required failure at each preapply/apply/marker point proving defer-before-begin or all-eight restart, never partial visibility | **Partial:** recovery fencing and marker reduction exist; ISP05 needs atomic visibility and bounded foreground timing evidence. |
| V21S12 | Preserve one persistent model-free compiled C++ service per node, producer-direct memfd/XPMEM, deterministic exact-token reduction, bounded libfabric `FI_EP_RDM` with exact Frontier `cxi`, and point-to-point redistribution; forbid MPI/all-rank waits, Python/Lustre dense bytes, and a central full-model broker. | R03, R08, R09, R10, R13; NDP01, NDP02, NDP03, NDP04, NDP05, NDP06, NDP07, NDP08, NDP09, NDP10, NDP11, NDP12, NDP13, NDP14, NDP16, NDP17 | v2.1 and coordination C ABI in `ndp.h`/`client.cpp`; versioned native fabric frames; `NativeManagerSession`; native 11/11 CTests, ABI/reference/failure/integration suites, and source/symbol audits. | **Present** locally with v2.1 fail-closed identities and coordination authority in the qualified bounded native service; the exact-code Frontier G2 artifact remains a V21S15 run gate. |
| V21S13 | Emit honest causal phase identities/intervals for freeze/snapshot, admission, publish/network, aggregation, checkpoint, result wait, apply/swap, and total foreground idle, plus all lag/bound/high-water facts. Gate every-event maximum/p99 snapshot/apply pauses and zero foreground result wait; never infer overlap. | R03, R06, R14, R15; NDP09, NDP13, NDP16 | runtime/validator; required missing-phase, mismatched-causal-ID, foreground-result-wait, tail-pause, and alternating-200-second-stall rejection tests | **Partial:** current cadence/idle/background telemetry is useful but does not satisfy ISP06/ISP07. |
| V21S14 | Publish one fenced immutable model/outer/token/identity/lag/apply-evidence bundle and digest-linked receipt, restore exact outer step/token/result/fence/apply state on a newer claim through peer recovery, cold-start from the exact step-2300930 final seed, and use submit-side SHA/attestation plus node-local `sbcast` offline verification. | R01, R07, R10, R12, R14; NDP01, NDP06, NDP10, NDP15, NDP16 | manifest peer authority; exact E97 render/launcher; fresh-allocation recovery and sqlite-poison tests; seed/SHA/sbcast/offline launcher tests. | **Present:** no database bootstrap is required; the pinned seed (`0239706e...6a72b2`) remains submit-verified and staged at `/tmp/emender-e97-seed-$SLURM_JOB_ID` with compute-node network fetch disabled. |
| V21S15 | Require current-source two-node clean/performance including ISP01–ISP07, fault/rejoin, and newer-fence fresh-recovery machine passes on `Partition=batch` and `QOS=debug`, retaining both scheduler fields separately; numerical/deterministic checks remain systems preflight and convergence/model quality remains separate. | R04, R05, R07, R08, R11, R12, R14, R15, R16; NDP05, NDP06, NDP08, NDP09, NDP10, NDP11, NDP12, NDP13, NDP14, NDP15, NDP16, NDP17 | qualification controller; clean/fault/fresh-recovery terminal collectors; causal phase/tail validator; separate `squeue`/`sacct` fields | **Present as a fail-closed authorization schema/controller; no new live pass is claimed.** |
| V21S16 | Promote systems scale only after the exact current-source two-node clean/fault/fresh-recovery pass, then require an immutable machine pass from each immediate predecessor in `2 -> 8 -> 32 -> 128`; 256 is explicit review-only, convergence is separate, and unchanged failures/missing evidence never advance. | R16; NDP17 | controller payload/state/authorization validation; `test_v21_direct_ladder_rejects_removed_rungs_and_review_only_256`; `test_v21_scale_rejects_wrong_exact_identity`; `test_v21_scale_rejects_incomplete_predecessor_systems_evidence`; exact predecessor and retry tests | **Present as a fail-closed submission boundary:** schemas are v2 direct-scale, every live rung verifies exact identities/evidence/predecessor, and no physical pass is claimed. |
| V21S17 | For scale only, pin a finite deterministic close over the leased READY snapshot, include every complete admissible arrival until that close, never key off launched ranks or close merely at `Q_min=2`, and derive close/deadline/cadence arithmetic from digested passing current-source two-node arrival/stage evidence rather than an unexplained constant. | R02, R03, R06, R14, R16; NDP01, NDP02, NDP07, NDP13, NDP16, NDP17 | `validate_scale_evidence`; `V21ScaleClosure`; scale `_pool_config`; preclose-arrival, missing-evidence, unexplained-constant, launched-rank, Q-min-early-close, and finite-nonbarrier tests. | **Present as a fail-closed reviewed formula mechanism:** no default constant exists; absent digested passing current-source two-node evidence and authorization, every scale render/preflight/submission is rejected. |

## Immutable snapshot pipeline requirements

ISP01–ISP07 are stable, additive requirements for any bounded asynchronous
mode. They codify the foreground/background contract that the older R/NDP/V21S
wording did not express precisely. The direct maps below identify the most
applicable rows; every R01–R16, NDP01–NDP17, and V21S01–V21S17 requirement
still applies independently through the compute-pool conformance checklist.

The named tests below are executable evidence requirements. A name that is not
yet implemented is an explicit bounded gap, not permission to substitute a
source-order assertion or checkpoint count.

| ID | Normative immutable-snapshot requirement | Direct R / NDP / V21S map | Required executable evidence | Status and bounded gap |
|---|---|---|---|---|
| ISP01 | The trainer exclusively owns live mutable model/optimizer state. At a safe K boundary it captures one coherent fenced immutable snapshot using a preallocated double buffer, copy-on-write view, or equivalent bounded mechanism. Concurrent copying from mutating weights and every background live-state read are forbidden. | R04, R09, R14, R15; NDP04–NDP06, NDP08, NDP10, NDP14, NDP16; V21S01, V21S04–V21S06, V21S09, V21S12–V21S13 | Unit/race: `test_snapshot_capture_is_coherent_and_background_never_reads_live_state` mutates model and optimizer immediately after admission, verifies the snapshot remains byte-stable and boundary-coherent, and fails any manager/service access to live objects. Native/source audit proves only sealed immutable extents cross the ABI. | **Gap:** ownership by process is present, but coherence under concurrent foreground mutation is not directly proven. |
| ISP02 | Snapshot capture/admission is a bounded foreground interruption (`1 s` through `OWNED` for exact two-node v2.1). Immediately after `OWNED`, the next K window starts without waiting for discovery, quorum, publication/hashing, network, aggregation, checkpoint, result readiness, or another trainer. | R03, R06, R08–R10, R13–R14; NDP01–NDP04, NDP07–NDP09, NDP11–NDP14, NDP16; V21S02, V21S05–V21S06, V21S09–V21S10, V21S12–V21S13, V21S17 | Integration: `test_admitted_snapshot_resumes_next_k_before_every_background_phase` independently blocks each named background phase after `OWNED`, asserts a complete next K window, and measures capture/admission within its configured bound. Live: causal JSONL proves overlap for all 16 trainers. | **Partial:** one blocked-result fixture overlaps K work, but it does not isolate every phase or prove coherent capture/admission timing. |
| ISP03 | Native/background workers publish, hash, aggregate, validate, and checkpoint only immutable admitted snapshots/results; checkpoint I/O never runs in the foreground apply/snapshot pause. | R04–R05, R07–R10, R12, R14–R15; NDP04–NDP12, NDP14–NDP16; V21S01, V21S03–V21S06, V21S08–V21S09, V21S12–V21S14 | Integration: `test_background_pipeline_uses_only_immutable_snapshots` blocks hashing/checkpoint I/O while mutating the live trainer and verifies stable roots plus advancing K windows; restart command verifies the emitted immutable checkpoint independently. | **Gap:** checkpoint correctness tests pass, but they do not establish immutable input ownership or overlap. |
| ISP04 | Snapshot buffers, queues, credits, and result mailboxes are capacity bounded. Every full-capacity path explicitly skips, replaces, drops, or defers background work; it never grows, spills, or waits in foreground. | R06, R08, R10, R14; NDP02, NDP08–NDP09, NDP11, NDP13, NDP16; V21S02, V21S06, V21S08–V21S10, V21S12–V21S13 | Unit/integration: `test_snapshot_and_mailbox_capacity_never_blocks_foreground` parameterizes snapshot slots, held mailbox view/staging, native credits, replay, and receipt bounds; asserts configured high-water, explicit outcome, no spill/allocation growth, and an advancing K counter. | **Partial:** individual memory/mailbox/credit bounds exist; the cross-layer nonblocking exhaustion gate does not. |
| ISP05 | A complete verified result is applied/swapped atomically at a later safe boundary under a separate foreground bound (`60 s` for the exact two-node all-eight transaction). Late, absent, invalid, failed, or unready results skip/defer without waiting; no partial model/optimizer/node application is visible. | R04, R07, R09, R11–R12, R14–R15; NDP06, NDP10, NDP12–NDP16; V21S02, V21S07–V21S08, V21S10–V21S14 | Unit/integration: `test_result_apply_is_atomic_bounded_and_nonblocking` injects absence, corruption, timeout, and failure before and during every lane apply; asserts defer-before-begin or all-eight fenced restart, exact once-only `x/z/interval` translation, no READY/partial visibility, and the apply bound. | **Partial:** correction math and restart markers exist; atomic visibility, tail timing, and nonblocking unready-result behavior are not one executable gate. |
| ISP06 | Telemetry causally and separately measures `freeze_snapshot`, `snapshot_admission`, `publish_network`, `aggregation`, `checkpoint`, `result_wait`, `apply_swap`, and total foreground idle. Foreground result-wait time is zero; snapshot/admission and apply/swap report every event, maximum, p99, and policy bounds. | R14–R16; NDP13, NDP16–NDP17; V21S13, V21S15–V21S17 | Validator unit tests reject each missing phase, mismatched causal identity, overlap inferred from thread names, nonzero foreground result wait, missing max/p99/bounds, and phase double counting. Live clean-gate JSONL retains all fields for every trainer/window/result. | **Gap:** current telemetry aggregates or conflates several required phases. |
| ISP07 | Checkpoint/restart correctness is necessary but cannot satisfy the overlap gate. Every-event tail evidence is mandatory: median cadence, checkpoint count, or aggregate idle cannot hide bursty alternating K windows; an approximately 200-second foreground pause is a hard failure. | R07, R12, R14–R16; NDP13, NDP15–NDP17; V21S13–V21S17 | Validator: `test_overlap_gate_rejects_200_second_bursty_alternation_despite_checkpoints_and_median` supplies healthy checkpoint/restart, median cadence, and idle summaries with alternating long stalls and must fail. Live clean gate retains raw timestamps and the validator command/result. | **Gap:** the existing cadence/idle test is median-oriented and does not encode this adversarial tail trace. |

The minimum executable unit/integration command is:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_snapshot_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_resilient_e97_runtime.py::test_fresh_process_restart_matches_uninterrupted_continuation \
  tests/test_async_diloco_v21.py::test_v21_checkpoint_and_fresh_allocation_restore
```

`tests/test_async_snapshot_pipeline.py` is the bounded deliverable containing
the named ISP01–ISP05 tests; its current absence is part of the gap. After
those tests pass, the live command is:

```bash
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean --nodes 2 --state "$ASYNC_V21_STATE" \
  --native-build-manifest "$ASYNC_V21_NATIVE_MANIFEST" \
  --full-layout-gate "$ASYNC_V21_G2_JSON" \
  --run-root "$ASYNC_V21_RUN_ROOT" \
  --evidence-root "$ASYNC_V21_EVIDENCE_ROOT" --submit
```

It requires `Partition=batch` and `QOS=debug` verified separately while live
and in terminal accounting.

The historical V2A01–V2A18 namespace describes
`async-decoupled-v2.0-exp` only. Its dated validation reports remain unchanged,
including hard lag 6/8, lag-adjusted weight, and half-step evidence. Those
requirements are incompatible historical records, not aliases or predecessors
that can be relabeled V21S.

The compute-pool conformance checklist remains mandatory and maps without
substitution:

| Checklist obligation | V2.1 / immutable-snapshot coverage |
|---|---|
| cite authorities and complete requirement sets | V21S01, V21S15, V21S16, V21S17, ISP01–ISP07 plus all R01–R16/NDP01–NDP17 |
| leased READY membership, bounded waits, no launched-rank invariant | V21S05, V21S09, V21S10, V21S17 |
| fenced identity, deterministic math, idempotence, stale/corrupt rejection, atomic evidence | V21S01–V21S05, V21S07, V21S08, V21S11, V21S14 |
| exclusive live-state ownership and coherent immutable snapshot | ISP01, ISP03; V21S06 |
| immediate foreground resume and bounded non-Lustre point-to-point transport | ISP02, ISP04; V21S06, V21S09, V21S12 |
| separately bounded atomic apply and nonblocking result failure | ISP05; V21S02, V21S07–V21S08, V21S11 |
| explicit failure/deadline/recovery path and minimum floor | ISP04–ISP05; V21S02, V21S05, V21S08–V21S11, V21S14, V21S17 |
| per-phase/tail overlap evidence and exact commands/artifacts | ISP06–ISP07; V21S13–V21S17 |

## Current architecture boundaries

Production E97 selects ADR-003's fixed-world `train.py` hierarchy. Its
launched-rank collectives are intentionally bounded by the whole child
execution epoch; they do not claim resilient global membership. The split-role
Python TCP/file implementation, native service/manager/direct-memfd path,
cell/owner-tree variants, and communicator-shrink ideas remain non-production
research or debug fixtures. `QuorumTransportServer` and its node-0 transport
remain adjacent legacy/reference code and are not production backends.

`docs/ASYNC_QUORUM_DILOCO.md`, `LocalAsyncDilocoConfig`, and the existing core
tests remain v1 stale-reject reference/scaffolding, not authority for v2.1.
ADR-002 remains the policy authority only for work claiming async-v2.1. Its
versioned ABI/wire/checkpoint identities, exact-token eta-one math, lag
admission, atomic node apply, snapshot pipeline, and semantic validator are
research requirements, not ADR-003 dependencies. Historical v2.0 records
remain readable only as explicit rejection/reference fixtures and cannot cross
a v2.1 boundary.
Current v1 implementation evidence is
`reports/integrate-resilient-pool-v1-20260718.md` plus its committed metrics
JSON. The retained job 5062348 report is in Git commit `20c9d1be`; it is
evidence of five atomic generations and a failed overlap/cadence/idle gate, not
v2.1 acceptance. Failed v2.0 jobs 5066495 and 5068873 are retained as
non-qualifying failure evidence.

## Bounded elastic-research successor tasks

The following backlog is retained for non-production v2.1 research and does
not gate or authorize ADR-003 production. The v2.1 implementation is locally
exercised without a Slurm mutation. Remaining research work must keep the full
elastic conformance checklist in every task's `## Validation`.

1. **Implement the immutable snapshot gate (ISP01–ISP07).** Add the named
   unit/integration/semantic-validator evidence, including coherent capture,
   every blocked background phase, all capacity edges, atomic apply fault
   injection, causal phase timing, and the approximately 200-second tail-stall
   rejection. No Slurm run may claim overlap before these pass.
2. **Qualify atomic restart (V21S11/V21S15/ISP05).** Run the implemented all-eight
   verified-latest reconstruction after a partial/timed-out node apply and
   retain the passing job-5068873-class fault artifact.
3. **Qualify the current source at exactly two nodes (all four namespaces).**
   Run clean/performance, fault/rejoin, and newer-fence fresh-recovery with
   explicit `Partition=batch`, `QOS=debug`; retain numerical/deterministic
   systems checks. Run convergence/model quality separately.
4. **Review promotion and closure (R16; NDP17; V21S16–V21S17; ISP06–ISP07).**
   Only the complete collector-backed exact-source two-node systems pass may
   derive the leased-READY finite scale closure and authorize 8 nodes.
5. **Run the direct systems ladder.** Require exact predecessor passes at
   `8 -> 32 -> 128`, then perform an explicit 256 evidence review. Never infer
   permission from a later, removed, convergence, or v2.0 artifact.

## Explicit unresolved decisions

| Decision | Owner/evidence required | Safe default now |
|---|---|---|
| Frontier allocation fencing/restart authority | Production: stable run identity, atomic latest selection, one fail-stop child, and human-approved fresh submission. Research: scheduler-fence monotonicity, immutable receipt lineage and fresh-allocation recovery. | ADR-003 uses one non-requeueing job and no database/protocol. A path claiming elastic identity still requires immutable claims/receipts/manifests. |
| v1 `Q_min`, `T_min`, optional READY fraction and retry limits | Training + reliability results per model/config | V1 uses test-only explicit values. V2.1 is fixed at `Q_min=2`, `T_min=3,934,080`, fraction disabled, zero retry for its two-node gate. |
| v1 outer optimizer and recovery/export cadence | Numerical parity and measured checkpoint cost | V2.1 is fixed to stateless exact-token `eta_outer=1.0`; restore its outer step/token clock and fail closed if unavailable. |
| Shard count and production byte bounds | Full E97 network/memory measurements at the 2-node rung | Deterministic full-layout mapping and bounded defaults; do not scale before telemetry is accepted. |
| Optional inner optimizer persistence | Cost/benefit measurement | Disposable; restart trainer from committed global state. |
| V2.1 scale closure | Passing current-source two-node arrival/stage distributions plus a separate systems authorization satisfying V21S16–V21S17 | No formula is guessed in advance. Missing evidence or an unexplained constant rejects every scale render/preflight/submission. |
| Simultaneous allocation federation | HA control-plane and partition ADR | Out of MVP; one exclusive scheduler-fenced allocation claim. |
