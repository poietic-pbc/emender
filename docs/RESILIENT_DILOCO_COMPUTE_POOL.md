# Resilient DiLoCo Compute Pool

**Status:** Architecture decision and design authority. Version 1
(2026-07-17) remains the strict fresh-only compatibility policy; reviewed
bounded-lag behavior is defined only by
[ADR-002: bounded-lag asynchronous DiLoCo v2](ASYNC_DECOUPLED_DILOCO_V2.md)
(2026-07-24).
**Authority:** Changes to resilient training behavior MUST conform to this
document and, when bounded-lag mode is selected, ADR-002. The normative compiled
transport, local handoff, wire protocol, and Python/native ABI specialization
is [Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md).
Detailed implementation evidence and gaps live in
[the companion matrix](RESILIENT_DILOCO_GAP_MATRIX.md). An implementation must
satisfy all applicable authorities; the native specialization cannot weaken
the lease, membership, weighting, fencing, atomicity, or recovery rules here.
Existing experiments may finish; these documents do not authorize cancelling
or mutating jobs.

The practical Frontier MVP is one Slurm allocation of any supported size. It acquires an exclusive, expiring, fenced lease for a logical run. Peers inside that allocation become contributors only after synchronizing and advertising READY; they may appear late, disappear, and return without defining a fixed world size or imposing an all-rank barrier. The same protocol applies to a future very large, potentially system-scale, single allocation.

Normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their RFC 2119 meanings.

## Scope and decisions

The goal is a versioned compute pool that makes bounded progress through committed DiLoCo generations despite ordinary process or node churn. Correctness means exact token/sample-weighted aggregation over an explicitly frozen accepted set, fenced atomic publication, bounded waiting, deterministic recovery, and evidence tied to committed generations.

The MVP includes an exclusive allocation lease, leased READY membership, fresh-generation contributions, quorum/deadline closure, sharded point-to-point aggregation, redistribution, periodic immutable checkpoints, and fresh-allocation continuation. It does **not** promise dynamic Slurm node addition, simultaneous allocations, survival of unlimited failures, exact resurrection of unfinished local work, or continuation after all compute disappears without a durable checkpoint. Version 1 remains strict `tau = 0`. The separately versioned v2 performance policy uses the finite bound and exact math in ADR-002; it is not enabled by changing a v1 configuration field. Neither mode may use a failure-sensitive all-rank collective/barrier, Lustre for update/aggregate/heartbeat/membership/redistribution payloads, or a central full-model broker.

Minimum progress is policy, not launch size: at least `Q_min` complete node-peer contributions and `T_min > 0` accepted tokens are required for a commit. A deployment MAY also require an active-membership fraction, but MUST cap the resulting threshold by the active READY snapshot, not launched ranks. If the floor is unavailable at the generation deadline, the generation does not commit; the owner retries only within a bounded run deadline, then checkpoints any previously committed state, releases the lease, and exits so a later allocation can resume.

## Model and terminology

- **Logical run:** stable run ID, configuration/code identity, generation history, token clock, and immutable checkpoints.
- **Allocation lease holder:** the sole MVP writer/committer; normally a model-free control process in the allocation.
- **Node peer / manager:** stable node identity plus a unique boot **incarnation**. It owns membership, local supervision, bounded spools, and network transport, but no model or optimizer.
- **Local trainers:** model-owning GPU processes. Their inner optimizer and unfinished generation are local and disposable.
- **Contribution pool:** protocol-visible READY peers and their fenced, checksummed generation contributions.
- **Shard owners:** deterministic owners of parameter/flat-range chunks; no owner is a central full-model broker.
- **Checkpoint publisher:** fenced role that publishes complete immutable global state and advances the durable pointer.
- **External scheduler:** Slurm today or another backend later; it supplies resources and termination signals, not training membership semantics.

```text
 scheduler -> allocation -> acquire RUN LEASE(epoch E)
                              |
             peers: DISCOVER -> BOOT -> SYNC(g) -> READY(lease,incarnation)
                              |                    |
                              +---- local train <--+
                                        |
          contribution(g,E,id,weight,chunks) -> deterministic shard owners
                                        |
                 quorum/deadline -> freeze set -> exact weighted aggregate
                                        |
                     atomic COMMIT(g+1,E) -> redistribute / checkpoint
                                        |
                         late/rejoining peers SYNC(g+1), join next round
```

## Lifecycle and membership

| State | Admission and transition |
|---|---|
| `DISCOVER` | Peer locates the lease holder and validates run/config identity. It is not active. |
| `BOOTING` | Manager starts trainers under first-heartbeat and boot deadlines. Slow peers do not block others. |
| `SYNCING` | Peer obtains and checksum-validates the latest committed base and required outer state. |
| `READY / ACTIVE` | Peer advertises `(worker_id, incarnation, base_generation, lease_expiry)` and renews it. It is ACTIVE only while READY, live, and synchronized to the open generation. |
| `DRAINING` | Peer stops new local work, releases buffers after receipts, and may report final state. It is excluded from later snapshots. |
| `EXPIRED` | Lease/heartbeat elapsed or incarnation was superseded. Contributions from it are inadmissible unless already frozen into a commit. |

Worker identity is stable for accounting; every manager restart generates a new incarnation. A slow or late peer synchronizes to the latest commit and normally enters the **next** generation. Disappearance is lease expiry, never an implicit wait. A returning identity uses a new incarnation, discards unfinished work, synchronizes, advertises READY, and becomes eligible for a subsequent round. Active world size is the observed set of live, leased READY peers—not an allocation, launch, or rank invariant.

## Generation protocol and invariants

For committed state `S_g`, the lease holder opens generation `g` with `(run_id, fence_epoch, generation, attempt, base_digest, policy_digest, deadline)`. It snapshots eligible READY incarnations for accounting, without waiting for every member. New peers normally defer to `g+1`.

Each admitted peer trains from exactly `S_g` for a bounded local-step or token budget and submits a contribution identity
`(run_id, epoch, g, attempt, worker_id, incarnation, contribution_seq)`, positive accepted-token/sample weight `w_i`, layout/code/base digests, and bounded checksummed chunks. An identity is idempotent: identical replay receives the original result; conflicting reuse is rejected. Corrupt, nonfinite, wrong-layout, wrong-fence, duplicate-conflicting, or stale input is rejected and recorded.

Shard owner `j` maintains exact incremental accumulators `(A_j, W)` where `A_j += w_i * delta_ij` once per accepted identity and `W += w_i` once per complete contribution. Partial contributions never enter the frozen set. Backpressure bounds in-flight bytes and retained generations; senders retain chunks until checksummed receipt or commit/reject, then promptly release them. Ownership is a deterministic function of run policy, generation attempt, and shard ID. Owner loss before commit causes bounded reassignment and replay from retained sender/node spools; otherwise the attempt aborts without publication.

The holder closes at the first configured condition: eligible complete contributions satisfy `Q_min` and `T_min` (and any capped fraction), or the generation deadline arrives. It deterministically freezes a complete accepted set. A deadline with the floor met MAY commit; without it MUST defer/abort. For each shard:

```text
delta_j = sum(i in accepted, w_i * delta_ij) / sum(i in accepted, w_i)
S_(g+1) = OuterApply(S_g, delta, committed_outer_state)
```

The full manifest binds the frozen identities, weights, rejection counts, shard checksums, base/result digests, optimizer state, token clock, policy, code, and fence. Commit is atomic: immutable state and manifest become durable before a compare-and-swap-like `latest` pointer advances under the current allocation fence. There is one authoritative result or none. Only after commit may redistribution announce `g+1`; peers checksum it, apply/catch up, and advertise READY for the next generation.

Version 1 stale policy is strict: accept only
`base_generation == open_generation` and the current attempt/epoch. Late work
is rejected, then the peer catches up. ADR-002 is the only reviewed exception:
its distinct experimental v2 policy admits `0 <= global lag <= 6` with a
promotion target of two, separately bounds speculative local-window lag,
uses exact token-times-linear-lag weights, applies a stateless half-step outer
update, and has bounded coalescing, safe-boundary correction/rebase, restart,
performance, and convergence gates. Those semantics are not compatible with
v1 by implication or telemetry relabeling.

## Admission lease, state, and recovery

Before loading a model or mutating run state, an allocation attempts to acquire a durable lease containing run ID, Slurm job/allocation identity (or backend equivalent), allocation incarnation, monotonically increasing fencing epoch/token, acquired/renewed/expires timestamps, and protocol/config identity. Acquisition and renewal MUST be atomic and linearizable enough to reject stale owners. Failure to acquire is a successful no-op exit: do not load the checkpoint or write run state.

A static lock is insufficient because its owner can die or partition indefinitely and a stale process can resume after replacement. The expiring lease permits takeover; the strictly newer fence makes every membership record, contribution attempt, commit, checkpoint, and latest-pointer update from the former owner fail. Loss of renewal stops new generation admission and publication immediately. Security assumes allocation identities cannot forge a newer fence and the durable control mechanism enforces authenticated writer/CAS permissions; tensor confidentiality is outside MVP scope.

Globally authoritative state is only a committed manifest/checkpoint chain: model state, required outer optimizer state, committed generation and accepted-token clock, policy/code/layout identities, and lease fence. Small lease/checkpoint metadata MAY use an approved durable control store, including Lustre if atomicity is proven. Dense hot-path data MUST NOT. Membership, heartbeats, accumulators, cached bases, unfinished trainer state, and inner optimizer work MAY be reconstructed or discarded. Local inner optimizer restoration is optional and never a prerequisite for correctness.

After all compute disappears, a later allocation acquires a newer fence and loads the latest complete immutable checkpoint; work after that checkpoint may be lost. Missing/corrupt global model or required outer state is unrecoverable and fails closed. Checkpoint publication verifies completeness/digests and current fence before and while advancing `latest`.

## Failure semantics

| Failure | Required response |
|---|---|
| Trainer loss | Manager expires/restarts it with a new incarnation; discard unfinished local work; progress if floor remains. |
| Slow/stuck boot | First-heartbeat/boot deadline; exclude without blocking READY peers. |
| Manager or node loss | Expire its lease and contributions not already frozen; allocation continues if floor remains. Rejoin via SYNC with new incarnation. |
| Aggregation owner loss before commit | Reassign deterministic shards and replay retained chunks, or abort the attempt at deadline; never partial commit. |
| Late/stale or duplicate input | Reject strict stale/conflicting duplicate; idempotently acknowledge identical replay; instruct catch-up. |
| Corrupt/nonfinite input | Reject, retain evidence, and quarantine/expire the source according to policy. |
| Network partition | Isolate unreachable peers; only the current fenced holder may commit. Pause when quorum/fence safety is uncertain. |
| Lease-holder loss | Stop publication; allocation may elect/restart only through the durable newer-fence protocol, otherwise exit for later allocation. |
| Whole-allocation loss | Later job takes a newer lease and resumes latest immutable checkpoint. |
| Shared-filesystem/control outage | Continue only while lease safety and hot-path state are valid; do not checkpoint/publish. Pause before lease expiry or checkpoint deadline. |
| Return/rejoin | New incarnation, latest-state sync, READY lease, next admissible generation; never resurrect old work. |

Whole-allocation restart is reserved for allocation/scheduler loss, run-lease loss, unrecoverable partition, quorum collapse, or an explicit fail-fast deadline—not an ordinary trainer/node failure.

## Correctness, observability, and gates

Changing participation changes effective batch and update variance. Weight by accepted tokens/samples (not nominal steps or ranks), and drive learning-rate/data schedules by committed accepted tokens and/or committed generations. Manifests MUST expose accepted/missing membership, weights, effective batch, deadline reason, staleness and rejection counts. Research must evaluate bias from heterogeneous data/token counts, quorum selection, outer-optimizer sensitivity, and reproducibility. Tests MUST compare incremental sharded math to a high-precision single-process reference; the all-fresh equal-weight/full-cohort case must match synchronous DiLoCo within a stated tolerance.

Every stage has a configured deadline: first heartbeat, boot/sync, generation progress, aggregation/freeze, apply/redistribution, lease renewal, checkpoint publication, and graceful shutdown. Logs and volatile heartbeats diagnose; an immutable committed generation/checkpoint is authoritative progress evidence.

Scale admission is sequential: deterministic simulation/unit/reference math; then a 2-node scripted gate covering delayed boot, late join, disappearance with continued commits, new-incarnation rejoin, stale/duplicate rejection, owner failure, and continuation by a fresh allocation/fence; then 4, 8, 32, 64, 256+ nodes. Each rung proves bounded memory/backpressure, deadlines, committed-token accounting, restart, and numerical tolerance before the next. A future enormous Frontier allocation uses this identical protocol.

### Conformance checklist (required in every implementation/runner/scale task Validation)

- Cite this document/version and name the requirement IDs from the companion matrix.
- Show READY leased membership, bounded waits, and no launched-rank/all-rank invariant.
- Show fenced generation identity, deterministic weighted math, idempotence, stale/corrupt rejection, and atomic committed evidence.
- Show bounded non-Lustre hot-path transport, backpressure/release, and no central full-model broker.
- Exercise the applicable failure/deadline and recovery path; state the minimum progress floor.
- Report exact validation commands and committed-generation/checkpoint artifacts; scale tasks must pass every prior rung.
- A bounded-lag task must additionally cite ADR-002 and V2A01–V2A18, report
  base/commit/apply/window lag honestly, and distinguish the training-lane SLO
  from checkpoint correctness latency. A v1 task must state `tau = 0` and
  cannot claim required generation-g work overlaps a generation-(g+1) window
  that starts from committed `S_(g+1)`.

## Bounded-lag asynchronous v2 policy

ADR-002 resolves the previously open stale-update decision with one concrete
initial policy. Local K40 windows continue from a worker-local state while
prior contributions move through the bounded background system. Contributions
carry fenced worker/incarnation/sequence/window identity, base version and
digest, exact tokens, policy/layout/code digests, payload digest, and measured
global and local-window lag. Each worker exposes at most one sealed immutable
descriptor and cumulatively coalesces adjacent K40 windows into one bounded
next interval while the native service owns it. The holder admits at most one
contribution per worker for each global transition, requires the exact
two-node `Q_min/T_min` floors, and rejects global lag greater than the
experimental hard bound six. Promotion requires the lower global-lag and
speculative-window targets of two.

The v2 aggregate uses exact integer
`tokens * (7 - commit_lag)` weights and the outer equation
`S_(g+1) = S_g + 0.5 * weighted_mean(delta)`. A verified capacity-one latest
mailbox is applied only at a K-window boundary by translating both
ScheduleFree `x` and `z` with
`(new_global - old_global) - accepted_local_delta_sum`; audited inner moments
remain local and restartable. One sealed descriptor, one mutable cumulative
interval, finite native credits/replay, a capacity-one visible mailbox,
explicit pause/drop/catch-up rules, and a 420-second correctness deadline bound
memory and time. Local `OWNED` transfers descriptor responsibility to the
persistent service; the trainer does not wait for send or receipt completion.

The v2 policy preserves the v1 lease/fence, READY membership, model-free
compiled point-to-point transport, no-all-rank-wait, no-Lustre-dense-hot-path,
no-central-full-model-broker, and atomic committed-state requirements. It
requires a versioned native protocol/ABI extension; the v1 generation field and
one-open-generation buffer contract cannot simply be renamed. Qualification
is two-node only until numerical, failure/restart, real-decoupling performance,
and convergence/reproducibility gates all pass and a later review authorizes
the next rung.

## Native data-plane binding

The production elastic dense path is bound to
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
version 1 (requirements NDP01–NDP17). Python remains the control plane for the
allocation lease/fence, READY membership, generation admission and closure,
owner reassignment policy, checkpoint policy/publication, and Slurm
supervision. A persistent model-free C++17 service on every node owns local
XPMEM/memfd handoff, exact native reduction, libfabric `FI_EP_RDM`/Frontier
`cxi` payload movement, bounded replay, and redistribution. Python TCP and
Python object serialization MUST NOT carry production dense contributions or
aggregates.

The native service is not a second committer and cannot infer membership or
closure from transport reachability. Python freezes only locally complete,
checksummed, retained contributions; native owners execute that immutable set.
All fabric operations are bounded point-to-point operations. The elastic
backend MUST NOT initialize MPI or require an all-rank collective, including
during endpoint exchange, failure handling, redistribution, or shutdown.

The compiled Cray-MPICH helper remains a numerical/performance reference and an
explicit fixed-world fallback. Its launched-rank collectives do not satisfy the
elastic failure-domain requirements. No real model or 4+ node native job is
admissible until the exact-code full-layout two-node synthetic CXI artifact
required by NDP17 has passed and been retained. This adds a gate; it does not
replace the sequential lifecycle/failure/restart ladder above.

## Backend mapping and decision record

Frontier/Slurm supplies a fixed allocation envelope; one model-free allocation holder acquires the run lease, launches independent node managers/trainers and one persistent native data service per node, maps shard owners deterministically among available peers, and reacts to Slurm shutdown signals. The native services select libfabric `FI_EP_RDM` with exact provider `cxi`; the Python holder exchanges their opaque endpoints through leased membership. Slurm node count is capacity only. Hyperscale-local infrastructure maps a service lease, host agents, and local/NVMe/network transports to the same identities, lifecycle, generations, and commits, using a separately qualified provider without changing the protocol. Other schedulers do likewise.

**ADR-001:** The MVP chooses exactly one allocation write/commit lease for operational simplicity and safe continuation across queued jobs. Simultaneous independent allocations do not join one live run. A future federation may allow them to join through the same pool protocol, but requires a highly available lease/control service, cross-allocation discovery/authentication, shard-owner placement, and partition semantics. It MUST preserve all fences, membership, generation, weighting, and commit invariants; it is an extension, not the organizing design.

Unresolved decisions are intentionally explicit: the approved durable
lease/CAS mechanism on Frontier; v1 production `Q_min`, `T_min`, optional
fraction and retry deadlines per model size; v1 outer optimizer and checkpoint
cadence; production shard placement/reassignment; and whether trainer inner
state is ever checkpointed. ADR-002 resolves bounded-lag math only for its
initial two-node v2 profile; broader promotion remains gated by its acceptance
criteria. Until resolved by a reviewed ADR/config, implementations fail closed
or use test-only values.
