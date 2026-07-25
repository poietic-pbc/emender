# Resilient DiLoCo Compute Pool

**Status:** Architecture decision and design authority. Version 1
(2026-07-17) remains the strict fresh-only compatibility policy; reviewed
bounded asynchronous behavior is defined only by
[ADR-002: simple asynchronous DiLoCo v2.1](ASYNC_DECOUPLED_DILOCO_V2.md)
(2026-07-25).
**Authority:** Changes to resilient training behavior MUST conform to this
document and, when v2.1 asynchronous mode is selected, ADR-002. The normative compiled
transport, local handoff, wire protocol, and Python/native ABI specialization
is [Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md).
Detailed implementation evidence and gaps live in
[the companion matrix](RESILIENT_DILOCO_GAP_MATRIX.md). An implementation must
satisfy all applicable authorities; the native specialization cannot weaken
the admission, membership, weighting, fencing, atomicity, or recovery rules here.
Existing experiments may finish; these documents do not authorize cancelling
or mutating jobs.

The practical Frontier MVP is one Slurm allocation of any supported size. It
binds a monotonically increasing scheduler fence to an immutable allocation
claim before model load. The in-memory native peer-control protocol owns live
membership, incarnation fencing, generation/commit state, and recovery
handshakes for the allocation. Peers become contributors only after
synchronizing and advertising READY; they may appear late, disappear, and
return without defining a fixed world size or imposing an all-rank barrier.
The same protocol applies to a future very large, potentially system-scale,
single allocation.

**2026-07-25 no-database amendment.** “Lease” below denotes the logical,
deadline-bounded READY/incarnation relationship maintained by native peer
control; it MUST NOT be implemented by a shared-filesystem database or lock.
No allocation supervisor, manager, trainer, native service, diagnostic,
heartbeat, generation, apply, checkpoint, or restart path may open SQLite.
The Slurm job ID (or an authenticated backend-equivalent monotonically
increasing token) is the allocation fence. Immutable allocation claims,
digest-linked commit receipts, complete checkpoint manifests, and node-apply
receipts are durable evidence, not a live coordination store. A compatibility
`latest.json` is never authority. Historical SQLite may be read only by an
offline submit-side migration tool and is not a production launch dependency.

Normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their RFC 2119 meanings.

## Scope and decisions

The goal is a versioned compute pool that makes bounded progress through committed DiLoCo generations despite ordinary process or node churn. Correctness means exact token/sample-weighted aggregation over an explicitly frozen accepted set, fenced atomic publication, bounded waiting, deterministic recovery, and evidence tied to committed generations.

The MVP includes an exclusive scheduler-fenced allocation claim, peer-owned
READY membership, fresh-generation contributions, quorum/deadline closure,
sharded point-to-point aggregation, redistribution, periodic immutable
checkpoints, and fresh-allocation continuation. It does **not** promise dynamic
Slurm node addition, simultaneous allocations, survival of unlimited failures,
exact resurrection of unfinished local work, or continuation after all compute
disappears without a durable checkpoint. Version 1 remains strict `tau = 0`.
The separately versioned `async-decoupled-v2.1-simple` policy uses the finite
bounds and exact math in ADR-002; it is not enabled by changing a v1
configuration field, and v2.0 artifacts are incompatible. Neither mode may
use a failure-sensitive all-rank collective/barrier, Lustre for
update/aggregate/heartbeat/membership/redistribution payloads, a shared
filesystem database for control, or a central full-model broker.

Minimum progress is policy, not launch size: at least `Q_min` complete node-peer contributions and `T_min > 0` accepted tokens are required for a commit. A deployment MAY also require an active-membership fraction, but MUST cap the resulting threshold by the active READY snapshot, not launched ranks. If the floor is unavailable at the generation deadline, the generation does not commit; the owner retries only within a bounded run deadline, then checkpoints any previously committed state, releases the lease, and exits so a later allocation can resume.

## Model and terminology

- **Logical run:** stable run ID, configuration/code identity, generation history, token clock, and immutable checkpoints.
- **Allocation claim:** immutable binding of run/config/allocation identity,
  unique incarnation, scheduler fence, and exact base commit receipt.
- **Native peer-control authority:** the model-free in-memory protocol that
  owns READY membership, incarnation expiry, generation closure, commit
  agreement, node-apply receipts, and recovery handshakes for one allocation.
- **Node peer / manager:** stable node identity plus a unique boot **incarnation**. It owns membership, local supervision, bounded spools, and network transport, but no model or optimizer.
- **Local trainers:** model-owning GPU processes. Their inner optimizer and unfinished generation are local and disposable.
- **Contribution pool:** protocol-visible READY peers and their fenced, checksummed generation contributions.
- **Shard owners:** deterministic owners of parameter/flat-range chunks; no owner is a central full-model broker.
- **Checkpoint publisher:** fenced role that publishes complete immutable global state and advances the durable pointer.
- **External scheduler:** Slurm today or another backend later; it supplies resources and termination signals, not training membership semantics.

```text
 scheduler -> allocation -> publish CLAIM(fence E, exact base receipt)
                              |
             peers: DISCOVER -> BOOT -> SYNC(g) -> READY(peer,incarnation)
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
| `DISCOVER` | Peer locates native peer control and validates run/config/allocation-claim identity. It is not active. |
| `BOOTING` | Manager starts trainers under first-heartbeat and boot deadlines. Slow peers do not block others. |
| `SYNCING` | Peer obtains and checksum-validates the latest committed base and required outer state. |
| `READY / ACTIVE` | Peer advertises `(worker_id, incarnation, base_generation, peer_deadline)` and renews it in memory. It is ACTIVE only while READY, live, and synchronized to the open generation. |
| `DRAINING` | Peer stops new local work, releases buffers after receipts, and may report final state. It is excluded from later snapshots. |
| `EXPIRED` | Lease/heartbeat elapsed or incarnation was superseded. Contributions from it are inadmissible unless already frozen into a commit. |

Worker identity is stable for accounting; every manager restart generates a new incarnation. A slow or late peer synchronizes to the latest commit and normally enters the **next** generation. Disappearance is lease expiry, never an implicit wait. A returning identity uses a new incarnation, discards unfinished work, synchronizes, advertises READY, and becomes eligible for a subsequent round. Active world size is the observed set of live, leased READY peers—not an allocation, launch, or rank invariant.

## Generation protocol and invariants

For committed state `S_g`, native peer control opens generation `g` with
`(run_id, fence_epoch, generation, attempt, base_digest, policy_digest,
deadline)`. It snapshots eligible READY incarnations for accounting, without
waiting for every member. New peers normally defer to `g+1`.

Each admitted peer trains from exactly `S_g` for a bounded local-step or token budget and submits a contribution identity
`(run_id, epoch, g, attempt, worker_id, incarnation, contribution_seq)`, positive accepted-token/sample weight `w_i`, layout/code/base digests, and bounded checksummed chunks. An identity is idempotent: identical replay receives the original result; conflicting reuse is rejected. Corrupt, nonfinite, wrong-layout, wrong-fence, duplicate-conflicting, or stale input is rejected and recorded.

Shard owner `j` maintains exact incremental accumulators `(A_j, W)` where `A_j += w_i * delta_ij` once per accepted identity and `W += w_i` once per complete contribution. Partial contributions never enter the frozen set. Backpressure bounds in-flight bytes and retained generations; senders retain chunks until checksummed receipt or commit/reject, then promptly release them. Ownership is a deterministic function of run policy, generation attempt, and shard ID. Owner loss before commit causes bounded reassignment and replay from retained sender/node spools; otherwise the attempt aborts without publication.

The holder closes at the first configured condition: eligible complete contributions satisfy `Q_min` and `T_min` (and any capped fraction), or the generation deadline arrives. It deterministically freezes a complete accepted set. A deadline with the floor met MAY commit; without it MUST defer/abort. For each shard:

```text
delta_j = sum(i in accepted, w_i * delta_ij) / sum(i in accepted, w_i)
S_(g+1) = OuterApply(S_g, delta, committed_outer_state)
```

The full manifest binds the frozen identities, weights, rejection counts,
shard checksums, base/result digests, optimizer state, token clock, policy,
code, and fence. Commit is atomic: the immutable complete state, manifest, and
digest-linked commit receipt extend the current allocation claim exactly once;
every native peer validates that exact generation/result/token/prior-receipt
identity before advancing its volatile live state or acknowledging the commit.
There is one authoritative result or none.
`latest.json` may mirror a receipt for operators but cannot authorize apply or
restart. Only after all eight local trainer apply receipts are reduced to one
peer-acknowledged node-apply receipt may that node advertise READY for the next
generation.

Version 1 stale policy is strict: accept only
`base_generation == open_generation` and the current attempt/epoch. Late work
is rejected, then the peer catches up. ADR-002 is the only reviewed exception:
its v2.1 policy keeps commit, applied-anchor, result-version, and speculative
window lag distinct, accepts each only through two, drops/catches up at three,
uses exact tokens as the sole quantitative quorum, accepted-token clock, and
deterministic numerical weight, and applies a stateless full average with
`eta_outer = 1.0`. It retains bounded coalescing, K-boundary correction,
atomic eight-trainer node apply, restart, performance, and convergence gates.
Those semantics are not compatible with v1 or historical v2.0 by implication,
field reuse, migration, or telemetry relabeling.

## Allocation claim, state, and recovery

Before loading a model or mutating run state, an allocation publishes an
immutable claim containing run ID, Slurm job/allocation identity (or backend
equivalent), unique allocation incarnation, monotonically increasing scheduler
fence, protocol/config identity, and the exact base generation/commit-receipt
digest. A claim whose fence is not strictly newer than the current claim is a
successful no-op exit: do not load the checkpoint or write run state. Claim
creation is create-once and conflict detecting; it has no renewal transaction,
database, mutable lock row, or filesystem heartbeat.

The scheduler owns allocation lifetime. Native peer control binds every live
command to the current claim/fence and owns peer expiry, manager/trainer
incarnations, READY membership, generation admission, exact-once commit
agreement, and recovery handshakes. A strictly newer claim makes every old
membership record, contribution, peer command, commit, checkpoint publication,
and apply receipt stale. Native services independently reject lower-fence or
wrong-incarnation commands and frames. Loss of peer-control authority stops
new generation admission and publication immediately.

Globally authoritative restart state is only the immutable
claim/commit-receipt/checkpoint-manifest chain: model state, required outer
optimizer state, committed generation and accepted-token clock,
policy/code/layout identities, result roots, frozen membership, node-apply
receipts, and fence/incarnation. These records MAY live on Lustre because they
are append-only restart evidence, never polled live membership/heartbeat state.
No correctness or liveness decision may depend on a shared-filesystem
database. Membership, heartbeats, accumulators, cached bases, unfinished
trainer state, and inner optimizer work are volatile and MAY be reconstructed
or discarded. Local inner optimizer restoration is optional and never a
prerequisite for correctness.

After all compute disappears, a later allocation publishes a strictly newer
claim anchored to the newest valid commit receipt, reloads and independently
verifies the complete immutable checkpoint/manifest chain, and rejoins through
the native peer recovery handshake. Missing/corrupt model, outer state, token
clock, result root, fence/incarnation, or apply evidence is unrecoverable and
fails closed. No database bootstrap or mutable `latest` pointer is consulted.

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
| Peer-control leader loss | Stop publication; reconstruct its volatile state from the exact peer-agreed commit and immutable receipt, with a new incarnation, or exit for a later allocation. |
| Whole-allocation loss | Later job publishes a newer scheduler-fenced claim and resumes the newest valid immutable commit/checkpoint chain. |
| Shared-filesystem/checkpoint outage | Native live control may finish only already admitted bounded work; do not publish a commit/checkpoint and never substitute mutable filesystem state for peer authority. |
| Return/rejoin | New incarnation, latest-state sync, READY lease, next admissible generation; never resurrect old work. |

Whole-allocation restart is reserved for allocation/scheduler loss, run-lease loss, unrecoverable partition, quorum collapse, or an explicit fail-fast deadline—not an ordinary trainer/node failure.

## Correctness, observability, and gates

Changing participation changes effective batch and update variance. Weight by accepted tokens/samples (not nominal steps or ranks), and drive learning-rate/data schedules by committed accepted tokens and/or committed generations. Manifests MUST expose accepted/missing membership, weights, effective batch, deadline reason, staleness and rejection counts. Research must evaluate bias from heterogeneous data/token counts, quorum selection, outer-optimizer sensitivity, and reproducibility. Tests MUST compare incremental sharded math to a high-precision single-process reference; the all-fresh equal-weight/full-cohort case must match synchronous DiLoCo within a stated tolerance.

Every stage has a configured deadline: first heartbeat, boot/sync, generation
progress, aggregation/freeze, apply/redistribution, peer recovery, checkpoint
publication, and graceful shutdown. Logs and volatile heartbeats diagnose; an
immutable committed generation/checkpoint is authoritative restart evidence.

Scale admission is sequential: deterministic simulation/unit/reference math;
then a 2-node scripted gate covering delayed boot, late join, disappearance,
new-incarnation rejoin, stale/duplicate rejection, owner failure, and
continuation by a fresh allocation/fence; then the strict
`4 -> 8 -> 16 -> 32 -> 64 -> 256` ladder. Each rung proves bounded
memory/backpressure, deadlines, committed-token accounting, restart, and
numerical tolerance before the next. The v2.1 scale path additionally requires
ADR-002's reviewed finite close over its leased READY snapshot; it never closes
at the two-node floor merely because two nodes arrived. A future enormous
Frontier allocation uses this identical protocol.

### Conformance checklist (required in every implementation/runner/scale task Validation)

- Cite this document/version and name the requirement IDs from the companion matrix.
- Show peer-owned READY membership, bounded waits, and no launched-rank/all-rank invariant.
- Prove the rendered compute-role closure has no SQLite import, connection,
  database path, store construction, filesystem lock, or metadata heartbeat.
- Show fenced generation identity, deterministic weighted math, idempotence, stale/corrupt rejection, and atomic committed evidence.
- Show bounded non-Lustre hot-path transport, backpressure/release, and no central full-model broker.
- Exercise the applicable failure/deadline and recovery path; state the minimum progress floor.
- Report exact validation commands and committed-generation/checkpoint artifacts; scale tasks must pass every prior rung.
- A bounded asynchronous task must additionally cite ADR-002 and
  V21S01–V21S17, report commit/applied-anchor/result/speculative clocks
  honestly, and distinguish the training-lane SLO from checkpoint correctness
  latency. A v1 task must state `tau = 0` and cannot claim required
  generation-g work overlaps a generation-(g+1) window that starts from
  committed `S_(g+1)`.

## Simple asynchronous v2.1 policy

ADR-002 resolves the bounded asynchronous policy with one concise profile.
Local K40 windows continue from resident worker state while prior
contributions move through the bounded background system. Contributions carry
fenced worker/incarnation/sequence/window identity, base version/digest, exact
tokens, policy/layout/code/payload digests, and distinct lag clocks. Each
stable node worker exposes at most one immutable owned eight-trainer descriptor
and one mutable cumulative adjacent interval. The holder admits at most one
contribution per stable worker per transition and requires the exact two-node
floors `Q_min = 2` and `T_min = 3,934,080`.

Commit, applied-anchor, result-version, and speculative-window lag are separate
integers with maximum two. Lag three drops the stale contribution/result or
pauses before another local window for point-to-point catch-up. The aggregate
is the deterministic binary64 exact-token mean and the stateless outer update
is `S_(g+1) = S_g + mean(delta)` with `eta_outer = 1.0`. There is no
staleness multiplier or second numerical-weight field.

A verified capacity-one latest mailbox is applied only at a K boundary by
translating ScheduleFree `x`, `z`, and the mutable interval start with
`(new_global - old_global) - accepted_local_delta_sum`. All eight node trainers
prepare and recover the same version before the node advertises READY. One
owned cohort, one mutable cohort, finite native credits/replay, bounded mailbox
staging, explicit pause/drop/catch-up rules, and fixed deadlines bound memory
and time; a third dense cohort is forbidden. `OWNED` transfers descriptor
responsibility to the persistent service, so trainers do not wait for fabric
send or receipt completion.

V2.1 preserves the v1 scheduler claim/fence, READY membership, model-free compiled
point-to-point transport, no-all-rank-wait, no-Lustre/Python-dense-hot-path,
no-central-full-model-broker, and atomic checkpoint requirements. It requires
the v2.1 policy/schema/digest and native ABI/protocol boundary; v1 and v2.0
records cannot be renamed. Qualification is exactly two nodes until numerical,
failure/restart, clean performance, deterministic replay, and predeclared
three-seed convergence gates pass and a separate review authorizes only the
next rung.

## Native data-plane binding

The production elastic dense path is bound to
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
version 1 (requirements NDP01–NDP17). The model-free native peer-control
protocol owns the allocation fence/incarnations, READY membership, generation
admission/closure/commit state, and recovery handshakes. Python remains
responsible for scheduler adaptation, outer/checkpoint policy and publication,
and Slurm supervision. A persistent model-free C++17 service on every node owns local
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

Frontier/Slurm supplies a fixed allocation envelope; the supervisor publishes
one immutable scheduler-fenced claim, launches independent node
managers/trainers and one persistent native data service per node, and reacts
to Slurm shutdown signals. Native peer control maps shard owners
deterministically among available peers and exchanges opaque service endpoints.
The native services select libfabric `FI_EP_RDM` with exact provider `cxi`.
Slurm node count is capacity only. Other backends map an authenticated
monotonic allocation fence, host agents, and local/network transports to the
same identities and protocol.

**ADR-001 (amended 2026-07-25):** The MVP chooses exactly one
scheduler-fenced allocation claim for operational simplicity and safe
continuation across queued jobs. Simultaneous independent allocations do not
join one live run. The native peer protocol, not a shared database, owns live
control. A future federation requires a separately reviewed highly available
control service, cross-allocation authentication, shard-owner placement, and
partition semantics while preserving every fence and commit invariant.

Unresolved decisions are intentionally explicit: v1 production `Q_min`,
`T_min`, optional
fraction and retry deadlines per model size; v1 outer optimizer and checkpoint
cadence; production shard placement/reassignment; and whether trainer inner
state is ever checkpointed. ADR-002 fixes asynchronous math only for its exact
two-node v2.1 profile; broader promotion remains gated by its acceptance
criteria and scale-closure review. Until resolved by a reviewed ADR/config,
implementations fail closed or use test-only values.
