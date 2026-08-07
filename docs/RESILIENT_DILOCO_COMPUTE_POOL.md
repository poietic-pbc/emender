# Resilient DiLoCo Compute Pool

**Status:** Architecture decision and design authority. Version 1
(2026-07-17) and the 2026-07-25 async-v2.1 amendment are retained as the
elastic-compute-pool research design. The **production E97 recovery path** was
amended on 2026-07-31 to use fixed-world same-allocation execution epochs, as
specified below.
**Authority:** Production E97 launchers MUST follow the 2026-07-31 decision in
this document and the applicable/retired crosswalk in
[the companion matrix](RESILIENT_DILOCO_GAP_MATRIX.md). Bounded async-v2.1
research additionally conforms to
[ADR-002: simple asynchronous DiLoCo v2.1](ASYNC_DECOUPLED_DILOCO_V2.md), and
native elastic research conforms to
[Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md).
Those research specializations remain normative for work claiming their
identities, but are not production E97 launch dependencies. Existing evidence
is retained; this decision does not authorize cancelling or relabeling it.

Async-v2.1 research qualification and scale submission additionally use the
reviewed
[execution-source identity and durable scheduler transaction](ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md).
Evidence-only commits may advance without changing that immutable execution
identity, but every operational tracked byte and every separately bound
native/data/tokenizer/seed identity remains fail closed.

The elastic research MVP is one Slurm allocation of any supported size. It
binds a monotonically increasing scheduler fence to an immutable allocation
claim before model load. The in-memory native peer-control protocol owns live
membership, incarnation fencing, generation/commit state, and recovery
handshakes for the allocation. Peers become contributors only after
synchronizing and advertising READY; they may appear late, disappear, and
return without defining a fixed world size or imposing an all-rank barrier.
This remains the research design for a future elastic, potentially
system-scale allocation; ADR-003, not this protocol, is the selected production
E97 path.

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

## ADR-003: production same-allocation execution epochs (2026-07-31)

**Decision evidence.** Frontier job **5125415** physically proved the selected
failure boundary on the actual `train.py` path: a 16-rank, two-node child
completed hierarchical 67,108,864-element-bucket merges and atomically
published/reloaded a checkpoint; a subsequently damaged child terminated
nonzero in a bounded 99 seconds without publishing a checkpoint; the parent
batch allocation survived; and a fresh 8-rank child on one whole node, with a
new process group and `MASTER_PORT`, resumed the committed checkpoint and
merged successfully. The retained report is
[`validation/direct-same-allocation-trainpy-restart-5125415.md`](validation/direct-same-allocation-trainpy-restart-5125415.md).
The report correctly called the experiment nonconforming research when it ran;
this reviewed amendment uses it as decision evidence rather than retroactively
claiming v1, v2.1, native-data-plane, or overlap conformance.

Production E97 uses one Slurm batch parent and fixed-world `train.py`. Following
the 2026-08-04 operator safety decision, each submitted job has exactly one
**execution epoch**: there is no automatic child restart or scheduler requeue.

1. A caller supplies a stable `RUN_ID`; its run directory MUST NOT contain the
   Slurm job ID. The same exact checkpoint directory and atomic
   `train/latest.pt` survive a failed job and a human-approved replacement job.
2. Each epoch is a new child `srun`, process set, distributed process group, and
   `MASTER_PORT`. It resumes only the stable atomic `latest.pt`. It MUST NOT
   reuse a damaged communicator, unfinished tensors, a partial checkpoint, or
   any failed process.
3. `train.py` remains the data plane: singleton GPU islands, hierarchical RCCL
   merges in 67,108,864-element buckets, an exact output directory shared by
   every supervised epoch (so retention applies across restarts), and synchronous rank-0 checkpoint
   publication by temporary file, `os.replace`, temporary symlink, and
   `os.replace`. The launcher may advance its stable pointer only from a
   readable epoch `latest.pt`; temporary or bare checkpoint files are never
   candidates.
4. `srun --kill-on-bad-exit` plus finite wait/TERM/KILL deadlines bounds failed
   step teardown. Production MUST NOT use `--no-kill`, relaunch a child, reduce
   the world, or preserve a failed allocation. Any rank, node, collective, or
   child failure terminates the batch job nonzero. There is no rank-level
   elasticity or communicator shrink.
5. Production submissions MUST request `--no-requeue` and verify scheduler
   `Requeue=0`. No launcher path may call `scontrol requeue`. Allocation loss,
   timeout, signal, or child failure stops the job; a human inspects durable
   evidence and explicitly approves a fresh immutable submission anchored to
   atomic `latest.pt`.
6. Default E97 policy is synchronous **K40**, `save_every=200` local steps
   (five outer merges), and `keep_checkpoints=2`, while retaining `train.py`'s
   final and pre-walltime checkpoint behavior. Production passes no validation
   dataset or validation/held-out option: training performs no inline
   validation. These values have explicit launcher overrides, but every
   periodic save MUST remain K-aligned.

The approved production systems ladder is **8 -> 32 -> 128**. Job 5125415 is
the direct two-node decision observation; it is not relabeled as an async-v2.1
qualification rung. Each production rung requires an immutable exact-source
pass from its immediate predecessor, and 256 or another topology requires a
new human review. No scale submission is authorized merely by editing this
architecture.

**Retired from the production path, retained as research.** Dynamic leased
READY membership, async-v2.1/V21S01–V21S17, ISP01–ISP07 background snapshot and
apply, cell layouts, owner-tree aggregation, the elastic native service, and
communicator shrink are not production E97 requirements. No evidence is
deleted. In particular, production does **not** claim R02–R06/R08–R11 dynamic
pool semantics, NDP02's no-all-rank property, NDP15 background checkpointing,
NDP17's native G2–G6 chain, or any V21S/ISP overlap gate. The applicable safety
intent is R07/R12 atomic committed restart, R14/NDP13 bounded termination,
R16 evidence discipline, and NDP15 checkpoint atomicity; their elastic/native
clauses are explicitly unclaimed. ADR-003 adds no hashing, background
checkpointing, database, membership service, or coordination protocol.

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

### Immutable snapshot overlap contract

For bounded asynchronous modes, the trainer exclusively owns the live mutable
model, optimizer, iterator, and hidden state. At a K boundary it may interrupt
foreground work only to capture and admit a coherent, fenced immutable
snapshot into a preallocated double buffer, copy-on-write view, or equivalent
bounded mechanism. Snapshot bytes MUST represent one safe boundary. Copying
directly from weights while an optimizer can mutate them, or allowing a
background worker to read the live model or optimizer, is forbidden.

Admission transfers the immutable snapshot and its lifetime to the bounded
background system. The trainer then resumes immediately on its mutable state:
it MUST NOT wait for discovery, quorum, publication or hashing, network
progress, aggregation, checkpoint I/O, result readiness, or another trainer.
Native background workers may publish, aggregate, validate, and checkpoint
only admitted immutable snapshots. Every snapshot buffer and result mailbox is
capacity bounded; full capacity causes an explicit skip, replacement, or
defer under policy and MUST NOT turn backpressure into a foreground collective
wait.

A complete, verified result may be atomically applied or swapped only at a
later safe K boundary under a separately bounded foreground pause. If it is
late, absent, invalid, failed, or cannot be prepared completely within the
apply bound, it is skipped or deferred under the accepted lag policy. Training
does not wait and no subset of model, optimizer points, or node trainers may
observe a partial application. In steady state, snapshot capture/admission and
result apply/swap are the only permitted foreground interruption categories;
bootstrap, recovery, and terminal drain are separately labeled lifecycle
states and cannot be reported as overlap.

Telemetry MUST time, with causal snapshot/result identities, these disjoint
phases: `freeze_snapshot`, `snapshot_admission`, `publish_network`,
`aggregation`, `checkpoint`, `result_wait`, and `apply_swap`, plus total
foreground idle. `result_wait` is background result age/availability; its
foreground-wait component must be zero. Snapshot/admission and apply/swap each
have predeclared finite pause budgets and report every event, maximum, and p99.
For ADR-002's exact two-node profile those budgets are respectively `1 s`
through `OWNED` and `60 s` for the complete all-eight apply/swap transaction.
Checkpoint/restart correctness is necessary but does not prove overlap.
Median-only cadence, checkpoint count, or foreground-idle fraction cannot hide
tail stalls: bursty alternating K windows with approximately 200-second gaps
fail the overlap gate even when their medians and checkpoints look healthy.

Scale admission is sequential. Deterministic simulation/unit/reference math
must pass first. The current execution source is then physically qualified at
exactly two nodes with a clean systems/overlap pass followed by the complete
fault campaign, including successful newer-fence fresh-allocation recovery.
Only that exact-source machine pass may authorize the short direct systems
ladder `2 -> 8 -> 32 -> 128`; every live rung requires the immutable pass from
its immediate predecessor. After 128, 256 nodes is an explicit evidence review
only: there is no automatically authorized 256-node runner or submission.
Four-, 16-, and 64-node rungs are not part of this policy.

Each live rung proves bounded memory/backpressure, deadlines,
committed-token accounting, restart, numerical tolerance, coherent immutable
snapshot admission, immediate trainer resume, background compiled-CXI
exchange/aggregation/checkpointing, later atomic apply, causal phase telemetry,
and bounded foreground interruption before the next. The v2.1 scale path
additionally requires ADR-002's reviewed finite close over its leased READY
snapshot; it never closes at the two-node floor merely because two nodes
arrived. Convergence and model quality remain a separate study and neither
authorize nor block this systems ladder.

### Conformance checklist (required in every implementation/runner/scale task Validation)

- Cite this document/version/decision and name the applicable and explicitly
  retired requirement IDs from the companion matrix. Production ADR-003 tasks
  MUST NOT claim elastic/native/v2.1 requirements that the fixed-world path
  intentionally retires.
- Elastic research must show peer-owned READY membership, bounded waits, and no
  launched-rank/all-rank invariant. Production ADR-003 instead shows one
  bounded fixed-world child and no attempt to preserve, shrink, or automatically
  relaunch a broken all-rank communicator.
- Prove the rendered compute-role closure has no SQLite import, connection,
  database path, store construction, filesystem lock, or metadata heartbeat.
- Show fenced generation identity, deterministic weighted math, idempotence, stale/corrupt rejection, and atomic committed evidence.
- Show bounded non-Lustre hot-path transport, backpressure/release, and no central full-model broker.
- Exercise the applicable failure/deadline and recovery path; state the minimum progress floor.
- Report exact validation commands and committed-generation/checkpoint artifacts; scale tasks must pass every prior rung.
- A bounded asynchronous **research** task must additionally cite ADR-002 and
  V21S01–V21S17 and ISP01–ISP07, report commit/applied-anchor/result/speculative
  clocks honestly, and provide causally matched per-phase timing for
  freeze/snapshot, admission, publish/network, aggregation, checkpoint, result
  wait, apply/swap, and total foreground idle. It must prove finite
  snapshot/admission and apply/swap pause bounds using every-event maximum and
  p99 evidence. Checkpoint count, restart success, median-only cadence, or an
  aggregate idle fraction cannot satisfy the overlap gate or conceal a
  bursty approximately 200-second foreground stall. A v1 task must state
  `tau = 0` and cannot claim required
  generation-g work overlaps a generation-(g+1) window that starts from
  committed `S_(g+1)`.

## Simple asynchronous v2.1 policy

ADR-002 resolves the bounded asynchronous policy with one concise profile.
Local K40 windows continue from resident worker state while prior
contributions move through the bounded background system. Contributions carry
fenced worker/incarnation/sequence/window identity, base version/digest, exact
tokens, policy/layout/code/payload digests, and distinct lag clocks. Each
stable node worker exposes at most one immutable owned eight-trainer snapshot
and one trainer-owned mutable cumulative adjacent interval. The snapshot is
captured coherently at a K boundary; neither native nor Python background work
may read a concurrently mutating live model. The holder admits at most one
contribution per stable worker per transition and requires the exact two-node
floors `Q_min = 2` and `T_min = 3,934,080`.

Commit, applied-anchor, result-version, and speculative-window lag are separate
integers with maximum two. Lag three drops the stale contribution/result or
defers further snapshot admission until a complete verified result can be
applied at a later boundary; it does not pause foreground training for
point-to-point catch-up. The aggregate is the deterministic binary64
exact-token mean and the stateless outer update is
`S_(g+1) = S_g + mean(delta)` with `eta_outer = 1.0`. There is no staleness
multiplier or second numerical-weight field.

A verified capacity-one latest mailbox is applied only at a K boundary by
translating ScheduleFree `x`, `z`, and the mutable interval start with
`(new_global - old_global) - accepted_local_delta_sum`. All eight node trainers
prepare and recover the same version before the node advertises READY. One
owned cohort, one mutable cohort, finite native credits/replay, bounded mailbox
staging, explicit skip/drop/defer rules, and fixed deadlines bound memory and
time; a third dense cohort is forbidden. `OWNED` transfers immutable snapshot
responsibility to the persistent service, so trainers do not wait for fabric
send, receipt, aggregation, checkpoint, or result completion.

V2.1 preserves the v1 scheduler claim/fence, READY membership, model-free compiled
point-to-point transport, no-all-rank-wait, no-Lustre/Python-dense-hot-path,
no-central-full-model-broker, and atomic checkpoint requirements. It requires
the v2.1 policy/schema/digest and native ABI/protocol boundary; v1 and v2.0
records cannot be renamed. Systems qualification is exactly two nodes until
current-source clean, fault/restart, and newer-fence fresh-recovery machine
verdicts pass and a separate authorization binds the exact source,
policy/schema, native bundle/ABI/wire, launcher, seed, durable collector,
causal telemetry, and V21S17 closure. That authorizes only the 8-node rung.
Convergence/model quality is evaluated separately and is not a systems-scale
prerequisite.

## Native data-plane binding

The elastic dense research path is bound to
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
version 1 (requirements NDP01–NDP17). One pure deterministic transition kernel
in the persistent model-free native service owns the allocation
fence/incarnations, READY membership, generation admission/closure/commit/apply
state, and recovery handshakes. Python remains responsible for scheduler
adaptation, authenticated endpoint exchange, clocks/timers, outer/checkpoint
policy and publication, explicit effect execution, and Slurm supervision. A
persistent model-free C++17 service on every node also owns local XPMEM/memfd
handoff, exact native reduction, libfabric `FI_EP_RDM`/Frontier `cxi` payload
movement, bounded replay, and redistribution. Python TCP and Python object
serialization MUST NOT carry dense contributions or aggregates on a path
claiming this native elastic identity.

The native service is the sole peer-coordination committer, but it cannot
infer membership, closure, expiry, or recovery from transport reachability.
Those external observations re-enter only as fenced typed events; the kernel
returns typed dispositions and explicit effects. Python supplies only locally
complete, checksummed, retained contribution metadata; native owners execute
the resulting immutable frozen set. All fabric operations are bounded
point-to-point operations. The elastic backend MUST NOT initialize MPI or
require an all-rank collective, including during endpoint exchange, failure
handling, redistribution, or shutdown.

The compiled Cray-MPICH helper remains a numerical/performance reference and an
explicit fixed-world fallback. Its launched-rank collectives do not satisfy the
elastic failure-domain requirements. No real model or scale native job is
admissible until the exact-code full-layout two-node synthetic CXI artifact
required by NDP17 has passed and been retained. This adds a gate; it does not
replace the sequential lifecycle/failure/restart ladder above.

## Backend mapping and decision record

For the elastic research path, Frontier/Slurm supplies a fixed allocation
envelope; the supervisor publishes one immutable scheduler-fenced claim,
launches independent node managers/trainers and one persistent native data
service per node, and reacts to Slurm shutdown signals. Native peer control maps shard owners
deterministically among available peers and exchanges opaque service endpoints.
The native services select libfabric `FI_EP_RDM` with exact provider `cxi`.
Slurm node count is capacity only. Other backends map an authenticated
monotonic allocation fence, host agents, and local/network transports to the
same identities and protocol.

**ADR-001 (amended 2026-07-25, research after ADR-003):** The elastic MVP chooses exactly one
scheduler-fenced allocation claim for operational simplicity and safe
continuation across queued jobs. Simultaneous independent allocations do not
join one live run. The native peer protocol, not a shared database, owns live
control. A future federation requires a separately reviewed highly available
control service, cross-allocation authentication, shard-owner placement, and
partition semantics while preserving every fence and commit invariant.

Unresolved **elastic research** decisions are intentionally explicit: v1
`Q_min`, `T_min`, optional fraction and retry deadlines per model size; v1
outer optimizer and checkpoint cadence; shard placement/reassignment; and
whether trainer inner state is ever checkpointed. ADR-002 fixes asynchronous
math only for v2.1 research. None of these gaps blocks ADR-003 production,
whose fixed-world K40/save-200/keep-2 policy and 8 -> 32 -> 128 ladder are
selected above. A path claiming v1/v2.1/native identity still fails closed or
uses test-only values until its own reviewed decisions are resolved.
