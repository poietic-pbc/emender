# ADR-003 research: thousand-GPU resilient DiLoCo topology

**Status:** Research record; **no architecture selection, production change, scale authorization, or Slurm submission**

**Date:** 2026-07-30
**Scope:** 1,024–16,384 GPUs on Frontier-class eight-GPU nodes

## Decision

Do not scale the current full-cohort collective and do not replace it with a
single full-model server. Carry two prototypes only as far as deterministic,
synthetic, and separately authorized small-machine evidence:

1. **P1 — transactional sharded async fragments:** the smallest useful
   availability-first baseline. One eight-GPU island publishes a complete,
   immutable, chunked contribution without waiting for another island. A
   metadata sequencer and replicated shard owners apply it as one idempotent
   transaction. This deliberately tests `Q_min=1`; it is an algorithmic
   experiment, not the current v2.1 policy.
2. **P2 — finite-quorum streaming owner trees:** eight-GPU local-SGD islands,
   balanced streaming fragments, a finite leased-READY close, exact-token
   averaging, and deterministic bounded-degree reduction/redistribution trees
   rooted at durable shard owners. Missing islands are omitted at the close;
   owner failure causes bounded remap/replay or no commit.

P1 falsifies the simplest fully asynchronous algorithm quickly. P2 preserves a
statistically less aggressive cohort mean while removing central fan-in. Run a
Moshpit/random-group arm as a convergence and failure **control**, not as the
durable authority. Do not choose P1 versus P2 until the benchmark plan below
measures convergence, tail latency, recovery, and hotspots.

This recommendation retains the correctness core of the project authorities:
[compute-pool v1](RESILIENT_DILOCO_COMPUTE_POOL.md) (scope and decisions begin
at line 49; conformance checklist at line 285),
[native data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md) (NDP01–NDP17 at
lines 40–67), [ADR-002](ASYNC_DECOUPLED_DILOCO_V2.md), and the
[gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md) (R at lines 18–37, NDP at 39–65,
V21S at 67–93, ISP at 95–115). Any prototype that changes their semantics is a
research backend with a new policy/schema; it cannot be called v1 or v2.1.

## Evidence and limits of the literature

| Primary source | What it supports | What it does **not** establish |
|---|---|---|
| DiLoCo [S1] | AdamW inner/Nesterov outer; one full model delta per worker every `H`; 500x fewer synchronizations at `H=500`; experiments through 64 replicas and 400M parameters; random outer-gradient-drop ablation. | The algorithm waits for all workers. Dropping an update in a simulation is not a fenced, durable, exact-once failure protocol. The paper identifies heterogeneous speed, true asynchrony, >8-worker efficiency, and larger models as limitations. |
| OpenDiLoCo [S2, C1] | 1.1B model, four eight-H100 workers; Hivemind P2P implementation; FP16 pseudo-gradient reduction; geographically distributed run with 67.5 min local compute and about 300 s averaging (90–95% utilization). | Its reference default is `WAIT_FOR_ALL` [C1]. Four workers and a DHT do not demonstrate thousand-island integrity, durable publication, or bounded hotspot behavior. |
| Streaming DiLoCo [S3] | Staggered fragments reduce peak bandwidth; communication may overlap a small `tau`; E3M0/int4 outer gradients; 1B overtraining exchanged 400x fewer bits, with 8x peak reduction; model-quality experiments used two replicas. | It explicitly assumes all workers are present and uses all-reduce. Streaming is a scheduling/compression technique, not a membership, partition, or commit protocol. `tau` is algorithmic staleness and must not be inferred from wall-clock overlap. |
| Decoupled DiLoCo [S4] | Independent learners; balanced fragments; minimum quorum (usually `K=1`), grace window, token/step weighting; `P=H=24`, `tau=2`; sharded CPU syncer; consistent distributed checkpoints and bounded recovery; experiments to 9B and eight learners, with a 12B/8-learner geo run. | It is still a logically central full-model syncer with internal shard all-reduce. “Millions of chips” are simulated event tapes; the paper says physical deployment is order 1K chips. Its weighting/RDA and overwrite semantics differ from this project's reviewed exact-token eta-one v2.1 math. It does not establish Frontier CXI behavior at 128–2,048 islands. |
| Moshpit SGD [S5, C2] | Random groups converge exponentially toward the global average; for `I` peers split into `r` groups the disagreement factor is `((r-1)/(I-1))^T`. Official code bounds group size/time and falls back to local updates after an averaging timeout. | Temporary group all-reduce can still fail locally; repeated mixing is an approximate consensus, not one atomic exact global generation. The code does not provide this project's scheduler fence, complete contribution transaction, or immutable checkpoint lineage. |
| Local/hierarchical/decentralized SGD [S6–S10] | Frequent synchronization inside fast sub-networks and infrequent hub/global mixing fit hardware hierarchy. D-PSGD removes the central-node hotspot; multi-level local SGD makes convergence depend on local/global periods, topology, and heterogeneity. | Most convergence results assume connected mixing over time, bounded delay, and non-adversarial stochastic gradients. They do not supply operational fencing, duplicate suppression, durable commit, or correlated-failure recovery. |
| SWARM Parallelism [S11] | Temporary randomized pipelines and rebalancing contain unreliable pipeline stages; a 1B shared-parameter transformer ran on preemptible T4s below 200 Mb/s. | It partitions the forward/backward pipeline, not DiLoCo outer updates. Its re-routing ideas inform failure placement but do not provide exact-token averaging or a durable global commit. |

**Citation correction.** The task's label “Hierarchical Local SGD
(arXiv:1812.02407)” conflates two sources. arXiv:1812.02407 is the *Elastic
Gossip* M.S. thesis [S6], not a paper titled Hierarchical Local SGD. This review
therefore cites that requested identifier and uses the actual hierarchical
primary sources *Don't Use Large Mini-Batches, Use Local SGD* [S7],
*Hier-AVG* [S8], and *Multi-Level Local SGD* [S9].

## Scale model and equations

Use binary scale labels: `G ∈ {1024,4096,16384}` GPUs and one eight-GPU island
per node, so `I=G/8 ∈ {128,512,2048}` islands. Let:

- `N=688,346,312` E97 elements and `U=N*c` wire bytes for codec `c`;
- the conservative current native binary64 contribution be
  `U=5,506,770,496 B = 5.129 GiB` [NDP canonical layout, lines 345–377];
- `K=40` inner steps per full-model-equivalent outer cycle;
- `P=40` balanced **semantic** fragments, one sent per inner step, each
  `U/P=137,669,262.4 B`; physical CXI frames remain at most 64 MiB and are a
  separate concept;
- `O=64` illustrative shard owners, tree arity `b=8`, Moshpit group `q=8`;
- `s` seconds per inner step.

For a full upload plus result return:

```text
central or direct-sharded fabric payload = 2 I U
central broker ingress = I U; broker egress = I U
owner-o ingress (and egress) = I U / O
balanced b-ary tree payload = 2 (I - 1) U
streaming aggregate rate per direction = I U / (K s)
int4 sensitivity = every U-based figure / 16
```

Streaming changes the burst from `U` every 40 steps to `U/40` every step; it
**does not** change full-cycle bytes. Likewise, sharding changes fan-in and
memory placement but not first-order payload.

### Concrete E97 binary64 estimates

| GPUs | islands/nodes `I` | broker ingress | broker egress | total upload+return | one fragment, aggregate per direction | aggregate per-direction rate (`s=1`) | tree depth `ceil(log_8 I)` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 128 | 0.705 TB | 0.705 TB | 1.410 TB | 17.622 GB | 17.622 GB/s | 3 |
| 4,096 | 512 | 2.819 TB | 2.819 TB | 5.639 TB | 70.487 GB | 70.487 GB/s | 3 |
| 16,384 | 2,048 | 11.278 TB | 11.278 TB | 22.556 TB | 281.947 GB | 281.947 GB/s | 4 |

For another measured step time, divide the rate column by `s`. Every ordinary
island uploads and downloads `U=5.507 GB` per cycle, or `137.669 MB` per
fragment per direction. With a research int4 codec these are `344.173 MB` per
cycle and `8.604 MB` per fragment; aggregate total upload+return becomes
0.088/0.352/1.410 TB. Int4 quality evidence in Streaming and Decoupled DiLoCo
is encouraging but does not replace NDP05 binary64/reference gates.

With `O=64` direct owners, each owner ingests and returns respectively
11.014/44.054/176.217 GB per cycle at the three scales, while retaining
`fan-in=I`; each island has `fan-out=64`. With a striped eight-ary tree, an
internal node sees at most eight children plus one parent for the active
fragment, the root receives at most `8(U/P)=1.101 GB` rather than `I(U/P)`, and
roles should rotate across fragments/failure groups. The price is 3/3/4
latency stages and replay/reroute logic.

### State, fan, cadence, and expected hotspots

| Topology | Fabric egress over a full exchange | per-island fan | densest process | durable/model state | expected hotspot |
|---|---:|---:|---:|---:|---|
| One full-model server | `2IU` | 1 | `I` in and `I` out | server at least base+accumulator+result `≈3U=16.52 GB`, plus outer state | server NIC/CPU/memory grows linearly; global failure domain |
| `O` direct shard owners | `2IU` | `O` | each owner fan-in `I` for its `U/O` stripe | aggregate `≈3U`; per owner `≈3U/O`, plus receipts/replay | per-owner bytes grow `I/O`; skewed shards and owner NICs |
| `b`-ary owner trees | `2(I-1)U` | `≤b+1` per active fragment | `b+1` | streaming accumulator plus replay; roots only own stripes | no linear fan-in; internal/root links and tail across depth |
| degree-4 gossip | about `I U R`, `R≈ceil(log2 I)=7/9/11` | 4 | 4 | model+neighbor/version/residual state | no fixed hotspot; 4.93/25.38/124.06 TB egress for one rough global-mixing epoch; partition divergence |
| Moshpit, `q=8` | `I·2(q-1)U/q·R`, `R=3/3/4` | ring physical degree 2; logical group 7 | group-local | model plus group/round lineage | no global hotspot; about 3.70/14.80/78.95 TB egress; group tail and only approximate global mean |
| P2 hybrid | `≈2(I-1)U`, spread across 40 steps | `≤9` active-tree links | bounded owner-tree role | current v2.1 bound remains **64,001,671,648 B/node**; minimal NDP service is roughly `2U + U/O + registered slots + receipts` | root/internal role imbalance, close tail, checkpoint publisher—not aggregate fan-in |

The current conservative 64.002 GB host bound stays constant per node, but its
pool-wide reserved state is 8.19/32.77/131.08 TB. The compact NDP receipt bound
`I·83·128 B` is only 1.36/5.44/21.76 MB, although actual manifests and replay
metadata add overhead. Memory, not just bytes on the wire, must be measured:
no third dense cohort or unbounded group/update queue is allowed.

## Topology comparison

| Candidate | Semantics and convergence | Failure domain / staleness | Operational assessment |
|---|---|---|---|
| Synchronous hierarchical local-SGD islands | RCCL/FSDP inside eight GPUs; outer mean every `K`. Local averaging can reduce divergence between rare global syncs [S7–S9]. | Any GPU/collective failure loses one island; a synchronous outer collective still loses the whole run. `tau=0` is simplest to reason about. | Keep as numerical/control baseline, not resilient outer topology. |
| Asynchronous inter-island aggregation | Serial or windowed application; token-weighted updates from named bases. Removes peer waits but changes update order, effective batch, and optimizer dynamics. | Island loss is omission. Sequencer/owners become authority failures. Staleness must be bounded, recorded, and tested; arbitrary Hogwild is not acceptable. | P1 is the smallest protocol and fastest falsification target. |
| Streaming DiLoCo | Orthogonal fragments/overlap/quantization smooth bursts; every parameter still updates every `K`. | Published algorithm remains all-worker lock-step [S3]. A late fragment cannot be partially applied. | Adopt balanced fragments and bounded immutable overlap, not its all-reduce membership. |
| Decoupled DiLoCo | `K_min`+grace, dynamic weights and fragment overwrite gave good physical/simulated results [S4]. | Learners do not wait, but the sharded syncer is a logical central failure/hotspot and its math differs from v2.1. | Use as algorithmic comparator. Replace full-model syncer with fenced owners/trees before Frontier scale. |
| Bounded-degree gossip/D-PSGD | Repeated doubly-stochastic mixing avoids central load; convergence depends on spectral gap and connectedness over time [S10]. | Node loss is local; partitions produce independently drifting models. Duplicate or out-of-round mixing biases state unless sequenced. | Excellent hotspot control, difficult authoritative checkpoint and exact recovery. Research control only. |
| Moshpit randomized groups | Expected disagreement contracts about `q^-R` for groups of size `q`; groups allow independent progress [S5]. | A failed member affects its temporary group; timeout/rematch contains it. Partitions stop global information flow without necessarily stopping training. | Practical decentralized control. Needs a separate durable anchor protocol and more bytes/rounds. |
| Sharded parameter server | Exact per-shard math and independent owners; direct fan-out `O`, owner fan-in `I`. | Learner loss is omission. Owner loss blocks its shard until replica/remap; split brain is prevented only by fencing. | Simpler than trees but likely owner-NIC hotspot at 2,048 islands. P1 starts here at small scale. |
| Static reduction tree | Exact mean in `log_b I` stages, bounded degree. | Parent/rack loss can orphan a subtree; static membership is still a barrier. | Insufficient alone. Add deadline omission, retained replay, rotated placement, and durable manifests. |
| **Hybrid durable owner trees (P2)** | Exact-token finite accepted-set mean, streamed fragments, transactional whole-result manifest. | Missing island never blocks after finite close if floors remain. Owner/parent loss remaps/replays; otherwise no commit. | Best hypothesis for bounded hotspots and project conformance; highest control-plane complexity. |

## Failure matrix

“Continue” means healthy islands keep inner steps; it does not imply an outer
commit is always possible.

| Event | Central | direct sharded | owner tree / P2 | gossip | Moshpit |
|---|---|---|---|---|---|
| GPU/node/island loss | continue only if server quorum policy omits it | omit learner; remap if it owned a shard | expire island; prune/replay branch; commit if token/diversity floor remains | neighbors remix; convergence slows | affected group times out/reforms; other groups continue |
| Rack/dragonfly-group loss | many omissions; server placement decisive | correlated owners can make shards unavailable | place owner replicas and parents across failure groups; re-root | graph may disconnect | many groups shrink; randomized rematch only helps if paths remain |
| Slow learner | deadline/grace excludes it | same | finite close never extends; late contribution gets stale disposition | delayed messages create asynchronous-mixing bias | group tail until timeout; rematch |
| Network partition | only fenced authority side publishes | only current-fence owner set publishes | only current claim/receipt lineage publishes; minority trains locally at most | both sides drift—no single authoritative result | both sides mix separately—durable anchoring must stop split-brain publication |
| Duplicate delivery | exact identity returns original receipt | per-shard receipt; transaction commits once | every hop and owner idempotent; no double numerator | must dedupe `(peer,round,msg)` or bias mixing | group/round identity must dedupe |
| Stale update/result | reject or bounded-lag policy | same at every owner | base digest and separate lag clocks checked before mutation/apply | normal but theory needs bounded delay/connectivity | old group/round rejected; state catch-up is separate |
| Aggregator/owner loss | global stop/recovery | affected shards stop, replicate/remap | bounded new owner epoch and sender replay; no partial commit | no central owner | group coordinator can be replaced; global durable anchor still a failure domain |
| Checkpoint outage | do not acknowledge durable commit | same | inner work may continue boundedly; publication/commit stops | local states continue but no authoritative recovery point | same unless separate anchor remains available |

A missing peer is an **availability observation**, never checksum/nonfinite/data
corruption. Corruption means a present envelope or payload violates its bound
identity, layout, checksum, finiteness, or transaction.

## Integrity invariants versus availability policy

### Non-negotiable integrity/correctness

1. **Complete finite contribution:** declared layout, positive exact tokens,
   every chunk present, checksummed, finite, and retained until terminal
   receipt. No partial contribution enters a numerator.
2. **Identity and idempotency:**
   `(run, fence, policy, base, fragment/full-cycle, worker, incarnation, seq)`
   and payload digest name one operation. Identical replay returns the same
   receipt; conflicting reuse is rejected without mutation.
3. **Explicit staleness:** commit, applied anchor, result, and speculative
   clocks remain distinct. A versioned policy says accept/drop/defer; no
   receiver guesses a base or rewrites a stale delta as fresh.
4. **Atomic durable publication:** owners prepare disjoint complete shards; one
   manifest binds accepted identities, exact weights, result roots, prior
   receipt, and policy. The fenced receipt publishes all shards exactly once or
   publishes none. A `latest` pointer is not authority.
5. **Checkpoint recovery:** restart loads an independently verified immutable
   model/outer/token/receipt chain under a strictly newer scheduler fence.
   Volatile queues, memberships, and unfinished inner work may be discarded.
6. **Atomic island apply:** all eight trainers apply one complete verified
   result at a safe boundary or restart together. No partial fragment/model or
   mixed-anchor island advertises READY.

### Explicit availability/quality choices

`Q_min`, token/diversity floor, READY fraction, close/grace deadline, maximum
lag, owner replicas, reassignment count, retry count, checkpoint cadence, and
whether local training continues while publication is unavailable are policy.
They trade goodput and statistical quality; they cannot weaken the six
integrity invariants. P1 chooses `Q_min=1` only to test availability and
convergence. P2 chooses a finite accepted-set window and a measured token floor.
Production values require a reviewed policy and evidence.

### Smallest nonblocking protocol

The smallest protocol in which **loss or delay of an island never waits on that
island** is a fenced transactional sharded apply:

1. an island snapshots a complete update from named base `v`, splits it across
   owners, and continues local work after bounded `OWNED`;
2. owners validate and idempotently `PREPARE` all shards under one transaction;
3. a metadata-only fenced sequencer orders the transaction (bounded staleness)
   and atomically publishes a manifest/receipt; owners expose result `v+1`;
4. islands nonblockingly fetch and atomically apply a verified result at a safe
   boundary.

There is no peer set and `Q_min=1`; therefore no absent peer can block it. The
sequencer is not a dense broker, and replicated owners hold only stripes. This
is only the **minimum liveness construction**—not a claim that single-island
outer updates converge acceptably. Removing the sequencer requires consensus
or accepting multiple divergent histories, so it is not a smaller protocol
with the same single-history integrity.

## Responsibility boundary

| Layer | Responsibilities | Must not do |
|---|---|---|
| GPU trainer + fast C++/RCCL island collective | inner forward/backward/optimizer; synchronous eight-GPU reduction inside the island; safe-boundary immutable snapshot; local exact-token lane reduction; later all-eight atomic apply | background network/checkpoint reads of live state; Python dense serialization; waiting after `OWNED` |
| Persistent model-free C++ node service | fence/incarnation checks; leased READY state; memfd/XPMEM ownership; deterministic f64 reduction; balanced fragment framing; CXI point-to-point credits, checksums, receipts, replay, tree routing, redistribution; bounded buffers/deadlines | model/optimizer policy, Slurm decisions, Lustre hot path, MPI/all-rank operations |
| Python trainer/control adapter | data iterator and inner optimizer orchestration; request safe snapshots/apply; timers and typed events; Slurm supervision; endpoint exchange; policy configuration and telemetry collation | production dense payloads, membership inference from transport reachability, mutating native authority |
| Durable inter-island protocol/publisher | scheduler claim/fence; transaction ordering; owner placement/replication; finite close and floors; immutable manifests/checkpoints/receipt lineage; newer-fence recovery | mutable `latest` as authority, missing-peer-as-corruption, central full-model payload broker |

This is the project boundary required by NDP01–NDP17, not an implementation
suggestion to put collectives in Python.

## Cohort/generation disposition

No current requirement is silently changed. These are changes a future
versioned research policy would have to review.

| Existing concept | Disposition | Rationale |
|---|---|---|
| Scheduler-fenced allocation, worker/incarnation, base/policy/layout/code identity | **Retain** | prevents stale writers and ambiguous recovery. |
| Complete contribution and deterministic frozen accepted set per published transition | **Retain** | exact math and reproducible evidence still need a finite set. |
| Global generation/receipt/checkpoint lineage | **Retain as durable commit version** | one recoverable history is still required even if local windows differ. |
| Every contribution based exactly on the open generation (`tau=0`) | **Relax only in a new policy** | P1/P2 need bounded named base lag; retain separate clocks and hard drop. V1 remains `tau=0`. |
| One monolithic full-model generation barrier | **Relax** to fragment preparation plus one atomic composite manifest | permits streaming and bounded overlap without partial publication. |
| Cohort equals all launched or all READY ranks | **Remove** | leased READY is accounting input; finite close freezes complete arrivals, never launched ranks. |
| Wait for every cohort member | **Remove** | floors/deadline are availability policy; a missing peer is omission. |
| Every island must apply a commit before healthy islands start the next local window | **Remove** | each island tracks its applied anchor; READY gates only that island's next admissible publication. Atomic eight-trainer apply remains. |
| Fixed `Q_min=2`, `T_min=3,934,080` outside exact two-node v2.1 | **Retain for that gate; relax only for research schemas** | V21S05 fixes the two-node profile; V21S17 forbids promoting its early close to scale. |
| Bounded generation/owner attempts, deadlines, replay, and abort-without-publication | **Retain** | availability must remain finite without weakening integrity. |

## Requirement conformance map

Legend: **R** retain; **E** extend/research specialization; **G** gate/no scale
claim. “E” requires a new schema/ADR and is not current conformance.

### R01–R16

| ID | Map |
|---|---|
| R01 | **R:** immutable newer scheduler fence before model load; no shared database. |
| R02 | **R:** stable worker/new incarnation and explicit lifecycle; add tree/owner roles. |
| R03 | **R:** leased READY membership only; no launched-rank invariant. |
| R04 | **E:** preserve fenced identity/idempotency; replace v1 fresh-only admission only under explicit bounded-lag policy. |
| R05 | **R:** exact-token deterministic sharded reference math; P1 adds a separately tested serial-apply rule. |
| R06 | **E:** positive floor and finite close remain; P1 `Q=1` and P2 scale floor are research policy, never inferred. |
| R07 | **R:** native exact-once agreement plus immutable receipt/checkpoint chain. |
| R08 | **R:** deterministic owners, bounded chunks/credits/replay/release, no broker; extend to tree hop receipts. |
| R09 | **R:** model-free manager/service; trainer-only mutable state. |
| R10 | **R:** no Lustre/database/dense Python hot path. |
| R11 | **R:** late join/rejoin with new incarnation and authoritative catch-up. |
| R12 | **R:** globally owned outer state/token clock and fresh-allocation recovery. |
| R13 | **R:** backend-neutral protocol; Frontier/CXI is the first adapter. |
| R14 | **R:** absolute deadlines, immutable evidence, causal foreground/background timing. |
| R15 | **R:** high-precision reference, full-fresh parity, changing-participation accounting, separate convergence study. |
| R16 | **G:** current-source 2-node clean/fault/fresh recovery and exact predecessor ladder remain; this ADR authorizes no rung. |

### NDP01–NDP17

| ID | Map |
|---|---|
| NDP01 | **R:** native authority and hard metadata/C++-dense boundary; add transactional/tree transitions. |
| NDP02 | **R:** bounded point-to-point only; intra-island RCCL is a contained island implementation, never elastic membership. |
| NDP03 | **R:** persistent C++17 `FI_EP_RDM`, exact Frontier `cxi`. |
| NDP04 | **R:** coherent sealed memfd/XPMEM snapshot, no live-state read/copy race. |
| NDP05 | **R:** deterministic binary64 baseline; int4 is a separately decoded/validated research codec with f64 accumulation. |
| NDP06 | **R:** identity on commands, frames, tree hops, transactions, receipts, results, checkpoints. |
| NDP07 | **R:** leased endpoint metadata and current-fence routes; tree plans are digested metadata. |
| NDP08 | **R:** pre-registered finite pools; no queue growth or foreground wait. |
| NDP09 | **R:** receiver credits distinct from completion/receipt; hop backpressure stays background. |
| NDP10 | **R:** CRC/SHA, finite checks, once-only application, idempotent receipts. |
| NDP11 | **R:** sender replay, at most bounded owner epochs; P2 adds bounded parent reroute. |
| NDP12 | **R:** owner-direct complete result into one node aggregate; no eight copies. |
| NDP13 | **R:** absolute deadlines and local fault containment; no allocation abort or foreground wait. |
| NDP14 | **R:** versioned ABI and metadata-only local seqpacket; new policy gets new ABI/wire identity. |
| NDP15 | **R:** fenced immutable background checkpoint and later atomic bounded all-eight apply. |
| NDP16 | **R:** add tree depth/role/link skew, close composition, and per-fragment causal telemetry to all existing fields. |
| NDP17 | **G:** full-layout two-node CXI and current-source gates precede any real/scale use. |

### V21S01–V21S17

| ID | Map |
|---|---|
| V21S01 | **E:** never relabel P1/P2 v2.1; pin new policy/contribution/manifest/ABI/wire digests. |
| V21S02 | **R/E:** retain four clocks and lag-3 nonblocking disposition; P1 serial ordering needs its own proven clock rule. |
| V21S03 | **R:** exact tokens remain sole quorum/clock/numerical weight; Decoupled paper's quantity×quality weight is only a comparator. |
| V21S04 | **R:** K40 and stateless exact-token eta-one are P2 baseline; P1 eta is a convergence sweep, not v2.1. |
| V21S05 | **G/E:** exact two-node floors/deadlines remain; scale/P1 values require new digested policy. |
| V21S06 | **R:** exclusive trainer state, coherent immutable boundary, immediate resume. |
| V21S07 | **R:** complete verified once-only ScheduleFree correction at a safe boundary, no result wait. |
| V21S08 | **R:** current-fence verified capacity-one mailbox and nonblocking replacement/defer. |
| V21S09 | **R:** finite resident/slot/credit/replay/receipt bounds; no third cohort/spill. |
| V21S10 | **R:** leased membership/new incarnation; missing island cannot become corruption or foreground wait. |
| V21S11 | **R:** one all-eight node apply/recovery transaction and one marker. |
| V21S12 | **R:** compiled model-free CXI point-to-point service; no MPI/Python/Lustre/broker. |
| V21S13 | **R:** honest causal phases, lag/high-water facts, max/p99, zero foreground result wait. |
| V21S14 | **R:** complete immutable bundle, newer-fence recovery, exact cold-start identity where E97 is used. |
| V21S15 | **G:** exact two-node numerical/clean/fault/fresh-recovery evidence on `Partition=batch`, `QOS=debug`; none produced here. |
| V21S16 | **G:** promotion sequence and immutable predecessor evidence unchanged; no scale authorization. |
| V21S17 | **R/G:** P2 finite leased-READY close must be derived from passing evidence, include every pre-close admissible arrival, and never close at two merely because two arrived. |

### ISP01–ISP07

| ID | Map |
|---|---|
| ISP01 | **R:** race-test coherent immutable capture and prohibit background live-state access. |
| ISP02 | **R:** `OWNED≤1 s` two-node bound; block each background phase while the next K window advances. |
| ISP03 | **R:** owners/hash/checkpoint consume immutable admitted state only. |
| ISP04 | **R:** exhaust snapshot/mailbox/credit/tree-replay/receipt capacity and observe explicit skip/drop/defer, no foreground wait. |
| ISP05 | **R:** complete atomic apply `≤60 s` in exact two-node profile; inject every pre/during-lane failure. |
| ISP06 | **R:** causal `freeze_snapshot`, `snapshot_admission`, `publish_network`, `aggregation`, `checkpoint`, `result_wait`, `apply_swap`, total idle, every-event max/p99. |
| ISP07 | **R:** raw tails; reject an approximately 200 s alternation regardless of healthy checkpoint/median summaries. |

## Falsifiable benchmark plan

No step below submits a job as part of this ADR.

### Stage 0 — deterministic workstation/simulation

- Reference `P1` serial stale apply and `P2` exact-token finite-set mean against
  binary64 single-process math; all-fresh P2 must equal synchronous eta-one.
- Event simulation at 128/512/2,048 islands: random and correlated island loss,
  slow-tail distributions, partitions, duplicates, stale/future frames, owner
  and internal-tree loss. Assert one result or none, bounded memory/replay, and
  no wait edge to a missing island.
- Compare direct owners, arity 4/8/16 trees, degree-4 gossip, and q=8 Moshpit.
  **Accept topology model:** maximum logical fan ≤9 for P2, max/median node
  payload ≤1.5 after rotating fragment roles, no unbounded queue, and analytic
  bytes within 1% of the equations above.
- Convergence proxy on a small transformer, identical tokens/data order:
  strict fresh eta-one, P1 lag/eta sweep, P2 quorum/close sweep, Moshpit control.
  Reject nonfinite runs, unrecovered >2x merge shock, or persistent validation
  loss/BPB regression; predeclare three seeds before selecting thresholds.

### Stage 1 — local native microbench (no Slurm)

- Full E97 layout, eight fake lanes per node, 40 balanced fragments and 83-or-
  more ≤64 MiB transport chunks; duplicate/corrupt/nonfinite/future/stale input,
  capacity exhaustion, owner loss after receipt, parent loss, two remaps,
  checkpoint failure and restart.
- **Accept:** bitwise baseline roots, identical replay receipts, zero partial
  commits, zero Python/Lustre dense bytes, admitted=released bytes, no third
  cohort, and bounded RSS within the declared formula.
- Calibrate every link with a point-to-point sustainable-bandwidth probe.
  **Network acceptance:** p99 offered load on any endpoint ≤70% of its measured
  sustainable payload bandwidth, ≥30% headroom, no CQ overflow, and payload
  imbalance ≤1.5. Treat saturation, not average bandwidth, as failure.

### Stage 2 — exact two-node Frontier qualification, only if separately authorized

Use the authority's G2/G3–G5 shape and explicitly retain both
`Partition=batch` and `QOS=debug`. Run P1 and P2 separately with the same source,
seed, layout and token schedule.

- **Throughput:** at least NDP G2's 222.6 MB/s logical full-layout baseline;
  full transfer+redistribution median ≤98.961 s and no sample >118.754 s, until
  a reviewed changed-topology threshold supersedes it.
- **Overlap/tail:** `OWNED≤1 s`; complete all-eight `apply_swap≤60 s`;
  foreground result wait exactly zero; after warm-up, ten K40 windows, median
  cadence ≤1.25x raw K40, foreground idle <0.10; every-event max/p99 pass and no
  approximately 200 s gaps.
- **Recovery:** kill one trainer, island service, owner, and tree parent; hold a
  mailbox; duplicate/conflict/corrupt one chunk; fail publication; recover under
  a newer scheduler fence. Healthy island K windows continue, and each attempt
  yields one durable result or none.
- **Convergence:** separate three-seed ≥100-commit study. As the current ADR-002
  gate specifies, paired 95% CI upper bound for async minus strict fresh BPB
  ≤+0.02 and no seed worse than +0.05. P1 is rejected if it misses this; it does
  not inherit P2's result.

### Stage 3 — small systems experiment, only after exact-source authorization

The existing policy permits no ad-hoc 4/16/64-node rung. After complete
current-source two-node clean/fault/fresh-recovery evidence and reviewed
V21S16–V21S17 closure, the next possible experiment is the separately
authorized **8-node** rung, followed only by its authority's predecessor chain.
At eight nodes inject one whole-island loss and one correlated owner/parent
loss, require the same tail/network/recovery criteria, and compare direct-owner
versus tree hotspot measurements. Stop if link p99 exceeds 70%, max/median bytes
exceed 1.5, recovery exceeds its original absolute deadline, or model-quality
proxy regresses.

The 1K/4K/16K rows in this ADR are equations and simulator targets only. They
are **not** a proposed live Frontier ladder.

## Validation record and compute-pool checklist

This research maps every **R01–R16**, **NDP01–NDP17**, **V21S01–V21S17**, and
**ISP01–ISP07** requirement above and applies the compute-pool conformance
checklist at `RESILIENT_DILOCO_COMPUTE_POOL.md:285`:

- leased READY membership, finite waits, and no launched-rank invariant are
  explicit in P2; P1 intentionally has no peer wait;
- the compute closure remains database/lock/SQLite-free, and Lustre carries no
  membership, heartbeat, update, aggregate, or redistribution hot-path bytes;
- fenced identity, exact-token deterministic math, idempotence, stale/corrupt
  rejection, complete accepted sets, and atomic durable evidence are retained;
- dense transport is bounded native point-to-point CXI with prompt release and
  no central full-model broker; Python carries metadata only;
- node/island, owner/parent, partition, duplicate, stale, corruption,
  publication, checkpoint, and newer-fence recovery paths have explicit
  dispositions and falsifiable tests; P1's research floor is one, while P2's
  floor must be evidence-derived;
- immutable snapshot ownership, immediate foreground resume, bounded atomic
  apply, causal per-phase telemetry, max/p99 tails, and the 200-second
  adversarial-stall rejection retain ISP01–ISP07; and
- R16/NDP17/V21S15–V21S17 remain hard gates. No command here is an artifact,
  committed-generation claim, current-source pass, predecessor pass, or scale
  authorization.

Research validation commands are `git diff --check`, an ID-coverage scan for
all four namespaces, local-link existence checks, and independent recomputation
of every scale row from `N=688346312`, `c=8`, `K=P=40`. No Python test, native
build, or Slurm command is required because this task changes documentation
only.

## Open decisions after prototypes

- Whether P1 single-contribution application has acceptable sample efficiency
  at any lag/eta; reject rather than paper over a negative result.
- Evidence-derived P2 close/token/diversity floor and whether closure bias
  correlates with island speed or data shard.
- Owner replication/placement across Frontier failure groups and whether tree
  latency tail beats `O=64` direct owners.
- Binary64 versus an explicitly versioned compressed wire representation; never
  infer int4 integrity from ML parity alone.
- Checkpoint frequency and whether a metadata sequencer can be made highly
  available without becoming a dense broker or split-brain authority.
- Whether decentralized mixing can periodically anchor without destroying its
  hotspot advantage or violating single-history recovery.

## Primary references

- **[S1]** Douillard et al., “DiLoCo,” arXiv:2311.08105,
  <https://arxiv.org/html/2311.08105v3>.
- **[S2]** Jaghouar et al., “OpenDiLoCo,” arXiv:2407.07852,
  <https://arxiv.org/html/2407.07852v1>.
- **[S3]** Douillard et al., “Streaming DiLoCo with overlapping
  communication,” arXiv:2501.18512,
  <https://arxiv.org/html/2501.18512v1>.
- **[S4]** Douillard et al., “Decoupled DiLoCo for Resilient Distributed
  Pre-training,” arXiv:2604.21428,
  <https://arxiv.org/html/2604.21428v1>.
- **[S5]** Ryabinin et al., “Moshpit SGD,” NeurIPS 2021,
  <https://proceedings.neurips.cc/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html>.
- **[S6]** Pramod, “Elastic Gossip,” arXiv:1812.02407,
  <https://arxiv.org/abs/1812.02407>.
- **[S7]** Lin et al., “Don't Use Large Mini-Batches, Use Local SGD,”
  arXiv:1808.07217, <https://arxiv.org/abs/1808.07217>.
- **[S8]** Zhou and Cong, “A Distributed Hierarchical SGD Algorithm with
  Sparse Global Reduction,” arXiv:1903.05133,
  <https://arxiv.org/abs/1903.05133>.
- **[S9]** Castiglia et al., “Multi-Level Local SGD for Heterogeneous
  Hierarchical Networks,” arXiv:2007.13819,
  <https://arxiv.org/abs/2007.13819>.
- **[S10]** Lian et al., “Can Decentralized Algorithms Outperform Centralized
  Algorithms?,” arXiv:1705.09056,
  <https://arxiv.org/abs/1705.09056>.
- **[S11]** Ryabinin et al., “SWARM Parallelism,” arXiv:2301.11913,
  <https://arxiv.org/abs/2301.11913>.
- **[C1]** OpenDiLoCo official code: default `WAIT_FOR_ALL` and timeout/matchmaking
  paths, <https://github.com/PrimeIntellect-ai/OpenDiLoCo/blob/main/open_diloco/hivemind_diloco.py#L340-L360>.
- **[C2]** Moshpit official code: randomized/Moshpit averagers and bounded
  collaboration timeouts,
  <https://github.com/yandex-research/moshpit-sgd/blob/main/averaging_experiments/averager.py> and
  <https://github.com/yandex-research/moshpit-sgd/blob/main/language_modeling/collaboration.py#L20-L27>.
