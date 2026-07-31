# ADR-002: simple asynchronous DiLoCo v2.1

**Status:** Normative implementation authority for
`async-decoupled-v2.1-simple`, accepted for implementation and amended with the
direct systems-scale policy on 2026-07-29. This document does not itself
authorize a Slurm submission: current-source physical qualification and every
immutable authorization/predecessor gate below remain mandatory.

**Authority:** This ADR specializes
[Resilient DiLoCo Compute Pool](RESILIENT_DILOCO_COMPUTE_POOL.md). That
authority still owns scheduler-fenced claims, READY membership, fencing,
atomic publication, and recovery. [Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
still owns the compiled point-to-point transport invariants. The only
normative definitions and R/NDP crosswalk for the stable v2.1 requirement
namespace are the V21S01–V21S17 rows in the
[gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md). A contradiction fails closed.

Normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their RFC
2119 meanings.

## Decision and compatibility boundary

The selected policy is exactly `async-decoupled-v2.1-simple`. Its canonical
policy document, contribution metadata, commit manifest, native ABI/protocol,
layout, code, and payload each have a versioned schema identity and digest.
The policy digest covers the canonical bytes of every constant and rule in the
two-node profile below. Unknown fields, missing fields, digest disagreement, or
a v2.0 identity fail before model load or accumulator mutation.

The required v2.1 identity boundary is:

| Boundary | Required identity |
|---|---|
| policy | `async-decoupled-v2.1-simple` |
| policy schema | `emender-async-policy-v2.1` |
| contribution metadata | `emender-native-e97-submission-v2.1` |
| committed result/manifest | `emender-native-e97-generation-v2.1` |
| native ABI | `NDP_ABI_V21 = 0x00020001` |
| native wire protocol | major `2`, minor `1` |

An implementation MUST add v2.1 ABI symbols/records rather than extend a v1
struct or v2.0 record under its old version. The v1 `NDP_ABI_V1` and wire
protocol 1.0 remain the strict fresh-only compatibility path. A decoder MUST
reject v2.0 experimental policy, schema, manifest, and digest identities; it
must not migrate or relabel them.

## Fixed two-node profile

| Item | Normative value |
|---|---:|
| local interval | exactly `K = 40` optimizer steps |
| accepted commit lag | integer `0..2`; lag `3` is dropped |
| accepted applied-anchor lag | integer `0..2`; lag `3` defers snapshot admission until a verified boundary apply |
| accepted result-version lag | integer `0..2`; lag `3` result is replaced by current latest |
| speculative-window lag | integer `0..2`; a third speculative snapshot is not admitted |
| stable-worker diversity floor | `Q_min = 2` distinct leased READY node workers |
| accepted-token floor | `T_min = 3,934,080` exact tokens |
| active-membership fraction | disabled |
| generation-attempt retries | `0` |
| local descriptor ownership response | at most `1 s` |
| READY/sync from role start | at most `180 s` |
| one K40 progress window | at most `420 s` from trainer start |
| open group to freeze/abort | at most `420 s` |
| freeze to reload-verified fenced `latest` | at most `420 s` |
| foreground result wait | forbidden; background result age retains the `420 s` deadline |
| snapshot capture/admission pause | at most `1 s` through local `OWNED` |
| result apply/swap pause | at most `60 s` for the complete all-eight transaction |
| allocation to first committed `latest` | at most `720 s` |
| outer update | stateless exact-token average, `eta_outer = 1.0` |

The NDP stage deadlines remain additional inner bounds; a larger enclosing
deadline never resets or extends them. The two-node group freezes only after
both distinct stable workers and the exact-token floor are present. `Q_min` is
an identity-diversity safety predicate, not a numerical weight or clock.
Exact tokens are the sole quantitative quorum measure, accepted-token clock,
and deterministic numerical weight.

## Contributions, clocks, and exact math

Committed model `S_g` is the canonical exported ScheduleFree `x` vector after
`g` outer transitions. A worker is one stable leased node peer with eight
trainer lanes and a random boot incarnation. Each contribution binds:

```text
run_id, allocation_fence, worker_id, worker_incarnation,
contribution_sequence, local_window_start, local_window_end,
base_global_version, base_global_digest,
policy_schema, policy_digest, layout_digest, code_digest,
exact_tokens, local_trainer_set_digest, interval_endpoint_digest,
payload_digest, finite_check, bounded_shard_roots
```

The local trainer-set digest binds all eight trainer incarnations, exact
tokens, interval ranges, and lane payload digests. The complete canonical
envelope digest is the idempotency key. An identical replay returns the
original receipt. Reuse of a logical identity with different bytes is a
conflict and cannot mutate an accumulator. At most one complete contribution
from a stable `worker_id`—across all of its incarnations—may be frozen into one
global transition.

Four lag clocks are independent integers:

```text
commit_lag                  = open_commit_version - contribution_base_version
anchor_lag_before_apply     = newest_verified_version - applied_anchor_version
result_version_lag_at_apply = newest_verified_version - mailbox_result_version
speculative_window_lag      = current_window - last_committed_apply_window
```

Base lag at seal and result-window age remain useful telemetry, but neither
substitutes for these clocks. A contribution with `commit_lag = 3` or more is
dropped. A worker whose anchor, mailbox result, or speculative snapshot would
reach lag 3 stops admitting snapshots from that stale base. It continues
foreground local training on its live mutable state and checks nonblocking at
later K boundaries for a complete verified result; it does not fetch or wait
in the foreground. When one is ready, the worker applies it atomically at a
safe boundary and resumes admissible snapshots at lag zero. Expiry of the
background result deadline skips or defers that result and may drain the
incarnation through the separately labeled recovery path, but cannot create a
foreground catch-up wait. A negative global lag or a mailbox result newer than
authoritative latest is a future-base protocol error and is rejected, never
normalized.

For the frozen set `F_g`, contribution `i` has exact positive integer tokens
`t_i` and cumulative interval delta `d_i`. In NDP deterministic digest order
and binary64 arithmetic:

```text
A_g = sum(i in F_g, t_i * d_i)
T_g = sum(i in F_g, t_i)
D_g = A_g / T_g

S_(g+1) = S_g + D_g
O_(g+1) = {
  "mode": "delta_sgd",
  "eta_outer": 1.0,
  "step": O_g.step + 1,
  "accepted_tokens": O_g.accepted_tokens + T_g
}
```

There is no outer momentum tensor and no distinct aggregation, effective, or
staleness-adjusted weight. The wire record, accumulator, manifest, telemetry,
and checkpoint carry `exact_tokens`; a production v2.1 field representing
`tokens * f(lag)` is forbidden. This restores the retained E97 stateless
plain-average regime selected in
[the ScheduleFree DiLoCo outer report](SF_DILOCO_P4_OUTER_REGIME_REPORT.md)
and the exact-token K40 baseline retained in
[the two-node resilient-pool record](../reports/frontier/validate-resilient-pool-v1-2n.md).
Those records support the fixed starting policy; they are not v2.1
qualification evidence.

## Persistent lane, ownership, correction, and mailbox

A model, ScheduleFree optimizer, iterator, and hidden state are initialized
once per trainer incarnation and remain resident across K40 windows. The
trainer exclusively owns and mutates this live state. A local interval is a
cumulative displacement over a named adjacent range:

```text
d_i[q0,q1) = L_i(q1) - L_i(q0)
exact_tokens_i = sum(actual tokens in every window in [q0,q1))
```

At a K boundary, trainers pause only to capture one coherent, fenced immutable
eight-lane snapshot and admit it into a preallocated double buffer,
copy-on-write representation, or equivalent bounded mechanism. The snapshot
must describe exactly one safe optimizer boundary. Copying directly from live
weights while an optimizer may mutate them is forbidden, as is any background
read of the live model or optimizer.

Per stable node worker, the native service may own exactly one such immutable
sealed snapshot while the trainers extend exactly one mutable eight-lane
cumulative adjacent interval. `OWNED` transfers buffer, transport, replay,
receipt, and release responsibility to the persistent native service. After
`OWNED`, foreground training resumes immediately. Trainers do not wait for
peer discovery, quorum, publication or hashing, fabric send/receipt, reduction,
commit, checkpoint I/O, result readiness, or another trainer. They never
mutate an owned snapshot.

The snapshot and result mailboxes are capacity bounded. A full snapshot slot
causes an explicit skip or defer; a newer verified result may replace an older
unread result under the lag policy. Native background workers publish,
aggregate, validate, and checkpoint only immutable admitted snapshots.
Backpressure applies only within those bounded background queues and must not
silently become a foreground wait or collective rendezvous.

Every committed result enters one capacity-one, verified-latest mailbox per
node. Immutable state and manifest MUST be reload-verified, native peers MUST
agree the exact next result/token identity, and a current-claim digest-linked
commit receipt MUST exist before mailbox publication. `latest.json` is
compatibility telemetry, never authority. A newer result atomically replaces
an unread older result; equal
identical bytes are idempotent; older, conflicting, corrupt, nonfinite, or
wrong-fence results are rejected. One bounded replacement staging view may be
used while a reader holds the visible view; otherwise publication
is skipped or deferred without blocking the trainer. There is no result FIFO.

At a K boundary, trainer `i`, applied anchor `S_a`, and newest verified result
`S_h` use the manifest chain to find every one of that trainer's accepted
interval deltas in `(a,h]`:

```text
accepted_local_i(a,h) = sum(own accepted interval deltas in commits (a,h])
correction_i = (S_h - S_a) - accepted_local_i(a,h)

x_i <- x_i + correction_i
z_i <- z_i + correction_i
mutable_interval_start_i <- mutable_interval_start_i + correction_i
```

Skipped mailbox versions do not skip ledger entries. Gradient moments,
variance state, loss scaling, and scalar steps remain local and unchanged.
Every parameter-valued optimizer point, including ScheduleFree `z`, receives
the same translation. The correction is applied once, only between K windows,
and never to an immutable owned descriptor.

Applying a result is one node transaction across all eight trainers and is the
only steady-state foreground interruption other than snapshot
capture/admission:

1. entirely in the background, the manager verifies the fenced result,
   manifest chain, mailbox version, and eight per-lane correction-ledger plans;
2. if all eight live trainers are at a safe K boundary and the complete
   prepared version can meet the predeclared apply bound, they acknowledge it;
   otherwise the result is deferred without waiting;
3. every trainer applies `x/z` correction and emits a fenced recovery marker;
4. only after all eight markers match does the manager publish one node-applied
   marker and advertise READY at the new version.

The apply/swap is atomic and has its own finite foreground pause bound. A late,
absent, invalid, failed, conflicting, or not-yet-prepared result is skipped or
deferred without blocking training. Missing, conflicting, or timed-out markers
prevent READY and never authorize partial application. A failure after apply
begins but before the node-applied marker causes all eight lanes to discard
disposable local state and restart together from the verified committed result
with new trainer and node incarnations. Recovery is separately labeled and
never accepts a mixture of old- and new-anchor lanes as one node contribution.

## Finite memory, liveness, and membership

All dense state is admitted before model load. For E97 float64 layout
`L = 5,506,770,496`, each trainer cohort contains eight `L/2`-byte f32 vectors.
The conservative native-service bound remains:

```text
two eight-trainer cohorts = 16 * (L/2) = 44,054,163,968
retained owner resident bound             14,440,737,184
one replacement result                     5,506,770,496
total                                     64,001,671,648 bytes
64 GiB cap                                68,719,476,736 bytes
headroom                                   4,717,805,088 bytes
```

The correction accumulator reuses owned-cohort storage. Fixed registered
slots, credits, sender replay, two owner reassignments, receipt ledgers,
mailbox views, and every deadline remain within NDP01–NDP17. A third dense
cohort is forbidden even transiently. Capacity exhaustion skips, replaces, or
defers background work; it does not stop a foreground K window. Failure to
prove the complete non-overlapping formula fails startup; no dense spill to
Lustre or on-demand queue growth is permitted.

Membership is the leased READY snapshot, never launched ranks. Loss or expiry
removes the old incarnation. A returning stable worker discards unfinished
local/inner/descriptor/mailbox state, creates a new incarnation, verifies the
run/fence/policy/code/layout and authoritative latest, and becomes eligible for
a later transition. Old-incarnation work may satisfy only an already frozen
exact identity. With the two-node floor, loss of either stable worker prevents
a commit. The survivor may continue local foreground training but admits no
out-of-policy snapshot and cannot invent one-node authority.

## Native path and atomic checkpoint

The production dense path remains one persistent model-free C++17 service per
node, service-allocated `memfd` or XPMEM handoff, deterministic native
binary64 accumulation, and bounded libfabric `FI_EP_RDM` point-to-point
traffic with exact Frontier provider `cxi`. The native peer-control protocol
owns live fence/incarnation validation, READY membership, group closure,
accepted identities, exact-once commit state, node apply, and recovery
handshakes. Python owns scheduler adaptation, outer/checkpoint policy and
immutable publication, and Slurm supervision. The elastic path MUST NOT initialize MPI,
wait on all launched/READY ranks, use a failure-sensitive collective, send
dense bytes through Python or Lustre, or introduce a central full-model
broker.

Each successful transition publishes one immutable fenced bundle containing
`S_(g+1)`, `O_(g+1)`, prior state digest, frozen identities, exact tokens, all
lag clocks known at commit, drops/rejections, membership, node-apply evidence,
result roots, and policy/layout/code digests. The bundle is reload-verified
before a digest-linked receipt extends the current claim's exact commit
lineage and native peers acknowledge it. A native result,
mailbox insertion, trainer marker, or checkpoint file alone is not a commit.
Fresh-allocation recovery publishes a newer scheduler-fenced claim anchored to
the exact prior receipt, restores the complete model, outer step,
accepted-token clock, result root, fence/incarnation, and apply evidence,
rejects all old-fence volatile state, rejoins through the native peer recovery
handshake, and starts trainers with fresh local inner state. No database or
mutable latest pointer bootstraps recovery.

Cold start is fixed to the retained final E97 authority:

- checkpoint:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`
- step manifest:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json`
- discovery pointer:
  `s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json`
- step `2300930`; accepted tokens `150793748480`; size `7719680116`;
  SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.

Immediately before submission, the submit side revalidates both authorities,
downloads or verifies a content-addressed cold-cache object, computes the exact
size/SHA, and produces a digest-pinned attestation. Inside the allocation,
`sbcast` copies checkpoint and attestation to
`/tmp/emender-e97-seed-${SLURM_JOB_ID}` on every node. One offline verifier per
node rechecks job scope, regular single-link bytes, authority/attestation,
size, SHA, and step and emits `network_fetches=0` before any model-bearing role
starts. Compute nodes MUST NOT fetch the seed or tokenizer from a network or
use the Lustre cold cache as trainer input. The retained
[seed integration](validation/integrate-final-e97-s3-seed-20260722.md) and
[sbcast review](../reports/merge-final-e97-seed-sbcast.md) are the bootstrap
evidence.

## Telemetry and exact two-node gates

Telemetry reports true intervals and identities, not inferred overlap. Every
snapshot/result identity separately times `freeze_snapshot`,
`snapshot_admission`, `publish_network`, `aggregation`, `checkpoint`,
`result_wait`, and `apply_swap`, plus K-window start/end/cadence and total
foreground idle. It also reports all four lag clocks, exact tokens, interval
ranges, resident/credit/replay high-water, skips/drops/deferrals, READY
membership, eight-lane apply markers, and terminal reasons. `result_wait`
measures background result age/availability and reports zero foreground wait.
Snapshot/admission and apply/swap report every event, maximum, p99, and their
separate predeclared bounds. Checkpoint correctness latency (background
snapshot admission through reload-verified fenced `latest`) has a separate
pass/fail result; it is not foreground idle and cannot be hidden from its own
deadline.

Every two-node live systems gate uses exactly `Partition=batch` and
`QOS=debug`. Both fields are retained separately while queued/running and in
terminal accounting. The current-source systems qualification consists of a
clean gate followed by fault/rejoin and newer-fence fresh-recovery phases on
the same reviewed policy, source, native bundle, seed, and configuration.
Focused numerical and deterministic replay checks remain source preflight and
live semantic requirements. Convergence is listed separately because it is a
model-quality study, not a systems-scale prerequisite:

1. **Numerical:** lag 0/1/2 admission and lag-3 drop/defer, unequal exact
   tokens, digest-order binary64 agreement, `eta_outer=1.0`, cumulative
   intervals, skipped mailbox versions, selected/nonselected correction,
   ScheduleFree `x/z`, and atomic eight-trainer apply.
2. **Clean performance:** after two warm-up windows, at least ten consecutive
   K40 windows from all 16 real trainers; local `OWNED <= 1 s`; maximum and
   p99 commit/anchor/result/speculative lag at most 2; median boundary cadence
   at most `1.25` times median raw K40 compute; foreground idle strictly below
   `0.10`; causally matched per-phase timings and maximum/p99
   snapshot/admission and apply/swap within their predeclared bounds; all
   bounds/release invariants; and independent correctness latency. Checkpoint
   count, restart success, median-only cadence, or aggregate idle cannot prove
   overlap. Bursty alternating K windows with approximately 200-second
   foreground pauses fail regardless of healthy medians or checkpoints.
3. **Fault/restart:** missing/late contribution, lag-3 drop, duplicate/conflict,
   checksum/nonfinite input, held mailbox view, trainer/service/owner loss,
   replay/reassignment, partial eight-trainer apply, failed publication, new
   incarnation, and fresh-allocation/new-fence recovery with no partial commit.
4. **Deterministic replay:** two executions of the same frozen contribution
   schedule produce byte-identical result, outer state, token clock, and
   manifest roots after removing timestamps and allocation identities.
5. **Separate convergence/model-quality study:** three predeclared seeds
   compare v2.1 with strict fresh
   stateless `eta_outer=1.0` from the same checkpoint, data order, exact-token
   budget, and K40 count for at least 100 commits. The paired 95% confidence
   interval upper bound for v2.1 minus baseline held-out BPB is at most
   `+0.02`; no seed is worse by more than `+0.05`; neither arm has nonfinite
   loss or an unrecovered greater-than-2x merge shock.

A clean performance pass cannot omit backpressured windows or call a transport
stage `tau=0` merely because it overlaps current compute. Failed jobs 5066495
and 5068873 remain historical v2.0 evidence, including the latter's missing
node-1 apply markers; they do not qualify v2.1.

## Promotion and scale-only closure

Two-node success creates no scale authorization. A separate review must verify
the current-source clean, fault/rejoin, and fresh-recovery machine passes,
including durable scheduler-owned afterany collectors, complete atomic
publication, causal phase/tail telemetry, bounded foreground interruptions,
and absence of forbidden data paths. It explicitly binds the exact
policy/schema, source, native bundle/ABI/wire, launcher, seed, and closure
identities. The convergence study above is separate and cannot substitute for,
authorize, or block this systems decision. Promotion is strictly ordered:

```text
current-source 2-node clean + fault/rejoin + fresh-recovery
  -> authorize 8 -> 32 -> 128 -> explicit 256 review
```

Every rung requires an immutable pass from its immediate predecessor. Failure,
missing evidence, or an unchanged failed payload does not advance the ladder.
Four-, 16-, and 64-node rungs are not part of this policy. The 256 review is a
go/no-go evidence review only and cannot submit or auto-create a runner.

The signed authorization schema is
`emender-async-v21-direct-scale-authorization-v2`; the signed predecessor
machine-pass schema is `emender-async-v21-direct-rung-pass-v2`. Both carry:

- the exact node identity and scheduler tuple (`Nodes=<bound rung>`,
  `Partition=batch`, `QOS=debug`);
- the exact source, v2.1 policy digest, native bundle, final seed, and launcher
  digests plus `emender-async-v21-direct-scale-identity-v1`, which pins the
  policy/schema, contribution/manifest schema, `NDP_ABI_V21 = 0x00020001`,
  wire `2.1`, and final E97 step/token/size/SHA identity;
- ordered `clean`, `fault`, and `fresh-recovery` phase identities and SHA-256
  references for their terminal verdicts, causal telemetry, complete
  publication receipt, and recoverable checkpoint manifest;
- a passed scheduler-owned durable `afterany` collector verdict, complete
  causal telemetry and publication, `snapshot_admission <= 1 s`,
  `apply_swap <= 60 s`, zero foreground result wait, and an empty forbidden
  data-path list;
- affirmative leased-READY finite closure, coherent immutable snapshot,
  immediate trainer resume, compiled-CXI background work, later atomic apply,
  checkpoint recovery, fencing/idempotency, exact-token `eta_outer=1`, and
  changed-payload-only retry facts; and
- `convergence_claim=false`. The authorization additionally binds
  `systems_scale_ladder=[2,8,32,128]`, `review_only_nodes=[256]`, and
  `convergence_required=false`.

Unknown, partial, wrongly scheduled, failed, skipped, wrong-identity, or
digest-inconsistent records fail closed. These records are machine
authorization, not evaluator prose.

The two-node `Q_min=2` close rule is never promotable as a scale early-close
rule. Before any scale rung, the authorization MUST pin a deterministic finite
closure function over the leased READY snapshot `R_g` taken at group open:

```text
C_g = close_time(open_time, R_g, passing_two_node_evidence)
F_g = every complete admissible contribution from R_g received by C_g
```

`C_g` is immutable after open and finite. Its exact calculation must name and
digest the passing two-node arrival and stage distributions, state the chosen
quantiles/margins and arithmetic, derive both close and stage/cadence
deadlines, and pin the exact-token floor as a function of `R_g`. A new
unexplained constant is invalid. The group remains open through `C_g` even
after two nodes arrive and includes every complete admissible pre-close
arrival. At `C_g`, it freezes that complete set only if the authorization's
exact-token and stable-diversity safety predicates hold; otherwise it aborts.
Missing or expired peers do not extend `C_g`.

This is neither a wait for launched ranks nor an all-READY barrier. Launched
rank count never appears in the formula, membership, floor, or evidence.
Without a reviewed, evidence-derived finite formula, every scale render,
preflight, or submission fails closed.

## Historical v2.0 disposition

`async-decoupled-v2.0-exp` and V2A01–V2A18 are retained only in dated
validation reports as historical evidence. V2.0 used hard lag 6, speculative
lag 8, `exact_tokens * (7 - commit_lag)`, a separate aggregation-weight field,
and `eta=0.5`. Those values and schemas are incompatible with v2.1. They are
not aliases, migration defaults, promotion evidence, or fallback behavior.
Historical reports remain unchanged so their failures and measurements stay
honest.

## Required conformance statement

Every implementation, runner, or scale task MUST cite the compute-pool
conformance checklist, all R01–R16, all NDP01–NDP17, all V21S01–V21S17, and
ISP01–ISP07. It must name the applicable failure/deadline path, exact
minimum-progress floor, policy/schema/digests, committed checkpoint evidence,
phase-timing/foreground-idle evidence, and exact validation commands. A scale
task must additionally cite the passed systems authorization and predecessor
manifests
plus the reviewed scale-closure calculation. No task may claim conformance by
satisfying only the v2.1 rows; the mapped R and NDP requirements remain
independently normative.
