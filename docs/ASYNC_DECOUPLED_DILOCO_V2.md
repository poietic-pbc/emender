# ADR-002: bounded-lag asynchronous DiLoCo v2

**Status:** Normative initial v2 experimental policy, accepted for
implementation and two-node qualification only (2026-07-24). Its lag bound,
weights, and half-step are hypotheses with exact reference behavior, not
established optima. It does not authorize a 4+ node or production run.

**Authority:** This ADR is the bounded-lag policy extension of
[Resilient DiLoCo Compute Pool](RESILIENT_DILOCO_COMPUTE_POOL.md). The compute
pool remains authoritative for leases, membership, fencing, atomic publication,
and recovery. [Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
remains authoritative for the model-free compiled point-to-point data-plane
properties identified below. The [gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md)
assigns this policy requirement IDs V2A01–V2A18. A contradiction fails closed.

Normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their
RFC 2119 meanings.

## Decision

The initial asynchronous policy is `async-decoupled-v2.0-exp`: continuous
local `K = 40` training windows, experimental hard global-version lag
`tau_hard = 6`, production-promotion target `tau_target = 2`, hard speculative
local-window lag `sigma_hard = 8`, and promotion target `sigma_target = 2`.
It admits at most one contribution per worker in each global commit, uses
exact-token quorum accounting, linear integer staleness weighting, and a
stateless half-step outer update. The weights and half-step are deliberately
fixed so they can be falsified by the convergence gate.

One immutable cumulative contribution represents one or more adjacent K40
windows. While the persistent native service owns a prior descriptor, the
trainer coalesces later K-windows into one bounded next interval rather than
enqueueing one dense object per K. Workers receive committed results through a
verified latest-only mailbox and translate the local ScheduleFree model
iterate `x` and base iterate `z` only between intervals/K-windows. Local inner
work is disposable; the committed global model, outer step, accepted-token
clock, and their manifest are authoritative.

This is not a promise that stale DiLoCo is equivalent to synchronous DiLoCo.
It is one explicit, testable policy whose convergence must pass the gates
below before promotion.

Version 1 `tau = 0` remains a strict compatibility/correctness mode. It keeps
the rule that the next window begins from the just-committed state and therefore
does not claim full pipeline overlap. It is not the production performance path.
No run or telemetry may describe lagged v2 work as `tau = 0`.

## Why the former contract is impossible

The old combination required both:

1. under v1, every worker's local window `g+1` starts from committed
   `S_(g+1)`, where committing `S_(g+1)` requires generation-g reduction,
   outer apply, verified redistribution, and atomic publication; and
2. under the former performance validator, that same required generation-g
   background work overlaps the local K-window `g+1`.

Let `t` be the start of K-window `g+1`, and let `B_g` be any required
generation-g reduction/apply/redistribution interval. Rule 1 implies
`end(B_g) <= t`. Positive overlap with the K interval `[t, t + K)` requires
`end(B_g) > t` and `start(B_g) < t + K`. The two end-time inequalities are
inconsistent. A post-commit log write may overlap, but relabeling it as the
required reduce/outer-apply/redistribution does not satisfy the contract.
Threads do not remove this causal edge.

Slurm job 5062348 is the retained empirical counterexample, not the proof's
premise. It completed five restartable K40 generations and atomically
published generation 5, but its 80 trainer inter-window observations measured:

| Metric | Retained value |
|---|---:|
| median raw K40 compute | 63.679326 s |
| median K-boundary cadence | 358.265159 s |
| median foreground idle | 294.940987 s |
| cadence / raw K40 | 5.626083x |
| aggregate foreground idle fraction | 0.817668 |
| native local reduction | approximately 17.49–17.98 s |
| owner redistribution | 22.04–22.66 s |
| trainer apply, generations 0–3 | generally 11.08–14.68 s |
| generation-g background overlapping generation-(g+1) K40 | none |

The validator correctly rejected `generation 0 background did not overlap 1
K40 compute`. The measurements are retained in Git commit
`20c9d1bec436b6aa6a2eba4e434d2202e9c45762`,
`reports/validate-pipelined-native-2-final-seed.md`. The result proves that the
existing implementation obeyed the serial causal edge at large cost; it does
not support a claim that `tau = 0` can be made fully overlapped.

Job 5062348 does **not** justify `tau_hard = 6`, `eta = 0.5`, or the linear
weights. At one dense descriptor emitted per K and one commit per 5.626083 K,
arrival rate exceeds service rate and any FIFO grows without bound. Increasing
`tau` only postpones that failure. V2 instead bounds the descriptor arrival
rate by coalescing adjacent local windows and admits at most one immutable
descriptor plus one mutable next interval per worker. The retained artifact
does not isolate a quorum service time, and in particular contains no evidence
that quorum takes six K-windows. Its unitemized remainder includes serialized
control, handoff, checkpoint, integrity, and publication work.

The implementation must first decouple and independently time the known
approximately 18-second local reduction, 22-second owner redistribution,
11–15-second trainer apply, and checkpoint/publication/control stages. Only
then may the two-node gates interpret the measured base, commit, applied-anchor,
result, and local-window lag. Six is a predeclared, falsifiable experimental
global-version ceiling for delayed/rejoining contributions, not a latency
model. Two is the lower promotion target. Eight is a separate hard bound on
speculative local windows; two is its promotion target. The convergence and
performance gates, not the 5062348 serial total or any stage estimate, decide
whether these values may be promoted.

## Versions, windows, and lag

`S_g` is the atomically committed canonical global model/exported-`x` vector
after `g` outer updates; global scalar clocks live in `O_g` and the manifest.
ScheduleFree `z` and its other inner optimizer buffers are worker-local and are
not contributed. A worker incarnation has monotonically increasing
local-window number `q`, contribution sequence `s`, applied global anchor `a`,
and local model state `L_(i,q)`.

In the initial E97 profile, a global worker is one leased node peer. Its eight
trainer lanes share the contribution's `[q0,q1)` range. The local-set digest
binds each lane's incarnation, endpoint, exact tokens, and interval delta; the
native service reduces them to the one node descriptor. The correction ledger
later retains each selected lane's own delta even though global admission is
per node.

A contribution interval `r` begins at boundary `q0` with model snapshot
`P_(i,r) = L_(i,q0)`. The trainer runs adjacent exact K40 windows without a
global wait or mid-window mutation. At boundary `q1 > q0`, when its one native
descriptor slot is available, it seals:

```text
E_(i,r) = L_(i,q1)
d_(i,r) = E_(i,r) - P_(i,r)
window_count_(i,r) = q1 - q0
exact_tokens_(i,r) = sum of actual tokens in windows [q0, q1)
```

Intervals from one incarnation are nonoverlapping and adjacent except where an
explicit stale/corrupt/drop outcome closes an interval. This is **cumulative
coalescing over a named local-window range**, not a per-K delta, total
displacement from global `S_b`, or mutable update after enqueue. It represents
every local displacement in that range exactly once. If a safe-boundary global
rebase occurs before the interval can seal, the worker translates both its
current model and `P_(i,r)` by the same correction; `d_(i,r)` remains
unchanged and the interval retains its original base version.

The seal operation is a bounded local ownership transfer. The trainer fills a
service-allocated buffer, verifies finite data, seals the immutable descriptor,
and receives either `OWNED` or a local admission rejection. After `OWNED`, the
persistent native service exclusively owns transport, receipt, replay, and
release. The trainer MUST NOT wait for a fabric send completion, remote
receipt, reduction, or commit before starting another K-window. It blocks only
at the safe-boundary application rule or the explicit lag/capacity bounds
below.

The following lag values are distinct and MUST be emitted as integers:

- `base_lag_at_seal = G_seal - b`, where `b` is the original global anchor at
  interval boundary `q0` and `G_seal` is the newest committed version the
  worker has verified when sealing;
- `commit_lag = g - b`, recomputed by the current holder when selecting the
  contribution for transition `S_g -> S_(g+1)`;
- `anchor_lag_before_apply = G_apply - a`, where `a` is the worker's currently
  applied global anchor and `G_apply` is the newest committed version the
  worker has verified at that safe boundary;
- `result_version_lag_at_apply = G_apply - h`, where `S_h` is the result taken
  from the mailbox and `G_apply` is the newest verified committed version then
  known; and
- `result_window_age = q_apply - q1`, recorded by an originating worker
  when it first applies a committed anchor whose manifest chain contains that
  contribution. A worker that did not originate a selected contribution emits
  an explicit null, not an invented age.

The worker additionally emits
`speculative_window_lag = q_now - q_last_committed_apply`. A contribution may
have `commit_lag = 0` while its result arrives six local windows later; these
facts MUST NOT share a field or be inferred from one another. Admission has the
experimental hard global bound `0 <= commit_lag <= tau_hard = 6`. Local
apply/catch-up has the same global-version hard bound
`0 <= anchor_lag_before_apply <= tau_hard`, and training has the separate
`speculative_window_lag <= sigma_hard = 8` bound. Clean promotion requires
both p99 and maximum accepted `commit_lag <= 2`, both p99 and maximum
`anchor_lag_before_apply <= 2`, and both p99 and maximum
`speculative_window_lag <= 2`; the hard ceilings are failure-containment
limits, not production targets. `result_version_lag_at_apply` is independently
required to expose whether a mailbox/catch-up result was already old when
applied; it is never substituted for anchor lag.

Negative global lag is a future-base protocol error. A global lag greater than
six is a stale-bound drop, never a weight of zero and never a
compatibility-mode update.

## Contribution identity and admission

The immutable contribution envelope and its digest include, in canonical
encoding:

```text
(
  run_id, allocation_fence,
  worker_id, worker_incarnation,
  contribution_sequence, local_window_start, local_window_end, window_count,
  base_global_version, base_global_digest,
  policy_digest, layout_digest, code_digest,
  exact_tokens, base_lag_at_seal,
  payload_digest
)
```

The envelope also binds every K-window start/end monotonic timestamp in the
range, the interval endpoint digest, local trainer-set digest, source dtype,
finite-check result, and bounded shard roots. `exact_tokens` is a positive
integer sum over actual windows and advances the accepted-token clock; it is
never replaced by a rank count, nominal step count, or `window_count`. The
holder recomputes all digests and global lag from its own fenced state.

For each transition from current committed version `g`, the holder opens one
bounded group. The initial two-node qualification profile is:

```text
K = 40
tau_hard = 6; tau_target = 2
sigma_hard = 8 local windows; sigma_target = 2
Q_min = 2 distinct READY node peers
T_min = 3,934,080 exact tokens
active-membership fraction = disabled
group deadline = 420 seconds
generation-attempt retries = 0
```

The group snapshots leased READY membership without creating an all-rank wait.
It admits only the current run/fence and exact policy/layout/code identities.
Each worker exposes at most one sealed descriptor; the holder selects it only
when its base is in `[g-6, g]`. At most one contribution from that stable
worker enters the group. This ownership bound prevents a fast worker from
replacing peer diversity with multiple updates. The frozen set is sorted by
the complete contribution digest.

The holder freezes in one atomic control transaction when the selected set
first satisfies both `Q_min` and `T_min`, or evaluates the selected set at the
deadline. A deadline group commits only if both floors are satisfied.
Otherwise it aborts with no `S_(g+1)`. Partial contributions never enter the
set. A contribution not selected remains the worker's single sealed descriptor
for a later transition while admissible; no second descriptor queues behind it.

For any time interval `[u,v]`, let `A_i(u,v)` be immutable descriptors locally
admitted for worker `i` and `R_i(u,v)` be its descriptor releases. The
single-slot transfer rule gives:

```text
A_i(u,v) <= R_i(u,v) + 1
sealed_backlog_i <= 1
mutable_coalescing_intervals_i <= 1
```

K-window arrival rate may exceed global service rate indefinitely without
growing dense FIFO state: additional windows extend the one mutable interval.
If that interval reaches eight windows while its sealed slot is still busy,
the trainer pauses at the boundary. Thus the system is bounded without
pretending job 5062348's 5.626083x service time is healthy.

## Exact aggregation and outer update

For selected contribution `i`, define:

```text
ell_i = g - base_global_version_i
token_weight_i = exact_tokens_i
aggregation_weight_i = exact_tokens_i * (7 - ell_i)
```

Thus a fresh token has integer weight 7 and a maximum-lag token has weight 1.
The exact token floor and accepted-token clock use `token_weight_i`; only the
mean uses `aggregation_weight_i`. Both values are checked positive and for
unsigned-integer overflow. A v2 wire record carries them separately.

The linear factor is the initial experimental choice because it is positive at
the hard bound, monotonically attenuates stale work, and remains an exact
integer compatible with deterministic native accumulation. It is not claimed
to be statistically optimal. The half-step `eta = 0.5` is an experimental
shock bound: the outer displacement is half the weighted-mean displacement
rather than a full stale average. Neither choice compensates for slow service;
the `tau_target/sigma_target` performance gate still fails a 5.626083-window
background cadence. Promotion requires the paired convergence result below.

For every scalar or vector component, in the NDP deterministic contribution
order and binary64 arithmetic:

```text
A_g = sum_i aggregation_weight_i * d_i
W_g = sum_i aggregation_weight_i
D_g = A_g / W_g

eta = 0.5
S_(g+1) = S_g + eta * D_g
O_(g+1) = {
  "mode": "delta_sgd",
  "eta": 0.5,
  "step": O_g.step + 1,
  "accepted_tokens": O_g.accepted_tokens + sum_i exact_tokens_i
}
```

There is no momentum tensor in `async-decoupled-v2.0-exp`. `O_g` is
nevertheless required global outer state: mode, exact hyperparameters, outer
step, and token clock must be atomically checkpointed and restored. Changing
`eta`, the staleness function, either lag bound/target, coalescing semantics, or
optimizer mode changes `policy_digest` and requires a new reviewed policy
version.

This executable reference fixes scalar and vector behavior:

```python
import numpy as np

TAU = 6
ETA = 0.5

def contribution(start, end, exact_tokens, window_start, window_end):
    assert isinstance(exact_tokens, int) and exact_tokens > 0
    assert isinstance(window_start, int) and window_end > window_start
    assert window_end - window_start <= 8
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    assert np.isfinite(delta).all()
    return delta

def aggregate(current_version, records):
    # records contain base_version, exact_tokens, digest, and delta
    assert records
    admitted = sorted(records, key=lambda r: bytes.fromhex(r["digest"]))
    numerator = np.zeros_like(np.asarray(admitted[0]["delta"], dtype=np.float64))
    denominator = 0
    exact_tokens = 0
    for record in admitted:
        lag = current_version - int(record["base_version"])
        assert 0 <= lag <= TAU
        tokens = int(record["exact_tokens"])
        assert tokens > 0
        weight = tokens * (TAU + 1 - lag)
        numerator = np.add(
            numerator,
            np.multiply(np.asarray(record["delta"], dtype=np.float64), weight),
        )
        denominator += weight
        exact_tokens += tokens
    assert denominator > 0 and np.isfinite(numerator).all()
    return np.divide(numerator, denominator), exact_tokens

def outer_apply(state_g, outer_g, mean_delta, accepted_tokens):
    assert outer_g["mode"] == "delta_sgd" and outer_g["eta"] == ETA
    state_next = np.add(
        np.asarray(state_g, dtype=np.float64),
        np.multiply(np.asarray(mean_delta, dtype=np.float64), ETA),
    )
    outer_next = {
        "mode": "delta_sgd",
        "eta": ETA,
        "step": int(outer_g["step"]) + 1,
        "accepted_tokens": int(outer_g["accepted_tokens"]) + int(accepted_tokens),
    }
    return state_next, outer_next
```

The native implementation additionally follows NDP05's fixed round-to-nearest,
ties-to-even conversion, accumulation order, overflow checks, and result
encoding. The reference above is an executable semantic oracle, not permission
to use a different reduction order.

## Latest-only mailbox and safe-boundary rebase

Each worker has a capacity-one verified result mailbox. A result contains the
complete committed `(run, fence, version h, result/base/policy/layout/code
digests, exact accepted tokens, S_h, O_h)`. It is eligible only after immutable
state and manifest have been reload-verified and authoritative `latest` has
advanced under the current fence.

Ingress validates every identity and digest before publishing the slot. A
newer verified version atomically replaces and releases an older unread slot.
An equal version with identical bytes is idempotently acknowledged. An equal
version with different bytes, an older version, a future fence, corruption, or
nonfinite state is rejected and quarantined. Replacement is forbidden while a
worker holds the view; the service uses exactly one bounded replacement staging
buffer and otherwise backpressures the result producer until release.
“Latest-only” therefore bounds visible results to one and staging results to
one; it does not authorize use-after-free.

At the boundary before the next forward pass, a worker with applied anchor
`S_a` takes the newest verified `S_h`, `h > a`. Let `C_i(a,h)` be the
deterministic sum of this trainer's own interval deltas whose identities were
accepted in commits `(a,h]`. The local accepted-set records in those manifests
define the sum. Skipping mailbox versions does not skip their identities.
The service retains or incrementally folds those immutable local deltas into
one bounded correction buffer until the worker acknowledges apply.

The exact conceptual rebase is:

```text
correction_i = (S_h - S_a) - C_i(a,h)
L_i <- L_i + correction_i
anchor_i <- (h, S_h, digest(S_h))
```

For a nonaccepted worker, `C_i(a,h) = 0`, so its explicit anchor is the old
applied global `S_a` and all speculative local displacement is preserved. For
an accepted worker, subtracting its accepted interval displacement before
adding the global result replaces that personal work with the weighted global
update instead of counting it twice.

The tempting endpoint rule `L_i <- L_i + S_h - E_i` is valid only when there
is exactly one accepted interval, that interval started at `S_a`, and no
unaccepted local displacement preceded its endpoint. Under coalescing,
nonselection, or skipped mailbox versions it can erase valid unaccepted work.
The correction-ledger equation is the selected multi-version rule. It reduces
to the endpoint equation in the restricted case above.

For ScheduleFree, the same `correction_i` is a coordinate translation applied
to both the local model iterate and every parameter-valued inner buffer:

```text
x_local <- x_local + correction_i
z_local <- z_local + correction_i
coalescing_start_snapshot <- coalescing_start_snapshot + correction_i
```

Translating the active coalescing start snapshot keeps its cumulative delta
unchanged across a safe-boundary apply. Results are never applied in the middle
of a K-window and never rewrite an immutable owned contribution.

The inner optimizer's gradient moments, variance accumulators, loss-scaler,
and step counter are retained. Any optimizer buffer representing a
parameter-space point, including ScheduleFree `z`, MUST receive the same
component shift. The initial profile supports only the audited ScheduleFree
`x/z` mapping; an unknown parameter-valued buffer fails admission. On
incarnation/fence restart, local parameters, sealed/coalescing/correction
state, and the entire inner optimizer are discarded; the worker initializes
fresh local inner state from the newest committed `S_h`. Inner state is never
global authority.

An executable scalar/vector boundary reference is:

```python
def safe_boundary_rebase(
    local, old_anchor, new_anchor, accepted_local_deltas, parameter_points=(),
):
    local = np.asarray(local, dtype=np.float64)
    old_anchor = np.asarray(old_anchor, dtype=np.float64)
    new_anchor = np.asarray(new_anchor, dtype=np.float64)
    accepted = np.zeros_like(local)
    for delta in accepted_local_deltas:
        accepted = np.add(accepted, np.asarray(delta, dtype=np.float64))
    correction = np.subtract(np.subtract(new_anchor, old_anchor), accepted)
    rebased = np.add(local, correction)
    translated_points = tuple(
        np.add(np.asarray(point, dtype=np.float64), correction)
        for point in parameter_points
    )
    expected_disposable = np.subtract(np.subtract(local, old_anchor), accepted)
    np.testing.assert_allclose(
        np.subtract(rebased, new_anchor), expected_disposable,
        rtol=1e-6, atol=1e-7,
    )
    return rebased, translated_points
```

The scalar/vector and full-E97 gates use these tolerances because binary
floating-point addition/subtraction need not invert bitwise.

## Bounds, backpressure, and failure rules

All queues are count- and byte-bounded before model load. The initial E97
profile has:

- one immutable sealed/frozen/inflight descriptor and one mutable cumulative
  coalescing interval per worker/trainer lane;
- one deterministic accepted-delta correction buffer per trainer, reusing the
  immutable cohort storage rather than adding an unbounded history;
- one open global group and one compact prior idempotency/receipt ledger;
- one visible verified result mailbox slot and exactly one replacement staging
  slot;
- one worker-applied anchor plus its local state; and
- the existing NDP fixed registered slots, owner bounds, replay bound, and at
  most two owner reassignments.

For E97 float64 layout `L = 5,506,770,496`, each trainer f32 vector is `L/2`.
Two bounded eight-trainer cohorts are `8L = 44,054,163,968` bytes. Adding the
retained v1 owner-resident bound `14,440,737,184` and one replacement result
`L` gives a conservative native-service admission of `64,001,671,648` bytes.
This is below the configured 64 GiB (`68,719,476,736` byte) shared cap with
`4,717,805,088` bytes remaining. The correction accumulator reuses the owned
cohort; the worker-applied anchor is trainer/GPU model state, not another
native-service shared allocation. No third cohort is permitted even
transiently. Implementations MUST still preflight the complete,
non-overlapping resident formula and prove all dtype/reuse assumptions. If the
formula does not fit, startup fails; it may not silently lower a lag bound,
spill dense records to Lustre, or allocate on demand.

At every safe boundary, actions occur in this order:

1. finish the current K-window and extend the one cumulative interval, exact
   token count, and window range;
2. reject and release a sealed contribution for which current
   `g - base_version > 6`;
3. take the newest verified mailbox result and apply the correction-ledger
   rule; if `anchor_lag_before_apply` reaches six and the result is missing,
   fetch it point-to-point and pause before another window;
4. drop an unsealed cumulative interval whose original base has moved beyond
   lag six, record its exact windows/tokens, and begin a new interval only
   after the safe-boundary apply;
5. transfer the interval descriptor locally if its one sealed slot is free;
   `OWNED` ends trainer responsibility, and no send/receipt wait follows;
6. if the mutable interval has reached eight windows while the sealed slot is
   still busy, pause before another window; and
7. if catch-up or capacity does not recover within the 420-second stage
   deadline, drain the incarnation, discard all disposable work, and rejoin
   from authoritative latest.

A contribution at lag exactly six is eligible only for the currently open
group. It is not carried into another group. A contribution beyond six is
always dropped. The system never discards a selected/frozen contribution to
make room: it completes bounded replay/reassignment or aborts that group.

Missing peers do not block local K-windows while buffers and lag remain below
their bounds. At the group deadline, a missing contribution is excluded; the
group commits only if `Q_min/T_min` still hold. A contribution arriving after
freeze is late for that group but may remain the worker's single eligible
descriptor for the next group. With the initial two-node `Q_min = 2`, loss of
either peer prevents global commit; surviving local work continues only until
the explicit speculative-window/global-lag bound, then pauses rather than
inventing one-node authority.

A returning stable worker always uses a new random incarnation. It discards
old local/inner/descriptor/coalescing state, verifies current
run/fence/policy/code/layout,
loads authoritative latest, and becomes eligible only for a subsequent group.
Old-incarnation frames and results can satisfy only an already frozen exact
identity; otherwise they are rejected.

The complete envelope digest is the idempotency key. An identical retry returns
the original receipt without a second add. Reuse of `(run, fence, worker,
incarnation, sequence, local_window_start, local_window_end)` with any
different metadata or payload is a conflict, quarantines that route for the
attempt, and cannot mutate an accumulator. Wrong fence, policy, layout, code,
base digest, token count, window range, checksum, bounds, nonfinite payload, or
lag is rejected before accumulation. Loss of the allocation lease stops
admission and publication immediately.

## Atomic state, checkpoint, and restart

For every successful transition, the publisher writes and reload-verifies one
immutable bundle containing `S_(g+1)`, `O_(g+1)`, the prior state digest, frozen
contribution identities, exact tokens and aggregation weights, lag values known
at commit (base-at-seal and commit lag), rejections/drops, membership, result
roots, and policy/layout/code digests. The current fenced
holder then atomically advances authoritative `latest`. There is one complete
committed version or none. Result-version-at-apply and result-window-age are
later fenced apply receipts cross-linked to this immutable bundle; they never
mutate it. A native result, mailbox insertion, local apply, or checkpoint file
without the fenced latest CAS is not a commit.

Local training may continue while reduction, outer apply, reload verification,
and checkpoint publication proceed. A failed publication never enters a
mailbox and never advances global version. Workers may continue from an older
anchor only while the finite lag/buffer rules permit it.

Fresh-allocation recovery acquires a strictly newer fence and restores the
latest complete `S_g` and exact `O_g`. It rejects all older-fence frames,
mailboxes, receipts, and local journals. Missing/corrupt model or outer state is
unrecoverable. Disposable local K-windows after `S_g` may be lost.

Checkpoint correctness latency is separate from training-lane performance.
The initial two-node maximum from group freeze through reload-verified latest
CAS is 420 seconds and is reported independently. Excluding it from
K-boundary cadence does not waive the correctness deadline, atomicity, or lag
backpressure.

## Native data-plane constraints

V2 preserves R01–R16 and NDP01–NDP17 except where this reviewed policy
explicitly replaces v1's fresh-only generation semantics. In particular:

- managers remain model-free; trainers alone own model and inner optimizer;
- dense handoff/reduction/redistribution remains a persistent compiled
  point-to-point native service with exact Frontier `cxi`;
- there is no MPI initialization, all-rank collective/barrier, or wait for all
  launched/READY ranks;
- dense contributions, descriptors, coalescing/correction buffers, mailboxes,
  and redistribution do not use Lustre, Python TCP, Python object
  serialization, or a central full-model broker; and
- fixed buffers, credits, checksums, deterministic arithmetic, replay,
  fencing, deadlines, release counters, and direct owner redistribution remain
  mandatory.

The v1 wire identity and one-open-generation ABI cannot be relabeled as v2.
Implementation requires a versioned protocol/ABI extension carrying local
window range/coalescing facts, base version, policy/code digests, exact tokens
separately from aggregation weight, both lag clocks, local ownership-transfer
status, the one-sealed/one-mutable bound, and accepted-delta correction
identity. The local `OWNED` response has a one-second metadata/control
deadline; it says nothing about send or receipt completion. This is an
extension of NDP01–NDP17, not permission to bypass them.

## Two-node acceptance only

No gate below authorizes 4+ nodes. Every artifact cites R01–R16, NDP01–NDP17,
and V2A01–V2A18.

### Numerical and deterministic-reference gate

- Scalar and vector fixtures exercise every global lag 0–6, cumulative
  intervals of 1–8 local windows, unequal exact tokens, one-slot coalescing,
  deterministic digest order, `eta = 0.5`, skipped mailbox versions, accepted
  correction sums, nonaccepted anchors, and ScheduleFree `x/z` translation.
- Native aggregation is bitwise equal to the executable binary64 reference for
  every arrival permutation. Replaying the same frozen manifest produces
  byte-identical result and outer-state roots.
- Full-cohort lag-zero v2 is checked against its explicit half-step equation.
  V1 synchronous equivalence remains a separate compatibility test and is not
  attributed to v2.
- Full-E97 worker rebase preserves the `x` and local `z` displacements from the
  global model anchor with
  `rtol <= 1e-6`, `atol <= 1e-7`; all states remain finite.

### Failure and restart gate

An exact two-node `Partition=batch`, `QOS=debug` run injects delayed/missing
contributions, a lag-six contribution, a lag-seven drop, duplicate and
conflicting identities, checksum/nonfinite corruption, one-second local
admission failure, an eight-window coalescer pause, mailbox replacement,
trainer/service loss, owner reassignment, failed publication, and
new-incarnation/fresh-allocation rejoin. It proves bounded pause/release, no
partial commit, exact old-fence rejection, and restart of both `S_g` and
`O_g`.

### Decoupled performance gate

After at least two warm-up windows, retain at least ten consecutive measured
K40 windows for every one of 16 real trainers. A clean interval is
background-healthy only while lease/quorum are valid, no failure is injected,
all native byte/credit/memory bounds hold, local `OWNED` admission is within one
second, p99 and maximum accepted `commit_lag` and
`anchor_lag_before_apply` are at most `tau_target = 2`, p99 and maximum
speculative lag are at most `sigma_target = 2`, and no hard-bound pause has
fired. Hard-ceiling
behavior remains visible but cannot be selected away to create a clean
interval. For every healthy interval:

- local K-windows continue while at least one accurately versioned prior
  contribution is being handed off, transported, reduced, committed,
  redistributed, or published;
- local reduction, owner redistribution, trainer apply, checkpoint/publication,
  and remaining control/handoff/integrity work have independent intervals and
  do not disappear into an inferred “quorum time”;
- median steady-state K-boundary cadence is no greater than `1.25` times
  median raw K40 compute;
- aggregate foreground idle divided by boundary cadence is strictly below
  `0.10`; and
- telemetry reports base-at-seal, commit, anchor-before-apply, result-version,
  and local-window lag,
  descriptor/coalescing/mailbox high-water, interval window ranges,
  local-ownership versus fabric-send/receipt times, pauses/drops, exact tokens,
  and background intervals with their true contribution/global versions.

The cadence bound is unconditional inside a healthy interval; a stage
“fitting” a K-window does not excuse slow cadence. Backpressured/unhealthy
intervals remain in the artifact and fail the clean performance gate rather
than disappearing from the denominator. Correctness latency
(`freeze -> durable latest`) and checkpoint bytes/duration have a separate
pass/fail result and are not charged as foreground idle. A run passes only if
both correctness and training-lane gates pass. No stage from a lagged
contribution may be renamed generation `g`/`tau=0` merely because it overlaps
the current local window.

### Convergence and reproducibility gate

- Two independent replays of an identical frozen contribution schedule are
  byte-identical after removing timestamps and allocation identities.
- Three predeclared seeds run v2 and v1 `tau=0` from the same checkpoint,
  data order, exact accepted-token budget, and K40 count for at least 100
  commits. The paired 95% confidence interval upper bound for v2 minus v1
  held-out BPB is at most `+0.02`, no individual seed is worse by more than
  `+0.05` BPB, and neither arm has nonfinite loss or an un-recovered
  greater-than-2x merge shock. The promotion arm keeps accepted global lag at
  0–2 and speculative p99 at 0–2; a separately labeled stress arm exercises
  hard lags 3–6 and local ages 3–8 without being treated as production health.
- The v2 artifact reports the accepted set, lag/weight distribution, effective
  sample size, coalesced window/token ranges, drops, pauses, and token clock for
  every commit. Timing-dependent membership differences are never called
  bitwise reproducibility.
- Passing these thresholds qualifies only this exact experimental
  `tau_hard/tau_target/sigma/eta/weight` tuple. Failure rejects it; it does not
  trigger an unreviewed search or silent fallback.

Only a reviewed follow-up may promote this exact policy beyond two nodes after
all numerical, failure, performance, restart, and convergence gates pass.

## Rejected alternatives

- **Keep `tau = 0` and require full overlap:** causally impossible as proved
  above.
- **Unbounded asynchronous updates:** no finite memory, convergence, or failure
  contract.
- **One immutable dense enqueue per K:** arrival exceeds service in the retained
  run and any FIFO is unstable; v2 cumulatively coalesces a named adjacent
  window range behind one sealed slot.
- **Latest-only update replacement:** discards intervening tokens and makes the
  update basis timing-dependent; v2 seals nonoverlapping cumulative intervals.
- **Equal/rank weighting:** discards exact work accounting; v2 uses exact tokens
  with an integer lag factor.
- **Always rebase from the accepted endpoint:** correct only in a restricted
  single-interval case and can erase unaccepted speculative work; v2 subtracts
  the exact accepted-delta ledger from the global shift.
- **Apply results mid-window or reset inner state on every result:** changes the
  meaning of K and destroys continuous local work; v2 rebases only at the safe
  boundary and preserves audited inner state.
- **Make checkpoint latency invisible:** local cadence and durable correctness
  are separate measurements, but both remain required.
