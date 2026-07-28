# Direct async v2.1 systems-scale quality pass

Date: 2026-07-28

Task: `.quality-pass-v21-direct-scale`

## Decision

The replacement systems-scale batch is internally coherent after the WG
description and edge edits recorded below. Its only execution path is:

```text
requalify-v21-durable-collector-2n-clean
  -> qualify-simple-async-v21-2n-faults
  -> codify-v21-direct-scale-policy
  -> scale-v21-direct-8n
  -> scale-v21-direct-32n
  -> scale-v21-direct-128n
  -> review-v21-256n-readiness
```

`scale-v21-direct-8n` also retains a direct dependency on
`qualify-simple-async-v21-2n-faults`, so it cannot be released by a policy
update that lacks the passed fault/restart artifact. The fault task already
depends on the completed clean task. Assignment/evaluation lifecycle tasks are
WG metadata and are not substitute scale gates.

There is no 4-, 16-, or 64-node rung in this batch. There is no 256-node
runner. The 256 task makes an evidence-backed go/no-go decision only and cannot
create, release, or submit a runner. The long three-seed/100-commit convergence
study is a separate model-quality question and is not a dependency of this
systems-scale path.

## Authorities reviewed

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, architecture decision and design
  authority, version 1 with the accepted v2.1 specialization.
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including the conformance checklist
  and the normative R01-R16, V21S01-V21S17, and ISP01-ISP07 crosswalks.
- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`, accepted ADR-002 for
  `async-decoupled-v2.1-simple`.
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, native data-plane authority
  version 1 and NDP01-NDP17.
- `docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`, execution-source identity and
  durable scheduler transaction.

The currently accepted authorities still contain the older five-gate
convergence prerequisite and `4 -> 8 -> 16 -> 32 -> 64 -> 256` scale ladder.
That contradiction is not waived by this quality pass. It is the first
fail-closed deliverable of `codify-v21-direct-scale-policy`, which now must
amend all five authority/operational documents together before 8 nodes can
become ready. Until that policy/controller task passes, no replacement rung is
authorized.

## Batch-wide requirement allocation

All R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 requirements apply to
the policy/controller update and each live systems runner because the
authorities require the complete conformance checklist for an asynchronous
scale task. The 256 review does not re-execute them; it audits their immutable
8/32/128 evidence and fails closed on any gap.

### Compute-pool requirements

| ID | Policy/controller responsibility | 8/32/128 runner evidence | 256 review use |
|---|---|---|---|
| R01 | Pin execution source, scheduler claim, fence, and no-database admission. | Prove exact source/claim/fence before model load and reject stale authority. | Audit one exact lineage and no stale/ambiguous claim. |
| R02 | Preserve DISCOVER through READY/DRAIN/EXPIRE and stable-worker incarnations. | Retain leased peer lifecycle, expiry, and incarnation evidence. | Compare lifecycle and churn behavior by rung. |
| R03 | Make leased READY membership the active world; forbid launched-rank closure. | Report leased/READY/accepted/missing cohorts and finite closure. | Reject launched-rank or hidden-cohort reasoning. |
| R04 | Bind contribution identity, lag/staleness, replay, and conflict rejection. | Prove stale/corrupt/conflicting rejection and identical idempotence. | Audit reject/duplicate invariants and hidden failures. |
| R05 | Preserve exact-token deterministic binary64 aggregation and reference tests. | Prove exact-token `eta_outer=1` accounting and finite numerical results. | Compare numerical/accounting integrity without a quality claim. |
| R06 | Pin minimum progress, finite close, stage/run deadlines, and no unbounded wait. | Retain floor, V21S17 close, timeout, abort, and nonblocking evidence. | Reject unexplained constants, extended close, or missing deadlines. |
| R07 | Require exact-once peer commit plus immutable checkpoint/receipt lineage. | Prove one result or none and a reload-verified fenced checkpoint chain. | Audit publication completeness and receipt continuity. |
| R08 | Bound sharded owners, chunks, credits, replay, release, and brokerlessness. | Retain high-water, replay/reassignment, receipt, and release counters. | Compare transport scaling and bound headroom. |
| R09 | Keep managers model-free and trainers exclusive owners of mutable state. | Prove coherent immutable snapshots and no background live-state read. | Audit snapshot ownership and interruption tails. |
| R10 | Forbid SQLite/shared locks and Lustre/Python dense hot paths. | Prove zero forbidden control/data-plane paths for each rung. | Reject any fallback that weakens the production path. |
| R11 | Preserve catch-up, expiry, new-incarnation rejoin, and old-work rejection. | Retain missing/late/rejoin and stale-incarnation evidence. | Assess recovery scaling and failure containment. |
| R12 | Bind and restore model, outer step, token clock, result root, and apply state. | Verify restartable immutable checkpoint/result/apply identity. | Audit recoverability at the exact reviewed state. |
| R13 | Keep the protocol backend-neutral while validating the Frontier adapter. | Use the reviewed Frontier adapter without scheduler semantics leaking into membership. | Reject a 256 proposal that changes membership semantics. |
| R14 | Pin absolute deadlines and causal foreground/background telemetry. | Retain every required phase, bound, maximum, p99, and terminal reason. | Compare hard tails and reject missing/median-only telemetry. |
| R15 | Prove numerical/reference and changing-participation accounting. | Report exact accepted tokens/effective cohort and finite loss. | State explicitly that this is systems evidence, not convergence. |
| R16 | Codify the replacement promotion policy and immediate-predecessor checks. | Require exact passed predecessor at 8, 32, and 128. | Make the post-128 decision without auto-promoting 256. |

### Native data-plane requirements

| ID | Policy/controller responsibility | 8/32/128 runner evidence | 256 review use |
|---|---|---|---|
| NDP01 | Preserve native peer-control authority and the Python/C++ boundary; forbid shared DB control. | Prove current-fence peer control, native dense ownership, and no database. | Audit that scale did not centralize live authority. |
| NDP02 | Reject MPI, all-rank collectives, and launched/READY unanimity. | Prove bounded point-to-point progress with missing/late peers. | Reject collective or fixed-world evidence. |
| NDP03 | Require one persistent C++17 `FI_EP_RDM` service and exact Frontier `cxi`. | Retain provider, endpoint, build, and persistent-service facts. | Compare provider/route scaling and errors. |
| NDP04 | Require coherent XPMEM/memfd immutable handoff and no extra full copy. | Retain handoff kind, coherent boundary, and zero full-copy evidence. | Audit snapshot/handoff costs and ownership. |
| NDP05 | Preserve fixed layout/conversion/order/rounding/result encoding. | Prove deterministic exact-token roots and finite arithmetic. | Reject changed math or incompatible layout. |
| NDP06 | Bind every command/frame/receipt/result/checkpoint handoff identity. | Retain exact fence/generation/worker/incarnation/digest evidence. | Audit identity continuity across all rungs. |
| NDP07 | Exchange opaque endpoints through leased membership and current-fence routes. | Prove route installation/expiry without PMI, DNS fan-out, or filesystem polling. | Assess route/membership scale behavior. |
| NDP08 | Preflight bounded snapshot/result/fabric pools and nonblocking exhaustion. | Retain admitted formulas, high-water values, and explicit skip/defer behavior. | Assess memory headroom for a possible 256 proposal. |
| NDP09 | Keep logical credit separate from fabric completion and foreground progress. | Prove bounded credits/receipts and no trainer wait after `OWNED`. | Compare credit pressure and foreground impact. |
| NDP10 | Require CRC/SHA, finite input, once-only apply, and idempotent receipts. | Retain checksum/reject/duplicate/receipt invariants. | Reject any corruption or receipt ambiguity. |
| NDP11 | Bound sender replay, owner reassignment, and optional local-only fallback. | Retain replay/reassignment bytes, deadlines, and zero Lustre spill. | Assess failure cost and retry headroom. |
| NDP12 | Redistribute owner results into one shared node aggregate. | Prove complete result roots and no eight-copy/file fallback. | Compare redistribution scaling. |
| NDP13 | Give every stage an absolute deadline and contain route/generation failure. | Prove expiry skips/defers without foreground wait or allocation abort. | Audit liveness and failure containment. |
| NDP14 | Use the versioned ABI and metadata-only local seqpacket control. | Retain v2.1 ABI/wire identity and zero dense control-channel bytes. | Reject ABI drift or dense Python control paths. |
| NDP15 | Publish from immutable inputs and apply a verified result atomically later. | Prove background checkpoint work, bounded all-eight apply, and node receipt. | Compare apply/checkpoint tails and atomicity. |
| NDP16 | Emit all structured identities, bounds, counters, phases, and terminal reasons. | Retain causal raw telemetry plus immutable collector verdict. | Missing telemetry is an automatic NO-GO. |
| NDP17 | Enforce exact native prerequisites and the reviewed replacement scale order. | Prove matching G2/native bundle and exact immediate predecessor. | Require a separate reviewed authorization before any 256 runner. |

### Async v2.1 requirements

| ID | Policy/controller responsibility | 8/32/128 runner evidence | 256 review use |
|---|---|---|---|
| V21S01 | Pin v2.1 policy/schema/ABI/wire/digest identities and reject v2.0. | Prove exact compatible identities before load/mutation. | Audit one immutable identity lineage. |
| V21S02 | Preserve four distinct lag clocks, lag-3 drop/defer, and no foreground catch-up. | Retain all clocks, drops/deferrals, and zero foreground result wait. | Compare lag and nonblocking behavior. |
| V21S03 | Use positive exact tokens as the sole quorum/clock/numerical weight. | Prove token floors, denominator, clock, and no alternate weight. | Audit identical math across rungs. |
| V21S04 | Keep exact K40 and stateless exact-token mean with `eta_outer=1`. | Retain K40, outer step, token clock, and finite result evidence. | Reject policy/math changes. |
| V21S05 | Bind full contribution identity, floors, deadlines, and one stable-worker contribution. | Prove bound identities and the scale-authorized close/floor. | Audit identity/diversity safety. |
| V21S06 | Require coherent immutable capture, `OWNED`, and immediate trainer resume. | Retain causal capture/admission and continuing-K evidence. | Compare snapshot tails and foreground progress. |
| V21S07 | Apply complete verified results once to `x/z/interval` at a bounded safe boundary. | Prove bounded atomic application and nonblocking absent/invalid behavior. | Audit apply correctness and tails. |
| V21S08 | Keep one verified-latest mailbox plus bounded replacement staging. | Retain replacement/rejection/high-water and nonblocking evidence. | Assess mailbox pressure and correctness. |
| V21S09 | Preflight resident/slot/credit/replay/mailbox bounds; forbid a third cohort/spill. | Prove formulas, high-water, and explicit capacity outcomes. | Assess 256 memory feasibility. |
| V21S10 | Use leased READY membership, new-incarnation rejoin, and no one-node authority. | Retain membership/floor/rejoin and continuing-foreground evidence. | Compare effective cohorts and closure. |
| V21S11 | Treat all-eight apply/recovery as one node transaction. | Prove no partial visibility, exact node marker, and restartability. | Audit atomicity and recovery trend. |
| V21S12 | Preserve compiled CXI/memfd deterministic point-to-point transport. | Retain provider, byte, root, bounds, and forbidden-path evidence. | Assess transport scaling and headroom. |
| V21S13 | Emit honest causal phases, lags, high-water values, and hard-tail gates. | Retain raw phases, every-event/max/p99 pauses, and idle. | Missing or median-only evidence forces NO-GO. |
| V21S14 | Publish and restore the complete fenced bundle from the exact final seed. | Prove seed/attestation, result/receipt/checkpoint, and restart identities. | Audit exact seed and publication lineage. |
| V21S15 | Rewrite the authority so systems gates are distinct from the separate convergence study. | Prove clean/fault-derived systems criteria only; make no model-quality claim. | State what the systems ladder does and does not establish. |
| V21S16 | Replace the legacy ladder with exact `8 -> 32 -> 128 -> review 256`. | Require an immutable machine pass from each immediate predecessor. | Make an explicit review-only decision; never auto-promote. |
| V21S17 | Pin evidence-derived finite closure over leased READY membership. | Retain open/close/cohort/token arithmetic and missing-peer behavior. | Audit closure scaling and 256 feasibility. |

### Immutable-snapshot requirements

| ID | Policy/controller responsibility | 8/32/128 runner evidence | 256 review use |
|---|---|---|---|
| ISP01 | Require coherent fenced capture and exclusive live-state ownership. | Prove byte-stable admitted snapshots and no background live-state access. | Compare coherent-capture cost and failures. |
| ISP02 | Bound snapshot/admission and resume the next K window after `OWNED`. | Retain causal next-K progress while every background phase runs. | Audit foreground interruption scaling. |
| ISP03 | Restrict background publish/hash/aggregate/checkpoint to immutable inputs. | Prove stable roots and continuing K windows during background I/O. | Reject foreground checkpoint or mutable input use. |
| ISP04 | Bound slots/queues/credits/mailboxes and make exhaustion nonblocking. | Retain high-water and explicit skip/replace/drop/defer outcomes. | Assess capacity headroom. |
| ISP05 | Bound complete atomic later apply; absence/failure must not wait or partially apply. | Prove all-eight apply/node receipt or clean defer/restart. | Compare atomic-apply tails and recovery. |
| ISP06 | Retain causal disjoint phases, zero foreground result wait, max/p99, and bounds. | Produce complete raw telemetry and validator output. | Missing causal telemetry forces NO-GO. |
| ISP07 | Reject checkpoint/median-based overlap claims and hard foreground stalls. | Prove every-event tails and reject approximately 200-second stalls. | Compare hard tails, not only summaries. |

## Systems-scale acceptance contract

Each live rung must establish all of the following without claiming
model-quality convergence:

- systems correctness: fenced identity, exact-token math, idempotent receipts,
  one atomic result or none, and no stale/corrupt mutation;
- liveness: finite READY-based close, absolute deadlines, no launched-rank
  invariant, and no foreground result wait;
- transport and throughput: versioned compiled point-to-point CXI, bounded
  credits/replay/memory, exact byte/root accounting, absolute and per-node
  local-training throughput;
- membership closure: leased READY/accepted/missing/late cohorts and honest
  sufficiency;
- atomic apply and checkpoint publication: coherent immutable snapshot,
  immediate trainer resume, later all-eight apply, node receipt, immutable
  checkpoint/result/receipt chain; and
- bounded training interruption: causal every-event/max/p99 snapshot/apply
  pauses, foreground idle, and hard-tail rejection.

Every model run uses the canonical Frontier activation and exact final E97 seed
identity: step `2300930`, accepted tokens `150793748480`, size `7719680116`,
SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The model allocation is exact `Partition=batch`, `QOS=debug`; the short
measured-generation budget and thresholds are immutable before submission.
The held model job is released only after its scheduler-owned `afterany`
collector is durably registered. Retries require a changed payload digest.

Queue delay or a running job causes `wg wait`. Failure or incomplete evidence
at any rung leaves the task non-done and therefore blocks its successor.
Missing telemetry, ambiguous scheduler state, partial publication, insufficient
participation, a missing collector verdict, or evaluator prose is never a
pass. Only the collector-backed immutable machine-readable terminal pass on
the exact predecessor can advance the ladder.

## WG edits made

- Added dependency:
  `codify-v21-direct-scale-policy after qualify-simple-async-v21-2n-faults`.
- Replaced the descriptions/validation checklists for:
  - `codify-v21-direct-scale-policy`
  - `scale-v21-direct-8n`
  - `scale-v21-direct-32n`
  - `scale-v21-direct-128n`
  - `review-v21-256n-readiness`
- Retained the exact live-rung edges:
  - `scale-v21-direct-8n after codify-v21-direct-scale-policy`
  - `scale-v21-direct-8n after qualify-simple-async-v21-2n-faults`
  - `scale-v21-direct-32n after scale-v21-direct-8n`
  - `scale-v21-direct-128n after scale-v21-direct-32n`
  - `review-v21-256n-readiness after scale-v21-direct-128n`

No implementation file was edited and no Slurm command was submitted by this
quality pass.
