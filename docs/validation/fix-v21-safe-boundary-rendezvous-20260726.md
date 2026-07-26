# Async-v2.1 safe-boundary rendezvous validation

Date: 2026-07-26

WG task: `fix-v21-safe-boundary-rendezvous`

Status: **LOCAL IMPLEMENTATION AND GATES PASSED; LIVE MODEL GATE UNRUN**

## Outcome and scope

The native async-v2.1 apply path now has two explicitly separate phases:

1. Each trainer prepares and reload-verifies immutable candidate bytes in the
   background, then publishes a fenced `candidate_prepared` receipt. This
   receipt cannot satisfy the apply gate and does not stop the active K lane.
2. After all eight candidates exist, the node manager opens a distinct,
   transaction-bound, 420-second safe-boundary rendezvous. Each trainer
   finishes its already-running K window without translating `x`, `z`, or the
   mutable interval start, and publishes a distinct `boundary_ready` receipt.
   Only eight matching boundary receipts permit manager release.

The immutable 60-second apply clock is constructed as
`released_monotonic_s + 60.0` at the all-eight manager release. It does not
exist during candidate preparation or partial boundary arrival. After release,
each quiescent trainer performs the already-prepared ScheduleFree translation
once. All model and optimizer destinations are validated before the first
resident mutation. Eight transaction-bound native-applied receipts reduce to
one node-applied marker; fewer than eight cannot publish the node marker or
advance READY.

A preparation or rendezvous failure publishes a fenced abort before release.
The abort records zero applies, candidate/boundary counts, and raw/max/p99
rendezvous timing. Trainers waiting at an early boundary retain their
individual foreground-idle interval and observe the abort before translation.
The node cohort can then restart from verified latest under a fresh node and
trainer incarnation; the aborted transaction identity cannot be replayed into
the new cohort, and a stopped training lane rejects a second correction.

This implementation task did not submit, launch, hold, cancel, requeue, or
otherwise mutate a Slurm job. No `sbatch`, `srun`, `salloc`, `scancel`, or
`scontrol update` command was run. It does not claim a clean, performance,
fault, promotion, or scale qualification.

## Authorities and conformance checklist

The implementation and this evidence were checked against:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including its conformance
  checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including NDP01–NDP17,
  V21S01–V21S17, and the ISP01–ISP07 amendment;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`;
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`; and
- the retained job-5081295 source/evidence at
  `/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/`
  `qualify-simple-async-v21-2n-clean/`
  `35f33399cb60e40d726fc290b5d9d6f524be9ad0/clean/clean-overlap/`
  `clean-overlap`.

The compute-pool conformance checklist is satisfied locally as follows:

- Allocation fence, run, result version/root, node incarnation, trainer
  incarnation, rank, candidate digest, and one derived transaction digest are
  present at every new phase. Identical duplicate receipts are idempotent;
  conflicting candidate, boundary, or apply receipts fail closed.
- Active membership, exact two-node quorum/token policy, deterministic
  exact-token reduction, current-fence commit authority, immutable checkpoint
  lineage, catch-up/rejoin, and outer-state restore are unchanged.
- Manager and service roles remain model-free. Only trainers own mutable
  model/optimizer state. Candidate preparation reads the existing shared
  service result through the reviewed capacity-one authenticated reader lane.
- No dense Python socket, Lustre hot path, shared SQLite/database, central
  full-model broker, MPI initialization, all-rank collective, or launched-rank
  barrier was added. The change adds compact node-local JSON receipts only.
- Candidate preparation, boundary rendezvous, and released apply have separate
  absolute deadlines and terminal outcomes. Pre-release failure has zero
  applies. Post-release partial receipt state cannot publish the all-eight node
  marker and is handled by the existing cohort restart.
- Raw event arrays plus recomputed maximum and p99 are retained for candidate
  preparation, boundary wait, release-to-apply, and total foreground idle.
  Existing snapshot/admission raw tail evidence remains required. The semantic
  validator rejects a 421-second boundary tail and the pre-existing
  approximately 200-second alternating K stall even when median summaries
  look healthy.
- Local commands and exact counts are recorded below. No local result is
  relabeled as Frontier qualification or scale authorization.

## Retained job 5081295 remains failed evidence

Job `5081295` remains **FAILED** with terminal state/exit `FAILED 143:0`. Its
source commit was
`35f33399cb60e40d726fc290b5d9d6f524be9ad0`; scheduler evidence separately
named `Partition=batch` and `QOS=debug`.

The retained control trees contain:

- 16/16 old `native-apply-ready` receipts;
- 16/16 serialized native-result preparation receipts;
- 2/2 old node release markers; and
- only 9/16 `native-applied` receipts.

All immutable candidates prepared in 42.176–46.543 seconds, but the old
receipt asserted only candidate/reload readiness. Several trainers had just
started an approximately 67-second K40 window. The old manager incorrectly
treated those candidate receipts as boundary receipts, released immediately,
and started the absolute 60-second all-eight apply deadline before the
trainers' K boundaries. This is the reproduced causal defect, not qualifying
evidence.

Payload
`34f4404a856cf7df966f76ef7e9e72ac19dd38518be9c0f88e925198034c5d43`
is permanently retired. This task neither reused nor resubmitted it.

## Causal implementation

`SafeBoundaryRendezvous` owns the in-memory manager state machine:

```text
8 candidate_prepared
        |
        v
open bounded boundary rendezvous
        |
        v
8 exact candidate/trainer boundary_ready
        |
        v
manager release; apply_deadline = release + 60 s
        |
        v
8 exact native-applied -> one node-applied marker
```

Candidate and boundary records occupy different dictionaries, schemas, files,
and methods. `release_apply` examines only the complete boundary set. The
transaction digest covers run/fence/node/node-incarnation/result version/result
root/eight-trainer cardinality. Every later marker carries that digest.

`PersistentAsyncTrainingLane.finish_at_boundary(corrections=None)` stops only
after the active K and returns a cached zero-translation boundary report.
`apply_at_boundary` is unavailable until that report exists and rejects a
second call. The manager release is read between those calls. Candidate
checkpoint serialization, hashing, reload verification, and latest-CAS checks
remain before the first call and outside the released 60-second interval.

The manager writes `native-boundary-abort` for any exception before release.
No trainer calls `apply_at_boundary` or `PersistentRealWorkerSession.translate`
until it has read the exact release marker. Aborted early arrivals retain wait
telemetry, then fail into the existing all-eight cohort recovery instead of
publishing mixed state.

## R01–R16 traceability

| ID | Local conformance result |
|---|---|
| R01 | The new receipts bind the existing scheduler fence and immutable run/result authority; no shared database or lease-renewal store was introduced. |
| R02 | Node/trainer incarnations bind candidate, boundary, release, apply, and abort. Existing cohort restart supplies fresh incarnations. |
| R03 | The change is eight local trainers under one already-admitted node; global active world remains leased READY membership, never launched ranks. |
| R04 | Candidate, boundary, and apply identities reject stale/conflicting replay while identical duplicates are idempotent. |
| R05 | Exact-token deterministic binary64 aggregation is unchanged; the rendezvous begins after the complete immutable result exists. |
| R06 | Candidate, 420-second rendezvous, and 60-second apply bounds are absolute; exact Q/T floors and generation closure are unchanged. |
| R07 | Peer commit/latest remains immutable authority. Candidate or partial boundary evidence cannot create apply/commit/READY. |
| R08 | Existing bounded native reader credit, ownership, checksums, replay, and prompt result-view release are preserved. |
| R09 | Managers remain model-free. Trainers alone stop at K boundaries and mutate resident `x/z` only after release. |
| R10 | New traffic is compact node-local control metadata; dense state remains native/memfd, non-Lustre, and database-free. |
| R11 | A failed rendezvous/apply cannot advertise READY; fresh-incarnation cohort restart reloads verified latest. |
| R12 | Immutable global/outer/token recovery is unchanged; disposable local inner state is never promoted by a partial transaction. |
| R13 | No backend-neutral membership/reduction protocol changed; the implementation stays in the existing Frontier role adapter and generic v2 state machine. |
| R14 | Preparation, rendezvous, release/apply, abort, and total-idle intervals now have distinct bounds and raw/max/p99 evidence. |
| R15 | Exact-token accounting and correction-ledger math are unchanged; atomic prevalidation prevents malformed `z` from partially changing `x`. |
| R16 | Native local gates pass; no live two-node or 4+ job was submitted, and no promotion/scale readiness is claimed. |

## NDP01–NDP17 traceability

| ID | Local conformance result |
|---|---|
| NDP01 | Python owns fenced apply coordination; C++ continues to own dense handoff/reduction/transport. No production Python dense TCP or shared control database was added. |
| NDP02 | Rendezvous uses local point-to-point metadata and has no MPI/all-rank operation. |
| NDP03 | One persistent native service/provider policy per node is unchanged. |
| NDP04 | The service still consumes immutable bounded snapshots/results only; no extra trainer-sized copy or live-state background read was added. |
| NDP05 | Fixed layout, exact-token arithmetic, rounding, and encoding are unchanged and pass 10/10 native CTests. |
| NDP06 | Every new receipt carries run/fence/generation/result/node/trainer/transaction identity; conflicts fail before mutation. |
| NDP07 | Endpoint exchange and current-fence AV route installation are unchanged. |
| NDP08 | Existing preallocated snapshots, capacity-one result reader, and resident limits are preserved; the rendezvous allocates compact metadata only. |
| NDP09 | Candidate preparation may consume the reviewed reader credit but remains background; credit restoration/fabric completion semantics are unchanged. |
| NDP10 | Existing CRC/SHA/once-only native semantics are preserved; the rendezvous adds candidate digests and idempotent exact-identity receipts. |
| NDP11 | Sender replay and owner reassignment remain bounded; abort/restart cannot fabricate a second correction identity. |
| NDP12 | Trainers still read one service-owned node aggregate; no eight-file result fan-out was added. |
| NDP13 | Background preparation, boundary rendezvous, and apply have separate absolute deadlines. Pre-release timeout aborts with zero applies. |
| NDP14 | The C ABI and metadata-only seqpacket channel are unchanged; rendezvous metadata remains Python/node-local control. |
| NDP15 | Reload-verified immutable candidate preparation precedes a later safe-boundary release. Checkpoint I/O is outside the 60-second apply clock; eight receipts gate one node marker. |
| NDP16 | Structured telemetry retains transaction identities, raw events, maximum, p99, bounds, abort reason/counts, and total foreground idle. |
| NDP17 | Canonical native build/CTest passed locally. No G2/G3/model/scale run is claimed. |

## V21S01–V21S17 traceability

| ID | Local conformance result |
|---|---|
| V21S01 | Canonical v2.1 identities remain pinned; the new schemas are v2.1-specific and no v2.0 work is relabeled. |
| V21S02 | Commit/anchor/result/speculative clocks and lag-two policy are unchanged; candidate readiness cannot pause or masquerade as apply readiness. |
| V21S03 | Positive exact tokens remain the sole quantitative floor, numerator weight, denominator, and accepted-token clock. |
| V21S04 | Exact K40 and stateless `eta_outer=1.0` behavior are unchanged; phase two explicitly finishes the currently active K. |
| V21S05 | Stable worker/incarnation/window/base/policy/layout/code/token identity and exact two-node Q/T policy remain unchanged; transaction identity adds stricter fencing. |
| V21S06 | One persistent trainer owns model/optimizer/iterator/hidden state; candidate work is background and the trainer stops only at the explicit K boundary. |
| V21S07 | Complete verified correction applies exactly once to `x`, `z`, and mutable start after release. All x/z destinations validate before mutation and double correction rejects. |
| V21S08 | Only reload/latest-CAS-verified candidates enter phase one. Capacity and invalid-result behavior remain fail-closed. |
| V21S09 | Resident/credit/replay/mailbox/deadline bounds and one-owned/one-mutable/no-third-cohort policy are unchanged. |
| V21S10 | Leased membership/rejoin semantics are unchanged; fresh node incarnation derives a different transaction and stale work cannot reduce the two-node floor. |
| V21S11 | Directly fixed: eight prepared candidates open rendezvous; eight matching boundary receipts start one 60-second apply; eight applies produce one node marker; pre-release failure produces none. |
| V21S12 | Persistent compiled service, producer-direct memfd, exact reduction, bounded point-to-point fabric, and no Python/Lustre dense/broker/collective invariants are preserved. |
| V21S13 | Preparation, boundary, release-to-apply, total-idle, snapshot, causal phase, lag, and high-water evidence includes raw events and hard-tail validation. |
| V21S14 | Immutable model/outer/token/checkpoint/receipt recovery remains authority; transaction/node markers add exact apply identity without changing seed handling. |
| V21S15 | Local implementation components pass. The required exact-two-node live numerical/clean/fault/replay/convergence gates were not run and remain unclaimed. |
| V21S16 | No promotion review or scale authorization was issued. |
| V21S17 | Scale-only deterministic closure is unchanged and unexercised; no 4+ constant or launched-rank close was introduced. |

## ISP01–ISP07 traceability

| ID | Local conformance result |
|---|---|
| ISP01 | Trainer ownership and coherent immutable snapshot/result preparation remain unchanged; the manager sees only compact receipts and candidate digests. Snapshot coherence/background-access tests remain in the focused suite. |
| ISP02 | Candidate preparation, hashing, reload, and capacity-one reading remain background while the persistent K lane runs. Only the explicit boundary rendezvous stops after the active K. |
| ISP03 | Native publication/checkpoint/hash/reload work consumes immutable state and completes before the release/apply clock. No checkpoint I/O was moved into boundary apply. |
| ISP04 | Existing snapshot/mailbox/credit/replay/receipt capacities are unchanged and exhaustion tests continue to prove advancing foreground work; the rendezvous has one finite timer. |
| ISP05 | Directly fixed: preparation is not boundary readiness; eight exact boundaries precede release; the 60-second clock begins at release; pre-release abort has zero applies; translation is atomic-prevalidated and once-only; one node marker requires eight. |
| ISP06 | Runtime and validator separately retain snapshot, preparation, boundary rendezvous, release-to-apply, and total-idle raw/max/p99/bounds. Foreground result wait remains zero outside the explicit rendezvous/apply phases. |
| ISP07 | The validator recomputes max/p99 from raw events, rejects inconsistent summaries and a hidden 421-second boundary event, and retains the existing 200-second alternating-stall rejection. Median-only evidence cannot pass. |

## Regression-first evidence

The job-5081295 regression was added before the production state machine and
first run after canonical Frontier activation:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_async_diloco_v21.py \
  -k 'job_5081295 or apply_deadline_starts'
```

The red result failed during collection:

```text
ImportError: cannot import name 'SafeBoundaryRendezvous'
```

The fixture creates two node transactions with 16/16 prepared candidates but
only four plus five boundary arrivals, matching the historical 9/16 shape.
The fixed implementation refuses release, constructs no apply deadline, and
records zero applies. Adjacent tests prove:

- candidate and boundary receipts have distinct states and exact candidate/
  trainer identity;
- the apply deadline is exactly release plus 60 seconds only after all eight
  boundaries;
- boundary timeout records seven early waiters, zero applies, no node marker,
  and a different transaction after fresh-incarnation retry;
- success records exactly eight applications and one exact-transaction node
  marker;
- a stopped persistent lane does not translate before release and rejects a
  second translation; and
- malformed ScheduleFree `z` validation leaves all `x/z` points unchanged.

## Validation commands and exact results

Every Python, pytest, native build, and CTest command used the canonical
Frontier activation and `"$EMENDER_PYTHON"`/`PYTHON_BIN="$EMENDER_PYTHON"`.

Focused v2.1/boundary/atomic-xz suite:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_diloco_v21.py \
  tests/test_async_snapshot_pipeline.py \
  tests/test_async_diloco_real_trainer.py
```

Result:

```text
49 passed in 38.50s
```

Broad affected role/launcher/semantic-validator suite:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_validate_pipelined_e97_performance.py
```

Result:

```text
129 passed in 119.04s
```

Canonical native configure/build/CTest/install/attestation:

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result:

```text
100% tests passed, 0 tests failed out of 10
build/native-resilient-dataplane/native-artifacts.json: status=recorded
manifest SHA-256:
15525b4c304e273964590595485d7943c315c7b73a336304b77b8a0a904c92db
```

Syntax and whitespace gates:

```bash
"$EMENDER_PYTHON" -m py_compile \
  ndm/async_diloco_v2.py \
  ndm/async_diloco_real.py \
  scripts/frontier/resilient_e97_role.py \
  scripts/frontier/validate_pipelined_e97_performance.py
git diff --check
```

Result: both commands exited zero.

## Git and publication evidence

The implementation and retained evidence are committed surgically. Final
commit hashes, the non-force push result, clean `HEAD == origin/main`, and
`git ls-remote origin refs/heads/main` equality are recorded in the WG task
log after publication. No live clean job was submitted here.
