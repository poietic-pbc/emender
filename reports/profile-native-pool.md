# Native pool foreground-barrier profile and pipeline decision

**Decision (2026-07-20).** Convert the current trainer-visible, generation-wide
exchange/checkpoint sequence to a two-slot immutable handoff plus one latest-only
result slot.  Native collection, redistribution preparation, and durable
publication run in the background; a trainer applies only a complete result at
the next K40 boundary after validating its run, fence, generation, incarnation,
layout, base and result digests.  Keep the MVP policy `tau=0` for contributions.
Permit at most one unapplied committed result (`result_generation ==
trainer_generation`) at a boundary; otherwise skip local work and catch up from
the newest durable checkpoint.  This is the smallest conversion that removes
the foreground exchange/checkpoint barrier without weakening fail-closed
semantics.

This report profiles retained 2-, 4-, and 8-node evidence.  No 32-node evidence
exists: the retained 8-node metrics explicitly record zero jobs with 32 or more
nodes, so none was launched for this analysis.

## Measured critical path

Ranges are min–max across retained stage records; the maximum is the relevant
foreground barrier.  `—` means the retained schema did not measure that stage,
not zero.  The 2n row is job 5033125, 4n is the two-generation peer-loss job
5034117, and 8n is fresh-fence job 5034875.  Exact-source G2 end-to-end native
medians were 19.492 s (startup source), 23.648 s (4n source), and 23.006 s (8n
source), but they are a separate transport gate and are not substituted for
missing production-stage spans.

| rung | GPU K40 | SHA hashing | finite validation | local reduction | owner collection | result redistribution | trainer apply | durable checkpoint/publication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2n | 135.133–137.010 s (16) | — | — | 17.082–17.164 s (2) | 47.069–48.059 s (2) | 18.598–18.625 s (2) | 3.399–45.866 s (16) | publication completed 29.931 s after manifest commit; serialization/hash/fsync/CAS split unknown |
| 4n | — | — | — | 17.008–17.911 s (8) | 81.520–84.029 s (8; three-peer exchange) | 19.924–20.703 s (8) | 3.174–13.065 s (64) | — |
| 8n | — | — | — | 17.301–17.793 s (8) | contribution 14.776–16.838 s (8) | owner redistribution 12.182–13.234 s, then full local redistribution 55.090–56.307 s (8) | 3.199–10.757 s (64) | — |

The current observed foreground native tail, summing dependent maxima and not
route setup, is **129.71 s at 2n**, **135.71 s at 4n**, and **114.93 s at 8n**
(8n includes both redistribution phases).  These totals intentionally exclude
unmeasured integrity and durable-publication work.  At 2n the measured tail is
94.7% of the 137.010 s K40 maximum even before publication.  The 4n and 8n
evidence therefore proves where barriers are, but cannot support a cadence
ratio until K40 and publication spans are added.

Required instrumentation for the first implementation run:

1. Emit monotonic begin/end, bytes, and `foreground_wait_ns` for K40, snapshot
   seal, each `fd_sha256`, each finite scan, local reduction, owner contribution,
   owner redistribution, full redistribution, boundary validation/apply,
   checkpoint encode/write/fsync/rename, manifest hash, SQLite transaction/CAS,
   and latest-directory fsync.  Attach `(run_id, fence, generation, attempt,
   worker_id, incarnation, buffer_id)` to every span.
2. In `scripts/frontier/resilient_e97_role.py`, wrap the digest calls near the
   remote/result paths and resume/recovery reads, the reduction and apply
   telemetry, `torch.save`, `os.replace`, checkpoint SHA, and
   `finalize_checkpoint`.  In `src/native_resilient_dataplane/src/ndp.cpp`,
   count bytes and time the finite checks at the local-input, accumulator,
   frozen-result, and projection boundaries.
3. Report wall time and CPU time so overlapped work is not incorrectly added to
   trainer idle.  Record queue depth/high-water and result age at every K40
   boundary.  Unknown values above must remain unknown until these spans are
   retained.

## Integrity scan classification

| scan | correctness boundary | class and pipeline treatment |
|---|---|---|
| source-buffer SHA (`fd_sha256`) before accepting a trainer snapshot | binds immutable bytes to the authenticated contribution identity | **required**; compute while sealing/streaming chunks and carry the digest, rather than reread the sealed memfd |
| per-frame CRC32C and payload SHA on transport | detects corruption/replay at the wire/receipt boundary | **required**; stream during send/receive; never replace with a metadata-only check |
| local source finite check (`ndp.cpp` input read) | rejects nonfinite trainer input before it enters f64 accumulation | **required**; fuse with f32→f64 weighted reduction |
| f64 term/sum finite checks | detects overflow/nonfinite incremental accumulator state | **required**; fuse with each accumulator update |
| frozen f64 accumulator full scan | proves the frozen owner result is finite | **fusible** with division/projection; a separate full-buffer pass is redundant if the projection loop checks accumulator, divisor, and output |
| projected f32 result finite scan | protects the published/apply boundary | **required**, but stream with projection plus result SHA; do not rescan unchanged sealed bytes |
| result SHA at producer, manager aggregate, and each trainer | binds owner result, redistributed aggregate, and applied bytes | producer SHA is **required**; downstream full rescans are **redundant** when a sealed read-only mapping is authenticated by the same extent/root and transport receipts.  Keep one consumer verification until sealing/root propagation is proven, then replace repeated scans with metadata/root verification plus sampled debug audit |
| checkpoint SHA after `torch.save`, again for recovery metadata, and again on resume via `read_bytes()` | durable checkpoint completeness and latest-load trust | publication SHA and resume SHA are **required**; the recovery-metadata reread is **redundant**.  Hash bytes during checkpoint write, fsync, and publish the resulting digest; resume must stream-verify before use (no 7.9 GB `read_bytes()` allocation) |
| manifest/config/bundle small-file SHA | binds control and code identity | **required**, negligible; cache immutable artifact digests by inode/extent and verify again only across a trust boundary |

No integrity check may be removed merely for speed.  A rescan is eliminated only
when immutable sealing, exact extent, authenticated identity, and the upstream
digest cover precisely the bytes consumed.

## Concrete bounded pipeline

Each trainer owns two preallocated, sealed-on-handoff delta slots `D[2]` and
continues K40 in the other slot.  Each node service owns bounded frame credits,
one f64 reduction/owner accumulator for the open attempt, and one immutable f32
result slot `R`.  The checkpoint publisher owns one staging file.  Thus memory
does not grow with peers, retries, or generations; a slot is released only on a
checksummed receipt, rejection, committed result, or bounded abort.

State transitions are:

`TRAINING(g,f,i) → SEALING(g,f,i) → HANDED_OFF(g,f,i) → TRAINING(g+1,f,i)`;
background service
`OPEN → COLLECTING → FROZEN → REDISTRIBUTING → RESULT_READY → PUBLISHING → COMMITTED`.
Any digest/finiteness/identity/deadline/fence failure goes to `REJECTED` or
`ABORTED`, never `RESULT_READY`.  Fence loss atomically closes admission and
publication, invalidates all slots tagged with the old fence, and permits no
transition to committed.

The result identity is `(run_id, allocation_fence, generation, attempt,
base_digest, policy_digest, frozen_membership_root, result_digest)`.  Every
producer identity also includes stable worker ID and boot incarnation.  An
identical replay is idempotent; conflicting reuse is corrupt.  A rejoining
worker has a new incarnation, discards its old slots, syncs the latest committed
checkpoint, and joins only a subsequent open generation.

The single-entry result mailbox is latest-only but not lossy with respect to
authority: an older **unapplied volatile** result may be replaced only after a
newer result is completely verified and durably committed.  The durable chain
retains both commits.  The trainer applies only at a K40 boundary and only when
`result.base_generation == trainer_base_generation` and the current lease fence
matches.  Otherwise it does not mutate state.  If no result is ready it starts
one more K40 window.  If a second result would make lag exceed one committed
generation, admission backpressures that trainer; it discards disposable local
work and catches up from newest committed state.  This is bounded memory and
bounded staleness (`max committed-result lag = 1`) without authorizing stale
contributions: contribution `tau` remains zero.

## Failure and checkpoint behavior

- **Late peer:** contribution after freeze/deadline is rejected; sync and join
  the next open generation. **Missed quorum:** abort that attempt with no result
  or publication; bounded retry/reassignment only under the same current fence.
- **Delayed result:** trainer continues while lag is zero; at lag one it applies
  at the next boundary; beyond one it stops new local work and catches up.  A
  partial or corrupt result is quarantined and never enters the mailbox.
- **Node/owner loss:** expire its incarnation.  Before freeze, replay retained
  checksummed chunks to deterministic bounded reassignment; if the floor or
  deadline cannot be met, abort.  After a result is durably committed, recovery
  uses it; no partial owner state is authoritative.
- **Rejoin:** new incarnation, current-fence checkpoint sync, empty queues, then
  READY for a later generation.  Old receipts and unfinished buffers cannot be
  resurrected.
- **Process/allocation death during checkpoint:** write a unique
  `(generation,fence,digest).tmp`, stream SHA while encoding, fsync file, rename
  immutable checkpoint, fsync directory, then atomically publish
  commit/checkpoint/latest under a current-fence CAS and fsync the control
  store.  Death before CAS leaves an unreferenced artifact for GC; death after
  CAS leaves a complete authoritative checkpoint.  A later allocation takes a
  strictly newer fence and verifies the complete digest before load.  Never
  advance `latest` to an incomplete file.

Checkpoint encoding, hashing, write, and fsync overlap K40.  The publisher may
hold only one staging checkpoint; if the prior checkpoint has not become
durable by the next required cadence, it backpressures result commitment rather
than overwriting staging or weakening cadence.  Small manifest construction can
proceed concurrently, but its final digest and CAS wait for the immutable file.

## Targets and implementation touch points

Acceptance for the pipelined implementation is measured steady state after one
warm-up generation:

- trainer foreground idle (handoff + boundary validation/apply + mandatory
  backpressure wait, excluding K40 compute) **<10%** of wall time at each rung;
- commit/apply cadence **≤1.25 × measured K40 maximum** whenever all background
  stages plus required checkpoint work fit inside that K40 window;
- zero stale/partial/corrupt applies, queue depth no more than the stated two
  delta slots and one result slot, and terminal retained/in-flight bytes zero;
- when background work does not fit, report the named limiting span and exert
  bounded backpressure—do not hide it in an enlarged deadline.

Primary touch points are `scripts/frontier/resilient_e97_role.py` (trainer loop,
handoff/apply lanes, digest calls, checkpoint creation and publication),
`src/native_resilient_dataplane/src/ndp.cpp` (streamed finite/SHA/reduction and
owner/result state), the native ABI/bridge beside those calls (generation-tagged
slot ownership and completion notification), and the resilient runtime's
`finalize_checkpoint`/fenced admission store (background publisher and CAS).
Tests should first model the state machine deterministically, then cover corrupt
and delayed slots, fence loss at every publication step, latest-only replacement,
lag backpressure, owner replay, rejoin, and death-before/after CAS before the 2n
live gate.

## Authority conformance and evidence

This decision preserves the compute-pool checklist: R01–R05 identity,
membership, exact weighted math and deterministic ownership; R06 bounded quorum
policy; R07 atomic immutable publication; R08 newer-fence recovery; R09
model-owning trainers/model-free services; R10 no dense Lustre hot path; R11–R13
bounded lifecycle/provider portability; R14 explicit stage evidence; R15
token-driven policy; and R16 auditable operations.  It also preserves native
requirements NDP01–NDP17, particularly fixed framed identities/checksums
(NDP06), bounded credits (NDP09), sealed memfd handoff (NDP10), deterministic
owner math (NDP11), result redistribution (NDP12), deadlines/failure containment
(NDP13), fenced checkpoint publication (NDP15), and telemetry/release evidence
(NDP16–NDP17).

Sources: `reports/validate-native-pool-v1-2n-startup-metrics.json`,
`reports/validate-native-pool-v1-4n-metrics.json`,
`reports/validate-native-pool-v1-8n-metrics.json`, their paired top-level and
Frontier reports, and retained job evidence/checksums.  The controlling designs
are `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

## Validation

- [x] GPU K40, native stages, integrity, apply, and publication are separated;
  absent spans are explicitly instrumented rather than estimated.
- [x] SHA/finite scans are classified at their correctness boundaries and
  redundant full-buffer rescans are identified.
- [x] Generation/fence/incarnation state, bounded two-delta/one-result memory,
  latest-only policy, and lag-one backpressure are explicit.
- [x] Late peers, missed quorum, delayed/corrupt results, node loss, rejoin, and
  process/allocation death during checkpoint are defined fail closed.
- [x] Targets are foreground idle below 10% and cadence at most 1.25× measured
  K40 when background work fits.
