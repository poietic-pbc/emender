# Eliminate shared SQLite from resilient E97 runtime

**Task:** `eliminate-shared-sqlite-runtime`
**Date:** 2026-07-25
**Scope:** implementation and deterministic qualification gates; no Slurm
submission
**Authorities:** `RESILIENT_DILOCO_COMPUTE_POOL.md`,
`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, ADR-002
`ASYNC_DECOUPLED_DILOCO_V2.md`, and the
`RESILIENT_DILOCO_GAP_MATRIX.md` R01–R16, NDP01–NDP17 and
V21S01–V21S17 crosswalk.

## Outcome

Every production compute-node role is independent of SQLite. The allocation
supervisor publishes one immutable claim keyed by the monotonic scheduler
fence before any model role starts. `PoolControlServer` is the in-memory
native peer-control authority for live membership, incarnation fencing,
generation closure, exact-once commit agreement, node-apply state, and
recovery handshakes. Complete checkpoints, manifests, allocation claims,
digest-linked commit receipts, and all-eight-trainer node-apply receipts are
the only durable restart authority.

No correctness or liveness decision reads a shared filesystem database,
mutable lease row, lock, heartbeat, or `latest.json`. The compatibility
pointer is written only after the immutable receipt exists. Historical
`ndm/fenced_admission.py` remains isolated for offline history/migration tests;
the rendered sbatch/supervisor/manager/trainer/native runtime closure does not
import it or construct its store.

## Causal trace

The introduction has three exact commits:

1. `f62898ac83bbf9f6e62d774c2c59fb313f16fcba`, 2026-07-17,
   `feat: add fenced Frontier admission lease
   (implement-fenced-frontier)`, created `SQLiteFencedControlStore` and its
   tests.
2. `c726c91c8f6bf270d189e769726b3ea5eaa07e19`, 2026-07-17,
   `feat: implement-fenced-frontier (agent-1170)`, is the squash commit with
   the identical tree
   `85633b9fdcef9395d5f2566fc7217a55d05ac8a8`.
3. `ae2e6f26046fb7a6b348e845fb4615092a7c37e0`, 2026-07-18,
   `feat: integrate resilient E97 compute pool runtime`, is the commit that
   actually wired the store into the resilient E97 allocation, role,
   checkpoint, and launcher path.

The original rationale was legitimate fail-closed work for R01/R07: acquire
an exclusive owner before model load, renew an expiring lease, reject an older
fence, and atomically publish checkpoint/commit/latest. The 2026-07-18
integration report explicitly described one current-fence SQLite transaction
and left its Frontier filesystem locking/durability as a downstream
qualification gap. The mistaken assumption was that a shared-filesystem
database could safely serve as both live control and durable commit CAS on
Frontier.

The production default was
`$RUN_DIR/control/pool-v1.sqlite3`. Qualification run directories are rooted
at `/lustre/orion/...`, so the database, its journal/locking protocol, repeated
constructor PRAGMAs, renewals, diagnostics, and per-role fence checks generated
Lustre metadata/lock traffic.

Subsequent commits expanded the dependency into restart, rejoin, asynchronous
boundary and atomic apply paths, including
`7e8490bec436aac045b0e265589e3783291c9e9c`,
`10c6fd418159d419bcdc4c92942d472f01e5401c`,
`8ca8b3327300bee36af709af5c40dc3ecde498a3`, and
`c5600a968b850412c42fbeea0d524f815ee643f3` (with their WG squash
counterparts). These were safety extensions of the original control-store
assumption, not independent reasons to retain SQLite.

### Former production call sites

The exact source snapshot used by job 5072235 was
`3b8a6530ac38f98b197b26d656037b3b53fd5afc`. Its complete SQLite surface was:

| Role/path | Former call |
|---|---|
| allocation admission | supervisor import; `_allocation_admission` constructed the default database, acquired a lease, exported `RESILIENT_E97_FENCE_DB`, and started the renewal guard |
| allocation liveness | `AllocationLeaseGuard` periodically renewed the shared row |
| diagnostics/deadlines/restart | supervisor `_durable_generation` reconstructed the store and called `assert_current`/authoritative latest during every deadline pass and cohort recovery |
| manager startup/rejoin | role `_fenced_control`, `_native_manager_resume_point`, and manager paths reconstructed the store and asserted the lease |
| trainer startup/steady apply | trainer `_fenced_control`, resume validation, generation loops, and `_reload_verified_async_v2_latest` asserted the store |
| checkpoint/commit | `ndm.resilient_e97_runtime.finalize_checkpoint` called `assert_current`, `publish_bundle`, and post-publication assertions |
| termination/restart | manager/trainer checkpoint/rejoin paths consulted the same store to authorize recovery |

The current replacements are `_allocation_admission` plus
`ManifestPeerAuthority.claim`, `PoolControlServer`/`PoolControlClient`
`recover`/`commit`/`node_apply`, `ManifestPeerAuthority.publish_checkpoint`,
and immutable receipt-chain recovery. Diagnostics resolve only the receipt
digests already acknowledged in manager peer state; they do not open or poll
the immutable authority tree. Non-publisher managers discover a commit by
bounded point-to-point `commit_state` RPC, then name the exact durable receipt
to their trainers through node-local control. Each role reads that named
receipt once for apply/reload validation.

## Retained job 5072235 evidence

The checked-in evidence summary is
`reports/frontier/job-5072235-generation-8-sqlite-failure.json`. The retained
external evidence proves:

- `Partition=batch`, `QOS=debug`, exactly two nodes;
- job 5072235 ran 51m46s and ended `FAILED`, exit `143:0`;
- immutable generation 8 was complete at step 2,301,250 with accepted-token
  clock 150,872,430,080, two-node membership, checkpoint SHA-256
  `882df659...18eb8d5`, and result root `f624cb17...50ac82b`;
- the shared database existed at
  `clean/clean-overlap/control/pool-v1.sqlite3` on Lustre;
- both `_deadline_reason -> _durable_generation` and trainer
  `_reload_verified_async_v2_latest` failed in
  `SQLiteFencedControlStore._connect` while executing
  `PRAGMA synchronous=FULL`, with
  `sqlite3.OperationalError: database is locked`; and
- the retained Slurm stderr SHA-256 is
  `d210bb9ee9525bb377203b6b62f4f91ea9200ee00bc60ad48cc7c4505e73c428`.

This was a shared-control-store failure after a valid generation-8
checkpoint, not a native transport, numerical, or checkpoint-integrity
failure.

## New authority and invariants

An allocation claim binds
`(run_id, allocation_id, allocation_incarnation, scheduler_fence,
protocol_id, config_id, base_generation, base_commit_digest,
previous_claim_digest)`. It is create-once, content-addressed, and accepted
only when its scheduler fence is strictly newer. An older allocation may
finish an already-open filesystem write, but its receipt cannot become an
ancestor of the newer claim.

A commit receipt binds the allocation claim/fence/incarnation, exact next
generation, accepted-token clock, outer step, manifest/checkpoint hashes,
result/previous-result roots, frozen-membership digest, and previous receipt
digest. Only one exact successor is accepted; identical retry is idempotent,
conflicting or discontinuous publication fails closed. Native peers validate
that exact generation/result/token/prior-receipt identity before their live
state advances or they acknowledge the commit.

A node-apply receipt binds the commit receipt and result root to one node
incarnation and eight independently hashed trainer apply/recovery receipts.
The native manager acknowledges that node receipt through peer control before
the incarnation may advertise next-generation READY. Partial apply restarts
the manager/service/all-eight-trainer cohort from the last immutable commit
under a new incarnation.

Fresh allocation recovery validates the receipt chain and complete manifest,
restores model, required outer state, outer step, accepted-token clock, result
roots, frozen membership, fence/incarnation provenance and apply receipts,
then installs that state through the peer `RECOVER` handshake. Node-local
unfinished work and old-incarnation messages are discarded. No database
bootstrap exists.

## Compute-pool conformance checklist

- [x] Cite this authority/version plus the matrix: this report maps R01–R16,
  NDP01–NDP17, and V21S01–V21S17 below.
- [x] Peer-owned READY membership uses bounded deadlines and never launched
  ranks or an all-rank operation.
- [x] Fenced identity, exact weighted math, idempotence, stale/conflict/corrupt
  rejection, exact-once peer commit and immutable evidence remain fail closed.
- [x] Native memfd/XPMEM and point-to-point transport remain bounded,
  release-accounted, non-Lustre, and without a central full-model broker.
- [x] Trainer, manager, service, partial apply, checkpoint publication and
  fresh-allocation recovery paths are deterministically exercised at the
  exact two-node floor `Q_min=2`, `T_min=3,934,080`.
- [x] The rendered compute-role closure rejects SQLite imports/connections,
  store construction, database paths, filesystem locks, and metadata
  heartbeats.
- [x] No Slurm job was submitted, changed, cancelled, or otherwise operated
  by this implementation task.

## R01–R16 map

| ID | No-database conformance |
|---|---|
| R01 | Immutable strictly increasing scheduler-fence claim is published before roles; stale/conflicting claimant exits zero-work. |
| R02 | `PeerMembership` remains the volatile DISCOVER/BOOT/SYNC/READY/DRAIN/EXPIRE authority with stable worker and fresh incarnation. |
| R03 | Generation admission snapshots live peer READY membership, never allocation size or launched ranks. |
| R04 | Full run/fence/generation/attempt/worker/incarnation/sequence identities retain stale rejection and idempotent replay. |
| R05 | Deterministic exact-token binary64 native reduction and one final projection are unchanged. |
| R06 | Explicit Q/T/deadline closure and fail-closed quorum collapse are unchanged. |
| R07 | Peer exact-once commit agreement plus digest-linked immutable manifest/checkpoint receipts replaces database CAS. |
| R08 | Deterministic sharded owners, bounded replay/credits/checksums/release and no full-model broker are unchanged. |
| R09 | Managers remain model-free; trainers alone own model/optimizer; unfinished work remains disposable. |
| R10 | Live control/heartbeats stay node-local or in peer memory; shared writes are immutable restart evidence only, never a database. |
| R11 | New-incarnation rejoin uses peer recovery from the exact receipt; stale incarnation/fence messages are rejected. |
| R12 | Manifest/receipt restores model, outer state/step, token clock, result roots, fence/incarnation and apply receipts. |
| R13 | Peer protocol remains scheduler/MPI independent; only the claim adapter reads Slurm identity. |
| R14 | READY/K40/exchange/first-commit/recovery deadlines and honest committed evidence are preserved. |
| R15 | Frozen exact-token weighting and accepted-token clock remain deterministic and reference tested. |
| R16 | Implementation adds no scale authorization; exact two-node gates still precede every 4+ rung. |

## NDP01–NDP17 map

| ID | No-database conformance |
|---|---|
| NDP01 | Native peer control owns live membership/fencing/incarnation/generation/commit/recovery; Python only adapts scheduler/checkpoint policy and C++ owns dense state. |
| NDP02 | Peer control and data plane use bounded point-to-point operations, never failure-sensitive all-rank operations. |
| NDP03 | Persistent C++17 `FI_EP_RDM`/exact-`cxi` service binding is unchanged. |
| NDP04 | Producer-direct XPMEM/memfd handoff and zero trainer-sized shared write are unchanged. |
| NDP05 | Fixed deterministic layout/conversion/weight/rounding/reference math is unchanged. |
| NDP06 | Claims, peer commands, frames, receipts, results and checkpoints carry exact fence/incarnation identities. |
| NDP07 | Opaque endpoint exchange remains peer membership metadata; no filesystem polling or native all-gather. |
| NDP08 | Pre-registered finite buffers and resident/credit/mailbox bounds are unchanged. |
| NDP09 | Receiver credit remains distinct from transport completion; commit receipts are compact metadata. |
| NDP10 | CRC32C/SHA-256, once-only owner apply and conflict rejection remain mandatory. |
| NDP11 | Sender-memory replay and bounded reassignment remain unchanged; SQLite is not a replay source. |
| NDP12 | Owner-direct redistribution produces one shared node aggregate; all-eight apply receipts attest it. |
| NDP13 | Absolute deadlines cover peer recovery/commit/apply; publication failure cannot advance READY. |
| NDP14 | Stable metadata-only local ABI/service path remains unchanged; no database symbol crosses it. |
| NDP15 | Peer agrees commit and all-eight node apply; Python publishes immutable state; drain stays collective-free. |
| NDP16 | Manifests/receipts bind provider/build/config/result/token/apply identities and bounded telemetry. |
| NDP17 | Native build/10-CTest gate and ordered two-node-before-scale admission remain required. |

## V21S01–V21S17 map

| ID | No-database conformance |
|---|---|
| V21S01 | Canonical v2.1 policy/wire/checkpoint/receipt identities remain fail closed; historical v2.0 is never relabeled. |
| V21S02 | Commit/applied/result/speculative clocks stay distinct and bounded; peer commit state seeds recovery. |
| V21S03 | Exact tokens remain the sole quorum, clock, numerator and denominator. |
| V21S04 | Exact K40 and stateless `eta_outer=1.0` outer update remain unchanged and receipt-bound. |
| V21S05 | Full fenced worker/incarnation/window/base/policy/layout/code/token identity and exact two-node floors remain pinned. |
| V21S06 | One immutable owned cohort plus one mutable interval and `OWNED` responsibility remain bounded. |
| V21S07 | ScheduleFree `x/z` correction stays exact once; receipt/result identity prevents duplicate correction after restart. |
| V21S08 | Mailbox admits only reload-verified peer-committed receipt lineage; compatibility `latest.json` cannot authorize apply. |
| V21S09 | Resident/credit/replay/receipt/mailbox/deadline bounds remain finite; no dense/database spill is introduced. |
| V21S10 | Peer READY membership, expiry, new incarnation and stale rejection preserve the two-node floor. |
| V21S11 | All-eight trainer apply is one receipt-bound node transaction; partial apply reconstructs the cohort under a new incarnation. |
| V21S12 | Persistent native point-to-point CXI path, deterministic reduction and no broker/MPI/Python-dense/Lustre-dense rules remain unchanged. |
| V21S13 | All lag/stage/token/cohort/apply/checkpoint evidence remains honest; receipt/checkpoint latency is separate from training overlap. |
| V21S14 | Immutable bundle/receipt restores model, outer/token/result/fence/apply state on a newer claim through peer recovery. |
| V21S15 | Exactly two-node numerical, clean, fault/restart, replay and convergence gates remain unclaimed until rerun; no job was submitted here. |
| V21S16 | No scale promotion or predecessor evidence was created or inferred. |
| V21S17 | Finite leased-READY scale closure remains authorization/evidence derived, never launched-rank or unexplained constant. |

## Validation

All Python/native work used the canonical environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

### Deterministic no-SQLite and runtime suites

The focused runtime/supervisor/peer suite:

```bash
"$EMENDER_PYTHON" -m pytest -q --basetemp=/tmp/emender-esr-1581-focused \
  tests/test_manifest_peer_authority.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py
```

Result: **119 passed in 118.67 seconds**. This includes fatal
`sqlite3.connect` poisoning and source-closure checks plus admission, steady
generation, live diagnostics, supervisor/manager/trainer loss, stale
allocation and incarnation, conflicting/exact-once publication, publication
failure, all-eight apply, checkpoint verification, and fresh-allocation
recovery.

The wider async v2.1/controller/checkpoint/native/failure/policy/performance
selection contains 319 tests. To avoid the known resource-contention
sensitivity of the 20-process fresh-restart fixture, the deterministic final
pass was split:

- **318 passed, 1 deselected in 188.81 seconds** for the full selection with
  only `test_fresh_process_restart_matches_uninterrupted_continuation`
  deselected; and
- **1 passed in 85.19 seconds** for that exact real fresh-process restart
  fixture in isolation.

An earlier unsplit run before the final peer-discovery tightening passed all
319 tests in 199.59 seconds. The final unsplit attempt passed 318 tests but
the same 20-process fixture exceeded its 15-second aggregate deadline under
concurrent load; its immediate isolated rerun above passed. No assertion,
protocol, checksum, or implementation failure was hidden.

### Native build and CTests

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: **10/10 CTests passed**. The exact bundle remains
`f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441`;
the build installs `libemender_ndp.so.1`,
`libemender_ndp_transport.so.1`, and `ndp_cxi_service`. The build manifest is
regenerated from the final clean commit before publication.

### Render and runtime/source closure

A two-node clean controller render used `--dry-run` only. It produced a
`Nodes=2`, `Partition=batch`, `QOS=debug` plan and the canonical
`resilient_e97_true_2n.sbatch` entry point; no `sbatch` command was executed.
After the final clean commit, that render is repeated with its exact native
bundle digest and the exact-source build manifest is validated separately.
The previous G2 attestation correctly cannot attest a new source commit; the
downstream qualification task must produce the new exact-source G2 evidence.
This implementation task neither relabels that older evidence nor submits a
replacement Slurm job.

The audit command:

```bash
rg -n --hidden --glob '!build/**' --glob '!.git/**' \
  '(sqlite3|SQLiteFencedControlStore|RESILIENT_E97_FENCE_DB|pool-v1\.sqlite|\.sqlite3|\.sqlite\b)' \
  ndm scripts configs native
```

finds only `ndm/fenced_admission.py`, the explicitly isolated historical
offline record implementation. No file in the rendered launcher,
supervisor, role, checkpoint, peer-control, native service, configuration,
diagnostic, generation, apply, or restart closure imports or constructs it.
The production closure also contains no `.wait_for_generation(` call:
steady commit discovery is peer RPC plus node-local handoff, not shared
manifest-directory polling.

`git diff --check`, `compileall`, both evidence-file JSON parses, the
checksum-linked native reference test, and the deterministic render audit
all pass. The final authoritative `origin/main` SHA and its equality with
local `HEAD`, fetched `origin/main`, and `git ls-remote` are retained in the
WG task log because a Git commit cannot embed its own SHA without changing
that SHA.
