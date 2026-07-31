# Native resilient pool v1 two-node failure acceptance

Date: 2026-07-19

WG task: `validate-native-pool-v1-2n-failures`

Live source commit: `f56e27a53c9985e95c4ab7d0fcfa28b8a2c513e6`

Live progress job: `5033384`

Live owner-abort job: `5033405`

Exact-source full-layout prerequisite: G2 job `5033380`

## Result

**Passed.** The accepted progress run completed three atomic real-E97 K40
generations on exactly two Frontier nodes after injecting one trainer loss and
one model-free manager/owner loss. The manager returned with a new incarnation,
synchronized the latest immutable commit, remained deliberately absent from
READY for 30 seconds, and joined both later generations. The committed token
clock advanced exactly
`5,245,440 -> 10,490,880 -> 15,736,320`; all 48 trainer applications are unique
and accounted for.

A separate exactly-two-node run terminated the node-1 persistent native service
during the following generation's `owner_transport`. It detected the lost
service in 19.015 seconds and completed fenced TERM handoff in 37.685 seconds.
That run intentionally exited nonzero. Its authoritative state remained the
previous generation-1 commit at 5,245,440 tokens: there is no next native
result, apply receipt, checkpoint, handoff manifest, or SQLite publication row.
Both services' final transport records have zero in-flight and retained bytes.

No four-or-more-node job was submitted. The machine-readable result is
[`validate-native-pool-v1-2n-failures-metrics.json`](validate-native-pool-v1-2n-failures-metrics.json).
Compact, independently checksummed evidence is under
[`frontier/evidence/job-5033384`](frontier/evidence/job-5033384),
[`frontier/evidence/job-5033405`](frontier/evidence/job-5033405), and
[`frontier/native-dataplane/5033380`](frontier/native-dataplane/5033380).
The four 7.90-GB immutable checkpoints remain at their recorded run paths and
are not duplicated into git; their byte counts and SHA-256 digests were
independently streamed and recorded in each evidence bundle.

## Scope and immutable launch

The launch recipe is
[`validate-native-pool-v1-2n-failures-submit-20260719T201710Z.sh`](frontier/validate-native-pool-v1-2n-failures-submit-20260719T201710Z.sh).
Before either `sbatch`, it required:

- local `HEAD` and the pushed task branch to equal
  `f56e27a53c9985e95c4ab7d0fcfa28b8a2c513e6`, with no tracked source or index
  differences;
- the approved Frontier Python 3.12 environment;
- the pinned step-1525000 seed SHA-256
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`;
- the pinned tokenizer SHA-256
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`;
- exact-source G2 `5033380`, attested native bundle
  `7b4c77696011cfc2b68fce541a34f9a1072ef767e13cce94ed2a4e2befc57d04`,
  production `native-cxi`, exact provider `cxi`, full E97 layout, and `job_vni`;
- an empty user Slurm queue, a new run directory, exactly `-N 2`, eight GPUs
  per node, a 30-minute debug-QoS limit, global quorum 2, and no fixed-world
  collective.

Both live runs used 16 real GPU trainers, two model-free Python managers, and
two persistent C++ native services. The configured global floor was
`Q_min=2`, `T_min=3,934,080`; the local native floor was all eight trainers per
participating node. Dense local reductions, owner exchange, redistribution,
and result views remained service-owned memfd/CXI operations. Python carried
membership, identities, receipts, and atomic publication policy only.

| Run | Slurm result | Nodes | Elapsed | Purpose |
|---|---|---:|---:|---|
| G2 `5033380` | `COMPLETED 0:0` | 2, `frontier[06089,06092]` | 2:49 | Three-generation exact full-layout CXI prerequisite |
| progress `5033384` | `COMPLETED 0:0` | 2, `frontier[06092,06105]` | 17:29 | Trainer loss, manager loss, delayed new-incarnation rejoin, three commits |
| owner-abort `5033405` | expected `FAILED 1:0` | 2, `frontier[03568,03570]` | 10:20 | Persistent native-service loss at owner transport, deterministic abort |

## Trainer and manager failure with continued progress

Both managers became READY before the 16 trainers started. No injection fired
before that startup gate. Generation 0 then froze exactly two node
contributions totaling 5,245,440 tokens and atomically published generation 1.

Only after the generation-1 handoff was durable, the supervisor killed
`node-0-trainer-3` with SIGKILL at its `applied` stage. It restarted that one
trainer exactly once. Global manager membership remained at the configured
`Q_min=2`, the trainer reconstructed its state from the committed handoff, and
the run did not wait for or apply unfinished pre-failure work. The replacement
trainer supplied exactly one rank-3 receipt in each later generation.

The supervisor next evicted `node-1-manager` after its generation-1
`published` heartbeat. The old manager exited zero and detached only its own
client/endpoint; it did not drain the persistent native service. The stable
worker `node-1` changed incarnation from
`bfeb1c7d9cdc4e2baabeb3f9971b0613` to
`6daf33b239474e9cbaee28d133111bbb`. The replacement manager recorded:

- `status=synchronized`, generation 1, fence 1;
- exact handoff SHA-256
  `a75c44bb1159621d0edb66d99c192c1e434c91188549ca05934be3851a8df411`;
- a completed 30-second delayed-READY marker for the new incarnation;
- generation-1 and generation-2 frozen contribution identities under the new
  incarnation, which became durable result generations 2 and 3.

The old incarnation's competing delay marker is `cancelled`, not completed.
The generation admission records therefore show old node-1 incarnation only
in generation 0, and the new incarnation in generations 1 and 2. No old work
was resurrected and no stale incarnation re-entered READY.

There were exactly two nonzero `restart` starts in the entire successful run:
the intended trainer and intended manager. There was no restart exhaustion,
unplanned eviction, or nonzero terminal completion. Sixteen trainers and both
managers completed zero; both services then exited zero under
`allocation_complete`.

## Three exact atomic generations

| Result generation | Step | Frozen global weight | Cumulative accepted tokens | Manifest SHA-256 | Checkpoint SHA-256 |
|---:|---:|---:|---:|---|---|
| 1 | 1,525,040 | 5,245,440 | 5,245,440 | `a75c44bb1159621d0edb66d99c192c1e434c91188549ca05934be3851a8df411` | `49e3380f0e23d636067342fe293e719855ebd95f876cf03c9f7a00d7abc9b6b6` |
| 2 | 1,525,080 | 5,245,440 | 10,490,880 | `5fa307a540cf6a2c4a0f1289bc9f11219adba5ce12e5923e5c82afbf9f9c3bb6` | `fa08e6fbb0dd0f7a011cdc7ead7fba7c14ab47bdd8e5ba3432d3b71a913f534c` |
| 3 | 1,525,120 | 5,245,440 | 15,736,320 | `1b7b06d7af106fbb86305d1bda6ee7e5a7f52a1ecbd9d48e230b3446843eb8cd` | `03a51ea4e065b902c825ad78d987acaff32f0f5b52ed6f713b05a16b445fd753` |

Each manifest is finalized under fence 1 and generation identity `attempt=0`.
Native result `attempt=2` is the specified second arithmetic stage combining
the two preweighted f64 node numerators; it is not a repeated generation.
Every generation records global membership `node-0,node-1`, a distinct result
root, the same exact 5,245,440 weight, the pinned native-runtime digests, and a
5,506,770,496-byte projected result.

The retained evidence has exactly:

- 48 sealed native submissions: 3 generations x 2 nodes x 8 trainers;
- six identical-per-generation node result markers;
- 48 ordered result-lane release markers;
- 48 durable apply receipts, ranks 0 through 7 exactly once on each node in
  every generation;
- three one-line pool close records, each with two distinct frozen worker /
  incarnation / contribution-sequence identities and `accepted_tokens=5245440`;
- three immutable manifests and three immutable checkpoints;
- three SQLite `commit` rows, three linked `checkpoint` rows, and one
  authoritative `latest` row pointing at generation 3.

SQLite `PRAGMA integrity_check` returned `ok`. A separate post-job process
recomputed every 7,899,873,331-byte checkpoint SHA-256 and the final manifest
SHA-256. All match their immutable manifests, compatibility latest pointer,
and authoritative SQLite transaction. These counts and identities rule out
duplicate application or cumulative-token double counting.

## Native-service owner loss and fail-closed publication

The owner-abort run first committed generation 1 at exactly 5,245,440 tokens.
During the next input generation, both node contributions reached the
control-plane admission floor. At node 1's `owner_transport` heartbeat, the
supervisor sent TERM to `node-1-native-service`; the service exited zero under
`injected_native_service_stage`. Nineteen seconds after injection, the node-1
manager and trainers were fenced as `native_service_lost`. With zero permitted
restarts and the global floor no longer safe, the other node performed
`allocation_term_handoff`, and the expected nonzero Slurm job ended 37.685
seconds after injection.

The retained generation-1 pool record has `status=commit_ready` because the
metadata contribution floor had frozen before service loss. That is an
admission receipt, not an atomic model publication. The authoritative proof of
the abort is:

- compatibility latest remains generation 1 / fence 1 / 5,245,440 tokens;
- exactly one immutable checkpoint and one handoff manifest exist;
- SQLite contains only that generation's `commit`, `checkpoint`, and
  `authoritative latest` rows;
- interrupted input generation 1 has no native result marker, no durable apply
  receipt, no new checkpoint, no result-generation-2 handoff, and no
  publication row;
- the prior checkpoint's independently recomputed digest is
  `657856c595a30577f8d91229a285a0fd1bff419f9b5fe444abe7b0d6132c868c`.

Thus the owner loss produced a bounded deterministic abort with idempotent
frozen contribution identities and no partial publication or corruption. A
later task may restart from the valid generation-1 fence handoff; this task did
not broaden scope into that downstream fresh-allocation test.

## Buffer, endpoint, and TERM release

Across the successful run, node 0 moved exactly 33,040,622,976 useful bytes in
each direction (three owner generations). Node 1's old and new manager
incarnations moved 11,013,540,992 plus 22,027,081,984 useful bytes in each
direction, also exactly three generations. All manager-incarnation shutdown
records have:

- `replay_bytes=0`, `route_errors=0`;
- `in_flight_bytes=0`, `retained_bytes=0`;
- owner state `DRAINING` after release.

Each endpoint reports two terminal CQ completions with status `-12` and
provider errno 125 (`ECANCELED`). They occur only while DRAINING, after useful
payload release, with zero useful/wire bytes. They are bounded cancellation of
outstanding receive completions, not payload failure or replay.

The manager-loss path proves that `allocation_term_handoff` tears down only
the disposable manager endpoint: the persistent node-1 service accepted the
replacement incarnation and completed two more generations. At final success,
both services were TERM-drained only after all roles completed.

In the owner-abort run, both persistent service transport logs end with
`event=shutdown`, exact `cxi/cxi0/FI_EP_RDM`, `in_flight_bytes=0`,
`retained_bytes=0`, `replay_bytes=0`, `route_errors=0`, and owner state
`DRAINING`. The injected node-1 service and the node-0 handoff service both
exit zero. The authoritative generation-1 latest/SQLite/checkpoint agreement
makes that TERM sequence a valid fenced handoff.

## Exact-source prerequisite

Canonical RelWithDebInfo build and install passed all 10 native CTests. The
selected current resilient/native Python suite passed 93/93 in 133.02 seconds.

G2 `5033380` used the same source commit and bundle on exactly two production
CXI `FI_EP_RDM` endpoints. Its three timed full-E97-layout generations all
matched the independent analytical f64 reference. Median time was 22.6903
seconds, 4.361x the pinned Python baseline and above the required 4x gate. It
reported zero MPI collectives, all-rank barriers, Python dense socket bytes,
trainer spool bytes, disk replay bytes, CQ errors, route errors, and
post-release transport bytes.

## Focused fixes found by the live ladder

Every implementation commit below was pushed before the final exact-source
build, G2 gate, and accepted jobs:

| Commit | Correction |
|---|---|
| `7e8490be` | Add exact generation/stage failure controls, delayed READY evidence, and authoritative new-incarnation synchronization. |
| `b4cea8c2` | Release the ordered native result-view lane before slow durable recovery I/O. |
| `d3c380e3` | Bound the aggregate wait for eight durable apply receipts by the existing exchange/commit deadline rather than one reader APPLY deadline. |
| `bdaf166b` | Detach a lost manager client/endpoint without draining the persistent node service needed by its replacement. |
| `8295b748` | Wait through a stale atomic latest pointer until the next generation is published, instead of treating the normal transition as a fatal identity mismatch. |
| `f56e27a5` | Lease the native endpoint identity across the finite configured generation sequence while leaving every per-stage SLO and 30-minute allocation cap unchanged. |

Each of the last two live-discovered races first received a failing regression
test, then passed after its fix. Superseded jobs remained fail-closed at their
last complete immutable generation and are not used as acceptance evidence.

## Normative conformance

This validation uses *Resilient DiLoCo Compute Pool*, version 1, as the design
authority and the companion gap matrix requirement IDs **R01-R16** and
**NDP01-NDP17**.

- **READY membership and bounded waits (R01-R03, R06, R11, R13-R14; NDP02,
  NDP03, NDP07, NDP13):** the active world is the two live leased manager
  incarnations, not launched ranks. Global floor is explicitly `Q_min=2`,
  `T_min=3,934,080`; all jobs are bounded at 30 minutes. The new incarnation
  synchronizes before delayed READY. Owner loss below the safe floor aborts.
- **Identity, arithmetic, and atomic evidence (R04-R07, R12, R15; NDP05,
  NDP06, NDP10, NDP15):** all generation/fence/incarnation/sequences are
  recorded, all three weights and cumulative clocks are exact, receipts are
  once-only, and manifests/checkpoints/SQLite/latest agree. Owner loss exposes
  no partial publication.
- **Native bounded hot path (R08-R10; NDP01, NDP04, NDP08-NDP12, NDP14,
  NDP16):** compiled service memfds and point-to-point CXI carry dense state;
  Python and shared storage carry bounded metadata plus the immutable
  checkpoint. There is no central full-model broker, Python dense TCP, global
  MPI collective, trainer dense spool, or disk replay. Endpoint/buffer counters
  are zero after release.
- **Sequential gate (R16; NDP17):** exact-source G2 passed before the real
  failure runs. Every submitted live/gate job used exactly two nodes. No 4+
  allocation was requested.

## Validation

- [x] Trainer loss does not block later commits while the node/global policy
  floors remain satisfied: generations 2 and 3 commit after one rank-3 loss and
  restart.
- [x] Manager loss reassigns the node-1 owner to a synchronized new
  incarnation without duplicated receipts; native-service owner loss produces
  a bounded deterministic abort with no partial publication.
- [x] The returning peer synchronizes generation 1, uses a new incarnation,
  completes a 30-second delayed READY, and contributes to later result
  generations 2 and 3.
- [x] Three atomic generations commit at exact token clocks 5,245,440,
  10,490,880, and 15,736,320, with 48/48 unique trainer applies.
- [x] Native endpoints and buffers release to zero; successful terminal drain
  and expected abort TERM both leave a valid fenced handoff.
- [x] The conformance checklist is satisfied for **R01, R02, R03, R04, R05,
  R06, R07, R08, R09, R10, R11, R12, R13, R14, R15, R16** and **NDP01, NDP02,
  NDP03, NDP04, NDP05, NDP06, NDP07, NDP08, NDP09, NDP10, NDP11, NDP12,
  NDP13, NDP14, NDP15, NDP16, NDP17**.
- [x] Exact evidence and all fixes are committed/pushed; no 4+ node job was
  submitted.

Exact validation commands, after the canonical Frontier activation, were:

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" scripts/frontier/build_native_resilient_dataplane.sh
env -u NDP_BUILD_MANIFEST -u NDP_SERVICE_BINARY -u NDP_LIBRARY \
  "$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_pool_integration.py
bash -n reports/frontier/validate-native-pool-v1-2n-failures-submit-20260719T201710Z.sh
reports/frontier/validate-native-pool-v1-2n-failures-submit-20260719T201710Z.sh progress
reports/frontier/validate-native-pool-v1-2n-failures-submit-20260719T201710Z.sh owner-abort
sqlite3 RUN/control/pool-v1.sqlite3 'pragma integrity_check;'
sha256sum -c reports/frontier/native-dataplane/5033380/SHA256SUMS
sha256sum -c reports/frontier/evidence/job-5033384/SHA256SUMS
sha256sum -c reports/frontier/evidence/job-5033405/SHA256SUMS
```

The G2 `submission.json` preserves its exact branch-safe `sbatch` arguments.
The two run evidence bundles preserve Slurm accounting, runtime identity,
launch attestation, full supervisor events, pool records, native receipts,
telemetry, SQLite control state, manifests, independent audits, and digest
verification. Their `SHA256SUMS` files cover every included byte.
