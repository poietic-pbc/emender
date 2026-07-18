# Resilient DiLoCo Compute Pool v1 — two-node Frontier validation

Date: 2026-07-18

Task: `validate-resilient-pool-v1-2n`

Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1, and companion
requirements R01–R16 in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

## Status

The startup rung has failed closed five times without an unchanged retry. Job
`5028225` failed at the post-K40 local-delta streaming stage. Changed payload
r2 job `5028347` then proved the 64 MiB/two-file local spool fix: both managers
were READY within 61 seconds of allocation start, all 16 real trainers finished
K=40 within 134 seconds of child start, every local contribution published in
139–145 seconds, both managers exactly reduced six trainers and froze two node
contributions carrying 3,934,080 tokens. It then failed in distributed owner
transport because the trainers' 180-second aggregate-apply timers had started
at their own local publication, about 72 seconds before the managers opened the
fenced exchange window. The remaining 1 MiB owner frame also required about
40,000 short-lived request/response connections per E97 generation. No atomic
publication, checkpoint, handoff, or production mutation occurred.

Changed payload r3 job `5028483` proved the aligned trainer/manager exchange
window: all 16 trainers remained live through K=40, two-file local publication,
exact local reduction, and the two-node/3,934,080-token deterministic freeze.
It failed closed during the first distributed owner transfer. The client kept
the former one-second socket timeout after connecting, so each 64 MiB frame was
closed partway through transfer or receiver reduction; the owner observed
repeated partial-frame EOFs and the idempotent sender exhausted the bounded
180-second owner deadline. There was no atomic publication or checkpoint.

Changed payload r4 job `5028638` proved that separating the established-stream
timeout from the one-second connection bound was necessary but not sufficient.
READY completed within 63 seconds of allocation start; all 16 real trainers
finished K=40 within 132 seconds of child start and published exactly two local
spool files each; both managers reduced six trainers and froze the same exact
two-node/3,934,080-token set by allocation +395 seconds. The owner sender then
again exhausted its bounded window, and the remote owner retained 20 partial
64 MiB frames. The live client used an unbuffered `socket.makefile()` stream,
whose raw `write()` may legally accept only part of a large byte string;
`send_frame` ignored that returned byte count and started reading a receipt
after a truncated payload. This is a framing correctness bug, not evidence for
an unbounded timeout increase. The fenced store still has zero publications.

Changed payload r5 job `5028767` proved the exact-write fix and completed the
entire distributed owner plane: 11,013,540,992 bytes sent plus the same amount
redistributed in 98.961 seconds (222.6 MB/s), with two owners, zero replay,
11,013,540,992-byte high-water, and full sender release. It then failed closed
at checkpoint proposal. Node 0's designated trainer was still writing and
hashing a redundant node-local 7.7 GB recovery checkpoint before beginning the
authoritative checkpoint/proposal; node 0's manager exhausted the unchanged
180-second exchange/commit deadline while waiting. The focused follow-up moves
the one complete leader checkpoint and proposal ahead of disposable local
recovery and reuses that immutable file for leader recovery. No deadline is
extended and no second full leader serialization is needed.

The ladder therefore remains on rung 1. No resilience or fresh-restart job has
been rendered or submitted. Changed startup payload r5 retained the one-second
connection bound, bounded 64 MiB frames, and unchanged 180-second overall
exchange deadline, but makes every framed segment write exact by advancing a
memoryview until all accepted bytes are sent. A regression using a deliberately
partial raw writer failed on the r4 tree and passes with the focused fix. It is
not an unchanged retry. The next payload will differ by the checkpoint-critical
path ordering described above and will not be rendered until the full pinned
gate passes.

No production allocation, normal-QoS allocation, 4+ node allocation, or
two-hour allocation is authorized by this report.

## Startup job 5028225 result

Exact Slurm state and accounting:

| Field | Observed |
|---|---|
| submit / eligible | `2026-07-18T10:40:10-04:00` |
| start | `2026-07-18T10:55:43-04:00` (`2026-07-18T14:55:43Z`) |
| queue time | `00:15:33` (recorded separately from runtime) |
| end / runtime | `2026-07-18T11:09:11-04:00`, `00:13:28` |
| allocation | exactly 2 nodes, 16 GPUs, debug QoS, `00:20:00` limit |
| state / exit | `FAILED`, allocation `137:0`, step `5028225.1` `1:0` |
| Slurm totals | `TotalCPU=18:58:31`, `energy=1356707`; step raw accounting reports `AveDiskRead=144840.53M`, `AveDiskWrite=7651.52M` |

The allocation lease was acquired before role/model load at Unix timestamp
`1784386573.075951` with fence 1 and renewed throughout the attempt. Manager
READY records occurred at `1784386602.112662` and `1784386607.406854`, or
59.113 and 64.407 seconds after allocation start. All original trainer children
started at `1784386586.643..1784386591.707`; their first K40 completions were
`1784386715.836..1784386724.547`, 129.140–134.317 seconds after child start and
172.836–181.547 seconds after allocation start.

Every original trainer then entered `streaming_delta`. The 180-second stage
deadline was detected and bounded eviction began; serialized stop/event
recording puts the first eviction events 196.163–304.500 seconds after each
K40 completion. No local contribution manifest reached the manager before its
420-second collection deadline. Both managers exited with
`TimeoutError: local trainer quorum lost at aggregation deadline`; the
supervisor recorded 23 `progress_deadline` evictions, two manager `exit:1`
evictions, two `first_atomic_generation_deadline` evictions, four bounded
shutdown evictions, and one `restart_exhausted`. The fenced SQLite store has
zero publication rows and only fence epoch 1, so the failure produced no
authoritative result.

The exact post-failure evidence collector is
`reports/frontier/validate-resilient-pool-v1-2n-job-5028225-evidence.sh`, SHA-256
`72899cbbdeac214965a834cdfe4b237770503f62556a3a5aab2c68bade08999e`.
It ran as completed CPU-only step `5028225.2` at 11:05:15 EDT and copied only
JSON/JSONL supervision, phase telemetry, and compact control records—never
mailbox/tensor/checkpoint data—to `reports/frontier/evidence/job-5028225`.
That retained tree has 94 files and aggregate path-bound digest
`b235b50fc29a4601d6e78f272b4cd0f9f7c07b7f352136754272ac3bed85159a`.
The top-level stdout, stderr, events, and control-database hashes are,
respectively, `c7b8c2ee0144f9781b534f2e62186979c2c3a8bf2fd7dec165609e23d1788edb`,
`2a1b4d1d4885fb7cd7fd215a8867147c04e31d39fa04e230fc8fc144532a828a`,
`9662baa8fd7eb3b731f72610b0f24112412a67a83584abdfff79564c33beafdd`,
and `e508bd348fce66e365cf7a1cb20d0598d37844a2b54ba1e503a72659c53a2efa`.

### Focused fix

The failure was not slow K40 or missing READY. It was the local handoff after
K40: each real trainer divided the 1.3B-parameter delta according to the 1 MiB
network frame, causing thousands of finite-check, conversion, hash, and
cross-process ledger operations while eight trainers concurrently streamed one
large file each. In addition, the 32 GiB ledger was arithmetically too small:
six f32 trainer contributions plus the f64 exact local aggregate require about
42 GB simultaneously before trainer release.

The fix separates these bounds. Local trainer-to-manager records are bounded at
64 MiB and still append to exactly one data file plus one manifest per trainer;
the manager repacks its exact float64 reduction into independently bounded
1 MiB network owner chunks. The shared node-local spool ledger is hard-bounded
at 64 GiB, enough for the configured six-trainer floor and aggregate without
unbounded admission. A `local_delta_spool` timing/byte metric records the stage.
Finally, node supervisors atomically retain JSON/JSONL evidence after success or
failure while explicitly excluding mailbox data, tensor streams, recovery `.pt`
files, and caches.

The changed-tree pre-submit gate passed with the approved Python 3.12 runtime:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q -p no:cacheprovider tests/test_fenced_admission.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py tests/test_resilient_e97_reducer.py tests/test_resilient_shard_owner.py tests/test_resilient_pool_runtime.py tests/test_resilient_e97_split_roles.py tests/test_resilient_node_transport.py tests/test_resilient_e97_runtime.py tests/test_resilient_e97_true_2n_launcher.py tests/test_async_diloco_real_trainer.py tests/test_train_helpers.py
122 passed in 134.64s
```

Approved-Python `compileall`, `json.tool`, `git diff --check`, and the live-path
forbidden `.tolist()`/collective/MPI token scan also passed. New regression
tests first failed against the job-5028225 code and now prove the separated
local/network bounds and JSON-only post-role evidence retention. The local TCP
pool, exact weighted reducer, stale/duplicate/corrupt rejection, owner replay,
fencing/newer-allocation restart, and production real-trainer parity remain in
the passing suite.

### Changed startup payload r2

The focused runtime/evidence commit is
`2c337a83d8effac04c54b1805dbb9451460f2818`; it was pushed, fetched, and
verified equal to authoritative `origin/main` before rendering. The changed
payload identity is
`2c337a8-20260718T152955Z-pool-v1-startup-r2-2n20m-k40-local64m-ledger64g`.
Its exact retained command is:

```bash
bash reports/frontier/validate-resilient-pool-v1-2n-startup-r2-submit-20260718T152955Z.sh
```

The submit script SHA-256 is
`6aab23241a0fb67359dbbbc917fbc1a609b73aadfee44bb6000c1de93008b3ca`.
Changed live-file hashes are:

| Artifact | SHA-256 |
|---|---|
| two-node sbatch launcher | `807a0bdfde8fd9056bccd2b0e7e2d0315802f504f8762d511eb6cc28383585bf` |
| allocation supervisor | `fd1a194767b0442e9ae6298547966fe3fc6fa8240bb41ec1c46707a2e5fd5e0b` |
| live role entrypoint | `62a7721a6c41e437532c6f4eafe88609adec19216b0a9fa8ad6b6e363289ae2e` |

All other runtime/config hashes remain the values in the immutable-identity
table below. Payload r2 is still exactly 2 nodes, debug QoS, `00:20:00`, one
generation, two model-free managers, 16 real GPU trainers, batch size 4, K=40,
and node quorum 2 with no injection or resume handoff. It sets a bounded 64 GiB
node-local spool, 64 MiB local spool records, and unchanged 1 MiB network
frames. Startup retries are zero: a new stage failure ends the allocation
without repeating the unchanged role payload.

The retained command passed its authoritative-origin, empty-queue, absent-run-
directory, and exact-hash gates and returned job ID `5028347`. Slurm recorded
submit/eligible time `2026-07-18T15:32:41Z`, start time
`2026-07-18T15:33:25Z`, queue time 44 seconds, debug QoS, `00:20:00`, exactly
two nodes (`frontier06911,frontier08316`), and 16 allocated GPUs. Runtime
identity was attested at allocation start. The allocation failed closed at
`2026-07-18T15:41:47Z` after 502 runtime seconds with allocation/step exit
`1:0`; Slurm reports `TotalCPU=06:18:03` and raw energy `756077`.

Retained measurements are:

| Stage | Job 5028347 observation |
|---|---|
| manager READY | +56.613 / +60.363 seconds from allocation start |
| all trainer K40 | 131.390–133.639 seconds from child start; +173.095–176.670 seconds from allocation start |
| two-file local delta spool | 5,506,770,496 bytes per trainer in 139.037–144.102 seconds; completion at +312.132–320.219 seconds |
| exact six-trainer local reduce | managers completed at +373.178/+375.851 seconds; accepted 1,967,040 tokens each |
| deterministic global freeze | two node peers and 3,934,080 tokens at +387.026 seconds; `commit_ready` metadata, no atomic model publication |
| terminal stage | managers live in `owner_transport`; trainer aggregate deadline expired; no configured role retry |

The automatically retained JSON/JSONL/control/log tree is
`reports/frontier/evidence/job-5028347`: 114 files, path-bound SHA-256
`9bda0ac24cc62fc5f39815ce70548f5a36cbd95b4d608ce7e6f4455c571ea81e`.
Its manifest SHA-256 is
`923410328b65d0ed5cfac49d4a4826a17d584b854f64ab4e5bdce6b2d360ae3a`;
`sha256sum -c` passes. Top-level stdout, stderr, events, and fenced SQLite
hashes are `c7b8c2ee0144f9781b534f2e62186979c2c3a8bf2fd7dec165609e23d1788edb`,
`f0123c112c0f002333daa6da39e8e69643fcf38b8b5bbe4013e43c0b5f375c73`,
`17f2705785b5eec6fa5bdf0c1bd5f6e97abe34bd3d793e16752d0161a450ec15`,
and `5825d346e4a38dd64ac574acd9983f0ed9d5b401fb855e3e5c92a705a13507f9`.
The SQLite publication count is zero and its last fence is one.

### Focused owner-window fix after job 5028347

The trainer now enters a bounded `local_reduce_wait` after its two-file
publication. It observes only the node-local manager heartbeat and opens its
distinct 180-second aggregate-apply timer when that manager reaches `freeze`
or owner transport; a missing transition still fails at the existing
420-second generation deadline. This removes the premature timer without an
unbounded wait. The network owner frame changes from 1 MiB to a hard 64 MiB:
the measured E97 flat update has 165 rather than about 10,504 shards, while
checksums, one-frame RPC bounds, sender replay retention, deterministic owners,
and the 64 GiB high-water ledger remain enforced. No all-rank collective,
central full-model broker, Lustre hot path, or dense elementwise packing was
introduced.

The changed-tree gate passed with the pinned Python 3.12 / torch 2.10 ROCm 7.1
/ Triton 3.6 environment: the focused owner-window tests first failed against
job-5028347 code, then the pool/launcher slice passed 35 tests in 97.97 seconds.
An intervening exact-suite run exposed that the shared-login fixture reserved
only its coordinator port, not the two derived owner ports; the fixture now
verifies a free contiguous three-port block, its focused reproducer passed in
32.48 seconds, and the final exact pre-submit suite passed 123 tests in 111.68
seconds. Approved-Python `compileall`, metrics `json.tool`, `git diff --check`,
the live-path
`.tolist()`/collective/MPI ban, and the job-5028347 evidence manifest check also
passed. The user Frontier queue was empty after the gates.

### Changed startup payload r3

Focused code/evidence commit
`a63e456ac2ffc58d78055e6983471e7b0284bf74` was pushed to the task branch
and authoritative `origin/main`, fetched, and verified equal before rendering.
The exact retained command is:

```bash
bash reports/frontier/validate-resilient-pool-v1-2n-startup-r3-submit-20260718T161234Z.sh
```

The script SHA-256 is
`c3bb021d38d452fc80d11ed139126a13145e3933091319893e47ca77f76e696c`.
Its payload identity is
`a63e456-20260718T161234Z-pool-v1-startup-r3-2n20m-k40-windowed-owner64m-ledger64g`
and run identity is
`validate-resilient-pool-v1-2n-startup-r3-20260718T161234Z-a63e456`.
Changed live-file hashes are:

| Artifact | SHA-256 |
|---|---|
| two-node sbatch launcher | `6bad3f77e6a20834a13452c04de35299bd53f1d4304e4ab089a3524f3596d6ec` |
| allocation supervisor | `1893cdef5413945a0c49e6bc219dc0e9227eb5e510679429416851637b4c122e` |
| live role entrypoint | `c82025213f74ba85e6e4b1bf95f3f1a85cb0444ae921dda3fc37273366516833` |

Payload r3 remains exactly two nodes, debug QoS, `00:20:00`, two model-free
managers, 16 real GPU trainers, batch size 4, K=40, one generation, and node
quorum two. It has no injection or resume handoff, uses a 64 GiB hard spool
ledger with separate 64 MiB local and owner-network records, and configures
zero role restarts. The script refuses a nonempty user queue, existing run
directory, non-authoritative origin, changed live tree, or absent integrated-v1
ancestry. The exact command returned job ID `5028483`. Slurm recorded
submit/eligible time `2026-07-18T16:14:33Z`, start `2026-07-18T16:14:55Z`
after 22 seconds of queue time, exactly two nodes
(`frontier00639,frontier09084`), debug QoS, `00:20:00`, and 16 GPUs. It failed
at `2026-07-18T16:24:40Z` after 585 runtime seconds with allocation/step exit
`1:0`. Accounting reports `TotalCPU=06:19:17` and raw energy `848312`.

Retained measurements are:

| Stage | Job 5028483 observation |
|---|---|
| manager READY | +60.710 / +65.693 seconds from allocation start |
| all trainer K40 | 131.999–134.442 seconds from child start; +177.607–182.986 seconds from allocation start |
| two-file local delta spool | 5,506,770,496 bytes per trainer in 138.124–144.150 seconds; completion at +319.989–325.987 seconds |
| exact six-trainer local reduce | managers completed at +378.265/+381.696 seconds; accepted 1,967,040 tokens each |
| deterministic global freeze | two node peers and 3,934,080 tokens at +401.731 seconds; both managers within the 15-second freeze SLO |
| terminal stage | owner transport; node-0 manager failed at +561.507 seconds after repeated partial-frame EOF/replay; zero atomic publications |

The retained JSON/JSONL/control/log tree is
`reports/frontier/evidence/job-5028483`: 114 files, path-bound SHA-256
`aae31063e7a37c2085aa49dd0fc563a2c8a77f9f7a053076317429df56533372`.
Its manifest SHA-256 is
`7195225745a354eb1c3a35b8539b845a03d65fef7b6e549cc368e2a1da82cc72`;
`sha256sum -c` passes. Top-level stdout, stderr, events, and fenced SQLite
hashes are `c7b8c2ee0144f9781b534f2e62186979c2c3a8bf2fd7dec165609e23d1788edb`,
`007de7a382f004be6561c8355c588bf64def491346f828e30c7ac8d029e27865`,
`a49ce6e00ef5c535c72ee40a7437b0db9799254857e48293a1339f52617c32ac`,
and `50e07a3637711062b16e9f47e18cabb20c85d470c234b7011f7bec4ce17d3455`.
The SQLite publication count is zero and its last fence is one.

### Focused established-stream timeout fix after job 5028483

`ndm/resilient_pool_runtime.py::_owner_rpc` previously applied
`min(1.0, remaining)` both to connection establishment and every established
socket read/write. The owner logs prove that bound closed live peers in the
middle of their approved 64 MiB frames: each server received a header and a
payload prefix, then raised `EOFError: peer closed framed transport`; the
client retried the idempotent contribution until its overall deadline and
raised `shard owner transport deadline expired: timed out`.

The focused fix leaves connection attempts at one second and computes a
separate established-stream timeout from the larger of request and permitted
response bytes. It budgets an 8 MiB/s minimum transfer rate, so a 64 MiB frame
gets eight seconds, with a 15-second absolute per-operation ceiling and the
caller-provided remaining deadline as the final ceiling. Small control/ping
traffic remains at one second. The 180-second owner exchange deadline, 64 GiB
retention ledger, checksummed frame maximum, deterministic owners, replay,
prompt release, and fail-closed semantics are unchanged.

The regression first failed at import against the r3 tree, then passed both
the bound table and a real delayed TCP response that takes longer than the old
one-second established-stream allowance. The complete pool-runtime file passed
five tests. The full exact pre-submit suite then passed 125 tests in 114.42
seconds; static gates are repeated immediately before rendering any r4
payload.

### Changed startup payload r4

Focused code/evidence commit
`4b9def7cc9657d8648b67d1e385ea9bf161786fa` was pushed to the task branch
and authoritative `origin/main`, fetched, and verified equal before rendering.
The exact retained command is:

```bash
bash reports/frontier/validate-resilient-pool-v1-2n-startup-r4-submit-20260718T164450Z.sh
```

The script SHA-256 is
`f843058a58201bdc2017203b5976a4f3c213e3c6c05bed7ae7a142dd3ee6431f`.
Its payload identity is
`4b9def7-20260718T164450Z-pool-v1-startup-r4-2n20m-k40-owner64m-io8mps15s`
and run identity is
`validate-resilient-pool-v1-2n-startup-r4-20260718T164450Z-4b9def7`.
Immutable live-file hashes are:

| Artifact | SHA-256 |
|---|---|
| two-node sbatch launcher | `6bad3f77e6a20834a13452c04de35299bd53f1d4304e4ab089a3524f3596d6ec` |
| allocation supervisor | `1893cdef5413945a0c49e6bc219dc0e9227eb5e510679429416851637b4c122e` |
| live role entrypoint | `c82025213f74ba85e6e4b1bf95f3f1a85cb0444ae921dda3fc37273366516833` |
| pool owner/control runtime | `37de7776e5aee7d66e5e7b5737e6849dc5cd344cc283166beb88876a5f52264f` |

The script verifies those hashes itself before `sbatch`. Payload r4 remains
exactly two nodes, debug QoS, `00:20:00`, two model-free managers, 16 real GPU
trainers, batch size 4, K=40, one generation, node quorum two, 64 MiB local and
owner frames, and a bounded 64 GiB node-local ledger. It has no injection,
resume handoff, or configured role retry. It refuses a nonempty user queue,
existing run directory, changed live tree, non-authoritative origin, or absent
integrated-v1 ancestry. It differs from r3 only by the focused bounded
established-stream timeout fix and its regression/evidence commit.

The exact command was executed once and returned job ID `5028638`. Slurm
recorded submit/eligible time `2026-07-18T16:47:31Z`; the initial state was
`PENDING (Resources)` as the sole user job, with exactly two requested nodes,
debug QoS, `00:20:00`, 16 GPUs, no dependency, no requeue, and zero restarts.
Queue time is recorded separately from runtime.

### Startup r4 result and exact-write fix

Slurm started job `5028638` at `2026-07-18T16:47:49Z` after 18 seconds of
queue time on exactly `frontier00251` and `frontier01023`. It failed closed at
`2026-07-18T16:57:27Z` after 578 seconds with allocation and node-step exit
`1:0`; `TotalCPU=06:18:13`, raw energy was `847293`, and the role step reported
`MaxRSS=113479508K`, `MaxDiskRead=104113.44M`, and
`MaxDiskWrite=44396.08M`. Runtime identity and the two-manager/16-trainer,
batch-size-4, K=40, no-collective topology were attested before role load.

Observed bounded stages were:

| Stage | Job 5028638 observation |
|---|---|
| manager READY | allocation +59.655 / +62.531 seconds |
| all trainer K40 | 127.615–131.926 seconds from each child start; completed by allocation +178.369 seconds |
| two-file local delta spool | 5,506,770,496 bytes per trainer in 123.815–143.323 seconds; all complete by allocation +319.715 seconds |
| exact six-trainer local reduce | managers completed at allocation +373.068/+374.877 seconds; 1,967,040 tokens each |
| deterministic global freeze | two complete node contributions and 3,934,080 tokens at allocation +394.752/+394.758 seconds |
| terminal owner stage | node 0 exited at allocation +556.303 seconds after the remote owner retained 20 partial-frame EOFs; zero publications |

The automatically retained evidence is
`reports/frontier/evidence/job-5028638`: 114 files including only top-level
logs/control records and node-local JSON/JSONL snapshots; tensor mailbox,
`.data`, `.pt`, and kernel-cache payloads are excluded. Its `SHA256SUMS` digest
is `910de4f3bacf28ced6de1480fda7cf6dfe5a8b1f079e0a3bd89ac1102f2dd965`
and `sha256sum -c` passes. The stdout, stderr, fenced SQLite, and supervisor
event hashes are respectively
`c7b8c2ee0144f9781b534f2e62186979c2c3a8bf2fd7dec165609e23d1788edb`,
`7465e33fa7befa1ac16d4781bdcd1fd19eb09f955c12a4880e97f0766c575af4`,
`59caacc11da8934b74e52847dcb5fd2bd8bab5358351aa06090b405858d20f5f`,
and `616ed62657056719aeaa5ae42595435cab8a922ffea8118f97ef46fb6bd8f51d`.
The database has fence epoch 1 and zero publication rows.

The repeated EOF was caused by `send_frame`, which issued one `write()` per
segment and ignored its return value. That is incorrect for the raw
`SocketIO` returned by `socket.makefile(..., buffering=0)`: a successful write
may be partial. The focused change uses a bounded memoryview of the existing
segment and advances it until every byte is accepted, failing on zero progress.
It neither copies the whole model into a broker nor changes frame, spool,
deadline, or retention bounds. The new
`test_send_frame_retries_partial_raw_writes` first failed with the r4 code at
`EOFError: peer closed framed transport` and then passed after this fix. The
focused transport/pool slice passed 16 tests in 35.20 seconds. The first exact
suite run had 125 passes and exposed one unrelated healthy-quorum fixture whose
two-second server deadline was too narrow under shared-login scheduling load;
its standalone rerun passed. That fixture now uses a bounded 30-second healthy
allowance, while the independent 0.2-second quorum-deadline test remains
unchanged. The final pinned exact suite passed all 126 tests in 117.86 seconds;
approved-Python compileall, JSON validation, `git diff --check`, forbidden
live-path scanning, r4 `sha256sum -c`, and the empty-queue gate also passed.

### Changed startup payload r5

Focused code/evidence commit
`bfc2050b3f2e922f4c965aa3f3bd02688eaaeb92` was pushed to the task branch
and authoritative `origin/main`, fetched, and verified equal before rendering.
The exact retained command is:

```bash
bash reports/frontier/validate-resilient-pool-v1-2n-startup-r5-submit-20260718T171539Z.sh
```

The script SHA-256 is
`0c2c8e70c16c98bcb226f785d7633b9310a3d20841c6754223a2e2fdee5ad11c`.
Its run identity is
`validate-resilient-pool-v1-2n-startup-r5-20260718T171539Z-bfc2050` and its
payload identity is
`bfc2050-20260718T171539Z-pool-v1-startup-r5-2n20m-k40-owner64m-exactwrite`.
The script pins the launcher, supervisor, role, pool runtime, node transport,
and batch-size-4 configuration hashes to `6bad3f77…`, `1893cdef…`,
`c8202521…`, `37de7776…`, `e322c4c8…`, and `afc2a65f…`, respectively, and
checks authoritative ancestry, live-tree equality, absent run directory, and
an empty user queue before calling `sbatch`.

Payload r5 remains exactly two nodes, debug QoS, `00:20:00`, two model-free
managers, 16 real GPU trainers, batch size 4, K=40, one generation, node quorum
two, 64 MiB local/network frames, and a 64 GiB bounded node-local ledger. It
has no failure injection, resume handoff, or role retry. Relative to r4 it uses
the exact-write implementation and associated r4 evidence/tests; no walltime,
shape, model, optimizer, quorum, or deadline value changed.

The retained command was executed exactly once and returned job ID `5028767`.
Slurm recorded submit/eligible time `2026-07-18T17:18:18Z`, exactly two nodes,
debug QoS, `00:20:00`, 16 requested GPUs, no dependency, no requeue, and zero
restarts. Its initial state was `PENDING (Priority)` as the sole user job. It
started at `2026-07-18T17:19:00Z` after 42 seconds of queue time on exactly
`frontier05858` and `frontier05859`, then failed closed at
`2026-07-18T17:28:55Z` after 595 runtime seconds. The allocation exit was
`137:0` and the role-step exit was `1:0`. Queue time is separate from runtime;
no resilience or restart job was submitted.

### Startup r5 result and checkpoint-first fix

Runtime identity and the required two-manager/16-real-trainer topology were
attested before model load. Node 0 retained the following complete critical
path; the peer node step was terminated before it copied its node-local
snapshot, but its role logs and the global supervisor records are retained:

| Stage | Job 5028767 observation |
|---|---|
| manager READY | both roles READY; node 0 at allocation +58.352 seconds |
| node-0 trainer K40 | all eight at allocation +174.403–176.262 seconds |
| node-0 two-file local spool | 5,506,770,496 bytes per trainer in 140.228–143.897 seconds; complete by allocation +319.370 seconds |
| exact local reduce | node 0 accepted six trainers and 1,967,040 tokens at allocation +375.467 seconds |
| deterministic global freeze | two complete node contributions and 3,934,080 tokens at allocation +395.578 seconds |
| owner transport and redistribution | 11,013,540,992 bytes each in 98.961 seconds, 222,582,520.49 bytes/s, two owners, zero replay, 11,013,540,992-byte high-water, and full release; complete at allocation +494.541 seconds |
| terminal checkpoint stage | node 0 manager reached `checkpoint_commit`, then expired the fixed proposal deadline; zero publications and no configured role retry |

The node-0 ownership ledger reports a 55,068,090,151-byte overall local
high-water below its 68,719,476,736-byte hard cap, 22,027,198,003 bytes after
release, 11,013,540,992 bytes written, 33,040,622,976 bytes read, exactly two
published aggregate files, and `shared_run_dir_is_bulk_path=false`. The frozen
generation record has two owners, zero replay, the exact accepted peer set,
and `central_full_model_broker=false`. The fenced database remains at fence 1
with zero publication rows, so the failed attempt cannot be mistaken for a
committed generation.

Slurm accounting records `TotalCPU=08:11:15`, raw energy `839538`, and role
step `MaxRSS=151239508K`, `MaxDiskRead=144016.86M`, and
`MaxDiskWrite=54899.40M`. The retained tree at
`reports/frontier/evidence/job-5028767` contains 81 files; its `SHA256SUMS`
digest is `07b96b59379bc335f5bcfb7ccb691009ec68d9c9c9e7071041335932370eb09a`
and `sha256sum -c` passes. Top-level stdout, stderr, fenced SQLite, and
supervisor-event hashes are respectively
`c7b8c2ee0144f9781b534f2e62186979c2c3a8bf2fd7dec165609e23d1788edb`,
`5955617aaaf81e185f8d2921e493047f3a5586cec828fb77c4843ae7bf62f966`,
`4210f5843322a5db6e74d8c6c6509acdcb8aee4b0705b66a4872ab037e5048fd`,
and `1388de0089780a2c907cd7a997d69a5ec1f489a7ffd84fc85ce3dbb3e5f46852`.

The designated trainer previously serialized and hashed a separate 7.7 GB
node-local recovery checkpoint before it created the complete authoritative
checkpoint and proposal. That ordering placed disposable recovery ahead of the
only durable operation needed for atomic commit. The focused change writes one
complete fence-specific authoritative checkpoint immediately after aggregate
application, asserts the lease, publishes its proposal, and reuses the same
immutable file as leader recovery. Non-leaders retain bounded node-local
recovery, but it no longer gates the manager. The exchange/commit deadline
remains 180 seconds. The new
`test_checkpoint_leader_proposes_before_disposable_local_recovery` first failed
against the r5 tree (`proposal` followed local recovery) and passes after the
reordering; the independent fresh-process restart proof now also verifies that
leader recovery names and reloads the authoritative checkpoint without a
duplicate local serialization.

The final changed-tree pre-submit gate passed in the pinned environment: all
127 lifecycle, exact-math, membership/quorum, bounded transport, owner replay,
fencing/restart, launcher, real-trainer, and helper tests passed in 123.05
seconds. Approved-Python `compileall`, metrics `json.tool`, `git diff --check`,
the forbidden dense-packing/collective/MPI live-path scan, the r5 retained
evidence checksum, and the empty Frontier queue check also passed. This is the
only code change permitted in the next startup payload.

## Authoritative integration and queue gate

- Integrated v1 commit `ae2e6f26046fb7a6b348e845fb4615092a7c37e0` is an
  ancestor of local `HEAD` and fetched authoritative `origin/main`.
- The local lifecycle reliability fix is
  `87355275976184898fc3fb1975473ce69952f49f`; local `HEAD` and fetched
  `origin/main` were identical before payload rendering.
- `squeue -u "$USER"` returned no rows at the initial and final pre-submit
  checks. The retained submit script repeats this check and fails closed if any
  user job appears before `sbatch`.
- The superseded job `5027064` used code `8d0b6a5`, broad 900-second stage
  maxima, and failed before a first training heartbeat. This payload uses the
  integrated pool runtime and 180/420/180/720 hard stage bounds, so it is not an
  unchanged retry.

## Local pre-submit validation

The first exact suite invocation exposed an unreliable two-second healthy
Python-launch allowance in the bounded step-supervisor test: 119 passed and one
failed. The standalone reproducer failed on the loaded login node; direct cold
approved-Python launches reached 1.77 seconds before pytest/fork overhead. The
test now gives healthy process launch a bounded 30 seconds while retaining the
independent 0.2-second stuck-child eviction proof. The focused reproducer then
passed, and the exact full suite passed:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q -p no:cacheprovider tests/test_fenced_admission.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py tests/test_resilient_e97_reducer.py tests/test_resilient_shard_owner.py tests/test_resilient_pool_runtime.py tests/test_resilient_e97_split_roles.py tests/test_resilient_node_transport.py tests/test_resilient_e97_runtime.py tests/test_resilient_e97_true_2n_launcher.py tests/test_async_diloco_real_trainer.py tests/test_train_helpers.py
120 passed in 220.14s
```

Additional passing gates:

- approved Python 3.12 `compileall` over the integrated runtime, launcher, and
  selected tests;
- `python -m json.tool` on the integration metrics and `git diff --check`;
- no `.tolist(`, all-reduce/all-gather/process-group/barrier/MPI collective
  token in the live integrated files;
- exact weighted/reference, changing-membership, idempotence, stale/corrupt
  rejection, owner-loss replay, newer-fence restart, trainer parity, and
  launcher topology tests within the 120-test suite;
- representative integration metrics: two retained spool files, zero
  per-microchunk files, zero retained bytes after release, no central
  full-model broker, and no Lustre tensor hot path;
- the exact `/lustre/orion/bif148/proj-shared/emender/runs` control-store parent
  passed independent-connection loser admission, atomic bundle visibility,
  monotonic fence `1 -> 2`, and stale-publication rejection. The batch repeats
  real allocation admission before any role/model load.

## Immutable identity

Runtime attested locally and is re-attested before roles start:

```json
{"executable":"/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python","python":"3.12.13","torch":"2.10.0+rocm7.1","torch_hip":"7.1.25424","triton":"3.6.0"}
```

Inputs and payload files:

| Artifact | SHA-256 |
|---|---|
| generation-0 E97 seed (7,719,679,924 bytes) | `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9` |
| pinned p50k tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| flat batch-size-4 E97 arguments | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| two-node sbatch launcher | `3c4e41ef28e6cecc42b68a68fdbc111e26ba1c304f6d785e7110ea4d49ef6ed6` |
| allocation supervisor | `75722c09b9329fd0f4741d9b0b5182403b0bd9f8cc57545f2d59e4a05dd8f88b` |
| live role entrypoint | `dde43114e04a0e0a7540c8ebd07e33a96b293d8887a1c8b31a5bb288cf57a7a6` |
| fenced admission | `4fb942149d9ebb7dc8e25300f55e612fc2c3c704c73141729781dabac7ab01bb` |
| exact weighted reducer | `1cb4c30cc23d20d4c893ac047371d441b9c61c4a70f6da5dbcda86a9ec6f743c` |
| split roles | `4ac5abff8980fb2f59606dd5ccdeaf2668c0a4d4a3b9486acdc9418baff09d0f` |
| fenced runtime | `eed968906587451b1371150e68d445f316532e6cd220db6d9e69667fef7f9c77` |
| pool owner/control runtime | `4fdae65ccd5a9ec10cc2c88acb8edbf368fd16230711414dfab1f3e54603de97` |
| exact startup submit script | `f73a8c7246c305770fab5397700af5f6e3097cdf5e11a248974277f271ee845c` |

The source seed identity is
`step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
The startup payload identity is
`8735527-20260718T143538Z-pool-v1-startup-2n20m-k40`.

## Startup command and acceptance

The exact retained command is:

```bash
bash reports/frontier/validate-resilient-pool-v1-2n-startup-submit-20260718T143538Z.sh
```

The script resolves to one `sbatch --parsable` request for exactly two nodes,
debug QoS, `00:20:00`, 16 GPUs, 2 model-free managers, 16 real trainers, batch
size 4, K=40, global node quorum 2, one generation, no injections, no resume
handoff, node-local `/tmp` bulk/kernel/tokenizer paths, and bounded 1 MiB
chunks under a 32 GiB shared spool ledger. It refuses a nonempty queue, an
existing run directory, non-authoritative checkout, or runtime-code drift.

Run identity:
`validate-resilient-pool-v1-2n-startup-20260718T143538Z-8735527`.

Required observed stages from allocation start are READY by 180 seconds, local
K=40 by 420 seconds after trainer start, exchange/commit within a further 180
seconds, and first atomic generation by 720 seconds. TERM@300 provides the
handoff lead. Success additionally requires runtime/topology attestation,
accepted-token/weighted-manifest evidence, byte throughput/high-water/release
telemetry, no forbidden transport, and a reloadable immutable checkpoint.

## Version-1 conformance checklist

- **R01–R03:** exclusive expiring fenced admission precedes model load; two
  leased READY node managers define active membership, not 16 launched ranks.
- **R04–R07:** fresh fence/generation identities, duplicate/stale/corrupt
  rejection, exact token weighting, explicit quorum/deadlines, and one atomic
  immutable commit/checkpoint/latest bundle are live acceptance conditions.
- **R08–R10:** deterministic distributed shard owners use bounded checksummed
  point-to-point chunks and prompt release; managers are model-free; tensor,
  membership, heartbeat, spool, and redistribution hot paths are node-local or
  network, never Lustre. Only bounded durable lease/checkpoint evidence uses the
  shared run directory.
- **R11–R12:** this startup rung establishes the fenced checkpoint. Manager and
  trainer loss/rejoin plus outer-state recovery are reserved for the ordered
  resilience and fresh-allocation rungs.
- **R13–R16:** the scheduler-neutral pool is mapped through the Frontier
  adapter; stage deadlines and immutable metrics are mandatory; reference math
  passed; this two-node gate must pass before any 4+ node consumer proceeds.

Minimum startup progress floor: two complete READY node-peer contributions,
positive accepted-token weight, and one atomic committed generation. No
all-rank collective, full-model broker, Lustre tensor exchange, microfile spool,
unbounded wait, unchanged retry, or production mutation is permitted.
