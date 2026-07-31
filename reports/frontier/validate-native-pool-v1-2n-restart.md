# Native resilient pool v1 fresh two-node restart acceptance

Date: 2026-07-19/20 UTC

WG task: `validate-native-pool-v1-2n-restart`

Live source commit: `56237c05d39eea23ee2e5818ce931c0ea3e81f06`

Live restart job: `5033539`

Exact-source full-layout prerequisite: G2 job `5033529`

## Result

**Passed.** A fresh exactly-two-node Frontier allocation acquired fence 2 for
the existing resilience-smoke run, while the prior allocation and handoff were
fence 1. Both new managers reconstructed generation 3 from the immutable
fence-1 handoff, both persistent native services created fresh production CXI
endpoints, and 16 fresh GPU trainers completed two more real E97 K40
generations. Job `5033539` ended `COMPLETED 0:0` after 12:02 of its requested
20-minute debug-QoS limit on `frontier[03099,03394]`.

The atomic token clock advanced exactly
`15,736,320 -> 20,981,760 -> 26,227,200`, at steps
`1,525,120 -> 1,525,160 -> 1,525,200`. Generation 4 and generation 5 each
froze exactly two new manager contributions totaling 5,245,440 tokens, applied
all 32 expected trainer result lanes exactly once, and published one immutable
checkpoint/manifest pair under fence 2. SQLite ends with five commit rows,
five linked checkpoint rows, one authoritative latest row at generation 5,
`last_fence=2`, no live lease, and `PRAGMA integrity_check=ok`.

A live probe used the old job-5033384 allocation identity and fence 1 after
job 5033539 had acquired fence 2. Even with an artificially future expiry in
the caller's stale lease object, renewal, contribution admission, commit
publication, checkpoint publication, and latest publication were all rejected.
The publication-row count remained 7 and the authoritative generation-3
latest value was byte-for-byte unchanged during the probe.

The machine-readable result is
[`validate-native-pool-v1-2n-restart-metrics.json`](../validate-native-pool-v1-2n-restart-metrics.json).
The compact, independently checksummed live evidence is under
[`frontier/evidence/job-5033539`](evidence/job-5033539), and the
exact-source prerequisite is under
[`frontier/native-dataplane/5033529`](native-dataplane/5033529).
The three 7,899,873,331-byte checkpoints remain at their recorded durable run
paths instead of being duplicated into git; a separate post-job process
streamed their entire contents and independently reloaded each one.

## Allocation and source accounting

The immutable launch recipe is
[`validate-native-pool-v1-2n-restart-submit-20260720T000459Z.sh`](validate-native-pool-v1-2n-restart-submit-20260720T000459Z.sh).
It pins the exact pushed source snapshot, approved Python 3.12 environment,
step-1525000 seed, tokenizer, data, runtime config, native build, G2 gate,
source handoff, run database, and fresh node-local roots before calling
`sbatch`. Its allocation command uses exactly `-N 2`, eight GPUs per node,
`--qos=debug`, `-t 00:20:00`, and `--network=job_vni`; no four-or-more-node
job was submitted.

| Job | Slurm result | Nodes | Elapsed | Purpose |
|---|---|---:|---:|---|
| `5033516` | cancelled while pending; no node assigned | 2 requested | 0:00 | Superseded pre-compute attempt, cancelled when source review exposed the older-fence resume rejection |
| G2 `5033529` | `COMPLETED 0:0` | 2, `frontier[06841,06907]` | 2:50 | Exact-source clean production-CXI full-layout prerequisite |
| restart `5033539` | `COMPLETED 0:0` | 2, `frontier[03099,03394]` | 12:02 | Fresh fence-2 reconstruction and two real E97 K40 commits |

Job 5033516 never left the pending state, acquired no nodes or fence, and
mutated no durable state. It was cancelled rather than allowed to run known
incorrect source. Its accounting and the pre-restart seed snapshot are retained
under [`frontier/evidence/job-5033516`](evidence/job-5033516).

The accepted job started from the immutable snapshot
`/lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-56237c05d39eea23`,
whose detached `HEAD` is
`56237c05d39eea23ee2e5818ce931c0ea3e81f06`. Slurm's captured submit line
records:

- run `npv1f-progress-20260719T223459Z-f56e27a`, generation 3, two requested
  generations, global quorum 2, no restart or fault injection;
- durable fence DB and exact
  `generation-00000003-fence-00000001.json` resume handoff;
- unique bulk root `/tmp/resilient-e97-restart-20260720T000459Z` and unique
  kernel-cache root `/tmp/resilient-e97-restart-kernel-cache-20260720T000459Z`;
- `DILOCO_DATAPLANE=native-cxi`, exact `FI_PROVIDER=cxi`, `FI_EP_RDM`,
  `FI_MR_CACHE_MONITOR=kdreg2`, and `FI_CXI_ATS=0`;
- source commit `56237c05...`, build-manifest SHA-256
  `bc40d57e1ebbf105424d4e153da5d43bd182e08cc7d51c6b3f9597b4cdba3ab3`,
  and G2 job 5033529.

The seed checkpoint SHA-256 remained
`1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`;
the tokenizer SHA-256 remained
`94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`.

## New fence and stale-owner exclusion

The allocation acquired fence 2 and incarnation
`c4eac01c344d43819dd9f1982a1d7f4f` before native-service start, manager
start, trainer start, or model load. The earlier allocation identity was job
5033384, incarnation `3eb3153d576f4ed2b14f1025c79d75a3`, fence 1.

The live stale-owner probe ran while the new fence-2 lease was active and
before any fence-2 generation publication. It called the real
`SQLiteFencedControlStore` against the production control database:

```text
store.renew(old_fence_1_lease, ttl_s=60)
store.assert_current(old_fence_1_lease)             # contribution guard
store.publish(old_fence_1_lease, kind="commit", ...)
store.publish(old_fence_1_lease, kind="checkpoint", ...)
store.publish(old_fence_1_lease, kind="latest", ...)
```

The exact results were:

```text
renew: rejected: stale or expired allocation lease renewal
contribute: rejected: operation rejected by newer or expired fence
commit: rejected: stale commit publication
checkpoint: rejected: stale checkpoint publication
latest: rejected: stale latest publication
```

[`live-stale-fence-probe.json`](evidence/job-5033539/live-stale-fence-probe.json)
contains the before/after authoritative value and row counts. There was no
stale durable mutation. In the actual new-generation controls, every manager
pool identity has coordinator epoch 2 and one of the two fresh manager
incarnations; every one of the 32 native submissions, 32 result-lane releases,
and 32 durable apply receipts has fence epoch 2. A recursive audit found no
fence-1 record or prior manager incarnation in generation-3/4 native controls.

Thus an old allocation cannot renew or publish metadata, and stale native
messages cannot contribute to the new pool: no old route, owner incarnation,
fence, or trainer buffer identity appears in either new committed generation.
The exact-source G2 additionally exercised two stale-frame rejections and two
checksum rejections with all three analytical-reference generations still
matching.

## Exact reconstruction and independent reload

Both fresh managers reported `status=synchronized`, generation 3, current
fence 2, source fence 1, and exact source handoff SHA-256
`1b7b06d7af106fbb86305d1bda6ee7e5a7f52a1ecbd9d48e230b3446843eb8cd`.
Their new manager incarnations were:

- node 0: `60894d6f56db4b1283febd91517b041a`;
- node 1: `558d53135a7e445b9b8b79bfbee4150c`.

The new services created distinct endpoint epochs
`1784506032964477041` and `1784506036429441374`, reported the submitted
source and artifact bundle, and made both managers READY at generation 3
before any trainer started. The trainer resume independently checked the old
handoff's code identity and the checkpoint's native-runtime identity.

After Slurm completion, a new process used approved Torch 2.10 with
`weights_only=True`, `mmap=True`, and CPU mapping to load each checkpoint. It
checked all complete-state fields and streamed each full file through SHA-256.
It separately hashed the exact sorted tensor names, dtypes, shapes, and bytes
using `ndm.native_e97_runtime.state_digest`.

| Generation | Fence | Step | Tokens | Manifest SHA-256 | Checkpoint SHA-256 | Model-state digest |
|---:|---:|---:|---:|---|---|---|
| 3 (source) | 1 | 1,525,120 | 15,736,320 | `1b7b06d7af106fbb86305d1bda6ee7e5a7f52a1ecbd9d48e230b3446843eb8cd` | `03a51ea4e065b902c825ad78d987acaff32f0f5b52ed6f713b05a16b445fd753` | `2e59f874572b356439154cc6b781d13eaf4187c58fe8ad6e3f4845ecb75b6c23` |
| 4 (new) | 2 | 1,525,160 | 20,981,760 | `44d18271f7b61f887b13dde866cf111b77f06e6231050f0fafc09506e2ab54bc` | `94c259b4acb9a908ee7da2178c2e3f99d5be5946bcb566e73b8f7aa71254514e` | `9972f590ad25ab062c16511abbae55a877df2034f778957d02966df1268da548` |
| 5 (new) | 2 | 1,525,200 | 26,227,200 | `0b0ed65a87a796c56ff267db52dff4fd2f843b7ac9c9ffc8211c97fba770c51c` | `1e68333f6392b4fbd11e17fa19da8fcbbe5b688ffecbe2b1200e3b1fa110c828` | `7243f1bebb7553d024c6dd999931853f94dec996f380e07c203114a5b589dc91` |

All three independent loads contain 146 model tensors, 145 inner-optimizer
state entries, one optimizer parameter group, global membership
`node-0,node-1`, local ranks 0 through 7, the exact run/source/payload
identities, and outer state `{"algorithm":"weighted-mean","eta_outer":1.0}`.
The recomputed outer-state SHA-256 is
`79661b97a27fce6f9057a16642b0cabdc6f6f7ab7782315e12da17a3a136c712`
in every load.

The generation-3 model digest exactly equals the generation-4 native
`base_digest`; the independently loaded generation-4 model digest exactly
equals the generation-5 native `base_digest`. This is stronger than checking
that files merely exist: it proves that each predecessor model was the exact
native input to the next real E97 generation. Generation 5 was then loaded and
hashed independently as the terminal result.

The old and new native-runtime records differ only in the provenance fields
`source_commit` and `build_manifest_sha256`. Schema, provider, provider digest,
config digest, build bundle, and all four executable/library artifact digests
are identical. A substantive runtime difference would have failed resume.

## Two new native generations

Each input generation 3 and 4 froze two distinct fresh manager identities at
coordinator epoch 2 and exactly 5,245,440 tokens. For each input generation,
the retained evidence has:

- 16 sealed native submissions: eight ranks on each node;
- two node-result records with identical result root, layout, base digest, and
  global weight;
- 16 ordered result-lane release records and 16 durable apply receipts;
- one pool close containing only the two new manager incarnations;
- one leader checkpoint, immutable handoff manifest, commit row, checkpoint
  row, and monotonic latest update.

Generation 3's native result root is
`111a58284494bec5863c01448f25ddd3886b4bcd03146f3e4237e9bb60de6d83`;
generation 4's is
`5b81e75b789d7321f7cf1cea91c191db6ef60b26cf6833fafe65891189f5b85d`.
Both projected 5,506,770,496 result bytes. There were no restarts, duplicate
applications, unplanned evictions, or nonzero role completions: all 16
trainers and both managers completed zero, and the two native services drained
zero for `allocation_complete`.

## Local buffers are not authoritative

The successful allocation used brand-new job-local roots. Its source snapshot
records explicitly state that retained evidence includes only supervision,
telemetry, and bounded JSON/JSONL control metadata. It explicitly excludes
mailbox data, `*.data`, `*.pt`, and the kernel cache. The checkpoint was loaded
from the durable fence-1 handoff; neither manager synchronized from a local
result, spool, trainer recovery buffer, or unfinished prior allocation file.

There were zero trainer-spool bytes, disk-replay bytes, and Python dense-socket
bytes. The two new manager transports each moved exactly 22,027,081,984 useful
bytes in each direction for the two generations. Their terminal shutdown
records are exact production `cxi/cxi0/FI_EP_RDM` and have
`in_flight_bytes=0`, `retained_bytes=0`, `replay_bytes=0`, and
`route_errors=0`, with owner state `DRAINING`. The two CQ errors per manager
are the expected terminal cancelled receives (`status=-12`, provider errno
125) after useful bytes and retained buffers reached zero.

This makes durable handoff/checkpoint state authoritative and leaves local
dense buffers disposable, in conformance with R02, R07, R11, NDP04, NDP12,
NDP13, and NDP17.

## Exact-source prerequisite and regression validation

Source review before the accepted launch found that manager restart required
the authoritative handoff fence to equal the new allocation fence, which made
a real fresh allocation impossible by construction. The failing regression
`test_fresh_allocation_manager_syncs_older_authoritative_handoff` reproduced
`ValueError: manager rejoin latest fence differs from allocation fence`.

Commit `56237c05d39eea23ee2e5818ce931c0ea3e81f06` changed manager
synchronization to accept only a positive authoritative source fence no newer
than the live allocation fence, while preserving the new fence for all
subsequent work. It also added
`test_native_restart_runtime_compatibility_rejects_substantive_digest_change`:
resume may cross provenance-only source/build-manifest changes when every
runtime artifact and semantic configuration digest is identical, but rejects
any substantive native-runtime drift. Trainer resume also verifies that the
handoff code ID agrees with the checkpoint runtime source.

The exact commit then passed:

- canonical RelWithDebInfo native build and all 10/10 CTests;
- 95/95 selected resilient/native Python tests in 131.99 seconds;
- exact-source G2 job 5033529 on two production CXI endpoints.

G2 completed three timed full-layout generations, all matching the independent
analytical f64 reference. Its 23.8302-second median was 4.153x the pinned
Python baseline and above the required 4x gate. It reported zero MPI
collectives, all-rank barriers, Python dense socket bytes, trainer spool bytes,
disk replay bytes, CQ errors, route errors, and post-release transport bytes.

## Complete validation commands

The Frontier environment was activated before every Python, native build,
test, and submission command:

```bash
source scripts/frontier/activate_emender_frontier.sh

PYTHON_BIN="$EMENDER_PYTHON" scripts/frontier/build_native_resilient_dataplane.sh
ctest --test-dir build/native-resilient-dataplane-build --output-on-failure

"$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_pool_integration.py

scripts/frontier/submit_native_dataplane_2n_gate.sh clean
bash reports/frontier/validate-native-pool-v1-2n-restart-submit-20260720T000459Z.sh

sacct -j 5033539 --starttime 2026-07-19T00:00:00 \
  --format=JobIDRaw,JobName%34,State,ExitCode,Elapsed,Timelimit,NNodes,NodeList%30 -P

sqlite3 -readonly reports/frontier/evidence/job-5033539/control/pool-v1.sqlite3 \
  'pragma integrity_check; select kind,name,fence,payload from publications order by kind,name;'

(cd reports/frontier/native-dataplane/5033529 && sha256sum -c SHA256SUMS)
(cd reports/frontier/evidence/job-5033539 && sha256sum -c SHA256SUMS)
sha256sum -c reports/validate-native-pool-v1-2n-restart-SHA256SUMS
```

The independent reload command used this complete-state core for generations
3, 4, and 5 after Slurm completion:

```python
loaded = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
assert loaded["generation"] == manifest["generation"]
assert loaded["step"] == manifest["step"]
assert loaded["accepted_tokens"] == manifest["accepted_tokens"]
assert loaded["accepted_peers"] == manifest["membership"] == ["node-0", "node-1"]
assert loaded["fence"] == manifest["fence"]
assert loaded["outer_update_state"] == manifest["outer_update_state"]
assert loaded["native_runtime_digests"] == manifest["digests"]["native_runtime"]
model_digest = ndm.native_e97_runtime.state_digest(loaded["model_state_dict"]).hex()
```

For every checkpoint, the verifier also read the entire file in 32-MiB chunks
through `hashlib.sha256`, rather than trusting Torch metadata or the manifest.
The complete structured output is
[`independent-audit.json`](evidence/job-5033539/independent-audit.json).

## Validation

The normative authority was
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../../docs/RESILIENT_DILOCO_COMPUTE_POOL.md),
with requirement tracking from
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../../docs/RESILIENT_DILOCO_GAP_MATRIX.md) and
the native protocol from
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md).
The conformance checklist was re-run for this exact source, G2 gate, and live
restart:

- [x] R01-R04: fence-2 admission preceded model work; durable source of truth,
  explicit stage limits, quorum 2, and 3,934,080-token minimum remained bound.
- [x] R05-R08: model-free managers, native dense movement, lease/heartbeat
  separation, strictly newer fence, and stale mutation rejection are proven.
- [x] R09-R12: bounded fresh incarnations, no fixed-world collective, atomic
  commit/checkpoint/latest transactions, and current membership snapshots are
  present for both commits.
- [x] R13-R16: no concurrent stale route, no global barrier, deterministic
  native sharding, and zero retained/in-flight shutdown state are proven.
- [x] NDP01-NDP04: exact production `native-cxi` artifact attestation, native
  lifecycle, service-owned dense path, and disposable local buffer authority.
- [x] NDP05-NDP08: canonical identities/fences, immutable input/result records,
  deterministic layout, and exact weighted arithmetic for both generations.
- [x] NDP09-NDP12: bounded admission/backpressure, release evidence, no
  Lustre dense hot path, and explicit TERM/drain behavior.
- [x] NDP13-NDP17: bounded replay identity, stable ordering, service-side
  aggregation/redistribution, native observability, and restart compatibility
  validation with no opaque payload replay.

Applicable requirement IDs checked: **R01, R02, R03, R04, R05, R06, R07,
R08, R09, R10, R11, R12, R13, R14, R15, R16** and **NDP01, NDP02, NDP03,
NDP04, NDP05, NDP06, NDP07, NDP08, NDP09, NDP10, NDP11, NDP12, NDP13,
NDP14, NDP15, NDP16, NDP17**.

All raw retained files, derived audits, submission evidence, manifests,
accounting, and reports are covered by nested and top-level SHA256SUMS files.
