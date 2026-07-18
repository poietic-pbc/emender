# Native resilient data plane v1 integration record

Date: 2026-07-18

Task: `integrate-native-resilient-dataplane-v1`

Design authority: [Resilient DiLoCo Compute Pool v1](../../docs/RESILIENT_DILOCO_COMPUTE_POOL.md) and [Native resilient DiLoCo data plane v1](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md)

## Result and admission scope

The native local reducer and persistent libfabric owner transport are merged,
ABI-compatible, built as one reproducible bundle, and joined beneath the
existing Python resilient control plane. The integrated implementation tip is
`6eaa52dedd232b2b1b4896181223d18b9ddc3063`; the core integration commit is
`9acd2a7af64f1049275f8a5f485cfa9560553681`.

This record proves gate G0 only. It does **not** claim a Frontier CXI G1 or G2
result. The full-layout launcher defaults to `native-cxi`, requires exact
`FI_PROVIDER=cxi`, verifies the installed binary/library bundle and an exact
matching G2 artifact before roles or model load, and rejects Python TCP and
test providers. Because no matching G2 artifact exists, production/full-layout
execution remains fail-closed. No Slurm job was submitted by this task.

## Dependency integration and ABI resolution

The integration retains both evaluated dependency histories as merge parents:

| Dependency | Published commit | Integration merge | Preserved evidence |
|---|---|---|---|
| Native local reducer/memfd ABI | `796a503eb24258b4c83bdd1629c3053f627b41d3` | `dbc1fc9294a390577cf680b6902d0325172f0d53` | Local C ABI, exact Python/native roots, fencing, failure, replay journal, GNU/Cray builds |
| C++17 libfabric owner transport | `00ce168478ed589ee1c5892f3b518fc6aa227a66` | `d1f8cc134612e24bd4bcefe9f9484ae8287e8c19` | Protocol/owner, C ABI, multiprocess RDM, provider/fault, release and sanitizer tests |
| Compiled fixed-world qualification | `2c2e827d9f03095060dc05b6d85130d2bb5fc4f3` | base of `dbc1fc9` | Reference report/JSON and retained Frontier job 4974616; reference-only, never elastic authority |

The only add/add merge collision was
`tests/test_native_dataplane_reference.py`. It was resolved as a union: all
three compiled-reference qualification tests and all four local-native
reference tests remain. The qualification JSON checksum was advanced with the
now-integrated gap matrix; its report remains checksum-linked. No dependency
test or evidence file was dropped.

The ABI join is additive. `libemender_ndp.so.1` remains local ABI `0x00010000`.
`libemender_ndp_transport.so.1` remains transport ABI `0x00010000` and adds the
size-versioned `ndp_transport_identity_v1`/
`ndp_transport_bind_identity_v1` interface. A service binds its full fenced
identity before publishing an endpoint. `ndp_transport_endpoint_v1` now emits
the normative, at-most-4,096-byte endpoint record; peer upsert validates and
strips that record before inserting only the opaque provider address into the
AV. Prefix ABI compatibility and stale/rejoin records are exercised in C.

The unified CMake entrypoint is `native/CMakeLists.txt`. It configures the
local and transport trees together, installs both SONAMEs plus
`ndp_cxi_service`, runs all eight CTests, and records one manifest with source,
compiler, ABI, artifact size/SHA-256, and bundle SHA-256.

## Startup, lifecycle, and ownership

Startup is deliberately ordered:

1. Python acquires/validates the allocation lease and current fence.
2. The launcher verifies the clean native build manifest and exact G2 artifact
   before node discovery, supervisor admission, or model load.
3. A model-free manager opens the local native controller ABI, opens the
   explicit native provider, binds `(run, fence, worker, incarnation,
   endpoint_epoch, expiry)`, and publishes the opaque endpoint record in
   leased READY metadata.
4. Python freezes the READY snapshot. Native services install only endpoint
   records whose backend, bundle, run, fence, incarnation, epoch, expiry and
   provider match that snapshot.
5. Trainers map service-allocated memfd lanes and produce directly. Python
   supplies the immutable accepted set, weight and deadlines; native code
   performs exact reduction, owner transfer and redistribution.
6. Trainers apply the shared read-only result. Native code emits a metadata
   checkpoint proposal but cannot publish authoritative progress. It releases
   generation state only after Python supplies a matching finalized handoff
   and authoritative-latest CAS identity/digest under the same fence.
7. TERM or normal exit stops admission, drains or aborts local operations,
   cancels routes independently, releases registered/shared buffers, and does
   not rendezvous with a lost peer.

`NativeManagerSession` is the joined lifecycle surface. The integration test
executes install → producer-direct buffer → weighted submit → freeze → owner
finalize → shared trainer view → proposal → deliberate pre-CAS commit rejection
→ matching fenced CAS approval → commit → route install → READY telemetry →
TERM drain. It asserts zero final local shared bytes and zero final transport
in-flight/retained bytes.

R8 valid atomic commit behavior is unchanged: `finalize_checkpoint` still
makes the complete checkpoint/handoff durable and
`SQLiteFencedControlStore.publish_bundle` atomically publishes commit,
checkpoint and authoritative latest only under the current lease. Native
`COMMIT` is lifetime/release acknowledgement, not a second publication
authority. A stale fence or mismatched authoritative digest cannot release a
result as committed.

The R9 observability correction is also retained. The node supervisor produces
one deduplicated manager READY record per node/incarnation before admitting
cold trainers; the native READY record adds provider, endpoint epoch, source
and bundle identity plus zero Python-dense/spool byte counters. The manager
does not import or own a model or optimizer.

## Dense byte flow

The exact production E97 logical constants are `L = 5,506,770,496` bytes,
`C = 67,108,864` bytes, 83 shards, two nodes and eight trainer lanes per node.
The G2 validator requires `2L = 11,013,540,992` contribution bytes and exactly
`2L = 11,013,540,992` redistribution bytes for each generation. Each data
frame adds one exact 320-byte header; endpoint metadata is bounded at 4,096
bytes. Those header/record bytes are not counted as logical tensor bytes.

```text
                         metadata only (Python; bounded deadlines)
 lease/fence ── READY endpoint record ── frozen membership/weights ── CAS commit
     │                         │                       │                  ▲
     │                         │                       │                  │
     ▼                         ▼                       ▼                  │
  manager A                 manager B          native owner plan         │
     │                         │                       │                  │
 8 trainer lanes          8 trainer lanes             │                  │
     │ direct memfd/XPMEM      │ direct memfd/XPMEM    │                  │
     ▼                         ▼                       │                  │
 exact f64 local reduce   exact f64 local reduce      │                  │
     │ one L-byte node contribution                    │                  │
     └──────────── FI_EP_RDM / cxi, 83 bounded shards ┴──────┐           │
                 logical contribution total = 2L             ▼           │
                                                    distributed owners    │
                                                    once-only receipts    │
                                                    bounded replay/credit │
                                                             │           │
                    owner-direct redistribution total = 2L    │           │
     ┌───────────────────────────────┬─────────────────────────┘           │
     ▼                               ▼                                     │
 one shared node aggregate      one shared node aggregate                  │
     │ 8 read-only views             │ 8 read-only views                   │
     ▼                               ▼                                     │
 trainer apply / proposer ───────── metadata checkpoint proposal ──────────┘
```

There is no Lustre dense update/aggregate path, Python dense socket, Python
object/scalar packing, all-rank collective, or central full-model broker in
the admitted native backend. Checkpoints and immutable evidence may use the
shared filesystem only after dense apply. Registered slots, byte credit,
resident memory, replay, event queues and every wait are bounded. The normative
owner resident admission bound remains
`2*L + A + 2*K*(C + 320) + R + 64 MiB`; arithmetic is overflow-checked before
generation admission.

The local eight-trainer G0 fixture measured 4,096 bytes shared high-water and
512 bytes mapped-lane high-water, admitted/released 4,608/4,608 bytes, and zero
default disk/trainer spool. The optional debug replay fallback wrote one
bounded 1,120-byte reduced-numerator journal and removed it at commit. These
small-fixture numbers are not substituted for the mandatory G2 full-layout
metrics.

## Failure boundaries

```text
 allocation lease/fence expires
          └──> Python stops admission; stale native command/endpoint rejected

 trainer absent before freeze
          └──> bounded Q_min/T_min/deadline decision: freeze accepted set or abort

 corrupt/stale/conflicting frame
          └──> route-local CRC32C/SHA-256/fence rejection; no owner mutation

 lost receipt / owner disappears
          └──> idempotent receipt lookup or sender replay
               └──> at most 2 owner reassignments and at most 2*L replay bytes
                    └──> success, or bounded no-commit abort

 service/node disappears
          └──> endpoint lease expiry/removal; surviving routes continue independently

 checkpoint or authoritative CAS fails
          └──> native result remains uncommitted; prior immutable latest stays authoritative

 SIGTERM
          └──> stop admission -> DRAIN/ABORT local state -> cancel each route
               -> release mappings/MRs -> exit; never wait for every peer
```

Minimum progress is the configured `Q_min` complete frozen READY node
contributions and `T_min > 0` accepted tokens before the absolute generation
deadline. Transport reachability never invents membership or relaxes that
floor.

## Artifact identity and packaging

The clean RelWithDebInfo build recorded by
`build/native-resilient-dataplane/native-artifacts.json` is:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `lib64/libemender_ndp.so.1` | 981,496 | `5345c26c606b3b3329087cb7cbecafb7c1ed4e0372bccda092917ff694a76b96` |
| `lib64/libemender_ndp_transport.so.1` | 1,129,152 | `1fb8b85e8c8cb1a1bab13c0a1ad00b59c6137b1cbef78a8dde5898961333269f` |
| `bin/ndp_cxi_service` | 777,640 | `63e43df6d72a56ad88d37836d75aa7f084418392df21b5717237fdc708140264` |

Bundle SHA-256:
`446613d84147dc4d63ffd114ec2ea9ea4129233889cafc8854de05e5f1c562bb`.
The recorded source is clean commit
`6eaa52dedd232b2b1b4896181223d18b9ddc3063`, CMake 3.28.3, Cray wrappers
`cc`/`CC`, XPMEM enabled, and both ABI values `0x00010000`. Manifest
validation recomputes sizes/digests, rejects path escape or source mismatch,
and scans dynamic exports for forbidden `MPI_`/`PMPI_` symbols.

## Validation

This integration applies the Compute Pool v1 conformance checklist to
R01–R16 and the native checklist to NDP01–NDP17:

- READY is leased/fenced metadata and precedes trainers; endpoint exchange is
  bounded metadata rather than PMI, filesystem fan-out or all-gather.
- Generation/run/fence/attempt/owner/worker/incarnation/sequence identities,
  exact weighted math, idempotent receipts and stale/corrupt/conflict rejection
  are exercised in C and Python.
- Dense movement is node-local memfd/XPMEM plus point-to-point RDM, with fixed
  pools, receiver credit, explicit byte/replay/deadline bounds, full release,
  and no central broker.
- Trainer/peer loss, replay, reassignment, expiry/removal, checkpoint/CAS
  failure, TERM drain and fresh-fence restart have bounded outcomes.
- Immutable committed checkpoint evidence remains Python-controlled and
  atomically fenced; native telemetry is diagnostic and cannot declare a
  commit.
- G0 is the only newly completed rung. G1/G2 and every later rung remain
  ordered, exact-code operational work.

Applicable matrix IDs: **R01–R16; NDP01–NDP17**. Detailed present/partial gate
status is recorded in
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../../docs/RESILIENT_DILOCO_GAP_MATRIX.md).

Exact local commands executed (the Python path is the approved immutable
Frontier runtime; no `sbatch` command was run):

```bash
git fetch origin main
git merge --ff-only origin/main
git merge --no-ff 796a503eb24258b4c83bdd1629c3053f627b41d3
git merge --no-ff 00ce168478ed589ee1c5892f3b518fc6aa227a66

PYTHON_BIN=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
BUILD_JOBS=8 scripts/frontier/build_native_resilient_dataplane.sh

EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1" \
EMENDER_NDP_TRANSPORT_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1" \
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -q \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_transport_bridge.py \
  tests/test_native_artifact_attestation.py \
  tests/test_native_pool_integration.py \
  tests/test_resilient_e97_reducer.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_peer_membership.py \
  tests/test_resilient_shard_owner.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py

/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m compileall -q ndm scripts/frontier
bash -n scripts/frontier/build_native_resilient_dataplane.sh \
  scripts/frontier/resilient_e97_true_2n.sbatch
jq empty reports/native-dataplane/integrate-native-resilient-dataplane-v1-metrics.json \
  reports/frontier/native-dataplane-reference-v1.json
nm -D --defined-only build/native-resilient-dataplane/lib64/libemender_ndp.so.1
nm -D --defined-only build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1
ldd build/native-resilient-dataplane/lib64/libemender_ndp.so.1
ldd build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1
git diff --check
cmp -s AGENTS.md CLAUDE.md
```

The unified CTest result is 8/8. The complete focused native/resilient Python
suite contains 103 tests. Its final result and publication verification are
machine-readable in
[`integrate-native-resilient-dataplane-v1-metrics.json`](integrate-native-resilient-dataplane-v1-metrics.json).

## Promotion boundary

This task intentionally did not run G1/G2 and did not submit Slurm. A
production/full-layout launcher remains unusable until a later approved runner
produces an exact-code G2 JSON with `cxi`/`FI_EP_RDM`, all three artifact
digests, 83 shards, the exact `2L` contribution and redistribution byte counts,
zero Python dense/spool/disk replay/full-copy counters, bitwise roots, admitted
bounds, full releases and required timings. That fail-closed boundary is a
successful integration property, not a claim that Frontier promotion has
already occurred.
