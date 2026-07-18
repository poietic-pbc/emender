# Frontier native data-plane reference v1

Date: 2026-07-18

WG task: `qualify-compiled-frontier-transport-v1`

Machine-readable companion:
[`native-dataplane-reference-v1.json`](native-dataplane-reference-v1.json),
SHA-256
`33496519e54b6e166262b60a8fb3727c370bb0f16f33ccbde6623d8b8a4faf29`.

## Result

Qualification status: **qualified as a bounded correctness, build, provider,
and historical performance reference; not qualified as the elastic data
plane**.

The current helper source built twice with the current `CC`; both executable
builds were byte-identical, both shared-library builds were byte-identical,
the standalone singleton MPI diagnostic passed with
`MPI_THREAD_SERIALIZED`, and all 12 focused Python/C++ contract tests passed.
The active libfabric 2.3.1 installation returned usable discovery records for
CXI (`cxi0`), both shared-memory test providers (`shm` and `sm2`), and the TCP
test provider. The current binary resolves Cray MPICH through that exact
libfabric installation and `libcxi.so.1.6.0`.

This is a local/build qualification. The singleton diagnostic did not move
bytes between Frontier nodes, and `fi_info` provider discovery is not an
inter-node bandwidth result. Local evidence was sufficient to qualify the
installed provider and link path, so the task's conditional permission for a
two-node diagnostic did not apply. **No Slurm command was submitted**, no job
was created, and no large training job was run.

The retained job-4974616 result remains valid and checksum-linked, but it is
historical lineage evidence rather than a measurement of today's source and
toolchain. Job 4974616 used commit `40eb8d4`, CPE 26.03,
`PrgEnv-gnu/8.7.0`, Cray MPICH 9.1.0, and helper binaries that differ from the
current build. Today the active default is the CPE 24.11 component tuple,
`PrgEnv-cray/8.6.0`, Cray clang 18.0.1, and Cray MPICH 8.1.31. The helper has
114 inserted and 46 deleted lines since the retained run, principally to use
node-local communicators and leader reduction/materialization and to bound
node aggregate memory.

The old full-world result therefore establishes a concrete E97 payload and a
useful fixed-world reference. It does not establish current-source 256-node
performance and does not establish failure tolerance. MPI_Reduce is not the elastic solution.
The retained helper result itself says
`strict_collective_all_launched_ranks=true`; a failed/nonjoining rank can
strand or abort the collective.

## Authority and scope

This qualification is checked against the **Resilient DiLoCo Compute Pool,
version 1** in
[`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`](../../docs/RESILIENT_DILOCO_COMPUTE_POOL.md)
and the specialized
[`Native resilient DiLoCo data plane v1`](../../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
plus the companion requirement matrix in
[`docs/RESILIENT_DILOCO_GAP_MATRIX.md`](../../docs/RESILIENT_DILOCO_GAP_MATRIX.md).
Applicable compute-pool IDs are **R03, R05, R06, R08, R10, R14, R15, and
R16**; the native authority's compiled-reference minimum is **R05, R10, R15,
R16 and NDP02, NDP03, NDP05, NDP16, NDP17**.

The helper is deliberately outside the conforming live backend:

- R03 and R06: its active world and progress are the launched MPI world, not a
  live leased READY set with a bounded Q/T floor.
- R05 and R15: its bucketed weighted-sum math and the focused tests are useful
  numerical references. The elastic backend must still compare incremental
  sharded results to the published float64/synchronous references.
- R08: the helper has no leased owner reassignment/replay path and cannot
  supply bounded progress after an all-rank collective participant disappears.
- R10: the current executable resolves the native OFI/CXI stack, and the
  historical run used node-local IPC for dense handoff. This task did not run
  an inter-node transfer and makes no new live-traffic or mount claim.
- R14: the retained metrics define payload, latency, and throughput scopes;
  the machine baseline defines the native stage deadlines and required
  telemetry.
- R16: the historical 256-node fixed-world job cannot bypass the mandatory
  elastic two-node lifecycle/failure gate. This task did not claim or submit a
  new R16 rung.
- NDP02: the helper's MPI initialization, all-gather, node barriers, reduce,
  and all-reduce are explicitly nonconforming for the elastic binary.
- NDP03: exact `cxi` discovery is recorded, but the login-node query and
  singleton diagnostic are not promoted to a G1/G2 CXI transfer result.
- NDP05: the compiled result is a numerical reference; native v1 synthetic
  output must meet the stronger zero-tolerance, bitwise reference contract.
- NDP16 and NDP17: the baseline records provider/build/payload/performance
  facts and preserves the ordered G0–G6 gates, including hard G2 before any
  real model or 4+ node native job.

## Active Frontier toolchain

Captured on `login04` at `2026-07-18T21:00:55Z`:

| Component | Active value | Evidence |
| --- | --- | --- |
| Effective CPE release | `24.11` | `/opt/cray/pe/cpe/default -> 24.11`; loaded components exactly match `module show cpe/24.11` |
| Module-tree marker | `Core/25.03` | Present in `module -t list`; no explicit `cpe` module is loaded, so this is not reported as CPE 25.03 |
| Programming environment | `PrgEnv-cray/8.6.0` | `module -t list` |
| `CC` | `/opt/cray/pe/craype/2.7.33/bin/CC` | `command -v CC` |
| Compiler | Cray clang `18.0.1` | `CC --version` |
| CrayPE | `2.7.33`, CPU `x86-trento`, network `ofi` | modules and `CRAYPE_*` environment |
| Cray MPICH | product `8.1.31`, upstream `3.4a2`, device `ch4:ofi` | module, `mpichversion` |
| Libfabric | `2.3.1`, API `2.3` | `fi_info --version`, `pkg-config --modversion libfabric` |

The current executable resolves these native libraries:

| Library | Resolved path | SHA-256 |
| --- | --- | --- |
| Cray MPI | `/opt/cray/pe/mpich/8.1.31/ofi/cray/17.0/lib/libmpi_cray.so.12.0.0` | `2c4c8ebf39a26c04e813c37a1c838793dbe893567890f92a9792f229bfacb545` |
| Libfabric | `/opt/cray/libfabric/2.3.1/lib64/libfabric.so.1.29.1` | `4ff4ea1009c6472d9f3b5b60f43ed339862f5af74bd0288832266a05e5ddf916` |
| CXI support | `/usr/lib64/libcxi.so.1.6.0` | `3ab52632ec6ccdb2d7bd42002841ef103e30794f200b47b641544bc152178532` |

### Provider inventory

`fi_info -l` listed `cxi`, `ofi_rxm`, `ofi_rxd`, `shm`, `udp`, `tcp`, the
debug/noop/hmem/dmabuf hooks, `off_coll`, `sm2`, and `lnx`. The task-specific
provider probes all returned exit zero:

| Purpose | Probe | Version | Fabric/domain | Endpoint/protocol |
| --- | --- | --- | --- | --- |
| Frontier native | `fi_info -p cxi` | `0.1` | `cxi` / `cxi0` | `FI_EP_RDM` / `FI_PROTO_CXI` |
| Shared-memory test | `fi_info -p shm` | `203.10` | `shm` / `shm` | `FI_EP_RDM` / `FI_PROTO_SHM` |
| Shared-memory test | `fi_info -p sm2` | `203.10` | `sm2` / `sm2` | `FI_EP_RDM` / `FI_PROTO_SM2` |
| TCP test | `fi_info -p tcp` | `203.10` | `hsn0`, `ens9f1np1`, `bond0`, `lo` | `FI_EP_RDM` / `FI_PROTO_XNET`; `tcp;ofi_rxm` also available |

The TCP fabrics observed were `10.128.0.0/16`, `128.219.135.128/27`,
`172.30.204.128/25`, `127.0.0.1/32`, and `::1/128`.

## Current helper build and local diagnostics

Checkout qualified: `8a95df372db1b2952398245a912c001c2b3ce010`.
The helper's last source change is
`dc372fcdb2d4140f39ce311d169dd29c9e3ca93c`.

Source identities:

| Artifact | SHA-256 |
| --- | --- |
| `scripts/frontier/compiled_mpich_dense_helper.cpp` | `4b84788ee2a22d1dbf8a04fcc1af9029c07e4d5e52dc95201d308e2972248a46` |
| `scripts/frontier/build_compiled_mpich_dense_helper.sh` | `cfe55e66ddb4886b70812926bcfb2b642b3bcd91891c0b52d357bc4fe955b17e` |

The build was repeated in two independent `mktemp -d` artifact directories
without changing source, flags, modules, or environment:

```bash
ARTIFACT_DIR="$qual_build_a" \
  OUT="$qual_build_a/compiled_mpich_dense_helper" \
  bash scripts/frontier/build_compiled_mpich_dense_helper.sh
ARTIFACT_DIR="$qual_build_b" \
  OUT="$qual_build_b/compiled_mpich_dense_helper" \
  bash scripts/frontier/build_compiled_mpich_dense_helper.sh
cmp "$qual_build_a/compiled_mpich_dense_helper" \
  "$qual_build_b/compiled_mpich_dense_helper"
cmp "$qual_build_a/compiled_mpich_dense_helper.so" \
  "$qual_build_b/compiled_mpich_dense_helper.so"
```

Both `cmp` commands returned zero. The exact compiler invocations printed by
the build wrapper were:

```text
CC -O2 -std=c++17 -Wall -Wextra scripts/frontier/compiled_mpich_dense_helper.cpp -o <artifact-dir>/compiled_mpich_dense_helper
CC -O2 -std=c++17 -Wall -Wextra -fPIC -shared scripts/frontier/compiled_mpich_dense_helper.cpp -o <artifact-dir>/compiled_mpich_dense_helper.so
```

There were no compiler warnings. Repeated output identities:

| Output | SHA-256 |
| --- | --- |
| Executable | `8c3844b06077e407dc05941ba5ed04295797bb73df64ea82ee539fd815e43b92` |
| Shared library | `28743e10bf99ba751769264defa83f63c47e8b3c4bfb03af63b99a785614c342` |

The local command
`<artifact-dir>/compiled_mpich_dense_helper --diagnostic` returned zero and
emitted:

```json
{"diagnostic":"compiled_mpich_dense_helper","transport":"compiled-cray-mpich-helper-collective-reduce","world_size":1,"provided_thread_level":"MPI_THREAD_SERIALIZED","rank0_received_from":0}
```

Focused tests used Python 3.12.13 and PyTorch 2.10.0+rocm7.1. They were split
at the compiler-heavy parser test so each exact worker-shell command remained
within its execution window:

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -q tests/test_async_diloco_compiled_mpich.py \
  -k 'not cpp_request_parser_preserves_all_bucket_paths'
```

Result: `11 passed, 1 deselected in 10.80s`.

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -vv \
  tests/test_async_diloco_compiled_mpich.py::test_compiled_mpich_cpp_request_parser_preserves_all_bucket_paths
```

Result: `1 passed in 23.39s`. Together these are all 12 tests collected from
the focused module. They cover the bucketed collective source contract,
checksummed file IPC, rank-local workspace replacement, corruption rejection,
in-process shared-library invocation, weighted aggregate/reference parity,
distinct node-local IPC roots, streamed in-place apply, error propagation, and
the real `CC` 80-bucket request parser.

## Retained 256-node evidence

The retained narrative is
[`e97-async-256-rerun-job4974616-20260712.md`](e97-async-256-rerun-job4974616-20260712.md),
SHA-256
`b812f28c29911011da4cf7f9903bc6f1fe793d81ce7c8bfbbcb4bb27e0a36f32`.
The referenced run root remains readable:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260712/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/4974616-20260712T125530Z
```

Checksum links from that root:

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/manifest.json` | `a0d96f1b3ce168ffdd101724a11024045d848ceb569e6c434b1a37308863b454` |
| `artifacts/metrics.json` | `ef60edbaec5a5c8f66435f06b5f38ed2be6a13b51288ef303cdde110bf7b22e4` |
| `artifacts/command.txt` | `26289eaea21c9a7aba18eca587e1d1af1f162693a467081fb6619f1f7e2ea7fe` |
| `artifacts/env.txt` | `e6b00660ba570a3cab1a47344466b901c5fa182b8fd1277e209d20053c0420b7` |
| `artifacts/rank-start.tsv` | `24a9e2a1fc204f35f4460443cdd03e223e31cf2b3440bae3ad9448b54aab8a4e` |
| `summaries/summary.md` | `7f41a5822748a3bbecd56d94075d406c901230010369caa386e49ad74445ebc1` |
| `async_run/generations/gen_000000/manifest.json` | `2e101d7a6749e2185a53859a2667f784da4fdfa042b74fc8c8297a6e0c33cf50` |
| `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json` | `a0d1d53b0aa0d1244496fb0a1505486d18800ebd0020c153504d746645eea4e4` |
| Historical helper executable | `b7441676ff2d891e730b22cce2b8d3ece4d88283b27fb3b17d5dadfdea4b41f4` |
| Historical helper shared library | `b16e1c216ac4ab8a247e051789192ed6870a309c6a6bbca05cd61e24d0d6d3b8` |

The manifest has `validation_status=pass`, no validation errors, and 2048 of
2048 rank starts. The rank-start ledger has exactly 2048 records across 256
unique Frontier hostnames. The metrics record 2048 requested, participating,
quorum, and accepted ranks; zero failed, invalid, missing, stale, rejected, or
timed-out updates; 671,416,320 accepted tokens; and one advanced generation.

### Source/toolchain reconciliation

The run's `artifacts/env.txt` records:

- commit `40eb8d48e6dfe414aae3bfccf056904433aecdcb`;
- CPE 26.03 and `PrgEnv-gnu/8.7.0`;
- `gcc-native/14.2`, Cray MPICH 9.1.0, and libfabric 2.3.1.

The helper source at that commit hashes to
`fab9d0bbbe18322060ad2ee5279d252bc281e04b4cf4d347c0f1eef17952e4ac`,
not today's `4b84788e...`. The historical/current ELF files also differ.
Accordingly, the exact scale result is attached to the historical source,
binary, environment, command, and metrics hashes above. The current build and
tests prove that the evolved helper remains a working reference under the
current native toolchain; they do not silently transfer the old performance
number to the evolved binary.

## Performance baseline and metric scopes

Job 4974616 used 256 nodes, eight ranks per node, 2048 ranks, and 80 buckets.
Each rank contributed the 5,506,770,496-byte E97 dense delta. The logical
collective ingress was 11,277,865,975,808 bytes and the aggregate output was
5,506,770,496 bytes.

Two timings must remain separate:

- `merge_duration_s=5.304643992334604` is the reported Python-side
  merge/apply metric. Its payload rate is
  `5,506,770,496 / 5.304643992334604 = 1,038,103,688.759788 B/s`
  (1.038104 GB/s, 0.966809 GiB/s).
- `reduce_duration_s=73.9047` is the helper's sequential 80-bucket collective
  schedule and is also the recorded generation duration. Logical ingress rate
  is `11,277,865,975,808 / 73.9047 = 152,600,118,474.30542 B/s`
  (152.600118 GB/s). Normalized per rank contribution, the payload rate is
  `5,506,770,496 / 73.9047 = 74,511,776.598782 B/s`.

The per-bucket latency min/upper-median/max values are
`0.0366269 / 0.625382 / 5.42538` seconds. The first bucket already exceeds
the reported merge metric, further demonstrating why the 5.3046-second value
must not be mislabeled as the collective transport latency.

## Elastic backend acceptance targets

The machine-readable baseline carries the exact native-v1 gate constants, not
an inferred MPI acceptance rule.

The E97 native layout has 688,346,312 elements, 5,506,770,496 float64 bytes,
64 MiB maximum frames, and 83 shards. One persistent model-free C++17 service
runs per live node; eight trainers hand off through XPMEM/memfd; production
uses exact `cxi`/`FI_EP_RDM` with at least two distributed owners. Native v1
synthetic bytes and roots must be bitwise equal to the offline reference with
zero absolute tolerance.

Stage hard maxima are 10 seconds for local control bind, 30 seconds for
provider/endpoint/pools readiness, 30 seconds for endpoint install/probe, 180
seconds for local handoff, 180 seconds for frozen transfer including replay,
30 seconds for owner finalization, 180 seconds for redistribution/root
validation, at most 180 seconds for two-node checkpoint handoff/apply/
publication, and 30 seconds for drain (45-second process-kill bound). A
reassignment never resets its parent deadline.

The hard full-layout two-node synthetic G2 gate uses two node contributions
with weights 1,966,080 and 1,968,000, global weight 3,934,080, exactly
11,013,540,992 logical contribution bytes, and exactly 11,013,540,992 logical
redistribution bytes. After one warm-up, three timed generations must have:

- transfer-plus-redistribution p50 no greater than `98.961446568 s`;
- at least `222,582,457.59 logical B/s` over the combined
  22,027,081,984-byte numerator;
- no timed iteration over `118.753735882 s`;
- exact `cxi`, two owners, bitwise reference roots on both nodes, bounded
  credit/resident high-water, full release, and zero Python dense bytes,
  trainer files, disk replay, rejects, replay, or leaks.

No real model or 4+ native job is admissible before exact-code G2 passes. G3
then covers a real two-node generation; G4 covers failure/rejoin; G5 covers a
fresh allocation/fence; only then does G6 proceed in strict 4, 8, 32, 64, 256
order.

At G6's 256-node rung, contribution and redistribution are each
1,409,733,246,976 logical bytes. Elastic acceptance requires clean
reduction-plus-redistribution median no greater than
`10.609287984669208 s`, equivalent to at least
`265,754,544,322.50568 logical B/s` over both directions. Matching or beating
`5.304643992334604 s` is the performance target, equivalent to
`531,509,088,645.01135 logical B/s`; no sample may exceed 1.2 times its rung
median. Correctness without the 10.609-second cap leaves native CXI
unpromoted. A faster fixed-world result still fails the elastic semantics.

## Slurm decision

No `sbatch`, `salloc`, or `srun` command was issued by this task. The exact
reason is machine-recorded: the current helper built twice, the diagnostic and
focused tests passed, the executable resolved the active native libraries, and
`fi_info -p cxi` returned `cxi0` with status zero. Submitting the optional
two-node job after those conditions passed would have exceeded the task's
authorization. The existing two-node diagnostic script was neither submitted
nor presented as proposed work.

## Validation

Conformance checklist against **Resilient DiLoCo Compute Pool, version 1**,
native data-plane authority version 1, applicable matrix IDs **R03, R05, R06,
R08, R10, R14, R15, R16**, and native IDs **NDP02, NDP03, NDP05, NDP16,
NDP17**:

- [x] Current helper builds twice with current `CC`; executable and shared
  library outputs are byte-identical across the two artifact directories.
- [x] Singleton helper diagnostic passes with `MPI_THREAD_SERIALIZED`; all 12
  focused helper tests pass, including a real-current-`CC` parser build.
- [x] Active CPE component tuple, compiler, MPICH, libfabric, CXI library, and
  `cxi`/`shm`/`sm2`/`tcp` provider records are versioned and checksum-linked
  where a library file is available.
- [x] Job 4974616's manifest, metrics, environment, command, rank ledger,
  summary, generation manifest, finalization record, and historical binaries
  are checksum-linked.
- [x] The 2048-rank `5.304643992334604`-second merge evidence is preserved with
  its distinct 73.9047-second collective duration, exact payload/throughput
  calculations, historical source/toolchain identity, and fixed-world
  limitation.
- [x] The JSON baseline defines the 83-shard float64 payload, service topology,
  stage deadlines, G2/G6 latency and logical-throughput targets, bitwise
  correctness, failure semantics, and promotion boundary for the elastic
  backend.
- [x] R03/R06/R08 nonconformance is explicit: launched ranks are not READY
  membership, waits are collective, and owner loss/replay is absent.
- [x] R05/R15 numerical reference scope and R10 provider/link scope are
  explicit; no unexecuted network or mount claim is made.
- [x] R14/R16 and NDP16/NDP17 telemetry, latency/goodput, and ordered G0–G6
  gates are preserved; the old 256-node fixed-world job cannot bypass G2.
- [x] NDP02/NDP03/NDP05 are explicit: MPI is forbidden in the elastic binary,
  production must select exact `cxi`, and synthetic native results require
  bitwise equality rather than only approximate reducer parity.
- [x] No Slurm command was submitted and no large job was run.

Exact post-artifact validation commands:

```bash
jq empty reports/frontier/native-dataplane-reference-v1.json
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -q tests/test_native_dataplane_reference.py
git diff --check
```
