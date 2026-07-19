# Native data-plane local v1 qualification

Date: 2026-07-19 (UTC)
Task: `validate-native-dataplane-local-v1`
Decision: **PASS for the complete non-Slurm local/G0 gate**
Code-under-test commit: `b1f4229c9252069e6538506c0f09f20f88de82c2`

This qualification passes the local prerequisite for downstream two-node
synthetic work. It does **not** claim a Frontier G1/G2 result, production CXI
promotion, or completion of the live native split-role integration gap recorded
in the audit. No `sbatch`, `srun`, `scancel`, or other Slurm operation was run.
The exact machine-readable stress and full-layout record is
[`validate-native-dataplane-local-v1-metrics.json`](validate-native-dataplane-local-v1-metrics.json),
SHA-256
`24274eaad9ce45f39961d4dc6d8d9c67729bedd41c63b129f185cb6298a0faa6`.

## Outcome

All required gates passed after two focused reproducible-packaging fixes:

| Gate | Final result |
|---|---:|
| Canonical unified native RelWithDebInfo CTest | 8/8 passed |
| ASan + UBSan + leak-detection native CTest | 8/8 passed |
| Independent short-path reproducibility build CTest | 8/8 passed |
| Independent long-path reproducibility build CTest | 8/8 passed |
| Full native/resilient Python selection | 107/107 passed in 140.03 s |
| Repeated exact-reference native stress | 384/384 measured generations passed |
| Warm-up generations included in lifecycle/resource accounting | 24/24 passed |
| Fresh-process clean restart | 128 measured + 8 warm-up generations passed |
| Exact E97/256-node arithmetic assertions | all passed |
| Canonical/short/long installed artifact and manifest `cmp` | byte-identical |
| ABI/SONAME/export/provider/package checks | all passed |

The repeated-generation gate used two concurrently spawned OS processes and a
third newly spawned process after the first pool exited. Each process completed
8 warm-ups followed by 128 consecutive measured native generations. The real
provider CTest separately used two OS processes and `FI_EP_RDM` through explicit
`tcp;ofi_rxm`. Every stress generation compared native output byte-for-byte
against the deterministic Python v1 weighted reference. The expected and
observed result SHA-256 in every process was
`a2b008301ea629dd23c92c2fe703a13663cc668d2096e29395e1a140ffce18c2`.

### Repeated-generation resource result

| Process | Measured / warm-up | FDs during run | Threads during run | RSS plateau growth | RSS final growth | Native admitted / released | Native current shared / mapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| worker 0 | 128 / 8 | 17 / 17 | 65 / 65 | 0 B | 0 B | 6,684,672 / 6,684,672 B | 0 / 0 B |
| worker 1 | 128 / 8 | 17 / 17 | 65 / 65 | 0 B | 0 B | 6,684,672 / 6,684,672 B | 0 / 0 B |
| clean restart worker 2 | 128 / 8 | 17 / 17 | 65 / 65 | 0 B | 0 B | 6,684,672 / 6,684,672 B | 0 / 0 B |

All three clients dropped from 17 to 16 descriptors when closed. Each reported
exactly 136 projections, zero buffer exhaustion, zero retained current bytes,
and zero trainer-spool, Python-dense-socket, handoff-copy, or disk-replay bytes.
The final evidence therefore has no descriptor, thread, resident-memory
plateau, or retained-byte growth across more than the required 100 consecutive
generations, including a fresh-library/process restart.

## Fault, fallback, replay, and restart coverage

The final native and Python selections jointly cover the requested scenarios:

| Scenario | Passing evidence |
|---|---|
| Peer kill/removal and route restart | `ndp_fabric_multiprocess_test` abruptly exits and removes a real provider peer; `test_restart_replays_retained_bucket_and_catches_up`; `test_supervisor_kills_only_stuck_step_and_healthy_step_survives` |
| Owner loss and bounded replay | `test_ready_token_floor_distributed_owner_loss_and_late_join` kills the selected owner after receipt, remaps over surviving READY peers, and replays sender-retained chunks |
| New-incarnation rejoin | The same pool test drains the old incarnation, admits `i2-rejoined` only for the next generation, and rejects the superseded incarnation |
| Identical and conflicting duplicate replay | Native owner/protocol and multiprocess tests prove once-only application and stable identical receipts; `test_stale_duplicate_and_corrupt_contribution_receipts` and `test_nonfinite_and_conflicting_duplicate_payloads_are_rejected` prove conflict rejection |
| Corrupt frame / payload / nonfinite input | Native CRC32C/SHA-256 corruption cases plus `test_corruption_and_nonfinite_are_rejected_before_accumulator_mutation`, checksum/oversize spool tests, and apply corruption tests |
| Timeout / bounded wait | Native expired-deadline and cancellation paths, `test_owner_rpc_io_timeout_scales_with_bounded_frame_size`, `test_stale_epoch_rejected_and_quorum_deadline_fails_closed`, and stage-SLO tests |
| Newer fence and clean restart | `test_newer_fence_cancels_older_generation_and_preserves_incarnation_boundary`, `test_fenced_atomic_global_commit_and_newer_allocation_restart`, and `test_fresh_process_restart_matches_uninterrupted_continuation` |
| Provider fallback and fail-closed production selection | `ndp_test_provider_explicit` accepts only explicit local `tcp;ofi_rxm`; production/local-provider mismatch, test-mode `cxi`, and weakened `FI_PROVIDER` cases all fail closed; the installed service probe reports `tcp;ofi_rxm`, `FI_EP_RDM`, and `production_provider=false` |
| Exact reference and arrival independence | Native F32/F64/BF16 unequal-weight permutations plus every one of the 408 stress generations; all use zero-tolerance comparison |
| Atomic publication / no partial result | Stale result fence, mismatched root, missing CAS, stale allocation, and newer allocation tests leave the previous immutable latest authoritative until the exact current-fence publication succeeds |

The concrete pool floor in the owner-loss/late-join gate is `Q_min=2`,
`T_min=10`, with READY fraction `0.5`; it commits from two complete
contributions while a late/missing launched peer is excluded. No test derives
progress from launched rank count or uses a failure-sensitive all-rank
operation.

## Exact full-E97 and Frontier projection

The local payload was deliberately scaled to 4,096 elements per process. The
same executable gate separately asserts the following exact integers from the
v1 authority, without allocating the full 1.3B payload:

| Quantity | Machine-checked value |
|---|---:|
| E97 elements | 688,346,312 |
| Float64 layout bytes | 5,506,770,496 |
| Maximum payload | 67,108,864 B |
| Shards | 83 = 82 full + 3,843,648 B final |
| Frontier nodes / trainer lanes | 256 / 2,048 |
| Logical contribution bytes | 1,409,733,246,976 |
| Logical redistribution bytes | 1,409,733,246,976 |
| Total logical dense bytes | 2,819,466,493,952 |
| Frames per direction / total | 21,248 / 42,496 |
| Header bytes per direction / total | 6,799,360 / 13,598,720 |
| Routes per service / directed cluster routes | 255 / 65,280 |
| Receipt ledger bound | 2,719,744 B |
| Four-TX/four-RX registered slot pool | 536,873,472 B |
| Two-owner assignment bound | 2,820,494,112 B |
| Two-owner resident admission bound | 14,440,737,184 B |
| 256-owner assignment / resident bounds | 88,619,687 / 11,708,862,759 B |
| One trainer F32 lane / eight lanes | 2,753,385,248 / 22,027,081,984 B |
| Current local ABI default | 17,179,869,184 B |

The arithmetic deliberately records `eight_lanes_fit_local_default=false`:
eight simultaneously retained full-E97 F32 lanes exceed the current 16 GiB
local default. That is a fail-closed live-integration configuration constraint,
not evidence that the downstream full-layout job may ignore lane lifetime or
raise a bound without accounting. The native service resident formula itself
fits the recorded two-owner bound, uses no central full-model broker, and is
independent of launched world size.

## Reproducible packaging and ABI attestation

The initial independent RelWithDebInfo builds exposed absolute build-directory
DWARF bytes. Prefix mapping fixed both libraries, after which a stronger
canonical-versus-fresh comparison exposed install-time RPATH padding in the
service binary. The final focused fixes make source/debug paths virtual and use
fixed origin-relative build/install RPATHs. Clean build paths with materially
different lengths now produce identical installed bytes, and the canonical
package is identical to both.

The clean manifest has schema `emender-native-dataplane-build-v1`, source commit
`b1f4229c9252069e6538506c0f09f20f88de82c2`, `source_tree_dirty=false`, protocol
`1.0`, and both ABI values `0x00010000`. The manifest SHA-256 is
`5e986edd69492d7aa4eec8efc4979970ba2aceb990136619b743fc8bb9928178`;
its bundle SHA-256 is
`66ffaf0cbcb6c873f0c4202bb01421234f6bb4515fd156cf22f86cc91cba8b62`.

| Installed artifact | Bytes | SHA-256 |
|---|---:|---|
| `libemender_ndp.so.1` | 985,472 | `80982b7e771832a2ac517d73f73d7e22e904d3b7e3e76182e8f7978b93c0f350` |
| `libemender_ndp_transport.so.1` | 1,150,768 | `d1e2b6aac59be2ccb54c81dd00c176d1943a1e07869874bac3f550c714a6fdac` |
| `ndp_cxi_service` | 783,072 | `9a87fb7102541303bd2ca3fc0859b0b852bda86ead30b5d863a4655ea8910298` |

`readelf` confirms SONAMEs `libemender_ndp.so.1` and
`libemender_ndp_transport.so.1`. The installed service resolves the package
through `$ORIGIN/../lib64`; its live local probe succeeds. Dynamic symbol scans
find no `MPI_` or `PMPI_` export in either elastic library. The canonical
attestation verifier returns `status=attested` for backend `native-test` and
binds all three digests above. It intentionally provides no G2 artifact and
cannot promote a local provider to production.

Focused commits before this report are:

- `58b1b10`: add the reusable 100+ generation/resource/layout gate and unit
  checks;
- `d8b2f3e`: remove build-directory dependence from debug-bearing package
  artifacts; and
- `b1f4229`: remove install RPATH path-length dependence and retain a portable
  origin-relative package lookup.

## Validation

This task applies the Resilient DiLoCo Compute Pool authority, version 1, to
**R01–R16** and the Native resilient DiLoCo data-plane authority, version 1, to
**NDP01–NDP17**. The local gate confirms the mechanisms applicable at G0 and
preserves the gap-matrix distinctions for live split-role wiring and Frontier
evidence.

Conformance checklist:

- **R01–R03, R06, R11, R13, R14, R16 / NDP02, NDP07, NDP13, NDP16, NDP17:**
  leased READY snapshots, bounded deadlines, late join, disappearance,
  new-incarnation rejoin, current/newer fences, route-local peer failure, clean
  restart, and structured evidence pass without launched-rank or all-rank
  invariants. G1/G2 remain downstream and no scale rung is authorized here.
- **R04, R05, R07, R12, R15 / NDP05, NDP06, NDP10, NDP15:** exact fenced
  identities, deterministic token-weighted arithmetic, bitwise reference
  equality, identical duplicate receipts, conflicting/stale/corrupt rejection,
  atomic current-fence publication, outer-state/latest restart, and continuous
  committed state pass.
- **R08–R10 / NDP01, NDP03, NDP04, NDP08, NDP09, NDP11, NDP12, NDP14:** local
  component handoff, registered provider pools, byte/slot credit, bounded
  replay/reassignment, shared redistribution, prompt release, and the absence
  of MPI symbols or Python/spool dense bytes pass. This is component/local
  evidence only; the audited live split-role production gap remains fail-closed.
- **Backend and hot path:** the local provider is explicit `tcp;ofi_rxm` over
  `FI_EP_RDM`, never production-labelled; dense stress uses memfd-backed native
  buffers; all spool/Python-dense/handoff-copy/disk-replay counters are zero;
  no Lustre payload or central full-model broker is used.
- **Progress floor and failure:** `Q_min=2`, `T_min=10`, capped READY fraction
  `0.5`; owner loss, peer loss, replay, duplicate, corruption, nonfinite input,
  deadline, stale/newer fence, and clean restart all pass or fail closed with no
  partial commit.
- **Artifacts:** commands, counts, exact arithmetic, resource samples, source
  commit, ABI, provider fact, binary digests, and terminal decision are retained
  here and in the machine-readable metrics. No Slurm or committed Frontier
  generation/checkpoint artifact is claimed by this local task.

Exact final command shapes were:

```bash
# Clean canonical package and normal native suite (8/8)
BUILD_DIR="$PWD/build/native-resilient-dataplane-build" \
INSTALL_DIR="$PWD/build/native-resilient-dataplane" \
PYTHON_BIN=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
BUILD_TYPE=RelWithDebInfo \
bash scripts/frontier/build_native_resilient_dataplane.sh

# Clean sanitizer native suite (8/8)
cmake -S native -B build/validate-native-dataplane-local-v1-asan-final2 \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DNDP_BUILD_TESTS=ON \
  -DNDP_ENABLE_SANITIZERS=ON -DNDP_ENABLE_XPMEM=ON
cmake --build build/validate-native-dataplane-local-v1-asan-final2 --parallel 8
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build/validate-native-dataplane-local-v1-asan-final2 \
  --output-on-failure

# Two clean RelWithDebInfo trees used differently shaped build paths; each was
# built, tested 8/8, installed, manifest-recorded, then compared with cmp.
cmake -S native -B <short-or-long-clean-build> \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=<matching-clean-prefix> \
  -DBUILD_TESTING=ON -DNDP_BUILD_TESTS=ON -DNDP_ENABLE_XPMEM=ON
cmake --build <short-or-long-clean-build> --parallel 8
ctest --test-dir <short-or-long-clean-build> --output-on-failure
cmake --install <short-or-long-clean-build>
python scripts/frontier/attest_native_dataplane.py record-build \
  --prefix <matching-clean-prefix> --source-root "$PWD" \
  --cmake-cache <short-or-long-clean-build>/CMakeCache.txt

# 384 measured exact-reference generations plus 24 warm-ups
python scripts/validate_native_dataplane_local.py \
  --library build/native-resilient-dataplane/lib64/libemender_ndp.so.1 \
  --output reports/validate-native-dataplane-local-v1-metrics.json \
  --generations 128 --warmup-generations 8 --elements 4096 --workers 2

# Full native/resilient Python suite (107/107)
EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1" \
EMENDER_NDP_TRANSPORT_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1" \
python -m pytest -q \
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
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_validate_native_dataplane_local.py
```

Final ancillary checks also passed: Python `compileall`; shell syntax; JSON
schema and invariant queries; exact-source manifest verification; installed
provider probe; SONAME, RPATH, forbidden-symbol, and absolute-build-path scans;
canonical/short/long artifact and manifest comparisons; `AGENTS.md`/`CLAUDE.md`
identity; and `git diff --check`.

## Promotion boundary

Local G0 is complete. The next task may use this exact commit/bundle as the
local prerequisite for a **two-node synthetic** Frontier rung, subject to the
live split-role dependency and all NDP17 admission checks. This report does not
provide exact `cxi`, G1, G2 full-layout performance, a committed real model
generation, or a checkpoint. Production and any 4+ node run remain
unauthorized until the ordered downstream gates retain those artifacts.

No Slurm job was submitted.
