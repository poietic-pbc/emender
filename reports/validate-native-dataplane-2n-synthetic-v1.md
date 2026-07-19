# Frontier two-node native data-plane validation

**Task:** `validate-native-dataplane-2n-synthetic-v1`<br>
**Result:** **PASSED**<br>
**Final clean job:** `5031461` (`G2`)<br>
**Final changed fault job:** `5031553` (`G2-fault-rejoin-replay`)<br>
**Exact source:** `1c179e4ba014b2e54989a552fa5c99df010d7bbe` on authoritative `main`<br>
**Retained native bundle:** `411d7d92a5e23ea6838b370c0086265cd46d160878b4c2b1b8efc64e88293df1`

The exact E97 1.3B synthetic float64 layout passed on exactly two Frontier
nodes using two persistent native libfabric `FI_EP_RDM` endpoints with exact
provider/fabric/domain `cxi`/`cxi`/`cxi0`. The clean gate completed one
rejection warm-up and three measured generations; the median measured
contribution-plus-redistribution time was **23.219041387 seconds**, or
**948,664,573.05 logical bytes/s** and **4.2620814925x** the retained Python
throughput. The subsequent changed payload closed rank 1's endpoint, reassigned
and replayed the exact bounded work, reopened it at endpoint epoch 2 with a new
incarnation, rejected one old-owner-epoch input, reproduced the independent
exact result, and exposed no partial commit.

This is the synthetic G2/G2-fault admission gate. It authorizes the downstream
native wiring task; it does not claim a real-model generation, optimizer
commit, or checkpoint.

## Design authority and conformance scope

The normative authority reviewed before all changes and runs was
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../docs/RESILIENT_DILOCO_COMPUTE_POOL.md),
with the requirement inventory in
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../docs/RESILIENT_DILOCO_GAP_MATRIX.md).
The native specialization was checked against
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md).

Applicable compute-pool IDs: **R01–R16**. Applicable native data-plane IDs:
**NDP01–NDP17**. The emitted gates retain the complete ID lists, `Q_min=2`,
`T_min=3,934,080`, READY-membership rather than launched-rank admission,
bounded non-Lustre hot-path status, and atomic-result-or-no-commit status.
For R11/R12 and the commit-facing parts of NDP11/NDP12, this synthetic test
proves the no-partial-result/fence behavior only; the downstream real-model
task must still prove the durable optimizer/checkpoint integration.

## Retained build identity

The final pair used one retained native bundle and one retained final manifest;
there was no source change, rebuild, or evidence commit between clean job
`5031461` and fault job `5031553`. The controller/validator corrections after
the first throughput-passing candidate did not change the native bundle bytes.

| Item | Retained value |
|---|---:|
| Source commit | `1c179e4ba014b2e54989a552fa5c99df010d7bbe` |
| Manifest SHA-256 | `16daf9781253b1eff05afda7cfd72230adffbb3eea3b5713b3e00a2d907dcb6c` |
| Bundle SHA-256 | `411d7d92a5e23ea6838b370c0086265cd46d160878b4c2b1b8efc64e88293df1` |
| Synthetic gate binary | `77b34e9899d3fdbbfcd44d79cc44eaf100cff12bc55c36868b92928910e95c42` |
| Native service binary | `665652b23edf2a4600136f412828e388faa22624d0c65987ca4524a59c95b70a` |
| Local library | `80982b7e771832a2ac517d73f73d7e22e904d3b7e3e76182e8f7978b93c0f350` |
| Transport library | `e96080478f6c7ca933a8b04eeabbeec4357b1fc2e137ebf91ba88ce7fa8b1ded` |
| Local / transport ABI | `0x00010000` / `0x00010000` |
| Build | Cray `cc`/`CC`, `RelWithDebInfo`, XPMEM ON, tests ON |

The byte-exact retained manifest record is
[`native-artifacts-1c179e4.json`](frontier/native-dataplane/native-artifacts-1c179e4.json),
and the complete build/CTest output is
[`build-1c179e4.log`](../logs/frontier/native-dataplane/build-1c179e4.log).
The manifest records `source_tree_dirty=false`. Its artifact paths remain
relative to the original `build/native-resilient-dataplane` install prefix;
the archival copy preserves identity, while executable revalidation uses the
materialized original prefix.

Exact build and local validation commands:

```bash
PYTHON_BIN="$PWD/.envs/olcf-rocm711-torch210-py312/bin/python" \
BUILD_JOBS=16 \
scripts/frontier/build_native_resilient_dataplane.sh \
  2>&1 | tee logs/frontier/native-dataplane/build-1c179e4.log

NDM_PIN_TRITON_AUTOTUNE=0 \
EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1" \
EMENDER_NDP_TRANSPORT_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1" \
"$PWD/.envs/olcf-rocm711-torch210-py312/bin/python" -m pytest -q \
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
  tests/test_validate_native_dataplane_local.py \
  tests/test_validate_native_dataplane_2n_gate.py \
  tests/test_native_dataplane_2n_controller.py
```

Native CTest passed **8/8**. The installed native/resilient Python suite passed
**121/121 in 156.52 seconds**. The added failing-first regressions cover direct
local contribution/result application, delayed/reverse-arrival clock sampling,
exact clean-versus-fault physical accounting, off-by-one replay rejection, and
approved-Python pre-submission attestation.

## Exact Frontier commands and allocation timing

Final clean submission:

```bash
NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

The wrapper emitted the retained `sbatch` command in
[`submission.json`](frontier/native-dataplane/5031461/submission.json): account
`bif148`, partition `batch`, QoS `debug`, exactly `-N 2`, walltime
`00:20:00`, and allocation network `job_vni`. It exported layout
`e97-f64-5506770496`, eight local trainers, weights `1966080,1968000`, one
warm-up, three measured generations, and exact `FI_PROVIDER=cxi`.

Final changed fault submission:

```bash
NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
NDP_CLEAN_GATE_JSON="$PWD/reports/frontier/native-dataplane/5031461/full-layout-gate.json" \
scripts/frontier/submit_native_dataplane_2n_gate.sh fault
```

The fault wrapper verified the passing clean gate, exact source/bundle, and a
changed payload ID before `sbatch`. It exported one rejection warm-up plus one
fault generation. Exact Slurm accounting is retained in each job directory.

| Gate | Job | Nodes | Submitted | Started | Ended | Queue | Runtime | Step | Exit |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| G2 clean | `5031461` | `frontier[00122-00123]` | 01:39:09 | 01:39:34 | 01:42:25 | 25 s | 171 s | 122 s | `0:0` |
| G2 fault | `5031553` | `frontier[00122-00123]` | 01:43:28 | 01:43:37 | 01:45:30 | 9 s | 113 s | 66 s | `0:0` |

Times are America/New_York on 2026-07-19. The same nodes were allocated by
chance; the second job was an independent allocation and changed payload, not
a continuation inside the clean allocation.

## Provider and endpoint attestation

Both final jobs ran a single two-rank native step, one rank per node. Each rank
opened a persistent `FI_EP_RDM` endpoint and attested exact provider `cxi`,
fabric `cxi`, and domain `cxi0`; no provider fallback was accepted. Allocation
and step logs show `job_vni` plus `SLINGSHOT_DEVICES`, `SLINGSHOT_SVC_IDS`,
`SLINGSHOT_TCS`, and `SLINGSHOT_VNIS`. The paired first-observed-hello clock
offset delta was 0.085080 ms for clean and at most 0.129761 ms across the two
fault phases.

Fault phase 0 retained two distinct endpoint incarnations. In phase 1 rank 0's
endpoint record remained byte-identical at epoch 1, while rank 1 changed from
incarnation `552022c8d08c551ee59617d598a6834d` to
`6f4658149756f64675da5608f70a88ab` and advanced endpoint epoch 1 to 2.

## Exact layout, reduction, ownership, and reference result

The gate exercised exactly:

- 5,506,770,496 layout bytes / 688,346,312 little-endian float64 elements;
- 83 shards with a 67,108,864-byte maximum payload;
- eight deterministic local lanes per node;
- 22,027,081,984 local-reduction input bytes per node;
- node weights 1,966,080 and 1,968,000, global weight 3,934,080;
- 11,013,540,992 logical contribution bytes and 11,013,540,992 logical
  redistribution bytes per generation;
- deterministic distributed owners, bounded four-slot contribution/fetch
  pipelines, result redistribution to both nodes, and explicit release.

The independent Python analytical reference computes alternating exact values
`11.753904343582235` and `12.253904343582235`. Every measured generation on
both nodes matched the independent per-shard roots and full payload digest
`e8d95487da2e901938bc5cdeb08fd5cceba9b537ec07eb63a167d27c0037c304`.
The clean warm-up rejected exactly one stale and one corrupt frame at each
endpoint (two stale and two checksum rejects total); timed clean generations
had no rejection.

## Clean throughput, wire, CQ, and resource results

| Metric | Observed | Required |
|---|---:|---:|
| Measured samples | 23.259562736, 23.219041387, 23.150270638 s | 3 samples |
| Median | **23.219041387 s** | <= 24.740361642 s |
| Maximum | 23.259562736 s | <= 118.753735882 s |
| Logical throughput | **948,664,573.05 B/s** | >= 890,329,830.36 B/s |
| Speedup over retained Python | **4.2620814925x** | >= 4x |
| Useful TX / RX | 44,322,599,424 / 44,322,599,424 B | exact accounting |
| Wire TX / RX | 44,323,138,304 / 44,323,138,304 B | recorded |
| Retries | 1,895 | recorded, bounded completion |
| CQ / route errors | 0 / 0 | 0 / 0 |

The clean result admitted at most 14,372,851,648 owner-resident bytes against
the 14,440,737,184-byte owner bound. Transport in-flight high-water was
134,219,008 bytes, retained high-water was 268,436,736 bytes, and process RSS
high-water was 14,695,383,040 bytes against a 20,215,943,136-byte process
bound. Release accounting recorded 88,646,276,608 bytes. Terminal transport
in-flight plus retained bytes was exactly zero, and post-release RSS returned
within the required floor tolerance.

## Changed peer-loss/rejoin/replay result

Fault job `5031553` depended on clean gate SHA-256
`daf47bcb3307756c501264eafb6a01674a80e74cc4bd3f60ec97c9e1977cd6f1`
and changed the payload from `...-clean-20260719T053908Z` to
`...-fault-20260719T054304Z`.

The native sequence primed the route, closed rank 1's provider endpoint,
reassigned ownership from epoch 1 to epoch 2, retained and replayed the bounded
work, reopened/rejoined rank 1 with a new incarnation, rejected the old epoch,
and only then exposed the complete exact result. Observed invariants:

- exactly one reassignment on each rank;
- exactly 134,217,728 logical replay bytes on each rank (two payload maxima);
- exactly one 67,108,864-byte remote wire replay, with the other replay shard
  applied to its local native owner;
- exact physical contribution `5,573,879,360 = layout + payload_max` bytes;
- exact physical redistribution 5,506,770,496 bytes;
- one old-owner-epoch rejection total;
- rank 1 endpoint epoch 1 -> 2 and a new incarnation;
- `partial_commit=false` on both ranks;
- exact independent result/root match;
- zero CQ and route errors;
- terminal in-flight and retained transport bytes zero;
- post-release RSS within the required floor.

Fault telemetry recorded 16,788,950,784 wire TX bytes,
16,721,841,600 wire RX bytes, 2,361 bounded provider retries, and
33,510,792,384 released bytes. Fault duration is reported for diagnosis but is
not substituted for the already-passed three-sample clean performance gate.

## Forbidden-path audit

The retained gates report:

- `python_dense_socket_bytes=0`; Python handled bounded endpoint metadata only;
- `mpi_collectives=0` and `all_rank_barriers=0`;
- `trainer_spool_bytes=0`, `trainer_spool_files=0`, and `disk_replay_bytes=0`;
- `handoff_full_copy_bytes=0` and no central full-model broker;
- no GPU training or model load (the Frontier allocation contains GPUs as a
  node resource, but the payload launched one CPU native service per node);
- exactly two nodes, never four or more;
- dense contribution, aggregation, replay, and redistribution stayed in
  bounded native memory/CXI paths rather than Lustre.

## Preserved failure and correction ledger

Every failed Slurm payload was retained before the next changed payload. Each
listed directory contains the exact submission, available native evidence,
Slurm accounting, failure description/correction, logs, and checksum manifest.

| Job | Queue / runtime | Exact failure boundary and scoped correction |
|---:|---:|---|
| `5030760` | 26 / 80 s | Allocation had no VNI; request `job_vni` at allocation. |
| `5030810` | 27 / 4 s | Batch-scope Slingshot environment was checked too early; attest inside the native step. |
| `5030839` | 27 / 76 s | Repeating the allocation VNI request on `srun` failed interconnect setup; inherit the allocation VNI. |
| `5030860` | 8 / 79 s | `single_node_vni` interfered with the multi-node step; retain only `job_vni` and run the metadata controller in the batch shell. |
| `5030889` | 33 / 76 s | Exact CXI probe found `cxi0` and `cxi1`; bind and attest the approved `cxi0` domain. |
| `5030957` | 30 / 38 s | Routed traffic failed after an incorrect FI source contract; restore the provider MR/source contract. |
| `5031006` | 39 / 83 s | One peer entered redistribution while the other still contributed; add an authenticated result-announcement fence. |
| `5031056` | 8 / 90 s | One peer advanced generations before late result fetches; add an authenticated generation-goodbye fence. |
| `5031115` | 21 / 269 s | Full native work passed, but resident-bound validation and 49.272852589 s performance did not; correct rank-specific admission and profile native copies/hashes. |
| `5031145` | 606 / 328 s | Apparent 1.438 s launch-time "skew" and 34.555993958 s median; measure clock-offset delta and remove redundant native payload validation. |
| `5031273` | 12 / 222 s | Correctness passed; 36.502354161 s / 2.7111x failed throughput; bypass the synthetic local result wire loop. |
| `5031382` | 42 / 180 s | Correctness passed; 26.321730799 s / 3.7597x still failed; bypass the synthetic local contribution wire loop with equivalent bounds/order/finite checks. |
| `5031429` | 14 / 151 s | Throughput passed at 19.909507535 s / 4.9706x, but a delayed phase read produced 266.662 ms apparent offset; pair client clocks with first controller observations. |
| `5031454` | 36 / 113 s | Peer loss/rejoin/replay passed natively; validator applied clean physical-byte equality to the intentional replay; require exact fault `layout + one remote payload` and retain two-payload logical replay per rank. |

Intermediate clean job `5031449` passed at 20.400408822 seconds and 4.851x,
then was intentionally superseded because the clock/validator source had to
change before an exact same-source fault pair could be accepted. Its evidence
is retained rather than silently discarded.

## Validation

- [x] **R01–R16 / NDP01–NDP17 conformance checklist reviewed and embedded in
  both final gates.** The synthetic boundary and downstream durable-commit
  obligations are stated explicitly.
- [x] Runtime attests exact native provider/fabric/domain `cxi`/`cxi`/`cxi0`,
  `FI_EP_RDM`, two persistent endpoints, exact source/bundle/artifact hashes,
  `job_vni`, and no provider fallback.
- [x] Exact weighted result matches an independently computed analytical and
  per-shard reference on both nodes; corrupt and stale inputs are rejected.
- [x] Clean contribution plus redistribution passes the 24.740361642-second
  target and 4x retained-Python threshold with CQ, wire, useful-byte, retry,
  and release metrics.
- [x] Owner admission, process RSS, transport in-flight/retained high-water,
  and released bytes are bounded; terminal transport bytes are zero and RSS
  returns to the post-release floor.
- [x] Changed payload performs real rank-1 close/reopen, epoch-2 rejoin,
  reassignment, exact replay, one old-epoch reject, exact result, and no partial
  commit.
- [x] No Python dense payload, MPI collective/all-rank barrier, Lustre dense hot
  path, trainer spool, disk replay, GPU training, central full-model broker, or
  allocation larger than two nodes was used.
- [x] Exact commands, job IDs, queue/runtime timing, nodes, hashes, raw logs,
  accounting, node metrics, membership, final gates, failure evidence, and
  checksum manifests are retained for authoritative-main commit/push.

Machine-readable aggregate metrics are in
[`validate-native-dataplane-2n-synthetic-v1-metrics.json`](validate-native-dataplane-2n-synthetic-v1-metrics.json).
Raw final evidence is in
[`5031461`](frontier/native-dataplane/5031461) and
[`5031553`](frontier/native-dataplane/5031553); the independently verifiable
SHA-256 manifests in those directories include their corresponding build and
Slurm logs.
