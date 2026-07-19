# Native resilient pool hyperscale-local validation

**Task:** `validate-native-pool-v1-hyperscale-local`

**Decision:** PASS for the scheduler-free G0 backend conformance fixture

**Validated implementation commit:** `557a643d2f11d8d999bc005920a3b514751440a4`

**Native bundle SHA-256:**
`66ffaf0cbcb6c873f0c4202bb01421234f6bb4515fd156cf22f86cc91cba8b62`

**Machine evidence:**
[`validate-native-pool-v1-hyperscale-local-metrics.json`](validate-native-pool-v1-hyperscale-local-metrics.json)

## Scope and authority

This report validates a backend-neutral local adapter against both normative
authorities:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1 (2026-07-17),
  requirements R01-R11 and R13-R15;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, version 1 (2026-07-18),
  requirements NDP01-NDP16.

NDP17 is deliberately not promoted by this result. The adapter selects only
`native-test` with explicit provider `tcp;ofi_rxm`. Frontier production remains
fail-closed on `native-cxi`, exact provider `cxi`, and the ordered G1/G2 and
later scale gates. This was a local test only: no `sbatch`, `srun`, `salloc`, or
other Slurm submission/launch command was executed.

The minimum progress policy for this gate was `Q_min=2`, `T_min=3`, strict
freshness (`tau=0`), finite 180-second generation and 60-second native stage
bounds, and one exclusive expiring run lease.

## Adapter boundary

`ndm/hyperscale_local_adapter.py` adds scheduler mapping, not a new pool
protocol. It composes existing authoritative components:

1. `SQLiteFencedControlStore` acquires the exclusive allocation lease before
   artifact inspection, pool-server creation, native process startup, model
   loading, or run evidence writes.
2. `PoolControlServer`, `PeerMembership`, `GenerationAdmission`, and
   `ContributionReceipt` own DISCOVER/BOOT/SYNC/READY membership, the live
   generation snapshot, fenced contribution admission, and deterministic
   close. No launched worker count is supplied anywhere.
3. Every simulated host is a separate spawned process with exactly one
   `NativeManagerSession`, matching the native service's one-controller-per-host
   ABI contract. Python exchanges endpoint and command metadata only.
4. A child allocates each native memfd contribution buffer and passes only a
   duplicated descriptor to the producer. The producer fills that mapping
   directly; contribution bytes are never serialized through the Python
   control socket.
5. `TensorLayout.owner` assigns one bounded shard to each owner in this small
   fixture. Frozen contribution identities and weights are applied by the
   native float64 accumulator. Only read-only result shard views return to the
   Python checkpoint publisher; no owner or control server brokers a full set
   of contributions.
6. Each owner emits a fenced checkpoint proposal. Python reload-verifies the
   complete checkpoint and publishes commit, checkpoint, and authoritative
   `latest` records in one live-fence SQLite transaction. The global manifest
   carries every nested owner-result identity; each native owner verifies its
   own result before accepting `COMMIT` and releasing state.

The validation uncovered and fixed two real lifecycle defects while building
this adapter:

- native local service controllers cannot share one process, so local host
  agents are process-isolated rather than mocked as multiple in-process
  controllers;
- SQLite connection context managers do not close descriptors. All internal
  `SQLiteFencedControlStore` connections now use `contextlib.closing`, and a
  32-transaction FD regression test proves zero descriptor growth.

## Retained gate result

The retained 12-cycle command committed generations 0 through 14:

- generation 0 opened with `node-0` and `node-1`; `node-2` appeared after the
  snapshot and received `rejected_not_ready` without changing the open world;
- generation 1 committed with `node-0` and `node-2` after `node-1`
  disappeared;
- generation 2 admitted `node-1` only under the new incarnation
  `node-1-rejoin-1` and committed with three owners;
- generations 3-14 each opened attempt 1, lost a frozen owner, verified that no
  commit publication existed, started that stable worker under a new
  incarnation, and committed attempt 2;
- the winner held fence 1, the simultaneous allocation loser exited status 0
  with zero native starts/model loads/data-plane bytes, and the successor
  acquired fence 2 only after the winner released the lease.

There were 15 committed generations, 12 deliberately failed attempts with no
publication, 16 bounded native process departures, and 85 accepted tokens.
Every committed vector was compared byte-for-byte after float32 result encoding
with the deterministic float64 token-weighted reference. Every checkpoint was
reloaded and SHA-256 verified before the atomic metadata bundle advanced.

Representative evidence:

| Property | Retained result |
|---|---|
| Late join | third receipt was `rejected_not_ready`; open snapshot remained two peers |
| Disappearance | next generation committed with two surviving READY peers |
| Rejoin | same stable worker admitted only as a new incarnation |
| Owner loss | 12/12 attempt-1 failures produced no commit; 12/12 attempt-2 retries committed |
| Lease exclusion | loser status 0; native starts/model loads/dense bytes all 0 |
| Fence takeover | winner fence 1; post-release successor fence 2 |
| Progress floor | `Q_min=2`, `T_min=3`; launched world is `null` |
| Native provider | explicit `tcp;ofi_rxm`, `production_provider=false` |
| Frontier policy | exact-CXI gate unchanged |
| Scheduler work | Slurm jobs submitted: 0 |

## Native resource bounds

Resource samples include the parent control adapter and all three independent
native host processes. Sampling was taken after each replacement and commit,
when the live shape must be constant.

| Metric | Baseline | Min | Max / final | Decision |
|---|---:|---:|---:|---|
| Live processes | 4 | 4 | 4 | fixed bound passed |
| File descriptors | 97 | 97 | 97 | exact plateau passed |
| Threads | 267 | 267 | 267 | exact plateau passed |
| RSS, first sample window | — | — | 2,173,579,264 bytes | bounded |
| RSS, last sample window | — | — | 2,177,261,568 bytes | +3,682,304 bytes; below 256 MiB bound |
| Parent FDs after close | 8 before | — | 9 after | bounded process-global tracker only |
| Parent threads after close | 65 before | — | 65 after | fully released |

All 16 native terminal records had:

- `shared_bytes_current=0` and `mapped_bytes_current=0`;
- transport `in_flight_bytes=0` and `retained_bytes=0`;
- `trainer_spool_bytes=0`, `trainer_spool_files=0`;
- `python_dense_socket_bytes=0`, `handoff_full_copy_bytes=0`;
- `disk_replay_bytes=0` and no nonfinite/checksum/conflict rejection.

This gate exercises the clean abort/restart choice for a missing frozen owner.
The inherited native-local qualification remains the retained evidence for
bounded credit/replay mechanics; the adapter does not invent a new replay
policy or weaken its byte/deadline caps.

## Conformance checklist

### Compute Pool v1

- **R01:** winner/loser/successor use the same durable lease CAS. The losing
  path returns before manifest read, evidence-directory creation, native start,
  or model/data-plane callback.
- **R02/R03/R11:** the live control server executes the existing leased state
  machine. Late workers do not alter an open snapshot; disappearance removes a
  worker from later snapshots; return requires a new incarnation. No launch
  count, rank, barrier, or all-peer wait exists.
- **R04/R06:** contributions carry the full fence/generation/attempt/stable
  worker/incarnation/sequence identity. The late contribution receives the
  normative rejection receipt; `Q_min/T_min` closure is bounded.
- **R05/R15:** deterministic sharded token-weighted native results match the
  high-precision reference for every commit, including unequal weights and
  changing participation. The committed accepted-token total is 85.
- **R07:** immutable checkpoint bytes and manifests precede one atomic
  commit/checkpoint/latest transaction. A failed owner attempt publishes none
  of those records.
- **R08/R09/R10:** owner selection is deterministic and shard-bounded; host
  services are model-free; native memfd handoff carries dense bytes without
  Lustre, trainer spool, or Python TCP. Python sees read-only aggregate shards
  only for checkpoint publication.
- **R13:** the adapter has no scheduler import or launched-world parameter and
  uses the exact pool contracts. `RESILIENT_DILOCO_GAP_MATRIX.md` now records
  the G0 backend fixture as present.
- **R14:** lease, membership, generation, native, checkpoint, and shutdown
  stages have explicit finite bounds. JSON evidence identifies every terminal
  reason, owner, checkpoint, resource sample, and fence.

### Native data plane v1

- **NDP01/NDP14/NDP15:** Python owns leases, membership, freeze, owners, and
  atomic publication. One model-free native service per host owns memfd
  reduction, endpoint routes, native lifetime, and read-only result handoff.
- **NDP02/NDP03/NDP07/NDP13:** all endpoint routes are current-fence,
  point-to-point, independently removable, and bounded. The test provider is
  explicit and cannot be promoted; production still requires exact `cxi`.
- **NDP04/NDP08/NDP09/NDP12:** producer-direct memfd buffers, fixed payload and
  resident bounds, deterministic sharded owners, and one read-only result per
  shard are used. Resource and release counters are retained.
- **NDP05/NDP06/NDP10:** native deterministic weighted math and complete fenced
  identities/result roots are verified. Owner result identities are nested in
  one global manifest and checked before native commit.
- **NDP11:** missing-owner attempt 1 aborts without publication and the sender
  retries only under a new attempt/incarnation; inherited native component
  tests retain the credit/replay/reassignment byte-bound evidence.
- **NDP16:** provider, artifact, fence, membership, receipt, owner,
  checkpoint, terminal counter, and process resource telemetry are present in
  the machine JSON.
- **NDP17:** explicitly unchanged. This G0 local provider result neither claims
  CXI nor authorizes a real model, two-node G2, or 4+ node native job.

## Exact validation commands

The task inherited the completed local-native dependency branch and merged it
with the live dense frame ABI commit before implementation. The final
implementation was validated with these command shapes:

```bash
# Unified exact-commit build, install, manifest, and native CTest (8/8)
BUILD_DIR="$PWD/build/native-resilient-dataplane-build" \
INSTALL_DIR="$PWD/build/native-resilient-dataplane" \
PYTHON_BIN=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
BUILD_TYPE=RelWithDebInfo \
bash scripts/frontier/build_native_resilient_dataplane.sh

# Focused unit/fence suite (20 passed; real stress test deselected here)
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -q tests/test_hyperscale_local_adapter.py \
  -k 'not repeated_real' tests/test_fenced_admission.py

# Independent compiled four-cycle pytest gate (1 passed in 171.04s)
NDP_TEST_PROVIDER='tcp;ofi_rxm' \
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  -m pytest -q \
  tests/test_hyperscale_local_adapter.py::test_repeated_real_native_failure_restart_gate_uses_dynamic_membership

# Retained 12-cycle qualification that produced the machine report
NDP_TEST_PROVIDER='tcp;ofi_rxm' \
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python \
  scripts/validate_native_pool_hyperscale_local.py \
  --build-manifest build/native-resilient-dataplane/native-artifacts.json \
  --output-root build/validate-native-pool-v1-hyperscale-local-evidence \
  --output reports/validate-native-pool-v1-hyperscale-local-metrics.json \
  --failure-restart-cycles 12 --elements 12
```

The retained JSON records `status=passed`, exact commit/bundle identity, all 15
committed generation/checkpoint hashes, all 12 no-publication failure attempts,
the fence 1→2 successor transition, exact resource bounds, zero forbidden-path
counters, and `slurm_jobs_submitted=0`.

## Decision

The hyperscale-local native adapter closes the R13 G0 backend-fixture gap. It
proves that scheduler capacity and training membership are separate: workers
may appear late, disappear, and rejoin under new incarnations while the same
fence, generation, owner, receipt, native ABI, and checkpoint contracts remain
in force. It does not claim Frontier provider or scale admission; the exact-CXI
and ordered NDP17 gates remain mandatory.
