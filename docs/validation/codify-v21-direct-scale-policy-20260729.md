# Direct async v2.1 systems-scale policy validation

Date: 2026-07-29

Task: `codify-v21-direct-scale-policy`

## Result

The normative authorities, authorization schema, controller, allocation
launcher, supervisor, role admission, and focused tests now implement one
systems-only path:

```text
current execution source:
  exact 2-node clean
  -> fault/rejoin
  -> newer-fence fresh-allocation recovery
  -> authorize 8
  -> exact 8-node machine pass
  -> authorize 32
  -> exact 32-node machine pass
  -> authorize 128
  -> exact 128-node machine pass
  -> explicit 256 evidence review only
```

There is no 4-, 16-, or 64-node rung and no 256-node runner. Convergence and
model quality are separate and cannot authorize, substitute for, or block this
systems ladder. No Slurm command was submitted while implementing or
validating this change.

The final E97 identity is unchanged and fail closed:

- step: `2300930`
- accepted tokens: `150793748480`
- bytes: `7719680116`
- SHA-256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`

## Authorities and conformance checklist

This implementation was checked against:

- [`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
  version 1 and its mandatory conformance checklist;
- [`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md),
  including every R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 row;
- [`ASYNC_DECOUPLED_DILOCO_V2.md`](../ASYNC_DECOUPLED_DILOCO_V2.md),
  ADR-002;
- [`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
  native data-plane version 1; and
- [`ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`](../ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md),
  including exact-source identity and the held-payload/durable-afterany/release
  transaction.

The controller and launcher preserve peer-owned leased READY membership,
bounded waits, no launched-rank/all-rank invariant, fenced identities,
deterministic exact-token math, idempotence, stale/corrupt/conflict rejection,
atomic committed evidence, bounded non-Lustre compiled-CXI transport,
backpressure/release, and no central full-model broker. Scale authorization
requires complete clean/fault/fresh-recovery evidence, the exact minimum
systems floor, causal phase/tail evidence, immutable publication/recovery
digests, and an exact predecessor machine pass. Missing or failed evidence
cannot be replaced by evaluator prose.

## Requirement mapping

### Compute-pool R01-R16

| ID | Direct-scale enforcement and retained evidence |
|---|---|
| R01 | Exact execution-source identity, scheduler tuple, allocation fence lineage, and no-database evidence are bound by the v2 signed identity/evidence contract. |
| R02 | `leased_ready_finite_closure=true` plus the existing native lifecycle/rejoin suites preserve stable worker/incarnation transitions. |
| R03 | V21S17 closure remains over `leased-ready-at-group-open`; launched ranks and all-READY waits are rejected. |
| R04 | Exact source/policy/schema/native/seed identity, signed manifest digest, fencing/idempotency evidence, and existing stale/duplicate/conflict suites apply. |
| R05 | `exact_token_eta_outer_one=true` is mandatory; v2.1 exact-token reference/native tests remain green. |
| R06 | The authorization requires evidence-derived finite close, absolute bounds, `<=1 s` snapshot admission, `<=60 s` apply, and zero foreground result wait. |
| R07 | Complete publication receipt, checkpoint manifest, collector terminal verdict, fencing/idempotency, and `publication_complete=true` are mandatory. |
| R08 | `background_compiled_cxi=true`, empty forbidden paths, bounded snapshot/mailbox tests, and native credit/replay tests preserve bounded brokerless transfer. |
| R09 | Coherent immutable safe-boundary snapshots and immediate resume are mandatory schema facts and executable snapshot-pipeline tests. |
| R10 | `forbidden_data_paths=[]` rejects SQLite, shared Lustre/Python dense traffic, collectives, and spill/fallback evidence. |
| R11 | The required `fault` and `fresh-recovery` phases, checkpoint recovery, new-fence runtime tests, and exact predecessor pass preserve catch-up/rejoin. |
| R12 | Checkpoint manifest/publication digests and `checkpoint_recovery=true` bind outer step/token/result/fence/apply restoration. |
| R13 | Scheduler facts remain an adapter boundary; membership/closure are not derived from Slurm ranks. |
| R14 | Causal telemetry digest/completeness, every-event pause bounds, zero result wait, and tail-stall validator coverage are mandatory. |
| R15 | Exact-token numerical/reference tests and changing-participation runtime tests pass; convergence is explicitly separate. |
| R16 | Direct order is exactly `2 -> 8 -> 32 -> 128 -> review 256`; removed/skipped/failed rungs and unchanged failed payloads reject. |

### Native data-plane NDP01-NDP17

| ID | Direct-scale enforcement and retained evidence |
|---|---|
| NDP01 | Signed contracts require compiled native ownership, exact identities, fencing/idempotency, and an empty forbidden-path list; no shared DB authority is introduced. |
| NDP02 | Launched-rank/all-READY closure and fixed-world MPI are rejected; the elastic path remains bounded point-to-point. |
| NDP03 | `background_compiled_cxi=true`, native bundle digest, ABI/wire contract, and existing exact-provider launcher checks are mandatory. |
| NDP04 | `immutable_safe_boundary_snapshots=true` and ISP01 executable coherence/no-live-read tests remain green. |
| NDP05 | Native exact-token binary64 reference and permutation tests pass without math changes. |
| NDP06 | The identity contract pins policy/contribution/generation schemas, ABI `0x00020001`, wire `2.1`, source/bundle/launcher/seed digests, and signed manifest digests. |
| NDP07 | V21S17 still consumes leased READY endpoints/current-fence membership, never launched-rank exchange. |
| NDP08 | Bounded snapshot/mailbox/credit evidence is mandatory; full-capacity paths remain skip/replace/defer. |
| NDP09 | Immediate resume and zero foreground result wait are mandatory; native credit tests remain green. |
| NDP10 | Fencing/idempotency, exact digests, empty forbidden paths, corruption rejection, and once-only apply tests remain green. |
| NDP11 | Existing bounded replay/reassignment suites pass and no Lustre spill path is authorized. |
| NDP12 | Later complete atomic apply and shared native redistribution remain required; no eight-file fallback is introduced. |
| NDP13 | Finite close/stage/cadence arithmetic and bounded foreground facts reject unbounded interruption or background-expiry waits. |
| NDP14 | Scale identity pins v2.1 ABI/wire/schema and launcher independently recomputes the identity digest. |
| NDP15 | Publication/checkpoint digests, immutable input facts, background compiled-CXI work, and later atomic apply are required. |
| NDP16 | Complete causal telemetry and immutable collector verdicts are mandatory; missing or partial records reject. |
| NDP17 | G0-G5 remain prerequisites; G6 is direct `8 -> 32 -> 128`, followed only by explicit 256 review. |

### Async v2.1 V21S01-V21S17

| ID | Direct-scale enforcement and retained evidence |
|---|---|
| V21S01 | New identity schema pins `async-decoupled-v2.1-simple`, policy schema, contribution/generation schemas, ABI/wire, and exact digests; v2.0/wrong schema rejects. |
| V21S02 | Existing distinct-lag/drop/defer tests remain green and zero foreground result wait is mandatory. |
| V21S03 | Exact tokens remain the only quantitative/numerical weight; alternate weighting is not added. |
| V21S04 | K40, stateless exact-token mean, and `eta_outer=1` tests remain green. |
| V21S05 | Exact fenced identities remain bound; scale changes only reviewed closure, not contribution admission semantics. |
| V21S06 | Coherent immutable capture, `OWNED`, and immediate resume are mandatory systems facts and tested. |
| V21S07 | `later_atomic_apply=true`, `apply_swap_seconds_max<=60`, and zero foreground result wait are mandatory. |
| V21S08 | Existing verified latest/capacity-one mailbox and nonblocking capacity tests remain green. |
| V21S09 | Bounded capacity/high-water behavior is required; forbidden-path list must be empty. |
| V21S10 | Leased READY closure and no one-node/Q-min early authority remain mandatory. |
| V21S11 | Atomic all-eight apply/recovery, publication, and checkpoint-recovery facts are mandatory and tested. |
| V21S12 | Compiled CXI, exact native bundle/ABI/wire, no Python/Lustre dense traffic, and no collective remain mandatory. |
| V21S13 | Causal telemetry digest/completeness, max bounds, and zero result wait are authorization gates. |
| V21S14 | Final E97 seed identity, checkpoint/publication digests, and newer-fence recovery are bound exactly. |
| V21S15 | Current-source two-node systems qualification is clean + fault + fresh recovery on `batch/debug`; convergence is separate. |
| V21S16 | Signed v2 direct authorization and exact predecessor pass enforce `2 -> 8 -> 32 -> 128`; 256 is review-only and unchanged failure cannot retry. |
| V21S17 | Evidence-derived immutable finite close includes all admissible pre-close leased-READY arrivals and rejects Q-min early close, launched ranks, unexplained constants, and all-READY barriers. |

### Immutable snapshot ISP01-ISP07

| ID | Direct-scale enforcement and retained evidence |
|---|---|
| ISP01 | `test_snapshot_capture_is_coherent_and_background_never_reads_live_state` proves coherent immutable capture and exclusive live-state ownership. |
| ISP02 | Immediate resume is mandatory; blocked-background-phase tests prove next-K progress after bounded `OWNED`. |
| ISP03 | Background compiled-CXI/checkpoint facts and immutable snapshot tests prove background work consumes immutable inputs. |
| ISP04 | Capacity tests cover snapshot slots, mailboxes, credits, replay, receipts, no growth/spill, and nonblocking outcomes. |
| ISP05 | Atomic bounded/nonblocking result-apply tests cover absent/corrupt/timeout/failure and all-eight recovery. |
| ISP06 | Authorization requires causal telemetry digest/completeness and separate snapshot/admission/publish/aggregation/checkpoint/result-wait/apply/idle evidence. |
| ISP07 | The semantic validator rejects approximately 200-second alternating stalls despite healthy checkpoints/medians; every-event tails remain required. |

## Fail-closed controller and launcher coverage

The v2 schemas are:

- authorization: `emender-async-v21-direct-scale-authorization-v2`
- predecessor pass: `emender-async-v21-direct-rung-pass-v2`
- exact protocol/seed identity:
  `emender-async-v21-direct-scale-identity-v1`

The controller rejects:

- wrong source, policy digest/schema, native bundle/ABI/wire, seed, or launcher
  identity;
- old/unknown authorization or predecessor schema and any manifest digest or
  Ed25519 signature mismatch;
- missing/failed durable afterany collector or immutable terminal machine pass;
- missing evidence digests, causal telemetry, complete publication, checkpoint
  recovery, fencing/idempotency, exact-token math, or changed-payload-only
  retry;
- wrong node count, `Partition`, or `QOS`;
- removed, skipped, failed, or non-immediate rungs and every 256 submission;
- missing finite leased-READY closure, Q-min early close, launched-rank close,
  all-READY wait, unexplained closure constant, or incomplete pre-close
  accounting;
- unbounded snapshot/apply interruption, nonzero foreground result wait,
  forbidden data paths, or a convergence claim; and
- an unchanged retired payload or a second active payload.

The launcher independently verifies both canonical manifest digests, both
Ed25519 review signatures against the trusted reviewer key, the exact
protocol/seed identity contract and recomputed identity digest, equal
authorization/predecessor identities, target and predecessor scheduler tuples,
complete systems evidence, the direct ladder, and the review-only 256 rule.
The supervisor and manager admit only `2`, `8`, `32`, or `128`.

## Test-first and validation record

All Python/native commands were run only after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

and used `"$EMENDER_PYTHON"` or
`PYTHON_BIN="$EMENDER_PYTHON"` as applicable.

The focused tests were written before implementation. The red run was:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py -x
```

Result: collection failed because
`SCALE_IDENTITY_SCHEMA` and the direct-scale contract did not yet exist.

After implementation:

```bash
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
"$EMENDER_PYTHON" -m py_compile \
  scripts/frontier/run_async_v21_qualification.py \
  scripts/frontier/resilient_e97_allocation_supervisor.py \
  scripts/frontier/resilient_e97_role.py
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py
```

Result: `128 passed`.

Canonical native build/install:

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: build/install succeeded; CTest `11/11` passed.

Consolidated controller/launcher/snapshot/runtime/native validation, with the
built bundle bound and a short node-local test socket root:

```bash
export TMPDIR=/tmp
export EMENDER_NDP_SERVICE="$PWD/build/native-resilient-dataplane/bin/ndp_cxi_service"
export EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so.1"
export EMENDER_NDP_TRANSPORT_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1"
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_async_snapshot_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_async_diloco_v21.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_manifest_peer_authority.py \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_pool_integration.py \
  tests/test_native_dataplane_2n_controller.py \
  tests/test_validate_native_dataplane_2n_gate.py \
  tests/test_validate_async_v21_fault_phase.py
```

Result: `297 passed`.

The repository has no `tests/smoke/manifest.toml`, so there is no
project-defined smoke scenario to run. `wg done --full-smoke` remains the final
WG completion gate.

## Scheduler and release audit

- No `sbatch`, `srun`, `squeue`, `sacct`, `scontrol`, or other Slurm mutation
  command was invoked outside test-owned fake scheduler fixtures.
- No real Slurm job was submitted.
- No shared SQLite/Lustre model data-plane path, simulated Slurm
  implementation, or broad Lean runtime rewrite was introduced.
- Native reference checksums were refreshed only for the amended normative
  authorities; the historical 256-node fixed-world measurement remains
  explicitly reference-only.
- The exact release-candidate commit and `HEAD == fetched origin/main ==
  ls-remote refs/heads/main` equality are recorded in the WG task log after
  the surgical non-force push.
