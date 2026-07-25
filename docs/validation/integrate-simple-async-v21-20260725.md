# Async DiLoCo v2.1 integration and pre-Slurm release record

**Task:** `integrate-simple-async-v21`
**Date:** 2026-07-25
**Release scope:** source integration, production-path audit, native rebuild,
focused local qualification, and non-submitting controller renders only
**Slurm mutations:** none

## Decision

The pre-Slurm integration gate for `async-decoupled-v2.1-simple` passes.
The accepted ADR-002 design, implementation, serial qualification controller,
and atomic all-eight-trainer node-apply recovery are reconciled with the
current `origin/main` history. The inherited task branch and the fetched
remote initially had divergent commit histories but the same tree
`ad78eeee7ab48674586fc7ef859139505ba446b0`; the merge retained both histories
without removing or rewriting unrelated work.

This result authorizes the downstream exact two-node qualification task to
start from the pushed integration SHA. It does not claim that any V21S15 live
gate passed, does not create a scale authorization, and does not authorize a
4+ node submission. Historical `async-decoupled-v2.0-exp` results remain
incompatible evidence only.

The normative authorities reviewed before Python, pytest, native build, CTest,
or rendering were:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including R01–R16, NDP01–NDP17,
  and V21S01–V21S17;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, version 1; and
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`.

Every Python, pytest, native build, CTest, or controller-render command below
was preceded by:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Python was invoked as `"$EMENDER_PYTHON"` and native wrappers received
`PYTHON_BIN="$EMENDER_PYTHON"`.

## Production-path audit

The rendered production path is:

```text
scripts/frontier/run_async_v21_qualification.py
  -> scripts/frontier/resilient_e97_true_2n.sbatch
  -> scripts/frontier/resilient_e97_allocation_supervisor.py
  -> scripts/frontier/resilient_e97_role.py
  -> ndm/async_diloco_v2.py
     ndm/native_e97_runtime.py
     ndm/resilient_e97_runtime.py
     ndm/resilient_pool_runtime.py
  -> installed libemender_ndp.so.1 + libemender_ndp_transport.so.1
     + persistent ndp_cxi_service
```

The controller renders `sbatch` arguments but calls it only through the
explicit `--submit` path. This task used `--dry-run` only. The two-node
controller pins `Nodes=2`, `Partition=batch`, `QOS=debug`, K40, exact-token
floor 3,934,080, all four maximum lags to 2, `eta_outer=1.0`, the canonical
policy/config/seed/bundle/launcher digests, and `network_fetches=0`.

The batch launcher rechecks those values before starting roles. The allocation
supervisor acquires the fenced lease before model-bearing work, starts one
persistent native service and one model-free manager per node, and constructs
or restarts one atomic node cohort of all eight trainers. The role validates
the v2.1 schema/digests before admission, uses exact tokens as the only
quantitative weight, maintains one immutable native-owned cohort and one
mutable adjacent interval, and admits only reload-verified results to the
capacity-one mailbox. READY follows one matching node-applied marker from all
eight trainer recovery markers; partial or timed-out apply reconstructs the
whole cohort from authoritative latest under new incarnations.

The production native branch uses direct memfd/native handles and bounded
point-to-point transport. `LocalTrainerSpool` and `DistributedOwnerServer`
remain imported for the explicitly selected `python-tcp-debug` branch, but
their constructors are not reachable from the native production selection.
The focused source and symbol tests reject MPI/all-rank collectives, Python
dense transport, Lustre dense spill, unbounded buffers, and a central
full-model broker.

## R01–R16 requirement-to-test/render matrix

| ID | Integrated production evidence | Test or render anchor | Pre-Slurm result |
|---|---|---|---|
| R01 | Admission lease/fence is acquired before roles; stale holders cannot mutate state. | `test_allocation_fence_is_acquired_before_roles_and_loser_is_zero_work`; fenced checkpoint/restart tests. | Pass locally; Frontier durable-store qualification remains a live gate. |
| R02 | Stable worker plus incarnation drives leased lifecycle and atomic cohort recovery. | `test_ready_token_floor_distributed_owner_loss_and_late_join`; `test_job_5068873_partial_apply_restarts_atomic_eight_trainer_cohort`. | Pass. |
| R03 | Active membership is leased READY state, never launched ranks. | `test_v21_scale_close_includes_all_preclose_arrivals_not_only_two`; pool changing-membership tests. | Pass. |
| R04 | Full fenced contribution identity is idempotent; stale, corrupt, and conflicting reuse rejects. | `test_generation_identity_rejects_stale_partial_corrupt_nonfinite_and_obsolete`; native failure suite. | Pass. |
| R05 | Exact tokens are the only v2.1 numerical weight and deterministic binary64 denominator. | `test_v21_exact_tokens_are_only_weight_and_eta_one`; native unequal-token permutation tests. | Pass. |
| R06 | Exact Q/T floors and finite deadlines close or abort; no unbounded wait. | pool token-floor tests; `test_v21_scale_closure_rejects_launched_rank_and_unexplained_constant`. | Pass. |
| R07 | Immutable state reload verification precedes fenced authoritative-latest publication. | `test_fenced_atomic_global_commit_and_newer_allocation_restart`; checkpoint-finalization suite. | Pass. |
| R08 | Deterministic owners, bounded chunks/credits/replay/release, no broker. | native failure/reference/integration suites; 10/10 CTests. | Pass. |
| R09 | Manager stays model-free; eight trainers retain model/optimizer state; partial apply restarts atomically. | `test_live_native_selection_is_wired_and_python_debug_remains_explicit`; atomic apply tests. | Pass. |
| R10 | Native dense hot path is memfd/CXI, not Lustre or Python sockets. | `test_cross_process_trainer_handoff_passes_sealed_memfd_without_dense_socket`; launcher source audits. | Pass. |
| R11 | Late/rejoined workers use new incarnation/latest; stale local state is rejected. | manager rejoin, newer-fence restore, stale-incarnation, and atomic cohort tests. | Pass. |
| R12 | Global model/outer step/accepted-token clock restore exactly; local inner work is disposable. | `test_v21_checkpoint_and_fresh_allocation_restore`; canonical cold-start and newer-fence tests. | Pass. |
| R13 | Pool protocol is scheduler-independent and native transport is point-to-point. | `test_two_model_free_managers_exchange_without_collective`; native route tests. | Pass locally; non-Frontier adapter remains outside this release. |
| R14 | First heartbeat, READY, K40, freeze, apply, publication, catch-up, and shutdown are bounded and observable. | launcher stage-deadline tests; semantic telemetry validator. | Pass. |
| R15 | Arrival-order-stable reference math, exact accepted-token accounting, and ScheduleFree correction are checked. | native reference tests; correction-ledger selected/nonselected/replay tests. | Pass. |
| R16 | Exactly two nodes precede the strict `4→8→16→32→64→256` ladder. | controller render/refusal audit and exact-predecessor tests. | Pass as a fail-closed gate; no rung is authorized. |

## NDP01–NDP17 requirement-to-test/render matrix

| ID | Integrated production evidence | Test or render anchor | Pre-Slurm result |
|---|---|---|---|
| NDP01 | Python owns control/policy; C++ owns dense buffers, reduction, transport, and lifetime. | native selection/source audit and persistent-service integration tests. | Pass. |
| NDP02 | Elastic native path contains no MPI or failure-sensitive all-rank operation. | ABI symbol boundary; two-manager exchange; route-loss tests. | Pass. |
| NDP03 | One persistent C++17 `FI_EP_RDM` service per node; production requires exact `cxi`. | launcher topology/provider tests; provider CTests. | Pass locally; exact-code live CXI remains downstream. |
| NDP04 | Producer-direct service memfd handoff adds no Python dense copy. | sealed-memfd cross-process and eight-trainer K40 tests. | Pass. |
| NDP05 | Fixed deterministic binary64 exact-token arithmetic and one f32 projection. | native reference suite and exact-global-numerator test. | Pass. |
| NDP06 | ABI, wire, frames, contributions, receipts, and results carry fixed fenced identities. | native ABI/protocol CTests and stale/corrupt tests. | Pass. |
| NDP07 | Current-fence endpoint records are exchanged through leased membership. | route installation, endpoint filtering, and rejoin tests. | Pass. |
| NDP08 | Layout, shared bytes, registered slots, owner memory, and 64-GiB v2.1 resident formula preflight. | byte exhaustion, sealed extent, and production-policy tests. | Pass. |
| NDP09 | Receiver credits are distinct from fabric completion and remain bounded. | credit exhaustion/recovery CTests and bounded native-transfer tests. | Pass. |
| NDP10 | CRC/SHA, nonfinite rejection, once-only apply, and idempotent replay are mandatory. | native failure/reference/integration suites. | Pass. |
| NDP11 | Sender replay and two owner reassignments are bounded; disk replay is disabled by default. | replay exhaustion, cancellation/release, and zero-spool tests. | Pass. |
| NDP12 | Owners redistribute one service-owned shared aggregate directly. | independent owner import, shared result views, and eight-trainer tests. | Pass. |
| NDP13 | Every native stage has an absolute deadline and route-local failure containment. | install lifetime, service-loss, TERM, and stage-deadline tests. | Pass. |
| NDP14 | Stable `libemender_ndp.so.1` ABI and metadata-only seqpacket control are installed. | ABI struct/SONAME/fd-cardinality tests; native build. | Pass. |
| NDP15 | Python applies/checkpoints/publishes; native result release is fenced and collective-free. | fenced checkpoint release-once and finalization tests. | Pass. |
| NDP16 | Provider, identities, byte highs, lags, release counters, and terminal reasons are structured. | native manifest and semantic telemetry tests. | Pass locally; observed live values remain downstream. |
| NDP17 | Exact-code full-layout G2 precedes real-model/scale; scale order is fixed. | exact launcher artifact-gate tests and controller predecessor/refusal tests. | Pass as an admission boundary; no new live G2 claim. |

## V21S01–V21S17 requirement-to-test/render matrix

| ID | Integrated production evidence | Test or render anchor | Pre-Slurm result |
|---|---|---|---|
| V21S01 | Policy/schema/manifest/ABI/wire identities are v2.1-only; v2.0 rejects before mutation. | `test_v21_rejects_v20_policy_schema_and_digest`; ABI/protocol tests. | Pass. |
| V21S02 | Commit, anchor, result, and speculative clocks are distinct, accept 0–2, and reject/catch up at 3. | `test_v21_lag_0_1_2_accepts_and_lag_3_drops_and_catches_up`; semantic validator. | Pass. |
| V21S03 | Positive exact tokens are sole quorum, clock, numerator weight, and denominator. | `test_v21_exact_tokens_are_only_weight_and_eta_one`; pool exact-token test. | Pass. |
| V21S04 | K40 and stateless exact-token `eta_outer=1.0`; no outer momentum. | v2.1 policy test; real trainer/persistent-session tests; controller render. | Pass. |
| V21S05 | Full worker/incarnation/window/base/digest identity; exact 2n Q/T/deadlines; one worker contribution. | contribution replay/conflict tests and exact two-node renders. | Pass. |
| V21S06 | Persistent trainer session; one native-owned cohort plus one mutable adjacent interval. | `test_v21_one_owned_one_mutable_and_no_third_cohort`; persistent trainer tests. | Pass. |
| V21S07 | K-boundary correction translates ScheduleFree `x`, `z`, and mutable start once, including skipped versions. | correction-ledger and real trainer boundary tests. | Pass. |
| V21S08 | Capacity-one reload-verified mailbox with bounded replacement and fenced latest. | mailbox/checkpoint/corrupt-result tests. | Pass. |
| V21S09 | Exact resident/credit/replay/mailbox bounds forbid a third dense cohort or dense spill. | production-policy, byte-bound, and one-owned/one-mutable tests. | Pass. |
| V21S10 | Leased READY only; rejoin uses new incarnation/latest; no one-node authority. | membership/rejoin/frozen-identity/token-floor tests. | Pass. |
| V21S11 | Node apply is one all-eight transaction; partial apply reconstructs all lanes before READY. | `test_v21_node_ready_requires_all_eight_apply_markers`; job-5068873-class regression. | Pass locally; live fault artifact remains downstream. |
| V21S12 | Persistent native service, direct memfd, exact native reduction, bounded point-to-point CXI, no collective/dense Python path. | native ABI/integration/failure suites and 10/10 CTests. | Pass locally. |
| V21S13 | Honest interval/stage/lag/K/OWNED/apply/high-water telemetry and separate correctness latency. | semantic-validator and retained-node telemetry tests. | Pass locally. |
| V21S14 | One fenced immutable model/outer/token bundle and exact final seed with submit verification, job-scoped sbcast, offline node verification. | checkpoint restore, exact seed render, sbcast/offline launcher tests. | Pass. |
| V21S15 | Serial controller pins the five conceptual 2n gates to `batch/debug` and retains scheduler fields separately. | clean/faults/convergence renders; exact queue/squeue/sacct tests. | Controller pass only; no live qualification claimed. |
| V21S16 | Reviewed promotion plus exact immediate predecessor for `4→8→16→32→64→256`; failures never advance. | every-rung exact-predecessor, unchanged-failure, active-job, and refusal tests. | Pass as fail-closed control. |
| V21S17 | Finite immutable close over leased READY; all pre-close arrivals included; no Q-min close, launched ranks, all-ready wait, or unexplained constants. | closure derivation/arrival tests and scale dry-run refusal audit. | Pass as fail-closed control; no production closure authorization exists. |

## Compute-pool conformance checklist

- **Leased READY membership, bounded waits, no launched-rank invariant:**
  R01–R03, R06, R11, R14, V21S05, V21S09, V21S10, and V21S17 are
  exercised. Exact two-node minimum progress is two distinct stable workers
  and 3,934,080 exact tokens. Loss of either worker pauses/aborts rather than
  inventing one-node authority.
- **Fenced identity, deterministic math, idempotence, rejection, atomic
  evidence:** R04–R07, R12, R15, NDP05, NDP06, NDP10, NDP15, V21S01–V21S05,
  V21S07, V21S08, V21S11, and V21S14 are exercised.
- **Bounded non-Lustre point-to-point transport and release:** R08–R10,
  NDP01–NDP14, NDP16, V21S06, V21S09, and V21S12 are exercised. A central
  full-model broker and failure-sensitive collective are absent.
- **Failure/deadline/recovery:** tests cover late/stale/corrupt input,
  missing peer, lag-3 catch-up, held mailbox view, owner/service/trainer loss,
  partial eight-trainer apply, failed publication, new incarnation, and
  fresh-allocation/new-fence restore. All waits retain absolute parent bounds.
- **Exact commands/artifacts and prior-rung gate:** commands and local artifacts
  are below. Every 4+ rung refuses without its signed exact authorization and
  immediate predecessor pass; the accepted order is only
  `4→8→16→32→64→256`.

## Historical v2.0 reachability audit

The rendered production files were searched for v2.0 policy/schema identities,
lag-adjusted weighting, half-step outer math, tau 6, sigma/speculation 8,
`max_commit_lag=6`, `max_speculative_windows=8`, and an
`aggregation_weight` field. No historical constant or schema is selected,
migrated, or used as fallback by the v2.1 launcher. The only
`aggregation_weight` occurrences in the production path are explicit rejection
conditions in:

- `ndm/native_e97_runtime.py`;
- `ndm/resilient_pool_runtime.py`;
- `scripts/frontier/resilient_e97_role.py`; and
- `scripts/frontier/validate_pipelined_e97_performance.py`.

The historical V2A records and dated v2.0 reports remain readable as explicit
rejection/reference fixtures only.

## Exact seed binding

The controller, canonical config, exact renderer, and launcher agree on:

| Field | Exact value |
|---|---|
| checkpoint | `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt` |
| manifest | `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json` |
| step | `2300930` |
| accepted tokens | `150793748480` |
| bytes | `7719680116` |
| SHA-256 | `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2` |
| node-local root | `/tmp/emender-e97-seed-$SLURM_JOB_ID` |
| compute-node network fetches | `0` |

Immediately before a future submission, the submit side must revalidate both
S3 authorities, size, SHA, and attestation. The launcher requires compiled
`sbcast`, preserves the literal job-scoped destination, and runs one offline
verifier per node before a model-bearing role starts. This integration did not
download the seed and did not submit a job.

## Validation commands and results

### Native configure/build/install/attestation

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: **10/10 CTests passed**. The installed bundle contains
`libemender_ndp.so.1`, `libemender_ndp_transport.so.1`, and
`ndp_cxi_service`. The binary bundle SHA-256 is
`f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441`.
The exact final source commit, manifest SHA-256, artifact digests, and build
paths are retained in the final release receipt named below.

### Focused Python/controller/native integration gate

```bash
source scripts/frontier/activate_emender_frontier.sh
integration_gate_tmp=$(mktemp -d /tmp/emender-integrate-v21-gate.XXXXXX)
"$EMENDER_PYTHON" -m pytest -q --basetemp="$integration_gate_tmp" \
  tests/test_async_diloco_checkpoint_manager.py \
  tests/test_async_diloco_real_trainer.py \
  tests/test_async_diloco_v2.py \
  tests/test_async_diloco_v21.py \
  tests/test_async_v21_qualification_controller.py \
  tests/test_checkpoint_finalization.py \
  tests/test_e97_checkpoint_retention_guard.py \
  tests/test_native_dataplane_2n_controller.py \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_pool_integration.py \
  tests/test_native_pool_production_policy.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_resilient_e97_rank_lane.py \
  tests/test_resilient_e97_reducer.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_split_roles.py \
  tests/test_resilient_e97_topology.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_validate_native_dataplane_2n_gate.py \
  tests/test_validate_native_dataplane_local.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_walltime_final_checkpoint.py
```

Final result: **308 passed in 205.05 seconds**.

The first isolated-worktree run intentionally established the native
prerequisite: 270 tests passed, one skipped, and 37 native fixtures refused
because no installed service existed. After the canonical build, the suite
found one real integration defect: the checksum-linked native reference still
named the pre-atomic-fix gap-matrix digest. Only
`reports/frontier/native-dataplane-reference-v1.json` and its linked Markdown
digest were refreshed. The focused reproducer passed, followed by the complete
308/308 run above.

### Controller command templates

For each conceptual two-node phase (`clean`, `faults`, `convergence`):

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate <clean|faults|convergence> --nodes 2 \
  --state <retained-state.json> --evidence-root <retained-evidence> \
  --bundle-digest <exact-native-bundle-sha256> \
  --dry-run --output <rendered-plan.json>
```

For each rung in `4 8 16 32 64 256`, the same CLI with
`--gate scale --nodes <rung>` refuses without both `--authorization` and
`--prior-rung`. When supplied, those manifests must be canonical,
digest-valid, signed by the trusted Ed25519 reviewer, bind the exact
source/policy/bundle/seed/launcher identities, name the immediate predecessor,
and contain the reviewed evidence-derived V21S17 close. Scale parameters must
explicitly set:

```json
{
  "close_on_q_min": false,
  "uses_launched_ranks": false,
  "wait_for_all_ready": false
}
```

No production authorization is present in this task, so a successful scale
plan was neither fabricated nor rendered. The retained refusal artifacts for
all six rungs are the correct locked-scale result.

## Retained artifacts

Tracked:

- this report;
- `reports/frontier/native-dataplane-reference-v1.json`;
- `reports/frontier/native-dataplane-reference-v1.md`;
- the reconciled integration history and final release commit.

Generated from the final committed source and retained outside the tracked
source inventory:

- `build/integrate-simple-async-v21/final/native/native-artifacts.json`;
- `build/integrate-simple-async-v21/final/renders/two-node-*.json`;
- `build/integrate-simple-async-v21/final/renders/scale-refusals.json`;
- `build/integrate-simple-async-v21/final/release-receipt.json`.

The receipt binds the final Git SHA/tree, native bundle and manifest digests,
controller source/policy/seed/launcher identities, render digests, six
fail-closed scale refusals, and the final remote equality proof. No artifact
in this record is a Slurm pass or scale authorization.
