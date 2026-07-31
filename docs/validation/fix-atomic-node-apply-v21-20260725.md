# Atomic async-v2.1 node apply and restart validation

Date: 2026-07-25

WG task: `fix-atomic-node-apply-v21`

Status: **LOCAL IMPLEMENTATION AND GATES PASSED; LIVE FAULT GATE UNRUN**

## Outcome and scope

The async-v2.1 node transition is now an explicit all-eight-trainer recovery
cohort. A manager, native-service, or trainer failure cannot leave healthy
sibling trainers running past a partial node apply. The node supervisor stops
the service, manager, and all eight trainers under one bound, retains the
volatile failed transaction below its old node incarnation, verifies the
authoritative committed generation, creates a new node incarnation, and starts
the service, manager, and all eight fresh trainer process incarnations in that
order.

The manager and every trainer validate the same fenced cohort-recovery record.
Restarted trainers ignore same-fence per-rank recovery, resolve the
authoritative handoff, restore committed model/outer/token state, and construct
fresh disposable inner state. A trainer also rejects generation metadata from
the prior manager incarnation before opening or attaching to the native
service. The manager rejects submissions whose stable-worker incarnation does
not match the new cohort. Only eight matching trainer recovery receipts permit
the node-applied marker, native commit, and next-version READY advertisement.

No Slurm job was submitted, cancelled, held, released, requeued, or otherwise
mutated by this task. In particular, no `sbatch`, `srun`, `salloc`, `scancel`,
`scontrol update`, or other scheduler mutation was executed.

## Authorities and conformance checklist

The implementation was reviewed against:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, version 1;
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`; and
- retained failure report
  `docs/validation/re-run-fixed-async-v2-exact-2n-20260724.md`.

The compute-pool conformance checklist is satisfied locally as follows:

- READY remains leased membership, not launched-rank state. A replacement
  manager uses a new node incarnation and rejoins only after authoritative
  synchronization. All waits remain bounded.
- Generation, node, trainer, result, policy, layout, code, and allocation-fence
  identities remain explicit. Identical duplicate/delayed trainer receipts are
  idempotent; conflicts and stale incarnations fail before mutation.
- Native dense transport remains bounded, point-to-point, non-Lustre, and
  model-free. The recovery transaction moves only compact node-local JSON
  control evidence; it adds no dense copy, Python dense socket, central broker,
  collective, or all-rank wait.
- The exercised failure path is partial/timed-out node apply followed by
  manager/service/trainer reconstruction. Exact two-node progress remains
  `Q_min=2` distinct leased READY workers and
  `T_min=3,934,080` exact accepted tokens. Loss of one node cannot create
  one-node commit authority.
- Exact commands, results, commits, and native manifest digest are recorded
  below. No live/promotion/scale pass is inferred from local tests.

## Retained evidence remains failed evidence

### Job 5066495

Job `5066495` remains **FAILED**. It used exactly two nodes on
`Partition=batch`, `QOS=debug`, reached generation-zero publication, then
failed at the first independent post-commit ScheduleFree boundary with:

```text
ValueError: ScheduleFree z point is missing or malformed
```

The later lazy-state repair does not retroactively qualify this job. It grants
no async-v2.1 clean, fault, promotion, or scale evidence.

### Job 5068873

Job `5068873` remains **FAILED**:

```text
5068873|resilient-e97-true-2n|FAILED|1:0|00:27:50|2|batch|debug
```

The job reached the generation-zero native handoff with result root
`04dfbf27ebfe52371e56142e55ddbbae93588440c2906744720d4cf6d6bbab3d`,
exact accepted tokens `5,245,440`, and separate historical v2.0 lag-zero
numerical denominator `36,718,080`. Node 1 emitted only a subset of its
generation-zero trainer apply/recovery markers. The missing-lane deadline was
the first independent failure. Generation-one native-submit metadata
deadlines, manager/service restarts, stale/invalid generation identities, and
local-reduction deadlines were downstream cascades.

The deterministic regression models the task-assigned four-applied/four-missing
ordering explicitly: ranks 0–3 have generation-zero receipts, ranks 4–7 are
missing, generation-one stale manager/submission records remain, and the
manager crosses the apply deadline. It proves that all four partial receipts
and the stale generation-one records are removed from the active namespace and
retained below the failed node incarnation before all eight trainers are
reconstructed.

Neither historical job is v2.1 qualification evidence. The serial two-node
gate remains closed until a changed-payload V21S15 fault/restart run passes.

## Causal repair

The retained failure exposed four coupled defects:

1. the node supervisor replaced only a failed manager and native service;
2. sibling trainers, including already-applied lanes, survived that restart;
3. same-fence per-rank recovery could be preferred over the one authoritative
   global model/outer/token handoff; and
4. generation metadata named the manager incarnation but a restarted trainer
   did not compare it with a supervisor-fenced node cohort before native
   attachment.

The repair establishes these invariants:

1. `AllocationSupervisor` owns one random node incarnation shared by its
   service, manager, and eight trainers.
2. Any native service, manager, or trainer loss reconstructs the entire local
   cohort; the restart budget applies to every member.
3. After all old writers stop, volatile generation/layout/submission/result/
   apply/proposal/recovery JSON is atomically moved to
   `control/failed-cohorts/<failed-incarnation>/`.
4. Reconstruction fails closed unless immutable `handoff/latest.json`, its
   manifest checksum/finalized generation, and (when configured) the current
   fenced control-store `latest/authoritative` publication agree.
5. One `atomic-cohort-recovery.json` binds run, allocation fence, node rank,
   failed/new node incarnations, restart sequence, authoritative generation,
   exact ranks 0–7, and reconstruction status.
6. The fresh service starts first; the fresh manager validates/reloads latest
   and crosses fresh-incarnation readiness; only then are all eight trainers
   started.
7. A reconstructed trainer ignores rank-local recovery, reloads authoritative
   model/outer/token state, validates the cohort record, and rejects stale
   manager metadata before `Client.open` or `attach_generation`.
8. The manager admits only submissions carrying its current node incarnation.
   Eight distinct trainer incarnations and matching recovery digests are still
   required before the idempotent node-applied marker and READY.

Publication failure is safe: candidate trainer state never creates READY; a
cohort restart reloads the still-authoritative prior latest. Owner retry remains
bounded and idempotent in the native layer. A duplicate or delayed matching
trainer receipt returns the original record; a conflict cannot alter the node
transaction. Service loss, manager loss, or trainer loss all take the same
fenced all-eight reconstruction path.

## ScheduleFree exactly-once and bounded progress

The numerical/correction implementation was not changed. The verified-latest
boundary still applies:

```text
(new_global - old_anchor) - accepted_local_delta_sum
```

exactly once to ScheduleFree `x`, `z`, and mutable interval start. Scalar and
moment state remains untouched in a live incarnation. A failed partial apply
does not serialize mixed local inner state: all eight trainers reload the same
committed global checkpoint and recreate fresh inner state. The archived
partial markers cannot be re-read as active receipts, so the correction cannot
be double-translated and partial apply cannot advertise READY.

At the exact two-node gate, `PoolControlConfig` remains pinned to
`Q_min=2`, `T_min=3,934,080`, no active fraction, and zero generation-attempt
retry. Expiring one node removes its READY incarnation. The survivor may use
only the existing bounded one-owned/one-mutable and lag-at-most-two capacity,
then pauses/catches up or drains. No code path reduces the diversity floor or
invents a one-node commit.

## R01–R16 traceability

| ID | Local conformance result |
|---|---|
| R01 | Reconstruction requires checksum-valid durable latest and, when configured, exact agreement with the current allocation lease/control-store CAS before trainer restart. Allocation ownership change still requires the existing newer fence. |
| R02 | The supervisor now binds service, manager, and trainers to one node incarnation; every reconstruction creates a different node incarnation and eight different process/trainer incarnations. |
| R03 | Active world remains leased READY membership. Recovery is node-local and never derives authority from launched ranks. |
| R04 | Stale node-incarnation generation metadata and submissions are rejected; identical delayed apply receipts are idempotent and conflicting receipts fail closed. |
| R05 | Exact-token deterministic binary64 aggregation is unchanged and covered by native reference/integration tests. |
| R06 | Exact two-node `Q_min=2`/`T_min=3,934,080` closure is unchanged; loss of one node cannot commit. All restart/apply waits are bounded. |
| R07 | Immutable checkpoint/latest CAS remains global authority. Eight matching local recovery receipts gate the node marker and READY; partial files never become a commit. |
| R08 | Native chunks, credit, checksums, replay, release, and deterministic ownership are unchanged. Recovery archives compact JSON only. |
| R09 | The manager remains model-free. All eight model-owning trainers discard disposable partial local/inner state together. |
| R10 | Dense data remains in memfd/XPMEM/native point-to-point paths. Failed-cohort retention contains no dense trainer payload. |
| R11 | Rejoin requires new node/trainer incarnations plus authoritative latest; stale volatile work is retained only as failed evidence. |
| R12 | Restart restores authoritative global model, outer step, and accepted-token clock; per-rank same-fence recovery is bypassed for cohort reconstruction. |
| R13 | No scheduler-specific membership or data-plane semantics were added; the change stays in the existing supervisor/role adapter boundary. |
| R14 | Apply/recovery and restart readiness retain absolute bounds and structured causal events for failed/reconstructed cohorts. |
| R15 | Accepted-token accounting and numerical/correction behavior are unchanged and pass reference, runtime, checkpoint, and replay suites. |
| R16 | No Slurm mutation or 4+ authorization occurred. Historical failed jobs remain failed; later rungs remain closed. |

## NDP01–NDP17 traceability

| ID | Local conformance result |
|---|---|
| NDP01 | Python control/supervision owns cohort failure, fence/latest verification, and READY. C++ continues to own dense buffers and transport. |
| NDP02 | The restart is node-local process supervision; no MPI initialization, collective, all-rank barrier, or peer-dependent shutdown was added. |
| NDP03 | One persistent service exists per live node incarnation; reconstruction replaces it before the manager/trainers and preserves exact-provider policy. |
| NDP04 | Producer-direct memfd/XPMEM handoff is unchanged; reconstruction adds no trainer-sized copy or file. |
| NDP05 | Fixed deterministic exact-token reduction and result encoding are unchanged; native reference and integration suites pass. |
| NDP06 | Cohort recovery binds run/fence/node/restart/incarnation/generation; generation/submission/result/apply identities remain fenced and stale input is rejected. |
| NDP07 | A fresh manager/service incarnation creates fresh endpoint identity before READY; old endpoint state cannot be adopted. |
| NDP08 | Resident, slot, contribution, and result bounds are unchanged; no third dense cohort is introduced. |
| NDP09 | Credit and fabric-completion semantics are unchanged and pass the native CTests/integration suite. |
| NDP10 | Checksums and once-only apply remain mandatory; duplicate delayed recovery receipts are idempotent, conflicts reject. |
| NDP11 | Sender replay/owner reassignment remains bounded. Cohort failure archives only control evidence and never fabricates a retry contribution identity. |
| NDP12 | Owner-direct shared-result redistribution is unchanged. One candidate result cannot make a partially applied node READY. |
| NDP13 | Service/manager/trainer loss is contained to one node cohort with existing absolute startup/apply/recovery limits. |
| NDP14 | The stable ABI and metadata-only control channel are unchanged; stale metadata is rejected before native open/attach. |
| NDP15 | Fenced read-only result, reload-verified checkpoint, latest CAS, all-eight receipt barrier, node marker, native commit, and READY remain ordered. |
| NDP16 | Supervision events record failed/new incarnations, authoritative generation, ranks 0–7, cause, status, and fresh trainer incarnations. |
| NDP17 | Native build/10 CTests and local integrations pass. No live G3/G4/G5 or scale artifact is claimed. |

## V21S01–V21S17 traceability

| ID | Local conformance result |
|---|---|
| V21S01 | v2.1 policy/schema/ABI/wire/checkpoint identities remain mandatory; no v2.0 record is relabeled. |
| V21S02 | Commit/anchor/result/speculative clocks and lag-at-most-two bounds are unchanged; node loss cannot extend capacity. |
| V21S03 | Positive exact tokens remain the only quantitative floor, clock, numerator weight, and denominator. |
| V21S04 | K40 and stateless `eta_outer=1.0` outer state are unchanged and restored from authoritative latest. |
| V21S05 | Stable worker, node/trainer incarnation, window/base/digest/token identities are preserved; two-node constants remain exact. |
| V21S06 | One owned plus one mutable interval remains the live bound. Reconstruction discards both rather than retaining a third/mixed cohort. |
| V21S07 | ScheduleFree `x/z` and mutable-start correction remains exactly once; failed partial local state is never serialized as authoritative. |
| V21S08 | Only reload-verified current-fence latest is eligible for reconstruction/apply; stale generation metadata rejects before native mutation. |
| V21S09 | The 64,001,671,648-byte resident formula, finite credits/replay/mailbox/deadlines, and no-third-cohort rule are unchanged. |
| V21S10 | Old READY incarnation expires; the replacement uses authoritative latest/new incarnation. `Q_min=2` cannot become one-node authority. |
| V21S11 | The job-5068873-class regression proves four applied/four missing lanes cause archived failure and service/manager/all-eight reconstruction under one new node incarnation before READY. |
| V21S12 | Persistent compiled point-to-point service, direct memfd, exact reduction, bounded fabric, and no MPI/Python-dense/Lustre/broker invariants remain intact. |
| V21S13 | Cohort restart sequence/incarnations/status/reason and eight required ranks are explicit; lag/K/OWNED/apply/checkpoint telemetry remains honest. |
| V21S14 | Trainer reconstruction restores immutable fenced model/outer/token authority; fresh-allocation newer-fence behavior and exact cold seed remain unchanged. |
| V21S15 | Local numerical/fault/restart/determinism components pass, but the five exact-two-node live gates are unrun and unclaimed. |
| V21S16 | No promotion manifest was issued and no scale rung was authorized. |
| V21S17 | Scale-only closure code/tests remain fail-closed; this task neither changes nor exercises 4+ closure. |

The especially direct atomic apply/restart mappings are:

- **V21S07:** same correction implementation, archived partial state, fresh
  inner state after incarnation recovery;
- **V21S08:** reload/CAS authority plus pre-native stale-incarnation rejection;
- **V21S10:** expired old READY incarnation, fresh rejoin, unchanged
  `Q_min=2`;
- **V21S11:** all-eight cohort failure/reconstruction and node READY barrier;
  and
- **V21S14:** exact committed model/outer/token restore on same-allocation
  cohort reconstruction and newer-fence allocation recovery.

## Regression-first evidence

The regression was written before the repair and first run under canonical
activation:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_true_2n_launcher.py::test_job_5068873_partial_apply_restarts_atomic_eight_trainer_cohort
```

Expected red result:

```text
FAILED ... AttributeError:
'AllocationSupervisor' object has no attribute 'node_incarnation'
1 failed
```

After the repair, the job-5068873 regression plus adjacent manager/service,
node-marker, and apply-progress checks passed `4/4`. The expanded v2.1/runtime/
supervisor batch passed:

```text
110 passed in 112.83s
```

The regression surface also includes:

- `test_reconstructed_trainer_rejects_stale_manager_metadata_before_native_open`;
- `test_atomic_cohort_recovery_rejects_stale_fence_incarnation_and_generation`;
- duplicate/delayed/conflicting receipt assertions in
  `test_v21_node_ready_requires_all_eight_apply_markers`; and
- preservation of failed per-trainer/node marker evidence by incarnation.

## Native build and test commands

Every Python, pytest, and native build command was preceded by:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The authority-specified native configure/build/CTest command passed:

```bash
cmake -S native/dataplane -B build/native-dataplane \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNDP_ENABLE_XPMEM=ON
cmake --build build/native-dataplane --parallel
ctest --test-dir build/native-dataplane --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 10
```

The canonical installed/attested bundle was also built with:

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result:

```text
100% tests passed, 0 tests failed out of 10
build/native-resilient-dataplane/native-artifacts.json: status=recorded
```

The recorded manifest SHA-256 is:

```text
0dbe3db785d25dd248a4cf1e9d035975280b3ea485de63512167ccecfd4bceac
```

## Focused production gate

The final Python gate used a short pytest base directory because the
WG-worktree prefix plus pytest's default node names can exceed Linux
`AF_UNIX` path length. The short path changes no test semantics:

```bash
atomic_gate_tmp=$(mktemp -d /tmp/emender-atomic-v21-gate.XXXXXX)
"$EMENDER_PYTHON" -m pytest -q --basetemp="$atomic_gate_tmp" \
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

Final result:

```text
308 passed in 216.73s
```

An earlier invocation before the canonical installed bundle existed and with
pytest's default long temporary path produced eight native-pool fixture
failures: five missing-manifest errors and three `AF_UNIX path too long`
errors. Building/attesting the required bundle and using a short node-local
pytest temp root resolved those environment prerequisites. The native-pool
suite then passed `16/16`, and the complete gate passed `308/308`.

## Git and publication record

The regression and implementation commit is:

```text
c5600a968b850412c42fbeea0d524f815ee643f3
fix: make async v2.1 node apply atomic (fix-atomic-node-apply-v21)
```

The non-force reconciliation merge with the already-published v2.1 history is:

```text
bd4a53be9f5b6c082bfedfe449d4add6ab0f51ad
Merge remote-tracking branch 'origin/main' into
wg/agent-1555/fix-atomic-node-apply-v21
```

The validated code was pushed with:

```bash
git push origin HEAD:main
git push origin HEAD:refs/heads/wg/agent-1555/fix-atomic-node-apply-v21
git ls-remote origin refs/heads/main \
  refs/heads/wg/agent-1555/fix-atomic-node-apply-v21
```

Both remote refs resolved to
`bd4a53be9f5b6c082bfedfe449d4add6ab0f51ad`. No force push was used.
The validation-report commit and its final fast-forward publication are
recorded in the WG task log.

## Remaining live gate

V21S11 is present locally. V21S15 remains deliberately incomplete: an
authorized runner must rebuild the exact changed source/bundle, pass its
source-pinned G2 prerequisite, and run the job-5068873-class fault/restart
phase at exactly two nodes with separately retained `Partition=batch` and
`QOS=debug`. That future live artifact must show the failed old incarnation,
all-eight reconstruction, authoritative reload, no one-node commit, and a
subsequent valid contribution/READY from the new incarnation. This task did
not authorize or attempt that scheduler action.
