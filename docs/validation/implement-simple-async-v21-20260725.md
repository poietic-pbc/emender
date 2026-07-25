# Simple asynchronous DiLoCo v2.1 implementation validation

Date: 2026-07-25 UTC

WG task: `implement-simple-async-v21`

Implementation commit:
`8ffe2018f8320d25622a248571aff0fbca31037d`

Implementation tree:
`c7f9a3082bce7c0e31380961207e00e3e5a07c9e`

## Result

The pre-Slurm implementation gate passes. The rendered E97 trainer, persistent
Python control plane, compiled native ABI and fabric frame identity, checkpoint
and manifest contracts, semantic validator, exact renderer/launcher, and
canonical serial qualification controller now use the reviewed
`async-decoupled-v2.1-simple` policy.

The focused Python/native suite passed 227/227 tests. The canonical native
bundle rebuilt under the Frontier project environment and passed 10/10 CTests.
Real controller dry-runs pinned all two-node gates to `Nodes=2`,
`Partition=batch`, and `QOS=debug`; a four-node dry-run without signed
authorization and an exact predecessor pass failed closed.

No `sbatch`, `srun`, `squeue`, `sacct`, cancellation, or other Slurm mutation
was executed by this task. Failed historical jobs 5066495 and 5068873 remain
non-qualifying evidence. In particular, this report does not claim that the
two-node qualification gates, the atomic-restart fault campaign, or any scale
rung has passed.

## Authorities read before implementation and validation

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`
- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`

Every Python, pytest, native build, CTest, renderer, and controller invocation
below followed:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Python commands used `"$EMENDER_PYTHON"` and native wrappers received
`PYTHON_BIN="$EMENDER_PYTHON"`. `TMPDIR=/tmp` was used for tests that create
AF_UNIX sockets, because the WG-managed Lustre worktree path exceeds the UNIX
socket path limit.

## TDD chronology

The first implementation attempt recorded the red phase in the WG task log at
2026-07-25 08:31 UTC: the newly added v2.1 policy and controller tests failed
during collection because `ASYNC_DECOUPLED_V21` and the canonical controller
did not exist. The implementation then added the v2.1 boundary and made the
following required cases pass:

- lag 0/1/2 admission and lag-3 rejection/catch-up;
- unequal exact-token aggregation with `eta_outer=1.0`;
- v2.0 policy/schema/digest rejection;
- one native-owned plus one mutable interval and third-cohort rejection;
- all-eight-trainer marker transaction and fresh-incarnation reset;
- missing authorization, wrong predecessor, unchanged failed payload, and
  more-than-one-active-job rejection;
- finite V21S17 close behavior over a leased READY snapshot.

The primary red/green tests are
`tests/test_async_diloco_v21.py` and
`tests/test_async_v21_qualification_controller.py`.

## V21S01–V21S17 conformance

| ID | Production anchor | Test/evidence anchor | Local status |
|---|---|---|---|
| V21S01 | `AsyncV21Policy`, `ContributionIdentity`, `ndp_submit_v21`, and v2.1 native fabric magic/version | v2.0 policy/schema/digest and native ABI/protocol rejection tests | Present |
| V21S02 | four independent lag clocks in the policy, contribution, native request, result admission, and semantic validator | lag 0/1/2, lag-3, skipped-result, and pause/catch-up tests | Present |
| V21S03 | `exact_tokens` is the sole v2.1 quantitative field; native v2.1 maps it directly to the deterministic binary64 reducer | unequal-token reference and native permutation tests; forbidden-field source audit | Present |
| V21S04 | K40 and stateless exact-token mean with eta one in `AsyncV21CommitAuthority` and the real trainer | eta-one reference and persistent-session tests | Present |
| V21S05 | fenced stable worker/incarnation/window/base/policy/layout/code/token/payload identity and exact two-node profile | replay/conflict/incarnation, render, and pool admission tests | Present |
| V21S06 | one `PersistentRealWorkerSession`, one background lane, one immutable native-owned descriptor, and one mutable adjacent interval | no-rebootstrap, held-result overlap, coalescing, and third-cohort tests | Present |
| V21S07 | streamed ScheduleFree x/z boundary translation plus accepted-contribution correction ledger | selected/nonselected, skipped mailbox, exact replay, and once-only correction tests | Present |
| V21S08 | `LatestResultMailbox` and fenced immutable checkpoint/latest publication | capacity-one/staging, corrupt/nonfinite/fence/incarnation, and failed-publication tests | Present |
| V21S09 | exact resident formula, native registered-slot/credit/replay bounds, and dense-cohort high-water checks | resident formula, third cohort, native credit/replay/release tests | Present |
| V21S10 | leased `PeerMembership` READY snapshots, expiry, stable-worker incarnation, and paused two-node floor | membership change, expiry/rejoin, frozen identity, and missing-peer tests | Present |
| V21S11 | `AtomicEightTrainerApply` gates the one node-applied marker and READY on eight matching fenced recovery receipts | seven-marker rejection, eight-marker commit, and new-incarnation reset test | Present for the local implementation contract; the job-5068873 fault class remains assigned to `fix-atomic-node-apply-v21` for deterministic orchestration repair and live fault evidence |
| V21S12 | persistent compiled service, memfd/XPMEM producer ownership, exact-token reduction, v2.1 fabric identity, and point-to-point owner redistribution | 10/10 CTests plus ABI/reference/failure/pool integration suites | Present locally |
| V21S13 | v2.1 stage/clock/ownership/high-water telemetry and strict semantic validation | semantic-validator tests for historical identity, missing clocks, inferred overlap, nonfinite fields, and wrong math | Present locally |
| V21S14 | versioned fenced checkpoint/restore plus canonical final-seed render, submit-side verification, sbcast, and node-local offline verification | checkpoint/fresh-allocation restore and exact seed/launcher tests | Present locally |
| V21S15 | canonical serial controller and launcher pin exact two-node scheduler fields | CLI dry-runs and parametrized clean/faults/convergence tests | Tooling present; Slurm gates intentionally not run |
| V21S16 | signed immutable authorization, exact predecessor map `2→4→8→16→32→64→256`, payload state, and one-active-job lock | every-rung predecessor, missing authorization, unchanged failure, and active-job tests | Fail-closed boundary present; no promotion claimed |
| V21S17 | `validate_scale_evidence`, `V21ScaleClosure`, and scale `_pool_config` derive the finite close from digested passed two-node arrival/stage evidence | pre-close inclusion, no Q-min early close, no launched-rank membership, no all-ready barrier, and missing/unexplained-evidence rejection | Fail-closed reviewed-formula mechanism present; no scale authorization claimed |

The exact production and test paths are also recorded in the normative
V21S01–V21S17 table in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

## R01–R16 conformance checklist

| ID | Implementation check |
|---|---|
| R01 | Allocation admission remains fenced before model load; v2.1 run/fence identity is checked at contribution, mailbox, checkpoint, manager, and controller boundaries. |
| R02 | Stable-worker/incarnation membership continues through DISCOVER/READY/ACTIVE/DRAIN/EXPIRE; v2.1 commits use leased READY snapshots. |
| R03 | The elastic path uses READY membership rather than launched ranks and contains no failure-sensitive all-rank operation. |
| R04 | Contribution identity binds a fresh fence/window/base and implements deterministic duplicate/conflict/stale handling. |
| R05 | Exact tokens are the only numerator/denominator weight and reduction is deterministic binary64. |
| R06 | Commit, anchor, result, and speculative lags are distinct, telemetry-visible, and bounded at two. |
| R07 | Immutable result/checkpoint publication precedes the authoritative latest pointer; reload and CAS facts are required. |
| R08 | Native ownership, result mailbox, replacement staging, credit, replay, and resident state are explicitly bounded. |
| R09 | Trainers own model/ScheduleFree state; the compiled service is model-free and trainer buffers cross the native boundary through memfd. |
| R10 | Dense elastic traffic remains node-local/native and point-to-point; Lustre is restricted to checkpoint/control evidence. |
| R11 | Loss/lag causes bounded pause, catch-up, expiry, or restart rather than a collective barrier or invented authority. |
| R12 | Checkpoint restore retains the exact outer step and accepted-token clock and rejects old identities before mutation. |
| R13 | The source/runtime audit finds no MPI initialization or elastic all-rank collective in the v2.1 path. |
| R14 | Policy, code, layout, endpoint, trainer set, payload, seed, launcher, and evidence digests are carried at their respective boundaries. |
| R15 | K40, eta one, exact-token math, telemetry stages, and semantic validation are pinned. |
| R16 | Two-node gates precede signed promotion; every scale rung requires its exact immediate passed predecessor. |

## NDP01–NDP17 conformance checklist

| ID | Implementation check |
|---|---|
| NDP01 | Native endpoints and contributions bind run, fence, stable worker, incarnation, and leased membership. |
| NDP02 | Fabric traffic is point-to-point over current endpoints and does not depend on a fixed world. |
| NDP03 | Production provider/domain remain exact Frontier `cxi`/`cxi0`; weakening or test-provider selection fails closed. |
| NDP04 | Producers register sealed memfd/XPMEM buffers directly; Python does not serialize dense production payloads. |
| NDP05 | The compiled deterministic reducer uses exact-token binary64 numerator and denominator arithmetic. |
| NDP06 | v2.1 ABI/wire/policy/schema/base/payload/fence validation rejects v2.0, stale, corrupt, nonfinite, and incompatible work. |
| NDP07 | Owner/peer exchange and reassignment are bounded and do not become an all-rank barrier. |
| NDP08 | Registered slots, credits, resident bytes, owner state, and result views are bounded. |
| NDP09 | Credits, replay tables, result views, and native-owned descriptors have explicit high-water/release tests. |
| NDP10 | The service rejects Lustre/NFS/home dense output paths and binds content digests at ingress and publication. |
| NDP11 | Prompt release paths are exercised for buffers, operations, result views, and native ownership. |
| NDP12 | Result/checkpoint reload and atomic node-applied evidence are required before READY advances. |
| NDP13 | Deadlines, replay, pause/catch-up, owner loss, manager loss, and invalid incarnation are bounded failure states. |
| NDP14 | The real trainer remains persistent and applies correction at a K boundary exactly once. |
| NDP15 | Immutable checkpoint/latest publication restores exact authoritative state on a newer fence. |
| NDP16 | Versioned native ABI/wire, compact manifests, telemetry, and semantic validators fail closed on missing or historical identity. |
| NDP17 | The controller serializes qualification and scale promotion and submits at most one active payload. |

## Canonical validation commands and results

### Focused Python/native suite

```bash
source scripts/frontier/activate_emender_frontier.sh
TMPDIR=/tmp "$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_diloco_real_trainer.py \
  tests/test_async_diloco_v2.py \
  tests/test_async_diloco_v21.py \
  tests/test_async_v21_qualification_controller.py \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_pool_integration.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_validate_pipelined_e97_performance.py
```

Result:

```text
227 passed in 191.08s (0:03:11)
```

### Canonical native build and CTests

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result:

```text
100% tests passed, 0 tests failed out of 10
Total Test time (real) = 2.28 sec
build/native-resilient-dataplane/native-artifacts.json: status=recorded
```

### Controller dry-runs

The real CLI was invoked with `--dry-run`, immutable dummy source/policy/bundle
digests, the canonical seed digest, and a temporary `/tmp` retained-state
root. The three exact two-node results were:

```text
{"gate":"clean","scheduler":{"Nodes":2,"Partition":"batch","QOS":"debug"},"command":["sbatch","--parsable","--nodes=2","--partition=batch"]}
{"gate":"faults","scheduler":{"Nodes":2,"Partition":"batch","QOS":"debug"},"command":["sbatch","--parsable","--nodes=2","--partition=batch"]}
{"gate":"convergence","scheduler":{"Nodes":2,"Partition":"batch","QOS":"debug"},"command":["sbatch","--parsable","--nodes=2","--partition=batch"]}
```

The analogous `--gate scale --nodes 4 --dry-run` without authorization and a
prior-rung manifest exited nonzero with:

```text
ValueError: signed scale authorization and immediate predecessor pass are required
```

The unit suite separately exercises valid test-signed manifests for every
permitted rung, real Ed25519 requirement enforcement, V21S17 derivation, and
the unchanged-failure/one-active-job state checks. The `--submit` option was
never invoked.

## Forbidden-policy and transport source audit

The reachable v2.1 production files were searched for:

```text
tokens*(7-lag)
tau=6
sigma=8
eta_outer=0.5
max_commit_lag=6
max_speculative_windows=8
aggregation_weight
```

There are no v2.1 numerical fields or validator calculations using the
historical expressions or constants. The only `aggregation_weight` occurrences
in reachable production files are explicit fail-closed rejection checks in:

- `ndm/native_e97_runtime.py`
- `ndm/resilient_pool_runtime.py`
- `scripts/frontier/resilient_e97_role.py`
- `scripts/frontier/validate_pipelined_e97_performance.py`

The elastic production files contain no `mpi4py`, `MPI_Init`,
`init_process_group`, `all_reduce`, `all_gather`, or `reduce_scatter`.
Native service tests and source assertions cover sealed-memfd submission,
zero dense socket serialization, non-Lustre dense paths, absence of a central
full-model broker, bounded CXI frames/credits/replay/resident state, and prompt
release. `torch.save` remains only at the fenced checkpoint/recovery boundary,
which is not the dense transport hot path.

## Exact seed and render binding

All v2.1 qualification plans bind
`configs/frontier/e97_async_256.yaml` with:

| Field | Exact value |
|---|---|
| step | `2300930` |
| accepted tokens | `150793748480` |
| bytes | `7719680116` |
| SHA-256 | `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2` |
| node-local seed root | `/tmp/emender-e97-seed-$SLURM_JOB_ID` |
| compute-node network fetches | `0` |

The launcher verifies the submit-side authority and digest, uses compiled
`sbcast`, verifies the staged file independently on each node, and keeps
compute-node network fetch disabled. Renderer and launcher mutation tests cover
all of those exact bindings.

## Retained artifacts

- implementation commit `8ffe2018f8320d25622a248571aff0fbca31037d`
- `scripts/frontier/run_async_v21_qualification.py`
- `tests/test_async_diloco_v21.py`
- `tests/test_async_v21_qualification_controller.py`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `build/native-resilient-dataplane/native-artifacts.json`
- this validation report

The checksum-linked historical native reference was refreshed for the current
authority documents. Its JSON SHA-256 is
`838e2eceb0d1cea68b6af0e6e091baf34c6b2d2ae7ca1e7a295effc50113e338`.
