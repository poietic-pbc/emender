# Merge async decoupled DiLoCo v2 validation

Date: 2026-07-24

Task: `merge-async-decoupled-diloco-v2`

Policy: `async-decoupled-v2.0-exp`

Scope: reviewed integration onto the fetched authoritative `origin/main`,
local/reference/native validation, and a non-submitting exact-two-node render.
No Slurm command was run, no allocation was submitted, and no 4+ node use is
authorized or claimed.

## Integration outcome

The fetched integration inputs were:

- `origin/main`:
  `2e485fa70588f6b5b764416e8efac3dcfa6aaee4`;
- reviewed async-v2 worktree:
  `8886bef188de2d2d221e85967d52bba0bab30aac`; and
- merge base, after the final-seed staging work:
  `03e75b1501008cac0632ed62b548ee5fdd4e8c5b`.

`git merge --no-commit --no-ff origin/main` completed without conflicts. The
automatic merge tree was byte-identical to the reviewed async-v2 worktree tree
(`a93b3afaf15c200f36ff6b37858d6904a7efcc1d`). This is the intended resolution:
the current-main ancestry and final seed/sbcast integration are retained, while
the older serial `tau=0` overlap and validator behavior does not replace the
reviewed v2 files.

The production launcher and role reject policy drift before model load.
Production selects `async-decoupled-v2.0-exp`, `K=40`, global lag hard/target
`6/2`, speculative-window hard/target `8/2`, `Q_min=2`,
`T_min=3,934,080`, one sealed descriptor, one mutable cumulative interval, one
visible result, one replacement staging result, and stateless outer
`eta=0.5`. The older serial generation scheduler remains only a named v1
compatibility fixture and is not imported or constructed by the production
trainer. False `tau=0` policy or telemetry labels fail closed.

## Changed paths relative to the fetched `origin/main`

- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`
- `docs/ASYNC_QUORUM_DILOCO.md`
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`
- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `docs/validation/codify-async-decoupled-diloco-v2-20260724.md`
- `docs/validation/implement-async-decoupled-diloco-v2-20260724.md`
- `docs/validation/merge-async-decoupled-diloco-v2-20260724.md`
- `docs/validation/wire-async-v2-real-trainer-loop-20260724.md`
- `ndm/async_diloco_real.py`
- `ndm/async_diloco_v2.py`
- `ndm/native_e97_runtime.py`
- `ndm/resilient_e97_runtime.py`
- `ndm/resilient_node_quorum.py`
- `ndm/resilient_pool_runtime.py`
- `reports/frontier/native-dataplane-reference-v1.json`
- `reports/frontier/native-dataplane-reference-v1.md`
- `scripts/frontier/render_resilient_e97_exact_2n_acceptance.py`
- `scripts/frontier/resilient_e97_allocation_supervisor.py`
- `scripts/frontier/resilient_e97_role.py`
- `scripts/frontier/resilient_e97_true_2n.sbatch`
- `scripts/frontier/validate_pipelined_e97_performance.py`
- `tests/test_async_diloco_real_trainer.py`
- `tests/test_async_diloco_v2.py`
- `tests/test_resilient_e97_exact_2n_acceptance.py`
- `tests/test_resilient_e97_runtime.py`
- `tests/test_resilient_e97_true_2n_launcher.py`
- `tests/test_validate_pipelined_e97_performance.py`

The already-pushed final seed/sbcast paths on `origin/main` remain present and
unchanged by the integration. In particular, the rendered launcher uses
compiled `sbcast`, node-local `--verify-local`, offline tokenizer staging, and
the immutable final E97 checkpoint:

- step: `2,300,930`;
- accepted tokens: `150,793,748,480`;
- bytes: `7,719,680,116`; and
- SHA-256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.

## Production-path audit

The audited production path is:

```text
resilient_e97_true_2n.sbatch
  -> exact native-cxi / FI_PROVIDER=cxi attestation
  -> resilient_e97_allocation_supervisor.py
  -> model-free NativeManagerSession plus eight model-owning trainers per node
  -> NativeTrainerDataPlane direct service-owned memfd submission
  -> local OWNED
  -> PersistentAsyncTrainingLane starts the next resident K40 window
  -> compiled point-to-point result/reduction/redistribution control lane
  -> immutable checkpoint and fenced authoritative latest CAS
  -> reload verification
  -> PersistentAsyncTrainingLane.finish_at_boundary
  -> accepted-delta correction of model x, ScheduleFree z, and interval start
```

The real trainer creates one `PersistentRealWorkerSession`, preserving the
model, optimizer, iterator, hidden state, and audited ScheduleFree state across
adjacent windows. After local `OWNED`, it does not wait for fabric send,
receipt, reduction, checkpoint, or publication before starting the next K
window. The result lane releases the bounded native view, verifies the
immutable global handoff and current-fence latest CAS, and applies only at an
exact K boundary.

The production role contains none of `pickle`, Python dense `sendall`/`recv`,
`mpi4py`, `MPI_`/`PMPI_`, `all_reduce`, `all_gather`, `barrier`, `TCPStore`,
`torch.distributed`, `NativeGenerationPipeline`, or
`LiveNativeGenerationScheduler`. Dense data stays in the compiled persistent
memfd/CXI service. Managers remain model-free, owners are sharded, native
operations are bounded point-to-point, and no launched-rank or all-rank wait is
introduced.

## Mathematical and protocol audit

The deterministic reference admits global lag `0..6`, rejects lag seven,
sorts the frozen set by contribution digest, and keeps exact tokens separate
from staleness-adjusted aggregation weight:

```text
aggregation_weight_i = exact_tokens_i * (7 - commit_lag_i)
S_(g+1) = S_g + 0.5 * weighted_mean(delta_i)
```

The accepted-token clock advances by exact tokens, not aggregation weight.
The outer state atomically retains `mode=delta_sgd`, `eta=0.5`, outer step, and
accepted tokens. The boundary correction is:

```text
correction_i = (S_h - S_a) - C_i(a,h)
```

It translates model `x`, audited ScheduleFree parameter point `z`, and the
mutable interval start, while preserving scalar/moment state. Tests exercise
accepted and nonaccepted workers, skipped mailbox versions, capacity-one
visible/staging mailbox semantics, one sealed plus one mutable interval,
sigma-bound pause, stale interval drop, duplicate replay, corrupt/nonfinite
rejection, current membership/incarnation/fence checks, failed publication,
and fresh-allocation restoration of global model and outer state.

The semantic performance validator does not infer lag from stage names. It
requires explicit base, commit, applied-anchor, result-version, and local
window clocks; true versioned overlap; exact token/lag weights; bounded queue
high-water; local `OWNED <= 1 s`; at least two warm-up plus ten measured K40
windows for each of 16 trainers; cadence `<= 1.25x`; foreground idle `< 0.10`;
promotion lag targets `<= 2`; and independent
freeze-to-reload-verified-latest latency `<= 420 s`.

## Conformance checklist

This integration was reviewed against the authoritative
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
[`ASYNC_DECOUPLED_DILOCO_V2.md`](../ASYNC_DECOUPLED_DILOCO_V2.md), and
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md).

- R01–R04: exclusive admission/fencing, READY membership, current active
  snapshots, stable worker/new incarnation identity, stale rejection, and
  idempotence remain in the production control path.
- R05–R08: deterministic binary64 weighted math, exact token floors, finite
  group deadlines, atomic publication, sharded ownership, backpressure,
  bounded replay, and release remain tested.
- R09–R12: managers/services are model-free; trainers own local inner state;
  no Python/Lustre dense hot path is added; late/rejoining workers synchronize
  under a new incarnation; fresh fences restore only authoritative global
  model/outer state.
- R13–R16: scheduler-specific code remains outside the protocol, every wait is
  bounded and observable, mathematical/reference tests cover changing
  participation, and qualification remains two-node only.
- NDP01–NDP04: Python owns metadata/policy; compiled C++ owns dense handoff and
  transport; the production path is point-to-point and collective-free; local
  handoff is service-owned memfd without a trainer-sized spool.
- NDP05–NDP12: deterministic reduction/layout, versioned fenced identity,
  leased endpoint routes, finite buffers/credits, checksum and once-only
  apply, bounded replay/reassignment, and owner-direct redistribution remain
  compiled and tested.
- NDP13–NDP17: deadlines, stable ABI, fenced checkpoint handoff, required
  telemetry, exact-CXI attestation, and the retained full-layout G2
  prerequisite remain enforced.
- V2A01–V2A07: explicit v2 mode, continuous adjacent K40 windows, full
  contribution identity, independent lag clocks, finite group/queue bounds,
  exact lag weights, and half-step outer math are present and tested.
- V2A08–V2A14: correction-ledger safe apply, latest-only verified mailbox,
  conservative resident admission, ordered boundary/backpressure rules,
  missing/rejoin behavior, corruption/idempotence, atomic publication, and
  fresh-fence restart are present and tested.
- V2A15–V2A17: the compiled model-free point-to-point dense path is retained;
  the validator measures honest overlap, cadence, foreground idle, lag, and
  correctness latency separately.
- V2A18: local numerical, failure/restart, native, launcher, and renderer gates
  pass. Live two-node failure/performance/convergence qualification remains the
  downstream task; this merge does not claim it.

## Exact validation commands and results

Every Python, pytest, native build, and renderer command was run after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Canonical native build and compiled tests:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: configure, build, install, and manifest recording succeeded; all
`10/10` CTests passed in `2.52 s`. The manifest was written to
`build/native-resilient-dataplane/native-artifacts.json`.

Canonical focused async/reference/production suite:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent-1494-focused-rerun \
  tests/test_async_diloco_v2.py \
  tests/test_async_diloco_real_trainer.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_native_dataplane_reference.py
```

Result: `181 passed in 203.51s`.

The first focused invocation was intentionally retained in the task log:
before the native build existed in this fresh worktree it produced
`173 passed, 1 failed, 7 errors in 194.49s`; every failure/error was a
`FileNotFoundError` for `libemender_ndp.so.1` or `ndp_cxi_service`. Building the
required native prerequisite and rerunning the complete suite produced the
clean result above.

Native Python/ABI/runtime suite:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent-1494-native \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_pipeline.py \
  tests/test_native_pool_integration.py \
  tests/test_native_pool_production_policy.py \
  tests/test_native_transport_bridge.py \
  tests/test_native_artifact_attestation.py \
  tests/test_validate_native_dataplane_local.py \
  tests/test_validate_native_dataplane_2n_gate.py
```

Result: `90 passed in 117.75s`.

Non-submitting exact-two-node renderer:

```bash
"$EMENDER_PYTHON" \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo . \
  --native-build-manifest \
    build/native-resilient-dataplane/native-artifacts.json \
  --full-layout-gate \
    reports/frontier/native-dataplane/5034807/full-layout-gate.json \
  --run-root /tmp/emender-agent-1494-rendered-runs \
  --output /tmp/emender-agent-1494-async-v2-plan.json \
  --allow-non-authoritative-dry-run
```

Result: rendered successfully without submission. The plan records schema
`emender-real-e97-exact-2n-acceptance-v2`, exactly two nodes,
`Partition=batch`, `QOS=debug`, the reviewed v2 constants, the final immutable
seed, all R01–R16/NDP01–NDP17/V2A01–V2A18 IDs, and clean-overlap,
fault/rejoin, invalid-result, failed-publication, and fresh-restart phases.

Repository integrity:

```bash
git diff --check
git diff --cached --check
cmp -s AGENTS.md CLAUDE.md
git status --short --untracked-files=all
```

Result: all checks exited zero and tracked/untracked status was clean before
this validation record was added. Native build products are ignored under
`build/`; no generated or untracked file is included in the integration.
