# Production generation-gap authoritative integration

Date: 2026-07-23  
Task: `merge-production-generation-gap`

## Integrated history

The authoritative remote started at
`5f180e25fd0a54852892d5ff59a97b3c1d8737ff`. The evaluated implementation
commit is `7e439b23500cc76aed48e74b3ddc02c026fe80ca`; WG had already produced the
patch-identical squash commit `d817ec740187ab68531f43204c6210ca82448beb`.
`git diff 7e439b23 d817ec74` was empty.

Integration reconciled the clean isolated task branch with the fetched old
`origin/main`, then made the evaluated commit an explicit merge parent. This
preserves review ancestry instead of relying only on patch equivalence:

```text
git merge --no-ff origin/main \
  -m "merge: reconcile authoritative main for generation-gap integration"
git merge --no-ff -s ours 7e439b23 \
  -m "merge: retain reviewed generation-gap ancestry (merge-production-generation-gap)"
git merge-base --is-ancestor 7e439b23 HEAD  # exit 0
git merge-base --is-ancestor 5f180e25 HEAD  # exit 0
```

The `ours` merge is intentionally content-neutral because the complete
evaluated patch was already present as `d817ec74`. No unevaluated runtime
change was introduced.

## Validation

Every Python, native, controller, and launcher command followed canonical
Frontier activation:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Canonical native build, CTest, installation, and artifact attestation:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
```

Result: build and install passed; canonical CTest **10/10 passed**; the native
artifact manifest was recorded.

Explicit native controller Release validation:

```bash
cmake -S src/native_resilient_dataplane \
  -B build/merge-production-generation-gap-release \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/merge-production-generation-gap-release -j 8
ctest --test-dir build/merge-production-generation-gap-release \
  --output-on-failure
```

Result: Release build passed and CTest **1/1 passed**.

The complete production overlap/timing, cadence-budget, strict telemetry,
freeze convergence, rank containment, quorum/fencing, checkpoint/restart,
controller/launcher, and native integration selection was:

```bash
export TMPDIR=/tmp
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_pool_integration.py \
  tests/test_native_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_topology.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_resilient_e97_exact_2n_acceptance.py
```

Result: **163 passed in 181.11 seconds**. `TMPDIR=/tmp` is required only to
keep test AF_UNIX socket names under Linux's 108-byte bound in the long
WG-managed worktree path. Before that correction, the broader selection
passed 195/198; its only failures were the three overlong socket paths.

The exact regressions prove direct monotonic `g`/`g+1` foreground/background
overlap, less than 10% foreground idle, and at most 1.25x steady-state cadence
when background work fits. The retained 5055899 cadence budget closes as an
exclusive interval union:

```text
63.369 s K40 + 0.817 s handoff + 293.770 s labeled synchronous tail
= 357.956 s cadence; 22.230 s nested overlap; 0.000 s unaccounted
```

The strict validator thresholds and freeze/rank/quorum/fence/checkpoint/restart
tests were not modified. Production-path regressions retain the marker
`emender-production-delayed-pipeline-v1` and verify that live selection does
not fall back to a synchronous Python dense or fixed-world hot path.

## Architecture conformance

This integration applies the mandatory conformance checklist from *Resilient
DiLoCo Compute Pool*, version 1 (2026-07-17), and its companion gap matrix.

- **R12:** pending delayed outer state, accepted-token clock, and fresh-process
  restart remain identity-bound and once-only across checkpoints.
- **R14 / NDP16:** monotonic stage spans, exclusive-union budget closure,
  provider/identity/byte/bound/release telemetry, and strict cadence/idle
  admission pass without weakening thresholds.
- **R16 / NDP17:** this is an authoritative-source integration after retained
  synthetic G2 and exact two-node evidence. It authorizes only the next exact
  two-node validation, not a Slurm submission here and not 4+ scale.
- **R02-R08, R10-R11, R15 and NDP01-NDP15:** READY leased membership, bounded
  waits, fenced identities, deterministic token-weighted math, stale/corrupt
  rejection, atomic publication, point-to-point bounded transport,
  backpressure/release, and no central full-model broker remain covered.

The minimum production floor remains two READY managers, `Q_min=2`, and
`T_min=3,934,080` accepted tokens. No launched-rank/all-rank invariant,
Python dense transport, Lustre dense hot path, or synchronous collective was
introduced.

Work occurred only in the WG-managed isolated worktree. The primary checkout
and its user changes were not modified. No `sbatch`, `srun`, or other Slurm
job command was executed.

The final evidence commit, pushed `origin/main` SHA, remote ancestry checks,
and `git ls-remote` verification are recorded in the task log because a commit
cannot embed its own content-derived SHA.
