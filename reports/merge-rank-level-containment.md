# Rank-level trainer failure containment integration

Date: 2026-07-22

## Result

The evaluated rank-level containment payload is integrated with the then-current
authoritative `origin/main`.  The old main was
`5a455bc132f8809cf46e8c2a719904ee7d89281b`; the reviewed WG squash commit was
`def4c25ac32f21596acf4da6eccb07b05cb747a2`; and the clean-history reconciliation
merge is `933655f82dd02edfd33528f20a28fd54f6f87967`.

The implementation commit is an ancestor of the reconciliation merge, as is
the old main:

```text
git merge-base --is-ancestor def4c25a 933655f8  # exit 0
git merge-base --is-ancestor 5a455bc1 933655f8  # exit 0
```

Only reviewed histories were reconciled.  `def4c25a` is the WG-produced squash
of the evaluated `implement-rank-level` branch (implementation tip
`f14818728c8b53d02a97a3802c5c55b16844ccd6`, failing-first specification
`70ea99fd`).  The other merge parent is exactly the fetched old `origin/main`.
The merge completed without a content conflict.

## Failure-domain behavior

The topology and supervisor tests prove that a single trainer/GPU process can
exhaust its restart budget and be retired without terminating its CPU manager,
native service, or seven healthy sibling trainers.  Membership is expressed as
leased `(stable rank, process incarnation)` identities, so no failure-sensitive
all-launched-rank wait is introduced.  Generation closure remains governed by
eligible contribution and positive accepted-token floors.

That rank retirement is intentionally distinct from whole-node loss.  A node
failure still removes the node's complete manager/service/trainer role group
and is handled by pool membership and quorum policy; a rank failure revokes
only that trainer incarnation when the configured failure scope permits.  A
restarted rank receives a later incarnation and must catch up under the current
fence.  Older-epoch/fence work is rejected.  Reducer coverage verifies that
accepted tokens, not participant count, determine contribution weight.

Production overlap behavior remains intact: launcher coverage still requires
independent `srun --overlap --no-kill` steps, and the native pipeline/runtime
tests retain delayed-result admission, generation identity, stale/future
rejection, and strict telemetry checks.

## Validation

All commands ran after the canonical Frontier activation.  No `sbatch`,
`srun`, Slurm launcher, or job-submission command was executed.

Focused rank membership, quorum, single-GPU exit, stale-work rejection,
actual-token weighting, fenced rejoin, and Slurm/supervisor containment:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_topology.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_resilient_e97_reducer.py \
  tests/test_resilient_e97_true_2n_launcher.py
```

Result: **85 passed in 41.62s**.

Canonical native build, controller/transport tests, install, and artifact
attestation:

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
```

Result: build and install passed; canonical CTest **10/10 passed**; native
artifact manifest recorded.

Explicit native controller Release build:

```bash
source scripts/frontier/activate_emender_frontier.sh
cmake -S src/native_resilient_dataplane \
  -B build/merge-rank-level-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/merge-rank-level-release -j 8
ctest --test-dir build/merge-rank-level-release --output-on-failure
```

Result: Release build passed; CTest **1/1 passed**.

Production overlap, runtime, exact two-node launcher, performance admission,
and strict telemetry regressions:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_pipeline.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_validate_pipelined_e97_performance.py
```

Result: **118 passed in 131.19s**.

## Architecture conformance

This integration was checked against the required conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and the companion gap matrix.

- R01-R04 and R11: allocation and generation fencing remain mandatory; READY
  membership now distinguishes stable rank from process incarnation, and a
  rejoin cannot reuse a revoked incarnation.
- R05-R07 and R15: explicit Q/T floors, deterministic aggregation, stale work
  rejection, and actual accepted-token weighting pass unchanged.
- R08-R10 and R12: owner selection, bounded native buffers, model-free manager
  ownership, checkpoint identity, and recovery remain intact.
- R13-R16: backend separation, bounded deadlines/telemetry, and the two-node
  gate are retained.  This integration does not claim a new live G3 result or
  readiness for 4+ nodes.
- NDP01-NDP17: no dense-path ABI was changed.  NDP02 and NDP13 are strengthened
  by rank-local containment; NDP06, NDP09-NDP11, NDP15, and NDP16 remain covered
  by identity, backpressure, replay, fenced publication, and telemetry tests.
  NDP17 remains at the previously retained synthetic G2 gate.

Checklist result: membership is leased and bounded; generation identity is
fenced; aggregation is deterministic and token-weighted; stale/corrupt work is
rejected; commits are atomic; dense transport remains bounded, point-to-point,
non-Lustre, and collectively independent.  Trainer loss and whole-node loss
remain separate policies.

## Worktree isolation

Work occurred only in WG worktree
`.wg-worktrees/agent-1421`.  The primary checkout remained at
`9d7d34dcacd09ed6c3b5ef8daa02151beb25be10`; its pre-existing user/untracked
files were neither staged nor modified.  No stash, reset, destructive checkout,
force-push, or Slurm submission was used.
