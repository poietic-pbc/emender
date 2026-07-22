# Overlap scheduler and final E97 seed integration

Date: 2026-07-22 UTC. Task: `merge-overlap-and-final-seed`.

## Integrated history and remote sources

The integration was performed in a clean sibling worktree created from the
then-current `origin/main`, `5c4950f16bd9ce7cb7d96ab9b67e24efb61ed3a6`.
The primary checkout, including its user-owned untracked GatedDeltaNet-2
handoff/launcher checkout and run artifacts, was neither modified nor cleaned.

Both reviewed source commits were verified by `git ls-remote` on their remote
task refs:

- `origin/wg/agent-1401/fix-live-overlap-scheduling`:
  `84a81e40168feff60367faf1f7a312b6e7ea5a0a`
- `origin/wg/agent-1402/integrate-final-e97-s3-seed`:
  `b559ade9fda003f4642e061ea5cf5617b004a56d`

The reviewed commits already form a deterministic, conflict-free linear
history. The final-seed feature commit
`7eb4b437bc43f7e64d560bd52c2b442069305683` has the overlap commit as its
parent, and `b559ade9` has `7eb4b437` as its parent. Consequently the exact
history was integrated with `git merge --ff-only b559ade9`, preserving both
source SHAs rather than synthesizing cherry-pick replacements. The only files
between old main and `b559ade9` are the 11 reviewed scheduler, S3 launcher,
test, and validation artifacts; no primary-checkout-only content is included.

The authoritative post-push `origin/main` SHA and `git ls-remote` verification
are recorded in the WG task log because a commit cannot embed its own SHA.

## Validation

The canonical Frontier environment was loaded with:

```text
source scripts/frontier/activate_emender_frontier.sh
```

The combined focused suite passed:

```text
export EMENDER_NDP_LIBRARY="$PWD/build/merge-overlap-final-seed-install/lib64/libemender_ndp.so.1"
export EMENDER_NDP_SERVICE="$PWD/build/merge-overlap-final-seed-install/bin/ndp_cxi_service"
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_e97_s3_seed.py \
  tests/test_e97_async_256_promotion.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_true_2n_launcher.py
# 186 passed in 288.93s
```

This covers generation-g background overlap with generation-(g+1) simulated
K40, the unchanged strict overlap validator, exact step-2300930 authority,
atomic job-scoped staging, byte/hash checks, pointer drift and legacy-path
rejection, canonical render/fingerprint parity, mutation coverage, stale and
inadequate promotion evidence rejection, controller behavior, and canonical
launcher behavior. Smoke and production retain the reviewed fingerprint
`ef6f52145e34c056c154f0d162dec47ec96d02d883f540ebdf6f793427801ec3`.

Shell parsing and Python compilation passed:

```text
bash -n scripts/frontier/trainpy_async_quorum_smoke_common.sh \
  scripts/frontier/build_native_resilient_dataplane.sh
"$EMENDER_PYTHON" -m py_compile ndm/native_pipeline.py \
  scripts/frontier/materialize_e97_s3_seed.py \
  scripts/frontier/render_e97_async_256.py
```

The native bundle was configured and built as Release through the canonical
wrapper, with the canonical ROCm and libfabric runtime directories and loader
environment. All native tests passed:

```text
BUILD_TYPE=Release \
BUILD_DIR="$PWD/build/merge-overlap-final-seed-release" \
INSTALL_DIR="$PWD/build/merge-overlap-final-seed-install" \
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
# 100% tests passed, 0 tests failed out of 10
```

An initial combined Python attempt reached 185 passes and failed one retained
runtime reconnect test because no native library had yet been built in the
fresh worktree. Building the Release bundle and exporting the attested install
paths fixed the environment precondition; the complete 186-test command then
passed. No ROCm/native loader failure was waived.

No `sbatch`, `srun`, or Slurm submission command was issued.

## Architecture conformance

Authority checked: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1 and
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. The integration preserves the complete
R01-R16 and NDP01-NDP17 architecture. The directly applicable scheduler
requirements are R01, R05, R08, R12, R14, R16 and NDP03, NDP06, NDP09,
NDP11, NDP14, NDP16, NDP17. The directly applicable immutable-seed requirements
are R07, R09, R10, R12, R16 and NDP15-NDP16. The combined tests preserve
bounded handoff, fenced identity, atomic publication, strict stale/corrupt
rejection, immutable job-scoped staging, no legacy seed mutation, and the
ordered two-node-before-scale gate. The downstream exact-two-node task owns
the next live Slurm run.
