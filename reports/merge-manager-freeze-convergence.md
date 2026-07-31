# Manager FREEZE convergence integration

Date: 2026-07-22

## Integration scope

This integration follows *Resilient DiLoCo Compute Pool*, version 1, and the
required conformance checklist for R02-R08, R11, R14, R16 and NDP01-NDP17.
It integrates the evaluated manager FREEZE lifecycle and terminal-harvest fix
without submitting a Slurm job.

The authoritative remote started at `53441395245af7fbe767c2e25cc3ad379db07b0e`.
The reviewed task commit was `8ad78178`; its final rebased integration commit is
`bc97fc296ec1f4060cee6083960fa3e0a91de515`.  Full diffs of those two commits
had the identical SHA-256
`c1efda672f29f53930083c92dbc9fd368f7313190b3e12048c5fbb33a677d0f7`.
The integration branch was a clean descendant of the old remote tip, with
merge base `9d7d34dcacd09ed6c3b5ef8daa02151beb25be10`, and contained the reviewed
rank-containment and overlap integration immediately before the FREEZE fix.

## Validation

Canonical Frontier activation was used for every Python and native command:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The explicit native Release build and canonical CTest command were:

```bash
cmake -S native -B build/merge-manager-freeze-release \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/merge-manager-freeze-release --parallel 8
ctest --test-dir build/merge-manager-freeze-release --output-on-failure
```

Result: Release build passed and CTest passed 10/10.  In particular,
`ndp_service_rpc_integration_test` passed the replayed FREEZE assertion after
`RESULT_READY`, proving that duplicate/delayed/reordered control delivery
returns the original fenced operation instead of creating split-brain state.

The production lifecycle, route, rank-membership, overlap, deadline, and
terminal-harvest regression selection was:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_dataplane_failure.py \
  tests/test_native_pipeline.py \
  tests/test_native_pool_integration.py \
  tests/test_native_transport_bridge.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_resilient_e97_rank_lane.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_topology.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_peer_membership.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_validate_pipelined_e97_performance.py
```

The first run proved 191 assertions passed and identified five tests requiring
the canonical installed artifact manifest.  The required canonical build was
then performed rather than weakening or skipping those tests:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_native_pool_integration.py
```

Result: canonical CTest passed 10/10, artifact installation and attestation
completed, and the complete native pool integration file passed 16/16.  Thus
all 196 collected production regression cases passed with their required
native artifacts.  Coverage includes route loss/replacement and stale-route
rejection, fenced rank membership without a fixed-eight wait, strict g/g+1
overlap, active deadline refresh, terminal lane/checkpoint behavior, and the
bounded `squeue`-to-`sacct` propagation window.

## Conformance and operational result

- R02-R06/R11/R14: READY leased membership, fenced generation identity,
  per-rank progress refresh, bounded stage deadlines, and rejoin/recovery stay
  independent of launched-rank unanimity.
- R07-R08 and NDP01-NDP17: atomic committed evidence, deterministic weighted
  native math, point-to-point bounded transport, replay/backpressure/release,
  stale/corrupt rejection, and model-free managers remain unchanged.
- The production minimum floor remains two READY managers, `Q_min=2`, and
  `T_min=3,934,080` accepted tokens.  No all-rank collective, central
  full-model broker, Python dense transport, or Lustre dense hot path was
  introduced.
- The WG-managed isolated checkout remained clean except for this report and
  generated ignored build outputs.  The primary checkout was not modified.
- No `sbatch`, `srun`, or other Slurm submission command was executed.

The exact pushed remote tip, its parent/ancestry check, and the post-push
`origin/main` fetch verification are recorded in the task log after this
evidence commit is created.
