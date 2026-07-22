# Production overlap entrypoint authoritative integration

Date: 2026-07-22  
Task: `merge-production-overlap-entrypoint`

## Integrated ancestry

The integration used a clean temporary Git worktree created directly from the
fetched `origin/main`; the WG-managed primary checkout and its unrelated state
were not modified.  The remote was `32fd9ab15c6908827d094b21ff638f8ec2a24c2b`
before integration.  Evaluation task
`.evaluate-fix-production-overlap-entrypoint` completed successfully at
2026-07-22T13:57:18Z.

The reviewed source history was merged without rebasing or squashing:

- `670ab7fb680ff18e83141b3afea0d3cf343d2261` — production role binding,
  marker, renderer-path regression, and investigation report.
- `283665dca92d52e3420d8f721c9e474e12e6e54d` — one-generation result
  admission and stale/future boundary regressions.
- `0de86fbd` — integration merge based on remote main. Both reviewed commits
  are parents/ancestors, not merely patch-equivalent cherry-picks.

The final pushed and remotely fetched `origin/main` SHA is recorded in the WG
task log after the push, because a commit cannot embed its own content-derived
SHA.  Remote verification also checks both reviewed SHAs with
`git merge-base --is-ancestor`.

## Production path verification

The exact acceptance renderer names
`scripts/frontier/resilient_e97_true_2n.sbatch`.  The rendered batch binds
`ROLE="$REPO/scripts/frontier/resilient_e97_role.py"`; that production role
constructs `ndm.native_pipeline.LiveNativeGenerationScheduler` with
`result_delay=1` and emits the exact runtime schema marker
`emender-production-delayed-pipeline-v1`.  The focused renderer regression
asserts this complete chain rather than invoking the scheduler only as a test
fixture.

The production boundary regression proves that committed result generation
`g` is rejected at boundary `g`, applied once at boundary `g+1`, and rejected
when future or two generations stale.  The strict live validator was unchanged
and all six of its tests passed, including the monotonic g/g+1 overlap rule.

## Validation commands and results

All Python and native commands ran after:

```text
source scripts/frontier/activate_emender_frontier.sh
```

Combined production/native/controller/launcher/strict suite:

```text
$EMENDER_PYTHON -m pytest -q \
  tests/test_native_pipeline.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_validate_pipelined_e97_performance.py
```

The clean pre-build pass produced 117 passes and one expected missing-native-
library setup failure.  After the required native build, the affected complete
runtime suite passed 36/36.  Thus every collected test assertion passed; the
initial failure was resolved by performing the required build, not by changing
code or validator thresholds.

Canonical native build and CTest:

```text
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
```

Result: build passed and canonical CTest passed 10/10.

Explicit Release build and CTest:

```text
cmake -S src/native_resilient_dataplane \
  -B build/merge-production-overlap-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/merge-production-overlap-release -j 8
ctest --test-dir build/merge-production-overlap-release --output-on-failure
```

Result: Release build passed and CTest passed 1/1.

No `sbatch`, `srun`, Slurm launcher, or job-submission command was executed.

## Architecture conformance

Authority is *Resilient DiLoCo Compute Pool*, version 1 (2026-07-17), and the
companion gap matrix requirements R01-R16 and NDP01-NDP17.

- R01-R07, R10-R11 and NDP06-NDP10: scheduler and result admission retain the
  full fenced run/generation/attempt/incarnation/base identity, exact stale and
  future rejection, idempotent once-only admission, and atomic committed
  evidence. The strict MVP remains `tau=0`.
- R02-R06, R12-R14 and NDP01-NDP04/NDP13: READY membership, quorum floors,
  deadlines, and native point-to-point ownership remain unchanged. There is no
  launched-rank barrier, Python dense transport, Lustre dense hot path, or
  central full-model broker.
- R07-R10, R15-R16 and NDP11-NDP17: bounded native replay/backpressure/release,
  deterministic token-weighted math, checkpoint fencing, exact source marker,
  and telemetry remain the reviewed implementation. The applicable regression
  and native gates passed locally.

The production minimum-progress policy remains the approved two READY manager
peers with `Q_min=2` and `T_min=3,934,080` accepted tokens. This integration is
an authoritative-source gate only and makes no new live-job or scale claim.
