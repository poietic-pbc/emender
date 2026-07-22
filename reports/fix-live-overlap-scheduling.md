# Live K40/background overlap scheduling fix

Date: 2026-07-22

## Root cause and implementation

Retained job 5047497 proved the serialization in the trainer control flow:
after `publish_model_delta()` and the ownership handoff, the foreground loop
immediately called `result_shards()`, waited for the native result and apply
lane, applied it, and synchronously wrote/promoted the recovery checkpoint.
Only then could the generation loop enter the next K40 window
(`scripts/frontier/resilient_e97_role.py:2391-2494` at the retained source).
The existing two-slot policy object did not create a service thread; its
handoff token was released immediately and the subsequent blocking calls
remained on the trainer thread.

`LiveNativeGenerationScheduler` implements the missing split.  The producer
enqueues an immutable, fully identified generation snapshot into a bounded
latest-only mailbox and returns.  One service thread owns collection,
reduction, integrity scanning, redistribution, and checkpoint publication.
Only a complete result returned after publication can enter the result
mailbox.  Background exceptions mean non-participation for that round.  The
trainer polls and applies an accepted result only through
`apply_at_safe_boundary`; generation, run, source, route, lease, fence,
incarnation, layout, and base identity are checked.  Result admission and
application are replay-safe and latest-only.

The handoff bound is one queued plus one running item.  A newer queued item
replaces the old item and increments both replacement and drop counters.
Optional safety backpressure has an explicit deadline, returns false on
expiry, and emits `backpressure_timeout` with the measured monotonic wait.
There is no unbounded queue or implicit all-rank wait.

All scheduler telemetry uses `time.monotonic_ns()` and carries the complete
`GenerationIdentity`. Events cover enqueue/replacement/backpressure, K40
start/end (foreground call sites), arbitrary native phase callbacks,
accepted-result ready/rejection, safe-boundary apply, background failure, and
checkpoint publication.

## Validation

Commands were run from the repository root. The canonical Frontier activation
was sourced before validation:

```text
source scripts/frontier/activate_emender_frontier.sh
```

Focused deterministic regression:

```text
.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_native_pipeline.py
13 passed in 22.36s
```

This includes a direct monotonic interval assertion that generation-0
collection/publication overlaps generation-1 simulated K40, plus foreground
non-wait, slow service, latest-only replacement, bounded backpressure
accounting, missing/failed publication, stale/duplicate/non-finite rejection,
wrong route identity, and restart with a pending delayed update applied once.

The strict five-generation validator was not changed:

```text
.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q \
  tests/test_validate_pipelined_e97_performance.py
6 passed
```

Native Release build and CTest:

```text
cmake -S src/native_resilient_dataplane \
  -B build/fix-live-overlap-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/fix-live-overlap-release -j 8
ctest --test-dir build/fix-live-overlap-release --output-on-failure
100% tests passed, 0 tests failed out of 1
```

The combined controller suite reached 41 passing tests; its retained live
artifact loader test could not load `libamdhip64.so.6` outside the fully
propagated ROCm loader environment. This is an environment/load failure before
the tested controller path, not a test assertion failure.

No `sbatch`, Slurm submission, or 4n+ command was run.

## Architecture conformance

Authority checked: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. The applicable checklist is R01-R16 and
NDP01-NDP17. The bounded ownership/mailbox and fail-closed identity rules
conform to R01, R05, R08, R12, R14, R16 and NDP03, NDP06, NDP09, NDP11,
NDP14, NDP16, NDP17. In particular, the implementation preserves immutable
ownership, bounded memory, non-participation on late/missing work, fenced
admission, atomic-publication-before-readiness, and direct monotonic overlap
evidence. The downstream exact-two-node task remains responsible for the sole
authorized live validation run.
