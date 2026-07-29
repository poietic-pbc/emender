# Current RC two-node qualification terminal recovery

The frozen release candidate did **not** qualify. The machine-readable
certificate in
`docs/validation/complete-v21-current-rc-2n-20260729.json` has literal
`full_pass=false` and `passed=false`. This is a fail-closed terminal record,
not an authorization or a substitute experiment.

## Decision

Immutable native G2 jobs `5111221` and `5111243` were reconciled without
resubmission. Their retained `SHA256SUMS` files verify in full, their semantic
gates both say `status=passed`, and their terminal scheduler records separately
name two nodes, `Partition=batch`, and `QOS=debug`.

The next and only model allocation, clean-overlap job `5111908`, ran the same
frozen source and payload on two `batch/debug` nodes. It failed `1:0` after
`00:05:38`; its scheduler-owned `afterany:5111908` collector `5111909`
completed independently and retained `passed=false`, `verdict=failed`. The
required `pipelined-performance.json` was absent because the model path failed
before the first K40 window.

The first semantic failure was:

```text
ValueError: native peer recovery handshake disagrees with manifest
```

At cold-start generation zero there is no prior commit. The immutable manifest
correctly carried an empty commit-receipt identity, while the compiled native
coordination reply exposed the fixed-width all-zero receipt sentinel. The
validator compared the representations literally and rejected node 0 before
READY. The trace therefore contains the allocation authority recovery and one
peer recovery only; it contains no opened generation, dense contribution,
commit, checkpoint, or apply receipt.

Per the task's first-terminal-failure rule, no delayed-READY/loss campaign and
no fresh-allocation restart were submitted. The unchanged digest was not
retried. Scale remains unauthorized.

## Frozen identities

- Source commit: `76385074da8e22bfef0044c99fe0063d2f346edf`
- Execution source: `09e9e970afd39a227c774a81be217b38ce57a083bbd074f65a9c55133becbd35`
- Clean payload: `bff64be952becfab405badcc67283631454c7be08145546101a427237d2dc646`
- Native bundle: `6e962075594cf2db36280b55e05a35fde1965e67d8beefb40a3fec776b26d908`
- Final seed: step `2300930`, tokens `150793748480`, bytes `7719680116`,
  SHA-256 `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`

Both allocated model nodes retained job-scoped offline seed materialization
records with zero network fetches and the exact seed size and digest.

## Scheduler and semantic gates

| Phase | Job | Scheduler result | Semantic result |
|---|---:|---|---|
| Native G2 clean | 5111221 | 2 nodes, batch/debug, COMPLETED 0:0, 00:03:01 | passed, SHA-256 `1b285872…3104bd` |
| Native G2 fault | 5111243 | 2 nodes, batch/debug, COMPLETED 0:0, 00:02:02 | passed, SHA-256 `25a44ee8…7fe83` |
| Model clean | 5111908 | 2 nodes, batch/debug, FAILED 1:0, 00:05:38 | failed; required performance verdict missing |
| Afterany collector | 5111909 | 1 node, batch/normal, COMPLETED 0:0, 00:00:03 | retained model `passed=false`; collector exit is not itself a pass |

The collector is deliberately a separate scheduler-owned evidence job; the
two-node `batch/debug` constraint applies to the model qualification
allocation. Its own one-node `batch/normal` identity is recorded honestly.

## Conformance and requirement mapping

The controlling authority is
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, Version 1, including its required
bounded-asynchronous conformance checklist. The applicable companion
requirements are all of R01–R16, NDP01–NDP17, V21S01–V21S17, and
ISP01–ISP07 from `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, with ADR-002 in
`docs/ASYNC_DECOUPLED_DILOCO_V2.md` and the native specialization in
`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`.

The JSON certificate maps every individual requirement ID to exact entries in
its digest-bearing `artifact_catalog`. It distinguishes:

- immutable synthetic G2 behavior that genuinely passed;
- startup facts actually observed in model job `5111908`;
- gates blocked before model semantics could be exercised; and
- the downstream fault, restart, and scale phases that were intentionally not
  submitted after the terminal failure.

No new live bookkeeping or coordination subsystem was added. The report only
indexes immutable artifacts already produced by the native gates, model
allocation, and scheduler-owned collector.

## Validation

The recovery used the canonical Frontier environment before Python/native or
scheduler preflight. Validation performed:

```text
sha256sum -c .../5111221.9d3stfes.owned/SHA256SUMS
sha256sum -c .../5111243.f3ot8ubb.owned/SHA256SUMS
sacct -j 5111908,5111909 -X -n -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,Elapsed,AllocNodes,Partition,QOS,NodeList,Submit,Start,End
jq empty docs/validation/complete-v21-current-rc-2n-20260729.json
```

The acceptance facts are unambiguous: zero completed K40 windows, zero atomic
commits, no fault campaign, no fresh-allocation recovery, and literal
`full_pass=false`. Consequently this task must not be called done and cannot
authorize its 8-node consumer.
