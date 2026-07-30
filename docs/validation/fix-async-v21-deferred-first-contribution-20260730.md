# Async v2.1 deferred first-contribution closure fix

**Task:** `fix-async-v2-1-2`  
**Date:** 2026-07-30  
**Scope:** local implementation and regression evidence only; **no Frontier pass or scale authorization is claimed**.

## Immutable physical failure input

The diagnosis is the immutable failed clean-phase record
`docs/validation/requalify-v21-fixed-rc-2n-20260730.json` from commit
`4bcd324d`, file SHA-256
`a16f70922b38ccac540f89a206d9f6692dad8fffb666d291f9bea12c5ca31b92`.
It binds model job `5120935`, payload
`cf065a49813efd552a935221c2837cbeefc060c148787aa34ff0eefd8003c769`,
trace SHA-256
`9e5bcd4fa8844e36a467daf7af4bbeba479a0c4c2541678a99cf3fbee04ea42b`,
and primary stderr SHA-256
`c73485ea319a9a561904349c926541e03b7282cfe8aca10160b5b1998fb6e756`.
That record was not edited, the failed payload was not submitted again, and
this task performed no Slurm operation.

Trace replay established the first divergence:

1. node A became READY and `OpenGeneration(0,1)` froze a one-node cohort;
2. node B became leased READY immediately afterward;
3. node B's exact-token contribution (`2,622,720`) returned non-mutating
   `deferred` because it was outside the already frozen cohort;
4. node A's contribution was accepted, while finite close remained
   non-mutating `insufficient-cohort`; and
5. the node-B manager promoted its nonterminal disposition to `TimeoutError`,
   after which peer expiry prevented generation-zero commit.

The failure remained fail closed: zero committed generations, zero result
receipts, zero accepted-token-clock delta, and no partial publication.

## Fix and invariants

`NativePoolControlServer` now executes elapsed lease events first and delays the
*external mutating* `OpenGeneration` event until the configured current-generation
leased-READY stable-worker floor exists. The compiled kernel remains sole
membership/cohort authority; the Python effect cache cannot add a member.
For the fixed two-node policy this prevents an immutable one-node cohort from
being created while the second READY handshake is in flight.

`PoolControlClient.contribute_and_freeze` retains one exact fenced request and:

- replays the identical worker/incarnation/sequence/token/payload identity only
  while contribution admission is non-mutating `deferred` or
  `insufficient-cohort`;
- stops contribution RPCs permanently after accepted/identical receipt and
  polls close only;
- uses the caller's original absolute deadline for every replay, close poll,
  and sleep, with a finite 10 ms maximum poll interval;
- returns terminal stale fence/incarnation/generation, conflict, corrupt,
  nonfinite, closed, abort, and catch-up dispositions without retry;
- exits promptly on manager shutdown; and
- never changes generation or attempt (`attempt=1` throughout; zero attempt
  retries).

The rendered native manager passes its original `native_deadline` and shutdown
predicate. A shutdown before closure releases the unpublished native result,
aborts/drains boundedly, and publishes nothing. Normal catch-up/rejoin and
exact-once commit handling are unchanged. There remains one authoritative
result or none.

## Regression evidence

`test_deferred_first_contribution_closes_when_second_ready_peer_arrives` drives
the production compiled coordination service through the real RPC adapter. It
starts OPEN with only node A READY, proves no `OpenGeneration` kernel mutation,
admits node B later, and then has node A contribute first. The first finite close
is `insufficient-cohort`; node B contributes later under the same absolute
close deadline; both callers receive `commit_ready`. Exact replay returns
`identical-duplicate`, contribution count remains two, both peers remain live
and READY, and the trace contains neither `ExpirePeer` nor
`RetryNextGeneration`.

Additional focused cases cover:

- transient contribution replay with byte-identical arguments;
- original-deadline expiry when the second contribution is missing, with a
  bounded number of close polls;
- shutdown during transient admission;
- stale fence, stale incarnation, stale generation, conflicting duplicate,
  corrupt, nonfinite, and closed-generation terminal dispositions; and
- rendered manager deadline/shutdown wiring.

No test permits partial mutation/publication, double contribution, unbounded
busy wait, or attempt-number retry.

## Authority and requirement map

This implementation was checked against the **Resilient DiLoCo Compute Pool,
Version 1 conformance checklist**
(`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`), the Native data plane Version 1,
ADR-002 `async-decoupled-v2.1-simple`, and the companion gap matrix.
All applicable namespaces remain independently binding:

- **R01–R16:** `R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R11,
  R12, R13, R14, R15, R16`;
- **NDP01–NDP17:** `NDP01, NDP02, NDP03, NDP04, NDP05, NDP06, NDP07,
  NDP08, NDP09, NDP10, NDP11, NDP12, NDP13, NDP14, NDP15, NDP16, NDP17`;
- **V21S01–V21S17:** `V21S01, V21S02, V21S03, V21S04, V21S05, V21S06,
  V21S07, V21S08, V21S09, V21S10, V21S11, V21S12, V21S13, V21S14,
  V21S15, V21S16, V21S17`; and
- **ISP01–ISP07:** `ISP01, ISP02, ISP03, ISP04, ISP05, ISP06, ISP07`.

| Direct obligation | Local conformance evidence |
|---|---|
| R02/R03, NDP01, V21S10 | OPEN uses current-generation leased READY effect records, never launched ranks; late READY is admitted before the immutable snapshot rather than expiring a peer. |
| **R06**, **NDP13** | READY and close waits are finite, use one original absolute deadline, sleep between polls, and fail closed on missing peer/shutdown. |
| R04/R07, NDP06/NDP10/NDP14 | Fenced identity is replayed exactly; identical replay acknowledges without a second add; stale/conflict/corrupt inputs return without mutation; commit authority is unchanged. |
| **V21S05** | Fixed Q/T policy is consumed from `PoolControlConfig`; the regression uses stable-worker diversity, positive exact tokens, one contribution per worker, and attempt 1 only. Production remains `Q_min=2`, `T_min=3,934,080`, active fraction disabled, and zero attempt retries. |
| V21S02/V21S09, ISP04 | The poll loop and effect caches are capacity/time bounded; no dense bytes, spill, new cohort, queue, or allocation are introduced. |
| ISP02, **ISP04** | Closure remains model-free background control; shutdown cannot turn it into an unbounded manager/trainer wait. Existing snapshot ownership and immediate-resume gates pass unchanged. |
| ISP05, NDP15 | No result/apply/publication begins before `commit_ready`; deadline/shutdown produces none, so partial trainer application cannot occur. |
| R14/NDP16/V21S13/ISP06/ISP07 | Existing causal/tail validators pass unchanged. This local result does not replace the required physical max/p99, zero-foreground-result-wait, and anti-200-second-stall evidence. |
| **V21S15**, R16/NDP17/V21S16/V21S17 | Job 5120935 remains a failed batch/debug record and authorizes nothing. A new exact-source 2-node clean/fault/fresh-recovery sequence is required before any direct scale rung. |

The rest of R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 is preserved
by the unchanged scheduler fence, exact-token binary64 path, point-to-point
native transport, bounded credits/replay, immutable checkpoint/receipt chain,
atomic apply/recovery, causal telemetry, and fail-closed promotion controllers,
all exercised by the suites below.

## Validation commands and results

Every Python/native command was run after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

and used `"$EMENDER_PYTHON"`; the build wrapper received
`PYTHON_BIN="$EMENDER_PYTHON"`.

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
# native build succeeded; CTest 11/11 passed

TMPDIR=/tmp/emender-agent-1712 "$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_pool_integration.py \
  tests/test_resilient_e97_true_2n_launcher.py::test_native_manager_freeze_wait_spans_the_open_generation_deadline
# 28 passed

TMPDIR=/tmp/emender-agent-1712 "$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_pool_runtime.py \
  tests/test_native_pool_production_policy.py \
  tests/test_native_pool_integration.py
# 53 passed

TMPDIR=/tmp/emender-agent-1712 "$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_async_diloco_v21.py
# project-equivalent async-v2.1 controller/runtime/launcher suite: 200 passed

TMPDIR=/tmp/emender-agent-1712 "$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_snapshot_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_validate_async_v21_fault_phase.py
# 44 passed

TMPDIR=/tmp/emender-agent-1712 "$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_pool_integration.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_native_pool_production_policy.py \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_async_diloco_v21.py \
  tests/test_async_snapshot_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_validate_async_v21_fault_phase.py
# final combined validation on the completed source: 297 passed
```

The default long worktree path exceeds Linux `AF_UNIX` path length for three
pre-existing handoff tests; `TMPDIR=/tmp/emender-agent-1712` is the documented
short node-local test root and all such tests pass there.

## New execution-source boundary

The reviewed execution-source algorithm
`emender-async-v21-execution-source-v1` excludes this evidence report but
includes every changed runtime and test byte. Before integration it computes:

```text
old failed execution-source digest (job 5120935):
  a97ac00bbcf21b08a858e6be67098ed2936aef30dc0a751929c2c90b81bb34bc
new fixed execution-source digest:
  070d0638a96dda8d5e96d391d5bc2c60705543f97a288679d78a8f6af5cd73cd
```

This is a different immutable execution identity. Any Frontier requalification
must first integrate the fix to clean authoritative `main = origin/main`,
recompute and bind that exact digest plus a newly matching source commit,
native bundle/G2 artifact, launcher, policy/schema/wire, tokenizer, data, and
seed identities, and create a **new changed payload** through the durable held
model/afterany-collector transaction. Payload
`cf065a49813efd552a935221c2837cbeefc060c148787aa34ff0eefd8003c769`
and job `5120935` must never be resubmitted.
