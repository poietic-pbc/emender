# Wire async v2 into the real E97 trainer loop

Date: 2026-07-24

Task: `wire-async-v2-real-trainer-loop`

Status: locally implemented and validated in the canonical Frontier
environment. No Slurm command was run and no allocation was submitted. This
artifact does not claim the still-pending live two-node performance,
failure-injection, or convergence qualification.

## Authority and scope

The implementation conforms to the
[Resilient DiLoCo Compute Pool](../RESILIENT_DILOCO_COMPUTE_POOL.md), including
requirements R01–R16 in the
[gap matrix](../RESILIENT_DILOCO_GAP_MATRIX.md); the
[Native resilient DiLoCo data plane v1](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
including NDP01–NDP17; and
[ADR-002: bounded-lag asynchronous DiLoCo v2](../ASYNC_DECOUPLED_DILOCO_V2.md),
including V2A01–V2A18.

The changed production path remains restricted to exactly two nodes with
`Partition=batch`, `QOS=debug`. It retains the final immutable step-2300930 S3
seed, job-local `sbcast` materialization, offline bootstrap, K40,
`async-decoupled-v2.0-exp`, the compiled memfd/CXI point-to-point dense path,
and the existing 64-GiB native resident ledger. No 4+ node authorization is
introduced.

## Production wiring

The real model-owning `trainer()` entrypoint in
`scripts/frontier/resilient_e97_role.py` constructs one
`PersistentRealWorkerSession` for the trainer incarnation. That object retains
the model, ScheduleFree optimizer and `z`, data iterator, and hidden state
across adjacent local K40 windows. The native branch does not call the legacy
per-generation `_run_real_worker`; that call remains only in the explicit
Python TCP compatibility branch.

After one exact interval is sealed into service-owned memfd storage and the
native RPC returns local `OWNED`, the entrypoint immediately starts
`PersistentAsyncTrainingLane`. That lane invokes the same resident worker
session for the next exact K40 window(s), coalescing adjacent windows into one
mutable interval with the ADR-002 sigma bound. The caller may block in native
result/reduction, outer commit, checkpoint publication, and integrity
verification, but none of those operations is present in the lane's
K-to-next-K loop. One descriptor and one mutable interval remain the only
dense local cohorts.

The manager now returns an explicit accepted-local descriptor ledger containing
the rank, incarnation, contribution sequence, local window range, original
base version, payload digest, and canonical descriptor digest. The trainer
matches this identity against its immutable `OWNED` descriptor before
subtracting any own work. A missing rank identity means `C_i=0`; a conflicting
identity fails closed.

`apply_delta_with_correction_ledger` constructs the exact boundary correction
from the same bounded native result shards used for the global-anchor update:

```text
correction_i = (S_h - S_a) - C_i(a,h)
```

For an accepted interval, `C_i` is exactly
`interval_endpoint - interval_start`; for an unaccepted interval it is zero.
The persistent boundary then translates model `x`, audited ScheduleFree `z`,
and the mutable interval start by this same correction. Moment/scalar state is
retained, accepted own work is subtracted once, unaccepted/post-snapshot work
survives, and unknown parameter-valued optimizer buffers continue to fail
closed.

A native result remains only a candidate until
`_reload_verified_async_v2_latest` reloads the immutable handoff, validates its
run/generation/fence/digest, compares it to the durable
`latest/authoritative` CAS record under the current lease, and reasserts the
lease. Only then does the trainer request a stop at the next exact K boundary
and apply the correction. The native result view is released before durable
publication, while correction and checkpoint timing remain separate bounded
stages.

## Deterministic production-entrypoint trace

`test_production_trainer_entrypoint_overlaps_blocked_native_result_and_applies_at_boundary`
calls the actual production `trainer()` orchestration with a controlled
one-parameter model, ScheduleFree optimizer, persistent iterator, fenced
control store, and native descriptor/result endpoint. It does not call the
probe or a validator.

The test deliberately holds generation-0 result/reduce/outer-commit/checkpoint
completion after local `OWNED`. While that chain is blocked, the production
entrypoint:

- starts and completes local window 1 from the honest worker-local S0 basis;
- starts local window 2 before the released result can be verified;
- uses exactly one model build, one optimizer build, and one iterator build;
- retains the same model, optimizer, `z`, iterator, and hidden state;
- emits no apply before the fenced latest result is reload verified; and
- coalesces the speculative interval as `[1,3)`, rather than queueing two dense
  per-K objects.

After release, the fake publisher exposes a digest-valid immutable generation-1
handoff and a matching fenced authoritative CAS record. Application occurs only
after window 2 finishes, at local boundary 3. The trace proves:

| Field | Observed |
|---|---:|
| First sealed interval | `[0,1)` |
| Coalesced mutable interval | `[1,3)` |
| First/second original base version | `0 / 0` |
| Generation-0 commit version | `0` |
| First result/applied global version | `1` |
| First anchor lag before apply | `1` |
| First speculative-window lag | `2` |
| Model/optimizer/iterator builds | `1 / 1 / 1` |
| Resident windows completed | `3` before first apply |
| `x,z` before first correction | approximately `3,3` |
| Accepted correction | approximately `-0.5` |
| `x,z` after first correction | approximately `2.5,2.5` |
| Second accepted correction | approximately `-1.0` |
| Final `x,z` | approximately `1.5,1.5` |

The test also parses the production JSONL trace and checks the K start,
worker-local basis, applied-anchor version, result version, anchor lag,
speculative lag, accepted descriptor digest, and fenced-CAS evidence. Source
assertions independently verify that `result_shards`, `apply_delta`,
`torch.save`, and `wait_metadata` do not occur in
`PersistentAsyncTrainingLane` between one K completion and its next K start.

## Requirement checks

- R01–R07: the existing lease, READY membership, exact floors, deterministic
  reduction, and atomic publication remain unchanged; normal boundary apply
  now additionally verifies exact durable CAS agreement.
- R08–R12: compiled bounded ownership/redistribution and model-free managers
  remain intact; trainer-local ScheduleFree state persists only within an
  incarnation, while restart authority remains global S/O.
- R13–R16: the backend-neutral control contract, stage deadlines, reference
  arithmetic, and strict two-node ladder remain rendered.
- NDP01–NDP04: Python coordinates only metadata/handles; service-owned memfd
  remains the dense local handoff and the manager remains model-free.
- NDP05–NDP12: deterministic native weighted reduction, fixed identities,
  credits, checksums, replay, and direct redistribution remain covered by the
  unchanged native suites. Accepted correction identity is cross-linked to the
  immutable native payload digest.
- NDP13–NDP17: independent stage deadlines, stable ABI prefix, fenced
  checkpoint publication, telemetry, and the retained synthetic G2 gate still
  pass.
- V2A01–V2A07: the explicit policy, exact K40 identity, separate local/global
  clocks, one sealed plus one mutable interval, token/lag weights, and
  half-step outer equation remain pinned.
- V2A08–V2A09: the production entrypoint now uses the accepted-delta ledger and
  verifies durable fenced latest before applying to resident `x/z` at a safe
  boundary.
- V2A10–V2A15: finite resident/queue bounds, sigma/tau behavior,
  failure/idempotence rules, restart authority, and compiled P2P transport
  remain covered by the focused and native suites.
- V2A16–V2A17: real K progression overlaps a deliberately blocked prior
  background chain in the deterministic entrypoint test; training-lane and
  checkpoint/correctness timing remain separate.
- V2A18: the non-submitting renderer remains exactly two-node batch/debug and
  names every R, NDP, and V2A requirement. Live qualification is still pending.

## Validation commands and results

Every Python, pytest, native build, and renderer command below was run after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Canonical focused Python/reference/production suite:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent-1492-focused \
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

Final result: `181 passed in 192.72s`.

Native Python/ABI/runtime suite:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent-1492-native \
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

Result: `90 passed in 117.64s`.

Canonical native build and compiled tests:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: build and install succeeded; `10/10` CTests passed in `2.68s`;
`build/native-resilient-dataplane/native-artifacts.json` was recorded.

Rendered non-submitting exact-two-node preflight:

```bash
"$EMENDER_PYTHON" \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo . \
  --native-build-manifest \
    build/native-resilient-dataplane/native-artifacts.json \
  --full-layout-gate \
    reports/frontier/native-dataplane/5034807/full-layout-gate.json \
  --run-root /tmp/emender-agent-1492-rendered-runs \
  --output /tmp/emender-agent-1492-async-v2-plan.json \
  --allow-non-authoritative-dry-run
```

Result: rendered without submission. The plan has schema
`emender-real-e97-exact-2n-acceptance-v2`, exactly 2 nodes,
`partition=batch`, `qos=debug`, K40, the reviewed v2 policy and bounds, all
R01–R16/NDP01–NDP17/V2A01–V2A18 IDs, the verified immutable step-2300930 S3
seed, and explicit 4/8/32/64/256-node prohibitions.

Repository integrity/source audit:

```bash
git diff --check
cmp -s AGENTS.md CLAUDE.md
! rg -n \
  'pickle|sendall|recv\(|mpi4py|MPI_|all_reduce|all_gather|barrier|TCPStore|torch\.distributed|NativeGenerationPipeline|LiveNativeGenerationScheduler' \
  scripts/frontier/resilient_e97_role.py
jq -e \
  '.node_count == 2 and .queue.partition == "batch" and
   .queue.qos == "debug" and .k_local_steps == 40 and
   .seed.step == 2300930 and
   (.conformance.requirements|length)==16 and
   (.conformance.native_requirements|length)==17 and
   (.conformance.async_v2_requirements|length)==18' \
  /tmp/emender-agent-1492-async-v2-plan.json
```

Result: every command exited zero; the forbidden-symbol scan produced no
matches. The implementation commit SHA is recorded in the WG task log and in
this artifact's final commit note.
