# Implement async decoupled DiLoCo v2 validation

Date: 2026-07-24

Task: `implement-async-decoupled-diloco-v2`

Policy: `async-decoupled-v2.0-exp`

Scope: local implementation, deterministic/reference validation, native build
and rendered non-submitting preflight. No Slurm job was submitted. No 4+ node
run is authorized or claimed.

## Outcome

The rendered E97 native trainer now has one continuous model-owning lane and a
separate result/publication control lane.
`PersistentRealWorkerSession` builds the model, ScheduleFree optimizer, data
iterator, and hidden state once and keeps them resident across successive exact
K40 windows. After the native service accepts a contribution and returns local
`OWNED`, `PersistentAsyncTrainingLane` immediately starts the next K window on
that resident session. The caller concurrently waits for native
transport/reduction/result completion, releases the bounded native result
view, writes the authoritative global checkpoint, verifies the immutable
handoff and fenced `latest` CAS, and only then requests a stop at the next
actual K boundary.

At that boundary the real lane translates the resident model `x`, the
ScheduleFree `z` parameter point, and the mutable coalescing interval start by
the same audited correction. Scalar state and optimizer moments are retained.
Adjacent speculative windows are represented as one cumulative
`[local_window_start, local_window_end)` interval, not a dense per-window FIFO.
The lane pauses at `sigma_hard=8`; it does not wait for fabric send, reduction,
commit, or checkpoint while inside the configured bound.

The obsolete serial `NativeGenerationPipeline` scheduler is no longer imported
or instantiated by `scripts/frontier/resilient_e97_role.py`. It remains a
separately tested generic library, but it is not a second rendered production
authority. The rendered production marker identifies
`ndm.async_diloco_real.PersistentAsyncTrainingLane`.

## Reviewed-policy constants and identities

The production preflight and runtime fail closed unless all reviewed constants
match ADR-002:

| Field | Implemented value |
|---|---:|
| policy | `async-decoupled-v2.0-exp` |
| local window | `K=40` |
| global lag hard / target | `tau=6 / 2` |
| speculative local hard / target | `sigma=8 / 2` |
| global quorum | `Q_min=2` |
| global token floor | `T_min=3,934,080` |
| group deadline | 420 seconds |
| generation retries | 0 |
| outer update | stateless delta SGD, `eta=0.5` |
| sealed descriptor capacity | 1 |
| mutable coalescing interval capacity | 1 |
| result visible / staging capacity | 1 / 1 |
| E97 conservative resident admission | 64,001,671,648 bytes |

`ContributionIdentity`, the native generation-v2 schema, and the native
submission-v2 schema bind run, allocation fence, stable worker and
incarnation, contribution sequence, exact adjacent local-window range and
count, base version and digest, base lag at seal, policy/layout/code digests,
exact tokens, endpoint digest, and payload digest. Exact tokens remain
separate from the deterministic integer aggregation weight
`exact_tokens * (7 - commit_lag)`.

The result lane separately records base-at-seal, commit version/lag, known
global version, applied-anchor version, result version at apply, and
speculative local-window lag. It never relabels an unavailable base as fresh.

## Conformance checklist

The implementation was checked against the authoritative
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
[`ASYNC_DECOUPLED_DILOCO_V2.md`](../ASYNC_DECOUPLED_DILOCO_V2.md), and the
single-row traceability in
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md).

### Compute-pool v1 requirements

| IDs | Implementation conformance |
|---|---|
| R01–R04 | Current allocation fence remains authoritative; stable identity plus new incarnation is retained; membership is a frozen READY snapshot; contribution replay is idempotent and conflicting identity rejects. |
| R05–R08 | Exact tokens and deterministic binary64 aggregation remain separate from lag weight; quorum/token/deadline bounds are explicit; publication is fenced and atomic; deterministic sharded ownership stays bounded. |
| R09–R12 | Trainers alone own models; services/managers are model-free; no Python/Lustre dense hot path is introduced; catch-up/rejoin rules are bounded; restart restores authoritative global model and outer state while local inner work is disposable. |
| R13–R16 | The policy remains scheduler-neutral; all waits and queue/memory highs are named; deterministic numerical/failure/restart tests are present; qualification remains exactly two-node and does not authorize 4+. |

### Native data-plane requirements

| IDs | Implementation conformance |
|---|---|
| NDP01–NDP04 | Python remains metadata/control; the persistent compiled C++ service owns dense memfd/CXI handoff; production is point-to-point and collective-free; local `OWNED` transfers descriptor responsibility without a Python dense copy. |
| NDP05–NDP08 | Fixed layout and deterministic binary64 arithmetic remain; v2 uses distinct versioned metadata; fenced opaque routes remain; exactly one immutable plus one mutable cohort and one visible plus one staging result fit the 64,001,671,648-byte admission formula. |
| NDP09–NDP12 | Credits remain distinct from send/receipt completion; CRC/digest/finite/identity checks precede apply; replay and owner reassignment are bounded; owner-direct redistribution produces one service-owned node result rather than a full-model broker. |
| NDP13–NDP17 | Every stage has a finite deadline; the stable native ABI carries metadata while dense bytes stay native; Python retains fenced checkpoint policy; provider/identity/lag/byte/high-water telemetry is emitted; the next live gate remains exact two-node CXI only. |

### Async v2 requirements

| IDs | Implementation and local evidence |
|---|---|
| V2A01 | Runtime, launcher, renderer, and semantic validator pin one explicit v2 policy and reject false `tau=0` labeling. |
| V2A02 | The actual rendered trainer starts `PersistentAsyncTrainingLane` after local native `OWNED` and before `result_shards`; model/optimizer/iterator/hidden state remain resident and exact K40 windows continue while completion is delayed. |
| V2A03–V2A04 | Versioned contribution/result identities preserve base/digest/range and all independent lag clocks through native submission, commit, publication, and apply. |
| V2A05–V2A06 | One sealed descriptor, one cumulative mutable interval, finite group bounds, exact tokens, and deterministic lag weights are enforced. Reference tests cover unequal tokens, changing membership, and lag 0/6 rejection at 7. |
| V2A07–V2A08 | The exact half-step outer equation and atomic outer state are implemented. Safe-boundary correction translates model `x`, ScheduleFree `z`, and interval start while retaining moments/scalars; fresh restart discards local inner state. |
| V2A09–V2A10 | Latest-only verified mailbox semantics, one replacement staging view, monotonic/idempotent replacement, bounded view lifetime, and the conservative resident byte formula are tested and validated. |
| V2A11–V2A12 | Stale unsealed work drops after verified catch-up, the lane pauses at sigma, late/rejoin work uses bounded incarnation/fence rules, and the two-node quorum floor cannot degrade to one-node authority. |
| V2A13–V2A14 | Duplicate replay, wrong fence/base/layout/code/policy, corrupt/nonfinite data, owner retry, publication failure, immutable reload verification, fenced latest CAS, and fresh-allocation checkpoint restore are covered. |
| V2A15 | Production source has no dense Python TCP/object serialization, no Lustre dense hot path, no MPI/all-rank collective or barrier, and no central full-model broker. Dense payloads stay in the persistent native memfd/CXI service. |
| V2A16–V2A17 | The semantic validator separately checks continuous K cadence/idle/OWNED/lag bounds and correctness publication latency. It accepts latest-only application rather than recreating a per-K barrier, and rejects stalls, unbounded state, rebootstrap, unverifiable apply, and missing versioned overlap. |
| V2A18 | Local numerical, failure, restart, source, launcher, and preflight tests pass. The required live two-node Frontier performance/failure/convergence evidence remains pending by design; no Slurm or 4+ node claim is made here. |

## Deterministic and integration coverage

`tests/test_async_diloco_v2.py` supplies the single-process high-precision
reference model. It covers:

- fresh and lagged contributions, unequal exact tokens, changing membership,
  deterministic ordering, `tau=6`, and rejection beyond the boundary;
- the exact `S_(g+1) = S_g + 0.5 * weighted_mean(delta)` equation and
  `{mode, eta, step, accepted_tokens}` outer state;
- real q+1 progress while q service/commit is deliberately blocked;
- one sealed plus one mutable cumulative interval, sigma pause, stale drop, and
  verified catch-up;
- latest-only visible/staging mailbox behavior with a held view, replacement,
  idempotence, older/conflicting/corrupt rejection;
- accepted-ledger and nonaccepted safe-boundary correction;
- duplicate replay, wrong fence/base/layout/code/policy, corrupt and nonfinite
  payloads, owner retry, and failed publication without authority mutation;
- fresh-allocation restore of global model/outer authority only; and
- ScheduleFree `x`/`z` translation with moment/scalar preservation.

`tests/test_async_diloco_real_trainer.py` proves that multiple exact local
windows perform exactly one model build, one optimizer build, and one data
iterator build; the model/optimizer object identities and hidden-state
sequence persist across windows. Its delayed-result lane test observes three
real local windows before result release, then verifies a single boundary
translation of model `x`, optimizer `z`, and interval start.

`tests/test_resilient_e97_runtime.py` contains a production-entrypoint source
gate that requires this order in the actual rendered trainer:

```text
native local OWNED
  -> PersistentAsyncTrainingLane.start
  -> native result wait/materialization
  -> checkpoint and fenced latest verification
  -> PersistentAsyncTrainingLane.finish_at_boundary
  -> verified apply receipt
```

The same source gate proves `run_window` and `translate` are not executed by
the result/checkpoint caller path. Multi-process control tests additionally
exercise exact-token generations, two model-free managers without a
collective, corruption rejection, fenced atomic commit, and fresh-process
restart matching uninterrupted continuation.

## Production source audit

The rendered production role contains none of:

```text
pickle
sendall
recv(
mpi4py
MPI_
all_reduce
all_gather
barrier
TCPStore
torch.distributed
NativeGenerationPipeline
LiveNativeGenerationScheduler
```

The explicit `python-tcp-debug` fixture and older MPI performance/reference
implementations remain elsewhere in the repository, but the launcher rejects
them for full-layout E97. Production uses `NativeTrainerDataPlane`,
`NativeManagerSession`, direct producer memfd lanes, compiled point-to-point
owner transport, and one shared read-only native result. Telemetry asserts
`python_dense_socket_bytes=0`, `lustre_dense_hot_path_bytes=0`, bounded native
views/queues, and no central full-model broker.

## Rendered two-node and seed preflight

The non-submitting renderer produced
`/tmp/emender-agent-1489-async-v2-plan-final.json` with:

- schema `emender-real-e97-exact-2n-acceptance-v2`;
- exactly 2 nodes and explicit rejection of 4, 8, 32, 64, and 256 nodes;
- `Partition=batch` and `QOS=debug` for every phase;
- K40 and policy `async-decoupled-v2.0-exp`, tau 6/2, sigma 8/2,
  `eta=0.5`, and every queue capacity equal to one;
- conformance lists R01–R16, NDP01–NDP17, and V2A01–V2A18;
- a 12-generation clean overlap phase followed by bounded fault, invalid
  result, failed publication, and fresh-restart phases; and
- the final E97 checkpoint at step 2,300,930, 150,793,748,480 tokens,
  7,719,680,116 bytes, SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.

The batch launcher retains submit-side content-addressed seed caching,
compiled `sbcast`, node-local `--verify-local`, attestation binding, and
offline tokenizer staging. The verified job-local seed remains available for
the whole supervisor run and is removed on script exit so retries cannot
observe stale `/tmp` state.

## Exact validation commands and results

The canonical Frontier environment was sourced before every Python, pytest,
native build, and rendered preflight command.

### Focused async/reference/production suite

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent-1489-focused-final2 \
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

Result: `177 passed in 206.71s`.

The short `--basetemp` is intentional: AF_UNIX fixture paths have a kernel
length limit and the WG Lustre worktree path exceeds it. This changes only
ephemeral test paths, not production behavior.

### Native Python/ABI/runtime suite

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  --basetemp=/tmp/emender-agent-1489-native-final \
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

Result: `90 passed in 109.21s`.

### Canonical native build and compiled tests

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: configuration and build succeeded; all `10/10` compiled CTests passed;
the installed native libraries/service/gate were up to date; and
`build/native-resilient-dataplane/native-artifacts.json` was recorded.

### Rendered non-submitting preflight

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo . \
  --native-build-manifest \
    build/native-resilient-dataplane/native-artifacts.json \
  --full-layout-gate \
    reports/frontier/native-dataplane/5034807/full-layout-gate.json \
  --run-root /tmp/emender-agent-1489-rendered-runs-final \
  --output /tmp/emender-agent-1489-async-v2-plan-final.json \
  --allow-non-authoritative-dry-run
```

Result: plan rendered successfully with the bindings recorded above. The
non-authoritative flag is required because validation ran before the task
commit; it does not submit, schedule, or promote the plan.

### Repository integrity and source audit

```bash
git diff --check
cmp -s AGENTS.md CLAUDE.md
rg -n \
  'pickle|sendall|recv\(|mpi4py|MPI_|all_reduce|all_gather|barrier|TCPStore|torch\.distributed|NativeGenerationPipeline|LiveNativeGenerationScheduler' \
  scripts/frontier/resilient_e97_role.py
```

Result: the diff and project-guide identity checks exit zero. The production
forbidden-symbol scan returns no matches.

The checksum-linked native reference remains internally consistent after the
gap-matrix update:

| Artifact | SHA-256 |
|---|---|
| `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` | `0420d94862f338636a09e0bc7cef16b4fc2e5fe1985c1f3a096d91f2d4aedc08` |
| `docs/RESILIENT_DILOCO_GAP_MATRIX.md` | `89c8739d881f5943c08c9c4aaa1f2b2567a4c40e8e8be969b4846f15eadd98c5` |
| `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md` | `c9aa27e8da1ad87d435695a6288469a78304caee410cc92a6a3bdfb2961b6f0f` |
| `reports/frontier/native-dataplane-reference-v1.json` | `70283d6756c97ac80dbe4cb6a1996e070768e3abe20de14716febfbefdcb0f4b` |

## Qualification boundary

This task implements and locally validates the reviewed production path. It
does not claim the V2A18 live two-node performance/failure/convergence gate.
The next authorized operational step is exactly two nodes on
`Partition=batch`, `QOS=debug`, with scheduler evidence retaining both fields.
Four or more nodes remain prohibited until the full two-node numerical,
failure/restart, decoupling/correctness, deterministic-replay, and three-seed
convergence gates are accepted.

The implementation commit SHA is recorded in the WG task log after the
surgical commit; a commit cannot contain its own resulting SHA.
