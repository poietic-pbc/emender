# Native service / Lean 4 coordination conformance

**Status:** executable local differential gate, schema/view version 1
**Authorities:** Resilient DiLoCo Compute Pool v1, its gap matrix, the
authoritative Lean transition, and the compiled native service transition
**Scope:** pure peer coordination only; no Slurm, GPU, model tensor,
checkpoint, remote libfabric peer, or external service

## Outcome and authority split

The same deterministic document in
`formal/resilient/trace-schema-v1.json` executes through two authorities:

1. `ResilientProtocol.transition` is the protocol/proof oracle.
2. `coordination::step` in the persistent compiled service is the production
   mutation authority used by the networking/process runtime.

`emender-native-lean-authority-view-v1` is their versioned intersection. The
adapter translates nominal strings to fixed ABI keys, the Lean source
generation/attempt/owner numbering to native wire numbering, and facts that
the fixed ABI intentionally carries only as opaque digests. It never
calculates the expected Lean disposition, implements admission/closure
policy, substitutes a coordinator object, or mutates a test-only state
machine.

Every canonical event compares:

- typed disposition;
- run, allocation, fence, policy, schema, toolchain, source, layout, and code
  identity;
- generation, attempt, owner epoch/reassignment, accepted exact-token clock,
  accepted contribution count/tokens, and generation phase;
- every current worker/node/incarnation/lifecycle/synchronized-generation
  identity;
- the frozen cohort and every contribution's worker/node/incarnation,
  sequence, exact tokens, payload digest, and service-derived receipt digest;
- frozen cohort digest, commit receipt ID/digest, result ID/digest, every
  result-receipt worker, and every atomic node-apply identity/digest; and
- the deterministic normalized post-state digest.

The report also retains the native kernel's independent pre/post state
digests for every underlying call. A first mismatch stops execution and emits
the valid canonical prefix through the divergent event plus a replay command.

## Production call path

The source audit in `audit_production_call_path` and the runtime identity
manifest bind this exact chain:

```text
NativeTraceAdapter._step
  -> NativeCoordinationAuthority.step
  -> Client.coordination_step
  -> ndp_coord_step_v1
  -> AF_UNIX/SOCK_SEQPACKET Opcode::CoordinationStep
  -> LocalServiceCore::coordination_step
  -> Service::coordination_step
  -> coordination::step
```

The implementation sites are:

- `ndm/native_lean_conformance.py`;
- `ndm/native_coordination.py`;
- `ndm/native_dataplane.py`;
- `src/native_resilient_dataplane/src/client.cpp`;
- `src/native_resilient_dataplane/src/rpc_server.cpp`;
- `src/native_resilient_dataplane/src/service_core.hpp`;
- `src/native_resilient_dataplane/src/ndp.cpp`; and
- `src/native_resilient_dataplane/src/coordination_kernel.cpp`.

The audit requires the named call sites to exist. Each run hashes all of those
sources, the Lean `Types`, `Kernel`, `Trace`, and `Conformance` modules, the
schema, the Lean executable, the installed native library/service/gate
binaries, and the complete native bundle. It records the source commit,
dirty-state bit, `NDP_COORD_ABI_V1`, and exact event/member/result structure
sizes. A mismatched or missing artifact fails before service launch.

The service runs persistently on the local `tcp;ofi_rxm` test provider. This
exercises the compiled service and real metadata RPC without claiming a
Frontier/CXI peer qualification. `NativeCoordinationAuthority` supplies the
same fixed C ABI events used by the production live role. Bootstrap recovery
loads the trace's durable base authority and node-apply identities before the
first canonical event.

Some canonical proof events have a deliberately thinner production ABI
projection:

| Canonical operation | Actual production mutation calls |
|---|---|
| admission/restart/READY/expiry | `RECOVER_PEER`, `READY`, `EXPIRE_PEER` |
| open/contribution/close | `OPEN_GENERATION`, `CONTRIBUTION`, `CLOSE_GENERATION` |
| commit | one `RESULT_RECEIPT` for each actual frozen contributor, then `COMMIT` |
| eight-to-one apply | `NODE_APPLY` with the current commit receipt |
| owner replay/abort | `OWNER_LOST` |
| fresh fence | `RECOVER_AUTHORITY` |
| proof-only trainer/publication identity | current-authority `QUERY_COMMIT`; trainer loss additionally expires and recovers the real node authority |

The auxiliary calls are retained in order beneath their one canonical event.
The final native disposition is normalized only for vocabulary-equivalent
outcomes (`corrupt`/`corrupt_nonfinite`, committed old-generation
`generation_closed`/`catch_up`, noncohort `deferred`/
`retry_next_generation`, and owner exhaustion
`retry_next_generation`/`aborted`).

## Deterministic fail-closed serialization

The single trace encoding is compact UTF-8 JSON with recursively sorted keys
and no alternate whitespace. The native loader rejects:

- duplicate object keys;
- pretty-printed or differently ordered aliases;
- missing or unknown top-level identity fields;
- wrong schema/toolchain identities; and
- malformed bounded nominal identities.

The Lean runner independently reconstructs every derived record and rejects
missing/unknown nested fields, ambiguous event tags, invalid causal ordering,
wrong predecessor/state/step digests, false invariants, and any claimed
disposition or post-state that does not replay. The differential runner calls
Lean before launching native state, so an identity-incomplete or forbidden
trace cannot partially mutate a service.

## Permanent corpus

`formal/resilient/corpus/native-v1/manifest.json` is grow-only. Each entry
binds a stable ID, path, SHA-256, event count, and replay command.

| Stable ID | Required behavior |
|---|---|
| `native-job-5105811-generation-3-close-restart-rejoin` | generation-3 close/commit, node-0 service loss/new incarnation, concurrent node-1 closed-generation catch-up, no fatal peer exit/extra cohort restart/rollback/partial apply/double commit, preserved recovery path, both node applies, generation-4 READY/open/rejoin |
| `native-duplicate-conflict-stale-late-corrupt` | identical/conflicting duplicate, stale fence/incarnation, corrupt nonfinite, late closed contribution, duplicate/conflicting commit |
| `native-leased-ready-delay-expiry-insufficient` | delayed READY omitted from the immutable snapshot, retry-next-generation, lease expiry, deadline below Q/T, no commit |
| `native-owner-replay-reassignment-abort` | two bounded reassignments followed by typed abort |
| `native-participant-restart-open` | participant loss/restart during open |
| `native-service-restart-{open,closed,committed-apply}` | service loss/restart in every generation/apply phase |
| `native-manager-restart-{open,closed,committed-apply}` | manager loss/restart in every generation/apply phase |
| `native-trainer-restart-{open,closed,committed-apply}` | trainer loss/restart in every generation/apply phase, including partial apply invalidation |
| `native-fresh-fence-recovery` | monotonic fresh-fence recovery and old-fence rejection |

The compiled Lean definitions are the generator. To add a future minimized
divergence, add a named scenario to `ConformanceExamples.lean`, regenerate
the corpus, retain the first-divergence report while fixing the production
path or shared view, and keep its manifest ID forever.

## Deliberate mutation proof

`--fault-event-index 25` changes the native `exact_tokens` field by one before
calling the actual `CONTRIBUTION` transition for the permanent job-5105811
trace. It does not alter a comparison result. The service accepts and mutates
the different token/receipt state; the harness observes different native
pre/post digests, stops at event 25, and writes a canonical 26-event prefix
that both Lean and the CLI can replay. The focused pytest makes this proof
permanent.

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" scripts/conformance/run_native_lean_conformance.py \
  --trace formal/resilient/corpus/native-v1/native-job-5105811-generation-3-close-restart-rejoin.json \
  --build-manifest build/native-resilient-dataplane/native-artifacts.json \
  --fault-event-index 25
```

## Checklist application and limits

This gate applies every namespace in the compute-pool conformance checklist
and the gap matrix.

Protocol trace obligations exercised and compared here:

- R01–R08 and R11–R16;
- NDP01, NDP06, NDP07, and NDP10–NDP16;
- V21S01–V21S05, V21S07–V21S11, V21S13, V21S14, and V21S17; and
- ISP04–ISP07 as typed protocol/telemetry identities only.

Production-boundary evidence retained by the adapter, binary/source manifest,
and service tests:

- R09, R10, and R13;
- NDP02–NDP05, NDP08, NDP09, NDP14, and NDP17;
- V21S06, V21S12, V21S15, and V21S16; and
- ISP01–ISP03.

No boundary row is discharged by trace agreement alone. In particular, this
gate does **not** establish ISP timing/overlap, native dense byte movement,
buffer ownership under real load, exact floating-point reduction, CXI
provider behavior, Frontier qualification, real-model behavior, or any
two-/multi-node scale gate. Those remain separate native/reference, G1–G6,
and ISP evidence.

## Canonical local validation

From the repository root:

```bash
scripts/conformance/smoke_native_lean.sh
```

That wrapper sources the authoritative Frontier environment, builds the
pinned Lean executables, regenerates the corpus, builds/tests/installs/attests
the unified native bundle, replays all 486 events through fresh persistent
services, and runs the deliberate fault proof. It does not invoke Slurm.
