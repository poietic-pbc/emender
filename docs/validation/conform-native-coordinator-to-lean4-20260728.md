# Native coordinator / Lean 4 conformance validation

Date: 2026-07-28

Task: `conform-native-coordinator-to-lean4`

Normative authorities:

- [`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
  including its complete conformance checklist and Native data-plane binding;
- [`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md);
- [`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md);
- [`NATIVE_LEAN_COORDINATION_CONFORMANCE.md`](../NATIVE_LEAN_COORDINATION_CONFORMANCE.md);
- `formal/resilient/ResilientProtocol/Kernel.lean`, the pure protocol oracle;
  and
- `src/native_resilient_dataplane/src/coordination_kernel.cpp`, the compiled
  production mutation authority.

## Result

PASS. One strict canonical trace now executes first through the pinned Lean 4
oracle and then through the installed persistent production service. Every
canonical event agrees on typed disposition, every normalized authoritative
post-state field, and the canonical structural digest. The adapter does not
carry admission, closure, duplicate, receipt, commit, apply, recovery, owner,
or restart policy.

The permanent corpus contains 15 traces and 486 events. It includes the
minimized job-5105811 generation-closed race, all requested duplicate/stale/
late/corrupt cases, delayed and expired READY leases, owner replay/abort,
participant loss, the service/manager/trainer by open/closed/committed-apply
matrix, result-versus-failure and apply-versus-restart interleavings, and
fresh-fence rejoin.

No Slurm command, GPU, model tensor, checkpoint, remote libfabric peer, or
external runtime service was used. The service used the explicitly test-only
local `tcp;ofi_rxm` provider. This validation does not claim CXI, numerical,
buffer-ownership, timing, Frontier, two-node, or scale qualification.

## Exact source and pushed identity

The final source identity used to generate the retained evidence is:

```text
c97b5d8fe0a62884f52e28b25aa25cc03bacee9c
```

It contains implementation commit
`1ef567b7c972fbe620782ff4c4477a1106f43193` and the directly reproducible
first-divergence follow-up, plus the semantic merge of `origin/main` required
by the WG completion gate. That merge preserved remote equal-generation
peer-control rejoin and the compiled production authority, with 10/10 focused
merge tests passing. `git ls-remote --heads origin
wg/agent-1644/conform-native-coordinator-to-lean4` returned the exact same
`c97b5d8fe0a62884f52e28b25aa25cc03bacee9c`; local/remote equality was true.
The final refreshed-evidence commit and its pushed equality are also recorded
in the WG task log.

The clean build attestation recorded:

| Identity | Exact value |
|---|---|
| source commit | `c97b5d8fe0a62884f52e28b25aa25cc03bacee9c` |
| source tree dirty | `false` |
| native bundle SHA-256 | `6e962075594cf2db36280b55e05a35fde1965e67d8beefb40a3fec776b26d908` |
| build-manifest SHA-256 | `3e3fe3a3c9229f19f3323820f4215f85002c18b379bccb07e669b3b48d0160ca` |
| `libemender_ndp.so.1` SHA-256 | `dbb0ea7dd163e0c4b600fe6185bc7ed4ba2bfcc43ad0a0e85512dc98d2ea4645` |
| `ndp_cxi_service` SHA-256 | `fbb6beb8164c8c6511f4d3726e4586f13538c6ab998abaf092dfca6710241594` |
| `libemender_ndp_transport.so.1` SHA-256 | `ebe232da8974b7205fd1b43efee41374262ea3472cf66a15dc547beae8ba5572` |
| `ndp_frontier_2n_gate` SHA-256 | `7d8990b56ad906549964f99bbdfbc1f605cb78f2759bba475748c24a3ec7d0ef` |
| local and transport ABI | `65536` (`0x00010000`) |
| coordination event/member/result bytes | `312 / 200 / 52016` |
| Lean toolchain | `leanprover/lean4:v4.26.0@d8204c9fd894f91bbb2cdfec5912ec8196fd8562` |
| Lean oracle executable SHA-256 | `83dd5f2cedfe03cddd028f1f38964c048969912f220172bf4673b825db795e33` |
| trace schema | `emender-resilient-coordination-trace-v1` |
| trace schema SHA-256 | `cf654525395e63b31b2d76e8109ee2bcc6a652f6273d1c6e4ca5bec9ecb776b4` |
| common state view | `emender-native-lean-authority-view-v1` |
| policy | `async-decoupled-v2.1-simple` / `emender-async-policy-v2.1` |
| policy digest | `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` |
| source schema | `emender-async-v21-execution-source-v1` |
| corpus-manifest SHA-256 | `028fdccc8619de3127f6241353f1b9420376b070d63956af5e7ac3bb9659d15d` |

The agreement report additionally hashes every source file in the audited call
path, the Lean `Types`, `Kernel`, `Trace`, and `Conformance` modules, the trace
schema, the installed binary bundle, and the Lean executable.

## Source and runtime call-path audit

The adapter reaches the same mutation used by the compiled service:

```text
NativeTraceAdapter._step
  -> NativeCoordinationAuthority.step
  -> Client.coordination_step
  -> ndp_coord_step_v1
  -> rpc::Opcode::CoordinationStep
  -> LocalServiceCore::coordination_step
  -> Service::coordination_step
  -> coordination::step
```

The source-level sites are:

- `ndm/native_lean_conformance.py:417`, the thin event projection;
- `ndm/native_coordination.py:292` and `:297`, the production authority
  wrapper and its client call;
- `ndm/native_dataplane.py:1265` and `:1283`, the public C ABI invocation;
- `src/native_resilient_dataplane/src/client.cpp:584` and `:601`, the ABI
  implementation and production coordination opcode;
- `src/native_resilient_dataplane/src/rpc_server.cpp:433` and `:442`, service
  RPC dispatch;
- `src/native_resilient_dataplane/src/ndp.cpp:1987`, local service-core entry;
- `src/native_resilient_dataplane/src/ndp.cpp:1742` and `:1777`, persistent
  service state and the sole `coordination::step(coordination_state_, event)`
  mutation; and
- `src/native_resilient_dataplane/src/coordination_kernel.cpp`, the production
  transition implementation.

`audit_production_call_path` at `ndm/native_lean_conformance.py:1291` fails if
any required link disappears. `runtime_identity_manifest` at line 1194 binds
the audited sources and runtime artifacts. The test launches the installed
`ndp_cxi_service`; it does not instantiate a Python coordinator, mock service,
or test-only transition.

On the oracle side,
`formal/resilient/ResilientProtocol/Conformance.lean:283` calls
`ResilientProtocol.transition` first and derives the common view only from its
returned state. It does not provide an alternate transition. The native runner
invokes the Lean executable before launching the service, so malformed or
identity-incomplete input cannot partially mutate native state.

## Compared authority

Every event comparison includes:

- disposition and invariant verdict;
- trace/source/policy/toolchain/run/allocation identities;
- fence, generation, attempt, worker, node, incarnation, sequence, owner
  epoch, reassignment count, and lifecycle/synchronized-generation state;
- exact-token clock, accepted token and contribution counts;
- immutable cohort, cohort digest, contribution identities, payload digests,
  receipt digests, and exact tokens;
- result ID/digest, result-receipt workers, commit receipt and authority;
- node-apply identities/digests and applied phase; and
- the versioned canonical post-state digest.

The native report also keeps each underlying production call's independent
kernel pre/post digest. Comparison stops at the first unequal event.

The native loader independently rejects duplicate keys, noncanonical key/
whitespace order, wrong schema/toolchain, missing top-level identities, and
invalid bounded nominal values. Lean independently rejects unknown or
ambiguous nested fields, bad causal order, wrong predecessor/state/step
digests, claimed dispositions that do not execute, and invariant failure.

## Permanent corpus identities

The grow-only manifest is
`formal/resilient/corpus/native-v1/manifest.json`. Each entry includes the full
trace SHA-256 and a repository-relative replay command.

| Stable corpus ID | Events |
|---|---:|
| `native-job-5105811-generation-3-close-restart-rejoin` | 63 |
| `native-duplicate-conflict-stale-late-corrupt` | 37 |
| `native-leased-ready-delay-expiry-insufficient` | 29 |
| `native-owner-replay-reassignment-abort` | 31 |
| `native-participant-restart-open` | 27 |
| `native-service-restart-open` | 27 |
| `native-service-restart-closed` | 30 |
| `native-service-restart-committed-apply` | 32 |
| `native-manager-restart-open` | 27 |
| `native-manager-restart-closed` | 30 |
| `native-manager-restart-committed-apply` | 32 |
| `native-trainer-restart-open` | 27 |
| `native-trainer-restart-closed` | 30 |
| `native-trainer-restart-committed-apply` | 33 |
| `native-fresh-fence-recovery` | 31 |
| **Total** | **486** |

The manifest's 15 SHA-256 values were recomputed from disk and matched. The
pinned Lean generator reproduced every checked JSON byte-for-byte.

### Job 5105811

The permanent trace SHA-256 is
`bdaa2ec4029c4bdbeaf7161281bf50f86687882d3ca7463a5100e6abefe16803`.
All 63 events agreed, ending at common-view state digest
`785e61f3f01402500a22071fbab59ee03800cd92b6573c2d923b20540cbd36f0`.

The trace retains generation-3 close/commit; node-0 trainer/service loss and a
new incarnation; concurrent node-1 contribution to the closed generation;
typed, nonfatal, non-mutating catch-up; preserved restart authority; full
node-apply before READY; and generation-4 rejoin. Assertions exclude an extra
cohort restart, rollback, partial apply, double commit, or fatal peer exit.

The retained agreement report is
`reports/conformance/native-lean-v1/job-5105811-agreement.json`, SHA-256
`4fb199d6b504aa9d5b36441d286e8e8c5617b9c4edc691e4bd68d9d5ea4bb99d`.

## Deliberate actual-mutation divergence

Fault event 25 changes the native `CONTRIBUTION.exact_tokens` field from
1,966,080 to 1,966,081 before invoking the actual production transition. It
does not change expected output or the comparator.

The service accepted and mutated the different event. Its underlying native
pre/post digests differ. The harness then stopped at canonical event 25 with
differences in accepted tokens, contribution exact tokens, derived receipt,
and canonical state digest.

Retained files:

- 26-event canonical replay:
  `reports/conformance/native-lean-v1/deliberate-fault/native-job-5105811-generation-3-close-restart-rejoin.first-divergence-25.json`,
  SHA-256
  `ce2a88d6ea3d5bb40aa777e404ce404fe4d164403fdd0c3a57ad5cff72bbe943`;
- first-divergence report:
  `reports/conformance/native-lean-v1/deliberate-fault/native-job-5105811-generation-3-close-restart-rejoin.first-divergence-25.report.json`,
  SHA-256
  `f077d1f514af31b8f0cc349d0e073b553bf2ec33c215ab111a8e858fc44811b6`.

Direct reproduction from the repository root:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" scripts/conformance/run_native_lean_conformance.py \
  --trace reports/conformance/native-lean-v1/deliberate-fault/native-job-5105811-generation-3-close-restart-rejoin.first-divergence-25.json \
  --build-manifest build/native-resilient-dataplane/native-artifacts.json \
  --lean-runner formal/resilient/.lake/build/bin/resilient-conformance \
  --fault-event-index 25 \
  --divergence-directory reports/conformance/native-lean-v1/deliberate-fault
```

Exit status 1 is the expected observed divergence. For a non-injected future
divergence, add the minimized named scenario to
`ConformanceExamples.lean`, regenerate the manifest, retain its stable ID and
first-divergence report, and keep that corpus entry permanently.

## Exact validation commands and results

All Python and native commands followed:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

and used `"$EMENDER_PYTHON"` / `PYTHON_BIN="$EMENDER_PYTHON"`.

### Pinned Lean build and smoke

```bash
formal/resilient/scripts/smoke.sh
```

Result: exit 0. The pinned schema digest passed; the Lean library and
executables built; 10 base executable scenarios passed; all 15 conformance
scenarios passed; every checked trace regenerated byte-for-byte; every trace
was accepted by the independent oracle; malformed, ambiguous, and
forbidden-reordered cases failed closed.

### Unified native build, install, tests, and attestation

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: exit 0; 11/11 CTests passed. This includes the direct production
kernel, local ABI, persistent service RPC, provider-selection, capacity, and
production-fail-closed tests. The installed bundle was then attested with the
clean final source identity above.

### Differential test suite

```bash
"$EMENDER_PYTHON" -m pytest -q tests/test_native_lean_conformance.py
```

Result on final integrated source `c97b5d8f`: `7 passed in 79.45s`. The seven
tests cover corpus identity and canonical encoding; duplicate/reordered/
incomplete rejection; Lean unknown event rejection before native launch;
source call-path audit; all 15 traces and 486 live-service events;
actual-mutation fault observation and replay; and Lean/Python common-view
digest equality.

### Syntax and manifest checks

```bash
"$EMENDER_PYTHON" -m py_compile \
  ndm/native_lean_conformance.py \
  scripts/conformance/run_native_lean_conformance.py \
  scripts/conformance/generate_native_lean_corpus.py \
  tests/test_native_lean_conformance.py
git diff --check
```

Result: exit 0. CI YAML also parsed successfully. Ruff was not installed in
the canonical environment, so no Ruff result is claimed.

## Compute-pool checklist application

The complete checklist in `RESILIENT_DILOCO_COMPUTE_POOL.md` and all four gap
matrix namespaces were reviewed. This task applies them in two explicit
classes.

Protocol trace obligations exercised by event-by-event agreement:

- R01–R08 and R11–R16;
- NDP01, NDP06, NDP07, and NDP10–NDP16;
- V21S01–V21S05, V21S07–V21S11, V21S13, V21S14, and V21S17; and
- ISP04–ISP07 only to the extent of their typed protocol and telemetry
  identities.

Production-boundary evidence retained by the adapter, binary/source/ABI
manifest, call-path audit, and compiled service tests:

- R09, R10, and R13;
- NDP02–NDP05, NDP08, NDP09, NDP14, and NDP17;
- V21S06, V21S12, V21S15, and V21S16; and
- ISP01–ISP03.

R13, NDP14, and related identities appear in both lists because they have a
pure trace component and a production-boundary component. No boundary row is
discharged by trace agreement alone.

Specifically retained outside this gate are coherent real-trainer snapshot
ownership and overlap (ISP01–ISP03/V21S06), native dense bytes/buffer lifetime
and credit behavior (NDP02–NDP05/NDP08/NDP09), exact binary64 aggregation and
model execution, real CXI/provider behavior, causal wall-clock latency and
tail timing, Frontier scheduler evidence, fault timing under real peers, and
two-/multi-node scale authorization.
