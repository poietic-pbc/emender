# Formal/native coordination scale-gate integration — 2026-07-29

## Result and scope

This local, non-submitting integration is **passed**. It adds the formal/native
authorization that a future exact 32-node render must join to the
collector-backed passed 8-node verdict. It does not claim that the 8-node rung
has run, does not create a signed 32-node authorization, and does not authorize
any node count by itself.

The immutable local manifest is
`reports/frontier/formal-coordination-scale-gate-v1.json`:

- schema:
  `emender-formal-native-coordination-local-evidence-v1`;
- status: `local_passed_pending_8_node_join`;
- authorizes nodes: `[]`;
- manifest digest:
  `edb04f4179f866fe074e562118a17d450fc662687d2ac006c92492ca7c5181e6`;
- file SHA-256:
  `ad97203b8b9d06854ef9a0b24c25c8cc24ca83b4736c5afa62abcf122a78d2aa`;
- exact-source commit:
  `1cd6c9e28a208d6f0f8feec726242961b4f9e63b`;
- execution-source digest:
  `e310f0b9c6d3ad66db42f6b6b586328d72006f716b0b4d43c4953acb1954315f`.

The `join` command in
`scripts/frontier/build_formal_coordination_scale_gate.py` will accept this
local manifest only alongside the exact reviewed 32-node scale authorization
and exact collector-backed passed 8-node machine verdict. It checks their
identity contract and the native hardening/schedule-stress lineage and emits
an unsigned review candidate. The production controller and compute-side
launcher still require an Ed25519 reviewer signature; an unsigned candidate
cannot authorize a render or launch.

## Normative authority and trust boundary

This validation applies the conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including its new “Formal/native
coordination gate at 32 nodes” item, and the full requirement matrix in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

Lean specifies and proves properties of the single pure deterministic
coordination transition. The persistent compiled native service remains the
production mutation/runtime authority. It owns networking, libfabric, timers,
bounded buffers, process supervision, and every runtime effect. Canonical
production traces connect the two authorities.

The boundary is fail-closed:

- a test result is never presented as a proof;
- a Lean theorem is never presented as evidence about networking, timing,
  native memory, transport, scheduling, or other runtime effects;
- compute nodes hash and validate prebuilt evidence but never build or execute
  Lean;
- formal agreement is additive and does not replace numerical/reference,
  immutable-snapshot, foreground timing/tail, transport-byte/provider, G2,
  exact-source two-node, scheduler `Partition` and `QOS`, native stress,
  passed-8, or scale-policy evidence;
- missing, corrupt, stale, evaluator-only, partial, failed, wrong-source, or
  digest-mismatched evidence independently rejects exact 32-node preflight.

## Evidence-class legend

The row mapping below deliberately separates authorities:

| Class | Authority and admissible claim |
|---|---|
| **L — Lean theorem** | Pure coordination safety, and bounded progress only under the seven assumptions explicitly listed in `proof-manifest-v1.json`. |
| **D — differential trace** | Equality of the production-native and Lean coordination-visible transition for the permanent corpus; not a runtime or scale claim. |
| **N — native local** | Compiled service/ABI/source identity, native unit/integration checks, hardening, bounded schedules, replay and shrinkability; not a physical-network or scheduler claim. |
| **M — machine/runtime** | Numerical, immutable snapshot, timing/tails, transport bytes/provider, G2, two-node, scheduler, and live rung evidence from the applicable machine collectors. |
| **P — policy/controller** | Exact-source identity, ladder, predecessor, signature, render, and launcher fail-closed policy. |
| **E8 — exact passed 8** | The future collector-backed passed 8-node verdict, including the exact native hardening/stress digests it consumed. It is intentionally absent here. |

“L + D” below never absorbs an **N**, **M**, **P**, or **E8** obligation.
Rows still marked partial/gap by the normative gap matrix remain partial/gap;
formal integration does not relabel them.

## R01–R16 evidence-class mapping

| ID | Correct evidence class | Integration disposition |
|---|---|---|
| R01 | N + M + P | Scheduler-fenced allocation and newer-fence-before-load remain launcher/native/live evidence. Lean models fence monotonicity only; it does not establish a Slurm claim. |
| R02 | L + D + N + M | Pure lifecycle/incarnation transitions are proved and differentially replayed. Real lease clocks and process lifecycle remain native/machine evidence. |
| R03 | L + D + N + M | READY-snapshot transition semantics are formal/native. Live leased membership and the absence of launched-rank admission remain runtime evidence. |
| R04 | L + D + N | Fresh identities, stale noninterference, duplicate handling, and deterministic dispositions are theorem, corpus, and native-stress claims. |
| R05 | N + M | Exact weighted numerical aggregation is the native/reference and machine numerical gate; the coordination proof makes no numerical claim. |
| R06 | L + D + N + M | Safety of finite close/abort is proved. Progress is conditional on the named deadline/quorum/token/failure/delivery/fairness assumptions; actual deadline and no-unbounded-wait behavior is runtime evidence. |
| R07 | L + D + N + M | Unique coordination commit and receipt-chain transitions are proved/replayed. Durable checkpoint publication and exact physical recovery remain machine evidence. |
| R08 | N + M | Chunk bytes, checksums, credit/replay bounds, release, and absence of a full-model broker are compiled/native and live transport evidence, never Lean evidence. |
| R09 | M | Trainer ownership and coherent immutable snapshots require source/race/live overlap evidence. This task preserves the normative partial status. |
| R10 | N + M + P | Node-local paths and forbidden Lustre/database hot paths remain compiled launcher and machine evidence. |
| R11 | L + D + N + M | Rejoin/incarnation/late-work coordination is proved and replayed; actual disappear/rejoin and catch-up execution remains machine evidence. |
| R12 | L + D + N + M | Pure authority restoration is formal/native; optimizer ownership, checkpoint restoration, and fresh allocation remain runtime evidence. |
| R13 | N + M | Backend-neutral protocol structure and adapters are implementation/runtime evidence. The missing hyperscale-local fixture remains a documented partial gap. |
| R14 | M | Stage deadlines and disjoint foreground/background every-event timing are collector evidence. Lean does not prove wall-clock or tail latency; the partial status remains. |
| R15 | N + M | Numerical/reference correctness and accepted-token arithmetic require native/reference and live evidence, not formal coordination agreement. |
| R16 | P + M + E8 + L + D + N | The controller preserves current-source G2/two-node/scheduler/policy/predecessor gates. At 32 it additionally requires one signed joined manifest containing the exact E8 lineage plus the formal/native artifacts. No live rung pass is claimed here. |

## NDP01–NDP17 evidence-class mapping

| ID | Correct evidence class | Integration disposition |
|---|---|---|
| NDP01 | L + D + N | Pure authority mutation is proved/replayed, while the persistent compiled service and Python/C++ boundary are native-source and binary evidence. |
| NDP02 | N + M | Absence of failure-sensitive all-rank operations is a binary/source audit and live fault property; Lean is out of scope. |
| NDP03 | N + M | Persistent C++17 service and exact Frontier `cxi` are compiled/provider/startup evidence. Local build covers the binary; exact provider operation remains machine evidence. |
| NDP04 | N + M | Producer-direct immutable handoff is native evidence; coherent capture against mutable trainer state remains the ISP01 machine gap. |
| NDP05 | N + M | Bitwise exact weighted arithmetic/layout is native numerical/reference evidence only. |
| NDP06 | L + D + N | Fixed coordination identities and receipt semantics are theorem/corpus/native ABI claims; the actual packed ABI is attested by the compiled binary. |
| NDP07 | N + M | Endpoint exchange and current-fence AV routes are native/live provider evidence. |
| NDP08 | N + M | Capacity formulas and bounded buffers are native evidence; nonblocking behavior at every foreground edge remains the ISP04 machine gap. |
| NDP09 | N + M | Credit/slot correctness is native; proving credit exhaustion cannot delay foreground progress remains live ISP02/ISP04 evidence. |
| NDP10 | L + D + N + M | Coordination receipt idempotence is proved/replayed; CRC/payload integrity and once-only physical apply remain native/runtime evidence. |
| NDP11 | L + D + N + M | Bounded owner reassignment/replay coordination is formal/native. Physical transfer replay, cleanup, and owner-loss timing remain native/machine evidence. |
| NDP12 | N + M | Owner-direct redistribution and shared aggregate bytes are compiled transport and live-layout evidence. |
| NDP13 | L + D + N + M | Typed defer/retry containment is formal/native; absolute clocks and zero foreground wait remain machine evidence and the row remains partial. |
| NDP14 | D + N | The fixed C ABI, seqpacket control, bounded trace, and service-owned registry are compiled/source evidence; differential traces exercise the actual ABI path. |
| NDP15 | N + M | Immutable checkpoint handoff and atomic bounded apply are native/live evidence. Formal coordination does not fill ISP03/ISP05, so the row remains partial. |
| NDP16 | D + N + M | Canonical coordination traces and bounded fields are native/differential. Full causal phase/tail telemetry remains machine evidence and remains partial. |
| NDP17 | P + M + E8 + L + D + N | Exact G2/two-node/ladder gates remain independent. The 32-node launcher additionally validates the signed joined formal/native manifest by digest without executing Lean. Later physical rungs remain unclaimed. |

## V21S01–V21S17 evidence-class mapping

| ID | Correct evidence class | Integration disposition |
|---|---|---|
| V21S01 | L + D + N + P | Pure policy identity rejection is represented in the kernel/corpus; schema/ABI/source admission remains native/controller evidence. |
| V21S02 | L + D + N + M | Lag/drop/defer transition safety is formal/native. Zero foreground catch-up wait is machine evidence and remains partial. |
| V21S03 | L + D + N + M | Coordination accepts exact-token identities; numerical weighting/denominator correctness remains native/reference/machine evidence. |
| V21S04 | N + M + P | K40 and stateless eta-one outer arithmetic are policy/runtime/numerical evidence, not Lean coordination evidence. |
| V21S05 | L + D + N + M + P | Fenced identity and one-contribution admission are formal/native; exact two-node constants and timing are render/machine policy evidence. |
| V21S06 | N + M | Exclusive mutable-state ownership and coherent snapshot capture are native/source/live evidence. The ISP01/ISP02 gap is unchanged. |
| V21S07 | N + M | Atomic ScheduleFree apply and the 60-second all-eight bound require numerical/runtime fault evidence; the ISP05 gap is unchanged. |
| V21S08 | L + D + N + M | Old/conflicting/corrupt result rejection has coordination evidence; mailbox capacity and nonblocking foreground behavior remain native/machine evidence. |
| V21S09 | N + M + P | Resident/slot/credit/replay/mailbox bounds are native/preflight evidence; every full-capacity foreground result remains ISP04 machine evidence. |
| V21S10 | L + D + N + M | Leased membership/incarnation/old-work behavior is formal/native; missing-peer foreground timing remains machine evidence. |
| V21S11 | L + D + N + M | Node transaction markers have coordination evidence; all-eight atomic visibility/restart and 60-second completion remain machine evidence. |
| V21S12 | D + N + M | The persistent compiled service/ABI is production-native and differentially traced. Memfd/XPMEM/libfabric/CXI/point-to-point bytes and no-broker assertions remain native/machine evidence. |
| V21S13 | M | Honest causal wall-clock phases and every-event maximum/p99/tails are machine collector evidence. Formal agreement supplies no timing claim and the row remains partial. |
| V21S14 | L + D + N + M + P | Fenced pure recovery is formal/native; immutable model/outer/checkpoint bundle, exact seed, `sbcast`, and offline verification remain launcher/machine evidence. |
| V21S15 | P + M + L + D + N | Pinned Lean build/hygiene/corpus and bounded schedules run locally in CI as additive evidence. Current-source clean/fault/fresh two-node passes and explicit `Partition`/`QOS` remain machine gates. |
| V21S16 | P + M + E8 + L + D + N | The direct ladder and exact immediate predecessor remain controller/machine policy. Exact 32 additionally joins E8 to proofs/conformance/stress; no physical pass is inferred. |
| V21S17 | P + M | Empirical finite close/deadline/cadence arithmetic and reviewed READY snapshot are controller plus two-node machine evidence. Lean proves only transition safety under supplied inputs. |

## ISP01–ISP07 evidence-class mapping

Every immutable-snapshot-pipeline row is a runtime obligation. Formal
coordination agreement cannot close any ISP gap.

| ID | Correct evidence class | Integration disposition |
|---|---|---|
| ISP01 | N + M | Coherent safe-boundary capture and prohibition on background live-state reads require source/race/live evidence; still a gap. |
| ISP02 | N + M | The 1-second capture/admission bound and immediate next-K progress under each blocked background phase require causal live timing; still partial. |
| ISP03 | N + M | Immutable background inputs and no checkpoint I/O in foreground pauses require integration/restart evidence; still a gap. |
| ISP04 | N + M | Bounded capacity plus explicit skip/replace/drop/defer at every exhaustion edge requires cross-layer foreground evidence; still partial. |
| ISP05 | N + M | Atomic all-eight apply, 60-second bound, no partial visibility, and nonblocking late/failed results require live fault/timing evidence; still partial. |
| ISP06 | M | Separate causal phases, zero result wait, every-event maximum/p99, and bounds are collector/validator evidence; still a gap. |
| ISP07 | M | Adversarial every-event tail evidence and rejection of an approximately 200-second alternating stall are collector/validator evidence; still a gap. |

## Immutable artifacts

| Artifact | Result | SHA-256 / digest |
|---|---|---|
| Native hardening validation | prerequisite passed | `ba5fb5052b7aa10840613d69b91f280e64ffa2a1a2f6c33beb0f4aa4d53a1f57` |
| Integrated native schedule-stress manifest | 50,353 schedules; 1,979,412 transitions; two byte-identical repeats; zero safety failures | file `6a7a100518bd38d43e6430f32395024a31591704d121c8d22b5d0abddd0a03e6`; transcript `324f14d8f086a98c3de209e658dff089ab802d78747f76846577c24838fcd1b8` |
| Lean proof manifest | 18 SHA-bound artifacts; pure safety separated from seven assumed progress conditions | `f44d535568e3efe96ed668889e9d1a3f53893c680b63b8df87b0652935e19079` |
| Executable Lean kernel source | pinned theorem/executable transition | `fd23e1100bdd692b67a38bf8b853848bd924bb5119406861b63a250056bc31ea` |
| Lean conformance executable | local built oracle | `c712211f1142a7a9366c22d420aa1c717cbed277dd03b8895941655d57e16ed7` |
| Canonical trace schema | strict schema | `cf654525395e63b31b2d76e8109ee2bcc6a652f6273d1c6e4ca5bec9ecb776b4` |
| Permanent native/Lean corpus manifest | 15 traces, including the exact 63-event job-5105811 trace | `028fdccc8619de3127f6241353f1b9420376b070d63956af5e7ac3bb9659d15d` |
| Production-native/Lean differential manifest | 15/15 traces, 486/486 events, zero divergence | file `5b6f3f391f907a0e8b9e55d0f5c73cef5e06c5baef52b0094f5c52c90c17f4f0`; manifest `d1e9016fd362f9aa80ef80cea5a63e4dcdbbaf3dcb5d3936340d107fdd2ac1e6` |
| Refreshed job-5105811 agreement report | 63 events, agreement, clean source `1cd6c9e2…` | `85fb85e86a50b0ef5c0220928f300fc4ad1ca1c90aeda39c5adba8d2f3f373fa` |
| Local formal coordination manifest | complete local evidence, pending E8 join, authorizes `[]` | file `ad97203b8b9d06854ef9a0b24c25c8cc24ca83b4736c5afa62abcf122a78d2aa`; manifest `edb04f4179f866fe074e562118a17d450fc662687d2ac006c92492ca7c5181e6` |

The native differential additionally binds the actual production bundle
`6e962075594cf2db36280b55e05a35fde1965e67d8beefb40a3fec776b26d908`,
service binary
`fbb6beb8164c8c6511f4d3726e4586f13538c6ab998abaf092dfca6710241594`,
local library
`dbb0ea7dd163e0c4b600fe6185bc7ed4ba2bfcc43ad0a0e85512dc98d2ea4645`,
transport library
`ebe232da8974b7205fd1b43efee41374262ea3472cf66a15dc547beae8ba5572`,
`NDP_COORD_ABI_V1`, fixed structure sizes, every required production source
digest, and the trace-adapter digest.

## Fail-closed controller and launcher validation

The controller requires `--formal-coordination-gate` for exact 32-node
preflight after it has independently validated all pre-existing source,
policy, seed, native bundle, snapshot/timing, clean/fault/fresh, scheduler,
authorization, and immediate-predecessor evidence. The formal validator then
requires:

1. one final
   `emender-formal-native-coordination-scale-gate-v1` manifest;
2. passed/complete/non-partial/non-evaluator/non-stale status;
3. the exact 32-node identity contract and signed scale authorization digest;
4. the exact passed 8-node manifest digest and file SHA;
5. the hardening and stress file hashes named by that passed-8 verdict;
6. all proof, coverage, trace-schema, corpus, job-5105811, production source,
   binary, ABI, adapter, and zero-divergence hashes;
7. an Ed25519 signature under the same trusted reviewer key as the scale
   authorization and predecessor verdict.

The launcher repeats manifest/file digest, identity, lineage, proof/native
content, and reviewer-signature verification in the compute closure. It
imports no Lean executable and runs no theorem prover. Missing or mutated
fields fail independently in
`tests/test_formal_coordination_scale_gate.py`, including corrupt JSON,
manifest-digest mismatch, failed/partial/evaluator/stale status, incomplete
requirement/artifact sets, wrong exact identity, hardening/stress/passed-8
lineage, proof-policy substitution, hidden progress assumptions, source,
bundle, binary, ABI, adapter, kernel, corpus, job-5105811, and nonzero
differential divergence.

## Exact validation commands and results

Every Python, native, and Lean command below followed:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Proof and permanent corpus:

```bash
formal/resilient/scripts/verify-proof-manifest.sh
(cd formal/resilient && scripts/smoke.sh)
```

Result: proof manifest passed with 18 bound artifacts; pinned Lake built the
complete safety/progress/regression/mutation/conformance package; 10 executable
scenarios passed; the canonical job-5105811 trace replayed exactly 63
transitions; all 15 permanent native traces generated/replayed/compared.

Focused controller, formal mutation, and launcher tests:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_formal_coordination_scale_gate.py \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py
```

Result: **162 passed**. The final consolidated run added the 7 native/Lean
differential and deliberate-fault cases and passed **169/169**.

Compiled production-native build/install and bounded schedules:

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result: **12/12 CTests passed**, including the direct kernel, persistent
service/RPC, ABI/provider, and 12,000-schedule bounded deterministic stress
test. The build manifest records clean source commit `1cd6c9e2…`.

Integrated long deterministic campaign:

```bash
RANDOM_SCHEDULES=50000 DETERMINISM_REPEATS=2 \
  scripts/frontier/run_native_coordination_stress.sh
```

Result: **50,353 schedules**, **1,979,412 native transitions**, 29/58 pair
orders, 4/24 three-race orders, all 32 restart role/phase cases, all event and
disposition classes, 2/2 known-bad witnesses detected and minimized, zero
safety failures, and byte-identical transcript
`324f14d8f086a98c3de209e658dff089ab802d78747f76846577c24838fcd1b8`.

Production differential and deliberate-fault detection:

```bash
"$EMENDER_PYTHON" scripts/conformance/generate_native_lean_manifest.py \
  --build-manifest build/native-resilient-dataplane/native-artifacts.json \
  --lean-runner formal/resilient/.lake/build/bin/resilient-conformance \
  --output reports/conformance/native-lean-v1/manifest.json \
  --job-5105811-output \
    reports/conformance/native-lean-v1/job-5105811-agreement.json

EMENDER_NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
EMENDER_LEAN_CONFORMANCE_RUNNER="$PWD/formal/resilient/.lake/build/bin/resilient-conformance" \
  "$EMENDER_PYTHON" -m pytest -q tests/test_native_lean_conformance.py
```

Result: **15 traces and 486 events agreed with zero divergence**; the
deliberate-fault/differential suite was **7 passed**.

Local non-authorizing manifest:

```bash
"$EMENDER_PYTHON" \
  scripts/frontier/build_formal_coordination_scale_gate.py local \
  --execution-source-digest \
    e310f0b9c6d3ad66db42f6b6b586328d72006f716b0b4d43c4953acb1954315f \
  --source-commit 1cd6c9e28a208d6f0f8feec726242961b4f9e63b \
  --output reports/frontier/formal-coordination-scale-gate-v1.json
```

Result: the exact immutable local manifest above was generated with
`authorizes_nodes=[]`.

No `sbatch`, `srun`, `squeue`, `sacct`, or other Slurm command was executed by
this task. The stress manifest independently records zero Slurm calls and zero
Frontier allocations.

## CI and promotion status

`.github/workflows/ci.yml` now runs:

- the pinned resilient Lake build;
- theorem/proof-manifest hygiene rejecting `sorry`, `admit`,
  `native_decide`, `axiom`, `opaque`, `unsafe`, and Boolean theorem
  substitutes;
- the permanent corpus including job 5105811;
- production-native differential conformance and deliberate fault detection;
- the formal-gate mutation suite;
- the bounded deterministic native schedule campaign.

Longer exploration remains explicitly seeded, replayable by seed/index or
permanent schedule, causally shrinkable with deterministic `ddmin`, and
byte-repeatable.

The next 32-node task remains unable to authorize a render until the separate
8-node runner supplies a signed collector-backed passed-8 verdict containing
the exact hardening/stress lineage. Only then may the local manifest be joined
to the reviewed exact 32-node authorization and signed. This record makes no
claim that those external prerequisites already exist.
