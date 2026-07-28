# Native-first coordination gate quality pass

Date: 2026-07-28

Task: `.quality-pass-native-first-coordination`

## Outcome

The batch graph is native-first at the 8-node admission boundary and
formal-plus-native at the 32-node boundary.

The production coordination path is required to use one compiled, pure,
deterministic transition kernel:

```text
step(authoritative_state, event)
    -> (new_state, disposition, effects)
```

Events are applied serially within each fenced authority. The kernel is the
only writer of authoritative generation, cohort, commit, apply, and recovery
state. Networking and libfabric operations, timers, storage, process
supervision, buffer ownership, and execution of returned effects remain
outside the kernel. Their results may re-enter only as typed, identity-complete
events.

The local native schedule-stress gate executes this exact production
transition path. It must not use a Lean runner, a proof-only model, a copied
test coordinator, or a mock decision kernel. The later conformance task
replays canonical traces through that same production kernel and the
executable Lean kernel and compares every disposition and authoritative state
digest.

## Final fork/join graph

Internal `.assign-*` and `.evaluate-*` lifecycle edges are omitted.

```text
.quality-pass-native-first-coordination
  -> harden-native-coordination-kernel
       -> stress-native-coordinator-schedules
            -> scale-v21-direct-8n
            -> integrate-formal-coordination-scale-gate
       -> conform-native-coordinator-to-lean4

requalify-v21-durable-collector-2n-clean
  -> qualify-simple-async-v21-2n-faults
       -> codify-v21-direct-scale-policy
            -> scale-v21-direct-8n
            -> integrate-formal-coordination-scale-gate
  -> scale-v21-direct-8n

bootstrap-lean4-resilient-protocol
  -> prove-lean4-resilient-protocol-safety
       -> integrate-formal-coordination-scale-gate
  -> conform-native-coordinator-to-lean4
       -> integrate-formal-coordination-scale-gate

scale-v21-direct-8n -----------------------------\
                                                   \
integrate-formal-coordination-scale-gate ----------> scale-v21-direct-32n
                                                       -> scale-v21-direct-128n
```

The direct dependency set relevant to admission is therefore:

| Task | Required WG predecessors, excluding lifecycle tasks |
|---|---|
| `harden-native-coordination-kernel` | `.quality-pass-native-first-coordination` |
| `stress-native-coordinator-schedules` | `harden-native-coordination-kernel` |
| `scale-v21-direct-8n` | `requalify-v21-durable-collector-2n-clean`, `qualify-simple-async-v21-2n-faults`, `codify-v21-direct-scale-policy`, `stress-native-coordinator-schedules` |
| `prove-lean4-resilient-protocol-safety` | `bootstrap-lean4-resilient-protocol` |
| `conform-native-coordinator-to-lean4` | `bootstrap-lean4-resilient-protocol`, `harden-native-coordination-kernel` |
| `integrate-formal-coordination-scale-gate` | `prove-lean4-resilient-protocol-safety`, `conform-native-coordinator-to-lean4`, `stress-native-coordinator-schedules`, `codify-v21-direct-scale-policy` |
| `scale-v21-direct-32n` | `scale-v21-direct-8n`, `integrate-formal-coordination-scale-gate` |
| `scale-v21-direct-128n` | `scale-v21-direct-32n` |

This makes the following admission rules explicit:

1. Eight nodes require immutable exact-source two-node clean evidence,
   immutable two-node fault/restart evidence, the direct scale-policy
   manifest, the production deterministic native kernel, and a passing local
   native schedule-stress manifest bound to that exact kernel.
2. Eight nodes do not require Lean bootstrap, completed Lean proofs,
   Lean/native trace conformance, or the formal integration task.
3. Thirty-two nodes require both an immutable collector-backed pass from the
   8-node rung and the formal integration manifest. The latter joins the
   matching Lean proof, actual-production-native trace conformance, native
   stress, and scale-policy evidence.
4. The 128-node rung remains serially after an immutable passed 32-node
   verdict.

## Graph corrections applied

- Added `requalify-v21-durable-collector-2n-clean` as a direct predecessor of
  `scale-v21-direct-8n`.
- Removed the redundant direct
  `harden-native-coordination-kernel -> scale-v21-direct-8n` edge. The
  required production lineage is now the explicit
  `harden-native-coordination-kernel -> stress-native-coordinator-schedules
  -> scale-v21-direct-8n` chain, and the stress manifest must bind the exact
  kernel manifest.
- Added `stress-native-coordinator-schedules` as a direct predecessor of
  `integrate-formal-coordination-scale-gate`.
- Removed `scale-v21-direct-8n` as a predecessor of
  `integrate-formal-coordination-scale-gate`. Formal integration is local and
  may complete independently; `scale-v21-direct-32n` remains the join that
  separately requires both formal integration and the passed 8-node verdict.
- Updated the `scale-v21-direct-8n` description to state explicitly that Lean
  formalization first gates the 32-node rung.
- Updated the `integrate-formal-coordination-scale-gate` description to remove
  stale wording that made the 8-node run a prerequisite for local formal
  integration.
- Updated the `scale-v21-direct-32n` description to describe the two-input
  join correctly and removed a stale `conformance/` typo.

## Compute-pool conformance checklist

The governing authorities reviewed for this pass are:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1 and its required
  conformance checklist.
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including the independent
  R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 namespaces.
- ADR-002 and the native data-plane/source-identity authorities referenced by
  those two normative documents remain mandatory inputs to the implementation,
  controller, runner, and scale tasks.

The current compute-pool document still contains its older generic serial
ladder text. `codify-v21-direct-scale-policy` is the explicit blocking task
that must update all five named normative/operational authorities together to
the reviewed direct systems ladder before any 8-node submission. This quality
pass changes WG coordination metadata only and does not silently treat the
pending normative update as already complete.

| Checklist obligation | Requirement mapping | Batch discharge |
|---|---|---|
| Cite the design authority and independent requirement sets | R01-R16; NDP01-NDP17; V21S01-V21S17; ISP01-ISP07 | Every implementation/stress/formal/integration/scale task retains a validation crosswalk; the formal integration task routes every row to proof, native runtime, stress, policy, two-node, or scale evidence without substitution. |
| Peer-owned leased READY membership, bounded waits, and no launched-rank/all-rank invariant | R02, R03, R06, R11, R13, R16; NDP01, NDP02, NDP07, NDP13, NDP16, NDP17; V21S02, V21S05, V21S09, V21S10, V21S13, V21S15-V21S17; ISP02, ISP04, ISP06, ISP07 | The native kernel models deterministic finite close over an immutable leased-READY snapshot. Stress explores delay/expiry/loss/rejoin. Policy and all scale rungs require V21S17 closure and reject launched-rank unanimity. |
| No SQLite/database, filesystem lock, or mutable metadata heartbeat in the rendered compute closure | R01, R10, R12; NDP01, NDP10, NDP15; V21S01, V21S12, V21S14; ISP03 | Native hardening preserves the no-database runtime boundary. Policy and scale preflight must retain rendered-closure audits. Lean models only pure decisions and is never used as a compute-node runtime or database substitute. |
| Fenced identity, deterministic weighted math, idempotence, stale/corrupt rejection, and atomic evidence | R01, R04, R05, R07, R08, R12, R15; NDP01, NDP05, NDP06, NDP10-NDP12, NDP15-NDP16; V21S01-V21S05, V21S07-V21S08, V21S11, V21S14; ISP01, ISP03, ISP05-ISP06 | The production kernel owns typed coordination dispositions and atomic authority. Stress checks native invariants after every event. Lean proves pure safety. Conformance compares exact native and Lean results. Numerical and byte-path claims remain native/two-node/scale evidence, not Lean claims. |
| Bounded non-Lustre point-to-point transport, backpressure/release, and no central full-model broker | R08-R10, R13-R14; NDP02-NDP14, NDP16-NDP17; V21S06, V21S09, V21S12-V21S13, V21S15-V21S17; ISP01-ISP04, ISP06-ISP07 | Transport, buffers, timers, and process effects stay outside the pure kernel. Native and scale gates retain compiled `cxi`, capacity, replay, release, no-MPI, no-Python-dense, and no-Lustre-hot-path evidence. Formal results cannot discharge these rows. |
| Applicable failure/deadline/recovery paths and an explicit minimum progress floor | R02, R06, R08, R11-R12, R14, R16; NDP07-NDP13, NDP15-NDP17; V21S02, V21S05, V21S07-V21S11, V21S14-V21S17; ISP02, ISP04-ISP07 | Kernel/stress/proofs cover non-mutating stale/closed input, bounded replay/abort, unique commit, all-eight apply, and non-rollback recovery. Two-node fault evidence remains a hard 8-node predecessor. Progress claims require named quorum, token floor, deadlines, bounded failures, eventual delivery, and fairness assumptions. |
| Exact commands, immutable artifacts, and prior-rung passes | R07, R12, R14, R16; NDP15-NDP17; V21S13-V21S17; ISP06-ISP07 | The 8-node task consumes exact clean/fault/policy/kernel/stress manifests. The 32-node controller consumes the passed 8 verdict plus the formal integration manifest. The 128-node runner accepts only its exact passed 32 predecessor. |
| Bounded asynchronous phase/tail evidence and no hidden foreground wait | R09, R14-R16; NDP04, NDP08-NDP09, NDP13, NDP15-NDP17; V21S02, V21S06-V21S09, V21S11-V21S17; ISP01-ISP07 | Policy and scale tasks require causal `freeze_snapshot`, `snapshot_admission`, `publish_network`, `aggregation`, `checkpoint`, `result_wait`, `apply_swap`, and total foreground-idle evidence, including every-event maximum/p99 bounds, zero foreground result wait, and rejection of approximately 200-second tail stalls. Native stress is complementary protocol evidence and cannot substitute for live timing. |

## Requirement routing across the revised batch

Every namespace remains independently normative; a proof or test in one
namespace does not replace a runtime obligation in another.

| Batch task or evidence class | Applicable requirement map and limits |
|---|---|
| `harden-native-coordination-kernel` | Pure production decision authority: R01-R08, R11-R12, R14-R16; NDP01, NDP06, NDP10-NDP13, NDP15-NDP16; V21S01-V21S05, V21S07-V21S11, V21S13-V21S14, V21S17; ISP04-ISP07 where represented by coordination transitions. Production call-path/boundary validation additionally retains R09-R10, R13; NDP02-NDP05, NDP07-NDP09, NDP14, NDP17; V21S06, V21S12, V21S15-V21S16; ISP01-ISP03 without claiming that pure state transitions prove transport, numerical, ownership, or timing behavior. |
| `stress-native-coordinator-schedules` | Applies all R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 as a coverage and evidence-routing checklist. It directly stresses coordination safety and bounded-state aspects on actual production code, while explicitly not substituting for numerical, native-byte-path, immutable-snapshot ownership, timing, two-node, scheduler, or Frontier evidence. |
| Exact-source two-node clean/fault plus `codify-v21-direct-scale-policy` | Together cover all R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07. Clean/fault evidence supplies actual native, scheduler, snapshot, timing, numerical, transport, failure, restart, and durable-publication facts; policy supplies the reviewed `8 -> 32 -> 128` admission schema and V21S17 closure. |
| `scale-v21-direct-8n`, `scale-v21-direct-32n`, and `scale-v21-direct-128n` | Each runner maps all R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 into identity/fencing/publication/recovery, membership/liveness/closure/evidence, and math/transport/bounds/throughput/atomic-overlap groups. Every rung requires exact `Partition=batch` and `QOS=debug` evidence separately, an immutable collector-backed machine verdict, and its exact predecessor chain. |
| `bootstrap-lean4-resilient-protocol` | Executable pure-model scope: R01-R08, R11-R16; NDP01-NDP02, NDP06-NDP13, NDP15-NDP17; V21S01-V21S05, V21S07-V21S11, V21S13-V21S17; ISP02, ISP04-ISP07. R09-R10, R13; NDP03-NDP05, NDP08-NDP09, NDP14; V21S06, V21S12; ISP01, ISP03 are represented only by typed evidence identities and trust boundaries and remain native obligations. |
| `prove-lean4-resilient-protocol-safety` | Proved coordination scope: R01-R08, R11-R12, R14-R16; NDP01, NDP06, NDP10-NDP13, NDP15-NDP16; V21S01-V21S05, V21S07-V21S11, V21S13-V21S14, V21S17; ISP04-ISP07 where represented by transitions. R09-R10, R13; NDP02-NDP05, NDP07-NDP09, NDP14, NDP17; V21S06, V21S12, V21S15-V21S16; ISP01-ISP03 remain explicit native/controller evidence obligations and are not theorem claims. |
| `conform-native-coordinator-to-lean4` | Protocol comparison: R01-R08, R11-R16; NDP01, NDP06-NDP07, NDP10-NDP16; V21S01-V21S05, V21S07-V21S11, V21S13-V21S14, V21S17; ISP04-ISP07. The adapter retains, but trace equality alone does not discharge, R09-R10, R13; NDP02-NDP05, NDP08-NDP09, NDP14, NDP17; V21S06, V21S12, V21S15-V21S16; ISP01-ISP03. |
| `integrate-formal-coordination-scale-gate` | Routes every R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 row to the matching proof, actual-native conformance, native stress, policy, exact-source two-node, passed-8, or controller evidence class. Missing or mismatched evidence from any class blocks 32 nodes. |

## Job 5105811 regression disposition

Job 5105811 remains permanent negative/regression evidence, not a qualifying
two-node pass. Its minimized ordering is retained in all relevant native and
formal paths:

1. A generation closes/commits.
2. Node 0 trainer/service state fails and returns under a new incarnation.
3. Node 1 concurrently submits a contribution to the already closed
   generation.
4. The correct production result is a typed, recoverable, non-mutating
   disposition.
5. Node 1 stays live, unrelated restart budget is unchanged, no second commit,
   partial apply, or rollback occurs, and next-generation participation is
   permitted only through normal fenced admission.

`harden-native-coordination-kernel` encodes this production regression,
`stress-native-coordinator-schedules` keeps it in the permanent native stress
corpus, `bootstrap-lean4-resilient-protocol` and
`prove-lean4-resilient-protocol-safety` retain the executable/formal case, and
`conform-native-coordinator-to-lean4` keeps the actual-native-versus-Lean trace
in the permanent differential corpus. The formal integration manifest for 32
nodes must name the same corpus identity and matching native source/binary/ABI,
Lean toolchain/model, and trace-schema digests.

## Quality-pass validation

- No implementation file was edited.
- No Python, native build, test runner, Slurm preflight, `squeue`, `sacct`,
  `scontrol`, `sbatch`, or other scheduler mutation was invoked.
- WG task descriptions and dependency edges were inspected after mutation.
- The 8-node task has no direct or transitive dependency on the Lean bootstrap,
  proof, conformance, or formal integration tasks.
- The 32-node task directly depends on both the passed 8-node rung and the
  formal integration gate.
- The 128-node task directly depends on the 32-node rung.
