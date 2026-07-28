# Resilient protocol proof coverage

This matrix applies the Version 1 conformance checklist in
[`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`](../../docs/RESILIENT_DILOCO_COMPUTE_POOL.md)
and keeps the R, NDP, V21S, and ISP namespaces independent as required by
[`docs/RESILIENT_DILOCO_GAP_MATRIX.md`](../../docs/RESILIENT_DILOCO_GAP_MATRIX.md).
Every theorem below refers to `ResilientProtocol.transition`, its executable
decision stages, or `executeEvents`, which folds that same transition.

Assumption classes:

- **S0 — unconditional structural:** no well-formedness, delivery, scheduling,
  failure, or fairness premise.
- **S1 — well-formed structural:** only `WellFormedState`, whose fields are
  valid finite policy/schema bounds and structural authority consistency.
  It has no fairness field.
- **S2 — executable admission/frame:** an equality produced by the executable
  decision function, or an event/frame identity premise; never a second
  proof-only model.
- **J — job ordering:** the exact generation-3 structural facts in
  `Job5105811FailureOrdering`; no fairness premise.
- **P — explicitly conditional progress:** every field of
  `BoundedProgressAssumptions`, including finite close/stage deadlines,
  surviving eligible stable-worker quorum, exact-token floor, bounded
  failures/reassignments, eventual delivery/processing, and fair scheduling.

## Transition-constructor matrix

| Executable `Event` constructor | Machine-checked theorem(s) | Assumption class | Coordination requirement IDs represented |
|---|---|---:|---|
| `claimFence` | `transition_safety_all_constructors`; `transition_fence_monotone`; `transition_generation_monotone`; `transition_token_clock_monotone`; `transition_receipt_authority_monotone`; `fence_recovery_cannot_roll_back_generation_or_tokens`; `stale_fence_noninterference` | S0/S1 | R01, R04, R07, R11, R12; NDP01, NDP06, NDP10, NDP15, NDP16; V21S01, V21S05, V21S10, V21S14 |
| `peerTransition` | `transition_safety_all_constructors`; `next_ready_gate_requires_current_node_authority`; `next_ready_gate_requires_eight_registered_trainers`; `node_ready_apply_authority_is_all_eight_atomic` | S0/S1/S2 | R02, R03, R11, R14; NDP01, NDP13, NDP15, NDP16; V21S05, V21S07, V21S10, V21S11; ISP05 |
| `registerTrainer` | `transition_safety_all_constructors`; `node_ready_apply_authority_is_all_eight_atomic`; duplicate/non-mutating theorems | S0/S1/S2 | R02, R04, R11, R14; NDP01, NDP06, NDP10, NDP15; V21S05, V21S07, V21S11; ISP05 |
| `expirePeer` | `transition_safety_all_constructors`; `stale_incarnation_noninterference`; `retry_has_typed_recovery`; authority/receipt monotonicity | S0/S1 | R02, R03, R04, R06, R11; NDP01, NDP06, NDP13; V21S05, V21S10, V21S17; ISP04, ISP05 |
| `openGeneration` | `transition_safety_all_constructors`; generation/token/receipt monotonicity; `accepted_cohort_is_immutable_over_frame_trace` (subsequent frame) | S0/S1/S2 | R02, R03, R06, R11, R14, R16; NDP01, NDP13, NDP16; V21S05, V21S10, V21S17; ISP04, ISP06, ISP07 |
| `contribution` | `accepted_contribution_requires_eligible_leased_ready`; `eligible_leased_ready_fields`; `admitted_contribution_record_is_event_bound`; duplicate, stale, corrupt, closed, late, catch-up, and retry noninterference/typed-recovery theorems | S0/S2 | R01–R06, R11, R14, R15; NDP01, NDP06, NDP10, NDP13, NDP16; V21S01–V21S05, V21S10, V21S14, V21S17; ISP04 |
| `closeGeneration` | `accepted_cohort_is_immutable_in_frame`; `accepted_cohort_is_immutable_over_frame_trace`; `transition_safety_all_constructors`; insufficient-cohort is the only structural aborting close result | S0/S1/S2 | R02, R03, R06, R11, R14, R16; NDP01, NDP11, NDP13, NDP16; V21S03, V21S05, V21S09, V21S10, V21S17; ISP04, ISP06, ISP07 |
| `ownerLoss` | `owner_loss_recovery_abort_cannot_publish_partial_result`; existing commit/cohort trace immutability; authority/receipt monotonicity | S0/S2 | R06, R07, R08, R11, R12, R14; NDP01, NDP11, NDP13, NDP15; V21S05, V21S08, V21S10, V21S11, V21S14; ISP04, ISP05 |
| `commitGeneration` | `commit_decision_requires_declared_closed_cohort`; `accepted_commit_uses_declared_admissible_quorum`; `existing_commit_is_unique_in_frame`; `existing_commit_is_unique_over_frame_trace`; receipt/token monotonicity | S0/S1/S2 | R04, R06, R07, R12, R14, R15; NDP01, NDP06, NDP10, NDP11, NDP15, NDP16; V21S01, V21S03–V21S05, V21S08, V21S11, V21S14; ISP05, ISP06 |
| `publishResult` | frame commit/cohort immutability; `transition_receipt_authority_monotone`; corrupt/conflicting/deferred noninterference; `mutation_partial_publication_is_rejected` | S0/S2 | R07, R08, R12, R14; NDP01, NDP06, NDP10, NDP12, NDP15, NDP16; V21S01, V21S08, V21S11, V21S14; ISP04–ISP06 |
| `trainerApply` | `transition_safety_all_constructors`; duplicate/conflict/stale/corrupt/deferred noninterference; receipt, commit, and cohort frame preservation | S0/S1/S2 | R04, R07, R11, R12, R14, R15; NDP06, NDP10, NDP12, NDP13, NDP15, NDP16; V21S07, V21S08, V21S11, V21S14; ISP04–ISP06 |
| `reduceNodeApply` | `reduce_decision_requires_all_eight_distinct_receipts`; `accepted_node_apply_requires_all_eight_distinct_receipts`; `partial_or_timed_out_apply_never_becomes_node_authority`; `node_ready_apply_authority_is_all_eight_atomic` | S0/S1/S2 | R07, R11, R12, R14, R15; NDP10, NDP12, NDP13, NDP15, NDP16; V21S07, V21S08, V21S11, V21S14; ISP04–ISP06 |
| `loss` | `transition_safety_all_constructors`; existing commit/cohort immutability; monotone authority/receipt lineage; stale and typed retry/catch-up noninterference | S0/S1/S2 | R02, R03, R06, R07, R11, R12, R14; NDP01, NDP11, NDP13, NDP15; V21S08, V21S10, V21S11, V21S14; ISP04, ISP05 |
| `restartPeer` | `restart_recovery_cannot_roll_back_generation`; `owner_loss_recovery_abort_cannot_publish_partial_result`; frame commit/cohort immutability; `exact_job_5105811_generation_3_failure_order_is_safe` | S0/J | R01, R02, R07, R11, R12, R14; NDP01, NDP06, NDP11, NDP15, NDP16; V21S05, V21S10, V21S11, V21S14; ISP05 |
| `abortGeneration` | `owner_loss_recovery_abort_cannot_publish_partial_result`; existing commit/cohort immutability; monotone authority/receipt lineage | S0/S2 | R06, R07, R11, R12, R14; NDP01, NDP11, NDP13, NDP15; V21S08, V21S10, V21S11, V21S14; ISP04, ISP05 |

## Cross-constructor obligations

| Obligation | Theorem / checked artifact | Assumption class |
|---|---|---:|
| All 15 constructors preserve independent fence, generation, accepted-token, and receipt authority | `transition_safety_all_constructors`, `transition_authority_monotone`, `transition_receipt_authority_monotone` | S0/S1 |
| Every typed rejection except the explicitly state-changing insufficient/abort classes is the literal pre-state | `transition_nonmutating_of_disposition` and its disposition-specific corollaries | S0 |
| One authoritative commit/result and one immutable accepted cohort per fence/generation | `existing_commit_is_unique_over_frame_trace`, `accepted_cohort_is_immutable_over_frame_trace` | S2 |
| Admission only from eligible leased READY snapshot members | `accepted_contribution_requires_eligible_leased_ready`, `eligible_leased_ready_fields` | S2 |
| Commit only from the declared frozen quorum and exact-token floor | `accepted_commit_uses_declared_admissible_quorum` | S1/S2 |
| Owner loss, replay/restart, and abort never create commit/mailbox publication | `owner_loss_recovery_abort_cannot_publish_partial_result` | S0 |
| Node authority requires eight distinct current event-bound trainer receipts; partial/timed-out reduction is non-mutating | `accepted_node_apply_requires_all_eight_distinct_receipts`, `partial_or_timed_out_apply_never_becomes_node_authority`, `node_ready_apply_authority_is_all_eight_atomic` | S0/S1/S2 |
| Restart/fence recovery cannot roll generation/token authority backward or resurrect a historical contribution | `restart_recovery_cannot_roll_back_generation`, `fence_recovery_cannot_roll_back_generation_or_tokens`, `historical_contribution_catches_up_without_resurrection` | S0/S2 |
| Exact job-5105811 generation-3 race is catch-up, typed, non-mutating, budget-neutral, and partial-apply/second-commit free | `exact_job_5105811_generation_3_failure_order_is_safe`, `job_5105811_canonical_trace_binds_exact_late_event`, executable `job5105811OrderingFailures` | J |
| Representative double-commit, stale-write, conflicting-duplicate, mutable-cohort, partial-publication, and partial-apply mutants violate the obligations | six `mutation_*_is_rejected` theorems in `Mutations.lean` | S0 |
| Bounded next-generation participation | `bounded_next_generation_participation_under_exact_assumptions`; job specialization | P |
| No progress premise exists under total participant loss, permanent quorum loss, expired deadline without token floor, over-budget/unbounded faults, or unfair scheduling | five `*_exclude_progress_assumptions` theorems | P |

## Independent requirement-namespace disposition

The proved coordination-kernel scope is exactly:

- R01–R08, R11, R12, R14–R16;
- NDP01, NDP06, NDP10–NDP13, NDP15, NDP16;
- V21S01–V21S05, V21S07–V21S11, V21S13, V21S14, V21S17;
- ISP04–ISP07 only where a requirement is represented by a pure coordination
  transition, typed disposition, finite bound, or evidence identity.

The following remain explicit native/controller evidence obligations and are
not Lean theorem claims:

- R09, R10, R13;
- NDP02–NDP05, NDP07–NDP09, NDP14, NDP17;
- V21S06, V21S12, V21S15, V21S16;
- ISP01–ISP03.

For ISP04–ISP07, Lean proves only the represented coordination half. Physical
buffer capacity, foreground nonblocking time, atomic model/optimizer
visibility, causal telemetry completeness, p99/maximum pauses, and live tail
behavior remain native/qualification evidence.
