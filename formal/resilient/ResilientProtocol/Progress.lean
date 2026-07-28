import ResilientProtocol.Safety

namespace ResilientProtocol

/-!
# Explicitly conditional bounded progress

This module deliberately contains no unconditional liveness theorem.  A
bounded result requires every assumption named by the compute-pool contract:
finite close and stage deadlines, a surviving eligible stable-worker quorum,
the exact-token floor, bounded permitted failures and owner reassignments,
eventual delivery/processing, and fair scheduling of an enabled transition.

Consequently there is no theorem here under total participant loss, permanent
quorum loss, unbounded faults, an expired deadline without the floor, or
unfair scheduling.  Those cases do not inhabit
`BoundedProgressAssumptions`.
-/

def stateBeforeIndex
    (initial : RunState) (schedule : List Event) (index : Nat) :
    RunState :=
  executeEvents initial (schedule.take index)

/--
The exact assumption package for one bounded next-generation participation
result.  `enabledTransition` uses the executable contribution decision that
feeds `transition`; it is not a second model or a Boolean liveness oracle.
-/
structure BoundedProgressAssumptions
    (initial : RunState) where
  closeDeadline : Tick
  stageDeadline : Tick
  finiteCloseAndStageDeadlines :
    closeDeadline.value < stageDeadline.value
  survivingEligibleStableWorkerQuorum :
    initial.generation.readySnapshot.length ≥ initial.policy.qMin
  survivingExactTokenFloor :
    contributionTokens initial.generation.accepted ≥ initial.policy.tMin
  permittedFailures : Nat
  observedFailures : Nat
  boundedPermittedFailures :
    observedFailures ≤ permittedFailures
  boundedOwnerReassignments :
    initial.generation.ownerReassignments ≤
      initial.policy.maxOwnerReassignments
  deliveryBound : Nat
  schedule : List Event
  enabledIndex : Nat
  nextContribution : ContributionEvent
  nextGeneration :
    initial.generation.generation.value <
      nextContribution.context.generation.value
  eventualDeliveryAndProcessing :
    enabledIndex < schedule.length ∧
      schedule.length ≤ deliveryBound
  fairSchedulingOfEnabledTransitions :
    schedule[enabledIndex]? =
      some (.contribution nextContribution)
  enabledTransition :
    decideContribution
      (stateBeforeIndex initial schedule enabledIndex)
      nextContribution = .acceptContribution

/--
Under all named assumptions, a next-generation contribution is accepted at a
finite schedule index bounded by `deliveryBound`.
-/
theorem bounded_next_generation_participation_under_exact_assumptions
    (initial : RunState)
    (assumptions : BoundedProgressAssumptions initial) :
    ∃ index : Nat, ∃ event : ContributionEvent,
      index < assumptions.deliveryBound ∧
      assumptions.schedule[index]? =
        some (.contribution event) ∧
      initial.generation.generation.value <
        event.context.generation.value ∧
      (transition
        (stateBeforeIndex initial assumptions.schedule index)
        (.contribution event)).disposition = .accepted := by
  refine
    ⟨assumptions.enabledIndex, assumptions.nextContribution,
      ?_, assumptions.fairSchedulingOfEnabledTransitions,
      assumptions.nextGeneration, ?_⟩
  · exact
      Nat.lt_of_lt_of_le
        assumptions.eventualDeliveryAndProcessing.1
        assumptions.eventualDeliveryAndProcessing.2
  · exact
      admitted_contribution_transition_is_accepted
        (stateBeforeIndex initial assumptions.schedule
          assumptions.enabledIndex)
        assumptions.nextContribution assumptions.enabledTransition

/--
The theorem also exposes every non-fairness premise in its conclusion, so a
consumer cannot silently discard the finite-deadline, quorum, token-floor, or
bounded-fault obligations when forwarding the result.
-/
theorem bounded_progress_assumptions_are_explicit
    (initial : RunState)
    (assumptions : BoundedProgressAssumptions initial) :
    assumptions.closeDeadline.value <
        assumptions.stageDeadline.value ∧
      initial.generation.readySnapshot.length ≥ initial.policy.qMin ∧
      contributionTokens initial.generation.accepted ≥
        initial.policy.tMin ∧
      assumptions.observedFailures ≤ assumptions.permittedFailures ∧
      initial.generation.ownerReassignments ≤
        initial.policy.maxOwnerReassignments ∧
      assumptions.enabledIndex < assumptions.schedule.length ∧
      assumptions.schedule.length ≤ assumptions.deliveryBound ∧
      assumptions.schedule[assumptions.enabledIndex]? =
        some (.contribution assumptions.nextContribution) := by
  exact
    ⟨assumptions.finiteCloseAndStageDeadlines,
      assumptions.survivingEligibleStableWorkerQuorum,
      assumptions.survivingExactTokenFloor,
      assumptions.boundedPermittedFailures,
      assumptions.boundedOwnerReassignments,
      assumptions.eventualDeliveryAndProcessing.1,
      assumptions.eventualDeliveryAndProcessing.2,
      assumptions.fairSchedulingOfEnabledTransitions⟩

theorem total_participant_loss_excludes_progress_assumptions
    (initial : RunState)
    (hq : 0 < initial.policy.qMin)
    (hloss : initial.generation.readySnapshot = []) :
    BoundedProgressAssumptions initial → False := by
  intro assumptions
  have hzero :
      initial.policy.qMin ≤ 0 := by
    simpa [hloss] using
      assumptions.survivingEligibleStableWorkerQuorum
  exact (Nat.not_le_of_lt hq) hzero

theorem permanent_quorum_loss_excludes_progress_assumptions
    (initial : RunState)
    (hloss :
      initial.generation.readySnapshot.length <
        initial.policy.qMin) :
    BoundedProgressAssumptions initial → False := by
  intro assumptions
  exact
    (Nat.not_le_of_lt hloss)
      assumptions.survivingEligibleStableWorkerQuorum

theorem expired_deadline_without_token_floor_excludes_progress_assumptions
    (initial : RunState)
    (_expired :
      initial.generation.closeTick.value ≤ initial.now.value)
    (hfloor :
      contributionTokens initial.generation.accepted <
        initial.policy.tMin) :
    BoundedProgressAssumptions initial → False := by
  intro assumptions
  exact
    (Nat.not_le_of_lt hfloor)
      assumptions.survivingExactTokenFloor

theorem unbounded_or_over_budget_faults_exclude_progress_assumptions
    (initial : RunState)
    (assumptions : BoundedProgressAssumptions initial)
    (hover :
      assumptions.permittedFailures < assumptions.observedFailures) :
    False :=
  (Nat.not_le_of_lt hover) assumptions.boundedPermittedFailures

theorem unfair_schedule_excludes_progress_assumptions
    (initial : RunState)
    (assumptions : BoundedProgressAssumptions initial)
    (hunfair :
      assumptions.schedule[assumptions.enabledIndex]? ≠
        some (.contribution assumptions.nextContribution)) :
    False :=
  hunfair assumptions.fairSchedulingOfEnabledTransitions

end ResilientProtocol
