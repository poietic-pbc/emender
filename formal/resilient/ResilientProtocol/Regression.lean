import ResilientProtocol.Examples
import ResilientProtocol.Progress

namespace ResilientProtocol

/-!
# Permanent job-5105811 regression

`Job5105811FailureOrdering` records the exact authority facts at the reported
race: generation 3 is closed and committed, node 0 has lost service/trainer
state and reincarnated for generation 4, and node 1 concurrently contributes
to generation 3.  The theorem evaluates the same `transition` used by the
canonical trace runner.  It has no scheduler or fairness hypothesis.
-/

structure Job5105811FailureOrdering
    (state : RunState) (late : ContributionEvent) : Prop where
  contextAuthority :
    contextAuthorityDisposition? state late.context = none
  generation3 : state.generation.generation = ⟨3⟩
  lateGeneration3 : late.context.generation = ⟨3⟩
  sameAttempt : late.context.attempt = state.generation.attempt
  sameOwnerEpoch :
    late.context.ownerEpoch = state.generation.ownerEpoch
  unseenContribution :
    state.generation.seen.find? (contributionKeyMatches late) = none
  committed : state.generation.status = .committed
  frozenCohortPresent : state.generation.cohort.isSome = true
  authoritativeCommitPresent : state.generation.commit.isSome = true
  resultAuthorityPresent : state.lastResult.isSome = true
  node0Reincarnated :
    ∃ peer,
      findPeer? state (workerId "worker-0") = some peer ∧
      peer.node = nodeId "node-0" ∧
      peer.incarnation = incarnation "peer-inc-0-restarted" ∧
      peer.phase = .sync ∧
      peer.syncedGeneration = ⟨4⟩
  node1StillParticipating :
    ∃ peer,
      findPeer? state (workerId "worker-1") = some peer ∧
      peer.node = nodeId "node-1" ∧
      peer.incarnation = incarnation "peer-inc-1" ∧
      peer.phase = .ready
  lateIsNode1 :
    late.worker = workerId "worker-1" ∧
      late.node = nodeId "node-1" ∧
      late.incarnation = incarnation "peer-inc-1"
  onlyExpectedNodeRestart : state.restartCount = 1
  noCohortOwnerBudgetConsumed :
    state.generation.ownerReassignments = 0
  noPartialNodeApply : state.nodeApplyReceipts = []

/--
The exact job-5105811 close/restart race is typed, recoverable, and
non-mutating.  Because the complete state is preserved, there is no second
commit, cohort mutation, partial publication/apply, restart-budget charge, or
generation/token/receipt rollback; node 1 remains in its pre-response state.
-/
theorem exact_job_5105811_generation_3_failure_order_is_safe
    (state : RunState) (late : ContributionEvent)
    (ordering : Job5105811FailureOrdering state late) :
    (transition state (.contribution late)).disposition = .catchUp ∧
      (transition state (.contribution late)).disposition.nextAction =
        .catchUpLatest ∧
      (transition state (.contribution late)).state = state ∧
      (transition state (.contribution late)).state.generation.commit =
        state.generation.commit ∧
      (transition state (.contribution late)).state.generation.cohort =
        state.generation.cohort ∧
      (transition state (.contribution late)).state.mailbox =
        state.mailbox ∧
      (transition state (.contribution late)).state.nodeApplyReceipts =
        [] ∧
      (transition state (.contribution late)).state.restartCount = 1 ∧
      (transition state
        (.contribution late)).state.generation.ownerReassignments = 0 ∧
      (transition state (.contribution late)).state.generation.generation =
        ⟨3⟩ ∧
      (transition state (.contribution late)).state.acceptedTokenClock =
        state.acceptedTokenClock ∧
      (transition state (.contribution late)).state.baseReceipt =
        state.baseReceipt ∧
      findPeer?
        (transition state (.contribution late)).state
        (workerId "worker-1") =
        findPeer? state (workerId "worker-1") := by
  have hsameGeneration :
      late.context.generation = state.generation.generation := by
    calc
      late.context.generation = ⟨3⟩ :=
        ordering.lateGeneration3
      _ = state.generation.generation :=
        ordering.generation3.symm
  have hprefix :
      decideContributionPrefix state late = .reject .catchUp := by
    simp [decideContributionPrefix, ordering.contextAuthority,
      hsameGeneration, ordering.sameAttempt, ordering.sameOwnerEpoch,
      ordering.unseenContribution, ordering.committed,
      ordering.resultAuthorityPresent]
  have hready :
      decideContributionReady state late = .reject .catchUp := by
    unfold decideContributionReady
    rw [hprefix]
  have hdecision :
      decideContribution state late = .reject .catchUp := by
    unfold decideContribution
    rw [hready]
  have hcontribution :
      transitionContribution state late = noChange state .catchUp := by
    unfold transitionContribution
    rw [hdecision]
    rfl
  have hdisposition :
      (transition state (.contribution late)).disposition =
        .catchUp := by
    change
      (enforceNonMutatingDisposition state
        (sealStableAuthority state
          (transitionContribution state late))).disposition =
        .catchUp
    rw [hcontribution]
    rfl
  have hstate :
      (transition state (.contribution late)).state = state :=
    catch_up_noninterference state (.contribution late) hdisposition
  exact
    ⟨hdisposition,
      by simp [hdisposition, Disposition.nextAction],
      hstate,
      congrArg (fun value => value.generation.commit) hstate,
      congrArg (fun value => value.generation.cohort) hstate,
      congrArg RunState.mailbox hstate,
      by simpa [hstate] using ordering.noPartialNodeApply,
      by simpa [hstate] using ordering.onlyExpectedNodeRestart,
      by simpa [hstate] using ordering.noCohortOwnerBudgetConsumed,
      by simpa [hstate] using ordering.generation3,
      congrArg RunState.acceptedTokenClock hstate,
      congrArg RunState.baseReceipt hstate,
      congrArg
        (fun value => findPeer? value (workerId "worker-1")) hstate⟩

def job5105811CanonicalLateIndex : Nat := 32

/--
The full canonical executable scenario binds the concrete job-5105811 event
at the checked position after peer bootstrap, close, commit, service loss, and
node-0 reincarnation.
-/
theorem job_5105811_canonical_trace_binds_exact_late_event :
    restartRejoinScenario.actions[job5105811CanonicalLateIndex]?.map
        (·.event) =
      some (.contribution job5105811LateContribution) := by
  rfl

/--
Generation-4 participation is deliberately conditional.  A caller must
supply finite close/stage deadlines, surviving eligible stable-worker quorum,
the exact-token floor, bounded permitted failures/reassignments, eventual
delivery/processing, and fair scheduling of the enabled contribution.
-/
theorem job_5105811_next_generation_participation_under_declared_assumptions
    (state : RunState) (late : ContributionEvent)
    (_ordering : Job5105811FailureOrdering state late)
    (assumptions : BoundedProgressAssumptions state) :
    ∃ index : Nat, ∃ event : ContributionEvent,
      index < assumptions.deliveryBound ∧
      assumptions.schedule[index]? = some (.contribution event) ∧
      state.generation.generation.value <
        event.context.generation.value ∧
      (transition
        (stateBeforeIndex state assumptions.schedule index)
        (.contribution event)).disposition = .accepted :=
  bounded_next_generation_participation_under_exact_assumptions
    state assumptions

end ResilientProtocol
