import ResilientProtocol.Kernel
import Std.Tactic

namespace ResilientProtocol

/-!
# Machine-checked safety of the executable coordination kernel

Every theorem in this module is stated over `transition`, the same total
function used by executable trace construction and replay.  All safety
definitions are propositions over records and relations; no theorem is
replaced by `invariantHolds` or another runtime Boolean.
-/

/-! ## Structural assumptions -/

/--
The finite, schema-bound hypotheses that may be used by safety theorems.
There is deliberately no delivery, scheduling, or fairness field.
-/
structure WellFormedState (state : RunState) : Prop where
  authorityValid : validAuthority state.authority = true
  policyValid : validPolicy state.policy = true
  authorityPolicyBound :
    authorityMatchesPolicy state.authority state.policy = true
  ownerReplayFinite :
    state.generation.ownerReassignments ≤
      state.policy.maxOwnerReassignments
  quorumPositive : 0 < state.policy.qMin
  tokenFloorPositive : 0 < state.policy.tMin
  cohortHasFloor :
    ∀ cohort,
      state.generation.cohort = some cohort →
      floorSatisfied state = true
  cohortIsAcceptedSet :
    ∀ cohort,
      state.generation.cohort = some cohort →
      cohort.contributions = state.generation.accepted
  cohortTokenExact :
    ∀ cohort,
      state.generation.cohort = some cohort →
      cohort.exactTokens =
        contributionTokens state.generation.accepted
  commitHasCohort :
    ∀ commit,
      state.generation.commit = some commit →
      ∃ cohort, state.generation.cohort = some cohort
  nodeApplyIsAtomic :
    ∀ nodeReceipt ∈ state.nodeApplyReceipts,
      let matching :=
        state.applyReceipts.filter fun receipt =>
          receipt.generation == nodeReceipt.generation &&
          receipt.node == nodeReceipt.node &&
          receipt.worker == nodeReceipt.worker &&
          receipt.peerIncarnation == nodeReceipt.peerIncarnation &&
          receipt.result == nodeReceipt.result &&
          receipt.resultDigest == nodeReceipt.resultDigest
      matching.length = 8 ∧
        uniqueBy (matching.map (·.trainer)) = true

theorem wellFormed_initialState
    (authority : AuthorityIdentity)
    (policy : PolicyConfig)
    (baseGeneration : Generation)
    (baseDigest : Digest)
    (baseReceipt : ReceiptId)
    (baseReceiptDigest : Digest)
    (acceptedTokenClock : Nat)
    (lastResult : Option ResultId)
    (ha : validAuthority authority = true)
    (hp : validPolicy policy = true)
    (hap : authorityMatchesPolicy authority policy = true)
    (hq : 0 < policy.qMin)
    (ht : 0 < policy.tMin) :
    WellFormedState
      (initialState authority policy baseGeneration baseDigest baseReceipt
        baseReceiptDigest acceptedTokenClock lastResult) := by
  exact
    { authorityValid := ha
      policyValid := hp
      authorityPolicyBound := hap
      ownerReplayFinite := by simp [initialState]
      quorumPositive := hq
      tokenFloorPositive := ht
      cohortHasFloor := by simp [initialState]
      cohortIsAcceptedSet := by simp [initialState]
      cohortTokenExact := by simp [initialState]
      commitHasCohort := by simp [initialState]
      nodeApplyIsAtomic := by simp [initialState] }

/-! ## Independent monotone authority namespaces -/

structure MonotoneAuthority
    (before after : RunState) : Prop where
  fence :
    before.authority.fence.value ≤ after.authority.fence.value
  generation :
    before.generation.generation.value ≤ after.generation.generation.value
  acceptedTokens :
    before.acceptedTokenClock ≤ after.acceptedTokenClock

private theorem sealStableAuthority_monotone
    (state : RunState) (result : StepResult) :
    MonotoneAuthority state (sealStableAuthority state result).state := by
  exact
    { fence := by simp [sealStableAuthority]
      generation := by simp [sealStableAuthority]
      acceptedTokens := by simp [sealStableAuthority] }

private theorem sealCommitAuthority_monotone
    (state : RunState) (result : StepResult) :
    MonotoneAuthority state (sealCommitAuthority state result).state := by
  exact
    { fence := by simp [sealCommitAuthority]
      generation := by simp [sealCommitAuthority]
      acceptedTokens := by
        simp [sealCommitAuthority]
        omega }

private theorem sealCloseAuthority_monotone
    (state : RunState) (result : StepResult) :
    MonotoneAuthority state (sealCloseAuthority state result).state := by
  exact
    { fence := by simp [sealCloseAuthority]
      generation := by simp [sealCloseAuthority]
      acceptedTokens := by simp [sealCloseAuthority] }

private theorem sealPublicationAuthority_monotone
    (state : RunState) (result : StepResult) :
    MonotoneAuthority state
      (sealPublicationAuthority state result).state := by
  exact
    { fence := by
        simp [sealPublicationAuthority, sealStableAuthority]
      generation := by
        simp [sealPublicationAuthority, sealStableAuthority]
      acceptedTokens := by
        simp [sealPublicationAuthority, sealStableAuthority] }

private theorem sealOpenAuthority_monotone
    (state : RunState) (result : StepResult) :
    MonotoneAuthority state (sealOpenAuthority state result).state := by
  exact
    { fence := by simp [sealOpenAuthority]
      generation := by
        simp [sealOpenAuthority]
        omega
      acceptedTokens := by simp [sealOpenAuthority] }

private theorem sealClaimAuthority_monotone
    (state : RunState) (result : StepResult) :
    MonotoneAuthority state (sealClaimAuthority state result).state := by
  exact
    { fence := by
        simp [sealClaimAuthority]
        omega
      generation := by
        simp [sealClaimAuthority]
        omega
      acceptedTokens := by
        simp [sealClaimAuthority]
        omega }

private theorem transitionRaw_authority_monotone
    (state : RunState) (event : Event) :
    MonotoneAuthority state (transitionRaw state event).state := by
  cases event with
  | claimFence event =>
      simpa [transitionRaw] using
        sealClaimAuthority_monotone state
          (transitionClaimFence state event)
  | openGeneration event =>
      simpa [transitionRaw] using
        sealOpenAuthority_monotone state (transitionOpen state event)
  | commitGeneration event =>
      simpa [transitionRaw] using
        sealCommitAuthority_monotone state (transitionCommit state event)
  | peerTransition event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state (transitionPeer state event)
  | registerTrainer event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state
          (transitionRegisterTrainer state event)
  | expirePeer event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state (transitionExpirePeer state event)
  | contribution event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state
          (transitionContribution state event)
  | closeGeneration event =>
      simpa [transitionRaw] using
        sealCloseAuthority_monotone state (transitionClose state event)
  | ownerLoss event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state (transitionOwnerLoss state event)
  | publishResult event =>
      simpa [transitionRaw] using
        sealPublicationAuthority_monotone state
          (transitionPublish state event)
  | trainerApply event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state
          (transitionTrainerApply state event)
  | reduceNodeApply event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state
          (transitionReduceNodeApply state event)
  | loss event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state (transitionLoss state event)
  | restartPeer event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state
          (transitionRestartPeer state event)
  | abortGeneration event =>
      simpa [transitionRaw] using
        sealStableAuthority_monotone state (transitionAbort state event)

private theorem enforceNonMutatingDisposition_monotone
    (state : RunState) (result : StepResult)
    (h : MonotoneAuthority state result.state) :
    MonotoneAuthority state
      (enforceNonMutatingDisposition state result).state := by
  cases hd : result.disposition <;>
    simp only [enforceNonMutatingDisposition, hd, noChange]
  all_goals
    first
    | exact h
    | exact
        { fence := Nat.le_refl _
          generation := Nat.le_refl _
          acceptedTokens := Nat.le_refl _ }

/--
Fence, generation, and accepted-token authority are monotone for every event
constructor.  The constructor analysis is over `transitionRaw`, and the
typed-disposition normalization is then proved monotone.  There is no fairness
hypothesis.
-/
theorem transition_authority_monotone
    (state : RunState) (event : Event) :
    MonotoneAuthority state (transition state event).state := by
  apply enforceNonMutatingDisposition_monotone
  exact transitionRaw_authority_monotone state event

theorem transition_fence_monotone
    (state : RunState) (event : Event) :
    state.authority.fence.value ≤
      (transition state event).state.authority.fence.value :=
  (transition_authority_monotone state event).fence

theorem transition_generation_monotone
    (state : RunState) (event : Event) :
    state.generation.generation.value ≤
      (transition state event).state.generation.generation.value :=
  (transition_authority_monotone state event).generation

theorem transition_token_clock_monotone
    (state : RunState) (event : Event) :
    state.acceptedTokenClock ≤
      (transition state event).state.acceptedTokenClock :=
  (transition_authority_monotone state event).acceptedTokens

private theorem commit_decision_admission_internal
    (state : RunState) (event : CommitGenerationEvent)
    (hdecision : decideCommit state event = .commit) :
    ∃ cohort : FrozenCohort,
      state.generation.commit = none ∧
      state.generation.cohort = some cohort ∧
      state.generation.status = .closed ∧
      cohort.cohortDigest = event.cohortDigest ∧
      event.priorReceipt = state.baseReceipt := by
  unfold decideCommit at hdecision
  split at hdecision <;> try { cases hdecision }
  split at hdecision <;> try { cases hdecision }
  split at hdecision <;> try { cases hdecision }
  split at hdecision <;> try { cases hdecision }
  split at hdecision <;> try { cases hdecision }
  split at hdecision <;> try { cases hdecision }
  simp_all

def ReceiptAuthorityStep (before after : RunState) : Prop :=
  (after.baseReceipt = before.baseReceipt ∧
    after.baseReceiptDigest = before.baseReceiptDigest ∧
    after.lastResult = before.lastResult) ∨
  ∃ commit : CommitRecord,
    after.generation.commit = some commit ∧
    commit.priorReceipt = before.baseReceipt ∧
    after.baseReceipt = commit.receipt ∧
    after.baseReceiptDigest = commit.receiptDigest ∧
    after.lastResult = some commit.result

private theorem enforce_preserves_receipt_authority
    (state : RunState) (result : StepResult)
    (hreceipt : result.state.baseReceipt = state.baseReceipt)
    (hdigest :
      result.state.baseReceiptDigest = state.baseReceiptDigest)
    (hresult : result.state.lastResult = state.lastResult) :
    ReceiptAuthorityStep state
      (enforceNonMutatingDisposition state result).state := by
  left
  cases hd : result.disposition <;>
    simp [enforceNonMutatingDisposition, hd, noChange,
      hreceipt, hdigest, hresult]

/--
Receipt authority either remains byte-for-byte unchanged or advances through
an accepted commit whose `priorReceipt` is the preceding authority.  The proof
is a constructor analysis of the same `transition` used by trace replay.
-/
theorem transition_receipt_authority_monotone
    (state : RunState) (event : Event) :
    ReceiptAuthorityStep state (transition state event).state := by
  cases event with
  | claimFence event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealClaimAuthority]
  | openGeneration event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealOpenAuthority]
  | commitGeneration event =>
      cases hdecision : decideCommit state event with
      | reject disposition =>
          left
          cases disposition <;>
            simp [transition, transitionRaw, transitionCommit,
              applyCommitDecision, hdecision, sealCommitAuthority,
              enforceNonMutatingDisposition, noChange]
      | commit =>
          obtain ⟨cohort, hadmission⟩ :=
            commit_decision_admission_internal
              state event hdecision
          right
          refine ⟨commitRecord state event cohort, ?_, ?_, ?_, ?_, ?_⟩
          · simp [transition, transitionRaw, transitionCommit,
              applyCommitDecision, hdecision, sealCommitAuthority,
              enforceNonMutatingDisposition, acceptedState,
              hadmission.1, hadmission.2.1, commitRecord]
          · simpa [commitRecord] using hadmission.2.2.2.2
          · simp [transition, transitionRaw, transitionCommit,
              applyCommitDecision, hdecision, sealCommitAuthority,
              enforceNonMutatingDisposition, acceptedState,
              hadmission.2.1, commitRecord]
          · simp [transition, transitionRaw, transitionCommit,
              applyCommitDecision, hdecision, sealCommitAuthority,
              enforceNonMutatingDisposition, acceptedState,
              hadmission.2.1, commitRecord]
          · simp [transition, transitionRaw, transitionCommit,
              applyCommitDecision, hdecision, sealCommitAuthority,
              enforceNonMutatingDisposition, acceptedState,
              hadmission.2.1, commitRecord]
  | publishResult event =>
      apply enforce_preserves_receipt_authority
      all_goals
        simp [transitionRaw, sealPublicationAuthority,
          sealStableAuthority]
  | closeGeneration event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealCloseAuthority]
  | peerTransition event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | registerTrainer event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | expirePeer event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | contribution event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | ownerLoss event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | trainerApply event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | reduceNodeApply event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | loss event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | restartPeer event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]
  | abortGeneration event =>
      apply enforce_preserves_receipt_authority
      all_goals simp [transitionRaw, sealStableAuthority]

/-! ## Typed rejection and recovery are exactly non-mutating -/

/--
Every outcome outside the three state-changing disposition classes is the
literal input state.  This theorem is about `transition` itself and is the
common proof used by the constructor-specific stale, duplicate, close-race,
and recovery corollaries below.
-/
theorem transition_nonmutating_of_disposition
    (state : RunState) (event : Event) (disposition : Disposition)
    (hd : (transition state event).disposition = disposition)
    (haccepted : disposition ≠ .accepted)
    (hinsufficient : disposition ≠ .insufficientCohort)
    (haborted : disposition ≠ .aborted) :
    (transition state event).state = state := by
  unfold transition at hd ⊢
  unfold enforceNonMutatingDisposition at hd ⊢
  split at * <;> simp_all [noChange]

theorem identical_duplicate_idempotent
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .identicalDuplicate) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .identicalDuplicate h (by simp) (by simp) (by simp)

theorem conflicting_duplicate_rejected
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .conflictingDuplicate) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .conflictingDuplicate h (by simp) (by simp) (by simp)

theorem stale_fence_noninterference
    (state : RunState) (event : Event)
    (h : (transition state event).disposition = .staleFence) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .staleFence h (by simp) (by simp) (by simp)

theorem stale_incarnation_noninterference
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .staleIncarnation) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .staleIncarnation h (by simp) (by simp) (by simp)

theorem corrupt_input_noninterference
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .corruptNonfinite) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .corruptNonfinite h (by simp) (by simp) (by simp)

theorem deferred_noninterference
    (state : RunState) (event : Event)
    (h : (transition state event).disposition = .deferred) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .deferred h (by simp) (by simp) (by simp)

theorem generation_closed_noninterference
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .generationClosed) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .generationClosed h (by simp) (by simp) (by simp)

theorem late_event_noninterference
    (state : RunState) (event : Event)
    (h : (transition state event).disposition = .late) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .late h (by simp) (by simp) (by simp)

theorem catch_up_noninterference
    (state : RunState) (event : Event)
    (h : (transition state event).disposition = .catchUp) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .catchUp h (by simp) (by simp) (by simp)

theorem retry_next_generation_noninterference
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .retryNextGeneration) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .retryNextGeneration h (by simp) (by simp) (by simp)

theorem unknown_identity_noninterference
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .unknownIdentity) :
    (transition state event).state = state :=
  transition_nonmutating_of_disposition state event
    .unknownIdentity h (by simp) (by simp) (by simp)

theorem closed_event_has_typed_recovery
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .generationClosed) :
    (transition state event).state = state ∧
      (transition state event).disposition.nextAction =
        .catchUpLatest := by
  constructor
  · exact generation_closed_noninterference state event h
  · simp [h, Disposition.nextAction]

theorem late_event_has_typed_recovery
    (state : RunState) (event : Event)
    (h : (transition state event).disposition = .late) :
    (transition state event).state = state ∧
      (transition state event).disposition.nextAction =
        .retryNextGeneration := by
  constructor
  · exact late_event_noninterference state event h
  · simp [h, Disposition.nextAction]

theorem catch_up_has_typed_recovery
    (state : RunState) (event : Event)
    (h : (transition state event).disposition = .catchUp) :
    (transition state event).state = state ∧
      (transition state event).disposition.nextAction =
        .catchUpLatest := by
  constructor
  · exact catch_up_noninterference state event h
  · simp [h, Disposition.nextAction]

theorem retry_has_typed_recovery
    (state : RunState) (event : Event)
    (h :
      (transition state event).disposition =
        .retryNextGeneration) :
    (transition state event).state = state ∧
      (transition state event).disposition.nextAction =
        .retryNextGeneration := by
  constructor
  · exact retry_next_generation_noninterference state event h
  · simp [h, Disposition.nextAction]

theorem historical_contribution_catches_up_without_resurrection
    (state : RunState) (event : ContributionEvent)
    (hauthority :
      contextAuthorityDisposition? state event.context = none)
    (hold :
      event.context.generation.value <
        state.generation.generation.value)
    (hresult : state.lastResult.isSome = true) :
    (transition state (.contribution event)).disposition = .catchUp ∧
      (transition state (.contribution event)).state = state := by
  have hprefix :
      decideContributionPrefix state event =
        .reject .catchUp := by
    simp [decideContributionPrefix, hauthority, hold, hresult]
  have hready :
      decideContributionReady state event =
        .reject .catchUp := by
    unfold decideContributionReady
    rw [hprefix]
  have hdecision :
      decideContribution state event =
        .reject .catchUp := by
    unfold decideContribution
    rw [hready]
  have hstep :
      transitionContribution state event =
        noChange state .catchUp := by
    unfold transitionContribution
    rw [hdecision]
    rfl
  simp [transition, transitionRaw, hstep, sealStableAuthority,
    enforceNonMutatingDisposition, noChange]

theorem restart_recovery_cannot_roll_back_generation
    (state : RunState) (event : RestartPeerEvent) :
    state.generation.generation.value ≤
      (transition state (.restartPeer event)).state.generation.generation.value :=
  transition_generation_monotone state (.restartPeer event)

theorem fence_recovery_cannot_roll_back_generation_or_tokens
    (state : RunState) (event : ClaimFenceEvent) :
    state.generation.generation.value ≤
        (transition state (.claimFence event)).state.generation.generation.value ∧
      state.acceptedTokenClock ≤
        (transition state (.claimFence event)).state.acceptedTokenClock :=
  ⟨transition_generation_monotone state (.claimFence event),
    transition_token_clock_monotone state (.claimFence event)⟩

structure ConstructorSafety
    (before : RunState) (event : Event) : Prop where
  authority :
    MonotoneAuthority before (transition before event).state
  receipt :
    ReceiptAuthorityStep before (transition before event).state
  rejectedExact :
    ∀ disposition,
      (transition before event).disposition = disposition →
      disposition ≠ .accepted →
      disposition ≠ .insufficientCohort →
      disposition ≠ .aborted →
      (transition before event).state = before

/--
The single theorem quantified over every `Event` constructor.  Its only
hypothesis is structural well-formedness; it contains no delivery, failure, or
scheduler fairness premise.
-/
theorem transition_safety_all_constructors
    (state : RunState) (event : Event)
    (_hwf : WellFormedState state) :
    ConstructorSafety state event := by
  exact
    { authority := transition_authority_monotone state event
      receipt := transition_receipt_authority_monotone state event
      rejectedExact := by
        intro disposition hd ha hi hb
        exact
          transition_nonmutating_of_disposition state event disposition
            hd ha hi hb }

/-! ## Immutable authority inside one fence/generation namespace -/

/--
Events that operate inside the current allocation-fence/generation frame.
`claimFence` and `openGeneration` are excluded because they are precisely the
two constructors that may enter a newer authority namespace.
-/
inductive FrameEvent : Event → Prop where
  | peer (event : PeerTransitionEvent) :
      FrameEvent (.peerTransition event)
  | registerTrainer (event : RegisterTrainerEvent) :
      FrameEvent (.registerTrainer event)
  | expirePeer (event : ExpirePeerEvent) :
      FrameEvent (.expirePeer event)
  | contribution (event : ContributionEvent) :
      FrameEvent (.contribution event)
  | close (event : CloseGenerationEvent) :
      FrameEvent (.closeGeneration event)
  | ownerLoss (event : OwnerLossEvent) :
      FrameEvent (.ownerLoss event)
  | commit (event : CommitGenerationEvent) :
      FrameEvent (.commitGeneration event)
  | publish (event : PublishResultEvent) :
      FrameEvent (.publishResult event)
  | trainerApply (event : TrainerApplyEvent) :
      FrameEvent (.trainerApply event)
  | reduceNodeApply (event : ReduceNodeApplyEvent) :
      FrameEvent (.reduceNodeApply event)
  | loss (event : LossEvent) :
      FrameEvent (.loss event)
  | restartPeer (event : RestartPeerEvent) :
      FrameEvent (.restartPeer event)
  | abort (event : AbortGenerationEvent) :
      FrameEvent (.abortGeneration event)

private theorem enforce_preserves_existing_commit
    (state : RunState) (result : StepResult) (commit : CommitRecord)
    (hbefore : state.generation.commit = some commit)
    (hresult :
      result.state.generation.commit =
        some commit) :
    (enforceNonMutatingDisposition state result).state.generation.commit =
      some commit := by
  cases hd : result.disposition <;>
    simp [enforceNonMutatingDisposition, hd, noChange, hbefore, hresult]

private theorem enforce_preserves_existing_cohort
    (state : RunState) (result : StepResult) (cohort : FrozenCohort)
    (hbefore : state.generation.cohort = some cohort)
    (hresult :
      result.state.generation.cohort =
        some cohort) :
    (enforceNonMutatingDisposition state result).state.generation.cohort =
      some cohort := by
  cases hd : result.disposition <;>
    simp [enforceNonMutatingDisposition, hd, noChange, hbefore, hresult]

private theorem enforce_preserves_mailbox
    (state : RunState) (result : StepResult)
    (hresult : result.state.mailbox = state.mailbox) :
    (enforceNonMutatingDisposition state result).state.mailbox =
      state.mailbox := by
  cases hd : result.disposition <;>
    simp [enforceNonMutatingDisposition, hd, noChange, hresult]

private theorem enforce_preserves_commit
    (state : RunState) (result : StepResult)
    (hresult :
      result.state.generation.commit =
        state.generation.commit) :
    (enforceNonMutatingDisposition state result).state.generation.commit =
      state.generation.commit := by
  cases hd : result.disposition <;>
    simp [enforceNonMutatingDisposition, hd, noChange, hresult]

/--
Once a commit exists, no event in that fence/generation frame can replace it.
Thus an authoritative `(result, receipt)` pair is unique in the executable
transition history for that namespace.
-/
theorem existing_commit_is_unique_in_frame
    (state : RunState) (event : Event) (commit : CommitRecord)
    (hevent : FrameEvent event)
    (hcommit : state.generation.commit = some commit) :
    (transition state event).state.generation.commit =
      some commit := by
  cases hevent with
  | peer event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | registerTrainer event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | expirePeer event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | contribution event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | close event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealCloseAuthority, hcommit]
  | ownerLoss event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | commit event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealCommitAuthority, hcommit]
  | publish event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealPublicationAuthority,
        sealStableAuthority, hcommit]
  | trainerApply event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | reduceNodeApply event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | loss event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | restartPeer event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]
  | abort event =>
      apply enforce_preserves_existing_commit state _ commit hcommit
      simp [transitionRaw, sealStableAuthority, hcommit]

/--
After deterministic close has installed a cohort, no event in that
fence/generation frame can alter its membership, order, exact-token sum, close
evidence, or digest: the complete `FrozenCohort` value is preserved.
-/
theorem accepted_cohort_is_immutable_in_frame
    (state : RunState) (event : Event) (cohort : FrozenCohort)
    (hevent : FrameEvent event)
    (hcohort : state.generation.cohort = some cohort) :
    (transition state event).state.generation.cohort =
      some cohort := by
  cases hevent with
  | peer event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | registerTrainer event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | expirePeer event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | contribution event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | close event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealCloseAuthority, hcohort]
  | ownerLoss event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | commit event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealCommitAuthority, hcohort]
  | publish event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealPublicationAuthority,
        sealStableAuthority, hcohort]
  | trainerApply event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | reduceNodeApply event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | loss event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | restartPeer event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]
  | abort event =>
      apply enforce_preserves_existing_cohort state _ cohort hcohort
      simp [transitionRaw, sealStableAuthority, hcohort]

def executeEvents : RunState → List Event → RunState
  | state, [] => state
  | state, event :: tail =>
      executeEvents (transition state event).state tail

theorem existing_commit_is_unique_over_frame_trace
    (state : RunState) (events : List Event) (commit : CommitRecord)
    (hcommit : state.generation.commit = some commit)
    (hevents : ∀ event ∈ events, FrameEvent event) :
    (executeEvents state events).generation.commit = some commit := by
  induction events generalizing state with
  | nil => simpa [executeEvents] using hcommit
  | cons event tail ih =>
      simp only [executeEvents]
      apply ih
      · exact
          existing_commit_is_unique_in_frame state event commit
            (hevents event (by simp)) hcommit
      · intro next hnext
        exact hevents next (by simp [hnext])

theorem accepted_cohort_is_immutable_over_frame_trace
    (state : RunState) (events : List Event) (cohort : FrozenCohort)
    (hcohort : state.generation.cohort = some cohort)
    (hevents : ∀ event ∈ events, FrameEvent event) :
    (executeEvents state events).generation.cohort = some cohort := by
  induction events generalizing state with
  | nil => simpa [executeEvents] using hcohort
  | cons event tail ih =>
      simp only [executeEvents]
      apply ih
      · exact
          accepted_cohort_is_immutable_in_frame state event cohort
            (hevents event (by simp)) hcohort
      · intro next hnext
        exact hevents next (by simp [hnext])

/--
Owner loss/replay, peer recovery, and explicit abort can change bounded
lifecycle state, but none can create a commit or mailbox publication.  Thus
they cannot expose a partial result.
-/
theorem owner_loss_recovery_abort_cannot_publish_partial_result
    (state : RunState)
    (ownerLoss : OwnerLossEvent)
    (restart : RestartPeerEvent)
    (abort : AbortGenerationEvent) :
    (transition state (.ownerLoss ownerLoss)).state.mailbox =
        state.mailbox ∧
    (transition state (.ownerLoss ownerLoss)).state.generation.commit =
        state.generation.commit ∧
    (transition state (.restartPeer restart)).state.mailbox =
        state.mailbox ∧
    (transition state (.restartPeer restart)).state.generation.commit =
        state.generation.commit ∧
    (transition state (.abortGeneration abort)).state.mailbox =
        state.mailbox ∧
    (transition state (.abortGeneration abort)).state.generation.commit =
        state.generation.commit := by
  constructor
  · apply enforce_preserves_mailbox
    simp [transitionRaw, sealStableAuthority]
  constructor
  · apply enforce_preserves_commit
    simp [transitionRaw, sealStableAuthority]
  constructor
  · apply enforce_preserves_mailbox
    simp [transitionRaw, sealStableAuthority]
  constructor
  · apply enforce_preserves_commit
    simp [transitionRaw, sealStableAuthority]
  constructor
  · apply enforce_preserves_mailbox
    simp [transitionRaw, sealStableAuthority]
  · apply enforce_preserves_commit
    simp [transitionRaw, sealStableAuthority]

/-! ## Admission is restricted to the deterministic leased-READY snapshot -/

def EligibleLeasedReady
    (state : RunState) (event : ContributionEvent) : Prop :=
  ∃ peer : PeerRecord,
    findPeer? state event.worker = some peer ∧
    peer.node = event.node ∧
    peer.incarnation = event.incarnation ∧
    leasedReadyAdmissionGate state event peer = true

theorem ready_decision_requires_eligible_leased_ready
    (state : RunState) (event : ContributionEvent)
    (peer : PeerRecord)
    (hready :
      decideContributionReady state event = .ready peer) :
    EligibleLeasedReady state event := by
  unfold decideContributionReady at hready
  split at hready <;> try { cases hready }
  split at hready <;> try { cases hready }
  split at hready <;> try { cases hready }
  split at hready <;> try { cases hready }
  split at hready <;> try { cases hready }
  split at hready <;> try { cases hready }
  cases hready
  simp_all [EligibleLeasedReady]

theorem eligible_leased_ready_fields
    (state : RunState) (event : ContributionEvent)
    (heligible : EligibleLeasedReady state event) :
    ∃ peer : PeerRecord,
      findPeer? state event.worker = some peer ∧
      peer.node = event.node ∧
      peer.incarnation = event.incarnation ∧
      peer.phase = .ready ∧
      event.observedAt.value < peer.leaseUntil.value ∧
      isReadySnapshotMember state.generation event.worker
        event.node event.incarnation = true := by
  rcases heligible with
    ⟨peer, hfind, hnode, hincarnation, hgate⟩
  refine ⟨peer, hfind, hnode, hincarnation, ?_⟩
  simp only [leasedReadyAdmissionGate, Bool.and_eq_true] at hgate
  rcases hgate with ⟨⟨hphase, hlease⟩, hsnapshot⟩
  have hphaseEq : peer.phase = .ready := by
    cases hp : peer.phase <;> simp_all
  exact ⟨hphaseEq, of_decide_eq_true hlease, hsnapshot⟩

theorem decide_contribution_admit_requires_eligible_leased_ready
    (state : RunState) (event : ContributionEvent)
    (hdecision :
      decideContribution state event = .acceptContribution) :
    EligibleLeasedReady state event := by
  cases hready : decideContributionReady state event with
  | reject disposition =>
      unfold decideContribution at hdecision
      rw [hready] at hdecision
      cases hdecision
  | ready peer =>
      exact
        ready_decision_requires_eligible_leased_ready
          state event peer hready

theorem accepted_contribution_requires_eligible_leased_ready
    (state : RunState) (event : ContributionEvent)
    (haccepted :
      (transition state (.contribution event)).disposition =
        .accepted) :
    EligibleLeasedReady state event := by
  cases hdecision : decideContribution state event with
  | reject disposition =>
      cases disposition <;>
      simp [transition, transitionRaw, enforceNonMutatingDisposition,
        transitionContribution, applyContributionDecision, hdecision,
        sealStableAuthority, noChange] at haccepted
  | acceptContribution =>
      exact
        decide_contribution_admit_requires_eligible_leased_ready
          state event hdecision

theorem admitted_contribution_transition_is_accepted
    (state : RunState) (event : ContributionEvent)
    (hdecision :
      decideContribution state event = .acceptContribution) :
    (transition state (.contribution event)).disposition =
      .accepted := by
  simp [transition, transitionRaw, transitionContribution,
    applyContributionDecision, hdecision, sealStableAuthority,
    enforceNonMutatingDisposition, acceptedState]

theorem admitted_contribution_record_is_event_bound
    (state : RunState) (event : ContributionEvent) :
    (applyContributionDecision state event
      .acceptContribution).state.generation.accepted =
      insertContribution (contributionRecord event)
        state.generation.accepted := by
  rfl

/-! ## Commit is possible only for the one frozen admissible cohort -/

def CommitAdmission
    (state : RunState) (event : CommitGenerationEvent)
    (cohort : FrozenCohort) : Prop :=
  state.generation.commit = none ∧
  state.generation.cohort = some cohort ∧
  state.generation.status = .closed ∧
  cohort.cohortDigest = event.cohortDigest ∧
  event.priorReceipt = state.baseReceipt

theorem commit_decision_requires_declared_closed_cohort
    (state : RunState) (event : CommitGenerationEvent)
    (hdecision : decideCommit state event = .commit) :
    ∃ cohort : FrozenCohort,
      CommitAdmission state event cohort := by
  simpa [CommitAdmission] using
    commit_decision_admission_internal state event hdecision

theorem accepted_commit_uses_declared_admissible_quorum
    (state : RunState) (event : CommitGenerationEvent)
    (hwf : WellFormedState state)
    (haccepted :
      (transition state (.commitGeneration event)).disposition =
        .accepted) :
    ∃ cohort : FrozenCohort,
      CommitAdmission state event cohort ∧
      floorSatisfied state = true ∧
      cohort.contributions = state.generation.accepted ∧
      cohort.exactTokens =
        contributionTokens state.generation.accepted ∧
      (transition state (.commitGeneration event)).state.generation.commit =
        some (commitRecord state event cohort) ∧
      (transition state (.commitGeneration event)).state.acceptedTokenClock =
        state.acceptedTokenClock + cohort.exactTokens ∧
      (transition state (.commitGeneration event)).state.baseReceipt =
        event.receipt ∧
      (transition state (.commitGeneration event)).state.lastResult =
        some event.result := by
  cases hdecision : decideCommit state event with
  | reject disposition =>
      cases disposition <;>
        simp [transition, transitionRaw, transitionCommit,
          applyCommitDecision, hdecision, sealCommitAuthority,
          enforceNonMutatingDisposition, noChange] at haccepted
  | commit =>
      obtain ⟨cohort, hadmission⟩ :=
        commit_decision_requires_declared_closed_cohort
          state event hdecision
      refine ⟨cohort, hadmission, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · exact hwf.cohortHasFloor cohort hadmission.2.1
      · exact hwf.cohortIsAcceptedSet cohort hadmission.2.1
      · exact hwf.cohortTokenExact cohort hadmission.2.1
      · simp [transition, transitionRaw, transitionCommit,
          applyCommitDecision, hdecision, sealCommitAuthority,
          enforceNonMutatingDisposition, acceptedState,
          hadmission.1, hadmission.2.1, commitRecord]
      · simp [transition, transitionRaw, transitionCommit,
          applyCommitDecision, hdecision, sealCommitAuthority,
          enforceNonMutatingDisposition, acceptedState,
          hadmission.2.1, commitRecord]
      · simp [transition, transitionRaw, transitionCommit,
          applyCommitDecision, hdecision, sealCommitAuthority,
          enforceNonMutatingDisposition, acceptedState,
          hadmission.2.1, commitRecord]
      · simp [transition, transitionRaw, transitionCommit,
          applyCommitDecision, hdecision, sealCommitAuthority,
          enforceNonMutatingDisposition, acceptedState,
          hadmission.2.1, commitRecord]

/-! ## Eight distinct trainer receipts form one atomic node authority -/

def NodeApplyAdmission
    (state : RunState) (event : ReduceNodeApplyEvent) : Prop :=
  ∃ peer : PeerRecord,
    findPeer? state event.worker = some peer ∧
    nodeApplyAdmissionGate state event peer = true

theorem reduce_decision_requires_all_eight_distinct_receipts
    (state : RunState) (event : ReduceNodeApplyEvent)
    (hdecision : decideNodeApply state event = .reduce) :
    NodeApplyAdmission state event := by
  grind (splits := 30)
    [decideNodeApply, NodeApplyAdmission]

theorem node_apply_admission_has_exactly_eight_distinct_receipts
    (state : RunState) (event : ReduceNodeApplyEvent)
    (hadmission : NodeApplyAdmission state event) :
    (matchingApplyReceipts state event).length = 8 ∧
      uniqueBy
        ((matchingApplyReceipts state event).map (·.trainer)) =
        true := by
  rcases hadmission with ⟨peer, _, hgate⟩
  simp only [nodeApplyAdmissionGate, Bool.and_eq_true] at hgate
  rcases hgate with ⟨⟨⟨_, hlength⟩, hunique⟩, _⟩
  exact ⟨of_decide_eq_true hlength, hunique⟩

theorem accepted_node_apply_requires_all_eight_distinct_receipts
    (state : RunState) (event : ReduceNodeApplyEvent)
    (haccepted :
      (transition state (.reduceNodeApply event)).disposition =
        .accepted) :
    (matchingApplyReceipts state event).length = 8 ∧
      uniqueBy
        ((matchingApplyReceipts state event).map (·.trainer)) =
        true := by
  cases hdecision : decideNodeApply state event with
  | reject disposition =>
      cases disposition <;>
        simp [transition, transitionRaw, transitionReduceNodeApply,
          applyNodeApplyDecision, hdecision, sealStableAuthority,
          enforceNonMutatingDisposition, noChange] at haccepted
  | reduce =>
      exact
        node_apply_admission_has_exactly_eight_distinct_receipts
          state event
            (reduce_decision_requires_all_eight_distinct_receipts
              state event hdecision)

theorem partial_or_timed_out_apply_never_becomes_node_authority
    (state : RunState) (event : ReduceNodeApplyEvent)
    (hdeferred :
      (transition state (.reduceNodeApply event)).disposition =
        .deferred) :
    (transition state (.reduceNodeApply event)).state = state ∧
      (transition state (.reduceNodeApply event)).state.nodeApplyReceipts =
        state.nodeApplyReceipts := by
  have hstate :=
    deferred_noninterference state (.reduceNodeApply event)
      hdeferred
  exact ⟨hstate, congrArg RunState.nodeApplyReceipts hstate⟩

theorem next_ready_gate_requires_current_node_authority
    (state : RunState) (event : PeerTransitionEvent)
    (peer : PeerRecord) (commit : CommitRecord)
    (hcommit : state.generation.commit = some commit)
    (hready : event.toPhase = .ready)
    (hgate :
      peerReadyTransitionAllowed state event peer = true) :
    peerHasNodeApplyAuthority
      state peer state.generation.generation = true := by
  simp [peerReadyTransitionAllowed, hcommit, hready] at hgate
  exact hgate.2

theorem next_ready_gate_requires_eight_registered_trainers
    (state : RunState) (event : PeerTransitionEvent)
    (peer : PeerRecord)
    (hready : event.toPhase = .ready)
    (hgate :
      peerReadyTransitionAllowed state event peer = true) :
    peer.trainers.length = 8 ∧
      uniqueBy (peer.trainers.map (·.trainer)) = true := by
  simp [peerReadyTransitionAllowed, hready] at hgate
  cases hcommit : state.generation.commit <;>
    simp [hcommit] at hgate
  · exact ⟨hgate.1.2, hgate.2⟩
  · exact ⟨hgate.1.1.2, hgate.1.2⟩

theorem node_ready_apply_authority_is_all_eight_atomic
    (state : RunState) (event : PeerTransitionEvent)
    (peer : PeerRecord) (commit : CommitRecord)
    (hwf : WellFormedState state)
    (hcommit : state.generation.commit = some commit)
    (hready : event.toPhase = .ready)
    (hgate :
      peerReadyTransitionAllowed state event peer = true) :
    peer.trainers.length = 8 ∧
      uniqueBy (peer.trainers.map (·.trainer)) = true ∧
      peerHasNodeApplyAuthority
        state peer state.generation.generation = true ∧
      (∀ nodeReceipt ∈ state.nodeApplyReceipts,
        let matching :=
          state.applyReceipts.filter fun receipt =>
            receipt.generation == nodeReceipt.generation &&
            receipt.node == nodeReceipt.node &&
            receipt.worker == nodeReceipt.worker &&
            receipt.peerIncarnation == nodeReceipt.peerIncarnation &&
            receipt.result == nodeReceipt.result &&
            receipt.resultDigest == nodeReceipt.resultDigest
        matching.length = 8 ∧
          uniqueBy (matching.map (·.trainer)) = true) := by
  have htrainers :=
    next_ready_gate_requires_eight_registered_trainers
      state event peer hready hgate
  exact
    ⟨htrainers.1, htrainers.2,
      next_ready_gate_requires_current_node_authority
        state event peer commit hcommit hready hgate,
      hwf.nodeApplyIsAtomic⟩

end ResilientProtocol
