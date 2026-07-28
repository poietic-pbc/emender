import ResilientProtocol.Safety

namespace ResilientProtocol

/-!
# Deliberately invalid transition variants

These definitions are mutation tests, not alternate protocol models.  Each
one encodes a representative bug and the adjacent theorem proves that it
violates the corresponding safety obligation.  If the authoritative kernel
is changed to behave like one of these mutants, the Safety/Regression theorem
suite can no longer establish its required conclusion.
-/

def mutantDoubleCommit
    (state : RunState) (replacement : CommitRecord) : RunState :=
  { state with
    generation :=
      { state.generation with commit := some replacement } }

theorem mutation_double_commit_is_rejected
    (state : RunState) (authoritative replacement : CommitRecord)
    (hauthoritative :
      state.generation.commit = some authoritative)
    (hconflict : replacement ≠ authoritative) :
    (mutantDoubleCommit state replacement).generation.commit ≠
      state.generation.commit := by
  simp [mutantDoubleCommit, hauthoritative, hconflict]

def mutantStaleFenceWrite (state : RunState) : StepResult :=
  { state :=
      { state with
        acceptedTokenClock := state.acceptedTokenClock + 1 }
    disposition := .staleFence }

theorem mutation_stale_fence_write_is_rejected
    (state : RunState) :
    (mutantStaleFenceWrite state).state ≠ state := by
  intro heq
  have hclock := congrArg RunState.acceptedTokenClock heq
  simp [mutantStaleFenceWrite] at hclock

def mutantConflictingDuplicateWrite (state : RunState) : StepResult :=
  { state := { state with restartCount := state.restartCount + 1 }
    disposition := .conflictingDuplicate }

theorem mutation_conflicting_duplicate_write_is_rejected
    (state : RunState) :
    (mutantConflictingDuplicateWrite state).state ≠ state := by
  intro heq
  have hrestarts := congrArg RunState.restartCount heq
  simp [mutantConflictingDuplicateWrite] at hrestarts

def mutantMutableClosedCohort
    (state : RunState) (replacement : FrozenCohort) : RunState :=
  { state with
    generation :=
      { state.generation with cohort := some replacement } }

theorem mutation_mutable_closed_cohort_is_rejected
    (state : RunState) (closed replacement : FrozenCohort)
    (hclosed : state.generation.cohort = some closed)
    (hconflict : replacement ≠ closed) :
    (mutantMutableClosedCohort state replacement).generation.cohort ≠
      state.generation.cohort := by
  simp [mutantMutableClosedCohort, hclosed, hconflict]

def CommitBackedPublication (state : RunState) : Prop :=
  state.mailbox.isSome = true →
    state.generation.commit.isSome = true

def mutantPartialPublication
    (state : RunState) (mailbox : MailboxRecord) : RunState :=
  { state with mailbox := some mailbox }

theorem mutation_partial_publication_is_rejected
    (state : RunState) (mailbox : MailboxRecord)
    (hcommit : state.generation.commit = none) :
    ¬ CommitBackedPublication
      (mutantPartialPublication state mailbox) := by
  intro backed
  have := backed (by simp [mutantPartialPublication])
  simp [mutantPartialPublication, hcommit] at this

def AtomicNodeApplyAuthority (state : RunState) : Prop :=
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

def mutantPartialNodeApply
    (state : RunState) (nodeReceipt : NodeApplyReceipt) : RunState :=
  { state with nodeApplyReceipts := [nodeReceipt] }

theorem mutation_partial_node_apply_is_rejected
    (state : RunState) (nodeReceipt : NodeApplyReceipt)
    (hpartial : state.applyReceipts = []) :
    ¬ AtomicNodeApplyAuthority
      (mutantPartialNodeApply state nodeReceipt) := by
  intro hatomic
  have hreceipt :=
    hatomic nodeReceipt (by simp [mutantPartialNodeApply])
  simp [mutantPartialNodeApply, hpartial] at hreceipt

end ResilientProtocol
