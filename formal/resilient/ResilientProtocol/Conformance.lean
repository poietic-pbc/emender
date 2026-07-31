import ResilientProtocol.Trace

namespace ResilientProtocol

open Lean

/-!
The production C ABI intentionally carries a smaller, opaque identity surface
than the proof model.  This view is the lossless intersection used by the
native differential adapter.  It contains every coordination identity owned by
the compiled service, but none of the physical transport, tensor, timer, or
process effects that remain outside the pure kernel.

The view is derived only after calling `transition`; it is not another
transition system.
-/

def nativeConformanceViewVersion : String :=
  "emender-native-lean-authority-view-v1"

structure ConformanceMemberView where
  worker : String
  node : String
  incarnation : String
  lifecycle : String
  syncedGeneration : Nat
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure ConformanceCohortMemberView where
  worker : String
  node : String
  incarnation : String
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure ConformanceContributionView where
  worker : String
  node : String
  incarnation : String
  sequence : Nat
  exactTokens : Nat
  payloadDigest : String
  receiptDigest : String
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure ConformanceNodeApplyView where
  generation : Nat
  worker : String
  node : String
  incarnation : String
  resultDigest : String
  receiptDigest : String
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure ConformanceStateView where
  viewVersion : String
  traceSchema : String
  traceSchemaDigest : String
  toolchain : String
  sourceSchema : String
  sourceDigest : String
  run : String
  allocation : String
  fence : Nat
  policyName : String
  policySchema : String
  policyDigest : String
  layoutDigest : String
  codeDigest : String
  generation : Nat
  attempt : Nat
  generationStatus : String
  ownerEpoch : Nat
  ownerReassignments : Nat
  acceptedTokenClock : Nat
  acceptedCount : Nat
  acceptedTokens : Nat
  cohortDigest : Option String
  authorityReceipt : String
  authorityReceiptDigest : String
  result : Option String
  resultDigest : Option String
  members : List ConformanceMemberView
  cohort : List ConformanceCohortMemberView
  contributions : List ConformanceContributionView
  resultReceiptWorkers : List String
  nodeApplies : List ConformanceNodeApplyView
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

def conformanceMemberKey (value : ConformanceMemberView) : String :=
  value.worker ++ "\u001f" ++ value.node ++ "\u001f" ++ value.incarnation

def insertConformanceMember
    (value : ConformanceMemberView) :
    List ConformanceMemberView → List ConformanceMemberView
  | [] => [value]
  | head :: tail =>
      if conformanceMemberKey value < conformanceMemberKey head then
        value :: head :: tail
      else
        head :: insertConformanceMember value tail

def conformanceCohortKey (value : ConformanceCohortMemberView) : String :=
  value.worker ++ "\u001f" ++ value.node ++ "\u001f" ++ value.incarnation

def insertConformanceCohort
    (value : ConformanceCohortMemberView) :
    List ConformanceCohortMemberView → List ConformanceCohortMemberView
  | [] => [value]
  | head :: tail =>
      if conformanceCohortKey value < conformanceCohortKey head then
        value :: head :: tail
      else
        head :: insertConformanceCohort value tail

def conformanceContributionKey
    (value : ConformanceContributionView) : String :=
  value.worker ++ "\u001f" ++ value.node ++ "\u001f" ++ value.incarnation ++
    "\u001f" ++ toString value.sequence

def insertConformanceContribution
    (value : ConformanceContributionView) :
    List ConformanceContributionView → List ConformanceContributionView
  | [] => [value]
  | head :: tail =>
      if conformanceContributionKey value <
          conformanceContributionKey head then
        value :: head :: tail
      else
        head :: insertConformanceContribution value tail

def conformanceNodeApplyKey (value : ConformanceNodeApplyView) : String :=
  toString value.generation ++ "\u001f" ++ value.worker ++ "\u001f" ++
    value.node ++ "\u001f" ++ value.incarnation

def insertConformanceNodeApply
    (value : ConformanceNodeApplyView) :
    List ConformanceNodeApplyView → List ConformanceNodeApplyView
  | [] => [value]
  | head :: tail =>
      if conformanceNodeApplyKey value < conformanceNodeApplyKey head then
        value :: head :: tail
      else
        head :: insertConformanceNodeApply value tail

def conformanceLifecycle : PeerPhase → String
  | .ready => "leased_ready"
  | .expire => "expired"
  | .discover | .boot | .sync | .drain => "recovering"

def conformanceStateView (state : RunState) : ConformanceStateView :=
  let members :=
    state.peers.foldl
      (fun values peer =>
        let synchronizedGeneration :=
          match state.nodeApplyReceipts.find? (fun receipt =>
              receipt.node == peer.node &&
              receipt.peerIncarnation == peer.incarnation) with
          | none => peer.syncedGeneration.value
          | some receipt =>
              max peer.syncedGeneration.value
                (receipt.generation.value + 1)
        insertConformanceMember
          { worker := peer.worker.value
            node := peer.node.value
            incarnation := peer.incarnation.value
            lifecycle := conformanceLifecycle peer.phase
            syncedGeneration := synchronizedGeneration }
          values)
      []
  let cohort :=
    state.generation.readySnapshot.foldl
      (fun values member =>
        insertConformanceCohort
          { worker := member.worker.value
            node := member.node.value
            incarnation := member.incarnation.value }
          values)
      []
  let contributions :=
    state.generation.accepted.foldl
      (fun values contribution =>
        insertConformanceContribution
          { worker := contribution.key.worker.value
            node := contribution.node.value
            incarnation := contribution.key.incarnation.value
            sequence := contribution.key.sequence.value
            exactTokens := contribution.exactTokens
            payloadDigest := contribution.payloadDigest.value
            receiptDigest := contribution.receiptDigest.value }
          values)
      []
  let resultReceiptWorkers :=
    match state.generation.commit with
    | none => []
    | some _ => contributions.map (·.worker)
  let nodeApplies :=
    state.nodeApplyReceipts.foldl
      (fun values receipt =>
        insertConformanceNodeApply
          { generation := receipt.generation.value
            worker := receipt.worker.value
            node := receipt.node.value
            incarnation := receipt.peerIncarnation.value
            resultDigest := receipt.resultDigest.value
            receiptDigest := receipt.receiptDigest.value }
          values)
      []
  let result :=
    match state.generation.commit with
    | some commit => some commit.result.value
    | none => state.lastResult.map (·.value)
  let resultDigest :=
    match state.generation.commit with
    | some commit => some commit.resultDigest.value
    | none =>
        match state.lastResult with
        | none => none
        | some _ => some state.generation.baseDigest.value
  { viewVersion := nativeConformanceViewVersion
    traceSchema := state.authority.traceSchema
    traceSchemaDigest := state.authority.traceSchemaDigest.value
    toolchain := state.authority.toolchain
    sourceSchema := state.authority.sourceSchema
    sourceDigest := state.authority.sourceDigest.value
    run := state.authority.run.value
    allocation := state.authority.allocation.value
    fence := state.authority.fence.value
    policyName := state.authority.policyName
    policySchema := state.authority.policySchema
    policyDigest := state.authority.policyDigest.value
    layoutDigest := state.authority.layoutDigest.value
    codeDigest := state.authority.codeDigest.value
    generation := state.generation.generation.value
    attempt := state.generation.attempt.value
    generationStatus := generationStatusLabel state.generation.status
    ownerEpoch := state.generation.ownerEpoch.value
    ownerReassignments := state.generation.ownerReassignments
    acceptedTokenClock := state.acceptedTokenClock
    acceptedCount := state.generation.accepted.length
    acceptedTokens := contributionTokens state.generation.accepted
    cohortDigest := state.generation.cohort.map (·.cohortDigest.value)
    authorityReceipt := state.baseReceipt.value
    authorityReceiptDigest := state.baseReceiptDigest.value
    result
    resultDigest
    members
    cohort
    contributions
    resultReceiptWorkers
    nodeApplies }

def conformanceStateDigest (state : RunState) : Digest :=
  coordinationDigest nativeConformanceViewVersion
    (toJson (conformanceStateView state)).compress

structure ConformanceStepOracle where
  eventId : String
  eventIndex : Nat
  disposition : String
  state : ConformanceStateView
  stateDigest : String
  leanStateDigest : String
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure ConformanceOracleDocument where
  schemaVersion : String
  schemaDigest : String
  viewVersion : String
  policyName : String
  policySchema : String
  policyDigest : String
  toolchain : String
  sourceSchema : String
  sourceDigest : String
  traceId : String
  steps : List ConformanceStepOracle
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

def buildConformanceOracles :
    RunState → List TraceStep → List ConformanceStepOracle
  | _, [] => []
  | state, step :: tail =>
      let result := transition state step.event
      let view := conformanceStateView result.state
      { eventId := step.causality.eventId
        eventIndex := step.causality.eventIndex
        disposition := result.disposition.label
        state := view
        stateDigest := (conformanceStateDigest result.state).value
        leanStateDigest := (stateDigest result.state).value } ::
        buildConformanceOracles result.state tail

def conformanceOracleDocument
    (document : TraceDocument) : ConformanceOracleDocument :=
  let traceId :=
    match document.steps with
    | [] => ""
    | step :: _ => step.causality.traceId
  { schemaVersion := document.schemaVersion
    schemaDigest := document.schemaDigest.value
    viewVersion := nativeConformanceViewVersion
    policyName := document.policyName
    policySchema := document.policySchema
    policyDigest := document.policyDigest.value
    toolchain := document.toolchain
    sourceSchema := document.sourceSchema
    sourceDigest := document.sourceDigest.value
    traceId
    steps :=
      buildConformanceOracles document.initialState.toState document.steps }

end ResilientProtocol
