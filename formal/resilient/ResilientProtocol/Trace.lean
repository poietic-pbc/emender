import ResilientProtocol.Kernel
import Lean.Data.Json

namespace ResilientProtocol

open Lean

/-! ## Deterministic structural digests

These digests identify pure coordination state and trace lineage.  They do not
replace SHA-256 payload/result/checkpoint digests supplied as opaque evidence
by the native/reference authorities.  The algorithm and its Lean toolchain are
both schema-bound so native differential adapters can treat the runner output
as an oracle without relying on an implementation-defined `Hashable` instance.
-/

def digestModulus : Nat := 18446744073709551616

def digestLane (seed : Nat) (material : String) : Nat :=
  material.toList.foldl
    (fun value character =>
      (value * 1099511628211 + character.toNat + 1469598103934665603) %
        digestModulus)
    seed

def hexDigit (value : Nat) : Char :=
  if value < 10 then
    Char.ofNat ('0'.toNat + value)
  else
    Char.ofNat ('a'.toNat + value - 10)

def fixedHex : Nat → Nat → List Char
  | 0, _ => []
  | width + 1, value =>
      fixedHex width (value / 16) ++ [hexDigit (value % 16)]

def coordinationDigest (domain material : String) : Digest :=
  let domainMaterial := domain ++ "\u0000" ++ material
  let lanes :=
    [ digestLane 1469598103934665603 domainMaterial
    , digestLane 1099511628211 domainMaterial
    , digestLane 7809847782465536322 domainMaterial
    , digestLane 9650029242287828579 domainMaterial ]
  ⟨String.ofList (lanes.flatMap (fixedHex 16))⟩

def generationStatusLabel : GenerationStatus → String
  | .open => "open"
  | .closed => "closed"
  | .aborted => "aborted"
  | .committed => "committed"
  | .applied => "applied"

def peerPhaseLabel : PeerPhase → String
  | .discover => "discover"
  | .boot => "boot"
  | .sync => "sync"
  | .ready => "leased_ready"
  | .drain => "drain"
  | .expire => "expire"

def trainerMaterial (trainer : TrainerRef) : String :=
  trainer.trainer.value ++ ":" ++ trainer.incarnation.value

def peerMaterial (peer : PeerRecord) : String :=
  String.intercalate ":"
    [ peer.worker.value
    , peer.node.value
    , peer.incarnation.value
    , peerPhaseLabel peer.phase
    , toString peer.syncedGeneration.value
    , toString peer.leaseUntil.value
    , peer.managerEvidence.value
    , peer.serviceEvidence.value
    , String.intercalate "," (peer.trainers.map trainerMaterial) ]

def contributionMaterial (contribution : ContributionRecord) : String :=
  String.intercalate ":"
    [ toString contribution.key.generation.value
    , toString contribution.key.attempt.value
    , contribution.key.worker.value
    , contribution.node.value
    , contribution.key.incarnation.value
    , toString contribution.key.sequence.value
    , toString contribution.baseGeneration.value
    , contribution.baseDigest.value
    , toString contribution.exactTokens
    , contribution.envelopeDigest.value
    , contribution.payloadDigest.value
    , contribution.trainerSetDigest.value
    , contribution.receipt.value
    , contribution.receiptDigest.value
    , toString contribution.localWindowStart
    , toString contribution.localWindowEnd
    , toString contribution.commitLag
    , toString contribution.anchorLag
    , toString contribution.resultLag
    , toString contribution.speculativeLag
    , toString contribution.admittedAt.value ]

def applyReceiptMaterial (receipt : ApplyReceipt) : String :=
  String.intercalate ":"
    [ toString receipt.generation.value
    , receipt.node.value
    , receipt.worker.value
    , receipt.peerIncarnation.value
    , receipt.trainer.value
    , receipt.trainerIncarnation.value
    , receipt.result.value
    , receipt.resultDigest.value
    , receipt.receipt.value
    , receipt.receiptDigest.value ]

def nodeReceiptMaterial (receipt : NodeApplyReceipt) : String :=
  String.intercalate ":"
    [ toString receipt.generation.value
    , receipt.node.value
    , receipt.worker.value
    , receipt.peerIncarnation.value
    , receipt.result.value
    , receipt.resultDigest.value
    , receipt.trainerReceiptDigest.value
    , receipt.receipt.value
    , receipt.receiptDigest.value ]

def stateMaterial (state : RunState) : String :=
  let cohortMaterial :=
    match state.generation.cohort with
    | none => "-"
    | some cohort =>
        String.intercalate ":"
          [ cohort.cohortDigest.value
          , toString cohort.exactTokens
          , toString cohort.closeTick.value
          , cohort.closeEvidence.value ]
  let commitMaterial :=
    match state.generation.commit with
    | none => "-"
    | some commit =>
        String.intercalate ":"
          [ commit.cohortDigest.value
          , commit.result.value
          , commit.resultDigest.value
          , commit.receipt.value
          , commit.receiptDigest.value
          , commit.priorReceipt.value
          , toString commit.exactTokens
          , toString commit.acceptedTokenClock ]
  let mailboxMaterial :=
    match state.mailbox with
    | none => "-"
    | some mailbox =>
        String.intercalate ":"
          [ mailbox.result.value
          , mailbox.resultDigest.value
          , mailbox.commitReceipt.value
          , mailbox.publicationReceipt.value
          , mailbox.publicationDigest.value ]
  String.intercalate "\u001e"
    [ state.authority.run.value
    , state.authority.allocation.value
    , toString state.authority.fence.value
    , state.authority.policyName
    , state.authority.policySchema
    , state.authority.policyDigest.value
    , state.authority.traceSchema
    , state.authority.traceSchemaDigest.value
    , state.authority.toolchain
    , state.authority.sourceSchema
    , state.authority.sourceDigest.value
    , state.authority.layoutDigest.value
    , state.authority.codeDigest.value
    , toString state.now.value
    , toString state.generation.generation.value
    , toString state.generation.attempt.value
    , generationStatusLabel state.generation.status
    , state.generation.baseDigest.value
    , toString state.generation.openedAt.value
    , toString state.generation.closeTick.value
    , toString state.generation.ownerEpoch.value
    , toString state.generation.ownerReassignments
    , String.intercalate "," (state.generation.readySnapshot.map memberKey)
    , String.intercalate ","
        (state.generation.accepted.map contributionMaterial)
    , String.intercalate ","
        (state.generation.seen.map contributionMaterial)
    , cohortMaterial
    , commitMaterial
    , String.intercalate "," (state.peers.map peerMaterial)
    , mailboxMaterial
    , String.intercalate ","
        (state.applyReceipts.map applyReceiptMaterial)
    , String.intercalate ","
        (state.nodeApplyReceipts.map nodeReceiptMaterial)
    , toString state.acceptedTokenClock
    , state.baseReceipt.value
    , state.baseReceiptDigest.value
    , match state.lastResult with
      | none => "-"
      | some result => result.value
    , toString state.restartCount ]

def stateDigest (state : RunState) : Digest :=
  coordinationDigest "emender-lean-coordination-state-v1" (stateMaterial state)

/-! ## Canonical trace records -/

structure AuthorityView where
  run : RunId
  allocation : AllocationId
  fence : Fence
  policyName : String
  policySchema : String
  policyDigest : Digest
  traceSchema : String
  traceSchemaDigest : Digest
  toolchain : String
  sourceSchema : String
  sourceDigest : Digest
  layoutDigest : Digest
  codeDigest : Digest
  generation : Generation
  attempt : Attempt
  generationStatus : String
  ownerEpoch : OwnerEpoch
  ownerReassignments : Nat
  acceptedTokenClock : Nat
  acceptedCount : Nat
  acceptedTokens : Nat
  cohortDigest : Option Digest
  commitReceipt : Option ReceiptId
  result : Option ResultId
  mailboxResult : Option ResultId
  nodeApplyCount : Nat
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

def authorityView (state : RunState) : AuthorityView :=
  { run := state.authority.run
    allocation := state.authority.allocation
    fence := state.authority.fence
    policyName := state.authority.policyName
    policySchema := state.authority.policySchema
    policyDigest := state.authority.policyDigest
    traceSchema := state.authority.traceSchema
    traceSchemaDigest := state.authority.traceSchemaDigest
    toolchain := state.authority.toolchain
    sourceSchema := state.authority.sourceSchema
    sourceDigest := state.authority.sourceDigest
    layoutDigest := state.authority.layoutDigest
    codeDigest := state.authority.codeDigest
    generation := state.generation.generation
    attempt := state.generation.attempt
    generationStatus := generationStatusLabel state.generation.status
    ownerEpoch := state.generation.ownerEpoch
    ownerReassignments := state.generation.ownerReassignments
    acceptedTokenClock := state.acceptedTokenClock
    acceptedCount := state.generation.accepted.length
    acceptedTokens := contributionTokens state.generation.accepted
    cohortDigest := state.generation.cohort.map (·.cohortDigest)
    commitReceipt := state.generation.commit.map (·.receipt)
    result := state.generation.commit.map (·.result)
    mailboxResult := state.mailbox.map (·.result)
    nodeApplyCount := state.nodeApplyReceipts.length }

structure TraceInitialState where
  authority : AuthorityIdentity
  policy : PolicyConfig
  baseGeneration : Generation
  baseDigest : Digest
  baseReceipt : ReceiptId
  baseReceiptDigest : Digest
  acceptedTokenClock : Nat
  lastResult : Option ResultId
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

def TraceInitialState.toState (initial : TraceInitialState) : RunState :=
  initialState initial.authority initial.policy initial.baseGeneration
    initial.baseDigest initial.baseReceipt initial.baseReceiptDigest
    initial.acceptedTokenClock initial.lastResult

structure CausalityMetadata where
  traceId : String
  eventId : String
  eventIndex : Nat
  sourceEventIndex : Nat
  predecessorStepDigest : Digest
  causalParents : List String
  replayOf : Option String
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure TraceStep where
  schemaVersion : String
  schemaDigest : Digest
  policyName : String
  policySchema : String
  policyDigest : Digest
  toolchain : String
  sourceSchema : String
  sourceDigest : Digest
  preStateDigest : Digest
  preAuthority : AuthorityView
  event : Event
  disposition : String
  postStateDigest : Digest
  postAuthority : AuthorityView
  invariantVerdict : Bool
  invariantViolations : List String
  causality : CausalityMetadata
  stepDigest : Digest
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure TraceDocument where
  schemaVersion : String
  schemaDigest : Digest
  policyName : String
  policySchema : String
  policyDigest : Digest
  toolchain : String
  sourceSchema : String
  sourceDigest : Digest
  initialState : TraceInitialState
  steps : List TraceStep
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure ScheduledEvent where
  eventId : String
  event : Event
  causalParents : List String
  replayOf : Option String := none
  deriving Repr, BEq, DecidableEq

def rootStepDigest (traceId : String) : Digest :=
  coordinationDigest "emender-lean-trace-root-v1" traceId

def traceStepMaterial
    (preDigest : Digest)
    (eventId : String)
    (eventIndex : Nat)
    (disposition : String)
    (postDigest : Digest)
    (predecessor : Digest) : String :=
  String.intercalate "\u001f"
    [ preDigest.value
    , eventId
    , toString eventIndex
    , disposition
    , postDigest.value
    , predecessor.value ]

def computedStepDigest
    (preDigest : Digest)
    (eventId : String)
    (eventIndex : Nat)
    (disposition : String)
    (postDigest : Digest)
    (predecessor : Digest) : Digest :=
  coordinationDigest "emender-lean-trace-step-v1"
    (traceStepMaterial preDigest eventId eventIndex disposition postDigest
      predecessor)

def traceInitialStateOf (state : RunState) : TraceInitialState :=
  { authority := state.authority
    policy := state.policy
    baseGeneration := state.generation.generation
    baseDigest := state.generation.baseDigest
    baseReceipt := state.baseReceipt
    baseReceiptDigest := state.baseReceiptDigest
    acceptedTokenClock := state.acceptedTokenClock
    lastResult := state.lastResult }

def buildTraceSteps
    (traceId : String) :
    Nat → RunState → Digest → List ScheduledEvent → List TraceStep
  | _, _, _, [] => []
  | index, state, predecessor, scheduled :: tail =>
      let preDigest := stateDigest state
      let result := transition state scheduled.event
      let postDigest := stateDigest result.state
      let disposition := result.disposition.label
      let violations := invariantViolations result.state
      let stepDigest :=
        computedStepDigest preDigest scheduled.eventId index disposition
          postDigest predecessor
      let step : TraceStep :=
        { schemaVersion := traceSchemaVersion
          schemaDigest := ⟨traceSchemaDigest⟩
          policyName := state.policy.name
          policySchema := state.policy.schema
          policyDigest := state.policy.digest
          toolchain := toolchainIdentity
          sourceSchema := state.authority.sourceSchema
          sourceDigest := state.authority.sourceDigest
          preStateDigest := preDigest
          preAuthority := authorityView state
          event := scheduled.event
          disposition
          postStateDigest := postDigest
          postAuthority := authorityView result.state
          invariantVerdict := violations.isEmpty
          invariantViolations := violations
          causality :=
            { traceId
              eventId := scheduled.eventId
              eventIndex := index
              sourceEventIndex := index
              predecessorStepDigest := predecessor
              causalParents := scheduled.causalParents
              replayOf := scheduled.replayOf }
          stepDigest }
      step :: buildTraceSteps traceId (index + 1) result.state stepDigest tail

def buildTrace
    (traceId : String)
    (initial : RunState)
    (events : List ScheduledEvent) : TraceDocument :=
  { schemaVersion := traceSchemaVersion
    schemaDigest := ⟨traceSchemaDigest⟩
    policyName := initial.policy.name
    policySchema := initial.policy.schema
    policyDigest := initial.policy.digest
    toolchain := toolchainIdentity
    sourceSchema := initial.authority.sourceSchema
    sourceDigest := initial.authority.sourceDigest
    initialState := traceInitialStateOf initial
    steps :=
      buildTraceSteps traceId 0 initial (rootStepDigest traceId) events }

def renderTrace (trace : TraceDocument) : String :=
  (toJson trace).compress

/--
Round-tripping is the strict unknown-field/ambiguous-encoding check.  Derived
decoders require every field; equality with the canonical encoder rejects
unknown fields at any nesting depth rather than silently ignoring them.
-/
def strictFromJson [FromJson α] [ToJson α] (json : Json) :
    Except String α := do
  let value : α ← fromJson? json
  if toJson value == json then
    return value
  else
    throw "noncanonical, ambiguous, or unknown JSON field"

def parseTrace (input : String) : Except String TraceDocument := do
  let json ← Json.parse input
  let document : TraceDocument ← strictFromJson json
  if input.trim == (toJson document).compress then
    return document
  else
    throw "noncanonical or ambiguous raw JSON encoding"

def stepBindingsMatch
    (document : TraceDocument) (step : TraceStep) : Bool :=
  step.schemaVersion == document.schemaVersion &&
  step.schemaDigest == document.schemaDigest &&
  step.policyName == document.policyName &&
  step.policySchema == document.policySchema &&
  step.policyDigest == document.policyDigest &&
  step.toolchain == document.toolchain &&
  step.sourceSchema == document.sourceSchema &&
  step.sourceDigest == document.sourceDigest

def documentBindingsValid (document : TraceDocument) : Bool :=
  document.schemaVersion == traceSchemaVersion &&
  document.schemaDigest.value == traceSchemaDigest &&
  document.toolchain == toolchainIdentity &&
  document.sourceSchema == executionSourceSchema &&
  validDigest document.schemaDigest &&
  validDigest document.sourceDigest &&
  validAuthority document.initialState.authority &&
  validPolicy document.initialState.policy &&
  authorityMatchesPolicy document.initialState.authority
    document.initialState.policy &&
  document.policyName == document.initialState.policy.name &&
  document.policySchema == document.initialState.policy.schema &&
  document.policyDigest == document.initialState.policy.digest &&
  document.sourceDigest == document.initialState.authority.sourceDigest

def replaySteps
    (document : TraceDocument) :
    Nat → RunState → Digest → List String → List TraceStep →
      Except String RunState
  | _, state, _, _, [] => return state
  | index, state, predecessor, priorEventIds, step :: tail => do
      if !stepBindingsMatch document step then
        throw s!"unknown_identity at step {index}: schema/policy/toolchain/source"
      if step.causality.traceId.isEmpty ||
          !validOpaque step.causality.eventId then
        throw s!"malformed_trace at step {index}: missing trace/event identity"
      if step.causality.eventIndex != index ||
          step.causality.sourceEventIndex != index ||
          step.causality.predecessorStepDigest != predecessor then
        throw s!"forbidden_reorder at step {index}"
      if priorEventIds.any (· == step.causality.eventId) then
        throw s!"ambiguous duplicate event_id at step {index}"
      if !step.causality.causalParents.all
          (fun parent => priorEventIds.any (· == parent)) then
        throw s!"forbidden_reorder at step {index}: unknown causal parent"
      if index > 0 then
        match priorEventIds.reverse with
        | [] => throw s!"forbidden_reorder at step {index}: missing predecessor"
        | immediate :: _ =>
            if !step.causality.causalParents.any (· == immediate) then
              throw s!"forbidden_reorder at step {index}: predecessor not causal"
      match step.causality.replayOf with
      | none => pure ()
      | some replayId =>
          if !priorEventIds.any (· == replayId) then
            throw s!"forbidden_reorder at step {index}: replay target is not prior"
      let actualPreDigest := stateDigest state
      if step.preStateDigest != actualPreDigest ||
          step.preAuthority != authorityView state then
        throw s!"stale_fence at step {index}: pre-state authority mismatch"
      let result := transition state step.event
      let actualPostDigest := stateDigest result.state
      let actualViolations := invariantViolations result.state
      if step.disposition != result.disposition.label then
        throw s!"disposition mismatch at step {index}: expected {step.disposition}, got {result.disposition.label}"
      if step.postStateDigest != actualPostDigest ||
          step.postAuthority != authorityView result.state then
        throw s!"post-state mismatch at step {index}"
      if step.invariantVerdict != actualViolations.isEmpty ||
          step.invariantViolations != actualViolations ||
          !step.invariantVerdict then
        throw s!"invariant failure at step {index}: {actualViolations}"
      let expectedStepDigest :=
        computedStepDigest actualPreDigest step.causality.eventId index
          result.disposition.label actualPostDigest predecessor
      if step.stepDigest != expectedStepDigest then
        throw s!"causality digest mismatch at step {index}"
      replaySteps document (index + 1) result.state expectedStepDigest
        (priorEventIds ++ [step.causality.eventId]) tail

def replayTrace (document : TraceDocument) : Except String RunState := do
  if !documentBindingsValid document then
    throw "unknown_identity: document bindings are not authoritative"
  let state := document.initialState.toState
  if !invariantHolds state then
    throw s!"initial invariant failure: {invariantViolations state}"
  match document.steps with
  | [] => return state
  | first :: _ =>
      replaySteps document 0 state
        (rootStepDigest first.causality.traceId) [] document.steps

def replayTraceString (input : String) : Except String RunState := do
  let document ← parseTrace input
  replayTrace document

end ResilientProtocol
