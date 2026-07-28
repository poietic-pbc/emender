import Std
import Lean.Data.Json

namespace ResilientProtocol

/-! ## Versioned authorities -/

def traceSchemaVersion : String :=
  "emender-resilient-coordination-trace-v1"

def traceSchemaDigest : String :=
  "cf654525395e63b31b2d76e8109ee2bcc6a652f6273d1c6e4ca5bec9ecb776b4"

def toolchainIdentity : String :=
  "leanprover/lean4:v4.26.0@d8204c9fd894f91bbb2cdfec5912ec8196fd8562"

def executionSourceSchema : String :=
  "emender-async-v21-execution-source-v1"

def strictV1PolicyName : String := "resilient-diloco-strict-v1"
def strictV1PolicySchema : String := "emender-resilient-policy-v1"
def asyncV21PolicyName : String := "async-decoupled-v2.1-simple"
def asyncV21PolicySchema : String := "emender-async-policy-v2.1"

/-! ## Nominal identities

These wrappers deliberately keep semantically distinct identities from being
interchanged.  Their string payloads are opaque to the kernel; the JSON
boundary enforces nonempty bounded canonical encodings.
-/

structure RunId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure AllocationId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure Fence where value : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson, Ord, Lean.FromJson, Lean.ToJson

structure Generation where value : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson, Ord, Lean.FromJson, Lean.ToJson

structure Attempt where value : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson, Ord, Lean.FromJson, Lean.ToJson

structure WorkerId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure NodeId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure TrainerId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure Incarnation where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ContributionSeq where value : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson, Ord, Lean.FromJson, Lean.ToJson

structure OwnerEpoch where value : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson, Ord, Lean.FromJson, Lean.ToJson

structure ReceiptId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ResultId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure Digest where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure EvidenceId where value : String
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure Tick where value : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson, Ord, Lean.FromJson, Lean.ToJson

/-! ## Policy and lifecycle -/

inductive PolicyKind where
  | strictV1
  | asyncV21
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

inductive ClosureMode where
  /-- Strict v1: close when the floor is reached or at the finite deadline. -/
  | floorOrDeadline
  /-- Reviewed scale rule: remain open through one immutable finite close. -/
  | fixedSnapshotDeadline
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure PolicyConfig where
  kind : PolicyKind
  name : String
  schema : String
  digest : Digest
  qMin : Nat
  tMin : Nat
  maxCommitLag : Nat
  maxAnchorLag : Nat
  maxResultLag : Nat
  maxSpeculativeLag : Nat
  maxOwnerReassignments : Nat
  attemptRetries : Nat
  kSteps : Nat
  closureMode : ClosureMode
  closureEvidence : EvidenceId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

def PolicyConfig.strictV1
    (digest : Digest) (qMin tMin : Nat) (closureEvidence : EvidenceId) :
    PolicyConfig :=
  { kind := .strictV1
    name := strictV1PolicyName
    schema := strictV1PolicySchema
    digest
    qMin
    tMin
    maxCommitLag := 0
    maxAnchorLag := 0
    maxResultLag := 0
    maxSpeculativeLag := 0
    maxOwnerReassignments := 2
    attemptRetries := 1
    kSteps := 0
    closureMode := .floorOrDeadline
    closureEvidence }

def PolicyConfig.asyncV21
    (digest : Digest) (closureMode : ClosureMode)
    (closureEvidence : EvidenceId) : PolicyConfig :=
  { kind := .asyncV21
    name := asyncV21PolicyName
    schema := asyncV21PolicySchema
    digest
    qMin := 2
    tMin := 3934080
    maxCommitLag := 2
    maxAnchorLag := 2
    maxResultLag := 2
    maxSpeculativeLag := 2
    maxOwnerReassignments := 2
    attemptRetries := 0
    kSteps := 40
    closureMode
    closureEvidence }

inductive PeerPhase where
  | discover
  | boot
  | sync
  | ready
  | drain
  | expire
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

inductive GenerationStatus where
  | open
  | closed
  | aborted
  | committed
  | applied
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

inductive LossRole where
  | participant
  | service
  | manager
  | trainer
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

inductive Disposition where
  | accepted
  | identicalDuplicate
  | conflictingDuplicate
  | staleFence
  | staleIncarnation
  | corruptNonfinite
  | generationClosed
  | late
  | deferred
  | insufficientCohort
  | aborted
  | catchUp
  | retryNextGeneration
  | unknownIdentity
  | malformedTrace
  | forbiddenReorder
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

def Disposition.label : Disposition → String
  | .accepted => "accepted"
  | .identicalDuplicate => "identical_duplicate"
  | .conflictingDuplicate => "conflicting_duplicate"
  | .staleFence => "stale_fence"
  | .staleIncarnation => "stale_incarnation"
  | .corruptNonfinite => "corrupt_nonfinite"
  | .generationClosed => "generation_closed"
  | .late => "late"
  | .deferred => "deferred"
  | .insufficientCohort => "insufficient_cohort"
  | .aborted => "aborted"
  | .catchUp => "catch_up"
  | .retryNextGeneration => "retry_next_generation"
  | .unknownIdentity => "unknown_identity"
  | .malformedTrace => "malformed_trace"
  | .forbiddenReorder => "forbidden_reorder"

/-! ## State records -/

structure AuthorityIdentity where
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
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure EventContext where
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
  ownerEpoch : OwnerEpoch
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure TrainerRef where
  trainer : TrainerId
  incarnation : Incarnation
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure PeerRecord where
  worker : WorkerId
  node : NodeId
  incarnation : Incarnation
  phase : PeerPhase
  syncedGeneration : Generation
  leaseUntil : Tick
  managerEvidence : EvidenceId
  serviceEvidence : EvidenceId
  trainers : List TrainerRef
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ReadyMember where
  worker : WorkerId
  node : NodeId
  incarnation : Incarnation
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ContributionKey where
  generation : Generation
  attempt : Attempt
  worker : WorkerId
  incarnation : Incarnation
  sequence : ContributionSeq
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ContributionRecord where
  key : ContributionKey
  node : NodeId
  baseGeneration : Generation
  baseDigest : Digest
  exactTokens : Nat
  envelopeDigest : Digest
  payloadDigest : Digest
  trainerSetDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  localWindowStart : Nat
  localWindowEnd : Nat
  commitLag : Nat
  anchorLag : Nat
  resultLag : Nat
  speculativeLag : Nat
  admittedAt : Tick
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure FrozenCohort where
  generation : Generation
  attempt : Attempt
  closeTick : Tick
  closeEvidence : EvidenceId
  contributions : List ContributionRecord
  exactTokens : Nat
  cohortDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure CommitRecord where
  generation : Generation
  attempt : Attempt
  ownerEpoch : OwnerEpoch
  cohortDigest : Digest
  result : ResultId
  resultDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  priorReceipt : ReceiptId
  exactTokens : Nat
  acceptedTokenClock : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure MailboxRecord where
  generation : Generation
  result : ResultId
  resultDigest : Digest
  commitReceipt : ReceiptId
  publicationReceipt : ReceiptId
  publicationDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ApplyReceipt where
  generation : Generation
  node : NodeId
  worker : WorkerId
  peerIncarnation : Incarnation
  trainer : TrainerId
  trainerIncarnation : Incarnation
  result : ResultId
  resultDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure NodeApplyReceipt where
  generation : Generation
  node : NodeId
  worker : WorkerId
  peerIncarnation : Incarnation
  result : ResultId
  resultDigest : Digest
  trainerReceiptDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure GenerationRecord where
  generation : Generation
  attempt : Attempt
  status : GenerationStatus
  baseDigest : Digest
  openedAt : Tick
  closeTick : Tick
  readySnapshot : List ReadyMember
  accepted : List ContributionRecord
  seen : List ContributionRecord
  cohort : Option FrozenCohort
  ownerEpoch : OwnerEpoch
  ownerReassignments : Nat
  commit : Option CommitRecord
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure RunState where
  authority : AuthorityIdentity
  policy : PolicyConfig
  now : Tick
  generation : GenerationRecord
  peers : List PeerRecord
  mailbox : Option MailboxRecord
  applyReceipts : List ApplyReceipt
  nodeApplyReceipts : List NodeApplyReceipt
  acceptedTokenClock : Nat
  baseReceipt : ReceiptId
  baseReceiptDigest : Digest
  lastResult : Option ResultId
  restartCount : Nat
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

/-! ## Typed events -/

structure PeerTransitionEvent where
  context : EventContext
  worker : WorkerId
  node : NodeId
  incarnation : Incarnation
  fromPhase : PeerPhase
  toPhase : PeerPhase
  syncedGeneration : Generation
  leaseUntil : Tick
  managerEvidence : EvidenceId
  serviceEvidence : EvidenceId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure RegisterTrainerEvent where
  context : EventContext
  worker : WorkerId
  node : NodeId
  peerIncarnation : Incarnation
  trainer : TrainerId
  trainerIncarnation : Incarnation
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ExpirePeerEvent where
  context : EventContext
  worker : WorkerId
  node : NodeId
  incarnation : Incarnation
  observedAt : Tick
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure OpenGenerationEvent where
  context : EventContext
  baseDigest : Digest
  openedAt : Tick
  closeTick : Tick
  closeEvidence : EvidenceId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ContributionEvent where
  context : EventContext
  worker : WorkerId
  node : NodeId
  incarnation : Incarnation
  sequence : ContributionSeq
  baseGeneration : Generation
  baseDigest : Digest
  exactTokens : Nat
  envelopeDigest : Digest
  payloadDigest : Digest
  trainerSetDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  localWindowStart : Nat
  localWindowEnd : Nat
  commitLag : Nat
  anchorLag : Nat
  resultLag : Nat
  speculativeLag : Nat
  finite : Bool
  checksumValid : Bool
  layoutValid : Bool
  observedAt : Tick
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure CloseGenerationEvent where
  context : EventContext
  observedAt : Tick
  cohortDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure OwnerLossEvent where
  context : EventContext
  owner : WorkerId
  ownerIncarnation : Incarnation
  replayEvidence : EvidenceId
  observedAt : Tick
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure CommitGenerationEvent where
  context : EventContext
  cohortDigest : Digest
  result : ResultId
  resultDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  priorReceipt : ReceiptId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure PublishResultEvent where
  context : EventContext
  result : ResultId
  resultDigest : Digest
  commitReceipt : ReceiptId
  publicationReceipt : ReceiptId
  publicationDigest : Digest
  verified : Bool
  finite : Bool
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure TrainerApplyEvent where
  context : EventContext
  node : NodeId
  worker : WorkerId
  peerIncarnation : Incarnation
  trainer : TrainerId
  trainerIncarnation : Incarnation
  result : ResultId
  resultDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ReduceNodeApplyEvent where
  context : EventContext
  node : NodeId
  worker : WorkerId
  peerIncarnation : Incarnation
  result : ResultId
  resultDigest : Digest
  trainerReceiptDigest : Digest
  receipt : ReceiptId
  receiptDigest : Digest
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure LossEvent where
  context : EventContext
  role : LossRole
  worker : WorkerId
  node : NodeId
  peerIncarnation : Incarnation
  trainer : Option TrainerId
  trainerIncarnation : Option Incarnation
  evidence : EvidenceId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure RestartPeerEvent where
  context : EventContext
  worker : WorkerId
  node : NodeId
  oldIncarnation : Incarnation
  newIncarnation : Incarnation
  managerEvidence : EvidenceId
  serviceEvidence : EvidenceId
  syncedGeneration : Generation
  leaseUntil : Tick
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure AbortGenerationEvent where
  context : EventContext
  reasonEvidence : EvidenceId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

structure ClaimFenceEvent where
  authority : AuthorityIdentity
  policy : PolicyConfig
  baseGeneration : Generation
  baseDigest : Digest
  baseReceipt : ReceiptId
  baseReceiptDigest : Digest
  acceptedTokenClock : Nat
  lastResult : Option ResultId
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

inductive Event where
  | claimFence (value : ClaimFenceEvent)
  | peerTransition (value : PeerTransitionEvent)
  | registerTrainer (value : RegisterTrainerEvent)
  | expirePeer (value : ExpirePeerEvent)
  | openGeneration (value : OpenGenerationEvent)
  | contribution (value : ContributionEvent)
  | closeGeneration (value : CloseGenerationEvent)
  | ownerLoss (value : OwnerLossEvent)
  | commitGeneration (value : CommitGenerationEvent)
  | publishResult (value : PublishResultEvent)
  | trainerApply (value : TrainerApplyEvent)
  | reduceNodeApply (value : ReduceNodeApplyEvent)
  | loss (value : LossEvent)
  | restartPeer (value : RestartPeerEvent)
  | abortGeneration (value : AbortGenerationEvent)
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

def Event.context? : Event → Option EventContext
  | .claimFence _ => none
  | .peerTransition e => some e.context
  | .registerTrainer e => some e.context
  | .expirePeer e => some e.context
  | .openGeneration e => some e.context
  | .contribution e => some e.context
  | .closeGeneration e => some e.context
  | .ownerLoss e => some e.context
  | .commitGeneration e => some e.context
  | .publishResult e => some e.context
  | .trainerApply e => some e.context
  | .reduceNodeApply e => some e.context
  | .loss e => some e.context
  | .restartPeer e => some e.context
  | .abortGeneration e => some e.context

structure StepResult where
  state : RunState
  disposition : Disposition
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

end ResilientProtocol
