import ResilientProtocol.Types

namespace ResilientProtocol

/-! ## Canonical validation helpers -/

def isLowerHexChar (c : Char) : Bool :=
  ('0' ≤ c && c ≤ '9') || ('a' ≤ c && c ≤ 'f')

def validDigest (digest : Digest) : Bool :=
  digest.value.length == 64 && digest.value.toList.all isLowerHexChar

def validOpaque (value : String) : Bool :=
  !value.isEmpty && value.length ≤ 128 &&
    value.toList.all (fun c => c ≥ '!' && c ≤ '~')

def validEvidence (value : EvidenceId) : Bool :=
  validOpaque value.value

def validAuthority (identity : AuthorityIdentity) : Bool :=
  validOpaque identity.run.value &&
  validOpaque identity.allocation.value &&
  (identity.policyName == strictV1PolicyName ||
    identity.policyName == asyncV21PolicyName) &&
  (identity.policySchema == strictV1PolicySchema ||
    identity.policySchema == asyncV21PolicySchema) &&
  validDigest identity.policyDigest &&
  identity.traceSchema == traceSchemaVersion &&
  identity.traceSchemaDigest.value == traceSchemaDigest &&
  identity.toolchain == toolchainIdentity &&
  identity.sourceSchema == executionSourceSchema &&
  validDigest identity.sourceDigest &&
  validDigest identity.layoutDigest &&
  validDigest identity.codeDigest

def validPolicy (policy : PolicyConfig) : Bool :=
  validDigest policy.digest &&
  validEvidence policy.closureEvidence &&
  policy.qMin > 0 &&
  policy.tMin > 0 &&
  policy.maxOwnerReassignments == 2 &&
  match policy.kind with
  | .strictV1 =>
      policy.name == strictV1PolicyName &&
      policy.schema == strictV1PolicySchema &&
      policy.maxCommitLag == 0 &&
      policy.maxAnchorLag == 0 &&
      policy.maxResultLag == 0 &&
      policy.maxSpeculativeLag == 0
  | .asyncV21 =>
      policy.name == asyncV21PolicyName &&
      policy.schema == asyncV21PolicySchema &&
      policy.qMin == 2 &&
      policy.tMin == 3934080 &&
      policy.maxCommitLag == 2 &&
      policy.maxAnchorLag == 2 &&
      policy.maxResultLag == 2 &&
      policy.maxSpeculativeLag == 2 &&
      policy.attemptRetries == 0 &&
      policy.kSteps == 40

def authorityMatchesPolicy
    (identity : AuthorityIdentity) (policy : PolicyConfig) : Bool :=
  identity.policyName == policy.name &&
  identity.policySchema == policy.schema &&
  identity.policyDigest == policy.digest

def initialState
    (authority : AuthorityIdentity)
    (policy : PolicyConfig)
    (baseGeneration : Generation)
    (baseDigest : Digest)
    (baseReceipt : ReceiptId)
    (baseReceiptDigest : Digest)
    (acceptedTokenClock : Nat)
    (lastResult : Option ResultId := none) : RunState :=
  { authority
    policy
    now := ⟨0⟩
    generation :=
      { generation := baseGeneration
        attempt := ⟨0⟩
        status := .applied
        baseDigest
        openedAt := ⟨0⟩
        closeTick := ⟨0⟩
        readySnapshot := []
        accepted := []
        seen := []
        cohort := none
        ownerEpoch := ⟨0⟩
        ownerReassignments := 0
        commit := none }
    peers := []
    mailbox := none
    applyReceipts := []
    nodeApplyReceipts := []
    acceptedTokenClock
    baseReceipt
    baseReceiptDigest
    lastResult
    restartCount := 0 }

def noChange (state : RunState) (disposition : Disposition) : StepResult :=
  { state, disposition }

def acceptedState (state : RunState) : StepResult :=
  { state, disposition := .accepted }

def findPeer? (state : RunState) (worker : WorkerId) : Option PeerRecord :=
  state.peers.find? (fun peer => peer.worker == worker)

def replacePeer (peers : List PeerRecord) (replacement : PeerRecord) :
    List PeerRecord :=
  peers.map fun peer =>
    if peer.worker == replacement.worker then replacement else peer

def removePeer (peers : List PeerRecord) (worker : WorkerId) :
    List PeerRecord :=
  peers.filter fun peer => peer.worker != worker

def uniqueBy [BEq α] (values : List α) : Bool :=
  match values with
  | [] => true
  | head :: tail => !(tail.any fun value => value == head) && uniqueBy tail

def contributionTokens (values : List ContributionRecord) : Nat :=
  values.foldl (fun total contribution => total + contribution.exactTokens) 0

def contributionWorkers (values : List ContributionRecord) : List WorkerId :=
  values.map (·.key.worker)

def contributionKeyMatches
    (event : ContributionEvent) (record : ContributionRecord) : Bool :=
  record.key.generation == event.context.generation &&
  record.key.attempt == event.context.attempt &&
  record.key.worker == event.worker &&
  record.key.incarnation == event.incarnation &&
  record.key.sequence == event.sequence

def contributionReplayMatches
    (event : ContributionEvent) (record : ContributionRecord) : Bool :=
  contributionKeyMatches event record &&
  record.node == event.node &&
  record.baseGeneration == event.baseGeneration &&
  record.baseDigest == event.baseDigest &&
  record.exactTokens == event.exactTokens &&
  record.envelopeDigest == event.envelopeDigest &&
  record.payloadDigest == event.payloadDigest &&
  record.trainerSetDigest == event.trainerSetDigest &&
  record.receipt == event.receipt &&
  record.receiptDigest == event.receiptDigest &&
  record.localWindowStart == event.localWindowStart &&
  record.localWindowEnd == event.localWindowEnd &&
  record.commitLag == event.commitLag &&
  record.anchorLag == event.anchorLag &&
  record.resultLag == event.resultLag &&
  record.speculativeLag == event.speculativeLag &&
  event.finite && event.checksumValid && event.layoutValid

def insertContribution
    (record : ContributionRecord) :
    List ContributionRecord → List ContributionRecord
  | [] => [record]
  | head :: tail =>
      if record.envelopeDigest.value < head.envelopeDigest.value then
        record :: head :: tail
      else
        head :: insertContribution record tail

def memberKey (member : ReadyMember) : String :=
  member.worker.value ++ "\u001f" ++ member.node.value ++ "\u001f" ++
    member.incarnation.value

def insertReadyMember
    (member : ReadyMember) : List ReadyMember → List ReadyMember
  | [] => [member]
  | head :: tail =>
      if memberKey member < memberKey head then
        member :: head :: tail
      else
        head :: insertReadyMember member tail

def readySnapshot
    (state : RunState) (generation : Generation) (atTick : Tick) :
    List ReadyMember :=
  state.peers.foldl
    (fun members peer =>
      if peer.phase == .ready &&
          peer.syncedGeneration == generation &&
          peer.leaseUntil.value > atTick.value then
        insertReadyMember
          { worker := peer.worker
            node := peer.node
            incarnation := peer.incarnation }
          members
      else
        members)
    []

def isReadySnapshotMember
    (generation : GenerationRecord)
    (worker : WorkerId)
    (node : NodeId)
    (incarnation : Incarnation) : Bool :=
  generation.readySnapshot.any fun member =>
    member.worker == worker &&
    member.node == node &&
    member.incarnation == incarnation

def phaseTransitionAllowed (fromPhase toPhase : PeerPhase) : Bool :=
  match fromPhase, toPhase with
  | .discover, .discover => true
  | .discover, .boot => true
  | .boot, .sync => true
  | .sync, .ready => true
  | .ready, .ready => true
  | .ready, .drain => true
  | .sync, .drain => true
  | .boot, .drain => true
  | .discover, .expire => true
  | .boot, .expire => true
  | .sync, .expire => true
  | .ready, .expire => true
  | .drain, .expire => true
  | _, _ => false

def contextAuthorityDisposition?
    (state : RunState) (context : EventContext) : Option Disposition :=
  if !validOpaque context.run.value ||
      !validOpaque context.allocation.value ||
      !validDigest context.policyDigest ||
      !validDigest context.traceSchemaDigest ||
      !validDigest context.sourceDigest ||
      !validDigest context.layoutDigest ||
      !validDigest context.codeDigest then
    some .unknownIdentity
  else if context.run != state.authority.run then
    some .unknownIdentity
  else if context.fence.value < state.authority.fence.value then
    some .staleFence
  else if context.fence != state.authority.fence ||
      context.allocation != state.authority.allocation then
    some .unknownIdentity
  else if context.policyName != state.policy.name ||
      context.policySchema != state.policy.schema ||
      context.policyDigest != state.policy.digest ||
      context.traceSchema != state.authority.traceSchema ||
      context.traceSchemaDigest != state.authority.traceSchemaDigest ||
      context.toolchain != state.authority.toolchain ||
      context.sourceSchema != state.authority.sourceSchema ||
      context.sourceDigest != state.authority.sourceDigest ||
      context.layoutDigest != state.authority.layoutDigest ||
      context.codeDigest != state.authority.codeDigest then
    some .unknownIdentity
  else
    none

def currentGenerationDisposition?
    (state : RunState) (context : EventContext) : Option Disposition :=
  match contextAuthorityDisposition? state context with
  | some disposition => some disposition
  | none =>
      if context.generation.value < state.generation.generation.value then
        if state.lastResult.isSome then some .catchUp else some .generationClosed
      else if context.generation.value > state.generation.generation.value then
        some .unknownIdentity
      else if context.attempt != state.generation.attempt then
        some .generationClosed
      else if context.ownerEpoch.value < state.generation.ownerEpoch.value then
        some .late
      else if context.ownerEpoch.value > state.generation.ownerEpoch.value then
        some .unknownIdentity
      else
        none

def peerHasNodeApplyAuthority
    (state : RunState) (peer : PeerRecord) (generation : Generation) : Bool :=
  state.nodeApplyReceipts.any fun receipt =>
    receipt.generation == generation &&
    receipt.node == peer.node &&
    receipt.worker == peer.worker &&
    receipt.peerIncarnation == peer.incarnation

def cohortNodes (cohort : FrozenCohort) : List NodeId :=
  cohort.contributions.foldl
    (fun nodes contribution =>
      if nodes.any (· == contribution.node) then nodes
      else contribution.node :: nodes)
    []

def nodeHasApplyReceipt
    (state : RunState) (generation : Generation) (node : NodeId) : Bool :=
  state.nodeApplyReceipts.any fun receipt =>
    receipt.generation == generation && receipt.node == node

def allCohortNodesApplied (state : RunState) : Bool :=
  match state.generation.cohort with
  | none => false
  | some cohort =>
      (cohortNodes cohort).all
        (nodeHasApplyReceipt state state.generation.generation)

/-! ## Invariant oracle

The downstream proof task strengthens these executable checks into theorems.
They are intentionally structural: native/reference evidence remains the
authority for physical bytes, IEEE-754 execution, and runtime scheduling.
-/

def invariantViolations (state : RunState) : List String :=
  let accepted := state.generation.accepted
  let positiveTokens :=
    accepted.all (fun contribution => contribution.exactTokens > 0)
  let uniqueContributions := uniqueBy (accepted.map (·.key))
  let uniqueWorkers := uniqueBy (contributionWorkers accepted)
  let cohortConsistent : Bool :=
    match state.generation.cohort with
    | none =>
        state.generation.status == .open ||
        state.generation.status == .aborted ||
        state.generation.status == .applied
    | some cohort =>
        cohort.generation == state.generation.generation &&
        cohort.attempt == state.generation.attempt &&
        cohort.contributions == accepted &&
        cohort.exactTokens == contributionTokens accepted
  let commitConsistent :=
    match state.generation.commit, state.generation.cohort with
    | none, _ => true
    | some _, none => false
    | some commit, some cohort =>
        commit.generation == state.generation.generation &&
        commit.attempt == state.generation.attempt &&
        commit.ownerEpoch == state.generation.ownerEpoch &&
        commit.cohortDigest == cohort.cohortDigest &&
        commit.exactTokens == cohort.exactTokens &&
        commit.acceptedTokenClock == state.acceptedTokenClock &&
        state.baseReceipt == commit.receipt &&
        state.lastResult == some commit.result
  let mailboxConsistent :=
    match state.mailbox, state.generation.commit with
    | none, _ => true
    | some _, none => false
    | some mailbox, some commit =>
        mailbox.generation == commit.generation &&
        mailbox.result == commit.result &&
        mailbox.resultDigest == commit.resultDigest &&
        mailbox.commitReceipt == commit.receipt
  let applyReceiptsValid :=
    state.nodeApplyReceipts.all fun nodeReceipt =>
      let matching :=
        state.applyReceipts.filter fun receipt =>
          receipt.generation == nodeReceipt.generation &&
          receipt.node == nodeReceipt.node &&
          receipt.worker == nodeReceipt.worker &&
          receipt.peerIncarnation == nodeReceipt.peerIncarnation &&
          receipt.result == nodeReceipt.result &&
          receipt.resultDigest == nodeReceipt.resultDigest
      matching.length == 8 &&
      uniqueBy (matching.map (·.trainer))
  let nextReadyValid :=
    state.peers.all fun peer =>
      if peer.phase != .ready then true
      else if peer.syncedGeneration.value ≤ state.generation.generation.value then true
      else
        peer.syncedGeneration.value == state.generation.generation.value + 1 &&
        peerHasNodeApplyAuthority state peer state.generation.generation
  let appliedConsistent :=
    if state.generation.status == .applied && state.generation.commit.isSome then
      allCohortNodesApplied state
    else
      true
  []
  |> (fun errors => if validAuthority state.authority then errors
      else "invalid_authority_identity" :: errors)
  |> (fun errors => if validPolicy state.policy then errors
      else "invalid_policy" :: errors)
  |> (fun errors => if authorityMatchesPolicy state.authority state.policy then errors
      else "authority_policy_mismatch" :: errors)
  |> (fun errors => if state.generation.ownerReassignments ≤
      state.policy.maxOwnerReassignments then errors
      else "owner_replay_bound" :: errors)
  |> (fun errors => if positiveTokens then errors
      else "nonpositive_accepted_tokens" :: errors)
  |> (fun errors => if uniqueContributions then errors
      else "duplicate_contribution_identity" :: errors)
  |> (fun errors => if uniqueWorkers then errors
      else "duplicate_stable_worker" :: errors)
  |> (fun errors => if cohortConsistent then errors
      else "mutable_or_inconsistent_cohort" :: errors)
  |> (fun errors => if commitConsistent then errors
      else "commit_identity_or_clock" :: errors)
  |> (fun errors => if mailboxConsistent then errors
      else "mailbox_without_authoritative_commit" :: errors)
  |> (fun errors => if applyReceiptsValid then errors
      else "node_apply_without_eight_trainers" :: errors)
  |> (fun errors => if nextReadyValid then errors
      else "ready_without_node_apply_authority" :: errors)
  |> (fun errors => if appliedConsistent then errors
      else "applied_without_all_cohort_nodes" :: errors)

def invariantHolds (state : RunState) : Bool :=
  (invariantViolations state).isEmpty

/-! ## Deterministic total transition -/

def transitionClaimFence
    (state : RunState) (event : ClaimFenceEvent) : StepResult :=
  if !validAuthority event.authority ||
      !validPolicy event.policy ||
      !authorityMatchesPolicy event.authority event.policy ||
      !validDigest event.baseDigest ||
      !validDigest event.baseReceiptDigest ||
      !validOpaque event.baseReceipt.value then
    noChange state .unknownIdentity
  else if event.authority.run != state.authority.run then
    noChange state .unknownIdentity
  else if event.authority.fence.value < state.authority.fence.value then
    noChange state .staleFence
  else if event.authority.fence == state.authority.fence then
    if event.authority == state.authority &&
        event.baseReceipt == state.baseReceipt &&
        event.baseReceiptDigest == state.baseReceiptDigest &&
        event.acceptedTokenClock == state.acceptedTokenClock then
      noChange state .identicalDuplicate
    else
      noChange state .conflictingDuplicate
  else if event.policy != state.policy ||
      event.baseGeneration.value < state.generation.generation.value ||
      event.baseReceipt != state.baseReceipt ||
      event.baseReceiptDigest != state.baseReceiptDigest ||
      event.acceptedTokenClock != state.acceptedTokenClock ||
      event.lastResult != state.lastResult then
    noChange state .unknownIdentity
  else
    let restarted :=
      initialState event.authority event.policy event.baseGeneration
        event.baseDigest event.baseReceipt event.baseReceiptDigest
        event.acceptedTokenClock event.lastResult
    acceptedState { restarted with restartCount := state.restartCount + 1 }

def peerReadyTransitionAllowed
    (state : RunState) (event : PeerTransitionEvent)
    (peer : PeerRecord) : Bool :=
  let targetGeneration :=
    match state.generation.commit with
    | none => state.generation.generation
    | some _ => ⟨state.generation.generation.value + 1⟩
  if event.toPhase != .ready then
    true
  else
    event.syncedGeneration == targetGeneration &&
    event.leaseUntil.value > state.now.value &&
    peer.trainers.length == 8 &&
    uniqueBy (peer.trainers.map (·.trainer)) &&
    match state.generation.commit with
    | none => true
    | some _ =>
        peerHasNodeApplyAuthority
          state peer state.generation.generation

def transitionPeer
    (state : RunState) (event : PeerTransitionEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      if !validOpaque event.worker.value ||
          !validOpaque event.node.value ||
          !validOpaque event.incarnation.value ||
          !validEvidence event.managerEvidence ||
          !validEvidence event.serviceEvidence ||
          !phaseTransitionAllowed event.fromPhase event.toPhase then
        noChange state .unknownIdentity
      else
        match findPeer? state event.worker with
        | none =>
            if event.fromPhase != .discover ||
                event.toPhase != .discover ||
                event.syncedGeneration != state.generation.generation then
              noChange state .unknownIdentity
            else
              let peer : PeerRecord :=
                { worker := event.worker
                  node := event.node
                  incarnation := event.incarnation
                  phase := .discover
                  syncedGeneration := event.syncedGeneration
                  leaseUntil := event.leaseUntil
                  managerEvidence := event.managerEvidence
                  serviceEvidence := event.serviceEvidence
                  trainers := [] }
              acceptedState { state with peers := state.peers ++ [peer] }
        | some peer =>
            if peer.node != event.node ||
                peer.incarnation != event.incarnation then
              noChange state .staleIncarnation
            else if peer.phase != event.fromPhase then
              noChange state .conflictingDuplicate
            else
              if !peerReadyTransitionAllowed state event peer then
                noChange state .deferred
              else
                let updated : PeerRecord :=
                  { peer with
                    phase := event.toPhase
                    syncedGeneration := event.syncedGeneration
                    leaseUntil := event.leaseUntil
                    managerEvidence := event.managerEvidence
                    serviceEvidence := event.serviceEvidence }
                acceptedState
                  { state with
                    peers := replacePeer state.peers updated }

def transitionRegisterTrainer
    (state : RunState) (event : RegisterTrainerEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      match findPeer? state event.worker with
      | none => noChange state .unknownIdentity
      | some peer =>
          if peer.node != event.node ||
              peer.incarnation != event.peerIncarnation then
            noChange state .staleIncarnation
          else if !validOpaque event.trainer.value ||
              !validOpaque event.trainerIncarnation.value then
            noChange state .unknownIdentity
          else
            match peer.trainers.find? (fun trainer =>
                trainer.trainer == event.trainer) with
            | some trainer =>
                if trainer.incarnation == event.trainerIncarnation then
                  noChange state .identicalDuplicate
                else
                  noChange state .conflictingDuplicate
            | none =>
                if peer.trainers.length ≥ 8 || peer.phase == .ready then
                  noChange state .deferred
                else
                  let updated :=
                    { peer with
                      trainers := peer.trainers ++
                        [{ trainer := event.trainer
                           incarnation := event.trainerIncarnation }] }
                  acceptedState
                    { state with peers := replacePeer state.peers updated }

def transitionExpirePeer
    (state : RunState) (event : ExpirePeerEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      match findPeer? state event.worker with
      | none => noChange state .identicalDuplicate
      | some peer =>
          if peer.node != event.node ||
              peer.incarnation != event.incarnation then
            noChange state .staleIncarnation
          else if peer.phase == .expire then
            noChange state .identicalDuplicate
          else if event.observedAt.value < peer.leaseUntil.value then
            noChange state .deferred
          else
            let updated := { peer with phase := .expire }
            acceptedState
              { state with
                peers := replacePeer state.peers updated
                now := ⟨max state.now.value event.observedAt.value⟩ }

def transitionOpen
    (state : RunState) (event : OpenGenerationEvent) : StepResult :=
  match contextAuthorityDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      let expectedGeneration :=
        match state.generation.commit with
        | none => state.generation.generation
        | some _ => ⟨state.generation.generation.value + 1⟩
      let expectedBase :=
        match state.generation.commit with
        | none => state.generation.baseDigest
        | some commit => commit.resultDigest
      if state.generation.status != .applied then
        noChange state .deferred
      else if event.context.generation != expectedGeneration ||
          event.context.attempt.value != 0 ||
          event.context.ownerEpoch.value != 0 ||
          event.baseDigest != expectedBase then
        noChange state .unknownIdentity
      else if event.closeEvidence != state.policy.closureEvidence ||
          event.closeTick.value ≤ event.openedAt.value then
        noChange state .unknownIdentity
      else
        let snapshot :=
          readySnapshot state event.context.generation event.openedAt
        let opened : GenerationRecord :=
          { generation := event.context.generation
            attempt := event.context.attempt
            status := .open
            baseDigest := event.baseDigest
            openedAt := event.openedAt
            closeTick := event.closeTick
            readySnapshot := snapshot
            accepted := []
            seen := []
            cohort := none
            ownerEpoch := ⟨0⟩
            ownerReassignments := 0
            commit := none }
        acceptedState
          { state with
            generation := opened
            now := event.openedAt
            mailbox := none
            applyReceipts := []
            nodeApplyReceipts := [] }

inductive ContributionDecision where
  | reject (disposition : Disposition)
  | acceptContribution
  deriving Repr, BEq, DecidableEq

inductive ContributionReadyDecision where
  | reject (disposition : Disposition)
  | ready (peer : PeerRecord)
  deriving Repr, BEq, DecidableEq

inductive ContributionPrefixDecision where
  | reject (disposition : Disposition)
  | checkPeer
  deriving Repr, BEq, DecidableEq

def leasedReadyAdmissionGate
    (state : RunState) (event : ContributionEvent)
    (peer : PeerRecord) : Bool :=
  peer.phase == .ready &&
  peer.leaseUntil.value > event.observedAt.value &&
  isReadySnapshotMember state.generation event.worker
    event.node event.incarnation

def contributionRecord (event : ContributionEvent) : ContributionRecord :=
  { key :=
      { generation := event.context.generation
        attempt := event.context.attempt
        worker := event.worker
        incarnation := event.incarnation
        sequence := event.sequence }
    node := event.node
    baseGeneration := event.baseGeneration
    baseDigest := event.baseDigest
    exactTokens := event.exactTokens
    envelopeDigest := event.envelopeDigest
    payloadDigest := event.payloadDigest
    trainerSetDigest := event.trainerSetDigest
    receipt := event.receipt
    receiptDigest := event.receiptDigest
    localWindowStart := event.localWindowStart
    localWindowEnd := event.localWindowEnd
    commitLag := event.commitLag
    anchorLag := event.anchorLag
    resultLag := event.resultLag
    speculativeLag := event.speculativeLag
    admittedAt := event.observedAt }

/-- Fenced generation/replay prefix shared by executable contribution admission. -/
def decideContributionPrefix
    (state : RunState)
    (event : ContributionEvent) : ContributionPrefixDecision :=
  match contextAuthorityDisposition? state event.context with
  | some disposition => .reject disposition
  | none =>
      if event.context.generation.value < state.generation.generation.value then
        if state.lastResult.isSome then .reject .catchUp
        else .reject .generationClosed
      else if event.context.generation.value >
          state.generation.generation.value then
        .reject .unknownIdentity
      else if event.context.attempt != state.generation.attempt then
        .reject .generationClosed
      else if event.context.ownerEpoch.value <
          state.generation.ownerEpoch.value then
        .reject .late
      else if event.context.ownerEpoch.value >
          state.generation.ownerEpoch.value then
        .reject .unknownIdentity
      else
        match state.generation.seen.find? (contributionKeyMatches event) with
        | some prior =>
            if contributionReplayMatches event prior then
              .reject .identicalDuplicate
            else
              .reject .conflictingDuplicate
        | none =>
            if state.generation.status != .open then
              if (state.generation.status == .committed ||
                  state.generation.status == .applied) &&
                  state.lastResult.isSome then
                .reject .catchUp
              else
                .reject .generationClosed
            else
              .checkPeer

/--
The identity, generation, replay, close-time, and leased-READY admission
decision.  It is a stage of the one executable transition, not a proof-only
model.
-/
def decideContributionReady
    (state : RunState)
    (event : ContributionEvent) : ContributionReadyDecision :=
  match decideContributionPrefix state event with
  | .reject disposition => .reject disposition
  | .checkPeer =>
      match findPeer? state event.worker with
      | none => .reject .staleIncarnation
      | some peer =>
          if peer.node != event.node ||
              peer.incarnation != event.incarnation then
            .reject .staleIncarnation
          else if event.observedAt.value >
              state.generation.closeTick.value then
            .reject .late
          else if !leasedReadyAdmissionGate state event peer then
            .reject .retryNextGeneration
          else if state.generation.accepted.any
              (fun prior => prior.key.worker == event.worker) then
            .reject .retryNextGeneration
          else
            .ready peer

def decideContribution
    (state : RunState) (event : ContributionEvent) : ContributionDecision :=
  match decideContributionReady state event with
  | .reject disposition => .reject disposition
  | .ready _ =>
      if event.exactTokens == 0 ||
          event.exactTokens > 9007199254740991 ||
          !event.finite || !event.checksumValid ||
          !event.layoutValid ||
          !validDigest event.envelopeDigest ||
          !validDigest event.payloadDigest ||
          !validDigest event.trainerSetDigest ||
          !validDigest event.receiptDigest ||
          !validOpaque event.receipt.value then
        .reject .corruptNonfinite
      else
        let baseLagValid :=
          event.context.generation.value ≥ event.baseGeneration.value &&
          event.commitLag ==
            event.context.generation.value - event.baseGeneration.value
        let lagValid :=
          baseLagValid &&
          event.commitLag ≤ state.policy.maxCommitLag &&
          event.anchorLag ≤ state.policy.maxAnchorLag &&
          event.resultLag ≤ state.policy.maxResultLag &&
          event.speculativeLag ≤ state.policy.maxSpeculativeLag
        let v1Valid :=
          event.baseGeneration == event.context.generation &&
          event.baseDigest == state.generation.baseDigest &&
          event.commitLag == 0 &&
          event.anchorLag == 0 &&
          event.resultLag == 0 &&
          event.speculativeLag == 0
        let v21Valid :=
          event.localWindowEnd > event.localWindowStart &&
          (event.localWindowEnd - event.localWindowStart) %
              state.policy.kSteps == 0
        let policyValid :=
          match state.policy.kind with
          | .strictV1 => v1Valid
          | .asyncV21 => lagValid && v21Valid
        if !policyValid then
          if event.commitLag > state.policy.maxCommitLag ||
              event.anchorLag > state.policy.maxAnchorLag ||
              event.resultLag > state.policy.maxResultLag ||
              event.speculativeLag >
                state.policy.maxSpeculativeLag then
            .reject .catchUp
          else
            .reject .unknownIdentity
        else
          .acceptContribution

def applyContributionDecision
    (state : RunState) (event : ContributionEvent) :
    ContributionDecision → StepResult
  | .reject .accepted => noChange state .unknownIdentity
  | .reject .insufficientCohort => noChange state .unknownIdentity
  | .reject .aborted => noChange state .unknownIdentity
  | .reject disposition => noChange state disposition
  | .acceptContribution =>
      let record := contributionRecord event
      let accepted :=
        insertContribution record state.generation.accepted
      let seen :=
        insertContribution record state.generation.seen
      acceptedState
        { state with
          generation := { state.generation with accepted, seen }
          now := ⟨max state.now.value event.observedAt.value⟩ }

def transitionContribution
    (state : RunState) (event : ContributionEvent) : StepResult :=
  applyContributionDecision state event (decideContribution state event)

def floorSatisfied (state : RunState) : Bool :=
  uniqueBy (contributionWorkers state.generation.accepted) &&
  state.generation.accepted.length ≥ state.policy.qMin &&
  contributionTokens state.generation.accepted ≥ state.policy.tMin

def transitionClose
    (state : RunState) (event : CloseGenerationEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      match state.generation.cohort with
      | some cohort =>
          if cohort.cohortDigest == event.cohortDigest then
            noChange state .identicalDuplicate
          else
            noChange state .conflictingDuplicate
      | none =>
          if state.generation.status != .open then
            noChange state .generationClosed
          else if !validDigest event.cohortDigest then
            noChange state .unknownIdentity
          else
            let atDeadline :=
              event.observedAt.value ≥ state.generation.closeTick.value
            let mayClose :=
              match state.policy.closureMode with
              | .floorOrDeadline => floorSatisfied state || atDeadline
              | .fixedSnapshotDeadline => atDeadline
            if !mayClose then
              noChange state .deferred
            else if !floorSatisfied state then
              noChange
                { state with
                  generation :=
                    { state.generation with status := .aborted }
                  now := ⟨max state.now.value event.observedAt.value⟩ }
                .insufficientCohort
            else
              let cohort : FrozenCohort :=
                { generation := state.generation.generation
                  attempt := state.generation.attempt
                  closeTick := state.generation.closeTick
                  closeEvidence := state.policy.closureEvidence
                  contributions := state.generation.accepted
                  exactTokens :=
                    contributionTokens state.generation.accepted
                  cohortDigest := event.cohortDigest }
              acceptedState
                { state with
                  generation :=
                    { state.generation with
                      status := .closed
                      cohort := some cohort }
                  now := ⟨max state.now.value event.observedAt.value⟩ }

def transitionOwnerLoss
    (state : RunState) (event : OwnerLossEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      if !validEvidence event.replayEvidence then
        noChange state .unknownIdentity
      else if state.generation.status == .committed ||
          state.generation.status == .applied ||
          state.generation.status == .aborted then
        noChange state .generationClosed
      else
        match findPeer? state event.owner with
        | none => noChange state .staleIncarnation
        | some peer =>
            if peer.incarnation != event.ownerIncarnation then
              noChange state .staleIncarnation
            else if state.generation.ownerReassignments ≥
                state.policy.maxOwnerReassignments then
              noChange
                { state with
                  generation :=
                    { state.generation with status := .aborted }
                  now := ⟨max state.now.value event.observedAt.value⟩ }
                .aborted
            else
              acceptedState
                { state with
                  generation :=
                    { state.generation with
                      ownerEpoch :=
                        ⟨state.generation.ownerEpoch.value + 1⟩
                      ownerReassignments :=
                        state.generation.ownerReassignments + 1 }
                  now := ⟨max state.now.value event.observedAt.value⟩ }

inductive CommitDecision where
  | reject (disposition : Disposition)
  | commit
  deriving Repr, BEq, DecidableEq

def commitRecord
    (state : RunState) (event : CommitGenerationEvent)
    (cohort : FrozenCohort) : CommitRecord :=
  { generation := state.generation.generation
    attempt := state.generation.attempt
    ownerEpoch := state.generation.ownerEpoch
    cohortDigest := cohort.cohortDigest
    result := event.result
    resultDigest := event.resultDigest
    receipt := event.receipt
    receiptDigest := event.receiptDigest
    priorReceipt := event.priorReceipt
    exactTokens := cohort.exactTokens
    acceptedTokenClock :=
      state.acceptedTokenClock + cohort.exactTokens }

def decideCommit
    (state : RunState) (event : CommitGenerationEvent) : CommitDecision :=
  match currentGenerationDisposition? state event.context with
  | some disposition => .reject disposition
  | none =>
      match state.generation.commit with
      | some commit =>
          if commit.cohortDigest == event.cohortDigest &&
              commit.result == event.result &&
              commit.resultDigest == event.resultDigest &&
              commit.receipt == event.receipt &&
              commit.receiptDigest == event.receiptDigest &&
              commit.priorReceipt == event.priorReceipt then
            .reject .identicalDuplicate
          else
            .reject .conflictingDuplicate
      | none =>
          match state.generation.cohort with
          | none => .reject .insufficientCohort
          | some cohort =>
              if state.generation.status != .closed then
                .reject .generationClosed
              else if cohort.cohortDigest != event.cohortDigest ||
                  event.priorReceipt != state.baseReceipt then
                .reject .conflictingDuplicate
              else if !validOpaque event.result.value ||
                  !validOpaque event.receipt.value ||
                  !validDigest event.resultDigest ||
                  !validDigest event.receiptDigest then
                .reject .corruptNonfinite
              else
                .commit

def applyCommitDecision
    (state : RunState) (event : CommitGenerationEvent) :
    CommitDecision → StepResult
  | .reject .accepted => noChange state .unknownIdentity
  | .reject disposition => noChange state disposition
  | .commit =>
      match state.generation.cohort with
      | none => noChange state .unknownIdentity
      | some cohort =>
          let commit := commitRecord state event cohort
          acceptedState
            { state with
              generation :=
                { state.generation with
                  status := .committed
                  commit := some commit }
              acceptedTokenClock := commit.acceptedTokenClock
              baseReceipt := commit.receipt
              baseReceiptDigest := commit.receiptDigest
              lastResult := some commit.result }

def transitionCommit
    (state : RunState) (event : CommitGenerationEvent) : StepResult :=
  applyCommitDecision state event (decideCommit state event)

def transitionPublish
    (state : RunState) (event : PublishResultEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      match state.mailbox with
      | some mailbox =>
          if mailbox.result == event.result &&
              mailbox.resultDigest == event.resultDigest &&
              mailbox.commitReceipt == event.commitReceipt &&
              mailbox.publicationReceipt == event.publicationReceipt &&
              mailbox.publicationDigest == event.publicationDigest then
            noChange state .identicalDuplicate
          else
            noChange state .conflictingDuplicate
      | none =>
          match state.generation.commit with
          | none => noChange state .deferred
          | some commit =>
              if !event.verified || !event.finite ||
                  !validDigest event.resultDigest ||
                  !validDigest event.publicationDigest then
                noChange state .corruptNonfinite
              else if event.result != commit.result ||
                  event.resultDigest != commit.resultDigest ||
                  event.commitReceipt != commit.receipt then
                noChange state .conflictingDuplicate
              else
                let mailbox : MailboxRecord :=
                  { generation := commit.generation
                    result := event.result
                    resultDigest := event.resultDigest
                    commitReceipt := event.commitReceipt
                    publicationReceipt := event.publicationReceipt
                    publicationDigest := event.publicationDigest }
                acceptedState { state with mailbox := some mailbox }

def transitionTrainerApply
    (state : RunState) (event : TrainerApplyEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      match state.mailbox, findPeer? state event.worker with
      | none, _ => noChange state .deferred
      | _, none => noChange state .staleIncarnation
      | some mailbox, some peer =>
          if peer.node != event.node ||
              peer.incarnation != event.peerIncarnation then
            noChange state .staleIncarnation
          else
            match peer.trainers.find? (fun trainer =>
                trainer.trainer == event.trainer) with
            | none => noChange state .staleIncarnation
            | some trainer =>
                if trainer.incarnation != event.trainerIncarnation then
                  noChange state .staleIncarnation
                else if mailbox.result != event.result ||
                    mailbox.resultDigest != event.resultDigest ||
                    !validDigest event.receiptDigest then
                  noChange state .corruptNonfinite
                else
                  match state.applyReceipts.find? (fun receipt =>
                      receipt.generation == event.context.generation &&
                      receipt.node == event.node &&
                      receipt.trainer == event.trainer &&
                      receipt.result == event.result) with
                  | some receipt =>
                      if receipt.receipt == event.receipt &&
                          receipt.receiptDigest == event.receiptDigest &&
                          receipt.trainerIncarnation ==
                            event.trainerIncarnation then
                        noChange state .identicalDuplicate
                      else
                        noChange state .conflictingDuplicate
                  | none =>
                      let receipt : ApplyReceipt :=
                        { generation := event.context.generation
                          node := event.node
                          worker := event.worker
                          peerIncarnation := event.peerIncarnation
                          trainer := event.trainer
                          trainerIncarnation :=
                            event.trainerIncarnation
                          result := event.result
                          resultDigest := event.resultDigest
                          receipt := event.receipt
                          receiptDigest := event.receiptDigest }
                      acceptedState
                        { state with
                          applyReceipts :=
                            state.applyReceipts ++ [receipt] }

def matchingApplyReceipts
    (state : RunState) (event : ReduceNodeApplyEvent) :
    List ApplyReceipt :=
  state.applyReceipts.filter fun receipt =>
    receipt.generation == event.context.generation &&
    receipt.node == event.node &&
    receipt.worker == event.worker &&
    receipt.peerIncarnation == event.peerIncarnation &&
    receipt.result == event.result &&
    receipt.resultDigest == event.resultDigest

def nodeApplyAdmissionGate
    (state : RunState) (event : ReduceNodeApplyEvent)
    (peer : PeerRecord) : Bool :=
  let matching := matchingApplyReceipts state event
  peer.trainers.length == 8 &&
  matching.length == 8 &&
  uniqueBy (matching.map (·.trainer)) &&
  (peer.trainers.map (·.trainer)).all
    (fun trainer =>
      matching.any (fun receipt => receipt.trainer == trainer))

inductive NodeApplyDecision where
  | reject (disposition : Disposition)
  | reduce
  deriving Repr, BEq, DecidableEq

def decideNodeApply
    (state : RunState) (event : ReduceNodeApplyEvent) :
    NodeApplyDecision :=
  match currentGenerationDisposition? state event.context with
  | some disposition => .reject disposition
  | none =>
      match state.nodeApplyReceipts.find? (fun receipt =>
          receipt.generation == event.context.generation &&
          receipt.node == event.node) with
      | some receipt =>
          if receipt.worker == event.worker &&
              receipt.peerIncarnation == event.peerIncarnation &&
              receipt.result == event.result &&
              receipt.resultDigest == event.resultDigest &&
              receipt.trainerReceiptDigest ==
                event.trainerReceiptDigest &&
              receipt.receipt == event.receipt &&
              receipt.receiptDigest == event.receiptDigest then
            .reject .identicalDuplicate
          else
            .reject .conflictingDuplicate
      | none =>
          match state.mailbox, findPeer? state event.worker with
          | none, _ => .reject .deferred
          | _, none => .reject .staleIncarnation
          | some mailbox, some peer =>
              let cohortIncludesNode :=
                match state.generation.cohort with
                | none => false
                | some cohort =>
                    (cohortNodes cohort).any (· == event.node)
              if peer.node != event.node ||
                  peer.incarnation != event.peerIncarnation then
                .reject .staleIncarnation
              else if mailbox.result != event.result ||
                  mailbox.resultDigest != event.resultDigest ||
                  !cohortIncludesNode then
                .reject .conflictingDuplicate
              else if !nodeApplyAdmissionGate state event peer then
                .reject .deferred
              else if !validDigest event.trainerReceiptDigest ||
                  !validDigest event.receiptDigest then
                .reject .corruptNonfinite
              else
                .reduce

def nodeApplyReceipt (event : ReduceNodeApplyEvent) : NodeApplyReceipt :=
  { generation := event.context.generation
    node := event.node
    worker := event.worker
    peerIncarnation := event.peerIncarnation
    result := event.result
    resultDigest := event.resultDigest
    trainerReceiptDigest := event.trainerReceiptDigest
    receipt := event.receipt
    receiptDigest := event.receiptDigest }

def applyNodeApplyDecision
    (state : RunState) (event : ReduceNodeApplyEvent) :
    NodeApplyDecision → StepResult
  | .reject .accepted => noChange state .unknownIdentity
  | .reject disposition => noChange state disposition
  | .reduce =>
      let updated :=
        { state with
          nodeApplyReceipts :=
            state.nodeApplyReceipts ++ [nodeApplyReceipt event] }
      let status :=
        if allCohortNodesApplied updated then .applied
        else updated.generation.status
      acceptedState
        { updated with
          generation := { updated.generation with status } }

def transitionReduceNodeApply
    (state : RunState) (event : ReduceNodeApplyEvent) : StepResult :=
  applyNodeApplyDecision state event (decideNodeApply state event)

def transitionLoss
    (state : RunState) (event : LossEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      if !validEvidence event.evidence then
        noChange state .unknownIdentity
      else
        match findPeer? state event.worker with
        | none => noChange state .identicalDuplicate
        | some peer =>
            if peer.node != event.node ||
                peer.incarnation != event.peerIncarnation then
              noChange state .staleIncarnation
            else
              match event.role with
              | .participant | .service | .manager =>
                  let updatedPeer := { peer with phase := .expire }
                  let reducedStatus :=
                    if state.generation.status == .applied then
                      .committed
                    else state.generation.status
                  acceptedState
                    { state with
                      peers := replacePeer state.peers updatedPeer
                      applyReceipts :=
                        state.applyReceipts.filter
                          (fun receipt => receipt.worker != event.worker)
                      nodeApplyReceipts :=
                        state.nodeApplyReceipts.filter
                          (fun receipt => receipt.worker != event.worker)
                      generation :=
                        { state.generation with
                          status := reducedStatus } }
              | .trainer =>
                  match event.trainer, event.trainerIncarnation with
                  | some trainerId, some trainerIncarnation =>
                      match peer.trainers.find? (fun trainer =>
                          trainer.trainer == trainerId) with
                      | none => noChange state .staleIncarnation
                      | some trainer =>
                          if trainer.incarnation != trainerIncarnation then
                            noChange state .staleIncarnation
                          else
                            let updatedPeer :=
                              { peer with
                                phase := .sync
                                trainers :=
                                  peer.trainers.filter
                                    (fun value =>
                                      value.trainer != trainerId) }
                            let reducedStatus :=
                              if state.generation.status == .applied then
                                .committed
                              else state.generation.status
                            acceptedState
                              { state with
                                peers :=
                                  replacePeer state.peers updatedPeer
                                applyReceipts :=
                                  state.applyReceipts.filter
                                    (fun receipt =>
                                      receipt.node != event.node)
                                nodeApplyReceipts :=
                                  state.nodeApplyReceipts.filter
                                    (fun receipt =>
                                      receipt.node != event.node)
                                generation :=
                                  { state.generation with
                                    status := reducedStatus } }
                  | _, _ => noChange state .unknownIdentity

def transitionRestartPeer
    (state : RunState) (event : RestartPeerEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      match findPeer? state event.worker with
      | none => noChange state .unknownIdentity
      | some peer =>
          let expectedGeneration :=
            match state.generation.commit with
            | none => state.generation.generation
            | some _ => ⟨state.generation.generation.value + 1⟩
          if peer.node != event.node ||
              peer.incarnation != event.oldIncarnation then
            noChange state .staleIncarnation
          else if event.newIncarnation == event.oldIncarnation ||
              event.syncedGeneration != expectedGeneration ||
              !validOpaque event.newIncarnation.value ||
              !validEvidence event.managerEvidence ||
              !validEvidence event.serviceEvidence then
            noChange state .unknownIdentity
          else
            let restarted : PeerRecord :=
              { worker := event.worker
                node := event.node
                incarnation := event.newIncarnation
                phase := .sync
                syncedGeneration := event.syncedGeneration
                leaseUntil := event.leaseUntil
                managerEvidence := event.managerEvidence
                serviceEvidence := event.serviceEvidence
                trainers := [] }
            let reducedStatus :=
              if state.generation.status == .applied then
                .committed
              else state.generation.status
            acceptedState
              { state with
                peers := replacePeer state.peers restarted
                applyReceipts :=
                  state.applyReceipts.filter
                    (fun receipt => receipt.worker != event.worker)
                nodeApplyReceipts :=
                  state.nodeApplyReceipts.filter
                    (fun receipt => receipt.worker != event.worker)
                generation :=
                  { state.generation with status := reducedStatus }
                restartCount := state.restartCount + 1 }

def transitionAbort
    (state : RunState) (event : AbortGenerationEvent) : StepResult :=
  match currentGenerationDisposition? state event.context with
  | some disposition => noChange state disposition
  | none =>
      if !validEvidence event.reasonEvidence then
        noChange state .unknownIdentity
      else if state.generation.status == .committed ||
          state.generation.status == .applied then
        noChange state .generationClosed
      else if state.generation.status == .aborted then
        noChange state .identicalDuplicate
      else
        noChange
          { state with
            generation := { state.generation with status := .aborted } }
          .aborted

/-!
The dispatcher seals authority fields that an event class is not permitted to
change.  This is part of the executable kernel, not a proof-only projection:
trace execution and theorem checking use the same wrappers.  The specialized
wrappers make the independent fence, generation, and token namespaces
explicit.
-/
def sealStableAuthority
    (before : RunState) (result : StepResult) : StepResult :=
  { result with
    state :=
      { result.state with
        authority := before.authority
        generation :=
          { result.state.generation with
            generation := before.generation.generation
            cohort := before.generation.cohort
            commit := before.generation.commit }
        acceptedTokenClock := before.acceptedTokenClock
        baseReceipt := before.baseReceipt
        baseReceiptDigest := before.baseReceiptDigest
        lastResult := before.lastResult
        mailbox := before.mailbox } }

def sealPublicationAuthority
    (before : RunState) (result : StepResult) : StepResult :=
  let sealed := sealStableAuthority before result
  { sealed with
    state := { sealed.state with mailbox := result.state.mailbox } }

def sealCloseAuthority
    (before : RunState) (result : StepResult) : StepResult :=
  { result with
    state :=
      { result.state with
        authority := before.authority
        generation :=
          { result.state.generation with
            generation := before.generation.generation
            cohort :=
              match before.generation.cohort with
              | some cohort => some cohort
              | none => result.state.generation.cohort
            commit := before.generation.commit }
        acceptedTokenClock := before.acceptedTokenClock
        baseReceipt := before.baseReceipt
        baseReceiptDigest := before.baseReceiptDigest
        lastResult := before.lastResult
        mailbox := before.mailbox } }

def sealCommitAuthority
    (before : RunState) (result : StepResult) : StepResult :=
  { result with
    state :=
      { result.state with
        authority := before.authority
        generation :=
          { result.state.generation with
            generation := before.generation.generation
            cohort := before.generation.cohort
            commit :=
              match before.generation.commit with
              | some commit => some commit
              | none => result.state.generation.commit }
        acceptedTokenClock :=
          max before.acceptedTokenClock result.state.acceptedTokenClock
        mailbox := before.mailbox } }

def sealOpenAuthority
    (before : RunState) (result : StepResult) : StepResult :=
  { result with
    state :=
      { result.state with
        authority := before.authority
        generation :=
          { result.state.generation with
            generation :=
              ⟨max before.generation.generation.value
                result.state.generation.generation.value⟩ }
        acceptedTokenClock := before.acceptedTokenClock
        baseReceipt := before.baseReceipt
        baseReceiptDigest := before.baseReceiptDigest
        lastResult := before.lastResult } }

def sealClaimAuthority
    (before : RunState) (result : StepResult) : StepResult :=
  { result with
    state :=
      { result.state with
        authority :=
          { result.state.authority with
            fence :=
              ⟨max before.authority.fence.value
                result.state.authority.fence.value⟩ }
        generation :=
          { result.state.generation with
            generation :=
              ⟨max before.generation.generation.value
                result.state.generation.generation.value⟩ }
        acceptedTokenClock :=
          max before.acceptedTokenClock result.state.acceptedTokenClock
        baseReceipt := before.baseReceipt
        baseReceiptDigest := before.baseReceiptDigest
        lastResult := before.lastResult } }

def enforceNonMutatingDisposition
    (before : RunState) (result : StepResult) : StepResult :=
  match result.disposition with
  | .accepted | .insufficientCohort | .aborted => result
  | disposition => noChange before disposition

def transitionRaw (state : RunState) : Event → StepResult
  | .claimFence event =>
      sealClaimAuthority state (transitionClaimFence state event)
  | .peerTransition event =>
      sealStableAuthority state (transitionPeer state event)
  | .registerTrainer event =>
      sealStableAuthority state (transitionRegisterTrainer state event)
  | .expirePeer event =>
      sealStableAuthority state (transitionExpirePeer state event)
  | .openGeneration event =>
      sealOpenAuthority state (transitionOpen state event)
  | .contribution event =>
      sealStableAuthority state (transitionContribution state event)
  | .closeGeneration event =>
      sealCloseAuthority state (transitionClose state event)
  | .ownerLoss event =>
      sealStableAuthority state (transitionOwnerLoss state event)
  | .commitGeneration event =>
      sealCommitAuthority state (transitionCommit state event)
  | .publishResult event =>
      sealPublicationAuthority state (transitionPublish state event)
  | .trainerApply event =>
      sealStableAuthority state (transitionTrainerApply state event)
  | .reduceNodeApply event =>
      sealStableAuthority state (transitionReduceNodeApply state event)
  | .loss event =>
      sealStableAuthority state (transitionLoss state event)
  | .restartPeer event =>
      sealStableAuthority state (transitionRestartPeer state event)
  | .abortGeneration event =>
      sealStableAuthority state (transitionAbort state event)

/--
The single authoritative pure transition.  It is structurally total: every
typed event produces a state and disposition, including stale, corrupt,
duplicate, closed, late, insufficient, catch-up, and bounded-abort outcomes.
Typed rejection/recovery dispositions are normalized to the exact pre-state.
-/
def transition (state : RunState) (event : Event) : StepResult :=
  enforceNonMutatingDisposition state (transitionRaw state event)

end ResilientProtocol
