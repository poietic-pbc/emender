import ResilientProtocol.Examples
import ResilientProtocol.Conformance

namespace ResilientProtocol

/-!
Permanent differential inputs use generation three because the retained
job-5105811 failure was a generation-three close/commit race.  The opaque
contribution receipt digests below are the production kernel's independently
derived receipts for these exact C-ABI identities; the differential runner
checks them rather than replacing them with model output.
-/

def nativeConformancePolicy : PolicyConfig := examplePolicy

def nativeConformanceAuthority
    (fence : Nat := 5105811)
    (allocation : String := "allocation-job-5105811") : AuthorityIdentity :=
  { run := ⟨"run-native-lean-conformance"⟩
    allocation := ⟨allocation⟩
    fence := ⟨fence⟩
    policyName := nativeConformancePolicy.name
    policySchema := nativeConformancePolicy.schema
    policyDigest := nativeConformancePolicy.digest
    traceSchema := traceSchemaVersion
    traceSchemaDigest := ⟨traceSchemaDigest⟩
    toolchain := toolchainIdentity
    sourceSchema := executionSourceSchema
    sourceDigest := repeatedHex 'b'
    layoutDigest := repeatedHex 'c'
    codeDigest := repeatedHex 'd' }

def nativeConformanceInitial : RunState :=
  initialState nativeConformanceAuthority nativeConformancePolicy ⟨3⟩
    (repeatedHex 'e') (rid "commit-receipt-3") (repeatedHex 'f')
    150793748480 (some (resultId "result-3"))

def nativeDigest (value : String) : Digest := ⟨value⟩

def nativeContext
    (state : RunState)
    (generation : Nat := 3)
    (ownerEpoch : Nat := 0) : EventContext :=
  contextFor state generation 0 ownerEpoch

def nativeOpen
    (state : RunState := nativeConformanceInitial)
    (generation : Nat := 3)
    (baseDigest : Digest := repeatedHex 'e')
    (closeTick : Nat := 20) : ExampleAction :=
  accept s!"generation-{generation}:open" <| .openGeneration
    { context := nativeContext state generation
      baseDigest
      openedAt := ⟨1⟩
      closeTick := ⟨closeTick⟩
      closeEvidence := state.policy.closureEvidence }

def nativeContribution
    (state : RunState)
    (worker node peerInc : String)
    (sequence tokens observedAt : Nat)
    (payload receiptDigest : String)
    (generation : Nat := 3)
    (ownerEpoch : Nat := 0)
    (baseGeneration : Nat := 3)
    (baseDigest : Digest := repeatedHex 'e')
    (receipt : String := "native-contribution") : ContributionEvent :=
  { context := nativeContext state generation ownerEpoch
    worker := workerId worker
    node := nodeId node
    incarnation := incarnation peerInc
    sequence := ⟨sequence⟩
    baseGeneration := ⟨baseGeneration⟩
    baseDigest
    exactTokens := tokens
    envelopeDigest := nativeDigest payload
    payloadDigest := nativeDigest payload
    trainerSetDigest := repeatedHex '7'
    receipt := rid receipt
    receiptDigest := nativeDigest receiptDigest
    localWindowStart := sequence * 40
    localWindowEnd := (sequence + 1) * 40
    commitLag := generation - baseGeneration
    anchorLag := 0
    resultLag := 0
    speculativeLag := 0
    finite := true
    checksumValid := true
    layoutValid := true
    observedAt := ⟨observedAt⟩ }

def nativeNode0Contribution
    (state : RunState := nativeConformanceInitial) : ContributionEvent :=
  nativeContribution state "worker-0" "node-0" "node-0-a" 1
    1966080 10
    (String.ofList (List.replicate 64 '2'))
    "587be10d252fe5640314a6f23dd7d57a1e6eda913864e43294790a657effdee3"
    (receipt := "contribution-receipt-node-0-generation-3")

def nativeNode1Contribution
    (state : RunState := nativeConformanceInitial) : ContributionEvent :=
  nativeContribution state "worker-1" "node-1" "node-1-a" 1
    1968000 11
    (String.ofList (List.replicate 64 '6'))
    "920fef868542e46b38a017682deefa6b44f4b29d6d0721f47cdb17a0c5a4b487"
    (receipt := "contribution-receipt-node-1-generation-3")

def nativeClose
    (state : RunState := nativeConformanceInitial)
    (observedAt : Nat := 12)
    (ownerEpoch : Nat := 0) : ExampleAction :=
  accept "generation-3:finite-close" <| .closeGeneration
    { context := nativeContext state 3 ownerEpoch
      observedAt := ⟨observedAt⟩
      cohortDigest := repeatedHex '0' }

def nativeCommit
    (state : RunState := nativeConformanceInitial)
    (ownerEpoch : Nat := 0) : ExampleAction :=
  accept "generation-3:exact-once-commit" <| .commitGeneration
    { context := nativeContext state 3 ownerEpoch
      cohortDigest := repeatedHex '0'
      result := resultId "result-4"
      resultDigest := repeatedHex 'a'
      receipt := rid "commit-receipt-4"
      receiptDigest := repeatedHex 'b'
      priorReceipt := rid "commit-receipt-3" }

def nativePublish
    (state : RunState := nativeConformanceInitial)
    (ownerEpoch : Nat := 0) : ExampleAction :=
  accept "generation-3:result-publish" <| .publishResult
    { context := nativeContext state 3 ownerEpoch
      result := resultId "result-4"
      resultDigest := repeatedHex 'a'
      commitReceipt := rid "commit-receipt-4"
      publicationReceipt := rid "publication-receipt-4"
      publicationDigest := repeatedHex 'c'
      verified := true
      finite := true }

def nativeTrainerApply
    (state : RunState)
    (node worker peerInc trainerIncPrefix : String)
    (lane : Nat)
    (ownerEpoch : Nat := 0) : TrainerApplyEvent :=
  { context := nativeContext state 3 ownerEpoch
    node := nodeId node
    worker := workerId worker
    peerIncarnation := incarnation peerInc
    trainer := trainerName node lane
    trainerIncarnation := incarnation s!"{trainerIncPrefix}-{lane}"
    result := resultId "result-4"
    resultDigest := repeatedHex 'a'
    receipt := rid s!"apply-{node}-{lane}-{peerInc}"
    receiptDigest :=
      repeatedHex (if node == "node-0" then 'd' else 'e') }

def nativeTrainerApplyActions
    (state : RunState)
    (node worker peerInc trainerIncPrefix : String)
    (ownerEpoch : Nat := 0) : List ExampleAction :=
  (List.range 8).map fun lane =>
    accept s!"{node}:apply-{lane}" <| .trainerApply
      (nativeTrainerApply state node worker peerInc trainerIncPrefix lane
        ownerEpoch)

def nativeReduce
    (state : RunState)
    (node worker peerInc : String)
    (ownerEpoch : Nat := 0) : ExampleAction :=
  accept s!"{node}:eight-to-one-node-apply" <| .reduceNodeApply
    { context := nativeContext state 3 ownerEpoch
      node := nodeId node
      worker := workerId worker
      peerIncarnation := incarnation peerInc
      result := resultId "result-4"
      resultDigest := repeatedHex 'a'
      trainerReceiptDigest :=
        repeatedHex (if node == "node-0" then '1' else '2')
      receipt := rid s!"node-apply-{node}-{peerInc}"
      receiptDigest :=
        repeatedHex (if node == "node-0" then '3' else '4') }

def nativeReadyNext
    (state : RunState)
    (node worker peerInc : String)
    (fromPhase : PeerPhase := .ready) : ExampleAction :=
  accept s!"{node}:ready-generation-4" <| .peerTransition
    { context := nativeContext state
      worker := workerId worker
      node := nodeId node
      incarnation := incarnation peerInc
      fromPhase
      toPhase := .ready
      syncedGeneration := ⟨4⟩
      leaseUntil := ⟨2000⟩
      managerEvidence := ev s!"manager-{node}-{peerInc}-generation-4"
      serviceEvidence := ev s!"service-{node}-{peerInc}-generation-4" }

def nativeBootstrap : List ExampleAction :=
  bootstrapPeerActions nativeConformanceInitial
      "worker-0" "node-0" "node-0-a" 3 ++
    bootstrapPeerActions nativeConformanceInitial
      "worker-1" "node-1" "node-1-a" 3

def nativeCommonOpen : List ExampleAction :=
  nativeBootstrap ++ [nativeOpen]

def nativeJob5105811Scenario : ExampleScenario :=
  let serviceLoss : LossEvent :=
    { context := nativeContext nativeConformanceInitial
      role := .service
      worker := workerId "worker-0"
      node := nodeId "node-0"
      peerIncarnation := incarnation "node-0-a"
      trainer := none
      trainerIncarnation := none
      evidence := ev "job-5105811-node-0-trainer-service-loss" }
  let restart : RestartPeerEvent :=
    { context := nativeContext nativeConformanceInitial
      worker := workerId "worker-0"
      node := nodeId "node-0"
      oldIncarnation := incarnation "node-0-a"
      newIncarnation := incarnation "node-0-b"
      managerEvidence := ev "job-5105811-manager-node-0-b"
      serviceEvidence := ev "job-5105811-service-node-0-b"
      syncedGeneration := ⟨4⟩
      leaseUntil := ⟨2000⟩ }
  let late :=
    { nativeNode1Contribution with
      incarnation := incarnation "node-1-a"
      sequence := ⟨5105811⟩
      envelopeDigest := repeatedHex '9'
      payloadDigest := repeatedHex '9'
      receipt := rid "job-5105811-concurrent-closed-generation"
      receiptDigest := repeatedHex '9'
      observedAt := ⟨21⟩ }
  let restartedTrainers :=
    (List.range 8).map fun lane =>
      accept s!"node-0:restart-trainer-{lane}" <| .registerTrainer
        { context := nativeContext nativeConformanceInitial
          worker := workerId "worker-0"
          node := nodeId "node-0"
          peerIncarnation := incarnation "node-0-b"
          trainer := trainerName "node-0" lane
          trainerIncarnation := incarnation s!"trainer-node-0-b-{lane}" }
  let nextContribution :=
    nativeContribution nativeConformanceInitial
      "worker-0" "node-0" "node-0-b" 2 1966080 30
      (String.ofList (List.replicate 64 '9'))
      "6c3cd3c4dc9a257eb902fd3c3076d6e942022dcf9cd34df1f59684276d10ff27"
      (generation := 4) (baseGeneration := 4)
      (baseDigest := repeatedHex 'a')
      (receipt := "rejoined-node-0-generation-4")
  { name := "native-job-5105811-generation-3-close-restart-rejoin"
    initial := nativeConformanceInitial
    actions :=
      nativeCommonOpen ++
      [ accept "generation-3:node-0-contribution"
          (.contribution nativeNode0Contribution)
      , accept "generation-3:node-1-contribution"
          (.contribution nativeNode1Contribution)
      , nativeClose
      , nativeCommit
      , nativePublish
      , accept "node-0:service-loss" (.loss serviceLoss)
      , accept "node-0:new-incarnation" (.restartPeer restart)
      , expect "job-5105811:closed-generation-catch-up" .catchUp
          (.contribution late) ] ++
      restartedTrainers ++
      nativeTrainerApplyActions nativeConformanceInitial
        "node-0" "worker-0" "node-0-b" "trainer-node-0-b" ++
      [nativeReduce nativeConformanceInitial
        "node-0" "worker-0" "node-0-b"] ++
      nativeTrainerApplyActions nativeConformanceInitial
        "node-1" "worker-1" "node-1-a" "trainer-inc-node-1" ++
      [ nativeReduce nativeConformanceInitial
          "node-1" "worker-1" "node-1-a"
      , nativeReadyNext nativeConformanceInitial
          "node-0" "worker-0" "node-0-b" (fromPhase := .sync)
      , nativeReadyNext nativeConformanceInitial
          "node-1" "worker-1" "node-1-a"
      , nativeOpen nativeConformanceInitial 4 (repeatedHex 'a') 40
      , accept "node-0:new-incarnation-generation-4"
          (.contribution nextContribution) ] }

def nativeRejectionScenario : ExampleScenario :=
  let conflict :=
    { nativeNode0Contribution with
      payloadDigest := repeatedHex '9'
      envelopeDigest := repeatedHex '9' }
  let staleFence :=
    { nativeNode1Contribution with
      context :=
        { nativeNode1Contribution.context with fence := ⟨5105810⟩ }
      sequence := ⟨7⟩ }
  let staleIncarnation :=
    { nativeNode1Contribution with
      incarnation := incarnation "node-1-superseded"
      sequence := ⟨8⟩ }
  let corrupt :=
    { nativeNode1Contribution with
      sequence := ⟨9⟩
      finite := false }
  let late :=
    { nativeNode1Contribution with
      sequence := ⟨10⟩
      observedAt := ⟨21⟩ }
  let conflictingCommit :=
    match nativeCommit.event with
    | .commitGeneration commit =>
        .commitGeneration
          { commit with
            result := resultId "conflicting-result-4"
            resultDigest := repeatedHex '9'
            receipt := rid "conflicting-commit-receipt-4"
            receiptDigest := repeatedHex '9' }
    | event => event
  { name := "native-duplicate-conflict-stale-late-corrupt"
    initial := nativeConformanceInitial
    actions :=
      nativeCommonOpen ++
      [ accept "node-0:accepted" (.contribution nativeNode0Contribution)
      , expect "node-0:identical" .identicalDuplicate
          (.contribution nativeNode0Contribution)
      , expect "node-0:conflict" .conflictingDuplicate
          (.contribution conflict)
      , expect "node-1:stale-fence" .staleFence
          (.contribution staleFence)
      , expect "node-1:stale-incarnation" .staleIncarnation
          (.contribution staleIncarnation)
      , expect "node-1:corrupt" .corruptNonfinite
          (.contribution corrupt)
      , accept "node-1:accepted" (.contribution nativeNode1Contribution)
      , nativeClose
      , expect "node-1:late-after-close" .generationClosed
          (.contribution late)
      , nativeCommit
      , expect "commit:identical" .identicalDuplicate nativeCommit.event
      , expect "commit:conflicting" .conflictingDuplicate
          conflictingCommit ] }

def nativeLeasedReadyScenario : ExampleScenario :=
  let node1Bootstrap :=
    bootstrapPeerActions nativeConformanceInitial
      "worker-1" "node-1" "node-1-a" 3
  let delayedNode1 := node1Bootstrap.dropLast
  let node1Ready :=
    match node1Bootstrap.getLast? with
    | some value => value
    | none => nativeOpen
  let expire : ExpirePeerEvent :=
    { context := nativeContext nativeConformanceInitial
      worker := workerId "worker-0"
      node := nodeId "node-0"
      incarnation := incarnation "node-0-a"
      observedAt := ⟨1000⟩ }
  { name := "native-leased-ready-delay-expiry-insufficient"
    initial := nativeConformanceInitial
    actions :=
      bootstrapPeerActions nativeConformanceInitial
        "worker-0" "node-0" "node-0-a" 3 ++
      delayedNode1 ++
      [ nativeOpen
      , node1Ready
      , expect "delayed-ready-not-in-frozen-snapshot"
          .retryNextGeneration (.contribution nativeNode1Contribution)
      , accept "snapshot-member-contribution"
          (.contribution nativeNode0Contribution)
      , accept "lease-expiry" (.expirePeer expire)
      , expect "deadline-below-floor" .insufficientCohort
          (nativeClose nativeConformanceInitial 20).event ] }

def nativeOwnerAbortScenario : ExampleScenario :=
  let loss epoch : ExampleAction :=
    expect s!"owner-loss-{epoch}" (if epoch < 2 then .accepted else .aborted) <|
      .ownerLoss
        { context := nativeContext nativeConformanceInitial 3 epoch
          owner := workerId "worker-0"
          ownerIncarnation := incarnation "node-0-a"
          replayEvidence := ev s!"owner-replay-{epoch}"
          observedAt := ⟨13 + epoch⟩ }
  { name := "native-owner-replay-reassignment-abort"
    initial := nativeConformanceInitial
    actions :=
      nativeCommonOpen ++
      [ accept "node-0:accepted" (.contribution nativeNode0Contribution)
      , accept "node-1:accepted" (.contribution nativeNode1Contribution)
      , nativeClose
      , loss 0
      , loss 1
      , loss 2 ] }

def nativeRoleRestartScenario
    (role : LossRole) (phaseName : String)
    (beforeActions afterActions : List ExampleAction) : ExampleScenario :=
  let roleName :=
    match role with
    | .participant => "participant"
    | .service => "service"
    | .manager => "manager"
    | .trainer => "trainer"
  let loss : LossEvent :=
    { context := nativeContext nativeConformanceInitial
      role
      worker := workerId "worker-0"
      node := nodeId "node-0"
      peerIncarnation := incarnation "node-0-a"
      trainer :=
        if role == .trainer then some (trainerName "node-0" 0) else none
      trainerIncarnation :=
        if role == .trainer then
          some (trainerIncarnation "node-0" 0)
        else none
      evidence := ev s!"{roleName}-{phaseName}-loss" }
  let restart : RestartPeerEvent :=
    { context := nativeContext nativeConformanceInitial
      worker := workerId "worker-0"
      node := nodeId "node-0"
      oldIncarnation := incarnation "node-0-a"
      newIncarnation := incarnation s!"node-0-{roleName}-{phaseName}"
      managerEvidence := ev s!"manager-{roleName}-{phaseName}"
      serviceEvidence := ev s!"service-{roleName}-{phaseName}"
      syncedGeneration :=
        if phaseName == "committed-apply" then ⟨4⟩ else ⟨3⟩
      leaseUntil := ⟨2000⟩ }
  { name := s!"native-{roleName}-restart-{phaseName}"
    initial := nativeConformanceInitial
    actions :=
      beforeActions ++
      [ accept s!"{roleName}:loss-{phaseName}" (.loss loss)
      , accept s!"{roleName}:restart-{phaseName}" (.restartPeer restart) ] ++
      afterActions }

def nativeParticipantOpenRestart : ExampleScenario :=
  nativeRoleRestartScenario .participant "open" nativeCommonOpen []

def nativeClosedActions : List ExampleAction :=
  nativeCommonOpen ++
    [ accept "node-0:accepted" (.contribution nativeNode0Contribution)
    , accept "node-1:accepted" (.contribution nativeNode1Contribution)
    , nativeClose ]

def nativeCommittedActions : List ExampleAction :=
  nativeClosedActions ++ [nativeCommit, nativePublish]

def nativeServiceOpenRestart : ExampleScenario :=
  nativeRoleRestartScenario .service "open" nativeCommonOpen []

def nativeServiceClosedRestart : ExampleScenario :=
  nativeRoleRestartScenario .service "closed"
    nativeClosedActions []

def nativeServiceCommittedRestart : ExampleScenario :=
  nativeRoleRestartScenario .service "committed-apply"
    nativeCommittedActions []

def nativeManagerOpenRestart : ExampleScenario :=
  nativeRoleRestartScenario .manager "open" nativeCommonOpen []

def nativeManagerClosedRestart : ExampleScenario :=
  nativeRoleRestartScenario .manager "closed"
    nativeClosedActions []

def nativeManagerCommittedRestart : ExampleScenario :=
  nativeRoleRestartScenario .manager "committed-apply"
    nativeCommittedActions []

def nativeTrainerOpenRestart : ExampleScenario :=
  nativeRoleRestartScenario .trainer "open" nativeCommonOpen []

def nativeTrainerClosedRestart : ExampleScenario :=
  nativeRoleRestartScenario .trainer "closed"
    nativeClosedActions []

def nativeTrainerApplyRestart : ExampleScenario :=
  let firstApply :=
    accept "node-0:partial-apply-0" <| .trainerApply
      (nativeTrainerApply nativeConformanceInitial
        "node-0" "worker-0" "node-0-a" "trainer-inc-node-0" 0)
  nativeRoleRestartScenario .trainer "committed-apply"
    (nativeCommittedActions ++ [firstApply])
    []

def nativeFreshFenceScenario : ExampleScenario :=
  let authority :=
    nativeConformanceAuthority 5105812 "allocation-job-5105811-resume"
  let claim : ClaimFenceEvent :=
    { authority
      policy := nativeConformancePolicy
      baseGeneration := ⟨4⟩
      baseDigest := repeatedHex 'a'
      baseReceipt := rid "commit-receipt-4"
      baseReceiptDigest := repeatedHex 'b'
      acceptedTokenClock := 150797682560
      lastResult := some (resultId "result-4") }
  let stale :=
    { nativeNode0Contribution with sequence := ⟨88⟩ }
  { name := "native-fresh-fence-recovery"
    initial := nativeConformanceInitial
    actions :=
      nativeCommonOpen ++
      [ accept "node-0:accepted" (.contribution nativeNode0Contribution)
      , accept "node-1:accepted" (.contribution nativeNode1Contribution)
      , nativeClose
      , nativeCommit
      , accept "fresh-fence" (.claimFence claim)
      , expect "old-fence-stale" .staleFence (.contribution stale) ] }

def nativeConformanceScenarios : List ExampleScenario :=
  [ nativeJob5105811Scenario
  , nativeRejectionScenario
  , nativeLeasedReadyScenario
  , nativeOwnerAbortScenario
  , nativeParticipantOpenRestart
  , nativeServiceOpenRestart
  , nativeServiceClosedRestart
  , nativeServiceCommittedRestart
  , nativeManagerOpenRestart
  , nativeManagerClosedRestart
  , nativeManagerCommittedRestart
  , nativeTrainerOpenRestart
  , nativeTrainerClosedRestart
  , nativeTrainerApplyRestart
  , nativeFreshFenceScenario ]

def nativeConformanceTraces : List TraceDocument :=
  nativeConformanceScenarios.map scenarioTrace

end ResilientProtocol
