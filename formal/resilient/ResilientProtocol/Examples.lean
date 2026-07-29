import ResilientProtocol.Trace

namespace ResilientProtocol

/-! ## Small opaque identities used by executable protocol examples -/

def repeatedHex (character : Char) : Digest :=
  ⟨String.ofList (List.replicate 64 character)⟩

def ev (value : String) : EvidenceId := ⟨value⟩
def rid (value : String) : ReceiptId := ⟨value⟩
def resultId (value : String) : ResultId := ⟨value⟩
def workerId (value : String) : WorkerId := ⟨value⟩
def nodeId (value : String) : NodeId := ⟨value⟩
def trainerId (value : String) : TrainerId := ⟨value⟩
def incarnation (value : String) : Incarnation := ⟨value⟩

def examplePolicy : PolicyConfig :=
  PolicyConfig.asyncV21 (repeatedHex 'a') .floorOrDeadline
    (ev "two-node-v21-finite-close-v1")

def exampleScalePolicy : PolicyConfig :=
  PolicyConfig.asyncV21 (repeatedHex 'a') .fixedSnapshotDeadline
    (ev "reviewed-v21s17-close-evidence")

def strictExamplePolicy : PolicyConfig :=
  PolicyConfig.strictV1 (repeatedHex '9') 1 1
    (ev "strict-v1-finite-deadline")

def exampleAuthorityFor
    (policy : PolicyConfig)
    (fence : Nat := 1)
    (allocation : String := "allocation-1") : AuthorityIdentity :=
  { run := ⟨"run-resilient-example"⟩
    allocation := ⟨allocation⟩
    fence := ⟨fence⟩
    policyName := policy.name
    policySchema := policy.schema
    policyDigest := policy.digest
    traceSchema := traceSchemaVersion
    traceSchemaDigest := ⟨traceSchemaDigest⟩
    toolchain := toolchainIdentity
    sourceSchema := executionSourceSchema
    sourceDigest := repeatedHex 'b'
    layoutDigest := repeatedHex 'c'
    codeDigest := repeatedHex 'd' }

def exampleInitialFor (policy : PolicyConfig) : RunState :=
  initialState (exampleAuthorityFor policy) policy ⟨0⟩ (repeatedHex 'e')
    (rid "base-receipt-0") (repeatedHex 'f') 150793748480 none

def exampleInitial : RunState := exampleInitialFor examplePolicy

/--
The permanent job-5105811 regression starts from the last applied authority
at generation 3.  It uses the same transition function and example trace
runner as every other executable scenario.
-/
def job5105811Initial : RunState :=
  initialState (exampleAuthorityFor examplePolicy) examplePolicy ⟨3⟩
    (repeatedHex 'e') (rid "base-receipt-0") (repeatedHex 'f')
    150793748480 none

def contextFor
    (state : RunState)
    (generation : Nat := 0)
    (attempt : Nat := 0)
    (ownerEpoch : Nat := 0) : EventContext :=
  { run := state.authority.run
    allocation := state.authority.allocation
    fence := state.authority.fence
    policyName := state.policy.name
    policySchema := state.policy.schema
    policyDigest := state.policy.digest
    traceSchema := state.authority.traceSchema
    traceSchemaDigest := state.authority.traceSchemaDigest
    toolchain := state.authority.toolchain
    sourceSchema := state.authority.sourceSchema
    sourceDigest := state.authority.sourceDigest
    layoutDigest := state.authority.layoutDigest
    codeDigest := state.authority.codeDigest
    generation := ⟨generation⟩
    attempt := ⟨attempt⟩
    ownerEpoch := ⟨ownerEpoch⟩ }

def trainerName (node : String) (lane : Nat) : TrainerId :=
  trainerId s!"trainer-{node}-{lane}"

def trainerIncarnation (node : String) (lane : Nat) : Incarnation :=
  incarnation s!"trainer-inc-{node}-{lane}"

structure ExampleAction where
  name : String
  event : Event
  expected : Disposition
  deriving Repr, BEq, DecidableEq

def accept (name : String) (event : Event) : ExampleAction :=
  { name, event, expected := .accepted }

def expect
    (name : String) (expected : Disposition) (event : Event) :
    ExampleAction :=
  { name, event, expected }

def bootstrapPeerActions
    (state : RunState)
    (worker node peerInc : String)
    (generation : Nat := 0) : List ExampleAction :=
  let context := contextFor state generation
  let workerValue := workerId worker
  let nodeValue := nodeId node
  let peerIncValue := incarnation peerInc
  let manager := ev s!"manager-{node}-{peerInc}"
  let service := ev s!"service-{node}-{peerInc}"
  let discover :=
    accept s!"{node}:discover" <| .peerTransition
      { context
        worker := workerValue
        node := nodeValue
        incarnation := peerIncValue
        fromPhase := .discover
        toPhase := .discover
        syncedGeneration := ⟨generation⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := manager
        serviceEvidence := service }
  let boot :=
    accept s!"{node}:boot" <| .peerTransition
      { context
        worker := workerValue
        node := nodeValue
        incarnation := peerIncValue
        fromPhase := .discover
        toPhase := .boot
        syncedGeneration := ⟨generation⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := manager
        serviceEvidence := service }
  let trainers :=
    (List.range 8).map fun lane =>
      accept s!"{node}:trainer-{lane}" <| .registerTrainer
        { context
          worker := workerValue
          node := nodeValue
          peerIncarnation := peerIncValue
          trainer := trainerName node lane
          trainerIncarnation := trainerIncarnation node lane }
  let sync :=
    accept s!"{node}:sync" <| .peerTransition
      { context
        worker := workerValue
        node := nodeValue
        incarnation := peerIncValue
        fromPhase := .boot
        toPhase := .sync
        syncedGeneration := ⟨generation⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := manager
        serviceEvidence := service }
  let ready :=
    accept s!"{node}:leased-ready" <| .peerTransition
      { context
        worker := workerValue
        node := nodeValue
        incarnation := peerIncValue
        fromPhase := .sync
        toPhase := .ready
        syncedGeneration := ⟨generation⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := manager
        serviceEvidence := service }
  [discover, boot] ++ trainers ++ [sync, ready]

def openGenerationAction
    (state : RunState)
    (generation : Nat := 0)
    (baseDigest : Digest := repeatedHex 'e')
    (closeTick : Nat := 20)
    (openedAt : Nat := 1) : ExampleAction :=
  accept s!"generation-{generation}:open" <| .openGeneration
    { context := contextFor state generation
      baseDigest
      openedAt := ⟨openedAt⟩
      closeTick := ⟨closeTick⟩
      closeEvidence := state.policy.closureEvidence }

def contributionEvent
    (state : RunState)
    (worker node peerInc : String)
    (sequence tokens observedAt : Nat)
    (envelope payload trainerSet receiptDigest : Char)
    (generation : Nat := 0)
    (ownerEpoch : Nat := 0)
    (baseGeneration : Nat := 0)
    (baseDigest : Digest := repeatedHex 'e')
    (receipt : String := "contribution-receipt")
    (finite : Bool := true) : ContributionEvent :=
  { context := contextFor state generation 0 ownerEpoch
    worker := workerId worker
    node := nodeId node
    incarnation := incarnation peerInc
    sequence := ⟨sequence⟩
    baseGeneration := ⟨baseGeneration⟩
    baseDigest
    exactTokens := tokens
    envelopeDigest := repeatedHex envelope
    payloadDigest := repeatedHex payload
    trainerSetDigest := repeatedHex trainerSet
    receipt := rid receipt
    receiptDigest := repeatedHex receiptDigest
    localWindowStart := sequence * 40
    localWindowEnd := (sequence + 1) * 40
    commitLag := generation - baseGeneration
    anchorLag := 0
    resultLag := 0
    speculativeLag := 0
    finite
    checksumValid := true
    layoutValid := true
    observedAt := ⟨observedAt⟩ }

def job5105811LateContribution : ContributionEvent :=
  contributionEvent job5105811Initial "worker-1" "node-1" "peer-inc-1"
    5105811 1968000 13 '9' '6' '7' '8'
    (generation := 3) (baseGeneration := 3)
    (receipt := "job-5105811-late-generation-closed")

def job5105811NextContribution : ContributionEvent :=
  contributionEvent job5105811Initial "worker-0" "node-0"
    "peer-inc-0-restarted" 1 1966080 30 '9' '8' '7' '6'
    (generation := 4) (baseGeneration := 4)
    (baseDigest := repeatedHex 'a')
    (receipt := "rejoined-next-generation")

def node0Contribution (state : RunState) : ContributionEvent :=
  contributionEvent state "worker-0" "node-0" "peer-inc-0" 0
    1966080 10 '1' '2' '3' '4' (receipt := "contribution-receipt-0")

def node1Contribution (state : RunState) : ContributionEvent :=
  contributionEvent state "worker-1" "node-1" "peer-inc-1" 0
    1968000 11 '5' '6' '7' '8' (receipt := "contribution-receipt-1")

def closeAction
    (state : RunState)
    (observedAt : Nat := 12)
    (ownerEpoch : Nat := 0)
    (generation : Nat := 0) : ExampleAction :=
  accept s!"generation-{generation}:finite-close" <| .closeGeneration
    { context := contextFor state generation 0 ownerEpoch
      observedAt := ⟨observedAt⟩
      cohortDigest := repeatedHex '0' }

def commitAction
    (state : RunState)
    (ownerEpoch : Nat := 0)
    (generation : Nat := 0) : ExampleAction :=
  accept s!"generation-{generation}:exact-once-commit" <| .commitGeneration
    { context := contextFor state generation 0 ownerEpoch
      cohortDigest := repeatedHex '0'
      result := resultId "result-1"
      resultDigest := repeatedHex 'a'
      receipt := rid "commit-receipt-1"
      receiptDigest := repeatedHex 'b'
      priorReceipt := rid "base-receipt-0" }

def publishAction
    (state : RunState)
    (ownerEpoch : Nat := 0)
    (generation : Nat := 0) : ExampleAction :=
  accept s!"generation-{generation}:mailbox-publish" <| .publishResult
    { context := contextFor state generation 0 ownerEpoch
      result := resultId "result-1"
      resultDigest := repeatedHex 'a'
      commitReceipt := rid "commit-receipt-1"
      publicationReceipt := rid "publication-receipt-1"
      publicationDigest := repeatedHex 'c'
      verified := true
      finite := true }

def trainerApplyEvent
    (state : RunState)
    (node worker peerInc : String)
    (lane : Nat)
    (ownerEpoch : Nat := 0)
    (receiptSuffix : String := "")
    (generation : Nat := 0) : TrainerApplyEvent :=
  { context := contextFor state generation 0 ownerEpoch
    node := nodeId node
    worker := workerId worker
    peerIncarnation := incarnation peerInc
    trainer := trainerName node lane
    trainerIncarnation := trainerIncarnation node lane
    result := resultId "result-1"
    resultDigest := repeatedHex 'a'
    receipt := rid s!"apply-{node}-{lane}{receiptSuffix}"
    receiptDigest :=
      repeatedHex (if node == "node-0" then 'd' else 'e') }

def trainerApplyActions
    (state : RunState)
    (node worker peerInc : String)
    (ownerEpoch : Nat := 0)
    (generation : Nat := 0) : List ExampleAction :=
  (List.range 8).map fun lane =>
    accept s!"{node}:apply-{lane}" <|
      .trainerApply
        (trainerApplyEvent state node worker peerInc lane ownerEpoch ""
          generation)

def reduceAction
    (state : RunState)
    (node worker peerInc : String)
    (ownerEpoch : Nat := 0)
    (generation : Nat := 0) : ExampleAction :=
  accept s!"{node}:eight-to-one-node-apply" <| .reduceNodeApply
    { context := contextFor state generation 0 ownerEpoch
      node := nodeId node
      worker := workerId worker
      peerIncarnation := incarnation peerInc
      result := resultId "result-1"
      resultDigest := repeatedHex 'a'
      trainerReceiptDigest :=
        repeatedHex (if node == "node-0" then '1' else '2')
      receipt := rid s!"node-apply-{node}"
      receiptDigest :=
        repeatedHex (if node == "node-0" then '3' else '4') }

structure ExampleScenario where
  name : String
  initial : RunState
  actions : List ExampleAction
  deriving Repr

def commonBootstrap (state : RunState) : List ExampleAction :=
  bootstrapPeerActions state "worker-0" "node-0" "peer-inc-0" ++
  bootstrapPeerActions state "worker-1" "node-1" "peer-inc-1" ++
  [openGenerationAction state]

def normalScenario : ExampleScenario :=
  let node0 := node0Contribution exampleInitial
  let conflict :=
    { node0 with
      envelopeDigest := repeatedHex '9'
      payloadDigest := repeatedHex '9' }
  let staleFence :=
    { node0 with
      context :=
        { node0.context with fence := ⟨0⟩ }
      sequence := ⟨7⟩
      envelopeDigest := repeatedHex '6' }
  let staleIncarnation :=
    { node0 with
      incarnation := incarnation "superseded-peer-inc"
      sequence := ⟨8⟩
      envelopeDigest := repeatedHex '7' }
  let corrupt :=
    { node1Contribution exampleInitial with
      sequence := ⟨10⟩
      envelopeDigest := repeatedHex '8'
      finite := false }
  let lagThree :=
    { node1Contribution exampleInitial with
      sequence := ⟨11⟩
      envelopeDigest := repeatedHex 'f'
      commitLag := 3 }
  let late :=
    contributionEvent exampleInitial "worker-0" "node-0" "peer-inc-0"
      9 1966080 21 '6' '5' '4' '3'
      (receipt := "late-after-close")
  let firstApply :=
    trainerApplyEvent exampleInitial "node-0" "worker-0" "peer-inc-0" 0
  let conflictingApply :=
    { firstApply with
      receipt := rid "apply-node-0-0-conflict"
      receiptDigest := repeatedHex 'f' }
  let conflictingCommit :=
    match (commitAction exampleInitial).event with
    | .commitGeneration commit =>
        .commitGeneration
          { commit with
            result := resultId "conflicting-result"
            resultDigest := repeatedHex 'f' }
    | event => event
  let conflictingPublish :=
    match (publishAction exampleInitial).event with
    | .publishResult publication =>
        .publishResult
          { publication with
            publicationReceipt := rid "conflicting-publication"
            publicationDigest := repeatedHex 'f' }
    | event => event
  let readyNext node worker peerInc : ExampleAction :=
    accept s!"{node}:ready-generation-1" <| .peerTransition
      { context := contextFor exampleInitial
        worker := workerId worker
        node := nodeId node
        incarnation := incarnation peerInc
        fromPhase := .ready
        toPhase := .ready
        syncedGeneration := ⟨1⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := ev s!"manager-{node}-next"
        serviceEvidence := ev s!"service-{node}-next" }
  { name := "normal-commit-apply-and-rejections"
    initial := exampleInitial
    actions :=
      commonBootstrap exampleInitial ++
      [ accept "contribution-node-0" (.contribution node0)
      , expect "identical-contribution-replay" .identicalDuplicate
          (.contribution node0)
      , expect "conflicting-contribution-replay" .conflictingDuplicate
          (.contribution conflict)
      , expect "stale-fence" .staleFence (.contribution staleFence)
      , expect "stale-incarnation" .staleIncarnation
          (.contribution staleIncarnation)
      , expect "corrupt-nonfinite" .corruptNonfinite
          (.contribution corrupt)
      , expect "async-v21-lag-three-catch-up" .catchUp
          (.contribution lagThree)
      , accept "contribution-node-1"
          (.contribution (node1Contribution exampleInitial))
      , closeAction exampleInitial
      , expect "late-after-close" .generationClosed (.contribution late)
      , commitAction exampleInitial
      , expect "duplicate-commit-receipt" .identicalDuplicate
          (commitAction exampleInitial).event
      , expect "conflicting-commit-receipt" .conflictingDuplicate
          conflictingCommit
      , publishAction exampleInitial
      , expect "duplicate-mailbox-publication" .identicalDuplicate
          (publishAction exampleInitial).event
      , expect "conflicting-mailbox-publication" .conflictingDuplicate
          conflictingPublish
      , accept "node-0:first-apply" (.trainerApply firstApply)
      , expect "duplicate-apply-receipt" .identicalDuplicate
          (.trainerApply firstApply)
      , expect "conflicting-apply-receipt" .conflictingDuplicate
          (.trainerApply conflictingApply) ] ++
      (trainerApplyActions exampleInitial "node-0" "worker-0" "peer-inc-0").drop 1 ++
      [reduceAction exampleInitial "node-0" "worker-0" "peer-inc-0"] ++
      trainerApplyActions exampleInitial "node-1" "worker-1" "peer-inc-1" ++
      [ reduceAction exampleInitial "node-1" "worker-1" "peer-inc-1"
      , readyNext "node-0" "worker-0" "peer-inc-0"
      , readyNext "node-1" "worker-1" "peer-inc-1" ] }

def peerLossScenario : ExampleScenario :=
  let loss : LossEvent :=
    { context := contextFor exampleInitial
      role := .manager
      worker := workerId "worker-0"
      node := nodeId "node-0"
      peerIncarnation := incarnation "peer-inc-0"
      trainer := none
      trainerIncarnation := none
      evidence := ev "manager-loss-before-close" }
  { name := "peer-loss-insufficient-cohort"
    initial := exampleInitial
    actions :=
      commonBootstrap exampleInitial ++
      [ accept "manager-loss" (.loss loss)
      , expect "lost-peer-contribution" .retryNextGeneration
          (.contribution (node0Contribution exampleInitial))
      , accept "surviving-peer-contribution"
          (.contribution (node1Contribution exampleInitial))
      , expect "finite-close-insufficient" .insufficientCohort
          (closeAction exampleInitial 20).event ] }

def ownerReplayAbortScenario : ExampleScenario :=
  let ownerLoss epoch name : ExampleAction :=
    expect s!"owner-loss-{epoch}" (if epoch < 2 then .accepted else .aborted) <|
      .ownerLoss
        { context := contextFor exampleInitial 0 0 epoch
          owner := workerId "worker-0"
          ownerIncarnation := incarnation "peer-inc-0"
          replayEvidence := ev name
          observedAt := ⟨13 + epoch⟩ }
  { name := "bounded-owner-replay-abort"
    initial := exampleInitial
    actions :=
      commonBootstrap exampleInitial ++
      [ accept "contribution-node-0"
          (.contribution (node0Contribution exampleInitial))
      , accept "contribution-node-1"
          (.contribution (node1Contribution exampleInitial))
      , closeAction exampleInitial
      , ownerLoss 0 "owner-replay-0"
      , ownerLoss 1 "owner-replay-1"
      , ownerLoss 2 "owner-replay-exhausted" ] }

def roleLossScenario (role : LossRole) : ExampleScenario :=
  let roleName :=
    match role with
    | .participant => "participant"
    | .service => "service"
    | .manager => "manager"
    | .trainer => "trainer"
  let loss : LossEvent :=
    { context := contextFor exampleInitial
      role
      worker := workerId "worker-0"
      node := nodeId "node-0"
      peerIncarnation := incarnation "peer-inc-0"
      trainer :=
        if role == .trainer then some (trainerName "node-0" 0) else none
      trainerIncarnation :=
        if role == .trainer then
          some (trainerIncarnation "node-0" 0)
        else none
      evidence := ev s!"{roleName}-loss-evidence" }
  { name := s!"{roleName}-loss"
    initial := exampleInitial
    actions :=
      commonBootstrap exampleInitial ++
      [accept s!"{roleName}-loss" (.loss loss)] }

def restartRejoinScenario : ExampleScenario :=
  let initial := job5105811Initial
  let node0 :=
    contributionEvent initial "worker-0" "node-0" "peer-inc-0" 0
      1966080 10 '1' '2' '3' '4'
      (generation := 3) (baseGeneration := 3)
      (receipt := "job-5105811-contribution-receipt-0")
  let node1 :=
    contributionEvent initial "worker-1" "node-1" "peer-inc-1" 0
      1968000 11 '5' '6' '7' '8'
      (generation := 3) (baseGeneration := 3)
      (receipt := "job-5105811-contribution-receipt-1")
  let serviceLoss : LossEvent :=
    { context := contextFor initial 3
      role := .service
      worker := workerId "worker-0"
      node := nodeId "node-0"
      peerIncarnation := incarnation "peer-inc-0"
      trainer := none
      trainerIncarnation := none
      evidence := ev "job-5105811-node-0-service-state-loss" }
  let restart : RestartPeerEvent :=
    { context := contextFor initial 3
      worker := workerId "worker-0"
      node := nodeId "node-0"
      oldIncarnation := incarnation "peer-inc-0"
      newIncarnation := incarnation "peer-inc-0-restarted"
      managerEvidence := ev "manager-node-0-restarted"
      serviceEvidence := ev "service-node-0-restarted"
      syncedGeneration := ⟨4⟩
      leaseUntil := ⟨1000⟩ }
  let restartedTrainerActions :=
    (List.range 8).map fun lane =>
      accept s!"node-0:restarted-trainer-{lane}" <| .registerTrainer
        { context := contextFor initial 3
          worker := workerId "worker-0"
          node := nodeId "node-0"
          peerIncarnation := incarnation "peer-inc-0-restarted"
          trainer := trainerName "node-0" lane
          trainerIncarnation :=
            incarnation s!"trainer-inc-node-0-restarted-{lane}" }
  let restartedApplyActions :=
    (List.range 8).map fun lane =>
      accept s!"node-0:restarted-apply-{lane}" <| .trainerApply
        { context := contextFor initial 3
          node := nodeId "node-0"
          worker := workerId "worker-0"
          peerIncarnation := incarnation "peer-inc-0-restarted"
          trainer := trainerName "node-0" lane
          trainerIncarnation :=
            incarnation s!"trainer-inc-node-0-restarted-{lane}"
          result := resultId "result-1"
          resultDigest := repeatedHex 'a'
          receipt := rid s!"restart-apply-node-0-{lane}"
          receiptDigest := repeatedHex 'd' }
  let restartedReduce : ExampleAction :=
    accept "node-0:restarted-eight-to-one" <| .reduceNodeApply
      { context := contextFor initial 3
        node := nodeId "node-0"
        worker := workerId "worker-0"
        peerIncarnation := incarnation "peer-inc-0-restarted"
        result := resultId "result-1"
        resultDigest := repeatedHex 'a'
        trainerReceiptDigest := repeatedHex '1'
        receipt := rid "node-apply-node-0-restarted"
        receiptDigest := repeatedHex '3' }
  let readyRestarted : ExampleAction :=
    accept "node-0:rejoin-ready-generation-4" <| .peerTransition
      { context := contextFor initial 3
        worker := workerId "worker-0"
        node := nodeId "node-0"
        incarnation := incarnation "peer-inc-0-restarted"
        fromPhase := .sync
        toPhase := .ready
        syncedGeneration := ⟨4⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := ev "manager-node-0-restarted"
        serviceEvidence := ev "service-node-0-restarted" }
  let readyNode1 : ExampleAction :=
    accept "node-1:ready-generation-4" <| .peerTransition
      { context := contextFor initial 3
        worker := workerId "worker-1"
        node := nodeId "node-1"
        incarnation := incarnation "peer-inc-1"
        fromPhase := .ready
        toPhase := .ready
        syncedGeneration := ⟨4⟩
        leaseUntil := ⟨1000⟩
        managerEvidence := ev "manager-node-1-next"
        serviceEvidence := ev "service-node-1-next" }
  let openNext : ExampleAction :=
    openGenerationAction initial 4 (repeatedHex 'a') 40 21
  { name := "job-5105811-generation-closed-restart-rejoin"
    initial
    actions :=
      bootstrapPeerActions initial "worker-0" "node-0" "peer-inc-0" 3 ++
      bootstrapPeerActions initial "worker-1" "node-1" "peer-inc-1" 3 ++
      [openGenerationAction initial 3] ++
      [ accept "contribution-node-0"
          (.contribution node0)
      , accept "contribution-node-1"
          (.contribution node1)
      , closeAction initial 12 0 3
      , commitAction initial 0 3
      , publishAction initial 0 3
      , accept "node-0:service-and-trainer-state-loss"
          (.loss serviceLoss)
      , accept "node-0:restart-new-incarnation" (.restartPeer restart)
      , expect "job-5105811-generation-closed-catch-up" .catchUp
          (.contribution job5105811LateContribution) ] ++
      restartedTrainerActions ++ restartedApplyActions ++
      [restartedReduce] ++
      trainerApplyActions initial "node-1" "worker-1" "peer-inc-1" 0 3 ++
      [ reduceAction initial "node-1" "worker-1" "peer-inc-1" 0 3
      , readyRestarted
      , readyNode1
      , openNext
      , accept "new-incarnation-next-generation-contribution"
          (.contribution job5105811NextContribution) ] }

def freshFenceScenario : ExampleScenario :=
  let authority2 :=
    exampleAuthorityFor examplePolicy 2 "allocation-2"
  let claim : ClaimFenceEvent :=
    { authority := authority2
      policy := examplePolicy
      baseGeneration := ⟨1⟩
      baseDigest := repeatedHex 'a'
      baseReceipt := rid "commit-receipt-1"
      baseReceiptDigest := repeatedHex 'b'
      acceptedTokenClock := 150797682560
      lastResult := some (resultId "result-1") }
  let staleAfterRestart :=
    { node0Contribution exampleInitial with sequence := ⟨88⟩ }
  { name := "fresh-allocation-fence-restart"
    initial := exampleInitial
    actions :=
      commonBootstrap exampleInitial ++
      [ accept "contribution-node-0"
          (.contribution (node0Contribution exampleInitial))
      , accept "contribution-node-1"
          (.contribution (node1Contribution exampleInitial))
      , closeAction exampleInitial
      , commitAction exampleInitial
      , accept "newer-allocation-claim" (.claimFence claim)
      , expect "old-fence-after-restart" .staleFence
          (.contribution staleAfterRestart) ] }

def finiteSnapshotClosureScenario : ExampleScenario :=
  let initial := exampleInitialFor exampleScalePolicy
  let node2 :=
    contributionEvent initial "worker-2" "node-2" "peer-inc-2"
      0 1 19 '9' '8' '7' '6'
      (receipt := "contribution-receipt-2")
  { name := "leased-ready-finite-snapshot-closure"
    initial
    actions :=
      bootstrapPeerActions initial "worker-0" "node-0" "peer-inc-0" ++
      bootstrapPeerActions initial "worker-1" "node-1" "peer-inc-1" ++
      bootstrapPeerActions initial "worker-2" "node-2" "peer-inc-2" ++
      [ openGenerationAction initial
      , accept "contribution-node-0"
          (.contribution (node0Contribution initial))
      , accept "contribution-node-1"
          (.contribution (node1Contribution initial))
      , expect "qmin-does-not-close-reviewed-snapshot" .deferred
          (closeAction initial 12).event
      , accept "preclose-arrival-node-2" (.contribution node2)
      , closeAction initial 20 ] }

def strictAndUnknownScenario : ExampleScenario :=
  let initial := exampleInitialFor strictExamplePolicy
  let contribution :=
    contributionEvent initial "strict-worker" "strict-node"
      "strict-inc" 0 1 10 '1' '2' '3' '4'
      (receipt := "strict-receipt")
  let historical :=
    { contribution with
      context :=
        { contribution.context with
          policyName := "async-decoupled-v2.0-exp"
          policySchema := "emender-async-policy-v2.0"
          policyDigest := repeatedHex '7' } }
  { name := "strict-v1-and-historical-fail-closed"
    initial
    actions :=
      bootstrapPeerActions initial "strict-worker" "strict-node"
        "strict-inc" ++
      [openGenerationAction initial] ++
      [ expect "historical-v2.0-identity" .unknownIdentity
          (.contribution historical)
      , accept "strict-fresh-contribution" (.contribution contribution) ] }

def allScenarios : List ExampleScenario :=
  [ normalScenario
  , peerLossScenario
  , ownerReplayAbortScenario
  , roleLossScenario .participant
  , roleLossScenario .service
  , roleLossScenario .trainer
  , restartRejoinScenario
  , freshFenceScenario
  , finiteSnapshotClosureScenario
  , strictAndUnknownScenario ]

structure ScenarioRun where
  state : RunState
  observed : List (String × Disposition)
  failures : List String
  deriving Repr

def runScenario (scenario : ExampleScenario) : ScenarioRun :=
  scenario.actions.foldl
    (fun run action =>
      let result := transition run.state action.event
      let failures :=
        run.failures ++
          (if result.disposition == action.expected then []
           else
             [s!"{scenario.name}/{action.name}: expected {action.expected.label}, got {result.disposition.label}"]) ++
          (if invariantHolds result.state then []
           else
             [s!"{scenario.name}/{action.name}: invariant {invariantViolations result.state}"])
      { state := result.state
        observed := run.observed ++ [(action.name, result.disposition)]
        failures })
    { state := scenario.initial, observed := [], failures := [] }

def scheduledActions (scenario : ExampleScenario) : List ScheduledEvent :=
  let rec go
      (index : Nat)
      (prior : List (String × Event))
      (previous : Option String)
      (actions : List ExampleAction) : List ScheduledEvent :=
    match actions with
    | [] => []
    | action :: tail =>
        let eventId := s!"{scenario.name}:{index}"
        let replayOf :=
          (prior.find? fun pair => pair.2 == action.event).map (·.1)
        let parents :=
          match previous with
          | none => []
          | some priorId => [priorId]
        { eventId
          event := action.event
          causalParents := parents
          replayOf } ::
          go (index + 1) (prior ++ [(eventId, action.event)])
            (some eventId) tail
  go 0 [] none scenario.actions

def scenarioTrace (scenario : ExampleScenario) : TraceDocument :=
  buildTrace scenario.name scenario.initial (scheduledActions scenario)

def allExampleTraces : List TraceDocument :=
  allScenarios.map scenarioTrace

end ResilientProtocol
