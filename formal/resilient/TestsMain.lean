import ResilientProtocol.Examples
import Lean.Data.Json

open Lean
open ResilientProtocol

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

def normallyNonMutating : Disposition → Bool
  | .identicalDuplicate
  | .conflictingDuplicate
  | .staleFence
  | .staleIncarnation
  | .corruptNonfinite
  | .generationClosed
  | .late
  | .deferred
  | .catchUp
  | .retryNextGeneration
  | .unknownIdentity
  | .malformedTrace
  | .forbiddenReorder => true
  | _ => false

def scenarioMutationFailures (scenario : ExampleScenario) : List String :=
  let result :=
    scenario.actions.foldl
      (fun pair action =>
        let state := pair.1
        let failures := pair.2
        let stepped := transition state action.event
        let failure :=
          if normallyNonMutating stepped.disposition &&
              stepped.state != state then
            [s!"{scenario.name}/{action.name}: {stepped.disposition.label} mutated state"]
          else
            []
        (stepped.state, failures ++ failure))
      (scenario.initial, [])
  result.2

def traceRoundTripFailures (scenario : ExampleScenario) : List String :=
  let trace := scenarioTrace scenario
  match parseTrace (renderTrace trace) with
  | .error message =>
      [s!"{scenario.name}: canonical trace parse failed: {message}"]
  | .ok decoded =>
      if decoded != trace then
        [s!"{scenario.name}: canonical JSON round trip changed trace"]
      else
        match replayTrace decoded with
        | .error message =>
            [s!"{scenario.name}: replay failed: {message}"]
        | .ok final =>
            let direct := (runScenario scenario).state
            if final != direct then
              [s!"{scenario.name}: replay and direct transition diverged"]
            else
              []

def reorderedTraceFailure : Option String :=
  let trace := scenarioTrace normalScenario
  match trace.steps with
  | [] => some "normal scenario unexpectedly has no trace steps"
  | first :: second :: tail =>
      let reorderedSecond :=
        { second with
          causality :=
            { second.causality with eventIndex := 99 } }
      let tampered := { trace with steps := first :: reorderedSecond :: tail }
      match replayTrace tampered with
      | .ok _ => some "forbidden reordered trace was accepted"
      | .error _ => none
  | _ => some "normal scenario unexpectedly has fewer than two trace steps"

def unknownFieldFailure : Option String :=
  let traceJson := toJson (scenarioTrace normalScenario)
  let ambiguous := traceJson.setObjVal! "unexpected" (Json.bool true)
  match strictFromJson ambiguous (α := TraceDocument) with
  | .ok _ => some "unknown JSON field was accepted"
  | .error _ => none

def incompleteIdentityFailure : Option String :=
  match parseTrace "{}" with
  | .ok _ => some "identity-incomplete trace was accepted"
  | .error _ => none

def duplicateKeyFailure : Option String :=
  let canonical := renderTrace (scenarioTrace normalScenario)
  let duplicate :=
    "{\"toolchain\":\"duplicate-ambiguous-value\"," ++ canonical.drop 1
  match parseTrace duplicate with
  | .ok _ => some "ambiguous duplicate JSON key was accepted"
  | .error _ => none

def normalAuthorityFailuresFor (run : ScenarioRun) : List String :=
  let nextReady :=
    run.state.peers.all fun peer =>
      peer.phase == .ready && peer.syncedGeneration.value == 1
  []
  |> (fun failures =>
      if run.state.generation.status == .applied then failures
      else "normal generation did not reach applied" :: failures)
  |> (fun failures =>
      if run.state.nodeApplyReceipts.length == 2 then failures
      else "normal generation lacks exactly two node apply receipts" :: failures)
  |> (fun failures =>
      if run.state.applyReceipts.length == 16 then failures
      else "normal generation lacks all sixteen trainer receipts" :: failures)
  |> (fun failures =>
      if nextReady then failures
      else "a node advertised next READY without current all-eight authority" :: failures)
  |> (fun failures =>
      match run.state.generation.cohort, run.state.generation.commit with
      | some cohort, some commit =>
          if cohort.contributions.length == 2 &&
              cohort.exactTokens == 3934080 &&
              commit.exactTokens == 3934080 then failures
          else "accepted cohort/token accounting mismatch" :: failures
      | _, _ => "normal generation lacks frozen cohort/commit" :: failures)

def restartAuthorityFailuresFor (run : ScenarioRun) : List String :=
  let rejoined :=
    run.state.generation.accepted.any fun contribution =>
      contribution.key.worker == workerId "worker-0" &&
      contribution.key.incarnation == incarnation "peer-inc-0-restarted" &&
      contribution.key.generation.value == 1
  if rejoined then []
  else ["new incarnation did not rejoin the next admissible generation"]

def main : IO UInt32 := do
  let mut failures : List String := []
  for scenario in allScenarios do
    let run := runScenario scenario
    failures := failures ++ run.failures
    failures := failures ++ scenarioMutationFailures scenario
    failures := failures ++ traceRoundTripFailures scenario
  failures :=
    failures ++ normalAuthorityFailuresFor (runScenario normalScenario)
  failures :=
    failures ++ restartAuthorityFailuresFor (runScenario restartRejoinScenario)
  match reorderedTraceFailure with
  | some failure => failures := failures ++ [failure]
  | none => pure ()
  match unknownFieldFailure with
  | some failure => failures := failures ++ [failure]
  | none => pure ()
  match incompleteIdentityFailure with
  | some failure => failures := failures ++ [failure]
  | none => pure ()
  match duplicateKeyFailure with
  | some failure => failures := failures ++ [failure]
  | none => pure ()
  if failures.isEmpty then
    IO.println s!"PASS resilient protocol: {allScenarios.length} scenarios"
    return 0
  else
    for failure in failures do
      IO.eprintln s!"FAIL {failure}"
    return 1
