import ResilientProtocol.Examples

open ResilientProtocol

def printScenario (scenario : ExampleScenario) : IO Bool := do
  let run := runScenario scenario
  if run.failures.isEmpty then
    IO.println s!"PASS {scenario.name}: {run.observed.length} transitions, final={generationStatusLabel run.state.generation.status}, digest={(stateDigest run.state).value}"
    return true
  else
    for failure in run.failures do
      IO.eprintln s!"FAIL {failure}"
    return false

def findScenario? (name : String) : Option ExampleScenario :=
  allScenarios.find? (fun scenario => scenario.name == name)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["--trace", name] =>
      match findScenario? name with
      | none =>
          IO.eprintln s!"unknown example scenario: {name}"
          return 2
      | some scenario =>
          IO.println (renderTrace (scenarioTrace scenario))
          return 0
  | ["--list"] =>
      for scenario in allScenarios do
        IO.println scenario.name
      return 0
  | [] =>
      let mut allPassed := true
      for scenario in allScenarios do
        let passed ← printScenario scenario
        if !passed then allPassed := false
      return if allPassed then 0 else 1
  | _ =>
      IO.eprintln "usage: resilient-examples [--list | --trace SCENARIO]"
      return 2
