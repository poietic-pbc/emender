import ResilientProtocol.ConformanceExamples

open ResilientProtocol

def findScenario? (name : String) : Option ExampleScenario :=
  nativeConformanceScenarios.find? (fun scenario => scenario.name == name)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["--list"] =>
      for scenario in nativeConformanceScenarios do
        IO.println scenario.name
      return 0
  | ["--trace", name] =>
      match findScenario? name with
      | none =>
          IO.eprintln s!"unknown native conformance scenario: {name}"
          return 2
      | some scenario =>
          let run := runScenario scenario
          if !run.failures.isEmpty then
            for failure in run.failures do
              IO.eprintln s!"FAIL {failure}"
            return 1
          IO.println (renderTrace (scenarioTrace scenario))
          return 0
  | [] =>
      let mut allPassed := true
      for scenario in nativeConformanceScenarios do
        let run := runScenario scenario
        if run.failures.isEmpty then
          IO.println s!"PASS {scenario.name}: {run.observed.length} transitions"
        else
          allPassed := false
          for failure in run.failures do
            IO.eprintln s!"FAIL {failure}"
      return if allPassed then 0 else 1
  | _ =>
      IO.eprintln "usage: resilient-conformance-corpus [--list | --trace SCENARIO]"
      return 2
