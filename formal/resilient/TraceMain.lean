import ResilientProtocol.Trace
import Lean.Data.Json

open Lean
open ResilientProtocol

def verdictJson (path : String) (state : RunState) : Json :=
  Json.mkObj
    [ ("schemaVersion", Json.str traceSchemaVersion)
    , ("schemaDigest", Json.str traceSchemaDigest)
    , ("input", Json.str path)
    , ("verdict", Json.str "accepted")
    , ("finalStateDigest", Json.str (stateDigest state).value)
    , ("finalAuthority", toJson (authorityView state))
    , ("invariantVerdict", Json.bool (invariantHolds state)) ]

def failureJson (path message : String) : Json :=
  Json.mkObj
    [ ("schemaVersion", Json.str traceSchemaVersion)
    , ("schemaDigest", Json.str traceSchemaDigest)
    , ("input", Json.str path)
    , ("verdict", Json.str "fail_closed")
    , ("error", Json.str message) ]

def main (args : List String) : IO UInt32 := do
  let path? :=
    match args with
    | [path] => some path
    | ["replay", path] => some path
    | _ => none
  match path? with
  | none =>
      IO.eprintln "usage: resilient-trace [replay] TRACE.json"
      return (2 : UInt32)
  | some path =>
      try
        let input ← IO.FS.readFile path
        match replayTraceString input with
        | .ok state =>
            IO.println (verdictJson path state).compress
            return (0 : UInt32)
        | .error message =>
            IO.eprintln (failureJson path message).compress
            return (2 : UInt32)
      catch error =>
        IO.eprintln (failureJson path (toString error)).compress
        return (2 : UInt32)
