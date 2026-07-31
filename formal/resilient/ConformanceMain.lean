import ResilientProtocol.Conformance
import Lean.Data.Json

open Lean
open ResilientProtocol

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
    | ["oracle", path] => some path
    | _ => none
  match path? with
  | none =>
      IO.eprintln "usage: resilient-conformance [oracle] TRACE.json"
      return (2 : UInt32)
  | some path =>
      try
        let input ← IO.FS.readFile path
        match parseTrace input with
        | .error message =>
            IO.eprintln (failureJson path message).compress
            return (2 : UInt32)
        | .ok document =>
            match replayTrace document with
            | .error message =>
                IO.eprintln (failureJson path message).compress
                return (2 : UInt32)
            | .ok _ =>
                IO.println (toJson (conformanceOracleDocument document)).compress
                return (0 : UInt32)
      catch error =>
        IO.eprintln (failureJson path (toString error)).compress
        return (2 : UInt32)
