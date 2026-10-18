-- SPDX-License-Identifier: Apache-2.0

import Lean.Data.Json
import XlsynthLean.Aig

open Lean
open XlsynthLean.Aig

namespace XlsynthLean.EvalMain

private def parseBits (width : Nat) (text : String) : Except String (BitVec width) := do
  let bytes := text.toUTF8
  unless bytes.size == width do
    throw s!"input bit length mismatch: expected {width}, got {bytes.size}"
  let mut value := 0
  -- Read the LSB-first input backwards so each step appends one bit.
  for offset in [:bytes.size] do
    let index := bytes.size - 1 - offset
    value := value * 2
    match bytes[index]!.toNat with
    | 48 => pure ()
    | 49 => value := value + 1
    | _ => throw s!"invalid input bit at position {index}; expected 0 or 1"
  return BitVec.ofNat width value

private def formatBits (value : BitVec width) : String :=
  String.ofList ((List.range width).map fun index => if value.getLsbD index then '1' else '0')

private def evaluate (circuit : Circuit inputCount outputCount) (requestText : String) :
    Except String String := do
  let request ← Json.parse requestText
  let object ← request.getObj?
  unless object.size == 1 do throw "request must contain exactly the cases field"
  let cases ← (← request.getObjVal? "cases").getArr?
  let mut outputs := #[]
  for index in [:cases.size] do
    let text ← cases[index]!.getStr?
    let inputs ← match parseBits inputCount text with
      | .ok inputs => pure inputs
      | .error error => throw s!"case {index}: {error}"
    outputs := outputs.push (.str (formatBits (circuit.eval inputs)))
  return (Json.mkObj [("outputs", .arr outputs)]).compress

private def runFiles (circuit : Circuit inputCount outputCount) (args : List String) : IO Unit := do
  let [requestPath, responsePath] := args
    | throw (IO.userError "usage: REQUEST_JSON RESPONSE_JSON")
  let requestText ← IO.FS.readFile requestPath
  let response ← match evaluate circuit requestText with
    | .ok response => pure response
    | .error error => throw (IO.userError error)
  IO.FS.writeFile responsePath response

/-- Evaluate the request JSON batch with a supplied circuit and write response JSON. -/
def run (circuit : Circuit inputCount outputCount) (args : List String) : IO UInt32 := do
  try
    runFiles circuit args
    return 0
  catch error =>
    (← IO.getStderr).putStrLn s!"{error}"
    return 1

end XlsynthLean.EvalMain
