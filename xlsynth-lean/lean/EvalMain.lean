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

/-- Evaluate one function's packed input vectors from a JSON request. -/
def evaluate (eval : BitVec inputCount → BitVec outputCount) (request : Json) :
    Except String Json := do
  let object ← request.getObj?
  unless object.size == 1 do throw "request must contain exactly the cases field"
  let cases ← (← request.getObjVal? "cases").getArr?
  let mut outputs := #[]
  for index in [:cases.size] do
    let text ← cases[index]!.getStr?
    let inputs ← match parseBits inputCount text with
      | .ok inputs => pure inputs
      | .error error => throw s!"case {index}: {error}"
    outputs := outputs.push (.str (formatBits (eval inputs)))
  return Json.mkObj [("outputs", .arr outputs)]

private def runFiles (process : Json → Except String Json) (args : List String) : IO UInt32 := do
  try
    let [requestPath, responsePath] := args
      | throw (IO.userError "usage: REQUEST_JSON RESPONSE_JSON")
    let request ← IO.ofExcept (Json.parse (← IO.FS.readFile requestPath))
    let response ← IO.ofExcept (process request)
    IO.FS.writeFile responsePath response.compress
    return 0
  catch error =>
    (← IO.getStderr).putStrLn s!"{error}"
    return 1

/-- Evaluate a batch with a packed function and write the outputs as JSON. -/
def runFunction (eval : BitVec inputCount → BitVec outputCount) := runFiles (evaluate eval)

/-- Run a separate request for each function in the driver, in order. -/
def runFunctions (functions : Array (Json → Except String Json)) := runFiles fun request => do
  let requests ← request.getArr?
  unless requests.size == functions.size do
    throw s!"expected {functions.size} function requests, got {requests.size}"
  let mut responses := #[]
  for i in [:functions.size] do
    let response ← (functions[i]! requests[i]!).mapError fun e => s!"function {i}: {e}"
    responses := responses.push response
  return .arr responses

/-- Evaluate a batch with an AIG circuit. -/
def run (circuit : Circuit inputCount outputCount) := runFunction circuit.eval

end XlsynthLean.EvalMain
