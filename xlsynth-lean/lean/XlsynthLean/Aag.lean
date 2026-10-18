-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aig

open Std.Sat

namespace XlsynthLean.Aag

private def readNatTerm (bytes : ByteArray) (start : Nat) (terminator : UInt8) :
    Except String (Nat × Nat) := do
  let mut position := start
  let mut value := 0
  while inBounds : position < bytes.size do
    let byte := bytes[position]
    if byte < 48 ∨ 57 < byte then
      break
    value := value * 10 + (byte.toNat - 48)
    position := position + 1
  unless start < position do
    throw s!"expected a natural number at byte {start}"
  unless bytes[position]? = some terminator do
    throw s!"expected delimiter {terminator} at byte {position}"
  return (value, position + 1)

/-- Parse the canonical combinational ASCII AIGER subset emitted by `xlsynth-lean`.

The header must be `aag M I 0 O A`, with `M = I + A`. Inputs and AND left-hand
sides must use consecutive variable numbers, and AND fanins must refer to earlier
variables. Either fanin order is accepted. Records use single spaces and LF line
endings. Symbols, comments, extensions, and trailing data are rejected. -/
def parse (inputCount outputCount : Nat) (text : String) :
    Except String (XlsynthLean.Aig.Circuit inputCount outputCount) := do
  let bytes := text.toUTF8
  unless bytes[0]? = some 97 ∧ bytes[1]? = some 97 ∧
      bytes[2]? = some 103 ∧ bytes[3]? = some 32 do
    throw "expected an ASCII AIGER header"
  let (maxVar, position) ← readNatTerm bytes 4 32
  let (inputs, position) ← readNatTerm bytes position 32
  let (latches, position) ← readNatTerm bytes position 32
  let (outputs, position) ← readNatTerm bytes position 32
  let (ands, position) ← readNatTerm bytes position 10
  unless inputs = inputCount do
    throw s!"input count mismatch: expected {inputCount}, got {inputs}"
  unless outputs = outputCount do
    throw s!"output count mismatch: expected {outputCount}, got {outputs}"
  unless latches = 0 do
    throw "latches are not supported"
  unless maxVar = inputs + ands do
    throw "expected consecutive input and AND variable numbers"
  -- Every record needs at least one byte. Check this before using counts as
  -- array capacities, so malformed headers cannot request unbounded allocation.
  unless inputs + outputs + ands ≤ bytes.size - position do
    throw "record counts exceed the remaining file size"
  let mut position := position
  let mut decls : Array (AIG.Decl Nat) := Array.mkEmpty (maxVar + 1)
  decls := decls.push .false
  for index in [:inputs] do
    let (literal, next) ← readNatTerm bytes position 10
    position := next
    unless literal = 2 * (index + 1) do
      throw s!"unexpected input literal at index {index}"
    decls := decls.push (.atom index)
  let mut outputRefs : Array AIG.Fanin := Array.mkEmpty outputs
  for index in [:outputs] do
    let (literal, next) ← readNatTerm bytes position 10
    position := next
    unless literal < 2 * (maxVar + 1) do
      throw s!"output literal out of bounds at index {index}"
    outputRefs := outputRefs.push (.ofRaw literal)
  for index in [:ands] do
    let (lhs, next) ← readNatTerm bytes position 32
    let (rhs0, next) ← readNatTerm bytes next 32
    let (rhs1, next) ← readNatTerm bytes next 10
    position := next
    unless lhs = 2 * (inputs + index + 1) do
      throw s!"unexpected AND lhs at index {index}"
    unless rhs0 < lhs ∧ rhs1 < lhs do
      throw s!"AND reference is not topological at index {index}"
    decls := decls.push (.gate (.ofRaw rhs0) (.ofRaw rhs1))
  unless position = bytes.size do
    throw "unexpected trailing data"
  XlsynthLean.Aig.Circuit.fromDecls inputCount outputCount decls outputRefs

end XlsynthLean.Aag
