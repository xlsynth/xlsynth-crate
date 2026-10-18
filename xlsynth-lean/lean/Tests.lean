-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aig
import XlsynthLean.Aag

open Std.Sat
open XlsynthLean.Aig

-- Two input nodes, one AND node, and positive/negative outputs from that AND.
private def knownDecls : Array (AIG.Decl Nat) := #[
  .false,
  .atom 0,
  .atom 1,
  .gate (.ofRaw 2) (.ofRaw 4)
]

private def knownOutputs : Array AIG.Fanin := #[.ofRaw 6, .ofRaw 7]

private def evaluateKnown (input : Nat) : Option Nat := do
  let circuit ← (Circuit.fromDecls 2 2 knownDecls knownOutputs).toOption
  return (circuit.eval (BitVec.ofNat 2 input)).toNat

example : (List.range 4).map evaluateKnown = [.some 2, .some 2, .some 2, .some 1] := by
  native_decide

private def evaluateZeroWidth : Option Nat := do
  let circuit ← (Circuit.fromDecls 0 0 #[.false] #[]).toOption
  return (circuit.eval 0).toNat

example : evaluateZeroWidth = .some 0 := by native_decide

private def rejected (inputs outputs : Nat) (decls : Array (AIG.Decl Nat))
    (refs : Array AIG.Fanin) : Bool :=
  match Circuit.fromDecls inputs outputs decls refs with
  | .ok _ => false
  | .error _ => true

-- Every graph must begin with a false node.
example : rejected 0 0 #[] #[] = true := by native_decide

example : rejected 1 0 #[.atom 0] #[] = true := by native_decide

-- Node 2 is an AND that refers to itself.
example : rejected 1 1 #[.false, .atom 0, .gate (.ofRaw 4) (.ofRaw 2)] #[.ofRaw 4] =
    true := by native_decide

-- The graph contains only nodes 0 and 1, but its output refers to node 2.
example : rejected 1 1 #[.false, .atom 0] #[.ofRaw 4] = true := by native_decide

-- Output count and input atom labels must agree with the interface.
example : rejected 0 1 #[.false] #[] = true := by native_decide

example : rejected 1 0 #[.false, .atom 1] #[] = true := by native_decide

-- The AND fanins intentionally use ascending order; both fanin orders are valid.
private def knownAag := "aag 3 2 0 2 1\n2\n4\n6\n7\n6 2 4\n"

private def evaluateKnownAag (input : Nat) : Option Nat := do
  let circuit ← (XlsynthLean.Aag.parse 2 2 knownAag).toOption
  return (circuit.eval (BitVec.ofNat 2 input)).toNat

example : (List.range 4).map evaluateKnownAag = [.some 2, .some 2, .some 2, .some 1] := by
  native_decide

private def evaluateZeroWidthAag : Option Nat := do
  let circuit ← (XlsynthLean.Aag.parse 0 0 "aag 0 0 0 0 0\n").toOption
  return (circuit.eval 0).toNat

example : evaluateZeroWidthAag = .some 0 := by native_decide

private def evaluateConstantAag : Option Nat := do
  let circuit ← (XlsynthLean.Aag.parse 0 2 "aag 0 0 0 2 0\n0\n1\n").toOption
  return (circuit.eval 0).toNat

example : evaluateConstantAag = .some 2 := by native_decide

private def aagRejected (inputs outputs : Nat) (text : String) : Bool :=
  match XlsynthLean.Aag.parse inputs outputs text with
  | .ok _ => false
  | .error _ => true

-- Reject malformed headers, unsupported latches, inconsistent counts, and
-- counts that cannot fit in the remaining file before allocating graph arrays.
example : [
    "aig 0 0 0 0 0\n",
    "aag -1 0 0 0 0\n",
    "aag 0 0 0 0 0 0\n",
    "aag 0 0 1 0 0\n",
    "aag 1 0 0 0 0\n",
    "aag 1000000000 0 0 0 1000000000\n"
  ].all (aagRejected 0 0) = true := by native_decide

example : aagRejected 1 0 "aag 0 0 0 0 0\n" = true := by native_decide

example : aagRejected 0 1 "aag 0 0 0 0 0\n" = true := by native_decide

-- Reject inverted input declarations, out-of-bounds outputs, nonconsecutive
-- AND left-hand sides, and AND references to the current or later node.
example : [
    "aag 1 1 0 1 0\n3\n2\n",
    "aag 1 1 0 1 0\n2\n4\n",
    "aag 2 1 0 1 1\n2\n4\n6 2 0\n",
    "aag 2 1 0 1 1\n2\n4\n4 4 2\n",
    "aag 2 1 0 1 1\n2\n4\n4 2 6\n"
  ].all (aagRejected 1 1) = true := by native_decide

-- Exact records are required, including line endings and the end of the file.
example : [
    "aag 1 1 0 1 0\n2\n",
    "aag 1 1 0 1 0\n2\n2",
    "aag 1 1 0 1 0\n2\n2\n\n",
    "aag 1 1 0 1 0\n2\n2\ni0 input\n",
    "aag 1 1 0 1 0\n2\n2\nc\ncomment\n"
  ].all (aagRejected 1 1) = true := by native_decide
