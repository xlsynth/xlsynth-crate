-- SPDX-License-Identifier: Apache-2.0

import Std.Sat.AIG.RefVec
import Std.Tactic

open Std.Sat
open scoped AIG

namespace XlsynthLean.Aig

/-- A combinational graph with an ordered, fixed-width interface. Input atom `i`
reads bit `i` of the input bit vector, and output reference `i` produces bit `i`. -/
structure Circuit (inputCount outputCount : Nat) where
  aig : AIG Nat
  outputs : AIG.RefVec aig outputCount

/-- Check that every gate refers only to earlier declarations. -/
private def checkDag (decls : Array (AIG.Decl Nat)) : Bool :=
  (List.range decls.size).all fun index =>
    match decls[index]? with
    | some (.gate lhs rhs) => decide (lhs.gate < index ∧ rhs.gate < index)
    | some _ => true
    | none => false

private theorem checkDag_sound (decls : Array (AIG.Decl Nat))
    (checked : checkDag decls = true) : AIG.IsDAG Nat decls := by
  intro index lhs rhs inBounds isGate
  unfold checkDag at checked
  have atIndex := (List.all_eq_true.mp checked) index (List.mem_range.mpr inBounds)
  simpa [Array.getElem?_eq_getElem inBounds, isGate] using atIndex

private def ofDecls (decls : Array (AIG.Decl Nat))
    (checked : checkDag decls = true)
    (nonempty : 0 < decls.size)
    (zero : decls[0]'nonempty = .false) : AIG Nat :=
  { decls := decls
    cache := AIG.Cache.empty
    hdag := checkDag_sound decls checked
    hzero := nonempty
    hconst := zero }

private def checkRefs (aig : AIG Nat) (refs : Array AIG.Fanin) : Bool :=
  refs.all fun ref => decide (ref.gate < aig.decls.size)

private def ofRefs (aig : AIG Nat) (refs : Array AIG.Fanin)
    (checked : checkRefs aig refs = true) : AIG.RefVec aig refs.size :=
  { refs := refs.toVector
    hrefs := by
      intro index inBounds
      unfold checkRefs at checked
      have atIndex := (Array.all_eq_true.mp checked) index inBounds
      simpa using atIndex }

/-- Construct a circuit from stock AIG declarations and ordered output references.
Checks input atom bounds, reference bounds, DAG order, and the output count. -/
def Circuit.fromDecls (inputCount outputCount : Nat) (decls : Array (AIG.Decl Nat))
    (outputs : Array AIG.Fanin) :
    Except String (Circuit inputCount outputCount) := do
  unless decls.all (fun declaration =>
      match declaration with
      | .atom atom => decide (atom < inputCount)
      | _ => true) do
    throw "input atom out of bounds"
  if checked : checkDag decls = true then
    if nonempty : 0 < decls.size then
      if zero : decls[0]'nonempty = .false then
        let aig := ofDecls decls checked nonempty zero
        if refsChecked : checkRefs aig outputs = true then
          if outputSize : outputs.size = outputCount then
            return { aig, outputs := outputSize ▸ ofRefs aig outputs refsChecked }
          else
            throw s!"output count mismatch: expected {outputCount}, got {outputs.size}"
        else
          throw "output reference out of bounds"
      else
        throw "graph node zero is not false"
    else
      throw "graph has no false node"
  else
    throw "graph is not topologically ordered"

private def evalFanin (values : Array Bool) (ref : AIG.Fanin) : Bool :=
  values[ref.gate]?.getD false ^^ ref.invert

private def evalStep (aig : AIG Nat) (assign : Nat → Bool)
    (values : Array Bool) (index : Nat) : Array Bool :=
  let value :=
    match aig.decls[index]? with
    | some .false => false
    | some (.atom atom) => assign atom
    | some (.gate lhs rhs) => evalFanin values lhs && evalFanin values rhs
    | none => false
  values.push value

private def valuesUpTo (aig : AIG Nat) (assign : Nat → Bool) (count : Nat) : Array Bool :=
  (List.range count).foldl (evalStep aig assign) #[]

private theorem valuesUpTo_succ (aig : AIG Nat) (assign : Nat → Bool) (count : Nat) :
    valuesUpTo aig assign (count + 1) = evalStep aig assign (valuesUpTo aig assign count) count := by
  simp only [valuesUpTo, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

private theorem valuesUpTo_size (aig : AIG Nat) (assign : Nat → Bool) (count : Nat) :
    (valuesUpTo aig assign count).size = count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    rw [valuesUpTo_succ]
    simp only [evalStep, Array.size_push, ih]

private theorem valuesUpTo_get (aig : AIG Nat) (assign : Nat → Bool) (count : Nat)
    (countBounds : count ≤ aig.decls.size) (index : Nat) (indexBounds : index < count) :
    (valuesUpTo aig assign count)[index]? =
      some ⟦aig, ⟨index, false, by omega⟩, assign⟧ := by
  induction count generalizing index with
  | zero => omega
  | succ count ih =>
    rw [valuesUpTo_succ]
    unfold evalStep
    rw [Array.getElem?_push]
    by_cases earlier : index < count
    · have different : index ≠ (valuesUpTo aig assign count).size := by
        rw [valuesUpTo_size]
        omega
      rw [ite_eq_right different]
      exact ih (by omega) index earlier
    · have same : index = count := by omega
      subst index
      rw [ite_eq_left (valuesUpTo_size aig assign count).symm]
      have inGraph : count < aig.decls.size := by omega
      rw [Array.getElem?_eq_getElem inGraph]
      cases declaration : aig.decls[count] with
      | false => simp only [AIG.denote_idx_false declaration]
      | atom atom => simp only [AIG.denote_idx_atom declaration, Bool.xor_false]
      | gate lhs rhs =>
        have dag := aig.hdag inGraph declaration
        have lhsValue := ih (by omega) lhs.gate dag.1
        have rhsValue := ih (by omega) rhs.gate dag.2
        have lhsBounds : lhs.gate < aig.decls.size := by omega
        have rhsBounds : rhs.gate < aig.decls.size := by omega
        have gateValue :
            ((⟦aig, ⟨lhs.gate, false, lhsBounds⟩, assign⟧ ^^ lhs.invert) &&
             (⟦aig, ⟨rhs.gate, false, rhsBounds⟩, assign⟧ ^^ rhs.invert)) =
              ⟦aig, ⟨count, false, inGraph⟩, assign⟧ := by
          rw [AIG.denote_idx_gate declaration]
          simp only [AIG.denote, Bool.xor_false]
        simp only [evalFanin, lhsValue, rhsValue, Option.getD_some]
        exact congrArg some gateValue

private theorem evalFanin_sound (aig : AIG Nat) (ref : AIG.Fanin)
    (bounds : ref.gate < aig.decls.size) (assign : Nat → Bool) :
    evalFanin (valuesUpTo aig assign aig.decls.size) ref =
      ⟦aig, ⟨ref.gate, ref.invert, bounds⟩, assign⟧ := by
  unfold evalFanin
  rw [valuesUpTo_get aig assign aig.decls.size (Nat.le_refl _) ref.gate bounds]
  simp only [Option.getD_some, AIG.denote, Bool.xor_false]

private def bitsToBitVec {width : Nat} (bits : Fin width → Bool) : BitVec width :=
  BitVec.cast (by simp) (BitVec.ofBoolListLE (List.ofFn bits))

/-- Evaluate all graph nodes once in topological order, then collect output bits. -/
def Circuit.eval (circuit : Circuit inputCount outputCount) (inputs : BitVec inputCount) :
    BitVec outputCount :=
  let values := valuesUpTo circuit.aig inputs.getLsbD circuit.aig.decls.size
  bitsToBitVec fun index => evalFanin values circuit.outputs.refs[index]

/-- The direct stock `Std.Sat.AIG` meaning of the ordered output vector. -/
def Circuit.denote (circuit : Circuit inputCount outputCount) (inputs : BitVec inputCount) :
    BitVec outputCount :=
  bitsToBitVec fun index =>
    AIG.denote inputs.getLsbD ⟨circuit.aig, circuit.outputs.get index.val index.isLt⟩

/-- Two circuits are equivalent when their outputs agree for every input vector. -/
def Circuit.Equivalent (lhs rhs : Circuit inputCount outputCount) : Prop :=
  ∀ inputs, lhs.denote inputs = rhs.denote inputs

/-- The topological evaluator computes the stock AIG semantics. -/
theorem Circuit.eval_eq_denote (circuit : Circuit inputCount outputCount)
    (inputs : BitVec inputCount) : circuit.eval inputs = circuit.denote inputs := by
  unfold Circuit.eval Circuit.denote
  apply congrArg bitsToBitVec
  funext index
  exact evalFanin_sound circuit.aig circuit.outputs.refs[index]
    (circuit.outputs.hrefs index.isLt) inputs.getLsbD

end XlsynthLean.Aig
