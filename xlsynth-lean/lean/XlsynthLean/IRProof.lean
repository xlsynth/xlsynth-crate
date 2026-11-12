-- SPDX-License-Identifier: Apache-2.0
import XlsynthLean.IR
import Std.Tactic.BVDecide
import Lean
import Lean.Meta.Native

namespace XlsynthLean.IR

open Lean in
/-- Reachable definitions in one IR export, in dependency order. Calls to other
    exported functions remain boundaries. -/
partial def exportedNodes (decl : Name) : CoreM (Array Name) := do
  -- The entrypoint `Generated.Lhs.eval` and user abbreviations can name an export.
  let mut root := decl
  let mut aliases := #[]
  repeat
    let .defnInfo info ← getConstInfo root | break
    unless info.hints.isAbbrev do break
    let .const target _ := info.value.consumeMData | break
    aliases := aliases.push root
    root := target
  let rec visit (name : Name) : StateT (NameSet × Array Name) CoreM Unit := do
    if (← get).1.contains name then return
    modify fun (seen, nodes) => (seen.insert name, nodes)
    let some body := (← getConstInfo name).value?
      | throwError "expected a transparent definition at {name}"
    for dep in body.getUsedConstants.qsort Name.quickLt do
      if dep.getPrefix == root && (← getConstInfo dep).isDefinition then visit dep
    modify fun (seen, nodes) => (seen, nodes.push name)
  let (_, _, nodes) ← (visit root).run ({}, #[])
  return nodes ++ aliases.reverse

open Lean Elab Tactic in
/-- Unfold a named-node XLS function without unfolding the functions it calls. -/
elab "xls_unfold " fn:ident : tactic => do
  let nodes ← exportedNodes (← resolveGlobalConstNoOverload fn)
  let names := nodes.map mkIdent
  evalTactic (← `(tactic| simp only [$[$names:ident],*]))

/-- Expose a one-bit XLS selector to bit-vector reasoning. -/
theorem sel_bit (s : BitVec 1) (x y d : α) :
    #[x, y].getD s.toNat d = (bif s.getLsbD 0 then y else x) := by
  have h : s = 0 ∨ s = 1 := by bv_decide
  rcases h with rfl | rfl <;> rfl

/-- A one-case priority selector reads bit zero. -/
theorem priority_one (s : BitVec 1) (x d : α) :
    prioritySel s #[x] d = (bif s.getLsbD 0 then x else d) := by
  have h : s = 0 ∨ s = 1 := by bv_decide
  rcases h with rfl | rfl <;> rfl

/-- Case zero has priority over case one. -/
theorem priority_two (s : BitVec 2) (x y d : α) :
    prioritySel s #[x, y] d = (bif s.getLsbD 0 then x else bif s.getLsbD 1 then y else d) := by
  have h : s = 0 ∨ s = 1 ∨ s = 2 ∨ s = 3 := by bv_decide
  rcases h with rfl | rfl | rfl | rfl <;> rfl

/-- The executable bound on the shift count preserves logical shift semantics. -/
theorem shrl_min (x : BitVec w) (d : BitVec s) :
    x >>> min d.toNat w = x >>> d := by
  change _ = x >>> d.toNat
  rcases Nat.le_total d.toNat w with h | h
  · rw [Nat.min_eq_left h]
  · rw [Nat.min_eq_right h, BitVec.ushiftRight_eq_zero (Nat.le_refl _), BitVec.ushiftRight_eq_zero h]

/-- Large left shifts can be proved with the usual bit-vector shift operator. -/
theorem shll_min (x : BitVec w) (d : BitVec s) :
    x <<< min d.toNat w = x <<< d := by
  change _ = x <<< d.toNat
  rcases Nat.le_total d.toNat w with h | h
  · rw [Nat.min_eq_left h]
  · rw [Nat.min_eq_right h, BitVec.shiftLeft_eq_zero (Nat.le_refl _), BitVec.shiftLeft_eq_zero h]

/-- Arithmetic right shifts saturate at the sign bit. -/
theorem shra_min (x : BitVec w) (d : BitVec s) :
    x.sshiftRight (min d.toNat w) = x.sshiftRight' d := by
  change _ = x.sshiftRight d.toNat
  rcases Nat.le_total d.toNat w with h | h
  · rw [Nat.min_eq_left h]
  · rw [Nat.min_eq_right h]
    exact BitVec.sshiftRight_eq_sshiftRight_of_le (Nat.le_refl _) h

/-- Introduce word-level hypotheses and expose selectors, shifts, and encoders
    to Lean's bit-vector decision procedures. -/
macro "ir_bv_normalize" : tactic => `(tactic| (
  (try dsimp only) <;> intros <;>
    (try simp only [sel_bit, priority_one, priority_two, shrl_min, shll_min, shra_min,
                    oneHot, encode, List.range, List.range.loop, List.foldl] at *)))

end XlsynthLean.IR

open Lean Elab Command Meta in
/-- Audit the root theorem. Closed Boolean verification obligations from native
    tactics are recomputed, including the LRAT checks used by bv_decide. -/
elab "#check_ir_axioms " theoremName:ident : command => do
  let decl ← resolveGlobalConstNoOverload theoremName
  liftTermElabM do
    let mut verified : Nat := 0
    for ax in ← collectAxioms decl do
      if [``propext, ``Classical.choice, ``Quot.sound].contains ax then
        continue
      let info ← getConstInfo ax
      let some (ty, computation, value) := info.type.eq?
        | throwError "unverified axiom in {decl}: {ax}"
      unless ty.isConstOf ``Bool && value.isConstOf ``Bool.true do
        throwError "unverified axiom in {decl}: {ax}"
      match ← nativeEqTrue `xlsynth_ir_check computation with
      | .success _ => verified := verified + 1
      | .notTrue => throwError "native Boolean check failed for {ax}"
    logInfo m!"{decl}: checked axiom dependencies and replayed {verified} native Boolean checks"
