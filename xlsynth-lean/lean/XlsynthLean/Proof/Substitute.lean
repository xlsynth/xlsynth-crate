-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Rebuild
import Std.Sat.AIG.RefVec

open Std.Sat
open scoped AIG

namespace XlsynthLean.Proof

/-- Rebuild only by binding source atoms. Each binding can be a signed target
reference, a constant, or a target input bit. -/
abbrev Claim.Substitution (c : Claim w) (base : AIG Nat) (bindings : Nat → Rebuild.Atom base) :=
  Rebuild.Result c base { atom := bindings }

def Claim.substitute (c : Claim w) (base : AIG Nat) (bindings : Nat → Rebuild.Atom base) :
    Except String (c.Substitution base bindings) :=
  Rebuild.claim c base { atom := bindings }

private theorem inputsCompatible (g base : AIG Nat) (bindings : Nat → Rebuild.Atom base) (a : Nat → Bool) :
    ({ atom := bindings } : Rebuild.Plan g base).Compatible a := by
  intro i r impossible
  contradiction

/-- Instantiate a proved child on any concrete input bindings. -/
theorem Claim.valid_substitute {c : Claim w} (sub : c.Substitution base bindings) (valid : c.Valid) :
    sub.child.Valid := by
  intro a h i
  have ok := inputsCompatible c.graph base bindings a
  exact (sub.lhs_eq a ok i).trans ((valid _ ((sub.guard_eq a ok).symm.trans h) i).trans
    (sub.rhs_eq a ok i).symm)

/-- The parent guard must imply the instantiated child guard. -/
def Claim.instanceGuard (parent : Claim w) {child : Claim w} {bindings : Nat → Rebuild.Atom parent.graph}
    (sub : child.Substitution parent.graph bindings) : Claim 1 :=
  let g := sub.child.graph
  let empty : AIG.RefVec g 0 := .emptyWithCapacity 1
  ⟨g, parent.guard.cast sub.extendsBase.size_le, empty.push sub.child.guard,
    empty.push (g.mkConstCached true)⟩

/-- Match each side independently, using the parent guard on the shared graph. -/
def Claim.instanceOutputs (parent : Claim w) {child : Claim w} {bindings : Nat → Rebuild.Atom parent.graph}
    (sub : child.Substitution parent.graph bindings) : Claim (w + w) :=
  let size := sub.extendsBase.size_le
  ⟨sub.child.graph, parent.guard.cast size, (parent.lhs.cast size).append (parent.rhs.cast size),
    sub.child.lhs.append sub.child.rhs⟩

/-- Reuse a previously proved child. Guard entailment and both output
correspondences are concrete AIG claims and can themselves be decomposed. -/
theorem Claim.valid_of_substitution (parent : Claim w) {child : Claim w} {bindings : Nat → Rebuild.Atom parent.graph}
    (sub : child.Substitution parent.graph bindings) (valid : child.Valid)
    (guard : (parent.instanceGuard sub).Valid) (outputs : (parent.instanceOutputs sub).Valid) :
    parent.Valid := by
  intro a h i
  have heval (r : AIG.Ref parent.graph) :
      ⟦sub.child.graph, r.cast sub.extendsBase.size_le, a⟧ = ⟦parent.graph, r, a⟧ :=
    AIG.denote.eq_of_isPrefix ⟨_, r⟩ _ sub.extendsBase
  have lifted := (heval parent.guard).trans h
  have hchild := guard a lifted 0
  simp only [instanceGuard, Fin.val_zero, AIG.RefVec.get_push_ref_eq, AIG.denote_mkConstCached] at hchild
  have solved := valid_substitute sub valid a hchild i
  have l := outputs a lifted ⟨i.val, by omega⟩
  have r := outputs a lifted ⟨w + i.val, by omega⟩
  have upper : ¬ w + i.val < w := by omega
  simp only [instanceOutputs, AIG.RefVec.get_append, dite_eq_left i.isLt, dite_eq_right upper,
    Nat.add_sub_cancel_left, AIG.RefVec.get_cast, heval] at l r
  exact l.trans (solved.trans r.symm)

/-- A partial assignment fixes selected inputs while leaving all other atoms
symbolic under their original numbering. -/
abbrev FixedInputs := Nat → Option Bool

def FixedInputs.binding (fixed : FixedInputs) (i : Nat) : Rebuild.Atom (AIG.empty : AIG Nat) :=
  match fixed i with
  | some b => .const b
  | none => .input i

def FixedInputs.Matches (fixed : FixedInputs) (a : Nat → Bool) : Prop :=
  ∀ i b, fixed i = some b → a i = b

def Claim.cofactor (c : Claim w) (fixed : FixedInputs) :
    Except String (c.Substitution AIG.empty fixed.binding) :=
  c.substitute AIG.empty fixed.binding

/-- Lift cofactor proofs through an exhaustive (possibly overlapping) cover of
the parent guard. Every child uses its *computed*, specialized guard and outputs. -/
theorem Claim.valid_of_cofactors (c : Claim w) (fixed : ι → FixedInputs)
    (sub : (j : ι) → c.Substitution AIG.empty (fixed j).binding)
    (cover : ∀ a, ⟦c.graph, c.guard, a⟧ = true → ∃ j, (fixed j).Matches a)
    (children : ∀ j, (sub j).child.Valid) : c.Valid := by
  intro a h i
  obtain ⟨j, hj⟩ := cover a h
  let bindings := (fixed j).binding
  have ok := inputsCompatible c.graph _ bindings a
  have same : ({atom := bindings} : Rebuild.Plan c.graph AIG.empty).assign a = a := by
    funext n
    cases hn : fixed j n with
    | none => simp [Rebuild.Plan.assign, bindings, FixedInputs.binding, hn, Rebuild.Atom.denote]
    | some b => simpa [Rebuild.Plan.assign, bindings, FixedInputs.binding, hn, Rebuild.Atom.denote]
          using (hj n b hn).symm
  have hg := (sub j).guard_eq a ok
  have hl := (sub j).lhs_eq a ok i
  have hr := (sub j).rhs_eq a ok i
  rw [same] at hg hl hr
  exact hl.symm.trans ((children j a (hg.trans h) i).trans hr)

end XlsynthLean.Proof
