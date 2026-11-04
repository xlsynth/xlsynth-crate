-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Leaves
import XlsynthLean.Proof.Rebuild

open Std.Sat
open scoped AIG

namespace XlsynthLean.Proof

/-- A compactly numbered leaf, with a proof back to the original input space. -/
structure SparseLeaf (parent : Claim w) (labels : List Nat) where
  child : Claim w
  valid_of_child : child.Valid → parent.Valid

-- Unlisted atoms remain distinct from every row bit and from one another.
private def relabel (labels : List Nat) (atom : Nat) : Rebuild.Atom (AIG.empty : AIG Nat) :=
  let slot := labels.idxOf atom
  .input (if slot < labels.length then slot else labels.length + atom)

private def rowAssign (labels : List Nat) (a : Nat → Bool) (i : Nat) : Bool :=
  if h : i < labels.length then a labels[i] else a (i - labels.length)

private theorem recover (labels : List Nat) (a : Nat → Bool) (i : Nat) :
    (relabel labels i).denote (rowAssign labels a) = a i := by
  have outside : ¬ labels.length + i < labels.length := by omega
  by_cases h : labels.idxOf i < labels.length <;>
    simp [relabel, Rebuild.Atom.denote, rowAssign, h, outside]

/-- Rebuild the guard and outputs, numbering the listed atoms from zero in
the supplied order. Unlisted atoms can never masquerade as enumerated bits. -/
def Claim.sparseLeaf (parent : Claim w) (labels : List Nat) : Except String (SparseLeaf parent labels) := do
  let plan : Rebuild.Plan parent.graph AIG.empty := {atom := relabel labels}
  let s ← Rebuild.claim parent AIG.empty plan
  let leaf : SparseLeaf parent labels := ⟨s.child, by
    intro solved a h i
    let row := rowAssign labels a
    have ok : plan.Compatible row := by intro j r hr; contradiction
    have same : plan.assign row = a := funext (recover labels a)
    have guard := s.guard_eq row ok
    have lhs := s.lhs_eq row ok i
    have rhs := s.rhs_eq row ok i
    rw [same] at guard lhs rhs
    exact lhs.symm.trans ((solved row (guard.trans h) i).trans rhs)⟩
  if !s.mayHaveDead then return leaf
  -- A folded-away cone can contain an unlisted atom. Remove it before checking
  -- the atom bound, so omissions matter only for the remaining cones.
  let identity : Rebuild.Plan leaf.child.graph AIG.empty := {}
  let swept ← Rebuild.claim leaf.child AIG.empty identity
  return ⟨swept.child, fun solved => leaf.valid_of_child (swept.valid_of_identity solved)⟩

/-- Check all remaining atoms are among the listed labels and enumerate exactly
the supplied row width, regardless of the numeric magnitude of the labels. -/
theorem SparseLeaf.valid (leaf : SparseLeaf parent labels)
    (bound : leaf.child.atomsBelow labels.length = true)
    (rows : ∀ row : BitVec labels.length,
      ((leaf.child.mismatchCircuit labels.length).eval row).getLsbD 0 = false) : parent.Valid :=
  leaf.valid_of_child (leaf.child.valid_of_exhaustive labels.length bound rows)

end XlsynthLean.Proof
