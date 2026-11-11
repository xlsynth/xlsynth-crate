-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Decide
import XlsynthLean.Proof.Domain
import XlsynthLean.Proof.Substitute

open Std.Sat
open scoped AIG
open XlsynthLean.Proof

namespace DomainExamples

private def x := (AIG.empty : AIG Nat).mkAtom 0
private def singleton {g : AIG Nat} (r : AIG.Ref g) : AIG.RefVec g 1 :=
  (AIG.RefVec.emptyWithCapacity 1).push r
private def one := x.aig.mkConstCached true
private def bitClaim (guard lhs rhs : AIG.Ref x.aig) : Claim 1 :=
  ⟨x.aig, guard, singleton lhs, singleton rhs⟩

private def parent := bitClaim one x.ref x.ref
private def rows : Bool → List (BitVec 1) | false => [0] | true => [1]

-- The selector can be any circuit reference, including a complemented node.
example : parent.Valid := by
  apply parent.valid_of_bool_split x.ref.not <;> aig_decide

example : parent.Valid := by
  apply parent.valid_of_member_cover (singleton x.ref.not) rows
  · intro a _
    have allWords : ∀ (v : BitVec 1), ∃ i, v ∈ rows i := by decide
    exact allWords _
  · intro i
    cases i <;> aig_decide

-- SAT can prove cover as well as the individual branches.
example : parent.Valid := by
  apply parent.valid_of_ref_cover ((singleton x.ref).push x.ref.not)
  · aig_decide
  · intro i
    have two : ∀ i : Fin 2, i = 0 ∨ i = 1 := by decide
    rcases two i with rfl | rfl <;> aig_decide

example : parent.Valid := by
  apply parent.valid_of_table_cover (singleton x.ref.not) [[0], [1]]
  · aig_decide
  · intro table included
    simp only [List.mem_cons, List.not_mem_nil, or_false] at included
    rcases included with rfl | rfl <;> aig_decide

-- A single case is sufficient to cover a guarded parent only when the guard implies it.
private def missingCover := parent.requiresDomain (.anyOf (singleton x.ref))
example : ¬ missingCover.Valid := by
  fail_if_success have _ : missingCover.Valid := by aig_decide
  intro valid
  have mismatch := valid (fun _ => false) (by native_decide) 0
  exact (by native_decide : ⟦_, missingCover.lhs.get 0 (by decide), fun _ => false⟧ ≠
    ⟦_, missingCover.rhs.get 0 (by decide), fun _ => false⟧) mismatch

-- Selecting only row 0 misses the assignment x = false.
example : ¬ (∀ a, ⟦parent.graph, parent.guard, a⟧ = true →
    selectorValue (singleton x.ref.not) a ∈ ([0] : List (BitVec 1))) := by
  intro cover
  have missed := cover (fun _ => false) (by native_decide)
  have absent : selectorValue (singleton x.ref.not) (fun _ => false) ∉ ([0] : List (BitVec 1)) :=
    by native_decide
  exact absent missed

-- Cofactor a real gate. Fixing x removes it from the rebuilt output cone.
private def both := x.aig.mkGate ⟨x.ref, x.ref⟩
private def repeated : Claim 1 := ⟨both.aig, both.aig.mkConstCached true, singleton both.ref,
  singleton (x.ref.cast (by decide))⟩
private def fixed (b : Bool) : FixedInputs := fun i => if i = 0 then some b else none
private def cofactored (b : Bool) :=
  (repeated.cofactor (fixed b)).toOption.get (by cases b <;> native_decide)

example : (cofactored false).child.graph.decls.size = 1 := by native_decide
example : repeated.Valid := by
  apply repeated.valid_of_cofactors fixed cofactored
  · intro a _
    refine ⟨a 0, ?_⟩
    intro i b h
    by_cases hi : i = 0 <;> simp_all [fixed]
  · intro i
    cases i <;> aig_decide

-- One guarded child theorem, used with two opposite signed parent bindings.
private def child := bitClaim x.ref x.ref one
private theorem childValid : child.Valid := by aig_decide

private def onHigh := bitClaim x.ref x.ref one
private def onLow := bitClaim x.ref.not x.ref.not one
private def plugged (r : AIG.Ref x.aig) := child.substitute x.aig (fun _ => .bound r)
private def positive := (plugged x.ref).toOption.get (by native_decide)
private def negative := (plugged x.ref.not).toOption.get (by native_decide)

example : onHigh.Valid :=
  onHigh.valid_of_substitution positive childValid (by aig_decide) (by aig_decide)
example : onLow.Valid :=
  onLow.valid_of_substitution negative childValid (by aig_decide) (by aig_decide)

-- Reuse cannot weaken a child guard. The positive binding is false under onLow.
private def overstrong := onLow.instanceGuard positive
example : ¬ overstrong.Valid := by
  intro valid
  have mismatch := valid (fun _ => false) (by native_decide) 0
  exact (by native_decide : ⟦_, overstrong.lhs.get 0 (by decide), fun _ => false⟧ ≠
    ⟦_, overstrong.rhs.get 0 (by decide), fun _ => false⟧) mismatch

end DomainExamples
