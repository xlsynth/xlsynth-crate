-- SPDX-License-Identifier: Apache-2.0

import Target
import XlsynthLean.Proof

open Std.Sat XlsynthLean.Proof

namespace Partitioned

-- Each design exposes [a && b && c, !(a && b && c), a] in LSB-first order.
private def root := rootClaim Generated.Lhs.circuit Generated.Rhs.circuit
example : (outputBits? 3 [2, 0, 2]).toOption.map (List.map Fin.val) = some [2, 0, 2] := by decide
example : (outputBits? 3 [3]).toOption.isNone := by decide

private def groups : List (List (Fin 3)) := [[2, 0], [1, 0]]

private def one {g : AIG Nat} (r : AIG.Ref g) : AIG.RefVec g 1 :=
  (AIG.RefVec.emptyWithCapacity 1).push r

-- Cut the first output before comparing it alongside the bypass input.
private def first := root.project [2, 0]
private def cutPoints : CutPairs first.graph 1 :=
  ⟨one (root.lhs.get 0 (by decide)), one (root.rhs.get 0 (by decide))⟩
private def cut := (first.cut cutPoints).toOption.get (by native_decide)

private theorem firstValid : first.Valid := by
  apply cut.valid
  · aig_decide (timeout := 20)
  · exact cut.range.valid_of_identical (by native_decide)
  · exact cut.downstream.valid_of_identical (by native_decide)

-- For the other group, transfer one signed input equivalence into the rebuilt
-- graph, then split on a resulting signal and check the two cases differently.
private def second := root.project [1, 0]
private def shared : Rebuild.Pairs second.graph 1 :=
  ⟨one (rootLeftRef Generated.Lhs.circuit Generated.Rhs.circuit ⟨1, true, by native_decide⟩),
   one (rootRightRef Generated.Lhs.circuit Generated.Rhs.circuit ⟨1, true, by native_decide⟩)⟩
private def round := (Rewrite.run second shared).toOption.get (by native_decide)

private theorem secondValid : second.Valid := by
  apply round.valid_of_rewrite
  · aig_decide
  · apply round.child.valid_of_bool_split (round.child.lhs.get 0 (by decide))
    · aig_decide
    · apply Claim.valid_of_exhaustive _ 3 <;> native_decide

-- This is the theorem the proof runner checks against its generated target.
theorem equivalent : Target.Equivalent := by
  apply equivalent_of_rootValid
  apply root.valid_of_cover groups (by decide)
  intro group member
  simp only [groups, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact firstValid
  · exact secondValid

-- A proof that projects away the only bad bit must fail the coverage premise.
private def bad : Claim 3 := { root with rhs :=
  (AIG.RefVec.emptyWithCapacity 3 |>.push (root.lhs.get 0 (by decide))
    |>.push (root.lhs.get 1 (by decide)) |>.push (root.lhs.get 2 (by decide)).not) }

example : (bad.project [1, 0]).Valid := by aig_decide
example : ¬ (∀ bit : Fin 3, bit ∈ ([[1, 0]] : List (List (Fin 3))).flatten) := by decide
example : ¬ bad.Valid := by
  intro h
  have bit := h (fun _ => false) (by native_decide) (2 : Fin 3)
  exact (by native_decide : AIG.denote (fun _ => false) ⟨bad.graph, bad.lhs.get 2 (by decide)⟩ ≠
    AIG.denote (fun _ => false) ⟨bad.graph, bad.rhs.get 2 (by decide)⟩) bit

-- Fewer input bits would omit assignments to atom 2 in this graph.
example : second.atomsBelow 2 = false := by native_decide

end Partitioned
