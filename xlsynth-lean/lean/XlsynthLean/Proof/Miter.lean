-- SPDX-License-Identifier: Apache-2.0

import Std.Tactic.BVDecide.Bitblast.BVExpr.Circuit.Impl.Operations.Eq
import XlsynthLean.Proof.Claim

open Std.Sat
open scoped AIG
open Std.Tactic.BVDecide

namespace XlsynthLean.Proof

/-- An entrypoint that is true exactly when the guard holds and some output bits disagree. -/
def Claim.bad (claim : Claim width) : AIG.Entrypoint Nat :=
  let pair : AIG.BinaryRefVec claim.graph width := ⟨claim.lhs, claim.rhs⟩
  let equal := BVPred.mkEq claim.graph pair
  let guard := claim.guard.cast (AIG.LawfulOperator.le_size (f := BVPred.mkEq) claim.graph pair)
  equal.aig.mkAndCached ⟨guard, equal.ref.not⟩

/-- The guarded mismatch is false precisely when the output equality holds under the guard. -/
theorem Claim.bad_eq_false_iff (claim : Claim width) (assign : Nat → Bool) :
    ⟦claim.bad, assign⟧ = false ↔
      (⟦claim.graph, claim.guard, assign⟧ = true → ∀ i : Fin width,
        ⟦claim.graph, claim.lhs.get i.val i.isLt, assign⟧ =
          ⟦claim.graph, claim.rhs.get i.val i.isLt, assign⟧) := by
  simp only [Claim.bad, AIG.denote_mkAndCached, AIG.Ref.denote_not, AIG.denote_projected_entry]
  rw [AIG.LawfulOperator.denote_cast_entry (f := BVPred.mkEq) ⟨claim.graph, claim.guard⟩]
  simp only [Bool.and_eq_false_imp, Bool.not_eq_false', BVPred.mkEq, AIG.RefVec.denote_fold_and,
    AIG.RefVec.denote_zip, AIG.denote_mkBEqCached, beq_iff_eq]
  constructor
  · intro h guard i
    exact h guard i.val i.isLt
  · intro h guard idx hidx
    exact h guard ⟨idx, hidx⟩

/-- An UNSAT proof for the guarded mismatch establishes the original guarded claim. -/
theorem Claim.valid_of_bad_unsat (claim : Claim width) (h : claim.bad.Unsat) : claim.Valid :=
  fun assign => (claim.bad_eq_false_iff assign).mp (h assign)

end XlsynthLean.Proof
