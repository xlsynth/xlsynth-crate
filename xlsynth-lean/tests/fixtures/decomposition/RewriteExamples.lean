-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Decide
import XlsynthLean.Proof.Leaves
import XlsynthLean.Proof.Rewrite

open Std.Sat
open scoped AIG
open XlsynthLean.Proof

namespace RewriteExamples

-- Node 4 is ¬a, node 5 is a, and outputs 6 and 7 are both a ∧ b.
-- Node 3 is an input outside both output cones.
private def design := XlsynthLean.Aig.Circuit.fromDecls 3 2
  #[.false, .atom 0, .atom 1, .atom 2,
    .gate (.ofRaw 3) (.ofRaw 1), .gate (.ofRaw 2) (.ofRaw 1),
    .gate (.ofRaw 9) (.ofRaw 4), .gate (.ofRaw 10) (.ofRaw 4),
    .gate (.ofRaw 3) (.ofRaw 5), .gate (.ofRaw 17) (.ofRaw 4), .gate (.ofRaw 2) (.ofRaw 17)]
  #[.ofRaw 12, .ofRaw 14]
private def g := (design.toOption.get (by native_decide)).aig
private def ref (n : Nat) (inv : Bool := false) (h : n < g.decls.size := by native_decide) : AIG.Ref g :=
  ⟨n, inv, h⟩
private def single (r : AIG.Ref g) := (AIG.RefVec.emptyWithCapacity 1).push r
private def parent : Claim 1 := ⟨g, ref 0 true, single (ref 6), single (ref 7)⟩
private def pair (l r : AIG.Ref g) : Rebuild.Pairs g 1 := ⟨single l, single r⟩
private def signedForward := pair (ref 4 true) (ref 5)
private def round := (Rewrite.run parent signedForward).toOption.get (by native_decide)

-- The batch refers to the old graph and may redirect an earlier node to a later one.
-- Its left endpoint is inverted, so the new positive node must be the negated target.
example : parent.Valid :=
  round.valid_of_rewrite (by aig_decide) (round.child.valid_of_identical (by native_decide))
example : parent.Valid :=
  Rewrite.valid_of_identical (Rewrite.run parent signedForward) (by aig_decide) (by native_decide)

example : (round.image (ref 4)).isSome = true := by native_decide
example : (round.image (ref 3)).isNone = true := by native_decide
example : round.child.graph.decls.size = 4 := by native_decide

-- A requested pin survives pruning and carries a signed denotation theorem.
private def pinned := (Rewrite.run parent signedForward #[ref 3]).toOption.get (by native_decide)
private def pinnedRef := (pinned.image (ref 3 true)).get (by native_decide)
example (a : Nat → Bool) : ⟦pinned.child.graph, pinnedRef.val, a⟧ = ⟦g, ref 3 true, a⟧ :=
  pinnedRef.property (by aig_decide) a (AIG.denote_idx_false g.hconst)

-- The false equality cannot close the rewrite premise. Restricting the guard
-- to b makes a ∧ b = a a valid rewrite on that domain.
private def underB : Claim 1 := { parent with guard := ref 2 }
private def whenB := pair (ref 6) (ref 5)
private def guarded := (Rewrite.run underB whenB).toOption.get (by native_decide)
example : underB.Valid := guarded.valid_of_rewrite (by aig_decide) (by aig_decide)
example : ¬ (whenB.claim parent.guard).Valid := by
  intro h
  have bit := h (fun i => i == 0) (by native_decide) 0
  exact (by native_decide : ⟦g, ref 6, fun i => i == 0⟧ ≠ ⟦g, ref 5, fun i => i == 0⟧) bit

-- Even unreachable replacement sources are checked for cycles. This example
-- has a cycle through an ordinary gate, and a second cycle via two redirects.
private def trivial : Claim 1 := ⟨g, ref 0 true, single (ref 0), single (ref 0)⟩
private def batch (a b c d : AIG.Ref g) : Rebuild.Pairs g 2 :=
  ⟨(single a).push b, (single c).push d⟩
example : (Rewrite.run trivial (pair (ref 5) (ref 7))).toOption.isNone := by native_decide
example : (Rewrite.run trivial (batch (ref 4) (ref 5) (ref 5) (ref 4))).toOption.isNone := by native_decide
example : (Rewrite.run trivial (batch (ref 4) (ref 4 true) (ref 1) (ref 2))).toOption.isNone := by native_decide
example : (Rebuild.refs? g #[22]).toOption.isNone := by native_decide
example : (Rebuild.refs? g #[0, 1, 14, 15]).toOption.isSome := by native_decide

-- Absorption with either operand order and either matching OR operand.
private def absorbed : Claim 2 :=
  ⟨g, ref 0 true, (single (ref 9)).push (ref 10), (single (ref 2)).push (ref 1)⟩
private def folded := (Rebuild.claim absorbed AIG.empty {}).toOption.get (by native_decide)
example : folded.child.lhs.refs = folded.child.rhs.refs := by native_decide

private def noPairs : Rebuild.Pairs g 0 :=
  ⟨AIG.RefVec.emptyWithCapacity 0, AIG.RefVec.emptyWithCapacity 0⟩
private def swept := (Rewrite.run absorbed noPairs).toOption.get (by native_decide)
example : folded.mayHaveDead := by native_decide
example : swept.child.graph.decls.size = 3 := by native_decide
example : (swept.image (ref 8)).isNone := by native_decide
example : absorbed.Valid :=
  swept.valid_of_rewrite (by intro _ _ i; exact i.elim0)
    (swept.child.valid_of_identical (by native_decide))

-- Checking an unused redirect must not retain its private cone.
private def deadPair := (Rewrite.run trivial signedForward).toOption.get (by native_decide)
example : deadPair.child.graph.decls.size = 1 := by native_decide
example : (deadPair.image (ref 4)).isNone := by native_decide

end RewriteExamples
