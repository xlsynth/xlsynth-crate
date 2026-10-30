-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Cut
import XlsynthLean.Proof.Decide

open Std.Sat
open scoped AIG
open XlsynthLean.Aig XlsynthLean.Proof

namespace CutExamples

private def gate (a b : Nat) : AIG.Decl Nat := .gate (.ofRaw a) (.ofRaw b)

-- The signed cuts represent k₀ = t, k₁ = !t on each side. Downstream, the left
-- computes bypass && (k₀ XOR k₁), the right computes bypass && (k₀ OR k₁).
private def circuit := (Circuit.fromDecls 2 2 #[
  .false, .atom 0, .atom 1, .atom 0,   -- 1, 3 = t; 2 = bypass
  gate 2 2, gate 6 6,                  -- 4, 5 = t
  gate 2 8, gate 6 10,                 -- !6, !7 = !t
  gate 8 12, gate 9 13, gate 17 19,    -- !10 = k₀ XOR k₁
  gate 21 4,                           -- 11 = left output
  gate 11 14, gate 25 4                -- 13 = right output
] #[.ofRaw 22, .ofRaw 26]).toOption.get (by native_decide)

private def parent : Claim 1 where
  graph := circuit.aig
  guard := circuit.aig.mkConstCached true
  lhs := (Rebuild.refs? circuit.aig #[22]).toOption.get (by native_decide)
  rhs := (Rebuild.refs? circuit.aig #[26]).toOption.get (by native_decide)

private def points : CutPairs parent.graph 2 where
  lhs := (Rebuild.refs? parent.graph #[8, 13]).toOption.get (by native_decide)
  rhs := (Rebuild.refs? parent.graph #[10, 15]).toOption.get (by native_decide)

private def free := (parent.cut points).toOption.get (by native_decide)
example : free.inputAtom 0 = 2 ∧ free.inputAtom 1 = 3 := by native_decide
private def spurious (atom : Nat) := atom == 1 || atom == 2 || atom == 3

-- The old bypass input keeps label 1; the fresh cut inputs use labels 2 and 3.
-- Setting all three high is a genuine counterexample to the free downstream claim.
example : ¬ free.downstream.Valid := by
  intro valid
  have eq := valid spurious (by native_decide) 0
  have ne : ⟦free.downstream.graph, free.downstream.lhs.get 0 (by decide), spurious⟧ ≠
      ⟦free.downstream.graph, free.downstream.rhs.get 0 (by decide), spurious⟧ := by native_decide
  exact ne eq

-- Append the contract R(x,k) = !bypass || (k₀ XOR k₁) to the graph. Only its
-- downstream occurrence is abstracted; its upstream occurrence uses real k values.
private def different := parent.graph.mkXorCached
  ⟨points.lhs.get 0 (by decide), points.lhs.get 1 (by decide)⟩
private theorem differentExt : AIG.IsPrefix parent.graph.decls different.aig.decls :=
  AIG.LawfulOperator.isPrefix_aig (f := AIG.mkXorCached) _ _
private def bypass : AIG.Ref parent.graph := ⟨2, false, by native_decide⟩
private def contract := different.aig.mkOrCached
  ⟨(bypass.cast differentExt.size_le).not, different.ref⟩
private theorem contractExt : AIG.IsPrefix parent.graph.decls contract.aig.decls := by
  have p := AIG.LawfulOperator.isPrefix_aig (f := AIG.mkOrCached) different.aig
    ⟨(bypass.cast differentExt.size_le).not, different.ref⟩
  exact ⟨Nat.le_trans differentExt.size_le p.size_le,
    fun i h => (p.idx_eq i (Nat.lt_of_lt_of_le h differentExt.size_le)).trans (differentExt.idx_eq i h)⟩

private def extended := parent.onExtension contract.aig contractExt
private def bounded := (extended.cut (points.onExtension contract.aig contractExt) contract.ref).toOption.get
  (by native_decide)

example : parent.Valid := by
  apply parent.valid_of_extension contract.aig contractExt
  exact bounded.valid (by aig_decide) (by aig_decide) (by aig_decide)

-- Forbidding every cut value would make the downstream claim trivial, but the
-- upstream range obligation detects that the original graph can produce a value.
private def impossible := (parent.cut points (parent.graph.mkConstCached false)).toOption.get (by native_decide)
example : impossible.downstream.Valid := by aig_decide
example : ¬ impossible.range.Valid := by
  intro valid
  have eq := valid (fun _ => false) (by native_decide) 0
  have ne : ⟦impossible.range.graph, impossible.range.lhs.get 0 (by decide), fun _ => false⟧ ≠
      ⟦impossible.range.graph, impossible.range.rhs.get 0 (by decide), fun _ => false⟧ := by native_decide
  exact ne eq

-- A repeated physical node with the opposite sign cannot denote two independent
-- cut inputs. The same signed point on both sides of one pair is unambiguous.
private def overlap : CutPairs parent.graph 2 := {points with
  rhs := (Rebuild.refs? parent.graph #[10, 9]).toOption.get (by native_decide)}
example : (parent.cut overlap).isOk = false := by native_decide
example : (parent.cut {points with rhs := points.lhs}).isOk = true := by native_decide
private def withConstant : CutPairs parent.graph 2 := {points with
  lhs := (Rebuild.refs? parent.graph #[0, 13]).toOption.get (by native_decide)}
example : (parent.cut withConstant).isOk = false := by native_decide

end CutExamples
