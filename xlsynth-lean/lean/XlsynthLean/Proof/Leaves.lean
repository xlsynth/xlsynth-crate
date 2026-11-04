-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Miter

open Std.Sat
open scoped AIG
open XlsynthLean.Aig

namespace XlsynthLean.Proof

/-- Close a leaf by comparing its signed output references after rebuilding. -/
theorem Claim.valid_of_identical (c : Claim w) (same : c.lhs.refs = c.rhs.refs) : c.Valid := by
  intro a _ i
  have eq : c.lhs = c.rhs := by
    obtain ⟨g, h, ⟨l, hl⟩, ⟨r, hr⟩⟩ := c
    cases same
    rfl
  rw [eq]

/-- Interpret a guarded mismatch as a one-output circuit on a finite input word. -/
def Claim.mismatchCircuit (c : Claim w) (n : Nat) : Circuit n 1 :=
  ⟨c.bad.aig, (AIG.RefVec.emptyWithCapacity 1).push c.bad.ref⟩

/-- All atom labels must fit in the enumerated input word, including after a cut. -/
def Claim.atomsBelow (c : Claim w) (n : Nat) : Bool :=
  c.bad.aig.decls.all fun | .atom i => decide (i < n) | _ => true

/-- Exhaustively evaluate a small leaf with the topological circuit evaluator.
The atom bound makes enumeration over finite words cover arbitrary assignments. -/
theorem Claim.valid_of_exhaustive (c : Claim w) (n : Nat) (bound : c.atomsBelow n = true)
    (rows : ∀ x : BitVec n, ((c.mismatchCircuit n).eval x).getLsbD 0 = false) : c.Valid := by
  apply c.valid_of_bad_unsat
  intro a
  let word : BitVec n := (BitVec.ofBoolListLE (List.ofFn fun i : Fin n => a i.val)).cast (by simp)
  have agree (i : Nat) (hi : i ∈ c.bad.aig) : a i = word.getLsbD i := by
    have hb : i < n := of_decide_eq_true
      ((Array.all_eq_true_iff_forall_mem.mp bound) (.atom i) (AIG.mem_def.mp hi))
    simp only [word, BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE]
    simp [List.getD_eq_getElem?_getD, hb]
  have row := rows word
  rw [Circuit.eval_eq_denote] at row
  have congruent := AIG.denote_congr a word.getLsbD c.bad.aig c.bad.ref.gate c.bad.ref.invert
    c.bad.ref.hgate agree
  apply congruent.trans
  let bits := (c.mismatchCircuit n).outputs
  change ((BitVec.ofBoolListLE (List.ofFn fun i : Fin 1 =>
    ⟦c.bad.aig, bits.get i.val i.isLt, word.getLsbD⟧)).cast (by simp) : BitVec 1).getLsbD 0 = false at row
  simp only [BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE] at row
  simpa [List.getD_eq_getElem?_getD, bits, Claim.mismatchCircuit, AIG.RefVec.get,
    AIG.RefVec.push, AIG.RefVec.emptyWithCapacity] using row

end XlsynthLean.Proof
