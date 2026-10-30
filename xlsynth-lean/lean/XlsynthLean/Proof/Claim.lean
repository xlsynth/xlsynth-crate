-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aig

open Std.Sat
open scoped AIG
open XlsynthLean.Aig

namespace XlsynthLean.Proof

/-- Successive extensions preserve every declaration in the starting graph. -/
theorem prefixTrans {a b c : Array (AIG.Decl Nat)}
    (ab : AIG.IsPrefix a b) (bc : AIG.IsPrefix b c) : AIG.IsPrefix a c :=
  ⟨Nat.le_trans ab.size_le bc.size_le,
   fun i hi => (bc.idx_eq i (Nat.lt_of_lt_of_le hi ab.size_le)).trans (ab.idx_eq i hi)⟩

/-- A guarded equality between ordered output vectors of one graph. -/
structure Claim (width : Nat) where
  graph : AIG Nat
  guard : AIG.Ref graph
  lhs : AIG.RefVec graph width
  rhs : AIG.RefVec graph width

/-- Every assignment satisfying the guard makes all corresponding bits equal. -/
def Claim.Valid (c : Claim width) : Prop :=
  ∀ assign : Nat → Bool, AIG.denote assign ⟨c.graph, c.guard⟩ = true →
    ∀ i : Fin width, AIG.denote assign ⟨c.graph, c.lhs.get i i.isLt⟩ =
      AIG.denote assign ⟨c.graph, c.rhs.get i i.isLt⟩

private def shiftFanin (offset : Nat) (f : AIG.Fanin) : AIG.Fanin :=
  .mk (offset + f.gate) f.invert

private def shiftDecl (offset : Nat) : AIG.Decl Nat → AIG.Decl Nat
  | .false => .false
  | .atom i => .atom i
  | .gate a b => .gate (shiftFanin offset a) (shiftFanin offset b)

/-- Keep both node spaces contiguous. Only node indices move. Atom i on either
side refers to the same assignment bit i. -/
private def join (left right : AIG Nat) : AIG Nat where
  decls := left.decls ++ right.decls.map (shiftDecl left.decls.size)
  cache := AIG.Cache.empty
  hzero := by simp; have := left.hzero; omega
  hconst := by rw [Array.getElem_append_left left.hzero]; exact left.hconst
  hdag := by
    intro i a b hi gate
    by_cases hl : i < left.decls.size
    · rw [Array.getElem_append_left hl] at gate
      exact left.hdag hl gate
    · rw [Array.getElem_append_right (by omega), Array.getElem_map] at gate
      have hr : i - left.decls.size < right.decls.size := by
        simp only [Array.size_append, Array.size_map] at hi
        omega
      cases decl : right.decls[i - left.decls.size] with
      | false => simp [decl, shiftDecl] at gate
      | atom _ => simp [decl, shiftDecl] at gate
      | gate x y =>
        have bounds := right.hdag hr decl
        simp only [decl, shiftDecl, AIG.Decl.gate.injEq] at gate
        obtain ⟨rfl, rfl⟩ := gate
        simp only [shiftFanin, AIG.Fanin.gate_mk]
        omega

private theorem leftPrefix (left right : AIG Nat) :
    AIG.IsPrefix left.decls (join left right).decls where
  size_le := by simp [join]
  idx_eq _ hi := Array.getElem_append_left hi

private theorem rightBound {left right : AIG Nat} {i : Nat} (hi : i < right.decls.size) :
    left.decls.size + i < (join left right).decls.size := by
  simpa [join] using hi

private def rightOutputs (left right : AIG Nat) (v : AIG.RefVec right w) :
    AIG.RefVec (join left right) w where
  refs := v.refs.map (shiftFanin left.decls.size)
  hrefs := by
    intro i hi
    simpa only [Vector.getElem_map, shiftFanin, AIG.Fanin.gate_mk] using
      (rightBound (left := left) (v.hrefs hi))

private theorem denote_right (left right : AIG Nat) (assign : Nat → Bool)
    (i : Nat) (hi : i < right.decls.size) (inv : Bool) :
    ⟦join left right, ⟨left.decls.size + i, inv, rightBound hi⟩, assign⟧ =
      ⟦right, ⟨i, inv, hi⟩, assign⟧ := by
  have shifted : (join left right).decls[left.decls.size + i]'(rightBound hi) =
      shiftDecl left.decls.size right.decls[i] := by
    simp [join, Array.getElem_append_right (Nat.le_add_right ..)]
  cases decl : right.decls[i] with
  | false =>
    rw [AIG.denote_idx_false (by simpa [decl, shiftDecl] using shifted),
        AIG.denote_idx_false decl]
  | atom _ =>
    rw [AIG.denote_idx_atom (by simpa [decl, shiftDecl] using shifted),
        AIG.denote_idx_atom decl]
  | gate a b =>
    rw [AIG.denote_idx_gate (by simpa [decl, shiftDecl] using shifted),
        AIG.denote_idx_gate decl]
    have bounds := right.hdag hi decl
    simp only [shiftFanin, AIG.Fanin.gate_mk, AIG.Fanin.invert_mk]
    rw [denote_right left right assign a.gate (by omega) a.invert,
        denote_right left right assign b.gate (by omega) b.invert]
termination_by i
decreasing_by all_goals have := right.hdag hi decl; omega

/-- The unconditional claim on two circuits sharing the same ordered input bits. -/
def rootClaim (lhs rhs : Circuit n w) : Claim w :=
  let graph := join lhs.aig rhs.aig
  { graph, guard := ⟨0, true, graph.hzero⟩
    lhs := lhs.outputs.cast (leftPrefix lhs.aig rhs.aig).size_le
    rhs := rightOutputs lhs.aig rhs.aig rhs.outputs }

/-- Lift a selected node from the original LHS into the shared root graph. -/
def rootLeftRef (lhs rhs : Circuit n w) (ref : AIG.Ref lhs.aig) :
    AIG.Ref (rootClaim lhs rhs).graph :=
  ref.cast (leftPrefix lhs.aig rhs.aig).size_le

/-- Lift a selected node from the original RHS, preserving its polarity. -/
def rootRightRef (lhs rhs : Circuit n w) (ref : AIG.Ref rhs.aig) :
    AIG.Ref (rootClaim lhs rhs).graph :=
  ⟨lhs.aig.decls.size + ref.gate, ref.invert, rightBound ref.hgate⟩

@[simp] theorem denote_rootLeftRef (lhs rhs : Circuit n w) (r : AIG.Ref lhs.aig)
    (assign : Nat → Bool) :
    ⟦(rootClaim lhs rhs).graph, rootLeftRef lhs rhs r, assign⟧ = ⟦lhs.aig, r, assign⟧ :=
  AIG.denote.eq_of_isPrefix (assign := assign) ⟨lhs.aig, r⟩ _ (leftPrefix lhs.aig rhs.aig)

@[simp] theorem denote_rootRightRef (lhs rhs : Circuit n w) (r : AIG.Ref rhs.aig)
    (assign : Nat → Bool) :
    ⟦(rootClaim lhs rhs).graph, rootRightRef lhs rhs r, assign⟧ = ⟦rhs.aig, r, assign⟧ :=
  denote_right lhs.aig rhs.aig assign r.gate r.hgate r.invert

/-- A valid root claim proves the exact equivalence proposition used by the runner. -/
theorem equivalent_of_rootValid {lhs rhs : Circuit n w} (valid : (rootClaim lhs rhs).Valid) :
    lhs.Equivalent rhs := by
  intro xs
  unfold Circuit.denote
  congr 1
  funext i
  let graph := join lhs.aig rhs.aig
  have allBits := valid xs.getLsbD (AIG.denote_idx_false graph.hconst)
  have bit := allBits i
  have hl := AIG.denote.eq_of_isPrefix (assign := xs.getLsbD)
    ⟨lhs.aig, lhs.outputs.get i i.isLt⟩ graph (leftPrefix lhs.aig rhs.aig)
  have hr := denote_right lhs.aig rhs.aig xs.getLsbD _ (rhs.outputs.hrefs i.isLt)
    rhs.outputs.refs[i].invert
  simp only [rootClaim, AIG.RefVec.get, AIG.RefVec.cast, AIG.RefVec.cast', rightOutputs,
    Vector.getElem_map, shiftFanin, AIG.Fanin.gate_mk, AIG.Fanin.invert_mk] at bit
  exact hl.symm.trans (bit.trans hr)

end XlsynthLean.Proof
