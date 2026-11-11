-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Claim
import Std.Tactic.BVDecide.Bitblast.BVExpr.Circuit.Impl.Operations.Eq
import Std.Tactic.BVDecide.Bitblast.BVExpr.Circuit.Lemmas.Const

open Std.Sat
open Std.Tactic.BVDecide
open scoped AIG

namespace XlsynthLean.Proof

/-- A computed domain predicate can extend the graph containing the claim. -/
structure Domain (base : AIG Nat) extends AIG.Entrypoint Nat where
  extendsBase : AIG.IsPrefix base.decls aig.decls

def Domain.ofRef (r : AIG.Ref g) : Domain g := ⟨⟨g, r⟩, .rfl⟩

private theorem denoteCast {g g' : AIG Nat} (p : AIG.IsPrefix g.decls g'.decls)
    (r : AIG.Ref g) (a : Nat → Bool) :
    ⟦g', r.cast p.size_le, a⟧ = ⟦g, r, a⟧ :=
  AIG.denote.eq_of_isPrefix ⟨g, r⟩ g' p

/-- Read a vector of selectors as an unsigned little-endian word. -/
def selectorValue {g : AIG Nat} (bits : AIG.RefVec g n) (assign : Nat → Bool) : BitVec n :=
  (BitVec.ofBoolListLE (List.ofFn fun i : Fin n => ⟦g, bits.get i i.isLt, assign⟧)).cast (by simp)

private theorem selectorValue_bit (bits : AIG.RefVec g n) (assign : Nat → Bool)
    (i : Nat) (hi : i < n) :
    (selectorValue bits assign).getLsbD i = ⟦g, bits.get i hi, assign⟧ := by
  simp only [selectorValue, BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE]
  simp [List.getD_eq_getElem?_getD, hi]

private def addRow (acc : Domain g) (bits : AIG.RefVec g n) (row : BitVec n) : Domain g :=
  let input : AIG.BinaryRefVec acc.aig n :=
    ⟨bits.cast acc.extendsBase.size_le, BVExpr.bitblast.blastConst acc.aig row⟩
  let eq := BVPred.mkEq acc.aig input
  let p := AIG.LawfulOperator.isPrefix_aig (f := BVPred.mkEq) acc.aig input
  let pair : AIG.BinaryInput eq.aig := ⟨acc.ref.cast p.size_le, eq.ref⟩
  ⟨eq.aig.mkOrCached pair, prefixTrans acc.extendsBase
      (prefixTrans p (AIG.LawfulOperator.isPrefix_aig (f := AIG.mkOrCached) eq.aig pair))⟩

private theorem addRow_true_iff (acc : Domain g) (bits : AIG.RefVec g n)
    (row : BitVec n) (assign : Nat → Bool) :
    ⟦(addRow acc bits row).toEntrypoint, assign⟧ = true ↔
      ⟦acc.toEntrypoint, assign⟧ = true ∨ selectorValue bits assign = row := by
  simp only [addRow, AIG.denote_mkOrCached, Bool.or_eq_true, AIG.denote_projected_entry]
  rw [AIG.LawfulOperator.denote_cast_entry (f := BVPred.mkEq) ⟨acc.aig, acc.ref⟩]
  apply or_congr_right
  simp only [BVPred.mkEq, AIG.RefVec.denote_fold_and, AIG.RefVec.denote_zip,
    AIG.denote_mkBEqCached, beq_iff_eq, AIG.RefVec.get_cast, BVExpr.bitblast.denote_blastConst]
  rw [BitVec.eq_of_getLsbD_eq_iff]
  constructor
  · intro h i hi
    rw [selectorValue_bit bits assign i hi]
    exact (AIG.denote.eq_of_isPrefix ⟨g, bits.get i hi⟩ acc.aig acc.extendsBase).symm.trans (h i hi)
  · intro h i hi
    exact (AIG.denote.eq_of_isPrefix ⟨g, bits.get i hi⟩ acc.aig acc.extendsBase).trans
      ((selectorValue_bit bits assign i hi).symm.trans (h i hi))

/-- Compile membership in a finite set of selector words to a graph predicate. -/
def Domain.members (bits : AIG.RefVec g n) (rows : List (BitVec n)) : Domain g :=
  rows.foldl (fun acc row => addRow acc bits row) (Domain.ofRef (g.mkConstCached false))

theorem Domain.members_true_iff (bits : AIG.RefVec g n) (rows : List (BitVec n))
    (assign : Nat → Bool) :
    ⟦(members bits rows).toEntrypoint, assign⟧ = true ↔ selectorValue bits assign ∈ rows := by
  have go (xs : List (BitVec n)) (start : Domain g) :
      ⟦(xs.foldl (fun acc row => addRow acc bits row) start).toEntrypoint, assign⟧ = true ↔
      ⟦start.toEntrypoint, assign⟧ = true ∨ selectorValue bits assign ∈ xs := by
    induction xs generalizing start with
    | nil => simp
    | cons r rs ih => simp only [List.foldl_cons, ih, addRow_true_iff, List.mem_cons, or_assoc]
  simp [members, go, ofRef, AIG.denote_mkConstCached]

/-- A disjunction of signed predicates in the same graph. -/
def Domain.anyOf (bits : AIG.RefVec g n) : Domain g :=
  let noneSet := members bits [0]
  ⟨⟨noneSet.aig, noneSet.ref.not⟩, noneSet.extendsBase⟩

private theorem anyOf_true_iff (bits : AIG.RefVec g n) (a : Nat → Bool) :
    ⟦(Domain.anyOf bits).toEntrypoint, a⟧ = true ↔
      ∃ i : Fin n, ⟦g, bits.get i i.isLt, a⟧ = true := by
  simp only [Domain.anyOf, AIG.Ref.denote_not, Bool.not_eq_true', ← Bool.not_eq_true]
  change (¬ ⟦(Domain.members bits [0]).toEntrypoint, a⟧ = true) ↔ _
  simp only [Domain.members_true_iff, List.mem_singleton, BitVec.eq_of_getLsbD_eq_iff]
  constructor
  · intro h
    apply Classical.byContradiction
    intro missing
    apply h
    intro j hj
    rw [selectorValue_bit bits a j hj]
    cases bit : ⟦g, bits.get j hj, a⟧ with
    | false => simp
    | true => exact False.elim (missing ⟨⟨j, hj⟩, bit⟩)
  · rintro ⟨i, hi⟩ zero
    have bit := zero i i.isLt
    rw [selectorValue_bit bits a i i.isLt, hi] at bit
    simp at bit

/-- A one-bit AIG obligation saying the parent guard implies this predicate. -/
def Claim.requiresDomain (c : Claim w) (d : Domain c.graph) : Claim 1 :=
  let empty : AIG.RefVec d.aig 0 := .emptyWithCapacity 1
  ⟨d.aig, c.guard.cast d.extendsBase.size_le, empty.push d.ref,
    empty.push (d.aig.mkConstCached true)⟩

theorem Claim.requiresDomain_valid_iff (c : Claim w) (d : Domain c.graph) :
    (c.requiresDomain d).Valid ↔ ∀ a, ⟦c.graph, c.guard, a⟧ = true →
      ⟦d.toEntrypoint, a⟧ = true := by
  constructor
  · intro h a hg
    have eq := h a ((denoteCast d.extendsBase _ _).trans hg) 0
    simpa only [requiresDomain, Fin.val_zero, AIG.RefVec.get_push_ref_eq, AIG.denote_mkConstCached]
      using eq
  · intro h a hg i
    have hi : i = 0 := Fin.ext (by omega)
    subst i
    simpa only [requiresDomain, Fin.val_zero, AIG.RefVec.get_push_ref_eq, AIG.denote_mkConstCached]
      using h a ((denoteCast d.extendsBase _ _).symm.trans hg)

/-- Add a case to the existing guard and retain both ordered output vectors. -/
def Claim.restrict (c : Claim w) (d : Domain c.graph) : Claim w :=
  let joined := d.aig.mkAndCached ⟨c.guard.cast d.extendsBase.size_le, d.ref⟩
  let pref := prefixTrans d.extendsBase
    (AIG.LawfulOperator.isPrefix_aig (f := AIG.mkAndCached) d.aig _)
  ⟨joined.aig, joined.ref, c.lhs.cast pref.size_le, c.rhs.cast pref.size_le⟩

theorem Claim.restrict_valid_iff (c : Claim w) (d : Domain c.graph) :
    (c.restrict d).Valid ↔ ∀ assign, ⟦c.graph, c.guard, assign⟧ = true →
      ⟦d.toEntrypoint, assign⟧ = true → ∀ i : Fin w,
      ⟦c.graph, c.lhs.get i i.isLt, assign⟧ = ⟦c.graph, c.rhs.get i i.isLt, assign⟧ := by
  let pair : AIG.BinaryInput d.aig := ⟨c.guard.cast d.extendsBase.size_le, d.ref⟩
  have pref := prefixTrans d.extendsBase
    (AIG.LawfulOperator.isPrefix_aig (f := AIG.mkAndCached) d.aig pair)
  simp only [Claim.Valid, Claim.restrict, AIG.denote_projected_entry, AIG.denote_mkAndCached,
    Bool.and_eq_true, AIG.RefVec.get_cast]
  simp only [denoteCast d.extendsBase]
  constructor
  · intro h a hg hd i
    exact (denoteCast pref _ a).symm.trans ((h a ⟨hg, hd⟩ i).trans (denoteCast pref _ a))
  · intro h a ⟨hg, hd⟩ i
    exact (denoteCast pref _ a).trans ((h a hg hd i).trans (denoteCast pref _ a).symm)

/-- Every parent assignment must land in a proved case. Cases may overlap. -/
theorem Claim.valid_of_domain_cover (c : Claim w) (parts : ι → Domain c.graph)
    (coverage : ∀ assign, ⟦c.graph, c.guard, assign⟧ = true →
      ∃ i, ⟦(parts i).toEntrypoint, assign⟧ = true)
    (children : ∀ i, (c.restrict (parts i)).Valid) : c.Valid := by
  intro a h
  obtain ⟨i, hi⟩ := coverage a h
  exact (c.restrict_valid_iff _).mp (children i) a h hi

/-- A SAT-checkable cover for predicates already in the parent graph. -/
theorem Claim.valid_of_ref_cover (c : Claim w) (guards : AIG.RefVec c.graph n)
    (coverage : (c.requiresDomain (.anyOf guards)).Valid)
    (children : ∀ i : Fin n, (c.restrict (.ofRef (guards.get i i.isLt))).Valid) : c.Valid :=
  c.valid_of_domain_cover _ (fun a h => (anyOf_true_iff guards a).mp
    ((c.requiresDomain_valid_iff _).mp coverage a h)) children

/-- Split on either polarity of any signed graph reference. -/
theorem Claim.valid_of_bool_split (c : Claim w) (r : AIG.Ref c.graph)
    (whenFalse : (c.restrict (.ofRef r.not)).Valid)
    (whenTrue : (c.restrict (.ofRef r)).Valid) : c.Valid := by
  intro a h
  cases hr : ⟦c.graph, r, a⟧ with
  | true => exact (c.restrict_valid_iff _).mp whenTrue a h hr
  | false => exact (c.restrict_valid_iff _).mp whenFalse a h (by simp [Domain.ofRef, hr])

/-- A cover of the reachable selector words suffices for a cover of parent assignments. -/
theorem Claim.valid_of_member_cover (c : Claim w) (bits : AIG.RefVec c.graph n)
    (tables : ι → List (BitVec n))
    (coverage : ∀ a, ⟦c.graph, c.guard, a⟧ = true → ∃ i, selectorValue bits a ∈ tables i)
    (children : ∀ i, (c.restrict (.members bits (tables i))).Valid) : c.Valid :=
  c.valid_of_domain_cover _ (fun a ha => by
    obtain ⟨i, hi⟩ := coverage a ha
    exact ⟨i, (Domain.members_true_iff ..).mpr hi⟩) children

/-- Check a finite list of selector tables for coverage directly on the AIG. -/
theorem Claim.valid_of_table_cover (c : Claim w) (bits : AIG.RefVec c.graph n)
    (tables : List (List (BitVec n)))
    (coverage : (c.requiresDomain (.members bits tables.flatten)).Valid)
    (children : ∀ table ∈ tables, (c.restrict (.members bits table)).Valid) : c.Valid := by
  intro a h
  have member := (Domain.members_true_iff ..).mp ((c.requiresDomain_valid_iff _).mp coverage a h)
  obtain ⟨table, ht, hr⟩ := List.mem_flatten.mp member
  exact (c.restrict_valid_iff _).mp (children table ht) a h ((Domain.members_true_iff ..).mpr hr)

end XlsynthLean.Proof
