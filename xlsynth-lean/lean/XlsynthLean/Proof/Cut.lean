-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Rebuild

open Std.Sat
open scoped AIG

namespace XlsynthLean.Proof

/-- Ordered, signed cutpoints. Both endpoints of pair i represent fresh input i. -/
abbrev CutPairs := Rebuild.Pairs

namespace CutPairs

private structure Endpoint (count : Nat) where
  index : Fin count
  right : Bool

private def endpoint (pairs : CutPairs graph n) (e : Endpoint n) : AIG.Ref graph :=
  (if e.right then pairs.rhs else pairs.lhs).get e.index e.index.isLt

private def insertEndpoint (pairs : CutPairs graph n) (table : Array (Option (Endpoint n)))
    (e : Endpoint n) : Except String (Array (Option (Endpoint n))) := do
  let node := (endpoint pairs e).gate
  if node = 0 then throw "cannot cut the constant node zero"
  if table[node]!.isSome then throw s!"ambiguous cut: node {node} selected more than once"
  return table.set! node (some e)

private def indexEndpoints (pairs : CutPairs graph n) : Except String (Array (Option (Endpoint n))) := do
  let mut table := Array.replicate graph.decls.size none
  for i in [:n] do
    if h : i < n then
      let e : Endpoint n := ⟨⟨i, h⟩, false⟩
      table ← insertEndpoint pairs table e
      let r := { e with right := true }
      if (endpoint pairs e).gate != (endpoint pairs r).gate ||
          (endpoint pairs e).invert != (endpoint pairs r).invert then
        table ← insertEndpoint pairs table r
  return table

private def atomEnd : AIG.Decl Nat → Nat
  | .atom n => n + 1
  | _ => 0

private def firstFresh (graph : AIG Nat) : Nat := (graph.decls.map atomEnd).foldl max 0

private theorem firstFresh_gt (graph : AIG Nat) (a : Nat) (ha : a ∈ graph) :
    a < firstFresh graph := by
  have mem : a + 1 ∈ graph.decls.map atomEnd := Array.mem_map.mpr ⟨.atom a, AIG.mem_def.mp ha, rfl⟩
  rw [firstFresh, Array.foldl_max_eq_max (Array.ne_empty_of_mem mem)]
  exact Nat.lt_of_succ_le (Nat.le_trans (Array.le_max_of_mem mem) (Nat.le_max_right ..))

private def plan (pairs : CutPairs graph n) (first : Nat) (table : Array (Option (Endpoint n))) :
    Rebuild.Plan graph AIG.empty where
  replace i := do
    let e ← (table[i.val]?).join
    let ref := endpoint pairs e
    if ref.gate = i.val then
      some (.leaf (.input (first + e.index)) ref.invert)
    else none

private def extendAssign (pairs : CutPairs graph n) (first : Nat) (assign : Nat → Bool)
    (atom : Nat) : Bool :=
  if first ≤ atom then
    if h : atom - first < n then ⟦graph, pairs.lhs.get (atom - first) h, assign⟧
    else assign atom
  else assign atom

private theorem extend_cut (pairs : CutPairs graph n) (first : Nat) (assign : Nat → Bool)
    (i : Fin n) :
    extendAssign pairs first assign (first + i) = ⟦graph, pairs.lhs.get i i.isLt, assign⟧ := by
  simp [extendAssign, Nat.le_add_right, i.isLt]

private theorem extend_ref (pairs : CutPairs graph n) (assign : Nat → Bool) (f : AIG.Ref graph) :
    ⟦graph, f, extendAssign pairs (firstFresh graph) assign⟧ = ⟦graph, f, assign⟧ := by
  apply AIG.denote_congr
  intro a ha
  simp [extendAssign, Nat.not_le_of_gt (firstFresh_gt graph a ha)]

private theorem compatible (pairs : CutPairs graph n) (table : Array (Option (Endpoint n)))
    (assign : Nat → Bool)
    (heq : ∀ i : Fin n, ⟦graph, pairs.lhs.get i i.isLt, assign⟧ =
      ⟦graph, pairs.rhs.get i i.isLt, assign⟧) :
    (plan pairs (firstFresh graph) table).Compatible (extendAssign pairs (firstFresh graph) assign) := by
  intro i rep selected
  change (do
    let e ← (table[i.val]?).join
    if (endpoint pairs e).gate = i.val then
      some (.leaf (.input (firstFresh graph + e.index)) (endpoint pairs e).invert)
    else none) = some rep at selected
  cases entry : (table[i.val]?).join with
  | none => simp [entry] at selected
  | some e =>
    by_cases node : (endpoint pairs e).gate = i.val
    · simp [entry, node] at selected
      subst rep
      change ⟦graph, ⟨i.val, false, i.isLt⟩, extendAssign pairs (firstFresh graph) assign⟧ =
        (extendAssign pairs (firstFresh graph) assign (firstFresh graph + e.index) ^^
          (endpoint pairs e).invert)
      rw [extend_ref, extend_cut]
      have endpointEq : ⟦graph, pairs.lhs.get e.index e.index.isLt, assign⟧ =
          ⟦graph, endpoint pairs e, assign⟧ := by
        cases hr : e.right
        · simp [endpoint, hr]
        · simpa [endpoint, hr] using heq e.index
      rw [endpointEq]
      simp [AIG.denote, node]
    · simp [entry, node] at selected

end CutPairs

/-- Transport the parent claim to a prefix extension, for example when adding
new gates that compute a cut range predicate. -/
def Claim.onExtension (parent : Claim w) (graph : AIG Nat)
    (p : AIG.IsPrefix parent.graph.decls graph.decls) : Claim w :=
  ⟨graph, parent.guard.cast p.size_le, parent.lhs.cast p.size_le, parent.rhs.cast p.size_le⟩

theorem Claim.valid_of_extension (parent : Claim w) (graph : AIG Nat)
    (p : AIG.IsPrefix parent.graph.decls graph.decls)
    (h : (parent.onExtension graph p).Valid) : parent.Valid := by
  intro a hg i
  have gc := AIG.denote.eq_of_isPrefix (assign := a) ⟨_, parent.guard⟩ graph p
  have l := AIG.denote.eq_of_isPrefix (assign := a) ⟨_, parent.lhs.get i i.isLt⟩ graph p
  have r := AIG.denote.eq_of_isPrefix (assign := a) ⟨_, parent.rhs.get i i.isLt⟩ graph p
  exact l.symm.trans ((h a (gc.trans hg) i).trans r)

/-- Preserve the chosen endpoints when new contract gates are appended. -/
def CutPairs.onExtension (pairs : CutPairs graph n) (target : AIG Nat)
    (p : AIG.IsPrefix graph.decls target.decls) : CutPairs target n :=
  ⟨pairs.lhs.cast p.size_le, pairs.rhs.cast p.size_le⟩

/-- The rebuilt graph and the upstream claims needed to use its fresh-input contract. -/
structure HorizontalCut (parent : Claim w) (count : Nat) where
  private mk ::
  pairs : CutPairs parent.graph count
  contract : AIG.Ref parent.graph
  private table : Array (Option (CutPairs.Endpoint count))
  private result : Rebuild.Result { parent with guard := contract } AIG.empty
    (CutPairs.plan pairs (CutPairs.firstFresh parent.graph) table)

/-- Replace each selected pair with one shared fresh input. The downstream guard is the
contract rebuilt under that substitution. Choose a contract that includes any needed
parent-domain restrictions. The default true contract allows arbitrary fresh values.

Node zero and repeated underlying nodes (including inverted repeats) are rejected.
A pair can name the same signed ref on its two sides. -/
def Claim.cut (parent : Claim w) (pairs : CutPairs parent.graph count)
    (contract : AIG.Ref parent.graph := parent.graph.mkConstCached true) :
    Except String (HorizontalCut parent count) := do
  let table ← CutPairs.indexEndpoints pairs
  let result ← Rebuild.claim { parent with guard := contract } AIG.empty
    (pairs.plan (CutPairs.firstFresh parent.graph) table)
  return ⟨pairs, contract, table, result⟩

namespace HorizontalCut

/-- Input label used for a cut pair in the downstream graph, starting above all
original atom labels. A cut input may disappear if rebuilding prunes its cone. -/
def inputAtom {parent : Claim w} (_cut : HorizontalCut parent count) (i : Fin count) : Nat :=
  CutPairs.firstFresh parent.graph + i

def downstream {parent : Claim w} (cut : HorizontalCut parent count) : Claim w := cut.result.child

def upstream (cut : HorizontalCut parent count) : Claim count :=
  cut.pairs.claim parent.guard

/-- On the original domain, the contract must hold for the actual cut values. -/
def range (cut : HorizontalCut parent count) : Claim 1 :=
  let v : AIG.RefVec parent.graph 0 := .emptyWithCapacity 1
  ⟨parent.graph, parent.guard, v.push cut.contract, v.push (parent.graph.mkConstCached true)⟩

/-- Instantiate the abstract inputs with the original left endpoint values, use the
upstream equality to justify the right endpoints and the range proof to enable the child. -/
theorem valid (cut : HorizontalCut parent count)
    (pairs : cut.upstream.Valid) (inRange : cut.range.Valid)
    (child : cut.downstream.Valid) : parent.Valid := by
  intro a hg i
  let ext := cut.pairs.extendAssign (CutPairs.firstFresh parent.graph) a
  have compat := cut.pairs.compatible cut.table a (pairs a hg)
  have rangeAt := inRange a hg ⟨0, by decide⟩
  have rangeTrue : ⟦parent.graph, cut.contract, a⟧ = true := by
    simp only [range, AIG.RefVec.get, AIG.RefVec.push, AIG.RefVec.emptyWithCapacity,
      Vector.getElem_push_eq, AIG.Fanin.gate_mk, AIG.Fanin.invert_mk] at rangeAt
    simpa [AIG.mkConstCached, AIG.denote_idx_false parent.graph.hconst] using rangeAt
  have cg := cut.result.guard_eq ext compat
  have l := cut.result.lhs_eq ext compat i
  have r := cut.result.rhs_eq ext compat i
  change ⟦cut.downstream.graph, cut.downstream.guard, ext⟧ = ⟦parent.graph, cut.contract, ext⟧ at cg
  change ⟦cut.downstream.graph, cut.downstream.lhs.get i i.isLt, ext⟧ =
    ⟦parent.graph, parent.lhs.get i i.isLt, ext⟧ at l
  change ⟦cut.downstream.graph, cut.downstream.rhs.get i i.isLt, ext⟧ =
    ⟦parent.graph, parent.rhs.get i i.isLt, ext⟧ at r
  rw [CutPairs.extend_ref] at cg l r
  exact l.symm.trans ((child ext (cg.trans rangeTrue) i).trans r)

end HorizontalCut

end XlsynthLean.Proof
