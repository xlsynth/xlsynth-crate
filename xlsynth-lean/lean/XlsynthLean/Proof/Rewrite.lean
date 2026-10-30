-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Rebuild
import XlsynthLean.Proof.Leaves

open Std.Sat
open scoped AIG

namespace XlsynthLean.Proof.Rewrite

/-- Index redirects by the node being replaced, independently of edge polarity. -/
private structure Index (pairs : Rebuild.Pairs g n) where
  table : Std.HashMap Nat (Fin n)
  key_eq : ∀ (k : Nat) (j : Fin n), table[k]? = some j → (pairs.lhs.get j j.isLt).gate = k

private def Index.build (pairs : Rebuild.Pairs g n) : Except String (Index pairs) :=
  Fin.foldlM n (init := ⟨∅, by simp⟩) fun idx j => do
    let k := (pairs.lhs.get j j.isLt).gate
    if idx.table.contains k then throw s!"duplicate rewrite source {k}"
    return ⟨idx.table.insert k j, by
      intro other v lookup
      by_cases h : k = other
      · subst other
        simp at lookup; subst v; rfl
      · exact idx.key_eq other v (by simpa [Std.HashMap.getElem?_insert, h] using lookup)⟩

private def Index.plan {pairs : Rebuild.Pairs g n} (idx : Index pairs) :
    Rebuild.Plan g AIG.empty where
  replace i := do
    let j ← idx.table[i.val]?
    let l := pairs.lhs.get j j.isLt
    let r := pairs.rhs.get j j.isLt
    if l.gate == r.gate && l.invert == r.invert then none
    else some (.redirect (r.flip l.invert))

private theorem Index.compatible {c : Claim w} {pairs : Rebuild.Pairs c.graph n}
    (idx : Index pairs) (proved : (pairs.claim c.guard).Valid)
    (a : Nat → Bool) (hg : ⟦c.graph, c.guard, a⟧ = true) : idx.plan.Compatible a := by
  intro i r applied
  simp only [Index.plan] at applied
  cases hx : idx.table[i.val]? with
  | none => simp [hx] at applied
  | some j =>
    simp only [hx, bind, Option.bind] at applied
    split at applied
    · contradiction
    · obtain rfl := Option.some.inj applied
      have hpair := proved a hg j
      dsimp only [Rebuild.Pairs.claim] at hpair
      let l := pairs.lhs.get j j.isLt
      have key : l.gate = i.val := idx.key_eq _ _ hx
      change ⟦c.graph, ⟨i.val, false, i.isLt⟩, a⟧ = ⟦c.graph, _, a⟧
      rw [AIG.Ref.denote_flip, ← hpair]
      change ⟦c.graph, ⟨i.val, false, i.isLt⟩, a⟧ = (⟦c.graph, l, a⟧ ^^ l.invert)
      simp [AIG.denote, key]

/-- The result carries a guarded correspondence for each retained reference.
Its parent proof requires both the pre-rewrite equalities and the child proof. -/
structure Round (c : Claim w) (pairs : Rebuild.Pairs c.graph n) where
  child : Claim w
  image : (r : AIG.Ref c.graph) →
    Option { t : AIG.Ref child.graph // (pairs.claim c.guard).Valid →
      ∀ a, ⟦c.graph, c.guard, a⟧ = true → ⟦child.graph, t, a⟧ = ⟦c.graph, r, a⟧ }
  valid_of_rewrite : (pairs.claim c.guard).Valid → child.Valid → c.Valid

/-- Sweep gates that became unreachable after folding. No search for new
equalities is involved; the second map composes with the first. -/
private def Round.sweep {c : Claim w} {pairs : Rebuild.Pairs c.graph n}
    (round : Round c pairs) (pins : Array (AIG.Ref c.graph)) : Except String (Round c pairs) := do
  let keep ← pins.mapM fun r => do
    let some t := round.image r | throw "missing pinned rewrite image"
    return t.val
  let identity : Rebuild.Plan round.child.graph AIG.empty := {}
  let s ← Rebuild.claim round.child AIG.empty identity keep
  have ok (a : Nat → Bool) : identity.Compatible a := by intro i r h; contradiction
  return {
    child := s.child
    image := fun r => do
      let old ← round.image r
      let fresh ← s.image old.val
      return ⟨fresh.val, fun h a hg => (fresh.property a (ok a)).trans (old.property h a hg)⟩
    valid_of_rewrite := fun h child => round.valid_of_rewrite h (s.valid_of_identity child) }

/-- Replace each left endpoint by its signed right endpoint and rebuild. The
sources are also checked for cycles outside the output cones. Extra pins
retain selected nodes for later rounds even if no output depends on them. -/
def run (c : Claim w) (pairs : Rebuild.Pairs c.graph n) (pins : Array (AIG.Ref c.graph) := #[]) :
    Except String (Round c pairs) := do
  let idx ← Index.build pairs
  let sources := Array.ofFn fun (i : Fin n) => pairs.lhs.get i i.isLt
  let s ← Rebuild.claim c AIG.empty idx.plan pins (validateOnly := sources)
  let round : Round c pairs := {
    child := s.child
    image := fun r => (s.image r).map fun t => ⟨t.val,
      fun h a hg => t.property a (idx.compatible h a hg)⟩
    valid_of_rewrite := by
      intro fact child a hg i
      have ok := idx.compatible fact a hg
      have enabled := (s.guard_eq a ok).trans hg
      exact (s.lhs_eq a ok i).symm.trans ((child a enabled i).trans (s.rhs_eq a ok i)) }

  if s.mayHaveDead then round.sweep pins else return round

/-- A closed Boolean check avoids elaborating the rebuilt graph to infer
dependent types at each use of a large, concrete round. -/
def identical (result : Except String (Round c pairs)) : Bool :=
  match result with
  | .error _ => false
  | .ok round => decide (round.child.lhs.refs = round.child.rhs.refs)

/-- Close a rewrite whose child has identical signed outputs. Both successful
rebuilding and output identity are checked by the one Boolean computation. -/
theorem valid_of_identical {c : Claim w} {pairs : Rebuild.Pairs c.graph n}
    (result : Except String (Round c pairs)) (fact : (pairs.claim c.guard).Valid)
    (checked : identical result = true) : c.Valid := by
  cases result with
  | error _ => contradiction
  | ok round =>
    exact round.valid_of_rewrite fact (round.child.valid_of_identical (of_decide_eq_true checked))

end XlsynthLean.Proof.Rewrite
