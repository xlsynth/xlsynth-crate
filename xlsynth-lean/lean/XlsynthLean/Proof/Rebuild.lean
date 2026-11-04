-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Claim
import Std.Sat.AIG.CachedLemmas

open Std.Sat
open scoped AIG

namespace XlsynthLean.Proof.Rebuild

/-- Ordered signed pairs for either an equivalence rewrite or a horizontal cut. -/
structure Pairs (graph : AIG Nat) (count : Nat) where
  lhs : AIG.RefVec graph count
  rhs : AIG.RefVec graph count

def Pairs.claim (pairs : Pairs graph count) (guard : AIG.Ref graph) : Claim count :=
  ⟨graph, guard, pairs.lhs, pairs.rhs⟩

/-- Resolve AIGER literals in this graph (even = positive, odd = inverted).
The result preserves literal order and rejects references outside the graph. -/
def refs? (graph : AIG Nat) (literals : Array Nat) :
    Except String (AIG.RefVec graph literals.size) :=
  let v := literals.toVector.map AIG.Fanin.ofRaw
  if h : ∀ i : Fin literals.size, v[i].gate < graph.decls.size then
    .ok ⟨v, fun hi => h ⟨_, hi⟩⟩
  else .error "AIG literal outside graph"

/-- An input binding can use any existing reference in the destination graph. -/
inductive Atom (base : AIG Nat) where
  | input (index : Nat)
  | const (value : Bool)
  | bound (ref : AIG.Ref base)

def Atom.denote (a : Atom base) (assign : Nat → Bool) : Bool :=
  match a with
  | .input n => assign n
  | .const v => v
  | .bound r => ⟦base, r, assign⟧

/-- Replace a positive source node by a signed source reference or a new leaf. -/
inductive Replacement (source base : AIG Nat) where
  | redirect (ref : AIG.Ref source)
  | leaf (atom : Atom base) (invert : Bool)

def Replacement.denote (r : Replacement source base) (src dst : Nat → Bool) : Bool :=
  match r with
  | .redirect f => ⟦source, f, src⟧
  | .leaf a inv => a.denote dst ^^ inv

structure Plan (source base : AIG Nat) where
  atom : Nat → Atom base := .input
  replace : Fin source.decls.size → Option (Replacement source base) := fun _ => none

def Plan.assign (p : Plan source base) (dst : Nat → Bool) (n : Nat) : Bool :=
  (p.atom n).denote dst

/-- Input bindings induce a source assignment. Internal replacements must also
agree with that assignment for the mapped nodes to preserve their meaning. -/
def Plan.Compatible (p : Plan source base) (dst : Nat → Bool) : Prop :=
  ∀ i r, p.replace i = some r →
    ⟦source, ⟨i.val, false, i.isLt⟩, p.assign dst⟧ = r.denote (p.assign dst) dst

private structure State (p : Plan source base) where
  graph : AIG Nat
  extendsBase : AIG.IsPrefix base.decls graph.decls
  memo : Std.HashMap Nat AIG.Fanin
  mayHaveDead : Bool
  valid : ∀ (i : Fin source.decls.size) f, memo[i.val]? = some f →
    ∃ h : f.gate < graph.decls.size, ∀ a, p.Compatible a →
      ⟦graph, ⟨f.gate, f.invert, h⟩, a⟧ = ⟦source, ⟨i.val, false, i.isLt⟩, p.assign a⟧

private def State.empty (p : Plan source base) : State p :=
  ⟨base, .rfl, ∅, false, by simp⟩

@[inline] private def State.record {p : Plan source base} (s : State p) (i : Fin source.decls.size)
    (e : AIG.Entrypoint Nat) (pref : AIG.IsPrefix s.graph.decls e.aig.decls)
    (sound : ∀ a, p.Compatible a →
      ⟦e, a⟧ = ⟦source, ⟨i.val, false, i.isLt⟩, p.assign a⟧) (droppedFanin := false) : State p where
  graph := e.aig
  extendsBase := prefixTrans s.extendsBase pref
  memo := s.memo.insert i.val (.mk e.ref.gate e.ref.invert)
  mayHaveDead := s.mayHaveDead || droppedFanin
  valid := by
    intro j f lookup
    by_cases same : i.val = j.val
    · have eq : i = j := Fin.ext same
      subst j
      simp at lookup
      subst f
      simp only [AIG.Fanin.gate_mk, AIG.Fanin.invert_mk]
      exact ⟨e.ref.hgate, sound⟩
    · simp [Std.HashMap.getElem?_insert, same] at lookup
      obtain ⟨h, hs⟩ := s.valid j f lookup
      exact ⟨Nat.lt_of_lt_of_le h pref.size_le, fun a ha =>
        (AIG.denote.eq_of_isPrefix (assign := a) ⟨s.graph, ⟨_, _, h⟩⟩ _ pref).trans (hs a ha)⟩

/-- A lookup returns the signed image and its conditional preservation theorem. -/
@[inline] private def State.image {p : Plan source base} (s : State p) (r : AIG.Ref source) :
    Option { t : AIG.Ref s.graph // ∀ a, p.Compatible a →
      ⟦s.graph, t, a⟧ = ⟦source, r, p.assign a⟧ } :=
  match hlookup : s.memo[r.gate]? with
  | none => none
  | some f =>
    have good := s.valid ⟨r.gate, r.hgate⟩ f hlookup
    let t : AIG.Ref s.graph := ⟨f.gate, f.invert, good.choose⟩
    some ⟨t.flip r.invert, by
      intro a ha
      rw [AIG.Ref.denote_flip, good.choose_spec a ha]
      simp [AIG.denote]⟩

private structure Emitted (g : AIG Nat) (meaning : (Nat → Bool) → Bool) where
  entry : AIG.Entrypoint Nat
  ext : AIG.IsPrefix g.decls entry.aig.decls
  sound : ∀ a, ⟦entry, a⟧ = meaning a

private def emitAtom (g : AIG Nat) (pref : AIG.IsPrefix base.decls g.decls)
    (b : Atom base) : Emitted g b.denote :=
  match b with
  | .input n => ⟨g.mkAtomCached n, AIG.LawfulOperator.isPrefix_aig (f := AIG.mkAtomCached) _ _, by
      intro a
      simp [Atom.denote, AIG.mkAtomCached_eval_eq_mkAtom_eval]⟩
  | .const v => ⟨⟨_, g.mkConstCached v⟩, .rfl, by simp [Atom.denote]⟩
  | .bound r => ⟨⟨_, r.cast pref.size_le⟩, .rfl, fun a =>
      AIG.denote.eq_of_isPrefix ⟨_, r⟩ _ pref⟩

/-- An inverted AND with a complemented fanin represents an OR containing
that literal, so the outer AND can reduce to the shared literal. -/
private def absorbs? {g : AIG Nat} (composite other : AIG.Ref g) :
    Option (PLift (∀ a, (⟦g, composite, a⟧ && ⟦g, other, a⟧) = ⟦g, other, a⟧)) := do
  have := composite.hgate
  letI : DecidableEq (AIG.Ref g) := fun u v =>
    decidable_of_iff (u.gate = v.gate ∧ u.invert = v.invert) (by cases u; cases v; simp)
  if inv : composite.invert = true then
    match hd : g.decls[composite.gate] with
    | .gate l r =>
      have bounds := g.hdag composite.hgate hd
      let x : AIG.Ref g := ⟨l.gate, l.invert, by omega⟩
      let y : AIG.Ref g := ⟨r.gate, r.invert, by omega⟩
      have isOr (a : Nat → Bool) :
          ⟦g, composite, a⟧ = !(⟦g, x, a⟧ && ⟦g, y, a⟧) := by
        change ⟦g, ⟨composite.gate, composite.invert, composite.hgate⟩, a⟧ = _
        rw [AIG.denote_idx_gate hd]
        simp [inv, x, y]
      if h : other = x.not then
        return ⟨fun a => by rw [isOr, h, AIG.Ref.denote_not]; cases ⟦g, x, a⟧ <;> simp⟩
      else if h : other = y.not then
        return ⟨fun a => by rw [isOr, h, AIG.Ref.denote_not]; cases ⟦g, y, a⟧ <;> simp⟩
      else none
    | _ => none
  else none

private def emitAnd {g : AIG Nat} (x y : AIG.Ref g) :
    Emitted g (fun a => ⟦g, x, a⟧ && ⟦g, y, a⟧) :=
  match absorbs? x y with
  | some h => ⟨⟨g, y⟩, .rfl, fun a => (h.down a).symm⟩
  | none =>
    match absorbs? y x with
    | some h => ⟨⟨g, x⟩, .rfl, fun a => (h.down a).symm.trans (Bool.and_comm ..)⟩
    | none => ⟨g.mkGateCached ⟨x, y⟩,
        AIG.LawfulOperator.isPrefix_aig (f := AIG.mkGateCached) _ _,
        by simp [AIG.mkGateCached_eval_eq_mkGate_eval]⟩

private def State.step {p : Plan source base} (s : State p) (i : Fin source.decls.size) :
    Except String (State p) := do
  -- Release the old graph before growing its declaration array and cache.
  let ⟨graph, ext, memo, mayHaveDead, valid⟩ := s
  let s : State p := ⟨graph, ext, memo, mayHaveDead, valid⟩
  match h : p.replace i with
  | some (.redirect target) =>
    let some t := s.image target | throw "cyclic or unavailable redirect dependency"
    return s.record i ⟨_, t.val⟩ .rfl (fun a ha => (t.property a ha).trans (ha i _ h).symm)
  | some (.leaf b inv) =>
    let out := emitAtom graph ext b
    return s.record i ⟨_, out.entry.ref.flip inv⟩ out.ext (by
      intro a ha
      rw [AIG.Ref.denote_flip, AIG.denote_projected_entry, out.sound]
      exact (ha i _ h).symm)
  | none =>
    match hd : source.decls[i.val] with
    | .false => return s.record i ⟨_, s.graph.mkConstCached false⟩ .rfl (by
        intro a ha
        simp [AIG.denote_idx_false hd])
    | .atom n =>
      let out := emitAtom graph ext (p.atom n)
      return s.record i out.entry out.ext (by
        intro a ha
        simp [out.sound, AIG.denote_idx_atom hd, Plan.assign])
    | .gate l r =>
      have bounds := source.hdag i.isLt hd
      let some x := s.image ⟨l.gate, l.invert, by omega⟩ | throw "unavailable left fanin"
      let some y := s.image ⟨r.gate, r.invert, by omega⟩ | throw "unavailable right fanin"
      let out := emitAnd x.val y.val
      let k := out.entry.ref.gate
      let dropped := (k == 0 || k == x.val.gate || k == y.val.gate) &&
        ((x.val.gate != 0 && x.val.gate != k) || (y.val.gate != 0 && y.val.gate != k))
      return s.record i out.entry out.ext (by
        intro a ha
        simp [out.sound, x.property a ha, y.property a ha, AIG.denote_idx_gate hd]) dropped

private inductive Visit where
  | enter (node : Nat)
  | finish (node : Nat)

/-- Postorder traversal of the effective graph, including redirect edges. The
explicit stack avoids recursion through a long chain of AIG declarations. -/
private def run (p : Plan source base) (roots : List Nat) (initial : State p) : Except String (State p) :=
  -- Each first visit schedules a finish event and at most two fanin visits.
  go (roots.length + 3 * source.decls.size) (roots.map Visit.enter) ∅ initial
where
  go : Nat → List Visit → Std.HashSet Nat → State p → Except String (State p)
    | _, [], _, s => .ok s
    | 0, _ :: _, _, _ => .error "rebuild exhausted its graph traversal bound"
    | fuel + 1, visit :: todo, active, s => do
      match visit with
      | .finish k =>
        if h : k < source.decls.size then
          let next ← s.step ⟨k, h⟩
          go fuel todo (active.erase k) next
        else throw "rebuild node outside source graph"
      | .enter k =>
        if s.memo.contains k then go fuel todo active s
        else if active.contains k then throw s!"cyclic rebuild dependency at node {k}"
        else if h : k < source.decls.size then
          let rest := .finish k :: todo
          let visits := match p.replace ⟨k, h⟩ with
            | some (.redirect ref) => .enter ref.gate :: rest
            | some (.leaf ..) => rest
            | none =>
              match source.decls[k] with
              | .gate l r => .enter l.gate :: .enter r.gate :: rest
              | _ => rest
          go fuel visits (active.insert k) s
        else throw "rebuild node outside source graph"

private def State.mapVec {p : Plan source base} (s : State p) (v : AIG.RefVec source w) :
    Option { out : AIG.RefVec s.graph w // ∀ a, p.Compatible a → ∀ i : Fin w,
      ⟦s.graph, out.get i i.isLt, a⟧ = ⟦source, v.get i i.isLt, p.assign a⟧ } :=
  if available : ∀ i : Fin w, (s.image (v.get i i.isLt)).isSome then
    let get := fun (i : Fin w) => (s.image (v.get i i.isLt)).get (available i)
    let out : AIG.RefVec s.graph w := {
      refs := Vector.ofFn fun i => .mk (get i).val.gate (get i).val.invert
      hrefs := by intro i hi; simpa using (get ⟨i, hi⟩).val.hgate }
    some ⟨out, by
      intro a ha i
      simpa [out, AIG.RefVec.get] using (get i).property a ha⟩
  else none

/-- A rebuilt claim, with signed images of live interior refs and proofs relating
the guard and both output vectors to the original graph. -/
structure Result (c : Claim w) (base : AIG Nat) (p : Plan c.graph base) where
  child : Claim w
  /-- A fold may have made an already emitted nonconstant fanin unreachable. -/
  mayHaveDead : Bool
  extendsBase : AIG.IsPrefix base.decls child.graph.decls
  image : (r : AIG.Ref c.graph) →
    Option { t : AIG.Ref child.graph // ∀ a, p.Compatible a →
      ⟦child.graph, t, a⟧ = ⟦c.graph, r, p.assign a⟧ }
  guard_eq : ∀ a, p.Compatible a →
    ⟦child.graph, child.guard, a⟧ = ⟦c.graph, c.guard, p.assign a⟧
  lhs_eq : ∀ a, p.Compatible a → ∀ i : Fin w,
    ⟦child.graph, child.lhs.get i i.isLt, a⟧ = ⟦c.graph, c.lhs.get i i.isLt, p.assign a⟧
  rhs_eq : ∀ a, p.Compatible a → ∀ i : Fin w,
    ⟦child.graph, child.rhs.get i i.isLt, a⟧ = ⟦c.graph, c.rhs.get i i.isLt, p.assign a⟧

/-- An ordinary rebuild changes neither the input assignment nor the meaning
of the claim, even when normalization changes every output reference. -/
theorem Result.valid_of_identity {c : Claim w} (s : Result c AIG.empty {})
    (valid : s.child.Valid) : c.Valid := by
  intro a hg i
  have compatible : ({} : Plan c.graph AIG.empty).Compatible a := by intro j r h; contradiction
  exact (s.lhs_eq a compatible i).symm.trans
    ((valid a ((s.guard_eq a compatible).trans hg) i).trans (s.rhs_eq a compatible i))

/-- Rebuild the cones of the guard, both output vectors, and optional pinned
interior refs into an extension of `base`. Shared nodes are built once and
the builder folds absorbed ORs, constants, identical and complementary
fanins, and structurally hashes the remaining gates. Validation-only roots
are checked for cycles without retaining their cones in the result. -/
def claim (c : Claim w) (base : AIG Nat) (p : Plan c.graph base)
    (pins : Array (AIG.Ref c.graph) := #[]) (validateOnly : Array (AIG.Ref c.graph) := #[]) :
    Except String (Result c base p) := do
  let roots := c.guard.gate ::
    (c.lhs.refs.toList.map AIG.Fanin.gate ++ c.rhs.refs.toList.map AIG.Fanin.gate ++
      pins.toList.map AIG.Ref.gate)
  let s ← run p roots (.empty p)
  let missing := validateOnly.filterMap fun r =>
    if s.memo.contains r.gate then none else some r.gate
  unless missing.isEmpty do
    discard <| run p missing.toList s
  let some guard := s.image c.guard | throw "rebuild missing guard"
  let some lhs := s.mapVec c.lhs | throw "rebuild missing left output"
  let some rhs := s.mapVec c.rhs | throw "rebuild missing right output"
  return {
    child := ⟨s.graph, guard.val, lhs.val, rhs.val⟩, mayHaveDead := s.mayHaveDead
    extendsBase := s.extendsBase
    image := s.image, guard_eq := guard.property, lhs_eq := lhs.property, rhs_eq := rhs.property }

end XlsynthLean.Proof.Rebuild
