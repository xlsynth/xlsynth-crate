-- SPDX-License-Identifier: Apache-2.0

import Std

namespace XlsynthLean.IR

open Lean in
/-- Append bit vectors or arrays in order. Splitting the operands in half keeps the
    nesting depth logarithmic in the number of operands. -/
macro "concatMany![" operands:term,+ "]" : term => do
  let args := operands.getElems
  let rec expand (start : Nat) : Nat → MacroM (TSyntax `term)
    | 0 => Lean.Macro.throwError "expected at least one concat operand"
    | 1 => pure args[start]!
    | n + 2 => do
      let count := n + 2
      let half := count / 2
      let high ← expand start half
      let low ← expand (start + half) (count - half)
      `($high ++ $low)
  decreasing_by all_goals omega
  expand 0 args.size

/-- An empty bit concatenation has width zero. -/
macro "concatBits![" operands:term,* "]" : term =>
  if operands.getElems.isEmpty then `(0#0) else `(concatMany![$operands,*])

/-- The LSB-first bit layout used at the evaluator boundary and by aggregate one-hot select. -/
class Packed (α : Type) where
  width : Nat
  pack : α → Nat
  unpack : Nat → α

instance : Packed (BitVec w) where
  width := w
  pack := BitVec.toNat
  unpack := BitVec.ofNat w

instance : Packed Unit where
  width := 0
  pack _ := 0
  unpack _ := ()

-- The last tuple field and the first array element occupy the low bits.
instance [a : Packed α] [b : Packed β] : Packed (α × β) where
  width := a.width + b.width
  pack x := (a.pack x.1 <<< b.width) ||| b.pack x.2
  unpack x := (a.unpack (x >>> b.width), b.unpack x)

instance [a : Packed α] : Packed (Vector α n) where
  width := n * a.width
  pack xs := xs.foldr (fun x acc => (acc <<< a.width) ||| a.pack x) 0
  unpack x := Vector.ofFn fun i => a.unpack (x >>> (i.val * a.width))

/-- Signed XLS division saturates according to the dividend when dividing by zero. -/
def sdiv (x y : BitVec w) : BitVec w :=
  if y == 0 then (if x.msb then BitVec.intMin w else BitVec.intMax w) else x.sdiv y

/-- XLS updates only bits inside the destination, even with an arbitrarily wide start. -/
def bitSliceUpdate (x : BitVec w) (start : BitVec i) (v : BitVec k) : BitVec w :=
  if start.toNat ≥ w then x else
    let mask := ((BitVec.allOnes k).setWidth w) <<< start.toNat
    (x &&& ~~~mask) ||| ((v.setWidth w) <<< start.toNat)

/-- Out-of-range reads clamp to the last element of the indexed dimension. -/
def arrayIndex [Inhabited α] (xs : Vector α n) (i : Nat) : α :=
  xs.getD (min i (n - 1)) default

/-- A slice that crosses the end repeats its last element. -/
def arraySlice [Inhabited α] (k : Nat) (xs : Vector α n) (start : Nat) : Vector α k :=
  Vector.ofFn fun i => arrayIndex xs (start + i.val)

/-- Applies a (possibly nested) update only when this dimension is in bounds. -/
def arrayModify (xs : Vector α n) (i : Nat) (f : α → α) : Vector α n :=
  if h : i < n then xs.set i (f xs[i]) else xs

/-- The zero input selects an extra, highest output bit. -/
def oneHot (lsbPriority : Bool) (x : BitVec w) : BitVec (w + 1) :=
  if x == 0 then BitVec.twoPow _ w
  else if lsbPriority then (x &&& -x).setWidth _
  else BitVec.twoPow _ (w - 1 - x.clz.toNat)

/-- Multi-hot input encodes the bitwise OR of every selected index. -/
def encode (outWidth : Nat) (x : BitVec w) : BitVec outWidth :=
  BitVec.ofNat _ ((List.range w).foldl (fun acc i =>
    if x.getLsbD i then acc ||| i else acc) 0)

/-- Case zero wins whenever selector bit zero is set. -/
def prioritySel (s : BitVec w) (cases : Array α) (fallback : α) : α :=
  cases.getD (if s == 0 then cases.size else s.ctz.toNat) fallback

/-- All selected case values contribute, including for arrays and tuples. -/
def oneHotSel [p : Packed α] (s : BitVec w) (cases : Array α) : α :=
  p.unpack (cases.toList.zipIdx.foldl (fun acc (v, i) =>
    if s.getLsbD i then acc ||| p.pack v else acc) 0)

end XlsynthLean.IR
