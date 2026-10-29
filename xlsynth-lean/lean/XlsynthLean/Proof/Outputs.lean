-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Claim

open Std.Sat

namespace XlsynthLean.Proof

variable {graph : AIG Nat}

/-- Resolve numeric output positions with bounds checking. In particular, a
numeric position equal to the width is an error instead of wrapping to zero. -/
def outputBits? (width : Nat) (positions : List Nat) : Except String (List (Fin width)) :=
  positions.mapM fun i =>
    if h : i < width then .ok ⟨i, h⟩
    else .error s!"output position {i} out of bounds for width {width}"

private def selectOutputs (outputs : AIG.RefVec graph width) (bits : List (Fin width)) :
    AIG.RefVec graph bits.length where
  refs := Vector.ofFn fun i => outputs.refs[bits[i.val].val]
  hrefs := by
    intro i hi
    simpa using outputs.hrefs (bits[i].isLt)

/-- Select the same ordered bits on both sides of a guarded claim. -/
def Claim.project (c : Claim width) (bits : List (Fin width)) : Claim bits.length :=
  { graph := c.graph, guard := c.guard
    lhs := selectOutputs c.lhs bits, rhs := selectOutputs c.rhs bits }

private theorem get_selectOutputs (outputs : AIG.RefVec graph width) (bits : List (Fin width))
    (i : Nat) (h : i < bits.length) :
    (selectOutputs outputs bits).get i h = outputs.get bits[i].val bits[i].isLt := by
  simp [selectOutputs, AIG.RefVec.get]

/-- Groups can overlap and be permuted. Coverage requires every original bit to
appear in at least one group whose projected claim is proved. -/
theorem Claim.valid_of_cover (c : Claim width) (groups : List (List (Fin width)))
    (covers : ∀ bit : Fin width, bit ∈ groups.flatten)
    (parts : ∀ group ∈ groups, (c.project group).Valid) : c.Valid := by
  intro assign enabled bit
  obtain ⟨group, included, membership⟩ := List.mem_flatten.mp (covers bit)
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem membership
  have eq := parts group included assign enabled ⟨i, hi⟩
  simpa only [project, get_selectOutputs] using eq

end XlsynthLean.Proof
