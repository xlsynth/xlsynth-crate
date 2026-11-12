-- SPDX-License-Identifier: Apache-2.0

import Target
import XlsynthLean.IRDecompose

namespace RouteProof

abbrev exchange := Generated.Lhs.ir_exchange
abbrev Pair := BitVec 7 × BitVec 7 × Unit

theorem exchange_spec (p : BitVec 1) (pair : Pair) :
    exchange p pair = if p = 0 then pair else (pair.2.1, pair.1, ()) := by
  rcases pair with ⟨x, y, ⟨⟩⟩
  have h : p = 0 ∨ p = 1 := by bv_decide
  rcases h with rfl | rfl <;> xls_unfold exchange <;> rfl

theorem unchanged_when_zero (p : BitVec 1) (pair : Pair) :
    p = 0 → exchange p pair = pair := by
  intro h
  simp [exchange_spec, h]

private theorem unrelated (_p : BitVec 1) (pair : Pair) : pair = pair := rfl

theorem pack_spec (x y : BitVec 7) : Generated.Rhs.ir_pack x y = (x, y, ()) := rfl

theorem equivalent : Target.Equivalent := by
  intro p x y
  fail_if_success xls_cut (Generated.Lhs.eval p x y) at exchange using exchange_spec expect 3
  fail_if_success xls_cut (Generated.Rhs.eval p x y) at exchange using exchange_spec
  fail_if_success xls_cut (Generated.Lhs.eval p x y) at exchange using unrelated
  xls_cut (Generated.Lhs.eval p x y) at exchange using exchange_spec as routed expect 2
  xls_cut (Generated.Rhs.eval p x y) at Generated.Rhs.ir_pack using pack_spec as ref expect 1
  change routed1 = ref0
  by_cases h : p = 0 <;> simp_all <;> first | rfl | exact ⟨rfl, rfl⟩

example (p : BitVec 1) (x y : BitVec 7) (hp : p = 0) :
    Generated.Lhs.eval p x y = (x, y, ()) := by
  xls_cut (Generated.Lhs.eval p x y) at exchange using unchanged_when_zero as routed
  exact (routed1_spec hp).trans (routed0_spec hp)

end RouteProof
