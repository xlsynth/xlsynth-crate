-- SPDX-License-Identifier: Apache-2.0

import Generated

open XlsBlock

-- The unequal inputs distinguish a swap even though both fields have type u1.
example : (observe {} { left := 1, right := 0 }).pair = (1#1, 0#1, ()) ∧
          (observe {} { left := 1, right := 0 }).selected = 1#1 := ⟨rfl, rfl⟩

-- Check every pair against its scalar inputs independently of any encoder.
example (q : State) (x : Input) :
    (observe q x).pair = (x.left, x.right, ()) ∧
    (observe q x).selected = x.left := ⟨rfl, rfl⟩
