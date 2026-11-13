-- SPDX-License-Identifier: Apache-2.0

import Target
import XlsynthLean.IRDecompose

namespace DuplicateProof
open Lean Meta Elab Tactic

abbrev stage := Generated.Lhs.eval
abbrev helper := Generated.Lhs.ir_helper
theorem helper_spec (x : BitVec 8) : helper x = ~~~x := rfl
theorem negate_spec (x : BitVec 8) : Generated.Lhs.ir_negate x = -x := rfl

theorem equivalent : Target.Equivalent := by
  intro x
  run_tac withMainContext do
    let some (_, app, _) := (← getMainTarget).eq? | throwError "expected an equality"
    XlsynthLean.IR.sliceCalls app ``helper fun s => do
      let #[first, second] := s.calls | throwError "expected two reachable calls"
      unless first.result.isFVar && second.result.isFVar && first.result != second.result &&
          first.application == second.application do
        throwError "expected distinct bindings of the same helper application"
      let #[_, _, output] := s.downstream | throwError "expected three downstream bindings"
      unless output == s.output && (← isDefEq (← mkLetFVars s.bindings output) app) do
        throwError "the downstream slice does not reconstruct the stage"
  -- Abstracting the results would invalidate this local definition's proof.
  let concrete : stage x = Generated.Rhs.eval x := rfl
  fail_if_success xls_cut (stage x) at helper using helper_spec expect 2
  clear concrete
  have h : stage x = Generated.Rhs.eval x := rfl
  change stage x = _
  xls_cut (Generated.Lhs.eval x) at helper using helper_spec as r expect 2
  xls_cut (stage x) at Generated.Lhs.ir_negate using negate_spec as negated expect 1
  change ((r0 + r0) ^^^ r1) + negated0 = Generated.Rhs.eval x at h ⊢
  rw [negated0_spec] at h ⊢
  exact h

end DuplicateProof
