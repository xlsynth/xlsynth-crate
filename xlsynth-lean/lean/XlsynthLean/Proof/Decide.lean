-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Proof.Miter
import Lean.Elab.Tactic.Basic
import Lean.Meta.Eval
import Lean.Meta.Native
import Lean.Meta.Tactic.BVDecide.LRAT.Cert
import Lean.Meta.Tactic.BVDecide.TacticContext
import Std.Tactic.BVDecide.Reflect

namespace XlsynthLean.Proof

open Std.Sat
open Std.Tactic.BVDecide.Reflect

private theorem valid_of_certificate (c : Claim width) (cert : String)
    (accepted : verifyCert (AIG.toCNF c.bad) cert = true) : c.Valid :=
  c.valid_of_bad_unsat ((AIG.toCNF_equisat c.bad).mp (verifyCert_correct _ cert accepted))

/-- Replay a DIMACS model on the claim itself before reporting it. Each input atom
    has a CNF variable after the miter's graph nodes, and DIMACS is one-based. -/
private def counterexample (c : Claim width) (miterSize : Nat)
    (model : Array (Bool × Nat)) : Option String := Id.run do
  let mut inputs := 0
  let mut trueAtoms := #[]
  for (positive, literal) in model do
    if positive && literal > miterSize then
      let atom := literal - miterSize - 1
      inputs := inputs ||| (1 <<< atom)
      trueAtoms := trueAtoms.push atom
  let inputWidth := inputs.log2 + 1
  let observed : Aig.Circuit inputWidth (width + width + 1) :=
    ⟨c.graph, (c.lhs.append c.rhs).push c.guard⟩
  let values := observed.eval (BitVec.ofNat inputWidth inputs)
  unless values.getLsbD (width + width) do return none
  for bit in [0:width] do
    let lhs := values.getLsbD bit
    let rhs := values.getLsbD (width + bit)
    if lhs != rhs then
      return some (s!"input atoms set to true: {trueAtoms.toList} (all others false)\n" ++
        s!"output bit {bit}: lhs={lhs}, rhs={rhs}")
  return none

open Lean Meta Elab Tactic

/-- Solve a closed guarded AIG equivalence claim with Lean's SAT solver and LRAT checker. -/
syntax "aig_decide" ("(" &"timeout" ":=" num ")")? : tactic

elab_rules : tactic
  | `(tactic| aig_decide $[(timeout := $seconds:num)]?) => withMainContext do
    let target ← zetaReduce (← instantiateMVars (← getMainTarget))
    let_expr Claim.Valid width c := target
      | throwError "aig_decide expects a goal of the form Claim.Valid c"
    if c.hasFVar || width.hasFVar || c.hasMVar || width.hasMVar then
      throwError "aig_decide requires a closed, concrete claim"

    -- Package the width with its claim so native evaluation has a fixed result type.
    let packedType := mkApp2 (mkConst ``Sigma [.zero, .zero]) (mkConst ``Nat) (mkConst ``Claim)
    let packed := mkApp4 (mkConst ``Sigma.mk [.zero, .zero]) (mkConst ``Nat) (mkConst ``Claim) width c
    let ⟨_, claim⟩ ← unsafe evalExpr (Σ n, Claim n) packedType packed
    let bad := claim.bad
    let result ← IO.FS.withTempFile fun _ path => do
      let config : Lean.Elab.Tactic.BVDecide.BVDecideConfig := {}
      let timeout := seconds.map (·.getNat) |>.getD config.timeout
      let ctx ← Lean.Meta.Tactic.BVDecide.TacticContext.new path { config with timeout }
      Lean.Meta.Tactic.BVDecide.runExternal (AIG.toCNF bad) ctx.solver path
        ctx.config.trimProofs ctx.config.timeout ctx.config.binaryProofs ctx.config.solverMode
    match result with
    | .error model =>
      match counterexample claim bad.aig.decls.size model with
      | some description => throwError "aig_decide found a counterexample:\n{description}"
      | none => throwError "aig_decide: the SAT model did not reproduce a guarded output mismatch"
    | .ok cert =>
      let entry := mkApp2 (mkConst ``Claim.bad) width c
      let cnf := mkApp (mkConst ``AIG.toCNF) entry
      let certExpr := toExpr cert
      let checkExpr := mkApp2 (mkConst ``verifyCert) cnf certExpr
      match ← nativeEqTrue `aig_decide checkExpr (axiomDeclRange? := (← getRef)) with
      | .notTrue => throwError "aig_decide: LRAT certificate failed verification"
      | .success checked =>
        closeMainGoal `aig_decide (mkApp4 (mkConst ``valid_of_certificate) width c certExpr checked)

end XlsynthLean.Proof
