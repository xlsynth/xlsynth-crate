-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.IRProof

namespace XlsynthLean.IR

open Lean Meta Elab Tactic

/-- One call to a selected helper in a fully applied exported stage. The result
    is the shared node in the original export, definitionally equal to the call. -/
structure CallSite where
  result : Expr
  application : Expr

/-- Selected calls and the node definitions that connect their results to users
    within this stage. Definitions upstream of every selected result stay folded. -/
structure CallSlice where
  calls : Array CallSite
  downstream : Array Name

/-- Find reachable direct calls to an arbitrary helper, in dependency order.
    Work on the exported Lean definitions so projections, aggregates, and the
    types of the stage arguments come from the actual theorem target. -/
def sliceCalls (stage : Expr) (helper : Name) : MetaM CallSlice := do
  let some root := stage.getAppFn.constName? | throwError "expected a named XLS stage application"
  if (← whnf (← inferType stage)).isForall then
    throwError "expected all arguments of the XLS stage {root}"
  let some caller := (← withReducible (whnf (mkConst root))).constName?
    | throwError "expected a named XLS function at {root}"
  let some callee := (← withReducible (whnf (mkConst helper))).constName?
    | throwError "expected a named XLS helper at {helper}"
  let args := stage.getAppArgs
  let mut reached : NameSet := {}
  let mut calls := #[]
  let mut downstream := #[]
  for name in ← exportedNodes root do
    let some body := (← getConstInfo name).value? | throwError "missing body of {name}"
    let rhs := (body.beta args).consumeMData
    if name.getPrefix == caller && rhs.getAppFn.constName? == some callee then
      calls := calls.push {result := mkAppN (mkConst name) args, application := rhs}
      reached := reached.insert name
    else if body.getUsedConstants.any reached.contains then
      downstream := downstream.push name
      reached := reached.insert name
  return {calls, downstream}

/-- Instantiate one contract with this call's actual operands, then refold the
    call in the proposition to its shared node by definitional equality. -/
private def specializeCall (law : Expr) (site : CallSite) : MetaM (Expr × Expr) := do
  let proof := mkAppN law site.application.getAppArgs
  unless ← isTypeCorrect proof do
    throwError "contract does not accept the arguments of {site.application}"
  let fact ← inferType proof
  unless ← isProp fact do throwError "expected a proof of a proposition, got {fact}"
  let env ← getEnv
  let refolded ← Meta.transform fact (pre := fun e => do
    if e.getAppNumArgs == site.application.getAppNumArgs then
      if let some (.defnInfo info) := e.getAppFn.constName?.bind env.find? then
        if info.hints.isAbbrev || e.getAppFn == site.application.getAppFn then
          if ← withReducible (isDefEq e site.application) then return .done site.result
    return .continue)
  if refolded == fact then
    throwError "the instantiated contract does not mention the selected call: {fact}"
  return (proof, refolded)

/-- Abstract the named SSA results simultaneously in the goal and its
    hypotheses. Matching the exact node applications preserves DAG sharing
    and avoids expanding cones during definitional equality checks. -/
private def abstractResults (goal : MVarId) (results : Array (Name × Expr)) : MetaM MVarId := goal.withContext do
  let nodes := results.map Prod.snd
  let indices : Std.HashMap Expr Nat := Std.HashMap.ofList (nodes.zipIdx.toList)
  let decls ← results.mapM fun (name, node) => do return (name, ← inferType node)
  let mentionsNode := fun (e : Expr) => (e.find? indices.contains).isSome
  let affected := (← getLCtx).foldl (init := #[]) fun acc decl =>
    if mentionsNode decl.type || decl.value?.any mentionsNode then acc.push decl.fvarId else acc
  let (reverted, goal) ← goal.revert affected true
  goal.withContext do
    let target ← goal.getType
    let generalized ← withLocalDeclsDND decls fun vars =>
      mkForallFVars vars (target.replace fun e => indices[e]?.map fun i => vars[i]!)
    let next ← mkFreshExprSyntheticOpaqueMVar generalized (← goal.getTag)
    -- Applying the generalized proof to the original nodes reconstructs the
    -- reverted goal. The kernel checks this application in the enclosing theorem.
    goal.assign (mkAppN next nodes)
    let (_, next) ← next.mvarId!.introNP nodes.size
    return (← next.introNP reverted.size).2

/-- Factor a stage proof at all calls to a helper. Each fresh result has the
    helper's proved contract at the actual operands. The remaining goal and
    conditional premises are left to the proof author. -/
syntax "xls_cut " term:max " at " ident " using " term:max (" as " ident)? (" expect " num)? : tactic

elab_rules : tactic
| `(tactic| xls_cut $stage:term at $helper:ident using $contract:term $[as $prefixName:ident]? $[expect $count:num]?) =>
  focus do
    let (slice, law) ← withMainContext do
      let app ← instantiateMVars (← elabTerm stage none)
      let op ← resolveGlobalConstNoOverload helper
      let slice ← sliceCalls app op
      if slice.calls.isEmpty then throwErrorAt helper "no reachable calls to {op} in {app.getAppFn}"
      if let some n := count then
        unless slice.calls.size == n.getNat do
          throwErrorAt helper "expected {n.getNat} calls, found {slice.calls.size}"
      return (slice, ← instantiateMVars (← elabTerm contract none))
    let prefixName := prefixName.map (·.getId) |>.getD `cut
    let mut goal ← getMainGoal
    let mut cuts := #[]
    for site in slice.calls, i in [:slice.calls.size] do
      let (newGoal, arg) ← goal.withContext do
        let (proof, fact) ← specializeCall law site
        let ctx ← getLCtx
        let name := ctx.getUnusedName (prefixName.appendAfter (toString i))
        let (_, next) ← goal.note (ctx.getUnusedName (name.appendAfter "_spec")) proof (some fact)
        return (next, (name, site.result))
      goal := newGoal
      cuts := cuts.push arg
    replaceMainGoal [goal]
    if !slice.downstream.isEmpty then
      let names := slice.downstream.map mkIdent
      evalTactic (← `(tactic| simp (config := { failIfUnchanged := false }) only [$[$names:ident],*] at *))
    if let some openGoal := (← getGoals).head? then
      replaceMainGoal [← abstractResults openGoal cuts]

end XlsynthLean.IR
