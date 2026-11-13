-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.IRProof

namespace XlsynthLean.IR

open Lean Meta Elab Tactic

/-- A direct call in an exported function. Its result is the local let variable
    for this particular call, even when two calls have identical applications. -/
structure CallSite where
  result : Expr
  application : Expr

/-- The opened let bindings and return expression of a fully applied XLS stage.
    Calls reachable from the return and the non-call bindings downstream of
    those calls are listed in source binding order (dependencies first). -/
structure CallSlice where
  bindings : Array Expr
  output : Expr
  calls : Array CallSite
  downstream : Array Expr

private def withStageBindings (stage : Expr) (k : Array Expr → Array Expr → Expr → MetaM α) : MetaM α := do
  let some root := stage.getAppFn.constName? | throwError "expected a named XLS stage application"
  if (← whnf (← inferType stage)).isForall then
    throwError "expected all arguments of the XLS stage {root}"
  let env ← getEnv
  let caller := (resolveXlsAliases env root).1
  let some body := (← getConstInfo caller).value? | throwError "missing body of {caller}"
  letTelescope (body.beta stage.getAppArgs) (preserveNondepLet := false) fun bindings output => do
    let mut live := collectFVars {} output
    for binding in bindings.reverse do
      if live.fvarSet.contains binding.fvarId! then
        live := collectFVars live (← binding.fvarId!.getDecl).value
    k bindings (bindings.filter fun b => live.fvarSet.contains b.fvarId!) output

/-- Inspect a stage with its local bindings in scope. Use the expressions only
    inside `k`, or close them with `mkLetFVars slice.bindings` before returning.
    Reachability follows local values, stopping at the bodies of called functions. -/
def sliceCalls (stage : Expr) (helper : Name) (k : CallSlice → MetaM α) : MetaM α := do
  let callee := (resolveXlsAliases (← getEnv) helper).1
  withStageBindings stage fun bindings live output => do
    let mut reached : FVarIdSet := {}
    let mut calls := #[]
    let mut downstream := #[]
    for binding in live do
      let rhs := (← binding.fvarId!.getDecl).value.consumeMData
      if rhs.getAppFn.constName? == some callee then
        calls := calls.push {result := binding, application := rhs}
        reached := reached.insert binding.fvarId!
      else if (rhs.find? fun e => e.isFVar && reached.contains e.fvarId!).isSome then
        downstream := downstream.push binding
        reached := reached.insert binding.fvarId!
    k {bindings, output, calls, downstream}

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

/-- Keep the owning stage application on an opened binding. The value is still
    definitionally the original RHS, and the let itself distinguishes equal calls. -/
private abbrev stageBinding {α β : Type} (_stage : α) (value : β) := value

private def openStage (goal : MVarId) (stage : Expr) (caller : Name) : MetaM MVarId := goal.withContext do
  let env ← getEnv
  let args := stage.getAppArgs
  let matchesStage := fun (e : Expr) => e.getAppNumArgs == args.size &&
    e.getAppFn.constName?.any (fun n => (resolveXlsAliases env n).1 == caller) &&
    e.getAppArgs == args
  let mentions := fun (e : Expr) => (e.find? matchesStage).isSome
  let affected := (← getLCtx).foldl (init := #[]) fun acc decl =>
    if mentions decl.type || decl.value?.any mentions then acc.push decl.fvarId else acc
  let (reverted, goal) ← goal.revert affected true
  goal.withContext do
    let target ← goal.getType
    unless mentions target do throwError "stage {stage} does not occur in the goal or hypotheses"
    withStageBindings stage fun _ bindings output => do
      -- Only calls to sibling functions in this export need a persistent
      -- identity. Close ordinary SSA computations as lets at their users.
      let mut calls := #[]
      let mut operands := #[]
      let mut ctx ← getLCtx
      for b in bindings do
        let rhs := (← b.fvarId!.getDecl).value
        if rhs.consumeMData.getAppFn.constName?.any (fun n => n.getPrefix == caller.getPrefix) then
          let closed ← mkLetFVars operands (← mkAppM ``stageBinding #[stage, rhs])
          ctx := ctx.modifyLocalDecl b.fvarId! (·.setValue closed)
          calls := calls.push b
        else
          operands := operands.push b
      let output ← mkLetFVars operands output
      let target := target.replace fun e => if matchesStage e then some output else none
      let type ← withLCtx' ctx <| mkLetFVars calls target (usedLetOnly := false)
      let opened ← goal.replaceTargetDefEq type
      return (← opened.introNP (calls.size + reverted.size)).2

/-- Specialize each contract at its live operands, then forget the selected
    definitions. Other bindings retain their identities for the next cut. -/
private def cutCalls (goal : MVarId) (stage law : Expr) (helper base : Name)
    (count : Option Nat) : MetaM MVarId := goal.withContext do
  let env ← getEnv
  let some root := stage.getAppFn.constName? | throwError "expected a named XLS stage application"
  let caller := (resolveXlsAliases env root).1
  let stage := mkAppN (mkConst caller) stage.getAppArgs
  let markedRhs? := fun (value : Expr) => do
    guard (value.isAppOfArity ``stageBinding 4 && value.getArg! 2 == stage)
    return value.getArg! 3
  let callee := (resolveXlsAliases env helper).1
  let sites (goal : MVarId) : MetaM (Bool × Array (FVarId × Expr × Expr)) := goal.withContext do
    let mut calls := #[]
    let mut opened := false
    for decl in ← getLCtx do
      if let some value := decl.value? then
        let (isStage, call) ← letTelescope value (preserveNondepLet := false) fun operands result => do
          let some rhs := markedRhs? result | return (false, none)
          if rhs.consumeMData.getAppFn.constName? != some callee then return (true, none)
          let site := {result := decl.toExpr, application := rhs.consumeMData : CallSite}
          let (proof, fact) ← specializeCall law site
          return (true, some (decl.fvarId, ← mkLetFVars operands proof, ← mkLetFVars operands fact))
        opened := opened || isStage
        if let some call := call then calls := calls.push call
    return (opened, calls)
  let (opened, calls) ← sites goal
  let (goal, calls) ← if opened then pure (goal, calls) else do
    let goal ← openStage goal stage caller
    pure (goal, (← sites goal).2)
  goal.withContext do
    let ctx ← getLCtx
    if calls.isEmpty then throwError "no reachable calls to {helper} in {stage.getAppFn}"
    if let some expected := count then
      unless calls.size == expected do throwError "expected {expected} calls, found {calls.size}"
    let mut hypotheses := #[]
    let resultName := fun (i : Nat) => ctx.getUnusedName (base.appendAfter (toString i))
    for ((_, proof, fact), i) in calls.zipIdx do
      hypotheses := hypotheses.push {
        userName := ctx.getUnusedName ((resultName i).appendAfter "_spec")
        type := fact, value := proof : Hypothesis}
    let (_, goal) ← goal.assertHypotheses hypotheses
    goal.withContext do
      -- The stronger goal forgets the selected values simultaneously. Its
      -- proof specializes to the original let values in the old context.
      let mut ctx ← getLCtx
      for ((site, _, _), i) in calls.zipIdx do
        let name := resultName i
        ctx := (ctx.setUserName site name).modifyLocalDecl site fun decl =>
          .cdecl decl.index site name decl.type .default decl.kind
      -- A local proof such as `let h : stage x = rhs := rfl` may depend
      -- on the concrete value. Check the generalized telescope once.
      let target ← goal.getType
      let first := (← calls[0]!.1.getDecl).index
      let later := ctx.foldl (init := #[]) fun xs d =>
        if d.index >= first then xs.push d.toExpr else xs
      let generalized ← withLCtx' ctx <| mkForallFVars later target
        (usedLetOnly := false) (generalizeNondepLet := false)
      unless ← isTypeCorrect (← instantiateMVars generalized) do
        throwError "cannot abstract calls to {helper}: the resulting goal or hypotheses depend on their values"
      let next ← mkFreshExprMVarAt ctx (← getLocalInstances) target .syntheticOpaque (← goal.getTag)
      goal.assign next
      return next.mvarId!

/-- Factor a stage proof at all calls to a helper. Each fresh result has the
    helper's proved contract at the actual operands. The remaining goal and
    conditional premises are left to the proof author. -/
syntax "xls_cut " term:max " at " ident " using " term:max (" as " ident)? (" expect " num)? : tactic

elab_rules : tactic
| `(tactic| xls_cut $stage:term at $helper:ident using $contract:term $[as $prefixName:ident]? $[expect $count:num]?) =>
  focus do
    let (app, op, law) ← withMainContext do
      let app ← instantiateMVars (← elabTerm stage none)
      let op ← resolveGlobalConstNoOverload helper
      return (app, op, ← instantiateMVars (← elabTerm contract none))
    let base := prefixName.map (·.getId) |>.getD `cut
    replaceMainGoal [← cutCalls (← getMainGoal) app law op base (count.map (·.getNat))]

end XlsynthLean.IR
