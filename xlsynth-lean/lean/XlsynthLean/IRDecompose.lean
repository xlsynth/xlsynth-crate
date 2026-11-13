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

/-- Inspect a stage with its local bindings in scope. Use the expressions only
    inside `k`, or close them with `mkLetFVars slice.bindings` before returning.
    Reachability follows local values, stopping at the bodies of called functions. -/
def sliceCalls (stage : Expr) (helper : Name) (k : CallSlice → MetaM α) : MetaM α := do
  let some root := stage.getAppFn.constName? | throwError "expected a named XLS stage application"
  if (← whnf (← inferType stage)).isForall then
    throwError "expected all arguments of the XLS stage {root}"
  let env ← getEnv
  let caller := (resolveXlsAliases env root).1
  let callee := (resolveXlsAliases env helper).1
  let some body := (← getConstInfo caller).value? | throwError "missing body of {caller}"
  letTelescope (body.beta stage.getAppArgs) (preserveNondepLet := false) fun bindings output => do
    let mut live := collectFVars {} output
    for binding in bindings.reverse do
      if live.fvarSet.contains binding.fvarId! then
        live := collectFVars live (← binding.fvarId!.getDecl).value
    let mut reached : FVarIdSet := {}
    let mut calls := #[]
    let mut downstream := #[]
    for binding in bindings do
      if live.fvarSet.contains binding.fvarId! then
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

/-- Abstract calls in the stage output and in their contracts. Close the other
    bindings as lets so shared operands occur once in each resulting expression. -/
private def cutCalls (goal : MVarId) (stage law : Expr) (helper base : Name)
    (count : Option Nat) : MetaM MVarId := goal.withContext do
  let env ← getEnv
  let some root := stage.getAppFn.constName? | throwError "expected a named XLS stage application"
  let caller := (resolveXlsAliases env root).1
  let args := stage.getAppArgs
  let matchesStage := fun (e : Expr) => e.getAppNumArgs == args.size &&
    e.getAppFn.constName?.any (fun name => (resolveXlsAliases env name).1 == caller) &&
    e.getAppArgs == args
  let mentions := fun (e : Expr) => (e.find? matchesStage).isSome
  let ctx ← getLCtx
  let affected := ctx.foldl (init := #[]) fun acc decl =>
    if mentions decl.type || decl.value?.any mentions then acc.push decl.fvarId else acc
  let (reverted, goal) ← goal.revert affected true
  goal.withContext do
    let target ← goal.getType
    let (generalized, actual, n) ← sliceCalls stage helper fun slice => do
      if slice.calls.isEmpty then throwError "no reachable calls to {helper} in {stage.getAppFn}"
      if let some expected := count then
        unless slice.calls.size == expected do
          throwError "expected {expected} calls, found {slice.calls.size}"
      let selected := slice.calls.map (·.result)
      let selectedIds := FVarIdSet.ofArray (selected.map (·.fvarId!))
      let remaining := slice.bindings.filter fun binding => !selectedIds.contains binding.fvarId!
      let specs ← slice.calls.mapM (specializeCall law)
      let actual ← (selected ++ specs.map Prod.fst).mapM (mkLetFVars slice.bindings ·)
      let names ← selected.mapIdxM fun i node => do
        return (ctx.getUnusedName (base.appendAfter (toString i)), ← inferType node)
      let generalized ← withLocalDeclsDND names fun vars => do
        let abstractCalls := fun e => do return (← mkLetFVars remaining e).replaceFVars selected vars
        let output ← abstractCalls slice.output
        let mut target := target.replace fun e => if matchesStage e then some output else none
        for ((name, _), (_, fact)) in (names.zip specs).reverse do
          let specName := ctx.getUnusedName (name.appendAfter "_spec")
          target := mkForall specName .default (← abstractCalls fact) target
        mkForallFVars vars target
      return (generalized, actual, selected.size)
    -- Instantiating the new goal with the original calls and their proofs
    -- reconstructs the old goal, including its reverted hypotheses.
    let next ← mkFreshExprSyntheticOpaqueMVar generalized (← goal.getTag)
    goal.assign (mkAppN next actual)
    return (← next.mvarId!.introNP (2 * n + reverted.size)).2

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
