-- SPDX-License-Identifier: Apache-2.0

import Lean
import Lean.Meta.Native
import Lean.Replay

open Lean

/-! This program is elaborated with only the pinned Lean toolchain on its search
path. Its input modules are opened as data, so their syntax and command
elaborators cannot change the checks below. -/

/-- The declaration must come from its designated module. -/
private def requireOrigin (env : Environment) (decl moduleName : Name) : IO Unit := do
  let some index := env.getModuleIdxFor? decl
    | throw <| IO.userError s!"missing declaration {decl}"
  unless env.header.modules[index.toNat]?.any (·.module == moduleName) do
    throw <| IO.userError s!"{decl} must be declared by module {moduleName}"

/-- Recheck the proof modules against their trusted imports, excluding the
previously elaborated user declarations from the starting kernel environment. -/
private def replayProof (env : Environment) (proofModules : List Name) : IO Environment := do
  let mut imports : Array Import := #[{ module := `Target }]
  let mut constants : Std.HashMap Name ConstantInfo := {}
  for name in proofModules do
    let some index := env.getModuleIdx? name
      | throw <| IO.userError s!"missing proof module {name}"
    let data := env.header.moduleData[index.toNat]!
    for dep in data.imports do
      unless proofModules.contains dep.module || imports.any (·.module == dep.module) do
        imports := imports.push dep
    for ci in data.constants do
      constants := constants.insert ci.name ci
  let trusted ← importModules imports {} (loadExts := false)
  for name in proofModules do
    if (trusted.getModuleIdx? name).isSome then
      throw <| IO.userError s!"trusted imports depend on proof module {name}"
  let checked ← trusted.toKernelEnv.replay constants
  return Environment.ofKernelEnv checked

/-- Apply the IR checker's axiom policy to the exact, kernel-checked declaration. -/
private def auditIr (checked : Environment) (name : Name) : MetaM Nat := do
  let mut verified := 0
  -- Read dependencies and axiom types from the replayed kernel declarations.
  -- The original environment is used only to execute native Boolean checks.
  let axioms ← withEnv checked (collectAxioms name)
  for ax in axioms do
    if [``propext, ``Classical.choice, ``Quot.sound].contains ax then continue
    let some (.axiomInfo info) := checked.toKernelEnv.find? ax
      | throwError "missing axiom in {name}: {ax}"
    let some (ty, computation, value) := info.type.eq?
      | throwError "unverified axiom in {name}: {ax}"
    unless ty.isConstOf ``Bool && value.isConstOf ``Bool.true do
      throwError "unverified axiom in {name}: {ax}"
    match ← Meta.nativeEqTrue `xlsynth_ir_check computation with
    | .success _ => verified := verified + 1
    | .notTrue => throwError "native Boolean check failed for {ax}"
  return verified

def main (args : List String) : IO Unit := do
  let root :: args := args | throw <| IO.userError "expected a compiled module directory"
  let (selected, modules) ← match args with
    | "ir" :: theoremName :: mainModule :: helpers =>
      pure (some (theoremName.toName, mainModule.toName), mainModule :: helpers)
    | "aig" :: helpers => pure (none, helpers)
    | _ => throw <| IO.userError "expected aig or ir with a theorem and its defining module"
  let builtin ← getBuiltinSearchPath (← getBuildDir)
  searchPathRef.set (builtin ++ [System.FilePath.mk root])
  let env ← importModules #[{ module := `Target }, { module := `UserProof }] {} (loadExts := false)
  let theoremName := `UserProof.equivalent
  let targetName := `Target.Equivalent
  requireOrigin env targetName `Target
  requireOrigin env theoremName `UserProof
  let imported := (modules.map String.toName).filter fun name => (env.getModuleIdx? name).isSome
  let checked ← replayProof env (`UserProof :: imported)
  let some (.thmInfo thm) := checked.toKernelEnv.find? theoremName
    | throw <| IO.userError s!"expected a theorem named {theoremName}"
  unless Kernel.isDefEqGuarded checked {} thm.type (mkConst targetName) do
    throw <| IO.userError s!"{theoremName} has the wrong type; expected {targetName}"
  if let some (entry, moduleName) := selected then
    requireOrigin env entry moduleName
    let some (.thmInfo decl) := checked.toKernelEnv.find? entry
      | throw <| IO.userError s!"expected a theorem named {entry}"
    unless Kernel.isDefEqGuarded checked {} decl.type (mkConst targetName) do
      throw <| IO.userError s!"{entry} has the wrong type; expected {targetName}"
    unless thm.value.consumeMData.isConstOf entry do
      throw <| IO.userError s!"{theoremName} must reference the selected theorem {entry}"
    let verified ← (auditIr checked entry).run' |>.toIO'
      { fileName := "<proof audit>", fileMap := default } { env }
    IO.println s!"{entry}: checked axiom dependencies and replayed {verified} native Boolean checks"
  IO.println "xlsynth: verified UserProof.equivalent"
