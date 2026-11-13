// SPDX-License-Identifier: Apache-2.0

use std::fs;
#[cfg(feature = "lean-tests")]
use std::path::PathBuf;

#[cfg(feature = "lean-tests")]
use xlsynth_lean::{
    LeanError, LeanIrAxiomPolicy, LeanIrProofSources, LeanTheoremName,
    run_ir_function_equivalence_proof, run_ir_function_equivalence_proof_with_policy,
};
use xlsynth_lean::{LeanModuleName, export_ir_function, prepare_ir_function_equivalence_target};
use xlsynth_pir::ir_parser::Parser;

const IDENTITY: &str = r#"package p
top fn passthrough(unused: (bits[1], bits[7]) id=1, values: bits[3][2] id=2) -> bits[3][2] {
  ret output: bits[3][2] = identity(values, id=3)
}
"#;

#[test]
fn proof_target_uses_the_ordinary_export_for_both_models() {
    let pkg = Parser::new(include_str!("fixtures/decomposition/route.ir"))
        .parse_and_validate_package()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    prepare_ir_function_equivalence_target(&pkg, None, &pkg, Some("reference"))
        .unwrap()
        .write_to(dir.path())
        .unwrap();
    for (side, top) in [("Lhs", None), ("Rhs", Some("reference"))] {
        let module = LeanModuleName::new(format!("Generated.{side}")).unwrap();
        let ordinary = export_ir_function(&pkg, top, &module).unwrap();
        let target = fs::read_to_string(dir.path().join(format!("Generated/{side}.lean"))).unwrap();
        assert_eq!(target, ordinary.lean_source(), "Generated.{side}");
    }
}

#[test]
fn pointwise_target_preserves_argument_and_result_shapes() {
    let pkg = Parser::new(IDENTITY).parse_and_validate_package().unwrap();
    let target =
        prepare_ir_function_equivalence_target(&pkg, Some("passthrough"), &pkg, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    target.write_to(dir.path()).unwrap();
    assert_eq!(
        fs::read_to_string(dir.path().join("Target.lean")).unwrap(),
        include_str!("goldens/ir_target.lean")
    );
    assert!(!dir.path().join("Check.lean").exists());

    // The argument has the same packed width, but different XLS structure.
    let text = IDENTITY.replace("(bits[1], bits[7])", "bits[8]");
    let other = Parser::new(&text).parse_and_validate_package().unwrap();
    assert!(prepare_ir_function_equivalence_target(&pkg, None, &other, None).is_err());
}

#[cfg(feature = "lean-tests")]
#[test]
fn checks_ir_proofs_and_rejects_unproved_assumptions() {
    let lean = PathBuf::from(
        std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable"),
    );
    let pkg = Parser::new(IDENTITY).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target");
    prepare_ir_function_equivalence_target(&pkg, None, &pkg, None)
        .unwrap()
        .write_to(&target)
        .unwrap();
    let sources = LeanIrProofSources {
        root: dir.path().into(),
        module: dir.path().join("Identity.lean"),
        theorem: LeanTheoremName::new("identity_proof").unwrap(),
        modules: vec![],
    };
    fs::write(
        &sources.module,
        "import Target\ntheorem identity_proof : Target.Equivalent := by intro first second; rfl\n",
    )
    .unwrap();
    run_ir_function_equivalence_proof(&lean, &target, &sources, &dir.path().join("good")).unwrap();
    assert!(!dir.path().join("good/Check.lean").exists());

    fs::write(
        &sources.module,
        r#"import Target
axiom assumed : Target.Equivalent
theorem identity_proof : Target.Equivalent := assumed
"#,
    )
    .unwrap();
    let failure =
        run_ir_function_equivalence_proof(&lean, &target, &sources, &dir.path().join("bad"))
            .unwrap_err();
    assert!(failure.to_string().contains("unverified axiom"));
    fs::write(
        &sources.module,
        r#"import Target
axiom falseCheck : false = true
theorem identity_proof : Target.Equivalent := by cases falseCheck
"#,
    )
    .unwrap();
    let false_check = run_ir_function_equivalence_proof(
        &lean,
        &target,
        &sources,
        &dir.path().join("false-check"),
    )
    .unwrap_err();
    assert!(
        false_check
            .to_string()
            .contains("native Boolean check failed"),
        "{false_check}"
    );
}

#[cfg(feature = "lean-tests")]
#[test]
fn audits_only_the_axioms_reachable_from_the_selected_theorem() {
    let lean = PathBuf::from(std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN"));
    let ir = "package p\ntop fn c() -> bits[8] { ret z: bits[8] = literal(value=3, id=1) }";
    let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target");
    prepare_ir_function_equivalence_target(&pkg, None, &pkg, None)
        .unwrap()
        .write_to(&target)
        .unwrap();
    let helper = dir.path().join("Helper.lean");
    fs::write(
        &helper,
        r#"import Target
theorem Helper.left : Generated.Lhs.eval = (3 : BitVec 8) := by native_decide
theorem Helper.right : Generated.Rhs.eval = (3 : BitVec 8) := by native_decide
theorem Helper.unused : (1 : Nat) + 1 = 2 := by native_decide
theorem Helper.unfinished : Target.Equivalent := by sorry
"#,
    )
    .unwrap();
    let mut sources = LeanIrProofSources {
        root: dir.path().into(),
        module: dir.path().join("Proof.lean"),
        theorem: LeanTheoremName::new("Proof.viaHelper").unwrap(),
        modules: vec![helper],
    };
    fs::write(
        &sources.module,
        r#"import Helper
theorem Proof.viaHelper : Target.Equivalent := Helper.left.trans Helper.right.symm
theorem Proof.direct : Target.Equivalent := by rfl
theorem Proof.unfinished : Target.Equivalent := Helper.unfinished
"#,
    )
    .unwrap();

    let replayed =
        run_ir_function_equivalence_proof(&lean, &target, &sources, &dir.path().join("default"))
            .unwrap();
    assert_eq!(replayed.native_checks_replayed, Some(2));

    let run = |sources: &LeanIrProofSources, policy, work_dir| {
        run_ir_function_equivalence_proof_with_policy(
            &lean,
            &target,
            sources,
            &dir.path().join(work_dir),
            policy,
        )
    };
    let strict = LeanIrAxiomPolicy::RejectNativeChecks;
    let failure = run(&sources, strict, "strict-native").unwrap_err();
    assert!(
        failure.stderr.contains("Helper.left._native")
            || failure.stderr.contains("Helper.right._native"),
        "{failure}"
    );

    sources.theorem = LeanTheoremName::new("Proof.direct").unwrap();
    let checked = run(&sources, strict, "strict-direct").unwrap();
    assert_eq!(checked.native_checks_replayed, Some(0));

    sources.theorem = LeanTheoremName::new("Proof.unfinished").unwrap();
    for (policy, work_dir) in [
        (LeanIrAxiomPolicy::ReplayNativeChecks, "default-sorry"),
        (strict, "strict-sorry"),
    ] {
        let failure = run(&sources, policy, work_dir).unwrap_err();
        assert!(failure.stderr.contains("sorryAx"), "{failure}");
    }
}

#[cfg(feature = "lean-tests")]
#[test]
fn checks_nullary_and_inequivalent_functions() {
    let lean = PathBuf::from(
        std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable"),
    );
    let text = "package p\ntop fn c() -> bits[1] { ret v: bits[1] = literal(value=0, id=1) }";
    let lhs = Parser::new(text).parse_and_validate_package().unwrap();
    let other = text.replace("value=0", "value=1");
    let rhs = Parser::new(&other).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let sources = LeanIrProofSources {
        root: dir.path().into(),
        // The IR runner no longer owns Check.lean, so a user module may use it.
        module: dir.path().join("Check.lean"),
        theorem: LeanTheoremName::new("Constant.equivalent").unwrap(),
        modules: vec![],
    };
    fs::write(
        &sources.module,
        "import Target\ntheorem Constant.equivalent : Target.Equivalent := by rfl\n",
    )
    .unwrap();
    for (name, right, valid) in [("same", &lhs, true), ("different", &rhs, false)] {
        let root = dir.path().join(name);
        let target = root.join("target");
        prepare_ir_function_equivalence_target(&lhs, None, right, None)
            .unwrap()
            .write_to(&target)
            .unwrap();
        let result =
            run_ir_function_equivalence_proof(&lean, &target, &sources, &root.join("work"));
        assert_eq!(result.is_ok(), valid, "{result:?}");
    }
}

#[cfg(feature = "lean-tests")]
#[test]
fn rejects_a_helper_that_reinterprets_the_generated_goal() {
    let lean = PathBuf::from(std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN"));
    let zero = "package p\ntop fn c() -> bits[1] { ret v: bits[1] = literal(value=0, id=1) }";
    let one = zero.replace("value=0", "value=1");
    let lhs = Parser::new(zero).parse_and_validate_package().unwrap();
    let rhs = Parser::new(&one).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target");
    prepare_ir_function_equivalence_target(&lhs, None, &rhs, None)
        .unwrap()
        .write_to(&target)
        .unwrap();

    // The helper exports syntax into the wrapper and its importers. The
    // declaration it produces proves True even though the two functions differ.
    let helper = dir.path().join("Helper.lean");
    fs::write(&helper, include_str!("fixtures/reinterpret_target.lean")).unwrap();
    let sources = LeanIrProofSources {
        root: dir.path().into(),
        module: dir.path().join("Proof.lean"),
        theorem: LeanTheoremName::new("Proof.equivalent").unwrap(),
        modules: vec![helper],
    };
    fs::write(
        &sources.module,
        "import Helper\ntheorem Proof.equivalent : Target.Equivalent := True.intro\n",
    )
    .unwrap();
    let failure =
        run_ir_function_equivalence_proof(&lean, &target, &sources, &dir.path().join("work"))
            .unwrap_err();
    assert!(
        matches!(
            failure.error.as_ref(),
            LeanError::ProcessFailed {
                operation: "equivalence check",
                stderr,
                ..
            } if stderr.contains("wrong type") && stderr.contains("Target.Equivalent")
        ),
        "unexpected error: {failure}"
    );
}

#[cfg(feature = "lean-tests")]
#[test]
fn proves_distinct_graphs_and_agrees_with_the_executable_export() {
    let lean = PathBuf::from(
        std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable"),
    );
    let lhs = Parser::new(
        r#"package p
fn sum(a: bits[4] id=1, b: bits[4] id=2) -> bits[4] {
  ret s: bits[4] = add(a, b, id=3)
}
top fn lhs(x: bits[4] id=4, y: bits[4] id=5) -> bits[4] {
  ret r: bits[4] = invoke(x, y, to_apply=sum, id=6)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let rhs = Parser::new(
        r#"package p
top fn rhs(first: bits[4] id=1, second: bits[4] id=2) -> bits[4] {
  ret r: bits[4] = add(second, first, id=3)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let target = root.join("target");
    prepare_ir_function_equivalence_target(&lhs, None, &rhs, None)
        .unwrap()
        .write_to(&target)
        .unwrap();
    let executable =
        export_ir_function(&lhs, None, &LeanModuleName::new("Plain").unwrap()).unwrap();
    let plain = executable.write_to(root).unwrap();
    let proof = root.join("Adder.lean");
    fs::write(
        &proof,
        r#"import Target
import Plain
import XlsynthLean.IRProof

theorem add_commutes : Target.Equivalent := by
  intro a b
  calc
    Generated.Lhs.eval a b = Plain.eval a b := rfl
    _ = b + a := by change a + b = b + a; ir_bv_normalize <;> bv_decide
    _ = Generated.Rhs.eval a b := rfl
"#,
    )
    .unwrap();
    let mut sources = LeanIrProofSources {
        root: root.into(),
        module: proof,
        theorem: LeanTheoremName::new("add_commutes").unwrap(),
        modules: vec![plain],
    };
    run_ir_function_equivalence_proof(&lean, &target, &sources, &root.join("work")).unwrap();

    // Importing a theorem does not make it a declaration of the selected
    // module.
    sources.modules.push(sources.module);
    sources.module = root.join("OnlyImports.lean");
    fs::write(&sources.module, "import Adder\n").unwrap();
    let failure =
        run_ir_function_equivalence_proof(&lean, &target, &sources, &root.join("wrong-origin"))
            .unwrap_err();
    assert!(
        failure
            .to_string()
            .contains("add_commutes must be declared by module OnlyImports"),
        "unexpected error: {failure}",
    );
}

/// The cut generalizes both tuple results and preserves the condition on each
/// local contract. Changing one control input invalidates the full proof.
#[cfg(feature = "lean-tests")]
#[test]
fn factors_repeated_helper_calls_with_pointwise_contracts() {
    let lean = PathBuf::from(std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN"));
    let ir = include_str!("fixtures/decomposition/route.ir");
    let reference = Parser::new(ir).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let module = dir.path().join("RouteProof.lean");
    fs::write(
        &module,
        include_str!("fixtures/decomposition/RouteProof.lean"),
    )
    .unwrap();
    let sources = LeanIrProofSources {
        root: dir.path().into(),
        module,
        theorem: LeanTheoremName::new("RouteProof.equivalent").unwrap(),
        modules: vec![],
    };
    let check = |case: &str, text: &str| {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let root = dir.path().join(case);
        let target = root.join("target");
        prepare_ir_function_equivalence_target(&package, None, &reference, Some("reference"))
            .unwrap()
            .write_to(&target)
            .unwrap();
        run_ir_function_equivalence_proof(&lean, &target, &sources, &root.join("work"))
    };
    check("same-control", ir).unwrap();
    let changed = ir
        .replace(
            "ret second:",
            "other: bits[1] = not(p, id=18)\n  ret second:",
        )
        .replace("invoke(p, wired,", "invoke(other, wired,");
    let failure = check("inverted-control", &changed).unwrap_err();
    assert!(
        matches!(
            failure.error.as_ref(),
            LeanError::ProcessFailed {
                operation: "user proof compilation",
                ..
            }
        ) && failure.stdout.contains("~~~p = 0")
            && failure.stdout.contains("routed1_spec hp"),
        "{failure}"
    );
}

#[cfg(feature = "lean-tests")]
#[test]
fn preserves_distinct_calls_with_identical_operands_and_shared_uses() {
    let lean = PathBuf::from(std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN"));
    let pkg = Parser::new(include_str!("fixtures/decomposition/duplicates.ir"))
        .parse_and_validate_package()
        .unwrap();
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target");
    prepare_ir_function_equivalence_target(&pkg, None, &pkg, None)
        .unwrap()
        .write_to(&target)
        .unwrap();
    let sources = LeanIrProofSources {
        root: root.path().into(),
        module: root.path().join("DuplicateProof.lean"),
        theorem: LeanTheoremName::new("DuplicateProof.equivalent").unwrap(),
        modules: vec![],
    };
    fs::write(
        &sources.module,
        include_str!("fixtures/decomposition/DuplicateProof.lean"),
    )
    .unwrap();
    run_ir_function_equivalence_proof(&lean, &target, &sources, &root.path().join("work")).unwrap();
}
