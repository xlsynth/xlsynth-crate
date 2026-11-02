// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::PathBuf;

use xlsynth_lean::{
    LeanError, LeanModuleName, LeanProofSources, export_ir_function,
    prepare_ir_function_equivalence_target, run_ir_function_equivalence_proof,
};
use xlsynth_pir::ir_parser::Parser;

const IDENTITY: &str = r#"package p
top fn passthrough(unused: (bits[1], bits[7]) id=1, values: bits[3][2] id=2) -> bits[3][2] {
  ret output: bits[3][2] = identity(values, id=3)
}
"#;

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

    // The argument has the same packed width, but different XLS structure.
    let text = IDENTITY.replace("(bits[1], bits[7])", "bits[8]");
    let other = Parser::new(&text).parse_and_validate_package().unwrap();
    assert!(prepare_ir_function_equivalence_target(&pkg, None, &other, None).is_err());
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
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
    let sources = LeanProofSources {
        root: dir.path().into(),
        body: dir.path().join("Body.lean"),
        modules: vec![dir.path().join("Helper.lean")],
    };
    fs::write(
        &sources.modules[0],
        "import Target\ntheorem identity_proof : Target.Equivalent := by intro first second; rfl\n",
    )
    .unwrap();
    fs::write(&sources.body, "identity_proof").unwrap();
    run_ir_function_equivalence_proof(&lean, &target, &sources, &dir.path().join("good")).unwrap();

    fs::write(
        &sources.modules[0],
        "import Target\naxiom identity_proof : Target.Equivalent\n",
    )
    .unwrap();
    // A modified file in a prepared bundle cannot turn off the built-in audit.
    fs::write(target.join("Check.lean"), "import UserProof\n").unwrap();
    let failure =
        run_ir_function_equivalence_proof(&lean, &target, &sources, &dir.path().join("bad"))
            .unwrap_err();
    assert!(failure.to_string().contains("unverified axiom"));
    fs::write(
        &sources.modules[0],
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

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn checks_nullary_and_inequivalent_functions() {
    let lean = PathBuf::from(
        std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable"),
    );
    let text = "package p\ntop fn c() -> bits[1] { ret v: bits[1] = literal(value=0, id=1) }";
    let lhs = Parser::new(text).parse_and_validate_package().unwrap();
    let other = text.replace("value=0", "value=1");
    let rhs = Parser::new(&other).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let sources = LeanProofSources {
        root: dir.path().into(),
        body: dir.path().join("Body.lean"),
        modules: vec![],
    };
    fs::write(&sources.body, "by rfl").unwrap();
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

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
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
    let helper = root.join("Helper.lean");
    fs::write(
        &helper,
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
    let sources = LeanProofSources {
        root: root.into(),
        body: root.join("Body.lean"),
        modules: vec![plain, helper],
    };
    fs::write(&sources.body, "add_commutes").unwrap();
    run_ir_function_equivalence_proof(&lean, &target, &sources, &root.join("work")).unwrap();
}

/// The cut generalizes both tuple results and preserves the condition on each
/// local contract. Changing one control input invalidates the full proof.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn factors_repeated_helper_calls_with_pointwise_contracts() {
    let lean = PathBuf::from(std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN"));
    let ir = include_str!("fixtures/decomposition/route.ir");
    let reference = Parser::new(ir).parse_and_validate_package().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let body = dir.path().join("Body.lean");
    fs::write(&body, "RouteProof.equivalent").unwrap();
    let helper = dir.path().join("RouteProof.lean");
    fs::write(
        &helper,
        include_str!("fixtures/decomposition/RouteProof.lean"),
    )
    .unwrap();
    let sources = LeanProofSources {
        root: dir.path().into(),
        body,
        modules: vec![helper],
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
    assert!(matches!(
        *failure.error,
        LeanError::ProcessFailed {
            operation: "user proof compilation",
            ..
        }
    ));
}
