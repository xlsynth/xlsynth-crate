// SPDX-License-Identifier: Apache-2.0

use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, lower_module, parse_module};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir::{NodePayload, PackageMember};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn imported_specializations_coexist_with_an_extern_and_drive_registered_rtl() {
    // This caller deliberately owns the same symbol DSLX generates for
    // umul(u8,u5).
    let supplied = r#"package arithmetic
file_number 0 "arithmetic.x"
fn __std__umul__5_8_13(a: bits[8] id=1, b: bits[5] id=2) -> bits[13] {
  ret literal.3: bits[13] = literal(value=17, id=3, pos=[(0,0,0)])
}"#;
    let bx = r#"
use std;
use std::{umul, smul};
extern fn supplied(a: u8, b: u5) -> u13 = "__std__umul__5_8_13";

struct Results { saved: u13, wide: u20, negative: bool, pow2: bool, caller: u13 }
block products(clk, rst, valid: bool, a: u8, b: u5, c: u12) -> Results {
    let product = umul(a, b);
    let signed_product = smul(a as s8, b as s5);
    reg q: u13 = u13:0;
    if valid { q := product; }
    Results {
        saved: q,
        wide: std::umul(a + u8:1, c),
        negative: signed_product < s13:0,
        pow2: std::is_pow2(std::umul(a, b)),
        caller: supplied(a, b),
    }
}"#;
    let input = Parser::new(supplied).parse_and_validate_package().unwrap();
    let package = compile(bx, input, ResetBehavior::default()).unwrap();
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    // Four std specializations, the preexisting extern and the BX block.
    assert_eq!(package.members.len(), 6);
    let block = package.get_block("products").unwrap();
    let targets: Vec<_> = block
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(targets.len(), 6);
    assert_eq!(targets[0], targets[3]); // Both umul(u8,u5) calls share a body.
    assert_ne!(targets[0], targets[5]);
    assert_eq!(targets[5], "__std__umul__5_8_13");
    assert_eq!(package.file_table.id_to_path[&0], "arithmetic.x");
    assert!(
        package
            .file_table
            .id_to_path
            .values()
            .any(|path| path == "xls/dslx/stdlib/std.x")
    );

    let sv = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let pipeline = compile_pipeline_module(&sv.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&pipeline).unwrap();
    let ports = runner.port_bindings();
    let a = ports.input("a").unwrap();
    let b = ports.input("b").unwrap();
    let c = ports.input("c").unwrap();
    let rst = ports.input("rst").unwrap();
    let valid = ports.input("valid").unwrap();
    for (x, y, z, reset, enable, saved, negative, pow2) in [
        (0, 0, 0, 1, 0, 0, 0, 0),
        (255, 31, 4095, 0, 1, 7905, 0, 0),
        (128, 2, 4095, 0, 0, 7905, 1, 1),
        (7, 3, 1234, 0, 1, 21, 0, 0),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_u64(&a, x)?;
                drive.set_u64(&b, y)?;
                drive.set_u64(&c, z)?;
                drive.set_u64(&rst, reset)?;
                drive.set_u64(&valid, enable)
            })
            .unwrap();
        for (name, expected) in [
            ("saved", saved),
            ("wide", ((x + 1) & 255) * z),
            ("negative", negative),
            ("pow2", pow2),
            ("caller", 17),
        ] {
            assert_eq!(
                edge.outputs[name].to_u64_if_known(),
                Some(expected),
                "{name}"
            );
        }
    }
}

#[test]
fn aggregate_and_wide_results_survive_separate_checking_and_lowering() {
    let source = r#"
use std;
block indices(clk, xs: (sN[80],)[2]) -> (u32, (sN[80],))[2] {
    std::enumerate(xs)
}
block rounding(clk, x: u32) -> u32 {
    std::round_up_to_nearest(x, u32:4)
}"#;
    let checked = check_module(&parse_module(source).unwrap()).unwrap();
    let enumerate = checked
        .imported_calls()
        .iter()
        .find(|call| call.function == "enumerate")
        .unwrap();
    assert_eq!(enumerate.result, checked.blocks()[0].result);
    let package =
        lower_module(&checked, empty_package("arrays"), ResetBehavior::default()).unwrap();
    assert_eq!(package.name, "arrays");
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    // enumerate retains its loop body and round_up_to_nearest its ceil_div
    // helper.
    assert!(
        package
            .members
            .iter()
            .any(|member| matches!(member, PackageMember::Function(f)
        if f.nodes.iter().any(|node| matches!(node.payload, NodePayload::CountedFor { .. }))))
    );
    assert!(package.get_fn("__std__ceil_div__32").is_some());
}

#[test]
fn invalid_imported_calls_report_the_function_and_the_reason() {
    for (expression, callee, reason) in [
        ("std::lsbs(x)", "std::lsbs", "Could not infer parametric"),
        ("std::clzt_pow2_1(u1:0)", "std::clzt_pow2_1", "public"),
        (
            "std::umul(x, 3)",
            "std::umul",
            "literal '3' requires a concrete bit type",
        ),
    ] {
        let bx = format!("use std; block b(clk, x: u8) -> u8 {{ let unused = {expression}; x }}");
        let error = check_module(&parse_module(&bx).unwrap()).unwrap_err();
        assert!(error.contains(callee) && error.contains(reason), "{error}");
    }
}
