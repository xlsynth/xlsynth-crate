// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, parse_module};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir::{Block, MemberType, NodePayload, Unop};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

const PIPELINE: &str = include_str!("fixtures/mul_add.bx");
const MULTIPLIER: &str = r#"pub fn mul_wide(a: u8, b: u8) -> u16 {
    (a as u16) * (b as u16)
}"#;

#[test]
fn dslx_function_links_into_a_two_stage_pipeline() {
    let ir = xlsynth::convert_dslx_to_ir_text(
        MULTIPLIER,
        Path::new("arithmetic.x"),
        &xlsynth::DslxConvertOptions::default(),
    )
    .unwrap();
    let functions = Parser::new(&ir.ir).parse_and_validate_package().unwrap();
    let package = compile(PIPELINE, functions, ResetBehavior::default()).unwrap();
    assert_eq!(package.top, Some(("mul_add".into(), MemberType::Block)));
    assert_eq!(
        package.to_string(),
        include_str!("fixtures/mul_add.golden.ir")
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();

    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let pipeline = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&pipeline).unwrap();
    let ports = runner.port_bindings();
    let a = ports.input("a").unwrap();
    let b = ports.input("b").unwrap();
    let bias = ports.input("bias").unwrap();
    let valid = ports.input("valid_in").unwrap();
    let rst = ports.input("rst").unwrap();
    for (operands, valid_in, reset, expected_valid, expected_data) in [
        ([0, 0, 0], 0, 1, 0, None),
        ([255, 2, 3], 1, 0, 0, None),
        ([7, 9, 5], 1, 0, 1, Some(513)),
        ([99, 99, 99], 0, 0, 1, Some(68)),
        ([1, 2, 3], 1, 0, 0, Some(68)),
        ([0, 0, 0], 0, 1, 0, Some(5)),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_u64(&a, operands[0])?;
                drive.set_u64(&b, operands[1])?;
                drive.set_u64(&bias, operands[2])?;
                drive.set_u64(&valid, valid_in)?;
                drive.set_u64(&rst, reset)
            })
            .unwrap();
        assert_eq!(
            edge.outputs["valid"].to_u64_if_known(),
            Some(expected_valid)
        );
        if let Some(data) = expected_data {
            assert_eq!(edge.outputs["value"].to_u64_if_known(), Some(data));
        }
    }
}

#[test]
fn hierarchy_accepts_registered_feedback_and_checks_combinational_feedback() {
    let source = r#"
pub block accumulator(clk, rst, step: u8) -> u8 {
    inst cell: delay;
    cell.d := cell.out + step;
    cell.out
}
block delay(clk, rst, d: u8) -> u8 {
    reg q: u8 = u8:0;
    q := d;
    q
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    assert_eq!(package.top, Some(("accumulator".into(), MemberType::Block)));
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    let combinational = source.replace("q := d;\n    q", "q := d;\n    d");
    assert!(
        compile(
            &combinational,
            empty_package("design"),
            ResetBehavior::default()
        )
        .is_err()
    );
}

#[test]
fn hierarchy_tracks_feedthrough_per_result_port() {
    let source = r#"
struct Ports { saved: u8, live: u8 }
pub block parent(clk, delta: u8) -> u8 {
    inst lane: mixed;
    lane.x := lane.saved + delta;
    lane.live
}
block mixed(clk, x: u8) -> Ports {
    reg q: u8;
    q := x;
    let bundle = Ports { saved: q, live: x };
    bundle
}"#;
    compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    let cycle = source.replace("lane.saved + delta", "lane.live + delta");
    assert!(compile(&cycle, empty_package("design"), ResetBehavior::default()).is_err());
}

#[test]
fn aggregate_extern_signatures_preserve_nested_port_shapes() {
    let ir = r#"package agg
fn echo(x: (bits[4][2], bits[100]) id=1) -> (bits[4][2], bits[100]) {
  ret x: (bits[4][2], bits[100]) = param(name=x, id=1)
}"#;
    let source = r#"
struct Payload { samples: u4[2], tag: uN[100] }
extern fn echo(x: Payload) -> Payload;
block capture(clk, rst, x: Payload) -> (u4, Payload) {
    reg saved: Payload = Payload { samples: [0, 1], tag: uN[100]:0x10000000000000000 };
    saved := echo(x);
    (saved.samples[1], saved)
}"#;
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let output = compile(source, package, ResetBehavior::default()).unwrap();
    let block = output.get_block("capture").unwrap();
    assert_eq!(
        block
            .output_ports()
            .map(|p| (block.port_name(p), block.port_type(p).to_string()))
            .collect::<Vec<_>>(),
        [
            ("out_0", "bits[4]".into()),
            ("out_1", "(bits[4][2], bits[100])".into())
        ]
    );
    xlsynth::IrPackage::parse_ir(&output.to_string(), None).unwrap();
}

#[test]
fn reading_rst_means_asserted_even_when_the_physical_reset_is_active_low() {
    let source = "block reset_probe(clk, rst) -> bool { rst }";
    for active_low in [false, true] {
        let policy = ResetBehavior {
            active_low,
            asynchronous: true,
        };
        let output = compile(source, empty_package("design"), policy).unwrap();
        let block = output.get_block("reset_probe").unwrap();
        let reset = block.reset.as_ref().unwrap();
        assert_eq!((reset.active_low, reset.asynchronous), (active_low, true));
        let port = block.get_output_port("out").unwrap();
        let NodePayload::OutputPort { arg, .. } = block.get_node(port).payload else {
            panic!("expected output_port");
        };
        if active_low {
            assert!(
                matches!(block.get_node(arg).payload, NodePayload::Unop(Unop::Not, input) if input == reset.port)
            );
        } else {
            assert_eq!(arg, reset.port);
        }
    }
}
#[test]
fn invalid_register_writes_are_rejected_before_linking() {
    for body in [
        "reg q: u8; if en { q := x; } if !en { q := x; } q",
        "reg q: u8; q",
        "reg q: u8; q := en; q",
        "reg q: u8 = u8:0; q := x; q",
        "reg q: u8; q := x; if x { let t = en; } q",
    ] {
        let source = format!("block stage(clk, x: u8, en: bool) -> u8 {{ {body} }}");
        let parsed = parse_module(&source).unwrap();
        assert!(check_module(&parsed).is_err(), "{body}");
    }
}

#[test]
fn extern_declarations_are_checked_without_a_dslx_frontend_and_linked_by_signature() {
    let source = r#"
extern fn widen(x: s8) -> s16;
block wide_delay(clk, x: s8) -> s16 {
    reg q: s16;
    q := widen(x);
    q
}"#;
    check_module(&parse_module(source).unwrap()).unwrap();
    let linked = r#"package math
fn __math__widen(x: bits[8] id=1) -> bits[16] {
  ret sign_ext.2: bits[16] = sign_ext(x, new_bit_count=16, id=2)
}
"#;
    let package = Parser::new(linked).parse_and_validate_package().unwrap();
    compile(source, package.clone(), ResetBehavior::default()).unwrap();
    let wrong_signature = source.replace("s16", "s17");
    assert!(compile(&wrong_signature, package, ResetBehavior::default()).is_err());
    assert!(compile(source, empty_package("math"), ResetBehavior::default()).is_err());
}

const LINKED_MODULES: &str = r#"package bundle
fn __arithmetic__transform(x: bits[8] id=1) -> bits[8] {
  ret neg.2: bits[8] = neg(x, id=2)
}
fn __logic__transform(x: bits[8] id=3) -> bits[8] {
  ret not.4: bits[8] = not(x, id=4)
}
fn transform(x: bits[8] id=5) -> bits[8] {
  ret x: bits[8] = param(name=x, id=5)
}
fn __bundle__transform(x: bits[8] id=6) -> bits[8] {
  ret reverse.7: bits[8] = reverse(x, id=7)
}"#;

/// Return the linked function names at the block's invocation sites in node
/// order.
fn invoked_symbols(block: &Block) -> Vec<&str> {
    block
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn qualified_externs_distinguish_same_named_functions_in_a_linked_package() {
    let source = r#"
extern fn arithmetic::transform(x: u8) -> u8;
extern fn logic::transform(x: u8) -> u8;
extern fn transform(x: u8) -> u8;
extern fn raw_alias(x: u8) -> u8 = "__bundle__transform";
block calls(clk, x: u8) -> (u8, u8, u8, u8) {
    (arithmetic::transform(x), logic::transform(x), transform(x), raw_alias(x))
}"#;
    let linked = Parser::new(LINKED_MODULES)
        .parse_and_validate_package()
        .unwrap();
    let package = compile(source, linked, ResetBehavior::default()).unwrap();
    assert_eq!(
        invoked_symbols(package.get_block("calls").unwrap()),
        [
            "__arithmetic__transform",
            "__logic__transform",
            "transform",
            "__bundle__transform",
        ]
    );
}

#[test]
fn a_qualified_extern_can_override_its_mangled_symbol() {
    let source = r#"
extern fn arithmetic::transform(x: u8) -> u8 = "__logic__transform";
block call(clk, x: u8) -> u8 { arithmetic::transform(x) }"#;
    let linked = Parser::new(LINKED_MODULES)
        .parse_and_validate_package()
        .unwrap();
    let package = compile(source, linked.clone(), ResetBehavior::default()).unwrap();
    assert_eq!(
        invoked_symbols(package.get_block("call").unwrap()),
        ["__logic__transform"]
    );

    let missing_override = source.replace("__logic__transform", "__missing__transform");
    assert!(compile(&missing_override, linked, ResetBehavior::default()).is_err());
}

#[test]
fn missing_qualified_extern_does_not_bind_a_bare_or_destination_package_symbol() {
    let source = r#"
extern fn absent::transform(x: u8) -> u8;
block call(clk, x: u8) -> u8 { absent::transform(x) }"#;
    let parsed = parse_module(source).unwrap();
    check_module(&parsed).unwrap();
    let linked = Parser::new(LINKED_MODULES)
        .parse_and_validate_package()
        .unwrap();
    assert!(compile(source, linked, ResetBehavior::default()).is_err());
}

#[test]
fn calls_resolve_only_by_the_declared_source_name() {
    for (declaration, call) in [
        ("arithmetic::transform", "transform"),
        ("transform", "arithmetic::transform"),
        ("arithmetic::transform", "logic::transform"),
    ] {
        let source = format!(
            r#"
extern fn {declaration}(x: u8) -> u8;
block call(clk, x: u8) -> u8 {{ {call}(x) }}"#
        );
        let parsed = parse_module(&source).unwrap();
        assert!(
            check_module(&parsed).is_err(),
            "declaration {declaration}, call {call}"
        );
    }
}

#[test]
fn a_guard_becomes_a_shared_enable_and_a_self_hold_expression_stays_a_mux() {
    let source = r#"
block enables(clk, a: bool, b: bool, x: u8) -> (u8, u8, u8) {
    reg p: u8;
    reg q: u8;
    reg r: u8;
    if a && b {
        p := x;
        q := x;
    }
    r := if a { x } else { r };
    (p, q, r)
}"#;
    let pkg = compile(source, empty_package("gates"), ResetBehavior::default()).unwrap();
    let block = pkg.get_block("enables").unwrap();
    let writes: Vec<_> = block
        .nodes
        .iter()
        .filter_map(|node| {
            if let NodePayload::RegisterWrite {
                load_enable, arg, ..
            } = node.payload
            {
                Some((load_enable, arg))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(writes.len(), 3);
    assert!(writes[0].0.is_some());
    assert_eq!(writes[0].0, writes[1].0);
    assert!(writes[2].0.is_none());
    assert!(matches!(
        block.get_node(writes[2].1).payload,
        NodePayload::Sel { .. }
    ));
}

#[test]
fn linking_rejects_transitive_effects_outside_the_return_cone() {
    let ir = r#"package effects
fn observed(x: bits[1] id=1) -> bits[1] {
  event: () = cover(x, label="seen", id=2)
  ret x: bits[1] = param(name=x, id=1)
}
fn wrapper(x: bits[1] id=3) -> bits[1] {
  ret invoke.4: bits[1] = invoke(x, to_apply=observed, id=4)
}"#;
    let functions = Parser::new(ir).parse_and_validate_package().unwrap();
    let source =
        "extern fn wrapper(x: bool) -> bool; block sample(clk, x: bool) -> bool { wrapper(x) }";
    let err = compile(source, functions, ResetBehavior::default()).unwrap_err();
    assert_eq!(
        err,
        "cannot lift effectful IR function 'observed' into BX because it contains 'cover'"
    );
}
