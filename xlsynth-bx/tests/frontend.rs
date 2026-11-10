// SPDX-License-Identifier: Apache-2.0

#[path = "frontend/array_update.rs"]
mod array_update;
#[path = "frontend/bits_array_casts.rs"]
mod bits_array_casts;
#[path = "frontend/counted_for.rs"]
mod counted_for;
#[path = "support/directed_expr_eval.rs"]
mod directed_expr_eval;
#[path = "support/expr_eval.rs"]
mod expr_eval;
#[path = "frontend/function_parametrics.rs"]
mod function_parametrics;
#[path = "frontend/function_type_parametrics.rs"]
mod function_type_parametrics;
#[path = "frontend/function_value_parametrics.rs"]
mod function_value_parametrics;
#[path = "frontend/imported_structs.rs"]
mod imported_structs;
#[path = "frontend/local_functions.rs"]
mod local_functions;
#[path = "frontend/map_lambda.rs"]
mod map_lambda;
#[path = "frontend/module_constants.rs"]
mod module_constants;
#[path = "frontend/named_map.rs"]
mod named_map;
#[path = "frontend/numeric_enums.rs"]
mod numeric_enums;
#[path = "frontend/scalar_match.rs"]
mod scalar_match;
#[path = "frontend/unroll_for.rs"]
mod unroll_for;

use std::path::Path;

use directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx, compile_dslx_module,
};
use expr_eval::ExpressionEvaluator;

use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
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
fn match_tracks_feedthrough_per_reachable_result_field() {
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
    match x as u2 {
        2..1 => Ports { saved: x, live: q },
        0 => Ports { saved: q, live: x },
        1..=2 => Ports { saved: q, live: !x },
        3 => Ports { saved: q, live: x + u8:1 },
        _ => Ports { saved: x, live: x },
    }
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
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
    let source = "#[allow(combinational_block)] block reset_probe(clk, rst) -> bool { rst }";
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
#[allow(combinational_block)]
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
#[allow(combinational_block)]
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
    assert_eq!(
        compile(source, linked, ResetBehavior::default()).unwrap_err(),
        "extern fn 'absent::transform': no linked IR function for '__absent__transform'"
    );
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

#[test]
fn signed_operand_unifies_prior_literal_widths_in_source_order() {
    assert_expr_rejected_by_both("y: s4", "bool", "(2 ^ 0x8) == y");
    assert_expr_matches_dslx(
        "y: s4",
        "bool",
        "y == (2 ^ 0x8)",
        [(
            vec![IrValue::make_ubits(4, 10).unwrap()],
            IrValue::bool(true),
        )],
    );
    for body in [
        "let a = 2 ^ (0x8 + y); a",
        "let a = (2 ^ 0x8) + y; a",
        "let a = 2 ^ ((0x8 + -1) + y); a",
    ] {
        assert_expr_rejected_by_both("y: s4", "s4", body);
    }
    for (body, expected) in [
        ("2 ^ (0x8 + y)", 11),
        ("let a: s4 = 2 ^ (0x8 + y); a", 11),
        ("let a = 2 ^ (y + 0x8); a", 11),
        ("let a = -1 + (2 ^ (0x8 + y)); a", 10),
        ("let a = 0x8 + y; a", 9),
    ] {
        assert_expr_matches_dslx(
            "y: s4",
            "s4",
            body,
            [(
                vec![IrValue::make_ubits(4, 1).unwrap()],
                IrValue::make_ubits(4, expected).unwrap(),
            )],
        );
    }
}

#[test]
fn conditional_checks_signed_literal_width_across_both_results() {
    assert_expr_matches_dslx(
        "flag: bool",
        "s8",
        "let y = if flag { -1 } else { 0xff + 0xff + s8:0 }; y",
        [(true, 255), (false, 254)].map(|(flag, value)| {
            (
                vec![IrValue::bool(flag)],
                IrValue::make_ubits(8, value).unwrap(),
            )
        }),
    );
}

#[test]
fn packed_slices_extract_lsb_first_and_zero_fill_past_the_end() {
    let body = r#"
    let word = words[1];
    (word[offset+:u4], word[u8:16+:uN[80]], word[64+:s4] as s8)
"#;
    let words =
        IrValue::parse_typed("[bits[96]:0, bits[96]:0xc000_00cd_0000_0000_0000_009b]").unwrap();
    let cases = [
        ("0", 0xb),
        ("1", 0xd),
        ("94", 3),
        ("96", 0),
        ("0x1_0000_0000_0000_0000", 0),
    ]
    .map(|(offset, expected_nibble)| {
        let offset = IrValue::parse_typed(&format!("bits[96]:{offset}")).unwrap();
        let expected = format!(
            "(bits[4]:{expected_nibble}, bits[80]:0xc000_00cd_0000_0000_0000, bits[8]:0xfd)"
        );
        (
            vec![words.clone(), offset],
            IrValue::parse_typed(&expected).unwrap(),
        )
    });
    assert_expr_matches_dslx(
        "words: uN[96][2], offset: uN[96]",
        "(u4, uN[80], s8)",
        body,
        cases,
    );
}

#[test]
fn packed_slice_operand_inference_matches_dslx() {
    let inputs = "x: u8, i: u3, flag: bool";
    let args = vec![
        IrValue::make_ubits(8, 0x3a).unwrap(),
        IrValue::make_ubits(3, 2).unwrap(),
        IrValue::bool(true),
    ];
    for (expr, result, expected) in [
        ("0b100111[2+:u3]", "u3", "bits[3]:1"),
        ("0[0+:bits[0]]", "bits[0]", "bits[0]:0"),
        ("x[i+:bits[0]]", "bits[0]", "bits[0]:0"),
        ("x[flag+:bool]", "bool", "bits[1]:1"),
        ("x[u8:true+:u1]", "u1", "bits[1]:1"),
        ("x[u3:(1)+:u1]", "u1", "bits[1]:1"),
        ("x[(u3:1, u3:2).0x0+:u1]", "u1", "bits[1]:1"),
        ("(x[0+:u3],)", "(u3,)", "(bits[3]:2)"),
        (r"x['\x01'+:u1]", "u1", "bits[1]:1"),
        ("x[-(1)+:u1]", "u1", "bits[1]:0"),
        ("x[i + -(1)+:u1]", "u1", "bits[1]:1"),
        ("flag[i + 6+:u1]", "u1", "bits[1]:1"),
        ("x[(i >> 3)+:u2]", "u2", "bits[2]:2"),
        ("x[(i >> (2 + 2))+:u2]", "u2", "bits[2]:2"),
        ("x[i ++ 1+:u1]", "u1", "bits[1]:1"),
        ("x[(1 < 2)+:u1]", "u1", "bits[1]:1"),
        ("x[([1] == [3])+:u1]", "u1", "bits[1]:0"),
        ("x[((1, x) == (3, x))+:u1]", "u1", "bits[1]:0"),
        ("x[(7 as u2)+:u1]", "u1", "bits[1]:1"),
        ("x[if flag { 4 } else { i }+:u3]", "u3", "bits[3]:3"),
    ] {
        assert_expr_matches_dslx(
            inputs,
            result,
            expr,
            [(args.clone(), IrValue::parse_typed(expected).unwrap())],
        );
    }
}

#[test]
fn packed_slices_infer_literal_types_in_aggregate_projections() {
    let body = r#"
        let offsets = [1, 2];
        let pair = (1, 2);
        let mixed = [(1, u2:1), (i, 2)];
        let nested = [[1, 2], [4, 5]];
        (x[offsets[0]+:u1], x[pair.0+:u1], [1, 2][1][0+:u2],
         (1, 2).1[0+:u2], [1, i][0][0+:u3], [i, 1][1][0+:u3],
         mixed[0].0[0+:u3], mixed[1].1[0+:u2], nested[0][0][0+:u3])
    "#;
    let expected = IrValue::parse_typed(
        "(bits[1]:1, bits[1]:1, bits[2]:2, bits[2]:2, bits[3]:1, bits[3]:1, bits[3]:1, bits[2]:2, bits[3]:1)",
    )
    .unwrap();
    assert_expr_matches_dslx(
        "x: u8, i: u3",
        "(u1, u1, u2, u2, u3, u3, u3, u2, u3)",
        body,
        [(
            vec![
                IrValue::make_ubits(8, 2).unwrap(),
                IrValue::make_ubits(3, 3).unwrap(),
            ],
            expected,
        )],
    );
}

#[test]
fn packed_slice_dimensions_retain_projected_literal_widths() {
    let body = r#"
        let widths = ([1, 2], (u32:2, 3));
        let pair = widths.1;
        let two = widths.0[1];
        (x[0+:uN[(1, 2).0]], x[0+:uN[[1, 2][1]]], x[0+:uN[pair.0]],
         x[0+:uN[pair.1]], x[0+:uN[two]], x[0+:xN[(0, 1).0][1]])
    "#;
    assert_expr_matches_dslx(
        "x: u8",
        "(u1, u2, u2, u3, u2, u1)",
        body,
        [(
            vec![IrValue::make_ubits(8, 7).unwrap()],
            IrValue::parse_typed(
                "(bits[1]:1, bits[2]:3, bits[2]:3, bits[3]:7, bits[2]:3, bits[1]:1)",
            )
            .unwrap(),
        )],
    );
    for expr in [
        "x[0+:uN[(u1:1, 2).0]]",
        "let a = [1, u3:2]; x[0+:uN[a[0]]]",
        "let a: (u1, u2) = (1, 2); x[0+:uN[a.0]]",
        "x[0+:uN[(1, 2).0 + 0]]",
        "true[[1, 2][0]+:u1]",
    ] {
        assert_expr_rejected_by_both("x: u8", "u1", expr);
    }
}

#[test]
fn packed_slice_infers_a_common_type_for_aggregate_branches() {
    let body = r#"
        let t = if flag { (1, 1) } else { (3, 3) };
        let a = [(if flag { [1] } else { [1] }), [i]];
        ((if flag { [1] } else { [3] })[0][0+:u2],
         t.0[0+:u2], (if flag { [1] } else { [i] })[0][0+:u2],
         (if flag { [1] } else { if flag { [3] } else { [5] } })[0][0+:u3],
         x[0+:uN[(if true { (1, 2) } else { (3, 2) }).1]], a[0][0][0+:u2])
    "#;
    let cases = [
        (
            true,
            "(bits[2]:1, bits[2]:1, bits[2]:1, bits[3]:1, bits[2]:2, bits[2]:1)",
        ),
        (
            false,
            "(bits[2]:3, bits[2]:3, bits[2]:2, bits[3]:5, bits[2]:2, bits[2]:1)",
        ),
    ]
    .map(|(flag, expected)| {
        (
            vec![
                IrValue::bool(flag),
                IrValue::make_ubits(3, 6).unwrap(),
                IrValue::make_ubits(8, 2).unwrap(),
            ],
            IrValue::parse_typed(expected).unwrap(),
        )
    });
    assert_expr_matches_dslx(
        "flag: bool, i: u3, x: u8",
        "(u2, u2, u2, u3, u2, u2)",
        body,
        cases,
    );
}

#[test]
fn packed_slice_width_spellings_match_dslx_at_the_shorthand_boundary() {
    let x = IrValue::make_ubits(84, 1).unwrap();
    let expected =
        IrValue::parse_typed("(bits[64]:1, bits[64]:1, bits[84]:1, bits[65]:1, bits[0]:0)")
            .unwrap();
    assert_expr_matches_dslx(
        "x: uN[84]",
        "(uN[64], sN[64], uN[84], sN[65], uN[0])",
        "(x[0+:u64], x[0+:s64], x[0+:uN[84]], x[0+:sN[65]], x[0+:uN[0]])",
        [(vec![x], expected)],
    );
}

#[test]
fn bit_width_shorthand_requires_a_canonical_width_from_one_to_sixty_four() {
    for (params, result, body) in [
        ("x: u65", "uN[65]", "x"),
        ("x: sN[84]", "s84", "x"),
        ("x: (u8, s0)[2]", "sN[0]", "x[0].1"),
        ("x: uN[84]", "uN[84]", "let y: u84 = x; y"),
        ("x: s1", "sN[65]", "x as s65"),
        ("", "uN[0]", "u0:0"),
        ("", "sN[65]", "s65:-1"),
        ("", "s64", "s64:s65:1"),
        ("x: u1", "u1", "let y: u01 = x; y"),
        ("x: s1", "s1", "let y: s01 = x; y"),
        ("x: u64", "s64", "x[0+:s064]"),
    ] {
        assert_expr_rejected_by_both(params, result, body);
    }

    assert_expr_matches_dslx(
        "x: s1, y: u64, z: uN[0]",
        "(u1, s64, sN[65], uN[84], sN[0])",
        r#"
        let sign: s64 = x as s64;
        let wide: sN[65] = sign as sN[65];
        (u1:1, sign, wide, y as uN[84], z as sN[0])
        "#,
        [(
            vec![
                IrValue::bool(true),
                IrValue::make_ubits(64, u64::MAX).unwrap(),
                IrValue::make_ubits(0, 0).unwrap(),
            ],
            IrValue::parse_typed(
                "(bits[1]:1, bits[64]:0xffffffffffffffff, bits[65]:0x1ffffffffffffffff, bits[84]:0xffffffffffffffff, bits[0]:0)",
            )
            .unwrap(),
        )],
    );
}

#[test]
fn leading_zero_bit_type_spellings_resolve_to_declared_structs()
-> Result<(), Box<dyn std::error::Error>> {
    let declarations = r#"struct u01 { value: u8 }
struct s01 { inner: u01 }"#;
    let body = r#"let y: u01 = u01 { value: x };
let z: s01 = s01 { inner: y };
z.inner.value"#;
    let bx = format!(
        "{declarations}\n{}",
        expr_eval::bx_source("x: u8", "u8", body)
    );
    let dslx = format!(
        "{declarations}\n{}",
        expr_eval::dslx_source("x: u8", "u8", body)
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    let x = IrValue::make_ubits(8, 42)?;
    assert_eq!(evaluator.compare(std::slice::from_ref(&x))?, x);
    Ok(())
}

#[test]
fn register_struct_and_extern_bit_widths_match_dslx_annotations() {
    for (bx, dslx) in [
        (
            "block sample(clk, x: sN[65]) -> sN[65] { reg q: WIDTH; q := x; q }",
            "fn sample(x: sN[65]) -> sN[65] { let q: WIDTH = x; q }",
        ),
        (
            "struct S { words: WIDTH[2] } block sample(clk, x: S) -> sN[65] { x.words[0] }",
            "struct S { words: WIDTH[2] } fn sample(x: S) -> sN[65] { x.words[0] }",
        ),
        (
            "extern fn echo(x: WIDTH) -> sN[65]; block sample(clk, x: sN[65]) -> sN[65] { echo(x) }",
            "fn echo(x: WIDTH) -> sN[65] { x } fn sample(x: sN[65]) -> sN[65] { echo(x) }",
        ),
        (
            "extern block echo(clk, x: sN[65]) -> WIDTH; block sample(clk, x: sN[65]) -> sN[65] { inst e: echo; e.x := x; e.out }",
            "fn echo(x: sN[65]) -> WIDTH { x } fn sample(x: sN[65]) -> sN[65] { echo(x) }",
        ),
    ] {
        for (width, accepted) in [("s65", false), ("sN[65]", true)] {
            let bx = bx.replace("WIDTH", width);
            let dslx = dslx.replace("WIDTH", width);
            let checked = parse_module(&bx).and_then(|module| check_module(&module));
            assert_eq!(checked.is_ok(), accepted, "{bx}\n{:?}", checked.err());
            let reference = compile_dslx_module(&dslx, &[]);
            assert_eq!(reference.is_ok(), accepted, "{dslx}\n{:?}", reference.err());
        }
    }
}

#[test]
fn identifiers_outside_the_bit_width_shorthand_range_match_dslx()
-> Result<(), Box<dyn std::error::Error>> {
    let root = tempfile::tempdir()?;
    std::fs::write(
        root.path().join("s65.x"),
        "pub fn u0(a: u8, b: u8) -> u8 { a + b }",
    )?;
    let declaration = "struct u84 { s0: u8 }";
    let body = "let s065: u84 = u84 { s0: u65 }; s65::u0(s065.s0, s0)";
    let params = "u65: u8, s0: u8";
    let bx = format!(
        "use s65;\n{declaration}\n{}",
        expr_eval::bx_source(params, "u8", body)
    );
    let dslx = format!(
        "import s65;\n{declaration}\n{}",
        expr_eval::dslx_source(params, "u8", body)
    );
    let options = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &options,
        compile_dslx_module(&dslx, &[root.path()]),
    )?;
    let result = evaluator.compare(&[IrValue::make_ubits(8, 5)?, IrValue::make_ubits(8, 7)?])?;
    assert_eq!(result, IrValue::make_ubits(8, 12)?);
    Ok(())
}

#[test]
fn packed_slice_constexpr_widths_and_signedness_match_dslx() {
    let body = r#"
    let width = u32:4;
    let inferred_width = 4;
    let two_bits = 3;
    (
        x[0+:bits[0x4]],
        x[0+:uN[2 + 2]],
        x[0+:sN[width]] as s8,
        x[0+:xN[!false][inferred_width]] as s8,
        x[0+:xN[u1:0][4]] as u8,
        x[0+:uN[two_bits + two_bits]],
        x[0+:uN[if true { two_bits } else { 4 }]],
        x[0+:uN[(if false { two_bits } else { 4 }) + 0]],
        x[0+:uN[two_bits >> u4:3]],
    )
"#;
    let x = IrValue::make_ubits(8, 0xba).unwrap();
    let expected = IrValue::parse_typed(
        "(bits[4]:10, bits[4]:10, bits[8]:0xfa, bits[8]:0xfa, bits[8]:10, bits[2]:2, bits[3]:2, bits[4]:10, bits[0]:0)",
    )
    .unwrap();
    assert_expr_matches_dslx(
        "x: u8",
        "(u4, u4, s8, s8, u8, u2, u3, u4, uN[0])",
        body,
        [(vec![x], expected)],
    );
}

#[test]
fn packed_slices_fold_in_register_resets() {
    let declarations = "let seed = uN[96]:0xd9; let offset = u3:1;";
    let result = "(u4, u4, u4, u4)";
    let initializer = r#"(
        seed[-(1 - 6)+:u4],
        seed[(6 + 1) + offset+:u4],
        (u8:0xd9)[6+:u4],
        seed[0x1_0000_0000_0000_0000+:u4],
    )"#;
    let evaluator = ExpressionEvaluator::compile(
        "",
        result,
        &format!("{declarations} {initializer}"),
        compile_dslx,
    )
    .unwrap();
    let actual = evaluator.compare(&[]).unwrap();
    assert_eq!(
        actual,
        IrValue::parse_typed("(bits[4]:6, bits[4]:9, bits[4]:3, bits[4]:0)").unwrap(),
    );
    evaluator
        .compare_reset(&[], result, declarations, initializer, &actual)
        .unwrap();
}

#[test]
fn packed_slices_require_unsigned_operands_and_a_bit_result_that_fits_the_source() {
    for (inputs, expr, result) in [
        ("x: s8", "x[0+:u4]", "u4"),
        ("x: u8, i: s4", "x[i+:u4]", "u4"),
        ("x: u8", "x[-1+:u4]", "u4"),
        ("x: u8", "x[uN[0]:false+:u1]", "u1"),
        ("x: u8", "x[1 == 1 == 1+:u1]", "u1"),
        ("x: u8, i: u3", "x[i + -1+:u1]", "u1"),
        ("flag: bool, i: u3", "flag[6 + i+:u1]", "u1"),
        (
            "flag: bool",
            "let n = if flag { 3 } else { 2 }; flag[n+:u1]",
            "u1",
        ),
        ("flag: bool, x: u8", "let p = (3, x); flag[p.0+:u1]", "u1"),
        (
            "flag: bool",
            "let a = [1, 2]; let n = a[flag]; flag[n+:u1]",
            "u1",
        ),
        ("x: u8", "x[256+:u1]", "u1"),
        ("x: u8", "0x07[0+:u4]", "u4"),
        ("x: u8", "0[0+:u1]", "u1"),
        ("x: u8[2]", "x[0+:u4]", "u4"),
        ("x: u8, i: u4[2]", "x[i+:u4]", "u4"),
        ("x: u8", "x[0+:u4[1]]", "u4[1]"),
        ("x: u8, width: u32", "x[0+:uN[width]]", "u4"),
        ("x: u8", "x[0+:uN[u3:4]]", "u4"),
        ("x: u8", "let a = 3; x[0+:uN[a - 2]]", "u1"),
        (
            "x: u8",
            "let a = 3; x[0+:uN[(if true { a } else { 4 }) + 0]]",
            "u3",
        ),
        ("x: u8, flag: bool", "x[0+:xN[flag][4]]", "s4"),
        ("x: u8", "x[0+:xN[u2:1][4]]", "s4"),
        ("x: u8", "x[0+:(u1)]", "u1"),
        ("x: u8", "u8:0xd9[6+:u4]", "u4"),
        ("x: u8, i: u3", "x[(i >> 4)+:u2]", "u2"),
        ("x: uN[84]", "x[0+:u65]", "uN[65]"),
        ("x: uN[84]", "x[0+:s65]", "sN[65]"),
        ("x: uN[84]", "x[0+:u84]", "uN[84]"),
        ("x: u8", "x[0+:u0]", "uN[0]"),
        ("x: u8", "x[0+:s0]", "sN[0]"),
        ("x: u8", "x[0+:u9]", "u9"),
    ] {
        assert_expr_rejected_by_both(inputs, result, expr);
    }
}

#[test]
fn packed_slice_dimensions_check_runtime_inputs_in_untaken_branches() {
    for (inputs, body, result) in [
        (
            "x: u8, width: u32",
            "x[0+:uN[if true { 1 } else { width }]]",
            "u1",
        ),
        (
            "x: u8, width: u32",
            "let w = if true { u32:1 } else { width }; let n = w; x[0+:uN[n]]",
            "u1",
        ),
        (
            "x: u8, flag: bool",
            "x[0+:xN[if false { flag } else { false }][1]]",
            "u1",
        ),
    ] {
        if std::env::var_os("BX_DSLX_IR_CONVERTER").is_some() {
            assert_expr_rejected_by_both(inputs, result, body);
        } else {
            // libxls v0.61.0 folds these branches. Current DSLX requires all
            // inputs to a type dimension to be constexpr. Move these cases to
            // the common rejection matrix when the bundled frontend catches up.
            let source = format!("block invalid(clk, {inputs}) -> {result} {{ {body} }}");
            assert!(
                check_module(&parse_module(&source).unwrap()).is_err(),
                "{source}"
            );
            assert!(compile_dslx(inputs, result, body).is_ok(), "{body}");
        }
    }
}

#[test]
fn packed_slice_rejects_inconsistent_widths_in_a_let_bound_start() {
    // The DSLX converter aborts when lowering the inferred u3 and u8 operands
    // of this addition. BX should diagnose the mismatch during type checking.
    let source = "block invalid(clk, x: u8) -> u1 { let a = 7; x[a + 1+:u1] }";
    let parsed = parse_module(source).unwrap();
    assert!(check_module(&parsed).is_err());
}
