// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx_module,
};
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};
use std::error::Error;
use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::parse_xls_ir_package;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn lambda_annotations_resolve_the_enclosing_block_type_parameter() -> Result<(), Box<dyn Error>> {
    let source = r#"
block identity<T>(clk, xs: T[2]) -> T[2] { map(xs, |x: T| -> T { x }) }
pub block wrapper(clk, a: (u8, bool)[2], b: u16[2]) -> ((u8, bool)[2], u16[2]) {
    let pairs = identity(a);
    let words = identity(b);
    (pairs, words)
}
"#;
    let mut package = compile(
        source,
        empty_package("generic_map"),
        ResetBehavior::default(),
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    inline_all_blocks_in_package(&mut package)?;
    let function = combinational_block_to_fn(package.get_top_block().unwrap())?;
    let args = [
        IrValue::parse_typed("[(bits[8]:0,bits[1]:1),(bits[8]:255,bits[1]:0)]")?,
        IrValue::parse_typed("[bits[16]:1234,bits[16]:65535]")?,
    ];
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &args),
        IrValue::make_tuple(&args)
    );
    Ok(())
}

#[test]
fn annotated_map_transforms_each_signed_element_to_a_tuple() -> Result<(), Box<dyn Error>> {
    let body = r#"map(xs, |x: s8| -> (s16, bool) {
        let wide = x as s16;
        (wide + s16:1, x < s8:0)
    })"#;
    assert_expr_matches_dslx(
        "xs: s8[3]",
        "(s16, bool)[3]",
        body,
        [(
            vec![IrValue::parse_typed(
                "[bits[8]:254, bits[8]:0, bits[8]:127]",
            )?],
            IrValue::parse_typed(
                "[(bits[16]:65535, bits[1]:1), (bits[16]:1, bits[1]:0), (bits[16]:128, bits[1]:0)]",
            )?,
        )],
    );
    Ok(())
}

#[test]
fn nested_capture_free_maps_shadow_parameters_and_call_module_helpers() -> Result<(), Box<dyn Error>>
{
    let functions = r#"
fn widen(x: u8) -> u16 { x as u16 }
fn nested(rows: u8[2][2], x: u8) -> u16[2][2] {
    map(rows, |x| {
        map(x, |x| { let rows = widen(x); rows + u16:1 })
    })
}
"#;
    let params = "rows: u8[2][2], x: u8";
    let result = "u16[2][2]";
    let body = "nested(rows, x)";
    let bx = format!("{functions}\n{}", bx_source(params, result, body));
    let native = format!("{functions}\n{}", dslx_source(params, result, body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    let rows = IrValue::parse_typed("[[bits[8]:0, bits[8]:7], [bits[8]:254, bits[8]:255]]")?;
    assert_eq!(
        evaluator.compare(&[rows, IrValue::make_ubits(8, 88)?])?,
        IrValue::parse_typed("[[bits[16]:1, bits[16]:8], [bits[16]:255, bits[16]:256]]")?,
    );
    Ok(())
}

#[test]
fn inferred_map_bodies_can_bind_tuples_and_produce_arrays() -> Result<(), Box<dyn Error>> {
    assert_expr_matches_dslx(
        "left: (u8,u8)[2], right: (u8,u8)[2], pick: bool",
        "u8[2][2]",
        "map(if pick {left} else {right}, |pair| -> u8[2] { let (x,y) = pair; [y, x ^ y] })",
        [(
            vec![
                IrValue::parse_typed("[(bits[8]:1,bits[8]:2), (bits[8]:3,bits[8]:4)]")?,
                IrValue::parse_typed("[(bits[8]:7,bits[8]:8), (bits[8]:9,bits[8]:10)]")?,
                IrValue::bool(false),
            ],
            IrValue::parse_typed("[[bits[8]:8,bits[8]:15], [bits[8]:10,bits[8]:3]]")?,
        )],
    );
    assert_expr_matches_dslx(
        "xs: u8[2]",
        "s8[2]",
        "map(xs, |_| -> s8 { 0xff })",
        [(
            vec![IrValue::parse_typed("[bits[8]:0,bits[8]:7]")?],
            IrValue::parse_typed("[bits[8]:255,bits[8]:255]")?,
        )],
    );
    assert_expr_matches_dslx(
        "xs: u8[2]",
        "u16[2]",
        "map(xs, |x| x as u16)",
        [(
            vec![IrValue::parse_typed("[bits[8]:0,bits[8]:255]")?],
            IrValue::parse_typed("[bits[16]:0,bits[16]:255]")?,
        )],
    );
    Ok(())
}

#[test]
fn map_preserves_range_values_and_uses_lambda_result_widths() -> Result<(), Box<dyn Error>> {
    for (result, body, expected) in [
        (
            "u16[4]",
            "map(u8:0..u8:4, |i| (i * i) as u16)",
            "[bits[16]:0, bits[16]:1, bits[16]:4, bits[16]:9]",
        ),
        (
            "u4[3]",
            "map((u8:15..=u8:17), |i| i as u4)",
            "[bits[4]:15, bits[4]:0, bits[4]:1]",
        ),
        (
            "s16[3]",
            "map(s8:-1..=s8:1, |i| i as s16)",
            "[bits[16]:65535, bits[16]:0, bits[16]:1]",
        ),
    ] {
        assert_expr_matches_dslx(
            "",
            result,
            body,
            [(Vec::new(), IrValue::parse_typed(expected)?)],
        );
    }
    Ok(())
}

#[test]
fn map_infers_the_input_literal_array_element_type() -> Result<(), Box<dyn Error>> {
    assert_expr_matches_dslx(
        "",
        "u8[3]",
        "map([1, 2, 3], |x| -> u8 { x as u8 })",
        [(
            Vec::new(),
            IrValue::parse_typed("[bits[8]:1,bits[8]:2,bits[8]:3]")?,
        )],
    );
    Ok(())
}

#[test]
fn map_lambda_uses_dslx_identifier_and_operator_tokenization() -> Result<(), Box<dyn Error>> {
    let xs = IrValue::parse_typed("[bits[8]:0, bits[8]:17]")?;
    assert_expr_matches_dslx(
        "xs: u8[2]",
        "u8[2]",
        "map(xs, |x| { let a!=x; a! })",
        [(vec![xs.clone()], xs.clone())],
    );
    assert_expr_matches_dslx(
        "xs: u8[2]",
        "bool[2]",
        "map(xs, |x| x != u8:0)",
        [(vec![xs], IrValue::parse_typed("[bits[1]:0, bits[1]:1]")?)],
    );
    assert_expr_rejected_by_both("xs: u8[2]", "bool[2]", "map(xs, |x| x!=u8:0)");
    assert_expr_rejected_by_both(
        "xs: u8[2]",
        "u8[2]",
        "map(xs, |x| { x /* comment */ + u8:1 })",
    );
    Ok(())
}

#[test]
fn lambda_annotation_must_match_the_executable_literal_element_width() -> Result<(), Box<dyn Error>>
{
    assert_expr_matches_dslx(
        "",
        "u2[2]",
        "map([1,3], |x:u2| x + 1)",
        [(Vec::new(), IrValue::parse_typed("[bits[2]:2, bits[2]:0]")?)],
    );
    for body in ["map([1,3], |x:u8| x + 1)", "map(1..3, |x:u8| x)"] {
        let bx = bx_source("", "u8[2]", body);
        let error = check_module(&parse_module(&bx)?).unwrap_err();
        assert!(
            error.contains("map lambda parameter expects"),
            "{error}\n{bx}"
        );
        let native = dslx_source("", "u8[2]", body);
        parse_and_typecheck(&native, "native.x", "native", &mut ImportData::default())?;
        let error = compile_dslx_module(&native, &[])
            .err()
            .ok_or("DSLX unexpectedly lowered an incompatible element width")?;
        assert!(
            error.contains("have type bits[8], has type bits[2]"),
            "{error}\n{native}"
        );
    }
    Ok(())
}

#[test]
fn invalid_map_inputs_and_lambda_signatures_match_dslx_rejection() {
    for (params, body) in [
        ("xs: u8", "map(xs, |x| { x })"),
        ("xs: u8[2]", "map(xs, |x: s8| -> u8 { x as u8 })"),
        ("xs: u8[2]", "map(xs, |x| -> u8 { x as u16 })"),
        ("xs: u8[2]", "map(xs, |x| { missing })"),
        ("xs: u8[2]", "map(xs, |x,y| { x + y })"),
        ("xs: u8[2]", "map(xs, || { u8:1 })"),
        ("xs: u8[2]", "map(xs, |x| -> u8 x)"),
    ] {
        assert_expr_rejected_by_both(params, "u8[2]", body);
    }
    assert_expr_rejected_by_both("xs: u8[0]", "u16[0]", "map(xs, |x| {x as u16})");
}

/// Leave 64 expansion units for the first callback. The two-element guard
/// bounds elaboration at ordinal 2 if callback charging regresses.
fn check_map_replay_budget(
    source: impl Fn(u32) -> String,
    owner: &str,
) -> Result<(), Box<dyn Error>> {
    compile(
        &source(2),
        empty_package("unrolled_map"),
        ResetBehavior::default(),
    )?;
    let oversized = parse_module(&source(1_048_576 - 64))?;
    let error = check_module(&oversized).unwrap_err();
    assert!(
        error.contains("unroll_for! iteration 0:")
            && error.ends_with(&format!(
                "map lambda exceeds the {owner} expansion limit of 1048576"
            )),
        "{error}"
    );
    Ok(())
}

#[test]
fn unrolled_maps_charge_each_replayed_callback_expression() -> Result<(), Box<dyn Error>> {
    let xor = (0..6).fold("x".to_owned(), |body, _| format!("({body} ^ {body})"));
    let zero_trip = format!("unroll_for! (_, carry) in u32:0..u32:0 {{ {xor} }}(x)");
    let source = |iterations, generic, callback: &str| {
        let fold = format!(
            r#"unroll_for! (i, acc) in u32:0..u32:{iterations} {{
                map(if guards[i] {{ acc }} else {{ xs }}, |x| {callback})
            }}(xs)"#
        );
        let params = "xs: u8[1], guards: bool[2]";
        if generic {
            let wrapper = bx_source(params, "u8[1]", "let result = fold(xs, guards); result");
            format!("block fold<T>(clk, xs: T[1], guards: bool[2]) -> T[1] {{ {fold} }}\n{wrapper}")
        } else {
            bx_source(params, "u8[1]", &fold)
        }
    };
    for (generic, owner) in [(false, "block"), (true, "generic block specialization")] {
        for callback in [&xor, &zero_trip] {
            check_map_replay_budget(|iterations| source(iterations, generic, callback), owner)?;
        }
    }
    // The zero-trip loop resolves all wildcards but never projects a row.
    // Keep the tuple type in a module helper so only the pattern is wide here.
    let row = format!("({})", ["u8"; 64].join(","));
    let helper = format!("fn rows() -> {row}[1] {{ zero!<{row}[1]>() }}");
    let wildcards = ["_"; 64].join(",");
    let pattern_loop = format!(
        "unroll_for! (_, carry) in u32:0..u32:0 {{ for (({wildcards}), a) in rows() {{ a }}(carry) }}(x)"
    );
    check_map_replay_budget(
        |iterations| format!("{helper}\n{}", source(iterations, false, &pattern_loop)),
        "block",
    )?;
    Ok(())
}

#[test]
fn unrolled_maps_charge_aggregate_parameters_and_intermediate_values() -> Result<(), Box<dyn Error>>
{
    // S5 expands to 63 tuples and 64 one-bit leaves in the callback's IR.
    let declarations = r#"
struct S0 { a: u1, b: u1 }
struct S1 { a: S0, b: S0 }
struct S2 { a: S1, b: S1 }
struct S3 { a: S2, b: S2 }
struct S4 { a: S3, b: S3 }
struct S5 { a: S4, b: S4 }
"#;
    for (element, callback) in [
        ("S5", "u1:1"),
        ("u1", "zero!<S5>().a.a.a.a.a.a"),
        ("u1", "zero!<u1[128]>()[u32:0]"),
    ] {
        let source = |iterations| {
            format!(
                r#"{declarations}
#[allow(combinational_block)]
block sample(clk, xs: {element}[1], guards: bool[2]) -> u1[1] {{
    unroll_for! (i, acc) in u32:0..u32:{iterations} {{
        if guards[i] {{ map(xs, |_| {callback}) }} else {{ acc }}
    }}([u1:0])
}}"#
            )
        };
        check_map_replay_budget(source, "block")?;
    }
    Ok(())
}

#[test]
fn map_callback_literal_bindings_infer_independently_of_enclosing_loops()
-> Result<(), Box<dyn Error>> {
    let xs = IrValue::parse_typed("[bits[8]:7, bits[8]:255]")?;
    let ones = IrValue::parse_typed("[bits[8]:1, bits[8]:1]")?;
    for wrapper in [
        "map(xs, |x| -> u8 { BODY })",
        "for (i, acc) in u32:0..u32:1 { map(acc, |x| -> u8 { BODY }) }(xs)",
        "unroll_for! (i, acc) in u32:0..u32:1 { map(acc, |x| -> u8 { BODY }) }(xs)",
    ] {
        // The pinned converter gives the bare let a u1 ABI even though the
        // lambda declares u8. Its counted-loop converter can abort on that ABI.
        let narrow = wrapper.replace("BODY", "let tiny=1; tiny");
        let bx = bx_source("xs:u8[2]", "u8[2]", &narrow);
        assert!(
            check_module(&parse_module(&bx)?).is_err(),
            "BX accepted {bx}"
        );
        for body in ["let tiny=u8:1; tiny", "let tiny=1; tiny as u8"] {
            assert_expr_matches_dslx(
                "xs: u8[2]",
                "u8[2]",
                &wrapper.replace("BODY", body),
                [(vec![xs.clone()], ones.clone())],
            );
        }
    }
    for loop_kind in ["for", "unroll_for!"] {
        let body = format!(
            "map(xs, |x| {{ {loop_kind} (i, acc) in u32:0..u32:2 {{ acc + (i as u8) }}(x) }})"
        );
        assert_expr_matches_dslx(
            "xs: u8[2]",
            "u8[2]",
            &body,
            [(
                vec![xs.clone()],
                IrValue::parse_typed("[bits[8]:8, bits[8]:0]")?,
            )],
        );
    }
    Ok(())
}

#[test]
fn map_captures_match_dslx_in_functions_and_blocks() -> Result<(), Box<dyn Error>> {
    let params = "xs: u8[2], bias: u8";
    let result = "u8[2]";
    let xs = IrValue::parse_typed("[bits[8]:2, bits[8]:9]")?;
    for (body, with_three, with_seven) in [
        ("map(xs, |x| x + bias)", [5, 12], [9, 16]),
        ("map(u8:0..u8:2, |i| i + bias)", [3, 4], [7, 8]),
        (
            "let local = bias + u8:1; map(xs, |x| x + local)",
            [6, 13],
            [10, 17],
        ),
        (
            "map(xs, |x| { let bias = x + bias; bias + x })",
            [7, 21],
            [11, 25],
        ),
        (
            "map(xs, |outer| map([u8:2], |inner| inner + outer + bias)[u32:0])",
            [7, 14],
            [11, 18],
        ),
        (
            "for (i, acc) in u8:0..u8:2 { map(acc, |x| x + i + bias) }(xs)",
            [9, 16],
            [17, 24],
        ),
        (
            "for (i, acc) in u32:0..u32:2 { map(acc, |i| -> u8 { i + u8:1 }) }(xs)",
            [4, 11],
            [4, 11],
        ),
        (
            "map(xs, |x| for (i, a) in u8:0..u8:2 { a + x + i + bias }(u8:0))",
            [11, 25],
            [19, 33],
        ),
        (
            "unroll_for! (i, acc) in u8:0..u8:2 { map(acc, |x| x + i) }(xs)",
            [3, 10],
            [3, 10],
        ),
        (
            "unroll_for! (_, acc) in u8:0..u8:0 { map(acc, |x| x + bias) }(xs)",
            [2, 9],
            [2, 9],
        ),
    ] {
        let native = dslx_source(params, result, body);
        for bx in [
            bx_source(params, result, body),
            format!(
                "fn local_fn({params}) -> {result} {{ {body} }}\n{}",
                bx_source(params, result, "local_fn(xs, bias)")
            ),
        ] {
            let evaluator = ExpressionEvaluator::compile_block(
                &bx,
                &ImportOptions::default(),
                compile_dslx_module(&native, &[]),
            )?;
            for (bias, [a, b]) in [(3, with_three), (7, with_seven)] {
                let args = [xs.clone(), IrValue::make_ubits(8, bias)?];
                let expected =
                    IrValue::make_array(&[IrValue::make_ubits(8, a)?, IrValue::make_ubits(8, b)?])?;
                assert_eq!(evaluator.compare(&args)?, expected, "{bx}");
            }
        }
    }
    Ok(())
}

#[test]
fn captured_maps_can_change_the_element_layout() -> Result<(), Box<dyn Error>> {
    assert_expr_matches_dslx(
        "xs: u8[2], bias: u16",
        "(u16[2], bool)[2]",
        "map(xs, |x| -> (u16[2], bool) { ([x as u16, bias], (x as u16) < bias) })",
        [(
            vec![
                IrValue::parse_typed("[bits[8]:3, bits[8]:10]")?,
                IrValue::make_ubits(16, 7)?,
            ],
            IrValue::parse_typed(
                "[([bits[16]:3, bits[16]:7], bits[1]:1), ([bits[16]:10, bits[16]:7], bits[1]:0)]",
            )?,
        )],
    );
    Ok(())
}

#[test]
fn captured_maps_with_empty_array_result_fields_survive_optimization() -> Result<(), Box<dyn Error>>
{
    let source = r#"
#[allow(combinational_block)]
pub block sample(clk, xs: u8[3], empty: u8[0]) -> (u8, u8[0])[3] {
    map(xs, |x| -> (u8, u8[0]) { (x, empty) })
}
"#;
    let package = compile(source, empty_package("captures"), ResetBehavior::default())?;
    let optimized = xlsynth::optimize_ir_text(&package.to_string(), "sample")?;
    parse_xls_ir_package(&optimized).map_err(|e| format!("{e:?}"))?;
    Ok(())
}

#[test]
fn map_reads_register_and_child_output_at_the_current_cycle() -> xlsynth_vastly::Result<()> {
    let source = r#"
#[allow(combinational_block)]
block child(clk, d: u8) -> u8 { d + u8:1 }
pub block sample(clk, en: bool, d: u8) -> (u8, u8) {
    reg q: u8 = u8:0;
    if en { q := d; }
    let c = child(d);
    let ys = map(u8:0..u8:2, |i| i + q + c);
    (ys[u32:0], ys[u32:1])
}
"#;
    let mut package = compile(source, empty_package("captures"), ResetBehavior::default()).unwrap();
    inline_all_blocks_in_package(&mut package).unwrap();
    let optimized = xlsynth::optimize_ir_text(&package.to_string(), "sample").unwrap();
    let package = parse_xls_ir_package(&optimized).unwrap();
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let module = compile_pipeline_module(&rtl.system_verilog)?;
    let mut runner = FixtureRunner::new(&module)?;
    let ports = runner.port_bindings();
    let (rst, en, d) = (ports.input("rst")?, ports.input("en")?, ports.input("d")?);
    for (reset, enable, data, expected) in [(1, 0, 4, 5), (0, 1, 8, 17), (0, 0, 10, 19)] {
        let edge = runner.step_cycle_with_drive(&mut [], |drive| {
            drive.set_u64(&rst, reset)?;
            drive.set_u64(&en, enable)?;
            drive.set_u64(&d, data)
        })?;
        assert_eq!(edge.outputs["out_0"].to_u64_if_known(), Some(expected));
        assert_eq!(edge.outputs["out_1"].to_u64_if_known(), Some(expected + 1));
    }
    Ok(())
}

#[test]
fn captures_keep_their_materialized_width_inside_counted_loops() -> Result<(), Box<dyn Error>> {
    let params = "xs: u8[2]";
    let source = |ty, body| format!("for (i, acc) in {ty} {{ {body} }}(xs)");
    // v0.61 can accept these declarations yet pass narrower bits to the
    // callback. Its converter can abort on the resulting ABI mismatch.
    for body in [
        source("u8:0..u8:1", "let a = 1; map(acc, |_| -> u8 { a })"),
        source("0..2", "map(acc, |x| x + i)"),
    ] {
        let bx = bx_source(params, "u8[2]", &body);
        assert!(check_module(&parse_module(&bx)?).is_err(), "{bx}");
    }
    let xs = IrValue::parse_typed("[bits[8]:1, bits[8]:3]")?;
    for (body, result) in [
        (
            source("u8:0..u8:1", "let a = 1; map(acc, |_| -> u8 { a as u8 })"),
            "[bits[8]:1, bits[8]:1]",
        ),
        (
            source("0..2", "map(acc, |x| x + (i as u8))"),
            "[bits[8]:2, bits[8]:4]",
        ),
    ] {
        assert_expr_matches_dslx(
            params,
            "u8[2]",
            &body,
            [(vec![xs.clone()], IrValue::parse_typed(result)?)],
        );
    }
    Ok(())
}

#[test]
fn aliases_preserve_the_materialized_width_of_a_capture() -> Result<(), Box<dyn Error>> {
    for (result, callback) in [
        ("u8", "let (b,) = (a,); b"),
        (
            "u8",
            "let b = a; for (_, acc) in u8:0..u8:1 { let c = b; acc + c }(x)",
        ),
        ("bool", "let (b,) = (a,); x == b"),
    ] {
        let bx = format!(
            "fn sample(xs: u8[2]) -> {result}[2] {{ let a = 1; map(xs, |x| -> {result} {{ {callback} }}) }}"
        );
        let error = check_module(&parse_module(&bx)?).unwrap_err();
        assert!(error.contains("map lambda body"), "{error}\n{bx}");
    }
    // A same-width hex bit pattern can still be compared with signed bits.
    let body = "let a = 0xff; map(xs, |x| { let (b,) = (a,); x == b })";
    let params = "xs: s8[2]";
    let result = "bool[2]";
    let bx = format!(
        "fn local_fn({params}) -> {result} {{ {body} }}\n{}",
        bx_source(params, result, "local_fn(xs)")
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx_source(params, result, body), &[]),
    )?;
    let xs = IrValue::parse_typed("[bits[8]:255, bits[8]:0]")?;
    assert_eq!(
        evaluator.compare(&[xs])?,
        IrValue::parse_typed("[bits[1]:1, bits[1]:0]")?,
    );
    Ok(())
}

#[test]
fn map_lambda_rejects_block_instantiation_in_an_empty_fold() -> Result<(), String> {
    let source = r#"
block child(clk, x: u8) -> u8 { x }
pub block sample(clk, xs: u8[2], bias: u8) -> u8[2] {
    map(xs, |x| unroll_for! (_, a) in u32:0..u32:0 { child(a + bias) }(x))
}
"#;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert!(
        error.contains("pure function cannot call block 'child'"),
        "{error}"
    );
    Ok(())
}

#[test]
fn map_lambda_cannot_call_a_helper_hidden_by_an_enclosing_let() -> Result<(), Box<dyn Error>> {
    let helper = "fn helper(x: u8) -> u8 { x + u8:1 }";
    let body = "let helper = u8:7; map(xs, |x| { helper(x) })";
    let bx = format!("{helper}\n{}", bx_source("xs: u8[2]", "u8[2]", body));
    let native = format!("{helper}\n{}", dslx_source("xs: u8[2]", "u8[2]", body));
    assert!(
        parse_and_typecheck(&native, "native.x", "native", &mut ImportData::default()).is_err()
    );
    assert!(
        check_module(&parse_module(&bx)?).is_err(),
        "BX accepted {bx}"
    );
    Ok(())
}

#[test]
fn local_map_binding_shadows_the_builtin_in_a_scoped_expression() -> Result<(), Box<dyn Error>> {
    for body in [
        "{ let map = true; map(xs, |x| x) }",
        "match pick { true => { let map = true; map(xs, |x| x) }, _ => xs }",
    ] {
        let params = "xs: u8[2], pick: bool";
        let helper = format!("fn f({params}) -> u8[2] {{ {body} }}");
        let bx = format!("{helper}\n{}", bx_source(params, "u8[2]", "f(xs, pick)"));
        let native = format!("{helper}\n{}", dslx_source(params, "u8[2]", "f(xs, pick)"));
        assert!(
            parse_and_typecheck(&native, "native.x", "native", &mut ImportData::default()).is_err(),
            "DSLX accepted {native}"
        );
        assert!(
            check_module(&parse_module(&bx)?).is_err(),
            "BX accepted {bx}"
        );
    }
    Ok(())
}

#[test]
fn map_computes_a_constant_register_reset_with_a_captured_binding() -> Result<(), Box<dyn Error>> {
    let source = r#"
block sample(clk, data: u16[3]) -> u16[3] {
    let offset = u16:1;
    reg q: u16[3] = map([u8:3,u8:5,u8:255], |v| {
        let wide = v as u16;
        if true { wide + offset } else { data[u32:0] }
    });
    q := data;
    q
}
"#;
    let package = compile(source, empty_package("map_reset"), ResetBehavior::default())?;
    let expected = IrValue::parse_typed("[bits[16]:4, bits[16]:6, bits[16]:256]")?;
    assert_eq!(
        package.get_top_block().unwrap().registers[0]
            .reset_value
            .as_ref(),
        Some(&expected)
    );
    let runtime_reset = source.replace("if true", "if false");
    let error = check_module(&parse_module(&runtime_reset)?).unwrap_err();
    assert!(error.contains("must be a compile-time constant"), "{error}");
    Ok(())
}
