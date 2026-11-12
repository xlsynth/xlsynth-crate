// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::ExpressionEvaluator;
use std::error::Error;
use xlsynth::{
    XlsIrValue,
    dslx::{self, ImportData, MatchableModuleMember},
};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::{MemberType, NodePayload};
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::libxls_bridge::value_from_libxls;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn block_tuple_patterns_can_rebind_previous_values_in_lexical_order() -> Result<(), String> {
    let body = r#"
        let (x, old) = (x + u8:1, x);
        let x = x + old;
        let ((old, x), ..) = ((x, old), false);
        (old, x)
    "#;
    let bx = format!("block sample(clk, x:u8)->(u8,u8) {{ {body} }}");
    let dslx = format!("fn sample(x:u8)->(u8,u8) {{ {body} }}");
    let eval = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    let actual = eval.compare(&[IrValue::make_ubits(8, 5).unwrap()])?;
    assert_eq!(
        actual,
        IrValue::parse_typed("(bits[8]:11,bits[8]:5)").unwrap()
    );
    Ok(())
}

#[test]
fn destructured_inferred_bits_can_widen_at_each_use() -> Result<(), Box<dyn Error>> {
    let helper = "fn use_part(x:s3)->s3 { x }";
    let body = r#"
        let ((raw, _), pair) = ((1, x), ([1,2], x));
        (use_part(raw), raw, pair)
    "#;
    let result = "(s3, u8, (s8[2], u8))";
    let dslx = format!("{helper} fn sample(x:u8)->{result} {{ {body} }}");
    dslx::parse_and_typecheck(&dslx, "width.x", "width", &mut ImportData::default())?;
    // The native checker widens the name references, but its converter
    // supplies the narrow initializer to the invoke. Spell the same concrete
    // types explicitly in the reference so it has a valid IR signature.
    let reference =
        format!("{helper} fn sample(x:u8)->{result} {{ (use_part(s3:1), u8:1, ([s8:1,s8:2],x)) }}");
    for bx in [
        format!("{helper} block sample(clk,x:u8)->{result} {{ {body} }}"),
        format!(
            "{helper} fn project(x:u8)->{result} {{ {body} }} block sample(clk,x:u8)->{result} {{ project(x) }}"
        ),
    ] {
        let eval = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&reference, &[]),
        )?;
        let actual = eval.compare(&[IrValue::make_ubits(8, 17)?])?;
        assert_eq!(
            actual,
            IrValue::parse_typed("(bits[3]:1, bits[8]:1, ([bits[8]:1,bits[8]:2],bits[8]:17))")?
        );
    }
    Ok(())
}

#[test]
fn comparisons_unify_destructured_literal_widths_in_blocks_and_unrolls()
-> Result<(), Box<dyn Error>> {
    for (initializer, expected) in [
        ("((1,), (17,))", false),
        ("((1,), (-1,))", false),
        ("((0xff,), (-1,))", true),
        ("((0xff,), (s8:-1,))", true),
        ("(0xff, -1)", true),
        ("(([1], [0xff]), ([-1], [-1]))", false),
    ] {
        let comparison = format!("let (lo, hi) = {initializer}; lo == hi");
        for body in [
            comparison.clone(),
            format!("unroll_for! (_, a) in u8:0..u8:1 {{ {comparison} }}(flag)"),
        ] {
            let bx = format!("block sample(clk, flag:bool)->bool {{ {body} }}");
            let dslx = format!("fn sample(flag:bool)->bool {{ {body} }}");
            dslx::parse_and_typecheck(&dslx, "compare.x", "compare", &mut ImportData::default())?;
            // The native converter reads the narrow binder values even when
            // DSLX assigns the references wider types for comparison.
            let reference = format!("fn sample(flag:bool)->bool {{ {expected} }}");
            let evaluator = ExpressionEvaluator::compile_block(
                &bx,
                &ImportOptions::default(),
                compile_dslx_module(&reference, &[]),
            )?;
            assert_eq!(
                evaluator.compare(&[IrValue::bool(false)])?,
                IrValue::bool(expected)
            );
        }
    }
    // Decimal 255 needs nine signed bits; the explicit s8 operand fixes the
    // comparison width, so the inferred member cannot supply that type.
    let body = "let (lo, hi) = ((255,), (s8:-1,)); lo == hi";
    let bx = format!("block sample(clk)->bool {{ {body} }}");
    let dslx = format!("fn sample()->bool {{ {body} }}");
    assert!(check_module(&parse_module(&bx)?).is_err());
    assert!(
        dslx::parse_and_typecheck(&dslx, "compare.x", "compare", &mut ImportData::default())
            .is_err()
    );
    Ok(())
}

#[test]
fn destructured_aggregates_unify_with_inline_constructors() -> Result<(), Box<dyn Error>> {
    for (stored, inline, result) in [
        ("(1,)", "(17,)", "false"),
        ("(0xff,)", "(-1,)", "true"),
        ("(1, flag, u8:5)", "(17 + 1, true, u8:5)", "false"),
        ("(2, flag, u8:5)", "(1 + 1, true, u8:5)", "flag"),
    ] {
        for comparison in [format!("lo == {inline}"), format!("{inline} == lo")] {
            let body = format!("let (lo,) = ({stored},); {comparison}");
            let bx = format!("block sample(clk, flag:bool)->bool {{ {body} }}");
            let dslx = format!("fn sample(flag:bool)->bool {{ {body} }}");
            dslx::parse_and_typecheck(&dslx, "compare.x", "compare", &mut ImportData::default())?;
            // The converter can use the stored member's original width. The
            // reference expresses the result at the checked comparison width.
            let reference = format!("fn sample(flag:bool)->bool {{ {result} }}");
            let evaluator = ExpressionEvaluator::compile_block(
                &bx,
                &ImportOptions::default(),
                compile_dslx_module(&reference, &[]),
            )?;
            for flag in [false, true] {
                evaluator.compare(&[IrValue::bool(flag)])?;
            }
        }
    }
    Ok(())
}

#[test]
fn comparisons_infer_destructured_members_across_calls_and_branches() -> Result<(), Box<dyn Error>>
{
    let helpers = "fn identity(flag:bool)->bool { flag } fn wide(x:u8)->u8 { x }";
    for (stored, other, result) in [
        ("(1, flag)", "(17, !flag)", "false"),
        ("(1, flag)", "(17, identity(flag))", "false"),
        ("(1, flag)", "(17, flag && flag)", "false"),
        ("(1, flag)", "(wide(x), identity(flag))", "x == u8:1"),
        ("(1, 1)", "(17, wide(x))", "false"),
        ("(2, 1)", "(1 + 1, wide(x))", "x == u8:1"),
        ("(1,)", "if flag { (17,) } else { (1,) }", "!flag"),
        ("(1,)", "match flag { true => (17,), _ => (1,) }", "!flag"),
        (
            "((1,), flag)",
            "(if flag { (17,) } else { (1,) }, identity(flag))",
            "!flag",
        ),
    ] {
        for comparison in [
            format!("a == ({other})"),
            format!("({other}) == a"),
            format!("(if flag {{ a }} else {{ a }}) == ({other})"),
            format!("({other}) == (match flag {{ true => a, _ => a }})"),
        ] {
            let body = format!("let (a,) = ({stored},); {comparison}");
            let bx = format!(
                "{helpers} #[allow(combinational_block)] block sample(clk, flag:bool, x:u8)->bool {{ {body} }}"
            );
            let dslx = format!("{helpers} fn sample(flag:bool, x:u8)->bool {{ {body} }}");
            dslx::parse_and_typecheck(&dslx, "compare.x", "compare", &mut ImportData::default())?;
            let reference = format!("fn sample(flag:bool, x:u8)->bool {{ {result} }}");
            let evaluator = ExpressionEvaluator::compile_block(
                &bx,
                &ImportOptions::default(),
                compile_dslx_module(&reference, &[]),
            )
            .map_err(|e| format!("{body}: {e}"))?;
            for flag in [false, true] {
                for x in [1, 17] {
                    evaluator.compare(&[IrValue::bool(flag), IrValue::make_ubits(8, x)?])?;
                }
            }
        }
    }
    let source = format!(
        "{helpers} #[allow(combinational_block)] block sample(clk, flag:bool, x:u8)->bool {{ let (a,) = ((1, flag),); a == (wide(x), identity(flag)) }}"
    );
    let package = compile(&source, empty_package("anchors"), ResetBehavior::default())?;
    let nodes = &package.get_block("sample").unwrap().nodes;
    for name in ["wide", "identity"] {
        assert_eq!(nodes.iter().filter(|n| matches!(&n.payload, NodePayload::Invoke { to_apply, .. } if to_apply == name)).count(), 1);
    }
    Ok(())
}

#[test]
fn captured_projections_retain_inferred_literal_widths() -> Result<(), Box<dyn Error>> {
    for projection in [
        "((0xff,), true).0",
        "[(0xff,)][u1:0]",
        "({ let raw = 0xff; ((raw,), true) }).0",
    ] {
        for (target, equal) in [("s8:-1", true), ("s16:255", true), ("s16:-1", false)] {
            for comparison in [
                format!("a == (1, {projection})"),
                format!("(1, {projection}) == a"),
            ] {
                let body = format!("let (a,) = ((1, ({target},)),); flag == ({comparison})");
                let function = format!("fn test(flag:bool)->bool {{ {body} }}");
                dslx::parse_and_typecheck(
                    &function,
                    "capture.x",
                    "capture",
                    &mut ImportData::default(),
                )?;
                // At wider use sites the native converter can read the original
                // u8. Its bits represent positive 255 after widening.
                let reference = format!("fn sample(flag:bool)->bool {{ flag == {equal} }}");
                for bx in [
                    format!("{function} block sample(clk, flag:bool)->bool {{ test(flag) }}"),
                    format!("block sample(clk, flag:bool)->bool {{ {body} }}"),
                ] {
                    let evaluator = ExpressionEvaluator::compile_block(
                        &bx,
                        &ImportOptions::default(),
                        compile_dslx_module(&reference, &[]),
                    )?;
                    for flag in [false, true] {
                        evaluator.compare(&[IrValue::bool(flag)])?;
                    }
                }
            }
        }
    }
    for literal in ["255", "u8:0xff"] {
        let body = format!("let (a,) = ((1, (s8:-1,)),); a == (1, (({literal},), true).0)");
        let function = format!("fn test()->bool {{ {body} }}");
        assert!(check_module(&parse_module(&function)?).is_err());
        assert!(
            dslx::parse_and_typecheck(
                &function,
                "capture.x",
                "capture",
                &mut ImportData::default()
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn a_projected_constant_can_initialize_a_register() -> Result<(), Box<dyn Error>> {
    let source = r#"
        block sample(clk, x:u8) -> u8 {
            let packet = (u8:1, (false, u8:7));
            let (_, (.., initial)) = packet;
            reg q:u8 = initial;
            q := x;
            q
        }
    "#;
    let package = compile(source, empty_package("patterns"), ResetBehavior::default())?;
    assert_eq!(
        package.get_block("sample").unwrap().registers[0].reset_value,
        Some(IrValue::make_ubits(8, 7)?)
    );
    Ok(())
}

#[test]
fn child_calls_resolve_before_subsequent_tuple_bindings() -> Result<(), String> {
    let source = r#"
        block child(clk) -> bool { true }
        block sample(clk, x:bool) -> bool {
            let cell = child();
            let (child, _) = (x, x);
            cell
        }
    "#;
    check_module(&parse_module(source)?)?;
    let later = source.replace("let cell = child();", "").replace(
        "            cell\n",
        "            let cell = child(); cell\n",
    );
    let error = check_module(&parse_module(&later)?).unwrap_err();
    assert!(
        error.contains("'child' names a local value here"),
        "{error}"
    );
    Ok(())
}

#[test]
fn block_if_arms_destructure_with_the_same_lexical_scope_as_dslx() -> Result<(), Box<dyn Error>> {
    let conditional = r#"if flag {
        let (x, ..) = pair;
        x
    } else {
        let (.., x) = pair;
        x
    }"#;
    let params = "flag: bool, pair: (u8, u8), x: u8";
    for body in [
        conditional.to_owned(),
        format!("let y = {conditional}; y + x"),
        "let (a, b) = { let (x, y) = pair; { (y, x) } }; if flag { a } else { b }".to_owned(),
    ] {
        let bx = format!("block sample(clk, {params}) -> u8 {{ {body} }}");
        let dslx = format!("fn sample({params}) -> u8 {{ {body} }}");
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&dslx, &[]),
        )?;
        let pair = IrValue::parse_typed("(bits[8]:13, bits[8]:29)")?;
        for flag in [false, true] {
            evaluator.compare(&[
                IrValue::bool(flag),
                pair.clone(),
                IrValue::make_ubits(8, 2)?,
            ])?;
        }
    }
    Ok(())
}

#[test]
fn bx_value_scopes_preserve_tuple_binder_spellings() -> Result<(), String> {
    let scope = "let (out, (tail!, flag')) = (x, (x, true)); if flag' { out } else { tail! }";
    for body in [
        format!("match x {{ _ => {{ {scope} }} }}"),
        format!("for (_, a) in u8:0..u8:1 {{ {scope} }}(x)"),
    ] {
        let block = format!("block sample(clk, x:u8) -> u8 {{ {body} }}");
        check_module(&parse_module(&block)?)?;
    }
    // Newly introduced block statement patterns use DSLX binder names.
    for body in [
        "let (out, _) = (x, x); out",
        "unroll_for! (_, a) in u8:0..u8:1 { let (out, _) = (a, x); out }(x)",
    ] {
        assert!(parse_module(&format!("block sample(clk, x:u8) -> u8 {{ {body} }}")).is_err());
    }
    Ok(())
}

#[test]
fn bx_if_arms_preserve_fold_suffixes() -> Result<(), String> {
    let conditional = r#"if flag {
        unroll_for! (_, a) in u8:0..u8:1 { a }(x) + x
    } else {
        for (_, a) in u8:0..u8:1 { a }((x, x)).0
    }"#;
    for body in [conditional.to_owned(), format!("let y = {conditional}; y")] {
        let bx = format!("block sample(clk, flag:bool, x:u8) -> u8 {{ {body} }}");
        let reference = "fn sample(flag:bool, x:u8) -> u8 { if flag { x + x } else { x } }";
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(reference, &[]),
        )?;
        for flag in [false, true] {
            evaluator.compare(&[IrValue::bool(flag), IrValue::make_ubits(8, 5).unwrap()])?;
        }
    }
    Ok(())
}

#[test]
fn unit_arms_and_shadowed_intrinsics_follow_dslx_pattern_bindings() -> Result<(), Box<dyn Error>> {
    for body in [
        "if flag { let (a, _) = pair; } else {}",
        "if flag { let (a, _) = pair; a; } else { () }",
        "let u = if flag {} else { let (.., b) = pair; }; u",
    ] {
        let params = "flag: bool, pair: (u8, u8)";
        let bx = format!("block sample(clk, {params}) -> () {{ {body} }}");
        let dslx = format!("fn sample({params}) -> () {{ {body} }}");
        check_module(&parse_module(&bx)?)?;
        dslx::parse_and_typecheck(&dslx, "unit.x", "unit", &mut ImportData::default())?;
    }
    let body = "let (zero!, a) = (x, x); zero!<u8>()";
    let bx = format!("block sample(clk, x: u8) -> u8 {{ {body} }}");
    let dslx = format!("fn sample(x: u8) -> u8 {{ {body} }}");
    assert!(check_module(&parse_module(&bx)?).is_err());
    assert!(
        dslx::parse_and_typecheck(&dslx, "zero.x", "zero", &mut ImportData::default()).is_err()
    );
    Ok(())
}

#[test]
fn scoped_if_arms_share_result_types_field_by_field() -> Result<(), Box<dyn Error>> {
    let helper = r#"
fn choose(flag: bool) -> (u8, u16) {
    let pair = if flag { let x = u8:7; (x, 255 + 1) }
               else { let x = u16:1; (3, x) };
    pair
}
"#;
    let bx = format!("{helper}\nblock sample(clk, flag: bool) -> (u8, u16) {{ choose(flag) }}");
    let dslx = format!("{helper}\nfn sample(flag: bool) -> (u8, u16) {{ choose(flag) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (flag, expected) in [
        (true, "(bits[8]:7, bits[16]:256)"),
        (false, "(bits[8]:3, bits[16]:1)"),
    ] {
        assert_eq!(
            evaluator.compare(&[IrValue::bool(flag)])?,
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

/// Evaluate a concrete call directly in DSLX, including callees with names
/// such as `zero!` that the pinned converter cannot serialize as IR symbols.
fn native_call(helpers: &str, call: &str) -> Result<XlsIrValue, Box<dyn Error>> {
    let source = format!("{helpers}\nconst RESULT = {call};");
    let native =
        dslx::parse_and_typecheck(&source, "calls.x", "calls", &mut ImportData::default())?;
    let module = native.get_module();
    let Some(MatchableModuleMember::ConstantDef(result)) = module
        .get_member(module.get_member_count() - 1)
        .to_matchable()
    else {
        return Err("missing DSLX result constant".into());
    };
    Ok(native
        .get_type_info()
        .get_const_expr(&result.get_value())?
        .convert_to_ir()?)
}

#[test]
fn a_block_can_call_a_later_function_with_aggregate_arguments() -> Result<(), Box<dyn Error>> {
    let lanes = "struct Lanes { values: s8[2] }";
    let widen = r#"
fn widen(lanes: Lanes, index: u1) -> s16 {
    let value: s16 = lanes.values[index] as s16;
    let value = value + s16:1;
    value
}"#;
    let route = r#"
pub fn route(lanes: Lanes, swap: bool) -> (s16, s16) {
    let first = widen(lanes, u1:0);
    let second = widen(lanes, u1:1);
    if swap { (second, first) } else { (first, second) }
}"#;
    let bx = format!(
        "{lanes}\nblock sample(clk, x: Lanes, swap: bool) -> (s16, s16) {{ route(x, swap) }}\n{widen}\n{route}"
    );
    let dslx = format!(
        "{lanes}\n{widen}\n{route}\nfn sample(x: Lanes, swap: bool) -> (s16, s16) {{ route(x, swap) }}"
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    let lanes = IrValue::make_tuple(&[IrValue::parse_typed("[bits[8]:128, bits[8]:127]")?]);
    for (swap, values) in [(false, [-127, 128]), (true, [128, -127])] {
        let expected = IrValue::make_tuple(&[
            IrValue::make_sbits(16, values[0])?,
            IrValue::make_sbits(16, values[1])?,
        ]);
        assert_eq!(
            evaluator.compare(&[lanes.clone(), IrValue::bool(swap)])?,
            expected
        );
    }
    Ok(())
}

#[test]
fn signed_literal_arguments_unify_before_the_parameter_type() -> Result<(), Box<dyn Error>> {
    // Existing block-to-extern calls supply the declared type as context.
    let extern_block = "extern fn take(x: s8) -> s8; block sample(clk) -> s8 { take(0x80 + 1) }";
    check_module(&parse_module(extern_block)?)?;
    for (ty, argument) in [
        ("s8", "0x80 + 1"),
        ("s8", "{ let _ = flag; 0x80 + 1 }"),
        ("s8", "if flag { 1 } else { 0xff }"),
        ("s8[2]", "[1, 0xff]"),
        ("s8[2]", "[{ let _ = flag; 0xff }, 1]"),
        ("(s8, s8)", "(0xff, 0x80 + 1)"),
        ("s8[2][2]", "[[s8:1, 1], [0xff, 1]]"),
    ] {
        let source = format!(
            "fn take(x: {ty}) -> {ty} {{ x }} fn caller(flag: bool) -> {ty} {{ take({argument}) }}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
        assert!(
            dslx::parse_and_typecheck(&source, "width.x", "width", &mut ImportData::default())
                .is_err(),
            "{source}"
        );
    }
    let helpers = r#"
fn take(x: s8) -> s8 { x }
fn pair(x: s8[2]) -> s8[2] { x }
fn compute(delta: s8) -> (s8, s8, s8, s8[2]) {
    (take(0x80), take(0x80 + delta), take(delta + (0x80 + 1)), pair([delta, 0xff]))
}"#;
    let bx = format!(
        "{helpers}\nblock sample(clk, delta: s8) -> (s8, s8, s8, s8[2]) {{ compute(delta) }}"
    );
    let dslx =
        format!("{helpers}\nfn sample(delta: s8) -> (s8, s8, s8, s8[2]) {{ compute(delta) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_sbits(8, 1)?])?,
        IrValue::parse_typed("(bits[8]:128, bits[8]:129, bits[8]:130, [bits[8]:1, bits[8]:255])")?,
    );
    Ok(())
}

#[test]
fn signed_call_arguments_follow_inline_projections() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn take(x: s8) -> s8 { x }
fn select(x: s8, flag: bool) -> (s8, s8, s8) {
    (take((0xff, x).0), take(([0x80], x).0[0]), take([(x,), (0xff,)][flag].0))
}"#;
    let bx = format!(
        "{helpers}\nblock sample(clk, x: s8, flag: bool) -> (s8, s8, s8) {{ select(x, flag) }}"
    );
    let dslx =
        format!("{helpers}\nfn sample(x: s8, flag: bool) -> (s8, s8, s8) {{ select(x, flag) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (flag, expected) in [
        (false, "(bits[8]:255, bits[8]:128, bits[8]:42)"),
        (true, "(bits[8]:255, bits[8]:128, bits[8]:255)"),
    ] {
        let actual = evaluator.compare(&[IrValue::make_sbits(8, 42)?, IrValue::bool(flag)])?;
        assert_eq!(actual, IrValue::parse_typed(expected)?);
        let native = native_call(helpers, &format!("select(s8:42, {flag})"))?;
        assert_eq!(actual, value_from_libxls(&native, &actual.type_())?);
    }
    let bx = "fn take(x: s8) -> s8 { x } block sample(clk) -> s8 { let raw = (0xff,); let alias = raw.0; take(alias) }";
    let dslx = "fn take(x: s8) -> s8 { x } fn sample() -> s8 { let raw = (0xff,); let alias = raw.0; take(alias) }";
    let evaluator = ExpressionEvaluator::compile_block(
        bx,
        &ImportOptions::default(),
        compile_dslx_module(dslx, &[]),
    )?;
    assert_eq!(evaluator.compare(&[])?, IrValue::make_sbits(8, -1)?);
    // A legacy block-to-extern call still requires the projection's
    // unsigned type to match the declared port, even inside a local call.
    for body in ["external((0xff,).0)", "take(external((0xff,).0))"] {
        let source = format!(
            "extern fn external(x: s8) -> s8; fn take(x: s8) -> s8 {{ x }} block sample(clk) -> s8 {{ {body} }}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    for argument in [
        "[0x80 + 1, x][flag]",
        "[(0x80 + 1,), (x,)][flag].0",
        "[(x,), (0x80 + 1,)][flag].0",
        "(if flag { (x,) } else { (0x80 + 1,) }).0",
        "(u8:0xff, x).0",
        "(255, x).0",
        "(0x80, x).0 + 1",
        "{ let raw = 0xff; raw + 1 }",
        "({ let raw = 0xff; (raw + 1,) }).0",
        "[{ let raw = 0xff; raw + 1 }, x][flag]",
    ] {
        let source = format!(
            "fn take(x: s8) -> s8 {{ x }} fn f(x: s8, flag: bool) -> s8 {{ take({argument}) }}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
        assert!(
            dslx::parse_and_typecheck(&source, "width.x", "width", &mut ImportData::default())
                .is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn equal_width_projected_results_preserve_aggregate_types() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn take_tuple(p: (s8,)) -> s8 { p.0 }
fn take_array(a: s8[1]) -> s8 { a[0] }
fn pattern() -> s8 { ({ let raw = 0xff; (raw,) }).0 }
fn bindings() -> (s8, s8, s8, s8) {
    let raw = 0xff;
    let alias = ({ let local = raw; (local,) }).0;
    let pair = ({ let high = 0x80; high }, if true { let _ = true; raw } else { raw });
    let (first, second) = pair;
    let sum = alias + second;
    (alias, first, take_tuple((second,)), sum)
}
fn select(x: u4) -> ((s8, u4), s8, s8, s8, (s8, s8, s8, s8)) {
    let partial: (s8, u4) = ({ let high = 0x80; ((high, x), x) }).0;
    (partial, take_tuple(((0xff,),).0), take_array(([0x80],).0), pattern(), bindings())
}
"#;
    let result = "((s8, u4), s8, s8, s8, (s8, s8, s8, s8))";
    let bx = format!("{helpers}\nblock sample(clk, x: u4) -> {result} {{ select(x) }}");
    let dslx = format!("{helpers}\nfn sample(x: u4) -> {result} {{ select(x) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    let actual = evaluator.compare(&[IrValue::make_ubits(4, 3)?])?;
    assert_eq!(
        actual,
        IrValue::parse_typed(
            "((bits[8]:128, bits[4]:3), bits[8]:255, bits[8]:128, bits[8]:255, (bits[8]:255, bits[8]:128, bits[8]:255, bits[8]:254))"
        )?
    );
    let native = native_call(helpers, "select(u4:3)")?;
    assert_eq!(actual, value_from_libxls(&native, &actual.type_())?);
    for (ty, value) in [
        ("(s8,)", "((u8:0xff,),).0"),
        ("(s8,)", "((255,),).0"),
        ("s8[2]", "([0xff, 0x80],).0"),
        ("s8", "[0x80 + 1, x][0]"),
        ("s8", "let a = 255; a"),
        ("s8", "let a: u8 = 0xff; a"),
        ("s8", "let a = 0xff; take_scoped({ let _ = true; a + 1 })"),
        ("s8", "let a = 0xff + 1; a"),
        ("s8", "let a = 0xff; let b = 0xff; let sum = a + b; sum"),
        ("s8", "let a = 0xff; let a: u8 = a; a"),
    ] {
        let source =
            format!("fn take_scoped(x: s8) -> s8 {{ x }} fn f(x: s8) -> {ty} {{ {value} }}");
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
        assert!(
            dslx::parse_and_typecheck(&source, "width.x", "width", &mut ImportData::default())
                .is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn signed_anchors_can_follow_bound_literals_in_runtime_expressions() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn select(x: s8, flag: bool) -> (bool, bool, bool, bool, s8, s8, s8, s8) {
    let raw = 0xff;
    let left = if flag { raw } else { x };
    let right = if flag { x } else { raw };
    (raw == x, x == raw, (raw,) == (x,), ({ let a = 0xff; a }) == x,
     left, right, [raw, x][flag], [x, raw][flag])
}"#;
    let result = "(bool, bool, bool, bool, s8, s8, s8, s8)";
    let bx = format!(
        "{helpers}\nblock sample(clk, x: s8, flag: bool) -> {result} {{ select(x, flag) }}"
    );
    let dslx = format!("{helpers}\nfn sample(x: s8, flag: bool) -> {result} {{ select(x, flag) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for x in [-1, 1] {
        for flag in [false, true] {
            let actual = evaluator.compare(&[IrValue::make_sbits(8, x)?, IrValue::bool(flag)])?;
            let native = native_call(helpers, &format!("select(s8:{x}, {flag})"))?;
            assert_eq!(actual, value_from_libxls(&native, &actual.type_())?);
        }
    }
    let aggregates = r#"
fn wrap(x: s8) -> (s8,) { (x,) }
fn compare(x: s8) -> (bool, bool, bool, bool, bool, bool, bool, bool) {
    let t = ((0xff,),);
    let a = [0xff];
    let left = (x, 0xff);
    let right = (0xff, x);
    (t.0 == (x,), a == [x], left == right, right == left, t.0 == wrap(x), t.0 == (x ^ s8:0,),
     ([t.0],) == ([(x,)],), ([(x,)],) == ([t.0],))
}"#;
    let result = "(bool, bool, bool, bool, bool, bool, bool, bool)";
    let bx = format!("{aggregates}\nblock sample(clk, x: s8) -> {result} {{ compare(x) }}");
    let dslx = format!("{aggregates}\nfn sample(x: s8) -> {result} {{ compare(x) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for x in [-1, 1] {
        let actual = evaluator.compare(&[IrValue::make_sbits(8, x)?])?;
        let native = native_call(aggregates, &format!("compare(s8:{x})"))?;
        assert_eq!(actual, value_from_libxls(&native, &actual.type_())?);
    }
    for expression in [
        "raw + 1 == x",
        "[raw, 1, x][0] == x",
        "if flag { raw + 1 } else { x } == x",
    ] {
        let source = format!("fn f(x: s8, flag: bool) -> bool {{ let raw = 0xff; {expression} }}");
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
        assert!(
            dslx::parse_and_typecheck(&source, "anchor.x", "anchor", &mut ImportData::default())
                .is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn local_function_drives_a_clock_enabled_register() -> xlsynth_vastly::Result<()> {
    let source = r#"
fn add_one(x: u8) -> u8 { x + u8:1 }
pub block capture(clk, en: bool, x: u8) -> u8 {
    reg q: u8 = u8:0;
    if en { q := add_one(x); }
    q
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    assert_eq!(package.top, Some(("capture".into(), MemberType::Block)));
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let pipeline = compile_pipeline_module(&rtl.system_verilog)?;
    let mut runner = FixtureRunner::new(&pipeline)?;
    let ports = runner.port_bindings();
    let reset = ports.input("rst")?;
    let enable = ports.input("en")?;
    let input = ports.input("x")?;
    for (rst, en, x, expected) in [
        (1, 1, 8, 0),
        (0, 1, 255, 0),
        (0, 1, 42, 43),
        (0, 0, 3, 43),
        (0, 0, 9, 43),
    ] {
        let cycle = runner.step_cycle_with_drive(&mut [], |drive| {
            drive.set_u64(&reset, rst)?;
            drive.set_u64(&enable, en)?;
            drive.set_u64(&input, x)
        })?;
        assert_eq!(cycle.outputs["out"].to_u64_if_known(), Some(expected));
    }
    Ok(())
}

#[test]
fn empty_parameter_lists_and_unit_functions_match_dslx() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn answer() -> u8 { u8:42 }
fn empty() {}
fn discard(x: u8) -> () { let _ = x; }
"#;
    let body = "let _ = empty(); let _ = discard(answer()); answer()";
    let bx = format!("{helpers}\nfn get() -> u8 {{ {body} }}\nblock sample(clk) -> u8 {{ get() }}");
    let dslx = format!("{helpers}\nfn sample() -> u8 {{ {body} }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    assert_eq!(evaluator.compare(&[])?, IrValue::make_ubits(8, 42)?);
    Ok(())
}

#[test]
fn invalid_function_definitions_and_calls_are_rejected_before_lowering() {
    for source in [
        "fn f(x: u8) -> u16 { x }",
        "fn f(x: u8, x: u8) -> u8 { x }",
        "fn f(x: u8) -> u8 { missing(x) }",
        "fn f() -> u8 { x } block capture(clk, x: u8) -> u8 { f() }",
        "fn f() -> u8 { let _ = f(); u8:0 }",
        "fn f() -> u8 { g() } fn g() -> u8 { f() }",
        "fn f() -> u8 { u8:0 } extern fn f() -> u8;",
        "fn f() -> u8 { u8:0 } block f(clk) -> u8 { u8:1 }",
        "block child(clk) -> u8 { u8:1 } fn f() -> u8 { let x = child(); x }",
        "fn f() -> u8 { reg q: u8; q := u8:0; q }",
        "fn f(x: s8) -> s8 { x } block capture(clk, x: u8) -> s8 { f(x) }",
        "fn f(x: u8) -> u8 { x } block capture(clk, x: u8) -> u8 { f(x, x) }",
    ] {
        assert!(
            parse_module(source)
                .and_then(|module| check_module(&module))
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn function_identifiers_match_dslx() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn offset!(x': u8) -> u8 {
    let reg = x' + u8:1;
    let self = reg;
    self
}
fn block(x: u8) -> u8 { offset!(x) }
"#;
    let bx = format!(
        "{helpers}\n#[allow(combinational_block)]\nblock sample(clk, x: u8) -> u8 {{ block(x) }}"
    );
    let package = compile(&bx, empty_package("identifiers"), ResetBehavior::default())?;
    let block = package.get_top_block().ok_or("missing sample block")?;
    let function = combinational_block_to_fn(block)?;
    let actual = eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(8, 2)?]);
    let expected = native_call(helpers, "block(u8:2)")?;
    assert_eq!(actual, value_from_libxls(&expected, &actual.type_())?);

    for invalid in [
        "fn match() {}",
        "fn bad(self: u8) {}",
        "fn bad(in: u8) {}",
        "fn bad() { let out = u8:1; }",
        "fn _(x: u8) -> u8 { x } fn bad() -> u8 { _(u8:1) }",
    ] {
        assert!(parse_module(invalid).is_err(), "{invalid}");
        assert!(
            dslx::parse_and_typecheck(invalid, "names.x", "names", &mut ImportData::default())
                .is_err(),
            "{invalid}"
        );
    }
    Ok(())
}

#[test]
fn zero_calls_resolve_in_source_order_and_obey_shadowing() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn before() -> u8 { zero!<u8>() }
fn zero!() -> u8 { u8:7 }
fn after() -> (u8, u8) { (before(), zero!<u8>()) }
"#;
    let bx = format!(
        "{helpers}\n#[allow(combinational_block)]\nblock sample(clk) -> (u8, u8) {{ after() }}"
    );
    let package = compile(&bx, empty_package("design"), ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("missing sample block")?;
    let actual = eval_fn_in_package_value(&package, &combinational_block_to_fn(top)?, &[]);
    assert_eq!(actual, IrValue::parse_typed("(bits[8]:0, bits[8]:7)")?);
    assert_eq!(
        actual,
        value_from_libxls(&native_call(helpers, "after()")?, &actual.type_())?
    );
    for invalid in [
        "fn f(x: u8) -> u8 { let zero! = x; zero!<u8>() }",
        "fn f(x: u8) -> u8 { later(x) } fn later(x: u8) -> u8 { x }",
        "fn f(x: u8, y: u8) -> bool { x!=y }",
    ] {
        assert!(
            parse_module(invalid)
                .and_then(|m| check_module(&m))
                .is_err(),
            "{invalid}"
        );
        assert!(
            dslx::parse_and_typecheck(invalid, "names.x", "names", &mut ImportData::default())
                .is_err(),
            "{invalid}"
        );
    }
    Ok(())
}

#[test]
fn block_tuple_patterns_parse_dslx_singleton_annotations_recursively() -> Result<(), Box<dyn Error>>
{
    let cases = [
        "let (x): (u8) = (u8:7,); x",
        "let ((x), _): ((u8), u8) = ((u8:7,), u8:0); x",
        "let (xs,): ((u8)[2]) = ([(u8:3,), (u8:7,)],); xs[u32:1].0",
        "if flag { let (x): (u8) = (u8:7,); x } else { u8:7 }",
    ];
    for body in cases {
        let bx = format!("block sample(clk, flag: bool) -> u8 {{ {body} }}");
        let dslx = format!("fn sample(flag: bool) -> u8 {{ {body} }}");
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&dslx, &[]),
        )?;
        assert_eq!(
            evaluator.compare(&[IrValue::bool(true)])?,
            IrValue::make_ubits(8, 7)?
        );
    }
    // Existing BX type contexts treat a single parenthesized type as grouping.
    let grouped = r#"block sample(clk, flag: bool) -> (u8) {
        let x: (u8) = u8:7;
        match flag { _ => { let (out, y): ((u8), u8) = (x, u8:0); out + y } }
    }"#;
    check_module(&parse_module(grouped)?)?;
    Ok(())
}

#[test]
fn singleton_type_annotations_preserve_the_tuple_at_call_sites() -> Result<(), Box<dyn Error>> {
    for spelling in ["(u8)", "(u8,)"] {
        let helpers = format!("fn echo(x: {spelling}) -> {spelling} {{ x }}");
        let bx = format!("{helpers}\nblock sample(clk, x: u8) -> (u8,) {{ echo((x,)) }}");
        let dslx = format!("{helpers}\nfn sample(x: u8) -> (u8,) {{ echo((x,)) }}");
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&dslx, &[]),
        )?;
        let x = IrValue::make_ubits(8, 13)?;
        assert_eq!(
            evaluator.compare(std::slice::from_ref(&x))?,
            IrValue::make_tuple(&[x])
        );
    }
    Ok(())
}

#[test]
fn scoped_tuple_bindings_preserve_outer_and_branch_values() -> Result<(), Box<dyn Error>> {
    let helper = r#"
fn examine(payload: (u8, (u8, u8)), swap: bool) -> (u8, u8, u8, u8) {
    let (x, (y, _)): (u8, (u8, u8)) = payload;
    let saved = { let x = x + u8:1; x };
    let (first, second) = if swap {
        let (x, y) = (y, x);
        (x + saved, y)
    } else {
        let x = y - x;
        (x, y)
    };
    (first, second, x, saved)
}
"#;
    let params = "payload: (u8, (u8, u8)), swap: bool";
    let result = "(u8, u8, u8, u8)";
    let bx =
        format!("{helper}\nblock sample(clk, {params}) -> {result} {{ examine(payload, swap) }}");
    let dslx = format!("{helper}\nfn sample({params}) -> {result} {{ examine(payload, swap) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (payload, swap, expected) in [
        (
            "(bits[8]:250, (bits[8]:9, bits[8]:4))",
            true,
            "(bits[8]:4, bits[8]:250, bits[8]:250, bits[8]:251)",
        ),
        (
            "(bits[8]:250, (bits[8]:9, bits[8]:4))",
            false,
            "(bits[8]:15, bits[8]:9, bits[8]:250, bits[8]:251)",
        ),
        (
            "(bits[8]:0, (bits[8]:255, bits[8]:37))",
            true,
            "(bits[8]:0, bits[8]:0, bits[8]:0, bits[8]:1)",
        ),
    ] {
        assert_eq!(
            evaluator.compare(&[IrValue::parse_typed(payload)?, IrValue::bool(swap)])?,
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn tuple_pattern_arity_and_lexical_scope_match_dslx() {
    for body in [
        "let (a, b) = (x,); a",
        "let (a, a) = (x, x); a",
        "let y = { let z = x; z }; z",
        "if x == u8:0 { let y = x; y } else { y }",
    ] {
        let helper = format!("fn helper(x: u8) -> u8 {{ {body} }}");
        let bx = format!("{helper}\nblock sample(clk, x: u8) -> u8 {{ helper(x) }}");
        let dslx = format!("{helper}\nfn sample(x: u8) -> u8 {{ helper(x) }}");
        assert!(
            parse_module(&bx)
                .and_then(|module| check_module(&module))
                .is_err(),
            "{body}"
        );
        assert!(compile_dslx_module(&dslx, &[]).is_err(), "{body}");
    }
}

#[test]
fn braced_function_expressions_follow_dslx_operand_syntax() -> Result<(), Box<dyn Error>> {
    let helper = r#"
fn id(x: u8) -> u8 { x }
fn pick(pair: (u8, u8), flag: bool) -> u8 {
    if { flag } { ({pair}).0 + ({u8:1}) } else { id({pair.1}) }
}"#;
    let bx = format!(
        "{helper}\nblock sample(clk, pair: (u8, u8), flag: bool) -> u8 {{ pick(pair, flag) }}"
    );
    let dslx =
        format!("{helper}\nfn sample(pair: (u8, u8), flag: bool) -> u8 {{ pick(pair, flag) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (flag, expected) in [(true, 0), (false, 37)] {
        let pair = IrValue::parse_typed("(bits[8]:255, bits[8]:37)")?;
        assert_eq!(
            evaluator.compare(&[pair, IrValue::bool(flag)])?,
            IrValue::make_ubits(8, expected)?
        );
    }
    for expression in [
        "{x} + x",
        "x + {x}",
        "!{x}",
        "{x} as u8",
        "{(x, x)}.0",
        "{[x]}[0]",
    ] {
        let source = format!("fn sample(x: u8) -> u8 {{ {expression} }}");
        assert!(parse_module(&source).is_err(), "{expression}");
        assert!(
            dslx::parse_and_typecheck(&source, "braces.x", "braces", &mut ImportData::default())
                .is_err(),
            "{expression}"
        );
    }
    Ok(())
}

#[test]
fn postfix_operations_on_function_conditionals_match_dslx() -> Result<(), Box<dyn Error>> {
    for (branch, postfix) in [("(x,)", ".0"), ("[x]", "[0]"), ("x", "[0+:u8]")] {
        let conditional = format!("if flag {{ {branch} }} else {{ {branch} }}");
        for prefix in ["", "x + "] {
            for (subject, valid) in [
                (conditional.clone(), false),
                (format!("({conditional})"), true),
            ] {
                let source =
                    format!("fn sample(x: u8, flag: bool) -> u8 {{ {prefix}{subject}{postfix} }}");
                assert_eq!(parse_module(&source).is_ok(), valid, "{source}");
                assert_eq!(
                    dslx::parse_and_typecheck(
                        &source,
                        "postfix.x",
                        "postfix",
                        &mut ImportData::default()
                    )
                    .is_ok(),
                    valid,
                    "{source}"
                );
            }
        }
        if postfix != "[0+:u8]" {
            let block =
                format!("block sample(clk, x: u8, flag: bool) -> u8 {{ {conditional}{postfix} }}");
            check_module(&parse_module(&block)?)?;
        }
    }
    Ok(())
}

#[test]
fn function_comments_follow_dslx_lexing() -> Result<(), Box<dyn Error>> {
    for (source, valid) in [
        ("fn sample(x: u8) -> u8 { // value\n x }", true),
        ("fn sample(x: u8) -> u8 { /* value */ x }", false),
        ("fn /* name */ sample(x: u8) -> u8 { x }", false),
    ] {
        assert_eq!(parse_module(source).is_ok(), valid, "{source}");
        assert_eq!(
            dslx::parse_and_typecheck(source, "comments.x", "comments", &mut ImportData::default())
                .is_ok(),
            valid,
            "{source}"
        );
    }
    let source = "fn id(x: u8) -> u8 { x } block sample(clk, x: u8) -> u8 { /* input */ id(x) }";
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn an_unroll_starting_a_function_expression_ends_at_its_seed() -> Result<(), Box<dyn Error>> {
    let fold = "unroll_for! (i, a): (u32, u8) in u32:0..u32:1 { a + x }(u8:0)";
    for expression in [
        fold.to_owned(),
        format!("x + {fold}"),
        format!("!{fold}"),
        format!("({fold}) + x"),
        format!("({fold}) as u16 as u8"),
    ] {
        let source = format!("fn sample(x: u8) -> u8 {{ {expression} }}");
        check_module(&parse_module(&source)?)?;
        dslx::parse_and_typecheck(&source, "fold.x", "fold", &mut ImportData::default())?;
    }
    for expression in [format!("{fold} + x"), format!("{fold} as u8")] {
        let source = format!("fn sample(x: u8) -> u8 {{ {expression} }}");
        assert!(parse_module(&source).is_err(), "{expression}");
        assert!(
            dslx::parse_and_typecheck(&source, "fold.x", "fold", &mut ImportData::default())
                .is_err(),
            "{expression}"
        );
    }
    Ok(())
}
