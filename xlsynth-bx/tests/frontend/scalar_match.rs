// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::{compile_dslx, compile_dslx_module};
use super::expr_eval::ExpressionEvaluator;
use std::error::Error;
use xlsynth::dslx::{self, ImportData};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::NodePayload;

#[test]
fn match_shares_the_subject_and_infers_scoped_tuple_results() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn key(x: sN[96]) -> sN[96] { x }
fn classify(x: sN[96]) -> (u8, u16) {
    let result = match key(x) {
        sN[96]:-2..=sN[96]:0 => { let n = u8:1; (n, 255 + 1) },
        sN[96]:0..=sN[96]:2 => { let n = u16:7; (2, n) },
        _ => (3, u16:9),
    };
    result
}"#;
    let body = "match key(x) { sN[96]:-3 | sN[96]:3 => (u8:4, u16:5), _ => classify(x) }";
    let bx = format!("{helpers}\nblock sample(clk, x: sN[96]) -> (u8, u16) {{ {body} }}");
    let reference = format!("{helpers}\nfn sample(x: sN[96]) -> (u8, u16) {{ {body} }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&reference, &[]),
    )?;
    for (x, a, b) in [
        (-4, 3, 9),
        (-3, 4, 5),
        (-2, 1, 256),
        (0, 1, 256),
        (1, 2, 7),
        (2, 2, 7),
        (3, 4, 5),
        (4, 3, 9),
    ] {
        let expected =
            IrValue::make_tuple(&[IrValue::make_ubits(8, a)?, IrValue::make_ubits(16, b)?]);
        assert_eq!(evaluator.compare(&[IrValue::make_sbits(96, x)?])?, expected);
    }
    let package = compile(
        &bx,
        empty_package("match_subject"),
        ResetBehavior::default(),
    )?;
    for nodes in [
        &package.get_fn("classify").unwrap().nodes,
        &package.get_block("sample").unwrap().nodes,
    ] {
        assert_eq!(nodes.iter().filter(|n| matches!(&n.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "key")).count(), 1);
    }
    Ok(())
}

#[test]
fn match_subject_width_and_coverage_agree_with_native_dslx() -> Result<(), Box<dyn Error>> {
    for (subject, arms, valid) in [
        ("1", "uN[96]:1 => 7, _ => u8:9", true),
        ("flag", "true => 7, false => u8:9", true),
        ("flag", "true => u8:7", false),
        ("x", "0..2 => u8:1, 2..=3 => u8:2", true),
        ("x", "0..2 => u8:1, 3 => u8:2", false),
        ("x", "0 | u2:1 => u8:1, 1 => u8:2, _ => u8:3", false),
        ("x", "0..=2 => u8:1, 2 => u8:2, _ => u8:3", false),
        ("x", "0..=2 => u8:1, 2..=3 => u8:2, _ => u8:3", true),
        ("uN[0]:0", "uN[0]:0 => u8:1", true),
    ] {
        let source =
            format!("fn check_match(flag: bool, x: u2) -> u8 {{ match {subject} {{ {arms} }} }}");
        let bx = parse_module(&source).and_then(|m| check_module(&m));
        let native = dslx::parse_and_typecheck(
            &source,
            "match_test.x",
            "match_test",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{source}", native.err());
        assert_eq!(bx.is_ok(), valid, "BX {:?}\n{source}", bx.err());
    }
    Ok(())
}

#[test]
fn match_patterns_infer_signed_widths_in_source_order() {
    for (subject, patterns, valid) in [
        ("x", "0 | 0x7", true),
        ("key(x)", "0x7", true),
        ("key(x)", "0 | 0x7", false),
        ("key(x)", "0..=0x7", false),
        ("key(x)", "0x7 | s3:2", true),
        ("key(x)", "0 | s3:2 | 0x7", true),
        ("key(x)", "0 | 0x7 | s3:2", false),
        ("!key(x)", "0 | 0x7", false),
        ("key(x) + 0", "0 | 0x7", false),
        ("if true { key(x) } else { key(x) }", "0 | 0x7", false),
        ("key(x) + s3:0", "0 | 0x7", true),
        ("key(x) as s3", "0 | 0x7", true),
        ("(key(x),).0", "0 | 0x7", true),
        ("if true { x } else { key(x) }", "0 | 0x7", true),
        ("0", "s3:2 | 0x7", true),
        ("0", "0x7 | s3:2", false),
    ] {
        let source = format!(
            "fn key(x: s3) -> s3 {{ x }} fn f(x: s3) -> u8 {{ match {subject} {{ {patterns} => u8:1, _ => u8:2 }} }}"
        );
        let bx = parse_module(&source).and_then(|m| check_module(&m));
        let native = dslx::parse_and_typecheck(
            &source,
            "match_test.x",
            "match_test",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{source}", native.err());
        assert_eq!(bx.is_ok(), valid, "BX {:?}\n{source}", bx.err());
    }
}

#[test]
fn explicitly_signed_negative_patterns_negate_at_the_declared_width() -> Result<(), Box<dyn Error>>
{
    let expr = "match x { s3:-5 => s3:-7, _ => s3:0 }";
    let evaluator = ExpressionEvaluator::compile("x: s3", "s3", expr, compile_dslx)?;
    for x in -4..=3 {
        let args = [IrValue::make_sbits(3, x)?];
        assert_eq!(
            evaluator.compare(&args)?,
            IrValue::make_sbits(3, i64::from(x == 3))?
        );
    }
    for pattern in ["-5", "s3:-8"] {
        let source = format!("fn f(x:s3)->s3 {{ match x {{ {pattern} => s3:1, _ => s3:0 }} }}");
        let bx = parse_module(&source).and_then(|m| check_module(&m));
        let native = dslx::parse_and_typecheck(
            &source,
            "match_test.x",
            "match_test",
            &mut ImportData::default(),
        );
        assert!(native.is_err(), "native accepted {source}");
        assert!(bx.is_err(), "BX accepted {source}");
    }
    Ok(())
}

#[test]
fn match_subject_distinguishes_the_zero_intrinsic_from_a_call() -> Result<(), Box<dyn Error>> {
    let local = "fn zero!() -> s1 { s1:0 }";
    for (declaration, subject, valid) in [
        ("", "zero!<s1>()", true),
        (local, "zero!<s1>()", false),
        (local, "!zero!<s1>()", false),
        (local, "zero!<s1>() as s1", true),
    ] {
        let body = format!("match {subject} {{ 0 | 0x1 => u8:1, _ => u8:2 }}");
        let dslx = format!("{declaration} fn f() -> u8 {{ {body} }}");
        let native = dslx::parse_and_typecheck(
            &dslx,
            "match_test.x",
            "match_test",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{dslx}", native.err());
        for source in [
            dslx,
            format!("{declaration} block b(clk) -> u8 {{ {body} }}"),
        ] {
            let bx = check_module(&parse_module(&source)?);
            assert_eq!(bx.is_ok(), valid, "BX {:?}\n{source}", bx.err());
        }
    }
    Ok(())
}

#[test]
fn match_result_widths_include_scoped_literals_across_call_arms() {
    for (anchor, valid) in [("anchor()", false), ("anchor() + 0", false), ("s3:3", true)] {
        let source = format!(
            r#"
fn anchor() -> s3 {{ s3:3 }}
fn f(x: u2) -> s3 {{
    let result = match x {{
        0 => 1,
        1 => {anchor},
        _ => {{ let raw = 0x7; raw }},
    }};
    result
}}"#
        );
        let native = dslx::parse_and_typecheck(
            &source,
            "match_test.x",
            "match_test",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{source}", native.err());
        let block = source.replace("fn f(x: u2)", "block f(clk, x: u2)");
        for program in [&source, &block] {
            let bx = parse_module(program).and_then(|m| check_module(&m));
            assert_eq!(bx.is_ok(), valid, "BX {:?}\n{program}", bx.err());
        }
    }
}

#[test]
fn match_constant_evaluation_selects_the_first_matching_reset_arm() -> Result<(), Box<dyn Error>> {
    let expr = "match x { u4:0..=u4:2 => u8:7, u4:2..u4:4 => u8:9, _ => u8:5 }";
    let evaluator = ExpressionEvaluator::compile("x: u4", "u8", expr, compile_dslx)?;
    for (x, result) in [(1, 7), (2, 7), (3, 9), (4, 5)] {
        let args = [IrValue::make_ubits(4, x)?];
        let expected = IrValue::make_ubits(8, result)?;
        assert_eq!(evaluator.compare(&args)?, expected);
        evaluator.compare_reset(&args, "u8", "", expr, &expected)?;
    }
    Ok(())
}

#[test]
fn inferred_match_results_apply_call_widths_after_literal_unification() -> Result<(), Box<dyn Error>>
{
    let helpers = "fn key(x:s3)->s3{x} fn pair_arg(x:(s3,))->s3{x.0}";
    let dispatch = "match k { 0 => 0x7, 1 => key(x), _ => 0 }";
    let tuples = "match k { 0 => (0x7,), 1 => (key(x),), _ => (0,) }";
    for (body, valid) in [
        (dispatch.to_owned(), true),
        (format!("let v: s3 = {dispatch}; v"), true),
        (format!("let v = {dispatch}; v"), false),
        (format!("key(!({dispatch}))"), false),
        (format!("({dispatch},).0"), false),
        (format!("let v = x + ({dispatch}); v"), true),
        (format!("let v = key(x) + ({dispatch}); v"), false),
        (format!("let v = ({dispatch}) + x; v"), false),
        (format!("[x, {dispatch}][0]"), true),
        (format!("[key(x), {dispatch}][0]"), false),
        (format!("key(if true {{ x }} else {{ {dispatch} }})"), true),
        (
            format!("key(if true {{ key(x) }} else {{ {dispatch} }})"),
            false,
        ),
        (format!("let v = {tuples}; v.0"), true),
        (format!("pair_arg({tuples})"), false),
        ("let v = match k { 0 => 0x7, _ => key(x) }; v".into(), true),
    ] {
        let source = format!("{helpers} fn f(x: s3, k: u2) -> s3 {{ {body} }}");
        let native = dslx::parse_and_typecheck(
            &source,
            "match_test.x",
            "match_test",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{source}", native.err());
        for source in [
            source,
            format!("{helpers} block b(clk, x: s3, k: u2) -> s3 {{ {body} }}"),
        ] {
            let bx = check_module(&parse_module(&source)?);
            assert_eq!(bx.is_ok(), valid, "BX {:?}\n{source}", bx.err());
        }
    }
    Ok(())
}
