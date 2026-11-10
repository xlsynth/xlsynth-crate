// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use xlsynth::dslx::{self, ImportData};
use xlsynth_bx::{ImportOptions, check_module, parse_module};
use xlsynth_pir::IrValue;

use super::directed_expr_eval::{compile_dslx, compile_dslx_module};
use super::expr_eval::ExpressionEvaluator;

#[test]
fn tuple_match_can_dispatch_on_rounding_flags() -> Result<(), Box<dyn Error>> {
    let body = r#"
        match (guard, sticky) {
            (true, true) => true,
            (true, false) => odd,
            (false, _) => false,
        }
    "#;
    let eval = ExpressionEvaluator::compile(
        "guard: bool, sticky: bool, odd: bool",
        "bool",
        body,
        compile_dslx,
    )?;
    for flags in 0..8u64 {
        let args = (0..3)
            .map(|i| IrValue::make_ubits(1, (flags >> i) & 1))
            .collect::<Result<Vec<_>, _>>()?;
        let increment = flags & 1 != 0 && flags & 6 != 0;
        assert_eq!(
            eval.compare(&args)?,
            IrValue::make_ubits(1, u64::from(increment))?
        );
    }
    Ok(())
}

#[test]
fn tuple_match_coverage_agrees_with_dslx() {
    for (arms, valid) in [
        ("(true, _) => u8:1, (false, _) => u8:2", true),
        (
            "(true, _) => u8:1, (false, true) | (false, false) => u8:2",
            true,
        ),
        ("(true, _) => u8:1, (_, false) => u8:2, _ => u8:3", true),
        ("(true, _) => u8:1, (false, true) => u8:2", false),
        ("(true, _) => u8:1, (true, false) => u8:2, _ => u8:3", false),
        ("(true, _) => u8:1, (true, _) => u8:2, _ => u8:3", false),
        ("(true,) => u8:1, _ => u8:2", false),
        ("(true, u2:0) => u8:1, _ => u8:2", false),
        (
            "(true, u1:1..u1:1) => u8:1, (true, u1:1..u1:1) => u8:2, _ => u8:3",
            false,
        ),
        (
            "(true, u1:1..u1:1) => u8:1, (true, uN[1]:1..uN[1]:1) => u8:2, _ => u8:3",
            true,
        ),
    ] {
        let source = format!("fn f(x: (bool, bool)) -> u8 {{ match x {{ {arms} }} }}");
        let bx = parse_module(&source).and_then(|module| check_module(&module));
        let native = dslx::parse_and_typecheck(
            &source,
            "tuple_match.x",
            "tuple_match",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{source}", native.err());
        assert_eq!(bx.is_ok(), valid, "BX {:?}\n{source}", bx.err());
    }
}

#[test]
fn tuple_pattern_literals_use_the_type_of_a_call_result_field() {
    let source = r#"
        fn key(x: s8) -> s8 { x }
        fn f(x: s8) -> u8 {
            match (key(x), false) {
                (0, true) => u8:1,
                (0xff, false) => u8:2,
                _ => u8:0,
            }
        }
        fn g(x: s8, choose: bool) -> u8 {
            match (match choose { true => key(x), false => x }, false) {
                (0, true) => u8:1,
                (0xff, false) => u8:2,
                _ => u8:0,
            }
        }
        fn h(x: s8, outer: bool, inner: bool) -> u8 {
            match if outer {
                (match inner { true => key(x), false => 0xff }, outer)
            } else {
                (0, outer)
            } {
                (0, false) => u8:1,
                _ => u8:0,
            }
        }
    "#;
    dslx::parse_and_typecheck(
        source,
        "tuple_match.x",
        "tuple_match",
        &mut ImportData::default(),
    )
    .unwrap();
    check_module(&parse_module(source).unwrap()).unwrap();
}

#[test]
fn tuple_patterns_can_supply_field_widths_through_branches() -> Result<(), Box<dyn Error>> {
    // The pinned DSLX converter emits the minimum subject width for the
    // unsized spelling even after typechecking against the wider pattern.
    for subject in [
        "(1, x)",
        "if x { (2, x) } else { (1, x) }",
        "(match x { true => (2, x), false => (1, x) })",
    ] {
        let bx = format!("match {subject} {{ (u8:1, false) => u8:1, _ => u8:0 }}");
        let eval = ExpressionEvaluator::compile("x: bool", "u8", &bx, |params, result, body| {
            let explicit = body
                .replace("(1, x)", "(u8:1, x)")
                .replace("(2, x)", "(u8:2, x)");
            compile_dslx(params, result, &explicit)
        })?;
        for flag in 0..=1 {
            assert_eq!(
                eval.compare(&[IrValue::make_ubits(1, flag)?])?,
                IrValue::make_ubits(8, 1 - flag)?
            );
        }
    }
    Ok(())
}

#[test]
fn tuple_match_preserves_the_dslx_default_for_enum_encodings() -> Result<(), Box<dyn Error>> {
    let source = r#"
        enum State: u2 { A = 1, B = 2 }
        fn classify(x: (State, bool)) -> u8 {
            match x {
                (State::A, _) => u8:1,
                (State::B, _) => u8:2,
                (_, false) => u8:3,
            }
        }
    "#;
    let dslx = format!("{source} fn sample(x: (State, bool)) -> u8 {{ classify(x) }}");
    let bx = format!(
        "{source} #[allow(combinational_block)] block sample(clk, x: (State, bool)) -> u8 {{ classify(x) }}"
    );
    let eval = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (state, result) in [(0, 3), (1, 1), (2, 2), (3, 3)] {
        let value =
            IrValue::make_tuple(&[IrValue::make_ubits(2, state)?, IrValue::make_ubits(1, 1)?]);
        assert_eq!(eval.compare(&[value])?, IrValue::make_ubits(8, result)?);
    }
    Ok(())
}

#[test]
fn tuple_match_in_an_empty_enum_domain_distinguishes_source_spellings() {
    for (first, second, valid) in [
        ("(_, true)", "(_, true,)", false),
        ("(_, true)", "(_, u1:1)", true),
        ("(_, u1:1)", "(_, uN[1]:1)", true),
        ("(_, true)", "(_, true..=true)", true),
    ] {
        let source = format!(
            "enum Empty: u2 {{}} fn f(x: (Empty, bool)) -> u8 {{ match x {{
                {first} => u8:1, {second} => u8:2, _ => u8:3
            }} }}"
        );
        let bx = parse_module(&source).and_then(|module| check_module(&module));
        let native = dslx::parse_and_typecheck(
            &source,
            "tuple_match.x",
            "tuple_match",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "DSLX {:?}\n{source}", native.err());
        assert_eq!(bx.is_ok(), valid, "BX {:?}\n{source}", bx.err());
    }
}

#[test]
fn tuple_match_accounts_for_empty_domains_inside_wildcard_fields() -> Result<(), Box<dyn Error>> {
    let empty = "enum Empty: u2 {}";
    let encoding = IrValue::make_ubits(2, 1)?;
    for (decl, ty, field) in [
        (
            "",
            "Empty[1]",
            IrValue::make_array(std::slice::from_ref(&encoding))?,
        ),
        (
            "struct Wrap { empty: Empty }",
            "Wrap",
            IrValue::make_tuple(&[encoding]),
        ),
    ] {
        let body = "match x { (_, true) => u8:9 }";
        let dslx = format!("{empty} {decl} fn sample(x: ({ty}, bool)) -> u8 {{ {body} }}");
        let bx = format!(
            "{empty} {decl} #[allow(combinational_block)] block sample(clk, x: ({ty}, bool)) -> u8 {{ {body} }}"
        );
        let eval = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&dslx, &[]),
        )?;
        for flag in 0..=1 {
            let x = IrValue::make_tuple(&[field.clone(), IrValue::make_ubits(1, flag)?]);
            assert_eq!(eval.compare(&[x])?, IrValue::make_ubits(8, 9)?);
        }
    }

    let source = format!(
        "{empty} fn sample(x: (Empty[0], bool)) -> u8 {{ match x {{ (_, true) => u8:9 }} }}"
    );
    assert!(check_module(&parse_module(&source)?).is_err());
    assert!(
        dslx::parse_and_typecheck(
            &source,
            "tuple_match.x",
            "tuple_match",
            &mut ImportData::default(),
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn tuple_match_selects_a_constant_slice_width() -> Result<(), Box<dyn Error>> {
    let body = r#"
        x[0+:uN[match (true, (s8:-2, ())) {
            (true, (s8:-3..=s8:0, ())) => u32:3,
            (true, (s8:-2..=s8:2, ())) => u32:5,
            _ => u32:7,
        }]]
    "#;
    let eval = ExpressionEvaluator::compile("x: u8", "u3", body, compile_dslx)?;
    assert_eq!(
        eval.compare(&[IrValue::make_ubits(8, 0b10111010)?])?,
        IrValue::make_ubits(3, 2)?
    );
    Ok(())
}
