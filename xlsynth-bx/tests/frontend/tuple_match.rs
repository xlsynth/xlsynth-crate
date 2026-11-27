// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use xlsynth::dslx::{self, ImportData};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::NodePayload;

use super::directed_expr_eval::{compile_dslx, compile_dslx_module};
use super::expr_eval::ExpressionEvaluator;

#[test]
fn tuple_match_shares_a_call_subject_across_projected_arm_bindings() -> Result<(), Box<dyn Error>> {
    let key = r#"
        fn key(first: bool, x: u8) -> (bool, (u8[2], u8)) {
            (first, ([x, x + u8:1], u8:3))
        }
    "#;
    let body = r#"
        match key(first, x) {
            (true, (lanes, bias)) => {
                let bumped = map(lanes, |element| element + bias);
                bumped[u32:0] + bias
            },
            (false, (lanes, bias)) => {
                for (i, acc) in u32:0..u32:2 { acc + lanes[i] }(bias)
            },
        }
    "#;
    let functions = format!("{key} fn classify(first: bool, x: u8) -> u8 {{ {body} }}");
    let dslx = format!("{functions} fn sample(first: bool, x: u8) -> u8 {{ classify(first, x) }}");
    for block_body in [body, "classify(first, x)"] {
        let bx = format!(
            "{functions} #[allow(combinational_block)] block sample(clk, first: bool, x: u8) -> u8 {{ {block_body} }}"
        );
        let eval = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&dslx, &[]),
        )?;
        for (first, expected) in [(true, 16), (false, 24)] {
            assert_eq!(
                eval.compare(&[IrValue::bool(first), IrValue::make_ubits(8, 10)?])?,
                IrValue::make_ubits(8, expected)?,
            );
        }
        let package = compile(
            &bx,
            empty_package("match_bindings"),
            ResetBehavior::default(),
        )?;
        let nodes = if block_body == body {
            &package.get_block("sample").unwrap().nodes
        } else {
            &package.get_fn("classify").unwrap().nodes
        };
        let subject_calls = nodes.iter().filter(|node| {
            matches!(&node.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "key")
        });
        assert_eq!(subject_calls.count(), 1);
    }
    Ok(())
}

#[test]
fn tuple_match_bindings_fold_on_the_selected_arm() -> Result<(), Box<dyn Error>> {
    let helper = r#"
        fn read_at(i: u32) -> bool { [true][i] }
        fn check(p: (bool, (u32, u32))) -> bool {
            match p {
                (true, (i, _)) => { let read = read_at(i); read },
                (false, (_, i)) => read_at(i),
            }
        }
    "#;
    for (flag, left, right, valid) in [
        (true, 0, 5, true),
        (false, 5, 0, true),
        (true, 5, 0, false),
        (false, 0, 5, false),
    ] {
        let source = format!("{helper} const_assert!(check(({flag}, (u32:{left}, u32:{right}))));");
        let native = dslx::parse_and_typecheck(
            &source,
            "tuple_assert.x",
            "tuple_assert",
            &mut ImportData::default(),
        );
        assert_eq!(native.is_ok(), valid, "{source}\nDSLX {:?}", native.err());
        let bx = check_module(&parse_module(&source)?);
        assert_eq!(bx.is_ok(), valid, "{source}\nBX {:?}", bx.err());
    }
    let runtime = r#"
        fn check(p: (bool, bool)) -> bool {
            match p { (true, b) => { const_assert!(b); b }, (false, _) => false }
        }
    "#;
    assert!(check_module(&parse_module(runtime)?).is_err());
    let literal = r#"
        fn check() -> bool {
            match (true, true) {
                (true, b) => { const_assert!(b); const if b { true } else { false } },
                (false, _) => false,
            }
        }
        const_assert!(check());
    "#;
    check_module(&parse_module(literal)?)?;
    Ok(())
}

#[test]
fn tuple_match_binders_retain_unsized_field_constraints() -> Result<(), Box<dyn Error>> {
    for (subject, explicit_subject, arms, result, expected) in [
        ("(1,)", "(u8:1,)", "(v,) => v", "u8", 1),
        (
            "(b, 0xff)",
            "(b, s8:-1)",
            "(true,v) => v, _ => s8:0",
            "s8",
            255,
        ),
        (
            "(b, 1)",
            "(b, u8:1)",
            "(false,128) => u8:0, (_,v) => v",
            "u8",
            1,
        ),
        (
            "(b, (1, 2))",
            "(b, (u3:1, u8:2))",
            "(false, (u3:0, _)) => u8:0, (_, pair) => pair.1",
            "u8",
            2,
        ),
    ] {
        let body = format!("match {subject} {{ {arms} }}");
        let source = format!("fn sample(b: bool) -> {result} {{ {body} }}");
        dslx::parse_and_typecheck(
            &source,
            "tuple_width.x",
            "tuple_width",
            &mut ImportData::default(),
        )?;
        // The pinned DSLX converter materializes unsized subject fields at
        // their minimum width even after typechecking a wider match result.
        let reference = format!("match {explicit_subject} {{ {arms} }}");
        let bx = super::expr_eval::bx_source("b: bool", result, &body);
        let eval = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx("b: bool", result, &reference),
        )?;
        assert_eq!(
            eval.compare(&[IrValue::bool(true)])?,
            IrValue::make_ubits(8, expected)?
        );
    }
    for (subject, first_pattern, result) in [
        ("1", "u3:1", "u8"),
        ("1", "KEY", "u8"),
        ("1", "u3:1..u3:3", "u8"),
        ("1", "128", "u7"),
        ("0xff", "255", "s8"),
        ("u2:1", "1", "u8"),
    ] {
        let source = format!(
            "const KEY = u3:1; fn f(b: bool) -> {result} {{ match (b, {subject}) {{
                (false, {first_pattern}) => {result}:0, (_,v) => v
            }} }}"
        );
        assert!(
            dslx::parse_and_typecheck(
                &source,
                "tuple_width.x",
                "tuple_width",
                &mut ImportData::default(),
            )
            .is_err(),
            "{source}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    Ok(())
}

#[test]
fn tuple_match_bindings_respect_module_names_and_builtin_names() -> Result<(), String> {
    let source = r#"
const TAG = u8:3;
fn pick(p: (u8, u8)) -> u8 {
    match p { (TAG, value) => value, (_, _) => u8:0 }
}
const_assert!(pick((u8:3, u8:9)) == u8:9);
const_assert!(pick((u8:4, u8:9)) == u8:0);
"#;
    check_module(&parse_module(source)?)?;
    let parametric = r#"
struct R<T: type> { val: uN[match (u32:4, true) { (T, _) => T }] }
fn read(x: R<u8>) -> u4 { x.val }
"#;
    assert!(
        parse_module(parametric)
            .and_then(|m| check_module(&m))
            .is_err()
    );
    let port = r#"
#[allow(combinational_block)]
block route(x: u32, data: uN[match (u32:3, true) { (x, _) => x }]) -> u3 { data }
"#;
    assert!(parse_module(port).and_then(|m| check_module(&m)).is_err());
    for name in ["self", "Source", "rev"] {
        let source = format!("fn f(p: (u8,)) -> u8 {{ match p {{ ({name},) => {name} }} }}");
        assert!(
            parse_module(&source)
                .and_then(|m| check_module(&m))
                .is_err(),
            "{name}"
        );
    }
    Ok(())
}

#[test]
fn tuple_match_rest_and_whole_subject_agree_with_dslx() -> Result<(), Box<dyn Error>> {
    let params = "x: u8, y: u8, z: u8";
    let body = r#"
        match (x, true, z, (y, z)) {
            (u8:4, .., (u8:4, last)) => last,
            whole => whole.0,
        }
    "#;
    let eval = ExpressionEvaluator::compile(params, "u8", body, compile_dslx)?;
    for (x, y, z, expected) in [(4, 4, 9, 9), (4, 5, 9, 4), (7, 7, 0, 7)] {
        let args = [
            IrValue::make_ubits(8, x)?,
            IrValue::make_ubits(8, y)?,
            IrValue::make_ubits(8, z)?,
        ];
        assert_eq!(eval.compare(&args)?, IrValue::make_ubits(8, expected)?);
    }
    Ok(())
}

#[test]
fn tuple_match_constant_references_and_whole_bindings_fold_and_lower() -> Result<(), Box<dyn Error>>
{
    let functions = r#"
        const TAG = u8:3;
        fn both_tag(left: u8, right: u8) -> bool {
            match (left, right) { (TAG, .., TAG) => true, _ => false }
        }
        fn scalar(x: u8) -> u8 { match x { u8:0 => u8:9, remaining => remaining } }
        fn first(xs: u8[2]) -> u8 { match xs { all => all[u32:0] } }
        const_assert!(both_tag(u8:3, u8:3));
        const_assert!(!both_tag(u8:3, u8:4));
        const_assert!(scalar(u8:0) == u8:9 && scalar(u8:5) == u8:5);
    "#;
    let params = "a: u8, b: u8";
    let body = "if both_tag(a, b) { u8:255 } else { scalar(first([a, b])) }";
    let dslx = format!("{functions} fn sample({params}) -> u8 {{ {body} }}");
    let bx = format!(
        "{functions} #[allow(combinational_block)] block sample({params}) -> u8 {{ {body} }}"
    );
    let eval = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (a, b, expected) in [(0, 0, 9), (3, 3, 255), (2, 3, 2), (3, 4, 3)] {
        assert_eq!(
            eval.compare(&[IrValue::make_ubits(8, a)?, IrValue::make_ubits(8, b)?])?,
            IrValue::make_ubits(8, expected)?,
        );
    }
    Ok(())
}

#[test]
fn tuple_match_can_infer_a_rest_pattern_from_a_call_subject() -> Result<(), String> {
    // Pinned DSLX v0.65.2 needs the call bound to a typed local for this rest
    // pattern.
    let source = r#"
        fn key(x: (bool, u8)) -> (bool, u8) { x }
        fn pick(x: (bool, u8)) -> u8 {
            match key(x) { (false, .., tail) => tail, (.., tail) => !tail }
        }
        const_assert!(pick((false, u8:3)) == u8:3);
        const_assert!(pick((true, u8:3)) == u8:252);
    "#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

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
        ("(true, ..) => u8:1, (false, ..) => u8:2", true),
        ("all => u8:1, all => u8:2", false),
        ("all => u8:1, other => u8:2", true),
        ("(a, a) => u8:1", false),
        (
            "(a, a) => u8:1, (true, false) => u8:2, (false, true) => u8:3",
            false,
        ),
        ("(.., ..) => u8:1", false),
        ("(a, a) | (false, true) => u8:1, _ => u8:0", false),
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

#[test]
fn match_rest_and_whole_binders_retain_literal_widths() -> Result<(), Box<dyn Error>> {
    for (subject, explicit, arms, expected) in [
        ("1", "u8:1", "v => v", 1),
        ("1", "u8:1", "v => !v", 254),
        ("(1, b)", "(u8:1, b)", "v => v.0", 1),
        ("[1, 2]", "[u8:1, u8:2]", "v => v[u32:0]", 1),
        (
            "(b, u4:7, 1)",
            "(b, u4:7, u8:1)",
            "(false, .., 128) => u8:0, (.., v) => v",
            1,
        ),
        (
            "(b, (2, 1))",
            "(b, (u2:2, u8:1))",
            "(false, (.., 128)) => u8:0, (_, (.., v)) => v",
            1,
        ),
    ] {
        let body = format!("match {subject} {{ {arms} }}");
        let source = format!("fn sample(b: bool) -> u8 {{ {body} }}");
        dslx::parse_and_typecheck(
            &source,
            "match_width.x",
            "match_width",
            &mut ImportData::default(),
        )?;
        check_module(&parse_module(&source)?)?;
        // The pinned converter needs explicit subject widths to generate
        // comparable IR after successfully inferring the source types.
        let reference = format!("match {explicit} {{ {arms} }}");
        let bx = super::expr_eval::bx_source("b: bool", "u8", &body);
        let eval = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx("b: bool", "u8", &reference),
        )?;
        assert_eq!(
            eval.compare(&[IrValue::bool(true)])?,
            IrValue::make_ubits(8, expected)?,
            "{body}"
        );
    }
    for (result, body) in [
        ("u8", "match 1 { u3:1 => u8:0, v => v }"),
        (
            "u8",
            "match (true, u4:7, 1) { (false, .., 256) => u8:0, (.., v) => v }",
        ),
        ("s8", "match key() { 0 | 0xff => s8:0, v => v }"),
    ] {
        let source = format!("fn key() -> s8 {{ s8:-1 }} fn sample() -> {result} {{ {body} }}");
        assert!(
            dslx::parse_and_typecheck(
                &source,
                "match_width.x",
                "match_width",
                &mut ImportData::default()
            )
            .is_err(),
            "{body}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{body}");
    }
    Ok(())
}
