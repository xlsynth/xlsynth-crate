// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};
use std::error::Error;
use xlsynth_bx::{ImportOptions, check_module, parse_module};
use xlsynth_pir::IrValue;

/// Call the same function body from a lowered BX block and a DSLX function.
fn compare_function(
    helpers: &str,
    params: &str,
    result: &str,
    call: &str,
) -> Result<ExpressionEvaluator, String> {
    let bx = format!("{helpers}\n{}", bx_source(params, result, call));
    let dslx = format!("{helpers}\n{}", dslx_source(params, result, call));
    ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )
}

#[test]
fn concrete_signature_and_local_alias_see_the_same_runtime_parameters() -> Result<(), Box<dyn Error>>
{
    let helpers = r#"
const W = u32:7;
fn narrow(W: u32, data: uN[if true { u32:5 } else { W }])
    -> uN[if false { W } else { u32:4 }] {
    type Result = uN[if false { W } else { u32:4 }];
    let value: Result = data as Result;
    value
}
"#;
    let evaluator = compare_function(helpers, "W: u32, data: u5", "u4", "narrow(W, data)")?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(32, 19)?, IrValue::make_ubits(5, 31)?])?,
        IrValue::make_ubits(4, 15)?
    );
    let selected = helpers.replace("if true", "if false");
    assert!(compile_dslx_module(&selected, &[]).is_err());
    assert!(check_module(&parse_module(&selected)?).is_err());
    Ok(())
}

#[test]
fn a_signature_uses_the_runtime_array_length_when_checking_a_typed_loop()
-> Result<(), Box<dyn Error>> {
    let source = r#"
const xs = u32[2]:[0, 0];
fn narrow(xs: u32[3]) -> uN[if true { u32:5 } else {
    for (v, acc): (u32, u32) in xs { v + acc }(u32:0)
}] { u5:1 }
const RESULT = narrow(u32[3]:[1, 2, 3]);
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants().last().unwrap().value,
        IrValue::make_ubits(5, 1)?
    );
    let runtime = source.replace("if true", "if false");
    let error = check_module(&parse_module(&runtime)?).unwrap_err();
    assert!(
        error.contains("requires a compile-time constant"),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_runtime_formal_shadows_a_named_map_callback_in_a_dimension() -> Result<(), Box<dyn Error>> {
    for (parameter, accepted) in [("other", true), ("incr", false)] {
        let source = format!(
            r#"
fn incr(x: u32) -> u32 {{ x + u32:1 }}
fn sample({parameter}: u32) -> uN[map([u32:4], incr)[u32:0]] {{ u5:0 }}
"#
        );
        assert_eq!(
            compile_dslx_module(&source, &[]).is_ok(),
            accepted,
            "{source}"
        );
        assert_eq!(
            check_module(&parse_module(&source)?).is_ok(),
            accepted,
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn aliases_and_constants_follow_lexical_scope() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
const W = u32:3;
type Word = u16;
fn unpack(x: u8, pick: bool) -> (u8, u8, u32) {
    let first = Word:257 as u8;
    let w = W + u32:1;
    type Word = uN[w];
    const K: Word = 9;
    const W = w + u32:1;
    type Before = uN[W];
    const W = W + u32:1;
    const_assert!(bit_count<Before>() == u32:5);
    let branch = if pick {
        const K = K + Word:1;
        type Word = uN[W];
        let z: Word = Word:17;
        (K as u8) + (z as u8)
    } else {
        let K = x;
        K
    };
    (first + (K as u8), branch, W)
}
const CHECKED = unpack(u8:91, false);
"#;
    let evaluator = compare_function(
        helpers,
        "x: u8, pick: bool",
        "(u8, u8, u32)",
        "unpack(x, pick)",
    )?;
    for (pick, branch) in [(true, 27), (false, 91)] {
        assert_eq!(
            evaluator.compare(&[IrValue::make_ubits(8, 91)?, IrValue::bool(pick)])?,
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, 10)?,
                IrValue::make_ubits(8, branch)?,
                IrValue::make_ubits(32, 6)?,
            ])
        );
    }
    let checked = check_module(&parse_module(helpers)?)?;
    assert_eq!(
        checked.constants().last().unwrap().value,
        IrValue::parse_typed("(bits[8]:10, bits[8]:91, bits[32]:6)")?
    );
    Ok(())
}

#[test]
fn local_types_resolve_per_function_specialization() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn increase<N: u32>(x: uN[N]) -> uN[N] {
    const WIDTH = N;
    type Word = uN[WIDTH];
    const STEP: Word = Word:3;
    const_assert!(bit_count<Word>() == WIDTH);
    x + STEP
}
"#;
    let evaluator = compare_function(
        helpers,
        "x: u8",
        "(u4, u8)",
        "(increase(x as u4), increase(x))",
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 31)?])?,
        IrValue::parse_typed("(bits[4]:2, bits[8]:34)")?
    );
    let bx = r#"
fn dup<T>(x: T) -> (T, T) {
    type Element = T;
    type Pair = (Element, Element);
    const SEED: Pair = zero!<Pair>();
    (SEED.0, x)
}
const RESULTS = (dup(u8:35), dup(true));
"#;
    let checked = check_module(&parse_module(bx)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("((bits[8]:0, bits[8]:35), (bits[1]:0, bits[1]:1))")?
    );
    Ok(())
}

#[test]
fn local_alias_keeps_a_struct_named_like_another_formal() -> Result<(), Box<dyn Error>> {
    let functions = r#"
struct U { x: u8 }
struct Box<E> { value: E }
fn read<T,U>(a: T, b: U) -> (u8, u8, u8, u8) {
    type A = T;
    let one: A = a;
    type Pair = (A[2], U);
    let pair: Pair = ([one, a], b);
    type Boxed = Box<A>;
    let boxed: Boxed = Boxed { value: a };
    (one.x, pair.0[u32:1].x, pair.1 as u8, boxed.value.x)
}
"#;
    let params = "x: u8, y: u8";
    let result = "(u8, u8, u8, u8)";
    let call = "read(U { x }, y)";
    let bx = format!("{functions}\n{}", bx_source(params, result, call));
    let native = functions
        .replace("Box<E>", "Box<E:type>")
        .replace("read<T,U>", "read<T:type,U:type>");
    let dslx = format!(
        "#![feature(generics)]\n{native}\n{}",
        dslx_source(params, result, call)
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 5)?, IrValue::make_ubits(8, 7)?])?,
        IrValue::parse_typed("(bits[8]:5, bits[8]:5, bits[8]:7, bits[8]:5)")?
    );
    Ok(())
}

#[test]
fn local_const_width_follows_constexpr_let_dependencies() -> Result<(), Box<dyn Error>> {
    for prefix in [
        "let n = width(); const C = n;",
        "const C = { let ignored = x; u32:8 };",
        "const C = if true { u32:8 } else { x as u32 };",
        "let n = if true { u32:8 } else { x as u32 }; const C = n;",
    ] {
        let helpers = format!(
            r#"
fn width() -> u32 {{ u32:8 }}
fn narrow(x: u8) -> u8 {{
    {prefix}
    type Word = uN[C];
    x as Word
}}
"#
        );
        let evaluator = compare_function(&helpers, "x: u8", "u8", "narrow(x)")?;
        let input = IrValue::make_ubits(8, 253)?;
        assert_eq!(
            evaluator.compare(std::slice::from_ref(&input))?,
            input,
            "{prefix}"
        );
    }
    Ok(())
}

#[test]
fn overwritten_literal_dimensions_resolve_local_operands() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
enum Mode: u1 { Active = 1 }
fn width() -> u32 { u32:4 }
fn literals(x: u8) -> u8 {
    const N = width();
    let first = u8:(uN[N + u32:1]:241);
    let second = u8:(uN[Mode::Ignored + width()]:2);
    let third = u8:(uN[match x { bound => (bound as u32) + N }]:3);
    let fourth = u8:(uN[match x { N | rev => N, _ => x }]:1);
    first + second + third + fourth
}
"#;
    let evaluator = compare_function(helpers, "x: u8", "u8", "literals(x)")?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 42)?])?,
        IrValue::make_ubits(8, 247)?
    );
    for dimension in [
        "N + MISSING",
        "LATER + N",
        "N + later()",
        "N + (u8:3 as Missing)",
        "N + (match x { bound => out_of_scope + bound })",
        "match x { bound | _ => bound }",
        "match (x, x) { (_, _) | (bound, _) => bound }",
    ] {
        let source = format!(
            "fn sample(x:u8)->u8 {{ const N=u32:4; u8:(uN[{dimension}]:7) }}\nconst LATER=u32:1; fn later()->u32 {{ LATER }}"
        );
        assert!(compile_dslx_module(&source, &[]).is_err(), "{source}");
        assert!(parse_module(&source).is_err(), "{source}");
    }
    Ok(())
}

#[test]
fn map_lambda_inherits_outer_alias_and_constant() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn adjust(xs: u8[2]) -> u8[2] {
    const WIDTH = 4;
    type Word = uN[WIDTH];
    const BIAS: Word = 3;
    map(xs, |x| {
        type Inner = Word;
        const STEP = BIAS + Inner:1;
        let narrow: Inner = x as Inner;
        (narrow + STEP) as u8
    })
}
"#;
    let evaluator = compare_function(helpers, "xs: u8[2]", "u8[2]", "adjust(xs)")?;
    assert_eq!(
        evaluator.compare(&[IrValue::parse_typed("[bits[8]:31, bits[8]:128]")?])?,
        IrValue::parse_typed("[bits[8]:3, bits[8]:4]")?
    );
    Ok(())
}

#[test]
fn aliases_preserve_struct_and_enum_identity_in_expressions() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
struct Packet { tag: u8, payload: u8 }
enum Mode: u1 { Idle = u1:0, Active = u1:1 }
fn decode(x: u8, mode: Mode) -> u8 {
    type P = Packet;
    type M = Mode;
    const HIT = M::Active;
    const KEY: P = P { tag: u8:3, payload: u8:7 };
    let packet = P { payload: x, ..KEY };
    match (mode, packet.tag) {
        (HIT, u8:3) => packet.payload + KEY.payload,
        _ => u8:0,
    }
}
"#;
    let evaluator = compare_function(
        helpers,
        "x: u8",
        "(u8, u8)",
        "(decode(x, Mode::Idle), decode(x, Mode::Active))",
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 45)?])?,
        IrValue::parse_typed("(bits[8]:0, bits[8]:52)")?
    );
    Ok(())
}

#[test]
fn invalid_declarations_fail_even_when_unused() {
    for body in [
        "const V = x; x",
        "let y = x + u8:1; const V = y; x",
        "if true { x } else { const V = x; x }",
        "const V = [u8:1][u32:3]; x",
        "const V = { x; u8:1 }; V",
        "let (a, _) = (u8:4, x); const V = a; V",
        "let n = x as u32; type Word = uN[n]; x",
        "type Word = Missing; x",
        "const FIRST = SECOND; const SECOND = u8:1; x",
        "let first: Word = u8:1; type Word = u8; x",
        "let first = { const N = u8:1; N }; N",
        "let first = { type Word = u8; Word:1 }; Word:1",
        "type Word = u8; const Word = u8:2; let v: Word = u8:2; v",
        "const A = u8:2; type A = u8; A",
    ] {
        let source = dslx_source("x: u8", "u8", body);
        assert!(
            compile_dslx_module(&source, &[]).is_err(),
            "DSLX accepted {body}"
        );
        assert!(
            parse_module(&source)
                .and_then(|m| check_module(&m))
                .is_err(),
            "BX accepted {body}"
        );
    }
}

#[test]
fn local_declarations_can_depend_on_a_specialized_callback() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn eight() -> u32 { u32:8 }
fn four() -> u32 { u32:4 }
fn apply<F: fn() -> u32>(x: u8) -> (u32, u8) {
    let n = F();
    const WIDTH = n;
    type Word = uN[F()];
    (WIDTH, (x as Word) as u8)
}
const RESULTS = (apply<eight>(u8:255), apply<four>(u8:255));
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("((bits[32]:8, bits[8]:255), (bits[32]:4, bits[8]:15))")?
    );
    Ok(())
}

#[test]
fn unrelated_local_declarations_preserve_callback_template_typechecking() {
    for body in [
        "type Word = u8; F(x, x)",
        "const K = u8:1; F(u16:0)",
        "type Word = u8; u16:2",
        "const K = F(x); x",
        "const K = F(u8:0); missing",
        "const K = F(u8:0); F(K, K)",
        "const K = F(u8:0); u16:2",
        "const K = F(u8:0); const BAD = x; K",
        "const K = F(u8:0); let v = [u8:1, u8:2][K]; u16:2",
    ] {
        let source = format!("fn unused<F:fn(u8)->u8>(x:u8)->u8 {{ {body} }}");
        assert!(
            parse_module(&source)
                .and_then(|m| check_module(&m))
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn abstract_callback_declarations_can_evaluate_specializations_from_lets()
-> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn id(x:u8)->u8 { x }
fn inc(x:u8)->u8 { x + u8:1 }
fn add<N:u32>(x:u8)->u8 { x + (N as u8) }
fn apply<F:fn(u8)->u8>(x:u8)->u8 { F(x) }
fn outer<G:fn(u8)->u8>(x:u8)->(u8,u8,u8) {
    let one = add<u32:1>(u8:0);
    let two = add<u32:2>(u8:0);
    let next = apply<inc>(two);
    const H = (one,two,next);
    H
}
"#;
    let source = format!("{helpers}\nconst RESULT = outer<id>(u8:3);");
    let checked = check_module(&parse_module(&source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("(bits[8]:1,bits[8]:2,bits[8]:3)")?
    );
    let emitted: Vec<_> = checked
        .functions()
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(emitted, ["id", "inc"]);

    let unused = r#"
fn only_zero<N:u32>(x:u8)->u8 { const_assert!(N == u32:0); x }
fn unused<F:fn(u8)->u8>(x:u8)->u8 {
    let y = only_zero<u32:1>(x);
    const K = u8:1;
    y + K
}
"#;
    let checked = check_module(&parse_module(unused)?)?;
    assert!(checked.functions().is_empty());
    let needed = unused
        .replace("const K = u8:1", "const K = y")
        .replace("only_zero<u32:1>(x)", "only_zero<u32:1>(u8:0)");
    assert!(check_module(&parse_module(&needed)?).is_err());
    Ok(())
}

#[test]
fn callback_dependent_initializers_check_eager_operands_in_either_order()
-> Result<(), Box<dyn Error>> {
    for (init, valid) in [
        ("F(u8:0) + x", false),
        ("x + F(u8:0)", false),
        ("(F(u8:0), x)", false),
        ("(x, F(u8:0))", false),
        ("[F(u8:0), x]", false),
        ("(F(u8:0), F(u8:1), F(x))", false),
        ("F(u8:0) + if true { x } else { u8:0 }", false),
        ("F(u8:0) + if false { x } else { u8:0 }", true),
        ("if true { F(u8:0) } else { x }", true),
        ("if false { F(u8:0) } else { x }", false),
        ("saved + x", false),
        ("x + saved", false),
        ("saved + if false { x } else { u8:1 }", true),
        ("id(saved, x)", false),
        ("id(x, saved)", false),
        ("match u8:1 { u8:0 => saved, _ => x }", false),
        ("match u8:0 { u8:0 => x, _ => saved }", false),
        ("map([saved], |v| v + x)", false),
        ("for (_, z) in [x] { z }(saved)", false),
        ("match u8:0 { u8:0 => saved, _ => u8:0 }", true),
        ("map([saved], |v| v + u8:1)", true),
        ("for (_, z) in [u8:1] { z }(saved)", true),
    ] {
        let source = format!(
            "fn id(a:u8, b:u8)->u8 {{ a+b }} fn unused<F:fn(u8)->u8>(x:u8)->u8 {{ let saved=F(u8:0); const K={init}; x }}"
        );
        let result = check_module(&parse_module(&source)?);
        assert_eq!(result.is_ok(), valid, "{source}: {:?}", result.err());
    }
    Ok(())
}

#[test]
fn deferred_constant_is_checked_at_uses_requiring_its_value() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn one() -> u32 { u32:1 }
fn two() -> u32 { u32:2 }
fn classify<F:fn()->u32>(x:u32)->u8 {
    const K = F();
    match x { K => u8:1, _ => u8:0 }
}
fn count<F:fn()->u32>()->u8 {
    const K = F();
    unroll_for! (_, acc) in u32:0..K { acc+u8:1 }(u8:0)
}
const RESULT = (classify<one>(u32:1), classify<two>(u32:1), count<two>());
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("(bits[8]:1, bits[8]:0, bits[8]:2)")?
    );
    Ok(())
}

#[test]
fn zero_trip_constructor_uses_the_declared_type_parameter() -> Result<(), Box<dyn Error>> {
    for (before, body) in [
        ("", "T { x:z }"),
        ("", "type A=T; A { x:z }"),
        ("type A=T; type B=A;", "B { x:z }"),
    ] {
        let helpers = format!(
            r#"
fn f<T:type>(c:T)->T {{
    {before}
    unroll_for! (_, z) in u32:0..u32:0 {{ {body} }}(c)
}}
"#
        );
        let bx = format!("{helpers}\n{}", bx_source("x:u8", "u8", "f(x)"));
        let dslx = format!(
            "#![feature(generics)]\n{helpers}\n{}",
            dslx_source("x:u8", "u8", "f(x)")
        );
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&dslx, &[]),
        )?;
        let input = IrValue::make_ubits(8, 45)?;
        assert_eq!(evaluator.compare(std::slice::from_ref(&input))?, input);
    }
    Ok(())
}
#[test]
fn enum_pattern_uses_the_alias_that_shadows_a_callback() -> Result<(), Box<dyn Error>> {
    let source = r#"
enum Mode: u1 { Idle = u1:0, Active = u1:1 }
fn id(x: u8) -> u8 { x }
fn lookup<F: fn(u8) -> u8>(mode: Mode) -> u8 {
    type F = Mode;
    match mode { F::Idle => u8:5, _ => u8:0 }
}
const RESULTS = (lookup<id>(Mode::Idle), lookup<id>(Mode::Active));
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::parse_typed("(bits[8]:5, bits[8]:0)")?
    );
    Ok(())
}
