// SPDX-License-Identifier: Apache-2.0

use xlsynth_bx::{ResetBehavior, check_module, empty_package, lower_module, parse_module};
use xlsynth_pir::{IrValue, ir_eval::eval_fn_in_package_value};

#[test]
fn lexical_assertions_use_preceding_bindings_and_helpers() -> Result<(), String> {
    let source = r#"
const OK = true;
fn within(x: u32) -> bool { x < u32:4 }
fn checked(x: u8) -> u8 {
    const_assert!(OK);
    let inside = within(u32:3);
    if false {
        let OK = inside;
        const_assert!(OK);
        x
    } else {
        const_assert!({ let value = (x, inside); value.1 });
        x
    }
}
"#;
    // The tuple's runtime field prevents the entire tuple from being constexpr.
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert!(error.contains("12:9: const_assert!"), "{error}");
    check_module(&parse_module(
        &source.replace("(x, inside)", "(u8:0, inside)"),
    )?)?;
    for invalid in [
        "fn f() { const_assert!(later); let later = true; }",
        "fn f(x:bool) { let v = true; { let v = x; const_assert!(v); }; }",
        "fn f() { if false { const_assert!(false); } }",
        "fn f() { match false { true => { const_assert!(false); }, _ => () } }",
    ] {
        assert!(check_module(&parse_module(invalid)?).is_err(), "{invalid}");
    }
    Ok(())
}

#[test]
fn assertions_require_one_boolean_and_a_semicolon() -> Result<(), String> {
    for predicate in ["true", "u1:1", "1"] {
        check_module(&parse_module(&format!(
            "fn f() {{ const_assert!({predicate}); }}"
        ))?)?;
    }
    for predicate in ["false", "u2:1", "s1:-1", "2", "(true,)"] {
        let source = format!("fn f() {{ const_assert!({predicate}); }}");
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    let zero = parse_module("fn f() { let b = 0; const_assert!(b); }")?;
    assert!(check_module(&zero).is_err());
    for invalid in [
        "fn f() { const_assert!(true) }",
        "fn f() { const_assert!(true,) ; }",
        "fn f() { let _ = const_assert!(true); }",
        "fn f() { (const_assert!(true),) }",
        "fn const_assert!(x: bool) { }",
        "fn f(const_assert!: bool) { }",
    ] {
        assert!(parse_module(invalid).is_err(), "{invalid}");
    }
    Ok(())
}

#[test]
fn predicates_follow_dslx_constexpr_control_flow() -> Result<(), String> {
    let declarations = "fn yes(x: bool) -> bool { true }";
    for (predicate, valid) in [
        ("if true { true } else { x }", true),
        ("if true { yes(true) } else { yes(x) }", true),
        ("if false { true } else { x }", false),
        ("match true { true => true, _ => x }", false),
        ("true || x", false),
        ("yes(x)", false),
        ("{ x; true }", false),
        ("{ let _ = x; true }", true),
        ("if true { true } else { x; true }", true),
        ("{ let (a, _) = (true, x); a }", false),
        (
            "{ let (a, _) = (true, if true { true } else { x }); a }",
            true,
        ),
    ] {
        for statement in [
            format!("const_assert!({predicate});"),
            format!("let p = {predicate}; const_assert!(p);"),
        ] {
            let source = format!("{declarations} fn f(x: bool) {{ {statement} }}");
            let checked = check_module(&parse_module(&source)?);
            assert_eq!(checked.is_ok(), valid, "{source}: {:?}", checked.err());
        }
    }
    Ok(())
}

#[test]
fn assertions_are_checked_at_each_function_specialization() -> Result<(), String> {
    let definitions = r#"
fn positive<N: u32>(x: u8) -> u8 { const_assert!(N > u32:0); x }
fn fits<T>(x: T) -> T { const_assert!(bit_count<T>() <= u32:8); x }
fn accept() -> bool { true }
fn reject() -> bool { false }
fn guarded<F: fn() -> bool>(x: u8) -> u8 { const_assert!(F()); x }
"#;
    for (call, valid) in [
        ("x", true),
        ("guarded<accept>(positive<2>(fits(x)))", true),
        ("positive<0>(positive<2>(x))", false),
        ("fits(x as u9) as u8", false),
        ("guarded<reject>(guarded<accept>(x))", false),
    ] {
        let source = format!("{definitions} fn main(x: u8) -> u8 {{ {call} }}");
        let checked = check_module(&parse_module(&source)?);
        assert_eq!(checked.is_ok(), valid, "{source}: {:?}", checked.err());
    }
    for (formals, params) in [
        ("N: u32", "x: bool"),
        ("T", "x: bool, y: T"),
        ("F: fn() -> bool", "x: bool"),
    ] {
        let f = format!("fn f<{formals}>({params}) {{ const_assert!(x); }}");
        check_module(&parse_module(&f)?)?;
        let f = f.replace("const_assert!(x)", "const_assert!(missing)");
        assert!(check_module(&parse_module(&f)?).is_err(), "{f}");
    }
    Ok(())
}

#[test]
fn destructuring_an_unrolled_tuple_preserves_earlier_bytecode_work() -> Result<(), String> {
    let helper = r#"
fn first(i: u32, p: (bool, bool)) -> bool {
    let t = unroll_for! (j, a) in u32:0..u32:2 {
        if j == u32:0 { ([true][i], true) } else { p }
    }(p);
    let (a, _) = t;
    a
}
"#;
    for (i, valid) in [(0, true), (1, false)] {
        let source = format!("{helper} const_assert!(first(u32:{i}, (true, true)));");
        let result = check_module(&parse_module(&source)?);
        assert_eq!(result.is_ok(), valid, "{source}: {result:?}");
    }
    Ok(())
}

#[test]
fn unrolled_carries_preserve_runtime_ir_and_constexpr_initializers() -> Result<(), String> {
    let lower = |source: &str| {
        lower_module(
            &check_module(&parse_module(source)?)?,
            empty_package("test"),
            ResetBehavior::default(),
        )
    };
    for (step, seed) in [("a + u32:1", "u32:0"), ("a", "x")] {
        let source = format!(
            "fn f(x:u32)->u32 {{ unroll_for! (i,a) in u32:0..u32:3 {{ let _ = x; {step} }}({seed}) }}"
        );
        assert_eq!(
            lower(&format!("{source}\nconst_assert!(true);"))?.to_string(),
            lower(&source)?.to_string(),
            "{source}"
        );
    }
    let helper = r#"
fn f(idx:u32)->u32 {
    unroll_for! (i,a) in u32:0..u32:3 { let _ = [true][idx]; a + u32:1 }(u32:0)
}
"#;
    check_module(&parse_module(&format!(
        "{helper} const_assert!(f(u32:0) == u32:3);"
    ))?)?;
    let error = check_module(&parse_module(&format!(
        "{helper} const_assert!(f(u32:1) == u32:3);"
    ))?)
    .unwrap_err();
    assert!(error.contains("compile-time constant"), "{error}");
    Ok(())
}

#[test]
fn assertions_in_loops_follow_body_elaboration() -> Result<(), String> {
    for (kind, limit, predicate, valid) in [
        ("unroll_for!", 0, "false", true),
        ("unroll_for!", 0, "missing", false),
        ("unroll_for!", 2, "i < u32:2", true),
        ("unroll_for!", 3, "i < u32:2", false),
        ("for", 0, "false", false),
        ("for", 2, "i < u32:2", false),
    ] {
        let source = format!(
            "fn f(x: u8) -> u8 {{ {kind} (i, acc) in u32:0..u32:{limit} {{ const_assert!({predicate}); acc }}(x) }}"
        );
        let checked = check_module(&parse_module(&source)?);
        assert_eq!(checked.is_ok(), valid, "{source}: {:?}", checked.err());
    }
    Ok(())
}

#[test]
fn concrete_counted_loop_and_match_results_can_be_asserted_directly() -> Result<(), String> {
    let read = "fn read_at(i:u32)->bool { [true][i] }";
    for (predicate, valid) in [
        ("for (i,a) in u32:0..u32:1 { true }(false)", true),
        ("for (i,a) in u32:0..u32:1 { false }(true)", false),
        ("match true { true => true, _ => read_at(u32:5) }", true),
        ("match false { true => true, _ => read_at(u32:5) }", false),
    ] {
        for assertion in [predicate, "helper()"] {
            let source =
                format!("{read} fn helper()->bool {{ {predicate} }} const_assert!({assertion});");
            let checked = check_module(&parse_module(&source)?);
            assert_eq!(checked.is_ok(), valid, "{source}: {:?}", checked.err());
        }
    }
    Ok(())
}

#[test]
fn constexpr_bytecode_checks_initializers_on_the_selected_path() -> Result<(), String> {
    for (body, valid) in [
        ("[u8:1][i]; true", false),
        ("let _ = [u8:1][i]; true", false),
        ("let (ok, _) = (true, [u8:1][i]); ok", false),
        ("if false { let _ = [u8:1][i]; false } else { true }", true),
        (
            "unroll_for! (j, a) in u32:0..u32:2 { let _ = [u8:1][i - u32:2 - j]; true }(true)",
            false,
        ),
        (
            "unroll_for! (j, a) in u32:0..u32:2 { let _ = [u8:1][j - j]; true }(true)",
            true,
        ),
        (
            "unroll_for! (j, a) in u32:0..u32:1 { true }([true][i])",
            false,
        ),
    ] {
        let source = format!(
            "fn helper(i:u32)->bool {{ {body} }} fn middle(i:u32)->bool {{ helper(i) }} const_assert!(middle(u32:3));"
        );
        let result = check_module(&parse_module(&source)?);
        assert_eq!(result.is_ok(), valid, "{source}: {:?}", result.err());
    }
    for predicate in [
        "match true { true => { let _ = x; true }, _ => true }",
        "for (i,a) in u32:0..u32:2 { let _ = [u8:1][i]; true }(true)",
    ] {
        let source = format!("fn f(x: bool) {{ const_assert!({predicate}); }}");
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    Ok(())
}

#[test]
fn specialized_lets_replay_helpers_with_dslx_semantics() -> Result<(), String> {
    let source = r#"
fn read_at(i: u32) -> bool { [u8:1, u8:2][i]; true }
fn guard<N: u32>() {
    if false {
        let first = read_at(N % u32:3);
        let condition = first && true;
        const_assert!(condition);
    }
}
fn sample() { guard<8>(); }
"#;
    assert!(check_module(&parse_module(source)?).is_err());
    check_module(&parse_module(&source.replace("guard<8>()", "guard<7>()"))?)?;
    Ok(())
}

#[test]
fn ordinary_bx_bounds_use_the_selected_tuple_field() -> Result<(), String> {
    for assertion in ["", "const_assert!(true);"] {
        let prefix = format!("fn f(x:u32)->u8 {{ {assertion} let (n,_)=(u32:3,x);");
        check_module(&parse_module(&format!(
            "{prefix} for (i,a) in u32:0..n {{ a }}(u8:1) }}"
        ))?)?;
        assert!(check_module(&parse_module(&format!("{prefix} [u8:1][n] }}"))?).is_err());
    }
    Ok(())
}

#[test]
fn shared_aggregate_projections_preserve_runtime_ir() -> Result<(), String> {
    let source = r#"
struct Pair { a: u8, b: u8 }
const WORDS = (u8:17, u8:2);
const FIELDS = Pair { a: u8:7, b: u8:9 };
fn sample(p: (u8, u8), x: u8) -> (u8, u8, u8, u8, u8[2], (u8, u8)) {
    let (t, _) = (p, x);
    let (a, b) = t;
    let (c, _) = { let q = p; q };
    let (words, _) = (WORDS, x);
    let (_, d) = words;
    let (base, _) = (FIELDS, x);
    let updated = Pair { a: x, ..base };
    let narrow = [1, 2];
    let (xs, _) = (narrow, x);
    let small = (3, 4);
    let (pair, _) = (small, x);
    (a + b, c, d, updated.a + updated.b, xs, pair)
}
"#;
    let lower = |source: &str| {
        lower_module(
            &check_module(&parse_module(source)?)?,
            empty_package("test"),
            ResetBehavior::default(),
        )
    };
    let reference = lower(source)?;
    let package = lower(&format!("const_assert!(true);\n{source}"))?;
    assert_eq!(package.to_string(), reference.to_string());
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[
                IrValue::parse_typed("(bits[8]:5, bits[8]:13)").map_err(|e| e.to_string())?,
                IrValue::make_ubits(8, 11).map_err(|e| e.to_string())?,
            ],
        ),
        IrValue::parse_typed(
            "(bits[8]:18, bits[8]:5, bits[8]:2, bits[8]:20, [bits[8]:1, bits[8]:2], (bits[8]:3, bits[8]:4))"
        ).map_err(|e| e.to_string())?,
    );
    Ok(())
}

#[test]
fn aggregate_projections_keep_their_initializer_constexpr_dependencies() -> Result<(), String> {
    for body in [
        "let (p, _) = (WORDS, INPUT); let (a, b) = p; const_assert!(a + b == u8:3);",
        "let (p, _) = (FIELDS, INPUT); let updated = Pair { a: u8:1, ..p }; const_assert!(updated.a == u8:1);",
        "let (p, _) = (FIELDS, INPUT); let updated = Pair { a: u8:1, b: u8:2, ..p }; const_assert!(updated.a == u8:1);",
        "let p = [1, 2]; let (xs, _) = (p, INPUT); let wide: u8[2] = xs; const_assert!(wide[u32:0] == u8:1);",
        "let p = (1, 2); let (pair, _) = (p, INPUT); let wide: (u8, u8) = pair; const_assert!(wide.0 == u8:1);",
    ] {
        let source = format!(
            r#"
struct Pair {{ a: u8, b: u8 }}
const WORDS = (u8:1, u8:2);
const FIELDS = Pair {{ a: u8:1, b: u8:2 }};
fn check(x: u8) {{ {body} }}
"#
        );
        check_module(&parse_module(&source.replace("INPUT", "u8:0"))?)?;
        let error = check_module(&parse_module(&source.replace("INPUT", "x"))?).unwrap_err();
        assert!(error.contains("compile-time constant"), "{body}: {error}");
    }
    Ok(())
}

#[test]
fn lexical_aliases_share_repeated_constexpr_scope_operands() -> Result<(), String> {
    let mut source = String::from("fn f(x: bool) -> bool { let a0 = x;");
    for i in 1..=32 {
        let previous = i - 1;
        source.push_str(&format!("let a{i} = {{ let t = a{previous}; t }};"));
    }
    source.push_str("a32 } const_assert!(f(true));");
    check_module(&parse_module(&source)?)?;
    Ok(())
}

#[test]
fn constexpr_calls_queue_independent_lets_on_the_selected_path() -> Result<(), String> {
    let mut source = String::from("fn helper(x: u32) -> bool { if x == u32:2 {");
    for i in 0..1000 {
        source.push_str(&format!("let a{i} = x + u32:1;"));
    }
    source.push_str(
        "a999 == u32:3 } else { let _ = [true][x]; true } } const_assert!(helper(u32:2));",
    );
    check_module(&parse_module(&source)?)?;
    let source = source.replace("helper(u32:2)", "helper(u32:3)");
    assert!(check_module(&parse_module(&source)?).is_err());
    Ok(())
}

#[test]
fn assertion_expression_statements_queue_outer_lets() -> Result<(), String> {
    let mut source = String::from("fn identity(x: bool) -> bool { x } fn check(x: bool) {");
    for i in 0..1000 {
        source.push_str(&format!("let b{i} = identity(true);"));
    }
    source.push_str("const_assert!({ let _ = x;");
    for i in 0..1000 {
        source.push_str(&format!("b{i};"));
    }
    source.push_str("b999 }); }");
    check_module(&parse_module(&source)?)?;
    Ok(())
}
