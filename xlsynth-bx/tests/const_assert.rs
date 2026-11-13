// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs, path::Path};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_source_with_options, empty_package,
    lower_module, parse_module, run_tests,
};
use xlsynth_pir::{IrValue, ir_eval::eval_fn_in_package_value};

#[test]
fn module_assertions_check_in_order_without_binding_names() -> Result<(), String> {
    let source = r#"
#![require_define_before_use]
const LIMIT = u32:4;
fn next<N: u32>() -> u32 { N + u32:1 }
const_assert!(next<LIMIT>() == u32:5);
const_assert!({ let v = "A"; v[u32:0] == u8:65 });
const LAST = u32:7;
const_assert!(LAST > LIMIT);
"#;
    let module = parse_module(source)?;
    assert_eq!(module.const_asserts.len(), 3);
    let checked = check_module(&module)?;
    let package = lower_module(&checked, empty_package("test"), ResetBehavior::default())?;
    assert!(package.members.is_empty(), "{package}");
    let source = format!("{source}\nfn sample() -> u32 {{ next<LIMIT>() }}");
    let shared = check_module(&parse_module(&source)?)?;
    let package = lower_module(&shared, empty_package("test"), ResetBehavior::default())?;
    assert_eq!(package.members.len(), 2);
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package.get_fn("sample").ok_or("missing sample")?,
            &[]
        ),
        IrValue::make_ubits(32, 5).map_err(|e| e.to_string())?
    );
    assert_eq!(
        checked
            .constants()
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["LIMIT", "LAST"]
    );
    for header in ["", "#![require_define_before_use]\n"] {
        for source in [
            "const_assert!(LATER); const LATER = true;",
            "const_assert!(later()); fn later() -> bool { true }",
            "const_assert!(zero!<Later>() == u8:0); type Later = u8;",
        ] {
            assert!(
                check_module(&parse_module(&format!("{header}{source}"))?).is_err(),
                "{source}"
            );
        }
    }
    Ok(())
}

#[test]
fn nested_test_assertions_use_the_source_loop_scope() -> Result<(), String> {
    let check = |predicate: &str| {
        let source = format!(
            r#"
#[test] fn probe() {{
    for i in u32:0..u32:1 {{
        assert!({{ let (a, _) = (true, i); const_assert!({predicate}); a }}, "ok");
    }}
}}
"#
        );
        run_tests(
            &source,
            Path::new("probe.bx"),
            empty_package("test"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
        )
    };
    assert!(check("a").unwrap_err().contains("compile-time constant"));
    let passed = check("true")?;
    assert_eq!(passed.tests[0].failure, None);
    Ok(())
}

#[test]
fn module_assertions_are_private_statements_in_dslx_syntax() {
    for source in [
        "pub const_assert!(true);",
        "#[allow(useless_struct_splat)] const_assert!(true);",
        "#[test] const_assert!(true);",
        "const_assert!(true,);",
        "const_assert!(true)",
        "const_assert!(/* a DSLX predicate */ true);",
        "const_assert!(true); #![require_define_before_use]",
    ] {
        assert!(parse_module(source).is_err(), "{source}");
    }
}

#[test]
fn assertion_operands_do_not_change_existing_constexpr_consumers() -> Result<(), String> {
    let source = r#"
fn get<N: u32>() -> u32 { N }
fn f(x: u32) -> (u32, u32, u32, u32[1]) {
    let (n, _) = (u32:3, x);
    let k = { x; n };
    let t = match true { true => k, _ => k };
    (get<t>(), ceillog2(t), configured_value_or("test", t), map([t], ceillog2))
}
"#;
    for assertion in ["", "const_assert!(true);"] {
        check_module(&parse_module(&format!("{source}{assertion}"))?)?;
    }
    for predicate in ["k == u32:3", "ceillog2(k) == u32:2", "get<k>() == u32:3"] {
        let source = format!(
            r#"fn get<N: u32>() -> u32 {{ N }}
fn f(x: u32) {{
    let (n, _) = (u32:3, x);
    let k = {{ x; n }};
    let _ = get<k>();
    const_assert!({predicate});
}}"#
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    Ok(())
}

#[test]
fn assertion_helper_replay_does_not_change_ordinary_bx_index_checking() -> Result<(), String> {
    for assertion in ["", "const_assert!(size == u32:3);"] {
        let source = format!(
            r#"fn n() -> u32 {{ u32:3 }}
fn f() -> u32 {{ let size = n(); {assertion} [u32:1][size] }}"#
        );
        check_module(&parse_module(&source)?)?;
    }
    Ok(())
}

#[test]
fn builtin_constant_cache_preserves_its_existing_helper_and_branch_rules() -> Result<(), String> {
    for assertion in ["", "const_assert!(true);"] {
        let source = format!(
            "fn f(x:u32)->u32 {{ let a = match true {{ true => u32:4, _ => x }}; ceillog2(a) }} {assertion}"
        );
        check_module(&parse_module(&source)?)?;

        let source =
            format!("fn n()->u32 {{ u32:3 }} fn f()->u32 {{ let a=n(); ceillog2(a) }} {assertion}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(
            error.contains("'ceillog2' requires a compile-time argument"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn match_bytecode_checks_nested_arm_lets_in_their_defining_scope() -> Result<(), String> {
    for (predicate, valid) in [
        (
            "match { let _=if true {true} else {x}; true } { true=>true, _=>true }",
            false,
        ),
        (
            "match true { true => { let t={let _=x;true}; t }, _=>true }",
            false,
        ),
        (
            "{ let t={let _=x;true}; match true { true => t, _=>true } }",
            true,
        ),
        (
            "match true { true => { let t=if true {true} else {x}; t }, _=>true }",
            false,
        ),
        (
            "{ let t=if true {true} else {x}; match true { true => t, _=>true } }",
            true,
        ),
        (
            "match true { true=>true, _=>{ let t=if true {true} else {id(x)}; t } }",
            false,
        ),
        (
            "match true { true=>true, _=>{ let t=read_at(u32:5); t } }",
            true,
        ),
    ] {
        let source = format!(
            "fn id(x:bool)->bool {{ x }} fn read_at(i:u32)->bool {{ [true][i] }} fn f(x:bool) {{ const_assert!({predicate}); }}"
        );
        let result = check_module(&parse_module(&source)?);
        assert_eq!(result.is_ok(), valid, "{source}: {:?}", result.err());
    }
    Ok(())
}

#[test]
fn assertion_in_a_match_arm_evaluates_its_let_independently() -> Result<(), String> {
    let source = "fn f(x:bool) { match false { true=>{ let b={let _=x; true}; const_assert!(b); }, _=>() } }";
    check_module(&parse_module(source)?)?;
    for (condition, valid) in [("true", true), ("false", false)] {
        let source = format!(
            r#"fn f(x: bool) {{
                match false {{
                    true => {{
                        let b = if {condition} {{ true }} else {{ x; true }};
                        const_assert!(b);
                    }},
                    _ => (),
                }}
            }}"#
        );
        let checked = check_module(&parse_module(&source)?);
        assert_eq!(checked.is_ok(), valid, "{source}: {:?}", checked.err());
    }
    Ok(())
}

#[test]
fn assertion_replay_checks_erased_type_dimensions_and_range_endpoints() -> Result<(), String> {
    for value in [
        "(uN[k]:0 as u8) == u8:0",
        "zero!<uN[k]>() == u3:0",
        "S<k> { a: u3:7 }.a == u3:7",
        "{ let t: uN[k] = 0; t == u3:0 }",
        "unroll_for! (i,a) in u32:0..k { true }(false)",
        "for (i,a) in u32:0..k { true }(false)",
    ] {
        for (operand, valid) in [("u32:3", true), ("{ x; u32:3 }", false)] {
            for (assertion, accepted) in [
                (format!("const_assert!({value});"), valid),
                (format!("let b={value}; const_assert!(b);"), valid),
                (format!("let b={value}; const_assert!(true);"), true),
            ] {
                let source = format!(
                    "struct S<N:u32> {{ a:uN[N] }} fn f(x:bool) {{ let k={operand}; {assertion} }}"
                );
                let checked = check_module(&parse_module(&source)?);
                assert_eq!(checked.is_ok(), accepted, "{source}: {:?}", checked.err());
            }
        }
    }
    Ok(())
}

#[test]
fn assertions_replay_explicit_value_arguments_of_let_bound_calls() -> Result<(), String> {
    for (operand, valid) in [("u32:3", true), ("{x; u32:3}", false)] {
        for call in ["val<k>()", "val<{k + u32:0}>()"] {
            let definitions = format!(
                "fn val<N:u32>()->u32 {{ N }} fn f(x:bool)->bool {{ let k={operand}; let ok={call}==u32:3;"
            );
            let baseline = check_module(&parse_module(&format!("{definitions} ok }}"))?)?;
            for (assertion, accepted) in [
                ("const_assert!(true);", true),
                ("const_assert!(ok);", valid),
            ] {
                let source = format!("{definitions} {assertion} ok }}");
                let checked = check_module(&parse_module(&source)?);
                assert_eq!(
                    checked.is_ok(),
                    accepted,
                    "{source}: {:?}",
                    checked.as_ref().err()
                );
                if let Ok(checked) = checked {
                    let lower = |module| {
                        lower_module(module, empty_package("test"), ResetBehavior::default())
                            .map(|pkg| pkg.to_string())
                    };
                    assert_eq!(lower(&checked)?, lower(&baseline)?, "{source}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn assertions_replay_erased_builtin_operands_without_changing_ir() -> Result<(), String> {
    for predicate in [
        "ceillog2(bad(I)) == u32:0",
        "map([bad(I)], ceillog2)[u32:0] == u32:0",
        "configured_value_or(\"key\", bad(I)) == u32:1",
        "one_hot(u1:1, { let n = bad(I); n == u32:1 }) == u2:1",
        "signex(u8:1, bad(I)) == u32:1",
        "array_slice([u32:1], u32:0, [bad(I)])[u32:0] == u32:1",
    ] {
        for (index, valid) in [("u32:0", true), ("u32:4", false)] {
            let predicate = predicate.replace('I', index);
            let definitions = format!(
                "fn bad(i:u32)->u32 {{ let _=[u32:1][i]; u32:1 }} fn mid()->bool {{ {predicate} }}"
            );
            let baseline = format!("{definitions} fn wrapper()->bool {{ true }}");
            let baseline = check_module(&parse_module(&baseline)?)?;
            for body in [
                "const_assert!(mid()); true".to_owned(),
                format!("let ok={predicate}; const_assert!(ok); true"),
            ] {
                let source = format!("{definitions} fn wrapper()->bool {{ {body} }}");
                let checked = check_module(&parse_module(&source)?);
                assert_eq!(
                    checked.is_ok(),
                    valid,
                    "{source}: {:?}",
                    checked.as_ref().err()
                );
            }
            let source =
                format!("{definitions} fn wrapper()->bool {{ const_assert!(true); true }}");
            let checked = check_module(&parse_module(&source)?)?;
            let lower = |module| {
                lower_module(module, empty_package("test"), ResetBehavior::default())
                    .map(|pkg| pkg.to_string())
            };
            assert_eq!(lower(&checked)?, lower(&baseline)?, "{source}");
        }
    }
    Ok(())
}

#[test]
fn linked_module_assertions_use_their_defining_scope() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    fs::write(
        dir.path().join("base.bx"),
        "pub const N = u32:3; pub fn zero!() -> u8 { u8:7 }",
    )?;
    let library = r#"#![require_define_before_use]
use base::N;
fn fits(x: u32) -> bool { x < u32:4 }
const_assert!(fits(N));
const_assert!(zero!<u8>() == u8:0);
use base::zero!;
const_assert!(zero!<u8>() == u8:7);
pub const GOOD = true;
#[test] fn hidden_test() { assert!(false); }
"#;
    let path = dir.path().join("gate.bx");
    fs::write(&path, library)?;
    let root = r#"use gate;
const N = u32:100;
fn fits(x: u32) -> bool { false }
const_assert!(gate::GOOD);
#[test] fn visible_test() { assert!(true); }
"#;
    let run = run_tests(
        root,
        Path::new("root.bx"),
        empty_package("test"),
        ResetBehavior::default(),
        &options,
        true,
    )?;
    assert_eq!(run.tests.len(), 1);
    assert_eq!(run.tests[0].name, "visible_test");
    assert_eq!(run.tests[0].failure, None);

    fs::write(&path, library.replace("fits(N)", "fits(u32:4)"))?;
    // Even an unused import checks its module assertions.
    let error = check_source_with_options("use gate;", &options).unwrap_err();
    assert!(
        error.contains("gate") && error.contains("4:1") && error.contains("const_assert!"),
        "{error}"
    );
    Ok(())
}
