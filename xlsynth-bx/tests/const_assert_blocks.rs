// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, compile_with_options, empty_package,
    parse_module,
};

#[test]
fn block_assertions_check_each_specialization_and_emit_no_hardware() -> Result<(), String> {
    let stages = r#"
const MAX_WIDTH = u32:16;
fn positive<N: u32>() -> bool { N > u32:0 }
block stage<T, N: u32>(clk, x: T) -> T {
    let width = bit_count<T>();
    const_assert!(width <= MAX_WIDTH);
    const_assert!(positive<N>());
    reg q: T;
    q := x;
    q
}
"#;
    for (second, valid) in [
        ("stage<2>(a)", true),
        ("stage<0>(a)", false),
        ("stage<2>(a as u17)", false),
    ] {
        let source = format!(
            "{stages} pub block pipeline(clk, x:u8)->u8 {{ let a=stage<1>(x); let b={second}; b as u8 }}"
        );
        let compiled = compile(&source, empty_package("stages"), ResetBehavior::default());
        if valid {
            let without = source
                .lines()
                .filter(|l| !l.trim_start().starts_with("const_assert!("))
                .collect::<Vec<_>>()
                .join("\n");
            let baseline = compile(&without, empty_package("stages"), ResetBehavior::default())?;
            assert_eq!(compiled?.to_string(), baseline.to_string());
        } else {
            let error = compiled.unwrap_err();
            let line = if second.contains("u17") { 6 } else { 7 };
            assert!(
                error.contains(&format!("{line}:5: const_assert!: predicate is false")),
                "{error}"
            );
        }
    }
    Ok(())
}

#[test]
fn assertion_uses_the_callback_bound_to_this_block_instance() -> Result<(), String> {
    let stage = r#"
fn nonzero(n:u32)->bool { n > u32:0 }
fn reject(n:u32)->bool { false }
block stage<N:u32, F:fn(u32)->bool>(clk, x:u8)->u8 {
    const_assert!(F(N));
    reg q:u8;
    q := x;
    q
}
"#;
    for (callback, valid) in [("nonzero", true), ("reject", false)] {
        let source = format!(
            "{stage} block pipeline(clk,x:u8)->u8 {{ let a=stage<1,nonzero>(x); let b=stage<1,{callback}>(a); b }}"
        );
        let checked = check_module(&parse_module(&source)?);
        assert_eq!(checked.is_ok(), valid, "{source}: {:?}", checked.err());
    }
    Ok(())
}

#[test]
fn block_assertion_requires_a_visible_boolean_constant() -> Result<(), String> {
    let source = r#"block stage(clk, x: bool) -> bool {
    reg q: bool = true;
    let w: bool;
    let (known, other) = (true, false);
    const_assert!(PREDICATE);
    let later = true;
    w = x;
    q := w;
    q
}"#;
    for (predicate, expected) in [
        ("known && !other", None),
        ("if true { known } else { q }", None),
        ("false", Some("predicate is false")),
        ("x", Some("compile-time constant")),
        ("q", Some("compile-time constant")),
        ("w", Some("compile-time constant")),
        ("later", Some("unknown value")),
        ("u8:1", Some("bool")),
    ] {
        let checked = check_module(&parse_module(&source.replace("PREDICATE", predicate))?);
        match expected {
            Some(reason) => {
                let error = checked.unwrap_err();
                assert!(
                    error.contains("5:5: const_assert!") && error.contains(reason),
                    "{predicate}: {error}"
                );
            }
            None => {
                checked?;
            }
        }
    }
    for bad in [
        "const_assert!(true,);",
        "const_assert!(true)",
        "let b = const_assert!(true);",
    ] {
        assert!(
            parse_module(&source.replace("const_assert!(PREDICATE);", bad)).is_err(),
            "{bad}"
        );
    }
    Ok(())
}

#[test]
fn assertion_rechecks_dependencies_of_a_block_let_annotation() -> Result<(), String> {
    let source = r#"
block stage(clk, x: bool) -> bool {
    let width = { let n = u32:3; x; n };
    let word: uN[width] = 0;
    const_assert!(PREDICATE);
    reg q: bool;
    q := x;
    q
}
"#;
    check_module(&parse_module(&source.replace("PREDICATE", "true"))?)?;
    let error =
        check_module(&parse_module(&source.replace("PREDICATE", "word == u3:0"))?).unwrap_err();
    assert!(
        error.contains("5:5: const_assert!") && error.contains("compile-time constant"),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_direct_assertion_cannot_use_a_child_as_a_constexpr_helper() -> Result<(), String> {
    let child = "block child(clk, x: bool) -> bool { reg q: bool; q := x; q }";
    for predicate in [
        "child(true)",
        "unroll_for! (i,a) in u32:0..u32:0 { let c = child(a); c }(true)",
    ] {
        let source =
            format!("{child}\nblock stage(clk, x:bool)->bool {{ const_assert!({predicate}); x }}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(
            error.contains("const_assert!") && error.contains("child"),
            "{error}"
        );
    }
    for declaration in ["stage", "stage<N: u32>"] {
        let source = format!(
            "block {declaration}(clk, x:bool)->bool {{ reg q:bool; if x {{ const_assert!(true); q := x; }} q }}"
        );
        let error = parse_module(&source).unwrap_err();
        assert!(
            error.contains("const_assert! requires the top level of a block body"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn library_block_assertions_resolve_private_helpers_in_the_defining_module()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(directory.path().join("limits.x"), "pub const LOW = u32:2;")?;
    let library = r#"
#![require_define_before_use]
use limits::LOW;
fn accepts(n: u32) -> bool { n >= LOW }
pub block stage<N: u32>(clk, x: u8) -> u8 {
    const_assert!(accepts(N));
    reg q: u8;
    q := x;
    q
}
"#;
    fs::write(directory.path().join("stages.bx"), library)?;
    let options = ImportOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    for (depth, valid) in [(2, true), (1, false)] {
        let root = format!(
            "use stages::stage; fn accepts(n:u32)->bool {{ true }} pub block pipeline(clk, x:u8)->u8 {{ let y=stage<{depth}>(x); y }}"
        );
        let compiled = compile_with_options(
            &root,
            empty_package("design"),
            ResetBehavior::default(),
            &options,
        );
        if valid {
            let actual = compiled?;
            let baseline = library.replace("const_assert!(accepts(N));", "");
            fs::write(directory.path().join("stages.bx"), baseline)?;
            let expected = compile_with_options(
                &root,
                empty_package("design"),
                ResetBehavior::default(),
                &options,
            )?;
            assert_eq!(actual.to_string(), expected.to_string());
            fs::write(directory.path().join("stages.bx"), library)?;
        } else {
            assert!(
                compiled
                    .unwrap_err()
                    .contains("6:5: const_assert!: predicate is false")
            );
        }
    }
    let helper = "fn accepts(n: u32) -> bool { n >= LOW }";
    let reordered = format!("{}\n{helper}", library.replace(helper, ""));
    fs::write(directory.path().join("stages.bx"), reordered)?;
    let error = compile_with_options(
        "use stages::stage; block pipeline(clk,x:u8)->u8 { let y=stage<2>(x); y }",
        empty_package("design"),
        ResetBehavior::default(),
        &options,
    )
    .unwrap_err();
    assert!(
        error.contains("declared later") && error.contains("accepts"),
        "{error}"
    );
    Ok(())
}
