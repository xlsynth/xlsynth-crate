// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::path::Path;

use xlsynth_bx::{
    ImportOptions, QuickCheckOptions, ResetBehavior, compile, empty_package, parse_module,
    run_tests_with_quickcheck,
};

#[test]
fn invariant_declarations_are_private_concrete_blocks() -> Result<(), String> {
    for invalid in [
        "#[invariant] pub block prop(clk) -> bool { true }",
        "#[invariant] fn prop() -> bool { true }",
        "#[invariant] block prop(clk) -> u8 { u8:0 }",
        "#[invariant] block prop<T>(clk, x: T) -> bool { true }",
        "#[invariant] block prop(clk) -> bool { #[lemma] let _ = true; true }",
        "#[invariant] block prop(clk) -> bool { #[lemma] let (p, q) = (true, true); p }",
        "#[invariant] block prop(clk) -> bool { #[lemma] let p = true; #[lemma] let p = false; p }",
        "#[invariant] block prop(clk) -> bool { if true { #[lemma] let p = true; } true }",
        "block dut(clk) -> bool { #[lemma] let p = true; p }",
        "#[invariant] block prop(clk) -> bool { true } #[test] fn prop() {}",
        "#[invariant] block prop(clk) -> bool { true } fn prop() -> bool { true }",
    ] {
        assert!(parse_module(invalid).is_err(), "{invalid}");
    }
    let source = "#[invariant] block prop(clk) -> bool { #[lemma] let fact = true; fact }";
    let module = parse_module(source)?;
    assert!(module.blocks.is_empty());
    assert_eq!(module.invariants[0].lemmas[0].name, "fact");
    let hardware = compile(source, empty_package("dut"), ResetBehavior::default())?;
    assert!(hardware.members.is_empty());
    for callee in ["prop", "bx_test_prop"] {
        let bad = format!("{source} block caller(clk) -> bool {{ let o = {callee}(); o }}");
        assert!(compile(&bad, empty_package("dut"), ResetBehavior::default()).is_err());
    }
    Ok(())
}

#[test]
fn invariant_name_leaves_builtins_visible_in_its_body() -> Result<(), String> {
    let source = "#[invariant] block bit_count(clk) -> bool { bit_count<u8>() == u32:8 }";
    let report = run_tests_with_quickcheck(
        source,
        Path::new("builtin.bx"),
        empty_package("builtin"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
        QuickCheckOptions {
            trials: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
    )?;
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}

#[test]
fn invariant_lemmas_capture_the_marked_binding_through_shadowing() -> Result<(), String> {
    let source = r#"
#[invariant]
block scoped(clk, out_0: bool) -> bool {
    #[lemma] let live = false;
    let (live, _) = (true, u8:0);
    let out_1 = out_0;
    live || out_1
}
"#;
    let report = run_tests_with_quickcheck(
        source,
        Path::new("scope.bx"),
        empty_package("scope"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
        QuickCheckOptions {
            trials: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
    )?;
    assert_eq!(report.tests.len(), 1);
    let failure = report.tests[0]
        .failure
        .as_deref()
        .expect("marked binding was false");
    assert!(failure.contains("live"), "{failure}");
    Ok(())
}

#[test]
fn root_invariant_resolves_library_helpers_and_excludes_library_properties() -> Result<(), String> {
    let temp = tempfile::tempdir().map_err(|e| e.to_string())?;
    std::fs::write(
        temp.path().join("lib.bx"),
        r#"
pub fn equal(a: bool, b: bool) -> bool { a == b }
#[invariant] block library_only(clk) -> bool { false }
"#,
    )
    .map_err(|e| e.to_string())?;
    let source = r#"
#![require_define_before_use]
use lib;
type Predicate = bool;
#[invariant] block compared(clk, x: bool) -> Predicate {
    #[lemma] let fact: Predicate = lib::equal(x, x);
    fact
}
"#;
    let report = run_tests_with_quickcheck(
        source,
        &temp.path().join("main.bx"),
        empty_package("root"),
        ResetBehavior::default(),
        &ImportOptions {
            additional_search_paths: vec![temp.path()],
            ..Default::default()
        },
        true,
        QuickCheckOptions {
            trials: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
    )?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].name, "compared");
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}

#[test]
fn invariant_result_unrolls_clocked_children_in_structural_context() -> Result<(), String> {
    let source = r#"
block delay(clk, value: bool) -> bool {
    reg q: bool = true;
    q := value;
    q
}
#[test] fn companion() { assert!(true); }
#[invariant] block composed(clk, rst) -> bool {
    #[lemma] let fact = true;
    unroll_for! (_, prev): (u32, bool) in u32:0..u32:2 {
        let next_stage = delay(prev);
        next_stage
    }(fact)
}
"#;
    let report = run_tests_with_quickcheck(
        source,
        Path::new("unroll.bx"),
        empty_package("unroll"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
        QuickCheckOptions {
            trials: NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
    )?;
    assert_eq!(report.tests.len(), 2);
    for test in report.tests {
        assert_eq!(test.failure, None);
    }
    Ok(())
}

#[test]
fn invariant_must_declare_reset_for_initializers_and_resetful_children() {
    for (body, diagnostic) in [
        (
            "reg q: bool = true; q := true; q",
            "initializer on register 'q' requires a reset",
        ),
        (
            "let q = child(); q",
            "instance 'q' of 'child' requires a parent reset",
        ),
    ] {
        let source = format!(
            r#"block child(clk) -> bool {{ reg q: bool = true; q := true; q }}
#[invariant] block prop(clk) -> bool {{ {body} }}"#
        );
        let error = run_tests_with_quickcheck(
            &source,
            Path::new("reset.bx"),
            empty_package("reset"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
            QuickCheckOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error, format!("reset.bx: invariant 'prop': {diagnostic}"));
    }
}

#[test]
fn invariant_timing_annotation_checks_only_the_scalar_property() -> Result<(), String> {
    let check = |annotation: &str| {
        let source = format!(
            r#"
#[invariant] block timing(clk, rst, x: bool @0) -> bool {annotation} {{
    reg delayed: bool = true;
    delayed := x || !x;
    #[lemma] let delayed_is_true = delayed;
    x || !x
}}"#
        );
        run_tests_with_quickcheck(
            &source,
            Path::new("timing.bx"),
            empty_package("timing"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
            QuickCheckOptions {
                trials: NonZeroUsize::new(1).unwrap(),
                ..Default::default()
            },
        )
    };
    for annotation in ["@0", "@(out: 0)"] {
        assert_eq!(check(annotation)?.tests[0].failure, None);
    }
    assert_eq!(
        check("@(out: 1)").unwrap_err(),
        "timing.bx: invariant 'timing': output 'out': expected cycle 1, got cycle 0"
    );
    Ok(())
}
