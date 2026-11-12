// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, empty_package, parse_module, run_tests,
};

fn check(source: &str) -> Result<(), String> {
    check_module(&parse_module(source)?)?;
    Ok(())
}

const PULSE: &str = include_str!("fixtures/issue_latency.bx");

#[test]
fn checks_each_declared_latency_with_other_clocks_free() -> Result<(), String> {
    let report = run_tests(
        PULSE,
        Path::new("issue_latency.bx"),
        empty_package("suite"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].failure, None);
    let err = check(&PULSE.replace("long_qq }", "long_q }")).unwrap_err();
    assert!(
        err.contains("op=Op::Slow, valid=true") && err.contains("output at +2"),
        "{err}"
    );
    Ok(())
}

#[test]
fn registered_mux_select_keeps_issue_payload_aligned() -> Result<(), String> {
    let source = r#"
fn latency(invert: bool) -> u32 { u32:1 }
block route(clk, invert: bool, data: u8) -> u8 @latency(invert) {
    reg invert_q: bool;
    reg data_q: u8;
    invert_q := invert;
    data_q := data;
    if invert_q { !data_q } else { data_q }
}
"#;
    check(source)?;
    for bad in [
        source.replace("if invert_q", "if invert"),
        source.replace("data_q := data;", "if invert { data_q := data; }"),
        source.replace("u32:1", "u32:2"),
    ] {
        assert!(check(&bad).unwrap_err().contains("cannot prove the output"));
    }
    Ok(())
}

#[test]
fn separate_cases_do_not_assume_an_issuer_spacing_rule() {
    let source = format!(
        r#"{PULSE}
block shared(clk, valid: bool, op: Op, data: u8) -> u8 @latency(valid, op) {{
    reg short_q: bool;
    reg d1: u8;
    reg d2: u8;
    short_q := if valid {{ op == Op::Fast }} else {{ true }};
    d1 := data;
    d2 := d1;
    if short_q {{ d1 }} else {{ d2 }}
}}
"#
    );
    let err = check(&source).unwrap_err();
    assert!(
        err.contains("op=Op::Slow, valid=true") && err.contains("output at +2"),
        "{err}"
    );
}

#[test]
fn bounds_domains_and_latency_before_unrolling() {
    for (source, diagnostic) in [
        (
            "fn f(x:u8)->u32 { u32:1 } block b(clk, x:u8)->u8 @f(x) { x }",
            "exceeds 64 cases",
        ),
        (
            "fn f(x:u4,y:u4)->u32 { u32:1 } block b(clk,x:u4,y:u4)->u4 @f(x,y) { x }",
            "1 to 64 cases",
        ),
        (
            "fn f(x:bool)->u32 { u32:65 } block b(clk,x:bool)->bool @f(x) { x }",
            "64-clock proof limit",
        ),
        (
            "fn f(x:bool)->s32 { s32:1 } block b(clk,x:bool)->bool @f(x) { x }",
            "unsigned integer",
        ),
        (
            "fn f(x:bool)->u32 { u32:0 } block b(clk,x:u2)->u2 @f(x) { x }",
            "expects bool",
        ),
        (
            "fn f(x:bool)->u32 { u32:0 } block b(clk,x:bool)->bool @f(missing) { x }",
            "unknown issue input",
        ),
    ] {
        let err = check(source).unwrap_err();
        assert!(err.contains(diagnostic), "{err}");
    }
}

#[test]
fn dynamic_child_has_no_single_fixed_stage_for_parent() {
    let source = r#"
fn latency(op:bool)->u32 { u32:0 }
block child(clk, op:bool, x:u8)->u8 @latency(op) { x }
block parent(clk, op:bool, x:u8)->u8 @0 { let value = child(op, x); value }
"#;
    assert!(
        check(source)
            .unwrap_err()
            .contains("cannot infer output cycle")
    );
}
