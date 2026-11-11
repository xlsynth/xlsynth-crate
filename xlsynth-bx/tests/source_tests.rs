// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{error::Error, fs, path::Path};
use support::search_paths;

use tempfile::tempdir;

use xlsynth_bx::{BxTestReport, ImportOptions, ResetBehavior, compile, empty_package, run_tests};
use xlsynth_pir::ir_parser::parse_xls_ir_package;

/// Run a self-contained source file with the default synchronous clock domain.
fn run(source: &str) -> Result<BxTestReport, String> {
    run_tests(
        source,
        Path::new("suite.bx"),
        empty_package("suite"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )
}

#[test]
fn one_edge_advances_all_children_and_procedural_lets_take_snapshots() -> Result<(), String> {
    let source = r#"
block delay<T>(clk, rst, x: T) -> T {
    reg q: T = zero!<T>();
    q := x;
    q
}
#[test] block pipeline() {
    reg x: u8;
    let first = delay(x);
    let second = delay(first);
    tick!(rst: true);
    for i in u8:1..=u8:4 {
        x := i;
        let before = first;
        assert_eq(before, i - u8:1);
        tick!();
        assert_eq(second, before);
        assert_eq(first, i);
    }
}"#;
    let report = run(source)?;
    assert_eq!(report.tests.len(), 1);
    assert_eq!(report.tests[0].failure, None);
    let hardware = compile(source, empty_package("hardware"), ResetBehavior::default())?;
    // The only synthesizable declaration is generic and the test is its only
    // caller.
    assert!(hardware.members.is_empty());
    Ok(())
}

#[test]
fn pure_tests_run_scoped_table_loops_and_report_the_executed_assertion() -> Result<(), String> {
    let source = r#"fn bx_test_pure(x: u4) -> u4 { x + u4:1 }
#[test] fn pure() {
    for (a, b) in [(u4:2, u4:3), (u4:15, u4:0)] {
        let inc = bx_test_pure(a);
        if b == u4:0 { assert_eq(inc, u4:0); }
        else { assert_eq(inc, b); }
    }
    for i in u8:4..u8:4 { assert!(false, "empty range"); }
}
#[test] fn scoped_failure() {
    let x = u8:1;
    if false { assert!(false, "skipped"); }
    if true { let x = u8:2; assert_eq(x, u8:2); }
    assert!(x == u8:3, "outer binding")
}"#;
    let report = run(source)?;
    assert_eq!(report.tests.len(), 2);
    assert_eq!(report.tests[0].failure, None);
    assert_eq!(
        report.tests[1].failure.as_deref(),
        Some("suite.bx:14:5: test 'scoped_failure': assertion failed: outer binding")
    );
    Ok(())
}

#[test]
fn clocked_test_preserves_procedural_constants_as_range_bounds() -> Result<(), String> {
    let source = r#"#[test] block count() {
    reg edges: u8;
    tick!();
    let base = u8:2;
    if true {
        let (start, (count, previous)) = (u8:0, (base + u8:1, edges));
        edges := u8:4;
        for _ in start..count {
            edges := edges + u8:1;
            tick!();
            assert_eq(previous, u8:0);
        }
    }
    assert_eq(edges, u8:7);
}"#;
    let report = run(source)?;
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}

#[test]
fn discarded_runtime_captures_do_not_change_constant_range_bounds() -> Result<(), String> {
    let source = r#"fn pick(x: u8) -> u8 {
    let start = for (i, a) in u8:0..u8:1 {
        let values = map([a], |word| { let (_, kept) = (x + i, word); kept });
        values[u8:0]
    }(u8:0);
    for (i, a) in start..u8:3 { a + i }(u8:0)
}
#[test] block smoke() { tick!(); }
#[test] fn pure() { assert_eq(pick(u8:42), u8:3); }"#;
    let report = run(source)?;
    assert_eq!(report.tests.len(), 2);
    for test in report.tests {
        assert_eq!(test.failure, None, "{}", test.name);
    }
    Ok(())
}

#[test]
fn only_pure_test_callees_link_discarded_fields() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("lib.bx"),
        r#"extern fn missing(x: u8) -> u8;
fn observe(x: u8) -> u8 { let (_, kept) = (missing(x), x); kept }
pub fn keep(x: u8) -> u8 { let (_, kept) = (observe(x), x); kept }
#[test] fn library_test() { assert_eq(keep(u8:0), u8:0); }"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let source = r#"use lib::keep;
fn unused(x: u8) -> u8 { keep(x) }
#[test] block clocked() { assert_eq(keep(u8:3), u8:3); }
#[test] fn zero_trip() {
    assert_eq(for (_, a) in u8:0..u8:0 { keep(a) }(u8:7), u8:7);
}"#;
    let run = |source: &str| {
        run_tests(
            source,
            Path::new("suite.bx"),
            empty_package("suite"),
            ResetBehavior::default(),
            &options,
            true,
        )
    };
    let report = run(source)?;
    assert_eq!(report.tests.len(), 2);
    for test in report.tests {
        assert_eq!(test.failure, None, "{}", test.name);
    }
    let reachable =
        format!("{source}\n#[test] fn reaches_helper() {{ assert_eq(unused(u8:3), u8:3); }}");
    assert_eq!(
        run(&reachable).unwrap_err(),
        "suite.bx: extern fn 'missing': no linked IR function for '__suite__missing'"
    );
    Ok(())
}

#[test]
fn pure_test_skips_work_in_untaken_branches() -> Result<(), String> {
    let source = r#"fn sum(x: u32) -> u32 {
    for (i, a) in u32:0..u32:1000000 { a + i }(x)
}
#[test] fn skipped_fold() {
    for i in u32:0..u32:4 {
        if i > u32:3 { assert_eq(sum(i), u32:0); }
        assert!(i <= u32:3);
    }
}"#;
    let report = run(source)?;
    assert_eq!(report.tests[0].failure, None);
    Ok(())
}

#[test]
fn pure_tests_report_linked_bounds_failures_at_the_executed_statement() -> Result<(), String> {
    let package = parse_xls_ir_package(
        r#"package linked
fn lookup(a: bits[8][2] id=1, i: bits[8] id=2) -> bits[8] {
    ret selected: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=3)
}"#,
    )
    .map_err(|e| e.to_string())?;
    let source = r#"extern fn lookup(a: u8[2], i: u8) -> u8;
const DATA: u8[2] = [u8:17, u8:29];
#[test] fn out_of_bounds() {
    for i in [u8:1, u8:2] {
        assert_eq(lookup(DATA, i), u8:29);
    }
}
#[test] fn unused_binding() {
    let _ = lookup(DATA, u8:2);
    assert!(true);
}
#[test] fn valid() {
    if false { let _ = lookup(DATA, u8:2); }
    assert_eq(lookup(DATA, u8:0), u8:17);
    for i in [u8:2] { assert_eq(DATA[i], u8:29); }
}
#[test] fn nested_unused_binding() {
    assert_eq({ let _ = lookup(DATA, u8:2); u8:7 }, u8:7);
}
#[test] fn condition_scope() {
    if { let _ = lookup(DATA, u8:2); false } { assert!(false); }
}
#[test] fn iterable_scope() {
    for _ in { let _ = lookup(DATA, u8:2); DATA } { assert!(true); }
}
#[test] fn skipped_scope() {
    if false { assert_eq({ let _ = lookup(DATA, u8:2); u8:7 }, u8:7); }
    assert_eq({ let value = lookup(DATA, u8:1); value }, u8:29);
}
fn last_item(index: u8) -> u8 {
    let (.., last) = (u8:0, lookup(DATA, index), u8:7);
    last
}
#[test] fn nested_wildcard() {
    assert_eq({ let (v, (_, w)) = (u8:7, (lookup(DATA, u8:2), u8:1)); v + w }, u8:8);
}
#[test] fn helper_rest() {
    assert_eq(last_item(u8:2), u8:7);
}
#[test] fn skipped_destructure() {
    if false {
        assert_eq({ let (_, v) = (lookup(DATA, u8:2), u8:7); v }, u8:7);
        assert_eq(last_item(u8:2), u8:7);
    }
    assert_eq(last_item(u8:1), u8:7);
}"#;
    let report = run_tests(
        source,
        Path::new("suite.bx"),
        package,
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    assert_eq!(report.tests.len(), 10);
    assert_eq!(
        report.tests[0].failure.as_deref(),
        Some(
            "suite.bx:5:9: test 'out_of_bounds': IR array_index at node 3 was out of bounds despite assumed_in_bounds=true"
        )
    );
    assert_eq!(
        report.tests[1].failure.as_deref(),
        Some(
            "suite.bx:9:5: test 'unused_binding': IR array_index at node 3 was out of bounds despite assumed_in_bounds=true"
        )
    );
    assert_eq!(report.tests[2].failure, None);
    for (index, line, name) in [
        (3, 18, "nested_unused_binding"),
        (4, 21, "condition_scope"),
        (5, 24, "iterable_scope"),
        (7, 35, "nested_wildcard"),
        (8, 38, "helper_rest"),
    ] {
        assert_eq!(
            report.tests[index].failure,
            Some(format!(
                "suite.bx:{line}:5: test '{name}': IR array_index at node 3 was out of bounds despite assumed_in_bounds=true"
            ))
        );
    }
    assert_eq!(report.tests[6].failure, None);
    assert_eq!(report.tests[9].failure, None);
    Ok(())
}

#[test]
fn test_checker_reports_misuse_at_the_statement_even_in_empty_loops() {
    for (source, expected) in [
        (
            r#"#[test] block example() {
    for _ in u8:0..u8:0 { missing := u8:3; }
}"#,
            "suite.bx:2:27: test 'example': 'missing' is not a stimulus register",
        ),
        (
            r#"#[test] block example() {
    tick!();
    reg drive: u8;
}"#,
            "suite.bx:3:5: test 'example': declare stimulus registers in the initial block test setup",
        ),
    ] {
        assert_eq!(run(source).unwrap_err(), expected);
    }
}

#[test]
fn quickcheck_checks_parameter_types_and_trailing_predicates_only_in_test_mode()
-> Result<(), String> {
    for (source, expected) in [
        (
            "#[quickcheck] block p(x: u8, x: u16) -> bool { true }",
            "suite.bx:1:21: quickcheck 'p': duplicate name 'x'",
        ),
        (
            "#[quickcheck] block p(x: u8) -> bool {\n    x := u8:0;\n    true\n}",
            "suite.bx:2:5: test 'p': 'x' is not a stimulus register",
        ),
        (
            "#[quickcheck] block p(x: u8) -> bool {\n    tick!();\n    x\n}",
            "suite.bx:3:5: test 'p': expected bool, got u8",
        ),
        (
            "type Word = u8;\n#[quickcheck] block p() -> Word { true }",
            "suite.bx:2:21: quickcheck 'p': BX quickcheck blocks must return bool, got u8",
        ),
    ] {
        let hardware = compile(source, empty_package("hardware"), ResetBehavior::default())?;
        assert!(hardware.members.is_empty());
        assert_eq!(run(source).unwrap_err(), expected);
    }
    Ok(())
}

#[test]
fn quickcheck_accepts_local_and_imported_predicate_type_aliases() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("types.bx"), "pub type Flag = bool;")?;
    fs::write(dir.path().join("native.x"), "pub type Flag = bool;")?;
    let source = r#"
use types;
use native;
type Pred = bool;
#[quickcheck] block local(v: Pred) -> Pred { v || !v }
#[quickcheck] block imported(v: types::Flag) -> types::Flag { v || !v }
#[quickcheck] block dslx(v: native::Flag) -> native::Flag { v || !v }
"#;
    let report = run_tests(
        source,
        Path::new("suite.bx"),
        empty_package("suite"),
        ResetBehavior::default(),
        &search_paths(&[dir.path()]),
        true,
    )?;
    assert_eq!(report.tests.len(), 3);
    for test in report.tests {
        assert_eq!(test.failure, None, "{}", test.name);
    }
    Ok(())
}
