// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::path::Path;
use xlsynth_pir::ir_parser::parse_xls_ir_package;

use super::{QuickCheckOptions, TestSuite, run_tests_with_quickcheck};
use crate::{ImportOptions, ResetBehavior, bx_import, check};

const HELPERS_AND_PURE_TEST: &str = r#"
extern fn lookup(xs: u8[2], i: u8) -> u8;
fn identity(x: u8) -> u8 { x }
fn keep(x: u8) -> u8 {
    let (_, y) = (lookup([x, x], x), identity(x));
    y
}
fn forward(x: u8) -> u8 { keep(x) }
block inner(clk, x: u8) -> u8 {
    reg q: u8;
    q := forward(x);
    q
}
block outer(clk, x: u8) -> u8 { let q = inner(x); q }
#[test] fn pure() { assert_eq(keep(u8:0), u8:0); }
"#;

const NESTED_QUICKCHECK: &str = r#"
#[quickcheck] block nested(x: u8) -> bool {
    let d = x | u8:2;
    let q = outer(d);
    tick!();
    q == d
}
"#;

#[test]
fn clocked_lowering_depends_on_reachable_functions_with_discards() -> Result<(), String> {
    for (clocked_test, needs_clocked_package) in [
        (
            "#[test] block disjoint() { assert_eq(identity(u8:7), u8:7); }",
            false,
        ),
        (
            "#[test] block direct() { assert_eq(keep(u8:7), u8:7); }",
            true,
        ),
        (
            "#[quickcheck] block zero_trip(x: u8) -> bool { (for (_, a) in u8:0..u8:0 { keep(a) }(x)) == x }",
            false,
        ),
        (NESTED_QUICKCHECK, true),
    ] {
        let source = format!("{HELPERS_AND_PURE_TEST}{clocked_test}");
        let options = ImportOptions::default();
        let (root, imports) = bx_import::read_for_tests(&source, &options)?;
        let mut suite = TestSuite {
            reserved: Default::default(),
            programs: Vec::new(),
        };
        let checked = check::check_sources_with_tests(&root, imports, &options, Some(&mut suite))?;
        assert!(checked.functions_with_test_discards.contains("keep"));
        assert_eq!(
            suite
                .pure_harnesses_for_clocked_lowering(&checked)
                .is_some(),
            needs_clocked_package,
            "{clocked_test}"
        );
    }
    Ok(())
}

#[test]
fn nested_quickcheck_skips_a_contract_checked_by_the_pure_helper() -> Result<(), String> {
    let source = format!("{HELPERS_AND_PURE_TEST}{NESTED_QUICKCHECK}");
    // The pure test uses index zero. Every clocked input sets bit one, which
    // would violate the bounds assumption if it used the pure test lowering.
    let package = parse_xls_ir_package(
        r#"package mixed
fn lookup(xs: bits[8][2] id=1, i: bits[8] id=2) -> bits[8] {
    ret selected: bits[8] = array_index(xs, indices=[i], assumed_in_bounds=true, id=3)
}"#,
    )
    .map_err(|e| e.to_string())?;
    let report = run_tests_with_quickcheck(
        &source,
        Path::new("mixed.bx"),
        package,
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
        QuickCheckOptions {
            trials: NonZeroUsize::MIN,
            cross_check_vastly: true,
            ..Default::default()
        },
    )?;
    let [pure, nested] = report.tests.as_slice() else {
        panic!("expected a pure test and a clocked quickcheck");
    };
    assert_eq!(pure.failure, None);
    assert_eq!(nested.failure, None);
    let quickcheck = nested.quickcheck.as_ref().unwrap();
    assert_eq!(quickcheck.trials, 1);
    assert_eq!(quickcheck.four_valued.as_ref().unwrap().passed, 1);
    Ok(())
}
