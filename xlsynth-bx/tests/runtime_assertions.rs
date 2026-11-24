// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs, num::NonZeroUsize, path::Path};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, QuickCheckOptions, ResetBehavior, compile, empty_package,
    run_tests_with_quickcheck,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::{
    IrValue,
    block_eval::{BlockEvalError, BlockEvaluator, RegisterInit},
};

#[test]
fn clockless_block_checks_each_enabled_assertion_and_preserves_its_output()
-> Result<(), Box<dyn Error>> {
    let source = r#"pub block inspect(valid: bool, value: u8) -> u8 {
    if valid {
        assert!(value != u8:0, "nonzero_payload");
        if value != u8:1 { assert!(value < u8:8); }
    }
    value
}"#;
    let package = compile(source, empty_package("checks"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let eval = BlockEvaluator::prepare(&package, "inspect")?;
    let state = eval.initial_state(RegisterInit::AllZeros)?;
    for (valid, data, failure) in [
        (false, 0, None),
        (false, 8, None),
        (true, 1, None),
        (true, 7, None),
        (true, 0, Some("assertion failed at 3:9")),
        (true, 8, Some("assertion failed at 4:28")),
    ] {
        let value = IrValue::make_ubits(8, data)?;
        let inputs = [IrValue::bool(valid), value.clone()];
        match (eval.step(&state, &inputs), failure) {
            (Ok(cycle), None) => assert_eq!(cycle.outputs, [value]),
            (Err(BlockEvalError::Runtime(err)), Some(expected)) => {
                assert_eq!(err.assertion_failures.len(), 1);
                assert_eq!(err.assertion_failures[0].message, expected);
            }
            (result, expected) => {
                panic!("valid={valid}, data={data}: got {result:?}, expected {expected:?}")
            }
        }
    }
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    assert_eq!(
        rtl.system_verilog
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("always_comb begin"))
            .collect::<Vec<_>>(),
        ["always_comb begin : nonzero_payload", "always_comb begin"]
    );
    Ok(())
}

#[test]
fn block_ir_masks_assertions_during_its_own_physical_reset() -> Result<(), Box<dyn Error>> {
    let source = r#"pub block hold(clk, enable: bool) -> bool {
    reg q: bool = false;
    assert!(enable, "enable_required");
    q := enable;
    q
}"#;
    for active_low in [false, true] {
        let package = compile(
            source,
            empty_package("checks"),
            ResetBehavior {
                asynchronous: false,
                active_low,
            },
        )?;
        let eval = BlockEvaluator::prepare(&package, "hold")?;
        let state = eval.initial_state(RegisterInit::AllZeros)?;
        let inputs = |reset: bool| [IrValue::bool(reset ^ active_low), IrValue::bool(false)];
        let reset = eval.step(&state, &inputs(true))?;
        let err = eval.step(&reset.next_state, &inputs(false)).unwrap_err();
        let BlockEvalError::Runtime(failure) = err else {
            panic!("{err}");
        };
        let assertion = &failure.assertion_failures[0];
        assert_eq!(assertion.label, "enable_required");
        assert_eq!(assertion.message, "assertion failed at 3:5");
        let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
        assert!(
            rtl.system_verilog
                .contains("enable_required: assert property (@(posedge clk)")
        );
    }
    Ok(())
}

#[test]
fn imported_child_assertions_sample_before_each_unmasked_test_edge() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("lib.bx"),
        r#"const RESERVED = u8:255;
fn allowed(x: u8) -> bool { x != RESERVED }
pub block stage<T>(clk, x: T) -> T {
    reg q: T = zero!<T>();
    assert!(allowed(q as u8), "reserved_payload");
    q := x;
    q
}"#,
    )?;
    let source = r#"use lib::stage;
#[test] block unit() {
    reg x: u8;
    let out = stage(x);
    tick!(rst: true);
    x := u8:255;
    tick!();
    assert_eq(out, u8:255);
    tick!(rst: true);
    assert_eq(out, u8:0);
    tick!();
    tick!();
}
#[quickcheck] block property() -> bool {
    let out = stage(u8:255);
    tick!(rst: true);
    tick!();
    tick!();
    out == u8:255
}"#;
    for active_low in [false, true] {
        let report = run_tests_with_quickcheck(
            source,
            Path::new("checks.bx"),
            empty_package("checks"),
            ResetBehavior {
                asynchronous: false,
                active_low,
            },
            &ImportOptions {
                additional_search_paths: vec![dir.path()],
                ..Default::default()
            },
            true,
            QuickCheckOptions {
                trials: NonZeroUsize::MIN,
                cross_check_vastly: true,
                ..Default::default()
            },
        )?;
        assert_eq!(report.tests.len(), 2);
        assert_eq!(
            report.tests[0].failure.as_deref(),
            Some(
                "checks.bx:12:5: test 'unit' at tick 4: IR assertion 'reserved_payload' failed: assertion failed at lib.bx:5:5"
            )
        );
        assert_eq!(
            report.tests[1]
                .failure
                .as_deref()
                .and_then(|s| s.lines().next()),
            Some(
                "checks.bx:18:5: test 'property' at tick 2: IR assertion 'reserved_payload' failed: assertion failed at lib.bx:5:5"
            )
        );
        let four_valued = report.tests[1]
            .quickcheck
            .as_ref()
            .unwrap()
            .four_valued
            .as_ref()
            .unwrap();
        assert_eq!(four_valued.failed, 1);
    }
    Ok(())
}
