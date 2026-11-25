// SPDX-License-Identifier: Apache-2.0

//! Check BX examples with both the existing checker and the new IR checker.

use crate::ast::{OutputCycles, StageAnnotation};
use crate::ir_timing::{CheckResult, Contract, check_block, register_cycle};
use crate::{ResetBehavior, check_module, empty_package, lower_module, parse_module};

/// Temporarily remove `@N` so BX will lower even an example that should fail
/// timing checking. Then ask the IR checker to check the original `@N`.
fn check_ir(source: &str) -> Result<CheckResult, String> {
    check_ir_with_reset(source, ResetBehavior::default())
}

/// Lower the example with a specific physical reset polarity and behavior.
fn check_ir_with_reset(source: &str, reset: ResetBehavior) -> Result<CheckResult, String> {
    let mut parsed = parse_module(source)?;
    let Some(OutputCycles::Uniform(StageAnnotation::Literal(expected))) =
        parsed.blocks[0].output_cycles.take()
    else {
        return Err("fixture requires a uniform literal output cycle".into());
    };
    let checked = check_module(&parsed)?;
    let package = lower_module(&checked, empty_package("timing"), reset)?;
    let block = &checked.blocks[0];
    let ir = package
        .get_block(&block.name)
        .ok_or_else(|| format!("missing lowered block '{}'", block.name))?;
    check_block(
        ir,
        &Contract {
            input_cycles: &block.input_cycles,
            output_cycles: &OutputCycles::Uniform(expected),
        },
    )
}

#[test]
fn accepts_aligned_operands_and_constants() -> Result<(), String> {
    let source = "block add(x: u32 @2, y: u32 @2) -> u32 @2 { x + y + u32:1 }";
    check_module(&parse_module(source)?)?;
    assert_eq!(check_ir(source)?, CheckResult::Checked);
    Ok(())
}

#[test]
fn rejects_the_wrong_output_cycle() -> Result<(), String> {
    let source = "block add(x: u32 @0) -> u32 @1 { x + u32:1 }";
    assert!(check_module(&parse_module(source)?).is_err());
    assert_eq!(
        check_ir(source).unwrap_err(),
        "output 'out' is at stage 0, expected stage 1"
    );
    Ok(())
}

#[test]
fn leaves_unannotated_inputs_unknown() -> Result<(), String> {
    let source = "block passthrough(x: u32) -> u32 @1 { x }";
    assert!(check_module(&parse_module(source)?).is_err());
    assert_eq!(check_ir(source)?, CheckResult::Unsupported);
    Ok(())
}

#[test]
fn checks_mismatched_operands_even_when_the_result_is_unused() -> Result<(), String> {
    let used = "block add(x: u32 @0, y: u32 @1) -> u32 @0 { x + y }";
    let unused = r#"block add(x: u32 @0, y: u32 @1) -> u32 @0 {
    let unused = x + y;
    x
}"#;
    assert!(check_module(&parse_module(used)?).is_err());
    // BX currently accepts the unused addition because the output is just x.
    // The IR checker should reject the bad addition whether it is used or not.
    check_module(&parse_module(unused)?)?;
    for source in [used, unused] {
        assert_eq!(
            check_ir(source).unwrap_err(),
            "timing mismatch at add.3: operands are at stages 0 and 1"
        );
    }
    Ok(())
}

#[test]
fn cannot_certify_an_unused_operation_with_unknown_timing() -> Result<(), String> {
    // The output x is fine, but y has no cycle. We cannot prove x + y is OK.
    let source = r#"block add(x: u32 @0, y: u32) -> u32 @0 {
    let unused = x + y;
    x
}"#;
    check_module(&parse_module(source)?)?;
    assert_eq!(check_ir(source)?, CheckResult::Unsupported);
    Ok(())
}

#[test]
fn constant_output_can_satisfy_any_cycle() -> Result<(), String> {
    // Returning 42 does not require seven registers: it is always available.
    let source = "block constant() -> u32 @7 { u32:42 }";
    check_module(&parse_module(source)?)?;
    assert_eq!(check_ir(source)?, CheckResult::Checked);
    Ok(())
}

#[test]
fn checks_an_ordinary_register() -> Result<(), String> {
    // Both checkers should see q one cycle after x.
    let source = "block delay(clk, x: u32 @0) -> u32 @1 { reg q: u32; q := x; q }";
    check_module(&parse_module(source)?)?;
    assert_eq!(check_ir(source)?, CheckResult::Checked);
    Ok(())
}

#[test]
fn checks_a_register_after_its_input_expression() -> Result<(), String> {
    let source =
        "block sum(clk, x: u32 @0, y: u32 @0) -> u32 @1 { reg q: u32; q := x + y; q }";
    assert_eq!(check_ir(source)?, CheckResult::Checked);
    Ok(())
}

#[test]
fn checks_a_two_register_pipeline() -> Result<(), String> {
    let source = r#"block delay(clk, x: u32 @0) -> u32 @2 {
    reg a: u32;
    reg b: u32;
    a := x;
    b := a;
    b
}"#;
    assert_eq!(check_ir(source)?, CheckResult::Checked);

    let wrong_cycle = source.replace("-> u32 @2", "-> u32 @1");
    assert_eq!(
        check_ir(&wrong_cycle).unwrap_err(),
        "output 'out' is at stage 2, expected stage 1"
    );
    Ok(())
}

#[test]
fn enabled_register_requires_aligned_data_and_enable() -> Result<(), String> {
    let source = r#"block capture(clk, en: bool @2, x: u32 @2) -> u32 @3 {
    reg q: u32;
    if en { q := x; }
    q
}"#;
    assert_eq!(check_ir(source)?, CheckResult::Checked);

    let wrong_enable = source.replace("en: bool @2", "en: bool @1");
    assert_eq!(
        check_ir(&wrong_enable).unwrap_err(),
        "write to 'q' has enable at stage 1 and data at stage 2"
    );
    Ok(())
}

#[test]
fn enabled_constant_uses_the_enables_cycle() -> Result<(), String> {
    let source =
        "block capture(clk, en: bool @2) -> u32 @3 { reg q: u32; if en { q := u32:7; } q }";
    assert_eq!(check_ir(source)?, CheckResult::Checked);
    Ok(())
}

#[test]
fn reset_does_not_change_the_capture_stage() -> Result<(), String> {
    let source = r#"block capture(clk, en: bool @2, x: u32 @2) -> u32 @3 {
    reg q: u32 = u32:0;
    if en { q := x; }
    q
}"#;
    check_module(&parse_module(source)?)?;
    assert_eq!(check_ir(source)?, CheckResult::Checked);

    let wrong_enable = source.replace("en: bool @2", "en: bool @1");
    assert_eq!(
        check_ir(&wrong_enable).unwrap_err(),
        "write to 'q' has enable at stage 1 and data at stage 2"
    );
    Ok(())
}

#[test]
fn active_low_reset_does_not_change_the_capture_stage() -> Result<(), String> {
    let source = "block delay(clk, x: u32 @0) -> u32 @1 { reg q: u32 = u32:0; q := x; q }";
    for asynchronous in [false, true] {
        let reset = ResetBehavior {
            asynchronous,
            active_low: true,
        };
        assert_eq!(check_ir_with_reset(source, reset)?, CheckResult::Checked);
    }
    Ok(())
}

#[test]
fn active_low_reset_is_not_timed_data() -> Result<(), String> {
    let source = r#"#![allow(block_reset_signal)]
block reset_value(clk, rst) -> bool @0 { rst }"#;
    let reset = ResetBehavior {
        active_low: true,
        ..ResetBehavior::default()
    };
    assert_eq!(check_ir_with_reset(source, reset)?, CheckResult::Unsupported);
    Ok(())
}

#[test]
fn nested_enables_must_align() -> Result<(), String> {
    let source = r#"block capture(clk, en: bool @2, ready: bool @2, x: u32 @2) -> u32 @3 {
    reg q: u32;
    if en {
        if ready { q := x; }
    }
    q
}"#;
    assert_eq!(check_ir(source)?, CheckResult::Checked);

    let wrong_guard = source.replace("ready: bool @2", "ready: bool @1");
    assert!(check_ir(&wrong_guard).is_err());
    Ok(())
}

#[test]
fn register_feedback_is_unsupported() -> Result<(), String> {
    let source = "block sum(clk, x: u32 @0) -> u32 @1 { reg q: u32; q := q + x; q }";
    assert_eq!(check_ir(source)?, CheckResult::Unsupported);
    Ok(())
}

#[test]
fn checks_a_register_even_when_nobody_uses_its_value() -> Result<(), String> {
    // The output x is fine, but the unused register would overflow the cycle
    // numbering. Its read still has to be checked.
    let source = format!(
        "block delay(clk, x: u32 @{0}) -> u32 @{0} {{ reg q: u32; q := x; x }}",
        u64::MAX
    );
    check_module(&parse_module(&source)?)?;
    assert_eq!(
        check_ir(&source).unwrap_err(),
        "cycle offset overflow at register 'q'"
    );
    Ok(())
}

#[test]
fn register_cycle_rejects_overflow() {
    assert_eq!(register_cycle("q", 0).unwrap(), 1);
    assert_eq!(register_cycle("q", u64::MAX - 1).unwrap(), u64::MAX);
    assert_eq!(
        register_cycle("q", u64::MAX).unwrap_err(),
        "cycle offset overflow at register 'q'"
    );
}
