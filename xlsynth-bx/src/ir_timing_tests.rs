// SPDX-License-Identifier: Apache-2.0

//! Check BX examples with both the existing checker and the new IR checker.

use crate::ast::{OutputCycles, StageAnnotation};
use crate::ir_timing::{CheckResult, Contract, check_block};
use crate::{ResetBehavior, check_module, empty_package, lower_module, parse_module};

/// Temporarily remove `@N` so BX will lower even an example that should fail
/// timing checking. Then ask the IR checker to check the original `@N`.
fn check_ir(source: &str) -> Result<CheckResult, String> {
    let mut parsed = parse_module(source)?;
    let Some(OutputCycles::Uniform(StageAnnotation::Literal(expected))) =
        parsed.blocks[0].output_cycles.take()
    else {
        return Err("fixture requires a uniform literal output cycle".into());
    };
    let checked = check_module(&parsed)?;
    let package = lower_module(&checked, empty_package("timing"), ResetBehavior::default())?;
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
fn register_timing_is_not_certified_yet() -> Result<(), String> {
    // BX knows q is one register after x; our new checker does not yet.
    let source = "block delay(clk, x: u32 @0) -> u32 @1 { reg q: u32; q := x; q }";
    check_module(&parse_module(source)?)?;
    assert_eq!(check_ir(source)?, CheckResult::Unsupported);
    Ok(())
}
