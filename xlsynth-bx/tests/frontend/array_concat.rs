// SPDX-License-Identifier: Apache-2.0

use std::error::Error;

use super::directed_expr_eval::{assert_expr_matches_dslx, assert_expr_rejected_by_both};
use super::expr_eval::BlockEvaluator;
use xlsynth_bx::{check_module, parse_module};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{NodePayload, Type};

#[test]
fn operands_infer_array_elements_independently() -> Result<(), Box<dyn Error>> {
    assert_expr_matches_dslx(
        "",
        "u2[2]",
        "[2] ++ [3]",
        [(Vec::new(), IrValue::parse_typed("[bits[2]:2, bits[2]:3]")?)],
    );
    assert_expr_rejected_by_both("", "u8[2]", "[1] ++ [2]");
    assert_expr_rejected_by_both("xs: u8[1]", "u8[2]", "xs ++ [1]");
    assert_expr_rejected_by_both("xs: u8[1]", "u8[2]", "[1] ++ xs");
    let bad_result = parse_module("fn f() -> u8[2] { [2] ++ [3] }")?;
    assert!(check_module(&bad_result).is_err());
    Ok(())
}

#[test]
fn concatenation_preserves_typed_empty_arrays_and_constant_order() -> Result<(), Box<dyn Error>> {
    let source = r#"
const EMPTY: u8[0] = [];
const BOTH = EMPTY ++ EMPTY;
const SEED = EMPTY ++ [u8:1] ++ EMPTY ++ [u8:2];
#[allow(combinational_block)]
block sample(clk, x: u8, empty: u8[0]) -> (u8[0], u8[3]) {
    (empty ++ BOTH, SEED ++ [x])
}
"#;
    let checked = check_module(&parse_module(source)?)?;
    let evaluator = BlockEvaluator::lower(&checked, source)?;
    assert!(
        evaluator
            .function
            .nodes
            .iter()
            .any(|node| matches!(node.payload, NodePayload::ArrayConcat(_)))
    );
    let empty = IrValue::make_array_typed(Type::Bits(8), &[])?;
    let expected = IrValue::parse_typed("[bits[8]:1, bits[8]:2, bits[8]:3]")?;
    assert_eq!(
        evaluator.evaluate(&[IrValue::make_ubits(8, 3)?, empty.clone()])?,
        IrValue::make_tuple(&[empty, expected])
    );
    Ok(())
}
