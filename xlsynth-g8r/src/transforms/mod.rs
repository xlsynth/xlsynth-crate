// SPDX-License-Identifier: Apache-2.0

pub mod double_negate;
pub mod duplicate;
pub mod false_and;
pub mod redundant_and;
pub mod rewire_operand;
pub mod rotate_and;
pub mod swap_operands;
pub mod swap_outputs;
pub mod toggle_operand_negation;
pub mod toggle_output;
pub mod transform_trait;
pub mod true_and;

use crate::transforms::false_and::{InsertFalseAndTransform, RemoveFalseAndTransform};
use crate::transforms::rotate_and::{RotateAndLeftTransform, RotateAndRightTransform};
use crate::transforms::true_and::{InsertTrueAndTransform, RemoveTrueAndTransform};
use double_negate::DoubleNegateTransform;
use duplicate::{DuplicateGateTransform, UnduplicateGateTransform};
use redundant_and::{InsertRedundantAndTransform, RemoveRedundantAndTransform};
use rewire_operand::RewireOperandTransform;
use swap_operands::SwapOperandsTransform;
use swap_outputs::SwapOutputBitsTransform;
use toggle_operand_negation::ToggleOperandNegationTransform;
use toggle_output::ToggleOutputBitTransform;
use transform_trait::Transform;

// Updated function to include all refactored transforms.
pub fn get_all_transforms() -> Vec<Box<dyn Transform>> {
    vec![
        Box::new(SwapOperandsTransform::new()),
        Box::new(ToggleOutputBitTransform::new()),
        Box::new(ToggleOperandNegationTransform::new()),
        Box::new(DoubleNegateTransform::new()),
        Box::new(DuplicateGateTransform::new()),
        Box::new(UnduplicateGateTransform::new()),
        Box::new(InsertRedundantAndTransform::new()),
        Box::new(RemoveRedundantAndTransform::new()),
        Box::new(InsertFalseAndTransform::new()),
        Box::new(RemoveFalseAndTransform::new()),
        Box::new(InsertTrueAndTransform::new()),
        Box::new(RemoveTrueAndTransform::new()),
        Box::new(SwapOutputBitsTransform::new()),
        Box::new(RotateAndRightTransform::new()),
        Box::new(RotateAndLeftTransform::new()),
        Box::new(RewireOperandTransform::new()),
    ]
}
