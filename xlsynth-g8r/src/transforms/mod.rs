// SPDX-License-Identifier: Apache-2.0

pub mod double_negate;
pub mod duplicate;
pub mod redundant_and;
pub mod rotate_and;
pub mod swap_operands;
pub mod toggle_output;
pub mod transform_trait;
pub mod true_and;

use double_negate::DoubleNegateTransform;
use duplicate::{DuplicateGateTransform, UnduplicateGateTransform};
use redundant_and::{InsertRedundantAndTransform, RemoveRedundantAndTransform};
use swap_operands::SwapOperandsTransform;
use toggle_output::ToggleOutputBitTransform;
use transform_trait::Transform;

// Updated function to include all refactored transforms.
pub fn get_all_transforms() -> Vec<Box<dyn Transform>> {
    vec![
        Box::new(SwapOperandsTransform::new()),
        Box::new(ToggleOutputBitTransform::new()),
        Box::new(DoubleNegateTransform::new()),
        Box::new(DuplicateGateTransform::new()),
        Box::new(UnduplicateGateTransform::new()),
        Box::new(InsertRedundantAndTransform::new()),
        Box::new(RemoveRedundantAndTransform::new()),
    ]
}
