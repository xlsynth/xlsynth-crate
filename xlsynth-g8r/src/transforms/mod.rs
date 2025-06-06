// SPDX-License-Identifier: Apache-2.0

pub mod and_absorb;
pub mod balance_and_tree;
pub mod double_negate;
pub mod duplicate;
pub mod factor_shared_and;
pub mod false_and;
pub mod merge_equiv_leaves;
pub mod push_negation;
pub mod redundant_and;
pub mod rewire_operand;
pub mod rotate_and;
pub mod split_fanout;
pub mod swap_operands;
pub mod swap_outputs;
pub mod toggle_operand_negation;
pub mod toggle_output;
pub mod transform_trait;
pub mod true_and;

use crate::transforms::and_absorb::{AndAbsorbLeftTransform, AndAbsorbRightTransform};
use crate::transforms::false_and::{InsertFalseAndTransform, RemoveFalseAndTransform};
use crate::transforms::rotate_and::{RotateAndLeftTransform, RotateAndRightTransform};
use crate::transforms::true_and::{InsertTrueAndTransform, RemoveTrueAndTransform};
use balance_and_tree::{BalanceAndTreeTransform, UnbalanceAndTreeTransform};
use double_negate::DoubleNegateTransform;
use duplicate::{DuplicateGateTransform, UnduplicateGateTransform};
use factor_shared_and::{FactorSharedAndTransform, UnfactorSharedAndTransform};
use merge_equiv_leaves::MergeEquivLeavesTransform;
use push_negation::PushNegationTransform;
use redundant_and::{InsertRedundantAndTransform, RemoveRedundantAndTransform};
use rewire_operand::RewireOperandTransform;
use split_fanout::{MergeFanoutTransform, SplitFanoutTransform};
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
        Box::new(AndAbsorbRightTransform::new()),
        Box::new(AndAbsorbLeftTransform::new()),
        Box::new(BalanceAndTreeTransform::new()),
        Box::new(UnbalanceAndTreeTransform::new()),
        Box::new(RewireOperandTransform::new()),
        Box::new(PushNegationTransform::new()),
        Box::new(MergeEquivLeavesTransform::new()),
        Box::new(SplitFanoutTransform::new()),
        Box::new(MergeFanoutTransform::new()),
        Box::new(FactorSharedAndTransform::new()),
        Box::new(UnfactorSharedAndTransform::new()),
    ]
}

/// Returns all transforms that are always semantics preserving.
pub fn get_equiv_transforms() -> Vec<Box<dyn Transform>> {
    get_all_transforms()
        .into_iter()
        .filter(|t| t.always_equivalent())
        .collect()
}
