// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigRef, GateFn};
use anyhow::Result;
use std::any::Any;
use std::fmt::{self, Debug};

/// Enum representing the different kinds of transformations that can be
/// applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransformKind {
    SwapOperands,
    ToggleOutputBit,
    DoubleNegate,
    InsertRedundantAnd,
    RemoveRedundantAnd,
    DuplicateGate,
    UnduplicateGate,
    InsertFalseAnd,
    RemoveFalseAnd,
    InsertTrueAnd,
    RemoveTrueAnd,
    SwapOutputBits,
    RotateAndRight,
    RotateAndLeft,
    AndAbsorbRight,
    AndAbsorbLeft,
    BalanceAndTree,
    UnbalanceAndTree,
    ToggleOperandNegation,
    RewireOperand,
    PushNegation,
}

impl fmt::Display for TransformKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformKind::SwapOperands => write!(f, "SwapOp"),
            TransformKind::ToggleOutputBit => write!(f, "ToggleOut"),
            TransformKind::DoubleNegate => write!(f, "DblNeg"),
            TransformKind::InsertRedundantAnd => write!(f, "InsRedAnd"),
            TransformKind::RemoveRedundantAnd => write!(f, "RemRedAnd"),
            TransformKind::DuplicateGate => write!(f, "DupGate"),
            TransformKind::UnduplicateGate => write!(f, "UndupGate"),
            TransformKind::InsertFalseAnd => write!(f, "InsFalseAnd"),
            TransformKind::RemoveFalseAnd => write!(f, "RemFalseAnd"),
            TransformKind::InsertTrueAnd => write!(f, "InsTrueAnd"),
            TransformKind::RemoveTrueAnd => write!(f, "RemTrueAnd"),
            TransformKind::SwapOutputBits => write!(f, "SwapOutBits"),
            TransformKind::RotateAndRight => write!(f, "RotAndR"),
            TransformKind::RotateAndLeft => write!(f, "RotAndL"),
            TransformKind::AndAbsorbRight => write!(f, "AbsorbR"),
            TransformKind::AndAbsorbLeft => write!(f, "AbsorbL"),
            TransformKind::BalanceAndTree => write!(f, "BalTree"),
            TransformKind::UnbalanceAndTree => write!(f, "UnbalTree"),
            TransformKind::ToggleOperandNegation => write!(f, "TogOpNeg"),
            TransformKind::RewireOperand => write!(f, "RewireOp"),
            TransformKind::PushNegation => write!(f, "PushNeg"),
        }
    }
}

/// Enum to specify the direction of transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformDirection {
    Forward,
    Backward,
}

/// Represents a specific location in the GateFn where a transform can be
/// applied.
#[derive(Debug)]
pub enum TransformLocation {
    Node(AigRef),
    Operand(AigRef, bool),
    OutputPortBit { output_idx: usize, bit_idx: usize },
    Custom(Box<dyn Any + Send + Sync>),
}

/// Defines a reversible transformation that can be applied to a `GateFn`.
pub trait Transform: Debug + Send + Sync {
    /// Returns the specific `TransformKind` that this trait object represents.
    fn kind(&self) -> TransformKind;

    /// Returns a human-readable display name for this transform.
    /// Defaults to the `Display` implementation of `TransformKind`.
    fn display_name(&self) -> String {
        format!("{:?}", self.kind())
    }

    /// Finds all possible application sites for this transform in the given
    /// GateFn for the specified direction.
    fn find_candidates(
        &mut self,
        g: &GateFn,
        direction: TransformDirection,
    ) -> Vec<TransformLocation>;

    /// Applies the transform to the GateFn at the given candidate site
    /// in the specified direction.
    /// The MCMC loop is expected to pass a clone of the GateFn if rejection
    /// implies reverting to the prior state.
    fn apply(
        &self,
        g: &mut GateFn,
        candidate_location: &TransformLocation,
        direction: TransformDirection,
    ) -> Result<()>;

    /// Indicates whether this transform is always semantics preserving.
    /// When `true`, applying the transform cannot change the functional
    /// behaviour of the circuit, so equivalence checks can be skipped.
    fn always_equivalent(&self) -> bool;
}
