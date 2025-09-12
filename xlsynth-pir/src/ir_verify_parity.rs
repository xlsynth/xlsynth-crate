// SPDX-License-Identifier: Apache-2.0

use crate::ir_validate::ValidationError;

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorCategory {
    UnknownCallee,
    NodeTypeMismatch,
    OperandOutOfBounds,
    OperandUsesUndefined,
    DuplicateTextId,
    ReturnTypeMismatch,
    MissingReturnNode,
    DuplicateParamName,
    MissingParamNode,
    ExtraParamNode,
    NodeNameOpMismatch,
    NodeNameIdSuffixMismatch,
    Other,
}

pub fn categorize_pir_error(err: &ValidationError) -> ErrorCategory {
    use ErrorCategory::*;
    match err {
        ValidationError::UnknownCallee { .. } => UnknownCallee,
        ValidationError::NodeTypeMismatch { .. } => NodeTypeMismatch,
        ValidationError::OperandOutOfBounds { .. } => OperandOutOfBounds,
        ValidationError::OperandUsesUndefined { .. } => OperandUsesUndefined,
        ValidationError::DuplicateTextId { .. } => DuplicateTextId,
        ValidationError::ReturnTypeMismatch { .. } => ReturnTypeMismatch,
        ValidationError::MissingReturnNode { .. } => MissingReturnNode,
        ValidationError::DuplicateParamName { .. } => DuplicateParamName,
        ValidationError::MissingParamNode { .. } => MissingParamNode,
        ValidationError::ExtraParamNode { .. } => ExtraParamNode,
        ValidationError::NodeNameOpMismatch { .. } => NodeNameOpMismatch,
        ValidationError::NodeNameIdSuffixMismatch { .. } => NodeNameIdSuffixMismatch,
        _ => Other,
    }
}

pub fn categorize_xls_error_text(s: &str) -> ErrorCategory {
    use ErrorCategory::*;
    let lower = s.to_lowercase();
    if lower.contains("unknown callee")
        || lower.contains("unknown function")
        || lower.contains("cannot find function")
        || lower.contains("does not have a function with name")
    {
        return UnknownCallee;
    }
    if lower.contains("type mismatch")
        || (lower.contains("type") && lower.contains("mismatch"))
        || lower.contains("does not match expected type")
    {
        return NodeTypeMismatch; // coarse bucket
    }
    if lower.contains("out of bounds") {
        return OperandOutOfBounds;
    }
    if lower.contains("before definition") || lower.contains("uses operand") {
        return OperandUsesUndefined;
    }
    if lower.contains("duplicate") && lower.contains("id") {
        return DuplicateTextId;
    }
    if lower.contains("return type") && lower.contains("mismatch") {
        return ReturnTypeMismatch;
    }
    if lower.contains("expected 'ret' in function") {
        return MissingReturnNode;
    }
    if lower.contains("duplicate param") || lower.contains("duplicate parameter") {
        return DuplicateParamName;
    }
    if lower.contains("missing getparam") || lower.contains("missing param") {
        return MissingParamNode;
    }
    if lower.contains("not declared in signature") || lower.contains("extra param") {
        return ExtraParamNode;
    }
    Other
}
