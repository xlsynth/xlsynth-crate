// SPDX-License-Identifier: Apache-2.0

use crate::ir_verify::VerifyError;

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorCategory {
    DuplicateMemberName,
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
    MissingBlockClock,
    Other,
}

pub fn categorize_pir_error(err: &VerifyError) -> ErrorCategory {
    use ErrorCategory::*;
    match err {
        VerifyError::DuplicateMemberName(_) => DuplicateMemberName,
        VerifyError::UnknownCallee { .. } => UnknownCallee,
        VerifyError::NodeTypeMismatch { .. } => NodeTypeMismatch,
        VerifyError::OperandOutOfBounds { .. } => OperandOutOfBounds,
        VerifyError::OperandUsesUndefined { .. } => OperandUsesUndefined,
        VerifyError::DuplicateTextId { .. } => DuplicateTextId,
        VerifyError::ReturnTypeMismatch { .. } => ReturnTypeMismatch,
        VerifyError::MissingReturnNode { .. } => MissingReturnNode,
        VerifyError::DuplicateParamName { .. } => DuplicateParamName,
        VerifyError::MissingParamNode { .. } => MissingParamNode,
        VerifyError::ExtraParamNode { .. } => ExtraParamNode,
        VerifyError::NodeNameOpMismatch { .. } => NodeNameOpMismatch,
        VerifyError::NodeNameIdSuffixMismatch { .. } => NodeNameIdSuffixMismatch,
        VerifyError::BlockInvariantViolation { reason, .. }
            if reason == "registers require a clock port" =>
        {
            MissingBlockClock
        }
        _ => Other,
    }
}

pub fn categorize_xls_error_text(s: &str) -> ErrorCategory {
    use ErrorCategory::*;
    let lower = s.to_lowercase();
    if lower.contains("block has registers but no clock port") {
        return MissingBlockClock;
    }
    if lower.contains("not unique within package") || lower.contains("duplicate member name") {
        return DuplicateMemberName;
    }
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

#[cfg(test)]
mod tests {
    use super::{ErrorCategory, categorize_pir_error, categorize_xls_error_text};
    use crate::ir_parser::Parser;
    use crate::ir_verify::verify_package;

    #[test]
    fn registers_without_clock_are_semantic_rejections_in_both_apis() {
        let ir = "package test\nblock b() { reg r(()) }\n";
        let package = Parser::new(ir).parse_package().unwrap();
        let error = verify_package(&package).unwrap_err();
        assert_eq!(
            categorize_pir_error(&error),
            ErrorCategory::MissingBlockClock
        );
        let error = match xlsynth::IrPackage::parse_ir(ir, None) {
            Ok(_) => panic!("XLS accepted a register declaration without a clock port"),
            Err(error) => error,
        };
        assert_eq!(
            categorize_xls_error_text(&error.to_string()),
            ErrorCategory::MissingBlockClock,
        );
    }
}
