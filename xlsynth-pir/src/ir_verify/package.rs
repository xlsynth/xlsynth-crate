// SPDX-License-Identifier: Apache-2.0

//! Package-context portions of PIR verification.

use std::collections::HashSet;

use crate::ir::{
    Binop, BlockMetadata, BlockPortKind, Fn, InstantiationKind, MemberType, NaryOp, NodePayload,
    Package, PackageMember, Type,
};
use crate::ir_deduce::deduce_result_type_with_registers;
use crate::ir_utils::operands;

/// Errors that can arise during validation of XLS IR structures.
#[derive(Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// Two package members share the same name.
    DuplicateMemberName(String),
    /// The `top` attribute references a missing function.
    MissingTop(String),
    /// A node references an undefined operand (index out of bounds).
    OperandOutOfBounds {
        func: String,
        node_index: usize,
        operand: usize,
    },
    /// A node references an operand defined after the node.
    OperandUsesUndefined {
        func: String,
        node_index: usize,
        operand: usize,
    },
    /// A static `bit_slice` selects bits beyond its operand's width.
    BitSliceOutOfBounds {
        func: String,
        node_index: usize,
        start: usize,
        width: usize,
        operand_width: usize,
    },
    /// A function's return node is missing.
    MissingReturnNode(String),
    /// A function's declared return type doesn't match the return node type.
    ReturnTypeMismatch {
        func: String,
        expected: Type,
        actual: Type,
    },
    /// A node's text id is not unique among non-parameter nodes.
    DuplicateTextId { func: String, text_id: usize },
    /// A parameter node's text id does not match its declared parameter id.
    ParamIdMismatch {
        func: String,
        param_name: String,
        expected: usize,
        actual: usize,
    },
    /// The function refers to another function that does not exist in the
    /// package.
    UnknownCallee { func: String, callee: String },
    /// Function references form a recursive call graph, which XLS forbids.
    RecursiveFunctionReference { cycle: Vec<String> },
    /// A standalone-function verification attempted to check an operation
    /// whose contract requires package context.
    RequiresPackageContext {
        func: String,
        node_index: usize,
        op: String,
    },
    /// A register op references a register that does not exist in the block.
    UnknownRegister {
        func: String,
        node_index: usize,
        register: String,
    },
    /// A register op appears in a function (not a block).
    RegisterOpInFunction { func: String, node_index: usize },
    /// A register_write arg type does not match the register type.
    RegisterWriteTypeMismatch {
        func: String,
        node_index: usize,
        register: String,
        expected: Type,
        actual: Type,
    },
    /// A register_write load_enable is not bits[1].
    RegisterWriteLoadEnableTypeMismatch {
        func: String,
        node_index: usize,
        actual: Type,
    },
    /// A register_write reset is not bits[1].
    RegisterWriteResetTypeMismatch {
        func: String,
        node_index: usize,
        actual: Type,
    },
    /// An instantiation op references an unknown instantiation.
    UnknownInstantiation {
        func: String,
        node_index: usize,
        instantiation: String,
    },
    /// An instantiation declaration references a missing or later block.
    InstantiationBlockNotFound {
        func: String,
        instantiation: String,
        block: String,
    },
    /// An instantiation op appears in a function (not a block).
    InstantiationOpInFunction { func: String, node_index: usize },
    /// instantiation port name not found on callee block.
    UnknownInstantiationPort {
        func: String,
        node_index: usize,
        instantiation: String,
        port_name: String,
        direction: InstantiationPortDirection,
    },
    /// instantiation port type mismatch with callee port type.
    InstantiationPortTypeMismatch {
        func: String,
        node_index: usize,
        instantiation: String,
        port_name: String,
        direction: InstantiationPortDirection,
        expected: Type,
        actual: Type,
    },
    /// Duplicate instantiation port mapping.
    DuplicateInstantiationPort {
        func: String,
        node_index: usize,
        instantiation: String,
        port_name: String,
        direction: InstantiationPortDirection,
    },
    /// Missing instantiation port mappings.
    MissingInstantiationPorts {
        func: String,
        instantiation: String,
        missing: Vec<String>,
        direction: InstantiationPortDirection,
    },
    /// Block output arity mismatch when mapping ports.
    BlockOutputArityMismatch {
        func: String,
        expected: usize,
        actual: usize,
    },
    /// Ordered block ports disagree with compatibility metadata or the
    /// function-shaped representation of the block.
    BlockPortMetadataMismatch { func: String, reason: String },
    /// Bitwise n-ary ops (and/or/xor/nand/nor) must have identical bits-typed
    /// operands.
    NaryBitwiseOperandTypeMismatch { func: String, node_index: usize },
    /// `ext_nary_add` requires all operands to be bits-typed.
    ExtNaryAddOperandTypeMismatch { func: String, node_index: usize },
    /// `ext_nary_add` requires a bits-typed result.
    ExtNaryAddResultTypeMismatch {
        func: String,
        node_index: usize,
        actual: Type,
    },
    /// `ext_mask_low` requires a bits-typed count operand.
    ExtMaskLowCountTypeMismatch { func: String, node_index: usize },
    /// `ext_mask_low` requires a bits-typed result.
    ExtMaskLowResultTypeMismatch {
        func: String,
        node_index: usize,
        actual: Type,
    },
    /// `smulp` and `umulp` require a pair of identically sized bits results.
    PartialProductResultTypeMismatch {
        func: String,
        node_index: usize,
        actual: Type,
    },
    /// Two parameters share the same name within a function.
    DuplicateParamName { func: String, param_name: String },
    /// A parameter declared in the function signature has no corresponding
    /// GetParam node in the node list.
    MissingParamNode {
        func: String,
        param_name: String,
        expected_id: usize,
    },
    /// A GetParam node exists in the node list that does not correspond to any
    /// declared parameter in the function signature.
    ExtraParamNode { func: String, text_id: usize },
    /// A node name looks like a default textual id (e.g. op.id) but the
    /// operator prefix does not match the node's actual operator.
    NodeNameOpMismatch {
        func: String,
        node_index: usize,
        name: String,
        expected_op: String,
    },
    /// A node name looks like a default textual id (e.g. op.id) but the numeric
    /// suffix does not match the node's text id.
    NodeNameIdSuffixMismatch {
        func: String,
        node_index: usize,
        name: String,
        expected_id: usize,
    },
    /// A node's declared type does not match the type deduced from its
    /// operator and operand types.
    NodeTypeMismatch {
        func: String,
        node_index: usize,
        deduced: Type,
        actual: Type,
    },
    /// Type deduction failed for a node due to an internal error.
    TypeDeductionFailure {
        func: String,
        node_index: usize,
        reason: String,
    },
    /// An XLS node-level semantic constraint was not satisfied.
    NodeSemanticViolation {
        func: String,
        node_index: usize,
        reason: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::DuplicateMemberName(name) => {
                write!(f, "duplicate member name '{}'", name)
            }
            ValidationError::MissingTop(name) => {
                write!(f, "top member '{}' not found", name)
            }
            ValidationError::OperandOutOfBounds {
                func,
                node_index,
                operand,
            } => {
                write!(
                    f,
                    "function '{}' node {} references operand {} out of bounds",
                    func, node_index, operand
                )
            }
            ValidationError::OperandUsesUndefined {
                func,
                node_index,
                operand,
            } => {
                write!(
                    f,
                    "function '{}' node {} uses operand {} before definition",
                    func, node_index, operand
                )
            }
            ValidationError::BitSliceOutOfBounds {
                func,
                node_index,
                start,
                width,
                operand_width,
            } => {
                write!(
                    f,
                    "function '{}' node {} bit_slice start {} + width {} exceeds operand width {}",
                    func, node_index, start, width, operand_width
                )
            }
            ValidationError::MissingReturnNode(func) => {
                write!(f, "function '{}' missing return node", func)
            }
            ValidationError::ReturnTypeMismatch {
                func,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' return type mismatch: expected {}, got {}",
                    func, expected, actual
                )
            }
            ValidationError::DuplicateTextId { func, text_id } => {
                write!(f, "function '{}' has duplicate text id {}", func, text_id)
            }
            ValidationError::ParamIdMismatch {
                func,
                param_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' param '{}' id mismatch: expected {}, got {}",
                    func, param_name, expected, actual
                )
            }
            ValidationError::UnknownCallee { func, callee } => {
                write!(
                    f,
                    "function '{}' references undefined callee '{}'",
                    func, callee
                )
            }
            ValidationError::RecursiveFunctionReference { cycle } => {
                write!(
                    f,
                    "recursive function references are not valid XLS IR: {}",
                    cycle.join(" -> ")
                )
            }
            ValidationError::RequiresPackageContext {
                func,
                node_index,
                op,
            } => {
                write!(
                    f,
                    "function '{}' node {} operation '{}' requires package context for verification",
                    func, node_index, op
                )
            }
            ValidationError::UnknownRegister {
                func,
                node_index,
                register,
            } => {
                write!(
                    f,
                    "function '{}' node {} references unknown register '{}'",
                    func, node_index, register
                )
            }
            ValidationError::RegisterOpInFunction { func, node_index } => {
                write!(
                    f,
                    "function '{}' node {} uses register op outside a block",
                    func, node_index
                )
            }
            ValidationError::RegisterWriteTypeMismatch {
                func,
                node_index,
                register,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} register '{}' type mismatch: expected {} got {}",
                    func, node_index, register, expected, actual
                )
            }
            ValidationError::RegisterWriteLoadEnableTypeMismatch {
                func,
                node_index,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} register_write load_enable type mismatch: expected bits[1] got {}",
                    func, node_index, actual
                )
            }
            ValidationError::RegisterWriteResetTypeMismatch {
                func,
                node_index,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} register_write reset type mismatch: expected bits[1] got {}",
                    func, node_index, actual
                )
            }
            ValidationError::UnknownInstantiation {
                func,
                node_index,
                instantiation,
            } => {
                write!(
                    f,
                    "function '{}' node {} references unknown instantiation '{}'",
                    func, node_index, instantiation
                )
            }
            ValidationError::InstantiationBlockNotFound {
                func,
                instantiation,
                block,
            } => {
                write!(
                    f,
                    "function '{}' instantiation '{}' references missing block '{}'",
                    func, instantiation, block
                )
            }
            ValidationError::InstantiationOpInFunction { func, node_index } => {
                write!(
                    f,
                    "function '{}' node {} uses instantiation op outside a block",
                    func, node_index
                )
            }
            ValidationError::UnknownInstantiationPort {
                func,
                node_index,
                instantiation,
                port_name,
                direction,
            } => {
                write!(
                    f,
                    "function '{}' node {} instantiation '{}' {} port '{}' not found",
                    func, node_index, instantiation, direction, port_name
                )
            }
            ValidationError::InstantiationPortTypeMismatch {
                func,
                node_index,
                instantiation,
                port_name,
                direction,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} instantiation '{}' {} '{}' type mismatch: expected {} got {}",
                    func, node_index, instantiation, direction, port_name, expected, actual
                )
            }
            ValidationError::DuplicateInstantiationPort {
                func,
                node_index,
                instantiation,
                port_name,
                direction,
            } => {
                write!(
                    f,
                    "function '{}' node {} instantiation '{}' {} '{}' mapped multiple times",
                    func, node_index, instantiation, direction, port_name
                )
            }
            ValidationError::MissingInstantiationPorts {
                func,
                instantiation,
                missing,
                direction,
            } => {
                write!(
                    f,
                    "function '{}' instantiation '{}' missing {} ports: {:?}",
                    func, instantiation, direction, missing
                )
            }
            ValidationError::BlockOutputArityMismatch {
                func,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' output arity mismatch: expected {} outputs, got {}",
                    func, expected, actual
                )
            }
            ValidationError::BlockPortMetadataMismatch { func, reason } => {
                write!(f, "block '{func}' has inconsistent port metadata: {reason}")
            }
            ValidationError::NaryBitwiseOperandTypeMismatch { func, node_index } => {
                write!(
                    f,
                    "function '{}' node {} has mismatched operand types for bitwise n-ary op",
                    func, node_index
                )
            }
            ValidationError::ExtNaryAddOperandTypeMismatch { func, node_index } => {
                write!(
                    f,
                    "function '{}' node {} ext_nary_add requires bits operands",
                    func, node_index
                )
            }
            ValidationError::ExtNaryAddResultTypeMismatch {
                func,
                node_index,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} ext_nary_add requires bits result type, got {}",
                    func, node_index, actual
                )
            }
            ValidationError::ExtMaskLowCountTypeMismatch { func, node_index } => {
                write!(
                    f,
                    "function '{}' node {} ext_mask_low requires bits count operand",
                    func, node_index
                )
            }
            ValidationError::ExtMaskLowResultTypeMismatch {
                func,
                node_index,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} ext_mask_low requires bits result type, got {}",
                    func, node_index, actual
                )
            }
            ValidationError::PartialProductResultTypeMismatch {
                func,
                node_index,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} partial-product result must be (bits[N], bits[N]); got {}",
                    func, node_index, actual
                )
            }
            ValidationError::DuplicateParamName { func, param_name } => {
                write!(
                    f,
                    "function '{}' has duplicate param name '{}'",
                    func, param_name
                )
            }
            ValidationError::MissingParamNode {
                func,
                param_name,
                expected_id,
            } => {
                write!(
                    f,
                    "function '{}' missing GetParam node for param '{}' (expected id={})",
                    func, param_name, expected_id
                )
            }
            ValidationError::ExtraParamNode { func, text_id } => {
                write!(
                    f,
                    "function '{}' has GetParam node with id {} not declared in signature",
                    func, text_id
                )
            }
            ValidationError::NodeNameOpMismatch {
                func,
                node_index,
                name,
                expected_op,
            } => {
                write!(
                    f,
                    "function '{}' node {} name '{}' operator prefix does not match op '{}'",
                    func, node_index, name, expected_op
                )
            }
            ValidationError::NodeNameIdSuffixMismatch {
                func,
                node_index,
                name,
                expected_id,
            } => {
                write!(
                    f,
                    "function '{}' node {} name '{}' id suffix does not match text id {}",
                    func, node_index, name, expected_id
                )
            }
            ValidationError::NodeTypeMismatch {
                func,
                node_index,
                deduced,
                actual,
            } => {
                write!(
                    f,
                    "function '{}' node {} type mismatch: deduced {} vs actual {}",
                    func, node_index, deduced, actual
                )
            }
            ValidationError::TypeDeductionFailure {
                func,
                node_index,
                reason,
            } => {
                write!(
                    f,
                    "function '{}' node {} type deduction failed: {}",
                    func, node_index, reason
                )
            }
            ValidationError::NodeSemanticViolation {
                func,
                node_index,
                reason,
            } => {
                write!(
                    f,
                    "function '{}' node {} violates XLS node semantics: {}",
                    func, node_index, reason
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

pub(crate) struct InstantiationInfo {
    input_types: std::collections::HashMap<String, Type>,
    output_types: std::collections::HashMap<String, Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstantiationPortDirection {
    Input,
    Output,
}

impl std::fmt::Display for InstantiationPortDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstantiationPortDirection::Input => write!(f, "input"),
            InstantiationPortDirection::Output => write!(f, "output"),
        }
    }
}

/// Validates an entire package, ensuring all member names are unique, the top
/// function (if set) exists, and all contained functions are valid.
pub fn validate_package(p: &Package) -> Result<(), ValidationError> {
    let mut names = HashSet::<(String, MemberType)>::new();
    for member in &p.members {
        let (name, member_type) = match member {
            PackageMember::Function(f) => (f.name.clone(), MemberType::Function),
            PackageMember::Block { func, .. } => (func.name.clone(), MemberType::Block),
        };
        if !names.insert((name.clone(), member_type)) {
            return Err(ValidationError::DuplicateMemberName(name));
        }
    }

    if let Some(top) = &p.top {
        if !names.contains(top) {
            return Err(ValidationError::MissingTop(top.0.clone()));
        }
    }

    for (idx, member) in p.members.iter().enumerate() {
        match member {
            PackageMember::Function(f) => validate_fn(f, p)?,
            PackageMember::Block { func, metadata } => validate_block(func, metadata, p, idx)?,
        }
    }
    validate_function_call_graph(p)?;

    // Enforce package-wide uniqueness of node text ids (including parameter nodes).
    let mut seen_ids: HashSet<usize> = HashSet::new();
    for member in &p.members {
        let (f, metadata, synthetic_block_ret) = match member {
            PackageMember::Function(f) => (f, None, None),
            PackageMember::Block { func, metadata } => (
                func,
                Some(metadata),
                if metadata.output_names.len() != 1 {
                    func.ret_node_ref
                } else {
                    None
                },
            ),
        };
        for (node_index, node) in f.nodes.iter().enumerate() {
            // Skip synthetic Nil node at index 0 which is never emitted to IR text.
            if matches!(node.payload, NodePayload::Nil) {
                continue;
            }
            // Multi-output and output-less blocks use an internal tuple return
            // node that is not emitted to XLS IR text.
            if synthetic_block_ret == Some(crate::ir::NodeRef { index: node_index })
                && matches!(node.payload, NodePayload::Tuple(_))
            {
                continue;
            }
            if !seen_ids.insert(node.text_id) {
                return Err(ValidationError::DuplicateTextId {
                    func: f.name.clone(),
                    text_id: node.text_id,
                });
            }
        }
        if let Some(metadata) = metadata {
            let mut output_ids = metadata
                .output_port_ids
                .values()
                .copied()
                .collect::<Vec<_>>();
            output_ids.sort_unstable();
            for output_id in output_ids {
                if output_id == 0 {
                    return Err(ValidationError::BlockPortMetadataMismatch {
                        func: f.name.clone(),
                        reason: "output port id 0 is reserved".to_string(),
                    });
                }
                if !seen_ids.insert(output_id) {
                    return Err(ValidationError::DuplicateTextId {
                        func: f.name.clone(),
                        text_id: output_id,
                    });
                }
            }
        }
    }

    Ok(())
}

fn validate_function_call_graph(p: &Package) -> Result<(), ValidationError> {
    fn visit(
        p: &Package,
        function_name: &str,
        visited: &mut HashSet<String>,
        active: &mut Vec<String>,
    ) -> Result<(), ValidationError> {
        if let Some(cycle_start) = active.iter().position(|name| name == function_name) {
            let mut cycle = active[cycle_start..].to_vec();
            cycle.push(function_name.to_string());
            return Err(ValidationError::RecursiveFunctionReference { cycle });
        }
        if visited.contains(function_name) {
            return Ok(());
        }
        active.push(function_name.to_string());
        let function = p
            .get_fn(function_name)
            .expect("callee existence is validated before call-graph traversal");
        let mut callees = std::collections::BTreeSet::new();
        for node in &function.nodes {
            match &node.payload {
                NodePayload::Invoke { to_apply, .. } => {
                    callees.insert(to_apply.as_str());
                }
                NodePayload::CountedFor { body, .. } => {
                    callees.insert(body.as_str());
                }
                _ => {
                    // All other node kinds do not reference functions.
                }
            }
        }
        for callee in callees {
            visit(p, callee, visited, active)?;
        }
        active.pop();
        visited.insert(function_name.to_string());
        Ok(())
    }

    let mut visited = HashSet::new();
    for member in &p.members {
        let PackageMember::Function(function) = member else {
            continue;
        };
        visit(p, &function.name, &mut visited, &mut Vec::new())?;
    }
    Ok(())
}

/// Validates a function within the context of its parent package.
pub fn validate_fn(f: &Fn, parent: &Package) -> Result<(), ValidationError> {
    validate_fn_with(
        f,
        Some(parent),
        |name: &str| parent.get_fn_type(name).map(|ft| ft.return_type),
        |_register| None,
        None,
        false,
    )
}

/// Verifies a function whose contract does not depend on other package members.
pub(super) fn validate_standalone_fn(f: &Fn) -> Result<(), ValidationError> {
    validate_fn_with(f, None, |_name: &str| None, |_register| None, None, false)
}

pub fn validate_block(
    f: &Fn,
    metadata: &BlockMetadata,
    parent: &Package,
    member_index: usize,
) -> Result<(), ValidationError> {
    validate_ordered_block_ports(f, metadata)?;
    let prior_blocks = collect_prior_blocks(parent, member_index);
    for inst in metadata.instantiations.iter() {
        match inst.kind {
            InstantiationKind::Block if !prior_blocks.contains_key(&inst.block) => {
                return Err(ValidationError::InstantiationBlockNotFound {
                    func: f.name.clone(),
                    instantiation: inst.name.clone(),
                    block: inst.block.clone(),
                });
            }
            InstantiationKind::Extern => {
                let foreign =
                    parent
                        .get_fn(&inst.block)
                        .ok_or_else(|| ValidationError::UnknownCallee {
                            func: f.name.clone(),
                            callee: inst.block.clone(),
                        })?;
                if !foreign
                    .outer_attrs
                    .iter()
                    .any(|attribute| attribute.trim_start().starts_with("#[ffi_proto("))
                {
                    return Err(ValidationError::NodeSemanticViolation {
                        func: f.name.clone(),
                        node_index: 0,
                        reason: format!(
                            "extern instantiation '{}' references function '{}' without ffi_proto metadata",
                            inst.name, inst.block
                        ),
                    });
                }
            }
            _ => {}
        }
    }
    let instantiation_info = build_instantiation_info(metadata, &prior_blocks, parent)?;
    validate_fn_with(
        f,
        Some(parent),
        |name: &str| parent.get_fn_type(name).map(|ft| ft.return_type),
        |register| {
            metadata
                .registers
                .iter()
                .find(|r| r.name == register)
                .map(|r| r.ty.clone())
        },
        Some(&instantiation_info),
        true,
    )
}

/// Checks the authoritative ordered port list against the compatibility maps
/// and the function-shaped inputs/outputs used by PIR internally.
fn validate_ordered_block_ports(f: &Fn, metadata: &BlockMetadata) -> Result<(), ValidationError> {
    if metadata.ports.is_empty() {
        // Legacy programmatic metadata predates the ordered port vector.
        return Ok(());
    }
    let mismatch = |reason: String| ValidationError::BlockPortMetadataMismatch {
        func: f.name.clone(),
        reason,
    };
    let mut names = HashSet::new();
    for port in &metadata.ports {
        if !names.insert(port.name.as_str()) {
            return Err(mismatch(format!("duplicate ordered port '{}'", port.name)));
        }
        match (port.kind, &port.ty) {
            (BlockPortKind::Clock, None)
            | (BlockPortKind::Input | BlockPortKind::Output, Some(_)) => {}
            (BlockPortKind::Clock, Some(_)) => {
                return Err(mismatch(format!(
                    "clock port '{}' unexpectedly has a value type",
                    port.name
                )));
            }
            (BlockPortKind::Input | BlockPortKind::Output, None) => {
                return Err(mismatch(format!(
                    "data port '{}' is missing its value type",
                    port.name
                )));
            }
        }
    }

    let clock_names = metadata
        .ports
        .iter()
        .filter(|port| port.kind == BlockPortKind::Clock)
        .map(|port| port.name.as_str())
        .collect::<Vec<_>>();
    match (metadata.clock_port_name.as_deref(), clock_names.as_slice()) {
        (None, []) => {}
        (Some(recorded), [ordered]) if recorded == *ordered => {}
        _ => {
            return Err(mismatch(format!(
                "clock_port_name {:?} does not match ordered clock ports {:?}",
                metadata.clock_port_name, clock_names
            )));
        }
    }

    let ordered_inputs = metadata
        .ports
        .iter()
        .filter(|port| port.kind == BlockPortKind::Input)
        .collect::<Vec<_>>();
    if ordered_inputs.len() != f.params.len() {
        return Err(mismatch(format!(
            "ordered input count {} does not match parameter count {}",
            ordered_inputs.len(),
            f.params.len()
        )));
    }
    if metadata.input_port_ids.len() != ordered_inputs.len() {
        return Err(mismatch(format!(
            "input_port_ids count {} does not match ordered input count {}",
            metadata.input_port_ids.len(),
            ordered_inputs.len()
        )));
    }
    for (ordered, param) in ordered_inputs.iter().zip(&f.params) {
        if ordered.name != param.name || ordered.ty.as_ref() != Some(&param.ty) {
            return Err(mismatch(format!(
                "ordered input '{}: {:?}' does not match parameter '{}: {}'",
                ordered.name, ordered.ty, param.name, param.ty
            )));
        }
        if metadata.input_port_ids.get(&ordered.name) != Some(&param.id.get_wrapped_id()) {
            return Err(mismatch(format!(
                "input id for '{}' does not match parameter id {}",
                ordered.name,
                param.id.get_wrapped_id()
            )));
        }
    }

    let ordered_outputs = metadata
        .ports
        .iter()
        .filter(|port| port.kind == BlockPortKind::Output)
        .collect::<Vec<_>>();
    let output_types = match ordered_outputs.len() {
        0 if f.ret_ty.is_nil() => Vec::new(),
        1 => vec![f.ret_ty.clone()],
        _ => match &f.ret_ty {
            Type::Tuple(types) if types.len() == ordered_outputs.len() => {
                types.iter().map(|ty| (**ty).clone()).collect()
            }
            _ => {
                return Err(mismatch(format!(
                    "return type {} does not match {} ordered outputs",
                    f.ret_ty,
                    ordered_outputs.len()
                )));
            }
        },
    };
    if metadata.output_names.len() != ordered_outputs.len()
        || metadata.output_port_ids.len() != ordered_outputs.len()
    {
        return Err(mismatch(format!(
            "output compatibility metadata counts ({}, {}) do not match {} ordered outputs",
            metadata.output_names.len(),
            metadata.output_port_ids.len(),
            ordered_outputs.len()
        )));
    }
    for (index, (ordered, ty)) in ordered_outputs.iter().zip(output_types).enumerate() {
        if ordered.ty.as_ref() != Some(&ty) {
            return Err(mismatch(format!(
                "ordered output '{}' type {:?} does not match return element {}",
                ordered.name, ordered.ty, ty
            )));
        }
        if metadata.output_names[index] != ordered.name
            || !metadata.output_port_ids.contains_key(&ordered.name)
        {
            return Err(mismatch(format!(
                "ordered output '{}' does not match output compatibility metadata",
                ordered.name
            )));
        }
    }
    if let Some(reset) = &metadata.reset {
        let reset_port = ordered_inputs
            .iter()
            .find(|port| port.name == reset.port_name)
            .ok_or_else(|| mismatch(format!("reset port '{}' is not an input", reset.port_name)))?;
        if reset_port.ty.as_ref() != Some(&Type::Bits(1)) {
            return Err(mismatch(format!(
                "reset port '{}' must have type bits[1]",
                reset.port_name
            )));
        }
    }
    Ok(())
}

/// Validates a function within the context of its parent package, using a
/// dependency-injected resolver for callee return types.
pub(crate) fn validate_fn_with<F, R>(
    f: &Fn,
    parent: Option<&Package>,
    callee_ret_type_resolver: F,
    register_type_resolver: R,
    instantiation_info: Option<&std::collections::HashMap<String, InstantiationInfo>>,
    allow_registers: bool,
) -> Result<(), ValidationError>
where
    F: std::ops::Fn(&str) -> Option<Type>,
    R: std::ops::Fn(&str) -> Option<Type>,
{
    // Track ids used by non-parameter nodes to ensure uniqueness.
    let mut seen_nonparam_ids: HashSet<usize> = HashSet::new();
    let mut used_instantiation_inputs: std::collections::HashMap<String, HashSet<String>> =
        std::collections::HashMap::new();
    let mut used_instantiation_outputs: std::collections::HashMap<String, HashSet<String>> =
        std::collections::HashMap::new();
    if let Some(info) = instantiation_info {
        for inst_name in info.keys() {
            used_instantiation_inputs.insert(inst_name.clone(), HashSet::new());
            used_instantiation_outputs.insert(inst_name.clone(), HashSet::new());
        }
    }
    // Track GetParam node ids to verify 1:1 mapping with signature params.
    let mut seen_param_ids: HashSet<usize> = HashSet::new();
    // Map parameter names to their declared ids from the function signature, and
    // check name uniqueness.
    let mut param_name_to_id: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for p in &f.params {
        let name = p.name.as_str();
        if param_name_to_id.contains_key(name) {
            return Err(ValidationError::DuplicateParamName {
                func: f.name.clone(),
                param_name: p.name.clone(),
            });
        }
        param_name_to_id.insert(name, p.id.get_wrapped_id());
    }
    for (i, node) in f.nodes.iter().enumerate() {
        // Enforce: if a node has a name that looks like a default textual id
        // pattern '<prefix>.<digits>', then '<prefix>' must match the operator
        // and the numeric suffix must match the node's text id. This aligns with
        // external xlsynth verifier expectations and prevents misleading names.
        if let Some(ref name) = node.name {
            if let Some(dot_pos) = name.rfind('.') {
                let (prefix, suffix) = name.split_at(dot_pos);
                let suffix_digits = &suffix[1..]; // skip '.'
                if !suffix_digits.is_empty() && suffix_digits.chars().all(|c| c.is_ascii_digit()) {
                    let op_str = node.payload.get_operator();
                    if prefix != op_str {
                        return Err(ValidationError::NodeNameOpMismatch {
                            func: f.name.clone(),
                            node_index: i,
                            name: name.clone(),
                            expected_op: op_str.to_string(),
                        });
                    }
                    if let Ok(parsed_id) = suffix_digits.parse::<usize>() {
                        if parsed_id != node.text_id {
                            return Err(ValidationError::NodeNameIdSuffixMismatch {
                                func: f.name.clone(),
                                node_index: i,
                                name: name.clone(),
                                expected_id: node.text_id,
                            });
                        }
                    }
                }
            }
        }
        match &node.payload {
            NodePayload::GetParam(pid) => {
                let declared = node
                    .name
                    .as_ref()
                    .and_then(|n| param_name_to_id.get(n.as_str()))
                    .copied()
                    .unwrap_or(pid.get_wrapped_id());
                let actual_pid = pid.get_wrapped_id();
                // First: mismatch between declared and actual -> ParamIdMismatch.
                if actual_pid != declared || node.text_id != declared {
                    let param_name = node
                        .name
                        .clone()
                        .unwrap_or_else(|| "<unnamed-param>".to_string());
                    return Err(ValidationError::ParamIdMismatch {
                        func: f.name.clone(),
                        param_name,
                        expected: declared,
                        actual: node.text_id,
                    });
                }
                // Ensure this GetParam refers to a declared param id.
                if !param_name_to_id.values().any(|&v| v == actual_pid) {
                    return Err(ValidationError::ExtraParamNode {
                        func: f.name.clone(),
                        text_id: node.text_id,
                    });
                }
                // Ensure each GetParam id appears exactly once in the node list.
                if !seen_param_ids.insert(actual_pid) {
                    return Err(ValidationError::DuplicateTextId {
                        func: f.name.clone(),
                        text_id: actual_pid,
                    });
                }
            }
            _ => {
                if !seen_nonparam_ids.insert(node.text_id) {
                    return Err(ValidationError::DuplicateTextId {
                        func: f.name.clone(),
                        text_id: node.text_id,
                    });
                }
            }
        }
        // Ensure all operands refer to already defined nodes.
        for op in operands(&node.payload) {
            if op.index >= f.nodes.len() {
                return Err(ValidationError::OperandOutOfBounds {
                    func: f.name.clone(),
                    node_index: i,
                    operand: op.index,
                });
            }
            if op.index >= i {
                return Err(ValidationError::OperandUsesUndefined {
                    func: f.name.clone(),
                    node_index: i,
                    operand: op.index,
                });
            }
        }

        if let NodePayload::BitSlice { arg, start, width } = &node.payload {
            if let Type::Bits(operand_width) = &f.get_node(*arg).ty {
                let exceeds_operand = start
                    .checked_add(*width)
                    .map_or(true, |end| end > *operand_width);
                if exceeds_operand {
                    return Err(ValidationError::BitSliceOutOfBounds {
                        func: f.name.clone(),
                        node_index: i,
                        start: *start,
                        width: *width,
                        operand_width: *operand_width,
                    });
                }
            }
        }

        // Validate cross-package references.
        match &node.payload {
            NodePayload::Invoke { to_apply, operands } => {
                let Some(parent) = parent else {
                    return Err(ValidationError::RequiresPackageContext {
                        func: f.name.clone(),
                        node_index: i,
                        op: node.payload.get_operator().to_string(),
                    });
                };
                let Some(callee) = parent.get_fn(to_apply) else {
                    return Err(ValidationError::UnknownCallee {
                        func: f.name.clone(),
                        callee: to_apply.clone(),
                    });
                };
                if operands.len() != callee.params.len() {
                    return Err(ValidationError::NodeSemanticViolation {
                        func: f.name.clone(),
                        node_index: i,
                        reason: format!(
                            "invoke of '{}' expects {} operands, got {}",
                            to_apply,
                            callee.params.len(),
                            operands.len()
                        ),
                    });
                }
                for (operand, parameter) in operands.iter().zip(callee.params.iter()) {
                    if f.get_node(*operand).ty != parameter.ty {
                        return Err(ValidationError::NodeSemanticViolation {
                            func: f.name.clone(),
                            node_index: i,
                            reason: format!(
                                "invoke operand for parameter '{}' must have type {}, got {}",
                                parameter.name,
                                parameter.ty,
                                f.get_node(*operand).ty
                            ),
                        });
                    }
                }
            }
            NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
                invariant_args,
            } => {
                let Some(parent) = parent else {
                    return Err(ValidationError::RequiresPackageContext {
                        func: f.name.clone(),
                        node_index: i,
                        op: node.payload.get_operator().to_string(),
                    });
                };
                let Some(body_fn) = parent.get_fn(body) else {
                    return Err(ValidationError::UnknownCallee {
                        func: f.name.clone(),
                        callee: body.clone(),
                    });
                };
                let expected_param_count = 2 + invariant_args.len();
                if body_fn.params.len() != expected_param_count {
                    return Err(ValidationError::NodeSemanticViolation {
                        func: f.name.clone(),
                        node_index: i,
                        reason: format!(
                            "counted_for body '{}' expects {} parameters, got {}",
                            body,
                            expected_param_count,
                            body_fn.params.len()
                        ),
                    });
                }
                let max_i = stride
                    .checked_mul(trip_count.saturating_sub(1))
                    .ok_or_else(|| ValidationError::NodeSemanticViolation {
                        func: f.name.clone(),
                        node_index: i,
                        reason: "counted_for induction range overflows usize".to_string(),
                    })?;
                let minimum_i_width = if *trip_count <= 1 {
                    1
                } else if max_i == 0 {
                    0
                } else {
                    (usize::BITS - max_i.leading_zeros()) as usize
                };
                match &body_fn.params[0].ty {
                    Type::Bits(width) if *width >= minimum_i_width => {}
                    actual => {
                        return Err(ValidationError::NodeSemanticViolation {
                            func: f.name.clone(),
                            node_index: i,
                            reason: format!(
                                "counted_for body induction parameter must be bits[N] with N >= {}, got {}",
                                minimum_i_width, actual
                            ),
                        });
                    }
                }
                let init_ty = &f.get_node(*init).ty;
                if body_fn.params[1].ty != *init_ty || body_fn.ret_ty != *init_ty {
                    return Err(ValidationError::NodeSemanticViolation {
                        func: f.name.clone(),
                        node_index: i,
                        reason: format!(
                            "counted_for body '{}' carry parameter and result must have type {}",
                            body, init_ty
                        ),
                    });
                }
                for (argument, parameter) in invariant_args.iter().zip(body_fn.params[2..].iter()) {
                    if f.get_node(*argument).ty != parameter.ty {
                        return Err(ValidationError::NodeSemanticViolation {
                            func: f.name.clone(),
                            node_index: i,
                            reason: format!(
                                "counted_for invariant parameter '{}' must have type {}, got {}",
                                parameter.name,
                                parameter.ty,
                                f.get_node(*argument).ty
                            ),
                        });
                    }
                }
            }
            _ => {}
        }

        // Validate register usage (block-only).
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                if !allow_registers {
                    return Err(ValidationError::RegisterOpInFunction {
                        func: f.name.clone(),
                        node_index: i,
                    });
                }
                if register_type_resolver(register).is_none() {
                    return Err(ValidationError::UnknownRegister {
                        func: f.name.clone(),
                        node_index: i,
                        register: register.clone(),
                    });
                }
            }
            NodePayload::RegisterWrite {
                register,
                arg,
                load_enable,
                reset,
            } => {
                if !allow_registers {
                    return Err(ValidationError::RegisterOpInFunction {
                        func: f.name.clone(),
                        node_index: i,
                    });
                }
                let reg_ty = register_type_resolver(register).ok_or_else(|| {
                    ValidationError::UnknownRegister {
                        func: f.name.clone(),
                        node_index: i,
                        register: register.clone(),
                    }
                })?;
                let arg_ty = f.get_node(*arg).ty.clone();
                if reg_ty != arg_ty {
                    return Err(ValidationError::RegisterWriteTypeMismatch {
                        func: f.name.clone(),
                        node_index: i,
                        register: register.clone(),
                        expected: reg_ty,
                        actual: arg_ty,
                    });
                }
                if let Some(le) = load_enable {
                    let le_ty = f.get_node(*le).ty.clone();
                    if le_ty != Type::Bits(1) {
                        return Err(ValidationError::RegisterWriteLoadEnableTypeMismatch {
                            func: f.name.clone(),
                            node_index: i,
                            actual: le_ty,
                        });
                    }
                }
                if let Some(rst) = reset {
                    let rst_ty = f.get_node(*rst).ty.clone();
                    if rst_ty != Type::Bits(1) {
                        return Err(ValidationError::RegisterWriteResetTypeMismatch {
                            func: f.name.clone(),
                            node_index: i,
                            actual: rst_ty,
                        });
                    }
                }
            }
            NodePayload::InstantiationInput {
                instantiation,
                port_name,
                arg,
            } => {
                let Some(info_map) = instantiation_info else {
                    return Err(ValidationError::InstantiationOpInFunction {
                        func: f.name.clone(),
                        node_index: i,
                    });
                };
                let inst_info = info_map.get(instantiation).ok_or_else(|| {
                    ValidationError::UnknownInstantiation {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                    }
                })?;
                let expected_ty = inst_info.input_types.get(port_name).ok_or_else(|| {
                    ValidationError::UnknownInstantiationPort {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                        port_name: port_name.clone(),
                        direction: InstantiationPortDirection::Input,
                    }
                })?;
                let arg_ty = f.get_node(*arg).ty.clone();
                if &arg_ty != expected_ty {
                    return Err(ValidationError::InstantiationPortTypeMismatch {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                        port_name: port_name.clone(),
                        direction: InstantiationPortDirection::Input,
                        expected: expected_ty.clone(),
                        actual: arg_ty,
                    });
                }
                let used_ports = used_instantiation_inputs
                    .get_mut(instantiation)
                    .expect("instantiation input map must exist");
                if !used_ports.insert(port_name.clone()) {
                    return Err(ValidationError::DuplicateInstantiationPort {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                        port_name: port_name.clone(),
                        direction: InstantiationPortDirection::Input,
                    });
                }
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } => {
                let Some(info_map) = instantiation_info else {
                    return Err(ValidationError::InstantiationOpInFunction {
                        func: f.name.clone(),
                        node_index: i,
                    });
                };
                let inst_info = info_map.get(instantiation).ok_or_else(|| {
                    ValidationError::UnknownInstantiation {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                    }
                })?;
                let expected_ty = inst_info.output_types.get(port_name).ok_or_else(|| {
                    ValidationError::UnknownInstantiationPort {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                        port_name: port_name.clone(),
                        direction: InstantiationPortDirection::Output,
                    }
                })?;
                if &node.ty != expected_ty {
                    return Err(ValidationError::InstantiationPortTypeMismatch {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                        port_name: port_name.clone(),
                        direction: InstantiationPortDirection::Output,
                        expected: expected_ty.clone(),
                        actual: node.ty.clone(),
                    });
                }
                let used_ports = used_instantiation_outputs
                    .get_mut(instantiation)
                    .expect("instantiation output map must exist");
                if !used_ports.insert(port_name.clone()) {
                    return Err(ValidationError::DuplicateInstantiationPort {
                        func: f.name.clone(),
                        node_index: i,
                        instantiation: instantiation.clone(),
                        port_name: port_name.clone(),
                        direction: InstantiationPortDirection::Output,
                    });
                }
            }
            _ => {}
        }

        // Enforce that bitwise n-ary ops have identically typed bit operands.
        if let NodePayload::Nary(op, elems) = &node.payload {
            match op {
                NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor => {
                    if elems.is_empty() {
                        return Err(ValidationError::NodeSemanticViolation {
                            func: f.name.clone(),
                            node_index: i,
                            reason: format!(
                                "{} requires at least one operand",
                                node.payload.get_operator()
                            ),
                        });
                    }
                    let first_ty = f.get_node(elems[0]).ty.clone();
                    log::trace!(
                        "validating nary op: {:?} first_ty: {:?}",
                        node.payload,
                        first_ty
                    );
                    // Require bits type and identical types across all operands.
                    for nr in elems.iter().skip(1) {
                        let operand_ty = &f.get_node(*nr).ty;
                        log::trace!(
                            "=> validating nary op: {:?} operand_ty: {:?}",
                            node.payload,
                            operand_ty
                        );
                        if operand_ty != &first_ty {
                            return Err(ValidationError::NaryBitwiseOperandTypeMismatch {
                                func: f.name.clone(),
                                node_index: i,
                            });
                        }
                    }
                }
                NaryOp::Concat => {
                    // Does not require identical types across all operands.
                }
            }
        }

        if let NodePayload::ExtNaryAdd { terms, arch: _ } = &node.payload {
            if !matches!(node.ty, Type::Bits(_)) {
                return Err(ValidationError::ExtNaryAddResultTypeMismatch {
                    func: f.name.clone(),
                    node_index: i,
                    actual: node.ty.clone(),
                });
            }
            for term in terms.iter() {
                if !matches!(f.get_node(term.operand).ty, Type::Bits(_)) {
                    return Err(ValidationError::ExtNaryAddOperandTypeMismatch {
                        func: f.name.clone(),
                        node_index: i,
                    });
                }
            }
        }

        if let NodePayload::ExtMaskLow { count } = &node.payload {
            if !matches!(node.ty, Type::Bits(_)) {
                return Err(ValidationError::ExtMaskLowResultTypeMismatch {
                    func: f.name.clone(),
                    node_index: i,
                    actual: node.ty.clone(),
                });
            }
            if !matches!(f.get_node(*count).ty, Type::Bits(_)) {
                return Err(ValidationError::ExtMaskLowCountTypeMismatch {
                    func: f.name.clone(),
                    node_index: i,
                });
            }
        }

        if matches!(
            node.payload,
            NodePayload::Binop(Binop::Smulp | Binop::Umulp, _, _)
        ) {
            let valid_result_type = matches!(
                &node.ty,
                Type::Tuple(members)
                    if members.len() == 2
                        && members[0] == members[1]
                        && matches!(members[0].as_ref(), Type::Bits(_))
            );
            if !valid_result_type {
                return Err(ValidationError::PartialProductResultTypeMismatch {
                    func: f.name.clone(),
                    node_index: i,
                    actual: node.ty.clone(),
                });
            }
        }

        crate::ir_verify::verify_node_xls_semantics(f, i).map_err(|reason| {
            ValidationError::NodeSemanticViolation {
                func: f.name.clone(),
                node_index: i,
                reason,
            }
        })?;

        // After structural checks, ensure deduced node type matches declared.
        let op_refs = operands(&node.payload);
        let mut op_types: Vec<Type> = Vec::with_capacity(op_refs.len());
        for nr in op_refs.iter() {
            op_types.push(f.get_node(*nr).ty.clone());
        }
        match deduce_result_type_with_registers(
            &node.payload,
            &op_types,
            |callee| callee_ret_type_resolver(callee),
            |register| register_type_resolver(register),
        ) {
            Ok(Some(deduced)) => {
                if deduced != node.ty {
                    return Err(ValidationError::NodeTypeMismatch {
                        func: f.name.clone(),
                        node_index: i,
                        deduced,
                        actual: node.ty.clone(),
                    });
                }
            }
            Ok(None) => {
                // No deduction available for this payload; skip.
            }
            Err(e) => {
                return Err(ValidationError::TypeDeductionFailure {
                    func: f.name.clone(),
                    node_index: i,
                    reason: e.to_string(),
                });
            }
        }
    }

    if let Some(info_map) = instantiation_info {
        for (inst_name, inst_info) in info_map.iter() {
            let used_inputs = used_instantiation_inputs
                .get(inst_name)
                .cloned()
                .unwrap_or_default();
            let missing_inputs: Vec<String> = inst_info
                .input_types
                .keys()
                .filter(|k| !used_inputs.contains(*k))
                .cloned()
                .collect();
            if !missing_inputs.is_empty() {
                return Err(ValidationError::MissingInstantiationPorts {
                    func: f.name.clone(),
                    instantiation: inst_name.clone(),
                    missing: missing_inputs,
                    direction: InstantiationPortDirection::Input,
                });
            }
            let used_outputs = used_instantiation_outputs
                .get(inst_name)
                .cloned()
                .unwrap_or_default();
            let missing_outputs: Vec<String> = inst_info
                .output_types
                .keys()
                .filter(|k| !used_outputs.contains(*k))
                .cloned()
                .collect();
            if !missing_outputs.is_empty() {
                return Err(ValidationError::MissingInstantiationPorts {
                    func: f.name.clone(),
                    instantiation: inst_name.clone(),
                    missing: missing_outputs,
                    direction: InstantiationPortDirection::Output,
                });
            }
        }
    }
    // Ensure every declared parameter has a corresponding GetParam node.
    for p in &f.params {
        let pid = p.id.get_wrapped_id();
        if !seen_param_ids.contains(&pid) {
            return Err(ValidationError::MissingParamNode {
                func: f.name.clone(),
                param_name: p.name.clone(),
                expected_id: pid,
            });
        }
    }

    let ret_node_ref = f
        .ret_node_ref
        .ok_or_else(|| ValidationError::MissingReturnNode(f.name.clone()))?;
    let ret_node = f.get_node(ret_node_ref);
    if ret_node.ty != f.ret_ty {
        return Err(ValidationError::ReturnTypeMismatch {
            func: f.name.clone(),
            expected: f.ret_ty.clone(),
            actual: ret_node.ty.clone(),
        });
    }

    Ok(())
}

fn collect_prior_blocks<'a>(
    p: &'a Package,
    member_index: usize,
) -> std::collections::HashMap<String, (&'a Fn, &'a BlockMetadata)> {
    let mut prior = std::collections::HashMap::new();
    for member in p.members.iter().take(member_index) {
        if let PackageMember::Block { func, metadata } = member {
            prior.insert(func.name.clone(), (func, metadata));
        }
    }
    prior
}

fn build_instantiation_info(
    metadata: &BlockMetadata,
    prior_blocks: &std::collections::HashMap<String, (&Fn, &BlockMetadata)>,
    parent: &Package,
) -> Result<std::collections::HashMap<String, InstantiationInfo>, ValidationError> {
    let mut info_map = std::collections::HashMap::new();
    for inst in metadata.instantiations.iter() {
        if inst.kind == InstantiationKind::Extern {
            let callee_fn = parent
                .get_fn(&inst.block)
                .expect("foreign function existence was validated");
            let input_types = callee_fn
                .params
                .iter()
                .map(|param| (param.name.clone(), param.ty.clone()))
                .collect();
            let output_types =
                std::collections::HashMap::from([("return".to_string(), callee_fn.ret_ty.clone())]);
            info_map.insert(
                inst.name.clone(),
                InstantiationInfo {
                    input_types,
                    output_types,
                },
            );
            continue;
        }
        let (callee_fn, callee_meta) = prior_blocks
            .get(&inst.block)
            .expect("prior block missing after check");
        let mut input_types = std::collections::HashMap::new();
        for p in callee_fn.params.iter() {
            input_types.insert(p.name.clone(), p.ty.clone());
        }
        let mut output_types = std::collections::HashMap::new();
        if callee_meta.output_names.is_empty() {
            // no outputs
        } else if callee_meta.output_names.len() == 1 {
            output_types.insert(
                callee_meta.output_names[0].clone(),
                callee_fn.ret_ty.clone(),
            );
        } else {
            match &callee_fn.ret_ty {
                Type::Tuple(tys) => {
                    if tys.len() != callee_meta.output_names.len() {
                        return Err(ValidationError::BlockOutputArityMismatch {
                            func: callee_fn.name.clone(),
                            expected: callee_meta.output_names.len(),
                            actual: tys.len(),
                        });
                    }
                    for (name, ty) in callee_meta.output_names.iter().zip(tys.iter()) {
                        output_types.insert(name.clone(), (**ty).clone());
                    }
                }
                _ => {
                    return Err(ValidationError::BlockOutputArityMismatch {
                        func: callee_fn.name.clone(),
                        expected: callee_meta.output_names.len(),
                        actual: 1,
                    });
                }
            }
        }
        info_map.insert(
            inst.name.clone(),
            InstantiationInfo {
                input_types,
                output_types,
            },
        );
    }
    Ok(info_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;
    use crate::ir_parser::Parser;

    #[test]
    fn validate_package_ok() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          ret add.2: bits[1] = add(x, x)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        validate_package(&pkg).unwrap();
    }

    #[test]
    fn ordered_block_ports_must_match_compatibility_metadata() {
        let ir = r#"
package test

top block mixed(x: bits[8], y: bits[8], clk: clock, rst: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  x: bits[8] = input_port(name=x, id=1)
  rst: bits[1] = input_port(name=rst, id=2)
  y: () = output_port(x, name=y, id=3)
}
"#;
        let mut pkg = Parser::new_preserving_block_port_order(ir)
            .parse_package()
            .unwrap();
        validate_package(&pkg).unwrap();
        let PackageMember::Block { metadata, .. } = &mut pkg.members[0] else {
            panic!("expected block");
        };
        metadata.output_names[0] = "wrong".to_string();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::BlockPortMetadataMismatch { .. })
        ));
    }

    #[test]
    fn output_port_ids_participate_in_package_wide_uniqueness() {
        let ir = r#"
package test

top block passthrough(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
"#;
        let mut pkg = Parser::new(ir).parse_package().unwrap();
        let PackageMember::Block { metadata, .. } = &mut pkg.members[0] else {
            panic!("expected block");
        };
        metadata.output_port_ids.insert("y".to_string(), 1);
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::DuplicateTextId { text_id: 1, .. })
        ));
        let PackageMember::Block { metadata, .. } = &mut pkg.members[0] else {
            panic!("expected block");
        };
        metadata.output_port_ids.insert("y".to_string(), 0);
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::BlockPortMetadataMismatch { .. })
        ));
    }

    #[test]
    fn partial_product_accepts_explicit_result_width_independent_of_operands() {
        let ir = r#"
        package test

        fn foo(x: bits[6]) -> (bits[29], bits[29]) {
          ret split: (bits[29], bits[29]) = umulp(x, x, id=2)
        }
        "#;
        let pkg = Parser::new(ir).parse_package().unwrap();
        validate_package(&pkg).unwrap();
    }

    #[test]
    fn partial_product_rejects_non_pair_result_type() {
        let ir = r#"
        package test

        fn foo(x: bits[6]) -> bits[29] {
          ret split: bits[29] = umulp(x, x, id=2)
        }
        "#;
        let pkg = Parser::new(ir).parse_package().unwrap();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::PartialProductResultTypeMismatch { .. })
        ));
    }

    #[test]
    fn static_bit_slice_rejects_range_beyond_operand_width() {
        let ir = r#"
        package test

        fn foo(x: bits[8] id=1) -> bits[2] {
          ret slice: bits[2] = bit_slice(x, start=7, width=2, id=2)
        }
        "#;
        let pkg = Parser::new(ir).parse_package().unwrap();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::BitSliceOutOfBounds {
                start: 7,
                width: 2,
                operand_width: 8,
                ..
            })
        ));
    }

    #[test]
    fn undefined_operand_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          tmp: bits[1] = add(x, x, id=2)
          ret neg: bits[1] = neg(tmp, id=3)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg
                .members
                .iter_mut()
                .find_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    _ => None,
                })
                .unwrap();
            // Make node 1 reference a future node (index 2).
            if let NodePayload::Binop(_, ref mut a, _) = f.nodes[2].payload {
                *a = ir::NodeRef { index: 2 };
            }
        }
        let f = pkg
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) => Some(f),
                _ => None,
            })
            .unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::OperandUsesUndefined { .. })
        ));
    }

    #[test]
    fn return_type_mismatch_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          ret add.2: bits[1] = add(x, x)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg
                .members
                .iter_mut()
                .find_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    _ => None,
                })
                .unwrap();
            f.ret_ty = Type::Bits(2);
        }
        let f = pkg
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) => Some(f),
                _ => None,
            })
            .unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::ReturnTypeMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_text_id_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1]) -> bits[1] {
          a: bits[1] = add(x, x, id=2)
          b: bits[1] = add(a, x, id=2)
          ret b: bits[1] = identity(b, id=3)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::DuplicateTextId { .. })
        ));
    }

    #[test]
    fn name_operator_prefix_mismatch_fails() {
        let ir = r#"
        package test

        fn foo() -> bits[8] {
          ret one.2: bits[8] = literal(value=1, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let err = parser.parse_package().unwrap_err();
        assert_eq!(
            format!("{}", err),
            "ParseError: node name dotted prefix 'one' does not match operator 'literal'"
        );
    }

    #[test]
    fn manual_construct_one_dot_id_literal_fails() {
        // Build a function programmatically containing a node named "one.2"
        // with operator literal(id=2). This should fail with NodeNameOpMismatch.
        let mut pkg = ir::Package {
            name: "test".to_string(),
            file_table: ir::FileTable::new(),
            members: Vec::new(),
            top: Some(("f".to_string(), ir::MemberType::Function)),
        };
        let lit_node = ir::Node {
            text_id: 2,
            name: Some("one.2".to_string()),
            ty: ir::Type::Bits(8),
            payload: ir::NodePayload::Literal(xlsynth::IrValue::make_ubits(8, 1).unwrap()),
            pos: None,
        };
        let f = ir::Fn {
            name: "f".to_string(),
            params: Vec::new(),
            ret_ty: ir::Type::Bits(8),
            nodes: vec![
                ir::Node {
                    text_id: 0,
                    name: Some("reserved_zero_node".to_string()),
                    ty: ir::Type::nil(),
                    payload: ir::NodePayload::Nil,
                    pos: None,
                },
                lit_node,
            ],
            ret_node_ref: Some(ir::NodeRef { index: 1 }),
            outer_attrs: Vec::new(),
            inner_attrs: Vec::new(),
        };
        pkg.members.push(ir::PackageMember::Function(f.clone()));
        let fref = pkg.get_top_fn().unwrap();
        assert!(matches!(
            super::validate_fn(fref, &pkg),
            Err(ValidationError::NodeNameOpMismatch { .. })
        ));
    }

    #[test]
    fn param_id_mismatch_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[1] id=7) -> bits[1] {
          x: bits[1] = param(name=x, id=1)
          ret x: bits[1] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        // Now rejected at parse-time due to name/id mismatch on param node.
        let err = parser.parse_package().unwrap_err();
        assert_eq!(
            format!("{}", err),
            "ParseError: param name/id mismatch: name=x id=1"
        );
    }

    #[test]
    fn unknown_callee_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8]) -> bits[8] {
          ret r: bits[8] = invoke(x, to_apply=bar, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::UnknownCallee { .. })
        ));
    }

    #[test]
    fn duplicate_getparam_node_id_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8]) -> bits[8] {
          ret x: bits[8] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg.get_top_fn_mut().unwrap();
            // Manually insert a duplicate GetParam node with the same id as 'x'.
            let pid = f.params[0].id;
            let dup = ir::Node {
                text_id: pid.get_wrapped_id(),
                name: Some(f.params[0].name.clone()),
                ty: f.params[0].ty.clone(),
                payload: ir::NodePayload::GetParam(pid),
                pos: None,
            };
            f.nodes.push(dup);
        }
        let f_ro = pkg.get_top_fn().unwrap();
        assert!(matches!(
            validate_fn(f_ro, &pkg),
            Err(ValidationError::DuplicateTextId { .. })
        ));
    }

    #[test]
    fn missing_getparam_node_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8]) -> bits[8] {
          ret x: bits[8] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let mut pkg = parser.parse_package().unwrap();
        {
            let f = pkg.get_top_fn_mut().unwrap();
            // Remove the GetParam node for 'x'. It should be at index 1.
            let idx = f
                .nodes
                .iter()
                .position(|n| matches!(n.payload, NodePayload::GetParam(_)))
                .unwrap();
            f.nodes.remove(idx);
        }
        let f_ro = pkg.get_top_fn().unwrap();
        let err = validate_fn(f_ro, &pkg).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::MissingParamNode { .. } | ValidationError::OperandUsesUndefined { .. }
        ));
    }

    #[test]
    fn duplicate_param_name_fails() {
        let ir = r#"
        package test

        fn foo(x: bits[8], x: bits[8]) -> bits[8] {
          ret x: bits[8] = identity(x, id=2)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        assert!(matches!(
            validate_fn(f, &pkg),
            Err(ValidationError::DuplicateParamName { .. })
        ));
    }

    #[test]
    fn package_level_invoke_type_mismatch_fails() {
        let ir = r#"
        package test

        fn callee(x: bits[1] id=1) -> (bits[1], bits[1]) {
          ret tuple.3: (bits[1], bits[1]) = tuple(x, x, id=3)
        }

        fn foo(x: bits[1] id=1) -> bits[1] {
          invoke.2: bits[1] = invoke(x, to_apply=callee, id=2)
          ret identity.3: bits[1] = identity(invoke.2, id=3)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        // Public entry point should surface a node type mismatch error.
        let err = validate_package(&pkg).unwrap_err();
        match err {
            ValidationError::NodeTypeMismatch { .. } => {}
            other => panic!("expected NodeTypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn package_level_recursive_invoke_fails() {
        let ir = r#"
        package test

        fn first(x: bits[1] id=1) -> bits[1] {
          ret invoke.2: bits[1] = invoke(x, to_apply=second, id=2)
        }

        fn second(x: bits[1] id=3) -> bits[1] {
          ret invoke.4: bits[1] = invoke(x, to_apply=first, id=4)
        }
        "#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::RecursiveFunctionReference { .. })
        ));
    }

    #[test]
    fn instantiation_requires_prior_block() {
        let ir = r#"
package inst_test

block my_block(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32]) {
  instantiation inst_add(block=add_block, kind=block)
  in0: bits[32] = input_port(name=in0, id=1)
  in1: bits[32] = input_port(name=in1, id=2)
  instantiation_input.10: () = instantiation_input(in0, instantiation=inst_add, port_name=a, id=10)
  instantiation_input.11: () = instantiation_input(in1, instantiation=inst_add, port_name=b, id=11)
  instantiation_output.12: bits[32] = instantiation_output(instantiation=inst_add, port_name=x, id=12)
  instantiation_output.13: bits[32] = instantiation_output(instantiation=inst_add, port_name=y, id=13)
  out0: () = output_port(instantiation_output.12, name=out0, id=14)
  out1: () = output_port(instantiation_output.13, name=out1, id=15)
}

block add_block(a: bits[32], b: bits[32], x: bits[32], y: bits[32]) {
  a: bits[32] = input_port(name=a, id=3)
  b: bits[32] = input_port(name=b, id=4)
  x: () = output_port(a, name=x, id=5)
  y: () = output_port(b, name=y, id=6)
}
"#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::InstantiationBlockNotFound { .. })
        ));
    }

    #[test]
    fn instantiation_missing_input_port_fails() {
        let ir = r#"
package inst_test

block add_block(a: bits[32], b: bits[32], x: bits[32], y: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  x: () = output_port(a, name=x, id=3)
  y: () = output_port(b, name=y, id=4)
}

block my_block(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32]) {
  instantiation inst_add(block=add_block, kind=block)
  in0: bits[32] = input_port(name=in0, id=5)
  in1: bits[32] = input_port(name=in1, id=6)
  instantiation_input.10: () = instantiation_input(in0, instantiation=inst_add, port_name=a, id=10)
  instantiation_output.12: bits[32] = instantiation_output(instantiation=inst_add, port_name=x, id=12)
  instantiation_output.13: bits[32] = instantiation_output(instantiation=inst_add, port_name=y, id=13)
  out0: () = output_port(instantiation_output.12, name=out0, id=14)
  out1: () = output_port(instantiation_output.13, name=out1, id=15)
}
"#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::MissingInstantiationPorts {
                direction: InstantiationPortDirection::Input,
                ..
            })
        ));
    }

    #[test]
    fn instantiation_missing_output_port_fails() {
        let ir = r#"
package inst_test

block add_block(a: bits[32], b: bits[32], x: bits[32], y: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  x: () = output_port(a, name=x, id=3)
  y: () = output_port(b, name=y, id=4)
}

block my_block(in0: bits[32], in1: bits[32], out0: bits[32], out1: bits[32]) {
  instantiation inst_add(block=add_block, kind=block)
  in0: bits[32] = input_port(name=in0, id=5)
  in1: bits[32] = input_port(name=in1, id=6)
  instantiation_input.10: () = instantiation_input(in0, instantiation=inst_add, port_name=a, id=10)
  instantiation_input.11: () = instantiation_input(in1, instantiation=inst_add, port_name=b, id=11)
  instantiation_output.12: bits[32] = instantiation_output(instantiation=inst_add, port_name=x, id=12)
  out0: () = output_port(instantiation_output.12, name=out0, id=14)
  out1: () = output_port(instantiation_output.12, name=out1, id=15)
}
"#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        assert!(matches!(
            validate_package(&pkg),
            Err(ValidationError::MissingInstantiationPorts {
                direction: InstantiationPortDirection::Output,
                ..
            })
        ));
    }
}
