// SPDX-License-Identifier: Apache-2.0

//! Package-context portions of PIR verification.

use std::collections::HashSet;

use crate::ir::{
    Binop, Block, BlockPort, Fn, InstantiationKind, MemberType, NaryOp, NodeGraph, NodePayload,
    NodeRef, Package, PackageMember, Type,
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
    /// A non-Nil node's textual ID is not unique within its graph.
    DuplicateTextId { func: String, text_id: usize },
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
    /// A block's reset metadata does not identify a bits[1] input port.
    InvalidBlockResetPort { block: String, port: String },
    /// A block's declared resources violate a structural invariant.
    BlockInvariantViolation { block: String, reason: String },
    /// A declared register does not have exactly one register_read operation.
    RegisterReadCountMismatch {
        block: String,
        register: String,
        actual: usize,
    },
    /// Register reset values and write reset operands must either both exist or
    /// both be absent.
    RegisterWriteResetPresenceMismatch {
        func: String,
        node_index: usize,
        register: String,
        register_has_reset_value: bool,
        write_has_reset: bool,
    },
    /// Every write to a multiply written register requires a load enable.
    MultipleRegisterWritesRequireLoadEnable {
        block: String,
        node_index: usize,
        register: String,
    },
    /// Every write to a multiply written register must use the same reset.
    MultipleRegisterWritesResetMismatch {
        block: String,
        node_index: usize,
        register: String,
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
    /// An external instantiation references a function absent from the package.
    InstantiationForeignFunctionNotFound {
        func: String,
        instantiation: String,
        foreign_function: String,
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
    /// A signature reference does not point to a parameter node.
    MissingParamNode { func: String, node_ref: NodeRef },
    /// A Param node exists in the node list that does not correspond to any
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
            ValidationError::BlockInvariantViolation { block, reason } => {
                write!(f, "block '{block}' violates a resource invariant: {reason}")
            }
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
            ValidationError::InvalidBlockResetPort { block, port } => {
                write!(
                    f,
                    "block '{}' reset port '{}' must be a declared bits[1] input",
                    block, port
                )
            }
            ValidationError::RegisterReadCountMismatch {
                block,
                register,
                actual,
            } => {
                write!(
                    f,
                    "block '{}' register '{}' requires exactly one register_read, found {}",
                    block, register, actual
                )
            }
            ValidationError::RegisterWriteResetPresenceMismatch {
                func,
                node_index,
                register,
                register_has_reset_value,
                write_has_reset,
            } => {
                write!(
                    f,
                    "block '{}' node {} register '{}' reset value presence ({}) does not match register_write reset operand presence ({})",
                    func, node_index, register, register_has_reset_value, write_has_reset
                )
            }
            ValidationError::MultipleRegisterWritesRequireLoadEnable {
                block,
                node_index,
                register,
            } => {
                write!(
                    f,
                    "block '{}' node {} register '{}' has multiple writes but this register_write has no load_enable",
                    block, node_index, register
                )
            }
            ValidationError::MultipleRegisterWritesResetMismatch {
                block,
                node_index,
                register,
            } => {
                write!(
                    f,
                    "block '{}' node {} register '{}' has multiple writes with different reset operands",
                    block, node_index, register
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
            ValidationError::InstantiationForeignFunctionNotFound {
                func,
                instantiation,
                foreign_function,
            } => {
                write!(
                    f,
                    "function '{}' external instantiation '{}' references missing foreign function '{}'",
                    func, instantiation, foreign_function
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
            ValidationError::MissingParamNode { func, node_ref } => {
                write!(
                    f,
                    "function '{}' signature references missing parameter node {}",
                    func, node_ref.index
                )
            }
            ValidationError::ExtraParamNode { func, text_id } => {
                write!(
                    f,
                    "function '{}' has Param node with id {} not declared in signature",
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
    require_complete_mapping: bool,
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
            PackageMember::Block(block) => (block.name.clone(), MemberType::Block),
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
            PackageMember::Block(block) => validate_block(block, p, idx)?,
        }
    }
    validate_function_call_graph(p)?;

    // Enforce package-wide uniqueness of node text ids (including parameter
    // nodes).
    let mut seen_ids: HashSet<usize> = HashSet::new();
    for member in &p.members {
        let f: &NodeGraph = match member {
            PackageMember::Function(f) => f,
            PackageMember::Block(block) => block,
        };
        for node in &f.nodes {
            if matches!(node.payload, NodePayload::Nil) {
                // The reserved sentinel is not emitted.
                continue;
            }
            if !seen_ids.insert(node.text_id) {
                return Err(ValidationError::DuplicateTextId {
                    func: f.name.clone(),
                    text_id: node.text_id,
                });
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
        |name: &str| parent.get_fn(name).map(|callee| callee.ret_ty.clone()),
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
    f: &Block,
    parent: &Package,
    member_index: usize,
) -> Result<(), ValidationError> {
    validate_block_ports(f)?;
    validate_block_resources(f)?;
    let prior_blocks = collect_prior_blocks(parent, member_index);
    for inst in f.instantiations.iter() {
        match inst.kind {
            InstantiationKind::Block if !prior_blocks.contains_key(&inst.block) => {
                return Err(ValidationError::InstantiationBlockNotFound {
                    func: f.name.clone(),
                    instantiation: inst.name.clone(),
                    block: inst.block.clone(),
                });
            }
            InstantiationKind::Extern if parent.get_fn(&inst.block).is_none() => {
                return Err(ValidationError::InstantiationForeignFunctionNotFound {
                    func: f.name.clone(),
                    instantiation: inst.name.clone(),
                    foreign_function: inst.block.clone(),
                });
            }
            _ => {
                // The declared target exists with the kind required by this
                // instance.
            }
        }
    }
    let instantiation_info = build_instantiation_info(f, &prior_blocks, parent)?;
    validate_graph_with(
        f,
        Some(parent),
        |name: &str| parent.get_fn(name).map(|callee| callee.ret_ty.clone()),
        |register| {
            f.registers
                .iter()
                .find(|r| r.name == register)
                .map(|r| r.ty.clone())
        },
        Some(&instantiation_info),
        true,
    )?;
    validate_block_registers(f)
}

/// Checks declarations before any resolver indexes resources by name.
fn validate_block_resources(block: &Block) -> Result<(), ValidationError> {
    let violation = |reason| ValidationError::BlockInvariantViolation {
        block: block.name.clone(),
        reason,
    };
    if !block.registers.is_empty() && block.clock_port_name().is_none() {
        return Err(violation("registers require a clock port".to_string()));
    }
    let mut names = HashSet::new();
    for register in &block.registers {
        if !names.insert(register.name.as_str()) {
            return Err(violation(format!("duplicate register '{}'", register.name)));
        }
        if let Some(value) = &register.reset_value {
            if value.type_() != register.ty {
                return Err(violation(format!(
                    "reset value for register '{}' has type {}, expected {}",
                    register.name,
                    value.type_(),
                    register.ty
                )));
            }
            if block.reset.is_none() {
                return Err(violation(format!(
                    "register '{}' has a reset value but the block has no reset port",
                    register.name
                )));
            }
        }
    }
    names.clear();
    for instance in &block.instantiations {
        if !names.insert(instance.name.as_str()) {
            return Err(violation(format!(
                "duplicate instantiation '{}'",
                instance.name
            )));
        }
    }
    Ok(())
}

/// Checks that the ordered block interface names each actual port exactly once.
fn validate_block_ports(block: &Block) -> Result<(), ValidationError> {
    let violation = |node_index, reason: String| ValidationError::NodeSemanticViolation {
        func: block.name.clone(),
        node_index,
        reason,
    };
    if !block
        .nodes
        .first()
        .is_some_and(|node| matches!(node.payload, NodePayload::Nil))
    {
        return Err(violation(
            0,
            "block graph must start with a Nil sentinel".to_string(),
        ));
    }
    let mut names = HashSet::new();
    let mut port_nodes = HashSet::new();
    let mut clock_seen = false;
    for port in &block.ports {
        let (name, node_index) = match port {
            BlockPort::Clock(name) => {
                if clock_seen {
                    return Err(violation(
                        0,
                        "block has more than one clock port".to_string(),
                    ));
                }
                clock_seen = true;
                (name.as_str(), 0)
            }
            BlockPort::Input(reference) | BlockPort::Output(reference) => {
                let node = block.nodes.get(reference.index).ok_or_else(|| {
                    violation(
                        reference.index,
                        "block port references a missing node".to_string(),
                    )
                })?;
                let name = match (port, &node.payload) {
                    (BlockPort::Input(_), NodePayload::InputPort { name, .. })
                    | (BlockPort::Output(_), NodePayload::OutputPort { name, .. }) => name,
                    _ => {
                        return Err(violation(
                            reference.index,
                            "block port direction does not match its node".to_string(),
                        ));
                    }
                };
                if !port_nodes.insert(*reference) {
                    return Err(violation(
                        reference.index,
                        "block port is listed more than once".to_string(),
                    ));
                }
                (name.as_str(), reference.index)
            }
        };
        if name.is_empty() || !names.insert(name) {
            return Err(violation(
                node_index,
                format!("empty or duplicate block port name '{name}'"),
            ));
        }
    }
    for (index, node) in block.nodes.iter().enumerate() {
        match node.payload {
            NodePayload::InputPort { .. } | NodePayload::OutputPort { .. } => {
                if !port_nodes.contains(&NodeRef { index }) {
                    return Err(violation(
                        index,
                        "port node is missing from the block interface".to_string(),
                    ));
                }
            }
            NodePayload::Param => {
                return Err(violation(
                    index,
                    "blocks must use input ports, not function parameters".to_string(),
                ));
            }
            _ => {
                // Ordinary graph nodes are not part of the declared interface.
            }
        }
    }
    Ok(())
}

/// Applies the reset and register-write invariants required by XLS blocks.
fn validate_block_registers(f: &Block) -> Result<(), ValidationError> {
    if let Some(reset) = &f.reset {
        let valid_reset = f.input_ports().any(|port| port == reset.port)
            && f.get_node(reset.port).ty == Type::Bits(1);
        if !valid_reset {
            return Err(ValidationError::InvalidBlockResetPort {
                block: f.name.clone(),
                port: f
                    .nodes
                    .get(reset.port.index)
                    .map(|node| {
                        node.name
                            .clone()
                            .unwrap_or_else(|| node.text_id.to_string())
                    })
                    .unwrap_or_else(|| format!("node {}", reset.port.index)),
            });
        }
    }

    let definitions = f
        .registers
        .iter()
        .map(|register| (register.name.as_str(), register))
        .collect::<std::collections::HashMap<_, _>>();
    let mut reads = std::collections::HashMap::<&str, usize>::new();
    let mut writes =
        std::collections::HashMap::<&str, Vec<(usize, Option<NodeRef>, Option<NodeRef>)>>::new();
    for (index, node) in f.nodes.iter().enumerate() {
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                *reads.entry(register).or_default() += 1;
            }
            NodePayload::RegisterWrite {
                register,
                load_enable,
                reset,
                ..
            } => {
                let definition = definitions[register.as_str()];
                if definition.reset_value.is_some() != reset.is_some() {
                    return Err(ValidationError::RegisterWriteResetPresenceMismatch {
                        func: f.name.clone(),
                        node_index: index,
                        register: register.clone(),
                        register_has_reset_value: definition.reset_value.is_some(),
                        write_has_reset: reset.is_some(),
                    });
                }
                writes
                    .entry(register)
                    .or_default()
                    .push((index, *load_enable, *reset));
            }
            _ => {
                // Other operations do not access block register state.
            }
        }
    }

    for register in &f.registers {
        let read_count = reads.get(register.name.as_str()).copied().unwrap_or(0);
        if read_count != 1 {
            return Err(ValidationError::RegisterReadCountMismatch {
                block: f.name.clone(),
                register: register.name.clone(),
                actual: read_count,
            });
        }
        let Some(register_writes) = writes.get(register.name.as_str()) else {
            return Err(ValidationError::BlockInvariantViolation {
                block: f.name.clone(),
                reason: format!("register '{}' has no write", register.name),
            });
        };
        if register_writes.len() > 1 {
            let first_reset = register_writes[0].2;
            for &(node_index, load_enable, reset) in register_writes {
                if load_enable.is_none() {
                    return Err(ValidationError::MultipleRegisterWritesRequireLoadEnable {
                        block: f.name.clone(),
                        node_index,
                        register: register.name.clone(),
                    });
                }
                if reset != first_reset {
                    return Err(ValidationError::MultipleRegisterWritesResetMismatch {
                        block: f.name.clone(),
                        node_index,
                        register: register.name.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Checks the signature before any caller dereferences its parameter nodes.
pub(super) fn validate_fn_signature(f: &Fn) -> Result<(), ValidationError> {
    let invalid_param = |node_index, reason: &str| ValidationError::NodeSemanticViolation {
        func: f.name.clone(),
        node_index,
        reason: reason.to_string(),
    };
    if !f
        .nodes
        .first()
        .is_some_and(|node| matches!(node.payload, NodePayload::Nil))
    {
        return Err(invalid_param(
            0,
            "function graph must start with a Nil sentinel",
        ));
    }
    let mut param_refs = HashSet::new();
    let mut param_names = HashSet::new();
    for &param_ref in &f.params {
        let node = f
            .nodes
            .get(param_ref.index)
            .filter(|node| matches!(node.payload, NodePayload::Param))
            .ok_or_else(|| ValidationError::MissingParamNode {
                func: f.name.clone(),
                node_ref: param_ref,
            })?;
        if !param_refs.insert(param_ref) {
            return Err(invalid_param(
                param_ref.index,
                "parameter reference occurs more than once in the signature",
            ));
        }
        let name = node
            .name
            .as_deref()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid_param(param_ref.index, "parameter node must have a name"))?;
        if !param_names.insert(name) {
            return Err(ValidationError::DuplicateParamName {
                func: f.name.clone(),
                param_name: name.to_string(),
            });
        }
    }
    for (index, node) in f.nodes.iter().enumerate() {
        if matches!(node.payload, NodePayload::Param) && !param_refs.contains(&NodeRef { index }) {
            return Err(ValidationError::ExtraParamNode {
                func: f.name.clone(),
                text_id: node.text_id,
            });
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
    validate_fn_signature(f)?;
    validate_graph_with(
        f,
        parent,
        callee_ret_type_resolver,
        register_type_resolver,
        instantiation_info,
        allow_registers,
    )?;
    let ret_node_ref = f
        .ret_node_ref
        .ok_or_else(|| ValidationError::MissingReturnNode(f.name.clone()))?;
    let ret_node = f
        .nodes
        .get(ret_node_ref.index)
        .filter(|node| !matches!(node.payload, NodePayload::Nil))
        .ok_or_else(|| ValidationError::MissingReturnNode(f.name.clone()))?;
    if ret_node.ty != f.ret_ty {
        return Err(ValidationError::ReturnTypeMismatch {
            func: f.name.clone(),
            expected: f.ret_ty.clone(),
            actual: ret_node.ty.clone(),
        });
    }

    Ok(())
}

fn validate_graph_with<F, R>(
    f: &NodeGraph,
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
    // Track IDs of all non-Nil nodes, including parameters.
    let mut seen_text_ids: HashSet<usize> = HashSet::new();
    // A callee may be referenced by many nodes; check its signature only once.
    let mut checked_callees = HashSet::new();
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
    for (i, node) in f.nodes.iter().enumerate() {
        if !matches!(node.payload, NodePayload::Nil) && !seen_text_ids.insert(node.text_id) {
            return Err(ValidationError::DuplicateTextId {
                func: f.name.clone(),
                text_id: node.text_id,
            });
        }
        // Enforce: if a node has a name that looks like a default textual id
        // pattern '<prefix>.<digits>', then '<prefix>' must match the operator
        // and the numeric suffix must match the node's text id. This aligns
        // with external xlsynth verifier expectations and prevents
        // misleading names.
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
            NodePayload::InputPort { .. } | NodePayload::OutputPort { .. } if !allow_registers => {
                return Err(ValidationError::NodeSemanticViolation {
                    func: f.name.clone(),
                    node_index: i,
                    reason: "port nodes are only permitted in blocks".to_string(),
                });
            }
            _ => {
                // Function signatures and block interfaces are checked by the
                // owner.
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
                if checked_callees.insert(callee.name.as_str()) {
                    validate_fn_signature(callee)?;
                }
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
                for (operand, parameter) in operands.iter().zip(callee.param_nodes()) {
                    if f.get_node(*operand).ty != parameter.ty {
                        return Err(ValidationError::NodeSemanticViolation {
                            func: f.name.clone(),
                            node_index: i,
                            reason: format!(
                                "invoke operand for parameter '{}' must have type {}, got {}",
                                parameter.param_name(),
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
                if checked_callees.insert(body_fn.name.as_str()) {
                    validate_fn_signature(body_fn)?;
                }
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
                match &body_fn.get_param(0).ty {
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
                if body_fn.get_param(1).ty != *init_ty || body_fn.ret_ty != *init_ty {
                    return Err(ValidationError::NodeSemanticViolation {
                        func: f.name.clone(),
                        node_index: i,
                        reason: format!(
                            "counted_for body '{}' carry parameter and result must have type {}",
                            body, init_ty
                        ),
                    });
                }
                for (argument, parameter) in
                    invariant_args.iter().zip(body_fn.param_nodes().skip(2))
                {
                    if f.get_node(*argument).ty != parameter.ty {
                        return Err(ValidationError::NodeSemanticViolation {
                            func: f.name.clone(),
                            node_index: i,
                            reason: format!(
                                "counted_for invariant parameter '{}' must have type {}, got {}",
                                parameter.param_name(),
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
                    // Require bits type and identical types across all
                    // operands.
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
            if !inst_info.require_complete_mapping {
                continue;
            }
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
    Ok(())
}

fn collect_prior_blocks<'a>(
    p: &'a Package,
    member_index: usize,
) -> std::collections::HashMap<String, &'a Block> {
    let mut prior = std::collections::HashMap::new();
    for member in p.members.iter().take(member_index) {
        if let PackageMember::Block(block) = member {
            prior.insert(block.name.clone(), block);
        }
    }
    prior
}

fn build_instantiation_info(
    block: &Block,
    prior_blocks: &std::collections::HashMap<String, &Block>,
    parent: &Package,
) -> Result<std::collections::HashMap<String, InstantiationInfo>, ValidationError> {
    let mut info_map = std::collections::HashMap::new();
    for inst in block.instantiations.iter() {
        if inst.kind == InstantiationKind::Extern {
            let foreign_function = parent
                .get_fn(&inst.block)
                .expect("foreign function missing after target validation");
            validate_fn_signature(foreign_function)?;
            let mut input_types = std::collections::HashMap::new();
            for param in foreign_function.param_nodes() {
                collect_external_port_types(param.param_name(), &param.ty, &mut input_types);
            }
            let mut output_types = std::collections::HashMap::new();
            collect_external_port_types("return", &foreign_function.ret_ty, &mut output_types);
            info_map.insert(
                inst.name.clone(),
                InstantiationInfo {
                    input_types,
                    output_types,
                    require_complete_mapping: false,
                },
            );
            continue;
        }
        let callee = prior_blocks
            .get(&inst.block)
            .expect("prior block missing after check");
        let input_types = callee
            .input_ports()
            .map(|port| {
                (
                    callee.port_name(port).to_string(),
                    callee.port_type(port).clone(),
                )
            })
            .collect();
        let output_types = callee
            .output_ports()
            .map(|port| {
                (
                    callee.port_name(port).to_string(),
                    callee.port_type(port).clone(),
                )
            })
            .collect();
        info_map.insert(
            inst.name.clone(),
            InstantiationInfo {
                input_types,
                output_types,
                require_complete_mapping: true,
            },
        );
    }
    Ok(info_map)
}

/// Records aggregate external ports and each addressable tuple component.
fn collect_external_port_types(
    name: &str,
    ty: &Type,
    port_types: &mut std::collections::HashMap<String, Type>,
) {
    port_types.insert(name.to_string(), ty.clone());
    if let Type::Tuple(elements) = ty {
        for (index, element) in elements.iter().enumerate() {
            collect_external_port_types(&format!("{}.{}", name, index), element, port_types);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;
    use crate::ir_parser::Parser;

    const VALID_REGISTER_BLOCK: &str = r#"package public_register_validation

top block register_block(clk: clock, rst: bits[1], enable: bits[1], data: bits[8], result: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg state(bits[8], reset_value=7)
  rst: bits[1] = input_port(name=rst, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  data: bits[8] = input_port(name=data, id=3)
  current: bits[8] = register_read(register=state, id=4)
  written: () = register_write(data, register=state, load_enable=enable, reset=rst, id=5)
  result: () = output_port(current, name=result, id=6)
}
"#;

    #[test]
    fn valid_resettable_register_passes_structural_validation() {
        let package = Parser::new(VALID_REGISTER_BLOCK).parse_package().unwrap();
        assert_eq!(validate_package(&package), Ok(()));
    }

    /// Mutates an otherwise valid block without letting the parser preempt
    /// checks.
    fn verify_mutated_register_block(
        mutate: impl FnOnce(&mut Block),
    ) -> Result<(), ValidationError> {
        let mut package = Parser::new(VALID_REGISTER_BLOCK).parse_package().unwrap();
        let PackageMember::Block(block) = &mut package.members[0] else {
            unreachable!()
        };
        mutate(block);
        validate_package(&package)
    }

    #[test]
    fn block_declarations_require_unique_resources_clock_writes_and_typed_resets() {
        let check = |result, reason: &str| {
            assert_eq!(
                result,
                Err(ValidationError::BlockInvariantViolation {
                    block: "register_block".to_string(),
                    reason: reason.to_string(),
                })
            )
        };
        check(
            verify_mutated_register_block(|block| {
                block
                    .ports
                    .retain(|port| !matches!(port, BlockPort::Clock(_)));
            }),
            "registers require a clock port",
        );
        check(
            verify_mutated_register_block(|block| {
                block.registers.push(block.registers[0].clone());
            }),
            "duplicate register 'state'",
        );
        check(
            verify_mutated_register_block(|block| {
                block.registers[0].reset_value = Some(crate::IrValue::make_ubits(7, 0).unwrap());
            }),
            "reset value for register 'state' has type bits[7], expected bits[8]",
        );
        check(
            verify_mutated_register_block(|block| {
                let node = block
                    .nodes
                    .iter_mut()
                    .find(|node| matches!(node.payload, NodePayload::RegisterWrite { .. }))
                    .unwrap();
                node.payload = NodePayload::Tuple(Vec::new());
            }),
            "register 'state' has no write",
        );
        check(
            verify_mutated_register_block(|block| {
                let instance = crate::ir::Instantiation {
                    name: "child".to_string(),
                    block: "target".to_string(),
                    kind: InstantiationKind::Block,
                };
                block.instantiations = vec![instance.clone(), instance];
            }),
            "duplicate instantiation 'child'",
        );
    }

    #[test]
    fn invalid_port_lists_and_duplicate_output_ids_are_rejected() {
        let check = |result: Result<(), ValidationError>, expected: &str| {
            let Err(ValidationError::NodeSemanticViolation { reason, .. }) = result else {
                panic!("expected a port invariant failure: {result:?}")
            };
            assert_eq!(reason, expected);
        };
        check(
            verify_mutated_register_block(|block| {
                let port = block.input_ports().next().unwrap();
                block.ports.push(BlockPort::Input(port));
            }),
            "block port is listed more than once",
        );
        check(
            verify_mutated_register_block(|block| {
                block
                    .ports
                    .retain(|port| !matches!(port, BlockPort::Output(_)));
            }),
            "port node is missing from the block interface",
        );
        check(
            verify_mutated_register_block(|block| {
                block
                    .ports
                    .push(BlockPort::Output(NodeRef { index: usize::MAX }));
            }),
            "block port references a missing node",
        );
        check(
            verify_mutated_register_block(|block| {
                let output = block.output_ports().next().unwrap();
                for port in &mut block.ports {
                    if *port == BlockPort::Output(output) {
                        *port = BlockPort::Input(output);
                    }
                }
            }),
            "block port direction does not match its node",
        );
        assert_eq!(
            verify_mutated_register_block(|block| {
                let output = block.output_ports().next().unwrap();
                let input = block.input_ports().next().unwrap();
                block.get_node_mut(output).text_id = block.get_node(input).text_id;
            }),
            Err(ValidationError::DuplicateTextId {
                func: "register_block".to_string(),
                text_id: 1
            })
        );
    }

    #[test]
    fn duplicate_register_reads_are_rejected() {
        let duplicated = VALID_REGISTER_BLOCK.replace(
            "  result: () = output_port",
            "  duplicate: bits[8] = register_read(register=state, id=7)\n  result: () = output_port",
        );
        let package = Parser::new(&duplicated).parse_package().unwrap();
        assert_eq!(
            validate_package(&package),
            Err(ValidationError::RegisterReadCountMismatch {
                block: "register_block".to_owned(),
                register: "state".to_owned(),
                actual: 2,
            })
        );
    }

    #[test]
    fn missing_register_reads_are_rejected() {
        let missing = VALID_REGISTER_BLOCK
            .replace(
                r#"  current: bits[8] = register_read(register=state, id=4)
"#,
                "",
            )
            .replace("output_port(current", "output_port(data");
        let package = Parser::new(&missing).parse_package().unwrap();
        assert_eq!(
            validate_package(&package),
            Err(ValidationError::RegisterReadCountMismatch {
                block: "register_block".to_owned(),
                register: "state".to_owned(),
                actual: 0,
            })
        );
    }

    #[test]
    fn resettable_register_write_requires_a_reset_operand() {
        let ir = VALID_REGISTER_BLOCK.replace(", reset=rst, id=5", ", id=5");
        let package = Parser::new(&ir).parse_package().unwrap();
        assert!(matches!(
            validate_package(&package),
            Err(ValidationError::RegisterWriteResetPresenceMismatch {
                register_has_reset_value: true,
                write_has_reset: false,
                ..
            })
        ));
    }

    #[test]
    fn register_write_reset_requires_a_declared_reset_value() {
        let ir = VALID_REGISTER_BLOCK.replace("state(bits[8], reset_value=7)", "state(bits[8])");
        let package = Parser::new(&ir).parse_package().unwrap();
        assert!(matches!(
            validate_package(&package),
            Err(ValidationError::RegisterWriteResetPresenceMismatch {
                register_has_reset_value: false,
                write_has_reset: true,
                ..
            })
        ));
    }

    #[test]
    fn resettable_register_requires_a_declared_block_reset_port() {
        let ir = VALID_REGISTER_BLOCK.replace(
            "  #![reset(port=\"rst\", asynchronous=false, active_low=false)]\n",
            "",
        );
        let package = Parser::new(&ir).parse_package().unwrap();
        assert_eq!(
            validate_package(&package),
            Err(ValidationError::BlockInvariantViolation {
                block: "register_block".to_string(),
                reason: "register 'state' has a reset value but the block has no reset port"
                    .to_string(),
            })
        );
    }

    #[test]
    fn block_reset_metadata_requires_a_single_bit_input() {
        let mut package = Parser::new(VALID_REGISTER_BLOCK).parse_package().unwrap();
        let PackageMember::Block(block) = &mut package.members[0] else {
            unreachable!()
        };
        block.reset.as_mut().unwrap().port = block.get_input_port("data").unwrap();
        assert_eq!(
            validate_package(&package),
            Err(ValidationError::InvalidBlockResetPort {
                block: "register_block".to_owned(),
                port: "data".to_owned(),
            })
        );
    }

    #[test]
    fn multiple_enabled_writes_to_one_register_are_valid() {
        let ir = VALID_REGISTER_BLOCK.replace(
            "  result: () = output_port",
            "  disabled: bits[1] = not(enable, id=7)\n  second_write: () = register_write(data, register=state, load_enable=disabled, reset=rst, id=8)\n  result: () = output_port",
        );
        let package = Parser::new(&ir).parse_package().unwrap();
        assert_eq!(validate_package(&package), Ok(()));
    }

    #[test]
    fn multiple_register_writes_all_require_load_enables() {
        let ir = VALID_REGISTER_BLOCK.replace(
            "  result: () = output_port",
            "  second_write: () = register_write(data, register=state, reset=rst, id=7)\n  result: () = output_port",
        );
        let package = Parser::new(&ir).parse_package().unwrap();
        assert!(matches!(
            validate_package(&package),
            Err(ValidationError::MultipleRegisterWritesRequireLoadEnable { register, .. })
                if register == "state"
        ));
    }

    #[test]
    fn multiple_register_writes_must_share_the_same_reset_operand() {
        let ir = VALID_REGISTER_BLOCK
            .replace("rst: bits[1], enable:", "rst: bits[1], other_reset: bits[1], enable:")
            .replace(
                "  enable: bits[1] = input_port",
                "  other_reset: bits[1] = input_port(name=other_reset, id=7)\n  enable: bits[1] = input_port",
            )
            .replace(
                "  result: () = output_port",
                "  second_write: () = register_write(data, register=state, load_enable=enable, reset=other_reset, id=8)\n  result: () = output_port",
            );
        let package = Parser::new(&ir).parse_package().unwrap();
        assert!(matches!(
            validate_package(&package),
            Err(ValidationError::MultipleRegisterWritesResetMismatch { register, .. })
                if register == "state"
        ));
    }

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
        // with operator literal(id=2). This should fail with
        // NodeNameOpMismatch.
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
            payload: ir::NodePayload::Literal(crate::IrValue::make_ubits(8, 1).unwrap()),
            pos: None,
        };
        let f = ir::Fn {
            graph: ir::NodeGraph {
                name: "f".to_string(),
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
                outer_attrs: Vec::new(),
                inner_attrs: Vec::new(),
            },
            params: Vec::new(),
            ret_ty: ir::Type::Bits(8),
            ret_node_ref: Some(ir::NodeRef { index: 1 }),
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
    fn duplicate_param_node_id_fails() {
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
            // Add another signature parameter sharing x's textual ID. Its
            // name and reference are distinct so the ID check is exercised.
            let text_id = f.get_param(0).text_id;
            let dup = ir::Node {
                text_id,
                name: Some("y".to_string()),
                ty: f.get_param(0).ty.clone(),
                payload: ir::NodePayload::Param,
                pos: None,
            };
            f.params.push(NodeRef {
                index: f.nodes.len(),
            });
            f.nodes.push(dup);
        }
        let f_ro = pkg.get_top_fn().unwrap();
        assert!(matches!(
            validate_fn(f_ro, &pkg),
            Err(ValidationError::DuplicateTextId { .. })
        ));
    }

    #[test]
    fn missing_param_node_fails() {
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
            // Remove the Param node for 'x'. It should be at index 1.
            let idx = f
                .nodes
                .iter()
                .position(|n| matches!(n.payload, NodePayload::Param))
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
    fn external_instantiation_requires_existing_foreign_function() {
        let ir = r#"package external_test

top block wrapper(value: bits[8], result: bits[8]) {
  instantiation external_instance(foreign_function=missing, kind=extern)
  value: bits[8] = input_port(name=value, id=1)
  instantiation_input.2: () = instantiation_input(value, instantiation=external_instance, port_name=value, id=2)
  instantiation_output.3: bits[8] = instantiation_output(instantiation=external_instance, port_name=return, id=3)
  result: () = output_port(instantiation_output.3, name=result, id=4)
}
"#;
        let mut parser = Parser::new(ir);
        let package = parser.parse_package().unwrap();
        assert_eq!(
            validate_package(&package),
            Err(ValidationError::InstantiationForeignFunctionNotFound {
                func: "wrapper".to_string(),
                instantiation: "external_instance".to_string(),
                foreign_function: "missing".to_string(),
            })
        );
    }

    #[test]
    fn external_instantiation_rejects_unknown_tuple_component() {
        let ir = r#"package external_test

fn external_pair(value: (bits[8], bits[16]) id=1) -> (bits[8], bits[16]) {
  ret value: (bits[8], bits[16]) = param(name=value, id=1)
}

top block wrapper(input: bits[8], result: bits[8]) {
  instantiation external_instance(foreign_function=external_pair, kind=extern)
  input: bits[8] = input_port(name=input, id=2)
  instantiation_input.3: () = instantiation_input(input, instantiation=external_instance, port_name=value.2, id=3)
  instantiation_output.4: bits[8] = instantiation_output(instantiation=external_instance, port_name=return.0, id=4)
  result: () = output_port(instantiation_output.4, name=result, id=5)
}
"#;
        let mut parser = Parser::new(ir);
        let package = parser.parse_package().unwrap();
        assert!(matches!(
            validate_package(&package),
            Err(ValidationError::UnknownInstantiationPort {
                direction: InstantiationPortDirection::Input,
                port_name,
                ..
            }) if port_name == "value.2"
        ));
    }

    #[test]
    fn external_instantiation_rejects_tuple_component_type_mismatch() {
        let ir = r#"package external_test

fn external_pair(value: (bits[8], bits[16]) id=1) -> (bits[8], bits[16]) {
  ret value: (bits[8], bits[16]) = param(name=value, id=1)
}

top block wrapper(input: bits[8], result: bits[8]) {
  instantiation external_instance(foreign_function=external_pair, kind=extern)
  input: bits[8] = input_port(name=input, id=2)
  instantiation_input.3: () = instantiation_input(input, instantiation=external_instance, port_name=value.1, id=3)
  instantiation_output.4: bits[8] = instantiation_output(instantiation=external_instance, port_name=return.0, id=4)
  result: () = output_port(instantiation_output.4, name=result, id=5)
}
"#;
        let mut parser = Parser::new(ir);
        let package = parser.parse_package().unwrap();
        assert!(matches!(
            validate_package(&package),
            Err(ValidationError::InstantiationPortTypeMismatch {
                direction: InstantiationPortDirection::Input,
                expected: Type::Bits(16),
                actual: Type::Bits(8),
                ..
            })
        ));
    }

    #[test]
    fn external_instantiation_accepts_aggregate_and_component_ports() {
        let ir = r#"package external_test

fn external_pair(value: (bits[8], bits[16]) id=1) -> (bits[8], bits[16]) {
  ret value: (bits[8], bits[16]) = param(name=value, id=1)
}

top block wrapper(value: (bits[8], bits[16]), result: bits[8]) {
  instantiation external_instance(foreign_function=external_pair, kind=extern)
  value: (bits[8], bits[16]) = input_port(name=value, id=2)
  instantiation_input.3: () = instantiation_input(value, instantiation=external_instance, port_name=value, id=3)
  instantiation_output.4: bits[8] = instantiation_output(instantiation=external_instance, port_name=return.0, id=4)
  result: () = output_port(instantiation_output.4, name=result, id=5)
}
"#;
        let mut parser = Parser::new(ir);
        let package = parser.parse_package().unwrap();
        assert_eq!(validate_package(&package), Ok(()));
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
