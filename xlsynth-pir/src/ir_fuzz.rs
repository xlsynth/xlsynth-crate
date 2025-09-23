// SPDX-License-Identifier: Apache-2.0

use crate::ir::Type as InternalType;
use crate::ir_parser;
use crate::{fuzz_utils::arbitrary_irbits, ir};
use arbitrary::Arbitrary;
use rand::Rng;
use std::collections::BTreeMap;
use xlsynth::{BValue, FnBuilder, IrFunction, IrType, XlsynthError};

const MAX_OPS_PER_SAMPLE: u64 = 20;
const MAX_ELEMENTS_PER_TUPLE: u64 = 8;
const MAX_ELEMENTS_PER_ARRAY: u64 = 8;
const MAX_PARAMS_PER_SAMPLE: u64 = 4;
// Maximum number of recursive elements in a type.
const MAX_TOTAL_ELEMENTS_PER_TYPE: u64 = 64;

// Return the number of leaf elements in the type.
pub fn count_type_leaves(ty: &InternalType) -> usize {
    match ty {
        InternalType::Bits(_) => 1,
        InternalType::Tuple(ts) => ts.iter().map(|t| count_type_leaves(t)).sum(),
        InternalType::Array(a) => a
            .element_count
            .saturating_mul(count_type_leaves(&a.element_type)),
        InternalType::Token => 0,
    }
}

#[derive(Debug, Arbitrary, Clone)]
pub enum FuzzUnop {
    // bitwise not (one's complement negation)
    Not,
    // two's complement negation
    Neg,

    OrReduce,
    AndReduce,
    XorReduce,

    Identity,
    Reverse,

    Encode,
}

#[derive(Debug, Arbitrary, Clone)]
pub enum FuzzBinop {
    // bitwise
    And,
    Nand,
    Nor,
    Or,
    Xor,

    // comparisons
    Eq,
    Ne,
    Ugt,
    Ule,
    Ult,
    Uge,
    Sge,
    Sgt,
    Slt,
    Sle,

    // shifts
    Shrl,
    Shra,
    Shll,

    // arithmetic
    Add,
    Sub,

    // division / modulus
    Udiv,
    Sdiv,
    Umod,
    Smod,

    Concat,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzOperand {
    index: usize,
}

#[derive(Debug, Arbitrary, Clone)]
pub enum FuzzOp {
    Param {
        bits: u8,
    },
    Literal {
        bits: u8,
        value: u64,
    },
    Unop(FuzzUnop, usize),
    Binop(FuzzBinop, usize, usize),
    ZeroExt {
        operand: FuzzOperand,
        new_bit_count: u64,
    },
    SignExt {
        operand: FuzzOperand,
        new_bit_count: u64,
    },
    BitSlice {
        operand: FuzzOperand,
        start: u64,
        width: u64,
    },
    OneHot {
        arg: FuzzOperand,
        lsb_prio: bool,
    },
    Sel {
        selector: FuzzOperand,
        cases: Vec<FuzzOperand>,
        default: FuzzOperand,
    },
    OneHotSel {
        selector: FuzzOperand,
        cases: Vec<FuzzOperand>,
    },
    PrioritySel {
        selector: FuzzOperand,
        cases: Vec<FuzzOperand>,
        default: FuzzOperand,
    },
    Array {
        elements: Vec<FuzzOperand>,
    },
    ArrayIndex {
        array: FuzzOperand,
        index: FuzzOperand,
    },
    Decode {
        arg: FuzzOperand,
        width: u64,
    },
    UMul {
        lhs: FuzzOperand,
        rhs: FuzzOperand,
    },
    SMul {
        lhs: FuzzOperand,
        rhs: FuzzOperand,
    },
    Tuple {
        elements: Vec<FuzzOperand>,
    },
    TupleIndex {
        tuple: FuzzOperand,
        index: usize,
    },
    DynamicBitSlice {
        arg: FuzzOperand,
        start: FuzzOperand,
        width: u64,
    },
    BitSliceUpdate {
        value: FuzzOperand,
        start: FuzzOperand,
        update: FuzzOperand,
    },
}

#[derive(Debug, Clone, Arbitrary, PartialEq, Eq, Hash)]
pub enum FuzzOpFlat {
    Param,
    Literal,
    Unop,
    Binop,
    ZeroExt,
    SignExt,
    BitSlice,
    OneHot,
    Sel,
    PrioritySel,
    ArrayIndex,
    Array,
    Decode,
    OneHotSel,
    UMul,
    SMul,
    Tuple,
    TupleIndex,
    DynamicBitSlice,
    BitSliceUpdate,
}

/// Flattened opcode-only version of FuzzOp so we can ensure we select among all
/// available ops when making an arbitrary op.
#[allow(dead_code)]
fn to_flat(op: &FuzzOp) -> FuzzOpFlat {
    match op {
        FuzzOp::Param { .. } => FuzzOpFlat::Param,
        FuzzOp::Literal { .. } => FuzzOpFlat::Literal,
        FuzzOp::Unop { .. } => FuzzOpFlat::Unop,
        FuzzOp::Binop { .. } => FuzzOpFlat::Binop,
        FuzzOp::ZeroExt { .. } => FuzzOpFlat::ZeroExt,
        FuzzOp::SignExt { .. } => FuzzOpFlat::SignExt,
        FuzzOp::BitSlice { .. } => FuzzOpFlat::BitSlice,
        FuzzOp::OneHot { .. } => FuzzOpFlat::OneHot,
        FuzzOp::Sel { .. } => FuzzOpFlat::Sel,
        FuzzOp::PrioritySel { .. } => FuzzOpFlat::PrioritySel,
        FuzzOp::ArrayIndex { .. } => FuzzOpFlat::ArrayIndex,
        FuzzOp::Array { .. } => FuzzOpFlat::Array,
        FuzzOp::Decode { .. } => FuzzOpFlat::Decode,
        FuzzOp::OneHotSel { .. } => FuzzOpFlat::OneHotSel,
        FuzzOp::UMul { .. } => FuzzOpFlat::UMul,
        FuzzOp::SMul { .. } => FuzzOpFlat::SMul,
        FuzzOp::Tuple { .. } => FuzzOpFlat::Tuple,
        FuzzOp::TupleIndex { .. } => FuzzOpFlat::TupleIndex,
        FuzzOp::DynamicBitSlice { .. } => FuzzOpFlat::DynamicBitSlice,
        FuzzOp::BitSliceUpdate { .. } => FuzzOpFlat::BitSliceUpdate,
    }
}

/// Converts our internal Type (from ir.rs) to xlsynth::IrType using the package
/// as context.
pub fn internal_type_to_xlsynth(pkg: &mut xlsynth::IrPackage, ty: &InternalType) -> IrType {
    match ty {
        InternalType::Bits(width) => pkg.get_bits_type(*width as u64),
        InternalType::Tuple(types) => {
            let elem_types: Vec<IrType> = types
                .iter()
                .map(|t| internal_type_to_xlsynth(pkg, t))
                .collect();
            pkg.get_tuple_type(&elem_types)
        }
        InternalType::Array(arr) => {
            let elem_type = internal_type_to_xlsynth(pkg, &arr.element_type);
            // arr.element_count is u64, but get_array_type expects i64. Convert safely.
            let count_i64: i64 = arr
                .element_count
                .try_into()
                .expect("element_count must fit in i64");
            pkg.get_array_type(&elem_type, count_i64)
        }
        InternalType::Token => pkg.get_token_type(),
    }
}

/// Produces an arbitrary `IrValue` for the given internal IR type using the
/// provided RNG.
pub fn arbitrary_ir_value_for_internal_type<R: Rng>(
    rng: &mut R,
    ty: &InternalType,
) -> xlsynth::IrValue {
    match ty {
        InternalType::Bits(width) => {
            let bits = arbitrary_irbits(rng, *width);
            xlsynth::IrValue::from_bits(&bits)
        }
        InternalType::Tuple(elem_types) => {
            let elems: Vec<xlsynth::IrValue> = elem_types
                .iter()
                .map(|t| arbitrary_ir_value_for_internal_type(rng, t))
                .collect();
            xlsynth::IrValue::make_tuple(&elems)
        }
        InternalType::Array(arr) => {
            let mut elems: Vec<xlsynth::IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(arbitrary_ir_value_for_internal_type(rng, &arr.element_type));
            }
            xlsynth::IrValue::make_array(&elems).expect("array elements should be same-typed")
        }
        InternalType::Token => xlsynth::IrValue::make_token(),
    }
}

/// Given a map from types to available nodes, recursively constructs a value of
/// the requested type. If a node of the exact type is available, returns it.
/// Otherwise, attempts to build the type from available components.
pub fn build_return_value_with_type(
    fn_builder: &mut FnBuilder,
    pkg: &mut xlsynth::IrPackage,
    type_map: &BTreeMap<InternalType, Vec<BValue>>,
    target_type: &InternalType,
) -> Option<BValue> {
    // If we have a node of the exact type, use it.
    if let Some(nodes) = type_map.get(target_type) {
        if let Some(node) = nodes.first() {
            return Some(node.clone());
        }
    }
    match target_type {
        InternalType::Tuple(types) => {
            // Recursively build each element of the tuple.
            let mut elems = Vec::with_capacity(types.len());
            for elem_ty in types {
                let elem = build_return_value_with_type(fn_builder, pkg, type_map, elem_ty)?;
                elems.push(elem);
            }
            let refs: Vec<&BValue> = elems.iter().collect();
            Some(fn_builder.tuple(&refs, None))
        }
        InternalType::Array(array_data) => {
            // Recursively build each element of the array.
            let mut elems = Vec::with_capacity(array_data.element_count);
            for _ in 0..array_data.element_count {
                let elem = build_return_value_with_type(
                    fn_builder,
                    pkg,
                    type_map,
                    &array_data.element_type,
                )?;
                elems.push(elem);
            }
            let refs: Vec<&BValue> = elems.iter().collect();
            // Use the correct IrType for the array
            let ir_elem_type = internal_type_to_xlsynth(pkg, &array_data.element_type);
            let count_i64: i64 = array_data
                .element_count
                .try_into()
                .expect("element_count must fit in i64");
            let ir_array_type = pkg.get_array_type(&ir_elem_type, count_i64);
            Some(fn_builder.array(&ir_array_type, &refs, None))
        }
        InternalType::Bits(width) => {
            // Try to find a bits node of the right width.
            if let Some(nodes) = type_map.get(target_type) {
                if let Some(node) = nodes.first() {
                    return Some(node.clone());
                }
            }
            // Otherwise, try to build from other bits nodes (slice, extend, concat).
            // Find all available bits nodes.
            let mut bits_nodes: Vec<(usize, &BValue, &InternalType)> = Vec::new();
            for (ty, nodes) in type_map.iter() {
                if let InternalType::Bits(w) = ty {
                    for node in nodes {
                        bits_nodes.push((*w as usize, node, ty));
                    }
                }
            }
            // Try to find a node to slice or extend.
            for (node_width, node, _node_ty) in &bits_nodes {
                if *node_width > *width {
                    // Slice down
                    return Some(fn_builder.bit_slice(node, 0, *width as u64, None));
                } else if *node_width < *width {
                    // Zero-extend up
                    return Some(fn_builder.zero_extend(node, *width as u64, None));
                } else if *node_width == *width {
                    return Some((*node).clone());
                }
            }
            // Try to concat smaller nodes to reach the width
            bits_nodes.sort_by_key(|(w, _, _)| *w);
            let mut acc = Vec::new();
            let mut acc_width = 0;
            for (node_width, node, _) in &bits_nodes {
                if acc_width + *node_width <= *width {
                    acc.push(*node);
                    acc_width += *node_width;
                    if acc_width == *width {
                        break;
                    }
                }
            }
            if acc_width == *width && !acc.is_empty() {
                let refs: Vec<&BValue> = acc.iter().map(|n| *n).collect();
                return Some(fn_builder.concat(&refs, None));
            }
            // As a last resort, synthesize a literal of the requested width.
            let zeros: xlsynth::IrValue =
                xlsynth::IrValue::make_ubits(*width as usize, 0u64).unwrap();
            Some(fn_builder.literal(&zeros, None))
        }
        _ => None,
    }
}

/// Builds an xlsynth function by enqueueing the operations given by `ops` into
/// a `FnBuilder` and returning the built result.
pub fn generate_ir_fn(
    ops: Vec<FuzzOp>,
    package: &mut xlsynth::IrPackage,
    target_return_type: Option<&InternalType>,
) -> Result<IrFunction, XlsynthError> {
    let mut fn_builder = FnBuilder::new(package, "fuzz_test", true);

    // Track all available nodes that can be used as operands
    let mut param_count: usize = 0;
    let mut available_nodes: Vec<BValue> = Vec::new();
    let mut type_map: BTreeMap<InternalType, Vec<BValue>> = BTreeMap::new();

    // Process each operation
    for op in ops {
        let node = match op {
            FuzzOp::Param { bits } => {
                assert!(bits > 0, "param bits must be > 0");
                let ty: IrType = package.get_bits_type(bits as u64);
                let name = format!("p{}", param_count);
                param_count += 1;
                fn_builder.param(&name, &ty)
            }
            FuzzOp::Literal { bits, value } => {
                assert!(bits > 0, "literal op has no bits");
                let ir_value = xlsynth::IrValue::make_ubits(bits as usize, value)?;
                fn_builder.literal(&ir_value, None)
            }
            FuzzOp::Unop(unop, idx) => {
                let idx = idx % available_nodes.len();
                let operand = &available_nodes[idx];
                match unop {
                    FuzzUnop::Not => fn_builder.not(operand, None),
                    FuzzUnop::Neg => fn_builder.neg(operand, None),
                    FuzzUnop::OrReduce => fn_builder.or_reduce(operand, None),
                    FuzzUnop::AndReduce => fn_builder.and_reduce(operand, None),
                    FuzzUnop::XorReduce => fn_builder.xor_reduce(operand, None),
                    FuzzUnop::Reverse => fn_builder.rev(operand, None),
                    FuzzUnop::Identity => fn_builder.identity(operand, None),
                    FuzzUnop::Encode => {
                        let bits_type: IrType = fn_builder.get_type(operand).unwrap();
                        let bit_count: u64 = bits_type.get_flat_bit_count();
                        if bit_count <= 1 {
                            return Err(XlsynthError(
                                "encode needs more than 1 bit input operand".to_string(),
                            ));
                        }
                        fn_builder.encode(operand, None)
                    }
                }
            }
            FuzzOp::Binop(binop, idx1, idx2) => {
                let idx1 = idx1 % available_nodes.len();
                let idx2 = idx2 % available_nodes.len();
                let operand1: &BValue = &available_nodes[idx1];
                let operand2: &BValue = &available_nodes[idx2];
                match binop {
                    FuzzBinop::Add => fn_builder.add(operand1, operand2, None),
                    FuzzBinop::Sub => fn_builder.sub(operand1, operand2, None),
                    FuzzBinop::And => fn_builder.and(operand1, operand2, None),
                    FuzzBinop::Nand => fn_builder.nand(operand1, operand2, None),
                    FuzzBinop::Nor => fn_builder.nor(operand1, operand2, None),
                    FuzzBinop::Or => fn_builder.or(operand1, operand2, None),
                    FuzzBinop::Xor => fn_builder.xor(operand1, operand2, None),
                    FuzzBinop::Eq => fn_builder.eq(operand1, operand2, None),
                    FuzzBinop::Ne => fn_builder.ne(operand1, operand2, None),
                    FuzzBinop::Ugt => fn_builder.ugt(operand1, operand2, None),
                    FuzzBinop::Ult => fn_builder.ult(operand1, operand2, None),
                    FuzzBinop::Uge => fn_builder.uge(operand1, operand2, None),
                    FuzzBinop::Ule => fn_builder.ule(operand1, operand2, None),
                    FuzzBinop::Sgt => fn_builder.sgt(operand1, operand2, None),
                    FuzzBinop::Sge => fn_builder.sge(operand1, operand2, None),
                    FuzzBinop::Slt => fn_builder.slt(operand1, operand2, None),
                    FuzzBinop::Sle => fn_builder.sle(operand1, operand2, None),
                    FuzzBinop::Shrl => fn_builder.shrl(operand1, operand2, None),
                    FuzzBinop::Shra => fn_builder.shra(operand1, operand2, None),
                    FuzzBinop::Shll => fn_builder.shll(operand1, operand2, None),
                    FuzzBinop::Udiv => fn_builder.udiv(operand1, operand2, None),
                    FuzzBinop::Sdiv => fn_builder.sdiv(operand1, operand2, None),
                    FuzzBinop::Umod => fn_builder.umod(operand1, operand2, None),
                    FuzzBinop::Smod => fn_builder.smod(operand1, operand2, None),
                    FuzzBinop::Concat => fn_builder.concat(&[operand1, operand2], None),
                }
            }
            FuzzOp::ZeroExt {
                operand,
                new_bit_count,
            } => {
                assert!(new_bit_count > 0, "zero extend has new bit count of 0");
                let operand = &available_nodes[operand.index % available_nodes.len()];
                let operand_width =
                    fn_builder.get_type(operand).unwrap().get_flat_bit_count() as u64;
                let clamped_new_bit_count = std::cmp::max(new_bit_count, operand_width);
                fn_builder.zero_extend(operand, clamped_new_bit_count as u64, None)
            }
            FuzzOp::SignExt {
                operand,
                new_bit_count,
            } => {
                assert!(new_bit_count > 0, "sign extend has new bit count of 0");
                let operand = &available_nodes[operand.index % available_nodes.len()];
                let operand_width =
                    fn_builder.get_type(operand).unwrap().get_flat_bit_count() as u64;
                let clamped_new_bit_count = std::cmp::max(new_bit_count, operand_width);
                fn_builder.sign_extend(operand, clamped_new_bit_count as u64, None)
            }
            FuzzOp::BitSlice {
                operand,
                start,
                width,
            } => {
                assert!(width > 0, "bit slice has no width");
                let operand = &available_nodes[operand.index % available_nodes.len()];
                fn_builder.bit_slice(operand, start as u64, width as u64, None)
            }
            FuzzOp::OneHot { arg, lsb_prio } => {
                let arg = &available_nodes[arg.index % available_nodes.len()];
                fn_builder.one_hot(arg, lsb_prio, None)
            }
            FuzzOp::Sel {
                selector,
                cases,
                default,
            } => {
                let selector = &available_nodes[selector.index % available_nodes.len()];
                let cases = cases
                    .iter()
                    .map(|idx| &available_nodes[idx.index % available_nodes.len()])
                    .collect::<Vec<_>>();
                let default = &available_nodes[default.index % available_nodes.len()];
                fn_builder.select(selector, cases.as_slice(), default, None)
            }
            FuzzOp::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let selector: &BValue = &available_nodes[selector.index % available_nodes.len()];
                let cases: Vec<BValue> = cases
                    .iter()
                    .map(|idx| available_nodes[idx.index % available_nodes.len()].clone())
                    .collect::<Vec<_>>();
                let default = &available_nodes[default.index % available_nodes.len()];
                fn_builder.priority_select(selector, cases.as_slice(), default, None)
            }
            FuzzOp::ArrayIndex { array, index } => {
                let array = &available_nodes[array.index % available_nodes.len()];
                let index = &available_nodes[index.index % available_nodes.len()];
                fn_builder.array_index(array, index, None)
            }
            FuzzOp::Array { elements } => {
                let elements: Vec<BValue> = elements
                    .iter()
                    .map(|idx| available_nodes[idx.index % available_nodes.len()].clone())
                    .collect::<Vec<_>>();
                let element_type: Option<IrType> = fn_builder.get_type(&elements[0]);
                if element_type.is_none() {
                    return Err(XlsynthError(
                        "no type available for array element BValue".to_string(),
                    ));
                }
                let element_type_ref: &IrType = element_type.as_ref().unwrap();
                let elements: &[BValue] = elements.as_slice();
                let elements_refs = elements.iter().map(|e: &BValue| e).collect::<Vec<_>>();
                fn_builder.array(element_type_ref, &elements_refs, None)
            }
            FuzzOp::Tuple { elements } => {
                let tuple_elems: Vec<BValue> = elements
                    .iter()
                    .map(|idx| available_nodes[idx.index % available_nodes.len()].clone())
                    .collect::<Vec<_>>();
                let refs: Vec<&BValue> = tuple_elems.iter().collect();
                fn_builder.tuple(&refs, None)
            }
            FuzzOp::TupleIndex { tuple, index } => {
                let tuple_bv = &available_nodes[tuple.index % available_nodes.len()];
                fn_builder.tuple_index(tuple_bv, index as u64, None)
            }
            FuzzOp::Decode { arg, width } => {
                assert!(width > 0, "decode has width of 0");
                let arg = &available_nodes[arg.index % available_nodes.len()];
                fn_builder.decode(arg, Some(width as u64), None)
            }
            FuzzOp::OneHotSel { selector, cases } => {
                let selector = &available_nodes[selector.index % available_nodes.len()];
                let cases: Vec<BValue> = cases
                    .iter()
                    .map(|idx| available_nodes[idx.index % available_nodes.len()].clone())
                    .collect::<Vec<_>>();
                fn_builder.one_hot_select(selector, cases.as_slice(), None)
            }
            FuzzOp::UMul { lhs, rhs } => {
                let lhs = &available_nodes[lhs.index % available_nodes.len()];
                let rhs = &available_nodes[rhs.index % available_nodes.len()];
                fn_builder.umul(lhs, rhs, None)
            }
            FuzzOp::SMul { lhs, rhs } => {
                let lhs = &available_nodes[lhs.index % available_nodes.len()];
                let rhs = &available_nodes[rhs.index % available_nodes.len()];
                fn_builder.smul(lhs, rhs, None)
            }
            FuzzOp::DynamicBitSlice { arg, start, width } => {
                assert!(width > 0, "dynamic bit slice has no width");
                let arg = &available_nodes[arg.index % available_nodes.len()];
                let start = &available_nodes[start.index % available_nodes.len()];
                fn_builder.dynamic_bit_slice(arg, start, width as u64, None)
            }
            FuzzOp::BitSliceUpdate {
                value,
                start,
                update,
            } => {
                let value_bv = &available_nodes[value.index % available_nodes.len()];
                let start_bv = &available_nodes[start.index % available_nodes.len()];
                let update_bv = &available_nodes[update.index % available_nodes.len()];
                fn_builder.bit_slice_update(value_bv, start_bv, update_bv, None)
            }
        };
        // If the builder entered an error state when adding this node, bail out
        // immediately before attempting to query its type or record it.
        if let Err(e) = fn_builder.last_value() {
            return Err(e);
        }
        // Track the node and its type
        if let Some(ty) = fn_builder.get_type(&node) {
            let ty_str = ty.to_string();
            let mut parser = ir_parser::Parser::new(&ty_str);
            if let Ok(internal_ty) = parser.parse_type() {
                type_map.entry(internal_ty).or_default().push(node.clone());
            }
        }
        available_nodes.push(node);
    }
    // Determine the target return type: prefer caller-provided, otherwise use the
    // type of the last node
    let internal_ret_type: InternalType = match target_return_type {
        Some(t) => t.clone(),
        None => {
            let ret_type = fn_builder
                .get_type(available_nodes.last().unwrap())
                .unwrap();
            let ret_type_str = ret_type.to_string();
            let mut parser = ir_parser::Parser::new(&ret_type_str);
            parser.parse_type().expect("Failed to parse return type")
        }
    };
    let ret_node =
        build_return_value_with_type(&mut fn_builder, package, &type_map, &internal_ret_type)
            .unwrap_or_else(|| available_nodes.last().unwrap().clone());
    fn_builder.build_with_return_value(&ret_node)
}

#[derive(Debug, Clone)]
pub struct FuzzSample {
    pub ops: Vec<FuzzOp>,
    /// Parallel to nodes: index 0 is the primary input, then one entry per op.
    pub op_types: Vec<InternalType>,
}

/// Picks and returns a Bits type from the given list.  Returns the index of the
/// chosen type and the type itself.
fn pick_bits_type(
    u: &mut arbitrary::Unstructured,
    types: &[InternalType],
) -> arbitrary::Result<(usize, InternalType)> {
    let candidates: Vec<usize> = types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match t {
            InternalType::Bits(_) => Some(i),
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    let which = u.int_in_range(0..=((candidates.len() as u64) - 1))? as usize;
    let idx = candidates[which];
    Ok((idx, types[idx].clone()))
}

/// Picks and returns an Array type from the given list.  Returns the index of
/// the chosen type and the type itself.
fn pick_array_type(
    u: &mut arbitrary::Unstructured,
    types: &[InternalType],
) -> arbitrary::Result<(usize, InternalType)> {
    let candidates: Vec<usize> = types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match t {
            InternalType::Array(_) => Some(i),
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    let which = u.int_in_range(0..=((candidates.len() as u64) - 1))? as usize;
    let idx = candidates[which];
    Ok((idx, types[idx].clone()))
}

/// Picks and returns a Tuple type from the given list.  Returns the index of
/// the chosen type and the type itself.
fn pick_tuple_type(
    u: &mut arbitrary::Unstructured,
    types: &[InternalType],
) -> arbitrary::Result<(usize, InternalType)> {
    let candidates: Vec<usize> = types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match t {
            InternalType::Tuple(_) => Some(i),
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    let which = u.int_in_range(0..=((candidates.len() as u64) - 1))? as usize;
    let idx = candidates[which];
    Ok((idx, types[idx].clone()))
}

/// Returns indices of `n` nodes that all have identical type.
fn pick_same_types(
    u: &mut arbitrary::Unstructured,
    types: &[InternalType],
    n: usize,
) -> arbitrary::Result<(Vec<usize>, InternalType)> {
    if types.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    // Pick a random pivot node; use its type as the target type.
    let pivot = u.int_in_range(0..=((types.len() as u64) - 1))? as usize;
    let target_ty = &types[pivot];
    // Gather all indices that share the same type as the pivot.
    let candidates: Vec<usize> = types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| if t == target_ty { Some(i) } else { None })
        .collect();
    if candidates.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    // Pick n indices with replacement from candidates.
    let mut out: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..n {
        let which = u.int_in_range(0..=((candidates.len() as u64) - 1))? as usize;
        out.push(candidates[which]);
    }
    Ok((out, target_ty.clone()))
}

/// Returns indices of `n` nodes that all have identical Bits type.
fn pick_same_bits_types(
    u: &mut arbitrary::Unstructured,
    types: &[InternalType],
    n: usize,
) -> arbitrary::Result<(Vec<usize>, InternalType)> {
    // Restrict pivot selection to Bits-typed nodes only.
    let bit_indices: Vec<usize> = types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match t {
            InternalType::Bits(_) => Some(i),
            _ => None,
        })
        .collect();
    if bit_indices.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    let pivot_in_bits = u.int_in_range(0..=((bit_indices.len() as u64) - 1))? as usize;
    let pivot = bit_indices[pivot_in_bits];
    let target_ty = &types[pivot];
    // Gather all indices that share the same bits type as the pivot.
    let candidates: Vec<usize> = types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| if t == target_ty { Some(i) } else { None })
        .collect();
    if candidates.is_empty() {
        return Err(arbitrary::Error::IncorrectFormat);
    }
    // Pick n indices with replacement from candidates.
    let mut out: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..n {
        let which = u.int_in_range(0..=((candidates.len() as u64) - 1))? as usize;
        out.push(candidates[which]);
    }
    Ok((out, target_ty.clone()))
}

fn generate_fuzz_op(
    u: &mut arbitrary::Unstructured,
    node_types: &[InternalType],
) -> arbitrary::Result<(FuzzOp, InternalType)> {
    log::trace!("Generating fuzz op");
    // Randomly decide which kind of operation to generate (excluding Param). Params
    // are generated explicitly at FuzzSample creation time.
    let op_type = {
        let mut attempts: u8 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            if attempts > 10 {
                return Err(arbitrary::Error::IncorrectFormat);
            }
            let t = u.arbitrary::<FuzzOpFlat>()?;
            if let FuzzOpFlat::Param = t {
                continue;
            }
            break t;
        }
    };
    let (op, out_ty) = match op_type {
        FuzzOpFlat::Param => unreachable!("Param is generated only as explicit ops at start"),
        FuzzOpFlat::Literal => {
            // Literal op: generate a literal byte value.
            let literal_bits = u.int_in_range(1..=8)?;
            let value = u.int_in_range(0..=((1 << literal_bits) - 1))?;
            (
                FuzzOp::Literal {
                    bits: literal_bits,
                    value,
                },
                InternalType::Bits(literal_bits as usize),
            )
        }
        FuzzOpFlat::Unop => {
            let (idx, ty) = pick_bits_type(u, node_types)?;
            let unop = u.arbitrary::<FuzzUnop>()?;
            let operand_bits = match ty {
                InternalType::Bits(w) => w,
                _ => 1usize,
            };
            let out_bits = match unop {
                FuzzUnop::OrReduce | FuzzUnop::AndReduce | FuzzUnop::XorReduce => 1usize,
                FuzzUnop::Encode => operand_bits.next_power_of_two().trailing_zeros() as usize,
                _ => operand_bits,
            };
            (FuzzOp::Unop(unop, idx), InternalType::Bits(out_bits))
        }
        FuzzOpFlat::Binop => {
            let (pair, _sty) = pick_same_bits_types(u, node_types, 2)?;
            let idx1 = pair[0];
            let idx2 = pair[1];
            let binop = u.arbitrary::<FuzzBinop>()?;
            let b1 = match node_types.get(idx1) {
                Some(InternalType::Bits(w)) => *w,
                _ => 1usize,
            };
            let b2 = match node_types.get(idx2) {
                Some(InternalType::Bits(w)) => *w,
                _ => 1usize,
            };
            let out_bits = match binop {
                FuzzBinop::Eq
                | FuzzBinop::Ne
                | FuzzBinop::Ugt
                | FuzzBinop::Ule
                | FuzzBinop::Ult
                | FuzzBinop::Uge
                | FuzzBinop::Sge
                | FuzzBinop::Sgt
                | FuzzBinop::Slt
                | FuzzBinop::Sle => 1usize,
                FuzzBinop::Concat => b1.saturating_add(b2),
                FuzzBinop::Shrl | FuzzBinop::Shra | FuzzBinop::Shll => b1,
                _ => b1.max(b2),
            };
            (
                FuzzOp::Binop(binop, idx1, idx2),
                InternalType::Bits(out_bits),
            )
        }
        FuzzOpFlat::ZeroExt => {
            let (index, _) = pick_bits_type(u, node_types)?;
            let new_bit_count = u.int_in_range(1..=8)? as u64;
            (
                FuzzOp::ZeroExt {
                    operand: FuzzOperand { index },
                    new_bit_count,
                },
                InternalType::Bits(new_bit_count as usize),
            )
        }
        FuzzOpFlat::SignExt => {
            let (index, _) = pick_bits_type(u, node_types)?;
            let new_bit_count = u.int_in_range(1..=8)? as u64;
            (
                FuzzOp::SignExt {
                    operand: FuzzOperand { index },
                    new_bit_count,
                },
                InternalType::Bits(new_bit_count as usize),
            )
        }
        FuzzOpFlat::BitSlice => {
            // Slice a random bits-typed operand using its own width.
            let (index, ty) = pick_bits_type(u, node_types)?;
            let op_bits: usize = match ty {
                InternalType::Bits(w) => w,
                _ => 1usize,
            };
            let start = if op_bits <= 1 {
                0u64
            } else {
                u.int_in_range(0..=((op_bits as u64) - 1))? as u64
            };
            let max_width = (op_bits as u64).saturating_sub(start);
            let width = u.int_in_range(1..=max_width)? as u64;
            (
                FuzzOp::BitSlice {
                    operand: FuzzOperand { index },
                    start,
                    width,
                },
                InternalType::Bits(width as usize),
            )
        }
        FuzzOpFlat::OneHot => {
            let (index, ty) = pick_bits_type(u, node_types)?;
            let lsb_prio = u.int_in_range(0..=1)? == 1;
            let operand_bits = match ty {
                InternalType::Bits(w) => w,
                _ => 1usize,
            };
            (
                FuzzOp::OneHot {
                    arg: FuzzOperand { index },
                    lsb_prio,
                },
                InternalType::Bits(operand_bits),
            )
        }
        FuzzOpFlat::PrioritySel => {
            let (index, _) = pick_bits_type(u, node_types)?;
            let num_cases = u.int_in_range(1..=8)?;
            let (same, shared_ty) = pick_same_types(u, node_types, num_cases as usize)?;
            let cases: Vec<FuzzOperand> = same
                .into_iter()
                .map(|index| FuzzOperand { index })
                .collect();
            let last_case: FuzzOperand = cases.last().unwrap().clone();
            (
                FuzzOp::PrioritySel {
                    selector: FuzzOperand { index },
                    cases: cases.clone(),
                    default: last_case.clone(),
                },
                shared_ty,
            )
        }
        FuzzOpFlat::OneHotSel => {
            let (index, _) = pick_bits_type(u, node_types)?;
            let num_cases = u.int_in_range(1..=8)?;
            let (same, shared_ty) = pick_same_types(u, node_types, num_cases as usize)?;
            let cases: Vec<FuzzOperand> = same
                .into_iter()
                .map(|index| FuzzOperand { index })
                .collect();
            (
                FuzzOp::OneHotSel {
                    selector: FuzzOperand { index },
                    cases: cases.clone(),
                },
                shared_ty,
            )
        }
        FuzzOpFlat::Decode => {
            let (index, _) = pick_bits_type(u, node_types)?;
            let width = u.int_in_range(1..=8)? as u64;
            (
                FuzzOp::Decode {
                    arg: FuzzOperand { index },
                    width,
                },
                InternalType::Bits(width as usize),
            )
        }
        FuzzOpFlat::ArrayIndex => {
            let (index, aty) = pick_array_type(u, node_types)?;
            let op = FuzzOp::ArrayIndex {
                array: FuzzOperand { index },
                index: FuzzOperand { index },
            };
            let out_ty = match aty {
                InternalType::Array(arr) => (*arr.element_type).clone(),
                _ => InternalType::Bits(1),
            };
            (op, out_ty)
        }
        FuzzOpFlat::Array => {
            let num_elements = u.int_in_range(1..=MAX_ELEMENTS_PER_ARRAY)?;
            let (same, shared_ty) = pick_same_types(u, node_types, num_elements as usize)?;
            let elem_leaves: u64 = count_type_leaves(&shared_ty) as u64;
            let allowed = if num_elements * elem_leaves > MAX_TOTAL_ELEMENTS_PER_TYPE {
                MAX_TOTAL_ELEMENTS_PER_TYPE / elem_leaves
            } else {
                num_elements
            };
            let mut elements: Vec<FuzzOperand> = same
                .into_iter()
                .map(|index| FuzzOperand { index })
                .collect();
            if elements.len() > allowed as usize {
                elements.truncate(allowed as usize);
            }
            (
                FuzzOp::Array {
                    elements: elements.clone(),
                },
                InternalType::Array(crate::ir::ArrayTypeData {
                    element_type: Box::new(shared_ty),
                    element_count: elements.len(),
                }),
            )
        }
        FuzzOpFlat::Tuple => {
            if node_types.is_empty() {
                // Explicitly generate an empty tuple when no operands exist.
                (
                    FuzzOp::Tuple {
                        elements: Vec::new(),
                    },
                    InternalType::Tuple(Vec::new()),
                )
            } else {
                let num_elements = u.int_in_range(1..=MAX_ELEMENTS_PER_TUPLE)?;
                // For tuple, allow mixed types; select indices within available_nodes (with
                // replacement).
                let mut elements: Vec<FuzzOperand> = Vec::with_capacity(num_elements as usize);
                let mut leaf_count: usize = 0;
                for _ in 0..num_elements {
                    let element_index =
                        u.int_in_range(0..=((node_types.len() as u64) - 1)).unwrap() as usize;
                    let element_leaf_count = count_type_leaves(&node_types[element_index]);
                    if leaf_count + element_leaf_count > MAX_TOTAL_ELEMENTS_PER_TYPE as usize {
                        break;
                    }
                    elements.push(FuzzOperand {
                        index: element_index,
                    });
                    leaf_count += element_leaf_count;
                }
                if elements.is_empty() {
                    return Ok((
                        FuzzOp::Tuple {
                            elements: Vec::new(),
                        },
                        InternalType::Tuple(Vec::new()),
                    ));
                }
                let elem_types: Vec<Box<InternalType>> = elements
                    .iter()
                    .map(|e| Box::new(node_types[e.index].clone()))
                    .collect();
                (FuzzOp::Tuple { elements }, InternalType::Tuple(elem_types))
            }
        }
        FuzzOpFlat::TupleIndex => {
            // Choose only from nodes that are tuples so we can stay in-bounds.
            let (tuple_idx, tty) = pick_tuple_type(u, node_types)?;
            let tuple_len = match &tty {
                InternalType::Tuple(elems) => elems.len(),
                _ => unreachable!("chosen index must be tuple typed"),
            };
            if tuple_len == 0 {
                return Err(arbitrary::Error::IncorrectFormat);
            }
            let index: usize = u.int_in_range(0..=((tuple_len - 1) as u64))? as usize;
            let elem_ty = match &tty {
                InternalType::Tuple(elems) => elems[index].as_ref().clone(),
                _ => unreachable!("tuple_candidates built from tuples only"),
            };
            (
                FuzzOp::TupleIndex {
                    tuple: FuzzOperand { index: tuple_idx },
                    index,
                },
                elem_ty,
            )
        }
        FuzzOpFlat::UMul => {
            let (pair, _ty) = pick_same_bits_types(u, node_types, 2)?;
            let idx1 = pair[0];
            let idx2 = pair[1];
            let b1 = match node_types.get(idx1) {
                Some(InternalType::Bits(w)) => *w,
                _ => 1usize,
            };
            let b2 = match node_types.get(idx2) {
                Some(InternalType::Bits(w)) => *w,
                _ => 1usize,
            };
            (
                FuzzOp::UMul {
                    lhs: FuzzOperand { index: idx1 },
                    rhs: FuzzOperand { index: idx2 },
                },
                InternalType::Bits(b1.max(b2)),
            )
        }
        FuzzOpFlat::SMul => {
            let (pair, _ty) = pick_same_bits_types(u, node_types, 2)?;
            let idx1 = pair[0];
            let idx2 = pair[1];
            let b1 = match node_types.get(idx1) {
                Some(InternalType::Bits(w)) => *w,
                _ => 1usize,
            };
            let b2 = match node_types.get(idx2) {
                Some(InternalType::Bits(w)) => *w,
                _ => 1usize,
            };
            (
                FuzzOp::SMul {
                    lhs: FuzzOperand { index: idx1 },
                    rhs: FuzzOperand { index: idx2 },
                },
                InternalType::Bits(b1.max(b2)),
            )
        }
        FuzzOpFlat::Sel => {
            let (index, _) = pick_bits_type(u, node_types)?;
            let num_cases = u.int_in_range(1..=8)?;
            let (same, shared_ty) = pick_same_types(u, node_types, num_cases as usize)?;
            let cases: Vec<FuzzOperand> = same
                .iter()
                .copied()
                .map(|index| FuzzOperand { index })
                .collect();
            let default = cases[0].index;
            let out_ty = shared_ty;
            (
                FuzzOp::Sel {
                    selector: FuzzOperand { index },
                    cases,
                    default: FuzzOperand { index: default },
                },
                out_ty,
            )
        }
        FuzzOpFlat::DynamicBitSlice => {
            let (arg, _) = pick_bits_type(u, node_types)?;
            let (start, _) = pick_bits_type(u, node_types)?;
            let width = u.int_in_range(1..=8)? as u64;
            (
                FuzzOp::DynamicBitSlice {
                    arg: FuzzOperand { index: arg },
                    start: FuzzOperand { index: start },
                    width,
                },
                InternalType::Bits(width as usize),
            )
        }
        FuzzOpFlat::BitSliceUpdate => {
            let (pair, shared_ty) = pick_same_bits_types(u, node_types, 2)?;
            let value_idx = pair[0];
            let update_idx = pair[1];
            let (start_idx, _) = pick_bits_type(u, node_types)?;
            let out_ty = shared_ty;
            (
                FuzzOp::BitSliceUpdate {
                    value: FuzzOperand { index: value_idx },
                    start: FuzzOperand { index: start_idx },
                    update: FuzzOperand { index: update_idx },
                },
                out_ty,
            )
        }
    };
    log::trace!("Generated fuzz op: {:?}", op);
    if let InternalType::Bits(w) = out_ty {
        // Zero-width types may cause downstream issues, so error out.
        if w == 0 {
            return Err(arbitrary::Error::IncorrectFormat);
        }
    }
    Ok((op, out_ty))
}

impl<'a> arbitrary::Arbitrary<'a> for FuzzSample {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        log::debug!("Generating fuzz sample");
        // Decide how many parameters and operations to generate.
        let num_params = u.int_in_range(1..=MAX_PARAMS_PER_SAMPLE)? as usize;
        // Decide how many operations to generate.
        let num_ops = u.int_in_range(0..=MAX_OPS_PER_SAMPLE - num_params as u64)?;
        // We maintain a parallel vector that records, for each existing IR node,
        // an over-approximate `InternalType`. Index 0 corresponds to the primary input.
        let mut node_types: Vec<InternalType> = Vec::with_capacity(num_params + num_ops as usize);
        let mut ops = Vec::with_capacity(num_params + num_ops as usize);

        // Generate parameter ops first.
        for _ in 0..num_params {
            let bits = u.int_in_range(1..=8)?;
            ops.push(FuzzOp::Param { bits });
            node_types.push(InternalType::Bits(bits as usize));
        }

        for _ in 0..num_ops {
            let (op, out_ty) = generate_fuzz_op(u, &node_types)?;
            // Push op and its output type.
            ops.push(op);
            node_types.push(out_ty);
        }
        log::debug!("Generated fuzz sample: {:?}", ops);
        Ok(FuzzSample {
            ops,
            op_types: node_types,
        })
    }
}

impl FuzzSample {
    pub fn gen_with_same_signature<'a>(
        orig: &FuzzSample,
        u: &mut arbitrary::Unstructured<'a>,
        pkg_for_orig: &mut xlsynth::IrPackage,
        _pkg_for_tmp: &mut xlsynth::IrPackage,
    ) -> arbitrary::Result<Self> {
        let (param_types, orig_internal_ret_type): (Vec<InternalType>, Option<InternalType>) =
            generate_ir_fn(orig.ops.clone(), pkg_for_orig, None)
                .ok()
                .and_then(|f| {
                    f.get_type().ok().map(|t| {
                        // Extract parameter types
                        let mut pts: Vec<InternalType> = Vec::with_capacity(t.param_count());
                        for i in 0..t.param_count() {
                            if let Ok(pt) = t.param_type(i) {
                                let s = pt.to_string();
                                let mut p = ir_parser::Parser::new(&s);
                                if let Ok(ip) = p.parse_type() {
                                    pts.push(ip);
                                }
                            }
                        }
                        // Extract return type
                        let ret_ty = t.return_type();
                        let ret_ty_str = ret_ty.to_string();
                        let mut parser = ir_parser::Parser::new(&ret_ty_str);
                        let internal = parser.parse_type().ok();
                        (pts, internal)
                    })
                })
                .unwrap_or((Vec::new(), None));

        if let Some(ref want_ty) = orig_internal_ret_type {
            return generate_fuzz_sample_with_type(u, &param_types, want_ty);
        }
        // If we couldn't determine the original function's return type, treat
        // this fuzz input as unusable instead of generating a random sample.
        Err(arbitrary::Error::IncorrectFormat)
    }
}

/// Appends operations to `sample` that synthesize a value of `ty` and returns
/// the corresponding operand reference.
fn append_ops_for_type(sample: &mut FuzzSample, ty: &InternalType) -> FuzzOperand {
    match ty {
        InternalType::Bits(w) => {
            let lit_bits: u8 = (*w as u8).clamp(1, u8::MAX);
            sample.ops.push(FuzzOp::Literal {
                bits: lit_bits,
                value: 0,
            });
            sample.op_types.push(InternalType::Bits(lit_bits as usize));
            FuzzOperand {
                index: sample.op_types.len() - 1,
            }
        }
        InternalType::Tuple(types) => {
            let mut elems: Vec<FuzzOperand> = Vec::with_capacity(types.len());
            for t in types {
                elems.push(append_ops_for_type(sample, t));
            }
            sample.ops.push(FuzzOp::Tuple {
                elements: elems.clone(),
            });
            sample.op_types.push(ty.clone());
            FuzzOperand {
                index: sample.op_types.len() - 1,
            }
        }
        InternalType::Array(arr) => {
            let mut elems: Vec<FuzzOperand> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(append_ops_for_type(sample, &arr.element_type));
            }
            sample.ops.push(FuzzOp::Array {
                elements: elems.clone(),
            });
            sample.op_types.push(ty.clone());
            FuzzOperand {
                index: sample.op_types.len() - 1,
            }
        }
        InternalType::Token => {
            // Not supported directly; produce a u1 literal to keep progress.
            sample.ops.push(FuzzOp::Literal { bits: 1, value: 0 });
            sample.op_types.push(InternalType::Bits(1));
            FuzzOperand {
                index: sample.op_types.len() - 1,
            }
        }
    }
}

/// Generates a FuzzSample that conforms to the provided function signature
/// (parameter types and return type). For now, parameters are encoded as
/// `Param { bits }` ops for bit-typed parameters; non-bits params are mapped to
/// `bits[1]`.
pub fn generate_fuzz_sample_with_type<'a>(
    u: &mut arbitrary::Unstructured<'a>,
    param_types: &[InternalType],
    return_type: &InternalType,
) -> arbitrary::Result<FuzzSample> {
    let mut sample = FuzzSample {
        ops: Vec::new(),
        op_types: Vec::new(),
    };
    // Emit params
    for pt in param_types {
        match pt {
            InternalType::Bits(w) => {
                let bits: u8 = (*w as u8).clamp(1, 8);
                sample.ops.push(FuzzOp::Param { bits });
                sample.op_types.push(InternalType::Bits(bits as usize));
            }
            _ => {
                sample.ops.push(FuzzOp::Param { bits: 1 });
                sample.op_types.push(InternalType::Bits(1));
            }
        }
    }
    let extra_ops = u.int_in_range(0..=MAX_OPS_PER_SAMPLE)?;
    for _ in 0..extra_ops {
        let (op, out_ty) = generate_fuzz_op(u, &sample.op_types)?;
        sample.ops.push(op);
        sample.op_types.push(out_ty);
    }
    // Ensure the sample ends with a value of the desired return type
    let _ = append_ops_for_type(&mut sample, return_type);
    Ok(sample)
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzSampleWithArgs {
    pub sample: FuzzSample,
    /// Random seeds used to derive argument values for the function parameters.
    pub arg_seeds: Vec<u64>,
}

impl FuzzSampleWithArgs {
    /// Generates `IrValue` arguments for the parameters of the given internal
    /// IR function.
    pub fn gen_args_for_fn(&self, f: &ir::Fn) -> Vec<xlsynth::IrValue> {
        use rand_pcg::Pcg64Mcg;
        let mut out: Vec<xlsynth::IrValue> = Vec::with_capacity(f.params.len());
        for (i, p) in f.params.iter().enumerate() {
            let seed = self.arg_seeds.get(i).copied().unwrap_or(0);
            let mut rng = Pcg64Mcg::new(seed as u128);
            out.push(arbitrary_ir_value_for_internal_type(&mut rng, &p.ty));
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct FuzzSampleSameTypedPair {
    pub first: FuzzSample,
    pub second: FuzzSample,
}

impl<'a> arbitrary::Arbitrary<'a> for FuzzSampleSameTypedPair {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let first = FuzzSample::arbitrary(u)?;

        // Dependency-inject packages for internal type probing to avoid creating them
        // inside.
        let mut pkg_for_orig = xlsynth::IrPackage::new("pair_gen_orig")
            .map_err(|_| arbitrary::Error::IncorrectFormat)?;
        let mut pkg_for_tmp = xlsynth::IrPackage::new("pair_gen_tmp")
            .map_err(|_| arbitrary::Error::IncorrectFormat)?;
        let second =
            FuzzSample::gen_with_same_signature(&first, u, &mut pkg_for_orig, &mut pkg_for_tmp)?;
        Ok(FuzzSampleSameTypedPair { first, second })
    }
}

// -- Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use xlsynth::IrPackage;

    fn make_bits_node(fn_builder: &mut FnBuilder, width: usize, value: u64) -> BValue {
        let ir_value = xlsynth::IrValue::make_ubits(width, value).unwrap();
        fn_builder.literal(&ir_value, None)
    }

    #[test]
    fn test_build_bits_exact() {
        let mut pkg = IrPackage::new("test").unwrap();
        let mut fn_builder = FnBuilder::new(&mut pkg, "f", true);
        let node = make_bits_node(&mut fn_builder, 8, 42);
        let ty_str = "bits[8]";
        let mut parser = ir_parser::Parser::new(ty_str);
        let ty = parser.parse_type().unwrap();
        let mut map = BTreeMap::new();
        map.entry(ty.clone()).or_insert_with(|| vec![node.clone()]);
        let out = build_return_value_with_type(&mut fn_builder, &mut pkg, &map, &ty);
        assert!(out == Some(node));
    }

    #[test]
    fn test_build_bits_slice_and_extend() {
        let mut pkg = IrPackage::new("test").unwrap();
        let mut fn_builder = FnBuilder::new(&mut pkg, "f", true);
        let node = make_bits_node(&mut fn_builder, 16, 0xFFFF);
        let ty_str = "bits[8]";
        let mut parser = ir_parser::Parser::new(ty_str);
        let ty = parser.parse_type().unwrap();
        let mut map = BTreeMap::new();
        let node_ty_str = "bits[16]";
        let mut parser2 = ir_parser::Parser::new(node_ty_str);
        let node_ty = parser2.parse_type().unwrap();
        map.entry(node_ty.clone())
            .or_insert_with(|| vec![node.clone()]);
        // Should slice down
        let out = build_return_value_with_type(&mut fn_builder, &mut pkg, &map, &ty);
        assert!(out.is_some());
        // Should extend up
        let node8 = make_bits_node(&mut fn_builder, 8, 0xAA);
        let ty8_str = "bits[8]";
        let mut parser3 = ir_parser::Parser::new(ty8_str);
        let ty8 = parser3.parse_type().unwrap();
        let mut map2 = BTreeMap::new();
        map2.entry(ty8.clone())
            .or_insert_with(|| vec![node8.clone()]);
        let ty16_str = "bits[16]";
        let mut parser4 = ir_parser::Parser::new(ty16_str);
        let ty16 = parser4.parse_type().unwrap();
        let out2 = build_return_value_with_type(&mut fn_builder, &mut pkg, &map2, &ty16);
        assert!(out2.is_some());
    }

    #[test]
    fn test_build_tuple() {
        let mut pkg = IrPackage::new("test").unwrap();
        let mut fn_builder = FnBuilder::new(&mut pkg, "f", true);
        let a = make_bits_node(&mut fn_builder, 3, 1);
        let b = make_bits_node(&mut fn_builder, 5, 2);
        let a_ty_str = "bits[3]";
        let b_ty_str = "bits[5]";
        let mut parser_a = ir_parser::Parser::new(a_ty_str);
        let mut parser_b = ir_parser::Parser::new(b_ty_str);
        let a_ty = parser_a.parse_type().unwrap();
        let b_ty = parser_b.parse_type().unwrap();
        let mut map = BTreeMap::new();
        map.entry(a_ty.clone()).or_insert_with(|| vec![a.clone()]);
        map.entry(b_ty.clone()).or_insert_with(|| vec![b.clone()]);
        let tuple_ty_str = "(bits[3], bits[5])";
        let mut parser_tuple = ir_parser::Parser::new(tuple_ty_str);
        let tuple_ty = parser_tuple.parse_type().unwrap();
        let out = build_return_value_with_type(&mut fn_builder, &mut pkg, &map, &tuple_ty);
        assert!(out.is_some());
    }

    #[test]
    fn test_build_array() {
        let mut pkg = IrPackage::new("test").unwrap();
        let mut fn_builder = FnBuilder::new(&mut pkg, "f", true);
        let elem = make_bits_node(&mut fn_builder, 4, 7);
        let elem_ty_str = "bits[4]";
        let mut parser_elem = ir_parser::Parser::new(elem_ty_str);
        let elem_ty = parser_elem.parse_type().unwrap();
        let mut map = BTreeMap::new();
        map.entry(elem_ty.clone())
            .or_insert_with(|| vec![elem.clone()]);
        let arr_ty_str = "bits[4][3]";
        let mut parser_arr = ir_parser::Parser::new(arr_ty_str);
        let arr_ty = parser_arr.parse_type().unwrap();
        let out = build_return_value_with_type(&mut fn_builder, &mut pkg, &map, &arr_ty);
        assert!(out.is_some());
    }

    #[test]
    fn test_build_succeeds_via_literal_synthesis_when_no_candidates() {
        let mut pkg = IrPackage::new("test").unwrap();
        let mut fn_builder = FnBuilder::new(&mut pkg, "f", true);
        let map = BTreeMap::new();
        let ty_str = "bits[8]";
        let mut parser = ir_parser::Parser::new(ty_str);
        let ty = parser.parse_type().unwrap();
        let out = build_return_value_with_type(&mut fn_builder, &mut pkg, &map, &ty);
        assert!(out.is_some());
        // Ensure the synthesized value has the requested type.
        let out_bv = out.unwrap();
        let got_ty = fn_builder.get_type(&out_bv).unwrap();
        assert_eq!(got_ty.to_string(), "bits[8]");
    }

    #[test]
    fn generates_variety_of_samples() {
        use rand::RngCore;
        use rand_pcg::Pcg64Mcg;
        use std::collections::HashSet;

        const SAMPLE_COUNT: usize = 10000;
        let mut successful_samples = 0usize;
        let mut max_num_ops = 0usize;
        let mut min_num_ops = usize::MAX;
        let mut max_num_params = 0usize;
        let mut min_num_params = usize::MAX;
        let mut max_op_kinds = 0usize;
        let mut min_op_kinds = usize::MAX;
        // Fixed seed for determinism across runs.
        let mut rng = Pcg64Mcg::new(0xDEADBEEFCAFEBABEu128);
        for _ in 0..SAMPLE_COUNT {
            // Generate a fresh buffer of random bytes for the Arbitrary source.
            let mut buf = [0u8; 4096];
            rng.fill_bytes(&mut buf);
            let mut un = arbitrary::Unstructured::new(&buf);
            if let Ok(sample) = FuzzSample::arbitrary(&mut un) {
                successful_samples += 1;

                let total_ops = sample.ops.len();
                min_num_ops = min_num_ops.min(total_ops);
                max_num_ops = max_num_ops.max(total_ops);

                // Count distinct kinds via FuzzOpFlat mapping
                let mut kinds: HashSet<FuzzOpFlat> = HashSet::new();
                for op in sample.ops.iter() {
                    kinds.insert(to_flat(op));
                }
                min_op_kinds = min_op_kinds.min(kinds.len());
                max_op_kinds = max_op_kinds.max(kinds.len());

                let total_params = sample
                    .ops
                    .iter()
                    .filter(|op| matches!(op, FuzzOp::Param { .. }))
                    .count();
                min_num_params = min_num_params.min(total_params);
                max_num_params = max_num_params.max(total_params);
            }
        }
        // At least 50% of the samples should be successfully generated.
        assert!(successful_samples > SAMPLE_COUNT / 2);

        assert!(
            max_num_ops <= MAX_OPS_PER_SAMPLE as usize,
            "Expected at most MAX_OPS_PER_SAMPLE ops"
        );
        assert!(
            max_num_ops == MAX_OPS_PER_SAMPLE as usize,
            "Expected a maximum sized sample with MAX_OPS_PER_SAMPLE ops"
        );
        assert!(min_num_ops == 1);
        assert!(max_num_params == MAX_PARAMS_PER_SAMPLE as usize);
        assert!(min_num_params == 1);
        assert!(max_op_kinds > 10);
        assert!(min_op_kinds == 1);
    }

    #[test]
    fn generate_ir_fn_success_rate() {
        use rand::RngCore;
        use rand_pcg::Pcg64Mcg;

        const SAMPLE_COUNT: usize = 10000;
        let mut successful_samples = 0usize;
        let mut generate_ir_succeeded = 0usize;
        let mut parse_ir_succeeded = 0usize;

        // Fixed seed for determinism across runs.
        let mut rng = Pcg64Mcg::new(0x1234ABCD9999EEEFu128);
        for _ in 0..SAMPLE_COUNT {
            let mut buf = [0u8; 4096];
            rng.fill_bytes(&mut buf);
            let mut un = arbitrary::Unstructured::new(&buf);
            if let Ok(sample) = FuzzSample::arbitrary(&mut un) {
                successful_samples += 1;
                if let Ok(mut pkg) = IrPackage::new("gen_ir_test") {
                    if generate_ir_fn(sample.ops.clone(), &mut pkg, None).is_ok() {
                        generate_ir_succeeded += 1;
                        let pkg_text = pkg.to_string();
                        if ir_parser::Parser::new(&pkg_text)
                            .parse_and_validate_package()
                            .is_ok()
                        {
                            parse_ir_succeeded += 1;
                        }
                    }
                }
            }
        }
        // Verify at least 25% of successfully generated FuzzSamples convert.
        assert!(
            generate_ir_succeeded >= successful_samples / 4,
            "expected at least 25% of samples to be convertible to IR: converted={}, samples={}",
            generate_ir_succeeded,
            successful_samples
        );
        // Verify at least 25% of successfully generated FuzzSamples produce parseable
        // IR.
        assert!(
            parse_ir_succeeded >= successful_samples / 4,
            "expected at least 25% of samples to produce parseable IR: converted={}, samples={}",
            parse_ir_succeeded,
            successful_samples
        );
    }

    #[test]
    fn generate_live_ir() {
        use rand::RngCore;
        use rand_pcg::Pcg64Mcg;

        const SAMPLE_COUNT: usize = 10000;
        let mut successful_samples = 0usize;
        let mut generate_ir_succeeded = 0usize;
        let mut parse_ir_succeeded = 0usize;

        // Fixed seed for determinism across runs.
        let mut rng = Pcg64Mcg::new(0x1234ABCD9999EEEFu128);
        for _ in 0..SAMPLE_COUNT {
            let mut buf = [0u8; 4096];
            rng.fill_bytes(&mut buf);
            let mut un = arbitrary::Unstructured::new(&buf);
            if let Ok(sample) = FuzzSample::arbitrary(&mut un) {
                successful_samples += 1;
                if let Ok(mut pkg) = IrPackage::new("gen_ir_test") {
                    if generate_ir_fn(sample.ops.clone(), &mut pkg, None).is_ok() {
                        generate_ir_succeeded += 1;
                        let pkg_text = pkg.to_string();
                        if ir_parser::Parser::new(&pkg_text)
                            .parse_and_validate_package()
                            .is_ok()
                        {
                            parse_ir_succeeded += 1;
                        }
                    }
                }
            }
        }
        // Verify at least 25% of successfully generated FuzzSamples convert.
        assert!(
            generate_ir_succeeded >= successful_samples / 4,
            "expected at least 25% of samples to be convertible to IR: converted={}, samples={}",
            generate_ir_succeeded,
            successful_samples
        );
        // Verify at least 25% of successfully generated FuzzSamples produce parseable
        // IR.
        assert!(
            parse_ir_succeeded >= successful_samples / 4,
            "expected at least 25% of samples to produce parseable IR: converted={}, samples={}",
            parse_ir_succeeded,
            successful_samples
        );
    }
}
