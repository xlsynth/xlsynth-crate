// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use std::collections::{HashMap, hash_map::OccupiedError};

use xlsynth::{IrValue, ir_value::IrFormatPreference};

use crate::{ir, ir_parser, ir_utils::operands};

/// Strongly-typed wrapper for parameter IDs.
///
/// Note: This is *not* a general node id. This is an ordinal referring to the
/// dense parameter space for a function signature (i.e., the Nth parameter),
/// not a node id in the IR graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(usize);

impl ParamId {
    /// Constructs a new ParamId, asserting that the id is greater than zero.
    pub fn new(id: usize) -> Self {
        assert!(id > 0, "ParamId must be greater than zero, got {}", id);
        ParamId(id)
    }

    /// Returns the wrapped id value.
    pub fn get_wrapped_id(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ArrayTypeData {
    pub element_type: Box<Type>,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Type {
    Token,
    Bits(usize),
    Tuple(Vec<Box<Type>>),
    Array(ArrayTypeData),
}

/// Represents an interval of the form `[start, limit)` i.e. inclusive start
/// exclusive limit.
pub struct StartAndLimit {
    pub start: usize,
    pub limit: usize,
}

impl Type {
    pub fn new_array(element_type: Type, element_count: usize) -> Self {
        Type::Array(ArrayTypeData {
            element_type: Box::new(element_type),
            element_count,
        })
    }

    pub fn nil() -> Self {
        Type::Tuple(vec![])
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Type::Tuple(types) if types.is_empty())
    }

    pub fn bit_count(&self) -> usize {
        match self {
            Type::Token => 0,
            Type::Bits(width) => *width,
            Type::Tuple(types) => types.iter().map(|t| t.bit_count()).sum(),
            Type::Array(ArrayTypeData {
                element_type,
                element_count,
            }) => element_type.bit_count() * element_count,
        }
    }

    /// Returns the start and limit bits for our bitwise representation of a
    /// tuple access at the given index.
    ///
    /// E.g. consider `tuple(a, b, c)`, we represent this in a bit vector as:
    /// `a_msb, ..., a_lsb, b_msb, ..., b_lsb, c_msb, ..., c_lsb` where c_lsb is
    /// the least significant bit of the overall bit vector. That means to
    /// access index `i` we have to slice out all the members that come
    /// after it; e.g. if we want to access b we have to skip `c`
    /// least significant bits.
    pub fn tuple_get_flat_bit_slice_for_index(
        &self,
        index: usize,
    ) -> Result<StartAndLimit, String> {
        match self {
            Type::Tuple(types) => {
                let bits_after_index = types[index + 1..].iter().map(|t| t.bit_count()).sum();
                let limit = types[index].bit_count() + bits_after_index;
                Ok(StartAndLimit {
                    start: bits_after_index,
                    limit,
                })
            }
            _ => Err(format!(
                "Attempted to get bit slice for non-tuple type: {:?}",
                self
            )),
        }
    }

    pub fn get_array_element_type(&self) -> &Type {
        match self {
            Type::Array(ArrayTypeData { element_type, .. }) => element_type,
            _ => panic!(
                "Attempted to get array element type for non-array type: {:?}",
                self
            ),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Token => write!(f, "token"),
            Type::Bits(width) => write!(f, "bits[{}]", width),
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::Array(ArrayTypeData {
                element_type,
                element_count,
            }) => {
                write!(f, "{}", element_type)?;
                write!(f, "[{}]", element_count)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Binop {
    Add,
    Sub,

    Shll,
    Shrl,
    Shra,

    ArrayConcat,
    Smulp,
    Umulp,

    Eq,
    Ne,

    Uge,
    Ugt,
    Ult,
    Ule,

    // signed comparisons
    Sgt,
    Sge,
    Slt,
    Sle,

    Umul,
    Smul,

    Sdiv,
    Udiv,
    Umod,
    Smod,

    Gate,
}

pub fn operator_to_binop(operator: &str) -> Option<Binop> {
    match operator {
        // Shifts
        "shll" => Some(Binop::Shll),
        "shrl" => Some(Binop::Shrl),
        "shra" => Some(Binop::Shra),

        "array_concat" => Some(Binop::ArrayConcat),
        "smulp" => Some(Binop::Smulp),
        "umulp" => Some(Binop::Umulp),
        "umod" => Some(Binop::Umod),
        "eq" => Some(Binop::Eq),
        "ne" => Some(Binop::Ne),
        // signed comparisons
        "sgt" => Some(Binop::Sgt),
        "sge" => Some(Binop::Sge),
        "slt" => Some(Binop::Slt),
        "sle" => Some(Binop::Sle),
        // unsigned comparisons
        "uge" => Some(Binop::Uge),
        "ugt" => Some(Binop::Ugt),
        "ult" => Some(Binop::Ult),
        "ule" => Some(Binop::Ule),
        // arithmetic
        "add" => Some(Binop::Add),
        "sub" => Some(Binop::Sub),
        "umul" => Some(Binop::Umul),
        "smul" => Some(Binop::Smul),
        "sdiv" => Some(Binop::Sdiv),
        "udiv" => Some(Binop::Udiv),
        "umod" => Some(Binop::Umod),
        "smod" => Some(Binop::Smod),

        // "special" operations
        "gate" => Some(Binop::Gate),
        _ => None,
    }
}

pub fn binop_to_operator(binop: Binop) -> &'static str {
    match binop {
        Binop::Add => "add",
        Binop::Shll => "shll",
        Binop::Shrl => "shrl",
        Binop::Shra => "shra",
        Binop::ArrayConcat => "array_concat",
        Binop::Smulp => "smulp",
        Binop::Umulp => "umulp",
        Binop::Eq => "eq",
        Binop::Ne => "ne",
        Binop::Uge => "uge",
        Binop::Ugt => "ugt",
        Binop::Ult => "ult",
        Binop::Ule => "ule",
        Binop::Sgt => "sgt",
        Binop::Sge => "sge",
        Binop::Slt => "slt",
        Binop::Sle => "sle",
        Binop::Sub => "sub",
        Binop::Umul => "umul",
        Binop::Smul => "smul",
        Binop::Gate => "gate",
        Binop::Sdiv => "sdiv",
        Binop::Udiv => "udiv",
        Binop::Umod => "umod",
        Binop::Smod => "smod",
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unop {
    Neg,
    Not,
    Identity,
    Reverse,
    OrReduce,
    AndReduce,
    XorReduce,
}

pub fn operator_to_unop(operator: &str) -> Option<Unop> {
    match operator {
        "neg" => Some(Unop::Neg),
        "not" => Some(Unop::Not),
        "identity" => Some(Unop::Identity),
        "reverse" => Some(Unop::Reverse),
        "or_reduce" => Some(Unop::OrReduce),
        "and_reduce" => Some(Unop::AndReduce),
        "xor_reduce" => Some(Unop::XorReduce),
        _ => None,
    }
}

pub fn unop_to_operator(unop: Unop) -> &'static str {
    match unop {
        Unop::Neg => "neg",
        Unop::Not => "not",
        Unop::Identity => "identity",
        Unop::Reverse => "reverse",
        Unop::OrReduce => "or_reduce",
        Unop::AndReduce => "and_reduce",
        Unop::XorReduce => "xor_reduce",
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum NaryOp {
    And,
    Nor,
    Or,
    Xor,
    Nand,
    Concat,
}

pub fn operator_to_nary_op(operator: &str) -> Option<NaryOp> {
    match operator {
        "and" => Some(NaryOp::And),
        "nor" => Some(NaryOp::Nor),
        "or" => Some(NaryOp::Or),
        "xor" => Some(NaryOp::Xor),
        "nand" => Some(NaryOp::Nand),
        "concat" => Some(NaryOp::Concat),
        _ => None,
    }
}

pub fn nary_op_to_operator(nary_op: NaryOp) -> &'static str {
    match nary_op {
        NaryOp::And => "and",
        NaryOp::Nor => "nor",
        NaryOp::Or => "or",
        NaryOp::Xor => "xor",
        NaryOp::Nand => "nand",
        NaryOp::Concat => "concat",
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct NodeRef {
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodePayload {
    Nil,
    GetParam(ParamId),
    Tuple(Vec<NodeRef>),
    Array(Vec<NodeRef>),
    /// array_slice(array, start, width=<width>) -> Array
    /// Returns a slice of the input array consisting of `width` consecutive
    /// elements starting at `start`. If any selected index is out-of-bounds,
    /// the last element of the array is used for that position (XLS semantics).
    ArraySlice {
        array: NodeRef,
        start: NodeRef,
        width: usize,
    },
    TupleIndex {
        tuple: NodeRef,
        index: usize,
    },
    Binop(Binop, NodeRef, NodeRef),
    Unop(Unop, NodeRef),
    Literal(IrValue),
    SignExt {
        arg: NodeRef,
        new_bit_count: usize,
    },
    ZeroExt {
        arg: NodeRef,
        new_bit_count: usize,
    },
    ArrayUpdate {
        array: NodeRef,
        value: NodeRef,
        indices: Vec<NodeRef>,
        assumed_in_bounds: bool,
    },
    ArrayIndex {
        array: NodeRef,
        indices: Vec<NodeRef>,
        assumed_in_bounds: bool,
    },
    DynamicBitSlice {
        arg: NodeRef,
        start: NodeRef,
        width: usize,
    },
    BitSlice {
        arg: NodeRef,
        start: usize,
        width: usize,
    },
    BitSliceUpdate {
        arg: NodeRef,
        start: NodeRef,
        update_value: NodeRef,
    },
    Assert {
        token: NodeRef,
        activate: NodeRef,
        message: String,
        label: String,
    },
    Trace {
        token: NodeRef,
        activated: NodeRef,
        format: String,
        operands: Vec<NodeRef>,
    },
    AfterAll(Vec<NodeRef>),
    Nary(NaryOp, Vec<NodeRef>),
    Invoke {
        to_apply: String,
        operands: Vec<NodeRef>,
    },
    PrioritySel {
        selector: NodeRef,
        cases: Vec<NodeRef>,
        default: Option<NodeRef>,
    },
    OneHotSel {
        selector: NodeRef,
        cases: Vec<NodeRef>,
    },
    OneHot {
        arg: NodeRef,
        lsb_prio: bool,
    },
    Sel {
        selector: NodeRef,
        cases: Vec<NodeRef>,
        default: Option<NodeRef>,
    },
    Cover {
        predicate: NodeRef,
        label: String,
    },
    Decode {
        arg: NodeRef,
        width: usize,
    },
    Encode {
        arg: NodeRef,
    },
    // Counted for loop: initializes with `init`, runs for `trip_count` times with stride
    // `stride`, applying the function `body` each iteration.
    CountedFor {
        init: NodeRef,
        trip_count: usize,
        stride: usize,
        body: String,
        /// Additional invariant operands passed to the loop body after the
        /// induction variable and loop-carry.
        ///
        /// If there are N entries here, the callee body is invoked with
        /// signature like: body(i, acc, inv0, ..., inv{N-1}).
        invariant_args: Vec<NodeRef>,
    },
}

impl NodePayload {
    pub fn get_operator(&self) -> &str {
        match self {
            NodePayload::Nil => "nil",
            NodePayload::GetParam(_) => "get_param",
            NodePayload::Tuple(_) => "tuple",
            NodePayload::Array(_) => "array",
            NodePayload::ArraySlice { .. } => "array_slice",
            NodePayload::TupleIndex { .. } => "tuple_index",
            NodePayload::Binop(op, _, _) => binop_to_operator(*op),
            NodePayload::Unop(op, _) => unop_to_operator(*op),
            NodePayload::Literal(_) => "literal",
            NodePayload::SignExt { .. } => "sign_ext",
            NodePayload::ZeroExt { .. } => "zero_ext",
            NodePayload::ArrayUpdate { .. } => "array_update",
            NodePayload::ArrayIndex { .. } => "array_index",
            NodePayload::DynamicBitSlice { .. } => "dynamic_bit_slice",
            NodePayload::BitSlice { .. } => "bit_slice",
            NodePayload::BitSliceUpdate { .. } => "bit_slice_update",
            NodePayload::Assert { .. } => "assert",
            NodePayload::Trace { .. } => "trace",
            NodePayload::AfterAll(_) => "after_all",
            NodePayload::Nary(op, _) => nary_op_to_operator(*op),
            NodePayload::Invoke { .. } => "invoke",
            NodePayload::PrioritySel { .. } => "priority_sel",
            NodePayload::OneHotSel { .. } => "one_hot_sel",
            NodePayload::OneHot { .. } => "one_hot",
            NodePayload::Sel { .. } => "sel",
            NodePayload::Cover { .. } => "cover",
            NodePayload::Decode { .. } => "decode",
            NodePayload::Encode { .. } => "encode",
            NodePayload::CountedFor { .. } => "counted_for",
        }
    }

    pub fn validate(&self, f: &Fn) -> Result<(), String> {
        match self {
            NodePayload::Nil => Ok(()),
            NodePayload::GetParam(_) => Ok(()),
            NodePayload::Tuple(_) => Ok(()),
            NodePayload::Array(_) => Ok(()),
            NodePayload::TupleIndex { tuple, index } => {
                if tuple.index >= f.nodes.len() {
                    return Err(format!(
                        "tuple index {} is out of bounds for tuple of length {}",
                        index,
                        f.nodes.len()
                    ));
                }
                Ok(())
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let case_ty = f.get_node_ty(cases[0]);
                for case in cases.iter() {
                    if f.get_node_ty(*case) != case_ty {
                        return Err(format!("all cases must be the same type"));
                    }
                }
                if let Some(default) = default {
                    if f.get_node_ty(*default) != case_ty {
                        return Err(format!("default must be the same type as the cases"));
                    }
                }
                Ok(())
            }
            NodePayload::OneHotSel { selector, cases } => {
                // Validate that the cases all are the same type.
                let case_ty = f.get_node_ty(cases[0]);
                for case in cases.iter() {
                    if f.get_node_ty(*case) != case_ty {
                        return Err(format!("all cases must be the same type"));
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    pub fn to_string(
        &self,
        f: &Fn,
        id: usize,
        pos: Option<Vec<(usize, usize, usize)>>,
    ) -> Option<String> {
        let get_name = |node_ref: NodeRef| -> String {
            let node = f.get_node(node_ref);
            match node.payload {
                NodePayload::GetParam(_) => {
                    node.name.clone().expect("GetParam node should have a name")
                }
                _ => {
                    if let Some(ref name) = node.name {
                        name.clone()
                    } else {
                        format!("{}.{}", node.payload.get_operator(), node.text_id)
                    }
                }
            }
        };
        let result = match self {
            NodePayload::Tuple(nodes) => {
                if nodes.is_empty() {
                    format!("tuple(id={})", id)
                } else {
                    format!(
                        "tuple({}, id={})",
                        nodes
                            .iter()
                            .map(|n| get_name(*n))
                            .collect::<Vec<String>>()
                            .join(", "),
                        id
                    )
                }
            }
            NodePayload::Array(nodes) => format!(
                "array({}, id={})",
                nodes
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", "),
                id
            ),
            NodePayload::ArraySlice {
                array,
                start,
                width,
            } => {
                format!(
                    "array_slice({}, {}, width={}, id={})",
                    get_name(*array),
                    get_name(*start),
                    width,
                    id
                )
            }
            NodePayload::TupleIndex { tuple, index } => {
                format!(
                    "tuple_index({}, index={}, id={})",
                    get_name(*tuple),
                    index,
                    id
                )
            }
            NodePayload::Binop(op, lhs, rhs) => format!(
                "{}({}, {}, id={})",
                binop_to_operator(*op),
                get_name(*lhs),
                get_name(*rhs),
                id
            ),
            NodePayload::Unop(op, arg) => {
                format!("{}({}, id={})", unop_to_operator(*op), get_name(*arg), id)
            }
            NodePayload::Literal(value) => {
                let value_str = value
                    .to_string_fmt_no_prefix(IrFormatPreference::Default)
                    .unwrap();
                format!("literal(value={}, id={})", value_str, id)
            }
            NodePayload::SignExt { arg, new_bit_count } => format!(
                "sign_ext({}, new_bit_count={}, id={})",
                get_name(*arg),
                new_bit_count,
                id
            ),
            NodePayload::ZeroExt { arg, new_bit_count } => format!(
                "zero_ext({}, new_bit_count={}, id={})",
                get_name(*arg),
                new_bit_count,
                id
            ),
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                assumed_in_bounds,
            } => {
                let idx_str = indices
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ");
                if *assumed_in_bounds {
                    format!(
                        "array_update({}, {}, indices=[{}], assumed_in_bounds=true, id={})",
                        get_name(*array),
                        get_name(*value),
                        idx_str,
                        id
                    )
                } else {
                    format!(
                        "array_update({}, {}, indices=[{}], id={})",
                        get_name(*array),
                        get_name(*value),
                        idx_str,
                        id
                    )
                }
            }
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => {
                let idx_str = indices
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ");
                if *assumed_in_bounds {
                    format!(
                        "array_index({}, indices=[{}], assumed_in_bounds=true, id={})",
                        get_name(*array),
                        idx_str,
                        id
                    )
                } else {
                    format!(
                        "array_index({}, indices=[{}], id={})",
                        get_name(*array),
                        idx_str,
                        id
                    )
                }
            }
            NodePayload::DynamicBitSlice { arg, start, width } => format!(
                "dynamic_bit_slice({}, {}, width={}, id={})",
                get_name(*arg),
                get_name(*start),
                width,
                id
            ),
            NodePayload::BitSlice { arg, start, width } => {
                format!(
                    "bit_slice({}, start={}, width={}, id={})",
                    get_name(*arg),
                    start,
                    width,
                    id
                )
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => format!(
                "bit_slice_update({}, {}, {}, id={})",
                get_name(*arg),
                get_name(*start),
                get_name(*update_value),
                id
            ),
            NodePayload::Assert {
                token,
                activate,
                message,
                label,
            } => {
                format!(
                    "assert({}, {}, message={:?}, label={:?}, id={})",
                    get_name(*token),
                    get_name(*activate),
                    message,
                    label,
                    id
                )
            }
            NodePayload::Trace {
                token,
                activated,
                format,
                operands,
            } => format!(
                "trace({}, {}, {}, {}, id={})",
                get_name(*token),
                get_name(*activated),
                format,
                operands
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", "),
                id
            ),
            NodePayload::AfterAll(nodes) => {
                if nodes.is_empty() {
                    format!("after_all(id={})", id)
                } else {
                    format!(
                        "after_all({}, id={})",
                        nodes
                            .iter()
                            .map(|n| get_name(*n))
                            .collect::<Vec<String>>()
                            .join(", "),
                        id
                    )
                }
            }
            NodePayload::Nary(op, nodes) => format!(
                "{}({}, id={})",
                nary_op_to_operator(*op),
                nodes
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", "),
                id
            ),
            NodePayload::Invoke { to_apply, operands } => {
                let operands_str = operands
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ");
                if operands.is_empty() {
                    format!("invoke(to_apply={}, id={})", to_apply, id)
                } else {
                    format!("invoke({}, to_apply={}, id={})", operands_str, to_apply, id)
                }
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let default_str = if let Some(default) = default {
                    format!(", default={}", get_name(*default))
                } else {
                    "".to_string()
                };
                format!(
                    "priority_sel({}, cases=[{}]{}, id={})",
                    get_name(*selector),
                    cases
                        .iter()
                        .map(|n| get_name(*n))
                        .collect::<Vec<String>>()
                        .join(", "),
                    default_str,
                    id
                )
            }
            NodePayload::OneHotSel { selector, cases } => format!(
                "one_hot_sel({}, cases=[{}], id={})",
                get_name(*selector),
                cases
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", "),
                id
            ),
            NodePayload::OneHot { arg, lsb_prio } => {
                format!(
                    "one_hot({}, lsb_prio={}, id={})",
                    get_name(*arg),
                    lsb_prio,
                    id
                )
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                let default_str = if let Some(default) = default {
                    format!(", default={}", get_name(*default))
                } else {
                    "".to_string()
                };
                format!(
                    "sel({}, cases=[{}]{}, id={})",
                    get_name(*selector),
                    cases
                        .iter()
                        .map(|n| get_name(*n))
                        .collect::<Vec<String>>()
                        .join(", "),
                    default_str,
                    id
                )
            }
            NodePayload::Cover { predicate, label } => {
                format!("cover({}, {}, id={})", get_name(*predicate), label, id)
            }
            NodePayload::Decode { arg, width } => {
                format!("decode({}, width={}, id={})", get_name(*arg), width, id)
            }
            NodePayload::Encode { arg } => {
                format!("encode({}, id={})", get_name(*arg), id)
            }
            NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
                invariant_args,
            } => {
                let inv_str = if invariant_args.is_empty() {
                    String::new()
                } else {
                    format!(
                        ", invariant_args=[{}]",
                        invariant_args
                            .iter()
                            .map(|n| get_name(*n))
                            .collect::<Vec<String>>()
                            .join(", ")
                    )
                };
                format!(
                    "counted_for({}, trip_count={}, stride={}, body={}{}, id={})",
                    get_name(*init),
                    trip_count,
                    stride,
                    body,
                    inv_str,
                    id
                )
            }
            NodePayload::GetParam(_) | NodePayload::Nil => return None,
        };
        Some(result)
    }
}

/// Returns a human-oriented textual identifier for a node reference.
///
/// - For `get_param` nodes, returns the parameter's name.
/// - For other nodes, returns the node's `name` if present, otherwise
///   `"<operator>.<text_id>"`.
pub fn node_textual_id(f: &Fn, nr: NodeRef) -> String {
    let node = f.get_node(nr);
    match node.payload {
        NodePayload::GetParam(_) => node.name.clone().expect("GetParam node should have a name"),
        _ => match &node.name {
            Some(n) => n.clone(),
            None => format!("{}.{}", node.payload.get_operator(), node.text_id),
        },
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    /// All nodes have known ids.
    pub text_id: usize,
    /// Some nodes also have names -- params must have names, other nodes
    /// optionally have names.
    pub name: Option<String>,
    pub ty: Type,
    pub payload: NodePayload,
    pub pos: Option<PosData>,
}

impl Node {
    pub fn to_string(&self, f: &Fn) -> Option<String> {
        match self.payload.to_string(f, self.text_id, None) {
            Some(result) => {
                let name_str = if let Some(name) = &self.name {
                    format!("{}", name)
                } else {
                    format!("{}.{}", self.payload.get_operator(), self.text_id)
                };
                let mut payload_str = result;
                if let Some(pos_list) = &self.pos {
                    if !pos_list.is_empty() {
                        // Format pos as: pos=[(a,b,c), (d,e,f)] with no spaces inside tuples
                        let items: Vec<String> = pos_list
                            .iter()
                            .map(|p| format!("({},{},{})", p.fileno, p.lineno, p.colno))
                            .collect();
                        let pos_attr = format!(", pos=[{}]", items.join(", "));
                        if let Some(idx) = payload_str.rfind(')') {
                            // Insert before the final ')'
                            payload_str = format!(
                                "{}{}{}",
                                &payload_str[..idx],
                                pos_attr,
                                &payload_str[idx..]
                            );
                        } else {
                            // Fallback: append
                            payload_str.push_str(&pos_attr);
                        }
                    }
                }
                Some(format!("{}: {} = {}", name_str, self.ty, payload_str))
            }
            None => None,
        }
    }

    pub fn to_signature_string(&self, f: &Fn) -> String {
        let operands_str = operands(&self.payload)
            .iter()
            .map(|o| f.get_node(*o).ty.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        let attrs_str = match &self.payload {
            NodePayload::Decode { width, .. } => format!(", width={}", width),
            NodePayload::BitSlice {
                start,
                width,
                arg: _,
            } => format!(", start={}, width={}", start, width),
            NodePayload::ArraySlice { width, .. } => format!(", width={}", width),
            NodePayload::SignExt { new_bit_count, .. }
            | NodePayload::ZeroExt { new_bit_count, .. } => {
                format!(", new_bit_count={}", new_bit_count)
            }
            _ => "".to_string(),
        };
        format!(
            "{}({}{}) -> {}",
            self.payload.get_operator(),
            operands_str,
            attrs_str,
            self.ty
        )
    }
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub id: ParamId,
}

#[derive(Debug, PartialEq)]
pub struct FunctionType {
    pub param_types: Vec<Type>,
    pub return_type: Type,
}

#[derive(Debug, Clone)]
pub struct Fn {
    pub name: String,
    pub params: Vec<Param>,
    pub ret_ty: Type,
    pub nodes: Vec<Node>,
    pub ret_node_ref: Option<NodeRef>,
    pub outer_attrs: Vec<String>,
    pub inner_attrs: Vec<String>,
}

impl Fn {
    pub fn get_type(&self) -> FunctionType {
        FunctionType {
            param_types: self.params.iter().map(|p| p.ty.clone()).collect(),
            return_type: self.ret_ty.clone(),
        }
    }

    pub fn node_refs(&self) -> Vec<NodeRef> {
        (0..self.nodes.len())
            .map(|i| NodeRef { index: i })
            .collect()
    }

    pub fn get_node(&self, node_ref: NodeRef) -> &Node {
        &self.nodes[node_ref.index]
    }

    pub fn get_node_ty(&self, node_ref: NodeRef) -> &Type {
        &self.get_node(node_ref).ty
    }

    pub fn get_node_mut(&mut self, node_ref: NodeRef) -> &mut Node {
        &mut self.nodes[node_ref.index]
    }

    /// Returns a new Fn with the given parameter names dropped.
    /// Returns Err if any dropped parameter is used in the function body.
    pub fn drop_params(mut self, params: &[String]) -> Result<Self, String> {
        let dropped: Vec<_> = self
            .params
            .iter()
            .filter(|p| params.contains(&p.name))
            .map(|p| p.id)
            .collect();
        self.params.retain(|p| !params.contains(&p.name));
        // Find all GetParam nodes for dropped params
        let mut offending = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if let ir::NodePayload::GetParam(param_id) = &node.payload {
                if dropped.contains(param_id) {
                    offending.push((i, *param_id));
                }
            }
        }
        // For each offending node, check if it is referenced by any other node
        let mut to_nil = Vec::new();
        for (idx, param_id) in offending {
            let node_ref = ir::NodeRef { index: idx };
            let is_used = self.nodes.iter().any(|n| {
                use ir::NodePayload::*;
                match &n.payload {
                    Tuple(nodes) | Array(nodes) => nodes.contains(&node_ref),
                    ArraySlice { array, start, .. } => array == &node_ref || start == &node_ref,
                    TupleIndex { tuple, .. } => tuple == &node_ref,
                    Binop(_, a, b) => a == &node_ref || b == &node_ref,
                    Unop(_, a) => a == &node_ref,
                    SignExt { arg, .. } | ZeroExt { arg, .. } => arg == &node_ref,
                    ArrayUpdate {
                        array,
                        value,
                        indices,
                        assumed_in_bounds: _,
                    } => array == &node_ref || value == &node_ref || indices.contains(&node_ref),
                    ArrayIndex { array, indices, .. } => {
                        array == &node_ref || indices.contains(&node_ref)
                    }
                    DynamicBitSlice { arg, start, .. } => arg == &node_ref || start == &node_ref,
                    BitSlice { arg, .. } => arg == &node_ref,
                    BitSliceUpdate {
                        arg,
                        start,
                        update_value,
                    } => arg == &node_ref || start == &node_ref || update_value == &node_ref,
                    Assert {
                        token, activate, ..
                    } => token == &node_ref || activate == &node_ref,
                    Trace {
                        token,
                        activated,
                        operands,
                        ..
                    } => {
                        token == &node_ref || activated == &node_ref || operands.contains(&node_ref)
                    }
                    AfterAll(nodes) => nodes.contains(&node_ref),
                    Nary(_, nodes) => nodes.contains(&node_ref),
                    Invoke { operands, .. } => operands.contains(&node_ref),
                    PrioritySel {
                        selector,
                        cases,
                        default,
                    } => {
                        selector == &node_ref
                            || cases.contains(&node_ref)
                            || default.map_or(false, |d| d == node_ref)
                    }
                    OneHotSel { selector, cases } => {
                        selector == &node_ref || cases.contains(&node_ref)
                    }
                    OneHot { arg, .. } => arg == &node_ref,
                    Sel {
                        selector,
                        cases,
                        default,
                    } => {
                        selector == &node_ref
                            || cases.contains(&node_ref)
                            || default.map_or(false, |d| d == node_ref)
                    }
                    Cover { predicate, .. } => predicate == &node_ref,
                    Decode { arg, .. } => arg == &node_ref,
                    Encode { arg } => arg == &node_ref,
                    _ => false,
                }
            });
            if is_used {
                return Err(format!(
                    "Dropped parameter with id {:?} is used in the function body (node {})!",
                    param_id, idx
                ));
            } else {
                to_nil.push(idx);
            }
        }
        // Clobber unused GetParam nodes with Nil
        for idx in to_nil {
            self.nodes[idx].payload = ir::NodePayload::Nil;
        }
        Ok(self)
    }
}

impl std::fmt::Display for Fn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", emit_fn(self, /* is_top= */ false))
    }
}

fn append_emitted_node_line(
    out: &mut String,
    func: &Fn,
    node_ref: NodeRef,
    comment: Option<String>,
) {
    let node = func.get_node(node_ref);
    let is_ret = if let Some(ret_node_ref) = func.ret_node_ref {
        ret_node_ref == node_ref
    } else {
        false
    };

    match &node.payload {
        NodePayload::GetParam(pid) if is_ret => {
            let name = node
                .name
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("<unnamed>");
            let mut line = String::new();
            line.push_str("  ret ");
            line.push_str(&format!(
                "{}: {} = param(name={}, id={})",
                name,
                node.ty,
                name,
                pid.get_wrapped_id()
            ));
            if let Some(comment) = comment {
                line.push_str("  // ");
                line.push_str(&comment);
            }
            out.push_str(&line);
            out.push('\n');
        }
        _ => {
            let Some(node_str) = node.to_string(func) else {
                return;
            };
            let mut line = String::new();
            line.push_str("  ");
            if is_ret {
                line.push_str("ret ");
            }
            line.push_str(&node_str);
            if let Some(comment) = comment {
                line.push_str("  // ");
                line.push_str(&comment);
            }
            out.push_str(&line);
            out.push('\n');
        }
    }
}

/// Emits a function as text, including outer/inner attributes and body.
pub fn emit_fn(func: &Fn, is_top: bool) -> String {
    let mut out = String::new();
    // Outer attributes first
    for attr in &func.outer_attrs {
        out.push_str(attr);
        out.push('\n');
    }

    // Signature line
    let params_str = func
        .params
        .iter()
        .map(|p| format!("{}: {} id={}", p.name, p.ty, p.id.get_wrapped_id()))
        .collect::<Vec<String>>()
        .join(", ");
    let return_type_str = func.ret_ty.to_string();
    if is_top {
        out.push_str("top ");
    }
    out.push_str(&format!(
        "fn {}({}) -> {} {{\n",
        func.name, params_str, return_type_str
    ));

    // Inner attributes after signature
    for attr in &func.inner_attrs {
        out.push_str("  ");
        out.push_str(attr);
        out.push('\n');
    }

    // Body nodes.
    for (i, _node) in func.nodes.iter().enumerate() {
        let node_ref = NodeRef { index: i };
        append_emitted_node_line(&mut out, func, node_ref, None);
    }

    out.push('}');
    out
}

/// Emits a function as text, including outer/inner attributes and body,
/// allowing an optional end-of-line comment per node.
///
/// The comment is produced by `comment_fn` given the function and the node ref.
pub fn emit_fn_with_node_comments_full<F>(func: &Fn, is_top: bool, mut comment_fn: F) -> String
where
    F: FnMut(&Fn, NodeRef) -> Option<String>,
{
    let mut out = String::new();
    // Outer attributes first.
    for attr in &func.outer_attrs {
        out.push_str(attr);
        out.push('\n');
    }

    // Signature line.
    let params_str = func
        .params
        .iter()
        .map(|p| format!("{}: {} id={}", p.name, p.ty, p.id.get_wrapped_id()))
        .collect::<Vec<String>>()
        .join(", ");
    let return_type_str = func.ret_ty.to_string();
    if is_top {
        out.push_str("top ");
    }
    out.push_str(&format!(
        "fn {}({}) -> {} {{\n",
        func.name, params_str, return_type_str
    ));

    // Inner attributes after signature.
    for attr in &func.inner_attrs {
        out.push_str("  ");
        out.push_str(attr);
        out.push('\n');
    }

    // Body nodes.
    for (i, _node) in func.nodes.iter().enumerate() {
        let node_ref = NodeRef { index: i };
        let comment = comment_fn(func, node_ref);
        append_emitted_node_line(&mut out, func, node_ref, comment);
    }

    out.push('}');
    out
}

/// Emits a package as text, allowing a targeted override of function emission.
///
/// If `override_fn` returns `Some(text)` for a given function member, that text
/// is used; otherwise the package uses the default function/block emission.
pub fn emit_package_with_fn_override<F>(pkg: &Package, mut override_fn: F) -> String
where
    F: FnMut(&Fn, bool) -> Option<String>,
{
    let mut out = String::new();
    out.push_str(&format!("package {}\n\n", pkg.name));

    let mut sorted_file_ids = pkg.file_table.id_to_path.keys().collect::<Vec<_>>();
    sorted_file_ids.sort();
    for file_id in sorted_file_ids {
        let path = pkg.file_table.id_to_path[file_id].as_str();
        out.push_str(&format!("file_number {} \"{}\"\n", file_id, path));
    }
    if !pkg.file_table.id_to_path.is_empty() {
        out.push('\n');
    }

    for (i, member) in pkg.members.iter().enumerate() {
        match member {
            PackageMember::Function(func) => {
                let is_top = match &pkg.top {
                    Some((top_name, MemberType::Function)) => func.name == top_name.as_str(),
                    _ => false,
                };
                let func_text = override_fn(func, is_top).unwrap_or_else(|| emit_fn(func, is_top));
                out.push_str(&func_text);
            }
            PackageMember::Block { func, port_info } => {
                let is_top = match &pkg.top {
                    Some((top_name, MemberType::Block)) => func.name == top_name.as_str(),
                    _ => false,
                };
                // Emit as a block using helper from the parser module.
                let block_text = ir_parser::emit_fn_as_block(func, None, Some(port_info), is_top);
                out.push_str(&block_text);
            }
        }
        if i < pkg.members.len().saturating_sub(1) {
            out.push_str("\n\n");
        } else {
            out.push('\n');
        }
    }

    out
}

/// Emits a function as text, allowing an optional end-of-line comment per node.
/// The comment is produced by `comment_fn` given the function and the node ref.
pub fn emit_fn_with_node_comments<F>(func: &Fn, mut comment_fn: F) -> String
where
    F: FnMut(&Fn, NodeRef) -> Option<String>,
{
    let params_str = func
        .params
        .iter()
        .map(|p| format!("{}: {} id={}", p.name, p.ty, p.id.get_wrapped_id()))
        .collect::<Vec<String>>()
        .join(", ");
    let return_type_str = func.ret_ty.to_string();
    let mut out = String::new();
    out.push_str(&format!(
        "fn {}({}) -> {} {{\n",
        func.name, params_str, return_type_str
    ));

    for (i, _node) in func.nodes.iter().enumerate() {
        let node_ref = NodeRef { index: i };
        let comment = comment_fn(func, node_ref);
        append_emitted_node_line(&mut out, func, node_ref, comment);
    }
    out.push('}');
    out
}

/// Convenience: Emits a function with a human-readable position comment per
/// node, if `pos` exists. The comment lists humanized positions (file:line:col)
/// space-separated.
pub fn emit_fn_with_human_pos_comments(func: &Fn, file_table: &FileTable) -> String {
    emit_fn_with_node_comments(func, |f, nr| {
        let node = f.get_node(nr);
        match &node.pos {
            Some(pos_list) if !pos_list.is_empty() => {
                let parts: Vec<String> = pos_list
                    .iter()
                    .filter_map(|p| p.to_human_string(file_table))
                    .collect();
                if parts.is_empty() {
                    None
                } else {
                    Some(parts.join(" "))
                }
            }
            _ => None,
        }
    })
}

#[derive(Debug, Clone)]
pub struct FileTable {
    pub id_to_path: HashMap<usize, String>,
}

impl FileTable {
    pub fn new() -> Self {
        Self {
            id_to_path: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pos {
    pub fileno: usize,
    pub lineno: usize,
    pub colno: usize,
}

impl Pos {
    pub fn to_human_string(&self, file_table: &FileTable) -> Option<String> {
        let path = file_table.id_to_path.get(&self.fileno)?;
        Some(format!("{}:{}:{}", path, self.lineno + 1, self.colno + 1))
    }
}

pub type PosData = Vec<Pos>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemberType {
    Function,
    Block,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub name: String,
    pub file_table: FileTable,
    pub members: Vec<PackageMember>,
    pub top: Option<(String, MemberType)>,
}

#[derive(Debug, Clone)]
pub enum PackageMember {
    Function(Fn),
    Block { func: Fn, port_info: BlockPortInfo },
}

#[derive(Debug, Clone)]
pub struct BlockPortInfo {
    pub clock_port_name: Option<String>,
    pub input_port_ids: std::collections::HashMap<String, usize>,
    pub output_port_ids: std::collections::HashMap<String, usize>,
    pub output_names: Vec<String>,
}

impl Package {
    /// Sets the package top to the given function name, if it exists.
    pub fn set_top_fn(&mut self, name: &str) -> Result<(), String> {
        let exists = self.members.iter().any(|m| match m {
            PackageMember::Function(f) => f.name == name,
            _ => false,
        });
        if exists {
            self.top = Some((name.to_string(), MemberType::Function));
            Ok(())
        } else {
            Err(format!("set_top_fn: function '{}' not found", name))
        }
    }

    /// Sets the package top to the given block name, if it exists.
    pub fn set_top_block(&mut self, name: &str) -> Result<(), String> {
        let exists = self.members.iter().any(|m| match m {
            PackageMember::Block { func, .. } => func.name == name,
            _ => false,
        });
        if exists {
            self.top = Some((name.to_string(), MemberType::Block));
            Ok(())
        } else {
            Err(format!("set_top_block: block '{}' not found", name))
        }
    }

    pub fn get_top_fn(&self) -> Option<&Fn> {
        match &self.top {
            Some((_, MemberType::Block)) => None,
            Some((name, MemberType::Function)) => self
                .members
                .iter()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    PackageMember::Block { .. } => None,
                })
                .find(|f| f.name == *name),
            None => self
                .members
                .iter()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    PackageMember::Block { .. } => None,
                })
                .next(),
        }
    }

    pub fn get_top_fn_mut(&mut self) -> Option<&mut Fn> {
        match &mut self.top {
            Some((_, MemberType::Block)) => None,
            Some((name, MemberType::Function)) => self
                .members
                .iter_mut()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    PackageMember::Block { .. } => None,
                })
                .find(|f| f.name == *name),
            None => self
                .members
                .iter_mut()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => Some(f),
                    PackageMember::Block { .. } => None,
                })
                .next(),
        }
    }

    pub fn get_top_block(&self) -> Option<&PackageMember> {
        match &self.top {
            Some((_, MemberType::Function)) => None,
            Some((name, MemberType::Block)) => self
                .members
                .iter()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => None,
                    PackageMember::Block { func, .. } => {
                        if func.name == *name {
                            Some(m)
                        } else {
                            None
                        }
                    }
                })
                .next(),
            None => self
                .members
                .iter()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => None,
                    PackageMember::Block { func, .. } => Some(m),
                })
                .next(),
        }
    }

    pub fn get_top_block_mut(&mut self) -> Option<&mut PackageMember> {
        match &mut self.top {
            Some((_, MemberType::Function)) => None,
            Some((name, MemberType::Block)) => self
                .members
                .iter_mut()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => None,
                    PackageMember::Block { func, .. } => {
                        if func.name == *name {
                            Some(m)
                        } else {
                            None
                        }
                    }
                })
                .next(),
            None => self
                .members
                .iter_mut()
                .filter_map(|m| match m {
                    PackageMember::Function(f) => None,
                    PackageMember::Block { func, .. } => Some(m),
                })
                .next(),
        }
    }
    pub fn get_fn(&self, name: &str) -> Option<&Fn> {
        self.members.iter().find_map(|m| match m {
            PackageMember::Function(f) if f.name == name => Some(f),
            PackageMember::Block { .. } => None,
            _ => None,
        })
    }

    pub fn get_fn_mut(&mut self, name: &str) -> Option<&mut Fn> {
        self.members.iter_mut().find_map(|m| match m {
            PackageMember::Function(f) if f.name == name => Some(f),
            PackageMember::Block { .. } => None,
            _ => None,
        })
    }

    /// Returns the block member (as a `PackageMember::Block`) whose internal
    /// function name equals `name`.
    pub fn get_block(&self, name: &str) -> Option<&PackageMember> {
        self.members.iter().find_map(|m| match m {
            PackageMember::Block { func, .. } if func.name == name => Some(m),
            _ => None,
        })
    }

    /// Mutable variant of `get_block`.
    pub fn get_block_mut(&mut self, name: &str) -> Option<&mut PackageMember> {
        self.members.iter_mut().find_map(|m| match m {
            PackageMember::Block { func, .. } if func.name == name => Some(m),
            _ => None,
        })
    }

    pub fn for_each_fn_mut<F: FnMut(&mut Fn)>(&mut self, mut f: F) {
        for m in self.members.iter_mut() {
            match m {
                PackageMember::Function(func) => f(func),
                PackageMember::Block { func, .. } => f(func),
            }
        }
    }

    /// Returns the function type (parameter types and return type) for the
    /// function named `name`, if present in the package.
    pub fn get_fn_type(&self, name: &str) -> Option<FunctionType> {
        self.get_fn(name).map(|f| f.get_type())
    }

    /// Replaces a `PackageMember::Block` whose internal function name equals
    /// `name` with `new_block`. Returns an error if no such block exists, or if
    /// `new_block` is not a block.
    pub fn replace_block(&mut self, name: &str, new_block: PackageMember) -> Result<(), String> {
        match &new_block {
            PackageMember::Block { .. } => {}
            _ => return Err("replace_block requires a Block package member".to_string()),
        }
        if let Some((idx, _)) = self.members.iter().enumerate().find(|(_, m)| match m {
            PackageMember::Block { func, .. } => func.name == name,
            _ => false,
        }) {
            self.members[idx] = new_block;
            Ok(())
        } else {
            Err(format!("replace_block: block '{}' not found", name))
        }
    }
}

impl std::fmt::Display for Package {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Delegate to the canonical package emitter so we don't duplicate the
        // `file_number`/member formatting logic in multiple places.
        write!(
            f,
            "{}",
            emit_package_with_fn_override(self, |_func, _is_top| None)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser;
    use crate::ir_utils::operands;

    use pretty_assertions::assert_eq;

    // -- Helpers
    fn assert_round_trip_fn(ir_text: &str) {
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();
        assert_eq!(ir_fn.to_string(), ir_text);
    }

    fn assert_round_trip_pkg(pkg_text: &str) {
        let mut parser = ir_parser::Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        assert_eq!(pkg.to_string(), pkg_text);
    }

    #[test]
    fn params_are_dense_node_refs() {
        let ir_text = r#"fn add(x: bits[8] id=10, y: bits[8] id=20) -> bits[8] {
  ret add.42: bits[8] = add(x, y, id=42)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();

        assert_eq!(ir_fn.params.len(), 2);
        assert_eq!(ir_fn.nodes.len(), 4);

        assert!(matches!(
            ir_fn.get_node(NodeRef { index: 0 }).payload,
            NodePayload::Nil
        ));

        for (ordinal, param) in ir_fn.params.iter().enumerate() {
            let node_ref = NodeRef { index: ordinal + 1 };
            let node = ir_fn.get_node(node_ref);
            assert_eq!(node.name.as_deref(), Some(param.name.as_str()));
            assert_eq!(node.ty, param.ty);
            assert_eq!(node.payload, NodePayload::GetParam(param.id));
            assert!(
                operands(&node.payload).is_empty(),
                "GetParam nodes are leaves"
            );
        }

        let add_node_ref = NodeRef {
            index: ir_fn.nodes.len() - 1,
        };
        match ir_fn.get_node(add_node_ref).payload {
            NodePayload::Binop(_, lhs, rhs) => {
                assert_eq!(lhs, NodeRef { index: 1 });
                assert_eq!(rhs, NodeRef { index: 2 });
            }
            ref payload => panic!("expected binop payload, got {:?}", payload),
        }
    }

    #[test]
    fn returning_parameter_uses_explicit_get_param_node() {
        let ir_text = r#"fn passthrough(x: bits[16] id=7) -> bits[16] {
  ret x: bits[16] = param(name=x, id=7)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();

        // With de-duplication of GetParam nodes, only the reserved zero node and
        // a single GetParam remain; the return refers to that same node.
        assert_eq!(ir_fn.nodes.len(), 2);
        assert_eq!(ir_fn.ret_node_ref, Some(NodeRef { index: 1 }));

        let param_node = ir_fn.get_node(NodeRef { index: 1 });
        assert!(matches!(
            param_node.payload,
            NodePayload::GetParam(pid) if pid.get_wrapped_id() == 7
        ));

        let ret_node = ir_fn.get_node(ir_fn.ret_node_ref.unwrap());
        assert!(matches!(
            ret_node.payload,
            NodePayload::GetParam(pid) if pid.get_wrapped_id() == 7
        ));

        assert_eq!(ir_fn.to_string(), ir_text);
    }

    #[test]
    fn test_round_trip_and_gate_ir() {
        let ir_text = r#"fn do_and(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#;
        assert_round_trip_fn(ir_text);
    }

    #[test]
    fn test_round_trip_multidim_array_ir() {
        let ir_text = r#"fn f() -> bits[32][2][3][1] {
  ret literal.1: bits[32][2][3][1] = literal(value=[[[0, 1], [2, 3], [4, 5]]], id=1)
}"#;
        assert_round_trip_fn(ir_text);
    }
    #[test]
    fn test_invoke_round_trip() {
        let ir_text = r#"fn f(x: bits[8] id=1) -> bits[8] {
  ret r: bits[8] = invoke(x, to_apply=g, id=2)
}"#;
        assert_round_trip_fn(ir_text);
    }

    #[test]
    fn test_invoke_no_operands_round_trip() {
        let ir_text = r#"fn f() -> bits[8] {
  ret r: bits[8] = invoke(to_apply=g, id=1)
}"#;
        assert_round_trip_fn(ir_text);
    }

    #[test]
    fn test_round_trip_nil_tuple_return_formats_as_tuple_id() {
        let ir_text = r#"fn unit() -> () {
  ret tuple.1: () = tuple(id=1)
}"#;
        assert_round_trip_fn(ir_text);
    }
}
