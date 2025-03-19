// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use std::collections::{hash_map::OccupiedError, HashMap};

use xlsynth::{ir_value::IrFormatPreference, IrValue};

use crate::ir_utils::operands;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayTypeData {
    pub element_type: Box<Type>,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Token,
    Bits(usize),
    Tuple(Vec<Box<Type>>),
    Array(ArrayTypeData),
}

/// Represents an interval of the form `[start, limit)` i.e. inclusive start exclusive limit.
pub struct StartAndLimit {
    pub start: usize,
    pub limit: usize,
}

impl Type {
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

    /// Returns the start and limit bits for our bitwise representation of a tuple access at the
    /// given index.
    ///
    /// E.g. consider `tuple(a, b, c)`, we represent this in a bit vector as:
    /// `a_msb, ..., a_lsb, b_msb, ..., b_lsb, c_msb, ..., c_lsb` where c_lsb is the least
    /// significant bit of the overall bit vector. That means to access index `i` we have to slice
    /// out all the members that come after it; e.g. if we want to access b we have to skip `c`
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

    Or,
    Xor,
    Nand,

    Shll,
    Shrl,

    ArrayConcat,
    Smulp,
    Umulp,

    Eq,
    Ne,

    Uge,
    Ugt,
    Ult,
    Ule,

    Sgt,
    Sge,
    Slt,
    Sle,
    Sub,
    Umul,
    Gate,
    Sdiv,
}

pub fn operator_to_binop(operator: &str) -> Option<Binop> {
    match operator {
        "shll" => Some(Binop::Shll),
        "shrl" => Some(Binop::Shrl),
        "add" => Some(Binop::Add),
        "array_concat" => Some(Binop::ArrayConcat),
        "smulp" => Some(Binop::Smulp),
        "umulp" => Some(Binop::Umulp),
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
        "sub" => Some(Binop::Sub),
        "umul" => Some(Binop::Umul),
        "or" => Some(Binop::Or),
        "xor" => Some(Binop::Xor),
        "gate" => Some(Binop::Gate),
        "sdiv" => Some(Binop::Sdiv),
        // boolean operations
        "nand" => Some(Binop::Nand),
        _ => None,
    }
}

pub fn binop_to_operator(binop: Binop) -> &'static str {
    match binop {
        Binop::Add => "add",
        Binop::Or => "or",
        Binop::Xor => "xor",
        Binop::Nand => "nand",
        Binop::Shll => "shll",
        Binop::Shrl => "shrl",
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
        Binop::Gate => "gate",
        Binop::Sdiv => "sdiv",
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unop {
    Neg,
    Not,
    Identity,
    Reverse,
    Encode,
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
        "encode" => Some(Unop::Encode),
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
        Unop::Encode => "encode",
        Unop::OrReduce => "or_reduce",
        Unop::AndReduce => "and_reduce",
        Unop::XorReduce => "xor_reduce",
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum NaryOp {
    And,
    Nor,
    Concat,
}

pub fn operator_to_nary_op(operator: &str) -> Option<NaryOp> {
    match operator {
        "and" => Some(NaryOp::And),
        "nor" => Some(NaryOp::Nor),
        "concat" => Some(NaryOp::Concat),
        _ => None,
    }
}

pub fn nary_op_to_operator(nary_op: NaryOp) -> &'static str {
    match nary_op {
        NaryOp::And => "and",
        NaryOp::Nor => "nor",
        NaryOp::Concat => "concat",
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct NodeRef {
    pub index: usize,
}

#[derive(Debug, PartialEq)]
pub enum NodePayload {
    Nil,
    GetParam(usize),
    Tuple(Vec<NodeRef>),
    Array(Vec<NodeRef>),
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
    },
    ArrayIndex {
        array: NodeRef,
        indices: Vec<NodeRef>,
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
}

impl NodePayload {
    pub fn get_operator(&self) -> &str {
        match self {
            NodePayload::Nil => "nil",
            NodePayload::GetParam(_) => "get_param",
            NodePayload::Tuple(_) => "tuple",
            NodePayload::Array(_) => "array",
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
                NodePayload::GetParam(_) => node.name.clone().unwrap(),
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
            NodePayload::Tuple(nodes) => format!(
                "tuple({})",
                nodes
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            NodePayload::Array(nodes) => format!(
                "array({})",
                nodes
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
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
            } => format!(
                "array_update({}, {}, {})",
                get_name(*array),
                get_name(*value),
                indices
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            NodePayload::ArrayIndex { array, indices } => format!(
                "array_index({}, indices=[{}])",
                get_name(*array),
                indices
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            NodePayload::DynamicBitSlice { arg, start, width } => format!(
                "dynamic_bit_slice({}, {}, {})",
                get_name(*arg),
                get_name(*start),
                width
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
                "bit_slice_update({}, {}, {})",
                get_name(*arg),
                get_name(*start),
                get_name(*update_value)
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
                "trace({}, {}, {}, {})",
                get_name(*token),
                get_name(*activated),
                format,
                operands
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            NodePayload::AfterAll(nodes) => format!(
                "after_all({})",
                nodes
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
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
            NodePayload::Invoke { to_apply, operands } => format!(
                "invoke({}, {})",
                to_apply,
                operands
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
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
                    "priority_sel({}, cases=[{}]{})",
                    get_name(*selector),
                    cases
                        .iter()
                        .map(|n| get_name(*n))
                        .collect::<Vec<String>>()
                        .join(", "),
                    default_str
                )
            }
            NodePayload::OneHotSel { selector, cases } => format!(
                "one_hot_sel({}, cases=[{}])",
                get_name(*selector),
                cases
                    .iter()
                    .map(|n| get_name(*n))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            NodePayload::OneHot { arg, lsb_prio } => {
                format!("one_hot({}, lsb_prio={})", get_name(*arg), lsb_prio)
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
                    "sel({}, cases=[{}]{})",
                    get_name(*selector),
                    cases
                        .iter()
                        .map(|n| get_name(*n))
                        .collect::<Vec<String>>()
                        .join(", "),
                    default_str
                )
            }
            NodePayload::Cover { predicate, label } => {
                format!("cover({}, {})", get_name(*predicate), label)
            }
            NodePayload::Decode { arg, width } => {
                format!("decode({}, width={}, id={})", get_name(*arg), width, id)
            }
            NodePayload::GetParam(_) | NodePayload::Nil => return None,
        };
        Some(result)
    }
}

#[derive(Debug)]
pub struct Node {
    /// All nodes have known ids.
    pub text_id: usize,
    /// Some nodes also have names -- params must have names, other nodes optionally have names.
    pub name: Option<String>,
    pub ty: Type,
    pub payload: NodePayload,
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
                Some(format!("{}: {} = {}", name_str, self.ty, result))
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

#[derive(Debug)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub id: usize,
}

#[derive(Debug, PartialEq)]
pub struct FunctionType {
    pub param_types: Vec<Type>,
    pub return_type: Type,
}

#[derive(Debug)]
pub struct Fn {
    pub name: String,
    pub params: Vec<Param>,
    pub ret_ty: Type,
    pub nodes: Vec<Node>,
    pub ret_node_ref: Option<NodeRef>,
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

    /*
    pub fn validate(&self) -> Result<(), String> {
        for node in self.nodes.iter() {
            node.payload.validate(self)?;
        }
        Ok(())
    }
    */
}

impl std::fmt::Display for Fn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params_str = self
            .params
            .iter()
            .map(|p| format!("{}: {} id={}", p.name, p.ty, p.id))
            .collect::<Vec<String>>()
            .join(", ");
        let return_type_str = self.ret_ty.to_string();
        write!(
            f,
            "fn {}({}) -> {} {{\n",
            self.name, params_str, return_type_str
        )?;

        // Now that we've emitted the signature, emit the nodes in the function body.
        for (i, node) in self.nodes.iter().enumerate() {
            let node_ref = NodeRef { index: i };
            let ret_prefix = if let Some(ret_node_ref) = self.ret_node_ref
                && ret_node_ref == node_ref
            {
                "ret "
            } else {
                ""
            };

            // Note: some nodes don't need anything to be emitted, e.g. get param nodes.
            if let Some(node_str) = node.to_string(self) {
                write!(f, "  {}{}\n", ret_prefix, node_str)?;
            }
        }

        // Now we're done with the function.
        write!(f, "}}")
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct Package {
    pub name: String,
    pub file_table: FileTable,
    pub fns: Vec<Fn>,
    pub top_name: Option<String>,
}

impl Package {
    pub fn get_top(&self) -> Option<&Fn> {
        match &self.top_name {
            Some(name) => self.fns.iter().find(|f| f.name == *name),
            None => self.fns.first(),
        }
    }

    pub fn get_top_mut(&mut self) -> Option<&mut Fn> {
        match &mut self.top_name {
            Some(name) => self.fns.iter_mut().find(|f| f.name == *name),
            None => self.fns.first_mut(),
        }
    }
}

impl std::fmt::Display for Package {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "package {}\n\n", self.name)?;

        // Emit the file table.
        let mut sorted_file_ids = self.file_table.id_to_path.keys().collect::<Vec<_>>();
        sorted_file_ids.sort();
        for file_id in sorted_file_ids {
            let path = self.file_table.id_to_path[file_id].as_str();
            write!(f, "file_number {} \"{}\"\n", file_id, path)?;
        }
        if !self.file_table.id_to_path.is_empty() {
            write!(f, "\n")?;
        }

        for (i, func) in self.fns.iter().enumerate() {
            if let Some(top_name) = &self.top_name
                && func.name == top_name.as_str()
            {
                write!(f, "top {}", func)?;
            } else {
                write!(f, "{}", func)?;
            }
            if i < self.fns.len() - 1 {
                write!(f, "\n\n")?;
            } else {
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser;

    use pretty_assertions::assert_eq;

    #[test]
    fn test_round_trip_and_gate_ir() {
        let ir_text = "fn do_and(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();
        assert_eq!(ir_fn.to_string(), ir_text);
    }
}
