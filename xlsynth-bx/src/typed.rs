// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use xlsynth_pir::{IrBits, IrValue, ir_value_utils::ir_bits_to_usize};

use crate::ast::{BinaryOp, ExternBlock, ExternFn, Param, StructDef, Type, UnaryOp};
use crate::dslx_import::{ImportedCall, ResolvedImports};
use TypedExprKind as K;

/// A module with concrete signatures, single drivers, and an acyclic instance
/// hierarchy.
#[derive(Clone, Debug)]
pub struct CheckedModule {
    pub(crate) structs: Vec<StructDef>,
    pub(crate) externs: Vec<ExternFn>,
    pub(crate) extern_blocks: Vec<ExternBlock>,
    pub(crate) imports: ResolvedImports,
    pub(crate) blocks: Vec<CheckedBlock>,
}

impl CheckedModule {
    pub fn structs(&self) -> &[StructDef] {
        &self.structs
    }
    pub fn externs(&self) -> &[ExternFn] {
        &self.externs
    }
    /// Clocked block interfaces that must be supplied by the linked IR package.
    pub fn extern_blocks(&self) -> &[ExternBlock] {
        &self.extern_blocks
    }
    /// Concrete DSLX specializations, indexed by
    /// [`TypedExprKind::ImportedCall`].
    pub fn imported_calls(&self) -> &[ImportedCall] {
        &self.imports.calls
    }
    /// Blocks in dependency order, with child definitions preceding their
    /// parents.
    pub fn blocks(&self) -> &[CheckedBlock] {
        &self.blocks
    }
}

/// Ports, static hardware declarations, and connections of a type-checked
/// block.
#[derive(Clone, Debug)]
pub struct CheckedBlock {
    pub name: String,
    pub public: bool,
    pub clock: String,
    pub reset: Option<String>,
    pub params: Vec<Param>,
    pub result: Type,
    pub statements: Vec<CheckedStmt>,
    pub output: TypedExpr,
}

/// A structural declaration or a single driver with its complete enable
/// condition.
#[derive(Clone, Debug)]
pub enum CheckedStmt {
    Reg {
        name: String,
        ty: Type,
        reset: Option<IrValue>,
    },
    Inst {
        name: String,
        block: String,
    },
    Let {
        name: String,
        value: TypedExpr,
    },
    Write {
        name: String,
        value: TypedExpr,
    },
    If {
        condition: TypedExpr,
        body: Vec<CheckedStmt>,
    },
    Connect {
        instance: String,
        port: String,
        value: TypedExpr,
    },
}

/// An expression whose type retains source-level signedness and aggregate
/// names.
#[derive(Clone, Debug)]
pub struct TypedExpr {
    pub ty: Type,
    pub kind: TypedExprKind,
}

/// Resolved operations, including field projections in declaration order.
#[derive(Clone, Debug)]
pub enum TypedExprKind {
    Name(String),
    Literal(IrValue),
    Tuple(Vec<TypedExpr>),
    Array(Vec<TypedExpr>),
    Index {
        value: Box<TypedExpr>,
        index: Box<TypedExpr>,
    },
    TupleIndex {
        value: Box<TypedExpr>,
        index: usize,
    },
    Call {
        name: String,
        args: Vec<TypedExpr>,
    },
    /// A concrete imported DSLX function specialization.
    ImportedCall {
        /// Index of the specialization in [`CheckedModule::imported_calls`].
        id: usize,
        args: Vec<TypedExpr>,
    },
    Unary {
        op: UnaryOp,
        value: Box<TypedExpr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<TypedExpr>,
        right: Box<TypedExpr>,
    },
    Cast {
        value: Box<TypedExpr>,
    },
    If {
        condition: Box<TypedExpr>,
        then_value: Box<TypedExpr>,
        else_value: Box<TypedExpr>,
    },
    InstanceOutput {
        instance: String,
        port: String,
    },
}

pub(super) fn bits(ty: &Type) -> Option<(bool, usize)> {
    match ty {
        Type::Bool => Some((false, 1)),
        Type::Bits { signed, width } => Some((*signed, *width)),
        Type::Tuple(_) | Type::Named(_) | Type::Array { .. } => None,
    }
}

impl TypedExpr {
    /// Evaluate register reset constants, allowing prior constant let bindings.
    pub(super) fn constant(&self, names: &BTreeMap<String, IrValue>) -> Option<IrValue> {
        match &self.kind {
            K::Literal(v) => Some(v.clone()),
            K::Name(name) => names.get(name).cloned(),
            K::Tuple(fields) => Some(IrValue::Tuple(
                fields
                    .iter()
                    .map(|f| f.constant(names))
                    .collect::<Option<_>>()?,
            )),
            K::Array(elements) => {
                let elements = elements
                    .iter()
                    .map(|e| e.constant(names))
                    .collect::<Option<Vec<_>>>()?;
                IrValue::make_array(&elements).ok()
            }
            K::Index { value, index } => {
                let array = value.constant(names)?;
                let last = array.get_element_count().ok()?.checked_sub(1)?;
                let i = ir_bits_to_usize(index.constant(names)?.as_bits().ok()?).unwrap_or(last);
                array.get_element(i.min(last)).ok()
            }
            K::TupleIndex { value, index } => value.constant(names)?.get_element(*index).ok(),
            K::Call { .. } | K::ImportedCall { .. } | K::InstanceOutput { .. } => None,
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                if condition.constant(names)?.to_bool().ok()? {
                    then_value.constant(names)
                } else {
                    else_value.constant(names)
                }
            }
            K::Cast { value } => {
                let source = value.constant(names)?.to_bits().ok()?;
                let fill = bits(&value.ty)?.0 && source.is_negative();
                let width = bits(&self.ty)?.1;
                Some(IrValue::Bits(IrBits::from_lsb_fn(width, |i| {
                    source.get_bit(i).unwrap_or(fill)
                })))
            }
            K::Unary { op, value } => {
                let v = value.constant(names)?.to_bits().ok()?;
                match op {
                    UnaryOp::Not => Some(IrValue::Bits(v.not())),
                    UnaryOp::Neg => Some(IrValue::Bits(v.negate())),
                }
            }
            K::Binary { op, left, right } => {
                let lhs = left.constant(names)?;
                let rhs = right.constant(names)?;
                if matches!(op, BinaryOp::Eq | BinaryOp::Ne) {
                    return Some(IrValue::bool((lhs == rhs) == (*op == BinaryOp::Eq)));
                }
                let a = lhs.as_bits().ok()?;
                let b = rhs.as_bits().ok()?;
                let (signed, width) = bits(&left.ty)?;
                let result = match op {
                    BinaryOp::Add => a.add(b),
                    BinaryOp::Sub => a.sub(b),
                    BinaryOp::Mul => a.umul(b).width_slice(0, width as i64),
                    BinaryOp::Div if signed => a.sdiv(b),
                    BinaryOp::Div => a.udiv(b),
                    BinaryOp::Mod if signed => a.smod(b),
                    BinaryOp::Mod => a.umod(b),
                    BinaryOp::And | BinaryOp::LogicalAnd => a.and(b),
                    BinaryOp::Or | BinaryOp::LogicalOr => a.or(b),
                    BinaryOp::Xor => a.xor(b),
                    BinaryOp::Lt => IrBits::bool(if signed { a.slt(b) } else { a.ult(b) }),
                    BinaryOp::Le => IrBits::bool(if signed { a.sle(b) } else { a.ule(b) }),
                    BinaryOp::Gt => IrBits::bool(if signed { a.sgt(b) } else { a.ugt(b) }),
                    BinaryOp::Ge => IrBits::bool(if signed { a.sge(b) } else { a.uge(b) }),
                    BinaryOp::Shl | BinaryOp::Shr => {
                        let shift = ir_bits_to_usize(b).unwrap_or(width).min(width) as i64;
                        if *op == BinaryOp::Shl {
                            a.shll(shift)
                        } else if signed {
                            a.shra(shift)
                        } else {
                            a.shrl(shift)
                        }
                    }
                    BinaryOp::Concat => IrBits::from_lsb_fn(bits(&self.ty)?.1, |i| {
                        if i < b.get_bit_count() {
                            b.get_bit(i).expect("index within right operand")
                        } else {
                            a.get_bit(i - b.get_bit_count())
                                .expect("index within left operand")
                        }
                    }),
                    BinaryOp::Eq | BinaryOp::Ne => {
                        unreachable!("equality evaluated on arbitrary values above")
                    }
                };
                Some(IrValue::Bits(result))
            }
        }
    }
}
