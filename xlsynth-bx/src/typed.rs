// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use xlsynth_pir::{
    IrBits, IrValue,
    ir_value_utils::{ir_bits_to_usize, ir_bits_to_usize_in_range},
};

use crate::ast::{BinaryOp, ExternBlock, ExternFn, Param, StructDef, Type, UnaryOp};
use crate::dslx_import::{ImportedCall, ResolvedImports};
use TypedExprKind as K;

/// A module with concrete signatures, single drivers, and an acyclic instance
/// hierarchy.
#[derive(Clone, Debug)]
pub struct CheckedModule {
    pub(crate) structs: Vec<StructDef>,
    pub(crate) externs: Vec<ExternFn>,
    pub(crate) functions: Vec<CheckedFunction>,
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
    /// Local functions in dependency order, with callees preceding callers.
    pub fn functions(&self) -> &[CheckedFunction] {
        &self.functions
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

/// A pure function whose bindings and return value have concrete BX types.
#[derive(Clone, Debug)]
pub struct CheckedFunction {
    pub name: String,
    pub public: bool,
    pub params: Vec<Param>,
    pub result: Type,
    pub bindings: Vec<CheckedBinding>,
    pub output: TypedExpr,
}

/// A checked function binding in evaluation order, including unrolled lets.
#[derive(Clone, Debug)]
pub struct CheckedBinding {
    pub name: String,
    pub value: TypedExpr,
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
    ArrayUpdate {
        array: Box<TypedExpr>,
        index: Box<TypedExpr>,
        replacement: Box<TypedExpr>,
    },
    WidthSlice {
        value: Box<TypedExpr>,
        start: Box<TypedExpr>,
    },
    TupleIndex {
        value: Box<TypedExpr>,
        index: usize,
    },
    /// Project one field of a shared tuple without copying the complete tuple
    /// type and expression into every projection of a destructuring pattern.
    BindingTupleIndex {
        name: String,
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
    /// A DSLX type dimension must have constant inputs in every branch, even
    /// when its value could be folded by selecting just one branch.
    pub(super) fn has_constant_inputs(&self, names: &BTreeSet<String>) -> bool {
        match &self.kind {
            K::Name(name) | K::BindingTupleIndex { name, .. } => names.contains(name),
            K::Literal(_) => true,
            K::InstanceOutput { .. } => false,
            K::Tuple(items)
            | K::Array(items)
            | K::Call { args: items, .. }
            | K::ImportedCall { args: items, .. } => {
                items.iter().all(|item| item.has_constant_inputs(names))
            }
            K::TupleIndex { value, .. } | K::Cast { value } | K::Unary { value, .. } => {
                value.has_constant_inputs(names)
            }
            K::Index {
                value: left,
                index: right,
            }
            | K::WidthSlice {
                value: left,
                start: right,
            }
            | K::Binary { left, right, .. } => {
                left.has_constant_inputs(names) && right.has_constant_inputs(names)
            }
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                condition.has_constant_inputs(names)
                    && then_value.has_constant_inputs(names)
                    && else_value.has_constant_inputs(names)
            }
            K::ArrayUpdate {
                array,
                index,
                replacement,
            } => {
                array.has_constant_inputs(names)
                    && index.has_constant_inputs(names)
                    && replacement.has_constant_inputs(names)
            }
        }
    }

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
            K::ArrayUpdate {
                array,
                index,
                replacement,
            } => {
                let IrValue::Array(array) = array.constant(names)? else {
                    return None;
                };
                // DSLX constexpr evaluation requires an in-bounds update.
                // Runtime IR handles an out-of-bounds index as a no-op.
                let i = ir_bits_to_usize_in_range(
                    index.constant(names)?.as_bits().ok()?,
                    array.elements().len(),
                )?;
                let replacement = replacement.constant(names)?;
                let mut elements = array.elements().to_vec();
                elements[i] = replacement;
                IrValue::make_array_typed(array.element_type().clone(), &elements).ok()
            }
            K::WidthSlice { value, start } => {
                let value = value.constant(names)?;
                let source = value.as_bits().ok()?;
                let start = start.constant(names)?;
                let width = bits(&self.ty)?.1;
                let slice = match ir_bits_to_usize_in_range(
                    start.as_bits().ok()?,
                    source.get_bit_count(),
                ) {
                    Some(offset) => source.width_slice(offset as i64, width as i64),
                    None => IrBits::zero(width),
                };
                Some(IrValue::Bits(slice))
            }
            K::TupleIndex { value, index } => value.constant(names)?.get_element(*index).ok(),
            K::BindingTupleIndex { name, index } => names.get(name)?.get_element(*index).ok(),
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
