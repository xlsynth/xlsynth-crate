// SPDX-License-Identifier: Apache-2.0

//! Lean expressions for local values in a PIR function or block graph.

use std::fmt::Write as _;

use xlsynth_pir::ir::{Binop, NaryOp, NodeGraph, NodePayload as P, NodeRef, Type, Unop};
use xlsynth_pir::{IrFormatPreference, IrValue};

/// Named proof nodes may be applications, and readable equations may inline
/// entire expressions. Group either before applying a projection or operator.
pub(crate) enum LeanOperand {
    Atom(String),
    Expression(String),
}

impl LeanOperand {
    pub(crate) fn into_argument(self) -> String {
        match self {
            Self::Atom(name) => name,
            Self::Expression(term) => format!("({term})"),
        }
    }
}

#[derive(Debug)]
pub(crate) enum RenderError<E> {
    Unsupported,
    Invalid(String),
    Operand(E),
}

pub(crate) fn bit_width(ty: &Type) -> Result<usize, String> {
    ty.checked_bit_count()
        .ok_or_else(|| format!("bit width overflows for {ty}"))
}

fn packed_instance_size(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 1,
        Type::Tuple(fields) => {
            1 + fields
                .iter()
                .map(|t| 1 + packed_instance_size(t))
                .sum::<usize>()
        }
        Type::Array(a) => 1 + packed_instance_size(&a.element_type),
    }
}

/// Give the elaborator room for the deepest sequence of lets and for Packed
/// instances over nested products and vectors.
pub(crate) fn write_elaboration_limits<'a>(
    source: &mut String,
    graphs: impl IntoIterator<Item = &'a NodeGraph>,
) {
    let mut max_nodes = 0;
    let mut instance_size = 1;
    for graph in graphs {
        max_nodes = max_nodes.max(graph.nodes.len());
        for n in &graph.nodes {
            instance_size = instance_size.max(packed_instance_size(&n.ty));
        }
    }
    let depth = max_nodes.max(instance_size);
    if depth >= 256 {
        writeln!(
            source,
            "set_option maxRecDepth {}\n",
            depth.saturating_mul(4).saturating_add(512)
        )
        .unwrap();
    }
    if instance_size >= 128 {
        writeln!(
            source,
            "set_option synthInstance.maxSize {}\n",
            instance_size.saturating_mul(2)
        )
        .unwrap();
    }
}

/// Preserve operand order while keeping the Lean expression depth logarithmic.
pub(crate) fn balanced(operands: &[String], prefix: &str, separator: &str) -> String {
    fn emit(out: &mut String, operands: &[String], prefix: &str, separator: &str) {
        match operands {
            [] => {}
            [operand] => out.push_str(operand),
            _ => {
                let (left, right) = operands.split_at(operands.len() / 2);
                out.push('(');
                out.push_str(prefix);
                emit(out, left, prefix, separator);
                out.push_str(separator);
                emit(out, right, prefix, separator);
                out.push(')');
            }
        }
    }

    let mut expression = String::new();
    emit(&mut expression, operands, prefix, separator);
    expression
}

fn tuple(fields: &[String]) -> String {
    if fields.is_empty() {
        "()".into()
    } else {
        format!("({}, ())", fields.join(", "))
    }
}

#[derive(Clone, Copy)]
pub(crate) enum LocalRenderer {
    Function,
    Block,
}

const IR: &str = "_root_.XlsynthLean.IR";

impl LocalRenderer {
    // Block fields retain source names, which may shadow library names in
    // dependent record declarations. Keep the public function source spelling.
    fn root(self) -> &'static str {
        match self {
            Self::Function => "",
            Self::Block => "L.",
        }
    }

    pub(crate) fn lean_type(self, ty: &Type) -> String {
        let root = self.root();
        match ty {
            Type::Bits(w) => format!("{root}BitVec {w}"),
            Type::Array(a) => format!(
                "{root}Vector ({}) {}",
                self.lean_type(&a.element_type),
                a.element_count
            ),
            Type::Tuple(fields) if !fields.is_empty() => format!(
                "({} × {root}Unit)",
                fields
                    .iter()
                    .map(|t| self.lean_type(t))
                    .collect::<Vec<_>>()
                    .join(" × ")
            ),
            Type::Tuple(_) | Type::Token => format!("{root}Unit"),
        }
    }

    /// `Array.toVector` avoids unfolding computed elements in a `#v` size
    /// proof. Explicit append lengths keep size checks local to each leaf.
    pub(crate) fn vector_literal(self, fields: &[String]) -> String {
        let root = self.root();
        if fields.len() <= 64 {
            let fields = fields.join(", ");
            return match self {
                Self::Function => format!("#[{fields}].toVector"),
                Self::Block => format!("(L.Array.toVector #[{fields}])"),
            };
        }
        let (left, right) = fields.split_at(fields.len() / 2);
        format!(
            "(@{root}Vector.append _ {} {} {} {})",
            left.len(),
            right.len(),
            self.vector_literal(left),
            self.vector_literal(right)
        )
    }

    pub(crate) fn literal(self, value: &IrValue) -> String {
        let fields = |xs: &[IrValue]| xs.iter().map(|x| self.literal(x)).collect::<Vec<_>>();
        match value {
            IrValue::Bits(bits) => format!(
                "{}BitVec.ofNat {} 0x{}",
                self.root(),
                bits.get_bit_count(),
                bits.to_string_fmt(IrFormatPreference::PlainHex, false)
            ),
            IrValue::Tuple(xs) => tuple(&fields(xs)),
            IrValue::Array(xs) => self.vector_literal(&fields(xs.elements())),
            IrValue::Token => "()".into(),
        }
    }

    /// Keep ordinary Lean syntax in public function models and name the library
    /// operations explicitly in block equations.
    fn infix(self, name: &str, symbol: &str, a: &str, b: &str) -> String {
        match self {
            Self::Function => format!("{a} {symbol} {b}"),
            Self::Block => format!("L.BitVec.{name} {a} {b}"),
        }
    }

    fn method(self, name: &str, a: &str, b: &str) -> String {
        match self {
            Self::Function => format!("{a}.{name} {b}"),
            Self::Block => format!("L.BitVec.{name} {a} {b}"),
        }
    }

    fn resize(self, name: &str, w: usize, a: &str) -> String {
        match self {
            Self::Function => format!("{a}.{name} {w}"),
            Self::Block => format!("L.BitVec.{name} {w} {a}"),
        }
    }

    fn complement(self, a: &str) -> String {
        match self {
            Self::Function => format!("~~~{a}"),
            Self::Block => format!("L.BitVec.not {a}"),
        }
    }

    fn to_nat(self, a: &str) -> String {
        match self {
            Self::Function => format!("{a}.toNat"),
            Self::Block => format!("(L.BitVec.toNat {a})"),
        }
    }

    fn limited_offset(self, a: &str, w: usize) -> String {
        let offset = self.to_nat(a);
        match self {
            Self::Function => format!("(min {offset} {w})"),
            Self::Block => format!("(L.Nat.min {offset} {w})"),
        }
    }

    /// The declared result width is significant for ordinary multiply and
    /// shifts.
    fn binary<E>(self, op: Binop, a: &str, b: &str, w: usize) -> Result<String, RenderError<E>> {
        let root = self.root();
        Ok(match op {
            Binop::Add => self.infix("add", "+", a, b),
            Binop::Sub => self.infix("sub", "-", a, b),
            Binop::Umul | Binop::Smul => {
                let resize = if op == Binop::Umul {
                    "setWidth"
                } else {
                    "signExtend"
                };
                let a = self.resize(resize, w, a);
                let b = self.resize(resize, w, b);
                match self {
                    Self::Function => format!("{a} * {b}"),
                    Self::Block => format!("L.BitVec.mul ({a}) ({b})"),
                }
            }
            Binop::Udiv => format!("{root}BitVec.smtUDiv {a} {b}"),
            Binop::Sdiv => format!("{IR}.sdiv {a} {b}"),
            Binop::Umod => format!("if {b} == 0 then 0 else {}", self.infix("umod", "%", a, b)),
            Binop::Smod => format!("if {b} == 0 then 0 else {}", self.method("srem", a, b)),
            Binop::Shll => self.infix("shiftLeft", "<<<", a, &self.limited_offset(b, w)),
            Binop::Shrl => self.infix("ushiftRight", ">>>", a, &self.limited_offset(b, w)),
            Binop::Shra => self.method("sshiftRight", a, &self.limited_offset(b, w)),
            Binop::Eq => format!("{root}BitVec.ofBool ({a} == {b})"),
            Binop::Ne => format!("{root}BitVec.ofBool ({a} != {b})"),
            Binop::Ult => format!("{root}BitVec.ofBool ({})", self.method("ult", a, b)),
            Binop::Ule => format!("{root}BitVec.ofBool ({})", self.method("ule", a, b)),
            Binop::Ugt => format!("{root}BitVec.ofBool ({})", self.method("ult", b, a)),
            Binop::Uge => format!("{root}BitVec.ofBool ({})", self.method("ule", b, a)),
            Binop::Slt => format!("{root}BitVec.ofBool ({})", self.method("slt", a, b)),
            Binop::Sle => format!("{root}BitVec.ofBool ({})", self.method("sle", a, b)),
            Binop::Sgt => format!("{root}BitVec.ofBool ({})", self.method("slt", b, a)),
            Binop::Sge => format!("{root}BitVec.ofBool ({})", self.method("sle", b, a)),
            Binop::Gate => {
                let default = match self {
                    Self::Function => "default",
                    Self::Block => "L.Inhabited.default",
                };
                format!("if {a} == 1 then {b} else {default}")
            }
            Binop::Smulp | Binop::Umulp => return Err(RenderError::Unsupported),
        })
    }

    /// Resolve operands only when used. The caller handles graph-specific
    /// bindings, effects and calls, and supplies diagnostic source context.
    pub(crate) fn local_rhs<E>(
        self,
        graph: &NodeGraph,
        at: NodeRef,
        operand: impl Fn(NodeRef) -> Result<LeanOperand, E>,
    ) -> Result<String, RenderError<E>> {
        let n = graph.get_node(at);
        let name = |r| {
            operand(r)
                .map(LeanOperand::into_argument)
                .map_err(RenderError::Operand)
        };
        let names = |rs: &[NodeRef]| rs.iter().copied().map(name).collect::<Result<Vec<_>, _>>();
        let width = |ty: &Type| bit_width(ty).map_err(RenderError::Invalid);
        let root = self.root();
        Ok(match &n.payload {
            P::Literal(v) => self.literal(v),
            P::Tuple(rs) => tuple(&names(rs)?),
            P::Array(rs) => self.vector_literal(&names(rs)?),
            P::ArrayConcat(rs) => match self {
                Self::Function => balanced(&names(rs)?, "", " ++ "),
                Self::Block => balanced(&names(rs)?, "L.Vector.append ", " "),
            },
            P::TupleIndex { tuple, index } => format!("{}{}.1", name(*tuple)?, ".2".repeat(*index)),
            P::ArrayIndex { array, indices, .. } => {
                indices.iter().try_fold(name(*array)?, |base, i| {
                    Ok(format!(
                        "({IR}.arrayIndex {base} {})",
                        self.to_nat(&name(*i)?)
                    ))
                })?
            }
            P::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } => {
                let mut expr = name(*value)?;
                for (depth, idx) in indices.iter().enumerate().rev() {
                    let base = if depth == 0 {
                        name(*array)?
                    } else {
                        format!("v{}", depth - 1)
                    };
                    let binder = if depth + 1 == indices.len() {
                        "_".to_owned()
                    } else {
                        format!("v{depth}")
                    };
                    expr = format!(
                        "{IR}.arrayModify {base} {} (fun {binder} => {expr})",
                        self.to_nat(&name(*idx)?)
                    );
                }
                expr
            }
            P::ArraySlice {
                array,
                start,
                width,
            } => format!(
                "{IR}.arraySlice {width} {} {}",
                name(*array)?,
                self.to_nat(&name(*start)?)
            ),
            P::Binop(op, a, b) => self.binary(*op, &name(*a)?, &name(*b)?, width(&n.ty)?)?,
            P::Unop(op, a) => {
                let a = name(*a)?;
                match op {
                    Unop::Identity => a,
                    Unop::Neg => match self {
                        Self::Function => format!("-{a}"),
                        Self::Block => format!("L.BitVec.neg {a}"),
                    },
                    Unop::Not => self.complement(&a),
                    Unop::Reverse => match self {
                        Self::Function => format!("{a}.reverse"),
                        Self::Block => format!("L.BitVec.reverse {a}"),
                    },
                    Unop::OrReduce => format!("{root}BitVec.ofBool ({a} != 0)"),
                    Unop::AndReduce => {
                        format!("{root}BitVec.ofBool ({a} == {root}BitVec.allOnes _)")
                    }
                    Unop::XorReduce => match self {
                        Self::Function => format!("{a}.cpop.setWidth 1"),
                        Self::Block => format!("L.BitVec.setWidth 1 (L.BitVec.cpop {a})"),
                    },
                }
            }
            P::Nary(NaryOp::Concat, rs) if rs.is_empty() => "0".to_owned(),
            P::Nary(op, rs) => {
                let (named, infix) = match op {
                    NaryOp::And | NaryOp::Nand => ("L.BitVec.and ", " &&& "),
                    NaryOp::Or | NaryOp::Nor => ("L.BitVec.or ", " ||| "),
                    NaryOp::Xor => ("L.BitVec.xor ", " ^^^ "),
                    NaryOp::Concat => ("L.BitVec.append ", " ++ "),
                };
                let args = match self {
                    Self::Function => balanced(&names(rs)?, "", infix),
                    Self::Block => balanced(&names(rs)?, named, " "),
                };
                if matches!(op, NaryOp::Nand | NaryOp::Nor) {
                    self.complement(&format!("({args})"))
                } else {
                    args
                }
            }
            P::SignExt { arg, new_bit_count } => {
                self.resize("signExtend", *new_bit_count, &name(*arg)?)
            }
            P::ZeroExt { arg, new_bit_count } => {
                self.resize("setWidth", *new_bit_count, &name(*arg)?)
            }
            P::BitSlice { arg, start, width } => {
                format!("{root}BitVec.extractLsb' {start} {width} {}", name(*arg)?)
            }
            P::DynamicBitSlice {
                arg,
                start,
                width: w,
            } => format!(
                "{root}BitVec.extractLsb' {} {w} {}",
                self.limited_offset(&name(*start)?, width(graph.get_node_ty(*arg))?),
                name(*arg)?
            ),
            P::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => format!(
                "{IR}.bitSliceUpdate {} {} {}",
                name(*arg)?,
                name(*start)?,
                name(*update_value)?
            ),
            P::Decode { arg, width } => {
                format!(
                    "{root}BitVec.twoPow {width} {}",
                    self.limited_offset(&name(*arg)?, *width)
                )
            }
            P::Encode { arg } => format!("{IR}.encode {} {}", width(&n.ty)?, name(*arg)?),
            P::OneHot { arg, lsb_prio } => {
                format!("{IR}.oneHot {lsb_prio} {}", name(*arg)?)
            }
            P::OneHotSel { selector, cases } => format!(
                "{IR}.oneHotSel {} #[{}]",
                name(*selector)?,
                names(cases)?.join(", ")
            ),
            P::PrioritySel {
                selector,
                cases,
                default,
            } => format!(
                "{IR}.prioritySel {} #[{}] {}",
                name(*selector)?,
                names(cases)?.join(", "),
                name(default.ok_or_else(|| RenderError::Invalid(
                    "priority_sel requires a default".into()
                ))?)?
            ),
            P::Sel {
                selector,
                cases,
                default,
            } => {
                let fallback = default
                    .or_else(|| cases.last().copied())
                    .ok_or_else(|| RenderError::Invalid("sel has no cases or default".into()))?;
                let cases = names(cases)?.join(", ");
                let index = self.to_nat(&name(*selector)?);
                let fallback = name(fallback)?;
                match self {
                    Self::Function => format!("#[{cases}].getD {index} {fallback}"),
                    Self::Block => format!("L.Array.getD #[{cases}] {index} {fallback}"),
                }
            }
            _ => return Err(RenderError::Unsupported),
        })
    }
}
