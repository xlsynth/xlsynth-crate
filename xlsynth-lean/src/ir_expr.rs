// SPDX-License-Identifier: Apache-2.0

//! Physical XLS value types and pure node expressions shared by Lean emitters.

use xlsynth_pir::ir::{Binop, NaryOp, Node, NodeGraph, NodePayload as P, NodeRef, Type, Unop};
use xlsynth_pir::{IrFormatPreference, IrValue};

use crate::LeanError;

pub(crate) fn width(ty: &Type) -> Result<usize, LeanError> {
    ty.checked_bit_count()
        .ok_or_else(|| LeanError::InvalidIr(format!("bit width overflows for {ty}")))
}

fn unsupported_type_reason(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Token => Some("token-valued type"),
        Type::Bits(_) => None,
        Type::Tuple(fields) => fields
            .iter()
            .find_map(|field| unsupported_type_reason(field)),
        Type::Array(a) if a.element_count == 0 => Some("zero-length array"),
        Type::Array(a) => unsupported_type_reason(&a.element_type),
    }
}

/// Check even unused nodes so an unsupported observation cannot silently
/// disappear.
pub(crate) fn check_supported(graph: &NodeGraph, node: &Node) -> Result<(), LeanError> {
    let reason = match &node.payload {
        P::Binop(Binop::Smulp | Binop::Umulp, ..) => "partial-product multiply",
        P::CountedFor { .. } => "loop",
        P::AfterAll(_) | P::Assert { .. } | P::Trace { .. } | P::Cover { .. } => "token or event",
        P::InputPort { .. }
        | P::OutputPort { .. }
        | P::InstantiationInput { .. }
        | P::InstantiationOutput { .. }
        | P::RegisterRead { .. }
        | P::RegisterWrite { .. } => "block operation",
        _ => match unsupported_type_reason(&node.ty) {
            Some(reason) => reason,
            None => return Ok(()),
        },
    };
    Err(LeanError::UnsupportedIr {
        function: graph.name.clone(),
        op: node.payload.get_operator().to_owned(),
        id: node.text_id,
        reason,
    })
}

/// Renders a verified, supported physical value type.
pub(crate) fn lean_type(ty: &Type) -> String {
    match ty {
        Type::Bits(w) => format!("BitVec {w}"),
        Type::Array(a) => format!(
            "Vector ({}) {}",
            lean_type(&a.element_type),
            a.element_count
        ),
        Type::Tuple(fields) if fields.is_empty() => "Unit".to_string(),
        Type::Tuple(fields) => format!(
            "({} × Unit)",
            fields
                .iter()
                .map(|t| lean_type(t))
                .collect::<Vec<_>>()
                .join(" × ")
        ),
        Type::Token => unreachable!("token types were rejected before emission"),
    }
}

/// Preserve operand order while keeping the Lean expression depth logarithmic.
fn balanced_infix(operands: &[String], separator: &str) -> String {
    fn emit(out: &mut String, operands: &[String], separator: &str) {
        match operands {
            [] => {}
            [operand] => out.push_str(operand),
            _ => {
                let (left, right) = operands.split_at(operands.len() / 2);
                out.push('(');
                emit(out, left, separator);
                out.push_str(separator);
                emit(out, right, separator);
                out.push(')');
            }
        }
    }

    let mut expression = String::new();
    emit(&mut expression, operands, separator);
    expression
}

fn tuple(fields: &[String]) -> String {
    if fields.is_empty() {
        "()".into()
    } else {
        format!("({}, ())", fields.join(", "))
    }
}

/// `Array.toVector` avoids unfolding computed elements in a `#v` size proof.
/// Explicit append lengths keep size checks local to each leaf.
fn vector_literal(fields: &[String]) -> String {
    if fields.len() <= 64 {
        return format!("#[{}].toVector", fields.join(", "));
    }
    let (left, right) = fields.split_at(fields.len() / 2);
    format!(
        "(@Vector.append _ {} {} {} {})",
        left.len(),
        right.len(),
        vector_literal(left),
        vector_literal(right)
    )
}

fn literal(value: &IrValue) -> Result<String, LeanError> {
    Ok(match value.type_() {
        Type::Bits(w) => format!(
            "BitVec.ofNat {w} 0x{}",
            value.to_string_fmt_no_prefix(IrFormatPreference::PlainHex)
        ),
        Type::Tuple(_) | Type::Array(_) => {
            let fields = value
                .as_elements()?
                .iter()
                .map(literal)
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(value.type_(), Type::Tuple(_)) {
                tuple(&fields)
            } else {
                vector_literal(&fields)
            }
        }
        Type::Token => unreachable!("token literals were rejected before emission"),
    })
}

/// The declared result width is significant for ordinary multiply and shifts.
fn binary(op: Binop, a: &str, b: &str, w: usize) -> String {
    match op {
        Binop::Add => format!("{a} + {b}"),
        Binop::Sub => format!("{a} - {b}"),
        Binop::Umul => format!("{a}.setWidth {w} * {b}.setWidth {w}"),
        Binop::Smul => format!("{a}.signExtend {w} * {b}.signExtend {w}"),
        Binop::Udiv => format!("BitVec.smtUDiv {a} {b}"),
        Binop::Sdiv => format!("_root_.XlsynthLean.IR.sdiv {a} {b}"),
        Binop::Umod => format!("if {b} == 0 then 0 else {a} % {b}"),
        Binop::Smod => format!("if {b} == 0 then 0 else {a}.srem {b}"),
        Binop::Shll => format!("{a} <<< (min {b}.toNat {w})"),
        Binop::Shrl => format!("{a} >>> (min {b}.toNat {w})"),
        Binop::Shra => format!("{a}.sshiftRight (min {b}.toNat {w})"),
        Binop::Eq => format!("BitVec.ofBool ({a} == {b})"),
        Binop::Ne => format!("BitVec.ofBool ({a} != {b})"),
        Binop::Ult => format!("BitVec.ofBool ({a}.ult {b})"),
        Binop::Ule => format!("BitVec.ofBool ({a}.ule {b})"),
        Binop::Ugt => format!("BitVec.ofBool ({b}.ult {a})"),
        Binop::Uge => format!("BitVec.ofBool ({b}.ule {a})"),
        Binop::Slt => format!("BitVec.ofBool ({a}.slt {b})"),
        Binop::Sle => format!("BitVec.ofBool ({a}.sle {b})"),
        Binop::Sgt => format!("BitVec.ofBool ({b}.slt {a})"),
        Binop::Sge => format!("BitVec.ofBool ({b}.sle {a})"),
        Binop::Gate => format!("if {a} == 1 then {b} else default"),
        Binop::Smulp | Binop::Umulp => unreachable!("unsupported partial product"),
    }
}

/// Renders one pure node of a verified, extension-desugared graph. Operands
/// must be Lean terms safe in argument and projection position (wrap compound
/// expressions in parentheses). The caller binds parameters or block ports,
/// chooses callee names, and handles registers and observable events itself.
pub(crate) fn emit_pure_expr(
    graph: &NodeGraph,
    reference: NodeRef,
    operand: &dyn Fn(NodeRef) -> Result<String, LeanError>,
    callee_name: &dyn Fn(&str) -> Result<String, LeanError>,
) -> Result<String, LeanError> {
    let n = graph.get_node(reference);
    check_supported(graph, n)?;
    let operands = |refs: &[NodeRef]| {
        refs.iter()
            .copied()
            .map(operand)
            .collect::<Result<Vec<_>, _>>()
    };
    Ok(match &n.payload {
        P::Literal(v) => literal(v)?,
        P::Tuple(rs) => tuple(&operands(rs)?),
        P::Array(rs) => vector_literal(&operands(rs)?),
        P::ArrayConcat(rs) => balanced_infix(&operands(rs)?, " ++ "),
        P::TupleIndex { tuple, index } => format!("{}{}.1", operand(*tuple)?, ".2".repeat(*index)),
        P::ArrayIndex { array, indices, .. } => {
            let mut element = operand(*array)?;
            for index in indices {
                element = format!(
                    "(_root_.XlsynthLean.IR.arrayIndex {element} {}.toNat)",
                    operand(*index)?
                );
            }
            element
        }
        P::ArrayUpdate {
            array,
            value,
            indices,
            ..
        } => {
            let value = operand(*value)?;
            let Some((outer, inner)) = indices.split_first() else {
                return Ok(value);
            };
            // Anonymous sections keep nested updates from capturing names in
            // caller-supplied index and value expressions.
            let mut update = format!("(fun _ => {value})");
            for index in inner.iter().rev() {
                update = format!(
                    "(_root_.XlsynthLean.IR.arrayModify · {}.toNat {update})",
                    operand(*index)?
                );
            }
            format!(
                "_root_.XlsynthLean.IR.arrayModify {} {}.toNat {update}",
                operand(*array)?,
                operand(*outer)?
            )
        }
        P::ArraySlice {
            array,
            start,
            width,
        } => format!(
            "_root_.XlsynthLean.IR.arraySlice {width} {} {}.toNat",
            operand(*array)?,
            operand(*start)?
        ),
        P::Binop(op, a, b) => binary(*op, &operand(*a)?, &operand(*b)?, width(&n.ty)?),
        P::Unop(op, a) => {
            let a = operand(*a)?;
            match op {
                Unop::Identity => a,
                Unop::Neg => format!("-{a}"),
                Unop::Not => format!("~~~{a}"),
                Unop::Reverse => format!("{a}.reverse"),
                Unop::OrReduce => format!("BitVec.ofBool ({a} != 0)"),
                Unop::AndReduce => format!("BitVec.ofBool ({a} == BitVec.allOnes _)"),
                Unop::XorReduce => format!("{a}.cpop.setWidth 1"),
            }
        }
        P::Nary(NaryOp::Concat, rs) if rs.is_empty() => "0".to_owned(),
        P::Nary(op, rs) => {
            let sep = match op {
                NaryOp::And | NaryOp::Nand => " &&& ",
                NaryOp::Or | NaryOp::Nor => " ||| ",
                NaryOp::Xor => " ^^^ ",
                NaryOp::Concat => " ++ ",
            };
            let args = balanced_infix(&operands(rs)?, sep);
            if matches!(op, NaryOp::Nand | NaryOp::Nor) {
                format!("~~~({args})")
            } else {
                args
            }
        }
        P::SignExt { arg, new_bit_count } => {
            format!("{}.signExtend {new_bit_count}", operand(*arg)?)
        }
        P::ZeroExt { arg, new_bit_count } => format!("{}.setWidth {new_bit_count}", operand(*arg)?),
        P::BitSlice { arg, start, width } => {
            format!("BitVec.extractLsb' {start} {width} {}", operand(*arg)?)
        }
        P::DynamicBitSlice {
            arg,
            start,
            width: w,
        } => format!(
            "BitVec.extractLsb' (min {}.toNat {}) {w} {}",
            operand(*start)?,
            width(graph.get_node_ty(*arg))?,
            operand(*arg)?
        ),
        P::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => format!(
            "_root_.XlsynthLean.IR.bitSliceUpdate {} {} {}",
            operand(*arg)?,
            operand(*start)?,
            operand(*update_value)?
        ),
        P::Decode { arg, width } => {
            format!(
                "BitVec.twoPow {width} (min {}.toNat {width})",
                operand(*arg)?
            )
        }
        P::Encode { arg } => format!(
            "_root_.XlsynthLean.IR.encode {} {}",
            width(&n.ty)?,
            operand(*arg)?
        ),
        P::OneHot { arg, lsb_prio } => {
            format!("_root_.XlsynthLean.IR.oneHot {lsb_prio} {}", operand(*arg)?)
        }
        P::OneHotSel { selector, cases } => format!(
            "_root_.XlsynthLean.IR.oneHotSel {} #[{}]",
            operand(*selector)?,
            operands(cases)?.join(", ")
        ),
        P::PrioritySel {
            selector,
            cases,
            default,
        } => format!(
            "_root_.XlsynthLean.IR.prioritySel {} #[{}] {}",
            operand(*selector)?,
            operands(cases)?.join(", "),
            operand(default.expect("verified priority default"))?
        ),
        P::Sel {
            selector,
            cases,
            default,
        } => {
            let fallback = default
                .or_else(|| cases.last().copied())
                .expect("verified selector has an arm");
            format!(
                "#[{}].getD {}.toNat {}",
                operands(cases)?.join(", "),
                operand(*selector)?,
                operand(fallback)?
            )
        }
        P::Map { array, to_apply } => {
            format!("Vector.map {} {}", callee_name(to_apply)?, operand(*array)?)
        }
        P::Invoke {
            to_apply,
            operands: args,
        } => std::iter::once(callee_name(to_apply)?)
            .chain(operands(args)?)
            .collect::<Vec<_>>()
            .join(" "),
        other => {
            return Err(LeanError::UnsupportedIr {
                function: graph.name.clone(),
                op: other.get_operator().to_owned(),
                id: n.text_id,
                reason: "no word-level translation",
            });
        }
    })
}
