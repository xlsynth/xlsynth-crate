// SPDX-License-Identifier: Apache-2.0

//! DSLX minimum bit widths for integer expressions without a concrete type.

use std::collections::BTreeMap;

use num_bigint::BigUint;

use crate::ast::{BinaryOp, Expr, Type};
use crate::parser::radix_digits;
use crate::typed::bits;

/// A literal's minimum bit type before the context supplies a concrete type.
struct Minimum {
    width: usize,
    signed: bool,
    /// A single hex or binary literal can spell a signed bit pattern directly.
    has_prefix: bool,
    /// An inline literal takes its width from the use site. A bound value
    /// retains the width at which it was evaluated.
    has_inline_literal: bool,
    /// An unannotated fold with an unsized seed needs a concrete use site.
    needs_concrete_type: bool,
}

/// A path in an aggregate type. All array elements share the same type.
#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
enum Member {
    TupleField(usize),
    ArrayElement,
}

/// Minimum annotations retained on a binding's unsized bit members.
#[derive(Default)]
pub(crate) struct BindingWidths(BTreeMap<Vec<Member>, Type>);

impl BindingWidths {
    /// The inferred scalar type of a range element before use-site constraints.
    pub(crate) fn scalar(ty: Type) -> Self {
        Self(BTreeMap::from([(Vec::new(), ty)]))
    }
}

pub(crate) type InferredBindings = BTreeMap<String, BindingWidths>;

/// Minimum annotations visible to a type-inference operation. In ordinary
/// expressions, only values local to an unrolled iteration are retyped
/// at use sites. Type dimensions also consult preceding outer let bindings.
#[derive(Clone, Copy)]
pub(crate) struct WidthLookup<'a> {
    widths: &'a InferredBindings,
    locals: Option<&'a BTreeMap<String, crate::typed::TypedExpr>>,
}

impl<'a> WidthLookup<'a> {
    pub(crate) fn loop_locals(
        widths: &'a InferredBindings,
        locals: &'a BTreeMap<String, crate::typed::TypedExpr>,
    ) -> Self {
        Self {
            widths,
            locals: Some(locals),
        }
    }

    fn all(widths: &'a InferredBindings) -> Self {
        Self {
            widths,
            locals: None,
        }
    }

    fn get(&self, name: &str) -> Option<&BindingWidths> {
        if self.locals.is_some_and(|locals| !locals.contains_key(name)) {
            return None;
        }
        self.widths.get(name)
    }

    /// Determine whether this use selects an implicitly typed bound value.
    pub(crate) fn is_inferred_reference(self, expr: &Expr) -> Result<bool, String> {
        match expr {
            Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. } => {
                collect_projected(expr, &[], self, &mut None)
            }
            _ => Ok(false),
        }
    }
}

impl Minimum {
    fn inferred_type(self) -> Result<Type, String> {
        if self.needs_concrete_type {
            return Err(
                "unroll_for! with an implicit seed needs a concrete accumulator type".into(),
            );
        }
        Ok(self.into_type())
    }

    fn into_type(self) -> Type {
        Type::Bits {
            signed: self.signed,
            width: self.width,
        }
    }

    /// Count the fewest bits that hold the unsigned or two's-complement value.
    fn from_literal(text: &str) -> Result<Self, String> {
        let signed = text.starts_with('-');
        let (digits, radix) = radix_digits(text.strip_prefix('-').unwrap_or(text));
        let magnitude = BigUint::parse_bytes(digits.as_bytes(), radix)
            .ok_or_else(|| format!("invalid integer literal '{text}'"))?;
        let mut width = magnitude.bits();
        // -2^k fits in k+1 signed bits. Other nonzero negative values need
        // another bit beyond their magnitude. Both 0 and -0 have width zero.
        if signed && width != 0 && magnitude.count_ones() != 1 {
            width += 1;
        }
        Ok(Self {
            width: usize::try_from(width).map_err(|_| "integer literal is too wide")?,
            signed,
            has_prefix: radix != 10,
            has_inline_literal: true,
            needs_concrete_type: false,
        })
    }

    /// Unify two minimum annotations as DSLX does for same-type operands.
    fn merge(self, other: Self) -> Self {
        let signed = self.signed || other.signed;
        let width = self.width.max(other.width);
        // A positive decimal needs a sign bit when its minimum width is
        // reinterpreted as signed. Hex and binary bit patterns already fit.
        let needs_sign_bit = |m: &Self| !m.signed && !m.has_prefix && m.width == width;
        Self {
            width: width + usize::from(signed && (needs_sign_bit(&self) || needs_sign_bit(&other))),
            signed,
            // Only individual literals carry a radix prefix.
            has_prefix: false,
            has_inline_literal: self.has_inline_literal || other.has_inline_literal,
            needs_concrete_type: self.needs_concrete_type || other.needs_concrete_type,
        }
    }
}

/// These bit operators share a single type across their operands and result.
pub(crate) fn has_common_bit_type(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Mod
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor
    )
}

/// Accumulate literal constraints in source order. A concrete type encountered
/// along a same-type edge takes precedence over minimum-width inference.
fn collect_minimum(
    expr: &Expr,
    bindings: Option<WidthLookup<'_>>,
    minimum: &mut Option<Minimum>,
) -> Result<bool, String> {
    let next = match expr {
        Expr::Literal { ty: None, text } => Minimum::from_literal(text)?,
        Expr::UnrollFor(fold) if fold.annotation.is_none() => {
            let mut seed = None;
            if !collect_minimum(&fold.seed, None, &mut seed)? {
                return Ok(false);
            }
            let Some(mut seed) = seed else {
                return Ok(false);
            };
            seed.needs_concrete_type = true;
            seed
        }
        Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. } => {
            let Some(bindings) = bindings else {
                return Ok(false);
            };
            let mut projected = None;
            if !collect_projected(expr, &[], bindings, &mut projected)? {
                return Ok(false);
            }
            let Some(mut projected) = projected else {
                return Ok(false);
            };
            // Projections and names retain the width of their stored value.
            projected.has_prefix = false;
            projected.has_inline_literal = false;
            projected
        }
        Expr::Unary { value, .. }
        | Expr::Binary {
            op: BinaryOp::Shl | BinaryOp::Shr,
            left: value,
            ..
        } => return collect_minimum(value, bindings, minimum),
        Expr::Binary { op, left, right } if has_common_bit_type(*op) => {
            return Ok(collect_minimum(left, bindings, minimum)?
                && collect_minimum(right, bindings, minimum)?);
        }
        Expr::If {
            then_value: left,
            else_value: right,
            ..
        } => {
            return Ok(collect_minimum(left, bindings, minimum)?
                && collect_minimum(right, bindings, minimum)?);
        }
        _ => return Ok(false),
    };
    *minimum = Some(match minimum.take() {
        Some(prior) => prior.merge(next),
        None => next,
    });
    Ok(true)
}

/// Infer a result width when all operands that constrain it are unsized
/// integers.
pub(crate) fn minimum_type(expr: &Expr) -> Result<Option<Type>, String> {
    let mut minimum = None;
    if collect_minimum(expr, None, &mut minimum)? {
        minimum.map(Minimum::inferred_type).transpose()
    } else {
        Ok(None)
    }
}

/// Collect only the scalar member selected by a projection. For an array
/// its minimum comes from all elements, regardless of the runtime index.
fn collect_projected(
    expr: &Expr,
    path: &[Member],
    bindings: WidthLookup<'_>,
    minimum: &mut Option<Minimum>,
) -> Result<bool, String> {
    match (expr, path.split_first()) {
        (Expr::Name(name), _) => {
            let Some((signed, width)) = bindings
                .get(name)
                .and_then(|binding| binding.0.get(path))
                .and_then(bits)
            else {
                return Ok(false);
            };
            let next = Minimum {
                signed,
                width,
                has_prefix: false,
                has_inline_literal: false,
                needs_concrete_type: false,
            };
            *minimum = Some(match minimum.take() {
                Some(previous) => previous.merge(next),
                None => next,
            });
            Ok(true)
        }
        (Expr::Field { value, name }, _) => {
            let Ok(field) = name.parse() else {
                return Ok(false);
            };
            let mut source_path = vec![Member::TupleField(field)];
            source_path.extend_from_slice(path);
            collect_projected(value, &source_path, bindings, minimum)
        }
        (Expr::Index { value, .. }, _) => {
            let mut source_path = vec![Member::ArrayElement];
            source_path.extend_from_slice(path);
            collect_projected(value, &source_path, bindings, minimum)
        }
        (Expr::Tuple(fields), Some((Member::TupleField(i), rest))) => match fields.get(*i) {
            Some(field) => collect_projected(field, rest, bindings, minimum),
            None => Ok(false),
        },
        (Expr::Array(items), Some((Member::ArrayElement, rest))) => {
            for item in items {
                if !collect_projected(item, rest, bindings, minimum)? {
                    return Ok(false);
                }
            }
            Ok(!items.is_empty())
        }
        (
            Expr::If {
                then_value,
                else_value,
                ..
            },
            Some(_),
        ) => Ok(collect_projected(then_value, path, bindings, minimum)?
            && collect_projected(else_value, path, bindings, minimum)?),
        (_, None) => collect_minimum(expr, Some(bindings), minimum),
        _ => Ok(false),
    }
}

/// Record inferred leaves once, so a later projection can reuse the type of a
/// bound value without walking its initializer or its dependencies again.
pub(crate) fn binding_widths(
    expr: &Expr,
    ty: &Type,
    bindings: &InferredBindings,
) -> Result<Option<BindingWidths>, String> {
    fn visit(
        expr: &Expr,
        ty: &Type,
        bindings: &InferredBindings,
        path: &mut Vec<Member>,
        out: &mut BindingWidths,
    ) -> Result<(), String> {
        match ty {
            Type::Array { element, .. } => {
                path.push(Member::ArrayElement);
                visit(expr, element, bindings, path, out)?;
                path.pop();
            }
            Type::Tuple(fields) => {
                for (i, field) in fields.iter().enumerate() {
                    path.push(Member::TupleField(i));
                    visit(expr, field, bindings, path, out)?;
                    path.pop();
                }
            }
            Type::Bits { .. } | Type::Bool => {
                let mut minimum = None;
                if collect_projected(expr, path, WidthLookup::all(bindings), &mut minimum)?
                    && let Some(minimum) = minimum
                {
                    out.0.insert(path.clone(), minimum.into_type());
                }
            }
            Type::Named(_) => { /* A struct declares concrete types for its fields. */ }
        }
        Ok(())
    }
    let mut widths = BindingWidths::default();
    visit(expr, ty, bindings, &mut Vec::new(), &mut widths)?;
    Ok((!widths.0.is_empty()).then_some(widths))
}

/// Which operands determine the width of a contextual unsigned expression.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ContextOperands {
    /// The first explicit operand supplies its own type.
    Explicit,
    /// Every operand with a common result type was already evaluated.
    Bound,
    /// At least one unsized literal must take the context width.
    Inline,
}

/// Apply the standard unsigned width to minimum annotations preceding the
/// first explicit operand, following DSLX's source-order unification.
pub(crate) fn context_operands(
    expr: &Expr,
    bindings: &InferredBindings,
    width: usize,
) -> Result<ContextOperands, String> {
    let mut minimum = None;
    let all_inferred = collect_minimum(expr, Some(WidthLookup::all(bindings)), &mut minimum)?;
    if let Some(minimum) = minimum {
        if minimum.signed || minimum.width > width {
            return Err(format!(
                "inferred type {} cannot satisfy unsigned width {width}",
                minimum.into_type()
            ));
        }
        if all_inferred {
            return Ok(if minimum.has_inline_literal {
                ContextOperands::Inline
            } else {
                ContextOperands::Bound
            });
        }
    }
    Ok(ContextOperands::Explicit)
}

/// The operand supplying a pair's type, with a common minimum for two unsized
/// operands.
#[derive(Debug, PartialEq)]
pub(crate) enum OperandHint {
    FromLeft,
    FromRight(Option<Type>),
}

/// Select a type for a same-type operand pair in one walk of its unsized
/// literals.
pub(crate) fn operand_hint(
    left: &Expr,
    right: &Expr,
    locals: Option<WidthLookup<'_>>,
) -> Result<OperandHint, String> {
    let mut minimum = None;
    if !collect_minimum(left, locals, &mut minimum)? {
        return Ok(OperandHint::FromLeft);
    }
    let hint = if collect_minimum(right, locals, &mut minimum)? {
        minimum.map(Minimum::inferred_type).transpose()?
    } else {
        None
    };
    Ok(OperandHint::FromRight(hint))
}

/// Check the accumulated literal minimum against the first concrete signed
/// operand. Once two unsized literals have unified, their minimum loses the
/// radix prefix that lets an individual hex or binary literal spell a sign bit.
pub(crate) fn check_inferred_signed_width(exprs: &[&Expr], ty: &Type) -> Result<(), String> {
    let Some((true, width)) = bits(ty) else {
        return Ok(());
    };
    let mut minimum = None;
    for expr in exprs {
        if !collect_minimum(expr, None, &mut minimum)? {
            break;
        }
    }
    if let Some(minimum) = minimum {
        let required = minimum.width + usize::from(!minimum.signed && !minimum.has_prefix);
        if required > width {
            return Err(format!(
                "preceding integer literals require {required} signed bits, got {ty}"
            ));
        }
    }
    Ok(())
}

/// Which type to apply to a sequence of array elements.
pub(crate) enum ArrayElementHint {
    Minimum(Type),
    FromElement(usize),
}

/// Whether this element's type is fully specified without any of its array
/// siblings. In that case checking the elements in source order suffices.
pub(crate) fn has_concrete_element_type(
    expr: &Expr,
    locals: Option<WidthLookup<'_>>,
) -> Result<bool, String> {
    match expr {
        Expr::If {
            then_value,
            else_value,
            ..
        } => Ok(has_concrete_element_type(then_value, locals)?
            || has_concrete_element_type(else_value, locals)?),
        Expr::Tuple(fields) => {
            for field in fields {
                if !has_concrete_element_type(field, locals)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Expr::Array(elements) => {
            for element in elements {
                if has_concrete_element_type(element, locals)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(!collect_minimum(expr, locals, &mut None)?),
    }
}

/// Merge unsized element constraints, or identify an element whose type can
/// determine the others. A concrete aggregate can also type a literal array
/// or tuple that precedes it.
pub(crate) fn array_element_hint(
    items: &[&Expr],
    locals: Option<WidthLookup<'_>>,
) -> Result<ArrayElementHint, String> {
    let mut minimum = None;
    let mut constructor = None;
    for (i, expr) in items.iter().enumerate() {
        if matches!(expr, Expr::Tuple(_) | Expr::Array(_)) {
            constructor.get_or_insert(i);
        } else if !collect_minimum(expr, locals, &mut minimum)? {
            return Ok(ArrayElementHint::FromElement(i));
        }
    }
    if let Some(i) = constructor {
        return Ok(ArrayElementHint::FromElement(i));
    }
    Ok(ArrayElementHint::Minimum(
        minimum
            .ok_or("cannot infer an empty array's element type")?
            .inferred_type()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{OperandHint, minimum_type, operand_hint};
    use crate::ast::{Expr, Type};

    #[test]
    fn minimum_signed_widths_follow_twos_complement_and_literal_radix() {
        let literal = |text: &str| Expr::Literal {
            ty: None,
            text: text.into(),
        };
        for (text, width) in [
            ("-0", 0),
            ("-1", 1),
            ("-3", 3),
            ("-4", 3),
            ("-0x10000000000000000", 65),
            ("-0x10000000000000001", 66),
        ] {
            assert_eq!(
                minimum_type(&literal(text)).unwrap(),
                Some(Type::Bits {
                    signed: true,
                    width
                }),
                "{text}"
            );
        }
        for (text, width) in [("255", 9), ("0xff", 8), ("0b11111111", 8)] {
            assert_eq!(
                operand_hint(&literal(text), &literal("-1"), None).unwrap(),
                OperandHint::FromRight(Some(Type::Bits {
                    signed: true,
                    width
                })),
                "{text} and -1"
            );
        }
    }
}
