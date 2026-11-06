// SPDX-License-Identifier: Apache-2.0

//! DSLX minimum bit widths for integer expressions without a concrete type.

use std::collections::BTreeMap;
use std::rc::Rc;

use num_bigint::BigUint;

use crate::ast::{BinaryOp, Expr, Type};
use crate::parser::radix_digits;
use crate::typed::bits;

/// A literal's minimum bit type before the context supplies a concrete type.
#[derive(Clone)]
struct Minimum {
    width: usize,
    signed: bool,
    /// A single hex or binary literal can spell a signed bit pattern directly.
    has_prefix: bool,
    /// Aliases of one bound prefixed literal share its minimum annotation.
    /// Distinct occurrences have distinct constraints even with equal bits.
    origin: Option<Rc<()>>,
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
#[derive(Clone, Default)]
pub(crate) struct BindingWidths(BTreeMap<Vec<Member>, Minimum>);

impl BindingWidths {
    /// Distribute the inferred widths of a destructured tuple to its members.
    pub(crate) fn tuple_fields(self, count: usize) -> Vec<Option<Self>> {
        let mut fields = (0..count).map(|_| None).collect::<Vec<_>>();
        for (path, ty) in self.0 {
            if let Some((Member::TupleField(i), rest)) = path.split_first() {
                fields[*i]
                    .get_or_insert_with(Self::default)
                    .0
                    .insert(rest.to_vec(), ty);
            }
        }
        fields
    }

    /// The inferred scalar type of a range element before use-site constraints.
    pub(crate) fn scalar(signed: bool, width: usize) -> Self {
        let minimum = Minimum {
            signed,
            width,
            has_prefix: false,
            origin: None,
            has_inline_literal: false,
            needs_concrete_type: false,
        };
        Self(BTreeMap::from([(Vec::new(), minimum)]))
    }
}

pub(crate) type InferredBindings = BTreeMap<String, BindingWidths>;

/// Minimum annotations visible to a type-inference operation. In ordinary
/// expressions, only values local to an unrolled iteration are retyped
/// at use sites. Type dimensions also consult preceding outer let bindings.
#[derive(Clone, Copy)]
pub(crate) enum WidthLookup<'a> {
    Bindings {
        widths: &'a InferredBindings,
        locals: Option<&'a BTreeMap<String, crate::typed::TypedExpr>>,
    },
    /// Local DSLX expressions retain the radix of an unsized projected or
    /// bound literal when assigning signedness at its existing hardware width.
    DslxLiterals(Option<&'a InferredBindings>),
}

impl<'a> WidthLookup<'a> {
    pub(crate) fn loop_locals(
        widths: &'a InferredBindings,
        locals: &'a BTreeMap<String, crate::typed::TypedExpr>,
    ) -> Self {
        Self::Bindings {
            widths,
            locals: Some(locals),
        }
    }

    fn all(widths: &'a InferredBindings) -> Self {
        Self::Bindings {
            widths,
            locals: None,
        }
    }

    fn get(&self, name: &str) -> Option<&'a BindingWidths> {
        match self {
            Self::Bindings { widths, locals } => {
                if locals.is_some_and(|locals| !locals.contains_key(name)) {
                    return None;
                }
                widths.get(name)
            }
            Self::DslxLiterals(widths) => widths.and_then(|w| w.get(name)),
        }
    }

    /// A new lexical scope may shadow an outer binding's minimum.
    fn without_names(self) -> Option<Self> {
        match self {
            Self::DslxLiterals(_) => Some(Self::DslxLiterals(None)),
            Self::Bindings { .. } => None,
        }
    }

    /// Determine whether this use selects an implicitly typed bound value.
    pub(crate) fn is_inferred_reference(self, expr: &Expr) -> Result<bool, String> {
        match expr {
            Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. } => {
                collect_projected(expr, &[], Some(self), &mut None)
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
            origin: None,
            has_inline_literal: true,
            needs_concrete_type: false,
        })
    }

    /// Unify two minimum annotations as DSLX does for same-type operands.
    fn merge(self, other: Self) -> Self {
        if let (Some(a), Some(b)) = (&self.origin, &other.origin)
            && Rc::ptr_eq(a, b)
        {
            // The same unsized type constraint can reach both operands
            // through aliases. Applying it twice still denotes one literal.
            return self;
        }
        let signed = self.signed || other.signed;
        let width = self.width.max(other.width);
        // A positive decimal needs a sign bit when its minimum width is
        // reinterpreted as signed. Hex and binary bit patterns already fit.
        let needs_sign_bit = |m: &Self| !m.signed && !m.has_prefix && m.width == width;
        Self {
            width: width + usize::from(signed && (needs_sign_bit(&self) || needs_sign_bit(&other))),
            signed,
            // A group of distinct minimum annotations loses its radix prefix.
            has_prefix: false,
            origin: None,
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
        Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. }
            if matches!(bindings, Some(WidthLookup::DslxLiterals(_))) =>
        {
            return collect_projected(expr, &[], bindings, minimum);
        }
        Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. } => {
            let Some(bindings) = bindings else {
                return Ok(false);
            };
            let mut projected = None;
            if !collect_projected(expr, &[], Some(bindings), &mut projected)? {
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
        // Bindings inside the scope may shadow any outer width annotations.
        Expr::Scope { result, .. } => {
            return collect_minimum(
                result,
                bindings.and_then(WidthLookup::without_names),
                minimum,
            );
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
    bindings: Option<WidthLookup<'_>>,
    minimum: &mut Option<Minimum>,
) -> Result<bool, String> {
    match (expr, path.split_first()) {
        (Expr::Name(name), _) => {
            let Some(bound) = bindings
                .and_then(|widths| widths.get(name))
                .and_then(|binding| binding.0.get(path))
            else {
                return Ok(false);
            };
            let dslx = matches!(bindings, Some(WidthLookup::DslxLiterals(_)));
            let next = Minimum {
                signed: bound.signed,
                width: bound.width,
                has_prefix: bound.has_prefix && dslx,
                origin: if dslx { bound.origin.clone() } else { None },
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
        (Expr::Scope { result, .. }, _) => collect_projected(
            result,
            path,
            bindings.and_then(WidthLookup::without_names),
            minimum,
        ),
        (_, None) => collect_minimum(expr, bindings, minimum),
        _ => Ok(false),
    }
}

/// Visit each scalar type that can have an inferred width, in member order.
fn visit_bit_members(
    ty: &Type,
    path: &mut Vec<Member>,
    visit: &mut impl FnMut(&[Member], &Type) -> Result<(), String>,
) -> Result<(), String> {
    match ty {
        Type::Array { element, .. } => {
            path.push(Member::ArrayElement);
            visit_bit_members(element, path, visit)?;
            path.pop();
        }
        Type::Tuple(fields) => {
            for (i, field) in fields.iter().enumerate() {
                path.push(Member::TupleField(i));
                visit_bit_members(field, path, visit)?;
                path.pop();
            }
        }
        Type::Bits { .. } | Type::Bool => visit(path, ty)?,
        Type::Named(_) => { /* A struct declares concrete types for its fields. */ }
    }
    Ok(())
}

/// Record inferred leaves once, so a later projection can reuse the type of a
/// bound value without walking its initializer or its dependencies again.
pub(crate) fn binding_widths(
    expr: &Expr,
    ty: &Type,
    bindings: &InferredBindings,
) -> Result<Option<BindingWidths>, String> {
    binding_widths_using(expr, ty, WidthLookup::all(bindings))
}

/// Record the literal radix as well as the minimum width for local DSLX lets.
pub(crate) fn binding_widths_using(
    expr: &Expr,
    ty: &Type,
    lookup: WidthLookup<'_>,
) -> Result<Option<BindingWidths>, String> {
    let mut widths = BindingWidths::default();
    visit_bit_members(ty, &mut Vec::new(), &mut |path, _| {
        let mut minimum = None;
        if collect_projected(expr, path, Some(lookup), &mut minimum)?
            && let Some(mut minimum) = minimum
        {
            if minimum.has_prefix && matches!(lookup, WidthLookup::DslxLiterals(_)) {
                minimum.origin.get_or_insert_with(|| Rc::new(()));
            }
            widths.0.insert(path.to_vec(), minimum);
        }
        Ok(())
    })?;
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

/// Check the accumulated literal minimum against the first concrete signed
/// operand. Once two unsized literals have unified, their minimum loses the
/// radix prefix that lets an individual hex or binary literal spell a sign bit.
pub(crate) fn check_inferred_signed_width(
    exprs: &[&Expr],
    ty: &Type,
    bindings: Option<WidthLookup>,
) -> Result<(), String> {
    let Some((true, width)) = bits(ty) else {
        return Ok(());
    };
    let mut minimum = None;
    for expr in exprs {
        if !collect_minimum(expr, bindings, &mut minimum)? {
            break;
        }
    }
    check_signed_minimum(minimum, ty, width)
}

fn check_signed_minimum(minimum: Option<Minimum>, ty: &Type, width: usize) -> Result<(), String> {
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

/// DSLX infers a call argument before unifying it with its parameter. Array
/// elements share their literal minimum, while tuple fields are independent.
pub(crate) fn check_argument_widths(
    expr: &Expr,
    ty: &Type,
    bindings: Option<&InferredBindings>,
) -> Result<(), String> {
    visit_bit_members(ty, &mut Vec::new(), &mut |path, ty| {
        if let Some((true, width)) = bits(ty) {
            check_projected_argument(expr, path, ty, width, bindings)?;
        }
        Ok(())
    })
}

/// Check one signed parameter leaf. The native frontend first infers each
/// nested aggregate, then unifies corresponding fields across the group.
fn check_projected_argument(
    expr: &Expr,
    path: &[Member],
    ty: &Type,
    width: usize,
    bindings: Option<&InferredBindings>,
) -> Result<(), String> {
    match (expr, path.split_first()) {
        (Expr::Scope { result, .. }, _) => {
            return check_projected_argument(result, path, ty, width, None);
        }
        (Expr::Field { value, name }, _) => {
            if let Ok(i) = name.parse() {
                let mut source_path = vec![Member::TupleField(i)];
                source_path.extend_from_slice(path);
                return check_projected_argument(value, &source_path, ty, width, bindings);
            }
        }
        (Expr::Index { value, .. }, _) => {
            let mut source_path = vec![Member::ArrayElement];
            source_path.extend_from_slice(path);
            return check_projected_argument(value, &source_path, ty, width, bindings);
        }
        (Expr::Tuple(items), Some((Member::TupleField(i), rest))) => {
            if let Some(item) = items.get(*i) {
                return check_projected_argument(item, rest, ty, width, bindings);
            }
        }
        (Expr::Array(items), Some((Member::ArrayElement, rest))) if !rest.is_empty() => {
            for item in items {
                check_projected_argument(item, rest, ty, width, bindings)?;
            }
        }
        (
            Expr::If {
                then_value,
                else_value,
                ..
            },
            Some(_),
        ) => {
            check_projected_argument(then_value, path, ty, width, bindings)?;
            check_projected_argument(else_value, path, ty, width, bindings)?;
        }
        _ => { /* Scalar operands contribute to the common minimum below. */ }
    }
    let mut minimum = None;
    collect_projected(
        expr,
        path,
        Some(WidthLookup::DslxLiterals(bindings)),
        &mut minimum,
    )?;
    check_signed_minimum(minimum, ty, width)
}

/// Check whether each differing field of a projection is an unsized hex or
/// binary literal whose bit pattern already occupies the target signed width.
pub(crate) fn projected_type_compatible(
    expr: &Expr,
    actual: &Type,
    expected: &Type,
    bindings: Option<&InferredBindings>,
) -> Result<bool, String> {
    fn member_matches(
        expr: &Expr,
        path: &mut Vec<Member>,
        actual: &Type,
        expected: &Type,
        bindings: Option<&InferredBindings>,
    ) -> Result<bool, String> {
        if actual == expected {
            return Ok(true);
        }
        match (actual, expected) {
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    path.push(Member::TupleField(index));
                    let matches = member_matches(expr, path, a, b, bindings)?;
                    path.pop();
                    if !matches {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (
                Type::Array {
                    element: a,
                    size: n,
                },
                Type::Array {
                    element: b,
                    size: m,
                },
            ) if n == m => {
                path.push(Member::ArrayElement);
                let matches = member_matches(expr, path, a, b, bindings)?;
                path.pop();
                Ok(matches)
            }
            _ => {
                let (Some((a_signed, a_width)), Some((b_signed, b_width))) =
                    (bits(actual), bits(expected))
                else {
                    return Ok(false);
                };
                if a_width != b_width || a_signed && !b_signed {
                    return Ok(false);
                }
                if a_signed == b_signed {
                    // bool and u1 have the same source representation.
                    return Ok(true);
                }
                let mut minimum = None;
                let inferred = collect_projected(
                    expr,
                    path,
                    Some(WidthLookup::DslxLiterals(bindings)),
                    &mut minimum,
                )?;
                Ok(inferred
                    && minimum.is_some_and(|m| !m.signed && m.has_prefix && m.width == a_width))
            }
        }
    }
    member_matches(expr, &mut Vec::new(), actual, expected, bindings)
}

/// The expression supplying a group's type, or the minimum type of its
/// literals.
#[derive(Debug, PartialEq)]
pub(crate) enum CommonTypeHint {
    Minimum(Type),
    FromExpr(usize),
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
pub(crate) fn common_type_hint(
    items: &[&Expr],
    locals: Option<WidthLookup<'_>>,
) -> Result<CommonTypeHint, String> {
    let mut minimum = None;
    let mut constructor = None;
    for (i, expr) in items.iter().enumerate() {
        if matches!(expr, Expr::Tuple(_) | Expr::Array(_)) {
            constructor.get_or_insert(i);
        } else if !collect_minimum(expr, locals, &mut minimum)? {
            return Ok(CommonTypeHint::FromExpr(i));
        }
    }
    if let Some(i) = constructor {
        return Ok(CommonTypeHint::FromExpr(i));
    }
    Ok(CommonTypeHint::Minimum(
        minimum
            .ok_or("cannot infer an empty array's element type")?
            .inferred_type()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{CommonTypeHint, common_type_hint, minimum_type};
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
                common_type_hint(&[&literal(text), &literal("-1")], None).unwrap(),
                CommonTypeHint::Minimum(Type::Bits {
                    signed: true,
                    width
                }),
                "{text} and -1"
            );
        }
    }
}
