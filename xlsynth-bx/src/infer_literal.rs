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
    /// Outermost scope in which a contributing value was materialized. A
    /// callback can reuse captured bits' signedness but cannot widen them.
    bound_scope: Option<usize>,
    /// A single hex or binary literal can spell a signed bit pattern directly.
    has_prefix: bool,
    /// Aliases of one bound prefixed literal share its minimum annotation.
    /// Distinct occurrences have distinct constraints even with equal bits.
    origin: Option<Rc<()>>,
    /// An inline literal takes its width from the use site. A bound value
    /// retains the width at which it was evaluated.
    has_inline_literal: bool,
    /// An unannotated fold with an unsized seed needs a concrete use site.
    needs_accumulator_type: Option<&'static str>,
}

/// A path in an aggregate type. All array elements share the same type.
#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Member {
    TupleField(usize),
    ArrayElement,
}

/// Minimum annotations retained on a binding's unsized bit members.
#[derive(Clone, Default)]
pub(crate) struct BindingWidths {
    members: BTreeMap<Vec<Member>, Minimum>,
    /// Destructured fields and captured aggregate operands can acquire their
    /// concrete member types at each use in a block expression.
    pub(crate) infer_at_use: bool,
}

impl BindingWidths {
    /// Distribute the inferred widths of a destructured tuple to its members.
    pub(crate) fn tuple_fields(self, count: usize) -> Vec<Option<Self>> {
        let mut fields = (0..count).map(|_| None).collect::<Vec<_>>();
        for (path, ty) in self.members {
            if let Some((Member::TupleField(i), rest)) = path.split_first() {
                let field = fields[*i].get_or_insert_with(Self::default);
                field.infer_at_use = true;
                field.members.insert(rest.to_vec(), ty);
            }
        }
        fields
    }

    /// Remove the shared array-element prefix when binding an iteration value.
    pub(crate) fn array_element(self) -> Option<Self> {
        let widths = self
            .members
            .into_iter()
            .filter_map(|(path, ty)| {
                let (Member::ArrayElement, rest) = path.split_first()? else {
                    return None;
                };
                Some((rest.to_vec(), ty))
            })
            .collect::<BTreeMap<_, _>>();
        (!widths.is_empty()).then_some(Self {
            members: widths,
            infer_at_use: self.infer_at_use,
        })
    }

    /// The inferred scalar type of a range element before use-site constraints.
    pub(crate) fn scalar(signed: bool, width: usize) -> Self {
        let minimum = Minimum {
            signed,
            width,
            bound_scope: None,
            has_prefix: false,
            origin: None,
            has_inline_literal: false,
            needs_accumulator_type: None,
        };
        Self {
            members: BTreeMap::from([(Vec::new(), minimum)]),
            infer_at_use: false,
        }
    }

    /// Assign a scope to newly bound literals, preserving the origin of each
    /// member forwarded from an earlier binding.
    pub(crate) fn bind_in_scope(&mut self, scope: usize) {
        for minimum in self.members.values_mut() {
            minimum.bound_scope.get_or_insert(scope);
        }
    }
}

pub(crate) type InferredBindings = BTreeMap<String, BindingWidths>;

/// Minimum annotations visible to a type-inference operation. Block expressions
/// can retype unrolled locals and destructured inferred values at their use
/// sites. Type dimensions also consult preceding outer let bindings.
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
    pub(crate) fn scoped_locals(
        widths: &'a InferredBindings,
        loops: &'a BTreeMap<String, crate::typed::TypedExpr>,
    ) -> Self {
        Self::Bindings {
            widths,
            locals: Some(loops),
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
                let width = widths.get(name)?;
                if locals.is_some_and(|loops| !loops.contains_key(name)) && !width.infer_at_use {
                    return None;
                }
                Some(width)
            }
            Self::DslxLiterals(widths) => widths.and_then(|w| w.get(name)),
        }
    }

    /// Retain the DSLX literal radix for an inferred member when a block
    /// expression determines the common type of several operands.
    fn dslx_reference(self, mut expr: &Expr) -> Option<Self> {
        let Self::Bindings { widths, .. } = self else {
            return Some(self);
        };
        while let Expr::Field { value, .. } | Expr::Index { value, .. } = expr {
            expr = value;
        }
        let Expr::Name(name) = expr else {
            return None;
        };
        self.get(name)?
            .infer_at_use
            .then_some(Self::DslxLiterals(Some(widths)))
    }

    /// A new lexical scope may shadow an outer binding's minimum.
    fn without_names(self) -> Option<Self> {
        match self {
            Self::DslxLiterals(_) => Some(Self::DslxLiterals(None)),
            Self::Bindings { .. } => None,
        }
    }

    /// Determine whether this use selects a value that can widen in this scope.
    pub(crate) fn can_widen_reference(self, expr: &Expr, scope: usize) -> Result<bool, String> {
        match expr {
            Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. } => {
                let mut minimum = None;
                Ok(collect_projected(expr, &[], Some(self), &mut minimum)?
                    && minimum.is_some_and(|m| m.can_widen_in(scope)))
            }
            _ => Ok(false),
        }
    }
}

impl Minimum {
    fn can_widen_in(&self, scope: usize) -> bool {
        self.bound_scope.is_none_or(|bound| bound == scope)
    }

    fn inferred_type(self) -> Result<Type, String> {
        if let Some(fold) = self.needs_accumulator_type {
            return Err(format!(
                "{fold} with an implicit seed needs a concrete accumulator type"
            ));
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
            bound_scope: None,
            has_prefix: radix != 10,
            origin: None,
            has_inline_literal: true,
            needs_accumulator_type: None,
        })
    }

    /// Unify two minimum annotations as DSLX does for same-type operands.
    fn merge(mut self, other: Self) -> Self {
        // Map scopes are numbered on entry, so an outer capture has a lower
        // number than every local constraint with which it can be combined.
        let bound_scope = match (self.bound_scope, other.bound_scope) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if let (Some(a), Some(b)) = (&self.origin, &other.origin)
            && Rc::ptr_eq(a, b)
        {
            // The same unsized type constraint can reach both operands
            // through aliases. Applying it twice still denotes one literal.
            self.bound_scope = bound_scope;
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
            bound_scope,
            // A group of distinct minimum annotations loses its radix prefix.
            has_prefix: false,
            origin: None,
            has_inline_literal: self.has_inline_literal || other.has_inline_literal,
            needs_accumulator_type: self.needs_accumulator_type.or(other.needs_accumulator_type),
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
    collect_minimum_with_calls(expr, bindings, minimum, MinimumContext::Concrete)
}

/// Which expressions supply a concrete result during minimum-width inference.
#[derive(Clone, Copy)]
enum MinimumContext<'a> {
    Concrete,
    /// Module declarations need a concrete result even when an operator
    /// constrains both implicit operands to one bit.
    ConstantInitializer,
    /// Match subjects unify with patterns before a call supplies its result.
    Deferred {
        zero_is_call: &'a dyn Fn() -> Result<bool, String>,
    },
}

/// An implicit scalar seed contributes literal constraints to its enclosing
/// expression, but cannot by itself establish the accumulator's type.
fn collect_fold_seed(
    seed: &Expr,
    fold_name: &'static str,
    minimum: &mut Option<Minimum>,
    calls: MinimumContext<'_>,
) -> Result<bool, String> {
    let mut seed_minimum = None;
    if !collect_minimum_with_calls(seed, None, &mut seed_minimum, calls)? {
        return Ok(false);
    }
    let Some(mut seed) = seed_minimum else {
        return Ok(matches!(calls, MinimumContext::Deferred { .. }));
    };
    seed.needs_accumulator_type = Some(fold_name);
    *minimum = Some(match minimum.take() {
        Some(prior) => prior.merge(seed),
        None => seed,
    });
    Ok(true)
}

fn collect_minimum_with_calls(
    expr: &Expr,
    bindings: Option<WidthLookup<'_>>,
    minimum: &mut Option<Minimum>,
    calls: MinimumContext<'_>,
) -> Result<bool, String> {
    let next = match expr {
        Expr::Literal { ty: None, text } => Minimum::from_literal(text)?,
        Expr::Call { .. } | Expr::MapBeforeUse { .. } | Expr::ParametricCall { .. }
            if matches!(calls, MinimumContext::Deferred { .. }) =>
        {
            return Ok(true);
        }
        Expr::Zero(_) => {
            // An intrinsic has an explicit result type; a local zero! function
            // has the same delayed result constraint as any other invocation.
            return match calls {
                MinimumContext::Concrete | MinimumContext::ConstantInitializer => Ok(false),
                MinimumContext::Deferred { zero_is_call } => zero_is_call(),
            };
        }
        Expr::UnrollFor(fold) if fold.annotation.is_none() => {
            return collect_fold_seed(&fold.seed, "unroll_for!", minimum, calls);
        }
        Expr::For(fold) if fold.annotation.is_none() => {
            return collect_fold_seed(&fold.seed, "for", minimum, calls);
        }
        Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. } => {
            let Some(bindings) = bindings else {
                return Ok(false);
            };
            if let Some(dslx) = bindings.dslx_reference(expr) {
                return collect_projected(expr, &[], Some(dslx), minimum);
            }
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
            return collect_minimum_with_calls(
                result,
                bindings.and_then(WidthLookup::without_names),
                minimum,
                calls,
            );
        }
        Expr::Unary { value, .. }
        | Expr::Binary {
            op: BinaryOp::Shl | BinaryOp::Shr,
            left: value,
            ..
        } => return collect_minimum_with_calls(value, bindings, minimum, calls),
        Expr::Binary { op, left, right }
            if has_common_bit_type(*op)
                || (matches!(calls, MinimumContext::ConstantInitializer)
                    && matches!(op, BinaryOp::LogicalAnd | BinaryOp::LogicalOr)) =>
        {
            return Ok(collect_minimum_with_calls(left, bindings, minimum, calls)?
                && collect_minimum_with_calls(right, bindings, minimum, calls)?);
        }
        Expr::If {
            then_value: left,
            else_value: right,
            ..
        } => {
            return Ok(collect_minimum_with_calls(left, bindings, minimum, calls)?
                && collect_minimum_with_calls(right, bindings, minimum, calls)?);
        }
        Expr::Match { arms, .. } => {
            for arm in arms {
                if !collect_minimum_with_calls(&arm.value, bindings, minimum, calls)? {
                    return Ok(false);
                }
            }
            return Ok(!arms.is_empty());
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
    expression_minimum(expr, None)?
        .map(Minimum::inferred_type)
        .transpose()
}

/// Supply the literal constraints of a BX data operand to a native DSLX
/// type probe. Its runtime value is elaborated separately at the selected
/// width. Signedness and decimal versus bit-pattern spelling affect inference.
pub(crate) fn import_literal_witness(expr: &Expr) -> Result<Option<String>, String> {
    let Some(minimum) = expression_minimum(expr, None)? else {
        return Ok(None);
    };
    literal_witness(minimum).map(Some)
}

/// A named literal supplies independent evidence if no parametric fixed its
/// width, and its radix can permit reinterpretation at the same width.
pub(crate) fn import_bound_literal_witness(
    expr: &Expr,
    ty: &Type,
    lookup: WidthLookup<'_>,
    scope: usize,
) -> Result<Option<String>, String> {
    if !lookup.can_widen_reference(expr, scope)? {
        return Ok(None);
    }
    let Some(minimum) = expression_minimum(expr, Some(lookup))? else {
        return Ok(None);
    };
    if bits(ty) != Some((minimum.signed, minimum.width)) {
        return Ok(None);
    }
    literal_witness(minimum).map(Some)
}

fn literal_witness(minimum: Minimum) -> Result<String, String> {
    if minimum.width > crate::check::MAX_TYPE_SIZE {
        return Err("integer literal exceeds maximum concrete type size".into());
    }
    let magnitude = if minimum.width == 0 {
        BigUint::from(0u8)
    } else {
        BigUint::from(1u8) << (minimum.width - 1)
    };
    let sign = if minimum.signed { "-" } else { "" };
    Ok(if minimum.has_prefix {
        format!("{sign}{magnitude:#x}")
    } else {
        format!("{sign}{magnitude}")
    })
}

/// An unannotated module constant needs at least one concrete constraint on
/// its scalar result. Logical operators also propagate implicit operands
/// along this edge, while comparisons supply their own boolean result type.
pub(crate) fn has_implicit_constant_type(expr: &Expr) -> Result<bool, String> {
    collect_minimum_with_calls(expr, None, &mut None, MinimumContext::ConstantInitializer)
}

/// An aggregate operand can supply its own literal width unless an implicit
/// fold seed must first receive a concrete type from another operand.
pub(crate) fn minimum_operand_type(
    expr: &Expr,
    bindings: Option<WidthLookup<'_>>,
) -> Result<Option<Type>, String> {
    Ok(expression_minimum(expr, bindings)?
        .filter(|m| m.needs_accumulator_type.is_none())
        .map(Minimum::into_type))
}

fn expression_minimum(
    expr: &Expr,
    bindings: Option<WidthLookup<'_>>,
) -> Result<Option<Minimum>, String> {
    let mut minimum = None;
    if collect_minimum(expr, bindings, &mut minimum)? {
        Ok(minimum)
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
    collect_projected_with_calls(expr, path, bindings, minimum, MinimumContext::Concrete)
}

/// Collect a member's literal constraints before any deferred invocation
/// results are applied. A projection reads an independently inferred aggregate.
fn collect_projected_with_calls(
    expr: &Expr,
    path: &[Member],
    bindings: Option<WidthLookup<'_>>,
    minimum: &mut Option<Minimum>,
    calls: MinimumContext<'_>,
) -> Result<bool, String> {
    match (expr, path.split_first()) {
        (Expr::Name(name), _) => {
            let Some(bound) = bindings
                .and_then(|widths| widths.get(name))
                .and_then(|binding| binding.members.get(path))
            else {
                return Ok(false);
            };
            let dslx = matches!(bindings, Some(WidthLookup::DslxLiterals(_)));
            let next = Minimum {
                signed: bound.signed,
                width: bound.width,
                bound_scope: bound.bound_scope,
                has_prefix: bound.has_prefix && dslx,
                origin: if dslx { bound.origin.clone() } else { None },
                has_inline_literal: false,
                needs_accumulator_type: None,
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
            Some(field) => collect_projected_with_calls(field, rest, bindings, minimum, calls),
            None => Ok(false),
        },
        (Expr::Array(items), Some((Member::ArrayElement, rest))) => {
            for item in items {
                if !collect_projected_with_calls(item, rest, bindings, minimum, calls)? {
                    return Ok(false);
                }
            }
            Ok(!items.is_empty())
        }
        (Expr::For(fold), Some(_)) if fold.annotation.is_none() => {
            collect_projected_with_calls(&fold.seed, path, bindings, minimum, calls)
        }
        (
            Expr::If {
                then_value,
                else_value,
                ..
            },
            Some(_),
        ) => Ok(
            collect_projected_with_calls(then_value, path, bindings, minimum, calls)?
                && collect_projected_with_calls(else_value, path, bindings, minimum, calls)?,
        ),
        (Expr::Match { arms, .. }, Some(_)) => {
            for arm in arms {
                if !collect_projected_with_calls(&arm.value, path, bindings, minimum, calls)? {
                    return Ok(false);
                }
            }
            Ok(!arms.is_empty())
        }
        (Expr::Scope { result, .. }, _) => collect_projected_with_calls(
            result,
            path,
            bindings.and_then(WidthLookup::without_names),
            minimum,
            calls,
        ),
        (_, None) => collect_minimum_with_calls(expr, bindings, minimum, calls),
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
        Type::ParametricArray { .. }
        | Type::ParametricBits { .. }
        | Type::ParametricStruct { .. } => {
            return Err("literal inference requires resolved type dimensions".into());
        }
        Type::Enum { .. } | Type::Named(_) | Type::Applied { .. } => { /* A struct declares concrete types for its fields. */
        }
    }
    Ok(())
}

/// Apply the scalar literal unification rules to corresponding inferred leaves
/// of aggregate operands. Concrete leaves already have their operand types.
pub(crate) fn unify_inferred_members(
    ty: &mut Type,
    operands: &[&Expr],
    bindings: &InferredBindings,
) -> Result<(), String> {
    fn visit(
        ty: &mut Type,
        path: &mut Vec<Member>,
        operands: &[&Expr],
        bindings: WidthLookup<'_>,
    ) -> Result<(), String> {
        match ty {
            Type::Tuple(fields) => {
                for (i, field) in fields.iter_mut().enumerate() {
                    path.push(Member::TupleField(i));
                    visit(field, path, operands, bindings)?;
                    path.pop();
                }
            }
            Type::Array { element, .. } => {
                path.push(Member::ArrayElement);
                visit(element, path, operands, bindings)?;
                path.pop();
            }
            Type::Bits { .. } | Type::Bool => {
                let mut minimum = None;
                for operand in operands {
                    if !collect_projected(operand, path, Some(bindings), &mut minimum)? {
                        return Ok(());
                    }
                }
                if let Some(minimum) = minimum {
                    *ty = minimum.inferred_type()?;
                }
            }
            Type::Enum { .. } | Type::Named(_) | Type::Applied { .. } => { /* Nominal types have concrete member types. */
            }
            Type::ParametricArray { .. }
            | Type::ParametricBits { .. }
            | Type::ParametricStruct { .. } => {
                return Err("literal inference requires resolved type dimensions".into());
            }
        }
        Ok(())
    }
    visit(
        ty,
        &mut Vec::new(),
        operands,
        WidthLookup::DslxLiterals(Some(bindings)),
    )
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
            widths.members.insert(path.to_vec(), minimum);
        }
        Ok(())
    })?;
    Ok((!widths.members.is_empty()).then_some(widths))
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

/// Is the selected result type constrained by a match? Selectors, projections
/// and call arguments have their own groups and are checked independently.
fn has_match_result(expr: &Expr, path: &[Member]) -> bool {
    match (expr, path.split_first()) {
        (Expr::Match { .. }, _) => true,
        (
            Expr::If {
                then_value,
                else_value,
                ..
            },
            _,
        ) => has_match_result(then_value, path) || has_match_result(else_value, path),
        (Expr::Scope { result, .. }, _) => has_match_result(result, path),
        (Expr::Tuple(fields), Some((Member::TupleField(i), rest))) => fields
            .get(*i)
            .is_some_and(|field| has_match_result(field, rest)),
        (Expr::Array(items), Some((Member::ArrayElement, rest))) => {
            items.iter().any(|item| has_match_result(item, rest))
        }
        (
            Expr::Unary { value, .. }
            | Expr::Binary {
                op: BinaryOp::Shl | BinaryOp::Shr,
                left: value,
                ..
            },
            None,
        ) => has_match_result(value, path),
        (Expr::Binary { op, left, right }, None) if has_common_bit_type(*op) => {
            has_match_result(left, path) || has_match_result(right, path)
        }
        _ => false,
    }
}

/// Whether a match contributes to this expression's scalar type group.
pub(crate) fn has_scalar_match_result(expr: &Expr) -> bool {
    has_match_result(expr, &[])
}

/// When a match supplies an inferred result, invocations on the group's
/// common-type edges acquire their types after its unsized literals unify.
pub(crate) fn check_result_width(
    exprs: &[&Expr],
    ty: &Type,
    bindings: Option<WidthLookup>,
    zero_is_call: &dyn Fn() -> Result<bool, String>,
) -> Result<(), String> {
    if bits(ty).is_some_and(|(signed, _)| signed) && exprs.iter().any(|e| has_match_result(e, &[]))
    {
        check_match_operand_width(exprs, ty, bindings, zero_is_call)
    } else {
        check_inferred_signed_width(exprs, ty, bindings)
    }
}

/// Check the accumulated literal minimum against the first concrete signed
/// operand. Once two unsized literals have unified, their minimum loses the
/// radix prefix that lets an individual hex or binary literal spell a sign bit.
pub(crate) fn check_inferred_signed_width(
    exprs: &[&Expr],
    ty: &Type,
    bindings: Option<WidthLookup>,
) -> Result<(), String> {
    check_signed_operand_prefix(exprs, ty, bindings, MinimumContext::Concrete)
}

/// A direct `s8` argument anchors patterns immediately. With an `s8`
/// invocation as subject, `0 | 0xff` first merges two unsized minima,
/// losing the hexadecimal affordance and requiring nine signed bits.
pub(crate) fn check_match_operand_width(
    operands: &[&Expr],
    ty: &Type,
    bindings: Option<WidthLookup>,
    zero_is_call: &dyn Fn() -> Result<bool, String>,
) -> Result<(), String> {
    check_signed_operand_prefix(
        operands,
        ty,
        bindings,
        MinimumContext::Deferred { zero_is_call },
    )
}

fn check_signed_operand_prefix(
    exprs: &[&Expr],
    ty: &Type,
    bindings: Option<WidthLookup>,
    calls: MinimumContext<'_>,
) -> Result<(), String> {
    let Some((true, width)) = bits(ty) else {
        return Ok(());
    };
    let mut minimum = None;
    for expr in exprs {
        if !collect_minimum_with_calls(expr, bindings, &mut minimum, calls)? {
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
    zero_is_call: Option<&dyn Fn() -> Result<bool, String>>,
) -> Result<(), String> {
    visit_bit_members(ty, &mut Vec::new(), &mut |path, ty| {
        if let Some((true, width)) = bits(ty) {
            check_projected_argument(expr, path, ty, width, bindings, zero_is_call)?;
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
    zero_is_call: Option<&dyn Fn() -> Result<bool, String>>,
) -> Result<(), String> {
    match (expr, path.split_first()) {
        (Expr::Scope { result, .. }, _) => {
            return check_projected_argument(result, path, ty, width, None, zero_is_call);
        }
        (Expr::Field { value, name }, _) => {
            if let Ok(i) = name.parse() {
                let mut source_path = vec![Member::TupleField(i)];
                source_path.extend_from_slice(path);
                return check_projected_argument(value, &source_path, ty, width, bindings, None);
            }
        }
        (Expr::Index { value, .. }, _) => {
            let mut source_path = vec![Member::ArrayElement];
            source_path.extend_from_slice(path);
            return check_projected_argument(value, &source_path, ty, width, bindings, None);
        }
        (Expr::Tuple(items), Some((Member::TupleField(i), rest))) => {
            if let Some(item) = items.get(*i) {
                return check_projected_argument(item, rest, ty, width, bindings, zero_is_call);
            }
        }
        (Expr::Array(items), Some((Member::ArrayElement, rest))) if !rest.is_empty() => {
            for item in items {
                check_projected_argument(item, rest, ty, width, bindings, zero_is_call)?;
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
            check_projected_argument(then_value, path, ty, width, bindings, zero_is_call)?;
            check_projected_argument(else_value, path, ty, width, bindings, zero_is_call)?;
        }
        (Expr::Match { arms, .. }, Some(_)) => {
            for arm in arms {
                check_projected_argument(&arm.value, path, ty, width, bindings, zero_is_call)?;
            }
        }
        _ => { /* Scalar operands contribute to the common minimum below. */ }
    }
    let mut minimum = None;
    let calls = match zero_is_call {
        Some(zero_is_call) if has_match_result(expr, path) => {
            MinimumContext::Deferred { zero_is_call }
        }
        _ => MinimumContext::Concrete,
    };
    collect_projected_with_calls(
        expr,
        path,
        Some(WidthLookup::DslxLiterals(bindings)),
        &mut minimum,
        calls,
    )?;
    check_signed_minimum(minimum, ty, width)
}

/// Check each differing bit member against the minimum type retained by the
/// projection. At equal widths only a prefixed literal can acquire a sign bit.
pub(crate) fn projected_type_compatible(
    expr: &Expr,
    actual: &Type,
    expected: &Type,
    bindings: Option<&InferredBindings>,
    widen_in_scope: Option<usize>,
) -> Result<bool, String> {
    fn member_matches(
        expr: &Expr,
        path: &mut Vec<Member>,
        actual: &Type,
        expected: &Type,
        bindings: Option<&InferredBindings>,
        widen_in_scope: Option<usize>,
    ) -> Result<bool, String> {
        if actual == expected {
            return Ok(true);
        }
        match (actual, expected) {
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    path.push(Member::TupleField(index));
                    let matches = member_matches(expr, path, a, b, bindings, widen_in_scope)?;
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
                let matches = member_matches(expr, path, a, b, bindings, widen_in_scope)?;
                path.pop();
                Ok(matches)
            }
            _ => {
                let (Some((a_signed, a_width)), Some((b_signed, b_width))) =
                    (bits(actual), bits(expected))
                else {
                    return Ok(false);
                };
                if a_width > b_width
                    || (widen_in_scope.is_none() && a_width != b_width)
                    || a_signed && !b_signed
                {
                    return Ok(false);
                }
                if a_width == b_width && a_signed == b_signed {
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
                    && minimum.is_some_and(|m| {
                        if a_width < b_width {
                            widen_in_scope.is_some_and(|scope| m.can_widen_in(scope))
                        } else {
                            !m.signed && m.has_prefix && m.width == a_width
                        }
                    }))
            }
        }
    }
    member_matches(
        expr,
        &mut Vec::new(),
        actual,
        expected,
        bindings,
        widen_in_scope,
    )
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
        Expr::For(fold) if fold.annotation.is_none() => has_concrete_element_type(&fold.seed, None),
        Expr::If {
            then_value,
            else_value,
            ..
        } => Ok(has_concrete_element_type(then_value, locals)?
            || has_concrete_element_type(else_value, locals)?),
        Expr::Match { arms, .. } => {
            for arm in arms {
                if has_concrete_element_type(&arm.value, locals)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
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
