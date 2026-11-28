// SPDX-License-Identifier: Apache-2.0

//! Bind DSLX-style function value parametrics from constexpr actuals, argument
//! types and defaults, then typecheck each concrete function specialization.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use xlsynth_pir::{IrValue, ir_value_utils::ir_bits_to_usize};

use super::callables::CallbackSource;
use super::function_specializations::{
    BoundValueSignature, ExplicitEnvironment, InferredCallback, SuppliedParametrics,
};
use super::generic_blocks::{Bindings, callback_argument_name, validate_inference_sources};
use super::generic_inputs::{Input, InputPattern};
use super::struct_parametrics::{
    BoundStructArgument, StructApplication, StructLiteralError, struct_field_map,
};
use super::{
    BodyChecker, BodyKind, CallbackFormal, CallbackTarget, Declarations, FunctionInterface, K,
    MAX_ELABORATION_DEPTH, StructTable, TypeResolver, TypeShape, bits, child_type_budget,
    compatible, declare, literal_copy_work, minimum_type, resolve_type, source_type_nodes,
};
use crate::ast::{
    CallbackParam, Expr, Function, FunctionParametricRef, Param, ParametricBinding, SourceNode,
    StructArgument, StructArgumentRef, StructLiteralArgs, StructParametricRef, Type,
};
use crate::dslx_import::ImportResolver;
use crate::typed::{BoundParametricValue, TypedExpr};

pub(super) fn dimension_type() -> Type {
    Type::unsigned(32)
}

/// Total source storage copied from bound type parameters in one annotation.
#[derive(Default)]
pub(super) struct TypeArgumentCopies {
    nodes: usize,
    name_bytes: usize,
}

impl<'m> Declarations<'m> {
    /// Retain the declaration's symbolic signature until a call supplies its
    /// environment. Formal values occupy the same lexical namespace as data.
    pub(super) fn add_value_template(
        &mut self,
        f: &'m Function,
        structs: &StructTable,
    ) -> Result<(), String> {
        let mut names = BTreeSet::new();
        for name in &f.type_params {
            if !crate::parser::is_use_identifier(name) {
                return Err(format!("invalid type parameter name '{name}'"));
            }
            declare(&mut names, name)?;
        }
        let needs_result_type = !f.type_params.is_empty()
            && validate_inference_sources(&f.type_params, &f.params, &f.callback_params, None)
                .is_err();
        if needs_result_type {
            validate_inference_sources(
                &f.type_params,
                &f.params,
                &f.callback_params,
                Some(&f.result),
            )?;
        }
        let mut last = None;
        for p in &f.value_parametrics {
            if p.index >= f.value_parametrics.len() + f.callback_params.len()
                || last.is_some_and(|i| i >= p.index)
            {
                return Err("invalid value parametric declaration order".into());
            }
            last = Some(p.index);
            declare(&mut names, &p.name)?;
            match &p.ty {
                Type::Bits { .. } | Type::Bool | Type::Enum { .. } => {
                    structs.validate_type(&p.ty)?
                }
                Type::ParametricBits { .. }
                | Type::Named(_)
                | Type::Applied { .. }
                | Type::ParametricStruct { .. }
                | Type::MixedApplied { .. }
                | Type::TypeApplied { .. } => {
                    // Resolve dependent widths and nominal enum or struct
                    // annotations in the template's declaration scope.
                }
                _ => {
                    return Err(format!(
                        "value parametric '{}' requires a bits, bool, enum, or struct type",
                        p.name
                    ));
                }
            }
        }
        let mut callback_indices = BTreeMap::new();
        for (index, callback) in f.callback_params.iter().enumerate() {
            declare(&mut names, &callback.name)?;
            callback_indices.insert(callback.name.clone(), index);
        }
        for p in &f.params {
            declare(&mut names, &p.name)?;
        }
        if needs_result_type {
            self.result_context
                .insert(&f.name, super::ResultContext::TypeParameter);
        } else if !f.value_parametrics.is_empty() {
            // The specialization binds missing dimensions from data arguments
            // before it checks callback signatures.
            let mut dimensions = BTreeSet::new();
            for param in &f.params {
                super::type_dependencies(
                    &param.ty,
                    |_, _| {},
                    |expr| {
                        if let Expr::Name(name) = expr {
                            dimensions.insert(name.as_str());
                        }
                    },
                );
            }
            if let Some(formal) = f
                .value_parametrics
                .iter()
                .rev()
                .find(|p| p.default.is_none() && !dimensions.contains(p.name.as_str()))
            {
                self.result_context.insert(
                    &f.name,
                    super::ResultContext::ExplicitValues(formal.index + 1),
                );
            }
        }
        self.value_template_callbacks
            .insert(&f.name, callback_indices);
        self.function_templates.insert(&f.name, f);
        Ok(())
    }
}

/// Convert a bits or bool value to a host-sized dimension without truncating
/// high bits or interpreting a negative signed value as a width.
fn nonnegative_dimension(ty: &Type, value: &IrValue) -> Result<usize, String> {
    let (signed, _) =
        bits(ty).ok_or_else(|| format!("type dimension requires bits or bool, got {ty}"))?;
    let bits = value.as_bits().map_err(|e| e.to_string())?;
    if signed && bits.msb() {
        return Err("type dimension cannot be negative".into());
    }
    ir_bits_to_usize(bits).ok_or_else(|| "dimension does not fit usize".to_owned())
}

/// A bit width or sign can adopt a struct formal's type. Array lengths and
/// forwarded nominal bindings supply a value of a specific type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WitnessType {
    BitDimension,
    Exact,
}

/// Only a bare binding in a dimension provides an inference constraint. A
/// computed dimension is checked after its operands have concrete values.
pub(super) struct InferredBindings<'f> {
    formals: &'f [ParametricBinding],
    indices: BTreeMap<&'f str, usize>,
    pub(super) values: Vec<Option<BoundParametricValue>>,
    explicit_count: usize,
    // Struct bit widths and signedness provide unsized integer witnesses.
    // Array lengths and nested nominal arguments retain their source types.
    struct_bit_dimensions: Option<BTreeSet<usize>>,
    // Gather all witnesses from one data argument before its bindings become
    // context for later arguments. Tuple fields must agree with one another.
    witnesses: BTreeMap<usize, BoundParametricValue>,
}

impl<'f> InferredBindings<'f> {
    pub(super) fn new(formals: &'f [ParametricBinding], explicit_count: usize) -> Self {
        Self {
            formals,
            indices: formals
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.as_str(), i))
                .collect(),
            values: vec![None; formals.len()],
            explicit_count,
            struct_bit_dimensions: None,
            witnesses: BTreeMap::new(),
        }
    }

    pub(super) fn for_struct(formals: &'f [ParametricBinding], explicit_count: usize) -> Self {
        let mut bindings = Self::new(formals, explicit_count);
        bindings.struct_bit_dimensions = Some(BTreeSet::new());
        bindings
    }

    /// Give a bit-width or signedness witness its resolved declaration type.
    /// Other witnesses, including a forwarded nominal argument, must match
    /// exactly.
    pub(super) fn struct_value(
        &self,
        index: usize,
        ty: &Type,
    ) -> Result<Option<BoundParametricValue>, String> {
        let Some(bound) = self.values[index].as_ref() else {
            return Ok(None);
        };
        if !self
            .struct_bit_dimensions
            .as_ref()
            .is_some_and(|dims| dims.contains(&index))
        {
            return Ok(Some(bound.clone()));
        }
        let (signed, width) = bits(ty).ok_or_else(|| {
            format!(
                "parametric '{}' cannot infer {ty} from a bit dimension",
                self.formals[index].name
            )
        })?;
        let n = nonnegative_dimension(&bound.ty, &bound.value)?;
        let value = if signed {
            IrValue::make_sbits(
                width,
                i64::try_from(n).map_err(|_| "dimension exceeds i64")?,
            )
        } else {
            IrValue::make_ubits(width, n as u64)
        }
        .map_err(|_| {
            format!(
                "inferred dimension {n} does not fit {ty} for parametric '{}'",
                self.formals[index].name
            )
        })?;
        Ok(Some(BoundParametricValue {
            ty: ty.clone(),
            value,
        }))
    }

    fn missing(&self, dimension: &Expr) -> Option<usize> {
        let Expr::Name(name) = dimension else {
            return None;
        };
        let i = *self.indices.get(name.as_str())?;
        (self.formals[i].index >= self.explicit_count
            && self.values[i].is_none()
            && self.formals[i].default.is_none())
        .then_some(i)
    }

    fn needs_witness(&self, ty: &Type) -> bool {
        match ty {
            Type::ParametricBits { signed, width } => {
                self.missing(signed).is_some() || self.missing(width).is_some()
            }
            Type::ParametricArray { element, size } => {
                self.missing(size).is_some_and(|i| {
                    self.witnesses.get(&i).is_none_or(|previous| {
                        // An array length supplies an exact u32. A prior bit
                        // dimension or a differently typed nominal argument
                        // still needs to agree with that constraint.
                        self.struct_bit_dimensions.as_ref().is_some_and(|dims| {
                            dims.contains(&i) || previous.ty != dimension_type()
                        })
                    })
                }) || self.needs_witness(element)
            }
            Type::Array { element, .. } => self.needs_witness(element),
            Type::Tuple(fields) | Type::TypeApplied { args: fields, .. } => {
                fields.iter().any(|f| self.needs_witness(f))
            }
            Type::MixedApplied { args, .. } => args.iter().any(|arg| match arg {
                StructArgument::Type(ty) => self.needs_witness(ty),
                StructArgument::Value(expr) => self.missing(expr).is_some(),
            }),
            Type::ParametricStruct { args, .. } => {
                args.iter().any(|arg| self.missing(arg).is_some())
            }
            _ => false,
        }
    }

    /// Merge evidence from one field. A typed nominal argument can also
    /// constrain a binding that supplies a numeric bit width in that field.
    fn bind_witness(
        &mut self,
        dimension: &Expr,
        ty: &Type,
        value: IrValue,
        witness_type: WitnessType,
    ) -> Result<(), String> {
        let Some(index) = self.missing(dimension) else {
            return Ok(());
        };
        let contextual =
            self.struct_bit_dimensions.is_some() && witness_type == WitnessType::BitDimension;
        let witness = BoundParametricValue {
            ty: ty.clone(),
            value,
        };
        if let Some(previous) = self.witnesses.get(&index) {
            let previous_contextual = self
                .struct_bit_dimensions
                .as_ref()
                .is_some_and(|dims| dims.contains(&index));
            let equal = (compatible(&previous.ty, &witness.ty) && previous.value == witness.value)
                || ((previous_contextual || contextual)
                    && nonnegative_dimension(&previous.ty, &previous.value)
                        .ok()
                        .is_some_and(|n| {
                            nonnegative_dimension(&witness.ty, &witness.value) == Ok(n)
                        }));
            if !equal {
                return Err(format!(
                    "conflicting values inferred for parametric '{}': {} and {}",
                    self.formals[index].name, previous.value, witness.value
                ));
            }
            if contextual && !previous_contextual {
                return Ok(());
            }
        }
        if let Some(dims) = &mut self.struct_bit_dimensions {
            if contextual {
                dims.insert(index);
            } else {
                dims.remove(&index);
            }
        }
        self.witnesses.insert(index, witness);
        Ok(())
    }

    pub(super) fn finish_argument(&mut self) {
        for (index, value) in std::mem::take(&mut self.witnesses) {
            self.values[index] = Some(value);
        }
    }

    fn bind_size(
        &mut self,
        dimension: &Expr,
        size: usize,
        witness_type: WitnessType,
    ) -> Result<(), String> {
        if self.missing(dimension).is_some() {
            let value = IrValue::make_ubits(32, size as u64).map_err(|e| e.to_string())?;
            self.bind_witness(dimension, &dimension_type(), value, witness_type)?;
        }
        Ok(())
    }

    fn infer(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
        resolver: &ImportResolver,
    ) -> Result<(), String> {
        match (formal, actual) {
            (Type::ParametricBits { signed, width }, actual) => {
                if let Some((sign, count)) = bits(actual) {
                    self.bind_witness(
                        signed,
                        &Type::Bool,
                        IrValue::bool(sign),
                        WitnessType::BitDimension,
                    )?;
                    self.bind_size(width, count, WitnessType::BitDimension)?;
                }
            }
            (
                Type::ParametricArray { element, size },
                Type::Array {
                    element: actual,
                    size: count,
                },
            ) => {
                self.bind_size(size, *count, WitnessType::Exact)?;
                self.infer(element, actual, structs, resolver)?;
            }
            (
                Type::Array { element, .. },
                Type::Array {
                    element: actual, ..
                },
            ) => self.infer(element, actual, structs, resolver)?,
            (Type::Tuple(fields), Type::Tuple(actuals)) => {
                for (field, actual) in fields.iter().zip(actuals) {
                    self.infer(field, actual, structs, resolver)?;
                }
            }
            (formal, Type::Named(canonical))
                if let Some((name, args)) = formal.struct_application()
                    && !matches!(args, StructLiteralArgs::Integers(_)) =>
            {
                if let Some(instance) = structs.instances.get(canonical) {
                    if instance.template.name != name {
                        return Ok(());
                    }
                    for (arg, bound) in args.iter().zip(&instance.args) {
                        match (arg, bound) {
                            (StructArgumentRef::Type(ty), BoundStructArgument::Type(actual)) => {
                                self.infer(ty, actual, structs, resolver)?;
                            }
                            (StructArgumentRef::Value(expr), BoundStructArgument::Value(bound)) => {
                                self.bind_witness(
                                    expr,
                                    &bound.ty,
                                    bound.value.clone(),
                                    WitnessType::Exact,
                                )?;
                            }
                            _ if matches!(args, StructLiteralArgs::Mixed(_)) => {
                                return Err(format!("argument kind mismatch for struct '{name}'"));
                            }
                            _ => {
                                // Homogeneous lists supply witnesses only for
                                // their own kind.
                            }
                        }
                    }
                } else if let StructLiteralArgs::Expressions(args) = args
                    && let Some(values) = resolver.struct_parametric_values(name, canonical)?
                {
                    for (arg, bound) in args.iter().zip(values) {
                        self.bind_witness(arg, &bound.ty, bound.value.clone(), WitnessType::Exact)?;
                    }
                }
            }
            _ => { /* Concrete constraints are checked against the final signature. */ }
        }
        Ok(())
    }

    /// Gather array lengths and explicit constructor values while their
    /// contextual elements or fields await a type witness.
    pub(super) fn infer_literal_values(
        &mut self,
        pattern: &Type,
        expr: &Expr,
        checker: &mut BodyChecker<'_, '_>,
        remaining: usize,
    ) -> Result<(), String> {
        match (pattern, expr) {
            (_, Expr::Scope { result, .. }) if !expr.needs_scope_preparation() => {
                // Discarded operands remain attached for constexpr evaluation.
                // The captured trailing value supplies the dimension witness.
                self.infer_literal_values(pattern, result, checker, child_type_budget(remaining)?)?;
            }
            (
                Type::Array { element, .. } | Type::ParametricArray { element, .. },
                Expr::UnrollMap(map),
            ) => {
                if let Some(count) = checker.prepared_map_count(expr) {
                    if let Type::ParametricArray { size, .. } = pattern {
                        self.bind_size(size, count, WitnessType::Exact)?;
                    }
                    if count > 0
                        && self.needs_witness(element)
                        && let Some(result) = checker.map_first_result(map)?
                    {
                        self.infer_literal_values(
                            element,
                            &result,
                            checker,
                            child_type_budget(remaining)?,
                        )?;
                    }
                }
            }
            (
                Type::Array { .. } | Type::ParametricArray { .. },
                Expr::If {
                    then_value,
                    else_value,
                    ..
                },
            ) => {
                let depth = child_type_budget(remaining)?;
                self.infer_literal_values(pattern, then_value, checker, depth)?;
                self.infer_literal_values(pattern, else_value, checker, depth)?;
            }
            (Type::Array { .. } | Type::ParametricArray { .. }, Expr::Match { arms, .. }) => {
                let depth = child_type_budget(remaining)?;
                for arm in arms {
                    self.infer_literal_values(pattern, &arm.value, checker, depth)?;
                }
            }
            (Type::ParametricArray { size, element }, Expr::Array(items)) => {
                if let Some(first) = items.first() {
                    self.bind_size(size, items.len(), WitnessType::Exact)?;
                    self.infer_literal_values(
                        element,
                        first,
                        checker,
                        child_type_budget(remaining)?,
                    )?;
                }
            }
            (Type::Array { element, .. }, Expr::Array(items)) => {
                if let Some(first) = items.first() {
                    self.infer_literal_values(
                        element,
                        first,
                        checker,
                        child_type_budget(remaining)?,
                    )?;
                }
            }
            (Type::Tuple(fields), Expr::Tuple(items)) => {
                for (field, item) in fields.iter().zip(items) {
                    self.infer_literal_values(field, item, checker, child_type_budget(remaining)?)?;
                }
            }
            (Type::MixedApplied { name, args }, _) if self.needs_witness(pattern) => {
                let depth = child_type_budget(remaining)?;
                match expr {
                    Expr::If {
                        then_value,
                        else_value,
                        ..
                    } => {
                        self.infer_literal_values(pattern, then_value, checker, depth)?;
                        self.infer_literal_values(pattern, else_value, checker, depth)?;
                    }
                    Expr::Match { arms, .. } => {
                        for arm in arms {
                            self.infer_literal_values(pattern, &arm.value, checker, depth)?;
                        }
                    }
                    _ => {
                        let Some(literal) = expr.struct_literal().filter(|s| s.name == name) else {
                            return Ok(());
                        };
                        let Some(template) = checker.structs.template(name) else {
                            return Ok(());
                        };
                        // The length of S { lanes: [1, 2] } supplies N in S<T,
                        // N> even while T awaits a sibling witness. Do not
                        // force the unsized elements to adopt a width during
                        // this pass.
                        let count = match literal.args {
                            StructLiteralArgs::Omitted => 0,
                            StructLiteralArgs::Mixed(args) => args.len(),
                            _ => return Ok(()),
                        };
                        let mut nested =
                            InferredBindings::for_struct(&template.value_parametrics, count);
                        let needs_prefix = args.iter().take(count).any(|arg| match arg {
                            StructArgument::Value(expr) => self.missing(expr).is_some(),
                            StructArgument::Type(_) => false,
                        });
                        if needs_prefix {
                            // The final constructor check retains assertion
                            // operands. This pass only needs their constexpr
                            // values.
                            let operands = checker.assert_operands.take();
                            let prefix = checker.struct_prefix_at(
                                name,
                                literal.args,
                                remaining,
                                checker.function_position,
                            );
                            checker.assert_operands = operands;
                            let values = prefix?.into_iter().filter_map(|arg| match arg {
                                BoundStructArgument::Value(bound) => Some(bound),
                                BoundStructArgument::Type(_) => None,
                            });
                            for (slot, value) in nested.values.iter_mut().zip(values) {
                                *slot = Some(value);
                            }
                        }
                        if let Some(base) = literal.base {
                            self.infer_literal_values(pattern, &base.value, checker, depth)?;
                        } else {
                            let fields: BTreeMap<_, _> = literal
                                .fields
                                .iter()
                                .map(|(name, value)| (name.as_str(), value))
                                .collect();
                            for field in &template.fields {
                                if let Some(value) = fields.get(field.name.as_str()) {
                                    nested
                                        .infer_literal_values(&field.ty, value, checker, depth)?;
                                    nested.finish_argument();
                                }
                            }
                        }
                        for (formal, bound) in template.value_parametrics.iter().zip(nested.values)
                        {
                            if let (Some(StructArgument::Value(expr)), Some(bound)) =
                                (args.get(formal.index), bound)
                            {
                                self.bind_witness(
                                    expr,
                                    &bound.ty,
                                    bound.value,
                                    WitnessType::Exact,
                                )?;
                            }
                        }
                    }
                }
            }
            _ => { /* Nonliteral arguments provide their checked type instead. */ }
        }
        Ok(())
    }
}

/// A data operand may already have a type from inference or from map's element.
enum DataArgument<'e> {
    Source(&'e Expr),
    Checked(TypedExpr),
}

/// Declaration scope for the current field or function input during constructor
/// inference.
#[derive(Clone, Copy)]
struct ConstructorScope<'s> {
    owner: &'s str,
    /// Function inputs through the current parameter, or an empty slice for a
    /// struct field.
    params: &'s [Param],
}

/// Follow each required binding's annotation and default back to preceding
/// declarations. A later formal cannot shadow a name used by an earlier one.
fn required_bindings<'n>(
    roots: impl Iterator<Item = SourceNode<'n>>,
    declarations: impl DoubleEndedIterator<Item = (&'n str, &'n Type, Option<&'n Expr>)>,
) -> BTreeSet<&'n str> {
    let mut pending = roots
        .flat_map(SourceNode::free_names)
        .collect::<BTreeSet<_>>();
    let mut required = BTreeSet::new();
    for (name, ty, default) in declarations.rev() {
        if pending.remove(name) {
            required.insert(name);
            pending.extend(SourceNode::Annotation(ty).free_names());
            if let Some(default) = default {
                pending.extend(SourceNode::Expression(default).free_names());
            }
        }
    }
    required
}

/// Earlier explicit formals stay in declaration scope while the caller
/// evaluates each constexpr actual with the type of the next formal.
#[derive(Default)]
pub(super) struct ExplicitScope {
    pub(super) types: Option<Bindings>,
    names: BTreeSet<String>,
    values: BTreeMap<String, Type>,
    literals: BTreeMap<String, TypedExpr>,
    constexpr_names: BTreeSet<String>,
    constant_pattern_names: BTreeSet<String>,
}

impl ExplicitScope {
    /// Move the lexical maps between successive checks without copying the
    /// types or literals of every preceding formal at each step.
    pub(super) fn exchange(&mut self, checker: &mut BodyChecker<'_, '_>) {
        std::mem::swap(&mut self.types, &mut checker.function_type_bindings);
        std::mem::swap(&mut self.names, &mut checker.names);
        std::mem::swap(&mut self.values, &mut checker.values);
        std::mem::swap(&mut self.literals, &mut checker.function_values);
        std::mem::swap(&mut self.constexpr_names, &mut checker.constexpr_names);
        std::mem::swap(
            &mut self.constant_pattern_names,
            &mut checker.constant_pattern_names,
        );
    }

    fn evaluate<'m, T>(
        &mut self,
        caller: &mut BodyChecker<'_, 'm>,
        f: &'m Function,
        preceding_value: Option<(&str, &BoundParametricValue)>,
        evaluate: impl FnOnce(&mut BodyChecker<'_, 'm>) -> Result<T, String>,
    ) -> Result<T, String> {
        caller.with_value_definition(&f.name, |definition| {
            self.exchange(definition);
            if let Some((name, value)) = preceding_value {
                definition.bind_parametric_value(name, value);
            }
            let result = evaluate(definition);
            self.exchange(definition);
            result
        })
    }
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// Apply the selected parametric signature to unsized expressions and
    /// materialized references in the caller's lexical scope.
    pub(super) fn with_parametric_context<T>(
        &mut self,
        check: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        self.with_argument_context(true, check)
    }

    /// A fixed formal can supply signedness at an existing width. Parametric
    /// inference can also widen eligible bindings to a specialized width.
    pub(super) fn with_argument_context<T>(
        &mut self,
        parametric: bool,
        check: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let saved = std::mem::replace(&mut self.parametric_argument_context, parametric);
        let result = self.with_dslx_value_context(check);
        self.parametric_argument_context = saved;
        result
    }

    pub(super) fn bind_parametric_value(&mut self, name: &str, bound: &BoundParametricValue) {
        self.names.insert(name.to_owned());
        self.constant_pattern_names.insert(name.to_owned());
        self.values.insert(name.to_owned(), bound.ty.clone());
        self.constexpr_names.insert(name.to_owned());
        self.function_values.insert(
            name.to_owned(),
            TypedExpr {
                ty: bound.ty.clone(),
                kind: K::Literal(bound.value.clone()),
            },
        );
    }

    /// Evaluate in the active lexical scope with the binding's declared type.
    pub(super) fn parametric_value(&mut self, expr: &Expr, ty: &Type) -> Result<IrValue, String> {
        let retain = self.assert_operands.is_some();
        let (constant, operand) = self.parametric_value_and_operand(expr, ty, retain)?;
        self.record_assert_operand(operand);
        Ok(constant)
    }

    /// Fold an actual while optionally keeping its typed expression and
    /// constexpr-only lexical lets for an operand erased during lowering.
    pub(super) fn parametric_value_and_operand(
        &mut self,
        expr: &Expr,
        ty: &Type,
        retain_operand: bool,
    ) -> Result<(IrValue, TypedExpr), String> {
        let had_functions = self.constant_functions.is_some();
        let first = self.statements.len();
        let callees = std::mem::take(&mut self.local_callees);
        let outer_erased = self.in_erased_operand;
        self.in_erased_operand |= retain_operand;
        self.enable_constant_functions();
        let result = self.with_parametric_context(|checker| {
            let value = checker.dslx_argument(expr, ty)?;
            let constant = checker
                .checked_constant_for(&value, checker.dslx_constant_use)?
                .ok_or_else(|| "value parametric requires a compile-time constant".to_owned())?;
            Ok((constant, value))
        });
        // Scope elaboration may materialize lets while computing an actual
        // or dimension. Their values have been consumed at compile time.
        if !retain_operand || result.is_err() {
            self.discard_temporary_statements(first);
        }
        self.in_erased_operand = outer_erased;
        self.local_callees = callees;
        if !had_functions {
            self.constant_functions = None;
        }
        result
    }

    /// Whether checking this operand may require a witness from the constructor
    /// itself.
    pub(super) fn constructor_binding_pending(&self, source: SourceNode<'_>) -> bool {
        self.probing_constructor_types
            && source.free_names().any(|name| {
                (self.names.contains(name) && !self.constexpr_names.contains(name))
                    || self
                        .type_bindings()
                        .is_some_and(|types| types.declares(name) && types.get(name).is_none())
            })
    }

    /// An intermediate binding may depend on the very dimension its
    /// constructor will establish. Retain successful partial evaluations and
    /// leave unresolved declarations shadowing their module names.
    fn probe_constructor_binding<T>(
        &mut self,
        source: SourceNode<'_>,
        resolve: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        match resolve(self) {
            Ok(bound) => Ok(Some(bound)),
            Err(_) if self.constructor_binding_pending(source) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// A directly named struct value parameter can supply a dimension at its
    /// declared width. Computed dimensions use the usual bool or u32 context.
    fn struct_dimension(&self, expr: &Expr) -> Result<Option<usize>, String> {
        self.direct_struct_binding(expr)
            .map(|(ty, value)| nonnegative_dimension(ty, value))
            .transpose()
    }

    pub(super) fn resolve_dimensions(
        &mut self,
        ty: &Type,
        remaining: usize,
    ) -> Result<Type, String> {
        self.resolve_annotation(ty, remaining, &mut None)
    }

    /// Substitute each written type parameter once, at its source reference.
    /// Already concrete names inside an alias or another argument retain their
    /// original identity even if a formal has the same spelling.
    pub(super) fn concrete_annotation(&mut self, ty: &Type) -> Result<Type, String> {
        let mut copies = Some(TypeArgumentCopies::default());
        self.resolve_annotation(ty, MAX_ELABORATION_DEPTH, &mut copies)
    }

    /// Resolve dimensions and aliases, keeping formals symbolic when checking
    /// an inference pattern. Concrete annotations budget all copied formals
    /// together before cloning their bound types.
    pub(super) fn resolve_annotation(
        &mut self,
        ty: &Type,
        remaining: usize,
        copies: &mut Option<TypeArgumentCopies>,
    ) -> Result<Type, String> {
        let size = |checker: &mut Self, expr: &Expr| {
            if let Some(n) = checker.struct_dimension(expr)? {
                return Ok(n);
            }
            let value = checker.parametric_value(expr, &dimension_type())?;
            ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
                .ok_or_else(|| "dimension does not fit usize".to_owned())
        };
        match ty {
            Type::Named(name) if self.local_type_aliases.contains_key(name) => {
                let alias = &self.local_type_aliases[name].ty;
                self.charge_reference_copy(super::source_type_nodes(alias))?;
                Ok(alias.clone())
            }
            Type::TypeApplied { name, args } => {
                if self.has_local_name(name)
                    || self
                        .callback_indices()
                        .is_some_and(|indices| indices.contains_key(name))
                {
                    return Err(format!("'{name}' names a local value here"));
                }
                self.reject_type_parameter_value(name)?;
                self.decls.check_type_name_visibility(
                    name,
                    self.function_position.or(self.block_position),
                )?;
                let template = self
                    .structs
                    .template(name)
                    .ok_or_else(|| format!("unknown type-parametric struct '{name}'"))?;
                super::struct_parametrics::check_type_arity(template, args.len())?;
                let budget = child_type_budget(remaining)?;
                let args = args
                    .iter()
                    .map(|arg| self.resolve_annotation(arg, budget, copies))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_type_resolver(|types, dimensions, bindings| {
                    let bindings = if copies.is_some() { None } else { bindings };
                    types.resolve_struct_type_application(
                        name, args, bindings, dimensions, remaining,
                    )
                })
            }
            Type::MixedApplied { name, args } => {
                if copies.is_none()
                    && self
                        .type_bindings()
                        .is_some_and(|b| b.has_unbound_variable(ty))
                {
                    let depth = child_type_budget(remaining)?;
                    let args = args
                        .iter()
                        .map(|arg| match arg {
                            StructArgument::Type(ty) => self
                                .resolve_annotation(ty, depth, copies)
                                .map(StructArgument::Type),
                            StructArgument::Value(_) => Ok(arg.clone()),
                        })
                        .collect::<Result<_, _>>()?;
                    return Ok(Type::MixedApplied {
                        name: name.clone(),
                        args,
                    });
                }
                let args =
                    self.struct_prefix_at(name, StructLiteralArgs::Mixed(args), remaining, None)?;
                self.struct_applied_type(name, &args, remaining)
            }
            Type::ParametricStruct { name, args } => {
                let args = self.struct_prefix(name, StructLiteralArgs::Expressions(args))?;
                self.struct_applied_type(name, &args, remaining)
            }
            Type::ParametricBits { signed, width } => {
                let signed = match self.struct_dimension(signed)? {
                    Some(0) => false,
                    Some(1) => true,
                    Some(n) => return Err(format!("bit signedness must be 0 or 1, got {n}")),
                    None => self
                        .parametric_value(signed, &Type::Bool)?
                        .to_bool()
                        .map_err(|e| e.to_string())?,
                };
                Ok(Type::Bits {
                    signed,
                    width: size(self, width)?,
                })
            }
            Type::ParametricArray {
                element,
                size: expr,
            } => Ok(Type::array(
                self.resolve_annotation(element, child_type_budget(remaining)?, copies)?,
                size(self, expr)?,
            )),
            Type::Array { element, size } => Ok(Type::array(
                self.resolve_annotation(element, child_type_budget(remaining)?, copies)?,
                *size,
            )),
            Type::Tuple(fields) => {
                let depth = child_type_budget(remaining)?;
                fields
                    .iter()
                    .map(|t| self.resolve_annotation(t, depth, copies))
                    .collect::<Result<_, _>>()
                    .map(Type::Tuple)
            }
            _ => {
                let callbacks = self.callback_indices();
                let resolved = resolve_type(
                    ty,
                    &|name| {
                        self.names.contains(name)
                            || self
                                .enclosing_type_locals
                                .is_some_and(|contains| contains(name))
                            || callbacks.is_some_and(|indices| indices.contains_key(name))
                    },
                    self.function_type_bindings
                        .as_ref()
                        .or_else(|| self.specialization.map(|s| &s.bindings)),
                    self.decls.dimensions_at(
                        self.function_position.or(self.block_position),
                        &self.zero_binding,
                        self.warning_owner,
                    ),
                    &mut TypeResolver {
                        resolver: self.resolver,
                        structs: self.structs,
                        functions: self.function_registry.as_deref_mut(),
                    },
                    remaining,
                )?;
                if let Some(copies) = copies
                    && let Type::Named(name) = ty
                    && let Some(bindings) = self.type_bindings()
                    && let Some(shape) = bindings.shape(name)
                {
                    let bound = bindings
                        .get(name)
                        .ok_or_else(|| format!("cannot infer type parameter '{name}'"))?;
                    copies.nodes = copies.nodes.saturating_add(shape.source_nodes);
                    copies.name_bytes = copies.name_bytes.saturating_add(shape.name_bytes);
                    TypeShape::check_source_budget(copies.nodes, copies.name_bytes)?;
                    return Ok(bound.clone());
                }
                Ok(resolved)
            }
        }
    }

    pub(super) fn value_function_call(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: &[Expr],
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        for arg in args {
            crate::infer_literal::check_argument_array_sizes(arg)?;
        }
        self.with_parametric_context(|checker| {
            checker.with_scope_aliases(|checker| {
                // Keep branch-local and aggregate-member bindings visible
                // through both witness collection and application of the final
                // signature.
                let prepared = args
                    .iter()
                    .map(|expr| checker.prepare_function_operand(expr, !f.type_params.is_empty()))
                    .collect::<Result<Vec<_>, _>>()?;
                let args = prepared
                    .iter()
                    .map(|expr| DataArgument::Source(expr.as_ref()))
                    .collect();
                checker.instantiate_value_function(f, actuals, args, expected)
            })
        })
    }

    pub(super) fn value_map_call_with_actuals(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        element: TypedExpr,
    ) -> Result<TypedExpr, String> {
        if !f.callback_params.is_empty() || actuals.len() > f.value_parametrics.len() {
            return Err(format!(
                "invalid parametric arguments for map callback '{}'",
                f.name
            ));
        }
        let (ty, kind) = self.instantiate_value_function(
            f,
            actuals,
            vec![DataArgument::Checked(element)],
            None,
        )?;
        Ok(TypedExpr { ty, kind })
    }

    /// Evaluate a declaration's annotations and defaults with its own imports
    /// and earlier module declarations, even when called from a different
    /// scope.
    pub(super) fn with_value_definition<T>(
        &mut self,
        owner: &'m str,
        evaluate: impl FnOnce(&mut BodyChecker<'_, 'm>) -> Result<T, String>,
    ) -> Result<T, String> {
        let registry = self
            .function_registry
            .as_deref_mut()
            .ok_or_else(|| format!("cannot specialize '{owner}' in this type annotation"))?;
        registry.with_definition(owner, |registry| {
            let mut definition =
                BodyChecker::new(self.decls, self.resolver, self.structs, BodyKind::Function);
            definition.function_position = self
                .decls
                .positions
                .get(owner)
                .map(|at| at.saturating_sub(1));
            definition.warning_owner = Some(owner);
            definition.zero_binding = self.decls.zero_binding(owner);
            definition.function_registry = Some(registry);
            evaluate(&mut definition)
        })
    }

    /// Check an explicitly supplied value or evaluate its default in the
    /// declaration scope containing earlier bindings.
    pub(super) fn value_or_default(
        &mut self,
        owner: &str,
        formal: &ParametricBinding,
        ty: Type,
        supplied: Option<&BoundParametricValue>,
    ) -> Result<BoundParametricValue, String> {
        self.names.insert(formal.name.clone());
        Ok(match (supplied, &formal.default) {
            (Some(bound), default) => {
                if !compatible(&ty, &bound.ty) {
                    return Err(format!(
                        "parametric '{}' expects {ty}, got {}",
                        formal.name, bound.ty
                    ));
                }
                if let Some(expr) = default {
                    // An overridden default still has to typecheck. The
                    // formal has its supplied value inside its own default.
                    self.bind_parametric_value(&formal.name, bound);
                    self.with_parametric_context(|c| c.dslx_argument(expr, &ty))
                        .map_err(|e| format!("default for '{}': {e}", formal.name))?;
                }
                BoundParametricValue {
                    ty,
                    value: bound.value.clone(),
                }
            }
            (None, Some(expr)) => {
                let value = self
                    .parametric_value(expr, &ty)
                    .map_err(|e| format!("default for '{}': {e}", formal.name))?;
                BoundParametricValue { ty, value }
            }
            (None, None) => {
                return Err(format!(
                    "cannot infer value parametric '{}' of '{}'",
                    formal.name, owner
                ));
            }
        })
    }

    /// Bind defaults in declaration order, then resolve each runtime parameter
    /// and the result in the completed value environment.
    fn bound_value_signature(
        &mut self,
        f: &'m Function,
        provided: &[Option<BoundParametricValue>],
    ) -> Result<BoundValueSignature, String> {
        let mut values = Vec::with_capacity(f.value_parametrics.len());
        let mut callbacks = Vec::with_capacity(f.callback_params.len());
        // One inferred tuple can expand at every formal. Bound the retained
        // signature incrementally, before resolving the next annotation.
        let charge = |checker: &mut Self, work| {
            checker
                .function_registry
                .as_deref_mut()
                .expect("module function registry")
                .charge_bindings(work)
        };
        for formal in f.parametric_bindings() {
            let formal = match formal {
                FunctionParametricRef::Value(formal) => formal,
                FunctionParametricRef::Callback(formal) => {
                    let mut binding_work = 0;
                    let mut resolve = |source: &Type| -> Result<Type, String> {
                        let ty = self.source_type(source)?;
                        let nodes = source_type_nodes(&ty);
                        charge(self, nodes)?;
                        binding_work += nodes;
                        Ok(ty)
                    };
                    let params = formal
                        .params
                        .iter()
                        .map(&mut resolve)
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = resolve(&formal.result)?;
                    callbacks.push(CallbackFormal {
                        signature: CallbackParam {
                            name: formal.name.clone(),
                            params,
                            result,
                        },
                        binding_work,
                    });
                    // Callback targets are selected after this signature has
                    // been resolved. The formal still hides
                    // the outer declaration.
                    self.names.insert(formal.name.clone());
                    continue;
                }
            };
            let supplied = &provided[values.len()];
            let ty = self.value_binding_type(formal)?;
            let bound = self.value_or_default(&f.name, formal, ty, supplied.as_ref())?;
            charge(self, 1 + literal_copy_work(&bound.value))?;
            self.bind_parametric_value(&formal.name, &bound);
            values.push(bound);
        }
        let mut params = Vec::new();
        for p in &f.params {
            params.push(self.function_runtime_formal(p)?);
        }
        let result = self.source_type(&f.result)?;
        Ok(BoundValueSignature {
            values,
            type_bindings: self
                .function_type_bindings
                .take()
                .unwrap_or_else(|| Bindings::new(&[])),
            interface: FunctionInterface {
                params,
                result,
                callback_params: callbacks,
                callback_indices: self.decls.value_template_callbacks[f.name.as_str()].clone(),
                local: true,
            },
        })
    }

    /// Charge each resolved data input before retaining its type in the
    /// specialization scope.
    fn function_runtime_formal(&mut self, formal: &Param) -> Result<Param, String> {
        self.resolve_runtime_formal(formal, |checker, source| {
            let ty = checker.source_type(source)?;
            checker
                .function_registry
                .as_deref_mut()
                .expect("module function registry")
                .charge_bindings(source_type_nodes(&ty))?;
            Ok(ty)
        })
    }

    /// An inferred annotation on a constexpr formal must select one of the
    /// value types accepted for an explicitly annotated binding.
    fn value_binding_type(&mut self, formal: &ParametricBinding) -> Result<Type, String> {
        let ty = self.source_type(&formal.ty)?;
        if let Type::Named(name) = &formal.ty
            && self
                .type_bindings()
                .is_some_and(|bindings| bindings.declares(name))
            && !matches!(
                ty,
                Type::Bits { .. } | Type::Bool | Type::Enum { .. } | Type::Named(_)
            )
        {
            return Err(format!(
                "value parametric '{}' requires a bits, bool, enum, or struct type, got {ty}",
                formal.name
            ));
        }
        Ok(ty)
    }

    /// Explicit actuals and input types come from the caller. Defaults and
    /// formal annotations are evaluated in the function's definition scope.
    fn specialized_signature(
        &mut self,
        f: &'m Function,
        types: Bindings,
        values: Vec<Option<BoundParametricValue>>,
    ) -> Result<Rc<BoundValueSignature>, String> {
        let supplied = SuppliedParametrics {
            types: types.arguments(&f.type_params)?,
            values,
        };
        let registry = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry");
        if let Some(bound) = registry.cached_value_signature(f, &supplied)? {
            return Ok(bound);
        }
        let signature = self.with_value_definition(&f.name, |definition| {
            definition.function_type_bindings = (!f.type_params.is_empty()).then_some(types);
            definition.bound_value_signature(f, &supplied.values)
        })?;
        self.function_registry
            .as_deref_mut()
            .expect("module function registry")
            .cache_value_signature(f, supplied, signature)
    }

    /// An unsized scalar in a tuple or struct supplies its minimum width.
    /// An unsized scalar passed directly to a function can wait for context.
    fn dimension_witness(
        &mut self,
        formal: &Type,
        expr: &Expr,
        inferred: &mut InferredBindings<'_>,
        tuple_member: bool,
    ) -> Result<Option<TypedExpr>, String> {
        if !inferred.needs_witness(formal)
            || self.prepared_map_count(expr).is_some() && !self.has_concrete_element_type(expr)?
        {
            return Ok(None);
        }
        let value = if self.contains_type_struct_constructor(expr) {
            let Some(value) = self.constructor_dimension_witness(expr)? else {
                return Ok(None);
            };
            value
        } else if tuple_member {
            self.inferred_expr(expr)?
        } else if minimum_type(expr)?.is_none() {
            self.expr(expr, None)?
        } else {
            return Ok(None);
        };
        inferred.infer(formal, &value.ty, self.structs, self.resolver)?;
        Ok(Some(value))
    }

    /// Tuple fields have independent widths. Checking a field that supplies a
    /// dimension must leave the other fields free to use their inferred types.
    pub(super) fn infer_type_input<'e>(
        &mut self,
        formal: &Type,
        expr: &'e Expr,
        types: &mut Bindings,
        inferred: &mut InferredBindings<'_>,
        tuple_member: bool,
    ) -> Result<Input<'e>, String> {
        if inferred.needs_witness(formal)
            && let (Type::Tuple(fields), Expr::Tuple(items)) = (formal, expr)
            && fields.len() == items.len()
        {
            let members = fields
                .iter()
                .zip(items)
                .map(|(ty, item)| {
                    self.infer_type_input(ty, item, types, inferred, /* tuple_member= */ true)
                })
                .collect::<Result<_, _>>()?;
            return Ok(Input::Aggregate {
                expr,
                members,
                charged_fill: 0,
            });
        }
        let witness = self.dimension_witness(formal, expr, inferred, tuple_member)?;
        self.infer_input(formal, expr, types, InputPattern::Function(witness))
    }

    /// Gather dimension witnesses from the same field across array rows or
    /// conditional arms. Other fields receive the completed signature later.
    fn dimension_input_group<'e>(
        &mut self,
        formal: &Type,
        exprs: &[&'e Expr],
        inferred: &mut InferredBindings<'_>,
        aggregate_member: bool,
    ) -> Result<Vec<Input<'e>>, String> {
        if exprs.is_empty() || !inferred.needs_witness(formal) {
            return Ok(exprs
                .iter()
                .map(|&expr| Input::Contextual {
                    expr,
                    prepared: None,
                })
                .collect());
        }
        if exprs
            .iter()
            .any(|expr| matches!(expr, Expr::If { .. } | Expr::Match { .. }))
            && exprs.iter().any(|expr| super::has_aggregate_literal(expr))
        {
            let (selectors, branches) = self.result_branches(exprs)?;
            let mut inputs = self
                .dimension_input_group(formal, &branches, inferred, aggregate_member)?
                .into_iter();
            return Ok(exprs
                .iter()
                .zip(selectors)
                .map(|(&expr, selector)| {
                    let count = match expr {
                        Expr::If { .. } => 2,
                        Expr::Match { arms, .. } => arms.len(),
                        _ => return inputs.next().expect("one input for an unbranched operand"),
                    };
                    Input::Branches {
                        expr,
                        selector: Some(selector),
                        results: inputs.by_ref().take(count).collect(),
                    }
                })
                .collect());
        }
        if let Type::Tuple(fields) = formal
            && let Some(rows) = exprs
                .iter()
                .map(|expr| match expr {
                    Expr::Tuple(items) if items.len() == fields.len() => Some(items),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            let mut members: Vec<Vec<Input>> = rows
                .iter()
                .map(|_| Vec::with_capacity(fields.len()))
                .collect();
            for (index, field) in fields.iter().enumerate() {
                let column = rows.iter().map(|row| &row[index]).collect::<Vec<_>>();
                let inputs = self.dimension_input_group(
                    field, &column, inferred, /* aggregate_member= */ true,
                )?;
                for (row, input) in members.iter_mut().zip(inputs) {
                    row.push(input);
                }
            }
            return Ok(exprs
                .iter()
                .zip(members)
                .map(|(&expr, members)| Input::Aggregate {
                    expr,
                    members,
                    charged_fill: 0,
                })
                .collect());
        }
        if let Type::Array { element, .. } | Type::ParametricArray { element, .. } = formal
            && let Some(rows) = exprs
                .iter()
                .map(|expr| expr.unannotated_array())
                .collect::<Option<Vec<_>>>()
        {
            if let Type::ParametricArray { size, .. } = formal {
                for (row, fill) in &rows {
                    if !fill && !row.is_empty() {
                        inferred.bind_size(size, row.len(), WitnessType::Exact)?;
                    }
                }
            }
            let column = rows
                .iter()
                .flat_map(|(items, _)| items.iter())
                .collect::<Vec<_>>();
            let mut inputs = self
                .dimension_input_group(
                    element, &column, inferred, /* aggregate_member= */ true,
                )?
                .into_iter();
            return Ok(exprs
                .iter()
                .zip(rows)
                .map(|(&expr, (row, _))| {
                    if row.is_empty() {
                        // The complete element type is needed to materialize an
                        // empty array.
                        Input::Contextual {
                            expr,
                            prepared: None,
                        }
                    } else {
                        Input::Aggregate {
                            expr,
                            members: inputs.by_ref().take(row.len()).collect(),
                            charged_fill: 0,
                        }
                    }
                })
                .collect());
        }
        let mut needs_sibling = true;
        for expr in exprs {
            let contextual = if self.prepared_map_count(expr).is_some() {
                !self.has_concrete_element_type(expr)?
            } else {
                crate::infer_literal::requires_sibling_type(
                    expr,
                    self.common_widths(),
                    &|name, count| self.call_needs_result(name, count),
                )?
            };
            if !contextual {
                needs_sibling = false;
                break;
            }
        }
        if needs_sibling {
            return Ok(exprs
                .iter()
                .map(|&expr| Input::Contextual {
                    expr,
                    prepared: None,
                })
                .collect());
        }
        if let [expr] = exprs {
            let witness = self.dimension_witness(formal, expr, inferred, aggregate_member)?;
            return Ok(vec![Input::Contextual {
                expr,
                prepared: witness,
            }]);
        }
        let witnesses = self.infer_constructors(exprs, /* infer_literal_widths= */ true)?;
        let mut witnesses = witnesses.into_iter().flatten().peekable();
        if let Some(value) = witnesses.peek() {
            // Sibling checking has given the entire column one type.
            inferred.infer(formal, &value.ty, self.structs, self.resolver)?;
        }
        Ok(exprs
            .iter()
            .map(|&expr| Input::Contextual {
                expr,
                prepared: witnesses.next(),
            })
            .collect())
    }

    /// Check each data argument for witnesses in source order, retaining its
    /// checked leaves for application of the complete function signature.
    fn infer_function_inputs<'e>(
        &mut self,
        f: &Function,
        args: Vec<DataArgument<'e>>,
        types: &mut Bindings,
        inferred: &mut InferredBindings<'_>,
    ) -> Result<Vec<Input<'e>>, String> {
        let mut inputs = Vec::with_capacity(args.len());
        for (arg, param) in args.into_iter().zip(&f.params) {
            let input = match arg {
                DataArgument::Source(expr) => {
                    inferred.infer_literal_values(&param.ty, expr, self, MAX_ELABORATION_DEPTH)?;
                    let mut input = self
                        .dimension_input_group(
                            &param.ty,
                            &[expr],
                            inferred,
                            /* aggregate_member= */ false,
                        )?
                        .pop()
                        .expect("one input per argument");
                    if !f.type_params.is_empty() {
                        // Dimension witnesses share a column; type parameters
                        // bind in the original order of the caller's rows and
                        // fields.
                        input.visit_leaves(&param.ty, &mut |formal, leaf| {
                            if let Input::Contextual { expr, prepared } = leaf {
                                *leaf = self.infer_input(
                                    formal,
                                    expr,
                                    types,
                                    InputPattern::Function(prepared.take()),
                                )?;
                            }
                            Ok(())
                        })?;
                    }
                    input
                }
                DataArgument::Checked(value) => {
                    types.infer_source_type(
                        &param.ty,
                        &value.ty,
                        self.structs,
                        /* contextual_projection= */ false,
                    )?;
                    inferred.infer(&param.ty, &value.ty, self.structs, self.resolver)?;
                    Input::Checked { expr: None, value }
                }
            };
            inferred.finish_argument();
            inputs.push(input);
        }
        Ok(inputs)
    }

    /// Apply the bound signature to remaining operands and reuse the witnesses.
    fn finish_function_inputs(
        &mut self,
        f: &Function,
        inputs: Vec<Input>,
        params: &[Param],
    ) -> Result<Vec<TypedExpr>, String> {
        let mut checked = Vec::with_capacity(inputs.len());
        for (input, param) in inputs.into_iter().zip(params) {
            if let Some(expr) = input.source() {
                self.check_dslx_argument_widths(expr, &param.ty)?;
            }
            checked
                .push(self.finish_input(input, &param.ty).map_err(|e| {
                    format!("argument '{}', call to '{}': {e}", param.name, f.name)
                })?);
        }
        Ok(checked)
    }

    /// A contextual constructor can supply a defaulted value once other data
    /// or callbacks have determined the types used to evaluate that default.
    pub(super) fn infer_contextual_defaults<'p>(
        &mut self,
        inputs: &mut [Input<'_>],
        params: impl IntoIterator<Item = &'p Param>,
        owner: &str,
        types: &Bindings,
        inferred: &mut InferredBindings<'_>,
    ) -> Result<(), String> {
        if inferred.values.iter().all(Option::is_some) {
            return Ok(());
        }
        let function = self.decls.function_templates.get(owner).copied();
        for (index, (input, param)) in inputs.iter_mut().zip(params).enumerate() {
            let scope = ConstructorScope {
                owner,
                params: function.map_or(&[][..], |f| &f.params[..=index]),
            };
            input.visit_leaves(&param.ty, &mut |pattern, leaf| {
                if let Input::Contextual {
                    expr,
                    prepared: None,
                } = leaf
                {
                    self.infer_constructor_default(
                        pattern,
                        expr,
                        scope,
                        types,
                        inferred,
                        MAX_ELABORATION_DEPTH,
                    )?;
                }
                Ok(())
            })?;
            inferred.finish_argument();
        }
        Ok(())
    }

    /// Follow the actual's nominal type through a prepared scope, branch, or
    /// update. Only an omitted default can supply a new value witness here.
    fn infer_constructor_default(
        &mut self,
        pattern: &Type,
        expr: &Expr,
        scope: ConstructorScope<'_>,
        types: &Bindings,
        inferred: &mut InferredBindings<'_>,
        remaining: usize,
    ) -> Result<(), String> {
        let Type::MixedApplied { name, args } = pattern else {
            return Ok(());
        };
        let depth = child_type_budget(remaining)?;
        match expr {
            Expr::Scope { result, .. } if !expr.needs_scope_preparation() => {
                return self
                    .infer_constructor_default(pattern, result, scope, types, inferred, depth);
            }
            Expr::If {
                then_value,
                else_value,
                ..
            } => {
                self.infer_constructor_default(pattern, then_value, scope, types, inferred, depth)?;
                return self
                    .infer_constructor_default(pattern, else_value, scope, types, inferred, depth);
            }
            Expr::Match { arms, .. } => {
                for arm in arms {
                    self.infer_constructor_default(
                        pattern, &arm.value, scope, types, inferred, depth,
                    )?;
                }
                return Ok(());
            }
            _ => { /* A constructor can supply its own specialization. */ }
        }
        let Some(literal) = expr.struct_literal().filter(|s| s.name == name) else {
            return Ok(());
        };
        if let Some(base) = literal.base {
            return self.infer_constructor_default(
                pattern,
                &base.value,
                scope,
                types,
                inferred,
                depth,
            );
        }
        let Some(template) = self.structs.template(name) else {
            return Ok(());
        };
        let count = literal.args.len();
        let needs_default = template.value_parametrics.iter().any(|param| {
            param.index >= count
                && param.default.is_some()
                && matches!(args.get(param.index), Some(StructArgument::Value(expr)) if inferred.missing(expr).is_some())
        });
        if !needs_default {
            return Ok(());
        }
        // The pattern's type arguments belong to the enclosing formal,
        // including its imports and previously inferred dimensions.
        let mut child_types = Bindings::new(&template.type_params);
        {
            let mut definition = BodyChecker::struct_type_scope(
                self.decls.module_dimensions(scope.owner),
                self.resolver,
                self.structs,
                self.function_registry.as_deref_mut(),
            );
            definition.in_struct_definition = definition.structs.template(scope.owner).is_some();
            definition.function_type_bindings = Some(types.clone());
            let context_types = template
                .parametric_bindings()
                .zip(args)
                .skip(count)
                .filter_map(|(formal, arg)| {
                    if let (StructParametricRef::Type(name), StructArgument::Type(ty)) =
                        (formal, arg)
                        && !types.has_unbound_variable(ty)
                        && !inferred.needs_witness(ty)
                    {
                        Some((name, ty))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            definition.bind_constructor_input_types(scope, &context_types, inferred)?;
            for (name, ty) in context_types {
                let actual = definition.source_type(ty)?;
                child_types.match_input(
                    &Type::Named(name.to_owned()),
                    &actual,
                    definition.structs,
                )?;
            }
        }
        if count == 0
            && !template
                .type_params
                .iter()
                .any(|p| child_types.get(p).is_some())
        {
            // An earlier unconstrained probe already tried these fields.
            return Ok(());
        }
        // The final field check in the caller keeps its own statements
        // and assertion operands. Only the nominal type is a witness.
        let start = self.statements.len();
        let operands = self.assert_operands.take();
        let result = (|| {
            let prefix = self.struct_prefix(name, literal.args)?;
            let source = StructApplication::Applied(name, &prefix);
            let row = struct_field_map(source, literal.fields)?;
            self.infer_struct_rows(source, template, [&row], child_types)
        })();
        self.assert_operands = operands;
        self.discard_temporary_statements(start);
        match result {
            Ok((ty, _)) => inferred.infer(pattern, &ty, self.structs, self.resolver),
            Err(StructLiteralError::Unbound(_)) => Ok(()),
            Err(error) => Err(error.into_message()),
        }
    }

    /// Resolve only the earlier inputs and omitted defaults needed by these
    /// type arguments. Other inputs may still depend on the witness this
    /// constructor is supplying.
    fn bind_constructor_input_types(
        &mut self,
        scope: ConstructorScope<'_>,
        context_types: &[(&str, &Type)],
        inferred: &InferredBindings<'_>,
    ) -> Result<(), String> {
        if context_types.is_empty() {
            return Ok(());
        }
        let Some((current, earlier)) = scope.params.split_last() else {
            for (formal, bound) in inferred.formals.iter().zip(&inferred.values) {
                if let Some(bound) = bound {
                    self.bind_parametric_value(&formal.name, bound);
                }
            }
            return Ok(());
        };
        let missing = inferred
            .formals
            .iter()
            .zip(&inferred.values)
            .filter(|(_, value)| value.is_none());
        let required = required_bindings(
            context_types
                .iter()
                .map(|(_, ty)| SourceNode::Annotation(ty)),
            missing
                .map(|(formal, _)| (formal.name.as_str(), &formal.ty, formal.default.as_ref()))
                .chain(
                    earlier
                        .iter()
                        .map(|param| (param.name.as_str(), &param.ty, None)),
                ),
        );
        let function = self.decls.function_templates[scope.owner];
        self.probing_constructor_types = true;
        let mut values = inferred.values.iter();
        for formal in function.parametric_bindings() {
            let formal = match formal {
                FunctionParametricRef::Callback(callback) => {
                    self.names.insert(callback.name.clone());
                    continue;
                }
                FunctionParametricRef::Value(formal) => formal,
            };
            self.names.insert(formal.name.clone());
            if let Some(bound) = values.next().expect("one slot per value formal") {
                self.bind_parametric_value(&formal.name, bound);
                continue;
            }
            if required.contains(formal.name.as_str())
                && let Some(ty) = self
                    .probe_constructor_binding(SourceNode::Annotation(&formal.ty), |checker| {
                        checker.value_binding_type(formal)
                    })?
            {
                if let Some(default) = &formal.default
                    && let Some(bound) = self
                        .probe_constructor_binding(SourceNode::Expression(default), |checker| {
                            checker.value_or_default(scope.owner, formal, ty.clone(), None)
                        })?
                {
                    self.bind_parametric_value(&formal.name, &bound);
                    continue;
                }
                self.values.insert(formal.name.clone(), ty);
            }
        }
        for param in earlier {
            self.names.insert(param.name.clone());
            if required.contains(param.name.as_str()) {
                self.probe_constructor_binding(SourceNode::Annotation(&param.ty), |checker| {
                    checker.function_runtime_formal(param)
                })?;
            }
        }
        self.names.insert(current.name.clone());
        Ok(())
    }

    /// Infer type variables from the result context, data and callbacks before
    /// checking contextual arguments and the specialized body.
    fn instantiate_type_function(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: Vec<DataArgument<'_>>,
        expected: Option<&Type>,
        mut inferred: InferredBindings<'_>,
    ) -> Result<(Type, K), String> {
        let mut types = Bindings::new(&f.type_params);
        if let Some(result) = expected {
            types.infer_source_type(
                &f.result,
                result,
                self.structs,
                /* contextual_projection= */ false,
            )?;
        }
        let mut inputs = self.infer_function_inputs(f, args, &mut types, &mut inferred)?;
        self.infer_input_groups(&mut inputs, &f.params, &mut types, &format!("'{}'", f.name))?;
        let (types, targets) = self.infer_explicit_formals(f, types, actuals, &mut inferred)?;
        self.infer_contextual_defaults(&mut inputs, &f.params, &f.name, &types, &mut inferred)?;
        let bound = self.specialized_signature(f, types, inferred.values)?;
        let checked = self.finish_function_inputs(f, inputs, &bound.interface.params)?;
        let result = bound.interface.result.clone();
        let name = self.intern_function(f, targets, Some(bound))?;
        Ok((
            result,
            K::Call {
                name,
                args: checked,
            },
        ))
    }

    /// Visit explicit formals in order. A callback's return can determine
    /// later callback inputs, constexpr annotations, and the function result.
    fn infer_explicit_formals(
        &mut self,
        f: &'m Function,
        types: Bindings,
        actuals: &[Expr],
        inferred: &mut InferredBindings<'_>,
    ) -> Result<(Bindings, Vec<CallbackTarget<'m>>), String> {
        if actuals.is_empty() {
            return Ok((types, Vec::new()));
        }
        let mut environment = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry")
            .explicit_environment(ExplicitEnvironment::DataTypes(
                &f.name,
                types.partial_arguments(),
            ))?;
        let mut scope = ExplicitScope {
            types: Some(types),
            ..Default::default()
        };
        let mut targets = Vec::with_capacity(f.callback_params.len());
        let mut preceding_value: Option<(&str, &BoundParametricValue)> = None;
        let mut explicit_values = inferred.values.iter_mut();
        for (index, (formal, actual)) in f.parametric_bindings().zip(actuals).enumerate() {
            let prior = preceding_value.take();
            match formal {
                FunctionParametricRef::Value(value) => {
                    let ty = scope.evaluate(self, f, prior, |d| {
                        if let Some(ty) = d
                            .function_registry
                            .as_deref()
                            .expect("module function registry")
                            .explicit_value_type(environment)
                        {
                            return Ok(ty);
                        }
                        let ty = d.value_binding_type(value)?;
                        d.function_registry
                            .as_deref_mut()
                            .expect("module function registry")
                            .record_explicit_value_type(environment, &ty);
                        Ok(ty)
                    })?;
                    let actual = self
                        .parametric_value(actual, &ty)
                        .map_err(|e| format!("parametric '{}': {e}", value.name))?;
                    let slot = explicit_values
                        .next()
                        .expect("value formal in declaration order");
                    let bound = slot.insert(BoundParametricValue { ty, value: actual });
                    if index + 1 < actuals.len() {
                        environment = self
                            .function_registry
                            .as_deref_mut()
                            .expect("module function registry")
                            .explicit_environment(ExplicitEnvironment::Value(
                                environment,
                                bound.clone(),
                            ))?;
                    }
                    preceding_value = Some((&value.name, bound));
                }
                FunctionParametricRef::Callback(callback) => {
                    let name = callback_argument_name(actual)?;
                    let source = self.callback_source(name, |head| self.has_local_name(head))?;
                    environment = self
                        .function_registry
                        .as_deref_mut()
                        .expect("module function registry")
                        .explicit_environment(ExplicitEnvironment::Callback(
                            environment,
                            source.clone(),
                        ))?;
                    let target = scope
                        .evaluate(self, f, prior, |d| {
                            d.infer_callback_result(callback, &source, environment)
                        })
                        .map_err(|e| {
                            format!(
                                "callable parameter '{}' of '{}': {e}",
                                callback.name, f.name
                            )
                        })?;
                    targets.push(target);
                }
            }
        }
        Ok((scope.types.expect("generic function bindings"), targets))
    }

    /// A callback's concrete result binds variables in its return pattern
    /// after its input annotations resolve in the template's lexical scope.
    fn infer_callback_result(
        &mut self,
        callback: &CallbackParam,
        source: &CallbackSource<'m>,
        environment: usize,
    ) -> Result<CallbackTarget<'m>, String> {
        let bound = match self
            .function_registry
            .as_deref()
            .expect("module function registry")
            .inferred_callback(environment)
        {
            Some(bound) => bound,
            None => {
                let bound = Rc::new(self.validate_inferred_callback(callback, source)?);
                self.function_registry
                    .as_deref_mut()
                    .expect("module function registry")
                    .record_inferred_callback(environment, Rc::clone(&bound));
                bound
            }
        };
        let registry = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry");
        self.function_type_bindings
            .as_mut()
            .expect("generic function bindings")
            .merge_callback_results(&bound.results, |nodes| registry.charge_bindings(nodes))?;
        self.names.insert(callback.name.clone());
        Ok(bound.target)
    }

    /// Resolve the dependent annotations once and retain only the bindings
    /// contributed by the callback's result.
    fn validate_inferred_callback(
        &mut self,
        callback: &CallbackParam,
        source: &CallbackSource<'m>,
    ) -> Result<InferredCallback<'m>, String> {
        let mut params = Vec::with_capacity(callback.params.len());
        for ty in &callback.params {
            let ty = self.source_type(ty)?;
            self.charge_expansion("callback bindings", source_type_nodes(&ty))?;
            params.push(ty);
        }
        let pattern = self.resolve_dimensions(&callback.result, MAX_ELABORATION_DEPTH)?;
        self.structs.update_imports(self.resolver)?;
        self.structs
            .formal_shape(&pattern, MAX_ELABORATION_DEPTH, self.type_bindings())?;
        let input_nodes = if matches!(source, CallbackSource::Imported { .. }) {
            Some(params.iter().try_fold(0usize, |sum, ty| {
                self.structs
                    .type_shape(ty, MAX_ELABORATION_DEPTH)
                    .map(|shape| sum.saturating_add(shape.ir_nodes))
            })?)
        } else {
            None
        };
        let (target, result) = self.callback_result(&params, source, input_nodes)?;
        self.charge_expansion("callback bindings", source_type_nodes(&result))?;
        let bindings = self
            .function_type_bindings
            .as_ref()
            .expect("generic function bindings");
        let mut results = Bindings::new(&bindings.referenced_parameters([&pattern]));
        results.match_input(&pattern, &result, self.structs)?;
        if super::has_mixed_application(&pattern) {
            let mut completed = bindings.clone();
            completed.merge_callback_results(&results, |work| {
                self.function_registry
                    .as_deref_mut()
                    .expect("module function registry")
                    .charge_bindings(work)
            })?;
            let previous = self.function_type_bindings.replace(completed);
            let expected = self.source_type(&callback.result);
            self.function_type_bindings = previous;
            let expected = expected?;
            if !compatible(&expected, &result) {
                return Err(format!(
                    "callback '{}' returns {result}, expected {expected}",
                    callback.name
                ));
            }
        }
        Ok(InferredCallback { target, results })
    }

    /// Bind a local template to its actuals and emit a call to its concrete
    /// symbol.
    fn instantiate_value_function(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: Vec<DataArgument<'_>>,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        if self.function_registry.is_none() {
            return Err(format!(
                "cannot specialize function '{}' in this type annotation",
                f.name
            ));
        }
        if args.len() != f.params.len() {
            return Err(format!(
                "'{}' expects {} arguments, got {}",
                f.name,
                f.params.len(),
                args.len()
            ));
        }
        let mut inferred = InferredBindings::new(&f.value_parametrics, actuals.len());
        if let Some(result) = expected {
            inferred.infer(&f.result, result, self.structs, self.resolver)?;
            inferred.finish_argument();
        }
        if !f.type_params.is_empty() {
            return self.instantiate_type_function(f, actuals, args, expected, inferred);
        }
        if !actuals.is_empty() {
            // Check explicit annotations before instantiating the callee.
            // Preceding formals hide module declarations even though none of
            // their values are available yet in this definition scope.
            let types = self.with_value_definition(&f.name, |definition| {
                let mut types = Vec::new();
                for formal in f.parametric_bindings().take(actuals.len()) {
                    let name = match formal {
                        FunctionParametricRef::Value(value) => {
                            types.push(definition.source_type(&value.ty)?);
                            &value.name
                        }
                        FunctionParametricRef::Callback(callback) => &callback.name,
                    };
                    definition.names.insert(name.clone());
                }
                Ok(types)
            })?;
            for (i, (formal, ty)) in f.value_parametrics.iter().zip(types).enumerate() {
                let value = self
                    .parametric_value(&actuals[formal.index], &ty)
                    .map_err(|e| format!("parametric '{}': {e}", formal.name))?;
                inferred.values[i] = Some(BoundParametricValue { ty, value });
            }
        }
        let mut types = Bindings::new(&[]);
        let mut inputs = self.infer_function_inputs(f, args, &mut types, &mut inferred)?;
        self.infer_contextual_defaults(&mut inputs, &f.params, &f.name, &types, &mut inferred)?;
        let bound = self.specialized_signature(f, types, inferred.values)?;
        let interface = &bound.interface;
        let mut targets = Vec::new();
        for (i, formal) in f.parametric_bindings().enumerate() {
            if let FunctionParametricRef::Callback(_) = formal {
                let actual = actuals
                    .get(i)
                    .ok_or_else(|| format!("'{}' requires explicit callable arguments", f.name))?;
                let name = super::generic_blocks::callback_argument_name(actual)?;
                let source = self.callback_source(name, |head| self.has_local_name(head))?;
                let formal = &interface.callback_params[targets.len()];
                if self.replaying_body() {
                    self.charge_expansion("callback bindings", formal.binding_work)?;
                }
                let signature = &formal.signature;
                let (target, result) = self.callback_result(&signature.params, &source, None)?;
                if !compatible(&result, &signature.result) {
                    return Err(format!(
                        "callback '{name}' returns {result}, expected {}",
                        signature.result
                    ));
                }
                targets.push(target);
            }
        }
        let checked = self.finish_function_inputs(f, inputs, &interface.params)?;
        let result = interface.result.clone();
        let name = self.intern_function(f, targets, Some(bound))?;
        Ok((
            result,
            K::Call {
                name,
                args: checked,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Bindings, BodyChecker, BodyKind, Declarations, StructTable, Type};
    use crate::check::{FunctionSpecializations, ImportResolver, MAX_TYPE_SIZE};
    use crate::{ImportOptions, check_module, parse_module};
    use xlsynth_pir::IrValue;

    #[test]
    fn repeated_type_formals_stop_signature_construction_at_the_expansion_limit()
    -> Result<(), String> {
        let module = parse_module("fn repeat<T>(a:T,b:T)->T { a }")?;
        let f = &module.functions[0];
        let mut structs = StructTable::default();
        let mut decls = Declarations::default();
        decls.add_value_template(f, &structs)?;
        let mut types = Bindings::new(&f.type_params);
        let tuple = Type::Tuple(vec![Type::Bool, Type::Bool]);
        types.match_input(&f.params[0].ty, &tuple, &structs)?;
        let mut resolver = ImportResolver::new(&[], &ImportOptions::default())?;
        let mut registry = FunctionSpecializations::new(Default::default());
        // The first tuple fits. Resolving the second must stop construction.
        registry.expansion_work = MAX_TYPE_SIZE - 3;
        let mut checker = BodyChecker::new(&decls, &mut resolver, &mut structs, BodyKind::Function);
        checker.function_registry = Some(&mut registry);
        checker.function_type_bindings = Some(types);
        let error = checker.bound_value_signature(f, &[]).err();
        assert_eq!(
            error,
            Some(format!(
                "parametric bindings exceed the function specialization expansion limit of {MAX_TYPE_SIZE}"
            ))
        );
        Ok(())
    }

    #[test]
    fn callback_results_type_later_explicit_values_and_callbacks() -> Result<(), String> {
        let source = r#"
fn widen(x:u8)->u16 { x as u16 }
fn add(a:u16,b:u16)->u16 { a+b }
fn apply<T,R,F:fn(T)->R,V:R,G:fn(R,R)->R>(x:T)->R { G(F(x),V) }
const ANSWER = apply<widen,37,add>(u8:5);
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::make_ubits(16, 42).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn nested_array_literal_supplies_a_length_while_its_element_type_awaits_context()
    -> Result<(), String> {
        let source = r#"
struct S<T, N:u32> { values:T[N] }
struct Outer<N:u32,T> { item:S<T,N> }
fn first<T,N:u32>(outer:Outer<N,T>, other:T)->T { outer.item.values[u32:0] }
fn sample(x:u8)->u8 { first(Outer { item:S { values:[1] } }, x) }
const RESULT = sample(u8:42);
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::make_ubits(8, 1).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn partial_struct_prefix_supplies_values_before_a_sibling_binds_the_type() -> Result<(), String>
    {
        let source = r#"
struct S<N:u32,T> { xs:T[N] }
struct W<N:u32,T> { item:S<N,T>, anchor:T }
struct Tag<U,N:U,T> { x:T }
fn read<N:u32,T>(s:S<N,T>, anchor:T)->T { s.xs[u32:1] }
fn tag_value<V:u8,T>(s:Tag<u8,V,T>, anchor:T)->u8 { V }
fn sample(t:u8)->(u8,u8,u8,u8) {
    const NUM = u32:2;
    let from_prefix = read(S<NUM>{xs:[1,2]}, t);
    let from_base = read(S{..S<2>{xs:[3,4]}}, t);
    let outer = W{item:S<2>{xs:[5,6]}, anchor:t};
    let phantom = tag_value(Tag<u8,7>{x:1}, t);
    (from_prefix, from_base, outer.item.xs[u32:1], phantom)
}
const RESULT = sample(u8:0);
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[8]:2, bits[8]:4, bits[8]:6, bits[8]:7)")
                .map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn contextual_constructor_evaluates_a_default_after_other_types_bind() -> Result<(), String> {
        let source = r#"
struct S<T, N:u32={bit_count<T>()}> { x:T }
struct W<T, N:u32> { inner:S<T,N>, witness:T }
fn read<T, N:u32>(p:S<T,N>, y:T)->u32 { N }
fn read_reversed<T, N:u32>(y:T, p:S<T,N>)->u32 { N }
fn read_fixed<N:u32>(p:S<u8,N>)->u32 { N }
fn outer<T, N:u32>(p:W<T,N>)->u32 { N }
fn sample(y:u8)->(u32,u32,u32,u32) {
    let a = read(S{x:1}, y);
    let b = read_reversed(y, S{..S{x:1}});
    let c = outer(W{inner:S{x:1}, witness:y});
    let d = read_fixed(S{x:1});
    (a,b,c,d)
}
const RESULT = sample(u8:42);
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[32]:8, bits[32]:8, bits[32]:8, bits[32]:8)")
                .map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn a_constructor_witness_can_precede_bindings_in_an_untaken_dimension_arm() -> Result<(), String>
    {
        let source = r#"
const A = u32:12;
const pending = u32:19;
struct S<T, W:u32={bit_count<T>()}> { item:T }
fn defaulted<M:u32, A:u32={M}>(y:S<uN[if true { u32:5 } else { A }], M>) -> u32 { A }
fn runtime<M:u32>(pending:uN[M], y:S<uN[if false { pending as u32 } else { u32:7 }], M>) -> u32 { M }
fn nested<M:u32, N:u32={u32:3}, D:u32={if true { N+u32:1 } else { M }}>(
    pending:uN[M],
    y:S<uN[(if D == u32:4 { D } else { let x = pending as u32; x }) + u32:1], M>
) -> u32 { M }
const RESULT = (defaulted(S{item:1}), runtime(1, S{item:1}), nested(1, S{item:1}));
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants().last().unwrap().value,
            IrValue::parse_typed("(bits[32]:5, bits[32]:7, bits[32]:5)")
                .map_err(|e| e.to_string())?
        );
        for (invalid, expected_error) in [
            (
                source.replace("if true { u32:5 }", "if false { u32:5 }"),
                "compile-time constant",
            ),
            (
                source.replace("if false { pending as u32 }", "if true { pending as u32 }"),
                "unknown value 'pending'",
            ),
            (
                source.replace(
                    "if false { pending as u32 }",
                    "if false { pending as bool }",
                ),
                "expected u32, got bool",
            ),
        ] {
            let error = check_module(&parse_module(&invalid)?).unwrap_err();
            assert!(error.contains(expected_error), "{error}");
        }
        Ok(())
    }

    #[test]
    fn constructor_witness_folds_only_required_predicates() -> Result<(), String> {
        let source = r#"
enum E:u2 { Z = u2:0 }
struct S<T, W:u32={bit_count<T>()}> { item:T }
fn read<M:u32>(y:S<uN[if true { u32:5 } else {
    if (u2:3 as E) == E::Z { u32:1 } else { u32:2 }
}], M>) -> u32 { M }
fn sibling<M:u32, D:u32={M}>(y:S<uN[(if true { 5 } else {
    if (u2:3 as E) == E::Z { D } else { u32:2 }
}) as u32], M>) -> u32 { M }
const RESULT = (read(S{item:1}), sibling(S{item:1}));
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[32]:5, bits[32]:5)").map_err(|e| e.to_string())?
        );
        let required = source.replace("if true { u32:5 }", "if false { u32:5 }");
        let error = check_module(&parse_module(&required)?).unwrap_err();
        assert!(error.contains("constant cast to enum 'E'"), "{error}");
        Ok(())
    }

    #[test]
    fn a_constructor_dimension_cast_preserves_its_branch_operand_width() -> Result<(), String> {
        let source = r#"
const D = u5:0;
struct S<T, W:u32={bit_count<T>()}> { item:T }
fn typed<M:u32, D:uN[M]={0}>(y:S<uN[(if true { u5:5 } else { D }) as u32], M>) -> u32 { M }
fn implicit<M:u32, D:u3={M as u3}>(y:S<uN[(!(if true { 0 } else { D })) as u32], M>) -> u32 { M }
fn scoped<M:u32, D:u3={M as u3}>(y:S<uN[(!(if false { D } else { let x = 0; x })) as u32], M>) -> u32 { M }
const R = (typed(S{item:1}), implicit(S{item:1}), scoped(S{item:1}));
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants().last().unwrap().value,
            IrValue::parse_typed("(bits[32]:5, bits[32]:7, bits[32]:7)")
                .map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn constructor_default_inference_uses_the_earlier_function_parameters() -> Result<(), String> {
        let source = r#"
struct S<T, M:u32={bit_count<T>()}> { item:T }
fn width<N:u32={u32:4}, M:u32>(
    pending:uN[M],
    previous:u32,
    x:uN[if N == u32:4 { u32:32 } else { previous }],
    y:S<uN[if N == u32:4 { u32:5 } else { x }], M>
) -> u32 { M }
const RESULT = width(1, u32:9, u32:7, S{item:1});
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::make_ubits(32, 5).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn a_generic_block_type_can_supply_only_a_struct_value_argument() -> Result<(), String> {
        let source = r#"
struct Tag<U, N:u32> { data:U }
fn wrap(x:u8)->Tag<u8,8> { Tag { data:x } }
#[allow(combinational_block)]
block unpack<T>(clk, x:T, tag:Tag<u8,{bit_count<T>()}>)->u8 { tag.data }
#[allow(combinational_block)]
block call<T, U, F:fn(u8)->Tag<U,{bit_count<T>()}>>(clk, x:T, arg:u8)->U { F(arg).data }
pub block sample(clk, x:u8, y:u16)->(u8,u8) {
    let a = unpack(x,wrap(x));
    let b = call<wrap>(x,x);
    (a,b)
}
"#;
        check_module(&parse_module(source)?)?;
        // Reuse the callback with the same data signature. The distinct T
        // changes the required phantom argument and therefore the cache key.
        let mismatch = source.replace("(a,b)", "let c = call<wrap>(y,x); (a,c)");
        let error = check_module(&parse_module(&mismatch)?).unwrap_err();
        assert!(
            error.contains("returns Tag<u8, 8>, expected Tag<u8, 16>"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn mixed_constructor_siblings_share_the_typed_field() -> Result<(), String> {
        let source = r#"
struct S<T,N:u32> { values:T[N] }
fn sample(x:u8) -> (u8,u8) {
    let branch = if x == u8:1 { S { values:[x] } } else { S { values:[255] } };
    let array = [S { values:[x] }, S { values:[255] }];
    (branch.values[u32:0], array[u32:1].values[u32:0])
}
const RESULT = sample(u8:42);
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[8]:255, bits[8]:255)").map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn callback_result_infers_a_mixed_struct_type_and_checks_its_phantom_value()
    -> Result<(), String> {
        let declarations = r#"
struct Tag<T, N:u32> { data:T }
fn wrap(x:u8)->Tag<u8,3> { Tag { data:x } }
"#;
        for caller in [
            r#"
fn apply<T, F:fn(u8)->Tag<T,3>>(x:u8)->T { F(x).data }
const RESULT = apply<wrap>(u8:42);
"#,
            r#"
#[allow(combinational_block)]
block apply<T, F:fn(u8)->Tag<T,3>>(clk, x:u8)->T { F(x).data }
pub block sample(clk, x:u8)->u8 { let result = apply<wrap>(x); result }
"#,
        ] {
            let source = format!("{declarations}{caller}");
            check_module(&parse_module(&source)?)?;
            let mismatch = source.replace("Tag<T,3>", "Tag<T,4>");
            let error = check_module(&parse_module(&mismatch)?).unwrap_err();
            assert!(
                error.contains("returns Tag<u8, 3>, expected Tag<u8, 4>"),
                "{error}"
            );
        }
        Ok(())
    }

    #[test]
    fn aggregate_arguments_share_complementary_type_witnesses() -> Result<(), String> {
        let source = r#"
fn pick<T>(a:T,b:T)->T { a }
const ANSWER = pick((u4:1,2), (3,u8:4));
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[4]:1,bits[8]:2)").map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn generic_functions_preserve_contextual_aggregate_signedness() -> Result<(), String> {
        let source = r#"
fn width<T,N:u32>(t:T,x:sN[N][1])->u32 { N }
fn fixed<T>(t:T,x:(s4,s4))->u32 { u32:4 }
fn mixed<T,N:u32>(x:(T,sN[N][1]))->u32 { N }
fn test()->(u32,u32,u32,u32,u32) {
    let a = [0xf];
    let t = (0xf,0xf);
    let m = (u8:0,[0xf]);
    (width<4>(u8:0,a), fixed(u8:0,t), mixed(m),
     width(u8:0,if true { a } else { a }),
     mixed(match false { true => m, _ => m }))
}
const ANSWER = test();
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[32]:4,bits[32]:4,bits[32]:4,bits[32]:4,bits[32]:4)")
                .map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn generic_function_dimensions_use_each_arguments_original_witness() -> Result<(), String> {
        let source = r#"
fn leading<T,N:u32>(t:T,x:uN[N],xs:uN[N][1])->u32 { N }
fn trailing<T,N:u32>(t:T,xs:uN[N][1],x:uN[N])->u32 { N }
fn nested<T,N:u32>(p:(T,uN[N]),x:uN[N])->u32 { N }
fn tuple_width<T,N:u32>(p:(T,uN[N]))->u32 { N }
"#;
        let accepted =
            format!("{source} const WIDTH = (leading(u8:0,u8:3,[1]),tuple_width((u8:0,1)));");
        assert_eq!(
            check_module(&parse_module(&accepted)?)?.constants()[0].value,
            IrValue::parse_typed("(bits[32]:8,bits[32]:1)").map_err(|e| e.to_string())?
        );
        for call in ["trailing(u8:0,[1],u8:3)", "nested((u8:0,1),u8:3)"] {
            let rejected = format!("{source} const WIDTH = {call};");
            let err = check_module(&parse_module(&rejected)?).unwrap_err();
            assert!(err.contains("expected u1, got u8"), "{call}: {err}");
        }
        Ok(())
    }

    #[test]
    fn value_formals_keep_their_supported_types_after_inference() -> Result<(), String> {
        let source = "fn f<T,V:T>(x:T)->T { V } fn test()->u8[1] { f<{[u8:0]}>([u8:1]) }";
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(
            error.contains("requires a bits, bool, enum, or struct type"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn mixed_formals_resolve_callback_types_in_declaration_order() -> Result<(), String> {
        let source = r#"
const N = u32:8;
fn step(x:u8)->u8{x}
fn f<F:fn(uN[N])->uN[N],N:u32>(x:u8)->u8{F(x)}
fn sample(x:u8)->u8{f<step,3>(x)}
"#;
        check_module(&parse_module(source)?)?;
        Ok(())
    }

    #[test]
    fn a_callable_formal_hides_globals_in_defaults_and_dimensions() -> Result<(), String> {
        let definitions = "fn F(x:u32)->u32{x+u32:100} fn step(x:u32)->u32{x+u32:1}";
        for (signature, call) in [
            (
                "fn f<F:fn(u32)->u32,N:u32={F(u32:1)}>()->u32{N}",
                "f<step>()",
            ),
            (
                "fn f<F:fn(u32)->u32,N:u32={F(u32:1)}>()->u32{N}",
                "f<step,2>()",
            ),
            (
                "fn f<F:fn(u32)->u32,N:u32>(x:uN[F(N)])->u32{x as u32}",
                "f<step,1>(uN[101]:1)",
            ),
            (
                "fn f<F:fn(u32)->u32,N:uN[F(u32:1)]>()->u32{N as u32}",
                "f<step,uN[101]:1>()",
            ),
        ] {
            let source = format!("{definitions}\n{signature}\nfn sample()->u32{{{call}}}");
            let error = check_module(&parse_module(&source)?).unwrap_err();
            assert!(error.contains("'F' names a local value"), "{error}");
        }
        Ok(())
    }
}
