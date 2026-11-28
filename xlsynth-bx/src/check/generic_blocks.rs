// SPDX-License-Identifier: Apache-2.0

//! Inference of concrete BX types for generic block input ports.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;

use crate::ast::{
    Block, CallbackParam, Expr, FunctionParametricRef, Param, SourceNode, StructArgument,
    StructArgumentRef, StructLiteralArgs, Type,
};
use crate::parser::{is_identifier, is_use_identifier};
use crate::typed::BoundParametricValue;

use super::module_constants::BlockDimensionFormals;
use super::struct_parametrics::{BoundStructArgument, references_formal};
use super::{
    BlockCallbackFormal, BlockPortLayout, BlockTemplate, BodyChecker, MAX_ELABORATION_DEPTH,
    ModuleDimensions, StructTable, TypeResolver, TypeShape, compatible, resolve_signature,
    resolve_type, source_type_nodes, type_parameter_references,
};

/// A callable argument identifies a function in the caller's lexical scope.
pub(super) fn callback_argument_name(expr: &Expr) -> Result<&str, String> {
    match expr {
        Expr::Name(name) => Ok(name),
        _ => Err("callable argument must name a function".into()),
    }
}

/// Check the template's parameter namespace and ensure every type parameter
/// can be inferred from data inputs and then callback results, in order.
pub(super) fn validate_declaration(block: &Block) -> Result<(), String> {
    let mut type_params = BTreeSet::new();
    for name in &block.type_params {
        if !is_identifier(name) {
            return Err(format!("invalid type parameter name '{name}'"));
        }
        if !type_params.insert(name.as_str()) {
            return Err(format!("duplicate type parameter '{name}'"));
        }
    }
    let mut callback_names = BTreeSet::new();
    for callback in &block.callback_params {
        let name = &callback.name;
        if !is_use_identifier(name) {
            return Err(format!("invalid callable parameter name '{name}'"));
        }
        if type_params.contains(name.as_str()) || !callback_names.insert(name.as_str()) {
            return Err(format!("duplicate block parameter '{name}'"));
        }
    }
    let mut value_names = BTreeSet::new();
    let explicit_count = block.value_parametrics.len() + block.callback_params.len();
    let mut previous_index = None;
    for parametric in &block.value_parametrics {
        let name = parametric.name.as_str();
        if !is_identifier(name) {
            return Err(format!("invalid value parametric name '{name}'"));
        }
        if type_params.contains(name) || callback_names.contains(name) || !value_names.insert(name)
        {
            return Err(format!("duplicate block parameter '{name}'"));
        }
        if parametric.index >= explicit_count
            || previous_index.is_some_and(|i| i >= parametric.index)
        {
            return Err(format!(
                "invalid argument position for block value parametric '{name}'"
            ));
        }
        previous_index = Some(parametric.index);
    }
    for name in block
        .clock
        .iter()
        .chain(&block.reset)
        .chain(block.params.iter().map(|param| &param.name))
    {
        if type_params.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a type parameter"));
        }
        if callback_names.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a callable parameter"));
        }
        if value_names.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a value parametric"));
        }
    }
    validate_inference_sources(
        &block.type_params,
        &block.params,
        &block.callback_params,
        None,
    )
}

/// Type formals must be inferable from data, the expected function result, or
/// the result of a callback with already inferable inputs.
pub(super) fn validate_inference_sources(
    types: &[String],
    params: &[Param],
    callbacks: &[CallbackParam],
    result: Option<&Type>,
) -> Result<(), String> {
    let mut type_params: BTreeSet<_> = types.iter().map(String::as_str).collect();
    let mut input_names = Vec::new();
    for ty in params.iter().map(|param| &param.ty).chain(result) {
        type_parameter_references(ty, &mut input_names);
    }
    let result_source = if result.is_some() {
        " or the function result"
    } else {
        ""
    };
    for name in input_names {
        type_params.remove(name);
    }
    for callback in callbacks {
        let mut dependencies = Vec::new();
        for ty in &callback.params {
            type_parameter_references(ty, &mut dependencies);
        }
        if let Some(missing) = dependencies
            .iter()
            .find(|name| type_params.contains(**name))
        {
            return Err(format!(
                "cannot infer type parameter '{missing}' before callable parameter '{}'; its input types must come from a data input or an earlier callback result{result_source}",
                callback.name
            ));
        }
        dependencies.clear();
        type_parameter_references(&callback.result, &mut dependencies);
        for name in dependencies {
            type_params.remove(name);
        }
    }
    if let Some(missing) = types
        .iter()
        .find(|name| type_params.contains(name.as_str()))
    {
        let source = if callbacks.is_empty() {
            "any data input"
        } else {
            "any data input or callback result"
        };
        return Err(format!(
            "cannot infer type parameter '{missing}': it does not appear in {source}{result_source}"
        ));
    }
    Ok(())
}

/// The data input and result types of a block signature in its declaration
/// scope.
pub(super) struct BlockPortTypes {
    pub(super) params: Vec<Param>,
    pub(super) result: Type,
}

impl BlockPortTypes {
    /// Compute the minimum layout using written formal occurrences. An alias
    /// expands to a concrete type, even if a struct in that expansion shares
    /// a name with a block type formal.
    fn minimum_layout(&self, block: &Block, formals: &Bindings) -> Self {
        fn minimum(source: &Type, ty: &Type, formals: &Bindings) -> Type {
            match (source, ty) {
                (Type::Named(name), _) if formals.declares(name) => Type::unsigned(1),
                (_, Type::ParametricBits { width, .. }) => {
                    // A symbolic sign does not change the cost of a known
                    // width.
                    let width = match width.as_ref() {
                        Expr::Literal { ty: None, text } => {
                            text.parse().expect("resolved decimal width")
                        }
                        _ => 1,
                    };
                    Type::unsigned(width)
                }
                (
                    Type::Array { element: src, .. } | Type::ParametricArray { element: src, .. },
                    Type::ParametricArray { element, .. },
                ) => Type::array(minimum(src, element, formals), 1),
                (
                    Type::Array { element: src, .. } | Type::ParametricArray { element: src, .. },
                    Type::Array { element, size },
                ) => Type::array(minimum(src, element, formals), *size),
                (Type::Tuple(src), Type::Tuple(fields)) => Type::Tuple(
                    src.iter()
                        .zip(fields)
                        .map(|(src, ty)| minimum(src, ty, formals))
                        .collect(),
                ),
                (Type::TypeApplied { args: src, .. }, Type::TypeApplied { name, args }) => {
                    Type::TypeApplied {
                        name: name.clone(),
                        args: src
                            .iter()
                            .zip(args)
                            .map(|(src, ty)| minimum(src, ty, formals))
                            .collect(),
                    }
                }
                (Type::MixedApplied { args: src, .. }, Type::MixedApplied { name, args }) => {
                    Type::MixedApplied {
                        name: name.clone(),
                        args: src
                            .iter()
                            .zip(args)
                            .map(|(src, arg)| match arg {
                                StructArgument::Type(ty) => StructArgument::Type(minimum(
                                    src.as_type().expect("resolved type argument"),
                                    ty,
                                    formals,
                                )),
                                StructArgument::Value(_) => arg.clone(),
                            })
                            .collect(),
                    }
                }
                _ => ty.clone(),
            }
        }
        Self {
            params: block
                .params
                .iter()
                .zip(&self.params)
                .map(|(src, p)| Param {
                    name: p.name.clone(),
                    ty: minimum(&src.ty, &p.ty, formals),
                })
                .collect(),
            result: minimum(&block.result, &self.result, formals),
        }
    }
}

impl<'m> BlockTemplate<'m> {
    /// Revisit a parameter-dependent signature in its definition scope. Only a
    /// written type formal is substituted, even when a module alias expands to
    /// a nominal type with the same name.
    pub(super) fn specialize_dependent_ports(
        &self,
        dimensions: ModuleDimensions<'_, 'm>,
        types: &mut TypeResolver<'_, 'm>,
        bindings: &Bindings,
        values: &[BoundParametricValue],
    ) -> Result<Option<BlockPortTypes>, String> {
        if self.port_binding_work == 0 {
            return Ok(None);
        }
        let block = self.block;
        let registry = types
            .functions
            .as_deref_mut()
            .ok_or_else(|| format!("cannot specialize '{}' in this type annotation", block.name))?;
        registry.with_definition(&block.name, |registry| {
            let mut definition = BodyChecker::struct_type_scope(
                dimensions,
                types.resolver,
                types.structs,
                Some(registry),
            );
            definition.charge_expansion("block port types", self.port_binding_work)?;
            definition.function_type_bindings = Some(bindings.clone());
            definition
                .names
                .extend(block.clock.iter().chain(&block.reset).cloned());
            definition
                .names
                .extend(block.callback_params.iter().map(|p| p.name.clone()));
            for (formal, value) in block.value_parametrics.iter().zip(values) {
                definition.bind_parametric_value(&formal.name, value);
            }
            // One binding can be expanded by many ports. Bound all the copied
            // source type arguments together before retaining the concrete
            // ports.
            let mut copies = Some(Default::default());
            let mut resolve = |definition: &mut BodyChecker<'_, 'm>, source: &Type| {
                let ty =
                    definition.resolve_annotation(source, MAX_ELABORATION_DEPTH, &mut copies)?;
                definition.structs.update_imports(definition.resolver)?;
                definition.structs.validate_type(&ty)?;
                Ok::<_, String>(ty)
            };
            let params = block
                .params
                .iter()
                .map(|p| definition.resolve_data_parameter(p, &mut resolve))
                .collect::<Result<_, String>>()?;
            let result = resolve(&mut definition, &block.result)?;
            Ok(Some(BlockPortTypes { params, result }))
        })
    }

    /// Resolve module types while retaining the block's symbolic type names
    /// for data inference and callback specialization at each call site.
    pub(super) fn new(
        block: &'m Block,
        bindings: &Bindings,
        dimensions: ModuleDimensions<'_, 'm>,
        types: &mut TypeResolver<'_, 'm>,
    ) -> Result<Self, String> {
        let mut value_types = Vec::with_capacity(block.value_parametrics.len());
        let mut required_arguments = 0;
        if block.value_parametrics.iter().any(|p| p.default.is_some()) {
            let mut checker = types.struct_scope(dimensions);
            checker.function_type_bindings = Some(bindings.clone());
            checker.check_block_default_names(block)?;
        }
        let mut formal_names = BTreeSet::new();
        for (index, formal) in block.parametric_bindings().enumerate() {
            match formal {
                FunctionParametricRef::Callback(callback) => {
                    required_arguments = index + 1;
                    formal_names.insert(callback.name.as_str());
                }
                FunctionParametricRef::Value(value) => {
                    // A value formal shadows module declarations starting at
                    // its own annotation, before its type or value is known.
                    formal_names.insert(value.name.as_str());
                    if !block.type_params.is_empty()
                        && references_formal(SourceNode::Annotation(&value.ty), |name| {
                            bindings.declares(name)
                        })
                    {
                        return Err(format!(
                            "block value parametric '{}' requires a concrete type independent of the block's type parameters",
                            value.name
                        ));
                    }
                    let ty = resolve_type(
                        &value.ty,
                        &|name| formal_names.contains(name) || bindings.declares(name),
                        None,
                        dimensions,
                        types,
                        MAX_ELABORATION_DEPTH,
                    )?;
                    if matches!(&ty, Type::Named(_)) {
                        types.structs.update_imports(types.resolver)?;
                        types.structs.validate_type(&ty)?;
                    } else if !matches!(crate::typed::bits(&ty), Some((false, 1 | 32))) {
                        return Err(format!(
                            "block value parametric '{}' must have type u32, bool, or a concrete struct",
                            value.name
                        ));
                    }
                    value_types.push(ty);
                    if value.default.is_none() {
                        required_arguments = index + 1;
                    }
                }
            }
        }
        let reserved_ports = block
            .clock
            .iter()
            .chain(&block.reset)
            .chain(block.callback_params.iter().map(|p| &p.name))
            .chain(block.value_parametrics.iter().map(|p| &p.name));
        let formals = BlockDimensionFormals {
            values: block
                .value_parametrics
                .iter()
                .map(|p| p.name.clone())
                .collect(),
            types: bindings,
        };
        let mut port_dimensions = dimensions;
        port_dimensions.block_formals = Some(&formals);
        let (params, result) = resolve_signature(
            &block.params,
            &block.result,
            reserved_ports,
            Some(bindings),
            port_dimensions,
            types,
        )?;
        let mut callback_indices = BTreeMap::new();
        let mut callback_params = Vec::with_capacity(block.callback_params.len());
        for (index, callback) in block.callback_params.iter().enumerate() {
            callback_indices.insert(callback.name.clone(), index);
            let mut resolve = |ty: &Type| {
                resolve_type(
                    ty,
                    &|name| callback_indices.contains_key(name) || formals.values.contains(name),
                    Some(bindings),
                    dimensions,
                    types,
                    MAX_ELABORATION_DEPTH,
                )
            };
            let signature = CallbackParam {
                name: callback.name.clone(),
                params: callback
                    .params
                    .iter()
                    .map(&mut resolve)
                    .collect::<Result<_, _>>()?,
                result: resolve(&callback.result)?,
            };
            callback_params.push(BlockCallbackFormal::new(signature, bindings));
        }
        let TypeResolver {
            resolver, structs, ..
        } = types;
        structs.update_imports(resolver)?;
        for formal in &callback_params {
            let signature = &formal.signature;
            for ty in signature.params.iter().chain([&signature.result]) {
                structs
                    .formal_shape(ty, MAX_ELABORATION_DEPTH, Some(bindings))
                    .map_err(|e| format!("callable parameter '{}': {e}", signature.name))?;
            }
        }
        let (input_pattern_work, port_binding_work) = {
            if port_dimensions.has_dependent_signature(&params, &result) {
                let types = params.iter().map(|p| &p.ty).chain([&result]);
                // Specialization revisits every written port, including fixed
                // expressions that declaration resolution folded to a size.
                let mut work: usize = types.map(source_type_nodes).sum();
                let mut expressions = Vec::new();
                for ty in block.params.iter().map(|p| &p.ty).chain([&block.result]) {
                    super::type_dependencies(
                        ty,
                        |_, _| {},
                        |expr| expressions.push(SourceNode::Expression(expr)),
                    );
                }
                while let Some(node) = expressions.pop() {
                    work += 1;
                    node.push_children(&mut expressions);
                }
                let input_work = block.params.iter().map(|p| source_type_nodes(&p.ty)).sum();
                (input_work, work)
            } else {
                (0, 0)
            }
        };
        let ports = BlockPortTypes { params, result };
        // Check the lower bound from the written ports and reserve their output
        // names. A symbolic struct counts as a nominal leaf here.
        // Specialization checks its expanded fields inside the enclosing port
        // type.
        let shadows_struct = block
            .type_params
            .iter()
            .any(|p| structs.indices.contains_key(p));
        let minimum_ports;
        let (declaration_ports, shape_bindings) = if port_binding_work != 0 || shadows_struct {
            minimum_ports = ports.minimum_layout(block, bindings);
            (&minimum_ports, None)
        } else {
            (&ports, Some(bindings))
        };
        // A bare type parameter can acquire any output names from its binding.
        let known_outputs = !matches!(&block.result, Type::Named(name) if bindings.declares(name));
        let mut layout = BlockPortLayout::new(
            block.clock.as_deref(),
            block.reset.as_deref(),
            &declaration_ports.params,
            known_outputs.then_some(&declaration_ports.result),
            structs,
            shape_bindings,
        )?;
        if known_outputs {
            layout.check_template_outputs(&ports.result, structs, resolver)?;
        }
        Ok(Self {
            block,
            stages: dimensions.block_stages(block)?,
            ports,
            port_binding_work,
            input_pattern_work,
            callback_params,
            value_types,
            required_arguments,
            callback_indices,
        })
    }

    /// Complete the caller's explicit value prefix in the block's own scope.
    pub(super) fn apply_value_defaults(
        &self,
        caller: &mut BodyChecker<'_, 'm>,
        supplied: Vec<BoundParametricValue>,
        types: &Bindings,
    ) -> Result<Vec<BoundParametricValue>, String> {
        if self
            .block
            .value_parametrics
            .iter()
            .all(|p| p.default.is_none())
        {
            return Ok(supplied);
        }
        caller.with_value_definition(&self.block.name, |definition| {
            definition.function_type_bindings = Some(types.clone());
            let mut supplied = supplied.iter();
            let mut bound = Vec::with_capacity(self.value_types.len());
            for formal in self.block.parametric_bindings() {
                match formal {
                    FunctionParametricRef::Callback(callback) => {
                        definition.names.insert(callback.name.clone());
                    }
                    FunctionParametricRef::Value(formal) => {
                        let ty = self.value_types[bound.len()].clone();
                        let value = definition.value_or_default(
                            &self.block.name,
                            formal,
                            ty,
                            supplied.next(),
                        )?;
                        definition.bind_parametric_value(&formal.name, &value);
                        bound.push(value);
                    }
                }
            }
            Ok(bound)
        })
    }
}

/// Type parameters of one template and their bindings at a single call site.
#[derive(Clone)]
pub(super) struct Bindings {
    types: BTreeMap<String, Option<TypeArgument>>,
}

/// A shared specialization-key component. Callback results reused as inputs
/// to another callback keep the same canonical type across call sites.
pub(super) type TypeArgument = Rc<BoundType>;

/// The concrete argument and its cached expanded layout, reused whenever the
/// parameter occurs in an annotation or port type.
pub(super) struct BoundType {
    ty: Type,
    shape: TypeShape,
    pub(super) source_nodes: usize,
    type_hash: u64,
}

impl BoundType {
    /// Only independently inferred types with the same hash require a
    /// structural comparison to resolve equality.
    pub(super) fn comparison_nodes(&self, other: &Self) -> usize {
        if std::ptr::eq(self, other) || self.type_hash != other.type_hash {
            0
        } else {
            self.source_nodes.min(other.source_nodes)
        }
    }
}

impl PartialEq for BoundType {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (self.type_hash == other.type_hash && self.ty == other.ty)
    }
}

impl Eq for BoundType {}

impl Hash for BoundType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_hash.hash(state);
    }
}

/// Whether dimensions are resolved and a reference can still use context.
#[derive(Clone, Copy, Eq, PartialEq)]
enum TypePattern {
    Resolved,
    Source,
    SourceProjection,
}

/// Type names used by a constexpr expression, including casts and builtins.
fn value_type_names(expr: &Expr) -> impl Iterator<Item = &str> {
    let mut pending = vec![SourceNode::Expression(expr)];
    std::iter::from_fn(move || {
        while let Some(node) = pending.pop() {
            let name = match &node {
                SourceNode::Annotation(Type::Named(name)) => Some(name.as_str()),
                SourceNode::Expression(expr) => expr.struct_literal().map(|s| s.name),
                _ => None,
            };
            node.push_children(&mut pending);
            if name.is_some() {
                return name;
            }
        }
        None
    })
}

impl Bindings {
    pub(super) fn new(type_params: &[String]) -> Self {
        Self {
            types: type_params
                .iter()
                .map(|name| (name.clone(), None))
                .collect(),
        }
    }

    /// Introduce a type formal for the annotations and defaults that follow it.
    pub(super) fn declare_type_parameter(&mut self, name: &str) {
        self.types.insert(name.to_owned(), None);
    }

    pub(super) fn get(&self, name: &str) -> Option<&Type> {
        Some(&self.types.get(name)?.as_ref()?.ty)
    }

    /// Use a concrete binding's cached layout, or the minimum layout of a
    /// still-symbolic type parameter when checking the declaration.
    pub(super) fn shape(&self, name: &str) -> Option<TypeShape> {
        Some(
            self.types
                .get(name)?
                .as_ref()
                .map_or(TypeShape::LEAF, |t| t.shape),
        )
    }

    pub(super) fn declares(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Check whether the pattern still needs a type witness.
    pub(super) fn has_unbound_variable(&self, pattern: &Type) -> bool {
        match pattern {
            Type::Named(name) => self.types.get(name).is_some_and(Option::is_none),
            Type::Tuple(fields) => fields.iter().any(|field| self.has_unbound_variable(field)),
            Type::MixedApplied { args, .. } => args.iter().any(|arg| match arg {
                StructArgument::Type(ty) => self.has_unbound_variable(ty),
                StructArgument::Value(expr) => value_type_names(expr)
                    .any(|name| self.types.get(name).is_some_and(Option::is_none)),
            }),
            Type::TypeApplied { args, .. } => args.iter().any(|arg| self.has_unbound_variable(arg)),
            Type::Array { element, .. } | Type::ParametricArray { element, .. } => {
                self.has_unbound_variable(element)
            }
            _ => false,
        }
    }

    /// A type parameter cannot resolve to a callee or an import qualifier.
    pub(super) fn reject_value(&self, name: &str) -> Result<(), String> {
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if self.declares(head) {
            return Err(format!("'{head}' names a type parameter here"));
        }
        Ok(())
    }

    pub(super) fn source_nodes(&self, name: &str) -> Option<usize> {
        Some(self.types.get(name)?.as_ref()?.source_nodes)
    }

    /// Apply a callback result's inferred variables in the caller. Unbound
    /// variables share their cached concrete type and layout. Only an existing
    /// independent binding requires another comparison.
    pub(super) fn merge_callback_results(
        &mut self,
        result: &Bindings,
        mut charge: impl FnMut(usize) -> Result<(), String>,
    ) -> Result<(), String> {
        charge(result.types.len())?;
        for (name, inferred) in &result.types {
            let inferred = inferred.as_ref().expect("matched result type parameter");
            let binding = self.types.get_mut(name).expect("declared type parameter");
            if let Some(prior) = binding {
                charge(prior.comparison_nodes(inferred))?;
                if prior != inferred {
                    return Err(format!(
                        "conflicting types for type parameter '{name}': inferred {} and {}",
                        prior.ty, inferred.ty
                    ));
                }
            }
            *binding = Some(Rc::clone(inferred));
        }
        Ok(())
    }

    /// Match a resolved formal port shape against the argument's checked type.
    /// Named structs retain their identity even when their layouts coincide.
    pub(super) fn match_input(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
    ) -> Result<(), String> {
        self.match_pattern(formal, actual, structs, TypePattern::Resolved)
    }

    /// Infer type variables while dimensions and nominal imports in a source
    /// annotation still await resolution in the function's declaration scope.
    /// An inferred projection still needs to be checked with its bound type.
    pub(super) fn infer_source_type(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
        contextual_projection: bool,
    ) -> Result<(), String> {
        let pattern = if contextual_projection {
            TypePattern::SourceProjection
        } else {
            TypePattern::Source
        };
        self.match_pattern(formal, actual, structs, pattern)
    }

    fn match_pattern(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
        pattern: TypePattern,
    ) -> Result<(), String> {
        if let Type::Named(name) = formal
            && let Some(binding) = self.types.get_mut(name)
        {
            match binding {
                Some(prior)
                    if pattern != TypePattern::SourceProjection
                        && !compatible(&prior.ty, actual) =>
                {
                    return Err(format!(
                        "conflicting types for type parameter '{name}': inferred {} and {actual}",
                        prior.ty
                    ));
                }
                Some(_) => { /* Keep the first binding as context for checking the argument. */ }
                None => {
                    let shape = structs.type_shape(actual, MAX_ELABORATION_DEPTH)?;
                    let ty = canonical_argument(actual);
                    let mut hasher = DefaultHasher::new();
                    ty.hash(&mut hasher);
                    *binding = Some(Rc::new(BoundType {
                        ty,
                        shape,
                        source_nodes: shape.source_nodes,
                        type_hash: hasher.finish(),
                    }));
                }
            }
            return Ok(());
        }
        match (formal, actual) {
            (formal, Type::Named(canonical))
                if let Some((name, args)) = formal.struct_application()
                    && let Some(instance) = structs.instances.get(canonical)
                    && instance.template.name == name
                    && match args {
                        StructLiteralArgs::Types(types) => types.len() == instance.args.len(),
                        StructLiteralArgs::Mixed(_) => true,
                        _ => false,
                    } =>
            {
                for (arg, bound) in args.iter().zip(&instance.args) {
                    match (arg, bound) {
                        (StructArgumentRef::Type(formal), BoundStructArgument::Type(actual)) => {
                            self.match_pattern(formal, actual, structs, pattern)?;
                        }
                        _ if matches!(args, StructLiteralArgs::Types(_)) => {
                            return Err(format!("expected a type argument for {name}"));
                        }
                        _ => { /* Mixed value arguments are checked after type inference. */ }
                    }
                }
                Ok(())
            }
            (Type::Tuple(formals), Type::Tuple(actuals)) if formals.len() == actuals.len() => {
                for (formal, actual) in formals.iter().zip(actuals) {
                    self.match_pattern(formal, actual, structs, pattern)?;
                }
                Ok(())
            }
            (
                Type::Array {
                    element: formal,
                    size,
                },
                Type::Array {
                    element: actual,
                    size: actual_size,
                },
            ) if size == actual_size => self.match_pattern(formal, actual, structs, pattern),
            (
                Type::ParametricArray { element, .. },
                Type::Array {
                    element: actual, ..
                },
            ) if pattern != TypePattern::Resolved => {
                self.match_pattern(element, actual, structs, pattern)
            }
            _ if pattern != TypePattern::Resolved => Ok(()),
            _ if compatible(formal, actual) => Ok(()),
            _ => Err(format!("expected {formal}, got {actual}")),
        }
    }

    /// Bound size and depth before copying type arguments into their use sites.
    /// A non-parameter name requires one AST node. Its full nominal layout is
    /// checked after source names (including import aliases) are resolved.
    pub(super) fn validate_substitution(&self, formal: &Type) -> Result<(), String> {
        TypeShape::for_type(
            formal,
            MAX_ELABORATION_DEPTH,
            &|name| match self.types.get(name) {
                Some(Some(binding)) => Ok(binding.shape),
                Some(None) => Err(format!("cannot infer type parameter '{name}'")),
                None => Ok(TypeShape::LEAF),
            },
        )
        .map(|_| ())
    }

    /// Instantiate a resolved formal type once its complete layout fits.
    pub(super) fn concrete_type(&self, formal: &Type) -> Result<Type, String> {
        self.validate_substitution(formal)?;
        Ok(self.substitute(formal))
    }

    fn substitute(&self, formal: &Type) -> Type {
        match formal {
            Type::Named(name) => self.get(name).unwrap_or(formal).clone(),
            Type::Tuple(fields) => {
                Type::Tuple(fields.iter().map(|field| self.substitute(field)).collect())
            }
            Type::Array { element, size } => Type::array(self.substitute(element), *size),
            Type::MixedApplied { name, args } => Type::MixedApplied {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| match arg {
                        StructArgument::Type(ty) => StructArgument::Type(self.substitute(ty)),
                        StructArgument::Value(_) => arg.clone(),
                    })
                    .collect(),
            },
            Type::TypeApplied { name, args } => Type::TypeApplied {
                name: name.clone(),
                args: args.iter().map(|arg| self.substitute(arg)).collect(),
            },
            Type::Bool
            | Type::Bits { .. }
            | Type::Enum { .. }
            | Type::ParametricBits { .. }
            | Type::ParametricArray { .. }
            | Type::Applied { .. }
            | Type::ParametricStruct { .. } => formal.clone(),
        }
    }

    /// Identify the type parameters referenced by callback annotations.
    pub(super) fn referenced_parameters<'t>(
        &self,
        types: impl IntoIterator<Item = &'t Type>,
    ) -> Vec<String> {
        let mut names = Vec::new();
        for ty in types {
            type_parameter_references(ty, &mut names);
        }
        names.retain(|name| self.declares(name));
        names.sort_unstable();
        names.dedup();
        names.into_iter().map(str::to_owned).collect()
    }

    /// Types needed to evaluate constexpr arguments are inputs to
    /// specialization, even when that value expression cannot infer the type
    /// from a port.
    pub(super) fn value_type_dependencies(&self, ty: &Type) -> Vec<String> {
        let mut names = BTreeSet::new();
        super::type_dependencies(
            ty,
            |_, _| {},
            |expr| {
                names.extend(
                    value_type_names(expr)
                        .filter(|name| self.declares(name))
                        .map(str::to_owned),
                );
            },
        );
        names.into_iter().collect()
    }

    /// Before callbacks are resolved, a template's data arguments can leave
    /// some types unbound. Name order is stable within that declaration.
    pub(super) fn partial_arguments(&self) -> Vec<Option<TypeArgument>> {
        self.types.values().cloned().collect()
    }

    /// Build the specialization key in the template's declaration order.
    pub(super) fn arguments(&self, type_params: &[String]) -> Result<Vec<TypeArgument>, String> {
        type_params
            .iter()
            .map(|name| {
                self.types
                    .get(name)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| format!("cannot infer type parameter '{name}'"))
            })
            .collect()
    }
}

/// Use one specialization for bool and u1, including inside aggregate inputs.
fn canonical_argument(ty: &Type) -> Type {
    match ty {
        Type::Bool => Type::unsigned(1),
        Type::Tuple(fields) => Type::Tuple(fields.iter().map(canonical_argument).collect()),
        Type::Array { element, size } => Type::array(canonical_argument(element), *size),
        Type::Bits { .. }
        | Type::Enum { .. }
        | Type::Named(_)
        | Type::ParametricBits { .. }
        | Type::ParametricArray { .. }
        | Type::Applied { .. }
        | Type::ParametricStruct { .. }
        | Type::MixedApplied { .. }
        | Type::TypeApplied { .. } => ty.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImportOptions;
    use crate::check::{
        BlockParametricArguments, BlockSpecializations, Declarations, FunctionSpecializations,
        ImportResolver, MAX_TYPE_SIZE,
    };
    use crate::parser::parse_module;
    use xlsynth_pir::IrValue;

    #[test]
    fn block_record_formal_respects_the_type_parameter_scope() -> Result<(), String> {
        let source = r#"
struct T { depth: u32 }
type Options = T;
struct Wrapped<E> { options: E }
const DEFAULT = T { depth: 3 };
const CONFIG = Wrapped<Options> { options: DEFAULT };
fn size<O: Options>(xs: u8[O.depth]) -> u32 { array_size(xs) }
block stage<T, O: Wrapped<Options>>(clk, x: T, xs: u8[O.options.depth]) -> (T, u32) {
    reg q: T = zero!();
    q := x;
    (q, size<{O.options}>(xs))
}
pub block circuit(clk, x: bool, xs: u8[3]) -> (bool, u32) {
    let y = stage<CONFIG>(x, xs);
    y
}
"#;
        crate::check_module(&parse_module(source)?)?;
        for ty in ["T", "Wrapped<T>"] {
            let symbolic = source.replace(
                "stage<T, O: Wrapped<Options>>",
                &format!("stage<T, O: {ty}>"),
            );
            assert_eq!(
                crate::check_module(&parse_module(&symbolic)?).map(|_| ()),
                Err("block 'stage': block value parametric 'O' requires a concrete type independent of the block's type parameters".into())
            );
        }
        Ok(())
    }

    #[test]
    fn distinct_block_port_bindings_share_the_constant_evaluation_budget() -> Result<(), String> {
        let module = parse_module("block stage<N: u32>(xs: u8[N + u32:1]) -> bool { true }")?;
        let block = &module.blocks[0];
        let decls = Declarations::default();
        let dimensions = decls.module_dimensions(&block.name);
        let mut functions = FunctionSpecializations::new(BTreeSet::new());
        // Folding the addition fits once, including its two scalar operands.
        functions.constant_budgets.required = 5;
        let mut resolver = ImportResolver::new(&[], &ImportOptions::default())?;
        let mut types = TypeResolver {
            resolver: &mut resolver,
            structs: &mut StructTable::default(),
            functions: Some(&mut functions),
        };
        let bindings = Bindings::new(&block.type_params);
        let template = BlockTemplate::new(block, &bindings, dimensions, &mut types)?;
        let mut blocks = BlockSpecializations::new(BTreeSet::new());
        let mut intern = |n| {
            let args = BlockParametricArguments {
                bindings: bindings.clone(),
                values: vec![BoundParametricValue {
                    ty: Type::unsigned(32),
                    value: IrValue::u32(n),
                }],
                callbacks: vec![],
            };
            blocks.intern(&template, args, &[], dimensions, &mut types)
        };
        let first = intern(1)?;
        assert_eq!(
            first.interface.params[0].ty,
            Type::array(Type::unsigned(8), 2)
        );
        assert!(Rc::ptr_eq(&first, &intern(1)?));
        assert_eq!(
            intern(2).map(|_| ()),
            Err(format!(
                "constant evaluation exceeds the work limit of {MAX_TYPE_SIZE}"
            ))
        );
        assert!(!functions.checking_definition());
        Ok(())
    }

    #[test]
    fn substitution_checks_combined_size_and_depth_before_copying() -> Result<(), String> {
        let name = String::from("T");
        let formal = Type::Named(name.clone());
        let structs = StructTable::default();
        let mut bindings = Bindings::new(std::slice::from_ref(&name));
        let actual = Type::Tuple(vec![Type::Bool; 1024]);
        bindings.match_input(&formal, &actual, &structs)?;
        let repeated = Type::Tuple(vec![formal.clone(); 1024]);
        assert_eq!(
            bindings.concrete_type(&repeated).unwrap_err(),
            format!("aggregate exceeds maximum concrete type size {MAX_TYPE_SIZE}"),
        );

        let nested = (0..MAX_ELABORATION_DEPTH).fold(formal, |element, _| Type::Array {
            element: Box::new(element),
            size: 1,
        });
        assert_eq!(
            bindings.concrete_type(&nested).unwrap_err(),
            format!("expanded type nesting exceeds {MAX_ELABORATION_DEPTH} levels"),
        );
        Ok(())
    }
}
