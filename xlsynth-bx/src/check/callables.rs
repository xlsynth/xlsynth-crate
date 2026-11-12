// SPDX-License-Identifier: Apache-2.0

//! Bind a named callback in the instantiating scope and call it from the
//! template's definition scope without introducing a runtime function value.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::rc::Rc;

use super::generic_blocks::TypeArgument;
use super::{
    BUILTIN_PREFIX, Bindings, BlockSpecializations, BodyChecker, BodyKind, CallbackFormal,
    CallbackTarget, K, MAX_ELABORATION_DEPTH, MAX_TYPE_SIZE, Param, TypedExpr, compatible,
    source_type_nodes,
};
use crate::ast::{CallbackParam, Expr, Function, Type};

/// A visible declaration before its DSLX specialization (if any) is known.
#[derive(Clone, Eq, Hash, PartialEq)]
pub(super) enum CallbackSource<'m> {
    Named(&'m str),
    Imported { module: String, member: String },
}

/// A named map callback resolved in the caller's scope.
pub(super) enum MapCallee<'m> {
    Formal(usize),
    ValueTemplate(&'m Function),
    Declaration(CallbackSource<'m>),
    Builtin(&'static str),
}

/// Unary DSLX builtins available as named map callbacks.
const MAP_BUILTINS: [&str; 10] = [
    "and_reduce",
    "array_rev",
    "array_size",
    "ceillog2",
    "clz",
    "ctz",
    "encode",
    "or_reduce",
    "rev",
    "xor_reduce",
];

/// One formal has a fixed resolved signature for the lifetime of the module.
/// Its validated actual can be reused at any site resolving the same source.
#[derive(Eq, Hash, PartialEq)]
pub(super) struct CallbackBinding<'m> {
    template: &'m str,
    formal: usize,
    source: CallbackSource<'m>,
}

/// A symbolic block callback and the type bindings needed to resolve its
/// input signature. Its return pattern may introduce further bindings.
pub(super) struct BlockCallbackFormal {
    pub(super) signature: CallbackParam,
    input_type_params: Vec<String>,
    result_type_params: Vec<String>,
}

impl BlockCallbackFormal {
    pub(super) fn new(signature: CallbackParam, bindings: &Bindings) -> Self {
        let input_type_params = bindings.referenced_parameters(&signature.params);
        let result_type_params = bindings.referenced_parameters([&signature.result]);
        Self {
            signature,
            input_type_params,
            result_type_params,
        }
    }
}

/// A resolved block callback shared by instances with the same concrete
/// input bindings. The actual function determines the result type.
pub(super) struct BoundCallback<'m> {
    pub(super) signature: CallbackParam,
    pub(super) target: CallbackTarget<'m>,
    result_bindings: Bindings,
}

/// A block formal can have different input types across specializations.
/// Only the variables used by its input signature belong in this lookup.
#[derive(Eq, Hash, PartialEq)]
pub(super) struct BlockCallbackBinding<'m> {
    binding: CallbackBinding<'m>,
    inputs: Vec<TypeArgument>,
}

impl BlockSpecializations<'_> {
    /// Reserve shared callback metadata or a newly emitted DSLX signature.
    fn charge_callback_storage(&mut self, nodes: usize) -> Result<(), String> {
        self.expansion_work = self.expansion_work.saturating_add(nodes);
        if self.expansion_work > MAX_TYPE_SIZE {
            return Err(format!(
                "callback bindings exceed the generic block specialization expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        Ok(())
    }
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// The declaration-site lookup table is shared by all specializations.
    pub(super) fn callback_indices(&self) -> Option<&'a BTreeMap<String, usize>> {
        if let Some(indices) = self.template_callbacks {
            return Some(indices);
        }
        if let Some(template) = self.source_template {
            return Some(&template.callback_indices);
        }
        let spec = self.specialization?;
        if spec.callbacks.is_empty() {
            return None;
        }
        Some(&self.decls.templates[spec.block.name.as_str()].callback_indices)
    }

    /// Look up a callable parameter for a direct call or a named map.
    pub(super) fn callback_index(&self, name: &str) -> Option<usize> {
        self.callback_indices()?.get(name).copied()
    }

    /// Resolve the template itself in the caller's lexical scope.
    pub(super) fn callable_template(
        &self,
        name: &str,
        hidden: bool,
        callback_count: usize,
    ) -> Result<&'m Function, String> {
        if hidden {
            let name = super::scope::source_binding_name(name, &self.scope_alias_names);
            return Err(format!("'{name}' names a local value here"));
        }
        if self.callback_index(name).is_some() {
            return Err(format!(
                "callable parameter '{name}' requires a direct call"
            ));
        }
        self.reject_type_parameter_value(name)?;
        if !self.decls.visible(name, self.function_position) {
            return Err(format!("function '{name}' is declared later in the module"));
        }
        let &template = self
            .decls
            .function_templates
            .get(name)
            .ok_or_else(|| format!("'{name}' is not a function with callable parameters"))?;
        if template.has_inferred_parametrics() {
            let count = template.callback_params.len() + template.value_parametrics.len();
            if callback_count > count {
                return Err(format!(
                    "'{name}' expects at most {count} parametric arguments, got {callback_count}"
                ));
            }
            if template
                .parametric_bindings()
                .skip(callback_count)
                .any(|p| matches!(p, crate::ast::FunctionParametricRef::Callback(_)))
            {
                return Err(format!("'{name}' requires an explicit callable argument"));
            }
            return Ok(template);
        }
        if callback_count != template.callback_params.len() {
            return Err(format!(
                "'{name}' expects {} callable arguments, got {callback_count}",
                template.callback_params.len()
            ));
        }
        Ok(template)
    }

    /// Resolve a local or external declaration or a public DSLX function in
    /// the caller's lexical scope.
    pub(super) fn callback_source(
        &self,
        name: &str,
        is_local: impl FnOnce(&str) -> bool,
    ) -> Result<CallbackSource<'m>, String> {
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if is_local(head) {
            let head = super::scope::source_binding_name(head, &self.scope_alias_names);
            return Err(format!("'{head}' names a local value here"));
        }
        if self.callback_index(head).is_some() {
            return Err(format!("cannot forward callable parameter '{head}'"));
        }
        self.reject_type_parameter_value(head)?;
        if !self.decls.visible(name, self.function_position) {
            return Err(format!("callback '{name}' is declared later in the module"));
        }
        if let Some((module, member)) = self.resolver.lookup(name)? {
            self.resolver.check_function_name(&module, &member)?;
            return Ok(CallbackSource::Imported { module, member });
        }
        if let Some((&callee, _)) = self.decls.functions.get_key_value(name)
            && !self.decls.function_templates.contains_key(callee)
        {
            return Ok(CallbackSource::Named(callee));
        }
        Err(format!(
            "callback '{name}' must name a concrete BX function, an extern function, or a public DSLX function"
        ))
    }

    /// Resolve the actual from the formal input types. Function templates
    /// can compare its return type or use it to infer further type bindings.
    pub(super) fn callback_result(
        &mut self,
        params: &[Type],
        source: &CallbackSource<'m>,
        new_import_input_nodes: Option<usize>,
    ) -> Result<(CallbackTarget<'m>, Cow<'a, Type>), String> {
        match source {
            CallbackSource::Named(callee) => {
                let f = &self.decls.functions[*callee];
                if f.params.len() != params.len()
                    || !f
                        .params
                        .iter()
                        .zip(params)
                        .all(|(p, expected)| compatible(&p.ty, expected))
                {
                    let actual = f
                        .params
                        .iter()
                        .map(|p| p.ty.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let expected = params
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "callback '{callee}' has parameter types ({actual}), expected ({expected})"
                    ));
                }
                let target = if f.local {
                    CallbackTarget::Local(callee)
                } else {
                    CallbackTarget::Extern(callee)
                };
                Ok((target, Cow::Borrowed(&f.result)))
            }
            CallbackSource::Imported { module, member } => {
                let mut charge_probe = |nodes| match self.specializations.as_deref_mut() {
                    Some(blocks) => blocks.charge_callback_storage(nodes),
                    None => self
                        .function_registry
                        .as_deref_mut()
                        .expect("module function registry")
                        .charge_bindings(nodes),
                };
                let mut created = false;
                let (id, result) = self.resolver.resolve_with_new_inputs(
                    module,
                    member,
                    params.to_vec(),
                    || {
                        if let Some(nodes) = new_import_input_nodes {
                            charge_probe(nodes)?;
                            created = true;
                        }
                        Ok(())
                    },
                )?;
                self.structs.update_imports(self.resolver)?;
                let shape = self.structs.type_shape(&result, MAX_ELABORATION_DEPTH)?;
                if created {
                    charge_probe(shape.ir_nodes)?;
                }
                Ok((CallbackTarget::Imported(id), Cow::Owned(result)))
            }
        }
    }

    /// Lexical lookup belongs to this particular call site. Only successful
    /// signature checks against the resolved declaration can be shared.
    fn resolve_callback(
        &mut self,
        actual: &str,
        template: &'m str,
        index: usize,
        formal: &CallbackFormal,
    ) -> Result<CallbackTarget<'m>, String> {
        let key = CallbackBinding {
            template,
            formal: index,
            source: self.callback_source(actual, |head| self.has_local_name(head))?,
        };
        if let Some(target) = self
            .function_registry
            .as_ref()
            .expect("module function registry")
            .validated_target(&key)
        {
            return Ok(target);
        }
        if self.replaying_body() {
            self.charge_expansion("callback bindings", formal.binding_work)?;
        }
        let (target, result) = self.callback_result(&formal.signature.params, &key.source, None)?;
        if !compatible(&result, &formal.signature.result) {
            return Err(format!(
                "callback '{actual}' returns {result}, expected {}",
                formal.signature.result
            ));
        }
        self.function_registry
            .as_mut()
            .expect("module function registry")
            .record_validated(key, target)?;
        Ok(target)
    }

    /// Resolve a symbolic block callback and use its result to extend or
    /// check the caller's type bindings. Fixed signature components are
    /// checked once per concrete input signature and function.
    pub(super) fn resolve_block_callback(
        &mut self,
        actual: &str,
        template: &'m str,
        index: usize,
        formal: &BlockCallbackFormal,
        bindings: &mut Bindings,
    ) -> Result<Rc<BoundCallback<'m>>, String> {
        let source = self.callback_source(actual, |head| self.has_local_name(head))?;
        self.charge_expansion("callback lookup", 1 + formal.input_type_params.len())?;
        let key = BlockCallbackBinding {
            binding: CallbackBinding {
                template,
                formal: index,
                source,
            },
            inputs: bindings.arguments(&formal.input_type_params)?,
        };
        let bound = match self
            .specializations
            .as_ref()
            .expect("block specialization registry")
            .validated_callbacks
            .get_key_value(&key)
        {
            Some((prior, bound)) => {
                let nodes = prior
                    .inputs
                    .iter()
                    .zip(&key.inputs)
                    .map(|(a, b)| a.comparison_nodes(b))
                    .sum();
                let bound = Rc::clone(bound);
                self.charge_expansion("callback lookup", nodes)?;
                bound
            }
            None => self.validate_block_callback(formal, bindings, key)?,
        };
        bindings.merge_callback_results(&bound.result_bindings, |nodes| {
            self.charge_expansion("callback result matching", nodes)
        })?;
        Ok(bound)
    }

    /// Cache the instantiated signature and the concrete bindings induced by
    /// its return pattern. Neither depends on other port or callback bindings.
    fn validate_block_callback(
        &mut self,
        formal: &BlockCallbackFormal,
        bindings: &Bindings,
        key: BlockCallbackBinding<'m>,
    ) -> Result<Rc<BoundCallback<'m>>, String> {
        let key_nodes = key.inputs.iter().fold(1usize, |total, input| {
            total.saturating_add(input.source_nodes)
        });
        self.specializations
            .as_mut()
            .expect("block specialization registry")
            .charge_callback_storage(key_nodes)?;
        let mut params = Vec::with_capacity(formal.signature.params.len());
        let mut input_nodes = 0usize;
        for param in &formal.signature.params {
            let ty = self.bound_block_type(param, bindings)?;
            let shape = self.structs.type_shape(&ty, MAX_ELABORATION_DEPTH)?;
            input_nodes = input_nodes.saturating_add(shape.ir_nodes);
            self.specializations
                .as_mut()
                .expect("block specialization registry")
                .charge_callback_storage(source_type_nodes(&ty))?;
            params.push(ty);
        }
        let (target, result) =
            self.callback_result(&params, &key.binding.source, Some(input_nodes))?;
        let registry = self
            .specializations
            .as_mut()
            .expect("block specialization registry");
        registry.charge_callback_storage(source_type_nodes(&result))?;
        let mut result_bindings = Bindings::new(&formal.result_type_params);
        result_bindings.match_input(&formal.signature.result, &result, self.structs)?;
        for name in &formal.result_type_params {
            registry.charge_callback_storage(
                result_bindings
                    .source_nodes(name)
                    .expect("matched result type parameter"),
            )?;
        }
        let bound = Rc::new(BoundCallback {
            signature: CallbackParam {
                name: formal.signature.name.clone(),
                params,
                result: result.into_owned(),
            },
            target,
            result_bindings,
        });
        registry.validated_callbacks.insert(key, Rc::clone(&bound));
        Ok(bound)
    }

    /// Bind callable identities at the instantiation site. Only data
    /// arguments become operands of the call to the generated function.
    pub(super) fn parametric_call(
        &mut self,
        name: &str,
        parametrics: &[Expr],
        args: &[Expr],
    ) -> Result<(Type, K), String> {
        if parametrics.is_empty() && !self.decls.function_templates.contains_key(name) {
            return self.function_call(name, args);
        }
        if !self.decls.function_templates.contains_key(name)
            && self.resolver.lookup(name)?.is_some()
            && let CallbackSource::Imported { module, member } =
                self.callback_source(name, |head| self.has_local_name(head))?
        {
            let value = self.explicit_import_call(&module, &member, parametrics, args)?;
            return Ok((value.ty, value.kind));
        }
        let template =
            self.callable_template(name, self.has_local_name(name), parametrics.len())?;
        if self.function_registry.is_none() {
            return Err(format!(
                "cannot specialize function '{name}' in this type annotation"
            ));
        }
        if template.has_inferred_parametrics() {
            return self.value_function_call(template, parametrics, args);
        }
        let callbacks = parametrics
            .iter()
            .map(super::generic_blocks::callback_argument_name)
            .collect::<Result<Vec<_>, _>>()?;
        let interface = &self.decls.functions[name];
        if self.replaying_body() {
            self.charge_expansion("callback bindings", callbacks.len())?;
        }
        let targets = callbacks
            .iter()
            .zip(&interface.callback_params)
            .enumerate()
            .map(|(index, (actual, formal))| {
                self.resolve_callback(actual, &template.name, index, formal)
                    .map_err(|e| {
                        format!(
                            "callable parameter '{}' of '{name}': {e}",
                            formal.signature.name
                        )
                    })
            })
            .collect::<Result<_, _>>()?;
        let args = self.call_arguments(
            name,
            args,
            interface
                .params
                .iter()
                .map(|p| (Some(p.name.as_str()), &p.ty)),
            true,
        )?;
        let symbol = self.intern_function(template, targets, None)?;
        Ok((interface.result.clone(), K::Call { name: symbol, args }))
    }

    fn bound_formal(&self, index: usize) -> (&'a CallbackParam, Option<CallbackTarget<'m>>) {
        if let Some(template) = self.source_template {
            (
                &template.callback_params[index].signature,
                self.function_spec.map(|spec| spec.targets[index]),
            )
        } else {
            let bound = &self.specialization.expect("bound block callback").callbacks[index];
            (&bound.signature, Some(bound.target))
        }
    }

    /// Typecheck against the declared signature in both the abstract body
    /// and its concrete instances. Abstract call nodes are never lowered.
    pub(super) fn call_formal(&mut self, index: usize, args: &[Expr]) -> Result<(Type, K), String> {
        let (formal, target) = self.bound_formal(index);
        let dslx_arguments =
            self.kind == BodyKind::Function || !matches!(target, Some(CallbackTarget::Extern(_)));
        let args = self.call_arguments(
            &formal.name,
            args,
            formal.params.iter().map(|ty| (None, ty)),
            dslx_arguments,
        )?;
        let call = self.callback_call(target, &formal.name, args);
        Ok((formal.result.clone(), call))
    }

    /// Resolve names even when an enclosing loop has no iterations.
    pub(super) fn map_callee(
        &self,
        name: &str,
        is_local: impl Fn(&str) -> bool,
    ) -> Result<MapCallee<'m>, String> {
        let qualified_builtin = name.strip_prefix(BUILTIN_PREFIX);
        let spelling = crate::bx_import::library_item(name).map_or(name, |(_, item)| item);
        let builtin = MAP_BUILTINS
            .into_iter()
            .find(|&builtin| builtin == qualified_builtin.unwrap_or(spelling));
        if qualified_builtin.is_some()
            && let Some(builtin) = builtin
        {
            return Ok(MapCallee::Builtin(builtin));
        }
        if let Some(index) = self.callback_index(name)
            && !is_local(name)
        {
            return Ok(MapCallee::Formal(index));
        }
        // An extern declaration binds its full qualified name. A value
        // sharing its qualifier can still shadow an imported module.
        if let Some(callee) = self.qualified_extern(name) {
            return self
                .callback_source(callee, |_| false)
                .map(MapCallee::Declaration);
        }
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if let Some(&template) = self.decls.function_templates.get(name)
            && template.has_inferred_parametrics()
            && template.callback_params.is_empty()
            && !is_local(head)
            && self.callback_index(head).is_none()
            && self.decls.visible(name, self.function_position)
        {
            self.reject_type_parameter_value(head)?;
            return Ok(MapCallee::ValueTemplate(template));
        }
        if let Some(builtin) = builtin
            && !is_local(head)
            && self.callback_index(head).is_none()
        {
            self.reject_type_parameter_value(head)?;
            let visible = self
                .decls
                .visible(name, self.function_position.or(self.block_position));
            // Blocks can call forward imports of functions. An imported type
            // shadows the builtin only after its use is visible.
            if !visible
                && let Some((module, member)) = self.resolver.lookup_enum_prefix(name)?
                && self.resolver.has_public_type(&module, &member)
            {
                return Ok(MapCallee::Builtin(builtin));
            }
            let visible_type = (self.structs.indices.contains_key(name)
                || self.structs.template(name).is_some()
                || self.structs.enums.contains_key(name)
                || self.decls.type_aliases.contains_key(name))
                && visible;
            if spelling == name
                && (!self.decls.visible(name, self.function_position)
                    || (!self.decls.declares_function(name)
                        && !self.decls.is_block(name)
                        && !visible_type
                        && self.resolver.lookup(name)?.is_none()))
            {
                return Ok(MapCallee::Builtin(builtin));
            }
        }
        self.callback_source(name, is_local)
            .map(MapCallee::Declaration)
    }

    /// Recover a qualified extern's declaration name if scope preparation
    /// substituted an alias for an unrelated value sharing its qualifier.
    pub(super) fn qualified_extern(&self, name: &str) -> Option<&'m str> {
        let (head, member) = name.split_once("::")?;
        let source = super::scope::source_binding_name(head, &self.scope_alias_names);
        let qualified = if source == head {
            Cow::Borrowed(name)
        } else {
            Cow::Owned(format!("{source}::{member}"))
        };
        let (&callee, f) = self.decls.functions.get_key_value(qualified.as_ref())?;
        (!f.local).then_some(callee)
    }

    /// Apply a unary callback to an already materialized array element.
    pub(super) fn map_element_call(
        &mut self,
        callee: MapCallee<'m>,
        name: &str,
        parametrics: &[Expr],
        parameter: &Param,
    ) -> Result<TypedExpr, String> {
        let element = TypedExpr {
            ty: parameter.ty.clone(),
            kind: K::Name(parameter.name.clone()),
        };
        let (target, ty) = match callee {
            MapCallee::ValueTemplate(template) => {
                return self.value_map_call_with_actuals(template, parametrics, element);
            }
            MapCallee::Declaration(CallbackSource::Imported { module, member })
                if !parametrics.is_empty() =>
            {
                let values = self.imported_value_parametrics(&module, &member, parametrics)?;
                return self.apply_explicit_import(&module, &member, values, vec![element]);
            }
            _ if !parametrics.is_empty() => {
                return Err(format!(
                    "map callback '{name}' requires a value-parametric function for explicit arguments"
                ));
            }
            MapCallee::Formal(index) => {
                let (formal, target) = self.bound_formal(index);
                let [input] = formal.params.as_slice() else {
                    return Err(format!(
                        "map callback '{name}' requires exactly one parameter"
                    ));
                };
                if !compatible(input, &element.ty) {
                    return Err(format!(
                        "map callback '{name}' expects {input}, got {}",
                        element.ty
                    ));
                }
                (target, formal.result.clone())
            }
            MapCallee::Declaration(source) => {
                let (target, ty) =
                    self.callback_result(std::slice::from_ref(&element.ty), &source, None)?;
                (Some(target), ty.into_owned())
            }
            MapCallee::Builtin(builtin) => {
                let mut saved = Vec::new();
                self.bind_scoped_value(&parameter.name, element, None, &mut saved);
                let result = self.builtin_call(builtin, &[Expr::Name(parameter.name.clone())]);
                self.leave_scope(saved);
                return result.map(|(ty, kind)| TypedExpr { ty, kind });
            }
        };
        Ok(TypedExpr {
            ty,
            kind: self.callback_call(target, name, vec![element]),
        })
    }

    fn callback_call(
        &mut self,
        target: Option<CallbackTarget<'m>>,
        name: &str,
        args: Vec<TypedExpr>,
    ) -> K {
        match target {
            Some(CallbackTarget::Local(name)) => {
                if self.source_kind == BodyKind::Function {
                    self.local_callees.insert(name.to_owned());
                }
                K::Call {
                    name: name.to_owned(),
                    args,
                }
            }
            Some(CallbackTarget::Extern(name)) => K::Call {
                name: name.to_owned(),
                args,
            },
            Some(CallbackTarget::Imported(id)) => K::ImportedCall { id, args },
            None => K::Call {
                name: name.to_owned(),
                args,
            },
        }
    }

    /// Apply the same data argument width and arity rules to a direct call,
    /// an instantiated template, and an invocation of its callback.
    pub(super) fn call_arguments<'t>(
        &mut self,
        name: &str,
        args: &[Expr],
        params: impl ExactSizeIterator<Item = (Option<&'t str>, &'t Type)>,
        dslx: bool,
    ) -> Result<Vec<TypedExpr>, String> {
        if args.len() != params.len() {
            return Err(format!(
                "'{name}' expects {} arguments, got {}",
                params.len(),
                args.len()
            ));
        }
        let outer = std::mem::replace(&mut self.dslx_value_context, dslx);
        let values = args
            .iter()
            .zip(params)
            .enumerate()
            .map(|(i, (arg, (label, ty)))| {
                let value = if dslx {
                    crate::infer_literal::check_argument_array_sizes(arg)
                        .and_then(|()| self.dslx_argument(arg, ty))
                } else {
                    self.expr(arg, Some(ty))
                };
                value.map_err(|e| match label {
                    Some(label) => format!("argument '{label}', call to '{name}': {e}"),
                    None => format!("argument {}, call to '{name}': {e}", i + 1),
                })
            })
            .collect();
        self.dslx_value_context = outer;
        values
    }
}

#[cfg(test)]
mod tests {
    use super::{Bindings, BlockCallbackFormal, BlockSpecializations, BodyChecker, BodyKind};
    use crate::ast::{CallbackParam, Type};
    use crate::check::{Declarations, MAX_TYPE_SIZE, StructTable};
    use crate::dslx_import::ImportResolver;
    use crate::{ImportOptions, check_module, parse_module};
    use std::collections::BTreeSet;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn block_callbacks_charge_a_concrete_imported_ir_signature_once() -> Result<(), String> {
        let dir = tempdir().map_err(|e| e.to_string())?;
        let mut source = String::from("pub struct S0 { value:u1 }\n");
        for level in 0..8 {
            source.push_str(&format!(
                "pub struct S{} {{ left:S{level}, right:S{level} }}\n",
                level + 1
            ));
        }
        source.push_str("pub fn echo<N:u32>(x:S8, unused:uN[N])->S8 { x }");
        fs::write(dir.path().join("tree.x"), source).map_err(|e| e.to_string())?;
        let module = parse_module("use tree; use tree::echo;")?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..ImportOptions::default()
        };
        let mut resolver = ImportResolver::new(&module.uses, &options)?;
        let tree = resolver.resolve_named("tree", "S8")?;
        let mut structs = StructTable::default();
        structs.update_imports(&resolver)?;
        let decls = Declarations::default();
        let mut registry = BlockSpecializations::new(BTreeSet::new());
        // S8 has 767 IR nodes. The remaining budget covers both sides of
        // one signature and the inputs of a second, even with metadata.
        registry.expansion_work = MAX_TYPE_SIZE - 2400;
        let mut checker = BodyChecker::new(
            &decls,
            &mut resolver,
            &mut structs,
            BodyKind::Block { has_reset: false },
        );
        checker.specializations = Some(&mut registry);
        let mut bindings = Bindings::new(&[]);
        let formal = |width| {
            BlockCallbackFormal::new(
                CallbackParam {
                    name: "F".into(),
                    params: vec![
                        tree.clone(),
                        Type::Bits {
                            signed: false,
                            width,
                        },
                    ],
                    result: tree.clone(),
                },
                &Bindings::new(&[]),
            )
        };
        let first =
            checker.resolve_block_callback("tree::echo", "left", 0, &formal(1), &mut bindings)?;
        let reused =
            checker.resolve_block_callback("echo", "right", 0, &formal(1), &mut bindings)?;
        assert_eq!(first.target, reused.target);
        let error = checker
            .resolve_block_callback("echo", "wide", 0, &formal(2), &mut bindings)
            .err()
            .expect("a second expanded signature exceeds the remaining budget");
        assert_eq!(
            error,
            format!(
                "callback bindings exceed the generic block specialization expansion limit of {MAX_TYPE_SIZE}"
            )
        );
        Ok(())
    }

    #[test]
    fn repeated_block_calls_share_fixed_signature_checks_and_inferred_results() -> Result<(), String>
    {
        let fields = vec!["u1"; 1024].join(",");
        // Each instance carries a scalar. The callback has wide fixed
        // components on both sides and binds Out to a wide tuple.
        let source = format!(
            r#"
fn pair(x: ({fields})) -> (({fields}), ({fields})) {{ (x, x) }}
fn head(x: ({fields})) -> u1 {{ x.0 }}
block keep<T, Out, F: fn(({fields})) -> (({fields}), Out), G: fn(Out) -> u1>(clk, x:T) -> T {{ x }}
block sample(clk, x:u8) -> u8 {{
    unroll_for! (_, acc) in u32:0..u32:1024 {{ let stage = keep<pair,head>(acc); stage }}(x)
}}
"#
        );
        let checked = check_module(&parse_module(&source)?)?;
        assert_eq!(checked.blocks().len(), 2);
        Ok(())
    }

    #[test]
    fn cached_block_callbacks_check_each_caller_scope_and_output_binding() -> Result<(), String> {
        let source = r#"
fn widen(x:u8)->u16 { x as u16 }
block keep<I,O,F:fn(I)->O>(clk,x:I,previous:O)->I { x }
block sample(clk,x:u8)->u8 {
    let first = keep<widen>(x, u16:0);
    let second = keep<widen>(first, u16:0);
    second
}
"#;
        check_module(&parse_module(source)?)?;
        for (replacement, expected) in [
            (
                "let second = keep<widen>(first, s16:0);",
                "conflicting types for type parameter 'O': inferred s16 and u16",
            ),
            (
                "let widen = first; let second = keep<widen>(first, u16:0);",
                "'widen' names a local value here",
            ),
        ] {
            let source = source.replace("let second = keep<widen>(first, u16:0);", replacement);
            let error = check_module(&parse_module(&source)?).unwrap_err();
            assert!(error.ends_with(expected), "{error}");
        }
        Ok(())
    }

    #[test]
    fn distinct_callbacks_charge_expanded_intermediate_block_types() -> Result<(), String> {
        let mut source = String::from("struct S0 { value:u1 }\n");
        for i in 0..17 {
            source.push_str(&format!("struct S{} {{ left:S{i}, right:S{i} }}\n", i + 1));
        }
        // The ports are small, but the unused callback result in each
        // specialized body lowers to a type with 393,215 IR nodes.
        source.push_str(
            r#"
extern fn a(x:u8)->S17;
extern fn b(x:u8)->S17;
extern fn c(x:u8)->S17;
block keep<T,F:fn(T)->S17>(clk,x:T)->T { let value = F(x); x }
block sample(clk,x:u8)->u8 {
    let first = keep<a>(x);
    let second = keep<b>(first);
    let third = keep<a>(second);
    third
}
"#,
        );
        check_module(&parse_module(&source)?)?;
        let distinct = source.replace("keep<a>(second)", "keep<c>(second)");
        let error = check_module(&parse_module(&distinct)?).unwrap_err();
        assert!(
            error.contains("block body exceeds the generic block specialization expansion limit"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn block_callback_calls_follow_lexical_value_bindings() -> Result<(), String> {
        let source = r#"
fn widen(x:u8)->u16 { x as u16 }
block child<T>(clk, x:T)->T { x }
block apply<I,O,F:fn(I)->O>(clk, x:I)->O { BODY }
block sample(clk, x:u8)->u16 { let stage = apply<widen>(x); stage }
"#;
        for body in ["let F = F(x); F", "map([x], |v| F(v))[0]"] {
            check_module(&parse_module(&source.replace("BODY", body))?)?;
        }
        for body in [
            "let F = x; F(x)",
            "reg F:I; F := x; F(x)",
            "let F = child(x); F(x)",
            "unroll_for! (_, F) in u32:0..u32:1 { F(x) }(zero!<O>())",
            "map([x], |F| F(F))[0]",
        ] {
            let error = check_module(&parse_module(&source.replace("BODY", body))?).unwrap_err();
            assert!(
                error.contains("names a local value") || error.contains("names a loop-local value"),
                "{body}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn block_extern_callbacks_use_the_direct_calls_literal_context() -> Result<(), String> {
        let source = r#"
extern fn signed(x:s8)->s8;
block stage<T,F:fn(s8)->s8>(clk, x:T)->s8 { BODY }
block sample(clk, x:u8)->s8 { let v = stage<signed>(x); v }
"#;
        for callee in ["signed", "F"] {
            let expr = format!("{callee}(0x80 + 1)");
            check_module(&parse_module(&source.replace("BODY", &expr))?)?;
            let lambda = format!("map([u8:0], |v| {expr})[0]");
            assert!(check_module(&parse_module(&source.replace("BODY", &lambda))?).is_err());
        }
        Ok(())
    }
}
