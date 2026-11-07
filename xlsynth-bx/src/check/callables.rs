// SPDX-License-Identifier: Apache-2.0

//! Bind a named callback in the instantiating scope and call it from the
//! template's definition scope without introducing a runtime function value.

use super::{BodyChecker, CallbackFormal, CallbackTarget, K, TypedExpr, compatible};
use crate::ast::{CallbackParam, Expr, Function, Type};

/// A visible declaration before its DSLX specialization (if any) is known.
#[derive(Eq, Hash, PartialEq)]
pub(super) enum CallbackSource<'m> {
    Local(&'m str),
    Imported { module: String, member: String },
}

/// One formal has a fixed resolved signature for the lifetime of the module.
/// Its validated actual can be reused at any site resolving the same source.
#[derive(Eq, Hash, PartialEq)]
pub(super) struct CallbackBinding<'m> {
    template: &'m str,
    formal: usize,
    source: CallbackSource<'m>,
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// A callback name is available only as the callee of a direct call.
    pub(super) fn callback_index(&self, name: &str) -> Option<usize> {
        self.source_template?.callback_indices.get(name).copied()
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
        if self
            .specialization
            .is_some_and(|spec| spec.bindings.declares(name))
        {
            return Err(format!("'{name}' names a type parameter here"));
        }
        if !self.decls.visible(name, self.function_position) {
            return Err(format!("function '{name}' is declared later in the module"));
        }
        let &template = self
            .decls
            .function_templates
            .get(name)
            .ok_or_else(|| format!("'{name}' is not a function with callable parameters"))?;
        if callback_count != template.callback_params.len() {
            return Err(format!(
                "'{name}' expects {} callable arguments, got {callback_count}",
                template.callback_params.len()
            ));
        }
        Ok(template)
    }

    /// Only a directly named, visible local function or imported DSLX member
    /// can supply a static callback. Its spelling never reaches the data IR.
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
        if self
            .specialization
            .is_some_and(|spec| spec.bindings.declares(head))
        {
            return Err(format!("'{head}' names a type parameter here"));
        }
        if !self.decls.visible(name, self.function_position) {
            return Err(format!("callback '{name}' is declared later in the module"));
        }
        if let Some((module, member)) = self.resolver.lookup(name)? {
            self.resolver.check_function_name(&module, &member)?;
            return Ok(CallbackSource::Imported { module, member });
        }
        if let Some((&callee, f)) = self.decls.functions.get_key_value(name)
            && f.local
            && f.callback_params.is_empty()
        {
            return Ok(CallbackSource::Local(callee));
        }
        Err(format!(
            "callback '{name}' must name a concrete BX function or a public DSLX function"
        ))
    }

    fn check_callback_signature(
        &mut self,
        name: &str,
        formal: &CallbackParam,
        source: &CallbackSource<'m>,
    ) -> Result<CallbackTarget<'m>, String> {
        match source {
            CallbackSource::Local(callee) => {
                let f = &self.decls.functions[*callee];
                if f.params.len() != formal.params.len()
                    || !f
                        .params
                        .iter()
                        .zip(&formal.params)
                        .all(|(p, expected)| compatible(&p.ty, expected))
                {
                    let actual = f
                        .params
                        .iter()
                        .map(|p| p.ty.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let expected = formal
                        .params
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "callback '{name}' has parameter types ({actual}), expected ({expected})"
                    ));
                }
                if !compatible(&f.result, &formal.result) {
                    return Err(format!(
                        "callback '{name}' returns {}, expected {}",
                        f.result, formal.result
                    ));
                }
                Ok(CallbackTarget::Local(callee))
            }
            CallbackSource::Imported { module, member } => {
                let (id, result) = self
                    .resolver
                    .resolve(module, member, formal.params.clone())?;
                self.structs.update_imports(self.resolver)?;
                self.structs.validate_type(&result)?;
                if !compatible(&result, &formal.result) {
                    return Err(format!(
                        "callback '{name}' returns {result}, expected {}",
                        formal.result
                    ));
                }
                Ok(CallbackTarget::Imported(id))
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
            source: self.callback_source(actual, |head| self.names.contains(head))?,
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
        let target = self.check_callback_signature(actual, &formal.signature, &key.source)?;
        self.function_registry
            .as_mut()
            .expect("module function registry")
            .record_validated(key, target)?;
        Ok(target)
    }

    /// Bind callable identities at the instantiation site. Only data
    /// arguments become operands of the call to the generated function.
    pub(super) fn parametric_call(
        &mut self,
        name: &str,
        callbacks: &[String],
        args: &[Expr],
    ) -> Result<(Type, K), String> {
        let template = self.callable_template(name, self.names.contains(name), callbacks.len())?;
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
        let symbol = self.intern_function(template, targets)?;
        Ok((interface.result.clone(), K::Call { name: symbol, args }))
    }

    /// Typecheck against the declared signature in both the abstract body
    /// and its concrete instances. Abstract call nodes are never lowered.
    pub(super) fn call_formal(&mut self, index: usize, args: &[Expr]) -> Result<(Type, K), String> {
        let formal = &self
            .source_template
            .expect("indexed callable parameter")
            .callback_params[index]
            .signature;
        let args = self.call_arguments(
            &formal.name,
            args,
            formal.params.iter().map(|ty| (None, ty)),
            true,
        )?;
        let call = match self.function_spec.map(|spec| spec.targets[index]) {
            Some(CallbackTarget::Local(name)) => {
                self.local_callees.insert(name.to_owned());
                K::Call {
                    name: name.to_owned(),
                    args,
                }
            }
            Some(CallbackTarget::Imported(id)) => K::ImportedCall { id, args },
            None => K::Call {
                name: formal.name.clone(),
                args,
            },
        };
        Ok((formal.result.clone(), call))
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
                    self.dslx_argument(arg, ty)
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
