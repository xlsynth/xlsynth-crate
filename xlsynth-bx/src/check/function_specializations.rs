// SPDX-License-Identifier: Apache-2.0

//! Intern and schedule local functions specialized on statically named callees.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use super::callables::CallbackBinding;
use super::{
    BodyChecker, BodyKind, CheckedFunction, ConstantBudgets, Declarations, Function,
    ImportResolver, MAX_ELABORATION_DEPTH, MAX_TYPE_SIZE, StructTable, dependency_order,
};
use crate::ast::{ConstDef, Module};

/// An actual callback resolved in the caller's scope. An import ID includes
/// its concrete DSLX specialization as well as its declaring module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum CallbackTarget<'m> {
    Local(&'m str),
    /// The BX declaration supplies the source types. Lowering verifies its
    /// linked IR function's signature and the purity of its reachable callees.
    Extern(&'m str),
    Imported(usize),
}

/// One emitted local function and its callbacks in formal declaration order.
pub(super) struct FunctionSpecialization<'m> {
    pub(super) template: &'m Function,
    pub(super) name: String,
    pub(super) targets: Vec<CallbackTarget<'m>>,
}

/// Keep concrete call edges until all block- and function-initiated
/// specializations have been discovered.
pub(super) struct FunctionBody {
    function: CheckedFunction,
    callees: BTreeSet<String>,
}

/// Share a single specialized function and elaboration budget across all
/// call sites in this module.
pub(super) struct FunctionSpecializations<'m> {
    cache: HashMap<(&'m str, Vec<CallbackTarget<'m>>), String>,
    validated: HashMap<CallbackBinding<'m>, CallbackTarget<'m>>,
    pending: VecDeque<FunctionSpecialization<'m>>,
    sources: BTreeMap<String, &'m str>,
    reserved: BTreeSet<String>,
    next_id: usize,
    pub(super) expansion_work: usize,
    pub(super) reference_copy_work: Cell<usize>,
    pub(super) constant_budgets: ConstantBudgets,
}

impl<'m> FunctionSpecializations<'m> {
    pub(super) fn new(reserved: BTreeSet<String>) -> Self {
        Self {
            cache: HashMap::new(),
            validated: HashMap::new(),
            pending: VecDeque::new(),
            sources: BTreeMap::new(),
            reserved,
            next_id: 0,
            expansion_work: 0,
            reference_copy_work: Cell::new(0),
            constant_budgets: ConstantBudgets::default(),
        }
    }

    pub(super) fn validated_target(&self, key: &CallbackBinding<'m>) -> Option<CallbackTarget<'m>> {
        self.validated.get(key).copied()
    }

    /// Abstract template checks can validate an actual without requesting an
    /// emitted specialization. Bound those entries along with concrete keys.
    pub(super) fn record_validated(
        &mut self,
        key: CallbackBinding<'m>,
        target: CallbackTarget<'m>,
    ) -> Result<(), String> {
        self.charge_bindings(1)?;
        self.validated.insert(key, target);
        Ok(())
    }

    fn charge_bindings(&mut self, count: usize) -> Result<(), String> {
        self.expansion_work = self.expansion_work.saturating_add(count);
        if self.expansion_work > MAX_TYPE_SIZE {
            return Err(format!(
                "callback bindings exceed the function specialization expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        Ok(())
    }

    /// Queue the first use of a concrete template/callback tuple. Generated
    /// function names and generated block names use disjoint suffixes.
    fn intern(
        &mut self,
        template: &'m Function,
        targets: Vec<CallbackTarget<'m>>,
    ) -> Result<String, String> {
        let key = (template.name.as_str(), targets);
        if let Some(name) = self.cache.get(&key) {
            return Ok(name.clone());
        }
        const MAX_SPECIALIZATIONS: usize = 4096;
        if self.cache.len() >= MAX_SPECIALIZATIONS {
            return Err(format!(
                "function specialization limit of {MAX_SPECIALIZATIONS} exceeded"
            ));
        }
        self.charge_bindings(key.1.len())?;
        let name = loop {
            let candidate = format!("{}__fn{}", template.name, self.next_id);
            self.next_id += 1;
            if !self.reserved.contains(&candidate) {
                break candidate;
            }
        };
        self.pending.push_back(FunctionSpecialization {
            template,
            name: name.clone(),
            targets: key.1.clone(),
        });
        self.sources.insert(name.clone(), key.0);
        self.cache.insert(key, name.clone());
        Ok(name)
    }

    /// Check functions and evaluate constants together in declaration order.
    /// Each initializer can call concrete functions already checked. A
    /// template's abstract callback signatures suffice to diagnose errors in an
    /// unused body. Only concrete calls schedule emitted definitions.
    pub(super) fn check_sources(
        &mut self,
        module: &'m Module,
        decls: &mut Declarations<'m>,
        resolver: &mut ImportResolver,
        structs: &mut StructTable,
    ) -> Result<Vec<FunctionBody>, String> {
        enum Definition<'m> {
            Constant(&'m ConstDef),
            Function(&'m Function),
        }
        let mut definitions = module
            .constants
            .iter()
            .map(|c| (c.name.as_str(), Definition::Constant(c)))
            .chain(
                module
                    .functions
                    .iter()
                    .map(|f| (f.name.as_str(), Definition::Function(f))),
            )
            .collect::<Vec<_>>();
        definitions
            .sort_by_key(|(name, _)| decls.positions.get(name).copied().unwrap_or(usize::MAX));
        let mut bodies: Vec<FunctionBody> = Vec::with_capacity(module.functions.len());
        let mut indices = BTreeMap::<&str, usize>::new();
        let mut depths = BTreeMap::new();
        for (name, definition) in definitions {
            let earlier = |name: &str| indices.get(name).map(|&i| &bodies[i].function);
            let mut checker = BodyChecker::new(decls, resolver, structs, BodyKind::Function);
            checker.function_registry = Some(self);
            let function = match definition {
                Definition::Function(f) => f,
                Definition::Constant(c) => {
                    let value = checker
                        .check_constant(c, &earlier)
                        .map_err(|e| format!("const '{name}': {e}"))?;
                    decls.constants.insert(name, value);
                    continue;
                }
            };
            let body = checker
                .check_function(function)
                .map_err(|e| format!("fn '{}': {e}", function.name))?;
            let callees = checker.local_callees;
            let mut depth = 1;
            for symbol in &callees {
                let callee = self.sources.get(symbol).copied().unwrap_or(symbol);
                if callee == function.name {
                    return Err(format!(
                        "recursive function dependency involving '{callee}'"
                    ));
                }
                let prior = depths.get(callee).ok_or_else(|| {
                    format!("function '{callee}' is declared later in the module")
                })?;
                depth = depth.max(prior + 1);
            }
            if depth > MAX_ELABORATION_DEPTH {
                return Err(format!(
                    "function dependency depth exceeds {MAX_ELABORATION_DEPTH} levels at '{}'",
                    function.name
                ));
            }
            depths.insert(function.name.as_str(), depth);
            if function.callback_params.is_empty() {
                indices.insert(name, bodies.len());
                bodies.push(FunctionBody {
                    function: body,
                    callees,
                });
            }
        }
        Ok(bodies)
    }

    /// Check queued bodies in their definition scope, then order every
    /// concrete call edge, including callbacks declared after a template.
    pub(super) fn finish(
        &mut self,
        mut bodies: Vec<FunctionBody>,
        decls: &Declarations<'m>,
        resolver: &mut ImportResolver,
        structs: &mut StructTable,
    ) -> Result<Vec<CheckedFunction>, String> {
        while let Some(spec) = self.pending.pop_front() {
            let mut checker = BodyChecker::new(decls, resolver, structs, BodyKind::Function);
            checker.function_registry = Some(self);
            checker.function_spec = Some(&spec);
            let function = checker
                .check_function(spec.template)
                .map_err(|e| format!("fn '{}' ({}): {e}", spec.template.name, spec.name))?;
            bodies.push(FunctionBody {
                function,
                callees: checker.local_callees,
            });
        }
        let graph = bodies
            .iter()
            .map(|body| {
                (
                    body.function.name.as_str(),
                    body.callees.iter().map(String::as_str).collect(),
                )
            })
            .collect::<Vec<_>>();
        let order = dependency_order("function", &graph)?;
        let mut functions: Vec<_> = bodies.into_iter().map(|b| Some(b.function)).collect();
        Ok(order
            .into_iter()
            .map(|i| functions[i].take().expect("unique function order"))
            .collect())
    }

    pub(super) fn generated_names(self) -> BTreeSet<String> {
        self.sources.into_keys().collect()
    }
}

impl<'m> BodyChecker<'_, 'm> {
    /// Resolve the name used by a checked call and record its dependency.
    /// While validating a template in isolation, a call contributes only a
    /// source dependency, since the checked abstract body is discarded.
    pub(super) fn intern_function(
        &mut self,
        template: &'m Function,
        targets: Vec<CallbackTarget<'m>>,
    ) -> Result<String, String> {
        let name = if self.source_template.is_some() && self.function_spec.is_none() {
            template.name.clone()
        } else {
            self.function_registry
                .as_mut()
                .expect("module function registry")
                .intern(template, targets)?
        };
        if self.source_kind == BodyKind::Function {
            self.local_callees.insert(name.clone());
        }
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{check_module, parse_module};

    /// S16 expands to 262,143 IR type nodes from one nominal BX reference.
    fn wide_struct_declarations() -> String {
        let mut declarations = String::from("struct S0 { a: u1, b: u1 }\n");
        for level in 1..=16 {
            let previous = level - 1;
            declarations.push_str(&format!(
                "struct S{level} {{ a: S{previous}, b: S{previous} }}\n"
            ));
        }
        declarations
    }

    #[test]
    fn cached_bindings_compare_nominal_types_without_expanding_their_fields() -> Result<(), String>
    {
        let declarations = wide_struct_declarations();
        let source = format!(
            r#"
{declarations}
fn id(x: S16) -> S16 {{ x }}
fn ignore<F: fn(S16) -> S16>(x: u8) -> u8 {{ x }}
block stage<T>(clk, x: T) -> u8 {{ ignore<id>(ignore<id>(ignore<id>(x))) }}
block sample(clk, x: u8) -> u8 {{ let result = stage(x); result }}
"#
        );
        let checked = check_module(&parse_module(&source)?)?;
        assert_eq!(checked.functions().len(), 2);
        Ok(())
    }

    #[test]
    fn distinct_specializations_share_the_expanded_signature_budget() -> Result<(), String> {
        let declarations = wide_struct_declarations();
        let source = format!(
            r#"
{declarations}
fn inc(x: u8) -> u8 {{ x + u8:1 }}
fn dec(x: u8) -> u8 {{ x - u8:1 }}
fn keep<F: fn(u8) -> u8>(x: S16) -> S16 {{ x }}
block sample(clk, x: S16) -> u1 {{
    let a = keep<inc>(x);
    let b = keep<inc>(x);
    u1:0
}}
"#
        );
        check_module(&parse_module(&source)?)?;
        let distinct = source.replace("let b = keep<inc>", "let b = keep<dec>");
        let error = check_module(&parse_module(&distinct)?).unwrap_err();
        assert!(
            error.contains("function specialization expansion limit"),
            "{error}"
        );
        Ok(())
    }
}
