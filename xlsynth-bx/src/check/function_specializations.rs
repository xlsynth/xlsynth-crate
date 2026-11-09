// SPDX-License-Identifier: Apache-2.0

//! Intern and schedule local functions specialized on named callees and values.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use super::callables::CallbackBinding;
use super::{
    BodyChecker, BodyKind, CheckedFunction, ConstantBudgets, Declarations, Function,
    FunctionInterface, ImportResolver, MAX_ELABORATION_DEPTH, MAX_SPECIALIZATIONS, MAX_TYPE_SIZE,
    StructTable, dependency_order, literal_copy_work, source_type_nodes,
};
use crate::ast::{ConstDef, EnumDef, ExternFn, Module};
use crate::typed::BoundParametricValue;
use crate::typed::{AvailableFunctions, constant_work};

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
    pub(super) name: String,
    pub(super) targets: Vec<CallbackTarget<'m>>,
    pub(super) value_signature: Option<Rc<BoundValueSignature>>,
}

/// Values and signature resolved in the template's definition scope. Callers
/// sharing supplied values also share the evaluation of dependent defaults.
pub(super) struct BoundValueSignature {
    pub(super) values: Vec<BoundParametricValue>,
    pub(super) interface: FunctionInterface,
}

/// Nominal callback identities and typed constexpr values in formal order.
#[derive(Clone, Eq, Hash, PartialEq)]
struct FunctionKey<'m> {
    template: &'m str,
    targets: Vec<CallbackTarget<'m>>,
    values: Vec<BoundParametricValue>,
}

/// Keep concrete call edges until all block- and function-initiated
/// specializations have been discovered.
pub(super) struct FunctionBody {
    function: Rc<CheckedFunction>,
    callees: BTreeSet<String>,
}

/// Share a single specialized function and elaboration budget across all
/// call sites in this module.
pub(super) struct FunctionSpecializations<'m> {
    cache: HashMap<FunctionKey<'m>, String>,
    value_signatures:
        HashMap<&'m str, HashMap<Vec<Option<BoundParametricValue>>, Rc<BoundValueSignature>>>,
    validated: HashMap<CallbackBinding<'m>, CallbackTarget<'m>>,
    active: Vec<&'m str>,
    generated: Vec<FunctionBody>,
    pub(super) available: AvailableFunctions,
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
            value_signatures: HashMap::new(),
            validated: HashMap::new(),
            active: Vec::new(),
            generated: Vec::new(),
            available: Default::default(),
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
                "parametric bindings exceed the function specialization expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        Ok(())
    }

    /// Resolve each partial value environment once. The definition's defaults
    /// and dimensions can themselves instantiate previously declared templates.
    pub(super) fn cached_value_signature(
        &mut self,
        f: &'m Function,
        provided: &[Option<BoundParametricValue>],
    ) -> Result<Option<Rc<BoundValueSignature>>, String> {
        let work = provided.iter().flatten().fold(provided.len(), |sum, v| {
            sum.saturating_add(constant_work(&v.value))
        });
        self.charge_bindings(work)?;
        Ok(self
            .value_signatures
            .get(f.name.as_str())
            .and_then(|signatures| signatures.get(provided))
            .cloned())
    }

    /// Retain a resolved signature after its definition and all nested defaults
    /// have been checked. Charge the types and concrete values held by the
    /// cache.
    pub(super) fn cache_value_signature(
        &mut self,
        f: &'m Function,
        provided: Vec<Option<BoundParametricValue>>,
        bound: BoundValueSignature,
    ) -> Result<Rc<BoundValueSignature>, String> {
        let interface = &bound.interface;
        let mut work = source_type_nodes(&interface.result);
        for param in &interface.params {
            work = work.saturating_add(source_type_nodes(&param.ty));
        }
        for callback in &interface.callback_params {
            work = work.saturating_add(callback.binding_work);
        }
        for value in &bound.values {
            work = work.saturating_add(1 + literal_copy_work(&value.value));
        }
        self.charge_bindings(work)?;
        let bound = Rc::new(bound);
        self.value_signatures
            .entry(&f.name)
            .or_default()
            .insert(provided, Rc::clone(&bound));
        Ok(bound)
    }

    /// During definition checking, defaults and dimensions consume the same
    /// module-wide budgets as the body that uses their concrete values.
    pub(super) fn checking_definition(&self) -> bool {
        !self.active.is_empty()
    }

    pub(super) fn enter(&mut self, name: &'m str) -> Result<(), String> {
        if self.active.contains(&name) {
            return Err(format!("recursive function dependency involving '{name}'"));
        }
        if self.active.len() >= MAX_ELABORATION_DEPTH {
            return Err(format!(
                "function dependency depth exceeds {MAX_ELABORATION_DEPTH} levels at '{name}'"
            ));
        }
        self.active.push(name);
        Ok(())
    }

    pub(super) fn leave(&mut self) {
        self.active.pop();
    }

    /// Allocate a unique symbol after accounting for the specialization key.
    fn allocate(&mut self, key: &FunctionKey<'m>) -> Result<String, String> {
        if self.cache.len() >= MAX_SPECIALIZATIONS {
            return Err(format!(
                "function specialization limit of {MAX_SPECIALIZATIONS} exceeded"
            ));
        }
        let name = loop {
            let candidate = format!("{}__fn{}", key.template, self.next_id);
            self.next_id += 1;
            if !self.reserved.contains(&candidate) {
                break candidate;
            }
        };
        self.sources.insert(name.clone(), key.template);
        self.cache.insert(key.clone(), name.clone());
        Ok(name)
    }

    fn record_body(&self, function: CheckedFunction, callees: BTreeSet<String>) -> FunctionBody {
        let function = Rc::new(function);
        self.available
            .borrow_mut()
            .insert(function.name.clone(), Rc::clone(&function));
        FunctionBody { function, callees }
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
            Struct(usize),
            Extern(&'m ExternFn),
            Enum(&'m EnumDef),
            Constant(&'m ConstDef),
            Function(&'m Function),
        }
        let position = |name: &str| decls.positions.get(name).copied().unwrap_or(usize::MAX);
        let mut definitions = module
            .enums
            .iter()
            .map(|e| (position(&e.name), Definition::Enum(e)))
            .chain(
                module
                    .externs
                    .iter()
                    .map(|f| (position(&f.name), Definition::Extern(f))),
            )
            .chain(
                module
                    .constants
                    .iter()
                    .map(|c| (position(&c.name), Definition::Constant(c))),
            )
            .chain(
                module
                    .functions
                    .iter()
                    .map(|f| (position(&f.name), Definition::Function(f))),
            )
            .collect::<Vec<_>>();
        // A struct can refer to another local struct declared later. Index it
        // in dependency order once the representations of its enums are known.
        definitions.extend(
            StructTable::struct_schedule(module, position)?
                .into_iter()
                .map(|(at, index)| (at, Definition::Struct(index))),
        );
        definitions.sort_by_key(|(position, _)| *position);
        structs.update_imports(resolver)?;
        let mut bodies: Vec<FunctionBody> = Vec::with_capacity(module.functions.len());
        let mut depths = BTreeMap::new();
        for (_, definition) in definitions {
            let function = match definition {
                Definition::Struct(index) => {
                    structs.definitions[index].fields =
                        super::resolve_fields(&module.structs[index].fields, resolver, structs)?;
                    structs.update_imports(resolver)?;
                    structs.index(index)?;
                    continue;
                }
                Definition::Extern(f) => {
                    decls
                        .add_function(&f.name, &f.params, &f.result, None, resolver, structs)
                        .map_err(|e| format!("extern '{}': {e}", f.name))?;
                    continue;
                }
                Definition::Function(f) => {
                    let at = decls.positions.get(f.name.as_str()).copied();
                    for ty in f.params.iter().map(|p| &p.ty).chain([&f.result]) {
                        decls.check_type_visibility(ty, at)?;
                    }
                    decls
                        .add_function(&f.name, &f.params, &f.result, Some(f), resolver, structs)
                        .map_err(|e| format!("fn '{}': {e}", f.name))?;
                    f
                }
                Definition::Constant(c) => {
                    let mut checker =
                        BodyChecker::new(decls, resolver, structs, BodyKind::Function);
                    checker.function_registry = Some(self);
                    let value = checker
                        .check_constant(c)
                        .map_err(|e| format!("const '{}': {e}", c.name))?;
                    decls.constants.insert(&c.name, value);
                    continue;
                }
                Definition::Enum(e) => {
                    let mut checker =
                        BodyChecker::new(decls, resolver, structs, BodyKind::Function);
                    checker.function_registry = Some(self);
                    let checked = checker
                        .check_enum(e)
                        .map_err(|err| format!("enum '{}': {err}", e.name))?;
                    structs.enums.insert(e.name.clone(), checked);
                    continue;
                }
            };
            if !function.value_parametrics.is_empty() {
                // Dimensions and the function body are checked with the actual
                // bindings of each instantiation, as in DSLX.
                let mut checker = BodyChecker::new(decls, resolver, structs, BodyKind::Function);
                checker
                    .check_value_template_names(function)
                    .map_err(|e| format!("fn '{}': {e}", function.name))?;
                depths.insert(function.name.as_str(), 1);
                continue;
            }
            let mut checker = BodyChecker::new(decls, resolver, structs, BodyKind::Function);
            checker.function_registry = Some(self);
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
                bodies.push(self.record_body(body, callees));
            }
        }
        Ok(bodies)
    }

    /// Check queued bodies in their definition scope, then order every
    /// concrete call edge, including callbacks declared after a template.
    pub(super) fn finish(
        &mut self,
        mut bodies: Vec<FunctionBody>,
    ) -> Result<Vec<CheckedFunction>, String> {
        bodies.append(&mut self.generated);
        self.available.borrow_mut().clear();
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
        let mut functions: Vec<_> = bodies
            .into_iter()
            .map(|b| Some(Rc::unwrap_or_clone(b.function)))
            .collect();
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
        value_signature: Option<Rc<BoundValueSignature>>,
    ) -> Result<String, String> {
        // Abstract callback bodies normally validate calls without emitting
        // them. A constexpr actual still needs its concrete callee to run.
        let name = if self.source_template.is_some()
            && self.function_spec.is_none()
            && self.constant_functions.is_none()
        {
            template.name.clone()
        } else {
            let registry = self
                .function_registry
                .as_deref_mut()
                .expect("module function registry");
            if registry.active.contains(&template.name.as_str()) {
                return Err(format!(
                    "recursive function dependency involving '{}'",
                    template.name
                ));
            }
            let values = value_signature
                .as_ref()
                .map_or(&[][..], |s| s.values.as_slice());
            let work = values.iter().fold(targets.len() + values.len(), |sum, v| {
                sum.saturating_add(constant_work(&v.value))
            });
            registry.charge_bindings(work)?;
            let key = FunctionKey {
                template: &template.name,
                targets,
                values: values.to_vec(),
            };
            if let Some(name) = registry.cache.get(&key) {
                name.clone()
            } else {
                let name = registry.allocate(&key)?;
                registry.enter(&template.name)?;
                let spec = FunctionSpecialization {
                    name: name.clone(),
                    targets: key.targets,
                    value_signature,
                };
                let body = stacker::maybe_grow(1024 * 1024, 2 * 1024 * 1024, || {
                    let mut checker = BodyChecker::new(
                        self.decls,
                        self.resolver,
                        self.structs,
                        BodyKind::Function,
                    );
                    checker.function_registry = Some(registry);
                    checker.function_spec = Some(&spec);
                    checker
                        .check_function(template)
                        .map(|body| (body, checker.local_callees))
                });
                registry.leave();
                let (body, callees) =
                    body.map_err(|e| format!("fn '{}' ({name}): {e}", template.name))?;
                let body = registry.record_body(body, callees);
                registry.generated.push(body);
                name
            }
        };
        if self.source_kind == BodyKind::Function {
            self.local_callees.insert(name.clone());
        }
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{Expr, Type};
    use crate::{check_module, parse_module};
    use std::fmt::Write;
    use xlsynth_pir::IrValue;

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
    fn calls_in_dependent_defaults_share_resolved_signatures() -> Result<(), String> {
        let mut source = String::from("fn f0<N:u32={u32:1}>()->u32{N}\n");
        for level in 1..31 {
            writeln!(
                source,
                "fn f{level}<N:u32={{f{prev}()+f{prev}()}}>()->u32{{N}}",
                prev = level - 1
            )
            .unwrap();
        }
        source.push_str("const ANSWER = f30();");
        let module = check_module(&parse_module(&source)?)?;
        assert_eq!(module.functions().len(), 31);
        assert_eq!(
            module.constants()[0].value,
            IrValue::make_ubits(32, 1 << 30).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn programmatic_symbolic_signatures_are_bounded_before_copying() -> Result<(), String> {
        let mut module = parse_module("fn probe<N:u32>(x:uN[N])->u32{N}")?;
        let input = &mut module.functions[0].params[0].ty;
        for _ in 0..16_384 {
            let inner = std::mem::replace(input, Type::Bool);
            *input = Type::ParametricBits {
                signed: Box::new(Expr::Bool(false)),
                width: Box::new(Expr::Zero(inner)),
            };
        }
        let checked = check_module(&module);
        // Dismantle the synthetic input iteratively on the test thread too.
        let input = &mut module.functions[0].params[0].ty;
        while let Type::ParametricBits { width, .. } = std::mem::replace(input, Type::Bool) {
            let Expr::Zero(inner) = *width else {
                break;
            };
            *input = inner;
        }
        assert_eq!(
            checked.unwrap_err(),
            "fn 'probe': value-parametric template nesting exceeds 64 levels"
        );
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
