// SPDX-License-Identifier: Apache-2.0

//! Intern and schedule local functions specialized on types, callees and
//! values.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use super::callables::{CallbackBinding, CallbackSource};
use super::generic_blocks::{Bindings, TypeArgument};
use super::{
    BodyChecker, BodyKind, CheckedFunction, ConstantBudgets, Declarations, Function,
    FunctionInterface, ImportResolver, MAX_ELABORATION_DEPTH, MAX_SPECIALIZATIONS, MAX_TYPE_SIZE,
    StructTable, TypeResolver, dependency_order, source_type_nodes,
};
use crate::ast::{ConstAssert, ConstDef, EnumDef, ExternFn, Module, Type, TypeAlias};
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

/// Concrete signature and parametric environment in the template's definition
/// scope.
pub(super) struct BoundValueSignature {
    pub(super) type_bindings: Bindings,
    pub(super) values: Vec<BoundParametricValue>,
    pub(super) interface: FunctionInterface,
}

/// Inferred types and supplied values before evaluating dependent defaults.
#[derive(Eq, Hash, PartialEq)]
pub(super) struct SuppliedParametrics {
    pub(super) types: Vec<TypeArgument>,
    pub(super) values: Vec<Option<BoundParametricValue>>,
}

/// An explicit formal sees the data-inferred types and the ordered prefix of
/// earlier callback sources and constexpr values. Intern each prefix once so
/// successive annotations need not copy or hash all preceding arguments.
#[derive(Eq, Hash, PartialEq)]
pub(super) enum ExplicitEnvironment<'m> {
    DataTypes(&'m str, Vec<Option<TypeArgument>>),
    Value(usize, BoundParametricValue),
    Callback(usize, CallbackSource<'m>),
}

/// The target and type variables determined by a callback in one environment.
pub(super) struct InferredCallback<'m> {
    pub(super) target: CallbackTarget<'m>,
    pub(super) results: Bindings,
}

/// Nominal types, callback identities and typed constexpr values in formal
/// order.
#[derive(Clone, Eq, Hash, PartialEq)]
struct FunctionKey<'m> {
    template: &'m str,
    types: Vec<TypeArgument>,
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
    value_signatures: HashMap<&'m str, HashMap<SuppliedParametrics, Rc<BoundValueSignature>>>,
    explicit_environments: HashMap<ExplicitEnvironment<'m>, usize>,
    explicit_value_types: HashMap<usize, Type>,
    inferred_callbacks: HashMap<usize, Rc<InferredCallback<'m>>>,
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
            explicit_environments: HashMap::new(),
            explicit_value_types: HashMap::new(),
            inferred_callbacks: HashMap::new(),
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

    pub(super) fn charge_bindings(&mut self, count: usize) -> Result<(), String> {
        self.expansion_work = self.expansion_work.saturating_add(count);
        if self.expansion_work > MAX_TYPE_SIZE {
            return Err(format!(
                "parametric bindings exceed the function specialization expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        Ok(())
    }

    /// Charge hashing, comparison and retained key storage before looking up
    /// the next formal's declaration environment.
    pub(super) fn explicit_environment(
        &mut self,
        key: ExplicitEnvironment<'m>,
    ) -> Result<usize, String> {
        let work = match &key {
            ExplicitEnvironment::DataTypes(_, types) => types
                .iter()
                .flatten()
                .fold(types.len(), |sum, ty| sum.saturating_add(ty.source_nodes)),
            ExplicitEnvironment::Value(_, value) => {
                source_type_nodes(&value.ty).saturating_add(constant_work(&value.value))
            }
            ExplicitEnvironment::Callback(_, _) => 1,
        };
        self.charge_bindings(1usize.saturating_add(work))?;
        let next = self.explicit_environments.len();
        Ok(*self.explicit_environments.entry(key).or_insert(next))
    }

    pub(super) fn explicit_value_type(&self, environment: usize) -> Option<Type> {
        self.explicit_value_types.get(&environment).cloned()
    }

    pub(super) fn record_explicit_value_type(&mut self, environment: usize, ty: &Type) {
        // Value annotations are scalar or nominal leaves. The environment
        // lookup already charged storage for this entry.
        self.explicit_value_types.insert(environment, ty.clone());
    }

    pub(super) fn inferred_callback(&self, environment: usize) -> Option<Rc<InferredCallback<'m>>> {
        self.inferred_callbacks.get(&environment).cloned()
    }

    pub(super) fn record_inferred_callback(
        &mut self,
        environment: usize,
        bound: Rc<InferredCallback<'m>>,
    ) {
        self.inferred_callbacks.insert(environment, bound);
    }

    /// Resolve each partial value environment once. The definition's defaults
    /// and dimensions can themselves instantiate previously declared templates.
    pub(super) fn cached_value_signature(
        &mut self,
        f: &'m Function,
        supplied: &SuppliedParametrics,
    ) -> Result<Option<Rc<BoundValueSignature>>, String> {
        let type_work: usize = supplied.types.iter().map(|t| t.source_nodes).sum();
        let work = supplied
            .values
            .iter()
            .flatten()
            .fold(type_work + supplied.values.len(), |sum, v| {
                sum.saturating_add(constant_work(&v.value))
            });
        self.charge_bindings(work)?;
        Ok(self
            .value_signatures
            .get(f.name.as_str())
            .and_then(|signatures| signatures.get(supplied))
            .cloned())
    }

    /// Charge the result and retain the signature. Its parameter types and
    /// values were charged individually as the definition was resolved.
    pub(super) fn cache_value_signature(
        &mut self,
        f: &'m Function,
        supplied: SuppliedParametrics,
        bound: BoundValueSignature,
    ) -> Result<Rc<BoundValueSignature>, String> {
        self.charge_bindings(source_type_nodes(&bound.interface.result))?;
        let bound = Rc::new(bound);
        self.value_signatures
            .entry(&f.name)
            .or_default()
            .insert(supplied, Rc::clone(&bound));
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
        structs: &mut StructTable<'m>,
    ) -> Result<Vec<FunctionBody>, String> {
        enum Definition<'m> {
            Struct(usize),
            Extern(&'m ExternFn),
            Enum(&'m EnumDef),
            Alias(&'m TypeAlias),
            Constant(&'m ConstDef),
            ConstAssert(&'m str, &'m ConstAssert),
            Function(&'m Function),
        }
        let position = |name: &str| decls.positions.get(name).copied().unwrap_or(usize::MAX);
        let mut definitions = module
            .enums
            .iter()
            .map(|e| (position(&e.name), Definition::Enum(e)))
            .chain(
                module
                    .type_aliases
                    .iter()
                    .map(|alias| (position(&alias.name), Definition::Alias(alias))),
            )
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
                    .const_asserts
                    .iter()
                    .map(|(label, a)| (position(label), Definition::ConstAssert(label, a))),
            )
            .chain(
                module
                    .functions
                    .iter()
                    .map(|f| (position(&f.name), Definition::Function(f))),
            )
            .collect::<Vec<_>>();
        // A struct can refer to another local struct declared later. Index it
        // in dependency order once its enum representations and dimensions are
        // known.
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
                Definition::Alias(alias) => {
                    let mut checker =
                        BodyChecker::new(decls, resolver, structs, BodyKind::Function);
                    checker.function_registry = Some(self);
                    checker.function_position = decls.positions.get(alias.name.as_str()).copied();
                    checker.warning_owner = Some(&alias.name);
                    checker.zero_binding = decls.zero_binding(&alias.name);
                    checker.enable_constant_functions();
                    let ty = checker
                        .source_type(&alias.ty)
                        .map_err(|e| format!("type alias '{}': {e}", alias.name))?;
                    let shape = structs.type_shape(&ty, MAX_ELABORATION_DEPTH)?;
                    let copy_work = (shape.source_nodes - 1)
                        .saturating_add(shape.name_bytes.saturating_sub(alias.name.len()));
                    decls
                        .type_aliases
                        .insert(&alias.name, super::CheckedTypeAlias { ty, copy_work });
                    continue;
                }
                Definition::Struct(index) => {
                    let def = &module.structs[index];
                    if def.is_parametric() {
                        let mut checker =
                            BodyChecker::new(decls, resolver, structs, BodyKind::Function);
                        checker
                            .check_struct_template_names(def)
                            .map_err(|e| format!("struct '{}': {e}", def.name))?;
                        continue;
                    }
                    let fields = super::resolve_fields(
                        &def.fields,
                        decls.module_dimensions(&def.name),
                        &mut TypeResolver {
                            resolver,
                            structs,
                            functions: Some(self),
                        },
                    )
                    .map_err(|e| format!("struct '{}': {e}", def.name))?;
                    structs.definitions[index].fields = fields;
                    structs.update_imports(resolver)?;
                    structs.index(index)?;
                    continue;
                }
                Definition::Extern(f) => {
                    decls
                        .add_function(
                            &f.name,
                            &f.params,
                            &f.result,
                            None,
                            &mut TypeResolver {
                                resolver,
                                structs,
                                functions: Some(self),
                            },
                        )
                        .map_err(|e| format!("extern '{}': {e}", f.name))?;
                    continue;
                }
                Definition::Function(f) => {
                    let at = decls.positions.get(f.name.as_str()).copied();
                    // Inferred templates check their symbolic signature and
                    // body together below, with type formals in scope.
                    if !f.has_inferred_parametrics() {
                        for ty in f.params.iter().map(|p| &p.ty).chain([&f.result]) {
                            decls.check_type_visibility(ty, at, None)?;
                        }
                    }
                    decls
                        .add_function(
                            &f.name,
                            &f.params,
                            &f.result,
                            Some(f),
                            &mut TypeResolver {
                                resolver,
                                structs,
                                functions: Some(self),
                            },
                        )
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
                Definition::ConstAssert(label, assertion) => {
                    let mut checker =
                        BodyChecker::new(decls, resolver, structs, BodyKind::Function);
                    checker.function_registry = Some(self);
                    checker.function_position = decls.positions.get(label).copied();
                    checker.warning_owner = Some(label);
                    checker.zero_binding = decls.zero_binding(label);
                    checker.enable_constant_functions();
                    checker.check_const_assert(assertion).map_err(|error| {
                        if let Some((subject, _)) = crate::bx_import::library_item(label) {
                            format!("BX import '{subject}': {error}")
                        } else {
                            error
                        }
                    })?;
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
            if function.has_inferred_parametrics() {
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

    pub(super) fn generated_count(&self) -> usize {
        self.generated.len()
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
            let types = match &value_signature {
                Some(bound) => bound.type_bindings.arguments(&template.type_params)?,
                None => Vec::new(),
            };
            let type_work: usize = types.iter().map(|ty| ty.source_nodes).sum();
            let work = values
                .iter()
                .fold(type_work + targets.len() + values.len(), |sum, v| {
                    sum.saturating_add(constant_work(&v.value))
                });
            registry.charge_bindings(work)?;
            let key = FunctionKey {
                template: &template.name,
                types,
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
    fn repeated_dependent_formal_annotations_share_their_inference() -> Result<(), String> {
        let mut source =
            String::from("fn id(x:u8)->u8{x} fn f0<T,F:fn(u8)->u8,V:u8>(x:T)->u32{u32:8}\n");
        for level in 1..20 {
            let previous = level - 1;
            // Each annotation asks for the same earlier specialization twice.
            // Both the callback input and the explicit value are eight bits.
            let call = format!("f{previous}<id,1>(u1:0)");
            let width = format!("{call}+{call}-u32:8");
            writeln!(
                source,
                "fn f{level}<T,F:fn(uN[{width}])->u8,V:uN[{width}]>(x:T)->u32{{u32:8}}"
            )
            .unwrap();
        }
        source.push_str("const ANSWER = f19<id,1>(u1:0);");
        let checked = check_module(&parse_module(&source)?)?;
        assert_eq!(checked.functions().len(), 21);
        assert_eq!(
            checked.constants()[0].value,
            IrValue::make_ubits(32, 8).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn dependent_formals_cache_the_preceding_values_and_callback_sources() -> Result<(), String> {
        let source = r#"
fn as_byte(x:u8)->u8 { x }
fn as_word(x:u8)->u16 { x as u16 }
fn apply<T,R,N:u32,F:fn(uN[N])->R,V:R>(x:T)->R { V }
const ANSWER = (apply<8,as_byte,7>(u8:1), apply<8,as_word,511>(u8:2), apply<8,as_word,511>(u8:3));
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(checked.functions().len(), 4);
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[8]:7,bits[16]:511,bits[16]:511)")
                .map_err(|e| e.to_string())?
        );
        let incompatible = source.replace(
            "apply<8,as_byte,7>(u8:1)",
            "apply<8,as_byte,7>(u8:1), apply<16,as_byte,7>(u8:1)",
        );
        assert!(check_module(&parse_module(&incompatible)?).is_err());
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
    fn type_specializations_charge_types_materialized_inside_the_body() -> Result<(), String> {
        let declarations = wide_struct_declarations();
        let source = format!(
            r#"
{declarations}
fn observe<T>(x: T, r: S15) -> T {{
    let a = r;
    let b = r;
    let c = r;
    let d = r;
    x
}}
block sample(clk, r: S15) -> u1 {{
    let a = observe(u8:0, r);
    let b = observe(u8:1, r);
    u1:0
}}
"#
        );
        check_module(&parse_module(&source)?)?;
        let distinct = source.replace("observe(u8:1", "observe(u16:1");
        let error = check_module(&parse_module(&distinct)?).unwrap_err();
        assert!(
            error.contains("function specialization expansion limit"),
            "{error}"
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
