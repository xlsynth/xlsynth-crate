// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use xlsynth::dslx;
use xlsynth_pir::{ir, libxls_bridge::value_from_libxls};

use super::binding_types::witness_names;
use super::outline_graph::SourceModule;
use super::{DeclId, ImportedStructs, instance_spelling};
use crate::ast::Type;
use crate::check::MAX_TYPE_SIZE;
use crate::dslx_import::{ProbeCapture, TypeWitness};
use crate::parser::outline::{WrittenParametric, WrittenStructTemplate};
use crate::parser::{Token, dslx_tokens};
use crate::typed::{BoundParametricValue, bits};

/// Results of one formal annotation or default, with its native dependencies
/// fixed before any result is cached.
pub(super) struct FormalResults<K, V> {
    indices: Vec<usize>,
    results: HashMap<K, V>,
}

pub(super) type StructFormalTypes = FormalResults<Vec<BoundParametricValue>, Type>;
pub(super) type StructDefaultValues = FormalResults<DefaultInputs, BoundParametricValue>;

/// A default's free values and the type used to interpret inferred literals.
#[derive(Eq, Hash, PartialEq)]
pub(super) struct DefaultInputs {
    expected_type: Type,
    free_values: Vec<BoundParametricValue>,
}

/// Visit possible lexical references, omitting qualified members and fields.
fn unqualified_tokens<'s>(tokens: &[Token<'s>]) -> impl Iterator<Item = (&'s str, &'s str)> {
    let mut member_follows = false;
    tokens.windows(2).filter_map(move |pair| {
        let name = pair[0].text;
        let unqualified = !member_follows;
        member_follows = matches!(name, "::" | ".");
        unqualified.then_some((name, pair[1].text))
    })
}

impl<K, V> FormalResults<K, V> {
    /// Resolve ambiguous binder and label spellings once for the frozen
    /// declaration, independent of which preceding inputs are known first.
    fn new(
        syntax: &str,
        preceding: &[WrittenParametric],
        capture: impl FnOnce(&[usize]) -> Result<Vec<usize>, String>,
    ) -> Result<Self, String> {
        let tokens = dslx_tokens(syntax)?;
        let mut names = BTreeSet::new();
        let mut labels = BTreeSet::new();
        for (name, next) in unqualified_tokens(&tokens) {
            names.insert(name);
            if next == ":" {
                labels.insert(name);
            }
        }
        let mut indices: Vec<_> = preceding
            .iter()
            .enumerate()
            .filter_map(|(i, f)| names.contains(f.name.as_str()).then_some(i))
            .collect();
        // Local declarations, loops and patterns can bind a name. An
        // explicit field label can share a formal's spelling. Braced
        // parametric actuals with ordinary arithmetic need no native probe.
        let needs_capture = names.iter().any(|name| {
            matches!(
                *name,
                "let" | "const" | "type" | "for" | "unroll_for!" | "match" | "|"
            )
        }) || preceding
            .iter()
            .any(|formal| labels.contains(formal.name.as_str()));
        if needs_capture && !indices.is_empty() {
            indices = capture(&indices)?;
        }
        Ok(Self {
            indices,
            results: HashMap::new(),
        })
    }

    fn actuals<'a>(
        &self,
        previous: impl Fn(usize) -> Option<&'a BoundParametricValue>,
    ) -> Option<Vec<BoundParametricValue>> {
        self.indices.iter().map(|&i| previous(i).cloned()).collect()
    }
}

impl WrittenStructTemplate {
    /// Mark values needed by later formal annotations, following defaults
    /// only for values already demanded. Dependencies always precede their
    /// consumer, so one reverse walk covers indirect chains.
    pub(super) fn default_demand(&self) -> Result<Vec<bool>, String> {
        let indices: HashMap<_, _> = self
            .formals
            .iter()
            .enumerate()
            .map(|(i, formal)| (formal.name.as_str(), i))
            .collect();
        let mut needed = vec![false; self.formals.len()];
        for (i, formal) in self.formals.iter().enumerate().rev() {
            let default = formal.default_expr.as_deref().filter(|_| needed[i]);
            for syntax in std::iter::once(formal.ty.syntax.as_str()).chain(default) {
                let tokens = dslx_tokens(syntax)?;
                for (name, _) in unqualified_tokens(&tokens) {
                    if let Some(&prior) = indices.get(name)
                        && prior < i
                    {
                        needed[prior] = true;
                    }
                }
            }
        }
        Ok(needed)
    }

    /// Insert witnesses before later module bindings can shadow names used
    /// in the struct, preserving the original source bytes on either side.
    fn insert_witness(&self, owner: &SourceModule, declarations: &str) -> Vec<u8> {
        let (before, after) = owner.source.split_at(self.declaration_end);
        [before, declarations.as_bytes(), after].concat()
    }
}

/// Instantiate an expression in its module with just its known preceding
/// bindings. Native DSLX checks the return annotation and evaluates the body.
fn binding_witness(
    owner: &SourceModule,
    template: &WrittenStructTemplate,
    indices: &[usize],
    actuals: &[BoundParametricValue],
    result_type: &str,
    body: &str,
) -> Result<(Vec<u8>, TypeWitness), String> {
    let name = witness_names(&owner.source, "BXStructBinding")
        .next()
        .expect("finite source reserves finitely many names");
    let params: Vec<_> = indices
        .iter()
        .zip(actuals)
        .map(|(&i, actual)| format!("{}: {}", template.formals[i].name, actual.ty))
        .collect();
    let (params, invocation) = if params.is_empty() {
        (String::new(), name.clone())
    } else {
        (
            format!("<{}>", params.join(", ")),
            instance_spelling(&name, actuals)?,
        )
    };
    let declaration = format!("\npub fn {name}{params}() -> {result_type} {{ {body} }}\n");
    Ok((
        template.insert_witness(owner, &declaration),
        TypeWitness {
            imports: format!("import {} as bx_owner;", owner.subject),
            params: String::new(),
            expr: format!("bx_owner::{invocation}()"),
        },
    ))
}

impl ImportedStructs {
    /// Use native lambda capture to distinguish free formals from local names
    /// and field labels. A tagged control binding identifies the outer lambda
    /// even when the expression contains nested lambdas.
    fn captured_formals(
        &self,
        owner: &SourceModule,
        template: &WrittenStructTemplate,
        candidates: &[usize],
        expression: &str,
    ) -> Result<Vec<usize>, String> {
        let mut names = witness_names(&owner.source, "BXStructCapture");
        let [witness, control, control_type] = std::array::from_fn(|_| {
            names
                .next()
                .expect("finite source reserves finitely many names")
        });
        let mut declarations = format!("\ntype {control_type} = u32;\n");
        let mut bindings = vec![format!("{control}: {control_type}")];
        let tags: Vec<_> = candidates.iter().copied().zip(names).collect();
        for (index, tag) in &tags {
            declarations.push_str(&format!("type {tag} = u32;\n"));
            bindings.push(format!("{}: {tag}", template.formals[*index].name));
        }
        // Only lexical identity matters: the witness is never instantiated.
        // libxls copies each captured parent's type annotation onto the
        // closure struct, so tags identify bindings across native rewrites.
        let bindings = bindings.join(", ");
        declarations.push_str(&format!(
            "fn {witness}<{bindings}>() {{ let _ = map([u32:0], |_: u32| {{ let _ = {control}; let _ = {expression}; }}); }}\n"
        ));
        let bytes = template.insert_witness(owner, &declarations);
        let (registry, mut isolated) = self.annotation_snapshot(&owner.subject, bytes)?;
        if let Ok((module, _)) = registry.native_owner(&owner.subject, &mut isolated) {
            // Native closure rewriting inserts structs just before their
            // containing function. Start there even if later declarations
            // include unrelated closures and aliases.
            let members = (0..module.get_member_count()).rev()
                .filter_map(|i| module.get_member(i).to_matchable())
                .skip_while(|member| {
                    !matches!(member, dslx::MatchableModuleMember::Function(f) if f.get_identifier() == witness)
                });
            for member in members {
                match member {
                    dslx::MatchableModuleMember::StructDef(closure) => {
                        let captured: BTreeSet<_> = closure
                            .parametric_bindings()
                            .into_iter()
                            .filter_map(|binding| {
                                binding
                                    .get_type_annotation()?
                                    .to_type_ref_type_annotation()?
                                    .get_type_ref()
                                    .get_type_definition()
                                    .to_type_alias()
                                    .map(|alias| alias.get_identifier())
                            })
                            .collect();
                        if captured.contains(&control_type) {
                            return Ok(tags
                                .iter()
                                .filter_map(|(i, tag)| captured.contains(tag).then_some(*i))
                                .collect());
                        }
                    }
                    dslx::MatchableModuleMember::TypeAlias(_) => break,
                    _ => { /* The witness and generated impls contain no capture declarations. */ }
                }
            }
        }
        // If the native frontend cannot reflect this optional witness, retain
        // every candidate. The concrete type/default probe checks the original
        // expression once these bindings are supplied.
        Ok(candidates.to_vec())
    }

    /// Resolve the next formal in its declaring scope with previously bound
    /// arguments. Native unsigned one-bit types are DSLX booleans.
    pub(crate) fn struct_parametric_type(
        &mut self,
        subject: &str,
        member: &str,
        previous: &[BoundParametricValue],
    ) -> Result<Type, String> {
        self.struct_parametric_type_at(subject, member, previous.len(), |i| previous.get(i))
            .map(|ty| ty.expect("all preceding struct arguments are bound"))
    }

    /// Resolve a formal even in a partly bound application if every preceding
    /// value referenced by its annotation is known.
    pub(crate) fn struct_parametric_type_at<'a>(
        &mut self,
        subject: &str,
        member: &str,
        index: usize,
        previous: impl Fn(usize) -> Option<&'a BoundParametricValue>,
    ) -> Result<Option<Type>, String> {
        let graph = Rc::clone(&self.graph);
        let template = graph
            .struct_template(subject, member)
            .ok_or_else(|| format!("missing DSLX struct template '{subject}::{member}'"))?;
        let formals = &template.formals;
        if formals.len() > MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX struct has more than {MAX_TYPE_SIZE} parametric bindings"
            ));
        }
        let formal = formals
            .get(index)
            .ok_or_else(|| format!("'{subject}::{member}' has no formal at index {index}"))?;
        let syntax = formal.ty.syntax.as_str();
        match syntax {
            "bool" | "u1" => return Ok(Some(Type::Bool)),
            "u32" => {
                return Ok(Some(Type::Bits {
                    signed: false,
                    width: 32,
                }));
            }
            "type" => {
                return Err(format!(
                    "DSLX parametric '{}' of '{subject}::{member}' expects a type",
                    formal.name
                ));
            }
            _ => { /* Resolve aliases and computed types in the native declaration scope. */ }
        }

        let key = (
            DeclId {
                subject: subject.into(),
                name: member.into(),
            },
            index,
        );
        let owner = graph
            .module_reference(subject, &[])
            .expect("template declaring module");
        let cache = if let Some(cache) = self.struct_formal_types.get(&key) {
            cache
        } else {
            let cache = StructFormalTypes::new(syntax, &formals[..index], |indices| {
                self.captured_formals(owner, template, indices, &format!("zero!<{syntax}>()"))
            })?;
            self.struct_formal_types.entry(key.clone()).or_insert(cache)
        };
        let Some(actuals) = cache.actuals(previous) else {
            return Ok(None);
        };
        if let Some(ty) = cache.results.get(&actuals) {
            return Ok(Some(ty.clone()));
        }

        let (bytes, witness) = binding_witness(
            owner,
            template,
            &cache.indices,
            &actuals,
            syntax,
            &format!("zero!<{syntax}>()"),
        )?;
        let (registry, mut isolated) = self.annotation_snapshot(subject, bytes)?;
        let source = format!(
            r#"#![feature(generics)]
{}
fn capture<T: type>(_v: T) {{}}
fn probe() {{ capture({}) }}"#,
            witness.imports, witness.expr
        );
        let checked = registry
            .check_probe(&source, "0bx_struct_binding_type", &mut isolated)
            .map_err(|e| {
                format!(
                    "checking parametric '{}' of '{subject}::{member}': {}",
                    formal.name,
                    registry.remap_diagnostic(e)
                )
            })?;
        let native = ProbeCapture::new(&checked, |_| None)?.param_type(0)?;
        let ty = match (!native.is_enum()).then(|| native.is_bits_like()).flatten() {
            Some((false, 1)) => Type::Bool,
            Some((false, 32)) => Type::Bits {
                signed: false,
                width: 32,
            },
            _ => {
                return Err(format!(
                    "DSLX struct parametric '{}' of '{subject}::{member}' has unsupported type '{native}'; expected bool or u32",
                    formal.name
                ));
            }
        };
        self.struct_formal_types
            .get_mut(&key)
            .expect("registered formal")
            .results
            .insert(actuals, ty.clone());
        Ok(Some(ty))
    }

    /// Evaluate a default in its declaring scope once all free formals are
    /// known.
    pub(super) fn partial_struct_default(
        &mut self,
        id: &DeclId,
        owner: &SourceModule,
        template: &WrittenStructTemplate,
        default: &str,
        ty: Type,
        previous: &[Option<BoundParametricValue>],
    ) -> Result<Option<BoundParametricValue>, String> {
        let key = (id.clone(), previous.len());
        let cache = if let Some(cache) = self.struct_default_values.get(&key) {
            cache
        } else {
            let cache = StructDefaultValues::new(
                default,
                &template.formals[..previous.len()],
                |indices| self.captured_formals(owner, template, indices, default),
            )?;
            self.struct_default_values
                .entry(key.clone())
                .or_insert(cache)
        };
        let Some(actuals) = cache.actuals(|i| previous[i].as_ref()) else {
            return Ok(None);
        };
        let inputs = DefaultInputs {
            expected_type: ty,
            free_values: actuals,
        };
        if let Some(value) = cache.results.get(&inputs) {
            return Ok(Some(value.clone()));
        }
        let (bytes, witness) = binding_witness(
            owner,
            template,
            &cache.indices,
            &inputs.free_values,
            &inputs.expected_type.to_string(),
            default,
        )?;
        let (mut registry, mut isolated) = self.annotation_snapshot(&owner.subject, bytes)?;
        let value =
            registry.evaluate_constant_call(&inputs.expected_type, &witness, &mut isolated)?;
        let bound = BoundParametricValue {
            ty: inputs.expected_type.clone(),
            value,
        };
        self.struct_default_values
            .get_mut(&key)
            .expect("registered default")
            .results
            .insert(inputs, bound.clone());
        Ok(Some(bound))
    }

    /// Read actuals in declaration order from the checked struct instance.
    /// Its environment includes defaults and arguments unused by the fields.
    pub(super) fn read_struct_bindings(
        &mut self,
        id: &DeclId,
        def: &dslx::StructDef,
        native: &dslx::Type,
    ) -> Result<Vec<BoundParametricValue>, String> {
        let count = def.get_parametric_binding_count();
        if self.public_struct_template_arity(&id.subject, &id.name) != Some(count) {
            return Err(format!(
                "Rust and DSLX disagree on the number of parametrics for '{}::{}'",
                id.subject, id.name
            ));
        }
        if count > MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX struct has more than {MAX_TYPE_SIZE} parametric bindings"
            ));
        }
        let env = native
            .get_struct_parametric_env()
            .map_err(|e| e.to_string())?;
        let bindings: BTreeMap<_, _> = env.bindings().into_iter().collect();
        let mut actuals = Vec::with_capacity(count);
        for formal in def.parametric_bindings() {
            let ty = self.struct_parametric_type(&id.subject, &id.name, &actuals)?;
            let name = formal.get_identifier();
            let value = bindings
                .get(&name)
                .ok_or_else(|| format!("missing DSLX struct value binding '{name}'"))?
                .convert_to_ir()
                .map_err(|e| e.to_string())?;
            let (_, width) = bits(&ty).expect("validated unsigned struct formal");
            let value =
                value_from_libxls(&value, &ir::Type::Bits(width)).map_err(|e| e.to_string())?;
            actuals.push(BoundParametricValue { ty, value });
        }
        Ok(actuals)
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::outline::{ImportedTypeDeclarations, outline_import};

    #[test]
    fn default_demand_follows_formal_types_and_their_default_chains() -> Result<(), String> {
        let mut declarations = ImportedTypeDeclarations::default();
        outline_import(
            r#"
struct Settings { UNUSED: u32 }
const CFG = Settings { UNUSED: u32:1 };
pub struct Word<BASE: u32 = {u32:1}, UNUSED: u32 = {u32:2},
                WIDTH: u32 = {BASE + u32:1}, DATA: u32 = {UNUSED + u32:1},
                X: uN[WIDTH * CFG.UNUSED] = {uN[WIDTH]:0}> { data: uN[DATA] }
"#,
            &mut declarations,
        )?;
        // X's type requires WIDTH, whose default uses BASE. DATA is only
        // used in a field and its default must not make UNUSED necessary.
        assert_eq!(
            declarations.struct_parametrics["Word"].default_demand()?,
            [true, false, true, false, false]
        );
        Ok(())
    }
}
