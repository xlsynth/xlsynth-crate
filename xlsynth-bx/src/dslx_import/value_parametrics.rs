// SPDX-License-Identifier: Apache-2.0

//! Carry constexpr BX values and contextual data arguments into DSLX invocation
//! probes.

use super::{ImportResolver, ImportedStructs, ProbeImports, Type, TypeWitness};
use crate::import_structs::probe_function;
use crate::parser::parse_value_type;
use crate::typed::BoundParametricValue;
use xlsynth_pir::{IrFormatPreference, IrValue};

/// A data argument's type evidence for DSLX contextual literal inference.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum NativeDataArg {
    Typed(Type),
    Inferred(String),
    BoundLiteral(String),
    Tuple(Vec<Self>),
    Array {
        members: Vec<Self>,
        has_ellipsis: bool,
    },
    Branches(Vec<Self>),
}

impl NativeDataArg {
    /// Account for a retained or replayed data-argument witness.
    pub(crate) fn storage_work(&self) -> usize {
        match self {
            Self::Typed(ty) => crate::check::source_type_nodes(ty),
            Self::Inferred(literal) | Self::BoundLiteral(literal) => literal.len().max(1),
            Self::Array { members: items, .. } | Self::Tuple(items) | Self::Branches(items) => {
                items
                    .iter()
                    .fold(1usize, |sum, item| sum.saturating_add(item.storage_work()))
            }
        }
    }
}

/// The contextual evidence used to infer each operand of one specialized native
/// call.
#[derive(Eq, Hash, PartialEq)]
pub(super) struct ContextInputs {
    subject: String,
    function: String,
    parametrics: Vec<BoundParametricValue>,
    args: Vec<NativeDataArg>,
}

impl ProbeImports {
    /// Render a value using its source type, retaining signedness and nominal
    /// declarations.
    fn render_value(
        &mut self,
        ty: &Type,
        value: &IrValue,
        structs: &ImportedStructs,
        aggregate_value: bool,
    ) -> Result<String, String> {
        let aggregate_value = aggregate_value || structs.aggregate_spelling(ty).is_some();
        match (ty, value) {
            (Type::Bits { signed, width }, IrValue::Bits(bits))
                if *width == bits.get_bit_count() =>
            {
                let format = if *signed {
                    IrFormatPreference::SignedDecimal
                } else {
                    IrFormatPreference::Hex
                };
                Ok(format!("{ty}:{}", bits.to_string_fmt(format, false)))
            }
            (Type::Bool, _) => Ok(value.to_bool().map_err(|e| e.to_string())?.to_string()),
            (Type::Enum { signed, width, .. }, IrValue::Bits(_)) => {
                let backing = Type::Bits {
                    signed: *signed,
                    width: *width,
                };
                Ok(format!(
                    "({} as {})",
                    self.render_value(&backing, value, structs, aggregate_value)?,
                    self.render_probe_type(ty, structs, aggregate_value)?
                ))
            }
            (Type::Named(name), IrValue::Tuple(values)) => {
                let def = structs
                    .definition(name)
                    .filter(|s| s.fields.len() == values.len())
                    .ok_or_else(|| format!("unknown DSLX struct layout '{name}'"))?;
                let fields = def
                    .fields
                    .iter()
                    .zip(values.iter())
                    .map(|(field, value)| {
                        Ok(format!(
                            "{}: {}",
                            field.name,
                            self.render_value(&field.ty, value, structs, aggregate_value)?
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(format!(
                    "{} {{ {} }}",
                    self.render_probe_type(ty, structs, aggregate_value)?,
                    fields.join(", ")
                ))
            }
            (Type::Array { element, size }, IrValue::Array(array))
                if *size == array.elements().len() =>
            {
                let items = array
                    .elements()
                    .iter()
                    .map(|v| self.render_value(element, v, structs, aggregate_value))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(format!(
                    "{}:[{}]",
                    self.render_probe_type(ty, structs, aggregate_value)?,
                    items.join(", ")
                ))
            }
            (Type::Tuple(types), IrValue::Tuple(values)) if types.len() == values.len() => {
                let fields = types
                    .iter()
                    .zip(values.iter())
                    .map(|(t, v)| self.render_value(t, v, structs, aggregate_value))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(tuple(&fields))
            }
            _ => Err(format!(
                "value '{value}' cannot be rendered as DSLX type '{ty}'"
            )),
        }
    }

    /// Reconstruct explicit compile-time operands in every native type and IR
    /// probe.
    pub(super) fn specialized_name(
        &mut self,
        subject: &str,
        function: &str,
        parametrics: &[BoundParametricValue],
        structs: &ImportedStructs,
    ) -> Result<String, String> {
        let mut name = self.qualify(subject, function);
        if !parametrics.is_empty() {
            let values = parametrics
                .iter()
                .map(|p| {
                    self.render_value(&p.ty, &p.value, structs, false)
                        .map(|v| format!("{{{v}}}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            name.push_str(&format!("<{}>", values.join(", ")));
        }
        Ok(name)
    }

    /// Give every typed leaf a probe parameter while preserving literal
    /// constraints.
    fn render_argument(
        &mut self,
        arg: &NativeDataArg,
        structs: &ImportedStructs,
        params: &mut Vec<String>,
        lets: &mut Vec<String>,
        in_aggregate: bool,
    ) -> Result<String, String> {
        match arg {
            NativeDataArg::Typed(ty) => {
                let name = format!("arg_{}", params.len());
                if in_aggregate
                    && structs.aggregate_spelling(ty).is_none()
                    && let Some((alias, suffix)) = structs.aggregate_projection(ty)
                {
                    let (subject, member) = alias.split_once("::").expect("qualified public alias");
                    params.push(format!("{name}: {}", self.qualify(subject, member)));
                    return Ok(format!("{name}{suffix}"));
                }
                // A private field can arrive from a public record before the
                // aggregate alias in the callee's formal has been reflected.
                params.push(format!(
                    "{name}: {}",
                    self.render_probe_type(ty, structs, in_aggregate)?
                ));
                Ok(name)
            }
            NativeDataArg::Inferred(literal) => Ok(literal.clone()),
            NativeDataArg::BoundLiteral(literal) => {
                // DSLX infers a let initializer before applying the concrete
                // callee's constraints at a use of its name.
                let name = format!("bound_{}", lets.len());
                lets.push(format!("let {name} = {literal};"));
                Ok(name)
            }
            NativeDataArg::Tuple(items)
            | NativeDataArg::Array { members: items, .. }
            | NativeDataArg::Branches(items) => {
                let in_aggregate = in_aggregate || !matches!(arg, NativeDataArg::Branches(_));
                let mut values = items
                    .iter()
                    .map(|a| self.render_argument(a, structs, params, lets, in_aggregate))
                    .collect::<Result<Vec<_>, _>>()?;
                if matches!(
                    arg,
                    NativeDataArg::Array {
                        has_ellipsis: true,
                        ..
                    }
                ) {
                    values.push("...".to_owned());
                }
                Ok(match arg {
                    NativeDataArg::Tuple(_) => tuple(&values),
                    NativeDataArg::Branches(_) => format!("[{}][u32:0]", values.join(", ")),
                    _ => format!("[{}]", values.join(", ")),
                })
            }
        }
    }
}

fn tuple(items: &[String]) -> String {
    let trailing = if items.len() == 1 { "," } else { "" };
    format!("({}{trailing})", items.join(", "))
}

impl ImportResolver {
    /// Resolve the explicitly supplied prefix of value annotations in their
    /// declaring scope.
    pub(crate) fn explicit_parametric_types(
        &mut self,
        module: &str,
        function: &str,
        count: usize,
    ) -> Result<Vec<Type>, String> {
        let key = (module.to_owned(), function.to_owned());
        let previous = self.binding_types.get(&key).map_or(0, Vec::len);
        if count > previous {
            let types = self.with_imports(|structs, data| {
                structs.binding_types(module, function, previous..count, data)
            })?;
            self.binding_types
                .entry(key.clone())
                .or_default()
                .extend(types);
        }
        Ok(self
            .binding_types
            .get(&key)
            .map_or(&[][..], |v| &v[..count])
            .to_vec())
    }

    /// Typecheck synthetic operands under the concrete native call's
    /// constraints.
    pub(crate) fn contextual_data_types(
        &mut self,
        subject: &str,
        function: &str,
        parametrics: &[BoundParametricValue],
        args: &[NativeDataArg],
    ) -> Result<Vec<Type>, String> {
        let key = ContextInputs {
            subject: subject.to_owned(),
            function: function.to_owned(),
            parametrics: parametrics.to_vec(),
            args: args.to_vec(),
        };
        if let Some(types) = self.context_types.get(&key) {
            return Ok(types.clone());
        }
        let work = args
            .iter()
            .fold(0usize, |sum, arg| sum.saturating_add(arg.storage_work()));
        self.reserve_explicit_probe(self.context_types.len(), parametrics, work, 1)?;
        let mut imports = ProbeImports::default();
        let target = imports.specialized_name(subject, function, parametrics, &self.structs)?;
        let mut params = Vec::new();
        let mut lets = Vec::new();
        let values = args
            .iter()
            .map(|arg| imports.render_argument(arg, &self.structs, &mut params, &mut lets, false))
            .collect::<Result<Vec<_>, _>>()?;
        let target_name = imports.qualify(subject, function);
        let specialization = target
            .strip_prefix(&target_name)
            .expect("specialized name starts with its callee");
        let declarations = imports.declarations();
        let params = params.join(", ");
        let lets = lets.join("\n");
        let values = values.join(", ");
        let source =
            format!("{declarations}fn probe({params}) {{ {lets} let _ = {target}({values}); }}");
        let module = format!("0bx_context_{}", self.context_types.len());
        let types: Vec<Type> = self.with_imports(|structs, data| {
            let checked = structs.check_probe(&source, &module, data)?;
            let root = checked.get_type_info();
            let owner = checked.get_module();
            let probe = probe_function(&checked)?;
            let graph = root
                .build_function_call_graph(&owner)
                .map_err(|e| e.to_string())?;
            let callee = (0..graph.callee_count(&probe))
                .filter_map(|i| graph.get_callee(&probe, i))
                .find(|f| f.get_identifier() == function)
                .ok_or("missing native invocation target")?;
            let sites = root
                .get_all_invocation_callee_data(&callee)
                .ok_or("missing native call sites")?;
            let info = (0..sites.len())
                .rev()
                .filter_map(|i| sites.get(i))
                .find_map(|site| {
                    let caller = root
                        .get_root_invocation_data(&site.invocation()?)?
                        .caller()?;
                    (caller == probe)
                        .then(|| site.derived_type_info())
                        .flatten()
                })
                .ok_or("missing type information for specialized native call")?;
            args.iter()
                .enumerate()
                .map(|(i, arg)| {
                    let annotation = callee.get_param(i).get_type_annotation();
                    let ty = info
                        .get_type_for_type_annotation(&annotation)
                        .ok_or("missing native operand type")?;
                    // Use the concrete formal for numeric dimensions and the
                    // checked caller's identity for typed nominal leaves.
                    let display = ty.to_string().map_err(|e| e.to_string())?;
                    // A formal can include several aggregate aliases. Bind
                    // matching candidates before rendering the final signature.
                    let mut aggregates = Vec::new();
                    let shape = contextual_shape(arg, &display, &mut aggregates);
                    for aggregate in aggregates {
                        structs.bind_aggregate_display(aggregate, data)?;
                    }
                    match shape {
                        Some(ty) => Ok(ty),
                        None => structs.contextual_argument_type(
                            subject,
                            function,
                            i,
                            |name| {
                                let identity = imports.qualify(subject, name);
                                TypeWitness {
                                    imports: declarations.clone(),
                                    params: params.clone(),
                                    expr: format!(
                                        "{{ {lets} {identity}{specialization}({values}) }}"
                                    ),
                                }
                            },
                            data,
                        ),
                    }
                })
                .collect()
        })?;
        self.context_types.insert(key, types.clone());
        Ok(types)
    }
}

/// Combine DSLX's concrete dimensions with known BX source types. Only the
/// delimiters of a displayed nominal type are inspected, never its identifier.
fn contextual_shape<'d>(
    arg: &NativeDataArg,
    display: &'d str,
    aggregates: &mut Vec<&'d str>,
) -> Option<Type> {
    let display = display.trim();
    let ty = match arg {
        NativeDataArg::Typed(ty) => Some(ty.clone()),
        NativeDataArg::Branches(arms) => {
            return arms
                .iter()
                .find_map(|arm| contextual_shape(arm, display, aggregates));
        }
        NativeDataArg::Inferred(_) | NativeDataArg::BoundLiteral(_) => {
            parse_value_type(display).ok()
        }
        NativeDataArg::Tuple(fields) => {
            let text = display.strip_prefix('(')?.strip_suffix(')')?;
            let mut depth = 0;
            let members = text
                .split(|c: char| match c {
                    '(' | '{' | '[' | '<' => {
                        depth += 1;
                        false
                    }
                    ')' | '}' | ']' | '>' => {
                        depth -= 1;
                        false
                    }
                    ',' => depth == 0,
                    _ => false,
                })
                .filter(|s| !s.trim().is_empty());
            let mut types = Vec::with_capacity(fields.len());
            for (i, member) in members.enumerate() {
                types.push(contextual_shape(fields.get(i)?, member, aggregates)?);
            }
            (types.len() == fields.len()).then_some(Type::Tuple(types))
        }
        NativeDataArg::Array { members: items, .. } => {
            let (element, size) = display.strip_suffix(']')?.rsplit_once('[')?;
            let size = size.parse().ok()?;
            let element = items
                .iter()
                .find_map(|a| contextual_shape(a, element, aggregates))
                .or_else(|| parse_value_type(element).ok())?;
            Some(Type::Array {
                element: Box::new(element),
                size,
            })
        }
    }?;
    if matches!(ty, Type::Tuple(_) | Type::Array { .. }) {
        aggregates.push(display);
    }
    Some(ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImportOptions, parse_module};
    use std::{error::Error, fs};
    use tempfile::tempdir;

    #[test]
    fn empty_array_operands_retain_nominal_elements_under_a_private_tuple_alias()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            r#"
pub enum Mode:u3 { A=0 }
pub struct Config<N:u32> { x:uN[N] }
type Operand = (u8, Config<8>[0], Mode[0]);
pub fn f<N:u32>(_:Operand)->u32 { N }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let mut resolver = ImportResolver::new(&parse_module("use wire;")?.uses, &options)?;
        let values = [BoundParametricValue {
            ty: Type::Bits {
                signed: false,
                width: 32,
            },
            value: IrValue::make_ubits(32, 8)?,
        }];
        let operand = NativeDataArg::Tuple(vec![
            NativeDataArg::Inferred("3".into()),
            NativeDataArg::Array {
                members: vec![],
                has_ellipsis: false,
            },
            NativeDataArg::Array {
                members: vec![],
                has_ellipsis: false,
            },
        ]);
        let types = resolver.contextual_data_types("wire", "f", &values, &[operand])?;
        let expected = Type::Tuple(vec![
            Type::Bits {
                signed: false,
                width: 8,
            },
            Type::Array {
                element: Box::new(resolver.resolve_applied("wire", "Config", &[8])?),
                size: 0,
            },
            Type::Array {
                element: Box::new(resolver.resolve_named("wire", "Mode")?),
                size: 0,
            },
        ]);
        assert_eq!(types, [expected]);
        Ok(())
    }

    #[test]
    fn cached_explicit_probes_remain_usable_at_the_specialization_limit()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            "pub fn tag<N:u32>() -> u32 { N }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let mut resolver = ImportResolver::new(&parse_module("use wire;")?.uses, &options)?;
        let mut values = vec![BoundParametricValue {
            ty: Type::Bits {
                signed: false,
                width: 32,
            },
            value: IrValue::make_ubits(32, 1)?,
        }];
        resolver.explicit_count = crate::check::MAX_SPECIALIZATIONS - 1;
        let first = resolver.resolve_explicit("wire", "tag", vec![], values.clone())?;
        assert_eq!(
            resolver.resolve_explicit("wire", "tag", vec![], values.clone())?,
            first
        );
        values[0].value = IrValue::make_ubits(32, 2)?;
        assert!(
            resolver
                .resolve_explicit("wire", "tag", vec![], values)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn unsized_operand_cannot_infer_a_missing_native_dimension() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            "pub fn f<K:u32,N:u32>(v:uN[N])->u32 { N }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let mut resolver = ImportResolver::new(&parse_module("use wire;")?.uses, &options)?;
        let parametrics = [BoundParametricValue {
            ty: Type::Bits {
                signed: false,
                width: 32,
            },
            value: IrValue::make_ubits(32, 2)?,
        }];
        assert!(
            resolver
                .contextual_data_types(
                    "wire",
                    "f",
                    &parametrics,
                    &[NativeDataArg::Inferred("3".into())]
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn native_invocation_exposes_contextual_widths_and_constexpr_specializations()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            r#"
const W = u32:80;
type Word = sN[W];
pub enum Mode: u2 { A = 0, B = 1 }
pub struct Config<N: u32> { x: uN[N], modes: Mode[2] }
type Operand = (u8, Config<8>);
pub fn inspect<N: u32, K: Word, C: Config<8>, M: Mode>(x: Operand) -> (uN[N], Word, u8, Mode) { (x.0 as uN[N], K, C.x, M) }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let mut resolver = ImportResolver::new(&parse_module("use wire;")?.uses, &options)?;
        assert!(resolver.has_imports());
        let bindings = resolver.explicit_parametric_types("wire", "inspect", 4)?;
        let values = [
            "bits[32]:8",
            "bits[80]:-7",
            "(bits[8]:11, [bits[2]:0, bits[2]:1])",
            "bits[2]:1",
        ]
        .iter()
        .map(|s| IrValue::parse_typed(s))
        .collect::<Result<Vec<_>, _>>()?;
        let mut actuals = bindings
            .iter()
            .cloned()
            .zip(values)
            .map(|(ty, value)| BoundParametricValue { ty, value })
            .collect::<Vec<_>>();
        let args = [NativeDataArg::Tuple(vec![
            NativeDataArg::Inferred("3".into()),
            NativeDataArg::Typed(bindings[2].clone()),
        ])];
        let types = resolver.contextual_data_types("wire", "inspect", &actuals, &args)?;
        let expected = vec![Type::Tuple(vec![
            Type::Bits {
                signed: false,
                width: 8,
            },
            bindings[2].clone(),
        ])];
        assert_eq!(types, expected);
        let (first, _) =
            resolver.resolve_explicit("wire", "inspect", types.clone(), actuals.clone())?;
        actuals[2].value = IrValue::parse_typed("(bits[8]:11, [bits[2]:1, bits[2]:0])")?;
        let (second, _) = resolver.resolve_explicit("wire", "inspect", types, actuals)?;
        assert_ne!(first, second);
        let imports = resolver.finish()?;
        assert_eq!(imports.batches.len(), 1);
        assert_ne!(imports.batches[0].roots[0], imports.batches[0].roots[1]);
        Ok(())
    }
}
