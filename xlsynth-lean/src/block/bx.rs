// SPDX-License-Identifier: Apache-2.0

//! Keep BX declarations paired with the package produced by lowering them.

use anyhow::{Context, Result, bail, ensure};
use std::{collections::BTreeMap, rc::Rc};
use xlsynth_bx::{
    ImportOptions, LinkedStruct, ResetBehavior,
    ast::{Module, Stmt, Type},
    check::output_ports,
};
use xlsynth_pir::ir;

use super::source_types::{
    SourceType, StatefulInterface, StructField, TypeDecl, resolve_aliases, source_shape,
};
use super::{BlockContext, LeanBlockBundle, readable, select_block};

/// A checked BX module and its linked block IR, from the same compilation.
pub struct CompiledBx {
    warnings: Vec<xlsynth_bx::BxWarning>,
    imported_structs: BTreeMap<String, Rc<LinkedStruct>>,
    source: Module,
    pub(super) package: ir::Package,
    dslx: Option<super::helper_types::DslxLink>,
    imports: Vec<super::helper_types::DslxLink>,
}

/// Check BX source with its import paths and lower it into the supplied
/// package. The package may already contain functions or blocks used by BX
/// externs.
pub fn compile_bx(
    source: &str,
    package: ir::Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let xlsynth_bx::CompiledModule {
        checked,
        source,
        package,
        imports,
        imported_structs,
        ..
    } = xlsynth_bx::compile_with_source(source, package, reset, options)
        .map_err(anyhow::Error::msg)?;
    Ok(CompiledBx {
        warnings: checked.warnings().to_vec(),
        imported_structs,
        source,
        package,
        dslx: None,
        imports: imports
            .into_iter()
            .map(super::helper_types::DslxLink::from_import)
            .collect(),
    })
}

/// Link the IR produced from this checked DSLX module and retain its source
/// types for calls from the BX block.
pub fn compile_bx_with_dslx(
    source: &str,
    module: xlsynth::dslx::TypecheckedModule,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let (package, link) = super::helper_types::DslxLink::compile(module)?;
    let compiled = compile_bx(source, package, reset, options)?;
    Ok(CompiledBx {
        dslx: Some(link),
        ..compiled
    })
}

impl CompiledBx {
    /// BX records each imported struct under its written type, including all
    /// arguments of a parametric application.
    fn imported_struct(&self, ty: &Type) -> Option<&LinkedStruct> {
        match ty {
            Type::Named(name) => self.imported_structs.get(name),
            Type::Applied { .. } => self.imported_structs.get(&ty.to_string()),
            _ => None,
        }
        .map(Rc::as_ref)
    }

    /// BX diagnostics collected while checking this compilation.
    pub fn warnings(&self) -> &[xlsynth_bx::BxWarning] {
        &self.warnings
    }

    /// The exact lowered package used for both raw and readable equations.
    pub fn package(&self) -> &ir::Package {
        &self.package
    }

    /// Emit the selected BX block, defaulting to the package's top or sole
    /// block.
    pub fn emit_readable(&self, block_name: Option<&str>) -> Result<LeanBlockBundle> {
        let graph = select_block(&self.package, block_name)?;
        let block = BlockContext::new(&self.package, graph)?;
        let (mut declarations, mut interface) = self.readable_interface(&block)?;
        for link in self.dslx.iter().chain(&self.imports) {
            link.add_signatures(&block, &mut declarations, &mut interface)?;
        }
        // Two specializations (or two linked copies of the same definition)
        // share a DSLX name but need separate Lean definitions.
        let mut names = std::collections::BTreeSet::new();
        for (symbol, signature) in &mut interface.helpers {
            if !names.insert(signature.name.clone()) {
                signature.name = format!("{}@{symbol}", signature.name);
            }
        }
        readable::emit(&block, &declarations, &interface)
    }

    pub(super) fn readable_interface(
        &self,
        block: &BlockContext<'_>,
    ) -> Result<(Vec<TypeDecl>, StatefulInterface)> {
        let graph = &block.graph;
        let source = self
            .source
            .blocks
            .iter()
            .find(|b| b.name == graph.name)
            .with_context(|| format!("block {} has no checked BX declaration", graph.name))?;
        ensure!(
            Some(source.clock.as_str()) == graph.clock_port_name(),
            "BX clock differs from block {}",
            graph.name
        );
        let reset = graph.reset.as_ref().map(|r| graph.port_name(r.port));
        ensure!(
            source
                .reset
                .as_deref()
                .is_none_or(|declared| Some(declared) == reset),
            "BX reset differs from block {}",
            graph.name
        );

        let mut declarations = Vec::new();
        let check = |kind: &str,
                     expected: Vec<(String, SourceType)>,
                     physical: Vec<(&str, &ir::Type)>,
                     declarations: &[TypeDecl]|
         -> Result<Vec<(String, SourceType)>> {
            ensure!(
                expected.len() == physical.len(),
                "BX {kind} count differs from block {}",
                graph.name
            );
            expected
                .into_iter()
                .zip(physical)
                .map(|((name, ty), (actual, shape))| {
                    ensure!(name == actual, "expected BX {kind} {name}, found {actual}");
                    ensure!(
                        &source_shape(&ty, declarations)? == shape,
                        "BX {kind} {name} has type {ty:?} but block has {shape}"
                    );
                    Ok((name, ty))
                })
                .collect()
        };
        let ports = |port| (graph.port_name(port), graph.port_type(port));
        let read_type = |kind: &str, (name, ty): (&str, &Type), decls: &mut Vec<TypeDecl>| {
            Ok((
                name.to_owned(),
                bx_type(ty, self, decls).with_context(|| format!("BX {kind} {name}"))?,
            ))
        };
        let inputs = check(
            "input",
            reset
                .into_iter()
                .map(|name| (name, &Type::Bool))
                .chain(source.params.iter().map(|p| (p.name.as_str(), &p.ty)))
                .map(|p| read_type("input", p, &mut declarations))
                .collect::<Result<_>>()?,
            graph.input_ports().map(ports).collect(),
            &declarations,
        )?
        .into_iter()
        .collect();
        let registers = check(
            "register",
            source
                .statements
                .iter()
                .filter_map(|s| match s {
                    Stmt::Reg { name, ty, .. } => Some((name.as_str(), ty)),
                    _ => None,
                })
                .map(|p| read_type("register", p, &mut declarations))
                .collect::<Result<_>>()?,
            graph
                .registers
                .iter()
                .map(|r| (r.name.as_str(), &r.ty))
                .collect(),
            &declarations,
        )?
        .into_iter()
        .collect();
        let outputs = if let Some(import) = self.imported_struct(&source.result) {
            let ty = super::helper_types::imported_struct(import, &mut declarations)?;
            let SourceType::Struct { name } = resolve_aliases(&ty, &declarations)? else {
                bail!("expected imported BX result struct {}", source.result);
            };
            let fields = declarations
                .iter()
                .find_map(|d| match d {
                    TypeDecl::Struct { name: n, fields } if *n == name => Some(fields),
                    _ => None,
                })
                .with_context(|| format!("missing imported BX result struct {name}"))?;
            fields
                .iter()
                .map(|f| (f.name.clone(), f.source_type.clone()))
                .collect()
        } else if let Type::Named(key) = &source.result
            && self.source.enums.iter().any(|decl| decl.name == *key)
        {
            vec![read_type(
                "output",
                ("out", &source.result),
                &mut declarations,
            )?]
        } else {
            output_ports(&source.result, &self.source.structs)
                .map_err(anyhow::Error::msg)?
                .iter()
                .map(|p| read_type("output", (&p.name, &p.ty), &mut declarations))
                .collect::<Result<_>>()?
        };
        let outputs = check(
            "output",
            outputs,
            graph.output_ports().map(ports).collect(),
            &declarations,
        )?
        .into_iter()
        .map(|(_, ty)| ty)
        .collect();
        Ok((
            declarations,
            StatefulInterface {
                inputs,
                registers,
                outputs,
                helpers: Default::default(),
            },
        ))
    }
}

/// Collect only interface-reachable types. Lean requires a nested struct to
/// precede its user even when the BX declarations appear in the reverse order.
fn bx_type(ty: &Type, module: &CompiledBx, declarations: &mut Vec<TypeDecl>) -> Result<SourceType> {
    if let Some(import) = module.imported_struct(ty) {
        return super::helper_types::imported_struct(import, declarations);
    }
    Ok(match ty {
        Type::Bool
        | Type::Bits {
            signed: false,
            width: 1,
        } => SourceType::Bool,
        Type::Bits {
            signed: false,
            width,
        } => SourceType::Bits { width: *width },
        Type::Bits { signed: true, .. } => {
            bail!("signed BX type {ty} is unsupported by readable Lean")
        }
        Type::Enum { .. } => bail!("BX numeric enum {ty} is unsupported by readable Lean"),
        Type::Applied { .. } => bail!("BX type application {ty} is unsupported by readable Lean"),
        Type::Tuple(items) => SourceType::Tuple {
            members: items
                .iter()
                .map(|t| bx_type(t, module, declarations))
                .collect::<Result<_>>()?,
        },
        Type::ParametricBits { .. }
        | Type::ParametricArray { .. }
        | Type::ParametricStruct { .. }
        | Type::TypeApplied { .. } => bail!("BX type {ty} is unsupported by readable Lean"),
        Type::Array { element, size } => SourceType::Array {
            element: Box::new(bx_type(element, module, declarations)?),
            size: *size,
        },
        Type::Named(key) => {
            if module.source.enums.iter().any(|decl| decl.name == *key) {
                bail!("BX numeric enum {ty} is unsupported by readable Lean");
            }
            if !declarations.iter().any(|d| d.name() == key) {
                let def = module
                    .source
                    .structs
                    .iter()
                    .find(|s| s.name == *key)
                    .with_context(|| format!("unknown BX struct {key}"))?;
                let fields = def
                    .fields
                    .iter()
                    .map(|field| {
                        Ok(StructField {
                            name: field.name.clone(),
                            source_type: bx_type(&field.ty, module, declarations)
                                .with_context(|| format!("field {key}.{}", field.name))?,
                        })
                    })
                    .collect::<Result<_>>()?;
                declarations.push(TypeDecl::Struct {
                    name: key.clone(),
                    fields,
                });
            }
            SourceType::Struct { name: key.clone() }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_width_output_reordering_is_rejected() {
        let mut compiled = compile_bx(
            "struct Ports { zebra: u8, apple: u8 } block pair(clk, x: u8) -> Ports { Ports { apple: u8:42, zebra: x } }",
            xlsynth_bx::empty_package("example"),
            Default::default(),
            &Default::default(),
        ).unwrap();
        // Mutate only the physical declaration order. Both port shapes still
        // match, and both output equations still have their original names.
        let graph = compiled.package.get_block_mut("pair").unwrap();
        let end = graph.ports.len();
        graph.ports.swap(end - 2, end - 1);
        let error = compiled.emit_readable(None).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("expected BX output zebra, found apple"),
            "{error:#}"
        );
    }
}
