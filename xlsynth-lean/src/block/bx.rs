// SPDX-License-Identifier: Apache-2.0

//! Keep BX declarations paired with the package produced by lowering them.

use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};
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
    module_name: String,
    imported_modules: BTreeSet<String>,
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
    module_name: &str,
    source: &str,
    package: ir::Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let compiled = xlsynth_bx::compile_with_source(source, package, reset, options)
        .map_err(anyhow::Error::msg)?;
    Ok(CompiledBx::from_bx(module_name, compiled, None))
}

/// Compile BX and retain checked loop bindings for the DSLX functions it
/// imports.
#[cfg(feature = "dslx-loop-bindings")]
pub fn compile_bx_with_loop_bindings(
    module_name: &str,
    source: &str,
    package: ir::Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let compiled =
        xlsynth_bx::compile_with_source_and_loop_bindings(source, package, reset, options)
            .map_err(anyhow::Error::msg)?;
    Ok(CompiledBx::from_bx(module_name, compiled, None))
}

/// Link the IR produced from this checked DSLX module and retain its source
/// types for calls from the BX block.
pub fn compile_bx_with_dslx(
    module_name: &str,
    source: &str,
    module: xlsynth::dslx::TypecheckedModule,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let (package, link) = super::helper_types::DslxLink::compile(module)?;
    let compiled = xlsynth_bx::compile_with_source(source, package, reset, options)
        .map_err(anyhow::Error::msg)?;
    Ok(CompiledBx::from_bx(module_name, compiled, Some(link)))
}

/// Link checked DSLX while recording loop binding names and types.
#[cfg(feature = "dslx-loop-bindings")]
pub fn compile_bx_with_dslx_loop_bindings(
    module_name: &str,
    source: &str,
    module: xlsynth::dslx::TypecheckedModule,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let (package, link) = super::helper_types::DslxLink::compile_with_bindings(module)?;
    let compiled =
        xlsynth_bx::compile_with_source_and_loop_bindings(source, package, reset, options)
            .map_err(anyhow::Error::msg)?;
    Ok(CompiledBx::from_bx(module_name, compiled, Some(link)))
}

impl CompiledBx {
    fn from_bx(
        module_name: &str,
        compiled: xlsynth_bx::CompiledModule,
        dslx: Option<super::helper_types::DslxLink>,
    ) -> Self {
        Self {
            module_name: module_name.to_owned(),
            imported_modules: compiled.imported_modules,
            imported_structs: compiled.imported_structs,
            source: compiled.source,
            package: compiled.package,
            dslx,
            imports: compiled
                .imports
                .into_iter()
                .map(super::helper_types::DslxLink::from_import)
                .collect(),
        }
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
        let independent = |owner: &str| {
            anyhow::anyhow!(
                "DSLX module '{owner}' was checked independently by the supplied DSLX module and BX imports; use one source for '{owner}' (remove the redundant BX use and call the supplied helper via extern, or omit the separate DSLX input)"
            )
        };
        let mut external = BTreeSet::new();
        if let Some(link) = &self.dslx {
            let mut bx_imports = self.imported_modules.clone();
            bx_imports.extend(declarations.iter().filter_map(|decl| {
                decl.name()
                    .split_once("::")
                    .map(|(owner, _)| owner.to_owned())
            }));
            link.module.collect_import_subjects(&mut external);
            external.insert(link.module.get_module().get_name());
            if let Some(owner) = bx_imports.intersection(&external).next() {
                return Err(independent(owner));
            }
            link.add_signatures(&block, &mut declarations, &mut interface, |owner| {
                if bx_imports.contains(owner) {
                    return Err(independent(owner));
                }
                external.insert(owner.to_owned());
                Ok(())
            })?;
        }
        for link in &self.imports {
            link.add_signatures(&block, &mut declarations, &mut interface, |owner| {
                if external.contains(owner) {
                    return Err(independent(owner));
                }
                Ok(())
            })?;
        }
        // BX structs have unqualified source keys. DSLX declarations keep the
        // owning module in theirs. Diagnose overlaps before rendering either.
        let bx_names: BTreeSet<_> = declarations
            .iter()
            .map(TypeDecl::name)
            .filter(|name| !name.contains("::"))
            .chain(std::iter::once(graph.name.as_str()))
            .collect();
        for (owner, local) in declarations
            .iter()
            .map(TypeDecl::name)
            .filter_map(|name| name.split_once("::"))
            .chain(
                interface
                    .helpers
                    .values()
                    .filter_map(|helper| helper.source_name.as_ref())
                    .map(|(owner, local)| (owner.as_str(), local.as_str())),
            )
        {
            ensure!(
                self.module_name != owner || !bx_names.contains(local),
                "duplicate source declaration '{owner}::{local}' in BX '{owner}.bx' and DSLX '{owner}.x'; rename one declaration"
            );
        }
        // Two linked copies can share a checked declaration and specialization.
        // Give it one public definition, keeping any other physical copy
        // callable.
        let mut names = std::collections::BTreeSet::new();
        for signature in interface.helpers.values_mut() {
            if let Some(name) = &signature.source_name
                && !names.insert(name.clone())
            {
                signature.source_name = None;
            }
        }
        readable::emit(&block, &declarations, &interface)
    }

    pub(super) fn readable_interface(
        &self,
        block: &BlockContext<'_>,
    ) -> Result<(Vec<TypeDecl>, StatefulInterface)> {
        let graph = block.graph;
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
        ensure!(
            source.reset.as_deref() == graph.reset.as_ref().map(|r| graph.port_name(r.port)),
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
            source
                .reset
                .iter()
                .map(|n| (n.as_str(), &Type::Bool))
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
        let outputs = if let Type::Named(key) = &source.result
            && let Some(import) = self.imported_structs.get(key)
        {
            let ty = super::helper_types::imported_struct(import, &mut declarations)?;
            let SourceType::Struct { name } = resolve_aliases(&ty, &declarations)? else {
                bail!("expected imported BX result struct {key}");
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
                bx_module: self.module_name.clone(),
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
        Type::Tuple(items) => SourceType::Tuple {
            members: items
                .iter()
                .map(|t| bx_type(t, module, declarations))
                .collect::<Result<_>>()?,
        },
        Type::Array { element, size } => SourceType::Array {
            element: Box::new(bx_type(element, module, declarations)?),
            size: *size,
        },
        Type::Named(key) => {
            if let Some(import) = module.imported_structs.get(key) {
                return super::helper_types::imported_struct(import, declarations);
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
            "example",
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
