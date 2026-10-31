// SPDX-License-Identifier: Apache-2.0

//! Keep BX declarations paired with the package produced by lowering them.
//!
//! Lowering erases the distinction between a struct and a tuple with the same
//! fields. Before assigning BX types to Lean values, compare each checked
//! port and register with the corresponding physical name, position and shape.

use anyhow::{Context, Result, anyhow, bail, ensure};
use xlsynth_bx::{
    CheckedModule, ImportOptions, ResetBehavior,
    ast::Type,
    check::{CheckedStmt, output_ports},
};
use xlsynth_pir::ir;

use super::source_types::{SourceType, StatefulInterface, StructField, TypeDecl, source_shape};
use super::{BlockContext, LeanBlockBundle, readable, select_block};

/// A checked BX module and its linked block IR, from the same compilation.
pub struct CompiledBx {
    pub(super) checked: CheckedModule,
    pub(super) package: ir::Package,
    dslx: Option<super::helper_types::DslxLink>,
}

/// Check BX source and lower it into the supplied IR package.
///
/// The package may already contain functions or blocks used by BX externs.
/// Linked functions have IR signatures here. Use [`compile_bx_with_dslx`] to
/// preserve source names and nominal types for supported helpers.
pub fn compile_bx(
    source: &str,
    package: ir::Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let checked =
        xlsynth_bx::check_source_with_options(source, options).map_err(anyhow::Error::msg)?;
    let package = xlsynth_bx::lower_module(&checked, package, reset).map_err(anyhow::Error::msg)?;
    Ok(CompiledBx {
        checked,
        package,
        dslx: None,
    })
}

/// Lower BX against the IR produced from a checked DSLX module.
///
/// The checked module supplies both the implementations for extern functions
/// and their declarations for readable Lean. Helpers with supported signatures
/// retain their declared types. Helpers with unsupported signatures use their
/// IR types.
///
/// # Errors
///
/// Returns an error if DSLX lowering, IR validation, BX checking, or linking
/// fails. BX extern declarations must match the linked functions' IR shapes.
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
    /// Borrow the warnings from checking the BX source that produced this
    /// package.
    ///
    /// Callers that treat warnings as errors can inspect them before emitting
    /// or writing the model.
    pub fn warnings(&self) -> &[xlsynth_bx::BxWarning] {
        self.checked.warnings()
    }

    /// The exact lowered package used for both raw and readable equations.
    pub fn package(&self) -> &ir::Package {
        &self.package
    }

    /// Emit the selected BX block, defaulting to the package's top or sole
    /// block.
    ///
    /// When the compilation retains checked DSLX, validate recovered helper
    /// signatures against their linked IR functions before emitting any
    /// modules.
    pub fn emit_readable(&self, block_name: Option<&str>) -> Result<LeanBlockBundle> {
        let graph = select_block(&self.package, block_name)?;
        let block = BlockContext::new(&self.package, graph)?;
        let (mut declarations, mut interface) = self.readable_interface(&block)?;
        if let Some(link) = &self.dslx {
            link.add_signatures(&block, &mut declarations, &mut interface)?;
        }
        readable::emit(&block, &declarations, &interface)
    }

    /// Pair the checked BX block's ordered ports and registers with their
    /// lowered IR shapes.
    ///
    /// Linked functions can be called at their physical IR types even when
    /// the interface supplies no source signature for them.
    pub(super) fn readable_interface(
        &self,
        block: &BlockContext<'_>,
    ) -> Result<(Vec<TypeDecl>, StatefulInterface)> {
        let graph = &block.graph;
        let source = self
            .checked
            .blocks()
            .iter()
            .find(|b| b.name == graph.name)
            .with_context(|| format!("block {} has no checked BX declaration", graph.name))?;
        ensure!(
            source.clock.as_deref() == graph.clock_port_name(),
            "BX clock differs from block {}",
            graph.name
        );
        ensure!(
            source.reset.as_deref() == graph.reset.as_ref().map(|r| graph.port_name(r.port)),
            "BX reset differs from block {}",
            graph.name
        );

        let mut declarations = Vec::new();
        let mut check = |kind: &str,
                         expected: Vec<(&str, &Type)>,
                         physical: Vec<(&str, &ir::Type)>|
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
                    let ty = bx_type(ty, &self.checked, &mut declarations)
                        .with_context(|| format!("BX {kind} {name}"))?;
                    ensure!(
                        &source_shape(&ty, &declarations)? == shape,
                        "BX {kind} {name} has type {ty:?} but block has {shape}"
                    );
                    Ok((name.to_owned(), ty))
                })
                .collect()
        };
        let ports = |port| (graph.port_name(port), graph.port_type(port));
        let inputs = check(
            "input",
            source
                .reset
                .iter()
                .map(|n| (n.as_str(), &Type::Bool))
                .chain(source.params.iter().map(|p| (p.name.as_str(), &p.ty)))
                .collect(),
            graph.input_ports().map(ports).collect(),
        )?
        .into_iter()
        .collect();
        let registers = check(
            "register",
            source
                .statements
                .iter()
                .filter_map(|s| match s {
                    CheckedStmt::Reg { name, ty, .. } => Some((name.as_str(), ty)),
                    _ => None,
                })
                .collect(),
            graph
                .registers
                .iter()
                .map(|r| (r.name.as_str(), &r.ty))
                .collect(),
        )?
        .into_iter()
        .collect();
        let outputs =
            output_ports(&source.result, self.checked.structs()).map_err(anyhow::Error::msg)?;
        let outputs = check(
            "output",
            outputs.iter().map(|p| (p.name.as_str(), &p.ty)).collect(),
            graph.output_ports().map(ports).collect(),
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
fn bx_type(
    ty: &Type,
    module: &CheckedModule,
    declarations: &mut Vec<TypeDecl>,
) -> Result<SourceType> {
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
        Type::Named(name) => {
            if !declarations.iter().any(|d| d.name() == name) {
                let def = module
                    .structs()
                    .iter()
                    .find(|s| s.name == *name)
                    .ok_or_else(|| anyhow!("unknown checked BX struct {name}"))?;
                let fields = def
                    .fields
                    .iter()
                    .map(|f| {
                        Ok(StructField {
                            name: f.name.clone(),
                            source_type: bx_type(&f.ty, module, declarations)
                                .with_context(|| format!("field {name}.{}", f.name))?,
                        })
                    })
                    .collect::<Result<_>>()?;
                declarations.push(TypeDecl::Struct {
                    name: name.clone(),
                    fields,
                });
            }
            SourceType::Struct { name: name.clone() }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Negative test: Swapped equal-width outputs must report mismatched names.
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
