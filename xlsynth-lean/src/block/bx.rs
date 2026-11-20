// SPDX-License-Identifier: Apache-2.0

//! Keep BX declarations paired with the package produced by lowering them.
//!
//! The checked BX module supplies concrete dimensions. Its written declarations
//! name records, while the linked graph fixes port order and the equations used
//! for both Lean models.

use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};
use xlsynth_bx::{
    CheckedModule, ImportOptions, LinkedStruct, ResetBehavior,
    ast::{Module, Param, Stmt, Type},
    check::CheckedStmt,
};
use xlsynth_pir::ir;

use super::source_types::{SourceType, StatefulInterface, StructField, TypeDecl, source_shape};
use super::{BlockContext, LeanBlockBundle, readable, select_block};

/// A checked BX module and its linked block IR, from the same compilation.
///
/// Retains the written BX annotations so readable emission can recover record
/// names without reopening the source files.
pub struct CompiledBx {
    checked: CheckedModule,
    block_symbols: xlsynth_bx::BlockSymbols,
    module_name: String,
    imported_structs: BTreeMap<String, Rc<LinkedStruct>>,
    source: Module,
    pub(super) package: ir::Package,
}

/// Check BX source with its import paths and lower it into the supplied IR
/// package.
///
/// The package may already contain functions or blocks used by BX externs.
///
/// `module_name` selects the Lean namespace for root BX declarations. For
/// example, passing `"design"` puts `Packet` at `XLS.design.Packet`, even
/// when the supplied IR package has a different name.
pub fn compile_bx(
    module_name: &str,
    source: &str,
    package: ir::Package,
    reset: ResetBehavior,
    options: &ImportOptions<'_>,
) -> Result<CompiledBx> {
    let xlsynth_bx::CompiledModule {
        checked,
        block_symbols,
        source,
        package,
        imported_structs,
        ..
    } = xlsynth_bx::compile_with_source(source, package, reset, options)
        .map_err(anyhow::Error::msg)?;
    Ok(CompiledBx {
        checked,
        block_symbols,
        module_name: module_name.to_owned(),
        imported_structs,
        source,
        package,
    })
}

impl CompiledBx {
    /// Follow transparent BX aliases back to their written declaration.
    fn written_type<'a>(&'a self, mut ty: &'a Type) -> &'a Type {
        while let Type::Named(name) = ty
            && let Some(alias) = self.source.type_aliases.iter().find(|a| a.name == *name)
        {
            ty = &alias.ty;
        }
        ty
    }

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
    /// The block and the types on its ports and registers keep their declaring
    /// modules. Emission rejects competing declarations at the same Lean path.
    pub fn emit_readable(&self, block_name: Option<&str>) -> Result<LeanBlockBundle> {
        let graph = select_block(&self.package, block_name)?;
        let block = BlockContext::new(&self.package, graph)?;
        let (declarations, interface) = self.readable_interface(&block)?;
        // Root and imported BX libraries can use the same logical module name.
        let mut bx_names = BTreeSet::new();
        for (owner, local) in declarations
            .iter()
            .map(TypeDecl::name)
            .chain([interface.block_name.as_str()])
            .filter(|name| !name.contains("::"))
            .map(|name| xlsynth_bx::library_item(name).unwrap_or((&self.module_name, name)))
        {
            ensure!(
                bx_names.insert((owner, local)),
                "duplicate BX source declaration '{owner}::{local}' in the root and an imported library"
            );
        }
        readable::emit(&block, &declarations, &interface)
    }

    /// Match written BX types to checked ports and registers in physical
    /// declaration order.
    ///
    /// Port names distinguish equal-width fields whose order affects the Lean
    /// result.
    pub(super) fn readable_interface(
        &self,
        block: &BlockContext<'_>,
    ) -> Result<(Vec<TypeDecl>, StatefulInterface)> {
        let graph = &block.graph;
        let checked = self
            .checked
            .blocks()
            .iter()
            .find(|block| self.block_symbols.ir_name(&block.name) == graph.name)
            .with_context(|| format!("block {} has no checked BX declaration", graph.name))?;
        let source = self
            .source
            .blocks
            .iter()
            .find(|block| block.name == checked.name)
            .with_context(|| format!("block {} has no checked BX declaration", graph.name))?;
        ensure!(
            source.clock.as_deref() == graph.clock_port_name(),
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
        let read_type =
            |kind: &str, name: &str, ty: &Type, written: &Type, decls: &mut Vec<TypeDecl>| {
                Ok((
                    name.to_owned(),
                    bx_type(ty, written, self, decls)
                        .with_context(|| format!("BX {kind} {name}"))?,
                ))
            };
        let inputs = check(
            "input",
            reset
                .into_iter()
                .map(|name| (name, &Type::Bool, &Type::Bool))
                .chain(
                    checked
                        .params
                        .iter()
                        .zip(&source.params)
                        .map(|(p, w)| (p.name.as_str(), &p.ty, &w.ty)),
                )
                .map(|(name, ty, written)| read_type("input", name, ty, written, &mut declarations))
                .collect::<Result<_>>()?,
            graph.input_ports().map(ports).collect(),
            &declarations,
        )?
        .into_iter()
        .collect();
        let written_registers: BTreeMap<_, _> = source
            .statements
            .iter()
            .filter_map(|s| match s {
                Stmt::Reg { name, ty, .. } => Some((name, ty)),
                _ => None,
            })
            .collect();
        let registers = check(
            "register",
            checked
                .statements
                .iter()
                .filter_map(|s| match s {
                    CheckedStmt::Reg { name, ty, .. } => Some((name, ty)),
                    _ => None,
                })
                .map(|(name, ty)| {
                    read_type(
                        "register",
                        name,
                        ty,
                        written_registers.get(name).copied().unwrap_or(ty),
                        &mut declarations,
                    )
                })
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
        let written_result = self.written_type(&source.result);
        ensure!(
            self.imported_struct(written_result).is_none(),
            "imported DSLX result type {written_result} is unsupported by readable Lean"
        );
        let outputs = {
            let fields = self.written_fields(written_result);
            checked
                .output_ports
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let written = match (written_result, fields) {
                        (Type::Tuple(items), _) => &items[i],
                        (_, Some(fields)) => &fields[i].ty,
                        _ => written_result,
                    };
                    read_type("output", &p.name, &p.ty, written, &mut declarations)
                })
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
                block_name: source.name.clone(),
                inputs,
                registers,
                outputs,
            },
        ))
    }

    /// Find a BX struct's written fields in declaration order, including for
    /// templates.
    fn written_fields(&self, ty: &Type) -> Option<&[Param]> {
        let name = match ty {
            Type::Named(name)
            | Type::Applied { name, .. }
            | Type::ParametricStruct { name, .. }
            | Type::TypeApplied { name, .. } => name,
            _ => return None,
        };
        self.source
            .structs
            .iter()
            .find(|s| s.name == *name)
            .map(|s| s.fields.as_slice())
    }
}

/// The checked type supplies concrete dimensions and specialized BX layouts.
///
/// Follow the written annotation to identify records in fields, tuple members
/// and array elements.
fn bx_type(
    ty: &Type,
    written: &Type,
    module: &CompiledBx,
    declarations: &mut Vec<TypeDecl>,
) -> Result<SourceType> {
    let written = module.written_type(written);
    ensure!(
        module.imported_struct(written).is_none(),
        "imported DSLX type {written} is unsupported by readable Lean"
    );
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
                .enumerate()
                .map(|(i, item)| {
                    let written_item = match written {
                        Type::Tuple(items) => &items[i],
                        _ => item,
                    };
                    bx_type(item, written_item, module, declarations)
                })
                .collect::<Result<_>>()?,
        },
        Type::ParametricBits { .. }
        | Type::ParametricArray { .. }
        | Type::ParametricStruct { .. }
        | Type::TypeApplied { .. } => bail!("BX type {ty} is unsupported by readable Lean"),
        Type::Array { element, size } => {
            let written_element = match written {
                Type::Array { element, .. } | Type::ParametricArray { element, .. } => element,
                _ => element,
            };
            SourceType::Array {
                element: Box::new(bx_type(element, written_element, module, declarations)?),
                size: *size,
            }
        }
        Type::Named(key) => {
            if !declarations.iter().any(|d| d.name() == key) {
                let def = module
                    .checked
                    .structs()
                    .iter()
                    .find(|s| s.name == *key)
                    .with_context(|| format!("unknown BX struct {key}"))?;
                let written_fields = module.written_fields(written);
                let fields = def
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, field)| {
                        Ok(StructField {
                            name: field.name.clone(),
                            source_type: bx_type(
                                &field.ty,
                                written_fields.map_or(&field.ty, |fields| &fields[i].ty),
                                module,
                                declarations,
                            )
                            .with_context(|| format!("field {key}.{}", field.name))?,
                        })
                    })
                    .collect::<Result<_>>()?;
                declarations.push(TypeDecl {
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

    // Verifies: Lean resolves BX aliases, constant widths and nested instances.
    // Catches: Taking port or register types from unresolved annotations.
    #[test]
    fn readable_interface_uses_concrete_bx_types() -> Result<()> {
        let compiled = compile_bx(
            "example",
            r#"
type Byte = u8;
const WIDTH = u32:8;
struct Packet<N: u32> { data: uN[N] }
struct Wrap<T> { item: T }
block scalar(clk, x: Byte) -> Byte { x }
block sized(clk, x: uN[WIDTH]) -> uN[WIDTH] { x }
block packet(clk, x: Packet<8>) -> Packet<8> { x }
block nested(clk, x: Wrap<Byte>) -> (Byte, Wrap<Byte>) {
    reg stored: Wrap<Byte>;
    stored := x;
    (x.item, stored)
}
"#,
            xlsynth_bx::empty_package("example"),
            Default::default(),
            &Default::default(),
        )?;
        for block in ["scalar", "sized", "packet", "nested"] {
            compiled.emit_readable(Some(block)).with_context(|| block)?;
        }
        Ok(())
    }

    // Negative test: Swapped BX outputs must report both mismatched port names.
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
