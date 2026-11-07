// SPDX-License-Identifier: Apache-2.0

//! Translate nominal catalogue identities into Lean namespace components.

use anyhow::{Result, bail};
use std::{collections::BTreeSet, fmt::Write};

use super::{SourceType, identifier, lean, product_type};
use crate::block::source_types::{StatefulInterface, StructField, TypeDecl};

pub(super) fn declaring_name<'a>(bx_namespace: &'a str, name: &'a str) -> (&'a str, &'a str) {
    name.split_once("::").unwrap_or((bx_namespace, name))
}

/// A type owns the namespace for its projections, constructors and other Lean
/// declarations. An emitted module must not also supply names under that type.
pub(super) fn check_type_namespaces(
    declarations: &[TypeDecl],
    interface: &StatefulInterface,
) -> Result<()> {
    let types: BTreeSet<_> = declarations
        .iter()
        .map(|decl| declaring_name(&interface.bx_module, decl.name()))
        .collect();
    let owners: BTreeSet<_> = types
        .iter()
        .map(|&(owner, _)| owner)
        .chain(std::iter::once(interface.bx_module.as_str()))
        .chain(
            interface
                .helpers
                .values()
                .filter_map(|helper| helper.source_name.as_ref().map(|(owner, _)| owner.as_str())),
        )
        .collect();
    for owner in owners {
        let mut prefix = owner;
        while let Some((module, local)) = prefix.rsplit_once('.') {
            if types.contains(&(module, local)) {
                bail!(
                    "source module '{owner}' would emit declarations under the Lean namespace of source type '{module}::{local}' (XLS.{prefix}); rename the type or module"
                );
            }
            prefix = module;
        }
    }
    Ok(())
}

pub(super) fn module_identifier(module: &str) -> Result<String> {
    let mut namespace = String::from("XLS");
    for component in module.split('.') {
        write!(namespace, ".{}", identifier(component)?)?;
    }
    Ok(namespace)
}

pub(super) fn nominal_reference(
    bx_namespace: &str,
    name: &str,
    current: Option<(&str, &[TypeDecl])>,
    shadowed: &impl Fn(&str) -> bool,
) -> Result<String> {
    let (owner, local) = declaring_name(bx_namespace, name);
    let local_name = identifier(local)?;
    if current.is_some_and(|(module, _)| module == owner) && !shadowed(local) {
        return Ok(local_name);
    }
    // Source namespaces may themselves contain a declaration or submodule
    // named XLS. In the generated evaluator only local binders can hide XLS.
    let nested_xls = current.is_some_and(|(_, decls)| {
        decls.iter().any(|decl| {
            let (module, local) = declaring_name(bx_namespace, decl.name());
            local == "XLS" || module.split('.').any(|component| component == "XLS")
        })
    });
    let root = if nested_xls || shadowed("XLS") {
        "_root_."
    } else {
        ""
    };
    Ok(format!("{root}{}.{local_name}", module_identifier(owner)?))
}

pub(super) fn source_type_in(
    bx_namespace: &str,
    ty: &SourceType,
    current: Option<(&str, &[TypeDecl])>,
    shadowed: &impl Fn(&str) -> bool,
) -> Result<String> {
    match ty {
        SourceType::Bool => Ok("_root_.Bool".into()),
        SourceType::Bits { width } => Ok(format!("_root_.BitVec {width}")),
        SourceType::Array { element, size } => Ok(format!(
            "_root_.Vector ({}) {size}",
            source_type_in(bx_namespace, element, current, shadowed)?
        )),
        SourceType::Tuple { members } => Ok(product_type(
            &members
                .iter()
                .map(|ty| source_type_in(bx_namespace, ty, current, shadowed))
                .collect::<Result<Vec<_>>>()?,
            true,
        )),
        SourceType::Enum { name } | SourceType::Struct { name } | SourceType::Alias { name } => {
            nominal_reference(bx_namespace, name, current, shadowed)
        }
    }
}

pub(super) fn write_struct(
    out: &mut String,
    bx_namespace: &str,
    module: &str,
    local: &str,
    fields: &[StructField],
    declarations: &[TypeDecl],
) -> Result<String> {
    let names = fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>();
    writeln!(
        out,
        "-- Source struct {}::{} {{ {} }}",
        lean::comment(module),
        lean::comment(local),
        lean::comment(&names.join(", "))
    )?;
    let rendered = fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            Ok((
                identifier(&field.name)?,
                source_type_in(
                    bx_namespace,
                    &field.source_type,
                    Some((module, declarations)),
                    &|n| names[..index].contains(&n),
                )?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let name = identifier(local)?;
    out.push_str(&lean::record_declaration(&name, &rendered, None, &[])?);
    Ok(lean::record_backing_name(&name, &rendered))
}
