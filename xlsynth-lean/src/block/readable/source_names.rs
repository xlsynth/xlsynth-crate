// SPDX-License-Identifier: Apache-2.0

//! Translate nominal catalogue identities into Lean namespace components.

use anyhow::Result;
use std::fmt::Write;

use super::{SourceType, identifier, lean, product_type};
use crate::block::source_types::{StructField, TypeDecl};

pub(super) fn declaring_name<'a>(bx_namespace: &'a str, name: &'a str) -> (&'a str, &'a str) {
    name.split_once("::").unwrap_or((bx_namespace, name))
}

pub(super) fn module_identifier(module: &str) -> Result<String> {
    match module {
        // The model already declares names such as XlsBlock.State. Keep
        // these modules distinct even when a source type has the same name.
        "XlsBlock" | "XlsReadable" | "XlsCodec" | "_root_" => {
            identifier(&format!("source:{module}"))
        }
        _ => identifier(module),
    }
}

pub(super) fn nominal_reference(
    bx_namespace: &str,
    name: &str,
    current: Option<(&str, &[TypeDecl])>,
    shadowed: &impl Fn(&str) -> bool,
) -> Result<String> {
    let (owner, local) = declaring_name(bx_namespace, name);
    let local_name = identifier(local)?;
    if current.is_some_and(|(m, _)| m == owner) && !shadowed(local) {
        return Ok(local_name);
    }
    // A dotted name can also start at a declaration in the surrounding
    // namespace. For example bus.Packet may denote a local bus's projection.
    let nested = match current {
        Some((module, decls)) => decls
            .iter()
            .any(|d| declaring_name(bx_namespace, d.name()) == (module, owner)),
        None => matches!(
            owner,
            "Helpers" | "EnumDispatch" | "State" | "Input" | "Output" | "Evaluation" | "Assertions"
        ),
    };
    let root = if nested || shadowed(owner) {
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
) -> Result<()> {
    let names = fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>();
    writeln!(
        out,
        "-- Source struct {}::{} {{ {} }}",
        lean::comment(module.strip_prefix("bx:").unwrap_or(module)),
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
    out.push_str(&lean::source_record(&identifier(local)?, &rendered)?);
    Ok(())
}
