// SPDX-License-Identifier: Apache-2.0

use crate::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;
use crate::netlist::gatefn_from_netlist::{
    GateFnProjectOptions, project_gatefn_from_netlist_and_liberty_with_options,
};
use crate::netlist::io::{load_liberty_from_path, parse_netlist_from_path, select_module};
use anyhow::{Result, anyhow};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Gv2IrOptions {
    /// Optional module name to select when the netlist contains multiple
    /// modules.
    pub module_name: Option<String>,

    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,

    /// Optional function name to use in the emitted XLS IR package.
    pub output_function_name: Option<String>,
}

impl Default for Gv2IrOptions {
    fn default() -> Self {
        Self {
            module_name: None,
            collapse_sequential: true,
            output_function_name: None,
        }
    }
}

fn select_output_function_name(raw: &str, override_name: Option<&str>) -> Result<String> {
    let name = override_name.unwrap_or(raw);
    if name == "top" {
        return Err(anyhow!(
            "gv2ir would emit XLS IR function name 'top', but 'top' is reserved; pass --output_function_name <NAME> to choose a valid emitted function name"
        ));
    }
    if !is_valid_xls_function_name(name) {
        return Err(anyhow!(
            "gv2ir would emit invalid XLS IR function name '{}'; pass --output_function_name <NAME> to choose a valid emitted function name",
            name
        ));
    }
    Ok(name.to_string())
}

fn is_valid_xls_function_name(name: &str) -> bool {
    if is_xls_ir_keyword(name) {
        return false;
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn is_xls_ir_keyword(name: &str) -> bool {
    matches!(
        name,
        "after_all"
            | "array"
            | "assert"
            | "bits"
            | "block"
            | "chan"
            | "cover"
            | "file_number"
            | "fn"
            | "gate"
            | "instantiation"
            | "map"
            | "next_value"
            | "package"
            | "proc"
            | "recv"
            | "reg"
            | "ret"
            | "send"
            | "state"
            | "token"
            | "trace"
    )
}

pub fn convert_gv2ir_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    collapse_sequential: bool,
) -> Result<String> {
    convert_gv2ir_paths_with_options(
        netlist_path,
        liberty_proto_path,
        &Gv2IrOptions {
            module_name: None,
            collapse_sequential,
            output_function_name: None,
        },
    )
}

pub fn convert_gv2ir_paths_with_options(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2IrOptions,
) -> Result<String> {
    let parsed = parse_netlist_from_path(netlist_path)?;
    let module = select_module(&parsed, opts.module_name.as_deref())?;
    let module_function_name = parsed
        .interner
        .resolve(module.name)
        .ok_or_else(|| anyhow!("could not resolve module name symbol"))?;
    let output_function_name =
        select_output_function_name(module_function_name, opts.output_function_name.as_deref())?;

    // Liberty
    let liberty_lib = load_liberty_from_path(liberty_proto_path)?;
    let empty_dff_cells = HashSet::new();

    // Project GateFn
    let mut gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
        module,
        &parsed.nets,
        &parsed.interner,
        &liberty_lib,
        &empty_dff_cells,
        &empty_dff_cells,
        &GateFnProjectOptions {
            collapse_sequential: opts.collapse_sequential,
        },
    )
    .map_err(|e| anyhow!(e))?;

    // Convert to IR text
    gate_fn.name = output_function_name;
    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type)?;
    Ok(ir_pkg.to_string())
}
