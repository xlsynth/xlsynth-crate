// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into a `GateFn` (AIG form).

use crate::aig::GateFn;
use crate::netlist::gatefn_from_netlist::{
    GateFnProjectOptions, project_gatefn_from_netlist_and_liberty_with_options,
};
use crate::netlist::io::{load_liberty_from_path, parse_netlist_from_path};
use anyhow::{Result, anyhow};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Gv2AigOptions {
    pub module_name: Option<String>,

    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,
}

impl Default for Gv2AigOptions {
    fn default() -> Self {
        Self {
            module_name: None,
            collapse_sequential: true,
        }
    }
}

pub fn convert_gv2aig_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
) -> Result<GateFn> {
    let parsed = parse_netlist_from_path(netlist_path)?;

    let module = if let Some(ref module_name) = opts.module_name {
        parsed
            .modules
            .iter()
            .find(|m| {
                parsed
                    .interner
                    .resolve(m.name)
                    .is_some_and(|s| s == module_name.as_str())
            })
            .ok_or_else(|| {
                let mut available: Vec<String> = parsed
                    .modules
                    .iter()
                    .map(|m| {
                        parsed
                            .interner
                            .resolve(m.name)
                            .unwrap_or("<unknown>")
                            .to_string()
                    })
                    .collect();
                available.sort();
                anyhow!(format!(
                    "module '{}' not found in netlist; available modules: [{}]",
                    module_name,
                    available.join(", ")
                ))
            })?
    } else if parsed.modules.len() == 1 {
        &parsed.modules[0]
    } else {
        let mut available: Vec<String> = parsed
            .modules
            .iter()
            .map(|m| {
                parsed
                    .interner
                    .resolve(m.name)
                    .unwrap_or("<unknown>")
                    .to_string()
            })
            .collect();
        available.sort();
        return Err(anyhow!(format!(
            "netlist contains {} modules; specify --module_name; available modules: [{}]",
            parsed.modules.len(),
            available.join(", ")
        )));
    };

    let liberty_lib = load_liberty_from_path(liberty_proto_path)?;

    let empty_dff_cells = std::collections::HashSet::new();

    let gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
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

    Ok(gate_fn)
}
