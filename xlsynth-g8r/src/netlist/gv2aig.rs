// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into a `GateFn` (AIG form).

use crate::aig::GateFn;
use crate::netlist::assigns_to_gatefn::project_gatefn_from_structural_assigns;
use crate::netlist::gatefn_from_netlist::{
    GateFnProjectOptions, project_gatefn_from_netlist_and_liberty_with_options,
};
use crate::netlist::io::{load_liberty_from_path, parse_netlist_from_path, select_module};
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

pub fn convert_gv2aig_paths_with_optional_liberty(
    netlist_path: &Path,
    liberty_proto_path: Option<&Path>,
    opts: &Gv2AigOptions,
) -> Result<GateFn> {
    let parsed = parse_netlist_from_path(netlist_path)?;
    let module = select_module(&parsed, opts.module_name.as_deref())?;

    let Some(liberty_proto_path) = liberty_proto_path else {
        // See STRUCTURAL_ASSIGNS.md for the exact Liberty-free assign subset and
        // sizing semantics accepted by this path.
        return project_gatefn_from_structural_assigns(module, &parsed.nets, &parsed.interner)
            .map_err(|e| anyhow!(e));
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

pub fn convert_gv2aig_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
) -> Result<GateFn> {
    convert_gv2aig_paths_with_optional_liberty(netlist_path, Some(liberty_proto_path), opts)
}
