// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into a `GateFn` (AIG form).

use crate::aig::GateFn;
use crate::netlist::dff_classify::classify_dff_cells_from_liberty;
use crate::netlist::gatefn_from_netlist::{
    GateFnProjectOptions, project_gatefn_from_netlist_and_liberty_with_options,
};
use crate::netlist::io::{load_liberty_from_path, parse_netlist_from_path};
use anyhow::{Result, anyhow};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Gv2AigOptions {
    pub module_name: Option<String>,

    /// Explicit DFF cell names treated as identity (D->Q), typically from
    /// `--dff_cells`.
    pub dff_cells_identity: HashSet<String>,

    /// If set, any cell with an output pin function exactly equal to this
    /// string is treated as an identity DFF (Q = D).
    pub dff_cell_formula: Option<String>,

    /// If set, any cell with an output pin function exactly equal to this
    /// string is treated as an inverted-output DFF (QN = NOT(D)).
    pub dff_cell_invert_formula: Option<String>,

    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,
}

impl Default for Gv2AigOptions {
    fn default() -> Self {
        Self {
            module_name: None,
            dff_cells_identity: HashSet::new(),
            dff_cell_formula: None,
            dff_cell_invert_formula: None,
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

    let dff_sets = classify_dff_cells_from_liberty(
        &liberty_lib,
        &opts.dff_cells_identity,
        opts.dff_cell_formula.as_deref(),
        opts.dff_cell_invert_formula.as_deref(),
    );

    let gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
        module,
        &parsed.nets,
        &parsed.interner,
        &liberty_lib,
        &dff_sets.identity,
        &dff_sets.inverted,
        &GateFnProjectOptions {
            collapse_sequential: opts.collapse_sequential,
        },
    )
    .map_err(|e| anyhow!(e))?;

    Ok(gate_fn)
}
