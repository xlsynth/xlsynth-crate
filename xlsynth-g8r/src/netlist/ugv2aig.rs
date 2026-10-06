// SPDX-License-Identifier: Apache-2.0

//! Convert combinational unmapped gate Verilog into a `GateFn`.

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::aig::GateFn;
use crate::netlist::assigns_to_gatefn::project_gatefn_from_structural_assigns;
use crate::netlist::io::{read_ugv_from_path, select_module};

#[derive(Debug, Clone, Default)]
pub struct Ugv2AigOptions {
    pub module_name: Option<String>,
}

/// Converts the supported combinational UGV subset to an AIG.
pub fn convert_ugv2aig_paths(netlist_path: &Path, opts: &Ugv2AigOptions) -> Result<GateFn> {
    let parsed = read_ugv_from_path(netlist_path)?;
    let module = select_module(&parsed, opts.module_name.as_deref())?;
    if !module.instances.is_empty() {
        return Err(anyhow!(
            "ugv2aig currently requires a flattened selected module; flatten structural helper modules before conversion"
        ));
    }
    project_gatefn_from_structural_assigns(module, &parsed.nets, &parsed.interner)
        .map_err(|error| anyhow!(error))
}
