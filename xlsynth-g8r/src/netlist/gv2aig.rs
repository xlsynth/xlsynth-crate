// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into a `GateFn` (AIG form).

use crate::aig::GateFn;
use crate::netlist::gv2aig_boundaries::{
    Gv2AigBoundaryExtraction, Gv2AigBoundaryRequest, Gv2AigBoundarySpec,
    extract_gatefn_with_boundaries, extract_gatefn_with_optional_boundary_request,
};
use crate::netlist::hierarchy::elaborate_hierarchy;
use crate::netlist::io::{load_liberty_from_path, read_gv_from_path, select_module};
use anyhow::{Result, anyhow};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Gv2AigOptions {
    pub module_name: Option<String>,

    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,

    /// If true, collapse recognized load-enable feedback while assuming prior Q
    /// is zero. Applies only when `collapse_sequential` is true; defaults to
    /// false.
    pub collapse_load_enable_feedback: bool,
}

impl Default for Gv2AigOptions {
    fn default() -> Self {
        Self {
            module_name: None,
            collapse_sequential: true,
            collapse_load_enable_feedback: false,
        }
    }
}

/// Converts strict GV and its Liberty library to a combinational AIG.
pub fn convert_gv2aig_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
) -> Result<GateFn> {
    convert_gv2aig_paths_with_optional_boundary_request(
        netlist_path,
        liberty_proto_path,
        opts,
        None,
    )
    .map(|extraction| extraction.gate_fn)
}

/// Converts selected source/sink cones after elaborating structural hierarchy.
pub fn convert_gv2aig_paths_with_boundaries(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
    boundaries: &Gv2AigBoundarySpec,
) -> Result<GateFn> {
    let parsed = read_gv_from_path(netlist_path)?;
    let module = select_module(&parsed, opts.module_name.as_deref())?;
    let elaborated = elaborate_hierarchy(&parsed, module)?;
    let liberty = load_liberty_from_path(liberty_proto_path)?;
    extract_gatefn_with_boundaries(
        &elaborated,
        &liberty,
        opts.collapse_sequential,
        opts.collapse_load_enable_feedback,
        boundaries,
    )
    .map_err(|error| anyhow!(error))
}

/// Converts compact source/sink selectors, or all top-module ports when no
/// request is supplied, using one hierarchy-aware extraction path.
pub fn convert_gv2aig_paths_with_optional_boundary_request(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
    request: Option<&Gv2AigBoundaryRequest>,
) -> Result<Gv2AigBoundaryExtraction> {
    let parsed = read_gv_from_path(netlist_path)?;
    let module = select_module(&parsed, opts.module_name.as_deref())?;
    let elaborated = elaborate_hierarchy(&parsed, module)?;
    let liberty = load_liberty_from_path(liberty_proto_path)?;
    extract_gatefn_with_optional_boundary_request(
        &elaborated,
        &liberty,
        opts.collapse_sequential,
        opts.collapse_load_enable_feedback,
        request,
    )
    .map_err(|error| anyhow!(error))
}
